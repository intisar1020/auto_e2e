"""SimpleBEV-style dense voxel unprojection for camera feature projection to BEV.

Reference implementation derived from simple_bev/ (segnet.py & utils/vox.py):
- Constructs an explicit 3D voxel grid in ego/vehicle frame meters.
- Unprojects 3D voxel centers to 2D image planes using the CameraProjectionModel operator.
- Samples 2D image features directly via bilinear grid_sample into the 3D voxel grid.
- Averages unprojected features across valid camera views (masked mean).
- Compresses the 3D height/pillar dimension into 2D BEV features via a 2D CNN (bev_compressor).

This module contains NO learned query embeddings and NO residual query drowning,
ensuring that the output BEV features directly reflect the camera image content.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .projection import (
    ImageTransform,
    PinholeProjection,
    PseudoProjection,
)


class SimpleBEVViewFusion(nn.Module):
    """Dense voxel unprojection (SimpleBEV style) from multi-view images to BEV.

    Args:
        num_views: Number of camera views (default 6).
        embed_dim: Channel dimension of per-view features and output BEV (default 256).
        bev_h: BEV grid height / rows (default 450).
        bev_w: BEV grid width / cols (default 300).
        num_points_in_pillar: Number of Z height voxels along each pillar (default 4).
        pc_range: Ego point cloud bounds (x_min, y_min, z_min, x_max, y_max, z_max).
        image_size: Model input image size (default 256).
    """

    def __init__(
        self,
        num_views: int = 6,
        embed_dim: int = 256,
        bev_h: int = 450,
        bev_w: int = 300,
        num_points_in_pillar: int = 4,
        pc_range: tuple[float, float, float, float, float, float] = (
            -60.0,
            -60.0,
            -5.0,
            120.0,
            60.0,
            3.0,
        ),
        image_size: int = 256,
        **kwargs,
    ):
        super().__init__()

        self.num_views = num_views
        self.embed_dim = embed_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.image_size = image_size
        self.image_transform = ImageTransform.square(image_size)

        # 3D Voxel -> 2D BEV feature compressor (matching SimpleBEV's bev_compressor)
        # Input channels: embed_dim * num_points_in_pillar (Z dimension collapsed into channels)
        in_ch = embed_dim * num_points_in_pillar
        self.bev_compressor = nn.Sequential(
            nn.Conv2d(in_ch, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(embed_dim),
            nn.GELU(),
        )

        # Fallback pseudo matrix parameter for calibration-free testing
        self.pseudo_projection = nn.Parameter(torch.randn(3, 4) * 0.01)

        self._init_3d_grid()

    def _init_3d_grid(self):
        """Pre-compute 3D physical coordinates (meters) for all voxel centers in ego frame.

        Ego frame conventions:
            X = forward (+x = front, -x = rear)
            Y = left (+y = left, -y = right)
            Z = up (+z = up, -z = down)
        """
        x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range

        # Row 0 -> x_max (front), Row bev_h-1 -> x_min (rear)
        xs = torch.linspace(x_max, x_min, self.bev_h)
        # Col 0 -> y_max (left), Col bev_w-1 -> y_min (right)
        ys = torch.linspace(y_max, y_min, self.bev_w)
        # Z height voxels from z_min to z_max
        zs = torch.linspace(z_min, z_max, self.num_points_in_pillar)

        # Grid of shape [bev_h, bev_w, num_points_in_pillar, 3]
        grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing="ij")
        grid_3d = torch.stack([grid_x, grid_y, grid_z], dim=-1)

        # Buffer for runtime reuse
        self.register_buffer("reference_points_3d", grid_3d)

    def forward(
        self,
        fused_per_view: torch.Tensor,
        B: int,
        V: int,
        projection=None,
        geometry_type: str | None = None,
        image_transform=None,
    ) -> torch.Tensor:
        """Unproject 2D camera image features into 3D voxel grid and compress to BEV.

        Args:
            fused_per_view: [B*V, C, Hf, Wf] multi-view feature maps from backbone.
            B: Batch size.
            V: Number of views.
            projection: CameraProjectionModel operator (PinholeProjection / FThetaProjection / etc.).
            geometry_type: Optional geometry label.
            image_transform: Optional ImageTransform override.

        Returns:
            bev_features: [B, C, bev_h, bev_w] 2D BEV feature map.
        """
        _, C, Hf, Wf = fused_per_view.shape
        dtype = fused_per_view.dtype
        device = fused_per_view.device

        # 0. Resolve projection operator
        if projection is None:
            projection = PseudoProjection(self.pseudo_projection, num_views=V)

        it = image_transform if image_transform is not None else self.image_transform

        # 1. Reshape 3D reference points to [M, 3] where M = bev_h * bev_w * num_z
        num_z = self.num_points_in_pillar
        ref_3d = self.reference_points_3d.to(device=device, dtype=dtype)
        ref_flat = ref_3d.reshape(-1, 3)  # [M, 3]

        # 2. Unproject 3D reference points into all camera image planes
        # result.uv_norm: [Bp, V, M, 2] in [0, 1] range
        # result.valid_mask: [Bp, V, M] boolean visibility mask
        result = projection.project_ego_to_image(ref_flat, it)
        Bp = result.uv_norm.shape[0]

        uv_norm = result.uv_norm  # [Bp, V, M, 2]
        valid_mask = result.valid_mask.float()  # [Bp, V, M]

        if Bp == 1 and B > 1:
            uv_norm = uv_norm.expand(B, -1, -1, -1)
            valid_mask = valid_mask.expand(B, -1, -1)

        # 3. Convert [0, 1] normalized pixel coords to grid_sample's [-1, 1] range
        grid = uv_norm * 2.0 - 1.0  # [B, V, M, 2]

        # 4. Reshape feature maps for batch indexing
        feat_views = fused_per_view.reshape(B, V, C, Hf, Wf)

        # Accumulators for multi-camera voxel fusion
        voxels_sum = torch.zeros(B, C, self.bev_h * self.bev_w * num_z, device=device, dtype=dtype)
        visible_counts = torch.zeros(B, 1, self.bev_h * self.bev_w * num_z, device=device, dtype=dtype)

        for v_idx in range(V):
            feat_v = feat_views[:, v_idx]  # [B, C, Hf, Wf]
            grid_v = grid[:, v_idx].unsqueeze(2)  # [B, M, 1, 2]
            mask_v = valid_mask[:, v_idx].unsqueeze(1)  # [B, 1, M]

            # Sample 2D features for all 3D voxel points
            sampled_v = F.grid_sample(
                feat_v, grid_v, mode="bilinear", padding_mode="zeros", align_corners=False
            )  # [B, C, M, 1]
            sampled_v = sampled_v.squeeze(3) * mask_v  # [B, C, M]

            voxels_sum = voxels_sum + sampled_v
            visible_counts = visible_counts + mask_v

        # Average across valid (visible) camera views (SimpleBEV reduce_masked_mean)
        voxels_avg = voxels_sum / visible_counts.clamp(min=1.0)  # [B, C, M]

        # 5. Reshape 3D voxel grid to [B, C, num_z, bev_h, bev_w]
        voxels_3d = voxels_avg.reshape(B, C, self.bev_h, self.bev_w, num_z)
        # Permute to [B, C, num_z, bev_h, bev_w] then collapse Z into channels -> [B, C * num_z, bev_h, bev_w]
        voxels_cat = voxels_3d.permute(0, 1, 4, 2, 3).reshape(
            B, C * num_z, self.bev_h, self.bev_w
        )

        # 6. Compress 3D voxel grid to 2D BEV feature map
        bev_features = self.bev_compressor(voxels_cat)  # [B, C, bev_h, bev_w]

        # Zero out BEV cells where no pillar point was visible by any camera
        bev_visible = (visible_counts.reshape(B, 1, self.bev_h, self.bev_w, num_z).sum(dim=-1) > 0).float()
        bev_features = bev_features * bev_visible

        return bev_features
