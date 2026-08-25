"""Pure-PyTorch BEVFormer-style encoder for KITScenes trajectory prediction.

This file is deliberately independent of the MMDetection3D/mmcv registries.
It ports the BEVFormer operations that matter for a Tesla-style end-to-end
planner:

    * multi-scale deformable attention (pure :func:`torch.nn.functional.grid_sample`)
    * spatial cross-attention (BEV queries -> camera feature maps)
    * temporal self-attention (previous BEV -> current BEV)
    * BEVFormer encoder layers / encoder
    * a small raster map encoder + image/map BEV fusion

Detection heads (bboxes, classes, Hungarian matching) are intentionally absent.
The network ends in :class:`QueryPlanner`, which predicts the 6.4 s
acceleration/curvature trajectory contract used by ``train_main.py``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_components.backbone import Backbone
from model_components.trajectory_planning.query_planner import QueryPlanner
from model_components.view_fusion.projection import ImageTransform


def _xavier_uniform(module: nn.Module) -> None:
    for p in module.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        elif p.dim() == 1:
            nn.init.zeros_(p)


def _multi_scale_deformable_attn(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Differentiable multi-scale deformable attention implemented with grid_sample.

    This replaces mmcv's CUDA ``multi_scale_deform_attn`` with pure PyTorch.

    Args:
        value: ``[B, sum_l(H_l*W_l), embed_dims]`` flattened value tokens.
        spatial_shapes: ``[num_levels, 2]`` ``(h, w)`` per feature level.
        level_start_index: ``[num_levels]`` flattened-token start index per level.
        sampling_locations: ``[B, num_query, num_heads, num_levels, num_points, 2]``
            normalized x/y in ``[0, 1]``.
        attention_weights: ``[B, num_query, num_heads, num_levels, num_points]``.

    Returns:
        ``[B, num_query, embed_dims]``.
    """
    B, Nq, H, L, P, _ = sampling_locations.shape
    C = value.shape[-1]
    if C % H != 0:
        raise ValueError(f"embed_dims ({C}) must be divisible by num_heads ({H})")
    dim_per_head = C // H
    out_dtype = value.dtype

    # Always run the geometry/math in fp32.  fp16 grid_sample with many sums is
    # the part that becomes unstable in original BEVFormer implementations.
    value = value.float()
    sampling_locations = sampling_locations.float()
    attention_weights = attention_weights.float()

    value = value.view(B, -1, H, dim_per_head).permute(0, 2, 3, 1)  # B,H,D,total
    output = value.new_zeros(B, H, Nq, dim_per_head)

    for level_idx, (h, w) in enumerate(spatial_shapes.tolist()):
        start = int(level_start_index[level_idx])
        end = start + int(h) * int(w)
        value_level = value[..., start:end].reshape(B, H, dim_per_head, h, w)

        # grid_sample wants [N, C, H_in, W_in] and [N, H_out, W_out, 2].
        grid = sampling_locations[:, :, :, level_idx]  # B,Nq,H,P,2
        grid = grid.permute(0, 2, 1, 3, 4).reshape(B * H, Nq, P, 2)
        value_level = value_level.reshape(B * H, dim_per_head, h, w)
        sampled = F.grid_sample(
            value_level,
            grid * 2.0 - 1.0,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )  # [B*H, dim, Nq, P]
        sampled = sampled.reshape(B, H, dim_per_head, Nq, P)
        sampled = sampled.permute(0, 1, 3, 4, 2)  # B,H,Nq,P,D

        w_lvl = attention_weights[:, :, :, level_idx]  # B,Nq,H,P
        w_lvl = w_lvl.permute(0, 2, 1, 3).unsqueeze(-1)  # B,H,Nq,P,1
        output = output + (sampled * w_lvl).sum(dim=3)  # B,H,Nq,D

    output = output.permute(0, 2, 1, 3).reshape(B, Nq, C)
    return output.to(out_dtype)


class PureMSDeformableAttention(nn.Module):
    """BEVFormer's ``MSDeformableAttention3D``, ported to pure PyTorch."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 8,
        dropout: float = 0.1,
        output_proj: bool = False,
    ) -> None:
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError("embed_dims must be divisible by num_heads")
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims) if output_proj else None
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.zeros_(self.sampling_offsets.weight)
        thetas = torch.arange(self.num_heads, dtype=torch.float32)
        thetas = thetas * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(
            1, self.num_levels, self.num_points, 1
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.reshape(-1)
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        if self.output_proj is not None:
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        B, Nq, _ = query.shape
        H = self.num_heads
        L = self.num_levels
        P = self.num_points

        value = self.value_proj(value)
        sampling_offsets = self.sampling_offsets(query).view(B, Nq, H, L, P, 2)
        attention_weights = self.attention_weights(query).view(
            B, Nq, H, L * P
        )
        attention_weights = attention_weights.softmax(dim=-1).view(
            B, Nq, H, L, P
        )

        if reference_points.shape[-1] != 2:
            raise ValueError(
                "reference_points last dimension must be 2 in this trajectory port"
            )
        num_z_anchors = reference_points.shape[2]
        if P % num_z_anchors != 0:
            raise ValueError(
                "num_points must be divisible by the number of 3D pillar anchors"
            )

        offset_normalizer = spatial_shapes.to(query.device).float()[:, [1, 0]]
        ref = reference_points[:, :, None, None, None, :, :]  # B,Nq,1,1,1,Z,2
        offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        offsets = offsets.view(B, Nq, H, L, P // num_z_anchors, num_z_anchors, 2)
        sampling_locations = ref + offsets
        sampling_locations = sampling_locations.reshape(B, Nq, H, L, P, 2)

        output = _multi_scale_deformable_attn(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
        )
        if self.output_proj is not None:
            output = self.output_proj(output)
        return self.dropout(output)


class PureSpatialCrossAttention(nn.Module):
    """BEVFormer spatial cross-attention with no mmcv dependency."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_cams: int = 6,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.deformable_attention = PureMSDeformableAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=0.0,
            output_proj=False,
        )
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        value_list: Sequence[torch.Tensor],
        reference_points_cam: torch.Tensor,
        bev_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attend each BEV query to each camera's multi-level feature maps.

        Args:
            query: ``[B, Nq, C]`` BEV queries.
            value_list: list of ``[B, V, C, H_l, W_l]`` image feature levels.
            reference_points_cam: ``[B, V, Nq, Z, 2]`` projected pillar anchors.
            bev_mask: ``[B, V, Nq, Z]`` valid projection mask.
            spatial_shapes: ``[num_levels, 2]``.
            level_start_index: ``[num_levels]``.
        """
        if residual is None:
            residual = query
        B, Nq, C = query.shape
        V = reference_points_cam.shape[1]

        flattened = []
        for feat in value_list:
            _, v, _, h, w = feat.shape
            flattened.append(feat.reshape(B, V, C, h * w).permute(0, 1, 3, 2))
        value_per_cam = torch.cat(flattened, dim=2)  # B,V,total_tokens,C

        output = query.new_zeros(B, Nq, C)
        visible_count = query.new_zeros(B, Nq, 1)
        for cam_idx in range(V):
            out_cam = self.deformable_attention(
                query,
                value_per_cam[:, cam_idx],
                reference_points_cam[:, cam_idx],
                spatial_shapes,
                level_start_index,
            )
            cam_visible = bev_mask[:, cam_idx].any(dim=-1).float().unsqueeze(-1)
            output = output + out_cam * cam_visible
            visible_count = visible_count + cam_visible

        output = output / visible_count.clamp(min=1.0)
        output = self.output_proj(output)
        return self.dropout(output) + residual


class PureTemporalSelfAttention(nn.Module):
    """BEVFormer temporal self-attention (current BEV attends to previous BEV)."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 1,
        num_points: int = 4,
        num_bev_queue: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError("embed_dims must be divisible by num_heads")
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue

        concat_dim = embed_dims * num_bev_queue
        self.sampling_offsets = nn.Linear(
            concat_dim,
            num_bev_queue * num_heads * num_levels * num_points * 2,
        )
        self.attention_weights = nn.Linear(
            concat_dim,
            num_bev_queue * num_heads * num_levels * num_points,
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.zeros_(self.sampling_offsets.weight)
        thetas = torch.arange(self.num_heads, dtype=torch.float32)
        thetas = thetas * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(
            1, self.num_levels * self.num_bev_queue, self.num_points, 1
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.reshape(-1)
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Nq, C = query.shape
        identity = query
        if query_pos is not None:
            query = query + query_pos

        if value is None:
            value = torch.stack([query, query], dim=1).reshape(B * 2, Nq, C)
        if reference_points is None:
            raise ValueError("reference_points are required for temporal attention")
        if spatial_shapes is None or level_start_index is None:
            raise ValueError("spatial_shapes/level_start_index are required")

        H = self.num_heads
        L = self.num_levels
        P = self.num_points
        queue = self.num_bev_queue

        # value: [2B, Nq, C].  The first half is history/current; concatenate the
        # first half with the current query to predict offsets, as in BEVFormer.
        query_input = torch.cat([value[:B], query], dim=-1)  # [B,Nq,2C]
        value_proj = self.value_proj(value)  # [2B,Nq,C]

        offsets = self.sampling_offsets(query_input).view(
            B, Nq, H, queue, L, P, 2
        )
        attn = self.attention_weights(query_input).view(
            B, Nq, H, queue, L * P
        )
        attn = attn.softmax(dim=-1).view(B, Nq, H, queue, L, P)

        offsets = offsets.permute(0, 3, 1, 2, 4, 5, 6).reshape(
            B * queue, Nq, H, L, P, 2
        )
        attn = attn.permute(0, 3, 1, 2, 4, 5).reshape(
            B * queue, Nq, H, L, P
        )

        offset_normalizer = spatial_shapes.to(query.device).float()[:, [1, 0]]
        ref = reference_points[:, :, None, :, None, :]  # 2B,Nq,1,1,1,2
        sampling_locations = ref + offsets / offset_normalizer[
            None, None, None, :, None, :
        ]

        output = _multi_scale_deformable_attn(
            value_proj,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attn,
        )  # [2B,Nq,C]
        output = output.reshape(B, queue, Nq, C).mean(dim=1)
        output = self.output_proj(output)
        return self.dropout(output) + identity


class PureBEVFormerEncoderLayer(nn.Module):
    """One BEVFormer encoder layer: temporal self-attn, spatial cross-attn, FFN."""

    def __init__(
        self,
        embed_dims: int,
        num_cams: int,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 8,
        dropout: float = 0.1,
        feedforward_channels: int = 512,
    ) -> None:
        super().__init__()
        self.temporal_attn = PureTemporalSelfAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=1,
            num_points=4,
            num_bev_queue=2,
            dropout=dropout,
        )
        self.spatial_attn = PureSpatialCrossAttention(
            embed_dims=embed_dims,
            num_cams=num_cams,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        value_list: Sequence[torch.Tensor],
        bev_pos: torch.Tensor,
        ref_2d: torch.Tensor,
        ref_3d_cam: torch.Tensor,
        bev_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        bev_h: int,
        bev_w: int,
        hybrid_value: torch.Tensor,
    ) -> torch.Tensor:
        query = self.temporal_attn(
            query=query,
            value=hybrid_value,
            reference_points=ref_2d,
            spatial_shapes=torch.tensor(
                [[bev_h, bev_w]], device=query.device, dtype=torch.long
            ),
            level_start_index=torch.tensor([0], device=query.device, dtype=torch.long),
            query_pos=bev_pos,
        )
        query = self.norm1(query)

        query = self.spatial_attn(
            query=query,
            value_list=value_list,
            reference_points_cam=ref_3d_cam,
            bev_mask=bev_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        query = self.norm2(query)
        query = query + self.ffn(query)
        return query


class PureBEVFormerEncoder(nn.Module):
    """Pure-PyTorch BEVFormer encoder (no detection decoder)."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_cams: int = 6,
        bev_h: int = 64,
        bev_w: int = 64,
        pc_range: Sequence[float] = (-60.0, -60.0, -5.0, 120.0, 60.0, 3.0),
        num_points_in_pillar: int = 4,
        num_encoder_layers: int = 2,
        num_heads: int = 4,
        num_levels: int = 4,
        num_points: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = tuple(pc_range)
        self.num_points_in_pillar = num_points_in_pillar
        self.layers = nn.ModuleList(
            [
                PureBEVFormerEncoderLayer(
                    embed_dims=embed_dims,
                    num_cams=num_cams,
                    num_heads=num_heads,
                    num_levels=num_levels,
                    num_points=num_points,
                    dropout=dropout,
                    feedforward_channels=embed_dims * 2,
                )
                for _ in range(num_encoder_layers)
            ]
        )

    @staticmethod
    def get_reference_points(
        H: int,
        W: int,
        Z: float,
        num_points_in_pillar: int,
        dim: str,
        bs: int,
        device,
        dtype,
    ) -> torch.Tensor:
        if dim == "3d":
            zs = (
                torch.linspace(
                    0.5, Z - 0.5, num_points_in_pillar, dtype=torch.float32, device=device
                )
                .view(-1, 1, 1)
                .expand(num_points_in_pillar, H, W)
                / Z
            )
            xs = (
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device)
                .view(1, 1, W)
                .expand(num_points_in_pillar, H, W)
                / W
            )
            ys = (
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device)
                .view(1, H, 1)
                .expand(num_points_in_pillar, H, W)
                / H
            )
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1).permute(0, 2, 1, 3)
            return ref_3d.to(dtype=dtype)

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device) / H,
            torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device) / W,
            indexing="ij",
        )
        ref_2d = torch.stack((ref_x, ref_y), -1).reshape(1, H * W, 2).unsqueeze(2)
        ref_2d = ref_2d.repeat(bs, 1, 1, 1)
        return ref_2d.to(dtype=dtype)

    def _project_ref_3d(
        self,
        ref_3d: torch.Tensor,
        projection=None,
        camera_params: Optional[torch.Tensor] = None,
        image_transform=None,
    ):
        B, N, Z, _ = ref_3d.shape
        pc = self.pc_range
        ego = ref_3d.clone()
        ego[..., 0] = ego[..., 0] * (pc[3] - pc[0]) + pc[0]
        ego[..., 1] = ego[..., 1] * (pc[4] - pc[1]) + pc[1]
        ego[..., 2] = ego[..., 2] * (pc[5] - pc[2]) + pc[2]
        if projection is not None:
            result = projection.project_ego_to_image(
                ego.reshape(N * Z, 3), image_transform
            )
            uv = result.uv_norm
            mask = result.valid_mask
            if uv.shape[0] == 1 and B > 1:
                uv = uv.expand(B, -1, -1, -1)
                mask = mask.expand(B, -1, -1)
            uv = uv.reshape(B, -1, N, Z, 2)
            mask = mask.reshape(B, -1, N, Z)
            return uv, mask

        if camera_params is None:
            raise ValueError("projection or camera_params is required")
        if camera_params.dim() != 4 or camera_params.shape[-2:] != (3, 4):
            raise ValueError(
                f"camera_params must be [B,V,3,4], got {tuple(camera_params.shape)}"
            )
        ones = torch.ones(
            B, N, Z, 1, device=ego.device, dtype=ego.dtype
        )
        pts = torch.cat([ego, ones], dim=-1)  # B,N,Z,4
        projected = torch.einsum(
            "bvij,bnzj->bvnzi", camera_params.to(ego.device), pts
        )
        depth = projected[..., 2]
        depth_safe = depth.clamp(min=1e-5).unsqueeze(-1)
        uv = projected[..., :2] / depth_safe
        it = image_transform or ImageTransform.square(256)
        w, h = it.wh
        wh = torch.tensor([w, h], device=uv.device, dtype=uv.dtype)
        uv = uv / wh
        in_bounds = (
            (uv[..., 0] >= 0.0)
            & (uv[..., 0] <= 1.0)
            & (uv[..., 1] >= 0.0)
            & (uv[..., 1] <= 1.0)
        )
        mask = (depth > 1e-5) & in_bounds
        return uv, mask

    def forward(
        self,
        bev_query: torch.Tensor,
        value_list: Sequence[torch.Tensor],
        bev_pos: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        bev_h: int,
        bev_w: int,
        prev_bev: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
        projection=None,
        camera_params: Optional[torch.Tensor] = None,
        image_transform=None,
    ) -> torch.Tensor:
        B, N, C = bev_query.shape
        Z = float(self.pc_range[5] - self.pc_range[2])
        ref_3d = self.get_reference_points(
            bev_h,
            bev_w,
            Z,
            self.num_points_in_pillar,
            "3d",
            B,
            bev_query.device,
            bev_query.dtype,
        )
        ref_2d = self.get_reference_points(
            bev_h,
            bev_w,
            Z,
            self.num_points_in_pillar,
            "2d",
            B,
            bev_query.device,
            bev_query.dtype,
        )
        ref_3d_cam, bev_mask = self._project_ref_3d(
            ref_3d,
            projection=projection,
            camera_params=camera_params,
            image_transform=image_transform,
        )

        shift_ref_2d = ref_2d.clone()
        if shift is not None:
            shift_ref_2d = shift_ref_2d + shift[:, None, None, :]

        if prev_bev is not None:
            hybrid_ref_2d = torch.stack([shift_ref_2d, ref_2d], dim=1).reshape(
                B * 2, N, 1, 2
            )
            hybrid_value = torch.stack([prev_bev, bev_query], dim=1).reshape(
                B * 2, N, C
            )
        else:
            hybrid_ref_2d = torch.stack([ref_2d, ref_2d], dim=1).reshape(
                B * 2, N, 1, 2
            )
            hybrid_value = torch.stack([bev_query, bev_query], dim=1).reshape(
                B * 2, N, C
            )

        for layer in self.layers:
            bev_query = layer(
                bev_query,
                value_list,
                bev_pos,
                hybrid_ref_2d,
                ref_3d_cam,
                bev_mask,
                spatial_shapes,
                level_start_index,
                bev_h,
                bev_w,
                hybrid_value,
            )
        return bev_query


class BEVFormerImageEncoder(nn.Module):
    """Turn current ``Backbone`` outputs into BEVFormer multi-level camera tokens."""

    def __init__(
        self,
        backbone: Backbone,
        embed_dims: int,
        num_views: int,
        num_levels: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embed_dims = embed_dims
        self.num_views = num_views
        available = len(backbone.feature_channels)
        self.num_levels = (
            min(int(num_levels), available) if num_levels is not None else available
        )
        self.level_projs = nn.ModuleList(
            [
                nn.Conv2d(ch, embed_dims, 1)
                for ch in backbone.feature_channels[: self.num_levels]
            ]
        )
        self.level_embeds = nn.Parameter(
            torch.randn(len(self.level_projs), embed_dims) * 0.02
        )
        self.cam_embeds = nn.Parameter(torch.randn(num_views, embed_dims) * 0.02)

    def forward(self, camera_tiles: torch.Tensor):
        B, V, C, H, W = camera_tiles.shape
        features = self.backbone(camera_tiles.reshape(B * V, C, H, W))
        features = features[: self.num_levels]
        out = []
        spatial_shapes = []
        for lvl, (feat, proj) in enumerate(zip(features, self.level_projs)):
            feat = proj(feat)
            h, w = feat.shape[-2:]
            feat = feat.reshape(B, V, self.embed_dims, h, w)
            feat = feat + self.cam_embeds[None, :, :, None, None]
            feat = feat + self.level_embeds[None, None, lvl, :, None, None]
            out.append(feat)
            spatial_shapes.append((h, w))
        spatial_shapes_t = torch.tensor(
            spatial_shapes, dtype=torch.long, device=feat.device
        )
        level_start_index = torch.cat(
            [
                spatial_shapes_t.new_zeros((1,)),
                spatial_shapes_t.prod(1).cumsum(0)[:-1],
            ]
        )
        return out, spatial_shapes_t, level_start_index


class BEVFormerMapEncoder(nn.Module):
    """Small raster-map encoder whose channel outputs are visualizable."""

    def __init__(
        self,
        in_channels: int,
        embed_dims: int,
        bev_h: int,
        bev_w: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dims // 2, embed_dims, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, nav_input: torch.Tensor) -> torch.Tensor:
        x = self.net(nav_input)
        if x.shape[-2:] != (self.bev_h, self.bev_w):
            x = F.interpolate(
                x,
                size=(self.bev_h, self.bev_w),
                mode="bilinear",
                align_corners=False,
            )
        return x


class BEVFormerMapFusion(nn.Module):
    """Fuse camera BEV and raster map BEV without changing channel count."""

    def __init__(self, embed_dims: int) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dims * 2, embed_dims, 1),
            nn.GELU(),
            nn.Conv2d(embed_dims, embed_dims, 1),
        )

    def forward(self, image_bev: torch.Tensor, map_bev: torch.Tensor) -> torch.Tensor:
        return image_bev + self.fusion(torch.cat([image_bev, map_bev], dim=1))


class BEVFormerTrajectoryNet(nn.Module):
    """BEVFormer encoder + trajectory head for KITScenes.

    The public forward signature intentionally mirrors ``AutoE2E`` enough that
    the new training script can reuse the data and metric helpers from
    ``train_main.py`` without changing the existing model package.
    """

    def __init__(
        self,
        backbone: str = "swin_v2_tiny",
        is_pretrained: bool = True,
        num_views: int = 6,
        embed_dim: int = 256,
        bev_h: int = 64,
        bev_w: int = 64,
        pc_range: Sequence[float] = (-60.0, -60.0, -5.0, 120.0, 60.0, 3.0),
        num_points_in_pillar: int = 4,
        num_encoder_layers: int = 2,
        num_heads: int = 4,
        num_levels: int = 4,
        num_points: int = 8,
        dropout: float = 0.1,
        map_context_channels: int = 14,
        route_channels: int = 2,
        egomotion_dim: int = 256,
        visual_history_dim: int = 896,
        num_timesteps: int = 64,
        num_signals: int = 2,
        image_size: int = 256,
        use_temporal_state: bool = False,
    ) -> None:
        super().__init__()
        if num_points % num_points_in_pillar != 0:
            raise ValueError(
                "num_points must be divisible by num_points_in_pillar for SCA"
            )

        self.num_views = num_views
        self.embed_dim = embed_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.egomotion_dim = egomotion_dim
        self.visual_history_dim = visual_history_dim
        self.use_temporal_state = use_temporal_state
        self.image_transform = ImageTransform.square(image_size)
        self.pc_range = tuple(pc_range)

        self.backbone = Backbone(backbone=backbone, is_pretrained=is_pretrained)
        actual_num_levels = min(num_levels, len(self.backbone.feature_channels))
        if actual_num_levels <= 0:
            raise ValueError("backbone returned no feature levels")
        self.num_levels = actual_num_levels
        self.image_encoder = BEVFormerImageEncoder(
            self.backbone,
            embed_dims=embed_dim,
            num_views=num_views,
            num_levels=actual_num_levels,
        )
        self.bev_queries = nn.Embedding(bev_h * bev_w, embed_dim)
        nn.init.normal_(self.bev_queries.weight, mean=0.0, std=0.02)
        self.bev_pos = nn.Parameter(
            torch.randn(1, embed_dim, bev_h, bev_w) * 0.02
        )
        self.bev_encoder = PureBEVFormerEncoder(
            embed_dims=embed_dim,
            num_cams=num_views,
            bev_h=bev_h,
            bev_w=bev_w,
            pc_range=pc_range,
            num_points_in_pillar=num_points_in_pillar,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            num_levels=actual_num_levels,
            num_points=num_points,
            dropout=dropout,
        )

        nav_in = map_context_channels + route_channels
        self.map_encoder = BEVFormerMapEncoder(
            in_channels=nav_in,
            embed_dims=embed_dim,
            bev_h=bev_h,
            bev_w=bev_w,
        )
        self.map_fusion = BEVFormerMapFusion(embed_dim)
        self.planner = QueryPlanner(
            embed_dim=embed_dim,
            num_timesteps=num_timesteps,
            num_signals=num_signals,
            num_controls=5,
            num_heads=8,
            egomotion_dim=egomotion_dim,
            visual_history_dim=visual_history_dim,
            reasoning_mode="none",
        )
        self.prev_bev: Optional[torch.Tensor] = None
        self.temporal_buffer: dict[tuple[str, int], torch.Tensor] = {}

    def reset_temporal_state(self) -> None:
        self.prev_bev = None
        self.temporal_buffer.clear()

    @staticmethod
    def _normalize_egomotion(egomotion_history: torch.Tensor) -> torch.Tensor:
        ego = egomotion_history.clone()
        ego[..., 0::4] = ego[..., 0::4] / 33.0
        ego[..., 1::4] = ego[..., 1::4] / 8.0
        return ego

    def _estimate_shift(self, egomotion_history: torch.Tensor, B: int):
        if egomotion_history.ndim == 3:
            ego = egomotion_history[:, -1]
        else:
            ego = egomotion_history
        speed = ego[..., -4].clamp(min=0.0)
        delta_x = speed * 0.1
        delta_y = torch.zeros_like(delta_x)
        # Rough grid-length in metres.  The full BEV is (x_max-x_min, y_max-y_min).
        grid_x = (self.pc_range[3] - self.pc_range[0]) / self.bev_w
        grid_y = (self.pc_range[4] - self.pc_range[1]) / self.bev_h
        shift_x = delta_x / grid_x / self.bev_w
        shift_y = delta_y / grid_y / self.bev_h
        return torch.stack([shift_x, shift_y], dim=-1).to(egomotion_history.dtype)

    def forward(
        self,
        camera_tiles: torch.Tensor,
        map_context: torch.Tensor,
        visual_history: torch.Tensor,
        egomotion_history: torch.Tensor,
        route_mask: Optional[torch.Tensor] = None,
        map_valid=None,
        route_valid=None,
        projection=None,
        geometry_type: Optional[str] = None,
        image_transform=None,
        mode: str = "train",
        trajectory_target=None,
        temporal_key=None,
        **kwargs,
    ) -> torch.Tensor:
        B, V, C, H, W = camera_tiles.shape
        shift = self._estimate_shift(egomotion_history, B)
        ego = self._normalize_egomotion(egomotion_history)

        value_list, spatial_shapes, level_start_index = self.image_encoder(
            camera_tiles
        )
        bev_query = self.bev_queries.weight.unsqueeze(0).expand(B, -1, -1)
        bev_pos = self.bev_pos.expand(B, -1, -1, -1).flatten(2).permute(0, 2, 1)

        prev_bev = None
        if self.use_temporal_state:
            if temporal_key is not None:
                scene_id, frame_idx = temporal_key
                prev_bev = self.temporal_buffer.get((str(scene_id), int(frame_idx) - 1))
            else:
                prev_bev = self.prev_bev
            if prev_bev is not None and prev_bev.shape[0] != B:
                prev_bev = None

        bev = self.bev_encoder(
            bev_query,
            value_list,
            bev_pos,
            spatial_shapes,
            level_start_index,
            self.bev_h,
            self.bev_w,
            prev_bev=prev_bev,
            shift=shift,
            projection=projection,
            image_transform=image_transform or self.image_transform,
        )  # [B,N,C]
        if self.use_temporal_state:
            if temporal_key is not None and B == 1:
                scene_id, frame_idx = temporal_key
                self.temporal_buffer[(str(scene_id), int(frame_idx))] = bev.detach()
            else:
                self.prev_bev = bev.detach()

        image_bev = bev.reshape(B, self.embed_dim, self.bev_h, self.bev_w)

        if route_mask is None:
            route_mask = map_context.new_zeros(
                B, 2, map_context.shape[-2], map_context.shape[-1]
            )

        def gate(x, valid, default):
            if valid is None:
                valid = torch.full(
                    (B,), default, dtype=torch.bool, device=x.device
                )
            valid = torch.as_tensor(valid, device=x.device)
            if valid.shape not in ((B,), (B, 1)):
                raise ValueError("validity gate must be [B] or [B,1]")
            return x * valid.reshape(B, 1, 1, 1).to(x.dtype)

        map_bev = self.map_encoder(
            torch.cat(
                [gate(map_context, map_valid, True), gate(route_mask, route_valid, False)],
                dim=1,
            )
        )
        fused_bev = self.map_fusion(image_bev, map_bev)
        trajectory = self.planner(fused_bev, visual_history, ego)
        return trajectory
