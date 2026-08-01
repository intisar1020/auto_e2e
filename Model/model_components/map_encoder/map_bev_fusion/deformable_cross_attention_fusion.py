"""Deformable cross-attention for map BEV fusion.

Each spatial query attends to K learned-offset sample points in the map BEV.
Uses F.grid_sample for sampling (no custom CUDA needed). O(N * K) memory
instead of O(N^2). At K=4, memory is ~0.5 GB for 135K tokens.

Compatible with the MapBEVFusion interface: takes (image_bev, map_bev) and
returns fused_image_bev.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MapDeformableCrossAttentionFusion(nn.Module):
    """Fuse image BEV and map BEV via deformable spatial cross-attention.

    Instead of dense Q@K^T attention over all 135K spatial tokens, each query
    pixel attends to only K learned-offset sample points in the map BEV.
    Sampling uses ``F.grid_sample`` (bilinear interpolation) — no custom
    CUDA ops needed.

    Args:
        embed_dim: Channel dimension (default 256).
        num_sample_points: K — number of offset points per query (default 4).
        num_heads: Attention heads for computing per-point attention weights.
        dropout: Dropout in FFN only.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_sample_points: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_points = num_sample_points
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Predict K 2D-offsests from the query feature
        self.offset_proj = nn.Linear(embed_dim, num_sample_points * 2)

        # Predict per-head attention weights for the K sample points
        self.attn_proj = nn.Linear(embed_dim, num_heads * num_sample_points)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Pre-normalization
        self.norm_query = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(
        self,
        image_bev: torch.Tensor,   # (B, C, H, W)
        map_bev: torch.Tensor,     # (B, C, H, W)
    ) -> torch.Tensor:
        B, C, H, W = image_bev.shape
        N = H * W
        K = self.num_points
        nH = self.num_heads

        # ---- Prepare reference grid (regular 2D grid in [-1, 1]) ----
        ys = torch.linspace(-1, 1, H, device=image_bev.device, dtype=image_bev.dtype)
        xs = torch.linspace(-1, 1, W, device=image_bev.device, dtype=image_bev.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        ref_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        ref_grid = ref_grid.reshape(1, H, W, 2)  # (1, H, W, 2)

        # ---- Flatten spatial dims ----
        q = image_bev.permute(0, 2, 3, 1).reshape(B, N, C)  # (B, N, C)
        map_flat = map_bev.permute(0, 2, 3, 1).reshape(B, N, C)

        # Normalize
        q_norm = self.norm_query(q)
        kv_norm = self.norm_kv(map_flat)

        # ---- Predict offsets and attention weights ----
        offsets = self.offset_proj(q_norm)  # (B, N, 2K)
        attn_logits = self.attn_proj(q_norm)  # (B, N, nH*K)

        # Offsets are learned displacements, normalized by image size
        offsets = offsets.reshape(B, N, K, 2)
        offsets[:, :, :, 0] = offsets[:, :, :, 0] / (W - 1) * 2.0  # scale x to grid coords
        offsets[:, :, :, 1] = offsets[:, :, :, 1] / (H - 1) * 2.0  # scale y to grid coords

        # Sampling positions = reference grid + learned offsets
        ref = ref_grid.reshape(1, H, W, 2)  # (1, H, W, 2)
        ref = ref.reshape(1, N, 1, 2)  # (1, N, 1, 2)
        sample_pos = ref + offsets.unsqueeze(1).mean(dim=1, keepdim=True)  # (1 or B, N, K, 2)
        # Actually need: (B, N, K, 2)
        # offsets is (B, N, K, 2), ref is (1, N, 1, 2)
        sample_pos = offsets + ref.expand(B, -1, -1, -1)  # (B, N, K, 2)

        # ---- Sample map features at offset positions ----
        # grid_sample expects (B, C, H_out, W_out) input and (B, H_out, W_out, 2) grid
        # We need to sample K positions per query: reshape to treat each sample as a batch
        sample_pos_grid = sample_pos.reshape(B * N, K, 1, 2)  # (B*N, K, 1, 2) for grid_sample
        # Broadcast map BEV to (B*N, C, H, W)
        map_expanded = map_bev.unsqueeze(2).expand(-1, -1, N, -1, -1)  # (B, C, N, H, W)
        # Reshape: treat each of the N queries as an independent sample
        # Simpler: use grid_sample once per batch with a (B, N*K, 2) grid
        sample_pos_2d = sample_pos.reshape(B, N * K, 2)  # (B, N*K, 2)
        # Reshape to (B, 1, N*K, 2) — grid_sample needs (B, H_out, W_out, 2)
        sample_grid = sample_pos_2d.reshape(B, N, K, 2)  # (B, N, K, 2)

        # grid_sample: for each (B, H_out, W_out) we sample from map_bev
        # We want: for each query i, sample K points → output (B, C, N, K)
        # grid_sample with (B, H_out, W_out, 2) where H_out=N, W_out=K
        sampled = F.grid_sample(
            map_bev, sample_grid, mode="bilinear", padding_mode="border", align_corners=True,
        )  # (B, C, N, K)
        sampled = sampled.permute(0, 2, 3, 1)  # (B, N, K, C)

        # ---- Multi-head attention weights ----
        attn_logits = attn_logits.reshape(B, N, nH, K)  # (B, N, nH, K)
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, N, nH, K)

        # Split head dim of sampled features
        sampled_heads = sampled.reshape(B, N, K, nH, self.head_dim).permute(0, 1, 3, 2, 4)  # (B, N, nH, K, d)

        # Weighted sum over sample points
        attn_out = (sampled_heads * attn_weights.unsqueeze(-1)).sum(dim=3)  # (B, N, nH, d)
        attn_out = attn_out.reshape(B, N, C)  # (B, N, C)

        # Output projection + residual
        q = q + self.out_proj(attn_out)

        # FFN
        q = q + self.ffn(self.norm_ffn(q))

        return q.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
