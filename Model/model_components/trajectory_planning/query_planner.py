"""Query-based Spatial Cross-Attention Trajectory Planner.

Unlike legacy vector planners that crush the BEV feature map into 1 channel
and flatten it to 1D, QueryPlanner preserves all 256 feature channels and
uses learned trajectory queries with spatial cross-attention to attend over
the 2D BEV feature map.

Trajectory output is expanded through a fixed Bernstein polynomial basis
over K control points, ensuring smooth (acceleration, curvature) profiles
by construction with zero jerk.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from .base import BasePlanner
from .reasoning_coupling import ReasoningCoupling


class QueryPlanner(BasePlanner):
    """Query-based Spatial Cross-Attention Trajectory Planner.

    Args:
        embed_dim: Channel dimension of BEV feature map and query tokens (default 256).
        num_timesteps: Target trajectory rollout steps (default 64 @ 10 Hz = 6.4 s).
        num_signals: Number of output control signals per step (default 2: accel, curvature).
        num_controls: Number of Bernstein control points / query tokens (default 5).
        num_heads: Attention heads for multi-head spatial cross-attention (default 8).
        egomotion_dim: Dimension of egomotion history vector (default 256).
        visual_history_dim: Dimension of visual history context (default 896).
        reasoning_mode: Zero-init reasoning coupling mode ("none", "pooled_latent", etc.).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_timesteps: int = 64,
        num_signals: int = 2,
        num_controls: int = 5,
        num_heads: int = 8,
        egomotion_dim: int = 256,
        visual_history_dim: int = 896,
        reasoning_mode: str = "none",
        **kwargs,
    ):
        super().__init__()

        if num_controls < 2:
            raise ValueError(f"num_controls must be >= 2, got {num_controls}")
        if num_controls > num_timesteps:
            raise ValueError(
                f"num_controls ({num_controls}) cannot exceed num_timesteps ({num_timesteps})"
            )

        self.num_timesteps = num_timesteps
        self.num_signals = num_signals
        self.num_controls = num_controls
        self.embed_dim = embed_dim
        self.egomotion_dim = egomotion_dim
        self.visual_history_dim = visual_history_dim

        # 1. Learned Trajectory Query Embeddings (K control point queries)
        self.query_embed = nn.Parameter(torch.randn(num_controls, embed_dim) * 0.02)

        # 2. Context Projection (Egomotion + Visual History)
        self.visual_history_proj = nn.Linear(visual_history_dim, visual_history_dim)
        nn.init.zeros_(self.visual_history_proj.weight)
        nn.init.zeros_(self.visual_history_proj.bias)

        self.context_proj = nn.Sequential(
            nn.Linear(egomotion_dim + visual_history_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 3. Spatial Cross-Attention over 2D BEV features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.norm_attn = nn.LayerNorm(embed_dim)

        # 4. Feed-Forward Processing Network
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # 5. Reasoning Coupling
        self.reasoning_coupling = ReasoningCoupling(embed_dim, mode=reasoning_mode)

        # 6. Control Point Output Head
        self.control_head = nn.Linear(embed_dim, num_signals)

        # 7. Bernstein Polynomial Basis [num_timesteps, num_controls]
        self.register_buffer(
            "bernstein_basis",
            self._bernstein_basis(num_timesteps, num_controls),
            persistent=False,
        )

    @staticmethod
    def _bernstein_basis(num_points: int, num_controls: int) -> torch.Tensor:
        """Evaluate Bernstein basis polynomials B_{i,n}(t) on a uniform grid t in [0, 1]."""
        n = num_controls - 1
        t = torch.linspace(0.0, 1.0, num_points).unsqueeze(1)  # [P, 1]
        i = torch.arange(num_controls, dtype=torch.float32).unsqueeze(0)  # [1, C]
        comb = torch.tensor(
            [math.comb(n, k) for k in range(num_controls)], dtype=torch.float32
        ).unsqueeze(0)  # [1, C]
        basis = comb * (t ** i) * ((1.0 - t) ** (n - i))  # [P, C]
        return basis

    def forward(
        self,
        bev_features: torch.Tensor,
        visual_history: torch.Tensor,
        egomotion_history: torch.Tensor,
        reasoning_latent: torch.Tensor | None = None,
        reasoning_horizon_tokens: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Decode control trajectory from BEV feature maps and ego/visual history context.

        Args:
            bev_features: [B, C, H, W] or [B, N, C] spatial BEV feature map.
            visual_history: [B, visual_history_dim] visual history embedding.
            egomotion_history: [B, egomotion_dim] egomotion state history.
            reasoning_latent: Optional [B, embed_dim] reasoning latent.
            reasoning_horizon_tokens: Optional reasoning tokens.

        Returns:
            trajectory: [B, num_timesteps * num_signals] flattened control trajectory.
        """
        B = bev_features.shape[0]

        # 1. Format spatial BEV features for cross-attention -> [B, N_spatial, C]
        if bev_features.ndim == 4:
            spatial_feats = bev_features.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        elif bev_features.ndim == 3:
            spatial_feats = bev_features  # [B, N, C]
        else:
            # Flattened 1D legacy vector [B, D] -> reshape to pseudo spatial tokens [B, D//C, C]
            spatial_feats = bev_features.view(B, -1, self.embed_dim)

        # 2. Context conditioning
        vis_proj = self.visual_history_proj(visual_history)
        ego_vis = torch.cat([egomotion_history, vis_proj], dim=-1)
        ctx_vec = self.context_proj(ego_vis).unsqueeze(1)  # [B, 1, C]

        # 3. Expand learned control queries and condition with ego/visual context
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1) + ctx_vec  # [B, K, C]

        # 4. Inject reasoning branch coupling (no-op when reasoning is off)
        queries = self.reasoning_coupling(
            queries,
            reasoning_latent=reasoning_latent,
            horizon_tokens=reasoning_horizon_tokens,
        )

        # 5. Multi-Head Spatial Cross-Attention over 2D BEV features
        attn_out, _ = self.cross_attn(
            query=queries, key=spatial_feats, value=spatial_feats
        )
        queries = self.norm_attn(queries + attn_out)

        # 6. FFN Residual
        queries = queries + self.ffn(self.norm_ffn(queries))  # [B, K, C]

        # 7. Predict Bernstein control points: [B, K, num_signals]
        control_points = self.control_head(queries)

        # 8. Bernstein basis expansion: [P, K] x [B, K, S] -> [B, P, S]
        trajectory = torch.einsum("pc,bcs->bps", self.bernstein_basis, control_points)

        # 9. Flatten to [B, P * S]
        return trajectory.reshape(B, self.num_timesteps * self.num_signals)
