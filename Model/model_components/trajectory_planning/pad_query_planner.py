"""Trajectory-query planner adapted from IPA-D/PAD for AutoE2E.

The planner follows PAD's proposal-query structure:

1. initialize a learned query for every proposal and future timestep;
2. decode a control trajectory from each query;
3. integrate the controls into ego-frame waypoints;
4. use those waypoints as reference points for deformable BEV sampling;
5. refine all proposal queries;
6. repeat the refinement for a fixed number of rounds;
7. score each proposal and select the best one.

The upstream implementations predict coarse ``(x, y, heading)`` poses and rely
on NavSim's simulator for score labels. AutoE2E instead needs the existing
``(acceleration, curvature)`` output contract, so each proposal here is a
control trajectory. Its integrated waypoints provide the proposal anchors.
Training scores use a continuous target derived from expert-trajectory error.

Source design:

* IPA-D/PAD: ``navsim/agents/pad/pad_model.py``,
  ``traj_refiner.py``, and ``score_module/scorer.py``.
* Drive-JEPA: the same proposal/refinement structure with additional
  NavSim-specific score calibration and pseudo-target losses.

An optional set of training-trajectory anchors can softly initialize the
proposal controls and query embeddings. The anchors are not frozen outputs:
every refinement predicts a residual update from the current BEV.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.losses.control_rollout import integrate_controls_torch

from .base import BasePlanner


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
    )


def _bev_grid_coordinates(
    xy_m: torch.Tensor,
    pc_range: Sequence[float],
) -> torch.Tensor:
    """Map ego-frame XY in metres to PyTorch ``grid_sample`` coordinates.

    AutoE2E's BEV rows run front-to-rear and columns left-to-right.
    ``grid_sample`` expects the final coordinate as ``(column, row)``.
    """
    x_min, y_min, _, x_max, y_max, _ = pc_range
    x = xy_m[..., 0]
    y = xy_m[..., 1]
    column = (y_max - y) / (y_max - y_min)
    row = (x_max - x) / (x_max - x_min)
    return torch.stack([2.0 * column - 1.0, 2.0 * row - 1.0], dim=-1)


class _ProposalRefinementLayer(nn.Module):
    """One shared PAD-style proposal decoding and BEV refinement round."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_points: int,
        num_heads: int,
        offset_scale: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_points = num_points
        self.offset_scale = offset_scale
        self.proposal_head = _mlp(embed_dim, embed_dim * 4, 2)
        self.sampling_offsets = nn.Linear(
            embed_dim,
            num_heads * num_points * 2,
        )
        self.attention_weights = nn.Linear(
            embed_dim,
            num_heads * num_points,
        )
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.bev_output_proj = nn.Linear(embed_dim, embed_dim)
        self.time_attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm_bev = nn.LayerNorm(embed_dim)
        self.norm_time = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        nn.init.normal_(self.proposal_head[-1].weight, std=1e-3)
        nn.init.normal_(self.proposal_head[-1].bias, std=1e-2)

    def _sample_bev(
        self,
        queries: torch.Tensor,
        bev_features: torch.Tensor,
        xy_m: torch.Tensor,
        pc_range: Sequence[float],
    ) -> torch.Tensor:
        batch_size, num_proposals, num_steps, channels = queries.shape
        num_queries = num_proposals * num_steps
        references = _bev_grid_coordinates(xy_m, pc_range).reshape(
            batch_size,
            num_queries,
            2,
        )
        flat_queries = queries.reshape(
            batch_size,
            num_queries,
            channels,
        )
        offsets = self.sampling_offsets(flat_queries).reshape(
            batch_size,
            num_queries,
            self.num_heads,
            self.num_points,
            2,
        )
        locations = references[:, :, None, None, :] + offsets * self.offset_scale

        values = (
            self.value_proj(bev_features.permute(0, 2, 3, 1))
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        values = values.reshape(
            batch_size,
            self.num_heads,
            self.head_dim,
            bev_features.shape[-2],
            bev_features.shape[-1],
        ).reshape(
            batch_size * self.num_heads,
            self.head_dim,
            bev_features.shape[-2],
            bev_features.shape[-1],
        )
        grid = locations.permute(0, 2, 1, 3, 4).reshape(
            batch_size * self.num_heads,
            num_queries,
            self.num_points,
            2,
        )
        sampled = F.grid_sample(
            values,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.reshape(
            batch_size,
            self.num_heads,
            self.head_dim,
            num_queries,
            self.num_points,
        ).permute(0, 3, 1, 4, 2)
        weights = (
            self.attention_weights(flat_queries)
            .reshape(
                batch_size,
                num_queries,
                self.num_heads,
                self.num_points,
            )
            .softmax(dim=-1)
        )
        attended = (sampled * weights.unsqueeze(-1)).sum(dim=-2)
        attended = attended.reshape(
            batch_size,
            num_queries,
            channels,
        )
        return self.bev_output_proj(attended).reshape_as(queries)

    def forward(
        self,
        queries: torch.Tensor,
        previous_controls: torch.Tensor,
        bev_features: torch.Tensor,
        initial_speed_mps: torch.Tensor,
        pc_range: Sequence[float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_proposals, num_steps, _ = queries.shape
        controls = previous_controls + self.proposal_head(queries)
        flat_controls = controls.reshape(
            batch_size * num_proposals,
            num_steps,
            2,
        )
        xy_m, _, _ = integrate_controls_torch(
            flat_controls,
            initial_speed_mps.repeat_interleave(num_proposals),
        )
        xy_m = xy_m.reshape(
            batch_size,
            num_proposals,
            num_steps,
            2,
        )

        attended = self._sample_bev(
            queries,
            bev_features,
            xy_m,
            pc_range,
        )
        queries = self.norm_bev(queries + attended)

        flat_queries = queries.reshape(
            batch_size * num_proposals,
            num_steps,
            self.embed_dim,
        )
        time_output, _ = self.time_attention(
            flat_queries,
            flat_queries,
            flat_queries,
            need_weights=False,
        )
        queries = self.norm_time(queries + time_output.reshape_as(queries))
        queries = self.norm_ffn(queries + self.ffn(queries))
        return queries, controls, xy_m


class PadQueryPlanner(BasePlanner):
    """PAD-derived multimodal control planner for AutoE2E.

    The return value intentionally remains the selected
    ``[B, num_timesteps * 2]`` control trajectory, so the existing runtime,
    rollout integration, and GRU baseline remain directly comparable.
    """

    def __init__(
        self,
        *,
        embed_dim: int = 256,
        num_timesteps: int = 64,
        num_signals: int = 2,
        num_proposals: int = 16,
        num_refinement_steps: int = 4,
        num_bev_points: int = 8,
        num_heads: int = 8,
        egomotion_dim: int = 256,
        visual_history_dim: int = 896,
        pc_range: Sequence[float] = (
            -60.0,
            -60.0,
            -5.0,
            120.0,
            60.0,
            3.0,
        ),
        offset_scale: float = 0.05,
        dropout: float = 0.1,
        score_temperature_m: float = 1.0,
        diversity_margin_m: float = 0.25,
        proposal_anchors: torch.Tensor | None = None,
        previous_refinement_weight: float = 0.1,
        diversity_weight: float = 0.1,
        proposal_loss_weight: float = 1.0,
        score_loss_weight: float = 1.0,
        reasoning_mode: str = "none",
        **_: object,
    ) -> None:
        super().__init__()
        if num_signals != 2:
            raise ValueError(
                "PAD query planner expects acceleration/curvature controls"
            )
        if (
            min(
                embed_dim,
                num_timesteps,
                num_proposals,
                num_refinement_steps,
                num_bev_points,
                num_heads,
            )
            <= 0
        ):
            raise ValueError("PAD query planner dimensions must be positive")
        if score_temperature_m <= 0.0:
            raise ValueError("score_temperature_m must be positive")
        if diversity_margin_m < 0.0:
            raise ValueError("diversity_margin_m must be non-negative")
        if reasoning_mode != "none":
            raise ValueError("PAD query planner currently supports no reasoning")
        self.embed_dim = int(embed_dim)
        self.num_timesteps = int(num_timesteps)
        self.num_signals = int(num_signals)
        self.num_proposals = int(num_proposals)
        self.num_refinement_steps = int(num_refinement_steps)
        self.egomotion_dim = int(egomotion_dim)
        self.visual_history_dim = int(visual_history_dim)
        self.pc_range = tuple(float(value) for value in pc_range)
        self.score_temperature_m = float(score_temperature_m)
        self.diversity_margin_m = float(diversity_margin_m)
        self.previous_refinement_weight = float(previous_refinement_weight)
        self.diversity_weight = float(diversity_weight)
        self.proposal_loss_weight = float(proposal_loss_weight)
        self.score_loss_weight = float(score_loss_weight)

        self.ego_state_proj = nn.Linear(egomotion_dim, embed_dim)
        self.visual_history_proj = nn.Linear(visual_history_dim, embed_dim)
        self.proposal_queries = nn.Embedding(
            num_proposals * num_timesteps,
            embed_dim,
        )
        self.anchor_encoder: nn.Linear | None
        if proposal_anchors is None:
            self.register_buffer("anchor_controls", None)
            self.anchor_encoder = None
        else:
            if tuple(proposal_anchors.shape) != (
                num_proposals,
                num_timesteps,
                num_signals,
            ):
                raise ValueError("proposal_anchors have the wrong shape")
            if not torch.isfinite(proposal_anchors).all():
                raise ValueError("proposal_anchors must be finite")
            self.register_buffer(
                "anchor_controls",
                proposal_anchors.to(dtype=torch.float32).clone(),
            )
            self.anchor_encoder = nn.Linear(num_signals, embed_dim)
        self.refinement = _ProposalRefinementLayer(
            embed_dim=embed_dim,
            num_points=num_bev_points,
            num_heads=num_heads,
            offset_scale=offset_scale,
            dropout=dropout,
        )
        if proposal_anchors is not None:
            nn.init.zeros_(self.refinement.proposal_head[-1].weight)
            nn.init.zeros_(self.refinement.proposal_head[-1].bias)
        self.score_head = _mlp(embed_dim, embed_dim * 2, 1)

        self.last_aux_loss: torch.Tensor | None = None
        self.last_proposal_loss: torch.Tensor | None = None
        self.last_score_loss: torch.Tensor | None = None
        self.last_diversity_loss: torch.Tensor | None = None
        self.last_proposal_controls: torch.Tensor | None = None
        self.last_proposal_xy: torch.Tensor | None = None
        self.last_proposal_scores: torch.Tensor | None = None
        self.last_selected_proposal: torch.Tensor | None = None

    def _proposal_loss(
        self,
        proposal_xy: Sequence[torch.Tensor],
        target_xy: torch.Tensor,
        target_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = target_valid[:, None, :, None].to(
            device=target_xy.device,
            dtype=target_xy.dtype,
        )
        denominator = valid.sum(dim=(2, 3)).clamp_min(1.0)
        total = target_xy.new_zeros(())
        final_diversity = target_xy.new_zeros(())
        for index, xy in enumerate(proposal_xy):
            expanded_target = target_xy[:, None].expand_as(xy)
            elementwise = F.smooth_l1_loss(
                xy,
                expanded_target,
                reduction="none",
                beta=1.0,
            )
            per_proposal = (elementwise * valid).sum(dim=(2, 3)) / denominator
            minimum = per_proposal.min(dim=1).values.mean()

            flat = xy.reshape(xy.shape[0], xy.shape[1], -1)
            distances = torch.cdist(flat, flat, p=1) / (self.num_timesteps * 2)
            distances = (
                distances
                + torch.eye(
                    self.num_proposals,
                    device=xy.device,
                    dtype=xy.dtype,
                )[None]
                * 1e6
            )
            # Upstream PAD maximizes proposal separation with an unbounded
            # negative term. Use a non-negative margin penalty so diversity is
            # encouraged only until proposals are sufficiently distinct.
            diversity = F.relu(
                self.diversity_margin_m - distances.min(dim=2).values
            ).mean()
            total = (
                self.previous_refinement_weight * total
                + minimum
                + self.diversity_weight * diversity
            )
            if index == len(proposal_xy) - 1:
                final_diversity = diversity
        return total, final_diversity

    def _score_loss(
        self,
        scores: torch.Tensor,
        proposal_xy: torch.Tensor,
        target_xy: torch.Tensor,
        target_valid: torch.Tensor,
    ) -> torch.Tensor:
        error = torch.linalg.vector_norm(
            proposal_xy - target_xy[:, None],
            dim=-1,
        )
        mask = target_valid[:, None, :].to(dtype=error.dtype)
        proposal_error = (error * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
        quality = torch.exp(-proposal_error / self.score_temperature_m).clamp(
            1e-4, 1.0 - 1e-4
        )
        return F.binary_cross_entropy_with_logits(
            scores.float(),
            quality.detach().float(),
        )

    def forward(
        self,
        bev_features: torch.Tensor,
        visual_history: torch.Tensor,
        egomotion_history: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        initial_speed = kwargs.get("initial_speed_mps")
        if not torch.is_tensor(initial_speed):
            raise ValueError("PAD query planner requires initial_speed_mps")
        if bev_features.ndim != 4 or bev_features.shape[1] != self.embed_dim:
            raise ValueError("PAD query planner expects camera BEV [B,C,H,W]")
        batch_size = bev_features.shape[0]
        context = self.ego_state_proj(egomotion_history) + self.visual_history_proj(
            visual_history
        )
        queries = context[:, None, None, :] + self.proposal_queries.weight.reshape(
            1,
            self.num_proposals,
            self.num_timesteps,
            self.embed_dim,
        )
        previous_controls = torch.zeros(
            batch_size,
            self.num_proposals,
            self.num_timesteps,
            self.num_signals,
            device=bev_features.device,
            dtype=bev_features.dtype,
        )
        if self.anchor_controls is not None:
            anchors = self.anchor_controls.to(
                device=bev_features.device,
                dtype=bev_features.dtype,
            )
            previous_controls = anchors[None].expand(
                batch_size,
                -1,
                -1,
                -1,
            )
            if self.anchor_encoder is None:
                raise RuntimeError("proposal anchors require an encoder")
            queries = queries + self.anchor_encoder(anchors)[None]
        initial_speed = initial_speed.to(
            device=bev_features.device,
            dtype=bev_features.dtype,
        )

        proposal_controls: list[torch.Tensor] = []
        proposal_xy: list[torch.Tensor] = []
        for _ in range(self.num_refinement_steps):
            queries, controls, xy_m = self.refinement(
                queries,
                previous_controls,
                bev_features,
                initial_speed,
                self.pc_range,
            )
            proposal_controls.append(controls)
            proposal_xy.append(xy_m)
            previous_controls = controls

        final_controls = proposal_controls[-1]
        final_xy = proposal_xy[-1]
        proposal_features = queries.amax(dim=2)
        scores = self.score_head(proposal_features).squeeze(-1)
        selected = scores.argmax(dim=1)

        target_xy = kwargs.get("trajectory_xy_m")
        target_valid = kwargs.get("trajectory_valid")
        self.last_aux_loss = None
        self.last_proposal_loss = None
        self.last_score_loss = None
        self.last_diversity_loss = None
        if torch.is_tensor(target_xy) and torch.is_tensor(target_valid):
            target_xy = target_xy.to(
                device=bev_features.device,
                dtype=final_xy.dtype,
            )
            target_valid = target_valid.to(
                device=bev_features.device,
                dtype=torch.bool,
            )
            proposal_loss, diversity_loss = self._proposal_loss(
                proposal_xy,
                target_xy,
                target_valid,
            )
            score_loss = self._score_loss(
                scores,
                final_xy,
                target_xy,
                target_valid,
            )
            self.last_proposal_loss = proposal_loss
            self.last_score_loss = score_loss
            self.last_diversity_loss = diversity_loss
            self.last_aux_loss = (
                self.proposal_loss_weight * proposal_loss
                + self.score_loss_weight * score_loss
            )

        rows = torch.arange(batch_size, device=bev_features.device)
        selected_controls = final_controls[rows, selected]
        self.last_proposal_controls = final_controls.detach()
        self.last_proposal_xy = final_xy.detach()
        self.last_proposal_scores = scores.detach()
        self.last_selected_proposal = selected.detach()
        return selected_controls.reshape(
            batch_size,
            self.num_timesteps * self.num_signals,
        )
