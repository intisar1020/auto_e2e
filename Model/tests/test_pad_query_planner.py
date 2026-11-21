"""Contract tests for the IPA-D/PAD-derived trajectory query planner."""

from __future__ import annotations

import torch

from model_components.trajectory_planning import (
    PLANNER_REGISTRY,
    PadQueryPlanner,
    build_planner,
)
from training.losses.control_rollout import integrate_controls_torch


def _planner(**kwargs) -> PadQueryPlanner:
    defaults = {
        "embed_dim": 32,
        "num_timesteps": 8,
        "num_proposals": 4,
        "num_refinement_steps": 3,
        "num_bev_points": 4,
        "num_heads": 4,
        "egomotion_dim": 24,
        "visual_history_dim": 40,
        "dropout": 0.0,
        "pc_range": (-32.0, -32.0, -5.0, 32.0, 32.0, 3.0),
    }
    defaults.update(kwargs)
    return PadQueryPlanner(**defaults)


def _inputs(
    *,
    batch_size: int = 2,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(
            batch_size,
            32,
            16,
            12,
            requires_grad=requires_grad,
        ),
        torch.randn(batch_size, 40),
        torch.randn(batch_size, 24),
        torch.full((batch_size,), 8.0),
    )


def _target_xy(
    plan: PadQueryPlanner,
    *,
    batch_size: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    controls = torch.zeros(batch_size, plan.num_timesteps, 2)
    controls[..., 1] = 0.01
    xy, _, _ = integrate_controls_torch(
        controls,
        torch.full((batch_size,), 8.0),
    )
    valid = torch.ones(batch_size, plan.num_timesteps, dtype=torch.bool)
    return xy, valid


def test_query_planner_is_registered() -> None:
    assert PLANNER_REGISTRY["query"] is PadQueryPlanner
    planner = build_planner(
        "query",
        embed_dim=32,
        num_timesteps=8,
        num_proposals=4,
        num_refinement_steps=3,
        num_bev_points=4,
        num_heads=4,
        egomotion_dim=24,
        visual_history_dim=40,
        dropout=0.0,
    )
    assert isinstance(planner, PadQueryPlanner)


def test_inference_selects_one_finite_proposal() -> None:
    torch.manual_seed(3)
    planner = _planner().eval()
    bev, visual, ego, speed = _inputs()
    with torch.no_grad():
        output = planner(
            bev,
            visual,
            ego,
            initial_speed_mps=speed,
        )
    assert output.shape == (2, 16)
    assert torch.isfinite(output).all()
    assert planner.last_proposal_controls is not None
    assert planner.last_proposal_controls.shape == (2, 4, 8, 2)
    assert planner.last_proposal_scores is not None
    assert planner.last_selected_proposal is not None
    assert torch.equal(
        planner.last_selected_proposal,
        planner.last_proposal_scores.argmax(dim=1),
    )
    assert planner.last_aux_loss is None


def test_training_losses_and_gradients_reach_bev() -> None:
    torch.manual_seed(7)
    planner = _planner().train()
    bev, visual, ego, speed = _inputs(requires_grad=True)
    target_xy, target_valid = _target_xy(planner)
    output = planner(
        bev,
        visual,
        ego,
        initial_speed_mps=speed,
        trajectory_xy_m=target_xy,
        trajectory_valid=target_valid,
    )
    assert planner.last_aux_loss is not None
    assert planner.last_proposal_loss is not None
    assert planner.last_score_loss is not None
    assert planner.last_diversity_loss is not None
    assert torch.isfinite(planner.last_aux_loss)
    loss = output.square().mean() + planner.last_aux_loss
    loss.backward()
    assert bev.grad is not None and bev.grad.abs().sum() > 0
    planner_grads = [
        parameter.grad
        for parameter in planner.parameters()
        if parameter.grad is not None
    ]
    assert planner_grads
    assert any(gradient.abs().sum() > 0 for gradient in planner_grads)


def test_repeated_depth_refines_each_round() -> None:
    torch.manual_seed(11)
    planner = _planner(num_refinement_steps=4).eval()
    bev, visual, ego, speed = _inputs()
    with torch.no_grad():
        output = planner(
            bev,
            visual,
            ego,
            initial_speed_mps=speed,
        )
    assert output.shape == (2, 16)
    assert planner.last_proposal_xy is not None
    assert planner.last_proposal_xy.shape == (2, 4, 8, 2)
    assert torch.isfinite(planner.last_proposal_xy).all()
