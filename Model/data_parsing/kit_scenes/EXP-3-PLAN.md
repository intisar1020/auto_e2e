# exp-3-route-consistency — Plan

Experiment to add the upstream `RouteConsistencyLoss` as an auxiliary training
term on the **current shared-encoder architecture**, and compare against the
exp-2-baseline (6-camera, imitation-only).

## Context

| | exp-2-baseline | exp-3-route-consistency |
|---|---|---|
| Architecture | current upstream (6-cam, shared NavigationEncoder, deformable fusion) | **same** (no arch change) |
| Camera | 6 | 6 |
| Data / split | splits.json (50 train tars / 1580 samples, 5 val / 145) | **same** |
| Init | exp-1 best (warm) | **exp-2-baseline best (warm)** |
| Loss | TrajectoryImitationLoss only | TrajectoryImitationLoss **+ RouteConsistencyLoss** (weighted) |
| Epochs | 30 (best 6.4s ADE 4.71, epoch 7) | 30 |

Goal: measure the delta from the route-consistency term alone (clean A/B,
architecture held fixed). Expectation: corridor compliance improves / off-route
deviation drops; ADE/FDE should not regress meaningfully.

## What RouteConsistencyLoss does

- Integrates predicted controls -> predicted positions (differentiable rollout).
- Samples supervision rasters at predicted positions (grid_sample):
  - `distance_to_corridor_m` (stay on selected route)
  - route heading (follow road direction)
  - destination (end near goal, relative to GT terminal distance)
- Weighted terms: corridor 1.0, branch 2.0 (2nd half, at intersections),
  destination 0.5, heading 0.25.
- Eligibility gating: only samples with GT corridor compliance >= 0.90 and
  valid route/quality are supervised (won't fight the imitation loss on
  off-route GT).
- Full loss: `imitation + route_consistency_weight * route_terms["total"]`.

Note: this is the loss already merged into our tree (from upstream main). It is
NOT the #186 route reconstruction loss (that PR is still open and not merged).

## Wiring changes (train_main.py)

1. Import `decode_route_supervision` and `RouteConsistencyLoss`.
2. New args (defaults keep exp-2-baseline behavior identical):
   - `--route-consistency-weight` float, default `0.10` (0 = off)
   - `--exp-name` (already added; default `exp-3-route-consistency`)
3. Per-sample supervision batch from `navigation_members`:
   - `sup = decode_route_supervision(members)`
   - `mc, rm, meta = decode_sample_navigation(members)` -> route_valid,
     route_intersection
   - build tensors `[B,H,W]` / `[B,2]` / `[B]` matching `pre_extracted.py`
     schema (distance_to_corridor_m, route_heading_sin/cos/valid,
     destination_xy_m, destination_visible, available)
4. In train loop only (not validation):
   ```
   route_terms = route_lfn(out, tg, v0_tensor, sup_dict, route_valid, route_intersection)
   loss = imitation_loss + weight * route_terms["total"]
   ```
   - `initial_speed` from existing `_extract_v0`
   - `out` is [1,128] flattened controls -> integrate_controls_torch handles it
5. Track per-epoch: corridor / branch / destination / heading means +
   `eligible_count` / `candidate_count` (must confirm the loss actually engages,
   not zero-gated).

## Run

- `python train_main.py --epochs 30 --exp-name exp-3-route-consistency \
     --init exp-2-baseline/checkpoints/best.pt`
- Outputs -> `exp-3-route-consistency/{checkpoints,history.json,trajectories}`
- ~1.1 s/iter -> ~28 min/epoch -> ~14 h for 30 epochs (same as baseline).

## Validation / comparison

- Same 3s + 6.4s ADE/FDE on the 145-sample val.
- Table vs exp-2-baseline (best 6.4s ADE 4.71, 3s ADE 1.16).
- Confirm route terms are active (eligible_count > 0) and corridor term
  decreases over training.

## Deferred (next steps)

- Route-swap / junction-branch counterfactual eval (reuse
  `route_swap_sample_metrics`).
- Route-consistency weight sweep (0.05 / 0.10 / 0.20).
- Step 2: separate Map/Route encoders + learned route gate (+ possibly #186
  reconstruction head once that PR lands on main).

## Files touched

- `Model/data_parsing/kit_scenes/train_main.py` (only)
