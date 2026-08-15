"""Main local training loop for the reactive AutoE2E branch on KITScenes.

This is the canonical research training script for this project. It trains the
reactive (single-frame, no world-model) AutoE2E to predict a 6.4 s future
trajectory (acceleration + curvature at 10 Hz) from 6 camera views + the
16-channel navigation input, using fully-local extracted scene data. There is
no streaming and no cloud dependency; the whole dataset sits under
``datasets/``.

Key design points
-----------------
* **Data source**: ``datasets/`` holds one directory per extracted KITScenes
  scene. The pinned SDK is loaded with ``split="train"``, which reads every
  scene dir under ``datasets/train/``; the role of each scene (train vs val)
  is NOT encoded on disk but assigned by the canonical ``splits.json`` IDs
  (50 train / 5 val tars -> 1580 / 145 samples).

* **Model contract**: 6-camera visual input (``NUM_VIEWS`` from ``camera.py``),
  14-channel map_context + 2-channel route_mask (all 16 channels live because
  ``route_valid=True``), deformable map-BEV fusion, imitation trajectory loss.
  See ``model_components/auto_e2e.py`` for the network itself.

* **Training setup**: batch size 1 with gradient accumulation (default 4),
  fp16 autocast (RTX 3060 12 GB), AdamW + cosine LR schedule, photometric
  augmentation (no geometric transforms -- those would break the calibrated
  camera projection), gradient clipping, and per-epoch validation on the fixed
  val split. The best checkpoint is selected by 6.4 s ADE.

* **Route-consistency loss (optional)**: with ``--route-consistency-weight``
  (default 0.10; 0 disables), the training objective is the weighted sum of the
  imitation loss and the upstream ``RouteConsistencyLoss``, which integrates the
  predicted controls into a rollout and penalizes going off the selected route
  corridor / road heading / destination (using the loss-only supervision rasters
  packed in each sample's ``route_supervision.npz``). This is the exp-3
  experiment objective; set it to 0 to reproduce the exp-2-baseline exactly.

* **Metrics**: ADE/FDE are reported at both the 3 s and full 6.4 s horizons
  (trajectory is 64 steps @ 10 Hz). Validation runs in ``mode="infer"`` with a
  real per-sample initial speed extracted from the egomotion history.

* **Artifacts**: every run writes into ``exp-<name>/`` (default
  ``exp-3-route-consistency``) containing ``checkpoints/`` (best.pt, latest.pt,
  curve.png), ``history.json`` (per-epoch val + train metrics), and
  ``trajectories/`` (GT-vs-prediction plots every ``PLOT_EVERY`` iterations).

* **Warm start**: by default the model is initialized from
  ``exp-2-baseline/checkpoints/best.pt`` (the 30-epoch imitation-only baseline,
  best 6.4 s ADE 4.71). Use ``--no-init`` to train from scratch, or ``--init``
  to point at a different checkpoint. Loading is ``strict=False`` so a
  checkpoint saved under a slightly different architecture transfers as much
  as possible and logs any dropped/extra keys.

Usage:
  cd Model/data_parsing/kit_scenes
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_main.py --epochs 30

  # custom experiment name + a different init checkpoint
  python train_main.py --epochs 30 --exp-name exp-3-route-consistency \\
      --init exp-2-baseline/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_MODEL_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_MODEL_DIR))

from data_parsing.kit_scenes import KitScenesDataset, NUM_VIEWS
from evaluation.metrics import integrate_trajectory
from model_components.auto_e2e import AutoE2E
from model_components.losses.trajectory_loss import TrajectoryImitationLoss
from model_components.view_fusion.projection import PinholeProjection
from navigation.artifacts import decode_route_supervision, decode_sample_navigation
from training.losses import RouteConsistencyLoss

KIT_DIR = Path(__file__).parent
DATA_ROOT = KIT_DIR / "datasets"
SPLITS = KIT_DIR / "splits.json"
DEFAULT_EXP = "exp-3-route-consistency"
PLOT_EVERY = 200


def _load_split():
    """Load the canonical train/val scene IDs from ``splits.json``.

    Returns:
        (train_ids, val_ids): lists of KITScenes scene IDs. The split is the
        fixed ~175 GB pool shared by every experiment; it is NOT re-derived
        per run so results across experiments are comparable.
    """
    s = json.loads(SPLITS.read_text())
    return s["train_ids"], s["val_ids"]


def _build_sets():
    """Build train/val ``KitScenesDataset`` instances for this run.

    Both datasets are constructed with ``split="train"`` (the SDK layout has
    every extracted scene under ``datasets/train/``) and ``scene_ids`` pinned
    to the canonical split, so scene roles come from ``splits.json`` rather
    than from disk layout. ``include_navigation=True`` makes each sample carry
    the 14-channel map + 2-channel route + route-supervision members.

    Returns:
        (train_ds, val_ds, train_ids, val_ids).
    """
    train_ids, val_ids = _load_split()
    train_ds = KitScenesDataset(data_root=str(DATA_ROOT), split="train",
                                include_navigation=True, scene_ids=train_ids)
    val_ds = KitScenesDataset(data_root=str(DATA_ROOT), split="train",
                              include_navigation=True, scene_ids=val_ids)
    return train_ds, val_ds, train_ids, val_ids


def _tensors(s, device):
    """Convert one dataset sample into the model input tensors.

    Args:
        s: A ``KitScenesSample`` from ``KitScenesDataset``.
        device: Target torch device.

    Returns:
        (v, eg, vh, tg, cp, mc, rm, mv, rv):
            v  - (1, 6, 3, 256, 256) camera tiles normalized to [0,1]
            eg - (1, 256) egomotion history (64 steps x 4 signals)
            vh - (1, 896) visual history placeholder (zeros; reactive mode)
            tg - (1, 128) trajectory target (64 steps x 2 signals)
            cp - (1, 6, 3, 4) real camera projection matrices
            mc - (1, 14, 256, 256) map_context semantic raster
            rm - (1, 2, 256, 256) route_mask raster
            mv - (1,) map_valid = True
            rv - (1,) route_valid = True (route channel ENABLED)

    The visual history is a zero placeholder because the reactive model is
    memory-free: the past is carried only by the compact egomotion vector, not
    by past image frames. Both navigation gates are on, so all 16 input
    channels are fed to the shared NavigationEncoder.
    """
    v = s["visual_tiles"].unsqueeze(0).to(device).float() / 255.0
    eg = s["egomotion_history"].unsqueeze(0).to(device).float()
    vh = s["visual_history"].unsqueeze(0).to(device).float()
    tg = s["trajectory_target"].unsqueeze(0).to(device).float()
    cp = s["camera_params"].unsqueeze(0).to(device).float()

    if "navigation_members" in s:
        mc, rm, _ = decode_sample_navigation(s["navigation_members"])
    else:
        # Navigation-less samples (should not happen for KITScenes): feed blank
        # rasters so the batch shapes stay consistent.
        mc = np.zeros((14, 256, 256), dtype=np.float32)
        rm = np.zeros((2, 256, 256), dtype=np.float32)

    mc = torch.from_numpy(mc.copy()).float().unsqueeze(0).to(device)
    rm = torch.from_numpy(rm.copy()).float().unsqueeze(0).to(device)
    # Both gates ON: map + route (all 16 channels live).
    mv = torch.ones(1, dtype=torch.bool, device=device)
    rv = torch.ones(1, dtype=torch.bool, device=device)
    return v, eg, vh, tg, cp, mc, rm, mv, rv


def _route_supervision_batch(s, device):
    """Build the route-supervision batch dict consumed by RouteConsistencyLoss.

    Decodes ``route_supervision.npz`` from the sample's navigation members and
    reshapes each field into the ``[B,H,W]`` / ``[B]`` / ``[B,2]`` tensor
    layout the loss expects (matching ``pre_extracted.py``). Also returns the
    route-validity flags the loss gates on.

    Args:
        s: A ``KitScenesSample`` with ``navigation_members``.
        device: Torch device.

    Returns:
        (route_supervision, route_valid, route_intersection):
            route_supervision - dict of tensors:
                distance_to_corridor_m [1,H,W], route_heading_sin/cos/valid
                [1,H,W], destination_xy_m [1,2], destination_visible [1],
                available [1].
            route_valid - [1] bool from navigation metadata.
            route_intersection - [1] bool from navigation metadata.
    """
    sup = decode_route_supervision(s["navigation_members"])
    _, _, meta = decode_sample_navigation(s["navigation_members"])
    route_valid = torch.tensor(
        [bool(meta["route_valid"])], dtype=torch.bool, device=device
    )
    route_intersection = torch.tensor(
        [bool(meta["route_intersection"])], dtype=torch.bool, device=device
    )

    def _t(arr):
        return torch.from_numpy(np.asarray(arr).copy()).float().unsqueeze(0).to(device)

    route_supervision = {
        "distance_to_corridor_m": _t(sup.distance_to_corridor_m),
        "distance_to_drivable_m": _t(sup.distance_to_drivable_m),
        "route_heading_sin": _t(sup.route_heading_sin),
        "route_heading_cos": _t(sup.route_heading_cos),
        "route_heading_valid": _t(sup.route_heading_valid),
        "destination_xy_m": _t(sup.destination_xy_m),
        "destination_visible": torch.tensor(
            [bool(sup.destination_visible)], dtype=torch.bool, device=device
        ),
        "available": route_valid.clone(),
        "drivable_available": torch.tensor(
            [bool(sup.drivable_available)], dtype=torch.bool, device=device
        ),
    }
    return route_supervision, route_valid, route_intersection


def _extract_v0(s) -> float:
    """Extract the current ground speed (m/s) from the egomotion history.

    The egomotion history is ``(256,)`` = 64 timesteps x [speed, acceleration,
    yaw_rate, curvature]. The last timestep's speed is the vehicle's current
    forward speed, used as the initial condition when integrating predicted
    controls into a trajectory for ADE/FDE. Falls back to 0 m/s when the
    history is too short.

    Args:
        s: A ``KitScenesSample``.

    Returns:
        float current speed in m/s (>= 0).
    """
    eg_hist = s["egomotion_history"]
    if eg_hist.numel() >= 4:
        return max(0.0, float(eg_hist[-4].item()))
    return 0.0


def _augment_image(v: torch.Tensor) -> torch.Tensor:
    """Photometric augmentation on (1, V, 3, H, W) float tiles.

    Applied in train mode only. Combines (stochastically, ~90% of samples):
      - shared color jitter across all V cameras (brightness/contrast/
        saturation) so all views undergo the same lighting change;
      - additive gaussian noise;
      - random erasing on one random camera view.

    Deliberately NO geometric transforms: flipping/cropping/rotation would
    break the fixed camera intrinsics and the calibrated BEV projection, so
    the augmentation is strictly photometric.

    Args:
        v: (1, V, 3, H, W) float tiles in [0, 1].

    Returns:
        Augmented tiles, clamped to [0, 1].
    """
    if not (torch.rand(1).item() < 0.9):
        return v
    b = 1.0 + torch.rand(1).item() * 0.2 - 0.1      # brightness ±0.1
    c = 1.0 + torch.rand(1).item() * 0.4 - 0.2      # contrast ±0.2
    s = 1.0 + torch.rand(1).item() * 0.4 - 0.2      # saturation ±0.2
    v = v * b
    mean = v.mean(dim=(3, 4), keepdim=True)
    v = (v - mean) * c + mean
    gray = v.mean(dim=2, keepdim=True)
    v = (v - gray) * s + gray
    if torch.rand(1).item() < 0.5:
        v = v + torch.randn_like(v) * 0.02
    if torch.rand(1).item() < 0.3:
        V = v.shape[1]
        cam = torch.randint(V, (1,)).item()
        _, _, _, h, w = v.shape
        rh = int(h * (0.05 + 0.15 * torch.rand(1).item()))
        rw = int(w * (0.05 + 0.15 * torch.rand(1).item()))
        y0 = torch.randint(h - rh, (1,)).item()
        x0 = torch.randint(w - rw, (1,)).item()
        v[0, cam, y0:y0 + rh, x0:x0 + rw] = 0.5
    return v.clamp(0.0, 1.0)


def _ade_fde(pred_np, tgt_np, v0=10.0):
    """Compute ADE/FDE at the 3 s and 6.4 s horizons from raw control outputs.

    The model outputs interleaved [accel, curvature] per step; this integrates
    both the prediction and the target into ego-frame (x,y) trajectories with a
    unicycle model (``evaluation.metrics.integrate_trajectory``), then computes
    per-step Euclidean error. The trajectory spans 64 steps @ 10 Hz = 6.4 s;
    the 3 s horizon is the first 30 steps.

    Args:
        pred_np: (128,) predicted controls [a0,c0,a1,c1,...].
        tgt_np:  (128,) target controls (same layout).
        v0: Initial speed (m/s) for integration.

    Returns:
        (ade_3s, fde_3s, ade_64s, fde_64s, px, tx):
            ade_3s - mean error over the first 30 steps
            fde_3s - error at step 30 (the 3 s endpoint)
            ade_64s - mean error over all 64 steps
            fde_64s - error at the final step (6.4 s endpoint)
            px - (64, 2) predicted trajectory
            tx - (64, 2) target trajectory
    """
    px = integrate_trajectory(pred_np[0::2], pred_np[1::2], v0=v0)
    tx = integrate_trajectory(tgt_np[0::2], tgt_np[1::2], v0=v0)
    er = np.linalg.norm(px - tx, axis=1)
    k3 = 30
    ade3 = float(er[:k3].mean())
    fde3 = float(er[k3 - 1])
    ade = float(er.mean())
    fde = float(er[-1])
    return ade3, fde3, ade, fde, px, tx


def _save_trajectory_plot(iter_idx, pred_np, tgt_np, loss, out_dir, v0=10.0):
    """Save a GT-vs-prediction trajectory plot for one training iteration.

    Renders the integrated predicted (red dashed) and target (blue solid)
    trajectories in the ego frame, with the starting point marked, and writes a
    5x5 PNG to ``out_dir/iter_<iter_idx:05d>.png``. Called every
    ``PLOT_EVERY`` steps to give a visual training progress diary.

    Args:
        iter_idx: Global iteration counter (for the filename).
        pred_np: (128,) predicted controls.
        tgt_np:  (128,) target controls.
        loss: Current scalar loss (shown in the title).
        out_dir: Directory to write the PNG into.
        v0: Initial speed (m/s) used for integration.
    """
    px = integrate_trajectory(pred_np[0::2], pred_np[1::2], v0=v0)
    tx = integrate_trajectory(tgt_np[0::2], tgt_np[1::2], v0=v0)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(tx[:, 0], tx[:, 1], "b-", lw=2, label="GT")
    ax.plot(px[:, 0], px[:, 1], "r--", lw=2, label="Pred")
    ax.scatter(tx[0, 0], tx[0, 1], c="k", s=30, marker="o", label="start")
    ax.set_title(f"iter {iter_idx}  loss {loss:.3f}  v0 {v0:.1f}m/s")
    ax.legend()
    ax.axis("equal")
    ax.grid(True)
    fig.savefig(out_dir / f"iter_{iter_idx:05d}.png", dpi=100)
    plt.close(fig)


def _validate(model, val_ds, device, lfn):
    """Run imitation-loss validation over the whole val split.

    Iterates every val sample in eval mode (no dropout/BN-update, no grad),
    computes the imitation loss and ADE/FDE at both horizons with a real
    per-sample initial speed, and returns the means.

    Args:
        model: AutoE2E in eval-capable state (train() is restored at the end).
        val_ds: Validation ``KitScenesDataset``.
        device: Torch device.
        lfn: TrajectoryImitationLoss instance.

    Returns:
        (acc, mean_loss, n):
            acc - dict with "ade3","fde3","ade","fde" means (meters)
            mean_loss - mean imitation loss
            n - number of validation samples
    """
    model.eval()
    tot_loss, n = 0.0, 0
    acc = {"ade3": 0.0, "fde3": 0.0, "ade": 0.0, "fde": 0.0}
    with torch.no_grad():
        for i in range(len(val_ds)):
            s = val_ds[i]
            v0 = _extract_v0(s)
            v, eg, vh, tg, cp, mc, rm, mv, rv = _tensors(s, device)
            with torch.amp.autocast("cuda"):
                out = model(v, mc, vh, eg, route_mask=rm, map_valid=mv,
                            route_valid=rv, projection=PinholeProjection(cp),
                            geometry_type="pinhole", trajectory_target=tg,
                            mode="infer")
            tot_loss += float(lfn(out, tg))
            ade3, fde3, ade, fde, _, _ = _ade_fde(out.cpu().numpy()[0],
                                                  tg.cpu().numpy()[0], v0=v0)
            acc["ade3"] += ade3; acc["fde3"] += fde3
            acc["ade"] += ade; acc["fde"] += fde
            n += 1
    model.train()
    return {k: v / max(1, n) for k, v in acc.items()}, tot_loss / max(1, n), n


def _forward(model, sample, device, *, augment: bool, route_weight: float = 0.0):
    """Run the model on one sample and return (prediction, target, route_batch).

    Args:
        model: AutoE2E model.
        sample: A ``KitScenesSample`` from ``KitScenesDataset``.
        device: Torch device.
        augment: Whether to apply photometric augmentation (train mode only).
        route_weight: If > 0, also decode the route-supervision batch for the
            RouteConsistencyLoss term.

    Returns:
        (out, tg, route_batch):
            out - predicted trajectory [1,128].
            tg - target trajectory [1,128].
            route_batch - (route_supervision, route_valid, route_intersection)
                or None when ``route_weight <= 0``.
    """
    v, eg, vh, tg, cp, mc, rm, mv, rv = _tensors(sample, device)
    if augment:
        v = _augment_image(v)
    route_batch = (
        _route_supervision_batch(sample, device) if route_weight > 0 else None
    )
    with torch.amp.autocast("cuda"):
        out = model(v, mc, vh, eg, route_mask=rm, map_valid=mv,
                    route_valid=rv, projection=PinholeProjection(cp),
                    geometry_type="pinhole", trajectory_target=tg, mode="train")
    return out, tg, route_batch


def train_epoch(epoch, model, train_ds, opt, sched, lfn, route_lfn, device,
                args, step, traj_dir, t_start):
    """Train one full epoch and return (mean_train_loss, route_terms_sum, step).

    Iterates the (shuffled) training samples once, running the forward pass and
    the imitation loss (+ optionally the route-consistency loss) with gradient
    accumulation, clipping, and a cosine LR step every ``grad_accum``
    micro-batches. Also emits a GT-vs-prediction trajectory plot every
    ``PLOT_EVERY`` steps as a visual diary.

    Args:
        epoch: 1-based epoch number (for logging).
        model: AutoE2E model (train mode).
        train_ds: Training ``KitScenesDataset``.
        opt: AdamW optimizer.
        sched: CosineAnnealingLR scheduler.
        lfn: TrajectoryImitationLoss.
        route_lfn: RouteConsistencyLoss or None (disabled when weight <= 0).
        device: Torch device.
        args: Parsed CLI args.
        step: Global optimizer-step counter (mutable via return value).
        traj_dir: Directory for trajectory plot PNGs.
        t_start: Wall-clock start time for progress logging.

    Returns:
        (mean_epoch_loss, route_terms_sum, step):
            mean_epoch_loss - mean total train loss over the epoch.
            route_terms_sum - dict of summed route term means per term
                (corridor/branch/destination/heading/eligible_count) or None.
            step - new global step counter.
    """
    n_train = len(train_ds)
    epoch_loss = 0.0
    route_terms_sum = (
        {"corridor": 0.0, "branch": 0.0, "destination": 0.0,
         "heading": 0.0, "eligible_count": 0.0, "candidate_count": 0.0,
         "count": 0}
        if route_lfn is not None else None
    )
    opt.zero_grad(set_to_none=True)

    # Shuffle the sample order each epoch (unless --no-shuffle) so the
    # gradient-accumulation windows see a different sample sequence.
    if not args.no_shuffle:
        perm = torch.randperm(n_train).tolist()
    else:
        perm = list(range(n_train))

    for idx_count, i in enumerate(perm):
        sample = train_ds[i]
        route_weight = args.route_consistency_weight if route_lfn is not None else 0.0
        out, tg, route_batch = _forward(
            model, sample, device, augment=not args.no_augment,
            route_weight=route_weight,
        )
        # Divide by grad_accum so accumulated gradients match a
        # grad_accum-sized "virtual batch" learning rate.
        loss = lfn(out, tg) / args.grad_accum
        if route_lfn is not None and route_batch is not None:
            route_sup, route_valid, route_intersection = route_batch
            v0 = torch.tensor([_extract_v0(sample)], device=device)
            # RouteConsistencyLoss integrates in fp32 (its own rollout casts to
            # float32); feed fp32 controls since autocast leaves out/tg in fp16.
            route_terms = route_lfn(out.float(), tg.float(), v0, route_sup,
                                    route_valid, route_intersection)
            loss = loss + (args.route_consistency_weight
                           * route_terms["total"] / args.grad_accum)
            for key in ("corridor", "branch", "destination", "heading",
                        "eligible_count", "candidate_count"):
                route_terms_sum[key] += float(route_terms[key])
            route_terms_sum["count"] += 1
        loss.backward()

        # Optimizer step every grad_accum micro-batches.
        if (idx_count + 1) % args.grad_accum == 0:
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

        epoch_loss += float(loss.item() * args.grad_accum)
        step += 1

        # Progress printout every 50 samples (RSS + VRAM in GB).
        if (idx_count + 1) % 50 == 0:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            vram = torch.cuda.memory_allocated() / 1e9
            print(f"  epoch {epoch} iter {idx_count+1}/{n_train} "
                  f"loss {epoch_loss/(idx_count+1):.4f} "
                  f"(lr {sched.get_last_lr()[0]:.2e}, "
                  f"RSS {rss:.2f}G VRAM {vram:.2f}G, "
                  f"{time.time()-t_start:.0f}s)", flush=True)
        # Visual diary: GT-vs-prediction trajectory plot every PLOT_EVERY steps.
        if step % PLOT_EVERY == 0:
            v0 = _extract_v0(sample)
            _save_trajectory_plot(step, out.detach().cpu().numpy()[0],
                                  tg.detach().cpu().numpy()[0],
                                  float(loss.item() * args.grad_accum),
                                  traj_dir, v0=v0)

    # Flush remaining accumulated gradients if epoch end doesn't align with grad_accum.
    if n_train % args.grad_accum != 0:
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

    if route_terms_sum is not None and route_terms_sum["count"] > 0:
        n_rt = route_terms_sum["count"]
        route_terms_sum = {k: (v / n_rt if k not in ("count",) else v)
                           for k, v in route_terms_sum.items()}
    return epoch_loss / n_train, route_terms_sum, step


def save_val_curve(history, ckpt_dir):
    """Render and save the 3s/6.4s val ADE/FDE curve vs epoch."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([h["epoch"] for h in history["val"]],
            [h["ade_64s"] for h in history["val"]], "ro-", label="6.4s ADE")
    ax.plot([h["epoch"] for h in history["val"]],
            [h["ade_3s"] for h in history["val"]], "bo-", label="3s ADE")
    ax.plot([h["epoch"] for h in history["val"]],
            [h["fde_64s"] for h in history["val"]], "rs--", label="6.4s FDE")
    ax.set_xlabel("epoch"); ax.set_ylabel("meters")
    ax.legend(); ax.grid(True); ax.set_title("train_main val curve (3s / 6.4s)")
    fig.savefig(ckpt_dir / "curve.png", dpi=120)
    plt.close(fig)


def train(args):
    """Run the full training loop given parsed args.

    High-level flow:
      1. Resolve the experiment output directory and build train/val datasets
         from the canonical splits.json split.
      2. Construct the AutoE2E model and warm-start from --init (strict=False).
      3. Loop over epochs: train_epoch() -> validate -> save best/latest
         checkpoints + history.
      4. Render the final val curve and report the best 6.4s ADE.

    Checkpoint selection: the epoch with the lowest 6.4 s val ADE is saved as
    ``best.pt``; ``latest.pt`` always holds the most recent epoch.
    """
    device = args.device if torch.cuda.is_available() else "cpu"
    # Per-experiment output layout: exp-<name>/{checkpoints,history.json,trajectories}
    exp_dir = KIT_DIR / args.exp_name
    ckpt_dir = exp_dir / "checkpoints"
    traj_dir = exp_dir / "trajectories"
    hist_path = exp_dir / "history.json"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, train_ids, val_ids = _build_sets()
    n_train = len(train_ds)
    print(f"train: {n_train} samples / {len(train_ids)} tars | "
          f"val: {len(val_ds)} samples / {len(val_ids)} tars | exp: {args.exp_name}")

    model = AutoE2E(enable_reasoning=False, num_views=NUM_VIEWS,
                    map_context_channels=14,
                    route_channels=2, map_fusion_mode="deformable").to(device)
    # Warm start: transfer weights from a prior run with strict=False so a
    # slightly different architecture still loads; dropped/extra keys are logged.
    if not args.no_init:
        init_ckpt = Path(args.init)
        if init_ckpt.is_file():
            missing, unexpected = model.load_state_dict(
                torch.load(init_ckpt, map_location=device), strict=False)
            if missing or unexpected:
                print(f"  init load: {len(missing)} missing, {len(unexpected)} "
                      f"unexpected keys (dropped)")
            print(f"initialized from {init_ckpt}")
        else:
            print(f"WARNING: init ckpt not found at {init_ckpt}; training from scratch")
    lfn = TrajectoryImitationLoss(loss_type="smooth_l1", temporal_decay=0.95,
                                  signal_scales=(0.778, 0.0350)).to(device)
    route_lfn = None
    if args.route_consistency_weight > 0:
        route_lfn = RouteConsistencyLoss().to(device)
        print(f"route-consistency loss ON (weight {args.route_consistency_weight})")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    # Cosine LR anneals over the TOTAL number of optimizer steps (not per epoch),
    # so the schedule is fixed regardless of where epochs fall on the boundary.
    steps_per_epoch = math.ceil(n_train / args.grad_accum)
    n_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=1e-6)

    history = {"train_loss": [], "val": [], "iter": [], "route": []}
    best_ade = float("inf")
    t_start = time.time()
    step = 0

    for epoch in range(args.epochs):
        print(f"\n=== EPOCH {epoch + 1}/{args.epochs} ===")
        train_avg, route_terms_sum, step = train_epoch(
            epoch + 1, model, train_ds, opt, sched, lfn, route_lfn, device,
            args, step, traj_dir, t_start,
        )

        # Epoch summary: mean train loss + val loss/ADE/FDE at both horizons.
        vac, vl, vn = _validate(model, val_ds, device, lfn)
        print(f"  EPOCH {epoch+1} done: train {train_avg:.4f} | "
              f"val loss {vl:.4f} | 3s ADE {vac['ade3']:.2f} FDE {vac['fde3']:.2f} "
              f"| 6.4s ADE {vac['ade']:.2f} FDE {vac['fde']:.2f} (n={vn})")
        history["train_loss"].append(train_avg)
        history["val"].append({"epoch": epoch + 1, "val_loss": vl,
                               "ade_3s": vac["ade3"], "fde_3s": vac["fde3"],
                               "ade_64s": vac["ade"], "fde_64s": vac["fde"],
                               "n": vn})
        history["iter"].append(step)
        if route_terms_sum is not None:
            history["route"].append({"epoch": epoch + 1, **route_terms_sum})
            print(f"    route: corridor {route_terms_sum['corridor']:.4f} "
                  f"branch {route_terms_sum['branch']:.4f} "
                  f"dest {route_terms_sum['destination']:.4f} "
                  f"heading {route_terms_sum['heading']:.4f} "
                  f"eligible {route_terms_sum['eligible_count']:.0f}/"
                  f"{route_terms_sum['candidate_count']:.0f}")
        # Best checkpoint = lowest 6.4 s val ADE; latest = most recent epoch.
        if vac["ade"] < best_ade:
            best_ade = vac["ade"]
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  * new best ADE {vac['ade']:.2f} -> saved best.pt")
        torch.save(model.state_dict(), ckpt_dir / "latest.pt")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        gc.collect()

    # Final val curve: 3s/6.4s ADE and 6.4s FDE vs epoch.
    save_val_curve(history, ckpt_dir)

    print(f"\nDone in {time.time()-t_start:.0f}s. best 6.4s ADE {best_ade:.2f}")
    print(f"Checkpoints: {ckpt_dir} | trajectories: {traj_dir}")
    return best_ade


def main():
    """Parse CLI args and launch training."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-grad-norm", type=float, default=5.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-init", action="store_true",
                    help="start from scratch instead of the --init checkpoint")
    ap.add_argument("--no-augment", action="store_true",
                    help="disable image augmentation")
    ap.add_argument("--no-shuffle", action="store_true",
                    help="disable random epoch shuffling")
    ap.add_argument("--exp-name", type=str, default=DEFAULT_EXP,
                    help="experiment output dir name (exp-<name>)")
    ap.add_argument("--init", type=str,
                    default=str(KIT_DIR / "exp-2-baseline" / "checkpoints"
                                / "best.pt"),
                    help="checkpoint to initialize from")
    ap.add_argument("--route-consistency-weight", type=float, default=0.10,
                    help="weight of the RouteConsistencyLoss term (0 disables)")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
