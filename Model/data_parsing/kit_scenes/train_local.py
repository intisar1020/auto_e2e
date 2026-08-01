"""Local training script for reactive-only KITScenes training with navigation maps.

Trains AutoE2E on a few scenes from local disk. No Flyte, MLflow, or S3 needed.

Usage:
  cd Model/data_parsing/kit_scenes
  python train_local.py --epochs 3 --lr 1e-4 --batch-size 2

Requires extracted scenes at /tmp/kitscenes_fwd/train/{scene_id}/
(see navigation_fwd_test.py for download logic)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_MODEL_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_MODEL_DIR))

from data_parsing.kit_scenes import KitScenesDataset
from evaluation.metrics import integrate_trajectory
from model_components.auto_e2e import AutoE2E
from model_components.losses.trajectory_loss import TrajectoryImitationLoss
from model_components.view_fusion.projection import PinholeProjection
from navigation.artifacts import decode_sample_navigation


def _load_navigation_batch(dataset, indices, device):
    """Load a batch of samples from dataset, decode navigation, return tensors."""
    B = len(indices)
    visual, ego_hist, vis_hist, target, cam_params = [], [], [], [], []
    map_ctx, route_msk, map_v, route_v = [], [], [], []

    for idx in indices:
        s = dataset[idx]
        visual.append(s["visual_tiles"])
        ego_hist.append(s["egomotion_history"])
        vis_hist.append(s["visual_history"])
        target.append(s["trajectory_target"])
        cam_params.append(s["camera_params"])

        mc, rm, _ = decode_sample_navigation(s["navigation_members"])
        map_ctx.append(torch.from_numpy(mc.copy()).float())
        route_msk.append(torch.from_numpy(rm.copy()).float())
        map_v.append(torch.tensor(True))
        route_v.append(torch.tensor(False))

    return {
        "visual": torch.stack(visual).to(device).float() / 255.0,
        "ego_hist": torch.stack(ego_hist).to(device).float(),
        "vis_hist": torch.stack(vis_hist).to(device).float(),
        "target": torch.stack(target).to(device).float(),
        "cam_params": torch.stack(cam_params).to(device).float(),
        "map_ctx": torch.stack(map_ctx).to(device).float(),
        "route_msk": torch.stack(route_msk).to(device).float(),
        "map_v": torch.stack(map_v).to(device),
        "route_v": torch.stack(route_v).to(device),
    }


def _split_indices(n_samples: int, val_frac: float, seed: int = 42):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_samples)
    n_val = max(1, int(n_samples * val_frac))
    return perm[n_val:].tolist(), perm[:n_val].tolist()


def _compute_ade_fde(pred: np.ndarray, tgt: np.ndarray, v0: float = 10.0):
    """Compute ADE / FDE from (T*2,) control vectors."""
    pred_xy = integrate_trajectory(pred[0::2], pred[1::2], v0=v0)
    tgt_xy = integrate_trajectory(tgt[0::2], tgt[1::2], v0=v0)
    err = np.linalg.norm(pred_xy - tgt_xy, axis=1)
    return float(err.mean()), float(err[-1])


def _evaluate(model, dataset, indices, device, loss_fn):
    model.eval()
    total_loss, total_ade, total_fde = 0.0, 0.0, 0.0
    B = min(2, len(indices))
    with torch.no_grad():
        for start in range(0, len(indices), B):
            batch_idx = indices[start : start + B]
            batch = _load_navigation_batch(dataset, batch_idx, device)
            with torch.amp.autocast("cuda"):
                out = model(
                    batch["visual"],
                    batch["map_ctx"],
                    batch["vis_hist"],
                    batch["ego_hist"],
                    route_mask=batch["route_msk"],
                    map_valid=batch["map_v"],
                    route_valid=batch["route_v"],
                    projection=PinholeProjection(batch["cam_params"]),
                    geometry_type="pinhole",
                    trajectory_target=batch["target"],
                    mode="train",
            )
            total_loss += float(loss_fn(out, batch["target"])) * len(batch_idx)

            pred_np = out.cpu().numpy()
            tgt_np = batch["target"].cpu().numpy()
            for i in range(len(batch_idx)):
                ade, fde = _compute_ade_fde(pred_np[i], tgt_np[i])
                total_ade += ade
                total_fde += fde

    n = len(indices)
    return total_loss / n, total_ade / n, total_fde / n


def _plot_trajectory(
    model, dataset, indices, device, checkpoint_dir: str, epoch: int,
):
    """Plot predicted vs GT trajectory for a few val samples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    samples_to_plot = min(3, len(indices))
    fig, axes = plt.subplots(1, samples_to_plot, figsize=(5 * samples_to_plot, 4))
    if samples_to_plot == 1:
        axes = [axes]
    fig.suptitle(f"Trajectory: Pred (blue) vs GT (red) — Epoch {epoch}", fontsize=11)

    with torch.no_grad():
        for plot_i in range(samples_to_plot):
            idx = indices[plot_i]
            batch = _load_navigation_batch(dataset, [idx], device)
            with torch.amp.autocast("cuda"):
                out = model(
                    batch["visual"],
                    batch["map_ctx"],
                    batch["vis_hist"],
                    batch["ego_hist"],
                    route_mask=batch["route_msk"],
                    map_valid=batch["map_v"],
                    route_valid=batch["route_v"],
                    projection=PinholeProjection(batch["cam_params"]),
                    geometry_type="pinhole",
                    trajectory_target=batch["target"],
                    mode="train",
            )
            pred = out.cpu().numpy()[0]
            tgt = batch["target"].cpu().numpy()[0]

            pred_xy = integrate_trajectory(pred[0::2], pred[1::2], v0=10.0)
            tgt_xy = integrate_trajectory(tgt[0::2], tgt[1::2], v0=10.0)

            ax = axes[plot_i]
            ax.plot(pred_xy[:, 0], pred_xy[:, 1], "b-", linewidth=2, label="Pred", alpha=0.8)
            ax.plot(tgt_xy[:, 0], tgt_xy[:, 1], "r-", linewidth=2, label="GT", alpha=0.8)
            ax.scatter(pred_xy[0, 0], pred_xy[0, 1], c="blue", s=40, zorder=5)
            ax.scatter(tgt_xy[0, 0], tgt_xy[0, 1], c="red", s=40, zorder=5)
            ax.scatter(pred_xy[-1, 0], pred_xy[-1, 1], c="blue", marker="x", s=60, zorder=5)
            ax.scatter(tgt_xy[-1, 0], tgt_xy[-1, 1], c="red", marker="x", s=60, zorder=5)
            ade = float(np.linalg.norm(pred_xy - tgt_xy, axis=1).mean())
            ax.set_title(f"Sample {idx} (ADE={ade:.1f}m)", fontsize=9)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.axis("equal")

    plt.tight_layout()
    out_path = f"{checkpoint_dir}/trajectory_epoch_{epoch:03d}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  Trajectory plot: {out_path}")
    plt.close(fig)


def _plot_history(history: list[dict], checkpoint_dir: str):
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_ade = [h["val_ade"] for h in history]
    val_fde = [h["val_fde"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training Curves — KITScenes Local", fontsize=12, fontweight="bold")

    ax_loss = axes[0]
    ax_loss.plot(epochs, train_loss, "b-o", label="Train Loss (control-space)")
    ax_loss.plot(epochs, val_loss, "r-s", label="Val Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("SmoothL1 Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_ade = axes[1]
    ax_ade.plot(epochs, val_ade, "r-s", label=f"ADE (m)")
    ax_ade.plot(epochs, val_fde, color="orange", marker="^", label=f"FDE (m)")
    ax_ade.set_xlabel("Epoch")
    ax_ade.set_ylabel("Meters")
    ax_ade.set_title("Displacement Error")
    ax_ade.legend()
    ax_ade.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{checkpoint_dir}/training_curve.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Training curve saved to {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=149)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--no-pretrain", action="store_true", default=False)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--data-root", type=str, default="/tmp/kitscenes_fwd")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")

    # ---- Data ----
    data_root = Path(args.data_root)
    scene_dirs = list((data_root / "train").iterdir())
    scene_ids = [d.name for d in scene_dirs if d.is_dir()]
    print(f"Scenes: {len(scene_ids)}")

    print("Loading dataset...")
    t0 = time.time()
    ds = KitScenesDataset(
        data_root=str(data_root),
        split="train",
        include_navigation=True,
        scene_ids=scene_ids,
    )
    print(f"  {len(ds)} samples from {len(scene_ids)} scenes ({time.time()-t0:.1f}s)")

    # ---- Train/val split ----
    train_idx, val_idx = _split_indices(len(ds), args.val_frac, seed=args.seed)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # ---- Model ----
    model = AutoE2E(
        is_pretrained=not args.no_pretrain,
        enable_world_model=False,
        enable_reasoning=False,
        map_context_channels=14,
        route_channels=2,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params/1e6:.1f}M")

    # ---- Optimizer & Loss ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1,
    )
    loss_fn = TrajectoryImitationLoss(
        loss_type="smooth_l1",
        temporal_decay=0.95,
        signal_scales=(0.778, 0.0350),
    ).to(device)

    # ---- Training ----
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_ade = float("inf")
    best_epoch = -1
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.RandomState(args.seed + epoch * 1000)
        epoch_order = rng.permutation(train_idx).tolist()

        total_loss = 0.0
        total_steps = 0
        t_start = time.time()
        optimizer.zero_grad()

        for step_start in range(0, len(epoch_order) - args.batch_size + 1, args.batch_size):
            batch_indices = epoch_order[step_start : step_start + args.batch_size]
            batch = _load_navigation_batch(ds, batch_indices, device)

            with torch.amp.autocast("cuda"):
                out = model(
                    batch["visual"],
                    batch["map_ctx"],
                    batch["vis_hist"],
                    batch["ego_hist"],
                    route_mask=batch["route_msk"],
                    map_valid=batch["map_v"],
                    route_valid=batch["route_v"],
                    projection=PinholeProjection(batch["cam_params"]),
                    geometry_type="pinhole",
                    trajectory_target=batch["target"],
                    mode="train",
            )

            loss = loss_fn(out, batch["target"]) / args.grad_accum
            loss.backward()
            total_loss += float(loss) * args.grad_accum

            total_steps += 1
            if total_steps % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        if total_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / total_steps if total_steps > 0 else 0
        t_epoch = time.time() - t_start

        # ---- Validation ----
        val_loss, val_ade, val_fde = _evaluate(model, ds, val_idx, device, loss_fn)
        scheduler.step(val_loss)

        # ---- Trajectory plot ----
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            _plot_trajectory(model, ds, val_idx, device, args.checkpoint_dir, epoch)

        is_best = val_ade < best_ade
        if is_best:
            best_ade = val_ade
            best_epoch = epoch

        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"loss={avg_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"ADE={val_ade:.2f}m | FDE={val_fde:.2f}m | "
            f"{t_epoch:.0f}s"
            + (" *" if is_best else "")
        )

        history.append({
            "epoch": epoch, "train_loss": avg_loss,
            "val_loss": val_loss, "val_ade": val_ade, "val_fde": val_fde,
        })

        # ---- Checkpoint (state_dict only to save disk) ----
        if is_best:
            torch.save(model.state_dict(), f"{args.checkpoint_dir}/best.pt")
            # Save history separately (small JSON)
            import json
            with open(f"{args.checkpoint_dir}/history.json", "w") as fh:
                json.dump(history, fh)

    print(f"\nBest: epoch {best_epoch}, ADE={best_ade:.2f}m")

    # ---- Plot training curves ----
    _plot_history(history, args.checkpoint_dir)


if __name__ == "__main__":
    main()
