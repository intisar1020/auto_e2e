"""exp-1-subset: fully-local end-to-end training on 30GB of extracted tars.

- 7 train tars + 3 val tars from exp-1-subset/data (all local, no streaming)
- 3 epochs, batch 1 / grad-accum 4, fp16 autocast (RTX 3060 12GB)
- AdamW + cosine LR 1e-4 -> 1e-6 (T_max = total steps over 3 epochs)
- Every epoch: val on all 3 val tars -> best.pt by ADE
- Every 200 iterations: GT-vs-prediction trajectory plot saved to
  exp-1-subset/trajectories/iter_%05d.png
- Logs: history.json (per-epoch val + per-iter train loss), curve.png

Usage:
  cd Model/data_parsing/kit_scenes
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_exp1.py
"""

from __future__ import annotations

import argparse
import gc
import json
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

from data_parsing.kit_scenes import KitScenesDataset
from evaluation.metrics import integrate_trajectory
from model_components.auto_e2e import AutoE2E
from model_components.losses.trajectory_loss import TrajectoryImitationLoss
from model_components.view_fusion.projection import PinholeProjection
from navigation.artifacts import decode_sample_navigation

BASE = Path(__file__).parent / "exp-1-subset"
PLOT_EVERY = 200


def _load_manifest():
    m = json.loads((BASE / "manifest.json").read_text())
    train = [e["sid"] for e in m["tars"] if e["role"] == "train"]
    val = [e["sid"] for e in m["tars"] if e["role"] == "val"]
    return train, val


def _build_sets():
    train_ids, val_ids = _load_manifest()
    root = BASE / "data"
    train_ds = KitScenesDataset(data_root=str(root), split="train",
                                include_navigation=True, scene_ids=train_ids)
    val_ds = KitScenesDataset(data_root=str(root), split="train",
                              include_navigation=True, scene_ids=val_ids)
    return train_ds, val_ds, train_ids, val_ids


def _tensors(s, device):
    v = s["visual_tiles"].unsqueeze(0).to(device).float() / 255.0
    eg = s["egomotion_history"].unsqueeze(0).to(device).float()
    vh = s["visual_history"].unsqueeze(0).to(device).float()
    tg = s["trajectory_target"].unsqueeze(0).to(device).float()
    cp = s["camera_params"].unsqueeze(0).to(device).float()
    mc, rm, _ = decode_sample_navigation(s["navigation_members"])
    mc = torch.from_numpy(mc.copy()).float().unsqueeze(0).to(device)
    rm = torch.from_numpy(rm.copy()).float().unsqueeze(0).to(device)
    mv = torch.ones(1, dtype=torch.bool, device=device)
    rv = torch.zeros(1, dtype=torch.bool, device=device)
    return v, eg, vh, tg, cp, mc, rm, mv, rv


def _augment_image(v: torch.Tensor) -> torch.Tensor:
    """Photometric augmentation on (1, V, 3, H, W) float tiles.

    Color jitter + gaussian noise + random erasing. No geometric transforms:
    flipping/cropping would break the fixed camera intrinsics and the
    calibrated BEV projection.
    """
    if not (torch.rand(1).item() < 0.9):
        return v
    # Shared color jitter across all V cameras (same lighting change).
    b = 1.0 + torch.rand(1).item() * 0.2 - 0.1      # brightness ±0.1
    c = 1.0 + torch.rand(1).item() * 0.4 - 0.2      # contrast ±0.2
    s = 1.0 + torch.rand(1).item() * 0.4 - 0.2      # saturation ±0.2
    v = v * b
    mean = v.mean(dim=(3, 4), keepdim=True)
    v = (v - mean) * c + mean
    gray = v.mean(dim=2, keepdim=True)
    v = (v - gray) * s + gray
    # Gaussian noise.
    if torch.rand(1).item() < 0.5:
        v = v + torch.randn_like(v) * 0.02
    # Random erasing on one random camera view.
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


def _forward(model, s, device, mode="train"):
    v, eg, vh, tg, cp, mc, rm, mv, rv = _tensors(s, device)
    if mode == "train":
        v = _augment_image(v)
    with torch.amp.autocast("cuda"):
        out = model(v, mc, vh, eg, route_mask=rm, map_valid=mv, route_valid=rv,
                    projection=PinholeProjection(cp), geometry_type="pinhole",
                    trajectory_target=tg, mode=mode)
    return out, tg


def _ade_fde(pred_np, tgt_np):
    px = integrate_trajectory(pred_np[0::2], pred_np[1::2], v0=10.0)
    tx = integrate_trajectory(tgt_np[0::2], tgt_np[1::2], v0=10.0)
    er = np.linalg.norm(px - tx, axis=1)
    return float(er.mean()), float(er[-1]), px, tx


def _save_trajectory_plot(iter_idx, pred_np, tgt_np, loss, out_dir):
    px = integrate_trajectory(pred_np[0::2], pred_np[1::2], v0=10.0)
    tx = integrate_trajectory(tgt_np[0::2], tgt_np[1::2], v0=10.0)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(tx[:, 0], tx[:, 1], "b-", lw=2, label="GT")
    ax.plot(px[:, 0], px[:, 1], "r--", lw=2, label="Pred")
    ax.scatter(tx[0, 0], tx[0, 1], c="k", s=30, marker="o", label="start")
    ax.set_title(f"iter {iter_idx}  loss {loss:.3f}")
    ax.legend()
    ax.axis("equal")
    ax.grid(True)
    fig.savefig(out_dir / f"iter_{iter_idx:05d}.png", dpi=100)
    plt.close(fig)


def _validate(model, val_ds, device, lfn):
    model.eval()
    tot_loss, tot_ade, tot_fde, n = 0.0, 0.0, 0.0, 0
    per_scene = {}
    with torch.no_grad():
        for i in range(len(val_ds)):
            s = val_ds[i]
            out, tg = _forward(model, s, device, mode="train")
            tot_loss += float(lfn(out, tg))
            ade, fde, _, _ = _ade_fde(out.cpu().numpy()[0], tg.cpu().numpy()[0])
            tot_ade += ade
            tot_fde += fde
            n += 1
    model.train()
    return tot_loss / n, tot_ade / n, tot_fde / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    train_ds, val_ds, train_ids, val_ids = _build_sets()
    n_train = len(train_ds)
    print(f"train: {n_train} samples / {len(train_ids)} tars | "
          f"val: {len(val_ds)} samples / {len(val_ids)} tars")

    model = AutoE2E(enable_reasoning=False, map_context_channels=14,
                    route_channels=2, map_fusion_mode="deformable").to(device)
    lfn = TrajectoryImitationLoss(loss_type="smooth_l1", temporal_decay=0.95,
                                  signal_scales=(0.778, 0.0350)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    n_steps = n_train * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=1e-6)

    ckpt_dir = BASE / "checkpoints"
    traj_dir = BASE / "trajectories"
    ckpt_dir.mkdir(exist_ok=True)
    traj_dir.mkdir(exist_ok=True)

    history = {"train_loss": [], "val": [], "iter": []}
    best_ade = float("inf")
    t_start = time.time()
    step = 0

    for epoch in range(args.epochs):
        print(f"\n=== EPOCH {epoch + 1}/{args.epochs} ===")
        epoch_loss = 0.0
        opt.zero_grad(set_to_none=True)
        for i in range(n_train):
            s = train_ds[i]
            out, tg = _forward(model, s, device, mode="train")
            loss = lfn(out, tg) / args.grad_accum
            loss.backward()
            if (i + 1) % args.grad_accum == 0:
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            epoch_loss += float(loss.item() * args.grad_accum)
            step += 1
            if (i + 1) % 50 == 0:
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
                vram = torch.cuda.memory_allocated() / 1e9
                print(f"  epoch {epoch+1} iter {i+1}/{n_train} "
                      f"loss {epoch_loss/(i+1):.4f} "
                      f"(lr {sched.get_last_lr()[0]:.2e}, "
                      f"RSS {rss:.2f}G VRAM {vram:.2f}G, "
                      f"{time.time()-t_start:.0f}s)", flush=True)
            if step % PLOT_EVERY == 0:
                _save_trajectory_plot(step, out.detach().cpu().numpy()[0],
                                      tg.detach().cpu().numpy()[0],
                                      float(loss.item() * args.grad_accum), traj_dir)
        train_avg = epoch_loss / n_train
        vl, va, vf, vn = _validate(model, val_ds, device, lfn)
        print(f"  EPOCH {epoch+1} done: train {train_avg:.4f} | "
              f"val loss {vl:.4f} ADE {va:.2f} FDE {vf:.2f} (n={vn})")
        history["train_loss"].append(train_avg)
        history["val"].append({"epoch": epoch + 1, "val_loss": vl,
                               "val_ade": va, "val_fde": vf, "n": vn})
        history["iter"].append(step)
        if va < best_ade:
            best_ade = va
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  * new best ADE {va:.2f} -> saved best.pt")
        torch.save(model.state_dict(), ckpt_dir / "latest.pt")
        with open(BASE / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    # curve.png
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([h["epoch"] for h in history["val"]],
            [h["val_ade"] for h in history["val"]], "ro-", label="val ADE")
    ax.plot([h["epoch"] for h in history["val"]],
            [h["val_fde"] for h in history["val"]], "rs--", label="val FDE")
    ax.set_xlabel("epoch"); ax.set_ylabel("meters")
    ax.legend(); ax.grid(True); ax.set_title("exp-1-subset val curve")
    fig.savefig(ckpt_dir / "curve.png", dpi=120)
    plt.close(fig)

    print(f"\nDone in {time.time()-t_start:.0f}s. best ADE {best_ade:.2f}")
    print(f"Checkpoints: {ckpt_dir} | trajectories: {traj_dir}")


if __name__ == "__main__":
    main()
