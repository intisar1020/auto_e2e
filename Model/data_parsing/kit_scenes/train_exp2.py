"""exp-2-subset: 50-epoch training on the full ~150GB split, all 16 channels.

Compared to exp-1-subset:
- Reads the canonical train/val split from splits.json (the fixed 175GB pool).
- Initializes weights from exp-1-subset/checkpoints/best.pt (pretrained baseline).
- route_valid=True: the 2 route channels are now ENABLED (all 16 input channels
  are live), so the model is trained map + route conditioned.
- Outputs to exp-2-subset/ (checkpoints, history.json, curve.png, trajectories).

Data root is shared with exp-1-subset (no re-download). The split IDs come from
splits.json, not the per-tar manifest.

Usage:
  cd Model/data_parsing/kit_scenes
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_exp2.py --epochs 50
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

from data_parsing.kit_scenes import KitScenesDataset
from evaluation.metrics import integrate_trajectory
from model_components.auto_e2e import AutoE2E
from model_components.losses.trajectory_loss import TrajectoryImitationLoss
from model_components.view_fusion.projection import PinholeProjection
from navigation.artifacts import decode_sample_navigation

BASE = Path(__file__).parent / "exp-2-subset"
EXP1_BASE = Path(__file__).parent / "exp-1-subset"
DATA_ROOT = EXP1_BASE / "data"
SPLITS = Path(__file__).parent / "splits.json"
PLOT_EVERY = 200


def _load_split():
    s = json.loads(SPLITS.read_text())
    return s["train_ids"], s["val_ids"]


def _build_sets():
    train_ids, val_ids = _load_split()
    train_ds = KitScenesDataset(data_root=str(DATA_ROOT), split="train",
                                include_navigation=True, scene_ids=train_ids)
    val_ds = KitScenesDataset(data_root=str(DATA_ROOT), split="train",
                              include_navigation=True, scene_ids=val_ids)
    return train_ds, val_ds, train_ids, val_ids


def _tensors(s, device):
    v = s["visual_tiles"].unsqueeze(0).to(device).float() / 255.0
    eg = s["egomotion_history"].unsqueeze(0).to(device).float()
    vh = s["visual_history"].unsqueeze(0).to(device).float()
    tg = s["trajectory_target"].unsqueeze(0).to(device).float()
    cp = s["camera_params"].unsqueeze(0).to(device).float()

    if "navigation_members" in s:
        mc, rm, _ = decode_sample_navigation(s["navigation_members"])
    else:
        mc = np.zeros((14, 256, 256), dtype=np.float32)
        rm = np.zeros((2, 256, 256), dtype=np.float32)

    mc = torch.from_numpy(mc.copy()).float().unsqueeze(0).to(device)
    rm = torch.from_numpy(rm.copy()).float().unsqueeze(0).to(device)
    # Both gates ON: map + route (all 16 channels live).
    mv = torch.ones(1, dtype=torch.bool, device=device)
    rv = torch.ones(1, dtype=torch.bool, device=device)
    return v, eg, vh, tg, cp, mc, rm, mv, rv


def _extract_v0(s) -> float:
    """Extract current ground speed (m/s) from egomotion history for trajectory integration."""
    eg_hist = s["egomotion_history"]
    if eg_hist.numel() >= 4:
        return max(0.0, float(eg_hist[-4].item()))
    return 0.0


def _augment_image(v: torch.Tensor) -> torch.Tensor:
    """Photometric augmentation on (1, V, 3, H, W) float tiles."""
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
    """Return (ade_3s, fde_3s, ade_64s, fde_64s, px, tx).

    Trajectory is 64 steps @ 10Hz = 6.4s. 3s horizon = first 30 steps.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-grad-norm", type=float, default=5.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-init", action="store_true",
                    help="start from scratch instead of exp-1 best.pt")
    ap.add_argument("--no-augment", action="store_true",
                    help="disable image augmentation")
    ap.add_argument("--no-shuffle", action="store_true",
                    help="disable random epoch shuffling")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    train_ds, val_ds, train_ids, val_ids = _build_sets()
    n_train = len(train_ds)
    print(f"train: {n_train} samples / {len(train_ids)} tars | "
          f"val: {len(val_ds)} samples / {len(val_ids)} tars")

    ckpt_dir = BASE / "checkpoints"
    traj_dir = BASE / "trajectories"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    model = AutoE2E(enable_reasoning=False, map_context_channels=14,
                    route_channels=2, map_fusion_mode="deformable").to(device)
    if not args.no_init:
        init_ckpt = EXP1_BASE / "checkpoints" / "best.pt"
        if init_ckpt.is_file():
            model.load_state_dict(torch.load(init_ckpt, map_location=device))
            print(f"initialized from {init_ckpt}")
        else:
            print(f"WARNING: init ckpt not found at {init_ckpt}; training from scratch")
    lfn = TrajectoryImitationLoss(loss_type="smooth_l1", temporal_decay=0.95,
                                  signal_scales=(0.778, 0.0350)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps_per_epoch = math.ceil(n_train / args.grad_accum)
    n_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=1e-6)

    history = {"train_loss": [], "val": [], "iter": []}
    best_ade = float("inf")
    t_start = time.time()
    step = 0

    def _fwd(s, mode="train"):
        v, eg, vh, tg, cp, mc, rm, mv, rv = _tensors(s, device)
        if mode == "train" and not args.no_augment:
            v = _augment_image(v)
        with torch.amp.autocast("cuda"):
            out = model(v, mc, vh, eg, route_mask=rm, map_valid=mv,
                        route_valid=rv, projection=PinholeProjection(cp),
                        geometry_type="pinhole", trajectory_target=tg, mode=mode)
        return out, tg

    for epoch in range(args.epochs):
        print(f"\n=== EPOCH {epoch + 1}/{args.epochs} ===")
        epoch_loss = 0.0
        opt.zero_grad(set_to_none=True)

        if not args.no_shuffle:
            perm = torch.randperm(n_train).tolist()
        else:
            perm = list(range(n_train))

        for idx_count, i in enumerate(perm):
            s = train_ds[i]
            out, tg = _fwd(s, mode="train")
            loss = lfn(out, tg) / args.grad_accum
            loss.backward()

            if (idx_count + 1) % args.grad_accum == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

            epoch_loss += float(loss.item() * args.grad_accum)
            step += 1

            if (idx_count + 1) % 50 == 0:
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
                vram = torch.cuda.memory_allocated() / 1e9
                print(f"  epoch {epoch+1} iter {idx_count+1}/{n_train} "
                      f"loss {epoch_loss/(idx_count+1):.4f} "
                      f"(lr {sched.get_last_lr()[0]:.2e}, "
                      f"RSS {rss:.2f}G VRAM {vram:.2f}G, "
                      f"{time.time()-t_start:.0f}s)", flush=True)
            if step % PLOT_EVERY == 0:
                v0 = _extract_v0(s)
                _save_trajectory_plot(step, out.detach().cpu().numpy()[0],
                                      tg.detach().cpu().numpy()[0],
                                      float(loss.item() * args.grad_accum), traj_dir, v0=v0)
        # Flush remaining accumulated gradients if epoch end doesn't align with grad_accum
        if n_train % args.grad_accum != 0:
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

        train_avg = epoch_loss / n_train
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
        if vac["ade"] < best_ade:
            best_ade = vac["ade"]
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  * new best ADE {vac['ade']:.2f} -> saved best.pt")
        torch.save(model.state_dict(), ckpt_dir / "latest.pt")
        with open(BASE / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        gc.collect()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([h["epoch"] for h in history["val"]],
            [h["ade_64s"] for h in history["val"]], "ro-", label="6.4s ADE")
    ax.plot([h["epoch"] for h in history["val"]],
            [h["ade_3s"] for h in history["val"]], "bo-", label="3s ADE")
    ax.plot([h["epoch"] for h in history["val"]],
            [h["fde_64s"] for h in history["val"]], "rs--", label="6.4s FDE")
    ax.set_xlabel("epoch"); ax.set_ylabel("meters")
    ax.legend(); ax.grid(True); ax.set_title("exp-2-subset val curve (3s / 6.4s)")
    fig.savefig(ckpt_dir / "curve.png", dpi=120)
    plt.close(fig)

    print(f"\nDone in {time.time()-t_start:.0f}s. best 6.4s ADE {best_ade:.2f}")
    print(f"Checkpoints: {ckpt_dir} | trajectories: {traj_dir}")


if __name__ == "__main__":
    main()
