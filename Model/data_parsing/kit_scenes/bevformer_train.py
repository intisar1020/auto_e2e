"""Training script for the pure-PyTorch BEVFormer trajectory planner.

This is a new, self-contained entrypoint next to ``train_main.py``.  It reuses
the KITScenes dataset, trajectory loss, and evaluation helpers, but trains
``BEVFormerTrajectoryNet`` from ``bevformer_pure_torch.py`` instead of the
existing reactive ``AutoE2E`` network.

The detection/bbox branch is intentionally absent: the BEVFormer encoder now
ends in ``QueryPlanner`` and predicts a 6.4 s acceleration/curvature trajectory
at 10 Hz (128 outputs).  During training the script writes:

    * ``<exp>/trajectories/``  GT vs prediction plots
    * ``<exp>/bev_features/``  PCA visualizations of the BEV encoder output
    * ``<exp>/map_channels/``  per-channel visualizations of the map encoder

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python bevformer_train.py --epochs 30 --exp-name exp-bevformer
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

_MODEL_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_MODEL_DIR))

from data_parsing.kit_scenes import KitScenesDataset, NUM_VIEWS
from evaluation.metrics import integrate_trajectory
from model_components.losses.trajectory_loss import TrajectoryImitationLoss
from model_components.view_fusion.projection import PinholeProjection
from navigation.artifacts import decode_sample_navigation

from bevformer_pure_torch import BEVFormerTrajectoryNet

KIT_DIR = Path(__file__).parent
DATA_ROOT = KIT_DIR / "datasets"
SPLITS = KIT_DIR / "splits.json"
DEFAULT_EXP = "exp-bevformer"

ROUTE_ONLY = True


def _load_split():
    s = json.loads(SPLITS.read_text())
    return s["train_ids"], s["val_ids"]


def _build_sets():
    train_ids, val_ids = _load_split()
    train_ds = KitScenesDataset(
        data_root=str(DATA_ROOT),
        split="train",
        include_navigation=True,
        scene_ids=train_ids,
    )
    val_ds = KitScenesDataset(
        data_root=str(DATA_ROOT),
        split="train",
        include_navigation=True,
        scene_ids=val_ids,
    )
    return train_ds, val_ds, train_ids, val_ids


def _tensors(sample, device):
    v = sample["visual_tiles"].unsqueeze(0).to(device).float() / 255.0
    eg = sample["egomotion_history"].unsqueeze(0).to(device).float()
    vh = sample["visual_history"].unsqueeze(0).to(device).float()
    tg = sample["trajectory_target"].unsqueeze(0).to(device).float()
    cp = sample["camera_params"].unsqueeze(0).to(device).float()

    if "navigation_members" in sample:
        mc, rm, _ = decode_sample_navigation(sample["navigation_members"])
        mc = np.asarray(mc)
        rm = np.asarray(rm)
        if ROUTE_ONLY:
            mc = np.zeros((0, 256, 256), dtype=np.float32)
    else:
        mc = np.zeros((14 if not ROUTE_ONLY else 0, 256, 256), dtype=np.float32)
        rm = np.zeros((2, 256, 256), dtype=np.float32)

    mc = torch.from_numpy(mc.copy()).float().unsqueeze(0).to(device)
    rm = torch.from_numpy(rm.copy()).float().unsqueeze(0).to(device)
    mv = torch.ones(1, dtype=torch.bool, device=device)
    rv = torch.ones(1, dtype=torch.bool, device=device)
    return v, eg, vh, tg, cp, mc, rm, mv, rv


def _extract_v0(sample) -> float:
    eg = sample["egomotion_history"]
    if eg.numel() >= 4:
        return max(0.0, float(eg[-4].item()))
    return 0.0


def _augment_image(v: torch.Tensor) -> torch.Tensor:
    if not (torch.rand(1).item() < 0.9):
        return v
    b = 1.0 + torch.rand(1).item() * 0.2 - 0.1
    c = 1.0 + torch.rand(1).item() * 0.4 - 0.2
    s = 1.0 + torch.rand(1).item() * 0.4 - 0.2
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
        v[0, cam, y0 : y0 + rh, x0 : x0 + rw] = 0.5
    return v.clamp(0.0, 1.0)


def _ade_fde(pred_np, tgt_np, v0=10.0):
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
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"iter_{iter_idx:05d}.png", dpi=100)
    plt.close(fig)


def _pca_rgb(feats: np.ndarray) -> np.ndarray:
    """Project a [N,H,W,C] feature stack to [N,H,W,3] with per-sample PCA."""
    n, h, w, c = feats.shape
    flat = feats.reshape(n * h * w, c)
    pca = PCA(n_components=3)
    proj = pca.fit_transform(flat).reshape(n, h, w, 3)
    proj = proj - proj.min(axis=(1, 2), keepdims=True)
    rng = proj.max(axis=(1, 2), keepdims=True) - proj.min(axis=(1, 2), keepdims=True)
    proj = proj / np.maximum(rng, 1e-6)
    return proj, pca


def _save_visualizations(
    model,
    val_ds,
    device,
    out_dir,
    tag: str,
    n_samples: int,
    viz_map_channels: int,
):
    """Run a few val samples once and save BEV PCA + map-channel montages."""
    if n_samples <= 0:
        return

    bev_captured = []
    map_captured = []
    handles = [
        model.bev_encoder.register_forward_hook(
            lambda _m, _a, o: bev_captured.append(o.detach().float().cpu())
        ),
        model.map_encoder.register_forward_hook(
            lambda _m, _a, o: map_captured.append(o.detach().float().cpu())
        ),
    ]

    model.eval()
    try:
        with torch.no_grad():
            for i in range(min(n_samples, len(val_ds))):
                s = val_ds[i]
                v, eg, vh, tg, cp, mc, rm, mv, rv = _tensors(s, device)
                with torch.amp.autocast("cuda"):
                    model(
                        v,
                        mc,
                        vh,
                        eg,
                        route_mask=rm,
                        map_valid=mv,
                        route_valid=rv,
                        projection=PinholeProjection(cp),
                        geometry_type="pinhole",
                        mode="infer",
                    )
    finally:
        for h in handles:
            h.remove()
        model.train()

    # BEV encoder output is [B, N, C] -> [N, bev_h, bev_w, C].
    if bev_captured:
        feats = torch.cat(bev_captured, dim=0)  # [n, N, C]
        feats = feats.view(feats.shape[0], model.bev_h, model.bev_w, -1)
        rgb, pca = _pca_rgb(feats.numpy())
        fig, axes = plt.subplots(
            1, rgb.shape[0], figsize=(4 * rgb.shape[0], 4)
        )
        if rgb.shape[0] == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.imshow(rgb[i])
            ax.axis("off")
            ax.set_title(
                f"{tag} s{i + 1}\nPCA {pca.explained_variance_ratio_[0]:.2f}/"
                f"{pca.explained_variance_ratio_[1]:.2f}/"
                f"{pca.explained_variance_ratio_[2]:.2f}"
            )
        fig.suptitle("BEVFormer BEV features (PCA)", fontsize=10)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{tag}_bev_pca.png", dpi=110)
        plt.close(fig)

    # Map encoder output [B, C, H, W].  Show raw channels as a small montage.
    if map_captured and viz_map_channels > 0:
        sample_map = map_captured[0][0]  # [C,H,W]
        c = min(viz_map_channels, sample_map.shape[0])
        grid = int(math.ceil(c**0.5))
        fig, axes = plt.subplots(grid, grid, figsize=(grid * 2.2, grid * 2.2))
        axes = np.asarray(axes).reshape(-1)
        for i in range(grid * grid):
            axes[i].axis("off")
            if i < c:
                ch = sample_map[i].numpy()
                ch = ch - ch.min()
                rng = ch.max() - ch.min()
                axes[i].imshow(ch / max(rng, 1e-6), cmap="viridis")
                axes[i].set_title(f"ch {i}")
        fig.suptitle("Map encoder channels", fontsize=10)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{tag}_map_channels.png", dpi=110)
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
                out = model(
                    v,
                    mc,
                    vh,
                    eg,
                    route_mask=rm,
                    map_valid=mv,
                    route_valid=rv,
                    projection=PinholeProjection(cp),
                    geometry_type="pinhole",
                    mode="infer",
                )
            tot_loss += float(lfn(out, tg))
            ade3, fde3, ade, fde, _, _ = _ade_fde(
                out.cpu().numpy()[0], tg.cpu().numpy()[0], v0=v0
            )
            acc["ade3"] += ade3
            acc["fde3"] += fde3
            acc["ade"] += ade
            acc["fde"] += fde
            n += 1
    model.train()
    return {k: v / max(1, n) for k, v in acc.items()}, tot_loss / max(1, n), n


def _forward(model, sample, device, augment: bool, temporal_state: bool = False):
    v, eg, vh, tg, cp, mc, rm, mv, rv = _tensors(sample, device)
    if augment:
        v = _augment_image(v)
    with torch.amp.autocast("cuda"):
        out = model(
            v,
            mc,
            vh,
            eg,
            route_mask=rm,
            map_valid=mv,
            route_valid=rv,
            projection=PinholeProjection(cp),
            geometry_type="pinhole",
            mode="train",
            temporal_key=(
                (sample["scene_id"], sample["frame_idx"]) if temporal_state else None
            ),
        )
    return out, tg


def train_epoch(
    epoch,
    model,
    train_ds,
    val_ds,
    opt,
    sched,
    lfn,
    device,
    args,
    step,
    traj_dir,
    viz_dir,
    t_start,
):
    n_train = len(train_ds)
    epoch_loss = 0.0
    opt.zero_grad(set_to_none=True)
    perm = list(range(n_train))
    if not args.no_shuffle:
        perm = torch.randperm(n_train).tolist()

    for idx_count, i in enumerate(perm):
        sample = train_ds[i]
        out, tg = _forward(
            model,
            sample,
            device,
            augment=not args.no_augment,
            temporal_state=args.temporal_state,
        )
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
            print(
                f"  epoch {epoch} iter {idx_count + 1}/{n_train} "
                f"loss {epoch_loss / (idx_count + 1):.4f} "
                f"(lr {sched.get_last_lr()[0]:.2e}, RSS {rss:.2f}G "
                f"VRAM {vram:.2f}G, {time.time() - t_start:.0f}s)",
                flush=True,
            )

        if step % args.viz_every == 0:
            v0 = _extract_v0(sample)
            _save_trajectory_plot(
                step,
                out.detach().cpu().numpy()[0],
                tg.detach().cpu().numpy()[0],
                float(loss.item() * args.grad_accum),
                traj_dir,
                v0=v0,
            )
            if args.viz_samples > 0:
                _save_visualizations(
                    model,
                    val_ds,
                    device,
                    viz_dir,
                    tag=f"iter_{step:05d}",
                    n_samples=args.viz_samples,
                    viz_map_channels=args.viz_map_channels,
                )

    if n_train % args.grad_accum != 0:
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

    return epoch_loss / n_train, step


def _save_curve(history, ckpt_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        [h["epoch"] for h in history["val"]],
        [h["ade_64s"] for h in history["val"]],
        "ro-",
        label="6.4s ADE",
    )
    ax.plot(
        [h["epoch"] for h in history["val"]],
        [h["ade_3s"] for h in history["val"]],
        "bo-",
        label="3s ADE",
    )
    ax.plot(
        [h["epoch"] for h in history["val"]],
        [h["fde_64s"] for h in history["val"]],
        "rs--",
        label="6.4s FDE",
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("meters")
    ax.legend()
    ax.grid(True)
    ax.set_title("BEVFormer trajectory planner val curve")
    fig.savefig(ckpt_dir / "curve.png", dpi=120)
    plt.close(fig)


def train(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    global ROUTE_ONLY
    ROUTE_ONLY = args.route_only

    exp_dir = KIT_DIR / args.exp_name
    ckpt_dir = exp_dir / "checkpoints"
    traj_dir = exp_dir / "trajectories"
    viz_dir = exp_dir / "bev_vis"
    hist_path = exp_dir / "history.json"
    for d in (ckpt_dir, traj_dir, viz_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, train_ids, val_ids = _build_sets()
    n_train = len(train_ds)
    print(
        f"train: {n_train} samples / {len(train_ids)} tars | "
        f"val: {len(val_ds)} samples / {len(val_ids)} tars | exp: {args.exp_name}"
    )
    print(
        "ROUTE-ONLY mode" if ROUTE_ONLY else "FULL MAP mode",
        "| BEVFormer encoder:",
        f"{args.num_encoder_layers} layers, embed={args.embed_dim}, "
        f"BEV={args.bev_h}x{args.bev_w}",
    )

    model = BEVFormerTrajectoryNet(
        backbone=args.backbone,
        is_pretrained=args.pretrained,
        num_views=NUM_VIEWS,
        embed_dim=args.embed_dim,
        bev_h=args.bev_h,
        bev_w=args.bev_w,
        num_points_in_pillar=args.num_z,
        num_encoder_layers=args.num_encoder_layers,
        num_heads=args.num_heads,
        num_levels=args.num_levels,
        num_points=args.num_points,
        dropout=args.dropout,
        map_context_channels=(0 if args.route_only else 14),
        route_channels=2,
        egomotion_dim=256,
        visual_history_dim=896,
        num_timesteps=64,
        num_signals=2,
        image_size=256,
        use_temporal_state=args.temporal_state,
    ).to(device)

    if args.init:
        init_ckpt = Path(args.init)
        if init_ckpt.is_file():
            missing, unexpected = model.load_state_dict(
                torch.load(init_ckpt, map_location=device), strict=False
            )
            print(
                f"initialized from {init_ckpt} "
                f"({len(missing)} missing, {len(unexpected)} unexpected keys)"
            )
        else:
            print(f"WARNING: init ckpt not found at {init_ckpt}; training from scratch")

    lfn = TrajectoryImitationLoss(
        loss_type="smooth_l1",
        temporal_decay=0.95,
        signal_scales=(0.778, 0.0350),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps_per_epoch = math.ceil(n_train / args.grad_accum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps_per_epoch * args.epochs, eta_min=1e-6
    )

    history = {"train_loss": [], "val": [], "iter": []}
    best_ade = float("inf")
    t_start = time.time()
    step = 0

    for epoch in range(args.epochs):
        print(f"\n=== EPOCH {epoch + 1}/{args.epochs} ===")
        model.reset_temporal_state()
        train_avg, step = train_epoch(
            epoch + 1,
            model,
            train_ds,
            val_ds,
            opt,
            sched,
            lfn,
            device,
            args,
            step,
            traj_dir,
            viz_dir,
            t_start,
        )
        vac, vl, vn = _validate(model, val_ds, device, lfn)
        print(
            f"  EPOCH {epoch + 1} done: train {train_avg:.4f} | "
            f"val loss {vl:.4f} | 3s ADE {vac['ade3']:.2f} FDE {vac['fde3']:.2f} "
            f"| 6.4s ADE {vac['ade']:.2f} FDE {vac['fde']:.2f} (n={vn})"
        )
        history["train_loss"].append(train_avg)
        history["val"].append(
            {
                "epoch": epoch + 1,
                "val_loss": vl,
                "ade_3s": vac["ade3"],
                "fde_3s": vac["fde3"],
                "ade_64s": vac["ade"],
                "fde_64s": vac["fde"],
                "n": vn,
            }
        )
        history["iter"].append(step)

        if args.viz_samples > 0:
            _save_visualizations(
                model,
                val_ds,
                device,
                viz_dir,
                tag=f"epoch_{epoch + 1:03d}",
                n_samples=args.viz_samples,
                viz_map_channels=args.viz_map_channels,
            )

        if vac["ade"] < best_ade:
            best_ade = vac["ade"]
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  * new best ADE {vac['ade']:.2f} -> saved best.pt")
        torch.save(model.state_dict(), ckpt_dir / "latest.pt")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        gc.collect()

    _save_curve(history, ckpt_dir)
    print(
        f"\nDone in {time.time() - t_start:.0f}s. best 6.4s ADE {best_ade:.2f}"
    )
    print(
        f"Checkpoints: {ckpt_dir} | trajectories: {traj_dir} | BEV/map viz: {viz_dir}"
    )
    return best_ade


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-grad-norm", type=float, default=5.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--exp-name", type=str, default=DEFAULT_EXP)
    ap.add_argument("--init", type=str, default="")
    ap.add_argument("--pretrained", action="store_true", default=True)
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    ap.add_argument("--backbone", type=str, default="swin_v2_tiny",
                    choices=["swin_v2_tiny", "conv_next_v2_tiny", "res_net_50"])
    ap.add_argument("--route-only", action="store_true", default=True)
    ap.add_argument("--full-map", dest="route_only", action="store_false")
    ap.add_argument("--temporal-state", action="store_true",
                    help="maintain previous BEV state between forward calls")

    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--bev-h", type=int, default=64)
    ap.add_argument("--bev-w", type=int, default=64)
    ap.add_argument("--num-encoder-layers", type=int, default=2)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--num-levels", type=int, default=4)
    ap.add_argument("--num-points", type=int, default=8)
    ap.add_argument("--num-z", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--viz-every", type=int, default=200)
    ap.add_argument("--viz-samples", type=int, default=3)
    ap.add_argument("--viz-map-channels", type=int, default=16)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
