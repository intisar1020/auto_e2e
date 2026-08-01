"""Final 1-epoch full-dataset streaming training.

Split (HF listing order):
  train = data/train (533) + first 52 of data/val = 585 shards
  val   = remaining 65 val shards

- Resume from checkpoints_full/best.pt
- Train end-to-end (no frozen backbone), 1 epoch
- AdamW + CosineAnnealingLR (1e-4 -> 1e-6 over the whole epoch)
- Stream cache: 1-2 tars (size-aware, >=20GB -> 1)
- best.pt + latest.pt after every 1-2 scenes
- Validation: one random val tar per checkpoint (streamed, discarded)

Usage:
  cd Model/data_parsing/kit_scenes
  python train_final.py
"""

from __future__ import annotations

import argparse, io, json, os, shutil, sys, tarfile, tempfile, time
from pathlib import Path

import numpy as np
import torch

_MODEL_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_MODEL_DIR))

from data_parsing.kit_scenes import KitScenesDataset
from evaluation.metrics import integrate_trajectory
from model_components.auto_e2e import AutoE2E
from model_components.losses.trajectory_loss import TrajectoryImitationLoss
from model_components.view_fusion.projection import PinholeProjection
from navigation.artifacts import decode_sample_navigation


def _download(sid: str, root: Path, split: str = "train"):
    """Download + extract a scene into root/<split>/<sid>. Val and train tars
    live in different split directories, matching the SDK layout."""
    d = root / split / sid
    if d.is_dir():
        return d
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    p = f"datasets/KIT-MRT/KITScenes-Multimodal/data/{split}/{sid}.tar"
    sz = fs.info(p)["size"]
    d.parent.mkdir(parents=True, exist_ok=True)
    tmp = d.parent / f".{sid}.tar.part"
    t0 = time.time()
    with fs.open(p, "rb") as fh, open(tmp, "wb") as out:
        while True:
            chunk = fh.read(64 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    t_dl = time.time() - t0
    t0 = time.time()
    tar = tarfile.open(tmp)
    td = Path(tempfile.mkdtemp(prefix="ks_"))
    tar.extractall(path=td)
    tar.close()
    tmp.unlink()
    (td / sid).rename(d)
    shutil.rmtree(td, ignore_errors=True)
    print(f"    ↓ {sid[:12]} ({sz/1e9:.1f}G) dl{t_dl:.0f}s x{time.time()-t0:.0f}s", flush=True)
    return d


def _clean_partial(root: Path):
    """Remove stale partial downloads left by interrupted runs."""
    removed = 0
    for sub in ("train", "val"):
        td = root / sub
        if not td.is_dir():
            continue
        for part in td.glob(".*.tar.part"):
            part.unlink(missing_ok=True)
            removed += 1
            print(f"  cleaned stale partial: {part}")
    if removed:
        print(f"  removed {removed} stale partial download(s)")


def _evict(root: Path, keep: set[str], cap: int):
    """Evict oldest scenes from train AND val, keeping `keep` and `cap` total."""
    for sub in ("train", "val"):
        td = root / sub
        if not td.is_dir():
            continue
        items = sorted([x for x in td.iterdir() if x.is_dir() and x.name not in keep],
                       key=lambda x: x.stat().st_mtime)
        for x in items[:max(0, len(items) + len(keep) - cap)]:
            shutil.rmtree(x, ignore_errors=True)


def _load_scene(sid: str, root: Path):
    try:
        ds = KitScenesDataset(data_root=str(root), split="train",
                              include_navigation=True, scene_ids=[sid])
    except ValueError as e:
        print(f"    ⚠ {sid[:12]}... train load failed: {str(e)[:80]}")
        return None, 0
    n = len(ds)
    if n == 0:
        print(f"    ⚠ {sid[:12]}... 0 usable samples (too few poses)")
        return None, 0
    V, E, H, T, C, M, R, MV, RV = [], [], [], [], [], [], [], [], []
    for i in range(n):
        s = ds[i]
        V.append(s["visual_tiles"]); E.append(s["egomotion_history"])
        H.append(s["visual_history"]); T.append(s["trajectory_target"])
        C.append(s["camera_params"])
        mc, rm, _ = decode_sample_navigation(s["navigation_members"])
        M.append(torch.from_numpy(mc.copy()).float())
        R.append(torch.from_numpy(rm.copy()).float())
        MV.append(torch.tensor(True)); RV.append(torch.tensor(False))
    return {
        "visual": torch.stack(V), "ego": torch.stack(E),
        "vis_hist": torch.stack(H), "target": torch.stack(T),
        "cam_params": torch.stack(C), "map_ctx": torch.stack(M),
        "route_msk": torch.stack(R), "map_v": torch.stack(MV),
        "route_v": torch.stack(RV),
    }, n


def _validate_single(model, sid, root, device, loss_fn):
    """Validate on ONE val tar. Returns (loss, ade, fde, n)."""
    model.eval()
    try:
        ds = KitScenesDataset(data_root=str(root), split="val",
                              include_navigation=True, scene_ids=[sid])
    except ValueError as e:
        print(f"    ⚠ val {sid[:12]}... load failed: {str(e)[:80]}")
        return 0, 0, 0, 0
    tl, ta, tf, total_n = 0.0, 0.0, 0.0, 0
    for i in range(len(ds)):
        s = ds[i]
        v = s["visual_tiles"].unsqueeze(0).to(device).float()/255.0
        eg = s["egomotion_history"].unsqueeze(0).to(device).float()
        vh = s["visual_history"].unsqueeze(0).to(device).float()
        tg = s["trajectory_target"].unsqueeze(0).to(device).float()
        cp = s["camera_params"].unsqueeze(0).to(device).float()
        mc_np, rm_np, _ = decode_sample_navigation(s["navigation_members"])
        mc = torch.from_numpy(mc_np.copy()).float().unsqueeze(0).to(device)
        rm = torch.from_numpy(rm_np.copy()).float().unsqueeze(0).to(device)
        mv = torch.ones(1, dtype=torch.bool, device=device)
        rv = torch.zeros(1, dtype=torch.bool, device=device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            out = model(v, mc, vh, eg, route_mask=rm, map_valid=mv,
                        route_valid=rv, projection=PinholeProjection(cp),
                        geometry_type="pinhole", trajectory_target=tg, mode="train")
        tl += float(loss_fn(out, tg))
        pn = out.cpu().numpy()[0]; tn = tg.cpu().numpy()[0]
        px = integrate_trajectory(pn[0::2], pn[1::2], v0=10.0)
        tx = integrate_trajectory(tn[0::2], tn[1::2], v0=10.0)
        er = np.linalg.norm(px - tx, axis=1)
        ta += float(er.mean()); tf += float(er[-1]); total_n += 1
    if total_n == 0:
        print(f"    ⚠ val {sid[:12]}... 0 usable samples")
    return tl, ta, tf, total_n


def _plot(history, ckpt_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if len(history) < 2:
        return
    st = [h["step"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Final 1-Epoch KITScenes Training", fontsize=11)
    ax1.plot(st, [h["val_loss"] for h in history], "r-s", label="Val Loss (single tar)")
    ax1.set_xlabel("Checkpoint"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(st, [h["val_ade"] for h in history], "r-s", label="ADE (m)")
    ax2.plot(st, [h["val_fde"] for h in history], color="orange", marker="^", label="FDE (m)")
    ax2.set_xlabel("Checkpoint"); ax2.set_ylabel("Meters"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{ckpt_dir}/final_curve.png", dpi=120); plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eta-min", type=float, default=1e-6)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=149)
    p.add_argument("--scenes-per-ckpt", type=int, default=2)
    p.add_argument("--cache-dir", type=str, default="data/cache")
    p.add_argument("--checkpoint-dir", type=str, default="./checkpoints_full")
    p.add_argument("--split-file", type=str, default="data/final_split.json")
    p.add_argument("--resume", type=str, default="./checkpoints_full/best.pt")
    p.add_argument("--log-file", type=str, default="./checkpoints_full/tar_log.txt")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda")
    print(f"Device: {device}")

    with open(args.split_file) as f:
        split = json.load(f)
    train_ids = split["train"]
    val_ids = split["val"]
    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}")

    root = Path(args.cache_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    _clean_partial(root)

    # ---- Document tar usage ----
    with open(args.log_file, "w") as fh:
        fh.write(f"=== TRAINING TARS ({len(train_ids)}) ===\n")
        for sid in train_ids:
            fh.write(f"{sid}\n")
        fh.write(f"\n=== VALIDATION TARS ({len(val_ids)}) ===\n")
        for sid in val_ids:
            fh.write(f"{sid}\n")
    print(f"Tar log: {args.log_file}")

    # ---- Model (resume from best.pt) ----
    print("\n=== Model ===")
    model = AutoE2E(is_pretrained=True, enable_world_model=False,
                    enable_reasoning=False, map_context_channels=14,
                    route_channels=2, map_fusion_mode="deformable").to(device)
    ckpt = torch.load(args.resume, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    print(f"  Resumed from {args.resume}")
    print(f"  {sum(p.numel() for p in model.parameters())/1e6:.1f}M params (end-to-end, 1 epoch)")

    # ---- Optimizer + Cosine LR ----
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lfn = TrajectoryImitationLoss(loss_type="smooth_l1", temporal_decay=0.95,
                                   signal_scales=(0.778, 0.0350)).to(device)

    # Estimate total optimizer steps for T_max
    # Rough: avg ~50 samples/scene, batch=1, grad_accum=4 → ~12 steps/scene
    EST_SAMPLES_PER_SCENE = 50
    total_steps = len(train_ids) * (EST_SAMPLES_PER_SCENE // args.batch_size) // args.grad_accum
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=args.eta_min)
    print(f"  Cosine LR: {args.lr} -> {args.eta_min} over ~{total_steps} steps")

    # ---- Log ----
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(train_ids).tolist()
    tot_step, ckpt_count = 0, 0
    history = [{"step": 0, "val_loss": None, "val_ade": None, "val_fde": None}]
    best_ade = float("inf")
    SPC = args.scenes_per_ckpt

    print(f"\n{'='*60}")
    print(f"Epoch 1/1  ({len(order)} scenes, ckpt every {SPC})")
    print(f"{'='*60}")

    # Keep the first 2 downloaded scenes so we don't re-download at start
    keep_set = set()

    si = 0
    while si < len(order):
        batch_sids = order[si:si + SPC]
        model.train()

        for sid in batch_sids:
            _download(sid, root)
            dat, n = _load_scene(sid, root)
            if not n:
                _evict(root, keep_set, 2)
                continue
            idxs = list(range(n)); rng.shuffle(idxs)
            B = args.batch_size
            for bs in range(0, n - B + 1, B):
                bi = idxs[bs:bs + B]
                v = dat["visual"][bi].to(device).float()/255.0
                eg = dat["ego"][bi].to(device).float()
                vh = dat["vis_hist"][bi].to(device).float()
                tg = dat["target"][bi].to(device).float()
                cp = dat["cam_params"][bi].to(device).float()
                mc = dat["map_ctx"][bi].to(device).float()
                rm = dat["route_msk"][bi].to(device).float()
                mb = dat["map_v"][bi].to(device)
                rb = dat["route_v"][bi].to(device)
                with torch.amp.autocast("cuda"):
                    out = model(v, mc, vh, eg, route_mask=rm, map_valid=mb,
                                route_valid=rb, projection=PinholeProjection(cp),
                                geometry_type="pinhole", trajectory_target=tg,
                                mode="train")
                loss = lfn(out, tg) / args.grad_accum
                loss.backward()
                tot_step += 1
                if tot_step % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); opt.zero_grad()
                    sched.step()
            del dat

        # Evict trained scenes, keep current batch
        keep_set = set(batch_sids)
        _evict(root, keep_set, 2)

        # ---- Checkpoint + validate on ONE random val tar ----
        ckpt_count += 1
        vsid = val_ids[rng.randint(len(val_ids))]
        _download(vsid, root, split="val")
        vl, va, vf, vn = _validate_single(model, vsid, root, device, lfn)
        _evict(root, keep_set, 2)  # remove val tar after eval

        best = vn > 0 and va < best_ade
        if best:
            best_ade = va
            torch.save(model.state_dict(), f"{args.checkpoint_dir}/best.pt")
        torch.save(model.state_dict(), f"{args.checkpoint_dir}/latest.pt")

        pct = (si + SPC) / len(order) * 100
        lr_now = sched.get_last_lr()[0]
        print(f"  [ckpt {ckpt_count:3d} | {pct:3.0f}%] val_tar={vsid[:12]} n={vn} "
              f"loss={vl:.4f} ADE={va:.2f}m FDE={vf:.2f}m lr={lr_now:.1e} "
              f"{'***' if best else ''}", flush=True)

        history.append({"step": ckpt_count, "val_loss": vl, "val_ade": va,
                        "val_fde": vf, "val_tar": vsid})
        _plot(history, args.checkpoint_dir)
        with open(f"{args.checkpoint_dir}/history.json", "w") as fh:
            json.dump(history, fh)

        si += SPC

    print(f"\nDone. Best ADE: {best_ade:.2f}m ({ckpt_count} checkpoints)")
    _plot(history, args.checkpoint_dir)


if __name__ == "__main__":
    main()
