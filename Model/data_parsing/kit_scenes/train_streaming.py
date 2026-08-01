"""Streaming training: download scene → train → delete, one at a time.

Trains on 50 KITScenes scenes with pretrained SwinV2 backbone, caching N
scenes on disk. Designed for limited GPU VRAM (12 GB, fp16).

Usage:
  cd Model/data_parsing/kit_scenes
  python train_streaming.py --epochs 5 --cache-size 3
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


def _download(scene_id: str, data_root: Path):
    d = data_root / "train" / scene_id
    if d.is_dir():
        return d
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    p = f"datasets/KIT-MRT/KITScenes-Multimodal/data/train/{scene_id}.tar"
    sz = fs.info(p)["size"]
    print(f"    ↓ {scene_id[:12]}... ({sz/1e9:.1f}G) ", end="", flush=True)
    t = time.time()
    with fs.open(p, "rb") as fh:
        raw = fh.read()
    print(f"{time.time()-t:.0f}s", end="", flush=True)
    t = time.time()
    tar = tarfile.open(fileobj=io.BytesIO(raw))
    tmp = Path(tempfile.mkdtemp(prefix="ks_"))
    tar.extractall(path=tmp)
    tar.close()
    del raw
    d.parent.mkdir(parents=True, exist_ok=True)
    (tmp / scene_id).rename(d)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f" extract{time.time()-t:.0f}s")
    return d


def _evict(data_root: Path, keep: set[str], cap: int):
    td = data_root / "train"
    if not td.is_dir():
        return
    items = sorted([x for x in td.iterdir() if x.is_dir() and x.name not in keep],
                   key=lambda x: x.stat().st_mtime)
    for x in items[:max(0, len(items) + len(keep) - cap)]:
        shutil.rmtree(x, ignore_errors=True)
        print(f"    ✗ {x.name[:12]}...")


def _load_samples(sid: str, data_root: Path):
    try:
        ds = KitScenesDataset(data_root=str(data_root), split="train",
                              include_navigation=True, scene_ids=[sid])
    except ValueError as e:
        return None, 0
    n = len(ds)
    if n == 0:
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


def _validate(model, data, device, loss_fn):
    model.eval()
    tl, ta, tf = 0.0, 0.0, 0.0
    n = len(data["visual"])
    B = 2
    with torch.no_grad():
        for s in range(0, n, B):
            e = min(s + B, n)
            b = {}
            for k in data:
                if k in ("map_v", "route_v"):
                    b[k] = data[k][s:e].to(device)
                else:
                    b[k] = data[k][s:e].to(device).float()
            b["visual"] = b["visual"] / 255.0
            with torch.amp.autocast("cuda"):
                out = model(b["visual"], b["map_ctx"], b["vis_hist"], b["ego"],
                            route_mask=b["route_msk"], map_valid=b["map_v"],
                            route_valid=b["route_v"],
                            projection=PinholeProjection(b["cam_params"]),
                            geometry_type="pinhole",
                            trajectory_target=b["target"], mode="train")
            tl += float(loss_fn(out, b["target"])) * (e - s)
            pn = out.cpu().numpy(); tn = b["target"].cpu().numpy()
            for i in range(e - s):
                px = integrate_trajectory(pn[i,0::2], pn[i,1::2], v0=10.0)
                tx = integrate_trajectory(tn[i,0::2], tn[i,1::2], v0=10.0)
                er = np.linalg.norm(px - tx, axis=1)
                ta += float(er.mean()); tf += float(er[-1])
    return tl/n, ta/n, tf/n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=149)
    p.add_argument("--cache-size", type=int, default=12)
    p.add_argument("--cache-dir", type=str, default="data/cache")
    p.add_argument("--checkpoint-dir", type=str, default="./checkpoints_stream")
    p.add_argument("--scene-list", type=str, default="data/train_10_scenes.json")
    p.add_argument("--val-scenes", type=int, default=3)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda")
    print(f"Device: {device}")
    root = Path(args.cache_dir)

    with open(args.scene_list) as f:
        all_sids = json.load(f)
    vsids = set(all_sids[:args.val_scenes])
    tsids = all_sids[args.val_scenes:]
    print(f"Train: {len(tsids)} scenes  Val: {len(vsids)} scenes")
    print(f"Cache: max {args.cache_size} scenes (~{args.cache_size*3} GB)")
    print()

    # ---- Validation data (load once, keep) ----
    print("=== Validation ===")
    va = {k: [] for k in ["visual","ego","vis_hist","target","cam_params",
                           "map_ctx","route_msk","map_v","route_v"]}
    for sid in sorted(vsids):
        _download(sid, root)
        d, n = _load_samples(sid, root)
        if n:
            for k in va:
                va[k].append(d[k])
            print(f"  {sid[:12]}... → {n}s")
    val = {k: torch.cat(va[k]) for k in va}
    print(f"  Total val: {len(val['visual'])} samples")
    _evict(root, vsids, args.cache_size)

    # ---- Model (PRETRAINED) ----
    print("\n=== Model ===")
    model = AutoE2E(is_pretrained=True, enable_world_model=False,
                    enable_reasoning=False, map_context_channels=14,
                    route_channels=2, map_fusion_mode="deformable").to(device)
    print(f"  {sum(p.numel() for p in model.parameters())/1e6:.1f}M params  (pretrained, deformable attn)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    lfn = TrajectoryImitationLoss(loss_type="smooth_l1", temporal_decay=0.95,
                                   signal_scales=(0.778, 0.0350)).to(device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---- Initial val ----
    vl, va_v, vf = _validate(model, val, device, lfn)
    print(f"\n  Initial: loss={vl:.4f} ADE={va_v:.2f}m FDE={vf:.2f}m")
    best_ade = va_v
    history = [{"epoch": 0, "train_loss": 0, "val_loss": vl, "val_ade": va_v, "val_fde": vf}]

    # ==== TRAINING ====
    rng = np.random.RandomState(args.seed)
    tot_step = 0

    for ep in range(1, args.epochs + 1):
        order = rng.permutation(tsids).tolist()
        ep_loss, ep_smp, ep_sc = 0.0, 0, 0
        ts = time.time()
        print(f"\n{'='*50}")
        print(f"Epoch {ep}/{args.epochs}  ({len(order)} scenes)")
        print(f"{'='*50}")

        for i, sid in enumerate(order):
            _download(sid, root)
            try:
                dat, n = _load_samples(sid, root)
            except ValueError as e:
                print(f"    ⚠ {sid[:12]}... SKIP ({e})")
                n = 0
                dat = None
            if not n:
                del dat
                _evict(root, vsids, args.cache_size)
                continue
            model.train()
            idx = list(range(n)); rng.shuffle(idx)
            sc_loss, sc_steps = 0.0, 0
            for bs in range(0, n - args.batch_size + 1, args.batch_size):
                bi = idx[bs:bs + args.batch_size]
                b = {}
                for k in dat:
                    if k in ("map_v", "route_v"):
                        b[k] = dat[k][bi].to(device)
                    else:
                        b[k] = dat[k][bi].to(device).float()
                b["visual"] = b["visual"] / 255.0

                with torch.amp.autocast("cuda"):
                    out = model(b["visual"], b["map_ctx"], b["vis_hist"], b["ego"],
                                route_mask=b["route_msk"], map_valid=b["map_v"],
                                route_valid=b["route_v"],
                                projection=PinholeProjection(b["cam_params"]),
                                geometry_type="pinhole",
                                trajectory_target=b["target"], mode="train")
                loss = lfn(out, b["target"]) / args.grad_accum
                loss.backward()
                sc_loss += float(loss) * args.grad_accum * len(bi)
                tot_step += 1; sc_steps += 1
                if tot_step % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); opt.zero_grad()

            ep_loss += sc_loss; ep_smp += n; ep_sc += 1
            del dat
            keep = vsids | {sid}
            _evict(root, keep, args.cache_size)
            pct = (i+1)/len(order)*100
            print(f"  [{pct:3.0f}%] {sid[:12]}... {n}s loss={sc_loss/max(n,1):.4f}")

        vl, va_v, vf = _validate(model, val, device, lfn)
        sch.step(vl)
        best = va_v < best_ade
        if best: best_ade = va_v
        avg = ep_loss / max(ep_smp, 1)
        te = time.time() - ts
        print(f"  --- loss={avg:.4f} val_loss={vl:.4f} ADE={va_v:.2f}m FDE={vf:.2f}m {te:.0f}s {'***' if best else ''}")

        torch.save(model.state_dict(), f"{args.checkpoint_dir}/stream_ep{ep:03d}.pt")
        if best:
            torch.save(model.state_dict(), f"{args.checkpoint_dir}/stream_best.pt")

        history.append({"epoch": ep, "train_loss": avg, "val_loss": vl, "val_ade": va_v, "val_fde": vf})
        _plot(history, args.checkpoint_dir)

    print(f"\nBest ADE: {best_ade:.2f}m")
    _plot(history, args.checkpoint_dir)


def _plot(history, ckpt_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if len(history) < 2:
        return
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Streaming Training — KITScenes 50 scenes, cross-attn fusion", fontsize=11)
    ax1.plot(epochs, [h["train_loss"] for h in history], "b-o", label="Train")
    ax1.plot(epochs, [h["val_loss"] for h in history], "r-s", label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, [h["val_ade"] for h in history], "r-s", label="ADE (m)")
    ax2.plot(epochs, [h["val_fde"] for h in history], color="orange", marker="^", label="FDE (m)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Meters"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{ckpt_dir}/stream_curve.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
