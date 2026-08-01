"""Full-dataset streaming training with 2-scene mini-batches.

- Downloads 2 scenes, trains, deletes, repeats (341 train scenes)
- Saves checkpoint after every 2 scenes (latest + best)
- 8 fixed validation scenes kept permanently on disk
- Validates after every checkpoint

Usage:
  cd Model/data_parsing/kit_scenes
  python train_full.py --epochs 5
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


def _download(sid: str, data_root: Path):
    d = data_root / "train" / sid
    if d.is_dir():
        return d
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    p = f"datasets/KIT-MRT/KITScenes-Multimodal/data/train/{sid}.tar"
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
    print(f"    ↓ {sid[:12]} ({sz/1e9:.1f}G) dl{time.time()-t0:.0f}s", end="", flush=True)
    t0 = time.time()
    tar = tarfile.open(tmp)
    td = Path(tempfile.mkdtemp(prefix="ks_"))
    tar.extractall(path=td)
    tar.close()
    tmp.unlink()
    (td / sid).rename(d)
    shutil.rmtree(td, ignore_errors=True)
    print(f" x{time.time()-t0:.0f}s")
    return d


def _evict(data_root: Path, keep: set[str], cap: int):
    td = data_root / "train"
    if not td.is_dir():
        return
    items = sorted([x for x in td.iterdir() if x.is_dir() and x.name not in keep],
                   key=lambda x: x.stat().st_mtime)
    for x in items[:max(0, len(items) + len(keep) - cap)]:
        shutil.rmtree(x, ignore_errors=True)


def _load_scene(sid: str, data_root: Path):
    try:
        ds = KitScenesDataset(data_root=str(data_root), split="train",
                              include_navigation=True, scene_ids=[sid])
    except ValueError:
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


def _validate(model, val_ids, data_root, device, loss_fn):
    model.eval()
    tl, ta, tf, total_n = 0.0, 0.0, 0.0, 0
    for sid in val_ids:
        try:
            ds = KitScenesDataset(data_root=str(data_root), split="train",
                                  include_navigation=True, scene_ids=[sid])
        except ValueError:
            continue
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
                            geometry_type="pinhole", trajectory_target=tg,
                            mode="train")
            tl += float(loss_fn(out, tg))
            pn = out.cpu().numpy()[0]; tn = tg.cpu().numpy()[0]
            px = integrate_trajectory(pn[0::2], pn[1::2], v0=10.0)
            tx = integrate_trajectory(tn[0::2], tn[1::2], v0=10.0)
            er = np.linalg.norm(px - tx, axis=1)
            ta += float(er.mean()); tf += float(er[-1]); total_n += 1
    if total_n == 0:
        return 0, 999, 999
    return tl/total_n, ta/total_n, tf/total_n


def _plot_curves(history, ckpt_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if len(history) < 2:
        return
    ep = [h["step"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Full KITScenes Training", fontsize=11)
    ax1.plot(ep, [h["val_loss"] for h in history], "r-s", label="Val Loss")
    ax1.set_xlabel("Checkpoint"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(ep, [h["val_ade"] for h in history], "r-s", label="ADE (m)")
    ax2.plot(ep, [h["val_fde"] for h in history], color="orange", marker="^", label="FDE (m)")
    ax2.set_xlabel("Checkpoint"); ax2.set_ylabel("Meters"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{ckpt_dir}/curve.png", dpi=120); plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=149)
    p.add_argument("--scenes-per-ckpt", type=int, default=2)
    p.add_argument("--cache-dir", type=str, default="data/cache")
    p.add_argument("--checkpoint-dir", type=str, default="./checkpoints_full")
    p.add_argument("--split-file", type=str, default="data/full_split.json")
    p.add_argument("--val-scenes", type=int, default=8)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda")
    print(f"Device: {device}")

    with open(args.split_file) as f:
        split = json.load(f)
    train_ids = split["train"]
    val_ids = split["val"][:args.val_scenes]
    print(f"Train: {len(train_ids)} scenes  Val: {len(val_ids)} scenes")
    print(f"Cache: 2 train + {len(val_ids)} val (~{len(val_ids)*3.5+7:.0f} GB)")

    root = Path(args.cache_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---- Permanent val scenes ----
    print(f"\n=== Val scenes ({len(val_ids)}, permanent) ===")
    val_keep = set(val_ids)
    for sid in val_ids:
        _download(sid, root)
        dat, n = _load_scene(sid, root)
        print(f"  {sid[:12]}... → {n}s")
        del dat
    _evict(root, val_keep, len(val_ids) + 2)

    # ---- Model ----
    print("\n=== Model ===")
    model = AutoE2E(is_pretrained=True, enable_world_model=False,
                    enable_reasoning=False, map_context_channels=14,
                    route_channels=2, map_fusion_mode="deformable").to(device)
    print(f"  {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    lfn = TrajectoryImitationLoss(loss_type="smooth_l1", temporal_decay=0.95,
                                   signal_scales=(0.778, 0.0350)).to(device)

    # ---- Initial val ----
    vl, va, vf = _validate(model, val_ids, root, device, lfn)
    print(f"\n  Initial: loss={vl:.4f} ADE={va:.2f}m FDE={vf:.2f}m")
    best_ade = va
    history = [{"step": 0, "val_loss": vl, "val_ade": va, "val_fde": vf}]
    torch.save(model.state_dict(), f"{args.checkpoint_dir}/best.pt")
    torch.save(model.state_dict(), f"{args.checkpoint_dir}/latest.pt")
    _plot_curves(history, args.checkpoint_dir)

    # ---- Training ----
    rng = np.random.RandomState(args.seed)
    tot_step, ckpt_count = 0, 0
    SPC = args.scenes_per_ckpt

    for ep in range(1, args.epochs + 1):
        order = rng.permutation(train_ids).tolist()
        print(f"\n{'='*60}")
        print(f"Epoch {ep}/{args.epochs}  ({len(order)} scenes, ckpt every {SPC})")
        print(f"{'='*60}")

        si = 0
        while si < len(order):
            # ---- Train on next 2 scenes ----
            batch_sids = order[si:si + SPC]
            model.train()
            for sid in batch_sids:
                _download(sid, root)
                dat, n = _load_scene(sid, root)
                if not n:
                    _evict(root, val_keep, len(val_ids) + 2)
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
                del dat

            # ---- Evict trained scenes, keep val ----
            _evict(root, val_keep, len(val_ids) + 2)

            # ---- Checkpoint + validate ----
            ckpt_count += 1
            vl, va, vf = _validate(model, val_ids, root, device, lfn)
            sch.step(vl)
            best = va < best_ade
            if best:
                best_ade = va
                torch.save(model.state_dict(), f"{args.checkpoint_dir}/best.pt")

            pct = (si + SPC) / len(order) * 100
            sc = si // SPC + 1
            print(f"  [ckpt {ckpt_count:3d} | {pct:3.0f}%] "
                  f"val_loss={vl:.4f} ADE={va:.2f}m FDE={vf:.2f}m {'***' if best else ''}")

            history.append({"step": ckpt_count, "val_loss": vl,
                            "val_ade": va, "val_fde": vf})
            _plot_curves(history, args.checkpoint_dir)

            torch.save(model.state_dict(), f"{args.checkpoint_dir}/latest.pt")
            with open(f"{args.checkpoint_dir}/history.json", "w") as fh:
                json.dump(history, fh)

            si += SPC

    print(f"\nBest ADE: {best_ade:.2f}m  ({ckpt_count} checkpoints)")
    _plot_curves(history, args.checkpoint_dir)


if __name__ == "__main__":
    main()
