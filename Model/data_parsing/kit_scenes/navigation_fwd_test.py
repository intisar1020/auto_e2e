"""Forward pass + ADE/FDE evaluation with 16-channel navigation maps.

Usage:
  cd Model/data_parsing/kit_scenes
  python navigation_fwd_test.py

Downloads 1 scene tar from HF (or uses cached), extracts it, loads data with
navigation maps, runs AutoE2E forward pass, and reports ADE/FDE.
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_MODEL_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_MODEL_DIR))

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"


def _resolve_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        tok = Path.home() / ".cache" / "huggingface" / "token"
        if tok.exists():
            token = tok.read_text().strip()
    return token


def _download_and_extract_scene(scene_id: str, output_root: Path) -> Path:
    """Download one train scene tar from HF and extract to ``output_root/train/{scene_id}/``."""
    token = _resolve_token()
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=token)

    tar_path = f"datasets/KIT-MRT/KITScenes-Multimodal/data/train/{scene_id}.tar"
    dest_dir = output_root / "train" / scene_id
    if dest_dir.is_dir():
        print(f"  Already extracted: {dest_dir}")
        return dest_dir

    print(f"  Streaming {scene_id} from HF ({fs.info(tar_path)['size']/1e9:.1f} GB)...")
    with fs.open(tar_path, "rb") as f:
        data = f.read()

    print(f"  Extracting...")
    tar = tarfile.open(fileobj=io.BytesIO(data))
    tmpdir = Path(tempfile.mkdtemp(prefix="kitscenes_"))
    tar.extractall(path=tmpdir)
    tar.close()
    del data

    # tar extracts as {tmpdir}/{scene_id}/... — move to split dir
    extracted = tmpdir / scene_id
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(extracted), str(dest_dir))
    shutil.rmtree(tmpdir, ignore_errors=True)
    return dest_dir


def _find_long_enough_scene(min_poses: int = 129) -> str:
    """Find the smallest train scene tar with at least ``min_poses`` ego poses."""
    token = _resolve_token()
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=token)
    paths = fs.glob("datasets/KIT-MRT/KITScenes-Multimodal/data/train/*.tar")
    entries = [(fs.info(p).get("size", 0), p) for p in paths]
    entries.sort(key=lambda x: x[0])

    print(f"Scanning {len(entries)} train scenes for one with >={min_poses} poses...")
    for sz, path in entries:
        sid = Path(path).stem
        try:
            # Read first 100 MB of tar — enough for poses.txt + map
            with fs.open(path, "rb") as f:
                header = f.read(100_000_000)
            tar = tarfile.open(fileobj=io.BytesIO(header))
            for m in tar.getmembers():
                if m.name.endswith("/poses.txt"):
                    n = len([l for l in tar.extractfile(m).read().decode().splitlines() if l.strip()])
                    break
            tar.close()
            if n >= min_poses:
                print(f"  Found: {sid} — {n} poses, {sz/1e9:.1f} GB")
                return sid
            else:
                print(f"  Skip {sid}: only {n} poses")
        except Exception:
            continue
    raise RuntimeError(f"No train scene found with >={min_poses} poses (scanned {len(entries)})")


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ------------------------------------------------------------------
    # 1. Locate or download a scene
    # ------------------------------------------------------------------
    # Try pre-extracted scene first, otherwise download
    PRE_EXTRACTED = Path("/tmp/kitscenes_fwd")
    data_root = PRE_EXTRACTED if PRE_EXTRACTED.is_dir() else Path(tempfile.mkdtemp(prefix="kitscenes_fwd_"))
    print(f"Data root: {data_root}")

    if not (data_root / "train").is_dir():
        scene_id = _find_long_enough_scene(min_poses=129)
        _download_and_extract_scene(scene_id, data_root)
    else:
        scenes = list((data_root / "train").iterdir())
        scene_id = scenes[0].name
        print(f"  Using pre-extracted scene: {scene_id}")

    # ------------------------------------------------------------------
    # 2. Load dataset with navigation maps
    # ------------------------------------------------------------------
    from data_parsing.kit_scenes import KitScenesDataset

    print(f"\nLoading dataset with navigation maps...")
    t0 = time.time()
    ds = KitScenesDataset(
        data_root=str(data_root),
        split="train",
        include_navigation=True,
        scene_ids=[scene_id],
    )
    print(f"  {len(ds)} samples loaded in {time.time() - t0:.1f}s")

    batch_size = min(4, len(ds))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ------------------------------------------------------------------
    # 3. Decode navigation maps from the dataset output
    # ------------------------------------------------------------------
    from navigation.artifacts import decode_sample_navigation
    from navigation.geometry import MapChannel

    all_map_context: list[torch.Tensor] = []
    all_route_mask: list[torch.Tensor] = []
    all_map_valid: list[torch.Tensor] = []

    print("\nDecoding navigation rasters from dataset...")
    t0 = time.time()
    for sample_idx in range(len(ds)):
        sample = ds[sample_idx]
        map_np, route_np, _meta = decode_sample_navigation(sample["navigation_members"])

        map_ctx = torch.from_numpy(map_np.copy()).float()   # [14, H, W]
        route = torch.from_numpy(route_np.copy()).float()    # [2, H, W]

        all_map_context.append(map_ctx)
        all_route_mask.append(route)
        all_map_valid.append(torch.tensor(True))
    print(f"  Decoded {len(ds)} samples in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 4. Build model
    # ------------------------------------------------------------------
    print("\nBuilding AutoE2E model...")
    from model_components.auto_e2e import AutoE2E
    from model_components.view_fusion.projection import PinholeProjection

    model = AutoE2E(
        is_pretrained=False,
        enable_world_model=False,
        enable_reasoning=False,
        map_context_channels=14,
        route_channels=2,
    ).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params/1e6:.1f}M")

    # ------------------------------------------------------------------
    # 5. Forward pass + ADE/FDE
    # ------------------------------------------------------------------
    from model_components.losses.trajectory_loss import TrajectoryImitationLoss
    from evaluation.metrics import integrate_trajectory

    loss_fn = TrajectoryImitationLoss(
        loss_type="smooth_l1",
        temporal_decay=0.95,
        signal_scales=(0.778, 0.0350),  # KITScenes scaling
    )

    all_ade: list[float] = []
    all_fde: list[float] = []
    all_loss: list[float] = []

    print(f"\nRunning forward pass on {len(ds)} samples...")
    batch_count = 0
    for batch in loader:
        B = batch["visual_tiles"].shape[0]
        batch_count += B

        visual = batch["visual_tiles"].to(device).float() / 255.0  # uint8 → [0,1]
        ego_hist = batch["egomotion_history"].to(device).float()
        vis_hist = batch["visual_history"].to(device).float()
        target = batch["trajectory_target"].to(device).float()
        cam_params = batch["camera_params"].to(device).float()

        # Map context for this batch
        batch_indices = range(batch_count - B, batch_count)
        map_ctx = torch.stack([all_map_context[i] for i in batch_indices]).to(device)
        route = torch.stack([all_route_mask[i] for i in batch_indices]).to(device)
        map_v = torch.stack([all_map_valid[i] for i in batch_indices]).to(device)
        route_v = torch.zeros(B, dtype=torch.bool, device=device)  # no route

        projection = PinholeProjection(cam_params)

        with torch.no_grad():
            out = model(
                visual,
                map_ctx,
                vis_hist,
                ego_hist,
                route_mask=route,
                map_valid=map_v,
                route_valid=route_v,
                projection=projection,
                geometry_type="pinhole",
                trajectory_target=target,
                mode="train",
            )

        # Compute control-space loss
        loss = loss_fn(out, target)
        all_loss.append(float(loss.item()))

        # Compute ADE/FDE from integrated positions
        pred_np = out.cpu().numpy()
        tgt_np = target.cpu().numpy()

        for i in range(B):
            pred_xy = integrate_trajectory(
                pred_np[i, 0::2], pred_np[i, 1::2], v0=10.0
            )
            tgt_xy = integrate_trajectory(
                tgt_np[i, 0::2], tgt_np[i, 1::2], v0=10.0
            )
            errors = np.linalg.norm(pred_xy - tgt_xy, axis=1)
            all_ade.append(float(errors.mean()))
            all_fde.append(float(errors[-1]))

    # ------------------------------------------------------------------
    # 6. Report
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"Results ({len(ds)} samples, {batch_size} batch, no-pretrain)")
    print(f"{'='*50}")
    print(f"Trajectory loss (control-space):  {np.mean(all_loss):.4f}")
    print(f"ADE (average displacement error): {np.mean(all_ade):.2f} m")
    print(f"FDE (final displacement error):   {np.mean(all_fde):.2f} m")
    print(f"{'='*50}")

    # Cleanup
    shutil.rmtree(data_root, ignore_errors=True)


if __name__ == "__main__":
    main()
