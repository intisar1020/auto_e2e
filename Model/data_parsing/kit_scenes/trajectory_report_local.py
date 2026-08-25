"""Local trajectory report from an exp-6 checkpoint and a KITScenes scene tar.

A local, self-contained counterpart of ``Tools/trajectory_visualization``.
Instead of reading a precomputed canonical AOVL overlay (overlay.bin.gz) plus a
WebDataset shard, it runs the local AutoE2E model checkpoint on a KITScenes
scene tar (or an already-extracted scene dir) and writes the same report layout:

    report/
    ├── manifest.json
    └── scenes/
        └── <scene_uid>/
            ├── thumbnail.jpg
            └── video.mp4

Each video frame is the OG layout: left panel = selected camera, right panel =
a synthetic metric BEV (prediction green, recorded-future blue). Predictions
come from the given checkpoint; the recorded future is the per-sample
``trajectory_target``. Both are integrated with the same control contract and
integrator the OG tool / Console use (64 steps x [acceleration, curvature] at
10 Hz, x_forward_y_left frame).

Usage:
  cd Model/data_parsing/kit_scenes
  python trajectory_report_local.py \
    --checkpoint exp-6-featsize32-gate/checkpoints/best.pt \
    --tar /path/to/<scene_id>.tar \
    --output-dir /tmp/report \
    --camera-index 0 \
    --max-frames-per-scene 300 \
    --fps 10

  # or against an already-extracted scene dir:
  python trajectory_report_local.py --scene-dir datasets/train/<scene_id> ...

The output directory must be empty.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import socket
import sys
import tarfile
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# --- path setup ------------------------------------------------------------
_MODEL_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_MODEL_DIR))
_REPO_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))

from data_parsing.kit_scenes import KitScenesDataset, NUM_VIEWS
from model_components.auto_e2e import AutoE2E
from model_components.view_fusion.projection import PinholeProjection
from Tools.trajectory_visualization import kinematics, rendering

socket.setdefaulttimeout(60)

_KIT_DIR = Path(__file__).parent
_KITSCENES_GROUND_Z_M = -2.1


def _sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_scene_segment(scene_uid: str) -> str:
    import re
    if re.fullmatch(r"^[A-Za-z0-9_-]+$", scene_uid):
        return scene_uid
    digest = hashlib.sha256(scene_uid.encode()).hexdigest()[:16]
    return f"scene-{digest}"


def _write_mp4_cv2(path: Path, frames, fps: float) -> None:
    """Encode frames with OpenCV (mp4v), the available local encoder."""
    import cv2
    writer = None
    for frame in frames:
        arr = np.asarray(frame)
        rgb = arr[..., :3]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if writer is None:
            h, w = bgr.shape[:2]
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h),
            )
        writer.write(bgr)
    if writer is not None:
        writer.release()


def _extract_tar(tar_path: Path, dest_root: Path) -> Path:
    """Extract a KITScenes scene tar into ``dest_root``; return the scene dir."""
    scene_dir = None
    with tarfile.open(tar_path, mode="r:*") as tar:
        # The tar contains one top-level scene directory (<scene_id>/...).
        tops = {
            member.name.split("/")[0]
            for member in tar.getmembers()
            if "/" in member.name or member.isdir()
        }
        tar.extractall(path=dest_root, filter="data")
    candidates = [d for d in dest_root.iterdir() if d.is_dir()]
    if len(candidates) != 1:
        raise ValueError(
            f"tar {tar_path.name} does not contain exactly one scene dir: "
            f"{[d.name for d in candidates]}"
        )
    return candidates[0]


def _build_dataset(scene_dir: Path) -> tuple[KitScenesDataset, str]:
    """Build a single-scene KitScenesDataset from an extracted scene dir.

    The SDK expects scenes under ``<data_root>/train/<scene_id>/``. We create a
    temporary data root with a ``train/`` symlink to the scene dir.
    """
    tmp_root = Path(tempfile.mkdtemp(prefix="ks_report_"))
    train_dir = tmp_root / "train"
    train_dir.mkdir()
    (train_dir / scene_dir.name).symlink_to(scene_dir.resolve())
    ds = KitScenesDataset(
        data_root=str(tmp_root),
        split="train",
        include_navigation=True,
        scene_ids=[scene_dir.name],
    )
    return ds, scene_dir.name


class _LocalShardSample:
    """Minimal stand-in for the OG ``ShardSample`` consumed by ``render_frame``."""

    def __init__(self, *, camera_jpeg, calibration, scene_uid, frame_idx,
                 sample_uid, dataset):
        self.camera_jpeg = camera_jpeg
        self.calibration = calibration
        self.scene_uid = scene_uid
        self.frame_idx = frame_idx
        self.sample_uid = sample_uid
        self.dataset = dataset


def _encode_camera_jpeg(rgb_tensor, camera_index: int) -> bytes:
    """JPEG-encode one camera view from the dataset's uint8 tile tensor."""
    tile = rgb_tensor[camera_index]  # (3, H, W) uint8
    arr = tile.permute(1, 2, 0).numpy()
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _calibration_for(camera_params, dataset: str) -> dict:
    """Build the calibration dict the OG rendering/projection expects."""
    return {
        "dataset": dataset,
        "projection": {
            "type": "pinhole",
            "matrix": [m.tolist() for m in camera_params],  # (V, 3, 4)
            "ground_z_m": _KITSCENES_GROUND_Z_M,
        },
    }


def generate_report(
    *,
    checkpoint_path: str | Path,
    scene_source: str | Path,
    output_dir: str | Path,
    scene_dir_arg: str | Path | None = None,
    camera_index: int = 0,
    max_frames_per_scene: int = 300,
    fps: float = 10.0,
    device: str = "cuda",
) -> dict:
    """Run the exp-6 model over one scene and write the OG-style report."""
    if max_frames_per_scene < 1:
        raise ValueError("max_frames_per_scene must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f"report output dir must be empty: {destination}")

    # --- 1. Resolve the scene (tar or extracted dir) ---
    tmp_root: Path | None = None
    scene_dir: Path
    if scene_dir_arg is not None:
        scene_dir = Path(scene_dir_arg).resolve()
    else:
        tar_path = Path(scene_source).resolve()
        tmp_root = Path(tempfile.mkdtemp(prefix="ks_extract_"))
        scene_dir = _extract_tar(tar_path, tmp_root)

    # --- 2. Build the dataset + model ---
    ds, scene_id = _build_dataset(scene_dir)
    print(f"scene {scene_id}: {len(ds)} samples")

    model = AutoE2E(
        enable_reasoning=False,
        num_views=NUM_VIEWS,
        map_context_channels=0,
        route_channels=2,
        map_fusion_mode="deformable",
        image_feature_size=32,
    ).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"checkpoint: {Path(checkpoint_path).name}")

    import train_main
    train_main.ROUTE_ONLY = True  # route-only input contract (matches exp-6).
    from train_main import _tensors, _extract_v0

    # --- 3. Infer per-sample predictions, integrate trajectories ---
    scene_samples = []
    n = min(len(ds), max_frames_per_scene)
    for idx in range(n):
        sample = ds[idx]
        v, eg, vh, tg, cp, mc, rm, mv, rv = _tensors(sample, device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            out = model(
                v, mc, vh, eg,
                route_mask=rm, map_valid=mv, route_valid=rv,
                projection=PinholeProjection(cp),
                geometry_type="pinhole",
                trajectory_target=tg, mode="infer",
            )
        pred_controls = out.detach().cpu().numpy()[0]           # (128,)
        target_controls = tg.cpu().numpy()[0]                    # (128,)
        v0 = _extract_v0(sample)
        camera_jpeg = _encode_camera_jpeg(sample["visual_tiles"], camera_index)
        calibration = _calibration_for(sample["camera_params"].numpy(),
                                        dataset="kitscenes")
        scene_samples.append({
            "sample_uid": f"{scene_id}:{sample['frame_idx']:06d}",
            "scene_uid": scene_id,
            "frame_idx": int(sample["frame_idx"]),
            "camera_jpeg": camera_jpeg,
            "calibration": calibration,
            "prediction": pred_controls,
            "target": target_controls,
            "v0": v0,
        })

    # --- 4. Integrate + render + write video/thumbnail ---
    sign = kinematics.curvature_sign_for_dataset("kitscenes")
    prepared = []
    for s in scene_samples:
        pred_xy = kinematics.integrate_controls(
            s["prediction"].reshape(64, 2), s["v0"], curvature_sign=sign
        )
        tgt_xy = kinematics.integrate_controls(
            s["target"].reshape(64, 2), s["v0"], curvature_sign=sign
        )
        prepared.append((s, pred_xy, tgt_xy))

    extent = rendering.trajectory_extent(
        t for _, p, t in prepared for t in (p, t)
    )

    scene_seg = _safe_scene_segment(scene_id)
    scene_dir_out = destination / "scenes" / scene_seg
    scene_dir_out.mkdir(parents=True)
    video_path = scene_dir_out / "video.mp4"
    thumbnail_path = scene_dir_out / "thumbnail.jpg"

    base_seed = 0  # deterministic local render; no AOVL seed concept.

    def make_sample(s):
        return _LocalShardSample(
            camera_jpeg=s["camera_jpeg"],
            calibration=s["calibration"],
            scene_uid=s["scene_uid"],
            frame_idx=s["frame_idx"],
            sample_uid=s["sample_uid"],
            dataset="kitscenes",
        )

    frames = []
    for s, pred_xy, tgt_xy in prepared:
        frame = rendering.render_frame(
            make_sample(s),
            prediction=pred_xy,
            target=tgt_xy,
            v0=s["v0"],
            base_seed=base_seed,
            extent=extent,
            camera_index=camera_index,
        )
        frames.append(frame)

    if frames:
        frames[0].save(thumbnail_path, format="JPEG", quality=90, optimize=True)

    if len(frames) > 1:
        _write_mp4_cv2(video_path, frames, fps)
    else:
        # Single frame: still write a valid (1-frame) mp4 so the layout matches.
        _write_mp4_cv2(video_path, frames, fps)

    # --- 5. Metrics + manifest ---
    errors = [
        np.linalg.norm(pred_xy - tgt_xy, axis=1)
        for _, pred_xy, tgt_xy in prepared
    ]
    metrics = {
        "ade_m": float(np.mean([e.mean() for e in errors])),
        "fde_m": float(np.mean([e[-1] for e in errors])),
        "max_error_m": float(max(e.max() for e in errors)),
    }

    manifest = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset": "kitscenes",
        "source": {
            "scene": scene_id,
            "checkpoint": Path(checkpoint_path).name,
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "scene_dir": scene_dir.name,
        },
        "render": {
            "camera_index": camera_index,
            "fps": fps,
            "control_contract": kinematics.AOVL_V1_CONTROL_CONTRACT.manifest(),
            "v0_source": "shard_history_speed",
            "curvature_sign": sign,
            "panel_order": ["camera", "metric_bev"],
            "prediction_source": "local_checkpoint_inference",
        },
        "scene_count": 1,
        "frame_count": len(frames),
        "scenes": [
            {
                "scene_uid": scene_id,
                "start_frame": prepared[0][0]["frame_idx"],
                "end_frame": prepared[-1][0]["frame_idx"],
                "frame_count": len(prepared),
                "sample_uids": [s["sample_uid"] for s, _, _ in prepared],
                "video": str(video_path.relative_to(destination)),
                "thumbnail": str(thumbnail_path.relative_to(destination)),
                "metrics": metrics,
                "bev_extent_m": extent,
            }
        ],
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if tmp_root is not None:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(json.dumps({
        "frame_count": len(frames),
        "scene_count": 1,
        "metrics": metrics,
        "output_dir": str(destination),
    }, sort_keys=True))
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Render an exp-6 checkpoint over a KITScenes scene as an "
                    "OG-style trajectory report (manifest.json + video + thumbnail)."
    )
    ap.add_argument("--checkpoint", required=True,
                    help="exp-6 model checkpoint (.pt)")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--tar", help="KITScenes scene tar to extract")
    source.add_argument("--scene-dir", help="already-extracted scene directory")
    ap.add_argument("--output-dir", required=True,
                    help="empty destination directory")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--max-frames-per-scene", type=int, default=300)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    generate_report(
        checkpoint_path=args.checkpoint,
        scene_source=args.tar,
        scene_dir_arg=args.scene_dir,
        output_dir=args.output_dir,
        camera_index=args.camera_index,
        max_frames_per_scene=args.max_frames_per_scene,
        fps=args.fps,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
