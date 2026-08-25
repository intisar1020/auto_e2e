"""BEVFormer trajectory video report for an extracted KITScenes scene.

This is the BEVFormer counterpart of ``trajectory_report_local.py``.  The
rendering and manifest layout are reused from that module, but the model is
``BEVFormerTrajectoryNet`` and the script can automatically choose a
contiguous window around the strongest left/right turn in the scene.

Usage:
  python trajectory_report_bevformer.py \
    --checkpoint exp-bevformer/checkpoints/best.pt \
    --scene-dir datasets/train/<scene_uid> \
    --output-dir /tmp/bevformer_turn_report \
    --turn-mode left \
    --window-frames 120 \
    --camera-index 0
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import trajectory_report_local as report
from data_parsing.kit_scenes import KitScenesDataset, NUM_VIEWS
from model_components.view_fusion.projection import PinholeProjection
from Tools.trajectory_visualization import kinematics, rendering

from bevformer_pure_torch import BEVFormerTrajectoryNet
from bevformer_train import _extract_v0, _tensors


def _turn_scores(samples, ds, device):
    """Return (final_lateral_y, max_abs_curvature) for each dataset sample."""
    out = []
    for i, sample in enumerate(samples):
        target = sample["trajectory_target"].numpy()
        controls = target.reshape(64, 2)
        v0 = _extract_v0(sample)
        xy = kinematics.integrate_controls(
            controls,
            v0,
            curvature_sign=kinematics.curvature_sign_for_dataset("kitscenes"),
        )
        lateral = float(xy[-1, 1])
        max_abs_curv = float(np.max(np.abs(controls[:, 1])))
        out.append((i, lateral, max_abs_curv))
    return out


def _choose_turn_window(
    ds: KitScenesDataset,
    *,
    turn_mode: str,
    window_frames: int,
    frame_idx: Optional[int] = None,
):
    """Choose a contiguous sample window centered on a turn."""
    samples = [ds[i] for i in range(len(ds))]
    if frame_idx is not None:
        positions = [i for i, s in enumerate(samples) if int(s["frame_idx"]) == frame_idx]
        if not positions:
            raise ValueError(f"frame_idx {frame_idx} is not a valid sample")
        center_pos = positions[0]
    else:
        scored = _turn_scores(samples, ds, "cpu")
        if turn_mode == "left":
            candidates = [s for s in scored if s[1] > 0]
            key = lambda s: s[1]
        elif turn_mode == "right":
            candidates = [s for s in scored if s[1] < 0]
            key = lambda s: -s[1]
        elif turn_mode == "auto":
            candidates = scored
            key = lambda s: abs(s[1])
        else:
            raise ValueError(f"unknown turn_mode {turn_mode!r}")
        if not candidates:
            candidates = scored
        center_pos = max(candidates, key=key)[0]

    start = max(0, center_pos - window_frames // 2)
    end = min(len(samples), start + window_frames)
    start = max(0, end - window_frames)
    return [samples[i] for i in range(start, end)], start


def _infer_sample(model, sample, device):
    v, eg, vh, tg, cp, mc, rm, mv, rv = _tensors(sample, device)
    with torch.no_grad(), torch.amp.autocast("cuda"):
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
    return out.detach().cpu().numpy()[0], tg.cpu().numpy()[0], _extract_v0(sample)


def generate_report(
    *,
    checkpoint_path: str | Path,
    scene_dir: str | Path,
    output_dir: str | Path,
    camera_index: int = 0,
    fps: float = 10.0,
    device: str = "cuda",
    turn_mode: str = "auto",
    window_frames: int = 120,
    frame_idx: Optional[int] = None,
) -> dict:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f"report output dir must be empty: {destination}")

    ds, scene_id = report._build_dataset(Path(scene_dir).resolve())
    samples, start_pos = _choose_turn_window(
        ds,
        turn_mode=turn_mode,
        window_frames=window_frames,
        frame_idx=frame_idx,
    )
    print(
        f"scene {scene_id}: {len(ds)} samples, rendering {len(samples)} "
        f"frames (start sample {start_pos})"
    )

    model = BEVFormerTrajectoryNet(
        backbone="swin_v2_tiny",
        is_pretrained=False,
        num_views=NUM_VIEWS,
        embed_dim=256,
        bev_h=64,
        bev_w=64,
        num_points_in_pillar=4,
        num_encoder_layers=2,
        num_heads=4,
        num_levels=4,
        num_points=8,
        dropout=0.1,
        map_context_channels=0,
        route_channels=2,
        egomotion_dim=256,
        visual_history_dim=896,
        num_timesteps=64,
        num_signals=2,
        image_size=256,
        use_temporal_state=False,
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    print(
        f"checkpoint {Path(checkpoint_path).name}: "
        f"{len(missing)} missing, {len(unexpected)} unexpected keys"
    )

    import bevformer_train

    bevformer_train.ROUTE_ONLY = True

    scene_samples = []
    for sample in samples:
        pred_controls, target_controls, v0 = _infer_sample(model, sample, device)
        camera_jpeg = report._encode_camera_jpeg(sample["visual_tiles"], camera_index)
        calibration = report._calibration_for(
            sample["camera_params"].numpy(), dataset="kitscenes"
        )
        scene_samples.append(
            {
                "sample_uid": f"{scene_id}:{sample['frame_idx']:06d}",
                "scene_uid": scene_id,
                "frame_idx": int(sample["frame_idx"]),
                "camera_jpeg": camera_jpeg,
                "calibration": calibration,
                "prediction": pred_controls,
                "target": target_controls,
                "v0": v0,
            }
        )

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

    extent = rendering.trajectory_extent(t for _, p, t in prepared for t in (p, t))
    scene_seg = report._safe_scene_segment(scene_id)
    scene_out = destination / "scenes" / scene_seg
    scene_out.mkdir(parents=True)
    video_path = scene_out / "video.mp4"
    thumbnail_path = scene_out / "thumbnail.jpg"

    frames = []
    for s, pred_xy, tgt_xy in prepared:
        shard_sample = report._LocalShardSample(
            camera_jpeg=s["camera_jpeg"],
            calibration=s["calibration"],
            scene_uid=s["scene_uid"],
            frame_idx=s["frame_idx"],
            sample_uid=s["sample_uid"],
            dataset="kitscenes",
        )
        frame = rendering.render_frame(
            shard_sample,
            prediction=pred_xy,
            target=tgt_xy,
            v0=s["v0"],
            base_seed=0,
            extent=extent,
            camera_index=camera_index,
        )
        frames.append(frame)

    if frames:
        frames[0].save(thumbnail_path, format="JPEG", quality=90, optimize=True)
    report._write_mp4_cv2(video_path, frames, fps)

    errors = [
        np.linalg.norm(pred_xy - tgt_xy, axis=1)
        for _, pred_xy, tgt_xy in prepared
    ]
    metrics = {
        "ade_m": float(np.mean([e.mean() for e in errors])),
        "fde_m": float(np.mean([e[-1] for e in errors])),
        "max_error_m": float(max(e.max() for e in errors)),
        "window_turn_mode": turn_mode,
    }
    manifest = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset": "kitscenes",
        "model": "BEVFormerTrajectoryNet",
        "source": {
            "scene": scene_id,
            "checkpoint": Path(checkpoint_path).name,
            "checkpoint_sha256": report._sha256_file(checkpoint_path),
            "scene_dir": Path(scene_dir).name,
        },
        "render": {
            "camera_index": camera_index,
            "fps": fps,
            "turn_mode": turn_mode,
            "window_frames": len(frames),
            "control_contract": kinematics.AOVL_V1_CONTROL_CONTRACT.manifest(),
            "curvature_sign": sign,
            "panel_order": ["camera", "metric_bev"],
            "prediction_source": "bevformer_checkpoint_inference",
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
    print(json.dumps({"output_dir": str(destination), "metrics": metrics}, sort_keys=True))
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--turn-mode", default="auto",
                    choices=["auto", "left", "right"])
    ap.add_argument("--window-frames", type=int, default=120)
    ap.add_argument("--frame-idx", type=int, default=None)
    args = ap.parse_args(argv)
    generate_report(
        checkpoint_path=args.checkpoint,
        scene_dir=args.scene_dir,
        output_dir=args.output_dir,
        camera_index=args.camera_index,
        fps=args.fps,
        device=args.device,
        turn_mode=args.turn_mode,
        window_frames=args.window_frames,
        frame_idx=args.frame_idx,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
