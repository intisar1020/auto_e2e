"""Generate trajectory prediction video from an already-extracted scene dir.

Adapted from generate_trajectory_video.py:
- Takes an extracted scene directory (data/train/<sid>) instead of a .tar.
- Defaults to the exp-2-subset best checkpoint.
- route_valid=True (all 16 navigation channels live), matching exp-2 training.

Usage:
  cd Model/data_parsing/kit_scenes
  python generate_trajectory_video_dir.py \
    --scene-dir datasets/train/<sid> \
    --output trajectory_video.mp4
"""

from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import cv2, matplotlib, numpy as np, torch
matplotlib.use("Agg")
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.pyplot as plt
from PIL import Image
from kitscenes.poses import load_ego_poses
from kitscenes.sensors import SensorDataLoader

_MODEL_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_MODEL_DIR))

from data_parsing.kit_scenes.camera import (
    CAMERA_NAMES,
    compute_camera_projection_matrices,
)
from data_parsing.kit_scenes.egomotion import pose_yaws, poses_to_arrays
from data_parsing.kit_scenes.map import generate_bev_map_tile
from data_parsing.kit_scenes.navigation import build_scene_navigation
from evaluation.metrics import integrate_trajectory
from model_components.auto_e2e import AutoE2E
from model_components.view_fusion.projection import PinholeProjection

# Display labels matching CAMERA_NAMES order (7 views):
# ["camera_base_front_center", "camera_ring_front", "camera_ring_front_left",
#  "camera_ring_front_right", "camera_ring_rear", "camera_ring_rear_left", "camera_ring_rear_right"]
_SHORT = ["front_ctr", "front", "front_L", "front_R", "rear", "rear_L", "rear_R"]

EXP2_BASE = Path(__file__).parent / "exp-2-baseline"


def _load_camera_frame_scene_dir(loader: SensorDataLoader, frame_idx: int) -> torch.Tensor:
    """Load multiview camera images in exact CAMERA_NAMES slot order (7 views)."""
    cams = []
    for cam in CAMERA_NAMES:
        rgb = loader.get_camera_image(cam, frame_idx)
        img = Image.fromarray(rgb).resize((256, 256), Image.Resampling.BILINEAR)
        cams.append(torch.from_numpy(np.array(img).copy()).permute(2, 0, 1))
    return torch.stack(cams, dim=0)


def _load_camera_calibration(loader: SensorDataLoader):
    """Load front-center camera intrinsics, camera-to-reference extrinsics, and resolution."""
    calib = loader.get_camera_calibration("camera_base_front_center")
    K_front = calib.intrinsic.copy().astype(np.float64)
    # calib.extrinsic is T_cam_to_ref; inverse is T_ref_to_cam (top_lidar_flu -> camera)
    T_ref_to_cam = np.linalg.inv(calib.extrinsic)
    orig_w, orig_h = calib.image_size
    return K_front, T_ref_to_cam, int(orig_w), int(orig_h)


def _project_trajectory_to_camera(
    pred_xy: np.ndarray,
    gt_xy: np.ndarray,
    K: np.ndarray,
    T_ref_to_cam: np.ndarray,
):
    """Project ego-frame XY waypoints (top_lidar_flu) onto front camera pixels.

    Points in ego frame (top_lidar_flu): +X = forward, +Y = left, ground Z = -2.1m.
    T_ref_to_cam transforms top_lidar_flu -> camera 3D space.
    Returns (pred_uv, gt_uv, pred_depth, gt_depth).
    """
    Z_GROUND = -2.1

    def _project(points_ego_xy):
        N = len(points_ego_xy)
        if N == 0:
            return np.zeros((0, 2)), np.zeros(0)
        pts_ego = np.zeros((N, 4), dtype=np.float64)
        pts_ego[:, 0] = points_ego_xy[:, 0]
        pts_ego[:, 1] = points_ego_xy[:, 1]
        pts_ego[:, 2] = Z_GROUND
        pts_ego[:, 3] = 1.0

        pts_cam = (T_ref_to_cam @ pts_ego.T).T  # (N, 4)
        depths = pts_cam[:, 2]

        uv = (K @ pts_cam[:, :3].T).T
        valid = depths > 0.01
        uv[valid, 0] /= uv[valid, 2]
        uv[valid, 1] /= uv[valid, 2]
        uv[~valid] = np.nan
        return uv[:, :2], depths

    pred_uv, pred_d = _project(pred_xy)
    gt_uv, gt_d = _project(gt_xy)
    return pred_uv, gt_uv, pred_d, gt_d


def _draw_path_ribbon(ax, uv_u, uv_v, depths, facecolor, linecolor, alpha=0.50,
                      zorder=4, endpoint_marker=True):
    """Draw a semi-transparent ribbon polygon + centerline on an axis."""
    front = ~np.isnan(uv_u) & ~np.isnan(uv_v) & (depths > 0.01)
    if not front.any():
        return
    uf = uv_u[front]; vf = uv_v[front]; df = depths[front]
    widths = np.clip(35 * (8.0 / np.clip(df, 3.0, 20.0)), 5, 90)
    left_u, left_v, right_u, right_v = [], [], [], []
    for i in range(len(uf)):
        if i < len(uf) - 1:
            dx = uf[i + 1] - uf[i]; dy = vf[i + 1] - vf[i]
        elif i > 0:
            dx = uf[i] - uf[i - 1]; dy = vf[i] - vf[i - 1]
        else:
            dx, dy = 1.0, 0.0
        mag = np.sqrt(dx * dx + dy * dy) + 1e-6
        px, py = -dy / mag, dx / mag
        w = widths[i]
        left_u.append(uf[i] + px * w); left_v.append(vf[i] + py * w)
        right_u.append(uf[i] - px * w); right_v.append(vf[i] - py * w)
    ribbon = np.column_stack([
        np.hstack([left_u, right_u[::-1]]),
        np.hstack([left_v, right_v[::-1]]),
    ])
    from matplotlib.patches import Polygon
    ax.add_patch(Polygon(ribbon, closed=True, facecolor=facecolor,
                          edgecolor="none", alpha=alpha, zorder=zorder))
    ax.plot(uf, vf, "-", color=linecolor, lw=5.0, zorder=zorder + 1)
    if endpoint_marker:
        ax.scatter(uf[-1], vf[-1], c=linecolor, edgecolors="black",
                   marker="o", s=80, zorder=zorder + 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default=str(EXP2_BASE / "checkpoints" / "best.pt"))
    parser.add_argument("--scene-dir", type=str, required=True)
    parser.add_argument("--output", type=str,
                        default=str(EXP2_BASE / "trajectory_video.mp4"))
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--horizon", type=float, default=6.4,
                        help="prediction horizon in seconds (default 6.4)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    scene_dir = Path(args.scene_dir).resolve()
    scene_id = scene_dir.name

    print("Loading model...")
    model = AutoE2E(
        is_pretrained=False, enable_world_model=False, enable_reasoning=False,
        map_context_channels=14, route_channels=2, map_fusion_mode="deformable",
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                     weights_only=True))
    model.to(device).eval()

    loader = SensorDataLoader(scene_dir)
    poses = tuple(load_ego_poses(scene_dir))
    n_poses = len(poses)
    if args.max_frames > 0:
        n_poses = min(n_poses, args.max_frames)
    print(f"Scene: {scene_id[:12]}... — {n_poses} poses")

    print("Loading real camera calibration...")
    cam_proj = compute_camera_projection_matrices(
        loader, camera_names=CAMERA_NAMES, image_size=256,
    )
    K_front, T_ref_to_cam, orig_w, orig_h = _load_camera_calibration(loader)

    print("Building navigation + pre-rendering...")
    egomotion, positions_local = poses_to_arrays(poses)
    yaws = pose_yaws(poses)
    timestamps_ns = np.asarray([p.timestamp_ns for p in poses], dtype=np.int64)

    scene_nav = build_scene_navigation(
        scene_id=scene_id,
        scene_path=scene_dir,
        positions_enu_m=positions_local,
        yaws_rad=yaws,
        timestamps_ns=timestamps_ns,
        source_revision="test",
    )

    nav_cache, route_cache = {}, {}
    for fi in range(n_poses):
        r = scene_nav.raster_for_frame(fi)
        nav_cache[fi] = r.map_context.astype(np.float32)
        route_cache[fi] = r.route_mask.astype(np.float32)
        if fi % 30 == 0:
            print(f"  .. {fi}/{n_poses}")

    print(f"Generating video ({n_poses} frames @ {args.fps} fps)...")
    os.makedirs(Path(args.output).parent, exist_ok=True)
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1920, 1080),
    )

    hz = 10.0
    n_steps = int(round(args.horizon * hz))

    for fi in range(n_poses):
        camera_tiles = _load_camera_frame_scene_dir(loader, fi)
        visual = camera_tiles.unsqueeze(0).to(device).float() / 255.0

        if fi >= 64:
            hist_arr = egomotion[fi - 64 : fi]
        else:
            hist_arr = np.pad(egomotion[:fi], ((64 - fi, 0), (0, 0)), mode="constant")
        ego_hist_np = hist_arr.flatten().astype(np.float32)
        ego_hist = torch.from_numpy(ego_hist_np).unsqueeze(0).to(device)

        map_ctx = torch.from_numpy(nav_cache[fi]).unsqueeze(0).to(device)
        route_msk = torch.from_numpy(route_cache[fi]).unsqueeze(0).to(device)
        vis_hist = torch.zeros(1, 896, device=device)
        proj = PinholeProjection(cam_proj.unsqueeze(0).to(device))

        with torch.no_grad(), torch.amp.autocast("cuda"):
            pred = model(
                visual, map_ctx, vis_hist, ego_hist,
                route_mask=route_msk,
                map_valid=torch.ones(1, dtype=torch.bool, device=device),
                route_valid=torch.ones(1, dtype=torch.bool, device=device),
                projection=proj, geometry_type="pinhole", mode="infer",
            )

        pred_np = pred.cpu().numpy()[0]
        ego_x = float(positions_local[fi, 0])
        ego_y = float(positions_local[fi, 1])
        ego_yaw = float(yaws[fi])
        v0 = float(egomotion[fi, 0])

        pred_sub = pred_np[:n_steps * 2]
        pred_xy = integrate_trajectory(pred_sub[0::2], pred_sub[1::2], v0=v0)

        end_idx = min(fi + 1 + n_steps, n_poses)
        gt_len = max(0, end_idx - (fi + 1))
        gt_xy = np.zeros((gt_len, 2))
        for t_off, t_frame in enumerate(range(fi + 1, end_idx)):
            dx = float(positions_local[t_frame, 0]) - ego_x
            dy = float(positions_local[t_frame, 1]) - ego_y
            c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
            gt_xy[t_off, 0] = c * dx - s * dy
            gt_xy[t_off, 1] = s * dx + c * dy

        min_len = min(len(pred_xy), gt_len)
        pred_xy_viz = pred_xy[:min_len]
        gt_xy_viz = gt_xy[:min_len]

        pred_uv, gt_uv, pred_depth, gt_depth = _project_trajectory_to_camera(
            pred_xy_viz, gt_xy_viz, K_front, T_ref_to_cam,
        )
        bev = generate_bev_map_tile(
            scene_path=scene_dir, ego_x=ego_x, ego_y=ego_y, ego_yaw=ego_yaw,
            canvas_size=500, radius_meters=80.0,
        )
        if bev is None:
            bev = np.full((500, 500, 3), 255, dtype=np.uint8)

        cam_np = camera_tiles.numpy()
        scale_uv = 256.0 / orig_w
        front_img = cam_np[0].transpose(1, 2, 0)

        fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="black")

        # ---- Large front-center hero camera with waypoint overlay ----
        gs_hero = GridSpec(1, 1, figure=fig, left=0.01, right=0.63, top=0.97,
                           bottom=0.03)
        ax_hero = fig.add_subplot(gs_hero[0])
        ax_hero.imshow(front_img)
        ax_hero.set_title("FRONT CENTER — blue=Pred  red=GT", fontsize=13,
                          color="white")
        ax_hero.axis("off")

        gu = gt_uv[:, 0] * scale_uv
        gv = gt_uv[:, 1] * scale_uv
        _draw_path_ribbon(ax_hero, gu, gv, gt_depth, facecolor="#dc2626",
                          linecolor="#ff4444", alpha=0.55, zorder=3,
                          endpoint_marker=False)

        pu = pred_uv[:, 0] * scale_uv
        pv = pred_uv[:, 1] * scale_uv
        _draw_path_ribbon(ax_hero, pu, pv, pred_depth, facecolor="#3b82f6",
                          linecolor="#1f6fff", alpha=0.60, zorder=5)

        from matplotlib.lines import Line2D
        ax_hero.legend(handles=[
            Line2D([0],[0], color="#1f6fff", lw=3, label="Pred"),
            Line2D([0],[0], color="#ff4444", lw=3, label="GT"),
        ], loc="lower left", fontsize=8, facecolor="black", labelcolor="white")

        # ---- Right column: BEV (top) + mini surround cameras (bottom) ----
        gs_right = GridSpec(2, 1, figure=fig, left=0.66, right=0.99, top=0.97,
                            bottom=0.03, hspace=0.12)

        ax_bev = fig.add_subplot(gs_right[0])
        ax_bev.imshow(bev)
        s = 500.0 / 160.0; cx, cy = 250.0, 250.0
        ade = float(np.linalg.norm(pred_xy_viz[:, :2] - gt_xy_viz, axis=1).mean()) if min_len > 0 else 0.0

        # Note: map.py maps +X_ego (fwd) -> row = cy - x*s, +Y_ego (left) -> col = cx - y*s
        ax_bev.plot(cx - pred_xy_viz[:, 1] * s, cy - pred_xy_viz[:, 0] * s, "b-", lw=2.5,
                    label=f"Pred (ADE={ade:.2f}m)")
        ax_bev.plot(cx - gt_xy_viz[:, 1] * s, cy - gt_xy_viz[:, 0] * s, "r-", lw=2.5,
                    label="GT")
        if min_len > 0:
            ax_bev.scatter(cx - pred_xy_viz[-1, 1] * s, cy - pred_xy_viz[-1, 0] * s,
                           c="blue", marker="x", s=80, zorder=10)
            ax_bev.scatter(cx - gt_xy_viz[-1, 1] * s, cy - gt_xy_viz[-1, 0] * s,
                           c="red", marker="x", s=80, zorder=10)
        ax_bev.scatter(cx, cy, c="white", marker="^", s=110, zorder=15,
                       edgecolors="black")
        ax_bev.set_title(f"BEV  F{fi+1}/{n_poses}  Speed: {v0:.1f} m/s", fontsize=10, color="white")
        ax_bev.legend(loc="upper left", fontsize=7, facecolor="black",
                      labelcolor="white")
        ax_bev.axis("off")

        # Surround mini views: indices 1..6 corresponding to CAMERA_NAMES[1..6]
        gs_mini = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_right[1],
                                           wspace=0.05, hspace=0.10)
        mini_views = [1, 2, 3, 4, 5, 6]  # skip index 0 (front-center)
        for i, idx in enumerate(mini_views):
            ax = fig.add_subplot(gs_mini[i // 3, i % 3])
            ax.imshow(cam_np[idx].transpose(1, 2, 0))
            ax.set_title(_SHORT[idx], fontsize=7, color="white")
            ax.axis("off")

        fig.canvas.draw()
        frame = np.array(fig.canvas.buffer_rgba())[:, :, :3]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if frame_bgr.shape[1] != 1920 or frame_bgr.shape[0] != 1080:
            frame_bgr = cv2.resize(frame_bgr, (1920, 1080))
        writer.write(frame_bgr)
        plt.close(fig)

        if fi % 10 == 0:
            print(f"  Frame {fi}/{n_poses}")

    writer.release()
    sz = os.path.getsize(args.output) / 1e6
    print(f"\nSaved: {args.output} ({sz:.1f} MB, {n_poses} frames, "
          f"{n_poses/args.fps:.0f}s)")


if __name__ == "__main__":
    main()
