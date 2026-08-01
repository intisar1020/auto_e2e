"""Generate trajectory prediction video with real calibration + ego history.

Usage:
  cd Model/data_parsing/kit_scenes
  python generate_trajectory_video.py \
    --checkpoint checkpoints/best.pt \
    --tar data/73197a6d-fd55-2fd2-4a47-ddb3ff3b7db7.tar \
    --output trajectory_video.mp4
"""

from __future__ import annotations

import argparse, io, json, os, shutil, sys, tarfile
from pathlib import Path

import cv2, matplotlib, numpy as np, torch
matplotlib.use("Agg")
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from PIL import Image

_MODEL_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_MODEL_DIR))

from data_parsing.kit_scenes.camera import compute_camera_projection_matrices
from data_parsing.kit_scenes.map import _cached_scene_map, generate_bev_map_tile
from data_parsing.kit_scenes.stream_debug_test import (
    _extract_map_to_tempdir, _parse_poses_from_tar, _quaternion_to_yaw,
)
from evaluation.metrics import integrate_trajectory
from model_components.auto_e2e import AutoE2E
from model_components.view_fusion.projection import PinholeProjection
from navigation.lanelet2_adapter import Lanelet2MapAdapter
from navigation.rasterizer import EgoPose, NativeNavigationRasterizer

_CAM_NAMES = [
    "camera_base_front_center",
    "camera_ring_front_left", "camera_ring_front", "camera_ring_front_right",
    "camera_ring_rear_left", "camera_ring_rear", "camera_ring_rear_right",
]
_SHORT = ["front_ctr", "front_L", "front", "front_R", "rear_L", "rear", "rear_R"]


def _load_camera_frame_scene_dir(scene_dir, scene_id, frame_idx):
    """Load cameras from extracted scene directory using the SDK."""
    from kitscenes.sensors import SensorDataLoader
    loader = SensorDataLoader(Path(scene_dir))
    cams = []
    for cam in _CAM_NAMES:
        rgb = loader.get_camera_image(cam, frame_idx)
        img = Image.fromarray(rgb).resize((256, 256), Image.Resampling.BILINEAR)
        cams.append(torch.from_numpy(np.array(img).copy()).permute(2, 0, 1))
    return torch.stack(cams, dim=0)


def _load_camera_calibration(scene_dir):
    """Load intrinsics + extrinsics for front camera from calib.json."""
    calib_path = Path(scene_dir) / "calibration" / "calib.json"
    calib = json.loads(calib_path.read_text())

    front = calib["camera_base_front_center"]
    T_cam_to_ref = np.array(front["T_to_reference"])
    T_ref_to_cam = np.linalg.inv(T_cam_to_ref)

    intr = front["intrinsics"]
    f = intr["focal_length"]
    cu = intr["principal_point_u"]
    cv = intr["principal_point_v"]
    K = np.array([[f, 0, cu], [0, f, cv], [0, 0, 1]], dtype=np.float64)

    res = front["resolution"]
    orig_w, orig_h = int(res["width"]), int(res["height"])
    return K, T_ref_to_cam, orig_w, orig_h


def _load_projection_from_scene(scene_dir, image_size=256):
    """Compute real 3x4 projection matrices from scene calibration."""
    from kitscenes.sensors import SensorDataLoader
    loader = SensorDataLoader(Path(scene_dir))
    return compute_camera_projection_matrices(
        loader, camera_names=_CAM_NAMES, image_size=image_size,
    )


def _project_trajectory_to_camera(
    pred_xy, gt_xy, ego_x, ego_y, ego_yaw, K, T_ref_to_cam, img_size=256,
):
    """Project predicted + GT XY (ego-frame) onto front camera pixels.

    XY trajectory is in ego frame (forward=X, left=Y). Convert to reference
    frame using ego pose, then project via K @ T_ref_to_cam.
    """
    Z_GROUND = -2.1  # ground plane z in reference frame

    def _project(points_ego_xy):
        """points_ego_xy: (N, 2) in ego frame."""
        N = len(points_ego_xy)
        # Ego → reference transform
        c, s = np.cos(ego_yaw), np.sin(ego_yaw)
        pts_ref = np.zeros((N, 3))
        pts_ref[:, 0] = c * points_ego_xy[:, 0] - s * points_ego_xy[:, 1] + ego_x
        pts_ref[:, 1] = s * points_ego_xy[:, 0] + c * points_ego_xy[:, 1] + ego_y
        pts_ref[:, 2] = Z_GROUND

        # Reference → camera 3D
        pts_h = np.hstack([pts_ref, np.ones((N, 1))])  # (N, 4)
        pts_cam = (T_ref_to_cam @ pts_h.T).T  # (N, 4)

        # Perspective division + intrinsics
        uv = (K @ pts_cam[:, :3].T).T  # (N, 3)
        uv[:, 0] /= uv[:, 2]
        uv[:, 1] /= uv[:, 2]
        return uv[:, :2]  # (N, 2) pixel coords

    pred_uv = _project(pred_xy)
    gt_uv = _project(gt_xy)

    # Scale from original resolution to img_size
    # (We need to know the original image size to scale correctly)
    return pred_uv, gt_uv
    """Compute real 3x4 projection matrices from scene calibration."""
    from kitscenes.sensors import SensorDataLoader
    loader = SensorDataLoader(Path(scene_dir))
    return compute_camera_projection_matrices(
        loader, camera_names=_CAM_NAMES, image_size=image_size,
    )  # [V, 3, 4]


def _build_egomotion_history(poses, frame_idx, num_history=64):
    """Build egomotion history [256] = 64 timesteps × 4 signals.

    Signal order: speed, acceleration, yaw_rate, curvature (matches egomotion.py).
    """
    if frame_idx < 2:
        return np.zeros(256, dtype=np.float32)

    start = max(0, frame_idx - num_history)
    hist_poses = poses[start : frame_idx + 1]

    speeds, accels, yaw_rates, curvatures = [], [], [], []
    prev_speed, prev_yaw = 10.0, _quaternion_to_yaw(*hist_poses[0, 4:8])

    for i in range(1, len(hist_poses)):
        dt = max(float(hist_poses[i, 0] - hist_poses[i - 1, 0]), 0.01)
        dx = float(hist_poses[i, 1] - hist_poses[i - 1, 1])
        dy = float(hist_poses[i, 2] - hist_poses[i - 1, 2])
        dist = np.sqrt(dx*dx + dy*dy)
        speed = dist / dt
        accel = (speed - prev_speed) / dt
        yaw = _quaternion_to_yaw(*hist_poses[i, 4:8])
        dyaw = yaw - prev_yaw
        while dyaw > np.pi: dyaw -= 2*np.pi
        while dyaw < -np.pi: dyaw += 2*np.pi
        yaw_rate = dyaw / dt
        curvature = dyaw / max(dist, 0.01)

        speeds.append(speed)
        accels.append(accel)
        yaw_rates.append(yaw_rate)
        curvatures.append(curvature)
        prev_speed = speed
        prev_yaw = yaw

    # Pad/trim to num_history timesteps
    for arr in (speeds, accels, yaw_rates, curvatures):
        if len(arr) < num_history:
            arr[:0] = [0.0] * (num_history - len(arr))
        else:
            arr[:] = arr[-num_history:]

    # Interleave: speed, accel, yaw_rate, curvature per timestep
    hist = np.zeros(256, dtype=np.float32)
    for t in range(min(num_history, len(speeds))):
        base = t * 4
        hist[base + 0] = speeds[t]
        hist[base + 1] = accels[t]
        hist[base + 2] = yaw_rates[t]
        hist[base + 3] = curvatures[t]
    return hist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tar", type=str, required=True)
    parser.add_argument("--output", type=str, default="trajectory_video.mp4")
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Model ----
    print("Loading model...")
    model = AutoE2E(
        is_pretrained=False, enable_world_model=False, enable_reasoning=False,
        map_context_channels=14, route_channels=2, map_fusion_mode="deformable",
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.to(device).eval()

    # ---- Extract scene ----
    print(f"Extracting {args.tar}...")
    tar = tarfile.open(args.tar)
    scene_id = Path(args.tar).stem
    scene_dir = _extract_map_to_tempdir(tar, scene_id)
    poses = _parse_poses_from_tar(tar)

    # Extract all files to scene_dir (needed by SDK for camera loading + calibration)
    print("  Extracting full scene (need calibration + images for SDK)...")
    tar.extractall(path=scene_dir.parent)
    tar.close()
    # The tar extracted as {scene_dir.parent}/{scene_id}/
    full_scene = scene_dir.parent / scene_id

    n_poses = len(poses)
    if args.max_frames > 0:
        n_poses = min(n_poses, args.max_frames)
    print(f"  Scene: {scene_id[:12]}... — {n_poses} poses")

    # ---- Real camera projection ----
    print("Loading real camera calibration...")
    cam_proj = _load_projection_from_scene(full_scene)  # [7, 3, 4]
    K_front, T_ref_to_cam, orig_w, orig_h = _load_camera_calibration(full_scene)

    # ---- Navigation pre-render ----
    print("Building navigation + pre-rendering...")
    rasterizer = NativeNavigationRasterizer()
    scene_map = _cached_scene_map(full_scene)
    nav_map = Lanelet2MapAdapter(
        scene_map, map_version=f"kitscenes:{scene_id}", map_sha256="0"*64,
        frame_id=f"kitscenes:{scene_id}:local_enu", source_revision="test",
    ).extract()

    nav_cache, route_cache = {}, {}
    for fi in range(n_poses):
        pose = EgoPose(
            x_enu_m=float(poses[fi, 1]), y_enu_m=float(poses[fi, 2]),
            yaw_rad=_quaternion_to_yaw(*poses[fi, 4:8]),
            timestamp_ns=int(poses[fi, 0] * 1e9),
        )
        r = rasterizer.render(nav_map, None, pose)
        nav_cache[fi] = r.map_context.astype(np.float32)
        route_cache[fi] = r.route_mask.astype(np.float32)
        if fi % 30 == 0:
            print(f"  .. {fi}/{n_poses}")

    # ---- Video loop ----
    print(f"Generating video ({n_poses} frames @ {args.fps} fps)...")
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1920, 1080),
    )

    for fi in range(n_poses):
        # Load camera images via SDK (uses real calibration paths)
        camera_tiles = _load_camera_frame_scene_dir(full_scene, scene_id, fi)
        visual = camera_tiles.unsqueeze(0).to(device).float() / 255.0

        # Real ego history from poses
        ego_hist_np = _build_egomotion_history(poses, fi)
        ego_hist = torch.from_numpy(ego_hist_np).unsqueeze(0).to(device)

        # Navigation
        map_ctx = torch.from_numpy(nav_cache[fi]).unsqueeze(0).to(device)
        route_msk = torch.from_numpy(route_cache[fi]).unsqueeze(0).to(device)
        vis_hist = torch.zeros(1, 896, device=device)

        # Real projection (batch it)
        proj = PinholeProjection(cam_proj.unsqueeze(0).to(device))

        with torch.no_grad(), torch.amp.autocast("cuda"):
            pred = model(
                visual, map_ctx, vis_hist, ego_hist,
                route_mask=route_msk,
                map_valid=torch.ones(1, dtype=torch.bool, device=device),
                route_valid=torch.zeros(1, dtype=torch.bool, device=device),
                projection=proj, geometry_type="pinhole", mode="infer",
            )

        pred_np = pred.cpu().numpy()[0]
        ego_x = float(poses[fi, 1])
        ego_y = float(poses[fi, 2])
        ego_yaw = _quaternion_to_yaw(*poses[fi, 4:8])

        # GT forward path
        end_idx = min(fi + 64, n_poses)
        gt_len = end_idx - fi
        gt_xy = np.zeros((gt_len, 2))
        for t in range(fi, end_idx):
            dx = float(poses[t, 1]) - ego_x
            dy = float(poses[t, 2]) - ego_y
            c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
            gt_xy[t - fi, 0] = c * dx - s * dy
            gt_xy[t - fi, 1] = s * dx + c * dy

        pred_xy = integrate_trajectory(pred_np[0::2], pred_np[1::2], v0=10.0)

        # Trim both to min length for ADE
        min_len = min(len(pred_xy), gt_len)
        pred_xy_viz = pred_xy[:min_len]

        # Project trajectory to front camera pixels
        pred_uv, gt_uv = _project_trajectory_to_camera(
            pred_xy, gt_xy, ego_x, ego_y, ego_yaw,
            K_front, T_ref_to_cam, img_size=256,
        )

        # BEV map
        bev = generate_bev_map_tile(
            scene_path=full_scene, ego_x=ego_x, ego_y=ego_y, ego_yaw=ego_yaw,
            canvas_size=500, radius_meters=80.0,
        )

        # ---- Draw ----
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="black")

        # Cameras
        gs_cam = GridSpec(2, 4, figure=fig, left=0.02, right=0.52, top=0.97, bottom=0.03,
                          wspace=0.05, hspace=0.08)
        cam_np = camera_tiles.numpy()
        scale_uv = 256.0 / orig_w
        for i, (r, c) in enumerate([(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2)]):
            ax = fig.add_subplot(gs_cam[r, c])
            ax.imshow(cam_np[i].transpose(1, 2, 0))

            # Overlay trajectory on front-center camera (i==0)
            if i == 0:
                # Scale UV coords to resized image
                pu = pred_uv[:, 0] * scale_uv
                pv = pred_uv[:, 1] * scale_uv
                gu = gt_uv[:, 0] * scale_uv
                gv = gt_uv[:, 1] * scale_uv
                ax.plot(pu, pv, "b-", lw=2, alpha=0.8)
                ax.plot(gu, gv, "r-", lw=2, alpha=0.8)
                ax.scatter(pu[-1], pv[-1], c="blue", marker="x", s=60, zorder=10)
                ax.scatter(gu[-1], gv[-1], c="red", marker="x", s=60, zorder=10)

            ax.set_title(_SHORT[i], fontsize=7, color="white")
            ax.axis("off")

        # BEV
        gs_bev = GridSpec(1, 1, figure=fig, left=0.55, right=0.98, top=0.97, bottom=0.03)
        ax_bev = fig.add_subplot(gs_bev[0])
        ax_bev.imshow(bev)

        s = 500 / 160.0; cx, cy = 250, 250
        ade = float(np.linalg.norm(pred_xy_viz[:, :2] - gt_xy[:min_len], axis=1).mean())

        ax_bev.plot(cx + pred_xy[:,1]*s, cy - pred_xy[:,0]*s, "b-", lw=2.5, label=f"Pred (ADE={ade:.1f}m)")
        ax_bev.scatter(cx + pred_xy[-1,1]*s, cy - pred_xy[-1,0]*s, c="blue", marker="x", s=100, zorder=10)
        ax_bev.plot(cx + gt_xy[:,1]*s, cy - gt_xy[:,0]*s, "r-", lw=2.5, label="GT")
        ax_bev.scatter(cx + gt_xy[-1,1]*s, cy - gt_xy[-1,0]*s, c="red", marker="x", s=100, zorder=10)
        ax_bev.scatter(cx, cy, c="white", marker="^", s=120, zorder=15, edgecolors="black")
        ax_bev.set_title(f"F{fi+1}/{n_poses}  ({ego_x:.0f},{ego_y:.0f})", fontsize=10, color="white")
        ax_bev.axis("off")

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
    shutil.rmtree(scene_dir.parent, ignore_errors=True)

    sz = os.path.getsize(args.output) / 1e6
    print(f"\nSaved: {args.output} ({sz:.1f} MB, {n_poses} frames, {n_poses/args.fps:.0f}s)")


if __name__ == "__main__":
    main()
