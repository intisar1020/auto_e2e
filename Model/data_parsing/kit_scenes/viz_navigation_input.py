"""Visualize navigation input as a video for one extracted scene.

For every frame in a scene, renders a single 1920x1080 panel:
  - Front-center camera (left)
  - 14-channel BEV map_context grid (top-right, 4x4)
  - route_mask: SELECTED_CORRIDOR + DESTINATION + overlay (bottom-right)

Uses decode_sample_navigation, so it shows the exact tensor the model consumes.

Usage:
  cd Model/data_parsing/kit_scenes
  python viz_navigation_input.py --scene <SID> [--fps 3] [--max-frames N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_MODEL_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_MODEL_DIR))

from data_parsing.kit_scenes import KitScenesDataset
from navigation.artifacts import decode_sample_navigation
from navigation.geometry import MapChannel, RouteChannel

BASE = Path(__file__).parent / "exp-1-subset"
VIZ_DIR = BASE / "viz"

MAP_NAMES = [c.name for c in MapChannel]
ROUTE_NAMES = [c.name for c in RouteChannel]


def _render_frame(fig, s, frame_idx: int) -> np.ndarray:
    """Render one frame; returns BGR image for cv2 writer."""
    front = s["visual_tiles"][0].permute(1, 2, 0).numpy()  # front-center, uint8
    mc, rm, meta = decode_sample_navigation(s["navigation_members"])

    # ---- Front camera (left) ----
    gs = fig.add_gridspec(1, 2, width_ratios=[0.38, 0.62], left=0.005,
                          right=0.995, top=0.97, bottom=0.03, wspace=0.05)
    ax_cam = fig.add_subplot(gs[0, 0])
    ax_cam.imshow(front)
    ax_cam.set_title(f"FRONT CENTER  F{frame_idx}  scene {s['scene_id'][:12]}",
                     fontsize=10, color="white")
    ax_cam.axis("off")

    # ---- Map context 14-channel grid (top-right) ----
    gs_right = gs[0, 1].subgridspec(2, 1, height_ratios=[0.62, 0.38], hspace=0.18)
    ax_grid = fig.add_subplot(gs_right[0])
    ax_grid.axis("off")
    for i, ch in enumerate(mc):
        ax = fig.add_subplot(gs_right[0].subgridspec(4, 4, hspace=0.15,
                                                     wspace=0.02)[i // 4, i % 4])
        ax.imshow(ch, cmap="gray", vmin=0, vmax=1)
        ax.set_title(MAP_NAMES[i], fontsize=4.5, color="white", pad=0.5)
        ax.set_xticks([]); ax.set_yticks([])
    ax_grid.set_title(
        f"map_context (14ch)  maneuver {meta.get('route_maneuver')}  "
        f"map_valid {meta.get('map_valid')}",
        fontsize=9, color="white")

    # ---- Route mask (bottom-right) ----
    corridor = rm[RouteChannel.SELECTED_CORRIDOR].astype(float)
    dest = rm[RouteChannel.DESTINATION].astype(float)
    canvas = np.zeros((*corridor.shape, 3), dtype=np.float32)
    canvas[corridor > 0] = [0.0, 0.6, 1.0]
    canvas[dest > 0] = [1.0, 0.3, 0.0]
    gs_route = gs_right[1].subgridspec(1, 3, wspace=0.1)
    for ax, ch, name in zip(
        [fig.add_subplot(gs_route[i]) for i in range(3)],
        [corridor, dest, canvas],
        ["SELECTED_CORRIDOR", "DESTINATION", "Corridor + Dest overlay"],
    ):
        ax.imshow(ch, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(name, fontsize=7, color="white")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        f"Navigation input  |  scene {s['scene_id']}  |  "
        f"route_valid {meta.get('route_valid')}  |  "
        f"route_id {meta.get('route_id', '')}",
        fontsize=11, color="white")

    fig.canvas.draw()
    frame = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if bgr.shape[1] != 1920 or bgr.shape[0] != 1080:
        bgr = cv2.resize(bgr, (1920, 1080))
    return bgr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--fps", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    ds = KitScenesDataset(
        data_root=str(BASE / "data"), split="train",
        include_navigation=True, scene_ids=[args.scene],
    )
    n = len(ds)
    if args.max_frames > 0:
        n = min(n, args.max_frames)
    print(f"scene {args.scene[:12]}  {len(ds)} samples; rendering {n}")

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    out = args.output or str(VIZ_DIR / f"nav_input_{args.scene[:12]}.mp4")
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                             (1920, 1080))

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="black")
    for i in range(n):
        s = ds[i]
        frame_idx = ds.frame_index(i)
        bgr = _render_frame(fig, s, frame_idx)
        writer.write(bgr)
        if i % 10 == 0:
            print(f"  frame {i}/{n} (F{frame_idx})")
    writer.release()
    plt.close(fig)
    print(f"saved {out} ({n} frames @ {args.fps}fps)")


if __name__ == "__main__":
    main()
