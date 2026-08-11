"""Visualize the route-supervision channels for one scene as a video.

For each frame, renders a 1920x1080 panel showing the loss-only route
supervision fields (decoded from navigation_members["route_supervision.npz"]):

  - distance_to_corridor_m   heatmap (0..30m, clipped)
  - distance_to_drivable_m   heatmap (0..30m)
  - route_heading            arrow field over corridor (from sin/cos where valid)
  - route_heading_valid      binary mask
  - destination              marker on the corridor map (visible -> orange)

Usage:
  cd Model/data_parsing/kit_scenes
  python viz_route_supervision.py --scene <SID> [--fps 3] [--max-frames N]
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
from navigation.artifacts import decode_route_supervision
from navigation.geometry import DEFAULT_NAVIGATION_GEOMETRY

DATA_ROOT = Path(__file__).parent / "datasets"
VIZ_DIR = Path(__file__).parent / "viz"


def _render_frame(fig, sup, s, frame_idx: int) -> np.ndarray:
    corridor = sup.distance_to_corridor_m
    drivable = sup.distance_to_drivable_m
    heading_valid = sup.route_heading_valid.astype(bool)
    heading = np.arctan2(
        np.where(heading_valid, sup.route_heading_sin, 0.0),
        np.where(heading_valid, sup.route_heading_cos, 1.0),
    )
    dest_xy = sup.destination_xy_m
    dest_visible = sup.destination_visible

    gs = fig.add_gridspec(2, 2, left=0.005, right=0.995, top=0.92,
                          bottom=0.03, wspace=0.08, hspace=0.22)

    # ---- distance to corridor ----
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(corridor, cmap="inferno", vmin=0, vmax=15)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"distance_to_corridor_m  F{frame_idx}", fontsize=10,
                 color="white")
    ax.set_xticks([]); ax.set_yticks([])

    # ---- distance to drivable ----
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(drivable, cmap="viridis", vmin=0, vmax=15)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"distance_to_drivable_m  drivable_available={sup.drivable_available}",
                 fontsize=10, color="white")
    ax.set_xticks([]); ax.set_yticks([])

    # ---- route heading arrows over corridor ----
    ax = fig.add_subplot(gs[1, 0])
    base = np.zeros((*corridor.shape, 3), dtype=np.float32)
    base[corridor <= 0.5] = [0.1, 0.1, 0.12]
    ax.imshow(base)
    step = 12
    rows = np.arange(0, corridor.shape[0], step)
    cols = np.arange(0, corridor.shape[1], step)
    gr, gc = np.meshgrid(rows, cols, indexing="ij")
    gv = heading_valid[gr, gc]
    if gv.any():
        gh = heading[gr, gc][gv]
        u = np.cos(gh)
        v = np.sin(gh)
        ax.quiver(gc[gv], gr[gv], u, v, color="cyan", angles="xy",
                  scale_units="xy", scale=3.0, width=0.004)
    ax.set_title("route_heading (sin/cos where valid)", fontsize=10,
                 color="white")
    ax.set_xticks([]); ax.set_yticks([])

    # ---- destination over corridor ----
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(corridor, cmap="gray", vmin=0, vmax=15)
    if dest_visible:
        row, col = DEFAULT_NAVIGATION_GEOMETRY.ego_to_pixel(
            dest_xy.reshape(1, 2)
        )[0]
        if 0 <= row < 256 and 0 <= col < 256:
            ax.scatter(col, row, c="orange", marker="*", s=220, zorder=10,
                       edgecolors="black")
            ax.text(col + 6, row + 6, f"dest ({dest_xy[0]:.0f},{dest_xy[1]:.0f})",
                    fontsize=8, color="orange")
    ax.set_title(f"destination  visible={dest_visible}", fontsize=10,
                 color="white")
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"Route supervision  scene {s['scene_id'][:12]}  frame {frame_idx}",
        fontsize=12, color="white")

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
        data_root=str(DATA_ROOT), split="train",
        include_navigation=True, scene_ids=[args.scene],
    )
    n = len(ds)
    if args.max_frames > 0:
        n = min(n, args.max_frames)
    print(f"scene {args.scene[:12]}  {len(ds)} samples; rendering {n}")

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    out = args.output or str(VIZ_DIR / f"route_sup_{args.scene[:12]}.mp4")
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                             (1920, 1080))

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="black")
    for i in range(n):
        s = ds[i]
        sup = decode_route_supervision(s["navigation_members"])
        bgr = _render_frame(fig, sup, s, ds.frame_index(i))
        writer.write(bgr)
        if i % 10 == 0:
            print(f"  frame {i}/{n} (F{ds.frame_index(i)})")
    writer.release()
    plt.close(fig)
    print(f"saved {out} ({n} frames @ {args.fps}fps)")


if __name__ == "__main__":
    main()
