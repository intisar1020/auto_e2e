"""Side-by-side comparison: Lanelet2 BEV tile vs Navigation Map channels.

Usage:
  cd Model/data_parsing/kit_scenes
  python map_comparison.py
"""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

from data_parsing.kit_scenes.map import (
    _cached_scene_map,
    _to_px,
    generate_bev_map_tile,
)
from data_parsing.kit_scenes.stream_debug_test import (
    _extract_map_to_tempdir,
    _parse_poses_from_tar,
    _quaternion_to_yaw,
)
from navigation.geometry import MapChannel
from navigation.lanelet2_adapter import Lanelet2MapAdapter

LOCAL_TAR = Path(__file__).parent / "data" / "73197a6d-fd55-2fd2-4a47-ddb3ff3b7db7.tar"


def _render_navigation_channels(
    nav_map,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    canvas_size: int = 256,
    radius_meters: float = 60.0,
) -> dict[str, np.ndarray]:
    """Render individual navigation map channels as separate 2D arrays."""
    rs = canvas_size * 4  # internal render resolution
    scale = rs / (radius_meters * 2.0)
    ego = np.array([ego_x, ego_y], dtype=np.float64)
    yaw = float(ego_yaw)

    def _polyline_to_px(pts_enu):
        pts = np.array(pts_enu, dtype=np.float64)
        if len(pts) < 2:
            return None
        return _to_px(pts, ego, yaw, scale, rs)

    channels: dict[str, np.ndarray] = {}

    # --- Drivable area ---
    canvas = np.zeros((rs, rs), dtype=np.uint8)
    for poly in nav_map.drivable_polygons:
        pts = poly.points_enu_m
        if len(pts) >= 2:
            px = _polyline_to_px(pts)
            if px is not None:
                cv2.fillPoly(canvas, [px], 255)
    channels["DRIVABLE_AREA"] = cv2.resize(canvas, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA)

    # --- Intersections ---
    canvas = np.zeros((rs, rs), dtype=np.uint8)
    for poly in nav_map.intersection_polygons:
        pts = poly.points_enu_m
        if len(pts) >= 2:
            px = _polyline_to_px(pts)
            if px is not None:
                cv2.fillPoly(canvas, [px], 255)
    channels["INTERSECTION"] = cv2.resize(canvas, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA)

    # --- Lane centerlines ---
    canvas = np.zeros((rs, rs), dtype=np.uint8)
    for cl in nav_map.lane_centerlines:
        pts = cl.points_enu_m
        px = _polyline_to_px(pts)
        if px is not None:
            cv2.polylines(canvas, [px], isClosed=False, color=255, thickness=1, lineType=cv2.LINE_AA)
    channels["LANE_CENTERLINE"] = cv2.resize(canvas, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA)

    # --- Stop lines ---
    canvas = np.zeros((rs, rs), dtype=np.uint8)
    for sl in nav_map.stop_lines:
        pts = sl.points_enu_m
        px = _polyline_to_px(pts)
        if px is not None:
            cv2.polylines(canvas, [px], isClosed=False, color=255, thickness=2, lineType=cv2.LINE_AA)
    channels["STOP_LINE"] = cv2.resize(canvas, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA)

    # --- Lane boundaries ---
    canvas = np.zeros((rs, rs), dtype=np.uint8)
    for lb in nav_map.lane_boundaries:
        pts = lb.points_enu_m
        px = _polyline_to_px(pts)
        if px is not None:
            cv2.polylines(canvas, [px], isClosed=False, color=255, thickness=1, lineType=cv2.LINE_AA)
    channels["LANE_BOUNDARY"] = cv2.resize(canvas, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA)

    # --- Composite (all channels overlaid in color) ---
    composite = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    # Drivable area = gray background
    composite[channels["DRIVABLE_AREA"] > 0] = [60, 60, 60]
    # Intersections = orange
    composite[channels["INTERSECTION"] > 0] = [0, 140, 255]
    # Lane boundaries = light gray
    composite[channels["LANE_BOUNDARY"] > 0] = [180, 180, 180]
    # Centerlines = cyan dashed (just thin line)
    composite[channels["LANE_CENTERLINE"] > 0] = [200, 200, 50]
    # Stop lines = red
    composite[channels["STOP_LINE"] > 0] = [0, 0, 255]

    channels["COMPOSITE"] = composite
    return channels


def main():
    if not LOCAL_TAR.is_file():
        print(f"Local tar not found: {LOCAL_TAR}")
        print("Run stream_debug_test.py first to cache a scene.")
        sys.exit(1)

    scene_id = LOCAL_TAR.stem
    print(f"Reading {scene_id}...")
    tar_bytes = LOCAL_TAR.read_bytes()
    tar = tarfile.open(fileobj=io.BytesIO(tar_bytes))

    # Extract map + poses
    scene_dir = _extract_map_to_tempdir(tar, scene_id)
    poses = _parse_poses_from_tar(tar)

    # Build Lanelet2 scene map
    print("Loading Lanelet2 scene map...")
    scene_map = _cached_scene_map(scene_dir)

    print(f"  {len(scene_map.lanelet_map.laneletLayer)} lanelets")

    # Build navigation map
    print("Extracting navigation map from Lanelet2...")
    adapter = Lanelet2MapAdapter(
        scene_map,
        map_version=f"kitscenes:{scene_id}",
        map_sha256="0" * 64,
        frame_id=f"kitscenes:{scene_id}:local_enu",
        source_revision="test",
    )
    nav_map = adapter.extract()
    print(f"  Drivable polygons: {len(nav_map.drivable_polygons)}")
    print(f"  Lane boundaries:   {len(nav_map.lane_boundaries)}")
    print(f"  Lane centerlines:  {len(nav_map.lane_centerlines)}")
    print(f"  Intersections:     {len(nav_map.intersection_polygons)}")
    print(f"  Stop lines:        {len(nav_map.stop_lines)}")
    print(f"  Crosswalks:        {len(nav_map.crosswalk_polygons)}")
    print(f"  Traffic signals:   {len(nav_map.static_traffic_signals)}")

    # Use frame 35 (mid-scene) for the comparison
    FRAME = 35
    ego_x = float(poses[FRAME, 1])
    ego_y = float(poses[FRAME, 2])
    ego_yaw = _quaternion_to_yaw(*poses[FRAME, 4:8].tolist())
    print(f"\nUsing frame {FRAME}: x={ego_x:.1f}, y={ego_y:.1f}, yaw={np.degrees(ego_yaw):.1f}°")

    # Generate both map types
    print("Rendering Lanelet2 tile...")
    lanelet2_tile = generate_bev_map_tile(
        scene_path=scene_dir,
        ego_x=ego_x,
        ego_y=ego_y,
        ego_yaw=ego_yaw,
        canvas_size=256,
        radius_meters=60.0,
    )

    print("Rendering navigation channels...")
    nav_channels = _render_navigation_channels(
        nav_map,
        ego_x=ego_x,
        ego_y=ego_y,
        ego_yaw=ego_yaw,
        canvas_size=256,
        radius_meters=60.0,
    )

    # --- Plot ---
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    fig.suptitle(
        f"Lanelet2 vs Navigation Map — Scene {scene_id[:8]}... Frame {FRAME}",
        fontsize=13,
        fontweight="bold",
    )

    # Row 0: Lanelet2 takes full width + label
    ax0 = axes[0, 0]
    ax0.imshow(lanelet2_tile)
    ax0.set_title("Lanelet2 Tile (256×256 RGB)", fontsize=10, fontweight="bold",
                  color="darkgreen")
    ax0.axis("off")

    # Lanelet2 info
    ax_info = axes[0, 1]
    ax_info.axis("off")
    ax_info.text(0.05, 0.95,
                 "Lanelet2 Map\n"
                 "─────────────\n"
                 "- Full HD map\n"
                 "- Lane-level detail\n"
                 "- Semantic colors\n"
                 "  (green=borders,\n"
                 "   blue=dividers,\n"
                 "   red=stop lines,\n"
                 "   yellow=crosswalks)\n"
                 "- 257 lanelets\n"
                 "- Raw OSM geometry\n"
                 "- Heavy to parse\n"
                 "- High precision",
                 transform=ax_info.transAxes,
                 fontsize=9, verticalalignment="top",
                 fontfamily="monospace")

    # Lanelet2 contour (B/W edge map for comparison)
    gray = cv2.cvtColor(lanelet2_tile, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    ax0b = axes[0, 2]
    ax0b.imshow(edges, cmap="gray")
    ax0b.set_title("Lanelet2 → Edges", fontsize=10)
    ax0b.axis("off")

    # Navigation composite
    ax0c = axes[0, 3]
    ax0c.imshow(nav_channels["COMPOSITE"])
    ax0c.set_title("Navigation Composite", fontsize=10, fontweight="bold",
                   color="darkorange")
    ax0c.axis("off")

    # Row 1: Individual navigation channels
    channel_names = ["DRIVABLE_AREA", "LANE_BOUNDARY", "LANE_CENTERLINE", "INTERSECTION"]
    channel_colors = ["Greens", "Greys", "Blues", "Oranges"]
    channel_labels = [
        "Drivable Area\n(road mask)",
        "Lane Boundaries\n(polylines)",
        "Lane Centerlines\n(driving path)",
        "Intersections\n(junction zones)",
    ]
    for j, (name, cmap, label) in enumerate(zip(channel_names, channel_colors, channel_labels)):
        ax = axes[1, j]
        ax.imshow(nav_channels[name], cmap=cmap)
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    # Row 2: More channels + info
    ax_stop = axes[2, 0]
    ax_stop.imshow(nav_channels["STOP_LINE"], cmap="Reds")
    ax_stop.set_title("Stop Lines", fontsize=9)
    ax_stop.axis("off")

    # Difference map (where Lanelet2 has edges that nav does/don't)
    nav_edges = cv2.Canny(nav_channels["DRIVABLE_AREA"], 30, 100)
    diff = np.zeros((256, 256, 3), dtype=np.uint8)
    diff[edges > 0] = [0, 255, 0]  # Lanelet2-only edges in green
    diff[nav_edges > 0] = [255, 0, 0]  # Nav-only edges in red
    # Overlap = yellow
    overlap = (edges > 0) & (nav_edges > 0)
    diff[overlap] = [0, 255, 255]
    ax_diff = axes[2, 1]
    ax_diff.imshow(diff)
    ax_diff.set_title("Overlap: Lanelet2(green)\nvs Nav(red)", fontsize=9)
    ax_diff.axis("off")

    # Blank
    axes[2, 2].axis("off")

    # Navigation info
    ax_info2 = axes[2, 3]
    ax_info2.axis("off")
    ax_info2.text(0.05, 0.95,
                  "Navigation Map\n"
                  "──────────────\n"
                  "- Multi-channel\n"
                  "  (14 map + 2 route)\n"
                  "- 256×256 per channel\n"
                  "- 1 m/pixel resolution\n"
                  "- Semantic channels:\n"
                  "  road, intersection,\n"
                  "  centerline, boundary,\n"
                  "  stop line, crosswalk,\n"
                  "  traffic direction\n"
                  "- Lightweight to render\n"
                  "- SD map compatible\n"
                  "- Explicit structure\n"
                  "  (no color decoding)",
                  transform=ax_info2.transAxes,
                  fontsize=9, verticalalignment="top",
                  fontfamily="monospace")

    plt.tight_layout()
    out_path = Path(__file__).parent / "map_comparison_output.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_path}")

    # Cleanup
    import shutil
    shutil.rmtree(scene_dir, ignore_errors=True)
    tar.close()


if __name__ == "__main__":
    main()
