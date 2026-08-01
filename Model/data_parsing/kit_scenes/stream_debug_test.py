"""Minimal streaming test: 6 camera samples + BEV map tile from KITScenes via HF.

**Setup required before running**:
  1. ``hf auth login`` and accept terms at
     https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal
  2. A single scene tar (~3 GB) is streamed from HF and cached in a temp dir.
     Images and map data are extracted in-memory from the tar.

**Usage**:
  cd Model/data_parsing/kit_scenes
  python stream_debug_test.py [--samples N]

If ``kitscenes`` + ``lanelet2`` are installed, ``generate_bev_map_tile`` from
``map.py`` is used. Otherwise a blank canvas with ego-position label is drawn.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from huggingface_hub import HfFileSystem

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

REPO_ID = "KIT-MRT/KITScenes-Multimodal"
TAR_GLOB = f"datasets/{REPO_ID}/data/train/*.tar"

# 6 surround ring cameras in top-to-bottom / left-to-right grid order
RING_CAMERAS = [
    "camera_ring_front_left",
    "camera_ring_front",
    "camera_ring_front_right",
    "camera_ring_rear_left",
    "camera_ring_rear",
    "camera_ring_rear_right",
]

# Grid layout for the 3x3 plot (position in 1-based axes)
GRID_POSITIONS = {
    "camera_ring_front_left": 1,
    "camera_ring_front": 2,
    "camera_ring_front_right": 3,
    "camera_ring_rear_left": 4,
    "camera_ring_rear": 6,
    "camera_ring_rear_right": 7,
}

# ---------------------------------------------------------------------------
# Map rendering
# ---------------------------------------------------------------------------


def _render_map_fallback(
    scene_id: str,
    frame_idx: int,
    canvas_size: int = 256,
) -> np.ndarray:
    """Blank canvas with scene + frame label (used when lanelet2 is absent)."""
    import cv2

    canvas = np.full((canvas_size, canvas_size, 3), 245, dtype=np.uint8)
    cv2.putText(
        canvas,
        f"{scene_id[:8]}...",
        (8, canvas_size // 2 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (100, 100, 100),
        1,
    )
    cv2.putText(
        canvas,
        f"frame {frame_idx}",
        (8, canvas_size // 2 + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (100, 100, 100),
        1,
    )
    return canvas


def _render_bev_tile(
    scene_dir: Path,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    canvas_size: int = 256,
    radius_meters: float = 50.0,
) -> np.ndarray:
    """Try the real ``generate_bev_map_tile``; fall back to a blank canvas."""
    try:
        from map import generate_bev_map_tile

        tile = generate_bev_map_tile(
            scene_path=scene_dir,
            ego_x=ego_x,
            ego_y=ego_y,
            ego_yaw=ego_yaw,
            canvas_size=canvas_size,
            radius_meters=radius_meters,
        )
        if tile is not None:
            return tile
    except ImportError as exc:
        print(f"  [Map] lanelet2/kitscenes not available ({exc}); using fallback canvas")
    except Exception as exc:
        print(f"  [Map] render error: {exc}; using fallback canvas")
    return _render_map_fallback(scene_dir.name, 0, canvas_size)


# ---------------------------------------------------------------------------
# Tar streaming
# ---------------------------------------------------------------------------


def _scene_id_from_tar_path(tar_path: str) -> str:
    return Path(tar_path).stem


def _parse_poses_from_tar(tar: tarfile.TarFile) -> np.ndarray | None:
    """Parse ``poses.txt`` from the tar into (N, 8) array: timestamp, x, y, z, qx, qy, qz, qw."""
    for member in tar.getmembers():
        if member.name.endswith("/poses.txt"):
            text = tar.extractfile(member).read().decode()
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            rows = []
            for ln in lines:
                parts = ln.split()
                if len(parts) >= 8:
                    rows.append([float(p) for p in parts[:8]])
            if rows:
                return np.array(rows, dtype=np.float64)
    return None


def _quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw (Z-rotation) from quaternion ``[qx, qy, qz, qw]``."""
    from scipy.spatial.transform import Rotation
    return float(Rotation.from_quat([qx, qy, qz, qw]).as_euler("ZYX")[0])


def _extract_map_to_tempdir(tar: tarfile.TarFile, scene_id: str) -> Path | None:
    """Extract maps/map.osm and maps/origin.json from the tar into a temp directory.

    Returns the temporary scene directory or None if the map files are absent.
    Handles both `{scene_id}/maps/...` and bare `maps/...` prefixes.
    """
    map_members: list[tuple[str, tarfile.TarInfo]] = []
    for member in tar.getmembers():
        name = member.name
        if name.endswith("/maps/map.osm") or name.endswith("/maps/origin.json"):
            map_members.append((name, member))

    if not map_members:
        print("  [Map] No OSM or origin.json in tar; using fallback canvas")
        return None

    tmpdir = Path(tempfile.mkdtemp(prefix=f"kitscenes_{scene_id}_"))
    for arcname, member in map_members:
        # Strip any leading scene_id dir to get canonical layout: maps/map.osm
        parts = arcname.split("/")
        if len(parts) > 1 and parts[0] == scene_id:
            rel = "/".join(parts[1:])
        else:
            rel = arcname
        dest = tmpdir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(tar.extractfile(member).read())
    return tmpdir


def _extract_images_from_tar(
    tar: tarfile.TarFile,
    scene_id: str,
    num_samples: int,
) -> list[dict]:
    """Extract camera JPEGs for ``num_samples`` frames from an in-memory tar.

    Returns a list of dicts: ``{cam_name: PIL.Image, frame_idx: int, ...}``.
    """
    members_by_name: dict[str, tarfile.TarInfo] = {}
    all_dirs: set[str] = set()
    for member in tar.getmembers():
        members_by_name[member.name] = member
        parts = member.name.split("/")
        if len(parts) >= 2:
            all_dirs.add("/".join(parts[:-1]))

    # Discover the camera directory prefix — it may or may not start with scene_id
    front_cam_dirs = sorted(
        d for d in all_dirs
        if d.endswith("camera_ring_front") and not d.endswith("_left") and not d.endswith("_right")
    )
    if not front_cam_dirs:
        # Try without scene_id prefix
        front_cam_dirs = [
            d for d in all_dirs
            if d == "camera_ring_front"
        ]
    if not front_cam_dirs:
        raise RuntimeError(
            f"No camera_ring_front directory found in tar. "
            f"Top-level dirs: {sorted(set(m.name.split('/')[0] for m in tar.getmembers()))[:10]}"
        )

    front_dir = front_cam_dirs[0]
    tar_root = front_dir.rsplit("/camera_ring_front", 1)[0]
    prefix = f"{tar_root}/" if tar_root else ""

    # Discover available frame indices from the front camera
    frames: list[int] = []
    for name in members_by_name:
        if not name.startswith(f"{prefix}camera_ring_front/"):
            continue
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = Path(name).stem
        try:
            frames.append(int(stem))
        except ValueError:
            parts = stem.split("_")
            for p in reversed(parts):
                try:
                    frames.append(int(p))
                    break
                except ValueError:
                    continue

    if not frames:
        raise RuntimeError(f"No camera frames found in tar for scene {scene_id}")

    frames = sorted(set(frames))
    step = max(1, len(frames) // num_samples)
    selected = frames[::step][:num_samples]

    print(f"  Tar prefix: {prefix!r}")
    print(f"  Scene has {len(frames)} frames; sampling {len(selected)}: {selected}")

    results: list[dict] = []
    for frame_idx in selected:
        sample: dict = {}
        for cam in RING_CAMERAS:
            matched = False
            for ext in (".jpg", ".jpeg", ".png"):
                for fmt in (f"{frame_idx:010d}", f"{frame_idx:06d}", f"{frame_idx:04d}", str(frame_idx)):
                    arcname = f"{prefix}{cam}/{fmt}{ext}"
                    if arcname in members_by_name:
                        raw = tar.extractfile(members_by_name[arcname]).read()
                        sample[cam] = Image.open(io.BytesIO(raw)).convert("RGB")
                        matched = True
                        break
                    # Also try uppercase extension
                    arcname_upper = f"{prefix}{cam}/{fmt}{ext.upper()}"
                    if arcname_upper in members_by_name:
                        raw = tar.extractfile(members_by_name[arcname_upper]).read()
                        sample[cam] = Image.open(io.BytesIO(raw)).convert("RGB")
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                sample[cam] = Image.new("RGB", (640, 480), color=(40, 40, 40))
        sample["frame_idx"] = frame_idx
        results.append(sample)
    return results


def _find_tar_files(fs: HfFileSystem) -> list[str]:
    """List train tar paths, sorted by size (smallest first for quick tests)."""
    paths = fs.glob(TAR_GLOB)
    if not paths:
        raise FileNotFoundError(f"No tar archives found at {TAR_GLOB}")
    entries: list[tuple[int, str]] = []
    for p in paths:
        info = fs.info(p)
        entries.append((info.get("size", 0), p))
    entries.sort(key=lambda x: x[0])  # smallest first
    return [name for _, name in entries]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def debug_stream_and_render(
    num_samples: int = 6,
    local_tar: str | None = None,
) -> None:
    """Stream a KITScenes scene tar from HF, extract images + map, render debug output."""
    if local_tar is not None:
        local_path = Path(local_tar)
        if not local_path.is_file():
            print(f"Local tar not found: {local_path}")
            sys.exit(1)
        print(f"Reading local tar: {local_path} ({local_path.stat().st_size / 1e9:.2f} GB)")
        tar_bytes = local_path.read_bytes()
        scene_id = local_path.stem
        tar = tarfile.open(fileobj=io.BytesIO(tar_bytes))
    else:
        try:
            from huggingface_hub import HfFileSystem
        except ImportError:
            print("huggingface_hub not installed. Run: pip install huggingface_hub")
            sys.exit(1)

        # Pick up token from env / HF cache
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not token:
            token_path = Path.home() / ".cache" / "huggingface" / "token"
            if token_path.exists():
                token = token_path.read_text().strip()
                os.environ["HF_TOKEN"] = token

        print("Initializing Hugging Face virtual filesystem...")
        fs = HfFileSystem(token=token)

        # 1. Find the smallest train tar for quickest download
        print(f"Listing archives in {REPO_ID}/data/train/ ...")
        try:
            tar_paths = _find_tar_files(fs)
        except Exception as exc:
            print(f"Failed to list tar files: {exc}")
            print("Check dataset access: https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal")
            sys.exit(1)

        print(f"Found {len(tar_paths)} training archives")
        tar_path = tar_paths[0]
        scene_id = _scene_id_from_tar_path(tar_path)
        print(f"Selected: {scene_id} ({tar_path})")

        # 2. Stream the tar from HF into memory
        print(f"Streaming tar from HF (this may take a minute for {tar_path})...")
        try:
            with fs.open(tar_path, "rb") as fh:
                tar_bytes = fh.read()
        except Exception as exc:
            print(f"Failed to read tar: {exc}")
            print("You may need to accept the dataset terms at:")
            print("  https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal")
            sys.exit(1)

        print(f"Downloaded {len(tar_bytes) / 1e6:.1f} MB")
        tar = tarfile.open(fileobj=io.BytesIO(tar_bytes))

    # 3. Extract map data to temp dir (for BEV rendering)
    print("Extracting map data...")
    scene_dir = _extract_map_to_tempdir(tar, scene_id)

    # 4. Parse ego poses from poses.txt
    print("Parsing ego poses from poses.txt...")
    poses = _parse_poses_from_tar(tar)
    if poses is not None:
        print(f"  Found {len(poses)} pose rows")
    else:
        print("  No poses.txt found; using origin (0,0)")

    # 5. Extract camera images for N frames
    print(f"Extracting {num_samples} camera frames...")
    samples = _extract_images_from_tar(tar, scene_id, num_samples)
    tar.close()
    del tar_bytes

    # 6. Render each sample
    for i, sample in enumerate(samples):
        frame_idx = sample["frame_idx"]
        print(f"\n--- Sample {i + 1}/{len(samples)} (frame {frame_idx}) ---")

        if poses is not None and frame_idx < len(poses):
            ego_x = float(poses[frame_idx, 1])
            ego_y = float(poses[frame_idx, 2])
            ego_yaw = _quaternion_to_yaw(*poses[frame_idx, 4:8].tolist())
            print(f"  Ego: x={ego_x:.1f}, y={ego_y:.1f}, yaw={np.degrees(ego_yaw):.1f} deg")
        else:
            ego_x = ego_y = ego_yaw = 0.0

        # Render BEV map tile
        print("  Rendering BEV map tile...")
        if scene_dir is not None:
            bev_map = _render_bev_tile(
                scene_dir=scene_dir,
                ego_x=ego_x,
                ego_y=ego_y,
                ego_yaw=ego_yaw,
                canvas_size=256,
            )
        else:
            bev_map = _render_map_fallback(scene_id, frame_idx)

        # 7. Plot 3x3 grid
        fig = plt.figure(figsize=(15, 9))
        fig.suptitle(
            f"KITScenes Debug — {scene_id[:12]}... / frame {frame_idx}",
            fontsize=13,
            fontweight="bold",
        )

        for cam_name, pos in GRID_POSITIONS.items():
            ax = fig.add_subplot(3, 3, pos)
            ax.imshow(sample[cam_name])
            label = cam_name.replace("camera_ring_", "")
            ax.set_title(label, fontsize=9)
            ax.axis("off")

        ax_map = fig.add_subplot(3, 3, 5)
        ax_map.imshow(bev_map)
        ax_map.set_title("BEV Map Tile", fontsize=10, color="darkblue", fontweight="bold")
        ax_map.axis("off")

        plt.tight_layout()
        out_name = f"debug_output_sample_{i + 1}.png"
        plt.savefig(out_name, dpi=150, bbox_inches="tight")
        print(f"  Saved {out_name}")
        plt.close(fig)

    # Cleanup
    if scene_dir is not None:
        import shutil
        shutil.rmtree(scene_dir, ignore_errors=True)

    print(f"\nDone. {len(samples)} debug plots saved to current directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal KITScenes streaming + map render test")
    parser.add_argument(
        "--samples", type=int, default=6,
        help="Number of camera frames to extract (default: 6)",
    )
    parser.add_argument(
        "--local-tar", type=str, default=None,
        help="Path to a locally cached scene tar (skips HF download)",
    )
    args = parser.parse_args()
    debug_stream_and_render(num_samples=args.samples, local_tar=args.local_tar)
