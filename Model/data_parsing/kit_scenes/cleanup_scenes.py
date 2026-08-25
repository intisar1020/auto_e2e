"""Remove unused sensor folders from extracted KITScenes scene dirs.

The AutoE2E training pipeline is camera-only. It reads exactly:
  - 7 camera dirs (camera_base_front_center + 6 camera_ring_*)
  - poses.txt, timestamp.reference.txt, maps/, calibration/

Everything else (lidar, radar, gnss, gnss_ins, processed, async, and the two
unused rect cameras) is not referenced by KitScenesDataset, so it is deleted
to free ~60GB across the downloaded split.

Verification: before deletion, one scene is loaded via KitScenesDataset and
its sample count recorded; after deletion the same check is re-run to confirm
training data is intact.

Usage:
  cd Model/data_parsing/kit_scenes
  python cleanup_scenes.py [--dry-run]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_MODEL_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_MODEL_DIR))

from data_parsing.kit_scenes import KitScenesDataset

DATA_ROOT = Path(__file__).parent / "datasets" / "train"

DELETABLE = [
    "lidar_corner_left", "lidar_corner_right", "lidar_front", "lidar_left",
    "lidar_rear", "lidar_right", "lidar_top",
    "radar_front", "radar_left", "radar_right",
    "gnss", "gnss_ins",
    "processed", "async",
    "camera_base_front_left_rect", "camera_base_front_right_rect",
]


def _sample_counts() -> dict[str, int]:
    """Sample count per scene (only the SDK-valid subset)."""
    ds = KitScenesDataset(data_root=str(DATA_ROOT.parent), split="train",
                          include_navigation=True)
    from collections import Counter
    return dict(Counter(sid for sid, _ in ds._samples))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be deleted without deleting")
    args = ap.parse_args()

    scenes = sorted(d for d in DATA_ROOT.iterdir() if d.is_dir())
    print(f"{len(scenes)} scenes under {DATA_ROOT}")

    before = _sample_counts()
    print(f"before: {sum(before.values())} samples / {len(before)} scenes "
          f"(SDK-valid)")

    freed = 0
    for scene in scenes:
        for folder in DELETABLE:
            target = scene / folder
            if not target.is_dir():
                continue
            size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            freed += size
            if args.dry_run:
                print(f"  would delete {target.relative_to(DATA_ROOT)} ({size/1e9:.2f}G)")
            else:
                shutil.rmtree(target)
        if not args.dry_run and not list(scene.iterdir()):
            shutil.rmtree(scene)
            print(f"  removed empty scene {scene.name}")

    print(f"\n{'would free' if args.dry_run else 'freed'} {freed/1e9:.1f} GB")

    if not args.dry_run:
        after = _sample_counts()
        print(f"after:  {sum(after.values())} samples / {len(after)} scenes")
        assert after == before, "sample counts changed after cleanup!"
        print("OK: training data intact after cleanup.")


if __name__ == "__main__":
    main()
