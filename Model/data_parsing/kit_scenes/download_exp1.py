"""One-shot download + extract for exp-1-subset (all local, no streaming).

Streams each tar from HF in 64MB chunks to a .part file, extracts to
exp-1-subset/data/<role>/<sid>, verifies sample counts, and updates the
manifest status. Run once; the dataset is then fully local.

Usage:
  cd Model/data_parsing/kit_scenes
  python download_exp1.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

_MODEL_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_MODEL_DIR))

from data_parsing.kit_scenes import KitScenesDataset

BASE = Path(__file__).parent / "exp-1-subset"
MANIFEST = BASE / "manifest.json"
CHUNK = 64 * 1024 * 1024


def _fs():
    from huggingface_hub import HfFileSystem
    return HfFileSystem()


def download_one(fs, sid: str, dest: Path) -> float:
    """Download + extract one tar; returns elapsed seconds."""
    p = f"datasets/KIT-MRT/KITScenes-Multimodal/data/train/{sid}.tar"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{sid}.tar.part"
    t0 = time.time()
    with fs.open(p, "rb") as fh, open(tmp, "wb") as out:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            out.write(chunk)
    t_dl = time.time() - t0
    t0 = time.time()
    with tarfile.open(tmp) as tar:
        td = Path(tempfile.mkdtemp(prefix="ks_"))
        tar.extractall(path=td)
        (td / sid).rename(dest)
        shutil.rmtree(td, ignore_errors=True)
    tmp.unlink()
    return time.time() - t0, t_dl


def count_samples(sid: str, root: Path, role: str) -> int:
    """Sample count for one scene dir under root/<role>/."""
    try:
        ds = KitScenesDataset(data_root=str(root), split="train",
                              include_navigation=True, scene_ids=[sid])
        return len(ds)
    except ValueError:
        return -1


def main():
    fs = _fs()
    manifest = json.loads(MANIFEST.read_text())
    for entry in manifest["tars"]:
        sid, role = entry["sid"], entry["role"]
        # All tars come from the HF train split; the SDK (split="train")
        # requires scene dirs under data/train/. Role is tracked in manifest.
        dest = BASE / "data" / "train" / sid
        if dest.is_dir():
            print(f"  skip {sid[:12]} (already extracted)")
            entry["status"] = "done"
            continue
        print(f"  ↓ {sid[:12]} ({entry['size_gb']:.2f} GB) ...", flush=True)
        x_time, dl_time = download_one(fs, sid, dest)
        n = count_samples(sid, BASE / "data", "train")
        entry["status"] = "done" if n > 0 else "empty-or-error"
        entry["samples"] = n
        print(f"    done in {x_time:.0f}s (dl {dl_time:.0f}s), samples={n}")
        MANIFEST.write_text(json.dumps(manifest, indent=2))
    done = sum(1 for e in manifest["tars"] if e["status"] == "done")
    print(f"\n{len(manifest['tars'])} tars, {done} extracted")


if __name__ == "__main__":
    main()
