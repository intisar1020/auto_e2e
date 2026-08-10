"""Batch download additional train tars for exp-1-subset (parallel, resumable).

Reads a list of tars (sid + size_gb) from JSON, downloads/extracts each into
exp-1-subset/data/train/<sid> (4 parallel workers), verifies sample counts,
and appends them to the manifest with status + samples. Tars with 0 samples
or load errors are marked 'empty-or-error' (skipped by training).

Usage:
  cd Model/data_parsing/kit_scenes
  python download_exp1_more.py /tmp/exp1_new_tars.json [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import sys
import tarfile
import tempfile
import time

# HfFileSystem's HTTP reads can hang forever on a stalled connection (we saw
# CLOSE-WAIT sockets with no progress). Bound each socket so a dead peer
# surfaces as an exception the retry loop can recover from.
socket.setdefaulttimeout(60)
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def download_one(sid: str) -> dict:
    """Download + extract one tar; returns result dict."""
    dest = BASE / "data" / "train" / sid
    if dest.is_dir():
        return {"sid": sid, "status": "done", "samples": -1, "note": "already-extracted"}
    p = f"datasets/KIT-MRT/KITScenes-Multimodal/data/train/{sid}.tar"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{sid}.tar.part"
    t0 = time.time()
    last_err = "timeout"
    for attempt in range(3):
        try:
            fs = _fs()
            with fs.open(p, "rb") as fh, open(tmp, "wb") as out:
                while True:
                    chunk = fh.read(CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
            with tarfile.open(tmp) as tar:
                td = Path(tempfile.mkdtemp(prefix="ks_"))
                tar.extractall(path=td)
                (td / sid).rename(dest)
                shutil.rmtree(td, ignore_errors=True)
            break
        except Exception as e:
            last_err = str(e)[:120]
            tmp.unlink(missing_ok=True)
            print(f"    retry {sid[:12]} ({attempt+1}/3): {last_err}", flush=True)
            time.sleep(5)
    else:
        return {"sid": sid, "status": "error", "samples": -1, "note": last_err}
    tmp.unlink(missing_ok=True)
    try:
        ds = KitScenesDataset(data_root=str(BASE / "data"), split="train",
                              include_navigation=True, scene_ids=[sid])
        n = len(ds)
    except ValueError:
        n = 0
    return {"sid": sid, "status": "done" if n > 0 else "empty-or-error",
            "samples": n, "note": f"{time.time()-t0:.0f}s"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tars_json", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    new_tars = json.loads(args.tars_json.read_text())
    manifest = json.loads(MANIFEST.read_text())
    have = {e["sid"] for e in manifest["tars"]}
    todo = [t for t in new_tars if t["sid"] not in have]
    print(f"{len(new_tars)} requested, {len(todo)} new (skipping already-in-manifest)")

    results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, t["sid"]): t for t in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            results.append(res)
            print(f"[{i}/{len(todo)}] {res['sid'][:12]} {res['status']} "
                  f"samples={res['samples']} ({res.get('note','')}) "
                  f"[{time.time()-t_start:.0f}s]", flush=True)

    for res in results:
        entry = next((e for e in manifest["tars"] if e["sid"] == res["sid"]), None)
        if entry is None:
            src = next((t for t in new_tars if t["sid"] == res["sid"]), {})
            entry = {"sid": res["sid"], "role": "train",
                     "size_gb": src.get("size_gb", 0.0)}
            manifest["tars"].append(entry)
        entry["status"] = res["status"]
        entry["samples"] = res["samples"]
    MANIFEST.write_text(json.dumps(manifest, indent=2))

    ok = sum(1 for r in results if r["status"] == "done")
    bad = [r for r in results if r["status"] != "done"]
    print(f"\n{ok}/{len(todo)} ok, {len(bad)} bad")
    for r in bad:
        print(f"  BAD {r['sid']}: {r['status']} {r.get('note','')}")


if __name__ == "__main__":
    main()
