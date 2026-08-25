"""Fast KITScenes tar validator: peek at the tar tail to count ego poses.

A scene is usable only if ``poses.txt`` has >= MIN_ROWS (129) lines. Reading
the whole tar to find out is wasteful (tars are ~2-4 GB); ``poses.txt`` is a
small file written near the END of the archive, so we fetch only the last
``TAIL_BYTES`` of the tar, scan it for the block-aligned ``poses.txt`` member
header, and count newlines in its payload.

Usage:
  python peek_tar_poses.py [--min-rows 129] --tail-mb 40 sid... | <sids on stdin>
"""

from __future__ import annotations

import argparse
import socket
import sys

import numpy as np

socket.setdefaulttimeout(30)

TAIL_BYTES = 40 * 1024 * 1024


def _fs():
    from huggingface_hub import HfFileSystem
    return HfFileSystem()


def count_poses(fs, tar_path: str, tail_bytes: int = TAIL_BYTES) -> int:
    """Return the ego-pose count for one tar by reading only its tail."""
    info = fs.info(tar_path)
    fsize = info["size"]
    with fs.open(tar_path, "rb") as fh:
        fh.seek(max(0, fsize - tail_bytes))
        chunk = fh.read(tail_bytes)
    for off in range(0, len(chunk) - 512, 512):
        nm = chunk[off:off + 100].split(b"\x00")[0]
        if nm.endswith(b"poses.txt"):
            sz_oct = chunk[off + 124:off + 136].split(b"\x00")[0]
            try:
                sz = int(sz_oct, 8)
            except ValueError:
                continue
            data = chunk[off + 512:off + 512 + sz]
            return data.count(b"\n")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-rows", type=int, default=129)
    ap.add_argument("--tail-mb", type=int, default=40)
    ap.add_argument("sids", nargs="*")
    args = ap.parse_args()

    if args.sids:
        sids = args.sids
    else:
        sids = [line.strip() for line in sys.stdin if line.strip()]

    fs = _fs()
    base = "datasets/KIT-MRT/KITScenes-Multimodal/data/train"
    # Resolve sid -> tar path once.
    tar_paths = {}
    for item in fs.ls(base, detail=True):
        nm = item["name"].split("/")[-1]
        if nm.endswith(".tar"):
            tar_paths[nm[:-4]] = item["name"]

    results = []
    for sid in sids:
        path = tar_paths.get(sid)
        if path is None:
            results.append((sid, -1))
            continue
        n = count_poses(fs, path, args.tail_mb * 1024 * 1024)
        results.append((sid, n))
        print(f"{sid} poses={n} {'VALID' if n >= args.min_rows else 'short'}",
              flush=True)

    n_valid = sum(1 for _, n in results if n >= args.min_rows)
    print(f"\n{n_valid}/{len(results)} valid (>= {args.min_rows} poses)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
