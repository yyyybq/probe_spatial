#!/usr/bin/env python3
"""Bound one tstep-probe job's disposable AOSS materialization cache."""

from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path


MARKER = ".tstep_probe_cache"
STOP = False


def stop(_signum, _frame) -> None:
    global STOP
    STOP = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--max-gib", type=float, default=900.0)
    parser.add_argument("--target-gib", type=float, default=700.0)
    parser.add_argument("--min-age-sec", type=int, default=300)
    parser.add_argument("--interval-sec", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--purge", action="store_true")
    return parser.parse_args()


def cache_files(root: Path) -> list[tuple[float, int, Path]]:
    files: list[tuple[float, int, Path]] = []
    for path in root.iterdir():
        if not path.is_file() or path.name == MARKER or ".tmp-" in path.name:
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        files.append((stat.st_mtime, stat.st_size, path))
    return files


def prune_once(root: Path, max_bytes: int, target_bytes: int, min_age: int) -> None:
    files = cache_files(root)
    total = sum(size for _mtime, size, _path in files)
    if total <= max_bytes:
        print(f"cache_ok bytes={total} files={len(files)}", flush=True)
        return

    cutoff = time.time() - min_age
    removed_files = 0
    removed_bytes = 0
    for mtime, size, path in sorted(files):
        if total <= target_bytes:
            break
        if mtime > cutoff:
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            print(f"cache_unlink_failed path={path} error={exc}", flush=True)
            continue
        total -= size
        removed_files += 1
        removed_bytes += size
    print(
        f"cache_pruned removed_files={removed_files} removed_bytes={removed_bytes} "
        f"remaining_bytes={total}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    root = Path(args.cache_dir).resolve()
    if args.max_gib <= 0 or args.target_gib < 0 or args.target_gib >= args.max_gib:
        raise SystemExit("require 0 <= target-gib < max-gib")
    if not (root / MARKER).is_file():
        raise SystemExit(f"refusing unmarked cache directory: {root}")

    if args.purge:
        removed_files = 0
        removed_bytes = 0
        for path in root.iterdir():
            if not path.is_file() or path.name == MARKER:
                continue
            try:
                size = path.stat().st_size
                path.unlink()
            except FileNotFoundError:
                continue
            removed_files += 1
            removed_bytes += size
        print(
            f"cache_purged removed_files={removed_files} removed_bytes={removed_bytes}",
            flush=True,
        )
        return

    max_bytes = int(args.max_gib * 1024**3)
    target_bytes = int(args.target_gib * 1024**3)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not STOP:
        prune_once(root, max_bytes, target_bytes, args.min_age_sec)
        if args.once:
            break
        for _ in range(max(args.interval_sec, 1)):
            if STOP:
                break
            time.sleep(1)


if __name__ == "__main__":
    main()
