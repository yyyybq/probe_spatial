#!/usr/bin/env python3
"""Multiprocess NFS stress test for atomic deterministic-label sidecar commits."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import pathlib
import sys


def worker(source: str, root: str, worker_id: int, iterations: int, queue) -> None:
    sys.path.insert(0, source)
    from vidfm3d.data.components.inscene15k_dataset import InsScene15KDataset
    import torch
    dataset = object.__new__(InsScene15KDataset)
    errors = []
    for i in range(iterations):
        slot = i % 64
        identity = {"label_version": "stress-v1", "slot": slot}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = pathlib.Path(root) / digest[:2] / f"{digest}.npz"
        try:
            dataset._store_label_cache(
                (identity, str(path)),
                {"value": torch.tensor([slot, worker_id], dtype=torch.long)},
            )
        except Exception as exc:
            errors.append({"iteration": i, "type": type(exc).__name__, "message": str(exc)})
            break
    queue.put({"worker": worker_id, "errors": errors})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=pathlib.Path)
    parser.add_argument("--sidecar-root", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()
    args.sidecar_root.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = [ctx.Process(target=worker, args=(str(args.source_root.resolve()),
                 str(args.sidecar_root.resolve()), i, args.iterations, queue))
                 for i in range(args.workers)]
    for process in processes: process.start()
    results = [queue.get() for _ in processes]
    for process in processes: process.join()
    files = list(args.sidecar_root.rglob("*.npz"))
    errors = [error for row in results for error in row["errors"]]
    exitcodes = [process.exitcode for process in processes]
    report = {
        "schema": "tstep-sidecar-atomic-stress-v1", "workers": args.workers,
        "iterations_per_worker": args.iterations,
        "attempted_commits": args.workers * args.iterations,
        "sidecar_files": len(files), "worker_exitcodes": exitcodes,
        "errors": errors, "passed": not errors and exitcodes == [0] * args.workers and len(files) == 64,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
