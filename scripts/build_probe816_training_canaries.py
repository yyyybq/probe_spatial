#!/usr/bin/env python3
"""Build eight short, receipt-backed probing training canaries.

The bundle is fail-closed: every normal-model sample is selected from an
``extraction_receipt_full_sha`` owner, while MiniMax samples are selected from
the physically verified ``minimax_bf16_v2_extraction`` route.  Production scientific
overrides come directly from the frozen execution queue; only trainer duration,
logging, and worker-count settings are shortened for the canary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any


MODELS = (
    "cogvideox", "videomae", "dino", "f3r", "vjepa2",
    "qwen2_5_vl", "lingbot", "minimax_h3",
)
NORMAL_RUNTIME = Path(
    "/mnt/umm/users/yinbaiqiao/probe_spatial/runtime/main_v2_probe_v2_r5/source"
)
MINIMAX_RUNTIME = Path(
    "/mnt/umm/users/yinbaiqiao/probe_spatial/runtime/main_v2_probe_minimax_bf16_v2/source"
)
PYTHON = Path("/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_exclusive(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise RuntimeError(f"refusing existing output: {args.output}")
    overlay = read_json(args.overlay.resolve())
    if overlay.get("status") != "READY" or overlay.get("fail_closed") is not True:
        raise RuntimeError("owner overlay is not fail-closed READY")
    if overlay.get("counts") != {
        "streaming_duplicate_owners": 0,
        "streaming_holes": 0,
        "target_duplicate_owners": 0,
        "target_holes": 0,
    }:
        raise RuntimeError("owner overlay closure audit is not clean")
    streaming_db = Path(overlay["streaming_database"]["path"])
    if digest(streaming_db) != overlay["streaming_database"]["sha256"]:
        raise RuntimeError("streaming owner DB SHA mismatch")
    if digest(args.queue.resolve()) != read_json(
        Path(overlay["frozen_queue_manifest"]["path"])
    )["queue"]["sha256"]:
        raise RuntimeError("execution queue SHA mismatch")

    queue = load_jsonl(args.queue.resolve())
    selected: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        matches = [row for row in queue if row["model"] == model and row["probe"] == "streaming_depth"]
        if not matches:
            raise RuntimeError(f"no streaming_depth execution for {model}")
        selected[model] = matches[0]

    connection = sqlite3.connect(f"file:{streaming_db}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    cases: list[dict[str, Any]] = []
    try:
        for model in MODELS:
            row = selected[model]
            source_manifest = Path(row["execution_manifest"]["path"])
            if digest(source_manifest) != row["execution_manifest"]["sha256"]:
                raise RuntimeError(f"source execution manifest SHA mismatch for {model}")
            manifest_rows = load_jsonl(source_manifest)
            by_key = {
                (item["scene_id"], int(item["streaming_window_id"])): item
                for item in manifest_rows
            }
            expected_origin = (
                "minimax_bf16_v2_extraction" if model == "minimax_h3"
                else "extraction_receipt_full_sha"
            )
            owners = connection.execute(
                "SELECT scene_id,window_id,prefix,raw_ids_json,source_uri,origin "
                "FROM streaming_owners WHERE vfm=? AND layer=? AND timestep=? "
                "AND prefix=? AND origin=? ORDER BY scene_id,window_id",
                (model, row["layer"], row["timestep"], row["prefix"], expected_origin),
            ).fetchall()
            candidates: dict[str, list[tuple[sqlite3.Row, dict[str, Any]]]] = {
                "train": [], "val": []
            }
            for owner in owners:
                manifest_row = by_key.get((owner["scene_id"], int(owner["window_id"])))
                if manifest_row is None:
                    continue
                split = manifest_row.get("split")
                if split in candidates:
                    candidates[split].append((owner, manifest_row))
            if len(candidates["train"]) < 4 or len(candidates["val"]) < 2:
                raise RuntimeError(f"insufficient receipt-backed train/val samples for {model}")
            chosen = candidates["train"][:4] + candidates["val"][:2]

            case_dir = args.output / "cases" / model
            manifest = case_dir / "manifest.jsonl"
            manifest_text = "".join(
                json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
                for _, item in chosen
            )
            write_exclusive(manifest, manifest_text)
            manifest_sha = digest(manifest)
            runtime = MINIMAX_RUNTIME if model == "minimax_h3" else NORMAL_RUNTIME
            output_dir = (case_dir / "output").resolve()
            command = [str(PYTHON), "vidfm3d/train.py", f"experiment={row['experiment_config']}"]
            command.extend(row["scientific_overrides"])
            command.extend([
                f"paths.log_dir={output_dir}",
                "paths.run_folder_name=run",
                "logger=canary_csv",
                "trainer.max_epochs=1",
                "++trainer.max_steps=1",
                "++trainer.limit_train_batches=2",
                "++trainer.limit_val_batches=0",
                "++trainer.num_sanity_val_steps=0",
                "++trainer.log_every_n_steps=1",
                "++trainer.enable_progress_bar=false",
                "data.data_module.num_workers=0",
                "data.data_module.num_workers_val=0",
            ])
            sample_evidence = [
                {
                    "sample_id": item["sample_id"],
                    "split": item["split"],
                    "scene_id": item["scene_id"],
                    "window_id": int(item["streaming_window_id"]),
                    "raw_ids": json.loads(owner["raw_ids_json"]),
                    "source_uri": owner["source_uri"],
                    "owner_origin": owner["origin"],
                }
                for owner, item in chosen
            ]
            case = {
                "schema": "PROBE816_RECEIPT_UNION_TRAINING_CANARY_V1",
                "model": model,
                "unique_key": row["unique_key"],
                "runtime": str(runtime),
                "overlay_manifest": str(args.overlay.resolve()),
                "overlay_manifest_sha256": digest(args.overlay.resolve()),
                "manifest": str(manifest.resolve()),
                "manifest_sha256": manifest_sha,
                "samples": sample_evidence,
                "command": command,
                "environment": {
                    "PROBE_OWNER_OVERLAY_MANIFEST": str(args.overlay.resolve()),
                    "INSCENE_MANIFEST": str(manifest.resolve()),
                    "INSCENE_MANIFEST_SHA256": manifest_sha,
                    "INSCENE_TRUST_EXPANDED_MANIFEST": "1",
                    "INSCENE_DATA_ROOT": "/mnt/umm/users/yinbaiqiao/InsScene-15K/data",
                    "INSCENE_STREAMING_FEAT_ROOT": "aoss://baiqiao/PROBE816_OWNER_OVERLAY_ONLY",
                    "AOSS_LOG_LOCAL_HITS": "0",
                },
                "canary_only_overrides": {
                    "max_steps": 1,
                    "limit_train_batches": 2,
                    "limit_val_batches": 0,
                    "num_workers": 0,
                },
                "resource": {
                    "aec2": "h800", "worker_nodes": 1,
                    "worker_spec": "N4lS.Iq.I80.1", "priority": "HIGHEST",
                },
            }
            case_path = case_dir / "case.json"
            write_exclusive(case_path, json.dumps(case, indent=2, sort_keys=True) + "\n")
            case["case_path"] = str(case_path.resolve())
            case["case_sha256"] = digest(case_path)
            cases.append(case)
    finally:
        connection.close()

    bundle = {
        "schema": "PROBE816_RECEIPT_UNION_TRAINING_CANARY_BUNDLE_V1",
        "status": "FROZEN_NOT_SUBMITTED",
        "models": list(MODELS),
        "case_count": len(cases),
        "queue": {"path": str(args.queue.resolve()), "sha256": digest(args.queue.resolve())},
        "overlay": {"path": str(args.overlay.resolve()), "sha256": digest(args.overlay.resolve())},
        "cases": [
            {key: case[key] for key in ("model", "unique_key", "case_path", "case_sha256", "resource")}
            for case in cases
        ],
    }
    write_exclusive(args.output / "bundle.json", json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(cases)}, sort_keys=True))


if __name__ == "__main__":
    main()
