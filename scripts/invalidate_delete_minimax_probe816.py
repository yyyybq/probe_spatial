#!/usr/bin/env python3
"""Audit, invalidate, and delete only completed MiniMax probe816 outputs.

This deliberately operates on one exact campaign/run namespace and refuses to
touch any experiment that is not already in the SUCCEEDED terminal state.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


MODEL_PREFIX = "minimax_h3__"
INVALID_STATE = "INVALID_SOURCE_FEATURE"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o644)
    Path(temporary).replace(path)


def inventory(root: Path) -> dict:
    files = []
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        total += size
        # Preserve a compact but useful checkpoint/completion inventory. Avoid
        # hashing multi-GB invalid checkpoints solely to delete them.
        if (
            path.name == "TRAIN_COMPLETE.json"
            or path.suffix in {".ckpt", ".json", ".yaml", ".yml"}
            or "checkpoint" in path.name.lower()
        ):
            files.append({"relative_path": str(path.relative_to(root)), "bytes": size})
    return {"total_bytes": total, "selected_files": files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-state-root", type=Path, required=True)
    parser.add_argument("--run-output-root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    campaign = args.campaign_state_root.resolve()
    run_root = args.run_output_root.resolve()
    audit_dir = args.audit_dir.resolve()
    experiments = campaign / "experiments"
    completed = campaign / "completed"

    targets = sorted(experiments.glob(f"{MODEL_PREFIX}*.json"))
    if len(targets) != 9:
        raise SystemExit(f"refusing: expected exactly 9 completed MiniMax states, found {len(targets)}")

    generated = datetime.now(timezone.utc).isoformat()
    report = {
        "schema": "PROBE816_INVALID_SOURCE_FEATURE_DELETION_V1",
        "generated_utc": generated,
        "reason": (
            "MiniMax source safetensors contain positive infinity values caused by "
            "FP16 serialization overflow; losses were zeroed by legacy non-finite handling."
        ),
        "campaign_state_root": str(campaign),
        "run_output_root": str(run_root),
        "execute": args.execute,
        "targets": [],
    }

    # Complete all validation and snapshots before deleting anything.
    for state_path in targets:
        key = state_path.stem
        state = read_json(state_path)
        if state.get("state") != "SUCCEEDED":
            raise SystemExit(f"refusing {key}: state={state.get('state')!r}, expected SUCCEEDED")
        receipt_path = completed / f"{key}.json"
        if not receipt_path.is_file():
            raise SystemExit(f"refusing {key}: missing completion receipt {receipt_path}")
        output = Path(state.get("output_dir", "")).resolve()
        if output.parent != run_root or output.name != key or not key.startswith(MODEL_PREFIX):
            raise SystemExit(f"refusing unsafe output binding for {key}: {output}")
        if not output.is_dir():
            raise SystemExit(f"refusing {key}: output directory missing: {output}")
        train_complete = output / "TRAIN_COMPLETE.json"
        report["targets"].append(
            {
                "experiment_key": key,
                "previous_experiment_state": state,
                "completion_receipt": read_json(receipt_path),
                "train_complete": read_json(train_complete) if train_complete.is_file() else None,
                "output_dir": str(output),
                "output_inventory": inventory(output),
            }
        )

    audit_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = audit_dir / "pre_delete_audit.json"
    if preflight_path.exists():
        raise SystemExit(f"refusing to overwrite existing audit: {preflight_path}")
    atomic_json(preflight_path, report)

    if not args.execute:
        print(json.dumps({"preflight": str(preflight_path), "targets": len(targets)}, indent=2))
        return

    deleted = []
    for item in report["targets"]:
        key = item["experiment_key"]
        output = Path(item["output_dir"])
        state_path = experiments / f"{key}.json"
        receipt_path = completed / f"{key}.json"

        # Recheck immediately before every destructive operation.
        current = read_json(state_path)
        if current.get("state") != "SUCCEEDED" or not receipt_path.is_file():
            raise SystemExit(f"state changed before deletion for {key}; stopping")
        if output.resolve().parent != run_root or output.resolve().name != key:
            raise SystemExit(f"output binding changed before deletion for {key}; stopping")

        shutil.rmtree(output)
        receipt_path.unlink()
        invalid = dict(current)
        invalid.update(
            {
                "state": INVALID_STATE,
                "invalidated_utc": datetime.now(timezone.utc).isoformat(),
                "invalid_reason": report["reason"],
                "deleted_output_dir": str(output),
                "invalidation_audit": str(preflight_path),
            }
        )
        atomic_json(state_path, invalid)
        deleted.append(key)

    final = {
        "schema": "PROBE816_INVALID_SOURCE_FEATURE_DELETION_RESULT_V1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "pre_delete_audit": str(preflight_path),
        "deleted_count": len(deleted),
        "deleted_experiment_keys": deleted,
    }
    atomic_json(audit_dir / "deletion_result.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
