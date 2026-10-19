#!/usr/bin/env python3
"""Re-publish locally complete canonical GAP payloads without re-extraction.

Use when extract already wrote payloads under
``FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1`` but AOSS sync aborted (e.g. ENOSPC
during hardlink staging).  This binds local-ready units, touches
``extract.finished`` immediately, and runs ``run_tstep_sync_loop.sh`` only.

Must run as the owner of the GAP feature tree (usually root on SCO workers)
because staging hardlinks and post-upload deletes require write access there.

``--trust-current-tools`` accepts the on-disk sync loop / adapter even when the
immutable shard still pins older SHAs (code drift after the extract job).
Worklist / protocol / shard content SHAs remain fail-closed.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.canonical_gap_worklist import canonical_json_sha, sha256_file  # noqa: E402
from scripts.windowfix_v2.canonical_gap_aoss_adapter import (  # noqa: E402
    GapSyncError,
    completed_units_from_receipt,
    expected_extraction_unit_keys,
    load_verified_receipt,
)
from scripts.windowfix_v2 import run_canonical_gap_shard as runner  # noqa: E402


def _light_load_shard(path: Path, *, trust_current_tools: bool) -> dict[str, Any]:
    """Load a READY streaming/target gap shard with worklist SHA checks.

    When ``trust_current_tools`` is set, skip SHA pins on runner / extractor /
    sync-loop / adapter / backend deps so reclaim can use the current tools.
    """

    if not trust_current_tools:
        return runner.validate_shard(path)

    shard_path = runner.regular_resolved_path(path, "gap shard")
    shard = runner.read_json(shard_path, "gap shard")
    stored = shard.get("content_sha256")
    unhashed = dict(shard)
    unhashed.pop("content_sha256", None)
    if shard.get("schema_version") != 1 or shard.get("status") != "READY":
        runner.fail("gap shard is not READY schema v1")
    if shard.get("contract") != "CANONICAL_V2_GAP_WORKLIST_V1":
        runner.fail("gap shard contract mismatch")
    if stored != canonical_json_sha(unhashed):
        runner.fail("gap shard content_sha256 mismatch")
    mode = str(shard.get("mode"))
    if shard.get("feature_namespace") != runner.GAP_NAMESPACES.get(mode):
        runner.fail("gap shard mode/feature namespace mismatch")
    worklist_path = runner.regular_resolved_path(shard.get("worklist_path"), "gap worklist")
    runner.require_sha(worklist_path, shard.get("worklist_file_sha256"), "gap worklist")
    protocol_path = runner.regular_resolved_path(shard.get("protocol_path"), "protocol")
    runner.require_sha(protocol_path, shard.get("protocol_file_sha256"), "protocol")
    shard["_binding_path"] = str(shard_path)
    shard["_binding_file_sha256"] = sha256_file(shard_path)
    return shard


def run_sync_only(shard: dict[str, Any], *, trust_current_tools: bool) -> dict[str, Any]:
    dataset_root = Path(str(shard["dataset_data_root"])).absolute().parent
    feature_root = Path(str(shard["feature_root"])).absolute()
    runner.ensure_no_symlink_components(feature_root, dataset_root)
    if feature_root.name != shard["feature_namespace"] or "SEEDED_V1" in feature_root.name:
        runner.fail("gap sync-only refuses non-gap or SEEDED_V1 destination")
    payload_root = feature_root / str(shard["vfm"]) / str(shard["source"])
    if not payload_root.is_dir():
        runner.fail(f"missing local payload root: {payload_root}")
    runner.ensure_no_symlink_components(payload_root, dataset_root)

    # Extract already wrote sidecars; reclaim only needs a complete file set.
    os.environ["GAP_SYNC_COMPLETENESS_ONLY"] = "1"

    # Receipts, staging directories, and cleanup are per worklist.  Concurrent
    # workers on disjoint worklists upload different scene trees.  Workers that
    # share one worklist still take the same exclusive lock.
    lock_dir = (
        feature_root / ".windowfix_v2_atomic_gap_locks" / shard["vfm"] / shard["source"]
    )
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (lock_dir / f"{shard['worklist_file_sha256']}.lock").open("a+")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)

    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-synconly-p{os.getpid()}"
    log_dir = (
        feature_root
        / ".windowfix_v2_gap_logs"
        / shard["vfm"]
        / shard["shard_id"]
        / attempt
    )
    log_dir.mkdir(parents=True, exist_ok=False)
    state_paths = runner._state_paths(shard, feature_root)
    state_paths["root"].mkdir(parents=True, exist_ok=True)

    receipt_path = log_dir / "receipt.json"
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "RUNNING_SYNC_ONLY",
        "attempt": attempt,
        "shard_id": shard["shard_id"],
        "shard_content_sha256": shard["content_sha256"],
        "worklist_file_sha256": shard["worklist_file_sha256"],
        "trust_current_tools": trust_current_tools,
        "remote_completion_receipt": str(state_paths["receipt"]),
        "aoss_done_index": str(state_paths["done_index"]),
    }
    runner.write_json_atomic(receipt_path, receipt)

    environment = os.environ.copy()
    environment.update({str(k): str(v) for k, v in shard.get("environment", {}).items()})
    environment.pop("EXTRACT_DONE_INDEX", None)
    environment["FEATURE_NAMESPACE"] = shard["feature_namespace"]
    environment["PYTHONPATH"] = ":".join(
        value
        for value in (
            str(shard["project_root"]),
            environment.get("PYTHONPATH", ""),
        )
        if value
    )

    exclusion_path = log_dir / "completed_scenes.json"
    completed_units_path = log_dir / "completed_units.json"
    local_ready_units_path = log_dir / "local_ready_units.json"
    try:
        completed_before, exclusion_sha = runner._bind_completed_scenes(
            shard, state_paths["receipt"], exclusion_path
        )
        completed_units_before, completed_units_sha = runner._bind_completed_units(
            shard, state_paths["receipt"], completed_units_path
        )
        local_ready_units, local_ready_units_sha, local_ready_unit_invalid = (
            runner._bind_local_ready_units(shard, payload_root, local_ready_units_path)
        )
    except GapSyncError as exc:
        receipt.update({"status": "FAILED_PREFLIGHT", "error": str(exc)})
        runner.write_json_atomic(receipt_path, receipt)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
        runner.fail(f"sync-only preflight failed: {exc}; receipt={receipt_path}")

    if exclusion_sha is not None:
        environment["CANONICAL_GAP_COMPLETED_SCENES_FILE"] = str(exclusion_path)
        environment["CANONICAL_GAP_COMPLETED_SCENES_FILE_SHA256"] = exclusion_sha
    if completed_units_sha is not None:
        environment["CANONICAL_GAP_COMPLETED_UNITS_FILE"] = str(completed_units_path)
        environment["CANONICAL_GAP_COMPLETED_UNITS_FILE_SHA256"] = completed_units_sha
    if local_ready_units:
        environment["CANONICAL_GAP_LOCAL_READY_UNITS_FILE"] = str(local_ready_units_path)
        environment["CANONICAL_GAP_LOCAL_READY_UNITS_FILE_SHA256"] = str(local_ready_units_sha)

    receipt.update(
        {
            "completed_scenes_before": sorted(completed_before),
            "completed_units_before": sorted(completed_units_before),
            "local_ready_units_before": sorted(local_ready_units),
            "local_ready_unit_invalid_count": len(local_ready_unit_invalid),
        }
    )
    runner.write_json_atomic(receipt_path, receipt)

    if not local_ready_units:
        receipt.update(
            {
                "status": "NO_LOCAL_READY_UNITS",
                "note": "nothing publishable under payload_root for this worklist",
            }
        )
        runner.write_json_atomic(receipt_path, receipt)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
        return {
            "dry_run": False,
            "shard_id": shard["shard_id"],
            "status": "NO_LOCAL_READY_UNITS",
            "receipt": str(receipt_path),
            "local_ready_units": 0,
        }

    stop_file = log_dir / "extract.finished"
    stop_file.touch(exist_ok=False)
    sync_log_dir = log_dir / "aoss_publish"
    sync_log_dir.mkdir()
    sync_log_path = sync_log_dir / "runner.log"

    sync_env = runner._sync_environment(
        shard,
        environment,
        payload_root=payload_root,
        log_dir=sync_log_dir,
        stop_file=stop_file,
        completion_receipt=state_paths["receipt"],
        done_index=state_paths["done_index"],
    )
    # Sync-only reclaim: allow oversized batches.  The runner hardcodes
    # GAP_SYNC_MAX_SCENES=4 for extract-time backpressure; that starves AOSS
    # bandwidth when millions of local units are already complete.
    for key in (
        "GAP_SYNC_MAX_SCENES",
        "SYNC_EVERY_GIB",
        "SYNC_EVERY_SEC",
        "SYNC_POLL_SEC",
        "GAP_SYNC_COMPLETENESS_ONLY",
        "GAP_SYNC_IGNORE_TEMPS_BEFORE",
    ):
        if key in os.environ and os.environ[key].strip():
            sync_env[key] = os.environ[key].strip()
    sync_env["GAP_SYNC_COMPLETENESS_ONLY"] = "1"
    if trust_current_tools:
        adapter = runner.GAP_AOSS_ADAPTER
        sync_loop = runner.AOSS_SYNC_LOOP
        sync_env["GAP_ADAPTER"] = str(adapter)
        sync_env["EXPECTED_GAP_ADAPTER_SHA256"] = sha256_file(adapter)
        sync_loop_path = sync_loop
    else:
        sync_loop_path = Path(str(shard["source_bindings"]["aoss_sync_loop"]["path"]))

    with sync_log_path.open("wb") as sync_handle:
        sync_process = subprocess.Popen(
            ["/bin/bash", str(sync_loop_path)],
            cwd=str(shard["project_root"]),
            env=sync_env,
            stdout=sync_handle,
            stderr=subprocess.STDOUT,
        )
        sync_returncode = sync_process.wait()

    remote_error = None
    completed_units_after: set[str] = set()
    try:
        remote = load_verified_receipt(
            state_paths["receipt"],
            worklist_file_sha256=str(shard["worklist_file_sha256"]),
            shard_content_sha256=str(shard["content_sha256"]),
        )
        raw = runner.read_json(Path(str(shard["worklist_path"])), "gap worklist")
        raw["_file_sha256"] = str(shard["worklist_file_sha256"])
        completed_units_after = completed_units_from_receipt(
            remote, raw, str(shard["worklist_file_sha256"])
        )
        missing_units = expected_extraction_unit_keys(raw) - completed_units_after
        # Sync-only may publish a subset (local ready). Fail only if sync failed.
        receipt["missing_worklist_units_after"] = len(missing_units)
        receipt["completed_units_after"] = sorted(completed_units_after)
    except (GapSyncError, FileNotFoundError) as exc:
        remote_error = str(exc)

    passed = sync_returncode == 0 and remote_error is None
    # For partial local inventory, require progress on previously local-ready units.
    if passed and local_ready_units:
        still_local = sorted(set(local_ready_units) - completed_units_after)
        receipt["local_ready_units_still_unpublished"] = still_local
        if still_local:
            passed = False
            remote_error = (
                f"{len(still_local)} local-ready units remain unpublished after sync"
            )

    receipt.update(
        {
            "status": "PASS" if passed else "FAILED",
            "aoss_sync": {
                "returncode": sync_returncode,
                "log": str(sync_log_path),
            },
            "remote_verification_error": remote_error,
        }
    )
    runner.write_json_atomic(receipt_path, receipt)
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    lock_handle.close()
    if not passed:
        runner.fail(
            "canonical gap sync-only failed closed; "
            f"sync_rc={sync_returncode} remote_error={remote_error!r}; "
            f"receipt={receipt_path}"
        )
    return {
        "dry_run": False,
        "shard_id": shard["shard_id"],
        "status": "PASS",
        "receipt": str(receipt_path),
        "remote_completion_receipt": str(state_paths["receipt"]),
        "local_ready_units": len(local_ready_units),
        "completed_units_after": len(completed_units_after),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--shard", type=Path)
    selection.add_argument("--batch", type=Path)
    parser.add_argument(
        "--trust-current-tools",
        action="store_true",
        help="use on-disk sync loop/adapter even if shard pins older SHAs",
    )
    parser.add_argument(
        "--streaming-only",
        action="store_true",
        help="when used with --batch, skip target_isolated shards",
    )
    args = parser.parse_args()

    if args.shard:
        shard = _light_load_shard(args.shard, trust_current_tools=args.trust_current_tools)
        value = run_sync_only(shard, trust_current_tools=args.trust_current_tools)
    else:
        batch_path = runner.regular_resolved_path(args.batch, "gap batch")
        batch = runner.read_json(batch_path, "gap batch")
        results = []
        for item in batch.get("shards", []):
            if not isinstance(item, dict):
                runner.fail(f"invalid batch shard binding: {item!r}")
            if args.streaming_only and item.get("mode") != "streaming_prefix":
                results.append(
                    {
                        "shard_id": item.get("shard_id"),
                        "status": "SKIPPED_TARGET",
                    }
                )
                continue
            shard_path = runner.regular_resolved_path(item.get("path"), "batch shard")
            shard = _light_load_shard(
                shard_path, trust_current_tools=args.trust_current_tools
            )
            results.append(
                run_sync_only(shard, trust_current_tools=args.trust_current_tools)
            )
        value = {
            "batch_id": batch.get("batch_id"),
            "shards": results,
        }
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
