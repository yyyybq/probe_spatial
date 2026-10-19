#!/usr/bin/env python3
"""Submit the complete-scope MiniMax BF16 V2 extraction as GPU shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
CAMPAIGN = ROOT / "artifacts/main_v2_probe_v2_816_20260917"
ROUTE = CAMPAIGN / "minimax_bf16_v2_production_route_v2/route_receipt.json"
RELEASE = ROOT / "runtime/main_v2_minimax_bf16_v3_extractor"
LAUNCHER = RELEASE / "run_production_shard_v3.sh"
VERIFIER = RELEASE / "verify_production_shard_v3.py"
SOURCE = RELEASE / "source"
CANARY_REPORT = (
    CAMPAIGN
    / "minimax_bf16_v2_production_extract_v4/shard_0_of_681/summary.json"
)
CANARY_LOG = CANARY_REPORT.with_name("worker.log")
LEDGER = CAMPAIGN / "minimax_bf16_v2_production_v4_submission_ledger.json"
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
SHARD_COUNT = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def reports_from_log(text: str) -> list[dict]:
    """Extract JSON reports embedded in a worker log."""
    decoder = json.JSONDecoder()
    reports = []
    offset = 0
    while True:
        start = text.find("{", offset)
        if start < 0:
            return reports
        try:
            value, length = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            offset = start + 1
            continue
        if isinstance(value, dict):
            reports.append(value)
        offset = start + length


def load_canary_report() -> tuple[dict, Path]:
    """Read the summary, falling back to the world-readable worker log."""
    try:
        return json.loads(CANARY_REPORT.read_text()), CANARY_REPORT
    except (OSError, json.JSONDecodeError) as report_error:
        try:
            reports = reports_from_log(CANARY_LOG.read_text())
        except OSError as log_error:
            raise SystemExit(
                f"canary evidence unreadable: summary={report_error}; log={log_error}"
            ) from log_error
        passing = [report for report in reports if report.get("status") == "PASS"]
        if not passing:
            raise SystemExit(
                f"canary summary unreadable and worker log has no PASS report: "
                f"{report_error}"
            )
        return passing[-1], CANARY_LOG


def expected_canary_counts(route: dict, selected: set[str]) -> tuple[int, int]:
    stream_keys: set[tuple[str, int, int, int]] = set()
    target_scenes: set[str] = set()
    for prefix, manifest in route["frozen_manifests"].items():
        path = Path(manifest["path"])
        if sha256(path) != manifest["sha256"]:
            raise SystemExit(f"route manifest SHA mismatch: {path}")
        lines = path.read_text().splitlines()
        if len(lines) != int(manifest["rows"]):
            raise SystemExit(f"route manifest row count mismatch: {path}")
        for line in lines:
            row = json.loads(line)
            scene = str(row["scene_id"])
            if scene not in selected:
                continue
            window_id = int(row["streaming_window_id"])
            for layer in route["streaming_layers"]:
                stream_keys.add((scene, window_id, int(prefix), int(layer)))
            window = row["streaming_window_indices"]
            anchor = int(prefix) - 1
            if any(anchor + horizon < len(window) for horizon in (1, 2, 4)):
                target_scenes.add(scene)
    return len(stream_keys), len(target_scenes) * len(route["target_layers"])


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o644)
    Path(temporary).replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards", default="0,1,2,3,4,5,6,7",
        help="comma-separated shard indices",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="validate the route and canary bindings without creating jobs",
    )
    args = parser.parse_args()
    requested = sorted({int(value) for value in args.shards.split(",") if value})
    if any(index < 0 or index >= SHARD_COUNT for index in requested):
        raise SystemExit("invalid shard index")

    route_sha = sha256(ROUTE)
    route = json.loads(ROUTE.read_text())
    if (
        route.get("status") != "PASS"
        or route.get("schema") != "MINIMAX_H3_BF16_PRODUCTION_ROUTE_V2"
        or route.get("scene_count") != 681
        or any(
            manifest.get("rows") != 2721
            for manifest in route.get("frozen_manifests", {}).values()
        )
    ):
        raise SystemExit("route receipt is not the complete 2,721-row PASS route")
    canary, canary_evidence = load_canary_report()
    canary_shard_count = int(canary.get("shard_count", -1))
    canary_shard_index = int(canary.get("shard_index", -1))
    if canary_shard_count <= 0 or not 0 <= canary_shard_index < canary_shard_count:
        raise SystemExit("canary report has invalid shard coordinates")
    scenes = route["ordered_scene_ids"]
    start = canary_shard_index * len(scenes) // canary_shard_count
    end = (canary_shard_index + 1) * len(scenes) // canary_shard_count
    expected_stream, expected_target = expected_canary_counts(
        route, set(scenes[start:end])
    )
    if (
        canary.get("status") != "PASS"
        or canary.get("route_sha256") != route_sha
        or canary_shard_count != 681
        or canary_shard_index != 0
        or canary.get("scene_count") != 1
        or canary.get("streaming_payloads_checked") != expected_stream
        or canary.get("target_payloads_checked") != expected_target
    ):
        raise SystemExit(
            "single-scene canary is not the route-derived expected PASS "
            f"(stream={expected_stream}, target={expected_target})"
        )

    bindings = {
        "route_sha256": route_sha,
        "launcher_sha256": sha256(LAUNCHER),
        "verifier_sha256": sha256(VERIFIER),
        "source_tree_sha256": source_tree_sha256(SOURCE),
        "canary_report_sha256": canonical_sha256(canary),
        "canary_evidence_sha256": sha256(canary_evidence),
    }
    if args.validate_only:
        print(json.dumps({
            "status": "PASS",
            "route": str(ROUTE),
            "canary_evidence": str(canary_evidence),
            "canary_expected_counts": {
                "streaming_payloads_checked": expected_stream,
                "target_payloads_checked": expected_target,
            },
            **bindings,
        }, indent=2, sort_keys=True))
        return
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {
        "schema": "MINIMAX_H3_BF16_V2_PRODUCTION_SUBMISSIONS_V4",
        "status": "SUBMITTING",
        "resource_pool": "zoetrope",
        "worker_spec": "N4lS.Iq.I80.1",
        "shard_count": SHARD_COUNT,
        "route_path": str(ROUTE),
        "launcher_path": str(LAUNCHER),
        "verifier_path": str(VERIFIER),
        "source_path": str(SOURCE),
        "canary_report_path": str(CANARY_REPORT),
        "canary_evidence_path": str(canary_evidence),
        "canary_expected_counts": {
            "streaming_payloads_checked": expected_stream,
            "target_payloads_checked": expected_target,
        },
        **bindings,
        "jobs": {},
    }
    for key, value in bindings.items():
        if ledger.get(key) != value:
            raise SystemExit(f"existing ledger binding mismatch: {key}")

    for shard in requested:
        key = f"{shard:02d}"
        if key in ledger["jobs"]:
            print(f"SKIP shard={key} job={ledger['jobs'][key]['job_id']}")
            continue
        command = (
            f"cd {ROOT} && env "
            f"EXPECTED_ROUTE_SHA256={bindings['route_sha256']} "
            f"EXPECTED_LAUNCHER_SHA256={bindings['launcher_sha256']} "
            f"EXPECTED_VERIFIER_SHA256={bindings['verifier_sha256']} "
            f"EXPECTED_SOURCE_TREE_SHA256={bindings['source_tree_sha256']} "
            f"SHARD_INDEX={shard} SHARD_COUNT={SHARD_COUNT} bash {LAUNCHER}"
        )
        argv = [
            SCO, "acp", "jobs", "create", "--workspace-name=aigc",
            "--aec2-name=zoetrope",
            f"--job-name=bq-mm-bf16v2-s{shard:02d}-r4",
            "--priority=HIGHEST", "--quota-type=reserved",
            "--container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617",
            "--storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm",
            "--training-framework=pytorch", "--worker-nodes=1",
            "--worker-spec=N4lS.Iq.I80.1", f"--command={command}",
        ]
        result = subprocess.run(argv, text=True, capture_output=True, timeout=120)
        output = (result.stdout or "") + (result.stderr or "")
        match = re.search(r"job id\s*:\s*(pt-[a-z0-9]+)", output, re.I)
        if result.returncode or not match:
            ledger["status"] = "PARTIAL_FAILURE"
            ledger["last_error"] = {
                "shard": shard, "returncode": result.returncode, "output": output,
            }
            atomic_json(LEDGER, ledger)
            raise SystemExit(
                f"submission failed shard={key} rc={result.returncode}: {output}"
            )
        ledger["jobs"][key] = {
            "job_id": match.group(1),
            "submitted_utc": datetime.now(timezone.utc).isoformat(),
            "startup_command": command,
            "scene_start": shard * 681 // SHARD_COUNT,
            "scene_end": (shard + 1) * 681 // SHARD_COUNT,
        }
        ledger["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(LEDGER, ledger)
        print(f"SUBMITTED shard={key} job={match.group(1)}")

    if len(ledger["jobs"]) == SHARD_COUNT:
        ledger["status"] = "SUBMITTED"
        ledger["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(LEDGER, ledger)


if __name__ == "__main__":
    main()
