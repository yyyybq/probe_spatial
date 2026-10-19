#!/usr/bin/env python3
"""Publish prepared canonical-v2 roots only after four full alignment audits.

Dry-run is the default.  Execution requires both ``--execute`` and the exact
plan id as ``--confirmation``.  Local tensors are never deleted by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"ERROR: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        fail(f"{label} must be a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"cannot read {label}: {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} is not a JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def credentials(conf: Path) -> tuple[str, str]:
    section = ""
    access = ""
    secret = ""
    for raw in conf.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section != "fj" or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key == "access_key":
            access = value
        elif key == "secret_key":
            secret = value
    if not access or not secret:
        fail(f"cannot load fj AOSS credentials from {conf}")
    return access, secret


def validate_audits(
    plan_dir: Path, canonical_binding: dict[str, Any]
) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    audit_plan = read_json(plan_dir / "audit_plan.json", "audit plan")
    entries = audit_plan.get("audits")
    if not isinstance(entries, list):
        fail("audit plan has no audits list")
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        if not isinstance(entry, dict):
            fail("invalid audit plan entry")
        report_path = Path(str(entry.get("output", "")))
        report = read_json(report_path, "full alignment audit")
        expected_prefix = int(entry["prefix"])
        expected = {
            "ok": True,
            "status": "pass",
            "scope": "full",
            "protocol_id": "inscene_streaming_canonical_v2_obs28_h124",
            "canonical_obs_len": 28,
            "vfm": entry["vfm"],
            "problem_count": 0,
            "include_anchors": True,
        }
        for key, value in expected.items():
            if report.get(key) != value:
                fail(f"audit {report_path} field {key} is {report.get(key)!r}, expected {value!r}")
        if expected_prefix not in report.get("prefixes", []):
            fail(f"audit {report_path} does not bind prefix {expected_prefix}")
        if report.get("binding_mode") != "compact_obs28_source_v1":
            fail(f"audit {report_path} does not use compact obs28 binding")
        if report.get("derived_prefix") != expected_prefix:
            fail(f"audit {report_path} derived-prefix mismatch")
        for field in ("canonical_source", "namespace_lock", "label_lineage_audit"):
            expected_binding = canonical_binding.get(field, {})
            actual_binding = report.get(field, {})
            if (
                actual_binding.get("path") != expected_binding.get("path")
                or actual_binding.get("sha256") != expected_binding.get("sha256")
            ):
                fail(f"audit {report_path} compact binding mismatch: {field}")
        if report.get("manifest_sha256") != sha256_file(Path(report["manifest"])):
            fail(f"audit {report_path} has stale manifest SHA")
        grouped[(str(entry["vfm"]), str(entry["model_id"]), int(entry["timestep"]))].append(entry)
    for key, group in grouped.items():
        if {int(item["prefix"]) for item in group} != {8, 12, 16, 24}:
            fail(f"model group lacks four full audits: {key}")
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--ads-cli", type=Path, default=Path("/mnt/umm/users/yinbaiqiao/aoss/ads-cli"))
    parser.add_argument("--aoss-conf", type=Path, default=Path("/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf"))
    parser.add_argument("--aoss-host", default="aoss-internal.cn-fz-01.fjscmsapi-oss.com")
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()

    plan_dir = args.plan_dir.resolve()
    summary = read_json(plan_dir / "summary.json", "plan summary")
    lock = read_json(plan_dir / "PLAN.lock.json", "plan lock")
    if lock.get("summary_sha256") != sha256_file(plan_dir / "summary.json"):
        fail("plan summary changed after lock creation")
    if summary.get("status") != "READY_FOR_REVIEW" or summary.get("blocked"):
        fail("plan contains blocked work and cannot be published")
    plan_id = str(summary.get("plan_id", ""))
    if not plan_id or lock.get("plan_id") != plan_id:
        fail("plan id mismatch")
    canonical_binding = summary.get("canonical_binding")
    if not isinstance(canonical_binding, dict):
        fail("plan summary has no compact canonical binding")
    audited = validate_audits(plan_dir, canonical_binding)

    for name, expected_sha in lock.get("batches", {}).items():
        path = plan_dir / "batches" / name
        if sha256_file(path) != expected_sha:
            fail(f"batch changed after lock creation: {path}")

    roots: dict[Path, str] = {}
    for name, expected_sha in lock.get("shards", {}).items():
        path = plan_dir / "shards" / name
        if sha256_file(path) != expected_sha:
            fail(f"shard changed after lock creation: {path}")
        shard = read_json(path, "shard")
        model_key = (str(shard["vfm"]), str(shard["model_id"]), int(shard["timestep"]))
        if model_key not in audited:
            fail(f"no complete audit set for shard {shard['shard_id']}")
        root = Path(str(shard["feature_root"]))
        if not root.is_dir() or root.is_symlink():
            fail(f"feature root is missing or symlinked: {root}")
        uri = str(shard["publish_uri"])
        previous = roots.get(root)
        if previous is not None and previous != uri:
            fail(f"one local root maps to multiple publish URIs: {root}")
        roots[root] = uri

    commands = [
        {
            "source": f"{root}/",
            "destination": f"{uri}/",
            "display": f"{args.ads_cli} -p {args.threads} sync {root}/ {uri}/",
        }
        for root, uri in sorted(roots.items(), key=lambda item: str(item[0]))
    ]
    if not args.execute:
        print(json.dumps({"dry_run": True, "plan_id": plan_id, "commands": commands}, indent=2))
        return
    if args.confirmation != plan_id:
        fail(f"execution requires --confirmation {plan_id}")
    if not args.ads_cli.is_file() or not os.access(args.ads_cli, os.X_OK):
        fail(f"ads-cli is unavailable: {args.ads_cli}")
    if not args.aoss_conf.is_file() or args.aoss_conf.is_symlink():
        fail(f"AOSS config is unavailable or symlinked: {args.aoss_conf}")
    access, secret = credentials(args.aoss_conf)
    host = args.aoss_host
    environment = os.environ.copy()
    environment["NO_PROXY"] = f"localhost,127.0.0.1,::1,baiqiao.{host},.{host},{host}"
    environment["no_proxy"] = environment["NO_PROXY"]
    receipts: list[dict[str, Any]] = []
    for root, uri in sorted(roots.items(), key=lambda item: str(item[0])):
        prefix = "aoss://baiqiao/"
        suffix = uri[len(prefix) :] if uri.startswith(prefix) else uri
        if suffix == uri or ".." in Path(suffix).parts:
            fail(f"invalid bound publish URI: {uri}")
        destination = f"s3://{access}:{secret}@baiqiao.{host}/{suffix}/"
        result = subprocess.run(
            [str(args.ads_cli), "-p", str(args.threads), "sync", f"{root}/", destination],
            env=environment,
            check=False,
        )
        receipts.append({"source": str(root), "publish_uri": uri, "returncode": result.returncode})
        if result.returncode:
            write_json_atomic(plan_dir / "publish_receipt.json", {"plan_id": plan_id, "status": "FAILED", "receipts": receipts})
            fail(f"AOSS sync failed for {uri}: rc={result.returncode}")
    write_json_atomic(
        plan_dir / "publish_receipt.json",
        {"schema_version": 1, "plan_id": plan_id, "status": "PUBLISHED", "receipts": receipts},
    )


if __name__ == "__main__":
    main()
