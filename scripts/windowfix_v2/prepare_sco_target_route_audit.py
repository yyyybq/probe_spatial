#!/usr/bin/env python3
"""Render review-only SCO commands for target route source gates.

The renderer never calls ``sco``.  It splits one immutable target merge plan by
VFM, writes SHA-bound shard plans, and records worker/SCO command text for human
review.  Audit jobs are read-only: their only outputs are route reports and gap
worklists under the chosen artifact directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyses.windowfix_v2.audit_target_row_routes import (
    CONTRACT,
    PLAN_TYPE,
    canonical_json_sha,
    sha256_file,
)


class RenderError(ValueError):
    pass


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RenderError(f"plan must be a regular non-symlink file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RenderError("plan is not a JSON object")
    return value


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RenderError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve()


def _atomic(path: Path, payload: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def render(
    *,
    plan_path: Path,
    output_dir: Path | None,
    source_timeout_seconds: float,
    workspace: str,
    worker_spec: str,
    image: str,
    storage_mount: str,
    worker_wrapper: Path | None = None,
    binding_schema: Path | None = None,
) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    plan = _json(plan_path)
    if plan.get("plan_type") != PLAN_TYPE or plan.get("contract") != CONTRACT:
        raise RenderError("target merge plan contract/type mismatch")
    actual_content = canonical_json_sha(
        {key: value for key, value in plan.items() if key != "content_sha256"}
    )
    if plan.get("content_sha256") != actual_content:
        raise RenderError("target merge plan content SHA mismatch")
    if (
        isinstance(source_timeout_seconds, bool)
        or not isinstance(source_timeout_seconds, (int, float))
        or not math.isfinite(source_timeout_seconds)
        or source_timeout_seconds <= 0
    ):
        raise RenderError("source timeout must be positive")
    if output_dir is None:
        output_dir = (
            plan_path.parent / f"sco_audit_review_{actual_content[:12]}"
        )
    output_dir = output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise RenderError(f"review output already exists; overwrite forbidden: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in plan.get("units", []):
        if not isinstance(raw, dict):
            raise RenderError("plan contains a non-object unit")
        groups[str(raw["vfm"])].append(raw)
    if len(groups) != 10 or sum(map(len, groups.values())) != 21_792:
        raise RenderError("expected 10 VFM shards and 21,792 units")

    audit_script = _regular(
        PROJECT_ROOT / "analyses/windowfix_v2/audit_target_row_routes.py", "auditor"
    )
    source_reader = _regular(
        PROJECT_ROOT / "analyses/windowfix_v2/target_row_source.py", "source reader"
    )
    worker_wrapper = _regular(
        worker_wrapper
        or PROJECT_ROOT / "scripts/windowfix_v2/run_target_route_audit_worker.py",
        "worker wrapper",
    )
    binding_schema = _regular(
        binding_schema
        or PROJECT_ROOT
        / "scripts/windowfix_v2/target_route_audit_worker_binding_v1.schema.json",
        "worker binding schema",
    )
    schema_value = _json(binding_schema)
    schema_id = "canonical_v2_target_route_audit_worker_binding_v1.schema.json"
    if schema_value.get("$id") != schema_id:
        raise RenderError("worker binding schema identity mismatch")
    parent_file_sha = sha256_file(plan_path)
    audit_script_sha = sha256_file(audit_script)
    source_reader_sha = sha256_file(source_reader)
    worker_wrapper_sha = sha256_file(worker_wrapper)
    binding_schema_sha = sha256_file(binding_schema)
    python = Path(sys.executable).resolve()
    shards = []
    worker_commands = []
    sco_commands = []
    for ordinal, vfm in enumerate(sorted(groups), 1):
        shard = {key: value for key, value in plan.items() if key not in {"content_sha256", "units"}}
        shard["parent_plan"] = {
            "path": str(plan_path),
            "file_sha256": parent_file_sha,
            "content_sha256": actual_content,
        }
        shard["units"] = groups[vfm]
        shard["content_sha256"] = canonical_json_sha(shard)
        shard_path = output_dir / "shards" / f"{ordinal:02d}-{vfm}.json"
        shard_bytes = _json_bytes(shard)
        _atomic(shard_path, shard_bytes)
        shard_sha = sha256_file(shard_path)
        audit_output = output_dir / "outputs" / vfm
        binding_receipt: dict[str, Any] = {
            "schema_version": 1,
            "receipt_type": "canonical_v2_target_route_audit_worker_binding",
            "status": "REVIEW_ONLY_NOT_EXECUTED",
            "parent_plan": {
                "path": str(plan_path),
                "file_sha256": parent_file_sha,
                "content_sha256": actual_content,
            },
            "shard": {
                "path": str(shard_path),
                "file_sha256": shard_sha,
                "content_sha256": shard["content_sha256"],
            },
            "auditor": {
                "path": str(audit_script),
                "file_sha256": audit_script_sha,
            },
            "source_reader": {
                "path": str(source_reader),
                "file_sha256": source_reader_sha,
            },
            "worker_wrapper": {
                "path": str(worker_wrapper),
                "file_sha256": worker_wrapper_sha,
            },
            "binding_schema": {
                "path": str(binding_schema),
                "file_sha256": binding_schema_sha,
                "schema_id": schema_id,
            },
            "audit_output": str(audit_output),
            "audit_arguments": {
                "remote_policy": "gate",
                "source_timeout_seconds": source_timeout_seconds,
                "verify_source_sha": True,
            },
        }
        binding_receipt["content_sha256"] = canonical_json_sha(binding_receipt)
        binding_path = output_dir / "bindings" / f"{ordinal:02d}-{vfm}.json"
        _atomic(binding_path, _json_bytes(binding_receipt))
        binding_sha = sha256_file(binding_path)
        worker_argv = [
            str(python),
            str(worker_wrapper),
            "--binding-receipt",
            str(binding_path),
            "--binding-schema",
            str(binding_schema),
            "--expected-binding-receipt-sha256",
            binding_sha,
            "--expected-binding-schema-sha256",
            binding_schema_sha,
            "--expected-parent-plan-sha256",
            parent_file_sha,
            "--expected-shard-sha256",
            shard_sha,
            "--expected-auditor-sha256",
            audit_script_sha,
            "--expected-source-reader-sha256",
            source_reader_sha,
            "--expected-worker-wrapper-sha256",
            worker_wrapper_sha,
        ]
        worker = "cd " + shlex.quote(str(PROJECT_ROOT)) + " && " + shlex.join(worker_argv)
        job_name = f"bq-wf2-tgt-audit-{ordinal:02d}-{vfm}"[:63]
        sco_argv = [
            "sco", "acp", "jobs", "create",
            f"--workspace-name={workspace}",
            "--aec2-name=h800",
            f"--job-name={job_name}",
            "--priority=HIGHEST",
            f"--container-image-url={image}",
            f"--storage-mount={storage_mount}",
            "--training-framework=pytorch",
            "--worker-nodes=1",
            f"--worker-spec={worker_spec}",
            f"--command={worker}",
        ]
        worker_commands.append(worker)
        sco_commands.append(shlex.join(sco_argv))
        layer_counts = Counter(int(unit["layer"]) for unit in groups[vfm])
        shards.append(
            {
                "ordinal": ordinal,
                "vfm": vfm,
                "plan": str(shard_path),
                "plan_file_sha256": shard_sha,
                "plan_content_sha256": shard["content_sha256"],
                "worker_binding_receipt": str(binding_path),
                "worker_binding_receipt_file_sha256": binding_sha,
                "worker_binding_receipt_content_sha256": binding_receipt[
                    "content_sha256"
                ],
                "units": len(groups[vfm]),
                "layers": sorted(layer_counts),
                "scenes": len({str(unit["scene_id"]) for unit in groups[vfm]}),
                "audit_output": str(audit_output),
                "worker_command": worker,
                "sco_argv": sco_argv,
            }
        )

    worker_path = output_dir / "worker_commands.txt"
    sco_path = output_dir / "sco_create_commands.txt"
    _atomic(worker_path, ("\n".join(worker_commands) + "\n").encode())
    _atomic(sco_path, ("\n".join(sco_commands) + "\n").encode())
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "plan_type": "canonical_v2_target_route_audit_sco_review",
        "status": "READY_FOR_REVIEW_NOT_SUBMITTED",
        "submission_performed": False,
        "feature_payload_written": False,
        "source_plan": {
            "path": str(plan_path),
            "file_sha256": parent_file_sha,
            "content_sha256": actual_content,
        },
        "auditor": {
            "path": str(audit_script),
            "sha256": audit_script_sha,
        },
        "source_reader": {
            "path": str(source_reader),
            "sha256": source_reader_sha,
        },
        "worker_wrapper": {
            "path": str(worker_wrapper),
            "sha256": worker_wrapper_sha,
        },
        "worker_binding_schema": {
            "path": str(binding_schema),
            "sha256": binding_schema_sha,
            "schema_id": schema_id,
        },
        "remote_policy": "gate",
        "source_timeout_seconds": source_timeout_seconds,
        "verify_local_source_sha": True,
        "counts": {
            "shards": len(shards),
            "worker_binding_receipts": len(shards),
            "units": sum(item["units"] for item in shards),
            "scenes_per_shard": 681,
            "sco_commands_rendered": len(sco_commands),
            "sco_commands_executed": 0,
        },
        "execution_preflight": {
            "required_before_remote_reads": True,
            "fail_closed": True,
            "full_file_sha256_bindings": [
                "parent_plan",
                "shard",
                "auditor",
                "source_reader",
                "worker_wrapper",
                "worker_binding_receipt",
                "worker_binding_schema",
            ],
            "content_sha256_bindings": [
                "parent_plan",
                "shard",
                "worker_binding_receipt",
            ],
        },
        "shards": shards,
        "worker_commands": {
            "path": str(worker_path),
            "sha256": sha256_file(worker_path),
        },
        "sco_create_commands": {
            "path": str(sco_path),
            "sha256": sha256_file(sco_path),
            "execution_policy": "review text only; this renderer never invokes sco",
        },
    }
    manifest["content_sha256"] = canonical_json_sha(manifest)
    manifest_path = output_dir / "manifest.json"
    _atomic(manifest_path, _json_bytes(manifest))
    return {**manifest, "manifest": str(manifest_path), "manifest_file_sha256": sha256_file(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--worker-wrapper", type=Path)
    parser.add_argument("--binding-schema", type=Path)
    parser.add_argument("--workspace", default="aigc")
    parser.add_argument("--worker-spec", default="N4lS.Iq.I80.8")
    parser.add_argument(
        "--image",
        default="registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617",
    )
    parser.add_argument(
        "--storage-mount",
        default="019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm",
    )
    args = parser.parse_args()
    result = render(
        plan_path=args.plan,
        output_dir=args.output_dir,
        source_timeout_seconds=args.source_timeout_seconds,
        workspace=args.workspace,
        worker_spec=args.worker_spec,
        image=args.image,
        storage_mount=args.storage_mount,
        worker_wrapper=args.worker_wrapper,
        binding_schema=args.binding_schema,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "submission_performed": result["submission_performed"],
                "counts": result["counts"],
                "manifest": result["manifest"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
