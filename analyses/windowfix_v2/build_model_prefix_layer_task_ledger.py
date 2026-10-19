#!/usr/bin/env python3
"""Build an immutable, positive-only canonical-v2 feature work ledger.

The ledger's *logical* identity is deliberately human-readable and stable:
``(model, prefix, layer, task)``.  A streaming prefix feature uses p8/p12/p16/
p24; an isolated target is shared across those prefixes and uses ``prefix=iso``.

This is not a job renderer.  A scheduler may serialise several logical units
for one model in one physical SCO job, but every extracted payload remains
accountable to one of the ledger keys below.  The input route audits are the
only authority: absent/PENDING rows are never inferred as extraction work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = "CANONICAL_V2_MODEL_PREFIX_LAYER_TASK_LEDGER_V1"
TASK_BY_MODE = {
    "streaming_prefix": "streaming_prefix",
    "target_isolated": "target_isolated",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def regular_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    return value


def verify_content(value: dict[str, Any], label: str) -> str:
    stored = value.get("content_sha256")
    unhashed = dict(value)
    unhashed.pop("content_sha256", None)
    actual = canonical_json_sha(unhashed)
    if stored != actual:
        raise ValueError(f"{label} content SHA mismatch")
    return actual


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o444)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def audit_worklists(audit_path: Path) -> Iterable[dict[str, Any]]:
    audit = regular_json(audit_path, "source audit")
    audit_content = verify_content(audit, "source audit")
    if audit.get("status") not in {"INCOMPLETE", "PENDING", "PASS"}:
        raise ValueError(f"source audit is unsafe for positive-only work: {audit_path}")
    if audit.get("hard_errors") or int(audit.get("counts", {}).get("hard_errors", 0)):
        raise ValueError(f"source audit has hard errors: {audit_path}")
    declared = audit.get("outputs", {}).get("gap_worklists")
    if not isinstance(declared, list) or not declared:
        raise ValueError(f"source audit has no gap worklists: {audit_path}")
    for binding in declared:
        if not isinstance(binding, dict):
            raise ValueError(f"source audit has an invalid worklist binding: {audit_path}")
        worklist_path = Path(str(binding.get("path", "")))
        worklist = regular_json(worklist_path, "audit-bound worklist")
        content = verify_content(worklist, "audit-bound worklist")
        actual_file_sha = sha256_file(worklist_path)
        if (
            binding.get("file_sha256") != actual_file_sha
            or binding.get("content_sha256") != content
        ):
            raise ValueError(f"audit/worklist SHA mismatch: {worklist_path}")
        yield {
            "audit_path": str(audit_path.resolve()),
            "audit_file_sha256": sha256_file(audit_path),
            "audit_content_sha256": audit_content,
            "worklist_path": str(worklist_path.resolve()),
            "worklist_file_sha256": actual_file_sha,
            "worklist_content_sha256": content,
            "worklist": worklist,
        }


def logical_key(
    *, vfm: str, model_id: str, timestep: int, prefix: str, layer: int, task: str
) -> tuple[str, str, int, str, int, str]:
    return (vfm, model_id, int(timestep), prefix, int(layer), task)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ready-vfm", action="append", default=[])
    parser.add_argument("--blocked-vfm", action="append", default=[])
    parser.add_argument("--physical-concurrency", type=int, default=2)
    args = parser.parse_args()
    if args.physical_concurrency < 1:
        raise SystemExit("--physical-concurrency must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to replace immutable output: {output_dir}")
    ready = set(args.ready_vfm)
    blocked = set(args.blocked_vfm)
    if ready & blocked:
        raise SystemExit(f"backend listed both ready and blocked: {sorted(ready & blocked)}")

    bindings = [
        binding
        for audit in args.source_audit
        for binding in audit_worklists(audit.resolve())
    ]
    units: dict[tuple[str, str, int, str, int, str], dict[str, Any]] = {}
    for binding in bindings:
        worklist = binding["worklist"]
        mode = str(worklist.get("mode"))
        if mode not in TASK_BY_MODE:
            raise SystemExit(f"unsupported worklist mode: {mode!r}")
        task = TASK_BY_MODE[mode]
        vfm = str(worklist["vfm"])
        model_id = str(worklist["model_id"])
        timestep = int(worklist["timestep"])
        for entry in worklist.get("entries", []):
            if not isinstance(entry, dict):
                raise SystemExit(f"invalid worklist entry: {binding['worklist_path']}")
            prefix = f"p{int(entry['prefix'])}" if mode == "streaming_prefix" else "iso"
            reason = str(entry["reason"])
            for layer in entry["layers"]:
                key = logical_key(
                    vfm=vfm,
                    model_id=model_id,
                    timestep=timestep,
                    prefix=prefix,
                    layer=int(layer),
                    task=task,
                )
                item = units.setdefault(
                    key,
                    {
                        "contract": CONTRACT,
                        "model": vfm,
                        "model_id": model_id,
                        "timestep": timestep,
                        "prefix": prefix,
                        "layer": int(layer),
                        "task": task,
                        "logical_unit_id": (
                            f"{vfm}__{prefix}__layer{int(layer)}__{task}"
                        ),
                        "layer_items": 0,
                        "reason_counts": Counter(),
                        "source_bindings": [],
                    },
                )
                item["layer_items"] += 1
                item["reason_counts"][reason] += 1
                source = {
                    key: binding[key]
                    for key in (
                        "audit_path",
                        "audit_file_sha256",
                        "audit_content_sha256",
                        "worklist_path",
                        "worklist_file_sha256",
                        "worklist_content_sha256",
                    )
                }
                if source not in item["source_bindings"]:
                    item["source_bindings"].append(source)

    rendered_units = []
    for key, item in sorted(units.items()):
        model = item["model"]
        state = "READY" if model in ready else "BLOCKED" if model in blocked else "UNCLASSIFIED"
        item["execution_state"] = state
        item["physical_batch_key"] = f"{model}__serial"
        item["reason_counts"] = dict(sorted(item["reason_counts"].items()))
        item["source_bindings"] = sorted(
            item["source_bindings"], key=lambda value: value["worklist_path"]
        )
        rendered_units.append(item)

    per_model: dict[str, dict[str, Any]] = {}
    for item in rendered_units:
        current = per_model.setdefault(
            item["model"],
            {
                "model": item["model"],
                "model_id": item["model_id"],
                "timestep": item["timestep"],
                "execution_state": item["execution_state"],
                "logical_units": 0,
                "layer_items": 0,
                "by_task": defaultdict(int),
            },
        )
        current["logical_units"] += 1
        current["layer_items"] += item["layer_items"]
        current["by_task"][item["task"]] += item["layer_items"]
    per_model_rows = []
    for row in sorted(per_model.values(), key=lambda item: item["model"]):
        row["by_task"] = dict(sorted(row["by_task"].items()))
        per_model_rows.append(row)

    counts = {
        "logical_units": len(rendered_units),
        "layer_items": sum(item["layer_items"] for item in rendered_units),
        "streaming_logical_units": sum(
            item["task"] == "streaming_prefix" for item in rendered_units
        ),
        "target_logical_units": sum(
            item["task"] == "target_isolated" for item in rendered_units
        ),
        "ready_layer_items": sum(
            item["layer_items"] for item in rendered_units if item["execution_state"] == "READY"
        ),
        "blocked_layer_items": sum(
            item["layer_items"] for item in rendered_units if item["execution_state"] == "BLOCKED"
        ),
        "unclassified_layer_items": sum(
            item["layer_items"] for item in rendered_units if item["execution_state"] == "UNCLASSIFIED"
        ),
        "physical_model_batches": len(per_model_rows),
    }
    payload = {
        "schema_version": 1,
        "contract": CONTRACT,
        "status": "READY_FOR_CONTROLLED_SUBMISSION",
        "submission_performed": False,
        "physical_submission_policy": {
            "logical_key": ["model", "prefix", "layer", "task"],
            "target_prefix_sentinel": "iso",
            "physical_batch_key": "model__serial",
            "max_active_sco_jobs": args.physical_concurrency,
            "why": (
                "A model batch reuses one forward pass for sibling layers and "
                "serializes namespace/index writers; it does not create one SCO job per logical unit."
            ),
        },
        "source_audits": [
            {"path": str(path.resolve()), "file_sha256": sha256_file(path.resolve())}
            for path in args.source_audit
        ],
        "counts": counts,
        "models": per_model_rows,
        "units": rendered_units,
    }
    payload["content_sha256"] = canonical_json_sha(payload)

    staging = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        atomic_json(staging / "ledger.json", payload)
        jsonl = "".join(json.dumps(item, sort_keys=True) + "\n" for item in rendered_units)
        atomic_text(staging / "units.jsonl", jsonl)
        fields = [
            "logical_unit_id", "execution_state", "model", "model_id", "timestep",
            "prefix", "layer", "task", "layer_items", "physical_batch_key", "reason_counts",
        ]
        from io import StringIO

        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        for item in rendered_units:
            writer.writerow({
                **{key: item[key] for key in fields if key != "reason_counts"},
                "reason_counts": json.dumps(item["reason_counts"], sort_keys=True),
            })
        atomic_text(staging / "units.csv", buffer.getvalue())
        lines = [
            "# Canonical-v2 model / prefix / layer / task ledger",
            "",
            f"- Logical units: `{counts['logical_units']}`",
            f"- Feature layer-items to produce: `{counts['layer_items']}`",
            f"- Ready now: `{counts['ready_layer_items']}`; backend-blocked: `{counts['blocked_layer_items']}`",
            f"- Physical policy: at most `{args.physical_concurrency}` SCO model batches active at once.",
            "- `prefix=iso` is intentional: one isolated target feature serves every prefix/horizon that needs that raw target frame.",
            "",
            "| Model | State | Logical units | Layer-items | streaming / target |",
            "| --- | --- | ---: | ---: | --- |",
        ]
        for row in per_model_rows:
            by_task = row["by_task"]
            lines.append(
                f"| {row['model']} | {row['execution_state']} | {row['logical_units']} | "
                f"{row['layer_items']} | {by_task.get('streaming_prefix', 0)} / {by_task.get('target_isolated', 0)} |"
            )
        lines.extend([
            "",
            "The full machine-readable ledger is `units.jsonl` / `units.csv`. No PENDING legacy route is included.",
        ])
        atomic_text(staging / "README.md", "\n".join(lines) + "\n")
        lock = {
            "schema_version": 1,
            "ledger_sha256": sha256_file(staging / "ledger.json"),
            "units_jsonl_sha256": sha256_file(staging / "units.jsonl"),
            "units_csv_sha256": sha256_file(staging / "units.csv"),
            "readme_sha256": sha256_file(staging / "README.md"),
        }
        atomic_json(staging / "PLAN.lock.json", lock)
        os.replace(staging, output_dir)
    except Exception:
        for path in sorted(staging.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        staging.rmdir()
        raise
    print(json.dumps({"output_dir": str(output_dir), "counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
