#!/usr/bin/env python3
"""Build disjoint single-GPU workers from the frozen probe528 atomic queue.

The scheduler unit is a long-lived single-GPU worker, while ownership and
retry identity remain one queue row: model/layer/scene and, for streaming,
prefix.  Rows assigned to a worker are merged only to avoid repeated model
loads; no row is added, dropped, or assigned twice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_WORKERS = {
    ("f3r", "streaming_prefix"): 2,
    ("f3r", "target_isolated"): 1,
    ("qwen2_5_vl", "streaming_prefix"): 2,
    ("cogvideox", "streaming_prefix"): 4,
    ("cogvideox", "target_isolated"): 2,
    ("dino", "streaming_prefix"): 1,
    ("dino", "target_isolated"): 1,
    ("lingbot", "streaming_prefix"): 4,
    ("lingbot", "target_isolated"): 2,
    ("videomae", "streaming_prefix"): 2,
    ("videomae", "target_isolated"): 2,
    ("vjepa2", "streaming_prefix"): 2,
    ("vjepa2", "target_isolated"): 2,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Any, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_text(path: Path, value: str, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def merged_entries(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        for entry in row["entries"]:
            if mode == "streaming_prefix":
                key = (
                    str(entry["scene"]), int(entry["window_id"]),
                    int(entry["prefix"]), str(entry["reason"]),
                )
            else:
                key = (
                    str(entry["scene"]), int(entry["frame_index"]),
                    str(entry["reason"]),
                )
            if key not in grouped:
                grouped[key] = {k: v for k, v in entry.items() if k != "layers"}
                grouped[key]["layers"] = []
            grouped[key]["layers"].extend(int(value) for value in entry["layers"])
    result = []
    for key in sorted(grouped):
        entry = grouped[key]
        entry["layers"] = sorted(set(entry["layers"]))
        result.append(entry)
    return result


def worker_spec(
    *, rows: list[dict[str, Any]], template: dict[str, Any], output: Path,
    owner: str, queue_sha: str, runner: Path, sync_loop: Path, claim_script: Path,
    claim_root: Path,
) -> dict[str, Any]:
    mode = str(rows[0]["mode"])
    model = str(rows[0]["model"])
    task_path = output / "tasks" / f"{owner}.jsonl"
    task_text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in sorted(rows, key=lambda value: value["unique_key"])
    )
    atomic_text(task_path, task_text)
    task_sha = sha256(task_path)

    parent_paths = {Path(str(row["parent_worklist"]["path"])).resolve() for row in rows}
    if len(parent_paths) != 1:
        raise RuntimeError(f"worker {owner} spans multiple parent worklists")
    parent_path = next(iter(parent_paths))
    parent = json.loads(parent_path.read_text())
    entries = merged_entries(rows, mode)
    worklist = {k: v for k, v in parent.items() if k not in {"entries", "counts", "content_sha256"}}
    worklist["output_layers"] = sorted({int(row["layer"]) for row in rows})
    worklist["entries"] = entries
    worklist["counts"] = {
        "entries": len(entries),
        "layer_items": sum(len(entry["layers"]) for entry in entries),
        "scenes": len({str(row["scene_id"]) for row in rows}),
    }
    worklist["content_sha256"] = canonical_sha(worklist)
    worklist_path = output / "worklists" / f"{owner}.json"
    atomic_json(worklist_path, worklist)
    worklist_sha = sha256(worklist_path)

    parent_audit_path = Path(str(template["source_audit_path"])).resolve()
    audit = json.loads(parent_audit_path.read_text())
    audit.pop("content_sha256", None)
    audit["audit_type"] = "PROBE528_ATOMIC_OWNER_ALLOWLIST_V1"
    audit["created_utc"] = datetime.now(timezone.utc).isoformat()
    audit["counts"] = {
        "hard_errors": 0,
        "streaming_layer_items": worklist["counts"]["layer_items"] if mode == "streaming_prefix" else 0,
        "target_layer_items": worklist["counts"]["layer_items"] if mode == "target_isolated" else 0,
        "worklists": 1,
    }
    audit["hard_errors"] = []
    audit["outputs"] = {"gap_worklists": [{
        "path": str(worklist_path),
        "file_sha256": worklist_sha,
        "content_sha256": worklist["content_sha256"],
        "counts": worklist["counts"],
        "mode": mode,
        "vfm": model,
        "timestep": int(worklist["timestep"]),
    }]}
    audit["atomic_owner"] = {
        "owner": owner,
        "queue_sha256": queue_sha,
        "task_list": str(task_path),
        "task_list_sha256": task_sha,
        "atomic_tasks": len(rows),
    }
    audit["content_sha256"] = canonical_sha(audit)
    audit_path = output / "audits" / f"{owner}.json"
    atomic_json(audit_path, audit)

    shard = dict(template)
    shard.update({
        "plan_id": f"probe528-atomic-owner-{queue_sha[:12]}",
        "shard_id": owner,
        "source_audit_path": str(audit_path),
        "source_audit_sha256": sha256(audit_path),
        "source_audit_content_sha256": audit["content_sha256"],
        "worklist_path": str(worklist_path),
        "worklist_file_sha256": worklist_sha,
        "worklist_content_sha256": worklist["content_sha256"],
        "layers": worklist["output_layers"],
        "counts": worklist["counts"],
        "scene_start": 0,
        "scene_end": 682,
        "n_gpus": 1,
    })
    shard["source_bindings"] = dict(shard["source_bindings"])
    for binding_name in ("record_filter", "extractor", "aoss_gap_adapter"):
        binding = dict(shard["source_bindings"][binding_name])
        binding_path = Path(str(binding["path"])).resolve()
        binding["sha256"] = sha256(binding_path)
        shard["source_bindings"][binding_name] = binding
    shard["source_bindings"]["backend_dependencies"] = {
        name: {**binding, "sha256": sha256(Path(str(binding["path"])).resolve())}
        for name, binding in shard["source_bindings"]["backend_dependencies"].items()
    }
    shard["source_bindings"]["runner"] = {"path": str(runner), "sha256": sha256(runner)}
    shard["source_bindings"]["aoss_sync_loop"] = {
        "path": str(sync_loop), "sha256": sha256(sync_loop)
    }
    shard.pop("content_sha256", None)
    shard["content_sha256"] = canonical_sha(shard)
    shard_path = output / "shards" / f"{owner}.json"
    atomic_json(shard_path, shard)

    batch = {
        "schema_version": 1,
        "status": "READY",
        "plan_id": shard["plan_id"],
        "batch_id": owner,
        "vfm": model,
        "serialization_scope": f"immutable atomic owner {owner}",
        "serialization_reason": "one disjoint scene set; atomic rows have one owner",
        "shards": [{
            "path": str(shard_path), "sha256": sha256(shard_path),
            "shard_id": owner, "mode": mode, "timestep": int(shard["timestep"]),
        }],
    }
    batch["content_sha256"] = canonical_sha(batch)
    batch_path = output / "batches" / f"{owner}.json"
    atomic_json(batch_path, batch)

    log_path = output / "logs" / f"{owner}.log"
    script_path = output / "scripts" / f"{owner}.sh"
    python = str(shard["python"])
    commands = [
        shlex.join([
            python, str(claim_script), "--tasks", str(task_path),
            "--tasks-sha256", task_sha, "--queue-sha256", queue_sha,
            "--owner", owner, "--claim-root", str(claim_root),
        ]),
        shlex.join([python, str(runner), "--batch", str(batch_path), "--execute"]),
    ]
    body = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"mkdir -p {shlex.quote(str(log_path.parent))}\n"
        f"exec >{shlex.quote(str(log_path))} 2>&1\n"
        + "\n".join(commands) + "\n"
    )
    atomic_text(script_path, body, 0o555)
    job_name = f"bq-p528-atomic-{model[:12]}-{'s' if mode == 'streaming_prefix' else 't'}-{owner[-2:]}-r1"
    argv = [
        "sco", "acp", "jobs", "create", "--workspace-name=aigc",
        "--aec2-name=zoetrope", f"--job-name={job_name}", "--priority=HIGHEST",
        "--quota-type=reserved",
        "--container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617",
        "--storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm",
        "--training-framework=pytorch", "--worker-nodes=1", "--worker-spec=N4lS.Iq.I80.1",
        f"--command=cd /mnt/umm/users/yinbaiqiao/probe_spatial && bash {script_path}",
    ]
    return {
        "owner": owner, "model": model, "mode": mode, "atomic_tasks": len(rows),
        "scenes": worklist["counts"]["scenes"], "layer_items": worklist["counts"]["layer_items"],
        "task_list": str(task_path), "task_list_sha256": task_sha,
        "worklist": str(worklist_path), "worklist_sha256": worklist_sha,
        "shard": str(shard_path), "batch": str(batch_path), "script": str(script_path),
        "job_name": job_name, "sco_argv": argv, "sco_command": shlex.join(argv),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--queue-sha256", required=True)
    parser.add_argument("--template", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--claim-root",
        type=Path,
        help="Reuse a prior immutable owner's claim root when resuming a revised runner.",
    )
    parser.add_argument(
        "--scene-limit",
        type=int,
        help="Canary only: retain the first N exact scene IDs per model/mode.",
    )
    parser.add_argument(
        "--exclude-task-list",
        action="append",
        type=Path,
        help="JSONL atomic task list whose successfully completed keys are excluded.",
    )
    args = parser.parse_args()
    queue = args.queue.resolve()
    if sha256(queue) != args.queue_sha256:
        raise RuntimeError("frozen atomic queue SHA mismatch")
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite immutable plan: {output}")
    output.mkdir(parents=True)
    claim_root = (
        args.claim_root.resolve()
        if args.claim_root is not None
        else output / "runtime" / "claims"
    )
    for name in ("tasks", "worklists", "audits", "shards", "batches", "scripts", "logs", "jobspecs", "runtime"):
        (output / name).mkdir()

    templates: dict[tuple[str, str], dict[str, Any]] = {}
    for path in args.template:
        value = json.loads(path.resolve().read_text())
        key = (str(value["vfm"]), str(value["mode"]))
        if key in templates:
            raise RuntimeError(f"duplicate template: {key}")
        templates[key] = value
    rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for line in queue.open():
        row = json.loads(line)
        key = (str(row["model"]), str(row["mode"]))
        if key in templates:
            rows_by_key[key].append(row)
    excluded_keys: set[str] = set()
    for exclude_path in args.exclude_task_list or []:
        for line in exclude_path.resolve().open():
            if line.strip():
                excluded_keys.add(str(json.loads(line)["unique_key"]))
    if excluded_keys:
        for key, rows in list(rows_by_key.items()):
            rows_by_key[key] = [
                row for row in rows if str(row["unique_key"]) not in excluded_keys
            ]
    if args.scene_limit is not None:
        if args.scene_limit < 1:
            raise RuntimeError("--scene-limit must be positive")
        for key, rows in list(rows_by_key.items()):
            keep = {
                scene for _index, scene in sorted(
                    {(int(row["scene_index"]), str(row["scene_id"])) for row in rows}
                )[: args.scene_limit]
            }
            rows_by_key[key] = [row for row in rows if str(row["scene_id"]) in keep]
    if set(rows_by_key) != set(templates):
        raise RuntimeError(f"template/queue key mismatch: templates={sorted(templates)} rows={sorted(rows_by_key)}")

    runner = (Path(__file__).resolve().parent / "windowfix_v2/run_atomic_gap_shard_v1.py").resolve()
    sync_loop = (Path(__file__).resolve().parent / "windowfix_v2/run_atomic_gap_sync_loop_v1.sh").resolve()
    claim_script = (Path(__file__).resolve().parent / "claim_probe528_atomic_tasks.py").resolve()
    workers = []
    all_keys: set[str] = set()
    for key in sorted(rows_by_key):
        rows = rows_by_key[key]
        n_workers = min(DEFAULT_WORKERS[key], len({row["scene_id"] for row in rows}))
        scenes = sorted({(int(row["scene_index"]), str(row["scene_id"])) for row in rows})
        slot_for_scene = {scene: index % n_workers for index, (_scene_index, scene) in enumerate(scenes)}
        slots: list[list[dict[str, Any]]] = [[] for _ in range(n_workers)]
        for row in rows:
            if row["unique_key"] in all_keys:
                raise RuntimeError(f"duplicate queue owner: {row['unique_key']}")
            all_keys.add(row["unique_key"])
            slots[slot_for_scene[str(row["scene_id"])]].append(row)
        for index, assigned in enumerate(slots):
            owner = f"p528-{key[0]}-{'stream' if key[1] == 'streaming_prefix' else 'target'}-{index:02d}"
            spec = worker_spec(
                rows=assigned, template=templates[key], output=output, owner=owner,
                queue_sha=args.queue_sha256, runner=runner, sync_loop=sync_loop,
                claim_script=claim_script, claim_root=claim_root,
            )
            workers.append(spec)
            atomic_json(output / "jobspecs" / f"{owner}.json", spec)
    expected = sum(len(value) for value in rows_by_key.values())
    if len(all_keys) != expected:
        raise RuntimeError("atomic owner reconciliation failed")
    commands = "\n".join(spec["sco_command"] for spec in workers) + "\n"
    atomic_text(output / "sco_create_commands.txt", commands)
    manifest = {
        "schema": "PROBE528_ATOMIC_SINGLE_GPU_WORKER_PLAN_V1",
        "status": "READY_NOT_SUBMITTED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "queue": str(queue), "queue_sha256": args.queue_sha256,
        "scene_limit": args.scene_limit,
        "atomic_tasks": expected, "unique_owned_tasks": len(all_keys),
        "excluded_completed_tasks": len(excluded_keys),
        "workers": workers,
        "scheduler": {"pool": "zoetrope", "worker_spec": "N4lS.Iq.I80.1", "priority": "HIGHEST"},
        "ownership": "one immutable worker and one O_EXCL claim per atomic queue key",
    }
    manifest["content_sha256"] = canonical_sha(manifest)
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"workers": len(workers), "atomic_tasks": expected, "keys": sorted(rows_by_key)}, sort_keys=True))


if __name__ == "__main__":
    main()
