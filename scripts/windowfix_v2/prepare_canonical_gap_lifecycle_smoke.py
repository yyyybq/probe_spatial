#!/usr/bin/env python3
"""Render a tiny, immutable production lifecycle smoke for canonical gaps.

The smoke is deliberately *not* a synthetic extractor test.  It selects one
already-positive item from each reviewed gap worklist and invokes the normal
``run_canonical_gap_shard.py --execute`` lifecycle when its rendered SCO
command is later submitted:

* extraction writes the canonical-v2 gap payload;
* the inherited AOSS sync loop stages, uploads, and verifies it;
* only a verified remote completion receipt permits local cleanup.

The renderer itself never invokes SCO and refuses both a broad source shard
and a source worklist that has lost any of its immutable bindings.  It writes
two one-scene/one-layer worklists: one streaming tensor unit (plus its
required structural ``prefix_index.npy``) and one isolated-target tensor unit
(plus its required delta index).  Both payload roots remain the normal
canonical-v2 gap namespaces.  Their *receipts, logs, state paths, worklists,
and plan artifacts* are distinct because each derived shard has its own
content SHA.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.canonical_gap_worklist import (  # noqa: E402
    CONTRACT,
    CanonicalGapWorklist,
    canonical_json_sha,
    sha256_file,
)
from scripts.windowfix_v2.run_canonical_gap_shard import (  # noqa: E402
    run_shard,
    validate_batch,
    validate_shard,
)


SMOKE_CONTRACT = "CANONICAL_V2_GAP_LIFECYCLE_SMOKE_V1"
DERIVATION_AUDIT_TYPE = "canonical_v2_gap_lifecycle_smoke_derivation"
DEFAULT_STREAMING_SHARD = (
    PROJECT_ROOT
    / "artifacts/windowfix_v2/sco_plans/"
    "canonical_v2_gap_modelunit_base_r2_20260831/"
    "shards/wf2-gap-019-wan-749-streaming_prefix.json"
)
DEFAULT_TARGET_SHARD = (
    PROJECT_ROOT
    / "artifacts/windowfix_v2/sco_plans/"
    "canonical_v2_gap_modelunit_base_r2_20260831/"
    "shards/wf2-gap-020-wan-749-target_isolated.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts/windowfix_v2/sco_plans/"
    "canonical_v2_gap_lifecycle_smoke_r3_20260831"
)


def fail(message: str) -> None:
    raise ValueError(message)


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        fail(f"{label} must be a regular non-symlink file: {path}")
    resolved = path.resolve()
    if resolved.is_symlink() or not resolved.is_file():
        fail(f"resolved {label} must be a regular file: {resolved}")
    return resolved


def _json(path: Path, label: str) -> dict[str, Any]:
    path = _regular(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"cannot parse {label}: {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must contain one JSON object: {path}")
    return value


def _content_valid(value: dict[str, Any], label: str) -> str:
    stored = value.get("content_sha256")
    unhashed = dict(value)
    unhashed.pop("content_sha256", None)
    actual = canonical_json_sha(unhashed)
    if stored != actual:
        fail(f"{label} content SHA mismatch")
    return actual


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o644)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _binding(path: Path) -> dict[str, str]:
    path = _regular(path, "bound artifact")
    return {"path": str(path), "sha256": sha256_file(path)}


def _render(argv: Iterable[str]) -> str:
    return shlex.join([str(item) for item in argv])


def _parent_plan_binding(source_shard_path: Path) -> dict[str, Any]:
    """Validate a source shard against its immutable parent plan lock."""

    source_shard_path = _regular(source_shard_path, "source shard")
    plan_dir = source_shard_path.parent.parent
    if plan_dir.is_symlink() or not plan_dir.is_dir():
        fail(f"source plan directory is invalid: {plan_dir}")
    lock_path = _regular(plan_dir / "PLAN.lock.json", "source plan lock")
    summary_path = _regular(plan_dir / "summary.json", "source plan summary")
    lock = _json(lock_path, "source plan lock")
    summary = _json(summary_path, "source plan summary")
    if lock.get("summary_sha256") != sha256_file(summary_path):
        fail("source plan lock does not bind its summary")
    shards = lock.get("shards")
    if not isinstance(shards, dict):
        fail("source plan lock has no shard map")
    if shards.get(source_shard_path.name) != sha256_file(source_shard_path):
        fail("source plan lock does not bind selected source shard")
    source_raw = _json(source_shard_path, "source shard")
    _content_valid(source_raw, "source shard")
    if summary.get("plan_id") != source_raw.get("plan_id") or lock.get("plan_id") != source_raw.get("plan_id"):
        fail("source shard/summary/lock plan_id mismatch")
    if summary.get("submission_performed") is not False:
        fail("source shard must originate from a review-only parent plan")
    # The canonical runner is the source-of-truth verifier for protocol,
    # source audit, worklist, source bindings, and current source files.
    validate_shard(source_shard_path)
    return {
        "plan_dir": str(plan_dir.resolve()),
        "plan_id": str(source_raw["plan_id"]),
        "plan_lock": _binding(lock_path),
        "summary": _binding(summary_path),
        "source_shard": {
            **_binding(source_shard_path),
            "content_sha256": str(source_raw["content_sha256"]),
            "shard_id": str(source_raw["shard_id"]),
        },
    }


def _source_shard(
    path: Path, *, mode: str, vfm: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    parent = _parent_plan_binding(path)
    raw = _json(path, "source shard")
    if raw.get("status") != "READY" or raw.get("contract") != CONTRACT:
        fail("source shard is not a READY canonical-v2 gap shard")
    if raw.get("mode") != mode or raw.get("vfm") != vfm:
        fail(f"source shard is not the requested {vfm}/{mode} source")
    worklist_path = _regular(Path(str(raw.get("worklist_path", ""))), "source worklist")
    if sha256_file(worklist_path) != raw.get("worklist_file_sha256"):
        fail("source worklist file SHA mismatch")
    worklist = _json(worklist_path, "source worklist")
    _content_valid(worklist, "source worklist")
    if worklist.get("content_sha256") != raw.get("worklist_content_sha256"):
        fail("source worklist content SHA mismatch")
    if worklist.get("mode") != mode or worklist.get("vfm") != vfm:
        fail("source worklist identity does not match source shard")
    return raw, worklist, parent


def _selected_entry(
    worklist: dict[str, Any],
    *,
    mode: str,
    scene: str,
    layer: int,
    window_id: int | None = None,
    prefix: int | None = None,
    frame_index: int | None = None,
) -> dict[str, Any]:
    entries = worklist.get("entries")
    if not isinstance(entries, list):
        fail("source worklist entries are invalid")
    candidates: list[dict[str, Any]] = []
    for item in entries:
        if not isinstance(item, dict) or str(item.get("scene")) != scene:
            continue
        if mode == "streaming_prefix":
            if int(item.get("window_id", -1)) != window_id or int(item.get("prefix", -1)) != prefix:
                continue
        elif int(item.get("frame_index", -1)) != frame_index:
            continue
        layers = item.get("layers")
        if isinstance(layers, list) and int(layer) in {int(value) for value in layers}:
            candidates.append(item)
    if len(candidates) != 1:
        key = (
            f"scene={scene} window={window_id} prefix={prefix}"
            if mode == "streaming_prefix"
            else f"scene={scene} frame_index={frame_index}"
        )
        fail(f"expected exactly one positive source worklist entry for {key}/layer={layer}; found {len(candidates)}")
    result = copy.deepcopy(candidates[0])
    result["layers"] = [int(layer)]
    return result


def _derived_worklist(
    source_worklist: dict[str, Any],
    *,
    source_worklist_path: Path,
    source_shard_binding: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    entry = selection["entry"]
    value = {
        key: copy.deepcopy(item)
        for key, item in source_worklist.items()
        if key not in {"entries", "counts", "output_layers", "content_sha256"}
    }
    value["output_layers"] = [int(selection["layer"])]
    value["entries"] = [entry]
    value["counts"] = {"entries": 1, "layer_items": 1, "scenes": 1}
    value["lifecycle_smoke_derivation"] = {
        "contract": SMOKE_CONTRACT,
        "source_worklist": {
            **_binding(source_worklist_path),
            "content_sha256": str(source_worklist["content_sha256"]),
        },
        "source_shard": source_shard_binding,
        "selection": {
            key: value for key, value in selection.items() if key != "entry"
        },
        "scope": "one positive allow-listed layer item only",
    }
    value["content_sha256"] = canonical_json_sha(value)
    return value


def _derived_audit(
    *,
    source_shard: dict[str, Any],
    parent: dict[str, Any],
    source_worklist_path: Path,
    source_worklist: dict[str, Any],
    derived_worklist_path: Path,
    derived_worklist_file_sha256: str,
    derived_worklist: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    source_audit_path = _regular(Path(str(source_shard["source_audit_path"])), "source audit")
    source_audit = _json(source_audit_path, "source audit")
    _content_valid(source_audit, "source audit")
    if source_audit.get("content_sha256") != source_shard.get("source_audit_content_sha256"):
        fail("source audit content SHA mismatches source shard")
    if sha256_file(source_audit_path) != source_shard.get("source_audit_sha256"):
        fail("source audit file SHA mismatches source shard")
    source_binding = source_audit.get("canonical_source")
    if not isinstance(source_binding, dict) or source_binding.get("sha256") != source_shard.get("canonical_source_sha256"):
        fail("source audit canonical source binding mismatch")
    value = {
        "schema_version": 1,
        "audit_type": DERIVATION_AUDIT_TYPE,
        "status": "PASS",
        "submission_performed": False,
        "protocol_file_sha256": source_shard["protocol_file_sha256"],
        "canonical_source": {
            "path": str(_regular(Path(str(source_shard["canonical_source_path"])), "canonical source")),
            "sha256": source_shard["canonical_source_sha256"],
        },
        "counts": {"hard_errors": 0, "gap_worklists": 1, "layer_items": 1, "scenes": 1},
        "hard_errors": [],
        "outputs": {
            "gap_worklists": [
                {
                    "path": str(derived_worklist_path.resolve()),
                    "file_sha256": derived_worklist_file_sha256,
                    "content_sha256": derived_worklist["content_sha256"],
                    "counts": derived_worklist["counts"],
                }
            ]
        },
        "lifecycle_smoke_derivation": {
            "contract": SMOKE_CONTRACT,
            "parent_plan": parent,
            "source_audit": {
                **_binding(source_audit_path),
                "content_sha256": source_audit["content_sha256"],
            },
            "source_worklist": {
                **_binding(source_worklist_path),
                "content_sha256": source_worklist["content_sha256"],
            },
            "selection": {
                key: value for key, value in selection.items() if key != "entry"
            },
            "prohibition": "no target materialization; delta payloads only",
        },
    }
    value["content_sha256"] = canonical_json_sha(value)
    return value


def _dataset_scene_range(
    source_shard: dict[str, Any], scene: str, requested_index: int | None
) -> tuple[int, int]:
    """Fail closed if a one-scene range is not the requested physical scene.

    The production extractor orders ScanNet++ directory names lexically.  The
    smoke intentionally uses its first canonical scene (``00777c41d4``), so
    this inexpensive check is sufficient without loading every scene's large
    metadata array during plan rendering.
    """

    data_root = Path(str(source_shard["dataset_data_root"]))
    root = data_root / "processed_scannetpp_v2"
    nested = root / "processed_scannetpp_v2"
    if nested.is_dir():
        root = nested
    if root.is_symlink() or not root.is_dir():
        fail(f"dataset ScanNet++ root is unavailable: {root}")
    candidates = []
    for name in sorted(path.name for path in root.iterdir() if path.is_dir() and not path.is_symlink()):
        candidate = root / name
        if (
            (candidate / "images").is_dir()
            and (candidate / "refined_ins_ids").is_dir()
            and (candidate / "scene_iphone_metadata.npz").is_file()
        ):
            candidates.append(name)
    if scene not in candidates:
        fail(f"selected scene is not a canonical ScanNet++ extraction candidate: {scene}")
    actual_index = candidates.index(scene)
    if requested_index is not None and actual_index != requested_index:
        fail(
            "requested scene index does not identify selected scene under the production collector: "
            f"index={requested_index} actual={actual_index} scene={scene}"
        )
    return actual_index, actual_index + 1


def _derived_shard(
    source_shard: dict[str, Any],
    *,
    plan_id: str,
    shard_id: str,
    derived_worklist_path: Path,
    derived_worklist_file_sha256: str,
    derived_worklist: dict[str, Any],
    derived_audit_path: Path,
    derived_audit_file_sha256: str,
    derived_audit: dict[str, Any],
    parent: dict[str, Any],
    selection: dict[str, Any],
    scene_start: int,
    scene_end: int,
    renderer_path: Path,
) -> dict[str, Any]:
    value = {
        key: copy.deepcopy(item)
        for key, item in source_shard.items()
        if key
        not in {
            "content_sha256",
            "plan_id",
            "shard_id",
            "source_audit_path",
            "source_audit_sha256",
            "source_audit_content_sha256",
            "worklist_path",
            "worklist_file_sha256",
            "worklist_content_sha256",
            "layers",
            "counts",
            "scene_start",
            "scene_end",
            "n_gpus",
        }
    }
    value.update(
        {
            "plan_id": plan_id,
            "shard_id": shard_id,
            "source_audit_path": str(derived_audit_path.resolve()),
            "source_audit_sha256": derived_audit_file_sha256,
            "source_audit_content_sha256": derived_audit["content_sha256"],
            "worklist_path": str(derived_worklist_path.resolve()),
            "worklist_file_sha256": derived_worklist_file_sha256,
            "worklist_content_sha256": derived_worklist["content_sha256"],
            "layers": [int(selection["layer"])],
            "counts": {"entries": 1, "layer_items": 1, "scenes": 1},
            "scene_start": scene_start,
            "scene_end": scene_end,
            "n_gpus": 1,
            "lifecycle_smoke": {
                "contract": SMOKE_CONTRACT,
                "scope": "one physical canonical scene, one positive allow-listed layer item",
                "selection": {
                    key: item for key, item in selection.items() if key != "entry"
                },
                "parent_plan": parent,
                "renderer": _binding(renderer_path),
                "payload_namespace_rule": (
                    "payloads remain in the normal canonical-v2 gap namespace; "
                    "receipt/log/state paths are shard-content-SHA isolated"
                ),
                "target_materialization": "FORBIDDEN",
            },
        }
    )
    value["content_sha256"] = canonical_json_sha(value)
    return value


def _batch(
    *,
    plan_id: str,
    batch_id: str,
    shard_path: Path,
    shard_file_sha256: str,
    shard: dict[str, Any],
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "status": "READY",
        "plan_id": plan_id,
        "batch_id": batch_id,
        "vfm": shard["vfm"],
        "serialization_scope": f"lifecycle smoke only for {shard['vfm']}/{shard['mode']}",
        "serialization_reason": "one exact canonical unit; preserve the normal gap runner lifecycle",
        "shards": [
            {
                "path": str(shard_path.resolve()),
                "sha256": shard_file_sha256,
                "shard_id": shard["shard_id"],
                "mode": shard["mode"],
                "timestep": shard["timestep"],
            }
        ],
    }
    value["content_sha256"] = canonical_json_sha(value)
    return value


def _jobspec(
    *,
    plan_id: str,
    batch_path: Path,
    batch_file_sha256: str,
    batch: dict[str, Any],
    shard: dict[str, Any],
    workspace: str,
    aec2_name: str,
    worker_spec: str,
    priority: str,
    container_image: str,
    storage_mount: str,
) -> dict[str, Any]:
    worker = [
        str(shard["python"]),
        str(PROJECT_ROOT / "scripts/windowfix_v2/run_canonical_gap_shard.py"),
        "--batch",
        str(batch_path.resolve()),
        "--execute",
    ]
    worker_command = f"cd {shlex.quote(str(PROJECT_ROOT))} && {_render(worker)}"
    job_name = f"bq-wf2-lifecycle-smoke-{shard['mode'].replace('_', '-')}-{shard['vfm']}"
    sco_argv = [
        "sco",
        "acp",
        "jobs",
        "create",
        f"--workspace-name={workspace}",
        f"--aec2-name={aec2_name}",
        f"--job-name={job_name}",
        f"--priority={priority}",
        f"--container-image-url={container_image}",
        f"--storage-mount={storage_mount}",
        "--training-framework=pytorch",
        "--worker-nodes=1",
        f"--worker-spec={worker_spec}",
        f"--command={worker_command}",
    ]
    return {
        "schema_version": 1,
        "plan_id": plan_id,
        "batch_id": batch["batch_id"],
        "batch_path": str(batch_path.resolve()),
        "batch_sha256": batch_file_sha256,
        "vfm": shard["vfm"],
        "mode": shard["mode"],
        "shard_count": 1,
        "layer_items": 1,
        "job_name": job_name,
        "resource": {
            "worker_nodes": 1,
            "worker_spec": worker_spec,
            "requested_runner_gpus": 1,
            "aec2_name": aec2_name,
            "priority": priority,
            "container_image": container_image,
        },
        "sco_argv": sco_argv,
        "sco_command": _render(sco_argv),
        "submission_policy": "REVIEW_ONLY_LIFECYCLE_SMOKE_NEVER_SUBMITTED_BY_RENDERER",
        "lifecycle_contract": "AOSS upload and remote verification precede local payload deletion",
        "target_materialization": "FORBIDDEN",
    }


def _write_readme(
    path: Path, *, plan_id: str, selected_scene: str, jobs: list[dict[str, Any]]
) -> None:
    lines = [
        "# Canonical-v2 gap AOSS lifecycle smoke",
        "",
        f"Immutable plan ID: `{plan_id}`.",
        "",
        "This plan is intentionally a real, bounded production smoke: it selects one", 
        f"positive allow-listed layer item from the canonical scene `{selected_scene}` for", 
        "each mode.  It does not run probing and it does not materialize target mixed cache.",
        "",
        "Payloads use the normal canonical-v2 gap namespaces so that the exact production", 
        "uploader, remote prefix, sidecar validation, receipt, and cleanup logic are exercised.",
        "The smoke's worklists, derived audits, shard-content-SHA state roots, logs, and", 
        "receipts are distinct.  A target smoke writes only an isolated `*.delta.sft` plus", 
        "`target_indices_layer*.npy`; full target materialization remains forbidden.",
        "",
        "The renderer never called SCO.  Before a real submit, run the two commands in", 
        "`dry_run_commands.txt`; they invoke the exact runner without `--execute` and", 
        "therefore only validate bindings and print the extractor commands.",
        "",
        "## Review-only SCO commands",
        "",
    ]
    for job in jobs:
        lines.extend([f"- `{job['job_name']}`: `{job['sco_command']}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o644)


def render_plan(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        fail(f"refusing to replace immutable output directory: {output_dir}")
    if args.scene_index is not None and args.scene_index < 0:
        fail("scene-index must be non-negative")
    if args.streaming_layer < 0 or args.target_layer < 0:
        fail("smoke layers must be non-negative")

    streaming_shard, streaming_worklist, streaming_parent = _source_shard(
        args.streaming_shard, mode="streaming_prefix", vfm="wan"
    )
    target_shard, target_worklist, target_parent = _source_shard(
        args.target_shard, mode="target_isolated", vfm="wan"
    )
    if streaming_shard["canonical_source_sha256"] != target_shard["canonical_source_sha256"]:
        fail("streaming and target source shards bind different canonical sources")
    if streaming_shard["protocol_file_sha256"] != target_shard["protocol_file_sha256"]:
        fail("streaming and target source shards bind different protocol files")
    if streaming_shard["feature_namespace"] != "FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1":
        fail("streaming smoke source is not in the canonical streaming gap namespace")
    if target_shard["feature_namespace"] != "FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1":
        fail("target smoke source is not in the canonical target gap namespace")

    scene_start, scene_end = _dataset_scene_range(
        streaming_shard, args.scene, args.scene_index
    )
    target_start, target_end = _dataset_scene_range(target_shard, args.scene, args.scene_index)
    if (scene_start, scene_end) != (target_start, target_end):
        fail("streaming and target source shards resolve different physical scene ranges")

    stream_entry = _selected_entry(
        streaming_worklist,
        mode="streaming_prefix",
        scene=args.scene,
        window_id=args.streaming_window_id,
        prefix=args.streaming_prefix,
        layer=args.streaming_layer,
    )
    target_entry = _selected_entry(
        target_worklist,
        mode="target_isolated",
        scene=args.scene,
        frame_index=args.target_frame_index,
        layer=args.target_layer,
    )
    selections = [
        {
            "mode": "streaming_prefix",
            "scene": args.scene,
            "scene_start": scene_start,
            "scene_end": scene_end,
            "window_id": args.streaming_window_id,
            "prefix": args.streaming_prefix,
            "layer": args.streaming_layer,
            "entry": stream_entry,
        },
        {
            "mode": "target_isolated",
            "scene": args.scene,
            "scene_start": scene_start,
            "scene_end": scene_end,
            "frame_index": args.target_frame_index,
            "layer": args.target_layer,
            "entry": target_entry,
        },
    ]
    renderer_path = _regular(Path(__file__), "lifecycle smoke renderer")
    plan_seed = {
        "contract": SMOKE_CONTRACT,
        "streaming_parent": streaming_parent,
        "target_parent": target_parent,
        "selections": [
            {key: value for key, value in selection.items() if key != "entry"}
            for selection in selections
        ],
        "renderer": _binding(renderer_path),
    }
    plan_id = f"windowfix-v2-gap-lifecycle-smoke-{canonical_json_sha(plan_seed)[:16]}"
    staging = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if staging.exists():
        fail(f"refusing to reuse staging directory: {staging}")
    for child in ("worklists", "source_audits", "shards", "batches", "jobspecs"):
        (staging / child).mkdir(parents=True, exist_ok=False)

    source_sets = [
        (streaming_shard, streaming_worklist, streaming_parent),
        (target_shard, target_worklist, target_parent),
    ]
    jobs: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    try:
        for (source, source_worklist, parent), selection in zip(source_sets, selections):
            mode = str(selection["mode"])
            source_worklist_path = _regular(Path(str(source["worklist_path"])), "source worklist")
            mode_token = "streaming" if mode == "streaming_prefix" else "target"
            worklist_path = staging / "worklists" / f"wan-{mode_token}-one-unit.json"
            final_worklist_path = output_dir / "worklists" / worklist_path.name
            derived_worklist = _derived_worklist(
                source_worklist,
                source_worklist_path=source_worklist_path,
                source_shard_binding=parent["source_shard"],
                selection=selection,
            )
            _atomic_json(worklist_path, derived_worklist)
            worklist_file_sha256 = sha256_file(worklist_path)
            # Validate the newly written worklist using exactly the production
            # parser before anything can be rendered from it.
            CanonicalGapWorklist.load(
                worklist_path,
                expected_sha256=sha256_file(worklist_path),
                mode=mode,
                vfm="wan",
                model_id=str(source["model_id"]),
                timestep=int(source["timestep"]),
                feature_namespace=str(source["feature_namespace"]),
                output_layers=[int(selection["layer"])],
                protocol_sha256=str(source["protocol_sha256"]),
                actual_prompt=(
                    str(source["prompt_contract"]["value"])
                    if source["prompt_contract"]["mode"] == "exact"
                    else str(source.get("runtime_prompt", ""))
                ),
                preprocess_identity=copy.deepcopy(source["preprocess_identity"]),
            )
            audit_path = staging / "source_audits" / f"wan-{mode_token}-one-unit.json"
            final_audit_path = output_dir / "source_audits" / audit_path.name
            derived_audit = _derived_audit(
                source_shard=source,
                parent=parent,
                source_worklist_path=source_worklist_path,
                source_worklist=source_worklist,
                derived_worklist_path=final_worklist_path,
                derived_worklist_file_sha256=worklist_file_sha256,
                derived_worklist=derived_worklist,
                selection=selection,
            )
            _atomic_json(audit_path, derived_audit)
            audit_file_sha256 = sha256_file(audit_path)
            shard_id = f"wf2-lifecycle-smoke-wan-{mode_token}"
            shard_path = staging / "shards" / f"{shard_id}.json"
            final_shard_path = output_dir / "shards" / shard_path.name
            derived_shard = _derived_shard(
                source,
                plan_id=plan_id,
                shard_id=shard_id,
                derived_worklist_path=final_worklist_path,
                derived_worklist_file_sha256=worklist_file_sha256,
                derived_worklist=derived_worklist,
                derived_audit_path=final_audit_path,
                derived_audit_file_sha256=audit_file_sha256,
                derived_audit=derived_audit,
                parent=parent,
                selection=selection,
                scene_start=scene_start,
                scene_end=scene_end,
                renderer_path=renderer_path,
            )
            _atomic_json(shard_path, derived_shard)
            shard_file_sha256 = sha256_file(shard_path)
            batch_id = f"wf2-lifecycle-smoke-{mode_token}-wan"
            batch_path = staging / "batches" / f"{batch_id}.json"
            final_batch_path = output_dir / "batches" / batch_path.name
            batch = _batch(
                plan_id=plan_id,
                batch_id=batch_id,
                shard_path=final_shard_path,
                shard_file_sha256=shard_file_sha256,
                shard=derived_shard,
            )
            _atomic_json(batch_path, batch)
            batch_file_sha256 = sha256_file(batch_path)
            jobspec = _jobspec(
                plan_id=plan_id,
                batch_path=final_batch_path,
                batch_file_sha256=batch_file_sha256,
                batch=batch,
                shard=derived_shard,
                workspace=args.workspace,
                aec2_name=args.aec2_name,
                worker_spec=args.worker_spec,
                priority=args.priority,
                container_image=args.container_image,
                storage_mount=args.storage_mount,
            )
            jobspec_path = staging / "jobspecs" / f"{batch_id}.json"
            _atomic_json(jobspec_path, jobspec)
            jobs.append(jobspec)
            artifacts.append(
                {
                    "mode": mode,
                    "selection": {
                        key: value for key, value in selection.items() if key != "entry"
                    },
                    "worklist": {
                        "path": str(final_worklist_path.resolve()),
                        "sha256": worklist_file_sha256,
                        "content_sha256": derived_worklist["content_sha256"],
                    },
                    "source_audit": {
                        "path": str(final_audit_path.resolve()),
                        "sha256": audit_file_sha256,
                        "content_sha256": derived_audit["content_sha256"],
                    },
                    "shard": {
                        "path": str(final_shard_path.resolve()),
                        "sha256": shard_file_sha256,
                        "content_sha256": derived_shard["content_sha256"],
                    },
                    "batch": {
                        "path": str(final_batch_path.resolve()),
                        "sha256": batch_file_sha256,
                        "content_sha256": batch["content_sha256"],
                    },
                    "jobspec": {
                        "path": str((output_dir / "jobspecs" / jobspec_path.name).resolve()),
                        "sha256": sha256_file(jobspec_path),
                    },
                    "dry_run": _render(
                        [
                            str(derived_shard["python"]),
                            str(PROJECT_ROOT / "scripts/windowfix_v2/run_canonical_gap_shard.py"),
                            "--batch",
                            str(final_batch_path.resolve()),
                        ]
                    ),
                }
            )
        commands = [
            "# REVIEW ARTIFACT ONLY. The lifecycle smoke renderer never invokes SCO.",
            *[job["sco_command"] for job in jobs],
        ]
        (staging / "sco_create_commands.txt").write_text(
            "\n".join(commands) + "\n", encoding="utf-8"
        )
        dry_runs = [
            "# Read-only binding/lifecycle preview. Do not add --execute.",
            *[
                f"cd {shlex.quote(str(PROJECT_ROOT))} && {item['dry_run']}"
                for item in artifacts
            ],
        ]
        (staging / "dry_run_commands.txt").write_text(
            "\n".join(dry_runs) + "\n", encoding="utf-8"
        )
        summary = {
            "schema_version": 1,
            "contract": SMOKE_CONTRACT,
            "plan_id": plan_id,
            "status": "READY_FOR_REVIEW_AND_REAL_SCO_LIFECYCLE_SMOKE",
            "submission_performed": False,
            "renderer": _binding(renderer_path),
            "parent_plans": {"streaming": streaming_parent, "target": target_parent},
            "canonical_source_sha256": streaming_shard["canonical_source_sha256"],
            "protocol_file_sha256": streaming_shard["protocol_file_sha256"],
            "scene": args.scene,
            "scene_range": {"start": scene_start, "end": scene_end},
            "counts": {
                "smoke_jobs": 2,
                "streaming_layer_items": 1,
                "target_row_layer_items": 1,
                "physical_scenes": 1,
                "target_materialization_rows": 0,
            },
            "artifacts": artifacts,
            "lifecycle": {
                "upload": "existing run_tstep_sync_loop.sh transaction engine",
                "remote_receipt": "CANONICAL_V2_GAP_AOSS_COMPLETION_RECEIPT_V1",
                "cleanup": "only after exact ads-cli remote verification and receipt commit",
                "resume": "canonical unit identity plus verified remote completion receipt",
            },
            "prohibitions": [
                "No SCO API/CLI was invoked by this renderer.",
                "No probing training/evaluation is included.",
                "No target mixed-cache materialization is included.",
                "No legacy feature root is mutable or writable by this plan.",
            ],
        }
        _atomic_json(staging / "summary.json", summary)
        _write_readme(
            staging / "README.md", plan_id=plan_id, selected_scene=args.scene, jobs=jobs
        )
        lock = {
            "schema_version": 1,
            "contract": SMOKE_CONTRACT,
            "plan_id": plan_id,
            "summary_sha256": sha256_file(staging / "summary.json"),
            "commands_sha256": sha256_file(staging / "sco_create_commands.txt"),
            "dry_run_commands_sha256": sha256_file(staging / "dry_run_commands.txt"),
            "worklists": {
                path.name: sha256_file(path)
                for path in sorted((staging / "worklists").glob("*.json"))
            },
            "source_audits": {
                path.name: sha256_file(path)
                for path in sorted((staging / "source_audits").glob("*.json"))
            },
            "shards": {
                path.name: sha256_file(path)
                for path in sorted((staging / "shards").glob("*.json"))
            },
            "batches": {
                path.name: sha256_file(path)
                for path in sorted((staging / "batches").glob("*.json"))
            },
            "jobspecs": {
                path.name: sha256_file(path)
                for path in sorted((staging / "jobspecs").glob("*.json"))
            },
        }
        _atomic_json(staging / "PLAN.lock.json", lock)
        os.replace(staging, output_dir)
        # All internal references deliberately name the final immutable
        # directory, so validate again only after its atomic publication.
        for item in artifacts:
            validate_shard(Path(item["shard"]["path"]))
            validate_batch(Path(item["batch"]["path"]))
        return summary
    except Exception:
        # Rendering failed before the immutable handoff.  Leave a diagnostic
        # staging directory intact rather than silently deleting evidence.
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streaming-shard", type=Path, default=DEFAULT_STREAMING_SHARD)
    parser.add_argument("--target-shard", type=Path, default=DEFAULT_TARGET_SHARD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scene", default="00777c41d4")
    parser.add_argument(
        "--scene-index", type=int, default=None,
        help="Optional lexical ScanNet++ index assertion for --scene.",
    )
    parser.add_argument("--streaming-window-id", type=int, default=0)
    parser.add_argument("--streaming-prefix", type=int, default=8)
    parser.add_argument("--streaming-layer", type=int, default=20)
    parser.add_argument("--target-frame-index", type=int, default=48)
    parser.add_argument("--target-layer", type=int, default=20)
    parser.add_argument("--workspace", default="aigc")
    parser.add_argument("--aec2-name", default="h800")
    parser.add_argument("--worker-spec", default="N4lS.Iq.I80.1")
    parser.add_argument("--priority", default="HIGHEST")
    parser.add_argument(
        "--container-image",
        default="registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617",
    )
    parser.add_argument(
        "--storage-mount",
        default="019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm",
    )
    args = parser.parse_args(argv)
    try:
        summary = render_plan(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
