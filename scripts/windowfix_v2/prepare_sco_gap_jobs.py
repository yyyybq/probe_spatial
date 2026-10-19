#!/usr/bin/env python3
"""Render, but never submit, SCO jobs for audited canonical-v2 gaps only.

Every work item must be listed by a SHA-bound route/row audit as a
``CANONICAL_V2_GAP_WORKLIST_V1`` positive allow-list.  The renderer refuses a
bare inventory and therefore cannot accidentally fall back to full extraction.
Shards for one VFM are placed in one serial SCO batch by default to protect
structural indexes and delta receipts from writer races.  ``--one-job-per-shard``
emits independent streaming and target jobs when those namespaces are disjoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.canonical_gap_worklist import (
    CANONICAL_PROTOCOL,
    CONTRACT,
    CanonicalGapWorklist,
    canonical_json_sha,
    sha256_file,
    streaming_worklist_preprocess_identity,
)
from scripts.windowfix_v2.run_backend_determinism_canary import (
    backend_provenance_binding as canary_backend_provenance_binding,
    canary_runtime_environment,
    target_contract_from_source_audit as canary_target_contract_from_source_audit,
    validate_qwen_streaming_tensor_contract,
    validate_target_isolated_tensor_contract,
)


CERTIFIED_VFMS = {"wan", "cogvideox", "minimax_h3", "dino", "vjepa2", "videomae"}
GATED_VFMS = {"aether", "f3r", "lingbot", "qwen2_5_vl"}
MLLM_VFMS = {"qwen2_5_vl", "bagel", "bagel_hf", "generic_hf"}
GAP_NAMESPACES = {
    "streaming_prefix": "FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1",
    "target_isolated": "FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1",
}
MINIMAX_H3_MIRROR_RECEIPT_CONTRACT = "MINIMAX_H3_AOSS_MODEL_MIRROR_RECEIPT_V1"
BACKEND_DEPENDENCY_FILES = {
    "wan": ["features/wan/wan_feature.py", "features/wan/extract_features.py"],
    "cogvideox": [
        "features/cogvideox/cogvideox_feature.py",
        "features/cogvideox/cogvideox_feature_i2v.py",
        "features/cogvideox/extract_features.py",
    ],
    "dino": ["features/dino/extract_features.py"],
    "vjepa2": ["features/vjepa2/vjepa2_feature.py"],
    "videomae": [],
    "aether": ["features/aether/aether_feature.py", "features/aether/extract_features.py"],
    "f3r": [
        "features/f3r/extract_features.py",
        "features/f3r/f3r/models/fast3r.py",
        "features/f3r/f3r/dust3r/inference_multiview.py",
    ],
    "minimax_h3": [
        "features/minimax_h3/minimax_h3_feature.py",
        "features/minimax_h3/extract_features.py",
    ],
    "lingbot": ["features/lingbot/lingbot_feature.py", "features/lingbot/extract_features.py"],
    "qwen2_5_vl": ["features/run_inscene15k.py"],
}


def backend_dependency_bindings(vfm: str, project_root: Path) -> dict[str, dict[str, str]]:
    if vfm not in BACKEND_DEPENDENCY_FILES:
        raise ValueError(f"no frozen backend dependency map for {vfm}")
    relative_paths = [
        "features/noise_seed.py",
        "features/offline_hf.py",
        *BACKEND_DEPENDENCY_FILES[vfm],
    ]
    result: dict[str, dict[str, str]] = {}
    for relative in relative_paths:
        path = project_root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"backend dependency must be regular: {path}")
        result[relative] = {"path": str(path), "sha256": sha256_file(path)}
    return result


def local_model_config_bindings(model_id: str) -> dict[str, dict[str, str]]:
    model_root = Path(str(model_id))
    if not model_root.is_absolute():
        return {}
    candidates = (
        "config.json",
        "model_index.json",
        "modular_model_index.json",
        "processor/config.json",
        "refiner/config.json",
        "text_encoder/config.json",
        "transformer/config.json",
        "vae/config.json",
    )
    result: dict[str, dict[str, str]] = {}
    for relative in candidates:
        raw_path = model_root / relative
        if raw_path.is_symlink():
            raise ValueError(f"local model config may not be a symlink: {raw_path}")
        if not raw_path.is_file():
            continue
        path = raw_path.resolve()
        result[relative] = {"path": str(path), "sha256": sha256_file(path)}
    if not result:
        raise ValueError(f"absolute local model has no bindable config files: {model_root}")
    return result


def minimax_h3_model_mirror_binding(
    receipt_path: Path | None, project_root: Path
) -> dict[str, Any] | None:
    if receipt_path is None:
        return None
    raw = receipt_path
    if raw.is_symlink() or not raw.is_file():
        raise ValueError(f"MiniMax-H3 model mirror receipt must be regular: {raw}")
    path = raw.resolve()
    receipt = read_json(path, "MiniMax-H3 model mirror receipt")
    unhashed = dict(receipt)
    stored_content = unhashed.pop("content_sha256", None)
    actual_content = canonical_json_sha(unhashed)
    if (
        receipt.get("contract") != MINIMAX_H3_MIRROR_RECEIPT_CONTRACT
        or receipt.get("status") != "PASS_REMOTE_SIZE_VERIFIED"
        or stored_content != actual_content
    ):
        raise ValueError("MiniMax-H3 model mirror receipt is not a verified immutable receipt")
    manifest_path = Path(str(receipt.get("manifest_path", "")))
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"MiniMax-H3 mirror manifest must be regular: {manifest_path}")
    manifest_path = manifest_path.resolve()
    if sha256_file(manifest_path) != receipt.get("manifest_file_sha256"):
        raise ValueError("MiniMax-H3 mirror manifest SHA mismatch")
    hydrate = project_root / "scripts/windowfix_v2/hydrate_minimax_h3_model_mirror.sh"
    if hydrate.is_symlink() or not hydrate.is_file():
        raise ValueError(f"missing MiniMax-H3 mirror hydrator: {hydrate}")
    mirror_sha = str(receipt.get("manifest_content_sha256"))
    remote_base = str(receipt.get("remote_base"))
    if len(mirror_sha) != 64 or not remote_base.startswith("aoss://"):
        raise ValueError("MiniMax-H3 mirror receipt has invalid content SHA or remote base")
    return {
        "receipt": {
            "path": str(path),
            "sha256": sha256_file(path),
            "content_sha256": stored_content,
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "content_sha256": mirror_sha,
        },
        "hydrate_script": {
            "path": str(hydrate.resolve()),
            "sha256": sha256_file(hydrate.resolve()),
        },
        "mirror_content_sha256": mirror_sha,
        "remote_base": remote_base,
        "contract": MINIMAX_H3_MIRROR_RECEIPT_CONTRACT,
    }


def runtime_num_frames_for_contract(preprocess: dict[str, Any]) -> int:
    """Return the CLI bound while retaining per-prefix unit identities."""
    if "num_frames" in preprocess:
        return int(preprocess["num_frames"])
    by_prefix = preprocess.get("num_frames_by_prefix")
    if not isinstance(by_prefix, dict) or not by_prefix:
        raise ValueError("preprocess identity has no frame-count contract")
    return max(int(value) for value in by_prefix.values())


def read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def slug(value: str, limit: int = 60) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._").lower() or "gap"
    if len(clean) <= limit:
        return clean
    return clean[: limit - 11].rstrip("-._") + "-" + hashlib.sha256(value.encode()).hexdigest()[:10]


def python_for(vfm: str) -> str:
    if vfm == "minimax_h3":
        return "/mnt/umm/users/yinbaiqiao/.conda/envs/minimax-h3/bin/python"
    if vfm == "lingbot":
        return "/mnt/umm/users/yinbaiqiao/.conda/envs/lingbot-video/bin/python"
    return "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"


def environment_for(vfm: str, project_root: Path) -> dict[str, str]:
    result = {
        "PROJECT_ROOT": str(project_root),
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "4",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    if vfm == "lingbot":
        result.update(
            {
                "HF_HOME": "/mnt/umm/shared_model/huggingface",
                "DIFFUSERS_ATTN_BACKEND": "_native_flash",
                "LINGBOT_MOE_PAD_BACKEND": "vectorized",
                "LINGBOT_MOE_EXPERT_BACKEND": "grouped_mm",
                "WINDOWFIX_EXTRA_PYTHONPATH": "/mnt/umm/users/yinbaiqiao/Consistency/lingbot-video",
            }
        )
    elif vfm == "minimax_h3":
        result["HF_HOME"] = "/mnt/umm/shared_model/huggingface"
    else:
        result.update(
            {
                "HF_HOME": "/mnt/umm/users/yinbaiqiao/.cache/huggingface",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
    return result


def environment_for_shard(
    vfm: str,
    project_root: Path,
    canary_binding: dict[str, Any] | None,
) -> dict[str, str]:
    result = environment_for(vfm, project_root)
    if not canary_binding:
        return result
    provenance = canary_binding.get("backend_provenance")
    if not isinstance(provenance, dict):
        return result
    required = provenance.get("required_environment")
    if not isinstance(required, dict):
        return result
    for key, value in required.items():
        result[str(key)] = str(value)
    return result


def render(argv: Sequence[str]) -> str:
    return shlex.join([str(value) for value in argv])


def validate_audits(
    paths: Sequence[Path], *, canonical_source_sha: str, protocol_file_sha: str
) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for raw_path in paths:
        if raw_path.is_symlink() or not raw_path.is_file():
            raise ValueError(f"source audit must be regular before resolve: {raw_path}")
        path = raw_path.resolve()
        audit = read_json(path, "route/row audit")
        # A route audit may be PENDING because unrelated remote candidates
        # could not be contacted.  Its gap_worklists remain a positive
        # allow-list of *definitive* missing rows; pending units are emitted in
        # a separate file and must never be inferred into extraction work.
        if audit.get("status") not in {"INCOMPLETE", "PENDING", "PASS"}:
            raise ValueError(f"source audit is not safe to schedule gaps: {path}")
        if int(audit.get("counts", {}).get("hard_errors", 0)) != 0 or audit.get("hard_errors"):
            raise ValueError(f"source audit contains hard errors: {path}")
        if audit.get("protocol_file_sha256") != protocol_file_sha:
            raise ValueError(f"source audit protocol SHA mismatch: {path}")
        source = audit.get("canonical_source", {})
        if source.get("sha256") != canonical_source_sha:
            raise ValueError(f"source audit canonical-source SHA mismatch: {path}")
        recorded_content = audit.get("content_sha256")
        unhashed = dict(audit)
        unhashed.pop("content_sha256", None)
        if recorded_content != canonical_json_sha(unhashed):
            raise ValueError(f"source audit content SHA mismatch: {path}")
        gap_values = audit.get("outputs", {}).get("gap_worklists")
        if not isinstance(gap_values, list):
            raise ValueError(f"source audit has no bound gap_worklists list: {path}")
        for item in gap_values:
            raw_worklist_path = Path(str(item.get("path", "")))
            if raw_worklist_path.is_symlink() or not raw_worklist_path.is_file():
                raise ValueError(
                    f"audit-bound worklist must be regular before resolve: {raw_worklist_path}"
                )
            worklist_path = raw_worklist_path.resolve()
            if worklist_path in seen:
                raise ValueError(f"duplicate worklist across audits: {worklist_path}")
            seen.add(worklist_path)
            if sha256_file(worklist_path) != item.get("file_sha256"):
                raise ValueError(f"audit-bound worklist file SHA mismatch: {worklist_path}")
            worklist = read_json(worklist_path, "gap worklist")
            if worklist.get("content_sha256") != item.get("content_sha256"):
                raise ValueError(f"audit-bound worklist content SHA mismatch: {worklist_path}")
            bindings.append(
                {
                    "audit_path": str(path),
                    "audit_sha256": sha256_file(path),
                    "audit_content_sha256": recorded_content,
                    "worklist_path": str(worklist_path),
                    "worklist_file_sha256": item["file_sha256"],
                    "worklist_content_sha256": item["content_sha256"],
                    "worklist": worklist,
                }
            )
    if not bindings:
        raise ValueError("audits bind zero gap worklists; refusing to render an empty/full fallback")
    return bindings


def valid_canary_bindings(
    directories: Sequence[Path] | Path | None,
    protocol_file_sha: str,
    worklist_bindings: Sequence[dict[str, Any]],
    project_root: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Return PASS receipts, keyed by ``(backend, mode)``.

    Streaming and isolated-target worklists for one backend are distinct
    immutable contracts.  They may therefore be certified by two separately
    rendered canary plans; collapsing the key to only the backend would let a
    streaming PASS incorrectly unlock a target delta (or vice versa).
    """
    if directories is None:
        return {}
    if isinstance(directories, Path):
        directories = [directories]
    if not directories:
        return {}
    main_extractor = project_root / "features/run_inscene15k.py"
    mllm_extractor = project_root / "features/run_inscene15k_mllm.py"
    canary_runner = project_root / "scripts/windowfix_v2/run_backend_determinism_canary.py"
    canary_renderer = project_root / "scripts/windowfix_v2/prepare_gap_backend_canary_jobs.py"
    gated_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for binding in worklist_bindings:
        vfm = str(binding["worklist"].get("vfm"))
        if vfm in GATED_VFMS:
            key = (vfm, str(binding["worklist"].get("mode")))
            gated_by_key[key].append(binding)
    expected_keys = set(gated_by_key)
    if not expected_keys:
        raise ValueError("backend canary directory supplied for zero gated worklists")
    ambiguous = sorted(key for key, values in gated_by_key.items() if len(values) != 1)
    if ambiguous:
        raise ValueError(f"multiple current gated worklists per backend: {ambiguous}")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_directory in directories:
        directory = raw_directory.resolve()
        if raw_directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"backend canary directory must be regular: {raw_directory}")
        summary_path = directory / "summary.json"
        lock_path = directory / "PLAN.lock.json"
        summary = read_json(summary_path, "gap backend canary plan summary")
        lock = read_json(lock_path, "gap backend canary plan lock")
        frozen_canary_renderer_sha = str(summary.get("canary_renderer_sha256", ""))
        if (
            summary.get("status") != "READY_FOR_REVIEW_NOT_SUBMITTED"
            or summary.get("submission_performed") is not False
            or summary.get("protocol_file_sha256") != protocol_file_sha
            or summary.get("main_extractor_sha256") != sha256_file(main_extractor)
            or summary.get("mllm_extractor_sha256") != sha256_file(mllm_extractor)
            or summary.get("canary_runner_sha256") != sha256_file(canary_runner)
            or sha256_file(summary_path) != lock.get("summary_sha256")
        ):
            raise ValueError("backend canary plan summary/lock identity mismatch")
        if len(frozen_canary_renderer_sha) != 64 or set(frozen_canary_renderer_sha) - set("0123456789abcdef"):
            raise ValueError("backend canary plan has an invalid frozen renderer SHA")
        specs_lock = lock.get("specs")
        if not isinstance(specs_lock, dict):
            raise ValueError("backend canary plan lock has no specs map")
        expected_spec_names = set(specs_lock)
        planned_vfms = {name[:-5] for name in expected_spec_names if name.endswith(".json")}
        modes = summary.get("worklist_modes")
        if not isinstance(modes, dict):
            raise ValueError("backend canary plan has no worklist-mode mapping")
        planned_keys = {(vfm, str(modes.get(vfm))) for vfm in planned_vfms}
        if (
            not planned_vfms
            or not planned_keys <= expected_keys
            or set(summary.get("backends", [])) != planned_vfms
            or int(summary.get("job_count", -1)) != len(planned_vfms)
            or set(modes) != planned_vfms
            or expected_spec_names != {f"{vfm}.json" for vfm in planned_vfms}
        ):
            raise ValueError("backend canary plan does not safely cover current gated worklists")
        for vfm, worklist_mode in sorted(planned_keys):
            key = (vfm, worklist_mode)
            if key in result:
                raise ValueError(f"duplicate PASS-canary binding for {key}")
            spec_path = directory / "specs" / f"{vfm}.json"
            require_spec_sha = specs_lock.get(spec_path.name)
            if not isinstance(require_spec_sha, str) or sha256_file(spec_path) != require_spec_sha:
                raise ValueError(f"backend canary spec SHA mismatch: {vfm}")
            spec = read_json(spec_path, "backend canary spec")
            stored_spec_content = spec.get("content_sha256")
            spec_unhashed = dict(spec)
            spec_unhashed.pop("content_sha256", None)
            if stored_spec_content != canonical_json_sha(spec_unhashed):
                raise ValueError(f"backend canary spec content SHA mismatch: {vfm}")
            candidates = [
                item for item in gated_by_key[key]
                if Path(item["worklist_path"]).resolve()
                == Path(str(spec.get("blocked_worklist", ""))).resolve()
            ]
            rebound_to_current_worklist = False
            if not candidates:
                # A PASS receipt can be rebound only when a successor immutable
                # catalogue retains the *identical* positive worklist bytes and
                # semantic content.  This is needed for the Qwen shape-certified
                # successor catalogue: its path changes intentionally, while its
                # canonical units must not.  Path-only matching would otherwise
                # force a redundant feature canary and obscure the actual binding.
                candidates = [
                    item for item in gated_by_key[key]
                    if item["worklist_file_sha256"]
                    == spec.get("blocked_worklist_file_sha256")
                    and item["worklist_content_sha256"]
                    == spec.get("blocked_worklist_content_sha256")
                ]
                rebound_to_current_worklist = bool(candidates)
            if len(candidates) != 1:
                raise ValueError(f"backend canary spec does not select one current worklist: {vfm}")
            binding = candidates[0]
            worklist = binding["worklist"]
            extractor = mllm_extractor if vfm in MLLM_VFMS else main_extractor
            provenance = canary_backend_provenance_binding(vfm, project_root)
            runtime_environment = canary_runtime_environment(vfm, provenance)
            worklist_mode = str(worklist["mode"])
            target_tensor_contract = None
            if worklist_mode == "target_isolated":
                audit = read_json(Path(binding["audit_path"]), "target source audit")
                try:
                    target_tensor_contract = canary_target_contract_from_source_audit(
                        audit,
                        args=argparse.Namespace(
                            vfm=vfm,
                            model_id=worklist["model_id"],
                            timestep=int(worklist["timestep"]),
                            layer=int(spec["layer"]),
                            protocol_file_sha256=protocol_file_sha,
                        ),
                        prompt_contract=worklist["prompt_contract"],
                        preprocess_identity=worklist["preprocess_identity"],
                        canonical_source_sha256=worklist["canonical_source_sha256"],
                    )
                except RuntimeError as exc:
                    raise ValueError(str(exc)) from exc
            sample_index = spec.get("runtime_contract", {}).get("sample_index")
            if not isinstance(sample_index, int) or sample_index < 0:
                raise ValueError("canary sample index mismatch")
            runtime_contract = {
                "prompt": (
                    worklist["prompt_contract"]["value"]
                    if worklist["prompt_contract"]["mode"] == "exact"
                    else ""
                ),
                "resize": worklist["preprocess_identity"]["resize"],
                "num_frames": runtime_num_frames_for_contract(
                    worklist["preprocess_identity"]
                ),
                "python": python_for(vfm),
                "sample_index": sample_index,
            }
            source_audit_sha_for_spec = (
                str(spec.get("source_audit_sha256"))
                if rebound_to_current_worklist
                else binding["audit_sha256"]
            )
            expected_spec = {
                "schema_version": 2,
                "audit_type": "canonical_v2_backend_canary_execution_spec",
                "status": "READY_FOR_REVIEW_NOT_SUBMITTED",
                "submission_performed": False,
                "vfm": vfm,
                "model_id": worklist["model_id"],
                "timestep": int(worklist["timestep"]),
                "blocked_mode": worklist_mode,
                "feature_namespace": worklist["feature_namespace"],
                "blocked_worklist_file_sha256": binding["worklist_file_sha256"],
                "blocked_worklist_content_sha256": binding["worklist_content_sha256"],
                "source_audit_sha256": source_audit_sha_for_spec,
                "canonical_source_sha256": worklist["canonical_source_sha256"],
                "protocol_file_sha256": protocol_file_sha,
                "extractor_sha256": sha256_file(extractor),
                "extractor": str(extractor),
                "canary_runner_sha256": sha256_file(canary_runner),
                "canary_runner": str(canary_runner),
                "canary_renderer_sha256": frozen_canary_renderer_sha,
                "canary_renderer": str(canary_renderer),
                "prompt_contract": worklist["prompt_contract"],
                "preprocess_identity": worklist["preprocess_identity"],
                "runtime_contract": runtime_contract,
                "runtime_environment": runtime_environment,
                "backend_provenance": provenance,
                "target_tensor_contract": target_tensor_contract,
                "receipt": str((directory / "receipts" / f"{vfm}.json").resolve()),
            }
            bad = [key for key, value in expected_spec.items() if spec.get(key) != value]
            if bad or int(spec.get("layer", 10**9)) not in [
                int(value) for value in worklist["output_layers"]
            ]:
                raise ValueError(f"backend canary spec identity mismatch ({vfm}): {bad}")
    
            receipt_path = directory / "receipts" / f"{vfm}.json"
            if not receipt_path.exists():
                continue
            receipt = read_json(receipt_path, "backend canary receipt")
            if receipt.get("sample_index") != sample_index:
                raise ValueError("canary receipt sample index mismatch")
            gap = receipt.get("gap_binding")
            comparisons = receipt.get("results")
            expected_receipt = {
                "schema_version": 2,
                "audit_type": "canonical_v2_backend_cross_process_order_canary",
                "status": "pass",
                "submission_performed": False,
                "vfm": vfm,
                "model_id": worklist["model_id"],
                "timestep": int(worklist["timestep"]),
                "layer": int(spec["layer"]),
                "protocol": CANONICAL_PROTOCOL,
                "protocol_file_sha256": protocol_file_sha,
                "extractor": str(extractor),
                "extractor_sha256": sha256_file(extractor),
                "canary_script": str(canary_runner),
                "canary_script_sha256": sha256_file(canary_runner),
                "runtime_prompt": (
                    worklist["prompt_contract"]["value"]
                    if worklist["prompt_contract"]["mode"] == "exact"
                    else ""
                ),
                "runtime_resize": worklist["preprocess_identity"]["resize"],
                "runtime_num_frames": runtime_num_frames_for_contract(
                    worklist["preprocess_identity"]
                ),
                "canary_spec_binding": {
                    "path": str(spec_path.resolve()),
                    "file_sha256": require_spec_sha,
                    "content_sha256": stored_spec_content,
                },
                "backend_provenance": provenance,
                "backend_provenance_revalidated_after_runs": True,
                "runtime_environment": runtime_environment,
            }
            receipt_bad = [
                key for key, value in expected_receipt.items() if receipt.get(key) != value
            ]
            expected_gap = {
                "blocked_worklist": (
                    str(Path(str(spec["blocked_worklist"])).resolve())
                    if rebound_to_current_worklist
                    else str(Path(binding["worklist_path"]).resolve())
                ),
                "blocked_worklist_file_sha256": binding["worklist_file_sha256"],
                "blocked_worklist_content_sha256": binding["worklist_content_sha256"],
                "canonical_source_sha256": worklist["canonical_source_sha256"],
                "worklist_mode": worklist_mode,
                "feature_namespace": worklist["feature_namespace"],
                "prompt_contract": worklist["prompt_contract"],
                "preprocess_identity": worklist["preprocess_identity"],
                "target_tensor_contract": target_tensor_contract,
                "selected_scene_layer_covered": True,
                "source_plan_summary": spec["source_plan_summary"],
                "source_plan_summary_sha256": spec["source_plan_summary_sha256"],
            }
            if rebound_to_current_worklist:
                old_audit_path = Path(str(gap.get("source_audit", ""))) if isinstance(gap, dict) else None
                if (
                    old_audit_path is None
                    or not old_audit_path.is_file()
                    or sha256_file(old_audit_path) != source_audit_sha_for_spec
                    or gap.get("source_audit_sha256") != source_audit_sha_for_spec
                ):
                    raise ValueError(f"rebound canary receipt lost its locked source audit: {vfm}")
                old_audit = read_json(old_audit_path, "rebound canary source audit")
                old_audit_unhashed = dict(old_audit)
                old_audit_unhashed.pop("content_sha256", None)
                if (
                    gap.get("source_audit_content_sha256")
                    != canonical_json_sha(old_audit_unhashed)
                ):
                    raise ValueError(f"rebound canary receipt source-audit content mismatch: {vfm}")
            else:
                expected_gap.update(
                    {
                        "source_audit": str(Path(binding["audit_path"]).resolve()),
                        "source_audit_sha256": binding["audit_sha256"],
                        "source_audit_content_sha256": binding["audit_content_sha256"],
                    }
                )
            gap_bad = [
                key
                for key, value in expected_gap.items()
                if not isinstance(gap, dict) or gap.get(key) != value
            ]
            comparison_ok = (
                isinstance(comparisons, dict)
                and set(comparisons) == {"streaming_prefix", "target_isolated"}
                and all(
                    isinstance(value, dict)
                    and int(value.get("comparison", {}).get("safetensors_compared", 0)) > 0
                    and int(value.get("comparison", {}).get("files_compared", 0))
                    >= int(value.get("comparison", {}).get("safetensors_compared", 0))
                    and isinstance(
                        value.get("comparison", {}).get("comparison_sha256"), str
                    )
                    and len(value["comparison"]["comparison_sha256"]) == 64
                    for value in comparisons.values()
                )
                and comparisons["streaming_prefix"]["comparison"].get("scene")
                == comparisons["target_isolated"]["comparison"].get("scene")
                == (gap.get("selected_scene") if isinstance(gap, dict) else None)
            )
            qwen_contract = None
            qwen_contract_ok = "qwen_streaming_tensor_contract" not in receipt
            if vfm == "qwen2_5_vl":
                try:
                    qwen_contract = validate_qwen_streaming_tensor_contract(
                        receipt.get("qwen_streaming_tensor_contract")
                    )
                    qwen_contract_ok = True
                except RuntimeError:
                    qwen_contract_ok = False
            target_contract_ok = "target_isolated_tensor_contract" not in receipt
            target_observed = None
            if worklist_mode == "target_isolated":
                try:
                    target_observed = validate_target_isolated_tensor_contract(
                        receipt.get("target_isolated_tensor_contract"),
                        target_tensor_contract,
                    )
                    target_contract_ok = True
                except RuntimeError:
                    target_contract_ok = False
            if (
                receipt_bad
                or gap_bad
                or not comparison_ok
                or not qwen_contract_ok
                or not target_contract_ok
            ):
                raise ValueError(
                    f"backend canary receipt is not certifiable ({vfm}): "
                    f"receipt={receipt_bad} gap={gap_bad} comparisons={comparison_ok} "
                    f"qwen_tensor_contract={qwen_contract_ok} "
                    f"target_tensor_contract={target_contract_ok}"
                )
            result[key] = {
                "path": str(receipt_path.resolve()),
                "sha256": sha256_file(receipt_path),
                "spec_path": str(spec_path.resolve()),
                "spec_sha256": require_spec_sha,
                "worklist_file_sha256": binding["worklist_file_sha256"],
                "worklist_content_sha256": binding["worklist_content_sha256"],
                "canary_runner_sha256": sha256_file(canary_runner),
                "backend_provenance": provenance,
                "qwen_streaming_expected": qwen_contract,
                "target_isolated_expected": target_observed,
                "receipt_rebound_to_current_worklist_by_identical_sha": rebound_to_current_worklist,
                "current_worklist_path": binding["worklist_path"],
                "current_worklist_file_sha256": binding["worklist_file_sha256"],
                "current_worklist_content_sha256": binding["worklist_content_sha256"],
            }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", action="append", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--canonical-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--include-vfm",
        action="append",
        default=None,
        help=(
            "Render only these VFM worklists (repeatable).  This is a strict "
            "positive subset of the bound audits, for independently runnable "
            "backends; it never derives additional work."
        ),
    )
    parser.add_argument(
        "--backend-canary-dir",
        action="append",
        type=Path,
        help="Immutable canary-plan directory; repeat once per certified mode.",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("/mnt/umm/users/yinbaiqiao/InsScene-15K"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--runner",
        type=Path,
        help=(
            "Optional immutable gap runner. Defaults to the legacy model-serial "
            "runner; atomic-owner plans bind the disjoint-worklist runner explicitly."
        ),
    )
    parser.add_argument("--scene-count", type=int, default=682)
    parser.add_argument("--n-gpus", type=int, default=8)
    parser.add_argument(
        "--minimax-h3-model-mirror-receipt",
        type=Path,
        help="Verified MiniMax-H3 AOSS model mirror receipt. Required to schedule minimax_h3.",
    )
    parser.add_argument("--workspace", default="aigc")
    parser.add_argument("--aec2-name", default="h800")
    parser.add_argument("--worker-spec", default="N4lS.Iq.I80.8")
    parser.add_argument("--priority", default="HIGHEST")
    parser.add_argument("--quota-type", choices=["reserved", "spot"], default="reserved")
    parser.add_argument("--container-image", default="registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617")
    parser.add_argument("--storage-mount", default="019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm")
    parser.add_argument(
        "--job-name-suffix",
        default="",
        help=(
            "Optional unique suffix appended to SCO job names. This only changes "
            "the scheduler display name; batch/worklist identities remain SHA-bound."
        ),
    )
    parser.add_argument(
        "--one-job-per-shard",
        action="store_true",
        help=(
            "Emit one SCO job per shard instead of one serial batch per VFM. "
            "Streaming and target write disjoint namespaces, so they can run "
            "as independent jobs when requested."
        ),
    )
    args = parser.parse_args(argv)

    include_vfms = (
        {str(value) for value in args.include_vfm}
        if args.include_vfm is not None
        else None
    )

    project_root = args.project_root.resolve()
    protocol_path = args.protocol.resolve()
    source_path = args.canonical_source.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to replace immutable output directory: {output_dir}")
    if args.scene_count < 1 or args.n_gpus < 1:
        raise SystemExit("scene-count and n-gpus must be positive")
    job_name_suffix = str(args.job_name_suffix).strip()
    if job_name_suffix:
        if slug(job_name_suffix, 64) != job_name_suffix:
            raise SystemExit("job-name-suffix must already be a safe lowercase slug")
    protocol = read_json(protocol_path, "canonical protocol")
    if protocol.get("canonical_protocol") != CANONICAL_PROTOCOL:
        raise SystemExit("wrong canonical protocol")
    protocol_file_sha = sha256_file(protocol_path)
    source_sha = sha256_file(source_path)
    bindings = validate_audits(
        args.source_audit,
        canonical_source_sha=source_sha,
        protocol_file_sha=protocol_file_sha,
    )
    runner = (
        args.runner.resolve()
        if args.runner is not None
        else project_root / "scripts/windowfix_v2/run_canonical_gap_shard.py"
    )
    aoss_sync_loop = project_root / "scripts/run_tstep_sync_loop.sh"
    aoss_gap_adapter = project_root / "scripts/windowfix_v2/canonical_gap_aoss_adapter.py"
    main_extractor = project_root / "features/run_inscene15k.py"
    mllm_extractor = project_root / "features/run_inscene15k_mllm.py"
    filter_source = project_root / "features/canonical_gap_worklist.py"
    for path in (
        runner,
        aoss_sync_loop,
        aoss_gap_adapter,
        main_extractor,
        mllm_extractor,
        filter_source,
    ):
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"missing bound source: {path}")
    canary_bindings = valid_canary_bindings(
        args.backend_canary_dir,
        protocol_file_sha,
        bindings,
        project_root,
    )
    minimax_model_mirror = minimax_h3_model_mirror_binding(
        args.minimax_h3_model_mirror_receipt,
        project_root,
    )
    ready_canary_keys = set(canary_bindings)
    plan_source_bindings = {
        "runner": {"path": str(runner), "sha256": sha256_file(runner)},
        "record_filter": {
            "path": str(filter_source),
            "sha256": sha256_file(filter_source),
        },
        "main_extractor": {
            "path": str(main_extractor),
            "sha256": sha256_file(main_extractor),
        },
        "mllm_extractor": {
            "path": str(mllm_extractor),
            "sha256": sha256_file(mllm_extractor),
        },
        "aoss_sync_loop": {
            "path": str(aoss_sync_loop),
            "sha256": sha256_file(aoss_sync_loop),
        },
        "aoss_gap_adapter": {
            "path": str(aoss_gap_adapter),
            "sha256": sha256_file(aoss_gap_adapter),
        },
    }
    plan_source_contract_sha = canonical_json_sha(plan_source_bindings)

    staging = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    staging.mkdir(parents=True)
    for child in ("shards", "batches", "jobspecs", "scripts", "logs"):
        (staging / child).mkdir()
    plan_id = (
        f"windowfix-v2-gaps-{protocol_file_sha[:12]}-"
        f"{canonical_json_sha([(b['worklist_file_sha256']) for b in bindings])[:12]}-"
        f"{plan_source_contract_sha[:12]}"
    )
    shards = []
    blocked = []
    selected_bindings = [
        binding
        for binding in bindings
        if include_vfms is None or str(binding["worklist"]["vfm"]) in include_vfms
    ]
    if not selected_bindings:
        raise SystemExit("--include-vfm selected zero audit-bound worklists")
    observed_vfms = {str(binding["worklist"]["vfm"]) for binding in bindings}
    if include_vfms is not None and not include_vfms.issubset(observed_vfms):
        unknown = sorted(include_vfms - observed_vfms)
        raise SystemExit(f"--include-vfm contains unknown audit VFM(s): {unknown}")

    for number, binding in enumerate(
        sorted(selected_bindings, key=lambda value: (
            value["worklist"]["vfm"], int(value["worklist"]["timestep"]),
            value["worklist"]["mode"], value["worklist_file_sha256"],
        )),
        1,
    ):
        worklist = binding["worklist"]
        mode = str(worklist["mode"])
        vfm = str(worklist["vfm"])
        namespace = str(worklist["feature_namespace"])
        if namespace != GAP_NAMESPACES.get(mode):
            raise SystemExit(f"worklist gap namespace mismatch: {binding['worklist_path']}")
        if worklist.get("protocol_sha256") != protocol.get("canonical_protocol_sha256"):
            raise SystemExit("worklist extractor-protocol SHA mismatch")
        if worklist.get("protocol_file_sha256") != protocol_file_sha:
            raise SystemExit("worklist protocol-file SHA mismatch")
        if worklist.get("canonical_source_sha256") != source_sha:
            raise SystemExit("worklist canonical-source SHA mismatch")
        preprocess = worklist.get("preprocess_identity")
        if not isinstance(preprocess, dict):
            raise SystemExit("worklist preprocess_identity must be an object")
        if mode == "streaming_prefix":
            try:
                fixed_frames = int(preprocess.get("num_frames", 1))
                expected_preprocess = streaming_worklist_preprocess_identity(
                    vfm, preprocess.get("resize", ()), fixed_frames
                )
            except (TypeError, ValueError):
                raise SystemExit("invalid streaming preprocess_identity") from None
            if preprocess != expected_preprocess:
                raise SystemExit("worklist streaming preprocess_identity is not canonical")
        elif set(preprocess) != {"resize", "num_frames"}:
            raise SystemExit("target preprocess_identity must be exactly {resize,num_frames}")
        if vfm == "minimax_h3" and minimax_model_mirror is None:
            blocked.append(
                {
                    "vfm": vfm, "model_id": worklist["model_id"],
                    "timestep": worklist["timestep"], "mode": mode,
                    "layers": worklist["output_layers"],
                    "worklist_path": binding["worklist_path"],
                    "worklist_file_sha256": binding["worklist_file_sha256"],
                    "layer_items": worklist["counts"]["layer_items"],
                    "reason": "BLOCKED_MINIMAX_H3_MODEL_MIRROR_NOT_BOUND",
                }
            )
            continue
        if vfm not in CERTIFIED_VFMS and (vfm, mode) not in ready_canary_keys:
            blocked.append(
                {
                    "vfm": vfm, "model_id": worklist["model_id"],
                    "timestep": worklist["timestep"], "mode": mode,
                    "layers": worklist["output_layers"],
                    "worklist_path": binding["worklist_path"],
                    "worklist_file_sha256": binding["worklist_file_sha256"],
                    "layer_items": worklist["counts"]["layer_items"],
                    "reason": "BLOCKED_BACKEND_CANARY_NOT_BOUND",
                }
            )
            continue
        job_token = slug(f"{vfm}-{worklist['timestep']}-{mode}", 40)
        shard_id = f"wf2-gap-{number:03d}-{job_token}"
        shard_path = output_dir / "shards" / f"{shard_id}.json"
        source_bindings = {
            "runner": plan_source_bindings["runner"],
            "record_filter": plan_source_bindings["record_filter"],
            "extractor": {
                "path": str(mllm_extractor if vfm in MLLM_VFMS else main_extractor),
                "sha256": sha256_file(mllm_extractor if vfm in MLLM_VFMS else main_extractor),
            },
            "aoss_sync_loop": plan_source_bindings["aoss_sync_loop"],
            "aoss_gap_adapter": plan_source_bindings["aoss_gap_adapter"],
            "backend_dependencies": backend_dependency_bindings(vfm, project_root),
            "model_configs": (
                {}
                if vfm == "minimax_h3"
                else local_model_config_bindings(str(worklist["model_id"]))
            ),
        }
        if vfm == "minimax_h3":
            source_bindings["model_mirror"] = minimax_model_mirror
        canary_binding = canary_bindings.get((vfm, mode))
        shard = {
            "schema_version": 1, "status": "READY", "contract": CONTRACT,
            "plan_id": plan_id, "shard_id": shard_id,
            "canonical_protocol": CANONICAL_PROTOCOL,
            "protocol_path": str(protocol_path), "protocol_file_sha256": protocol_file_sha,
            "protocol_sha256": protocol["canonical_protocol_sha256"],
            "canonical_source_path": str(source_path), "canonical_source_sha256": source_sha,
            "source_audit_path": binding["audit_path"],
            "source_audit_sha256": binding["audit_sha256"],
            "source_audit_content_sha256": binding["audit_content_sha256"],
            "worklist_path": binding["worklist_path"],
            "worklist_file_sha256": binding["worklist_file_sha256"],
            "worklist_content_sha256": binding["worklist_content_sha256"],
            "mode": mode, "vfm": vfm, "model_id": worklist["model_id"],
            "timestep": int(worklist["timestep"]),
            "layers": [int(value) for value in worklist["output_layers"]],
            "source": str(worklist.get("source", "scannetpp")),
            "feature_namespace": namespace,
            "feature_root": str(args.dataset_root.resolve() / namespace),
            "dataset_data_root": str(args.dataset_root.resolve() / "data"),
            "prompt_contract": worklist["prompt_contract"],
            "runtime_prompt": "",
            "preprocess_identity": preprocess,
            "counts": worklist["counts"],
            "scene_start": 0, "scene_end": args.scene_count, "n_gpus": args.n_gpus,
            "python": python_for(vfm), "project_root": str(project_root),
            "environment": environment_for_shard(vfm, project_root, canary_binding),
            "source_bindings": source_bindings,
            "backend_canary_binding": canary_binding,
        }
        shard["content_sha256"] = canonical_json_sha(shard)
        atomic_json(staging / "shards" / f"{shard_id}.json", shard)
        shards.append(shard)

    by_vfm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for shard in shards:
        by_vfm[shard["vfm"]].append(shard)
    job_groups: list[tuple[str, list[dict[str, Any]]]] = []
    for vfm in sorted(by_vfm):
        ordered = sorted(
            by_vfm[vfm],
            key=lambda value: (
                int(value["timestep"]),
                0 if value["mode"] == "streaming_prefix" else 1,
                value["shard_id"],
            ),
        )
        if args.one_job_per_shard:
            job_groups.extend((vfm, [item]) for item in ordered)
        else:
            job_groups.append((vfm, ordered))
    for batch_number, (vfm, ordered) in enumerate(job_groups, 1):
        mode_token = ""
        if args.one_job_per_shard:
            mode = str(ordered[0]["mode"])
            mode_token = "stream" if mode == "streaming_prefix" else "target"
            batch_id = f"wf2-gap-batch-{batch_number:02d}-{slug(vfm, 20)}-{mode_token}"
            serialization_scope = f"{mode} only for {vfm}"
            serialization_reason = (
                "one SCO job per shard; streaming and target write disjoint namespaces"
            )
            job_stem = f"bq-wf2-gap-{vfm}-{mode_token}-{batch_number}"
        else:
            batch_id = f"wf2-gap-batch-{batch_number:02d}-{slug(vfm, 30)}"
            serialization_scope = f"all canonical-v2 gap namespaces for {vfm}"
            serialization_reason = "one VFM per serial batch prevents index/delta writer races"
            job_stem = f"bq-wf2-gap-{vfm}-serial-{batch_number}"
        batch_path = output_dir / "batches" / f"{batch_id}.json"
        batch = {
            "schema_version": 1, "status": "READY", "plan_id": plan_id,
            "batch_id": batch_id, "vfm": vfm,
            "serialization_scope": serialization_scope,
            "serialization_reason": serialization_reason,
            "shards": [
                {
                    "path": str(output_dir / "shards" / f"{item['shard_id']}.json"),
                    "sha256": sha256_file(staging / "shards" / f"{item['shard_id']}.json"),
                    "shard_id": item["shard_id"], "mode": item["mode"],
                    "timestep": item["timestep"],
                }
                for item in ordered
            ],
        }
        batch["content_sha256"] = canonical_json_sha(batch)
        atomic_json(staging / "batches" / f"{batch_id}.json", batch)
        worker = [python_for(vfm), str(runner), "--batch", str(batch_path), "--execute"]
        batch_log = output_dir / "logs" / f"{batch_id}.log"
        worker_script_path = output_dir / "scripts" / f"{batch_id}.sh"
        worker_inner = f"cd {shlex.quote(str(project_root))} && {render(worker)}"
        worker_script = (
            "#!/usr/bin/env bash\n"
            "set -o pipefail; "
            f"mkdir -p {shlex.quote(str(batch_log.parent))}; "
            f"({worker_inner}) > {shlex.quote(str(batch_log))} 2>&1; "
            "gap_rc=$?; "
            f"chmod 0644 {shlex.quote(str(batch_log))} || true; "
            "exit ${gap_rc}"
        )
        staged_worker_script = staging / "scripts" / f"{batch_id}.sh"
        staged_worker_script.write_text(worker_script + "\n", encoding="utf-8")
        staged_worker_script.chmod(0o755)
        worker_command = (
            f"cd {shlex.quote(str(project_root))} && "
            f"{render(['bash', str(worker_script_path)])}"
        )
        job_name = slug(job_stem)
        if job_name_suffix:
            job_name = slug(f"{job_name}-{job_name_suffix}")
        sco_argv = [
            "sco", "acp", "jobs", "create",
            f"--workspace-name={args.workspace}", f"--aec2-name={args.aec2_name}",
            f"--job-name={job_name}",
            f"--priority={args.priority}", f"--quota-type={args.quota_type}",
            f"--container-image-url={args.container_image}",
            f"--storage-mount={args.storage_mount}", "--training-framework=pytorch",
            "--worker-nodes=1", f"--worker-spec={args.worker_spec}",
            f"--command={worker_command}",
        ]
        jobspec = {
            "schema_version": 1, "plan_id": plan_id, "batch_id": batch_id,
            "batch_path": str(batch_path),
            "batch_sha256": sha256_file(staging / "batches" / f"{batch_id}.json"),
            "vfm": vfm, "shard_count": len(ordered),
            "batch_log": str(batch_log),
            "worker_script": str(worker_script_path),
            "worker_script_sha256": sha256_file(staged_worker_script),
            "sco_argv": sco_argv, "sco_command": render(sco_argv),
            "submission_policy": "REVIEW_ONLY_NEVER_SUBMITTED_BY_PREPARER",
        }
        atomic_json(staging / "jobspecs" / f"{batch_id}.json", jobspec)

    commands = [
        "# REVIEW ARTIFACT ONLY. The renderer never invokes SCO.",
        *[
            read_json(path, "jobspec")["sco_command"]
            for path in sorted((staging / "jobspecs").glob("*.json"))
        ],
    ]
    (staging / "sco_create_commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    atomic_json(staging / "blocked.json", {"schema_version": 1, "blocked": blocked})
    summary = {
        "schema_version": 1, "plan_id": plan_id,
        "status": "BLOCKED_PARTIAL" if blocked else "READY_FOR_REVIEW",
        "submission_performed": False,
        "plan_source_contract_sha256": plan_source_contract_sha,
        "plan_source_bindings": plan_source_bindings,
        "protocol_file_sha256": protocol_file_sha,
        "canonical_source_sha256": source_sha,
        "source_audits": [
            {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}
            for path in args.source_audit
        ],
        "backend_canary_bindings": {
            f"{vfm}.{mode}": binding
            for (vfm, mode), binding in sorted(canary_bindings.items())
        },
        "counts": {
            "audited_worklists": len(bindings),
            "selected_audited_worklists": len(selected_bindings),
            "ready_shards": len(shards),
            "blocked_worklists": len(blocked), "ready_sco_batches": len(job_groups),
            "ready_layer_items": sum(item["counts"]["layer_items"] for item in shards),
            "blocked_layer_items": sum(item["layer_items"] for item in blocked),
            "streaming_layer_items": sum(
                item["counts"]["layer_items"] for item in shards if item["mode"] == "streaming_prefix"
            ),
            "target_layer_items": sum(
                item["counts"]["layer_items"] for item in shards if item["mode"] == "target_isolated"
            ),
        },
        "blocked": blocked,
        "notes": [
            "No SCO API/CLI was invoked.",
            "Only positive allow-listed gaps from SHA-bound audits are scheduled.",
            "Target jobs write per-layer deltas only; full materialization requires a duplicate-free row-route audit.",
            (
                "VFM subset: all audit-bound worklists"
                if include_vfms is None
                else "VFM subset: " + ",".join(sorted(include_vfms))
            ),
            (
                "One SCO job per shard."
                if args.one_job_per_shard
                else "One serial SCO batch per VFM."
            ),
        ],
    }
    atomic_json(staging / "summary.json", summary)
    lock = {
        "schema_version": 1, "plan_id": plan_id,
        "summary_sha256": sha256_file(staging / "summary.json"),
        "blocked_sha256": sha256_file(staging / "blocked.json"),
        "commands_sha256": sha256_file(staging / "sco_create_commands.txt"),
        "shards": {path.name: sha256_file(path) for path in sorted((staging / "shards").glob("*.json"))},
        "batches": {path.name: sha256_file(path) for path in sorted((staging / "batches").glob("*.json"))},
        "jobspecs": {path.name: sha256_file(path) for path in sorted((staging / "jobspecs").glob("*.json"))},
        "worker_scripts": {path.name: sha256_file(path) for path in sorted((staging / "scripts").glob("*.sh"))},
    }
    atomic_json(staging / "PLAN.lock.json", lock)
    os.replace(staging, output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
