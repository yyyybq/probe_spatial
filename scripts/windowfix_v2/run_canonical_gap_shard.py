#!/usr/bin/env python3
"""Validate and run one reviewed canonical-v2 gap-only extraction shard.

The runner accepts only immutable ``CANONICAL_V2_GAP_WORKLIST_V1`` positive
allow-lists.  It never derives work from missing files.  Without ``--execute``
it is read-only and prints the exact per-GPU extractor commands.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.canonical_gap_worklist import CanonicalGapWorklist, canonical_json_sha, sha256_file
from features.canonical_gap_worklist import (
    COMPLETED_SCENES_CONTRACT,
    COMPLETED_UNITS_CONTRACT,
    LOCAL_READY_UNITS_CONTRACT,
)
from scripts.windowfix_v2.canonical_gap_aoss_adapter import (
    GapSyncError,
    build_snapshot,
    completed_scenes_from_receipt,
    completed_units_from_receipt,
    expected_extraction_unit_keys,
    expected_scene_unit_keys,
    expected_upload_units,
    load_verified_receipt,
)


GAP_NAMESPACES = {
    "streaming_prefix": "FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1",
    "target_isolated": "FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1",
}
MLLM_VFMS = {"qwen2_5_vl", "bagel", "bagel_hf", "generic_hf"}
AOSS_SYNC_LOOP = PROJECT_ROOT / "scripts/run_tstep_sync_loop.sh"
GAP_AOSS_ADAPTER = PROJECT_ROOT / "scripts/windowfix_v2/canonical_gap_aoss_adapter.py"
SYNC_PYTHON = "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"
MINIMAX_H3_MIRROR_RECEIPT_CONTRACT = "MINIMAX_H3_AOSS_MODEL_MIRROR_RECEIPT_V1"


def fail(message: str) -> None:
    raise SystemExit(message)


def read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        fail(f"{label} must be a regular non-symlink file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        fail(f"{label} must contain one JSON object")
    return value


def regular_resolved_path(value: Any, label: str) -> Path:
    """Reject a symlink at the supplied path before resolving it."""
    raw = Path(str(value or ""))
    if raw.is_symlink() or not raw.is_file():
        fail(f"{label} must be a regular non-symlink file: {raw}")
    resolved = raw.resolve()
    if resolved.is_symlink() or not resolved.is_file():
        fail(f"resolved {label} must be a regular file: {resolved}")
    return resolved


def require_sha(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64 or sha256_file(path) != expected:
        fail(f"{label} SHA-256 mismatch: {path}")


def ensure_no_symlink_components(path: Path, stop: Path) -> None:
    path_abs = path.absolute()
    stop_abs = stop.absolute()
    try:
        relative = path_abs.relative_to(stop_abs)
    except ValueError:
        fail(f"gap feature root escapes dataset root: {path_abs}")
    current = stop_abs
    if current.is_symlink():
        fail(f"dataset root is a symlink: {current}")
    for component in relative.parts:
        current = current / component
        if current.exists() and current.is_symlink():
            fail(f"gap feature root contains a symlink component: {current}")


def validate_shard(path: Path, *, trust_current_tools: bool = False) -> dict[str, Any]:
    shard_path = regular_resolved_path(path, "gap shard")
    shard = read_json(shard_path, "gap shard")
    stored = shard.get("content_sha256")
    unhashed = dict(shard)
    unhashed.pop("content_sha256", None)
    if shard.get("schema_version") != 1 or shard.get("status") != "READY":
        fail("gap shard is not READY schema v1")
    if shard.get("contract") != "CANONICAL_V2_GAP_WORKLIST_V1":
        fail("gap shard contract mismatch")
    if stored != canonical_json_sha(unhashed):
        fail("gap shard content_sha256 mismatch")
    mode = str(shard.get("mode"))
    if shard.get("feature_namespace") != GAP_NAMESPACES.get(mode):
        fail("gap shard mode/feature namespace mismatch")
    if not shard.get("layers") or len(set(shard["layers"])) != len(shard["layers"]):
        fail("gap shard layers must be a unique non-empty list")
    if int(shard.get("scene_end", 0)) <= int(shard.get("scene_start", 0)):
        fail("gap shard has an empty scene range")
    if int(shard.get("n_gpus", 0)) < 1:
        fail("gap shard n_gpus must be positive")

    protocol_path = regular_resolved_path(shard.get("protocol_path"), "protocol")
    source_path = regular_resolved_path(
        shard.get("canonical_source_path"), "canonical source"
    )
    worklist_path = regular_resolved_path(shard.get("worklist_path"), "gap worklist")
    audit_path = regular_resolved_path(shard.get("source_audit_path"), "source audit")
    require_sha(protocol_path, shard.get("protocol_file_sha256"), "protocol")
    require_sha(source_path, shard.get("canonical_source_sha256"), "canonical source")
    require_sha(worklist_path, shard.get("worklist_file_sha256"), "gap worklist")
    require_sha(audit_path, shard.get("source_audit_sha256"), "source route/row audit")
    source_audit = read_json(audit_path, "source route/row audit")
    audit_unhashed = dict(source_audit)
    audit_content_sha = audit_unhashed.pop("content_sha256", None)
    if (
        audit_content_sha != canonical_json_sha(audit_unhashed)
        or audit_content_sha != shard.get("source_audit_content_sha256")
    ):
        fail("source route/row audit content SHA mismatch")
    audit_source = source_audit.get("canonical_source", {})
    if audit_source.get("sha256") != shard.get("canonical_source_sha256"):
        fail("source audit canonical-source binding mismatch")
    audit_worklists = source_audit.get("outputs", {}).get("gap_worklists", [])
    if not any(
        not Path(str(item.get("path", ""))).is_symlink()
        and Path(str(item.get("path", ""))).resolve() == worklist_path
        and item.get("file_sha256") == shard.get("worklist_file_sha256")
        and item.get("content_sha256") == shard.get("worklist_content_sha256")
        for item in audit_worklists
        if isinstance(item, dict)
    ):
        fail("source audit does not bind this gap worklist")
    source_bindings = shard.get("source_bindings")
    expected_source_bindings = {
        "runner", "record_filter", "extractor", "aoss_sync_loop",
        "aoss_gap_adapter", "backend_dependencies", "model_configs"
    }
    if str(shard.get("vfm")) == "minimax_h3":
        expected_source_bindings.add("model_mirror")
    if not isinstance(source_bindings, dict) or set(source_bindings) != expected_source_bindings:
        fail("gap shard source_bindings are incomplete")
    skip_tool_pins = {"runner", "aoss_sync_loop", "aoss_gap_adapter"} if trust_current_tools else set()
    skip_backend_pins = trust_current_tools and os.environ.get(
        "GAP_TRUST_BACKEND", ""
    ).strip() in {"1", "true", "TRUE", "yes"}
    for label in (
        "runner", "record_filter", "extractor", "aoss_sync_loop", "aoss_gap_adapter"
    ):
        binding = source_bindings[label]
        if not isinstance(binding, dict):
            fail(f"invalid gap shard source binding: {label}")
        bound_path = regular_resolved_path(binding.get("path"), f"bound source {label}")
        if label in skip_tool_pins:
            continue
        require_sha(
            bound_path,
            binding.get("sha256"),
            f"bound source {label}",
        )
    dependencies = source_bindings["backend_dependencies"]
    if not isinstance(dependencies, dict) or not dependencies:
        fail("gap shard has no backend dependency bindings")
    for dependency_name, binding in dependencies.items():
        if not isinstance(dependency_name, str) or not isinstance(binding, dict):
            fail("invalid backend dependency binding")
        dependency_path = regular_resolved_path(
            binding.get("path"), f"backend dependency {dependency_name}"
        )
        try:
            expected_relative = dependency_path.relative_to(
                Path(str(shard["project_root"])).resolve()
            ).as_posix()
        except ValueError:
            fail(f"backend dependency escapes project root: {dependency_path}")
        if expected_relative != dependency_name:
            fail(f"backend dependency key/path mismatch: {dependency_name}")
        if skip_backend_pins:
            continue
        require_sha(
            dependency_path,
            binding.get("sha256"),
            f"backend dependency {dependency_name}",
        )
    model_configs = source_bindings["model_configs"]
    if not isinstance(model_configs, dict):
        fail("gap shard model_configs binding must be an object")
    if (
        Path(str(shard["model_id"])).is_absolute()
        and not model_configs
        and str(shard.get("vfm")) != "minimax_h3"
    ):
        fail("absolute local model has no bound config files")
    for config_name, binding in model_configs.items():
        if not isinstance(config_name, str) or not isinstance(binding, dict):
            fail("invalid model config binding")
        config_path = regular_resolved_path(
            binding.get("path"), f"model config {config_name}"
        )
        require_sha(
            config_path,
            binding.get("sha256"),
            f"model config {config_name}",
        )
    prompt_contract = shard.get("prompt_contract")
    if not isinstance(prompt_contract, dict):
        fail("gap shard prompt_contract is missing")
    actual_prompt = (
        str(prompt_contract.get("value"))
        if prompt_contract.get("mode") == "exact"
        else str(shard.get("runtime_prompt", ""))
    )
    preprocess = shard.get("preprocess_identity")
    if not isinstance(preprocess, dict):
        fail("gap shard preprocess_identity is missing")
    worklist = CanonicalGapWorklist.load(
        worklist_path,
        expected_sha256=str(shard["worklist_file_sha256"]),
        mode=mode,
        vfm=str(shard["vfm"]),
        model_id=str(shard["model_id"]),
        timestep=int(shard["timestep"]),
        feature_namespace=str(shard["feature_namespace"]),
        output_layers=[int(value) for value in shard["layers"]],
        protocol_sha256=str(shard["protocol_sha256"]),
        actual_prompt=actual_prompt,
        preprocess_identity=preprocess,
    )
    if worklist.content_sha256 != shard.get("worklist_content_sha256"):
        fail("gap shard/worklist content SHA mismatch")
    if worklist.canonical_source_sha256 != shard.get("canonical_source_sha256"):
        fail("gap worklist canonical source binding mismatch")
    if str(shard.get("vfm")) == "minimax_h3":
        validate_minimax_h3_model_mirror_binding(source_bindings.get("model_mirror"))
    if worklist.protocol_file_sha256 != shard.get("protocol_file_sha256"):
        fail("gap worklist protocol-file binding mismatch")
    shard["_binding_path"] = str(shard_path)
    shard["_binding_file_sha256"] = sha256_file(shard_path)
    return shard


def validate_minimax_h3_model_mirror_binding(binding: Any) -> dict[str, Any]:
    if not isinstance(binding, dict):
        fail("MiniMax-H3 shard has no model mirror binding")
    if binding.get("contract") != MINIMAX_H3_MIRROR_RECEIPT_CONTRACT:
        fail("MiniMax-H3 model mirror contract mismatch")
    receipt = binding.get("receipt")
    manifest = binding.get("manifest")
    hydrate_script = binding.get("hydrate_script")
    for label, item in (
        ("MiniMax-H3 mirror receipt", receipt),
        ("MiniMax-H3 mirror manifest", manifest),
        ("MiniMax-H3 mirror hydrator", hydrate_script),
    ):
        if not isinstance(item, dict):
            fail(f"{label} binding is missing")
        path = regular_resolved_path(item.get("path"), label)
        require_sha(path, item.get("sha256"), label)
    receipt_path = Path(str(receipt["path"])).resolve()
    receipt_value = read_json(receipt_path, "MiniMax-H3 model mirror receipt")
    unhashed = dict(receipt_value)
    stored_content = unhashed.pop("content_sha256", None)
    if (
        receipt_value.get("contract") != MINIMAX_H3_MIRROR_RECEIPT_CONTRACT
        or receipt_value.get("status") != "PASS_REMOTE_SIZE_VERIFIED"
        or stored_content != canonical_json_sha(unhashed)
        or stored_content != receipt.get("content_sha256")
    ):
        fail("MiniMax-H3 model mirror receipt is not verified")
    if receipt_value.get("manifest_file_sha256") != manifest.get("sha256"):
        fail("MiniMax-H3 model mirror manifest binding mismatch")
    if (
        receipt_value.get("manifest_content_sha256")
        != binding.get("mirror_content_sha256")
        or receipt_value.get("remote_base") != binding.get("remote_base")
    ):
        fail("MiniMax-H3 model mirror receipt remote binding mismatch")
    return binding


def validate_batch(
    path: Path, *, trust_current_tools: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    batch_path = regular_resolved_path(path, "gap batch")
    batch = read_json(batch_path, "gap batch")
    stored = batch.get("content_sha256")
    unhashed = dict(batch)
    unhashed.pop("content_sha256", None)
    if batch.get("schema_version") != 1 or batch.get("status") != "READY":
        fail("gap batch is not READY schema v1")
    if stored != canonical_json_sha(unhashed):
        fail("gap batch content_sha256 mismatch")
    shards = []
    seen = set()
    for binding in batch.get("shards", []):
        shard_path = regular_resolved_path(binding.get("path"), "batch shard")
        require_sha(shard_path, binding.get("sha256"), "batch shard")
        shard = validate_shard(shard_path, trust_current_tools=trust_current_tools)
        if shard["shard_id"] in seen or shard["shard_id"] != binding.get("shard_id"):
            fail("duplicate or mismatched shard binding in gap batch")
        if shard["vfm"] != batch.get("vfm"):
            fail("gap batch contains another VFM")
        seen.add(shard["shard_id"])
        shards.append(shard)
    if not shards:
        fail("gap batch has no shards")
    return batch, shards


def extractor_command(shard: dict[str, Any], start: int, end: int) -> list[str]:
    is_mllm = shard["vfm"] in MLLM_VFMS
    command = [
        str(shard["python"]),
        "-m",
        "features.run_inscene15k_mllm" if is_mllm else "features.run_inscene15k",
    ]
    if is_mllm:
        command.extend(["--backend", "qwen2_5_vl", "--vfm-name", str(shard["vfm"])])
    else:
        command.extend(["--vfm", str(shard["vfm"])])
    prompt_contract = shard["prompt_contract"]
    runtime_prompt = (
        str(prompt_contract["value"])
        if prompt_contract["mode"] == "exact"
        else str(shard.get("runtime_prompt", ""))
    )
    preprocess = shard["preprocess_identity"]
    runtime_num_frames = preprocess.get("num_frames")
    if runtime_num_frames is None:
        by_prefix = preprocess.get("num_frames_by_prefix")
        if not isinstance(by_prefix, dict) or not by_prefix:
            fail("streaming preprocess contract has no runtime frame counts")
        runtime_num_frames = max(int(value) for value in by_prefix.values())
    command.extend(
        [
            "--mode", str(shard["mode"]),
            "--data-root", str(shard["dataset_data_root"]),
            "--out-root", str(shard["feature_root"]),
            "--source", "scannetpp",
            "--t", str(shard["timestep"]),
            "--model-id", str(shard["model_id"]),
            "--output-layers", *[str(layer) for layer in shard["layers"]],
            # Qwen streaming is no-pad and resolves the numeric input count
            # per record.  This CLI maximum is not part of its unit identity.
            "--num-frames", str(runtime_num_frames),
            "--prompt", runtime_prompt,
            "--prefix-lengths", "8,12,16,24",
            "--prefix-max-len", "24",
            "--streaming-window-obs-len", "28",
            "--streaming-motion-step", "0.35",
            "--streaming-rotation-weight", "0.5",
            "--streaming-window-stride", "8",
            "--streaming-max-windows-per-scene", "4",
            "--canonical-gap-worklist", str(shard["worklist_path"]),
            "--canonical-gap-worklist-sha256", str(shard["worklist_file_sha256"]),
            "--start", str(start), "--end", str(end),
        ]
    )
    if is_mllm:
        command.extend(["--resize", *[str(value) for value in preprocess["resize"]]])
    if shard["mode"] == "target_isolated":
        command.extend(
            [
                "--target-from-streaming-windows",
                "--target-prefix-lengths", "8,12,16,24",
                "--target-horizons", "1,2,4",
            ]
        )
    return command


def _dataset_scene_names(shard: dict[str, Any]) -> list[str]:
    import numpy as np

    source_path = Path(str(shard["dataset_data_root"])) / "processed_scannetpp_v2"
    nested = source_path / "processed_scannetpp_v2"
    if nested.is_dir():
        source_path = nested
    names: list[str] = []
    for scene_id in sorted(os.listdir(source_path)):
        scene_dir = source_path / scene_id
        img_dir = scene_dir / "images"
        mask_dir = scene_dir / "refined_ins_ids"
        meta_path = scene_dir / "scene_iphone_metadata.npz"
        if not scene_dir.is_dir() or not img_dir.is_dir() or not mask_dir.is_dir() or not meta_path.is_file():
            continue
        mask_files = set(os.listdir(mask_dir))
        image_files = {name for name in os.listdir(img_dir) if name.endswith(".jpg")}
        meta = np.load(meta_path)
        meta_images = [str(name) for name in meta["images"].tolist()]
        valid = 0
        for fname in meta_images:
            if fname in image_files and f"{fname}.npy" in mask_files:
                valid += 1
        if valid >= 5:
            names.append(scene_id)
            needed = int(shard.get("scene_end", 0))
            if needed and len(names) >= needed:
                break
    return names


def _remaining_scene_indices(
    shard: dict[str, Any],
    completed_units: set[str],
    completed_scenes: set[str],
) -> list[int]:
    names = _dataset_scene_names(shard)
    start, end = int(shard["scene_start"]), int(shard["scene_end"])
    remaining_scenes = _worklist_scenes(shard)
    if completed_units:
        raw = read_json(Path(str(shard["worklist_path"])), "gap worklist")
        raw["_file_sha256"] = str(shard["worklist_file_sha256"])
        remaining_scenes = {
            scene
            for scene, units in expected_scene_unit_keys(raw).items()
            if units - completed_units
        }
    elif completed_scenes:
        remaining_scenes = remaining_scenes - completed_scenes
    return [
        index
        for index in range(start, min(end, len(names)))
        if names[index] in remaining_scenes
    ]


def cuda_visible_devices_for_process(*, process_index: int, n_processes: int) -> str:
    """Pin one extract process onto a visible GPU, packing evenly.

    An 8-GPU SCO job with two processes per card exposes devices
    ``0,1,2,3,4,5,6,7`` and ``n_processes=16``. Adjacent process indices
    then share a card. A 1-GPU job with two processes exposes a single
    device and both children inherit it.
    """
    visible = [
        token.strip()
        for token in str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).split(",")
        if token.strip()
    ]
    if not visible:
        return str(process_index)
    n_cards = len(visible)
    if n_processes < 1:
        fail("n_processes must be positive")
    if n_processes % n_cards == 0:
        return visible[process_index * n_cards // n_processes]
    return visible[process_index % n_cards]


def shard_commands(
    shard: dict[str, Any],
    *,
    n_gpus: int | None = None,
    remaining_indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    start, end, count = (
        int(shard["scene_start"]),
        int(shard["scene_end"]),
        int(n_gpus if n_gpus is not None else shard["n_gpus"]),
    )
    if count < 1:
        fail("gap shard n_gpus must be positive")
    if remaining_indices is None:
        remaining_indices = list(range(start, end))
    else:
        remaining_indices = [index for index in remaining_indices if start <= index < end]
    commands = []
    total = len(remaining_indices)
    for gpu in range(count):
        item_start_i = total * gpu // count
        item_end_i = total * (gpu + 1) // count
        if item_start_i >= item_end_i:
            continue
        chunk = remaining_indices[item_start_i:item_end_i]
        item_start = chunk[0]
        item_end = chunk[-1] + 1
        commands.append(
            {"gpu": gpu, "start": item_start, "end": item_end,
             "argv": extractor_command(shard, item_start, item_end)}
        )
    return commands


def write_json_atomic(path: Path, value: Any) -> None:
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


def _worklist_scenes(shard: dict[str, Any]) -> set[str]:
    raw = read_json(Path(str(shard["worklist_path"])), "gap worklist")
    entries = raw.get("entries")
    if not isinstance(entries, list):
        fail("gap worklist entries disappeared after validation")
    return {str(entry["scene"]) for entry in entries if isinstance(entry, dict)}


def _state_paths(shard: dict[str, Any], feature_root: Path) -> dict[str, Path]:
    # The content SHA is the immutable transaction identity.  Keeping state
    # outside the VFM/source payload subtree prevents ads-cli from ever
    # treating receipts or done indexes as feature objects.
    state = (
        feature_root
        / ".windowfix_v2_gap_remote_receipts"
        / str(shard["content_sha256"])
    )
    return {
        "root": state,
        "receipt": state / "remote_completion.json",
        "done_index": state / "done_index.txt",
    }


def _bind_completed_scenes(
    shard: dict[str, Any], completion_receipt: Path, output: Path
) -> tuple[set[str], str | None]:
    if not completion_receipt.exists():
        return set(), None
    remote = load_verified_receipt(
        completion_receipt,
        worklist_file_sha256=str(shard["worklist_file_sha256"]),
        shard_content_sha256=str(shard["content_sha256"]),
    )
    raw = read_json(Path(str(shard["worklist_path"])), "gap worklist")
    completed = completed_scenes_from_receipt(
        remote, raw, str(shard["worklist_file_sha256"])
    )
    binding = {
        "schema_version": 1,
        "contract": COMPLETED_SCENES_CONTRACT,
        "worklist_file_sha256": str(shard["worklist_file_sha256"]),
        "completion_receipt_content_sha256": str(remote["content_sha256"]),
        "scenes": sorted(completed),
    }
    binding["content_sha256"] = canonical_json_sha(binding)
    write_json_atomic(output, binding)
    return completed, sha256_file(output)


def _bind_completed_units(
    shard: dict[str, Any], completion_receipt: Path, output: Path
) -> tuple[set[str], str | None]:
    if not completion_receipt.exists():
        return set(), None
    remote = load_verified_receipt(
        completion_receipt,
        worklist_file_sha256=str(shard["worklist_file_sha256"]),
        shard_content_sha256=str(shard["content_sha256"]),
    )
    raw = read_json(Path(str(shard["worklist_path"])), "gap worklist")
    completed = completed_units_from_receipt(
        remote, raw, str(shard["worklist_file_sha256"])
    )
    binding = {
        "schema_version": 1,
        "contract": COMPLETED_UNITS_CONTRACT,
        "worklist_file_sha256": str(shard["worklist_file_sha256"]),
        "completion_receipt_content_sha256": str(remote["content_sha256"]),
        "units": sorted(completed),
    }
    binding["content_sha256"] = canonical_json_sha(binding)
    write_json_atomic(output, binding)
    return completed, sha256_file(output)


def _bind_local_ready_scenes(
    shard: dict[str, Any], payload_root: Path, output: Path
) -> tuple[set[str], str | None, list[str]]:
    """Bind locally complete units for this *one* execution attempt.

    A canonical GAP root can contain a payload written by an interrupted
    earlier attempt.  If its stable semantic identity is exactly the current
    worklist unit, uploading it is safe; re-running the extractor in parallel
    is not, because it can replace the sidecar between the AOSS snapshot and
    receipt commit.  This binding prevents only that overwrite.  It is not a
    resume authority: the next run still relies solely on the AOSS completion
    receipt.
    """

    preprocess = shard["preprocess_identity"]
    runtime_num_frames = preprocess.get("num_frames")
    if runtime_num_frames is None:
        by_prefix = preprocess.get("num_frames_by_prefix")
        if not isinstance(by_prefix, dict) or not by_prefix:
            fail("local-ready preflight has no runtime frame-count contract")
        runtime_num_frames = max(int(value) for value in by_prefix.values())
    prompt_contract = shard["prompt_contract"]
    prompt = (
        str(prompt_contract["value"])
        if prompt_contract.get("mode") == "exact"
        else ""
    )
    rows, invalid = build_snapshot(
        root=payload_root,
        scan_root=payload_root,
        worklist_path=Path(str(shard["worklist_path"])),
        worklist_sha256=str(shard["worklist_file_sha256"]),
        mode=str(shard["mode"]),
        vfm=str(shard["vfm"]),
        model_id=str(shard["model_id"]),
        timestep=int(shard["timestep"]),
        namespace=str(shard["feature_namespace"]),
        layers=[int(value) for value in shard["layers"]],
        protocol_sha256=str(shard["protocol_sha256"]),
        prompt=prompt,
        resize=[int(value) for value in preprocess["resize"]],
        num_frames=int(runtime_num_frames),
        max_scenes=len(_worklist_scenes(shard)),
    )
    scenes = sorted({str(row["unit"]) for row in rows})
    if not scenes:
        return set(), None, invalid
    snapshot_content = canonical_json_sha(rows)
    binding = {
        "schema_version": 1,
        "contract": "CANONICAL_V2_GAP_LOCAL_READY_SCENES_V1",
        "worklist_file_sha256": str(shard["worklist_file_sha256"]),
        "local_snapshot_content_sha256": snapshot_content,
        "scenes": scenes,
    }
    binding["content_sha256"] = canonical_json_sha(binding)
    write_json_atomic(output, binding)
    return set(scenes), sha256_file(output), invalid


def _bind_local_ready_units(
    shard: dict[str, Any],
    payload_root: Path,
    output: Path,
    *,
    exclude_units: set[str] | None = None,
) -> tuple[set[str], str | None, list[str]]:
    preprocess = shard["preprocess_identity"]
    runtime_num_frames = preprocess.get("num_frames")
    if runtime_num_frames is None:
        by_prefix = preprocess.get("num_frames_by_prefix")
        if not isinstance(by_prefix, dict) or not by_prefix:
            fail("local-ready preflight has no runtime frame-count contract")
        runtime_num_frames = max(int(value) for value in by_prefix.values())
    prompt_contract = shard["prompt_contract"]
    prompt = (
        str(prompt_contract["value"])
        if prompt_contract.get("mode") == "exact"
        else ""
    )
    previous_completeness = os.environ.get("GAP_SYNC_COMPLETENESS_ONLY")
    os.environ["GAP_SYNC_COMPLETENESS_ONLY"] = "1"
    try:
        rows, invalid = build_snapshot(
            root=payload_root,
            scan_root=payload_root,
            worklist_path=Path(str(shard["worklist_path"])),
            worklist_sha256=str(shard["worklist_file_sha256"]),
            mode=str(shard["mode"]),
            vfm=str(shard["vfm"]),
            model_id=str(shard["model_id"]),
            timestep=int(shard["timestep"]),
            namespace=str(shard["feature_namespace"]),
            layers=[int(value) for value in shard["layers"]],
            protocol_sha256=str(shard["protocol_sha256"]),
            prompt=prompt,
            resize=[int(value) for value in preprocess["resize"]],
            num_frames=int(runtime_num_frames),
            max_scenes=1_000_000,
            exclude_units=exclude_units,
        )
    finally:
        if previous_completeness is None:
            os.environ.pop("GAP_SYNC_COMPLETENESS_ONLY", None)
        else:
            os.environ["GAP_SYNC_COMPLETENESS_ONLY"] = previous_completeness
    raw = read_json(Path(str(shard["worklist_path"])), "gap worklist")
    raw["_file_sha256"] = str(shard["worklist_file_sha256"])
    upload_units = expected_upload_units(raw)
    # Worklist exclusions are extraction keys (target_isolated/frame_...),
    # not publishable upload keys (target_delta_layer/...).  Binding the
    # latter made CanonicalGapWorklist.load reject the file and the sync
    # loop exit with "no remaining publishable units".
    units = sorted(
        {
            str(extraction_unit)
            for row in rows
            if not row.get("keep_local")
            for extraction_unit in upload_units.get(str(row["unit"]), {}).get(
                "extraction_units", ()
            )
        }
    )
    if not units:
        return set(), None, invalid
    snapshot_content = canonical_json_sha(rows)
    binding = {
        "schema_version": 1,
        "contract": LOCAL_READY_UNITS_CONTRACT,
        "worklist_file_sha256": str(shard["worklist_file_sha256"]),
        "local_snapshot_content_sha256": snapshot_content,
        "units": units,
    }
    binding["content_sha256"] = canonical_json_sha(binding)
    write_json_atomic(output, binding)
    return set(units), sha256_file(output), invalid


def _sync_environment(
    shard: dict[str, Any],
    environment: dict[str, str],
    *,
    payload_root: Path,
    log_dir: Path,
    stop_file: Path,
    completion_receipt: Path,
    done_index: Path,
) -> dict[str, str]:
    preprocess = shard["preprocess_identity"]
    runtime_num_frames = preprocess.get("num_frames")
    if runtime_num_frames is None:
        by_prefix = preprocess.get("num_frames_by_prefix")
        if not isinstance(by_prefix, dict) or not by_prefix:
            fail("gap sync has no runtime frame-count contract")
        runtime_num_frames = max(int(value) for value in by_prefix.values())
    prompt_contract = shard["prompt_contract"]
    prompt = (
        str(prompt_contract["value"])
        if prompt_contract["mode"] == "exact"
        else str(shard.get("runtime_prompt", ""))
    )
    kind = "FEAT_STREAMING" if shard["mode"] == "streaming_prefix" else "FEAT_TARGET"
    result = dict(environment)
    # These two variables affect only extractor record filtering.  The sync
    # adapter must always validate the complete immutable worklist universe.
    result.pop("CANONICAL_GAP_COMPLETED_SCENES_FILE", None)
    result.pop("CANONICAL_GAP_COMPLETED_SCENES_FILE_SHA256", None)
    result.pop("CANONICAL_GAP_COMPLETED_UNITS_FILE", None)
    result.pop("CANONICAL_GAP_COMPLETED_UNITS_FILE_SHA256", None)
    result.update(
        {
            "FEATURE_KIND": kind,
            "FEATURE_NAMESPACE": str(shard["feature_namespace"]),
            "VFM": str(shard["vfm"]),
            "T": str(shard["timestep"]),
            "SOURCE": str(shard["source"]),
            "LAYERS": " ".join(str(value) for value in shard["layers"]),
            "DATASET_ROOT": str(Path(str(shard["dataset_data_root"])).resolve().parent),
            "FEATURE_ROOT": str(payload_root),
            "FEATURE_SCAN_ROOT": str(payload_root),
            "EXTRACT_DONE_INDEX": str(done_index),
            "LOG_DIR": str(log_dir),
            "STOP_FILE": str(stop_file),
            "CANONICAL_GAP_WORKLIST": str(shard["worklist_path"]),
            "CANONICAL_GAP_WORKLIST_SHA256": str(shard["worklist_file_sha256"]),
            "GAP_SHARD_PATH": str(shard["_binding_path"]),
            "GAP_SHARD_FILE_SHA256": str(shard["_binding_file_sha256"]),
            "GAP_SHARD_CONTENT_SHA256": str(shard["content_sha256"]),
            "GAP_REMOTE_RECEIPT": str(completion_receipt),
            "GAP_MODE": str(shard["mode"]),
            "GAP_PROTOCOL_SHA256": str(shard["protocol_sha256"]),
            "EXPECTED_PROTOCOL": str(shard["canonical_protocol"]),
            "EXPECTED_OBS_LEN": "28",
            "EXPECTED_MODEL_ID": str(shard["model_id"]),
            "EXPECTED_PROMPT": prompt,
            "EXPECTED_HEIGHT": str(preprocess["resize"][0]),
            "EXPECTED_WIDTH": str(preprocess["resize"][1]),
            "EXPECTED_NUM_FRAMES": str(runtime_num_frames),
            "EXPECTED_GAP_ADAPTER_SHA256": str(
                shard["source_bindings"]["aoss_gap_adapter"]["sha256"]
            ),
            "GAP_ADAPTER": str(
                shard["source_bindings"]["aoss_gap_adapter"]["path"]
            ),
            "SYNC_DISJOINT_SCAN": "0",
            # Production backpressure: a model worker may write far faster
            # than AOSS can publish a large multi-layer scene.  Publish a
            # small complete-scene batch at a time and probe frequently,
            # keeping the original sync loop's immutable
            # staging/coverage/receipt logic.  MAX_SCENES=1 let extract
            # outrun ads-cli and pile payloads on local quota; 8 keeps
            # publish caught up without accepting partial units.
            "GAP_SYNC_MAX_SCENES": "8",
            "SYNC_EVERY_SEC": "15",
            "SYNC_EVERY_GIB": "1",
        }
    )
    result.setdefault("SYNC_PYTHON", SYNC_PYTHON)
    return result


def hydrate_minimax_h3_model_mirror(shard: dict[str, Any], environment: dict[str, str]) -> str:
    binding = validate_minimax_h3_model_mirror_binding(
        shard["source_bindings"].get("model_mirror")
    )
    hydrate_script = Path(str(binding["hydrate_script"]["path"])).resolve()
    env = dict(environment)
    env.update(
        {
            "MINIMAX_H3_MIRROR_SHA256": str(binding["mirror_content_sha256"]),
            "MINIMAX_H3_MIRROR_REMOTE_BASE": str(binding["remote_base"]),
        }
    )
    completed = subprocess.run(
        ["/bin/bash", str(hydrate_script)],
        cwd=str(shard["project_root"]),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        fail(
            "MiniMax-H3 AOSS model hydration failed: "
            f"rc={completed.returncode} stderr={completed.stderr[-2000:]}"
        )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        fail("MiniMax-H3 hydrator did not print a model path")
    model_path = Path(lines[-1])
    if model_path.is_symlink() or not model_path.is_dir():
        fail(f"MiniMax-H3 hydrated model path is not a regular directory: {model_path}")
    return str(model_path.resolve())


def run_shard(
    shard: dict[str, Any], *, execute: bool, trust_current_tools: bool = False
) -> dict[str, Any]:
    if not execute:
        return {"dry_run": True, "shard_id": shard["shard_id"], "commands": shard_commands(shard)}
    dataset_root = Path(str(shard["dataset_data_root"])).absolute().parent
    feature_root = Path(str(shard["feature_root"])).absolute()
    ensure_no_symlink_components(feature_root, dataset_root)
    if feature_root.name != shard["feature_namespace"] or "SEEDED_V1" in feature_root.name:
        fail("gap runner refuses non-gap or SEEDED_V1 destination")
    feature_root.mkdir(parents=True, exist_ok=True)
    ensure_no_symlink_components(feature_root, dataset_root)
    payload_root = feature_root / str(shard["vfm"]) / str(shard["source"])
    ensure_no_symlink_components(payload_root, dataset_root)
    payload_root.mkdir(parents=True, exist_ok=True)
    ensure_no_symlink_components(payload_root, dataset_root)
    lock_dir = feature_root / ".windowfix_v2_gap_locks" / shard["vfm"] / shard["source"]
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (lock_dir / "all-modes.lock").open("a+")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-p{os.getpid()}"
    log_dir = feature_root / ".windowfix_v2_gap_logs" / shard["vfm"] / shard["shard_id"] / attempt
    log_dir.mkdir(parents=True, exist_ok=False)
    ensure_no_symlink_components(log_dir, dataset_root)
    state_paths = _state_paths(shard, feature_root)
    ensure_no_symlink_components(state_paths["root"], dataset_root)
    state_paths["root"].mkdir(parents=True, exist_ok=True)
    ensure_no_symlink_components(state_paths["root"], dataset_root)
    receipt_path = log_dir / "receipt.json"
    receipt = {
        "schema_version": 1, "status": "RUNNING", "attempt": attempt,
        "shard_id": shard["shard_id"], "shard_content_sha256": shard["content_sha256"],
        "worklist_file_sha256": shard["worklist_file_sha256"],
        "worklist_content_sha256": shard["worklist_content_sha256"],
        "canonical_source_sha256": shard["canonical_source_sha256"],
        "remote_completion_receipt": str(state_paths["receipt"]),
        "aoss_done_index": str(state_paths["done_index"]),
    }
    write_json_atomic(receipt_path, receipt)
    environment = os.environ.copy()
    environment.update({str(k): str(v) for k, v in shard.get("environment", {}).items()})
    environment.pop("EXTRACT_DONE_INDEX", None)
    environment["FEATURE_NAMESPACE"] = shard["feature_namespace"]
    if str(shard["vfm"]) == "minimax_h3":
        model_path = hydrate_minimax_h3_model_mirror(shard, environment)
        model_mirror = shard["source_bindings"]["model_mirror"]
        environment["MINIMAX_H3_MODEL_LOAD_ID"] = model_path
        environment["MINIMAX_H3_MIRROR_SHA256"] = str(model_mirror["mirror_content_sha256"])
        environment["MINIMAX_H3_MIRROR_REMOTE_BASE"] = str(model_mirror["remote_base"])
    project_root = str(shard["project_root"])
    extra = environment.pop("WINDOWFIX_EXTRA_PYTHONPATH", "")
    environment["PYTHONPATH"] = ":".join(
        value for value in (project_root, extra, environment.get("PYTHONPATH", "")) if value
    )
    for key in (
        "CANONICAL_GAP_COMPLETED_SCENES_FILE",
        "CANONICAL_GAP_COMPLETED_SCENES_FILE_SHA256",
        "CANONICAL_GAP_LOCAL_READY_SCENES_FILE",
        "CANONICAL_GAP_LOCAL_READY_SCENES_FILE_SHA256",
        "CANONICAL_GAP_COMPLETED_UNITS_FILE",
        "CANONICAL_GAP_COMPLETED_UNITS_FILE_SHA256",
        "CANONICAL_GAP_LOCAL_READY_UNITS_FILE",
        "CANONICAL_GAP_LOCAL_READY_UNITS_FILE_SHA256",
    ):
        environment.pop(key, None)
    exclusion_path = log_dir / "completed_scenes.json"
    try:
        completed_before, exclusion_sha = _bind_completed_scenes(
            shard, state_paths["receipt"], exclusion_path
        )
    except GapSyncError as exc:
        receipt.update({"status": "FAILED_PREFLIGHT", "error": str(exc)})
        write_json_atomic(receipt_path, receipt)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
        fail(f"invalid prior AOSS completion receipt: {exc}; receipt={receipt_path}")
    if exclusion_sha is not None:
        environment["CANONICAL_GAP_COMPLETED_SCENES_FILE"] = str(exclusion_path)
        environment["CANONICAL_GAP_COMPLETED_SCENES_FILE_SHA256"] = exclusion_sha
    completed_units_path = log_dir / "completed_units.json"
    try:
        completed_units_before, completed_units_sha = _bind_completed_units(
            shard, state_paths["receipt"], completed_units_path
        )
    except GapSyncError as exc:
        receipt.update({"status": "FAILED_PREFLIGHT", "error": str(exc)})
        write_json_atomic(receipt_path, receipt)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
        fail(f"invalid prior AOSS completion receipt: {exc}; receipt={receipt_path}")
    if completed_units_sha is not None:
        environment["CANONICAL_GAP_COMPLETED_UNITS_FILE"] = str(completed_units_path)
        environment["CANONICAL_GAP_COMPLETED_UNITS_FILE_SHA256"] = completed_units_sha
    receipt["completed_scenes_before"] = sorted(completed_before)
    receipt["completed_scenes_binding"] = (
        {"path": str(exclusion_path), "sha256": exclusion_sha}
        if exclusion_sha is not None
        else None
    )
    receipt["completed_units_before"] = sorted(completed_units_before)
    receipt["completed_units_binding"] = (
        {"path": str(completed_units_path), "sha256": completed_units_sha}
        if completed_units_sha is not None
        else None
    )
    local_ready_path = log_dir / "local_ready_scenes.json"
    local_ready: set[str] = set()
    local_ready_sha = None
    local_invalid: list[str] = []
    # Unit-level local-ready binding below is the active overwrite guard.
    # Scene-level local-ready is intentionally disabled for partial publishing:
    # a scene may have one publishable unit while adjacent units are still
    # being written.
    local_ready_units_path = log_dir / "local_ready_units.json"
    try:
        local_ready_units, local_ready_units_sha, local_ready_unit_invalid = (
            _bind_local_ready_units(
                shard,
                payload_root,
                local_ready_units_path,
                exclude_units=set(completed_units_before),
            )
        )
    except GapSyncError as exc:
        receipt.update({"status": "FAILED_PREFLIGHT", "error": str(exc)})
        write_json_atomic(receipt_path, receipt)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
        fail(f"invalid local GAP unit preflight: {exc}; receipt={receipt_path}")
    if local_ready_units:
        environment["CANONICAL_GAP_LOCAL_READY_UNITS_FILE"] = str(local_ready_units_path)
        environment["CANONICAL_GAP_LOCAL_READY_UNITS_FILE_SHA256"] = str(local_ready_units_sha)
    receipt["local_ready_scenes_before"] = sorted(local_ready)
    receipt["local_ready_scenes_binding"] = (
        {"path": str(local_ready_path), "sha256": local_ready_sha}
        if local_ready_sha is not None
        else None
    )
    receipt["local_ready_invalid_count"] = len(local_invalid)
    receipt["local_ready_units_before"] = sorted(local_ready_units)
    receipt["local_ready_units_binding"] = (
        {"path": str(local_ready_units_path), "sha256": local_ready_units_sha}
        if local_ready_units_sha is not None
        else None
    )
    receipt["local_ready_unit_invalid_count"] = len(local_ready_unit_invalid)
    write_json_atomic(receipt_path, receipt)

    n_gpus = int(shard["n_gpus"])
    if trust_current_tools:
        override = os.environ.get("GAP_N_GPUS", "").strip()
        if override:
            n_gpus = int(override)
            if n_gpus < 1:
                fail("GAP_N_GPUS must be positive")
    remaining = _remaining_scene_indices(
        shard, set(completed_units_before), set(completed_before)
    )
    commands = shard_commands(
        shard, n_gpus=n_gpus, remaining_indices=remaining
    )
    receipt["n_gpus"] = n_gpus
    receipt["remaining_scene_count"] = len(remaining)
    receipt["extract_gpu_ranges"] = [
        {"gpu": item["gpu"], "start": item["start"], "end": item["end"]}
        for item in commands
    ]
    write_json_atomic(receipt_path, receipt)

    stop_file = log_dir / "extract.finished"
    sync_log_dir = log_dir / "aoss_publish"
    sync_log_dir.mkdir()
    sync_log_path = sync_log_dir / "runner.log"
    sync_handle = sync_log_path.open("wb")
    sync_env = _sync_environment(
        shard,
        environment,
        payload_root=payload_root,
        log_dir=sync_log_dir,
        stop_file=stop_file,
        completion_receipt=state_paths["receipt"],
        done_index=state_paths["done_index"],
    )
    if trust_current_tools:
        sync_env["GAP_ADAPTER"] = str(GAP_AOSS_ADAPTER)
        sync_env["EXPECTED_GAP_ADAPTER_SHA256"] = sha256_file(GAP_AOSS_ADAPTER)
        sync_loop = str(AOSS_SYNC_LOOP)
    else:
        sync_loop = str(shard["source_bindings"]["aoss_sync_loop"]["path"])
    sync_process = subprocess.Popen(
        ["/bin/bash", sync_loop],
        cwd=project_root,
        env=sync_env,
        stdout=sync_handle,
        stderr=subprocess.STDOUT,
    )
    processes: list[tuple[subprocess.Popen, Any, dict[str, Any], Path]] = []
    interrupted = False

    def terminate(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        for process, _handle, _item, _path in processes:
            if process.poll() is None:
                process.terminate()
        if sync_process.poll() is None:
            sync_process.terminate()

    old = {sig: signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for item in commands:
            log_path = log_dir / f"gpu{item['gpu']}_{item['start']}_{item['end']}.log"
            handle = log_path.open("wb")
            child_env = dict(environment)
            child_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices_for_process(
                process_index=int(item["gpu"]),
                n_processes=n_gpus,
            )
            process = subprocess.Popen(
                item["argv"], cwd=project_root, env=child_env,
                stdout=handle, stderr=subprocess.STDOUT,
            )
            processes.append((process, handle, item, log_path))
        sync_early_failure = False
        while any(process.poll() is None for process, *_rest in processes):
            if sync_process.poll() is not None:
                sync_early_failure = True
                for process, _handle, _item, _path in processes:
                    if process.poll() is None:
                        process.terminate()
                break
            time.sleep(0.25)
        results = []
        for process, handle, item, log_path in processes:
            returncode = process.wait()
            handle.close()
            results.append({**{k: item[k] for k in ("gpu", "start", "end")},
                            "returncode": returncode, "log": str(log_path)})
        stop_file.touch(exist_ok=False)
        sync_returncode = sync_process.wait()
        sync_handle.close()
        failed = [value for value in results if value["returncode"]]
        remote_error = None
        completed_after: set[str] = set()
        completed_units_after: set[str] = set()
        try:
            remote = load_verified_receipt(
                state_paths["receipt"],
                worklist_file_sha256=str(shard["worklist_file_sha256"]),
                shard_content_sha256=str(shard["content_sha256"]),
            )
            raw = read_json(Path(str(shard["worklist_path"])), "gap worklist")
            raw["_file_sha256"] = str(shard["worklist_file_sha256"])
            completed_after = completed_scenes_from_receipt(
                remote, raw, str(shard["worklist_file_sha256"])
            )
            completed_units_after = completed_units_from_receipt(
                remote, raw, str(shard["worklist_file_sha256"])
            )
            missing_units = expected_extraction_unit_keys(raw) - completed_units_after
            if missing_units:
                remote_error = f"remote receipt is missing {len(missing_units)} worklist units"
        except (GapSyncError, FileNotFoundError) as exc:
            remote_error = str(exc)
        passed = (
            not interrupted
            and not sync_early_failure
            and sync_returncode == 0
            and not failed
            and remote_error is None
        )
        receipt.update(
            {
                "status": "PASS" if passed else "FAILED",
                "processes": results,
                "aoss_sync": {
                    "returncode": sync_returncode,
                    "early_exit": sync_early_failure,
                    "log": str(sync_log_path),
                },
                "completed_scenes_after": sorted(completed_after),
                "completed_units_after": sorted(completed_units_after),
                "remote_verification_error": remote_error,
            }
        )
        write_json_atomic(receipt_path, receipt)
        if not passed:
            fail(
                "canonical gap extraction/AOSS publication failed closed; "
                f"extractor_failures={len(failed)} sync_rc={sync_returncode} "
                f"remote_error={remote_error!r}; receipt={receipt_path}"
            )
    finally:
        for process, handle, _item, _path in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            if not handle.closed:
                handle.close()
        if sync_process.poll() is None:
            sync_process.terminate()
            sync_process.wait(timeout=30)
        if not sync_handle.closed:
            sync_handle.close()
        for sig, handler in old.items():
            signal.signal(sig, handler)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
    return {
        "dry_run": False,
        "shard_id": shard["shard_id"],
        "receipt": str(receipt_path),
        "remote_completion_receipt": str(state_paths["receipt"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--shard", type=Path)
    selection.add_argument("--batch", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--trust-current-tools",
        action="store_true",
        help=(
            "reuse an existing shard identity (receipt/worklist SHAs) with the "
            "on-disk runner/sync-loop even if the shard pins older tool SHAs"
        ),
    )
    args = parser.parse_args()
    if args.shard:
        value = run_shard(
            validate_shard(args.shard, trust_current_tools=args.trust_current_tools),
            execute=args.execute,
            trust_current_tools=args.trust_current_tools,
        )
    else:
        batch, shards = validate_batch(
            args.batch, trust_current_tools=args.trust_current_tools
        )
        value = {
            "dry_run": not args.execute, "batch_id": batch["batch_id"],
            "shards": [
                run_shard(
                    shard,
                    execute=args.execute,
                    trust_current_tools=args.trust_current_tools,
                )
                for shard in shards
            ],
        }
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
