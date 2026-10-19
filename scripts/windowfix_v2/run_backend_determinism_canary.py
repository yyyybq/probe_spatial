#!/usr/bin/env python3
"""Cross-process/order canary for blocked canonical-v2 backends.

The canary runs both streaming-prefix and isolated-target extraction twice in
fresh temporary roots.  Run A processes only the selected scene; run B first
processes earlier scenes, so the selected logical samples occur at a different
call position.  Every selected-scene safetensor, index, and deterministic-noise
sidecar must then match exactly.  Nothing is published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL_ID = "inscene_streaming_canonical_v2_obs28_h124"
STREAM_NAMESPACE = "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
TARGET_NAMESPACE = "FEAT_TARGET_CANONICAL_V2_SEEDED_V1"
CANARY_VFMS = {"aether", "f3r", "lingbot", "qwen2_5_vl"}
SAMPLE_LOCAL_VFMS = {"aether", "f3r", "lingbot"}
PREFIX_FRAME_COUNTS = {"8": 8, "12": 12, "16": 16, "24": 24}
GAP_NAMESPACES = {
    "streaming_prefix": "FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1",
    "target_isolated": "FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1",
}
REQUIRED_TARGET_TAIL_SHAPES = {
    "aether": [30, 45, 3072],
    "qwen2_5_vl": [12, 12, 3584],
}
# ``features.run_inscene15k`` intentionally owns resize selection through its
# VFM defaults; it does not expose a ``--resize`` CLI option.  A canary still
# binds the expected resize in its immutable spec, but must not pass an option
# that the extractor cannot parse.  Keep this table deliberately small and
# fail closed for a new backend until its main-extractor contract is reviewed.
MAIN_EXTRACTOR_FIXED_RESIZES = {
    "aether": (480, 720),
    "f3r": (288, 512),
    "lingbot": (480, 832),
}
HEX40_OR_64 = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
DTYPE_ALIASES = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
}
DTYPE_BYTES = {
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}
BACKEND_DEPENDENCY_FILES = {
    "aether": [
        "features/aether/aether_feature.py",
        "features/aether/extract_features.py",
        "features/aether/aether/pipelines/aetherv1_pipeline_cogvideox.py",
        "features/aether/aether/utils/preprocess_utils.py",
        "features/aether/aether/utils/postprocess_utils.py",
    ],
    "f3r": [
        "features/f3r/extract_features.py",
        "features/f3r/f3r/models/fast3r.py",
        "features/f3r/f3r/dust3r/inference_multiview.py",
    ],
    "lingbot": [
        "features/lingbot/lingbot_feature.py",
        "features/lingbot/extract_features.py",
    ],
    # The MLLM extractor imports canonical window construction from this file.
    "qwen2_5_vl": [
        "features/run_inscene15k.py",
        "features/offline_hf.py",
        "vidfm3d/utils/feature_layers.py",
        "vidfm3d/utils/temporal_sampling.py",
    ],
}
BACKEND_DEPENDENCY_TREES = {
    "aether": ["features/aether"],
    "f3r": ["features/f3r"],
    "lingbot": ["features/lingbot"],
    "qwen2_5_vl": [],
}
BACKEND_EXTERNAL_DEPENDENCY_TREES = {
    "aether": [],
    "f3r": [],
    "lingbot": [
        Path("/mnt/umm/users/yinbaiqiao/Consistency/lingbot-video/lingbot_video")
    ],
    "qwen2_5_vl": [],
}
BACKEND_MODEL_ARTIFACTS = {
    # Aether combines the CogVideoX tokenizer/text/VAE with its own transformer.
    "aether": ["THUDM/CogVideoX-5b-I2V", "AetherWorldModel/AetherV1"],
    "f3r": ["jedyang97/Fast3R_ViT_Large_512"],
    "lingbot": [
        "/mnt/umm/users/yinbaiqiao/Consistency/lingbot-video/models/"
        "lingbot-video-moe-30b-a3b"
    ],
    "qwen2_5_vl": ["Qwen/Qwen2.5-VL-7B-Instruct"],
}
CANARY_BASE_ENVIRONMENT = {
    "aether": {},
    "f3r": {},
    "lingbot": {
        "HF_HOME": "/mnt/umm/shared_model/huggingface",
        "DIFFUSERS_ATTN_BACKEND": "_native_flash",
        "LINGBOT_MOE_PAD_BACKEND": "vectorized",
        "LINGBOT_MOE_EXPERT_BACKEND": "grouped_mm",
        "PYTHONPATH": (
            "/mnt/umm/users/yinbaiqiao/Consistency/lingbot-video:"
            "/mnt/umm/users/yinbaiqiao/probe_spatial"
        ),
    },
    "qwen2_5_vl": {},
}
MODEL_CONFIG_CANDIDATES = (
    "config.json",
    "generation_config.json",
    "model_index.json",
    "modular_model_index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "processor/config.json",
    "refiner/config.json",
    "text_encoder/config.json",
    "transformer/config.json",
    "vae/config.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def read_regular_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain one JSON object: {path}")
    return value


def backend_dependency_bindings(
    vfm: str, project_root: Path
) -> dict[str, dict[str, str]]:
    """Hash every repository source that is part of a gated backend path."""
    if vfm not in BACKEND_DEPENDENCY_FILES:
        raise RuntimeError(f"no frozen canary dependency map for {vfm}")
    relatives = {"features/noise_seed.py", *BACKEND_DEPENDENCY_FILES[vfm]}
    for root_name in BACKEND_DEPENDENCY_TREES[vfm]:
        root = project_root / root_name
        if root.is_symlink() or not root.is_dir():
            raise RuntimeError(f"backend dependency tree must be regular: {root}")
        relatives.update(
            path.relative_to(project_root).as_posix()
            for path in root.rglob("*.py")
            if path.is_file() and not path.is_symlink()
        )
    result: dict[str, dict[str, str]] = {}
    for relative in sorted(relatives):
        path = project_root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"backend dependency must be regular: {path}")
        result[relative] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    for root in BACKEND_EXTERNAL_DEPENDENCY_TREES[vfm]:
        if root.is_symlink() or not root.is_dir():
            raise RuntimeError(f"external backend dependency tree must be regular: {root}")
        for path in sorted(root.rglob("*.py")):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"external backend dependency must be regular: {path}")
            relative = path.relative_to(root).as_posix()
            key = f"external:{root}:{relative}"
            result[key] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    return result


def _config_bindings(root: Path) -> dict[str, dict[str, str]]:
    """Bind small model/processor configs while allowing HF blob symlinks."""
    result: dict[str, dict[str, str]] = {}
    for relative in MODEL_CONFIG_CANDIDATES:
        logical = root / relative
        if not logical.exists():
            continue
        resolved = logical.resolve()
        if not resolved.is_file():
            continue
        result[relative] = {
            "logical_path": str(logical),
            "resolved_path": str(resolved),
            "sha256": sha256_file(resolved),
        }
    return result


def _hf_homes() -> list[Path]:
    candidates: list[Path] = []
    if os.environ.get("HF_HOME"):
        candidates.append(Path(os.environ["HF_HOME"]))
    candidates.extend(
        [
            Path.home() / ".cache/huggingface",
            Path("/mnt/umm/shared_model/huggingface"),
        ]
    )
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def model_artifact_binding(model_id: str) -> dict[str, Any]:
    """Resolve the exact local model config/revision used by a canary.

    Hub ids are pinned through the local ``refs/main`` value and its snapshot
    configs.  The generated canary runs offline with the same ``HF_HOME`` so a
    moving network branch cannot silently replace the bound revision.
    """
    model_root = Path(model_id)
    if model_root.is_absolute():
        if not model_root.exists():
            raise RuntimeError(f"local model root is missing: {model_root}")
        resolved_root = model_root.resolve()
        configs = _config_bindings(model_root)
        if not configs:
            raise RuntimeError(f"local model has no bindable config: {model_root}")
        revision = None
        parts = resolved_root.parts
        if "snapshots" in parts:
            index = parts.index("snapshots")
            if index + 1 < len(parts) and HEX40_OR_64.fullmatch(parts[index + 1]):
                revision = parts[index + 1]
        return {
            "model_id": model_id,
            "resolution": "local_path",
            "requested_revision": None,
            "resolved_revision": revision,
            "model_root": str(model_root),
            "resolved_model_root": str(resolved_root),
            "revision_ref": None,
            "config_files": configs,
            "required_environment": {},
        }

    cache_name = "models--" + model_id.replace("/", "--")
    for hf_home in _hf_homes():
        root = hf_home / "hub" / cache_name
        ref = root / "refs/main"
        if ref.is_symlink() or not ref.is_file():
            continue
        revision = ref.read_text(encoding="utf-8").strip()
        if not HEX40_OR_64.fullmatch(revision):
            continue
        snapshot = root / "snapshots" / revision
        if not snapshot.is_dir():
            continue
        configs = _config_bindings(snapshot)
        if not configs:
            continue
        return {
            "model_id": model_id,
            "resolution": "huggingface_cache_main_ref",
            "requested_revision": None,
            "resolved_revision": revision,
            "model_root": str(root),
            "resolved_model_root": str(snapshot.resolve()),
            "revision_ref": {
                "path": str(ref.resolve()),
                "sha256": sha256_file(ref),
                "value": revision,
            },
            "config_files": configs,
            "required_environment": {
                "HF_HOME": str(hf_home),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
        }
    raise RuntimeError(f"no revision/config-bound local Hugging Face snapshot for {model_id}")


def backend_provenance_binding(vfm: str, project_root: Path) -> dict[str, Any]:
    if vfm not in BACKEND_MODEL_ARTIFACTS:
        raise RuntimeError(f"no frozen canary model-artifact map for {vfm}")
    artifacts = {
        model_id: model_artifact_binding(model_id)
        for model_id in BACKEND_MODEL_ARTIFACTS[vfm]
    }
    required_environment: dict[str, str] = {}
    for binding in artifacts.values():
        for key, value in binding["required_environment"].items():
            previous = required_environment.setdefault(key, value)
            if previous != value:
                raise RuntimeError(
                    f"backend model artifacts require conflicting {key}: "
                    f"{previous!r} != {value!r}"
                )
    return {
        "backend_dependencies": backend_dependency_bindings(vfm, project_root),
        "model_artifacts": artifacts,
        "required_environment": required_environment,
    }


def canary_runtime_environment(vfm: str, provenance: dict[str, Any]) -> dict[str, str]:
    if vfm not in CANARY_BASE_ENVIRONMENT:
        raise RuntimeError(f"no frozen canary environment for {vfm}")
    result = dict(CANARY_BASE_ENVIRONMENT[vfm])
    required = provenance.get("required_environment")
    if not isinstance(required, dict):
        raise RuntimeError("backend provenance required_environment must be an object")
    for key, value in required.items():
        previous = result.setdefault(key, value)
        if previous != value:
            raise RuntimeError(
                f"canary environment conflicts with model provenance ({vfm}/{key})"
            )
    return result


def validate_backend_provenance(
    provenance: Any, *, vfm: str, project_root: Path
) -> dict[str, Any]:
    """Recompute and compare every source/config/revision binding."""
    if not isinstance(provenance, dict):
        raise RuntimeError("backend_provenance must be an object")
    expected = backend_provenance_binding(vfm, project_root)
    if provenance != expected:
        raise RuntimeError(f"backend provenance changed for {vfm}")
    for key, value in expected["required_environment"].items():
        if os.environ.get(key) != value:
            raise RuntimeError(
                f"backend provenance requires environment {key}={value!r}, "
                f"found {os.environ.get(key)!r}"
            )
    return expected


def normalize_dtype(value: Any) -> str:
    text = str(value).strip()
    text = text.removeprefix("torch.").removeprefix("numpy.").removeprefix("np.")
    return DTYPE_ALIASES.get(text.upper(), text.lower())


def safetensor_contract(path: Path, tensor_key: str = "feat") -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"safetensor must be regular: {path}")
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise RuntimeError(f"truncated safetensors header length: {path}")
        header_size = struct.unpack("<Q", raw_length)[0]
        if header_size <= 1 or header_size > min(size - 8, 128 * 1024 * 1024):
            raise RuntimeError(f"invalid safetensors header length: {path}")
        raw_header = handle.read(header_size)
    if len(raw_header) != header_size:
        raise RuntimeError(f"truncated safetensors header: {path}")
    header = json.loads(raw_header.decode("utf-8"))
    tensor = header.get(tensor_key)
    if not isinstance(tensor, dict):
        raise RuntimeError(f"missing tensor {tensor_key!r}: {path}")
    shape = tensor.get("shape")
    if (
        not isinstance(shape, list)
        or not shape
        or any(not isinstance(value, int) or value <= 0 for value in shape)
    ):
        raise RuntimeError(f"invalid tensor shape: {path}")
    dtype = normalize_dtype(tensor.get("dtype"))
    if dtype not in {"float16", "bfloat16", "float32", "float64"}:
        raise RuntimeError(f"unsupported canary tensor dtype {dtype!r}: {path}")
    offsets = tensor.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise RuntimeError(f"invalid tensor offsets: {path}")
    start, end = (int(offsets[0]), int(offsets[1]))
    payload_size = size - 8 - header_size
    elements = 1
    for dimension in shape:
        elements *= dimension
    if (
        start < 0
        or end < start
        or end > payload_size
        or end - start != elements * DTYPE_BYTES[dtype]
    ):
        raise RuntimeError(f"tensor offsets disagree with shape/dtype: {path}")
    return {
        "shape": list(shape),
        "dtype": dtype,
        "header_sha256": hashlib.sha256(raw_length + raw_header).hexdigest(),
    }


def _stream_prefix_lookup(scene: Path) -> dict[tuple[int, int], int]:
    index = scene / "prefix_index.npy"
    if index.is_symlink() or not index.is_file():
        raise RuntimeError(f"Qwen canary prefix index is missing: {index}")
    raw = np.load(index, allow_pickle=True).tolist()
    if not isinstance(raw, list):
        raise RuntimeError(f"Qwen canary prefix index is not a record list: {index}")
    result: dict[tuple[int, int], int] = {}
    for record in raw:
        if not isinstance(record, dict):
            raise RuntimeError(f"Qwen canary prefix record is invalid: {index}")
        key = (int(record["window_id"]), int(record["tail"]))
        prefix = int(record["valid_length"])
        if prefix not in {int(value) for value in PREFIX_FRAME_COUNTS}:
            raise RuntimeError(f"unexpected Qwen prefix p{prefix}: {index}")
        if key in result and result[key] != prefix:
            raise RuntimeError(f"ambiguous Qwen prefix lookup key {key}: {index}")
        result[key] = prefix
    return result


def inspect_qwen_streaming_scene(scene: Path) -> dict[str, dict[str, Any]]:
    """Return one internally consistent shape/dtype contract for every prefix."""
    lookup = _stream_prefix_lookup(scene)
    observations: dict[str, list[dict[str, Any]]] = {
        key: [] for key in PREFIX_FRAME_COUNTS
    }
    for path in sorted(scene.glob("window_*/prefix_*/*.sft")):
        try:
            window_id = int(path.parent.parent.name.removeprefix("window_"))
            tail = int(path.parent.name.removeprefix("prefix_"))
            prefix = lookup[(window_id, tail)]
        except (KeyError, ValueError) as exc:
            raise RuntimeError(f"cannot map Qwen tensor to a canonical prefix: {path}") from exc
        observations[str(prefix)].append(safetensor_contract(path))

    result: dict[str, dict[str, Any]] = {}
    for prefix in PREFIX_FRAME_COUNTS:
        values = observations[prefix]
        if not values:
            raise RuntimeError(f"Qwen canary emitted no p{prefix} streaming tensor")
        identities = {(tuple(value["shape"]), value["dtype"]) for value in values}
        if len(identities) != 1:
            raise RuntimeError(
                f"Qwen p{prefix} shape/dtype varies within one run: {sorted(identities)}"
            )
        shape, dtype = next(iter(identities))
        result[prefix] = {
            "shape": list(shape),
            "dtype": dtype,
            "tensors_checked": len(values),
            "header_set_sha256": canonical_json_sha(
                sorted(value["header_sha256"] for value in values)
            ),
        }
    return result


def qwen_streaming_tensor_contract(root_a: Path, root_b: Path) -> dict[str, Any]:
    scenes_a = scene_directories(root_a, "qwen2_5_vl")
    if len(scenes_a) != 1:
        raise RuntimeError(f"Qwen single run emitted {len(scenes_a)} scene dirs")
    scene_a = scenes_a[0]
    scene_b = root_b / "qwen2_5_vl" / "scannetpp" / scene_a.name
    left = inspect_qwen_streaming_scene(scene_a)
    right = inspect_qwen_streaming_scene(scene_b)
    for prefix in PREFIX_FRAME_COUNTS:
        left_identity = (left[prefix]["shape"], left[prefix]["dtype"])
        right_identity = (right[prefix]["shape"], right[prefix]["dtype"])
        if left_identity != right_identity:
            raise RuntimeError(
                f"Qwen p{prefix} shape/dtype differs across canary order: "
                f"{left_identity!r} != {right_identity!r}"
            )
    return {
        "schema_version": 1,
        "contract": "QWEN_STREAMING_PREFIX_TENSOR_CONTRACT_V1",
        "tensor_key": "feat",
        "all_prefixes_present": True,
        "same_prefix_shape_dtype_consistent": True,
        "single_ordered_shape_dtype_equal": True,
        "prefixes": {
            prefix: {
                "shape": left[prefix]["shape"],
                "dtype": left[prefix]["dtype"],
                "single_run_tensors_checked": left[prefix]["tensors_checked"],
                "ordered_run_tensors_checked": right[prefix]["tensors_checked"],
                "single_run_header_set_sha256": left[prefix]["header_set_sha256"],
                "ordered_run_header_set_sha256": right[prefix]["header_set_sha256"],
            }
            for prefix in PREFIX_FRAME_COUNTS
        },
    }


def validate_qwen_streaming_tensor_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("Qwen receipt lacks streaming tensor contract")
    expected_top = {
        "schema_version": 1,
        "contract": "QWEN_STREAMING_PREFIX_TENSOR_CONTRACT_V1",
        "tensor_key": "feat",
        "all_prefixes_present": True,
        "same_prefix_shape_dtype_consistent": True,
        "single_ordered_shape_dtype_equal": True,
    }
    bad = [key for key, expected in expected_top.items() if value.get(key) != expected]
    prefixes = value.get("prefixes")
    if bad or not isinstance(prefixes, dict) or set(prefixes) != set(PREFIX_FRAME_COUNTS):
        raise RuntimeError(f"invalid Qwen streaming tensor contract: {bad}")
    normalized: dict[str, dict[str, Any]] = {}
    for prefix in PREFIX_FRAME_COUNTS:
        row = prefixes[prefix]
        if not isinstance(row, dict):
            raise RuntimeError(f"invalid Qwen p{prefix} tensor contract")
        shape = row.get("shape")
        dtype = normalize_dtype(row.get("dtype"))
        if (
            not isinstance(shape, list)
            or not shape
            or any(not isinstance(item, int) or item <= 0 for item in shape)
            or dtype != "float16"
            or int(row.get("single_run_tensors_checked", 0)) <= 0
            or int(row.get("ordered_run_tensors_checked", 0)) <= 0
            or int(row.get("single_run_tensors_checked", 0))
            != int(row.get("ordered_run_tensors_checked", 0))
        ):
            raise RuntimeError(f"invalid Qwen p{prefix} shape/dtype evidence")
        for digest_key in (
            "single_run_header_set_sha256",
            "ordered_run_header_set_sha256",
        ):
            digest = row.get(digest_key)
            if not isinstance(digest, str) or len(digest) != 64:
                raise RuntimeError(f"invalid Qwen p{prefix} {digest_key}")
        if (
            row["single_run_header_set_sha256"]
            != row["ordered_run_header_set_sha256"]
        ):
            raise RuntimeError(f"Qwen p{prefix} header contracts differ across runs")
        normalized[prefix] = {"shape": list(shape), "dtype": dtype}
    return normalized


def inspect_target_isolated_scene(
    scene: Path,
    *,
    tensor_key: str,
    expected_tail_shape: list[int],
    expected_dtype: str,
) -> dict[str, Any]:
    """Read target safetensors headers and enforce their per-row contract."""
    observations = [
        safetensor_contract(path, tensor_key=tensor_key)
        for path in sorted(scene.rglob("*.sft"))
        if path.is_file() and not path.is_symlink()
    ]
    if not observations:
        raise RuntimeError(f"target canary emitted no safetensor: {scene}")
    if any(len(value["shape"]) < 2 for value in observations):
        raise RuntimeError("target safetensor has no leading row dimension")
    expected_identity = (tuple(expected_tail_shape), normalize_dtype(expected_dtype))
    identities = {
        (tuple(value["shape"][1:]), value["dtype"])
        for value in observations
        if len(value["shape"]) >= 2
    }
    if identities != {expected_identity} or len(identities) != 1:
        raise RuntimeError(
            "target row tail shape/dtype differs from merge-plan contract: "
            f"expected={expected_identity!r} observed={sorted(identities)!r}"
        )
    return {
        "observed_tail_shape": list(expected_identity[0]),
        "observed_dtype": expected_identity[1],
        "tensors_checked": len(observations),
        "header_set_sha256": canonical_json_sha(
            sorted(value["header_sha256"] for value in observations)
        ),
    }


def target_isolated_tensor_contract(
    root_a: Path,
    root_b: Path,
    *,
    vfm: str,
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Record target-row shape/dtype evidence for both canary call orders."""
    scenes_a = scene_directories(root_a, vfm)
    if len(scenes_a) != 1:
        raise RuntimeError(f"target single run emitted {len(scenes_a)} scene dirs")
    scene_a = scenes_a[0]
    scene_b = root_b / vfm / "scannetpp" / scene_a.name
    if not scene_b.is_dir():
        raise RuntimeError(f"target ordered run is missing selected scene: {scene_b}")
    kwargs = {
        "tensor_key": str(expected["tensor_key"]),
        "expected_tail_shape": [int(value) for value in expected["expected_tail_shape"]],
        "expected_dtype": str(expected["expected_dtype"]),
    }
    left = inspect_target_isolated_scene(scene_a, **kwargs)
    right = inspect_target_isolated_scene(scene_b, **kwargs)
    if (
        left["observed_tail_shape"] != right["observed_tail_shape"]
        or left["observed_dtype"] != right["observed_dtype"]
        or left["tensors_checked"] != right["tensors_checked"]
        or left["header_set_sha256"] != right["header_set_sha256"]
    ):
        raise RuntimeError("target tensor header contract differs across canary order")
    return {
        "schema_version": 1,
        "contract": "TARGET_ISOLATED_ROW_TENSOR_CONTRACT_V1",
        "tensor_key": kwargs["tensor_key"],
        "expected_tail_shape": kwargs["expected_tail_shape"],
        "expected_dtype": normalize_dtype(kwargs["expected_dtype"]),
        "observed_tail_shape": left["observed_tail_shape"],
        "observed_dtype": left["observed_dtype"],
        "single_run_tensors_checked": left["tensors_checked"],
        "ordered_run_tensors_checked": right["tensors_checked"],
        "single_run_header_set_sha256": left["header_set_sha256"],
        "ordered_run_header_set_sha256": right["header_set_sha256"],
    }


def validate_target_isolated_tensor_contract(
    value: Any,
    expected: Any,
) -> dict[str, Any]:
    """Fail closed unless target receipt evidence matches its bound plan unit."""
    if not isinstance(value, dict) or not isinstance(expected, dict):
        raise RuntimeError("target-bound canary lacks target tensor contract")
    expected_tail = [int(item) for item in expected.get("expected_tail_shape", [])]
    expected_dtype = normalize_dtype(expected.get("expected_dtype"))
    expected_top = {
        "schema_version": 1,
        "contract": "TARGET_ISOLATED_ROW_TENSOR_CONTRACT_V1",
        "tensor_key": expected.get("tensor_key"),
        "expected_tail_shape": expected_tail,
        "expected_dtype": expected_dtype,
        "observed_tail_shape": expected_tail,
        "observed_dtype": expected_dtype,
    }
    bad = [key for key, item in expected_top.items() if value.get(key) != item]
    if (
        bad
        or not expected_tail
        or any(item <= 0 for item in expected_tail)
        or expected_dtype != "float16"
        or int(value.get("single_run_tensors_checked", 0)) <= 0
        or int(value.get("single_run_tensors_checked", 0))
        != int(value.get("ordered_run_tensors_checked", 0))
    ):
        raise RuntimeError(f"invalid target tensor contract: {bad}")
    for key in ("single_run_header_set_sha256", "ordered_run_header_set_sha256"):
        digest = value.get(key)
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f"invalid target tensor contract {key}")
    if value["single_run_header_set_sha256"] != value["ordered_run_header_set_sha256"]:
        raise RuntimeError("target tensor header contracts differ across runs")
    return {"tail_shape": expected_tail, "dtype": expected_dtype}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def validate_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} is missing or symlinked: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA changed: expected={expected} actual={actual} path={path}"
        )


def validate_canary_spec(
    args: argparse.Namespace,
    *,
    extractor: Path,
    canary_script: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate the immutable execution spec and all of its live provenance."""
    spec_path = args.canary_spec.resolve()
    spec = read_regular_json(spec_path, "backend canary spec")
    stored_content_sha = spec.get("content_sha256")
    unhashed = dict(spec)
    unhashed.pop("content_sha256", None)
    if stored_content_sha != canonical_json_sha(unhashed):
        raise RuntimeError("backend canary spec content SHA mismatch")
    expected = {
        "schema_version": 2,
        "audit_type": "canonical_v2_backend_canary_execution_spec",
        "status": "READY_FOR_REVIEW_NOT_SUBMITTED",
        "submission_performed": False,
        "vfm": args.vfm,
        "model_id": args.model_id,
        "layer": args.layer,
        "timestep": args.timestep,
        "protocol_file": str(args.protocol),
        "protocol_file_sha256": args.protocol_file_sha256,
        "extractor": str(extractor),
        "extractor_sha256": args.extractor_sha256,
        "canary_runner": str(canary_script),
        "canary_runner_sha256": sha256_file(canary_script),
        "canary_renderer": str(
            args.project_root
            / "scripts/windowfix_v2/prepare_gap_backend_canary_jobs.py"
        ),
        "canary_renderer_sha256": sha256_file(
            args.project_root
            / "scripts/windowfix_v2/prepare_gap_backend_canary_jobs.py"
        ),
        "receipt": str(args.output.resolve()),
    }
    bad = [key for key, value in expected.items() if spec.get(key) != value]
    if bad:
        raise RuntimeError(f"backend canary spec runtime identity mismatch: {bad}")
    runtime = spec.get("runtime_contract")
    expected_runtime = {
        "prompt": args.prompt,
        "resize": list(args.resize) if args.resize is not None else None,
        "num_frames": args.num_frames,
        "python": str(args.python),
        "sample_index": args.sample_index,
    }
    if runtime != expected_runtime:
        raise RuntimeError("backend canary spec runtime contract mismatch")
    provenance = validate_backend_provenance(
        spec.get("backend_provenance"), vfm=args.vfm, project_root=args.project_root
    )
    runtime_environment = spec.get("runtime_environment")
    expected_environment = canary_runtime_environment(args.vfm, provenance)
    if runtime_environment != expected_environment:
        raise RuntimeError("backend canary spec runtime_environment contract mismatch")
    if any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or os.environ.get(key) != value
        for key, value in runtime_environment.items()
    ):
        raise RuntimeError("backend canary runtime environment differs from spec")
    for key, value in provenance["required_environment"].items():
        if runtime_environment.get(key) != value:
            raise RuntimeError(
                f"backend canary spec does not enforce model environment {key}"
            )
    return spec, {
        "path": str(spec_path),
        "file_sha256": sha256_file(spec_path),
        "content_sha256": str(stored_content_sha),
    }


def extractor_command(
    args: argparse.Namespace,
    *,
    mode: str,
    out_root: Path,
    start: int,
    end: int,
) -> list[str]:
    is_mllm = args.vfm == "qwen2_5_vl"
    command = [
        str(args.python),
        "-m",
        "features.run_inscene15k_mllm" if is_mllm else "features.run_inscene15k",
    ]
    if is_mllm:
        command.extend(["--backend", "qwen2_5_vl", "--vfm-name", args.vfm])
    else:
        command.extend(["--vfm", args.vfm])
    command.extend([
        "--mode",
        mode,
        "--data-root",
        str(args.dataset_root / "data"),
        "--out-root",
        str(out_root),
        "--source",
        "scannetpp",
        "--model-id",
        args.model_id,
        "--t",
        str(args.timestep),
        "--output-layers",
        str(args.layer),
        "--prefix-lengths",
        "8,12,16,24",
        "--prefix-max-len",
        "24",
        "--streaming-window-obs-len",
        "28",
        "--streaming-motion-step",
        "0.35",
        "--streaming-rotation-weight",
        "0.5",
        "--streaming-window-stride",
        "8",
        "--streaming-max-windows-per-scene",
        "4",
        "--start",
        str(start),
        "--end",
        str(end),
        "--prompt",
        (
            args.prompt
            if args.prompt is not None
            else ("Describe the spatial layout of this scene." if is_mllm else ".")
        ),
    ])
    if args.resize is not None:
        if is_mllm:
            # The MLLM extractor has an explicit resize interface.
            command.extend(["--resize", *[str(value) for value in args.resize]])
        else:
            # The main extractor does not accept ``--resize``.  Refuse a
            # canary whose immutable contract disagrees with its built-in VFM
            # default rather than silently exercising a different preprocess.
            expected_resize = MAIN_EXTRACTOR_FIXED_RESIZES.get(args.vfm)
            if expected_resize is None:
                raise RuntimeError(
                    f"no reviewed fixed-resize contract for main-extractor VFM {args.vfm!r}"
                )
            if tuple(int(value) for value in args.resize) != expected_resize:
                raise RuntimeError(
                    f"main-extractor VFM {args.vfm} fixes resize={expected_resize}; "
                    f"canary requested resize={tuple(args.resize)}"
                )
    if args.num_frames is not None:
        command.extend(["--num-frames", str(args.num_frames)])
    if mode == "target_isolated":
        command.extend(
            [
                "--target-from-streaming-windows",
                "--target-prefix-lengths",
                "8,12,16,24",
                "--target-horizons",
                "1,2,4",
            ]
        )
    return command


def run_once(
    args: argparse.Namespace,
    *,
    mode: str,
    out_root: Path,
    start: int,
    end: int,
    swap_first_two_scenes: bool = False,
) -> dict[str, Any]:
    namespace = out_root.name
    environment = dict(os.environ)
    environment["FEATURE_NAMESPACE"] = namespace
    environment.pop("EXTRACT_DONE_INDEX", None)
    environment.pop("EXTRACT_CACHE_CHECK_ONLY", None)
    command = extractor_command(
        args, mode=mode, out_root=out_root, start=start, end=end
    )
    if swap_first_two_scenes:
        # Canary only: exercise physical scene zero AFTER another scene, without
        # changing the production extractor or the frame order inside a scene.
        wrapper = (
            "import importlib,sys; name=sys.argv.pop(1); "
            "m=importlib.import_module(name); collect=m.collect_scannetpp_scenes; "
            "\ndef reordered(path):\n"
            " scenes=collect(path)\n"
            " assert len(scenes)>=2, 'order canary needs two scenes'\n"
            " return [scenes[1],scenes[0],*scenes[2:]]\n"
            "m.collect_scannetpp_scenes=reordered\n"
            "import torch; torch.set_float32_matmul_precision('high')\n"
            "m.main()\n"
        )
        command = [command[0], "-c", wrapper, command[2], *command[3:]]
    completed = subprocess.run(
        command,
        cwd=args.project_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-80:])
        raise RuntimeError(
            f"extractor failed rc={completed.returncode} mode={mode} "
            f"range=[{start},{end})\n{tail}"
        )
    return {
        "argv": command,
        "range": [start, end],
        "swap_first_two_scenes": swap_first_two_scenes,
        "log_tail": completed.stdout.splitlines()[-20:],
    }


def scene_directories(root: Path, vfm: str) -> list[Path]:
    base = root / vfm / "scannetpp"
    if not base.is_dir():
        raise RuntimeError(f"extractor produced no scene root: {base}")
    return sorted(path for path in base.iterdir() if path.is_dir())


def compare_selected_scene(
    *,
    root_a: Path,
    root_b: Path,
    vfm: str,
) -> dict[str, Any]:
    scenes_a = scene_directories(root_a, vfm)
    if len(scenes_a) != 1:
        raise RuntimeError(f"single-scene canary emitted {len(scenes_a)} scene dirs")
    selected = scenes_a[0]
    counterpart = root_b / vfm / "scannetpp" / selected.name
    if not counterpart.is_dir():
        raise RuntimeError(f"ordered run is missing selected scene: {counterpart}")

    relative_files = sorted(
        path.relative_to(selected)
        for path in selected.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    if not any(str(path).endswith(".sft") for path in relative_files):
        raise RuntimeError(f"selected scene contains no safetensor: {selected}")

    compared = []
    for relative in relative_files:
        left = selected / relative
        right = counterpart / relative
        if not right.is_file() or right.is_symlink():
            raise RuntimeError(f"ordered run is missing regular file: {right}")
        left_sha = sha256_file(left)
        right_sha = sha256_file(right)
        if left_sha != right_sha:
            raise RuntimeError(
                f"cross-order mismatch: {relative} left={left_sha} right={right_sha}"
            )
        if str(relative).endswith(".sft.manifest.json"):
            metadata = json.loads(left.read_text())
            if metadata.get("feature_generation_contract") != "inscene_deterministic_noise_v1":
                raise RuntimeError(f"missing deterministic provenance: {left}")
            if vfm in SAMPLE_LOCAL_VFMS:
                if (
                    metadata.get("noise_seed_scheme") != "inscene_noise_seed_v1"
                    or not metadata.get("noise_seed_samples")
                ):
                    raise RuntimeError(f"missing sample-local provenance: {left}")
            elif (
                metadata.get("noise_seed_scheme") != "deterministic_forward_no_noise_v1"
                or metadata.get("noise_seed_scope") != "no_random_draws"
                or "noise_seed_samples" in metadata
            ):
                raise RuntimeError(f"missing deterministic no-noise provenance: {left}")
        compared.append({"path": str(relative), "sha256": left_sha})
    return {
        "scene": selected.name,
        "files_compared": len(compared),
        "safetensors_compared": sum(item["path"].endswith(".sft") for item in compared),
        "comparison_sha256": hashlib.sha256(
            json.dumps(compared, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vfm", required=True, choices=sorted(CANARY_VFMS))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--timestep", required=True, type=int)
    parser.add_argument("--sample-index", type=int, default=1)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"),
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/mnt/umm/users/yinbaiqiao/InsScene-15K")
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-file-sha256", required=True)
    parser.add_argument("--extractor-sha256", required=True)
    parser.add_argument("--canary-spec", type=Path, required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--resize", nargs=2, type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--blocked-worklist", type=Path)
    parser.add_argument("--blocked-worklist-file-sha256")
    parser.add_argument("--blocked-worklist-content-sha256")
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--source-audit-sha256")
    parser.add_argument("--source-plan-summary", type=Path)
    parser.add_argument("--source-plan-summary-sha256")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def target_contract_from_source_audit(
    audit: dict[str, Any],
    *,
    args: argparse.Namespace,
    prompt_contract: dict[str, Any],
    preprocess_identity: dict[str, Any],
    canonical_source_sha256: str,
) -> dict[str, Any]:
    """Re-derive the target tensor contract from the audit-bound merge plan."""
    plan_path = Path(str(audit.get("plan", ""))).resolve()
    plan_sha = audit.get("plan_sha256")
    if not isinstance(plan_sha, str):
        raise RuntimeError("target source audit has no merge-plan SHA")
    validate_sha(plan_path, plan_sha, "target merge plan")
    plan = read_regular_json(plan_path, "target merge plan")
    unhashed = dict(plan)
    plan_content_sha = unhashed.pop("content_sha256", None)
    if plan_content_sha != canonical_json_sha(unhashed):
        raise RuntimeError("target merge plan content SHA mismatch")
    if (
        plan.get("contract") != "CANONICAL_V2_TARGET_ROW_MERGE_V1"
        or plan.get("protocol_file_sha256") != args.protocol_file_sha256
        or plan.get("canonical_source", {}).get("sha256")
        != canonical_source_sha256
    ):
        raise RuntimeError("target merge plan canonical identity mismatch")
    matching = [
        unit
        for unit in plan.get("units", [])
        if isinstance(unit, dict)
        and unit.get("vfm") == args.vfm
        and unit.get("model_id") == args.model_id
        and int(unit.get("timestep", -1)) == args.timestep
        and int(unit.get("layer", 10**9)) == args.layer
        and "target_isolated" in unit.get("accepted_modes", [])
        and unit.get("prompt_contract") == prompt_contract
        and unit.get("preprocess_identity") == preprocess_identity
    ]
    identities = {
        (
            str(unit.get("tensor_key")),
            tuple(int(value) for value in unit.get("expected_tail_shape", [])),
            normalize_dtype(unit.get("expected_dtype")),
        )
        for unit in matching
    }
    if not matching or len(identities) != 1:
        raise RuntimeError("target merge plan has no unique backend tensor contract")
    tensor_key, tail_shape, dtype = next(iter(identities))
    if (
        tensor_key != "feat"
        or not tail_shape
        or any(value <= 0 for value in tail_shape)
        or dtype != "float16"
        or (
            args.vfm in REQUIRED_TARGET_TAIL_SHAPES
            and list(tail_shape) != REQUIRED_TARGET_TAIL_SHAPES[args.vfm]
        )
    ):
        raise RuntimeError("target merge plan tensor contract is unsupported")
    return {
        "schema_version": 1,
        "contract": "CANONICAL_V2_TARGET_ROW_TENSOR_CONTRACT_V1",
        "source_target_merge_plan": str(plan_path),
        "source_target_merge_plan_file_sha256": plan_sha,
        "source_target_merge_plan_content_sha256": str(plan_content_sha),
        "tensor_key": tensor_key,
        "expected_tail_shape": list(tail_shape),
        "expected_dtype": dtype,
    }


def validate_gap_binding(
    args: argparse.Namespace,
    spec: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    fields = {
        "blocked_worklist": args.blocked_worklist,
        "blocked_worklist_file_sha256": args.blocked_worklist_file_sha256,
        "blocked_worklist_content_sha256": args.blocked_worklist_content_sha256,
        "source_audit": args.source_audit,
        "source_audit_sha256": args.source_audit_sha256,
        "source_plan_summary": args.source_plan_summary,
        "source_plan_summary_sha256": args.source_plan_summary_sha256,
    }
    present = {key for key, value in fields.items() if value is not None}
    if not present:
        return None
    if present != set(fields):
        missing = sorted(set(fields) - present)
        raise RuntimeError(f"partial gap canary binding; missing={missing}")
    if args.resize is None or args.num_frames is None or args.prompt is None:
        raise RuntimeError("gap-bound canary requires exact prompt/resize/num_frames")

    worklist_path = args.blocked_worklist.resolve()
    audit_path = args.source_audit.resolve()
    plan_summary_path = args.source_plan_summary.resolve()
    validate_sha(
        worklist_path, args.blocked_worklist_file_sha256, "blocked gap worklist"
    )
    validate_sha(audit_path, args.source_audit_sha256, "gap source audit")
    validate_sha(
        plan_summary_path, args.source_plan_summary_sha256, "gap source plan summary"
    )
    worklist = json.loads(worklist_path.read_text())
    stored_content = worklist.get("content_sha256")
    unhashed = dict(worklist)
    unhashed.pop("content_sha256", None)
    if (
        stored_content != canonical_json_sha(unhashed)
        or stored_content != args.blocked_worklist_content_sha256
    ):
        raise RuntimeError("blocked gap worklist content SHA mismatch")
    worklist_mode = worklist.get("mode")
    feature_namespace = worklist.get("feature_namespace")
    if (
        worklist.get("contract") != "CANONICAL_V2_GAP_WORKLIST_V1"
        or worklist_mode not in GAP_NAMESPACES
        or feature_namespace != GAP_NAMESPACES.get(worklist_mode)
        or worklist.get("vfm") != args.vfm
        or worklist.get("model_id") != args.model_id
        or int(worklist.get("timestep", -1)) != args.timestep
        or args.layer not in [int(value) for value in worklist.get("output_layers", [])]
        or worklist.get("protocol_file_sha256") != args.protocol_file_sha256
    ):
        raise RuntimeError("blocked gap worklist identity mismatch")
    prompt_contract = worklist.get("prompt_contract")
    expected_prompt = (
        prompt_contract.get("value")
        if isinstance(prompt_contract, dict) and prompt_contract.get("mode") == "exact"
        else ""
    )
    if expected_prompt != args.prompt:
        raise RuntimeError("blocked gap worklist prompt mismatch")
    preprocess = worklist.get("preprocess_identity")
    if worklist_mode == "streaming_prefix":
        expected_preprocess = {
            "frame_index_schema": "image_mask_intersection_v1",
            "resize": [int(value) for value in args.resize],
            "temporal_padding": (
                "none" if args.vfm == "qwen2_5_vl" else "repeat_tail"
            ),
        }
        if args.vfm == "qwen2_5_vl":
            expected_preprocess["num_frames_by_prefix"] = dict(PREFIX_FRAME_COUNTS)
            if int(args.num_frames) != max(PREFIX_FRAME_COUNTS.values()):
                raise RuntimeError("Qwen canary CLI frame bound must cover p24")
        else:
            expected_preprocess["num_frames"] = int(args.num_frames)
    else:
        expected_preprocess = {
            "resize": [int(value) for value in args.resize],
            "num_frames": int(args.num_frames),
        }
    if preprocess != expected_preprocess:
        raise RuntimeError("blocked gap worklist preprocessing mismatch")

    audit = json.loads(audit_path.read_text())
    audit_unhashed = dict(audit)
    audit_content = audit_unhashed.pop("content_sha256", None)
    if audit_content != canonical_json_sha(audit_unhashed):
        raise RuntimeError("gap source audit content SHA mismatch")
    bindings = audit.get("outputs", {}).get("gap_worklists", [])
    if not any(
        Path(str(item.get("path", ""))).resolve() == worklist_path
        and item.get("file_sha256") == args.blocked_worklist_file_sha256
        and item.get("content_sha256") == args.blocked_worklist_content_sha256
        for item in bindings
        if isinstance(item, dict)
    ):
        raise RuntimeError("gap source audit does not bind blocked worklist")
    if audit.get("canonical_source", {}).get("sha256") != worklist.get(
        "canonical_source_sha256"
    ):
        raise RuntimeError("gap source audit canonical-source mismatch")
    if spec is not None:
        expected_spec_gap = {
            "blocked_mode": worklist_mode,
            "feature_namespace": feature_namespace,
            "blocked_worklist": str(worklist_path),
            "blocked_worklist_file_sha256": args.blocked_worklist_file_sha256,
            "blocked_worklist_content_sha256": args.blocked_worklist_content_sha256,
            "source_audit": str(audit_path),
            "source_audit_sha256": args.source_audit_sha256,
            "prompt_contract": prompt_contract,
            "preprocess_identity": preprocess,
        }
        bad = [
            key for key, value in expected_spec_gap.items() if spec.get(key) != value
        ]
        if bad:
            raise RuntimeError(f"canary spec/gap worklist identity mismatch: {bad}")

    target_contract = None
    if worklist_mode == "target_isolated":
        target_contract = target_contract_from_source_audit(
            audit,
            args=args,
            prompt_contract=prompt_contract,
            preprocess_identity=preprocess,
            canonical_source_sha256=worklist["canonical_source_sha256"],
        )
        if spec is not None and spec.get("target_tensor_contract") != target_contract:
            raise RuntimeError("canary spec target tensor contract mismatch")
    elif spec is not None and spec.get("target_tensor_contract") is not None:
        raise RuntimeError("streaming canary spec carries a target tensor contract")

    plan = json.loads(plan_summary_path.read_text())
    if (
        plan.get("status") != "BLOCKED_PARTIAL"
        or plan.get("submission_performed") is not False
        or plan.get("protocol_file_sha256") != args.protocol_file_sha256
        or plan.get("canonical_source_sha256") != worklist.get("canonical_source_sha256")
    ):
        raise RuntimeError("gap source plan identity mismatch")
    if not any(
        Path(str(item.get("path", ""))).resolve() == audit_path
        and item.get("sha256") == args.source_audit_sha256
        for item in plan.get("source_audits", [])
        if isinstance(item, dict)
    ):
        raise RuntimeError("gap source plan does not bind source audit")
    if not any(
        item.get("vfm") == args.vfm
        and item.get("mode") == worklist_mode
        and item.get("model_id") == args.model_id
        and int(item.get("timestep", -1)) == args.timestep
        and Path(str(item.get("worklist_path", ""))).resolve() == worklist_path
        and item.get("worklist_file_sha256") == args.blocked_worklist_file_sha256
        and args.layer in [int(value) for value in item.get("layers", [])]
        and item.get("reason") == "BLOCKED_BACKEND_CANARY_NOT_BOUND"
        for item in plan.get("blocked", [])
        if isinstance(item, dict)
    ):
        raise RuntimeError("gap source plan does not bind blocked backend worklist")
    return {
        "blocked_worklist": str(worklist_path),
        "blocked_worklist_file_sha256": args.blocked_worklist_file_sha256,
        "blocked_worklist_content_sha256": args.blocked_worklist_content_sha256,
        "source_audit": str(audit_path),
        "source_audit_sha256": args.source_audit_sha256,
        "source_audit_content_sha256": audit_content,
        "source_plan_summary": str(plan_summary_path),
        "source_plan_summary_sha256": args.source_plan_summary_sha256,
        "canonical_source_sha256": worklist["canonical_source_sha256"],
        "worklist_mode": worklist_mode,
        "feature_namespace": feature_namespace,
        "prompt_contract": prompt_contract,
        "preprocess_identity": preprocess,
        "target_tensor_contract": target_contract,
    }


def main() -> int:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.protocol = args.protocol.resolve()
    extractor = args.project_root / (
        "features/run_inscene15k_mllm.py"
        if args.vfm == "qwen2_5_vl"
        else "features/run_inscene15k.py"
    )
    canary_script = Path(__file__).resolve()
    validate_sha(args.protocol, args.protocol_file_sha256, "protocol")
    validate_sha(extractor, args.extractor_sha256, "extractor")
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("canonical_protocol") != PROTOCOL_ID:
        raise RuntimeError("bound protocol is not canonical-v2")
    if args.sample_index < 0:
        raise RuntimeError("--sample-index must be >=0")
    spec, spec_binding = validate_canary_spec(
        args, extractor=extractor, canary_script=canary_script
    )
    gap_binding = validate_gap_binding(args, spec)

    modes = [
        ("streaming_prefix", STREAM_NAMESPACE),
        ("target_isolated", TARGET_NAMESPACE),
    ]
    results: dict[str, Any] = {}
    qwen_tensor_contract: dict[str, Any] | None = None
    target_tensor_evidence: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix=f"windowfix-v2-{args.vfm}-") as temporary:
        base = Path(temporary)
        for mode, namespace in modes:
            root_a = base / f"{mode}-single" / namespace
            root_b = base / f"{mode}-ordered" / namespace
            run_a = run_once(
                args,
                mode=mode,
                out_root=root_a,
                start=args.sample_index,
                end=args.sample_index + 1,
            )
            run_b = run_once(
                args,
                mode=mode,
                out_root=root_b,
                start=0,
                end=max(2, args.sample_index + 1),
                swap_first_two_scenes=args.sample_index == 0,
            )
            comparison = compare_selected_scene(
                root_a=root_a, root_b=root_b, vfm=args.vfm
            )
            results[mode] = {
                "single_run": run_a,
                "ordered_run": run_b,
                "comparison": comparison,
            }
            if args.vfm == "qwen2_5_vl" and mode == "streaming_prefix":
                qwen_tensor_contract = qwen_streaming_tensor_contract(root_a, root_b)
            if (
                mode == "target_isolated"
                and gap_binding is not None
                and gap_binding.get("worklist_mode") == "target_isolated"
            ):
                target_tensor_evidence = target_isolated_tensor_contract(
                    root_a,
                    root_b,
                    vfm=args.vfm,
                    expected=gap_binding["target_tensor_contract"],
                )

    if gap_binding is not None:
        selected_scene = results[gap_binding["worklist_mode"]]["comparison"]["scene"]
        worklist = json.loads(Path(gap_binding["blocked_worklist"]).read_text())
        if not any(
            entry.get("scene") == selected_scene
            and args.layer in [int(value) for value in entry.get("layers", [])]
            for entry in worklist.get("entries", [])
            if isinstance(entry, dict)
        ):
            raise RuntimeError(
                f"canary selected scene/layer is absent from blocked worklist: "
                f"{selected_scene}/layer{args.layer}"
            )
        gap_binding["selected_scene"] = selected_scene
        gap_binding["selected_scene_layer_covered"] = True

    # Close the common model-ref/config TOCTOU window: a receipt is emitted
    # only if every bound source and model artifact still matches after both
    # extraction modes and both call orders have completed.
    validate_backend_provenance(
        spec["backend_provenance"], vfm=args.vfm, project_root=args.project_root
    )

    receipt = {
        "schema_version": 2,
        "audit_type": "canonical_v2_backend_cross_process_order_canary",
        "status": "pass",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submission_performed": False,
        "vfm": args.vfm,
        "model_id": args.model_id,
        "layer": args.layer,
        "timestep": args.timestep,
        "sample_index": args.sample_index,
        "protocol": PROTOCOL_ID,
        "protocol_file": str(args.protocol),
        "protocol_file_sha256": args.protocol_file_sha256,
        "extractor": str(extractor),
        "extractor_sha256": args.extractor_sha256,
        "canary_script": str(canary_script),
        "canary_script_sha256": sha256_file(canary_script),
        "canary_spec_binding": spec_binding,
        "backend_provenance": spec["backend_provenance"],
        "backend_provenance_revalidated_after_runs": True,
        "runtime_environment": spec["runtime_environment"],
        "runtime_prompt": (
            args.prompt
            if args.prompt is not None
            else (
                "Describe the spatial layout of this scene."
                if args.vfm == "qwen2_5_vl"
                else "."
            )
        ),
        "runtime_resize": list(args.resize) if args.resize is not None else None,
        "runtime_num_frames": args.num_frames,
        "gap_binding": gap_binding,
        "results": results,
    }
    if args.vfm == "qwen2_5_vl":
        if qwen_tensor_contract is None:  # pragma: no cover - loop invariant
            raise RuntimeError("Qwen canary did not produce a streaming tensor contract")
        validate_qwen_streaming_tensor_contract(qwen_tensor_contract)
        receipt["qwen_streaming_tensor_contract"] = qwen_tensor_contract
    if gap_binding is not None and gap_binding["worklist_mode"] == "target_isolated":
        if target_tensor_evidence is None:  # pragma: no cover - loop invariant
            raise RuntimeError("target-bound canary produced no target tensor evidence")
        validate_target_isolated_tensor_contract(
            target_tensor_evidence,
            gap_binding["target_tensor_contract"],
        )
        receipt["target_isolated_tensor_contract"] = target_tensor_evidence
    atomic_json(args.output.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
