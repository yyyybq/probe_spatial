"""Fail-closed canonical-v2 record filters for gap-only extraction.

The normal InsScene extractor discovers every canonical record in a scene.  A
reference-overlay repair must not recompute records that are served by an exact
legacy route.  This module binds an immutable worklist and exposes only the
``(scene, window, prefix, layer)`` or ``(scene, raw frame, layer)`` items that
the extractor is allowed to generate.

The worklist is intentionally a positive allow-list.  An absent item is never
interpreted as work.  This prevents a truncated route audit from silently
turning into a full-cache extraction campaign.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CONTRACT = "CANONICAL_V2_GAP_WORKLIST_V1"
CANONICAL_PROTOCOL = "inscene_streaming_canonical_v2_obs28_h124"
CANONICAL_PREFIXES = frozenset({8, 12, 16, 24})
MODES = frozenset({"streaming_prefix", "target_isolated"})
GAP_NAMESPACES = {
    "streaming_prefix": "FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1",
    "target_isolated": "FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1",
}
STREAM_SEMANTIC_IDENTITY = "CANONICAL_V2_STREAM_UNIT_SEMANTIC_V1"
COMPLETED_SCENES_CONTRACT = "CANONICAL_V2_GAP_COMPLETED_SCENES_V1"
LOCAL_READY_SCENES_CONTRACT = "CANONICAL_V2_GAP_LOCAL_READY_SCENES_V1"
COMPLETED_UNITS_CONTRACT = "CANONICAL_V2_GAP_COMPLETED_UNITS_V1"
LOCAL_READY_UNITS_CONTRACT = "CANONICAL_V2_GAP_LOCAL_READY_UNITS_V1"
FRAME_INDEX_SCHEMA = "image_mask_intersection_v1"
CANONICAL_PREFIX_FRAME_COUNTS = {"8": 8, "12": 12, "16": 16, "24": 24}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _scene_set_from_binding(
    *,
    path_env: str,
    sha_env: str,
    expected_worklist_sha256: str,
    contract: str,
    content_field: str,
    label: str,
) -> frozenset[str]:
    """Load one SHA-bound extractor-only scene exclusion set.

    ``COMPLETED`` sets are backed by a verified remote receipt.  ``LOCAL_READY``
    sets are intentionally narrower: they only prevent the current extractor
    process from overwriting a fully validated local scene while the sync loop
    publishes it.  A local-ready entry is never a resume authority; only the
    remote completion receipt is used on a later invocation.
    """

    raw_path = os.environ.get(path_env, "")
    expected_file_sha = os.environ.get(sha_env, "")
    if not raw_path and not expected_file_sha:
        return frozenset()
    if not raw_path or not _SHA256.fullmatch(expected_file_sha):
        raise GapWorklistError(f"{label} file/path SHA binding is incomplete")
    path = Path(raw_path)
    if path.is_symlink() or not path.is_file():
        raise GapWorklistError(f"{label} file must be regular and non-symlink: {path}")
    if sha256_file(path) != expected_file_sha:
        raise GapWorklistError(f"{label} file SHA-256 mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GapWorklistError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise GapWorklistError(f"{label} file must contain one JSON object")
    content_sha = payload.get("content_sha256")
    unhashed = dict(payload)
    unhashed.pop("content_sha256", None)
    if (
        payload.get("schema_version") != 1
        or payload.get("contract") != contract
        or payload.get("worklist_file_sha256") != expected_worklist_sha256
        or not _SHA256.fullmatch(str(payload.get(content_field)))
        or not _SHA256.fullmatch(str(content_sha))
        or canonical_json_sha(unhashed) != content_sha
    ):
        raise GapWorklistError(f"{label} file identity/content mismatch")
    scenes = payload.get("scenes")
    if (
        not isinstance(scenes, list)
        or any(not isinstance(scene, str) for scene in scenes)
        or scenes != sorted(set(scenes))
    ):
        raise GapWorklistError(f"{label} scenes must be sorted and unique")
    result = frozenset(str(scene) for scene in scenes)
    if any(not scene or "/" in scene or scene in {".", ".."} for scene in result):
        raise GapWorklistError(f"{label} contains an invalid scene")
    return result


def _unit_set_from_binding(
    *,
    path_env: str,
    sha_env: str,
    expected_worklist_sha256: str,
    contract: str,
    content_field: str,
    label: str,
) -> frozenset[str]:
    raw_path = os.environ.get(path_env, "")
    expected_file_sha = os.environ.get(sha_env, "")
    if not raw_path and not expected_file_sha:
        return frozenset()
    if not raw_path or not _SHA256.fullmatch(expected_file_sha):
        raise GapWorklistError(f"{label} file/path SHA binding is incomplete")
    path = Path(raw_path)
    if path.is_symlink() or not path.is_file():
        raise GapWorklistError(f"{label} file must be regular and non-symlink: {path}")
    if sha256_file(path) != expected_file_sha:
        raise GapWorklistError(f"{label} file SHA-256 mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GapWorklistError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise GapWorklistError(f"{label} file must contain one JSON object")
    content_sha = payload.get("content_sha256")
    unhashed = dict(payload)
    unhashed.pop("content_sha256", None)
    if (
        payload.get("schema_version") != 1
        or payload.get("contract") != contract
        or payload.get("worklist_file_sha256") != expected_worklist_sha256
        or not _SHA256.fullmatch(str(payload.get(content_field)))
        or not _SHA256.fullmatch(str(content_sha))
        or canonical_json_sha(unhashed) != content_sha
    ):
        raise GapWorklistError(f"{label} file identity/content mismatch")
    units = payload.get("units")
    if (
        not isinstance(units, list)
        or any(not isinstance(unit, str) for unit in units)
        or units != sorted(set(units))
    ):
        raise GapWorklistError(f"{label} units must be sorted and unique")
    result = frozenset(str(unit) for unit in units)
    if any(not unit or unit.startswith("/") or ".." in Path(unit).parts for unit in result):
        raise GapWorklistError(f"{label} contains an invalid unit")
    return result


def _completion_exclusions(expected_worklist_sha256: str) -> frozenset[str]:
    """Load the runner-authored, SHA-bound remote-completion exclusion set.

    The normal/default path has no exclusions.  The gap runner opts in after
    it has verified an AOSS completion receipt; this keeps restart decisions
    independent of deleted local payloads without changing extractor CLIs.
    """

    return _scene_set_from_binding(
        path_env="CANONICAL_GAP_COMPLETED_SCENES_FILE",
        sha_env="CANONICAL_GAP_COMPLETED_SCENES_FILE_SHA256",
        expected_worklist_sha256=expected_worklist_sha256,
        contract=COMPLETED_SCENES_CONTRACT,
        content_field="completion_receipt_content_sha256",
        label="completed-scenes",
    )


def _local_ready_exclusions(expected_worklist_sha256: str) -> frozenset[str]:
    return _scene_set_from_binding(
        path_env="CANONICAL_GAP_LOCAL_READY_SCENES_FILE",
        sha_env="CANONICAL_GAP_LOCAL_READY_SCENES_FILE_SHA256",
        expected_worklist_sha256=expected_worklist_sha256,
        contract=LOCAL_READY_SCENES_CONTRACT,
        content_field="local_snapshot_content_sha256",
        label="local-ready-scenes",
    )


def _completion_unit_exclusions(expected_worklist_sha256: str) -> frozenset[str]:
    return _unit_set_from_binding(
        path_env="CANONICAL_GAP_COMPLETED_UNITS_FILE",
        sha_env="CANONICAL_GAP_COMPLETED_UNITS_FILE_SHA256",
        expected_worklist_sha256=expected_worklist_sha256,
        contract=COMPLETED_UNITS_CONTRACT,
        content_field="completion_receipt_content_sha256",
        label="completed-units",
    )


def _local_ready_unit_exclusions(expected_worklist_sha256: str) -> frozenset[str]:
    return _unit_set_from_binding(
        path_env="CANONICAL_GAP_LOCAL_READY_UNITS_FILE",
        sha_env="CANONICAL_GAP_LOCAL_READY_UNITS_FILE_SHA256",
        expected_worklist_sha256=expected_worklist_sha256,
        contract=LOCAL_READY_UNITS_CONTRACT,
        content_field="local_snapshot_content_sha256",
        label="local-ready-units",
    )


def streaming_gap_unit_key(scene: str, window_id: int, prefix: int, layer: int) -> str:
    return (
        f"streaming_prefix/{scene}/window_{int(window_id):04d}/"
        f"prefix_{int(prefix):02d}/layer_{int(layer)}"
    )


def target_gap_unit_key(scene: str, frame_index: int, layer: int) -> str:
    return f"target_isolated/{scene}/frame_{int(frame_index):06d}/layer_{int(layer)}"


class GapWorklistError(ValueError):
    """Raised when a gap worklist is incomplete, ambiguous, or mis-bound."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def streaming_worklist_preprocess_identity(
    vfm: str,
    resize: Iterable[int],
    num_frames: int,
) -> dict[str, Any]:
    """Return the exact top-level preprocessing contract for a stream worklist.

    Qwen consumes the selected prefix without temporal padding, so a worklist
    spanning all four prefixes cannot truthfully carry one scalar frame count.
    Fixed-length backends repeat the final selected frame to their model input
    length.
    """
    identity: dict[str, Any] = {
        "frame_index_schema": FRAME_INDEX_SCHEMA,
        "resize": [int(value) for value in resize],
        "temporal_padding": "none" if str(vfm) == "qwen2_5_vl" else "repeat_tail",
    }
    if str(vfm) == "qwen2_5_vl":
        identity["num_frames_by_prefix"] = dict(CANONICAL_PREFIX_FRAME_COUNTS)
    else:
        identity["num_frames"] = int(num_frames)
    return identity


def stream_unit_preprocess_metadata(
    worklist: "CanonicalGapWorklist",
    record: dict[str, Any],
) -> dict[str, Any]:
    """Resolve a top-level stream contract to one numeric payload identity."""
    prefix = int(record["valid_length"])
    raw_ids = [int(value) for value in record["indices"]]
    contract = worklist.preprocess_identity
    if worklist.vfm == "qwen2_5_vl":
        by_prefix = contract.get("num_frames_by_prefix")
        if not isinstance(by_prefix, dict) or str(prefix) not in by_prefix:
            raise GapWorklistError(f"Qwen preprocess contract has no p{prefix} frame count")
        num_frames = _integer(by_prefix[str(prefix)], "num_frames_by_prefix")
        if num_frames != prefix or len(raw_ids) != prefix:
            raise GapWorklistError(
                "Qwen no-pad unit frame count must equal prefix and raw-ID count"
            )
    else:
        num_frames = _integer(contract.get("num_frames"), "preprocess_identity.num_frames")
        if len(raw_ids) > num_frames and worklist.vfm != "videomae":
            raise GapWorklistError("stream unit exceeds fixed backend input length")
    return {
        "frame_index_schema": FRAME_INDEX_SCHEMA,
        "resize": list(contract["resize"]),
        # ``num_frames`` is the canonical remote/local gate field.  The
        # explicit alias makes the per-unit runtime meaning unambiguous.
        "num_frames": int(num_frames),
        "input_num_frames": int(num_frames),
        "temporal_padding": str(contract["temporal_padding"]),
    }


def stream_unit_semantic_metadata(
    worklist: "CanonicalGapWorklist",
    scene: str,
    record: dict[str, Any],
    layer: int,
) -> dict[str, Any]:
    """Stable identity of one physical streaming payload.

    Deliberately excludes the global worklist SHA and the set of sibling
    layers requested in one campaign.  Those values may change when a later
    remote gate adds disjoint work without changing this payload.
    """
    raw_ids = [int(value) for value in record["indices"]]
    return {
        "gap_contract": CONTRACT,
        "gap_semantic_identity": STREAM_SEMANTIC_IDENTITY,
        "canonical_source_sha256": worklist.canonical_source_sha256,
        "canonical_protocol_sha256": worklist.protocol_sha256,
        "layer": int(layer),
        "gap_record_key": {
            "scene": str(scene),
            "window_id": int(record["window_id"]),
            "prefix": int(record["valid_length"]),
        },
        "gap_record_raw_ids_sha256": hashlib.sha256(
            json.dumps(raw_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        **stream_unit_preprocess_metadata(worklist, record),
    }


def stream_unit_producer_metadata(
    worklist: "CanonicalGapWorklist",
) -> dict[str, Any]:
    """Audit provenance for the campaign that produced a semantic unit."""
    return {
        "produced_by_gap_worklist_file_sha256": worklist.file_sha256,
        "produced_by_gap_worklist_content_sha256": worklist.content_sha256,
    }


def static_prefix_index_metadata(
    worklist: "CanonicalGapWorklist",
) -> dict[str, Any]:
    """Worklist-independent provenance for the full canonical prefix index."""
    return {
        "gap_index_contract": "CANONICAL_V2_STATIC_PREFIX_INDEX_V1",
        "canonical_source_sha256": worklist.canonical_source_sha256,
    }


def stream_unit_semantic_matches(
    metadata: Any,
    expected: dict[str, Any],
) -> bool:
    """Match only stable per-unit fields; producer provenance is informative."""
    return isinstance(metadata, dict) and all(
        metadata.get(key) == value for key, value in expected.items()
    )


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise GapWorklistError(f"{label} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise GapWorklistError(f"{label} must be an integer") from exc


def _layers(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise GapWorklistError(f"{label} must be a non-empty layer list")
    layers = tuple(sorted({_integer(item, label) for item in value}))
    if len(layers) != len(value):
        raise GapWorklistError(f"{label} contains duplicate layers")
    return layers


@dataclass(frozen=True)
class CanonicalGapWorklist:
    """Validated positive allow-list used by one physical extraction group."""

    path: Path
    file_sha256: str
    content_sha256: str
    mode: str
    vfm: str
    model_id: str
    timestep: int
    feature_namespace: str
    output_layers: tuple[int, ...]
    protocol_sha256: str
    protocol_file_sha256: str
    canonical_source_sha256: str
    prompt_contract: dict[str, Any]
    preprocess_identity: dict[str, Any]
    _stream: dict[tuple[str, int, int], frozenset[int]]
    _target: dict[tuple[str, int], frozenset[int]]
    _completed_scenes: frozenset[str]
    _completed_units: frozenset[str]
    reason_counts: dict[str, int]

    @property
    def scenes(self) -> frozenset[str]:
        if self.mode == "streaming_prefix":
            return frozenset(
                scene
                for scene, window_id, prefix in self._stream
                if self.stream_layers(scene, window_id, prefix)
            )
        return frozenset(
            scene
            for scene, frame_index in self._target
            if self.target_layers(scene, frame_index)
        )

    @property
    def entry_count(self) -> int:
        return len(self._stream) + len(self._target)

    @property
    def layer_item_count(self) -> int:
        return sum(len(value) for value in self._stream.values()) + sum(
            len(value) for value in self._target.values()
        )

    def stream_layers(self, scene: str, window_id: int, prefix: int) -> frozenset[int]:
        if self.mode != "streaming_prefix":
            raise GapWorklistError("stream_layers called for a target worklist")
        if str(scene) in self._completed_scenes:
            return frozenset()
        layers = self._stream.get((str(scene), int(window_id), int(prefix)), frozenset())
        return frozenset(
            layer
            for layer in layers
            if streaming_gap_unit_key(scene, window_id, prefix, layer)
            not in self._completed_units
        )

    def target_layers(self, scene: str, frame_index: int) -> frozenset[int]:
        if self.mode != "target_isolated":
            raise GapWorklistError("target_layers called for a streaming worklist")
        if str(scene) in self._completed_scenes:
            return frozenset()
        layers = self._target.get((str(scene), int(frame_index)), frozenset())
        return frozenset(
            layer
            for layer in layers
            if target_gap_unit_key(scene, frame_index, layer) not in self._completed_units
        )

    def target_frame_indices(self, scene: str) -> tuple[int, ...]:
        if self.mode != "target_isolated":
            raise GapWorklistError("target_frame_indices called for a streaming worklist")
        if str(scene) in self._completed_scenes:
            return ()
        return tuple(sorted(key[1] for key in self._target if key[0] == str(scene)))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
        mode: str,
        vfm: str,
        model_id: str,
        timestep: int,
        feature_namespace: str,
        output_layers: Iterable[int],
        protocol_sha256: str,
        actual_prompt: str,
        resize: Iterable[int] | None = None,
        num_frames: int | None = None,
        preprocess_identity: dict[str, Any] | None = None,
    ) -> "CanonicalGapWorklist":
        raw_source = Path(path)
        if raw_source.is_symlink() or not raw_source.is_file():
            raise GapWorklistError(
                f"gap worklist must be a regular non-symlink file: {raw_source}"
            )
        source = raw_source.resolve()
        if source.is_symlink() or not source.is_file():
            raise GapWorklistError(f"resolved gap worklist is not regular: {source}")
        if not _SHA256.fullmatch(str(expected_sha256)):
            raise GapWorklistError("--gap-worklist-sha256 must be a lowercase SHA-256")
        file_sha = sha256_file(source)
        if file_sha != expected_sha256:
            raise GapWorklistError("gap worklist file SHA-256 mismatch")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except Exception as exc:
            raise GapWorklistError(f"invalid gap worklist JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise GapWorklistError("gap worklist must contain one JSON object")
        content_sha = payload.get("content_sha256")
        unhashed = dict(payload)
        unhashed.pop("content_sha256", None)
        if not _SHA256.fullmatch(str(content_sha)) or canonical_json_sha(unhashed) != content_sha:
            raise GapWorklistError("gap worklist content_sha256 mismatch")

        if preprocess_identity is None:
            if resize is None or num_frames is None:
                raise GapWorklistError(
                    "runtime preprocess identity or resize/num_frames is required"
                )
            if mode == "streaming_prefix":
                runtime_preprocess = streaming_worklist_preprocess_identity(
                    vfm, resize, int(num_frames)
                )
            else:
                # Target-delta worklists retain their fixed row-extraction
                # contract; they do not share streaming prefix padding.
                runtime_preprocess = {
                    "resize": [int(value) for value in resize],
                    "num_frames": int(num_frames),
                }
        else:
            if not isinstance(preprocess_identity, dict):
                raise GapWorklistError("runtime preprocess_identity must be an object")
            runtime_preprocess = dict(preprocess_identity)
        if mode == "streaming_prefix":
            if resize is None:
                resize = runtime_preprocess.get("resize")
            if num_frames is None:
                num_frames = runtime_preprocess.get("num_frames", 1)
            canonical_preprocess = streaming_worklist_preprocess_identity(
                vfm, resize or (), int(num_frames)
            )
            if runtime_preprocess != canonical_preprocess:
                raise GapWorklistError(
                    "runtime streaming preprocess_identity is not canonical"
                )
        else:
            if set(runtime_preprocess) != {"resize", "num_frames"}:
                raise GapWorklistError(
                    "target preprocess_identity must be exactly {resize,num_frames}"
                )
            runtime_preprocess = {
                "resize": [int(value) for value in runtime_preprocess["resize"]],
                "num_frames": int(runtime_preprocess["num_frames"]),
            }

        expected = {
            "schema_version": 1,
            "contract": CONTRACT,
            "canonical_protocol": CANONICAL_PROTOCOL,
            "mode": str(mode),
            "vfm": str(vfm),
            "model_id": str(model_id),
            "timestep": int(timestep),
            "feature_namespace": str(feature_namespace),
            "protocol_sha256": str(protocol_sha256),
            "preprocess_identity": runtime_preprocess,
        }
        bad = [field for field, value in expected.items() if payload.get(field) != value]
        if bad:
            raise GapWorklistError("gap worklist identity mismatch: " + ",".join(bad))
        if mode not in MODES:
            raise GapWorklistError(f"unsupported gap mode: {mode!r}")
        if str(feature_namespace) != GAP_NAMESPACES[mode]:
            raise GapWorklistError("gap worklist mode/feature_namespace mismatch")
        prompt_contract = payload.get("prompt_contract")
        if not isinstance(prompt_contract, dict):
            raise GapWorklistError("prompt_contract must be a structured object")
        prompt_mode = prompt_contract.get("mode")
        prompt_value = prompt_contract.get("value")
        if prompt_mode == "exact":
            if not isinstance(prompt_value, str) or prompt_value != str(actual_prompt):
                raise GapWorklistError("exact prompt_contract does not match runtime prompt")
        elif prompt_mode == "not_applicable":
            if prompt_value is not None:
                raise GapWorklistError("not_applicable prompt_contract must have value=null")
        else:
            raise GapWorklistError("prompt_contract.mode must be exact or not_applicable")
        protocol_file_sha = str(payload.get("protocol_file_sha256") or "")
        canonical_source_sha = str(payload.get("canonical_source_sha256") or "")
        if not _SHA256.fullmatch(protocol_file_sha):
            raise GapWorklistError("gap worklist has no valid protocol_file_sha256")
        if not _SHA256.fullmatch(canonical_source_sha):
            raise GapWorklistError("gap worklist has no valid canonical_source_sha256")

        expected_layers = tuple(sorted({int(layer) for layer in output_layers}))
        declared_layers = _layers(payload.get("output_layers"), "output_layers")
        if declared_layers != expected_layers:
            raise GapWorklistError("gap worklist output_layers mismatch")
        entries = payload.get("entries")
        if not isinstance(entries, list) or not entries:
            raise GapWorklistError("gap worklist entries must be a non-empty list")

        stream: dict[tuple[str, int, int], frozenset[int]] = {}
        target: dict[tuple[str, int], frozenset[int]] = {}
        reasons: dict[str, int] = {}
        allowed_layers = set(expected_layers)
        layer_items = 0
        for index, entry in enumerate(entries):
            label = f"entries[{index}]"
            if not isinstance(entry, dict):
                raise GapWorklistError(f"{label} must be an object")
            scene = str(entry.get("scene") or "").strip()
            if not scene or "/" in scene or scene in {".", ".."}:
                raise GapWorklistError(f"{label}.scene is invalid")
            layers = _layers(entry.get("layers"), f"{label}.layers")
            if not set(layers).issubset(allowed_layers):
                raise GapWorklistError(f"{label}.layers escapes output_layers")
            reason = str(entry.get("reason") or "").strip().upper()
            if not reason:
                raise GapWorklistError(f"{label}.reason is required")
            reasons[reason] = reasons.get(reason, 0) + len(layers)
            layer_items += len(layers)
            if mode == "streaming_prefix":
                window_id = _integer(entry.get("window_id"), f"{label}.window_id")
                prefix = _integer(entry.get("prefix"), f"{label}.prefix")
                if window_id < 0 or prefix not in CANONICAL_PREFIXES:
                    raise GapWorklistError(f"{label} has a non-canonical stream key")
                key = (scene, window_id, prefix)
                if key in stream:
                    raise GapWorklistError(f"duplicate streaming gap key: {key}")
                stream[key] = frozenset(layers)
            else:
                frame_index = _integer(entry.get("frame_index"), f"{label}.frame_index")
                if frame_index < 0:
                    raise GapWorklistError(f"{label}.frame_index must be non-negative")
                key = (scene, frame_index)
                if key in target:
                    raise GapWorklistError(f"duplicate target gap key: {key}")
                target[key] = frozenset(layers)

        summary = payload.get("counts")
        if not isinstance(summary, dict):
            raise GapWorklistError("gap worklist counts object is required")
        expected_counts = {
            "entries": len(entries),
            "layer_items": layer_items,
            "scenes": len({entry[0] for entry in (*stream.keys(), *target.keys())}),
        }
        count_bad = [
            field for field, value in expected_counts.items() if summary.get(field) != value
        ]
        if count_bad:
            raise GapWorklistError("gap worklist count mismatch: " + ",".join(count_bad))

        remote_completed_scenes = _completion_exclusions(file_sha)
        local_ready_scenes = _local_ready_exclusions(file_sha)
        remote_completed_units = _completion_unit_exclusions(file_sha)
        local_ready_units = _local_ready_unit_exclusions(file_sha)
        # Local-ready is deliberately only an in-process write exclusion.  It
        # is merged here so the extractor cannot overwrite a scene which the
        # companion sync loop is about to publish, but it is never persisted
        # as remote completion state by the runner.
        completed_scenes = remote_completed_scenes | local_ready_scenes
        completed_units = remote_completed_units | local_ready_units
        declared_scenes = {entry[0] for entry in (*stream.keys(), *target.keys())}
        if not completed_scenes.issubset(declared_scenes):
            raise GapWorklistError(
                "completed/local-ready scenes file contains a scene outside the gap worklist"
            )
        declared_units = {
            streaming_gap_unit_key(scene, window_id, prefix, layer)
            for (scene, window_id, prefix), layers in stream.items()
            for layer in layers
        } | {
            target_gap_unit_key(scene, frame_index, layer)
            for (scene, frame_index), layers in target.items()
            for layer in layers
        }
        if not completed_units.issubset(declared_units):
            raise GapWorklistError(
                "completed/local-ready units file contains a unit outside the gap worklist"
            )

        return cls(
            path=source,
            file_sha256=file_sha,
            content_sha256=str(content_sha),
            mode=mode,
            vfm=vfm,
            model_id=model_id,
            timestep=int(timestep),
            feature_namespace=feature_namespace,
            output_layers=expected_layers,
            protocol_sha256=protocol_sha256,
            protocol_file_sha256=protocol_file_sha,
            canonical_source_sha256=canonical_source_sha,
            prompt_contract={"mode": prompt_mode, "value": prompt_value},
            preprocess_identity=runtime_preprocess,
            _stream=stream,
            _target=target,
            _completed_scenes=completed_scenes,
            _completed_units=completed_units,
            reason_counts=dict(sorted(reasons.items())),
        )
