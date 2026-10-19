from __future__ import annotations

import hashlib
import importlib
import json
import struct
from pathlib import Path

import numpy as np

from analyses.windowfix_v2.classify_feature_reuse import (
    DEFAULT_PROTOCOL,
    FeatureSpec,
    LegacyRoot,
    classify,
)


RAW = tuple(range(100, 108))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_safetensors(path: Path, shape=(2, 3), dtype="F16") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    element_sizes = {"F16": 2, "F32": 4}
    data_size = int(np.prod(shape)) * element_sizes[dtype]
    header = json.dumps(
        {"feat": {"dtype": dtype, "shape": list(shape), "data_offsets": [0, data_size]}},
        separators=(",", ":"),
    ).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(data_size))


def _write_feature(scene_root: Path, scene: str, *, model_id="model/a", shape=(2, 3)) -> Path:
    feature = (
        scene_root
        / scene
        / "window_0000"
        / "prefix_000007"
        / "feature_t749_layer20.sft"
    )
    _write_safetensors(feature, shape=shape)
    sidecar = {
        "schema_version": 1,
        "file": feature.name,
        "size_bytes": feature.stat().st_size,
        "sha256": _sha(feature),
        "vfm": "wan",
        "model_id": model_id,
        "frame_index_schema": "image_mask_intersection_v1",
        "mode": "streaming_prefix",
        "timestep": 749,
        "tensors": {"feat": {"shape": list(shape), "dtype": "torch.float16"}},
    }
    Path(f"{feature}.manifest.json").write_text(json.dumps(sidecar) + "\n")
    return feature


def _write_index(scene_root: Path, scene: str, raw_ids: tuple[int, ...]) -> Path:
    path = scene_root / scene / "prefix_index.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"window_id": 0, "tail": 7, "indices": list(raw_ids)}
    np.save(path, np.asarray([record], dtype=object))
    sidecar = {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha(path),
    }
    Path(f"{path}.manifest.json").write_text(json.dumps(sidecar) + "\n")
    return path


def _canonical(scene: str) -> dict:
    return {
        "scene_id": scene,
        "window_id": 0,
        "tail": 7,
        "raw_ids": RAW,
        "sample_id": f"{scene}_w000_p8",
        "source": "scannetpp",
    }


def _spec(**overrides) -> FeatureSpec:
    values = {
        "spec_id": "wan_t749_l20_p8",
        "root_ids": ("legacy",),
        "prefix": 8,
        "vfm": "wan",
        "model_id": "model/a",
        "layer": 20,
        "timestep": 749,
        "expected_shape": (2, 3),
        "expected_dtype": "torch.float16",
        "tensor_key": "feat",
        "feature_name": None,
        "feature_name_template": None,
        "expected_mode": "streaming_prefix",
        "expected_sidecar": {},
    }
    values.update(overrides)
    return FeatureSpec(**values)


def _root(index_root: Path, feature_root: Path) -> LegacyRoot:
    return LegacyRoot(
        root_id="legacy",
        index_scene_root=str(index_root),
        feature_scene_root=str(feature_root),
    )


def test_classifies_exact_mismatch_and_missing_per_canonical_record(tmp_path: Path) -> None:
    index_root = tmp_path / "indexes"
    feature_root = tmp_path / "features"
    _write_index(index_root, "exact", RAW)
    _write_feature(feature_root, "exact")
    _write_index(index_root, "mismatch", RAW[:-1] + (999,))
    _write_feature(feature_root, "mismatch")

    manifest = tmp_path / "p8.jsonl"
    canonical = [_canonical("exact"), _canonical("mismatch"), _canonical("missing")]
    details, summary = classify(
        specs=[_spec()],
        roots={"legacy": _root(index_root, feature_root)},
        manifests={8: (manifest, canonical)},
        protocol=DEFAULT_PROTOCOL,
    )

    assert {row["scene_id"]: row["status"] for row in details} == {
        "exact": "REUSE_EXACT",
        "mismatch": "REEXTRACT_MISMATCH",
        "missing": "REEXTRACT_MISSING",
    }
    assert summary["status_counts"] == {
        "REUSE_EXACT": 1,
        "REEXTRACT_MISMATCH": 1,
        "REEXTRACT_MISSING": 1,
        "UNVERIFIED": 0,
    }


def test_sidecar_only_hash_policy_never_claims_exact(tmp_path: Path) -> None:
    index_root = tmp_path / "indexes"
    feature_root = tmp_path / "features"
    _write_index(index_root, "scene", RAW)
    _write_feature(feature_root, "scene")
    details, _summary = classify(
        specs=[_spec()],
        roots={"legacy": _root(index_root, feature_root)},
        manifests={8: (tmp_path / "p8.jsonl", [_canonical("scene")])},
        protocol=DEFAULT_PROTOCOL,
        payload_hash_policy="sidecar",
    )
    assert details[0]["status"] == "UNVERIFIED"
    assert details[0]["reason_codes"] == ["payload_sha256_not_computed"]


def test_existing_payload_with_wrong_model_is_reextract_mismatch(tmp_path: Path) -> None:
    index_root = tmp_path / "indexes"
    feature_root = tmp_path / "features"
    _write_index(index_root, "scene", RAW)
    _write_feature(feature_root, "scene", model_id="wrong/model")
    details, _summary = classify(
        specs=[_spec()],
        roots={"legacy": _root(index_root, feature_root)},
        manifests={8: (tmp_path / "p8.jsonl", [_canonical("scene")])},
        protocol=DEFAULT_PROTOCOL,
    )
    assert details[0]["status"] == "REEXTRACT_MISMATCH"
    assert "sidecar_model_id_mismatch" in details[0]["reason_codes"]


def test_incomplete_spec_is_unverified_even_when_raw_ids_match(tmp_path: Path) -> None:
    index_root = tmp_path / "indexes"
    feature_root = tmp_path / "features"
    _write_index(index_root, "scene", RAW)
    _write_feature(feature_root, "scene")
    details, _summary = classify(
        specs=[_spec(expected_shape=None)],
        roots={"legacy": _root(index_root, feature_root)},
        manifests={8: (tmp_path / "p8.jsonl", [_canonical("scene")])},
        protocol=DEFAULT_PROTOCOL,
    )
    assert details[0]["status"] == "UNVERIFIED"
    assert details[0]["reason_codes"] == ["spec_missing_expected_shape"]


def test_legacy_import_recomputes_missing_historical_sha(tmp_path: Path) -> None:
    index_root = tmp_path / "indexes"
    feature_root = tmp_path / "features"
    _write_index(index_root, "scene", RAW)
    feature = _write_feature(feature_root, "scene")
    sidecar_path = Path(f"{feature}.manifest.json")
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["sha256"] = None
    sidecar["prompt"] = "."
    sidecar["resize"] = [480, 832]
    sidecar_path.write_text(json.dumps(sidecar) + "\n")

    details, summary = classify(
        specs=[_spec(expected_sidecar={
            "prompt": ".",
            "resize": [480, 832],
            "frame_index_schema": "image_mask_intersection_v1",
        })],
        roots={"legacy": _root(index_root, feature_root)},
        manifests={8: (tmp_path / "p8.jsonl", [_canonical("scene")])},
        protocol=DEFAULT_PROTOCOL,
        payload_hash_policy="legacy-import",
    )

    assert details[0]["status"] == "REUSE_EXACT"
    assert details[0]["source_payload_sha256"] == _sha(feature)
    assert summary["generation_contract"] == "LEGACY_EXACT_IMPORT_V1"
    assert summary["historical_sidecar_sha256_required"] is False


def test_legacy_import_requires_prompt_and_preprocess_contract(tmp_path: Path) -> None:
    index_root = tmp_path / "indexes"
    feature_root = tmp_path / "features"
    _write_index(index_root, "scene", RAW)
    _write_feature(feature_root, "scene")

    details, _summary = classify(
        specs=[_spec()],
        roots={"legacy": _root(index_root, feature_root)},
        manifests={8: (tmp_path / "p8.jsonl", [_canonical("scene")])},
        protocol=DEFAULT_PROTOCOL,
        payload_hash_policy="legacy-import",
    )

    assert details[0]["status"] == "UNVERIFIED"
    assert set(details[0]["reason_codes"]) == {
        "spec_missing_expected_prompt_contract",
        "spec_missing_expected_preprocess_contract",
    }


def test_legacy_import_rejects_wrong_existing_historical_sha(tmp_path: Path) -> None:
    index_root = tmp_path / "indexes"
    feature_root = tmp_path / "features"
    _write_index(index_root, "scene", RAW)
    feature = _write_feature(feature_root, "scene")
    sidecar_path = Path(f"{feature}.manifest.json")
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["sha256"] = "0" * 64
    sidecar["prompt"] = "."
    sidecar["resize"] = [480, 832]
    sidecar_path.write_text(json.dumps(sidecar) + "\n")

    details, _summary = classify(
        specs=[_spec(expected_sidecar={
            "prompt": ".",
            "resize": [480, 832],
            "frame_index_schema": "image_mask_intersection_v1",
        })],
        roots={"legacy": _root(index_root, feature_root)},
        manifests={8: (tmp_path / "p8.jsonl", [_canonical("scene")])},
        protocol=DEFAULT_PROTOCOL,
        payload_hash_policy="legacy-import",
    )

    assert details[0]["status"] == "REEXTRACT_MISMATCH"
    assert "payload_sha256_mismatch" in details[0]["reason_codes"]


def test_reference_mode_does_not_hash_payload_and_records_declared_sha(
    tmp_path: Path, monkeypatch
) -> None:
    index_root = tmp_path / "indexes"
    feature_root = tmp_path / "features"
    _write_index(index_root, "scene", RAW)
    feature = _write_feature(feature_root, "scene")
    sidecar_path = Path(f"{feature}.manifest.json")
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["sha256"] = "0" * 64
    sidecar["prompt"] = "."
    sidecar["resize"] = [480, 832]
    sidecar_path.write_text(json.dumps(sidecar) + "\n")

    classifier = importlib.import_module("analyses.windowfix_v2.classify_feature_reuse")
    original_sha256_file = classifier.sha256_file

    def forbid_payload_hash(path: Path) -> str:
        if Path(path).suffix == ".sft":
            raise AssertionError("reference mode must not hash feature payloads")
        return original_sha256_file(path)

    monkeypatch.setattr(classifier, "sha256_file", forbid_payload_hash)
    details, summary = classify(
        specs=[_spec(expected_sidecar={
            "prompt": ".",
            "resize": [480, 832],
            "frame_index_schema": "image_mask_intersection_v1",
        })],
        roots={"legacy": _root(index_root, feature_root)},
        manifests={8: (tmp_path / "p8.jsonl", [_canonical("scene")])},
        protocol=DEFAULT_PROTOCOL,
        payload_hash_policy="reference",
    )

    assert details[0]["status"] == "REUSE_EXACT"
    assert details[0]["source_payload_sha256"] is None
    assert details[0]["declared_source_payload_sha256"] == "0" * 64
    assert details[0]["payload_sha256_verified"] is False
    assert summary["generation_contract"] == "LEGACY_EXACT_REFERENCE_V1"
    assert summary["payload_full_read_performed"] is False
    assert summary["byte_level_certified"] is False
