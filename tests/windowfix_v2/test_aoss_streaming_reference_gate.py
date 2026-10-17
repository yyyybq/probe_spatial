from __future__ import annotations

import io
import json
import signal
import struct
from pathlib import Path

import numpy as np
import pytest

import analyses.windowfix_v2.build_streaming_reference_catalog as catalog_module
from analyses.windowfix_v2.gate_aoss_streaming_references import (
    FAIL,
    PASS,
    PENDING,
    RemoteReader,
    RequestTimeout,
    check_request,
    normalize_uri,
    request_timeout,
    request_timeout_receipt,
)
from analyses.windowfix_v2.build_streaming_reference_catalog import (
    EXPECTED_STREAMING_SHAPES,
    expected_shapes_by_prefix,
    preprocess_contract,
    qwen_contract_from_canary_receipt,
)


class FakeClient:
    def __init__(self, values, error=False):
        self.values = values
        self.error = error
        self.ranges = []

    def contains(self, uri):
        if self.error:
            raise RuntimeError("network unavailable")
        return uri in self.values

    def size(self, uri):
        return len(self.values[uri])

    def get(self, uri, **kwargs):
        data = self.values[uri]
        byte_range = kwargs.get("range")
        if byte_range is None:
            return data
        self.ranges.append((uri, byte_range))
        start, end = (int(value) for value in byte_range.split("-"))
        return data[start : end + 1]


def npy_bytes(record):
    output = io.BytesIO()
    np.save(output, np.array([record], dtype=object), allow_pickle=True)
    return output.getvalue()


def safetensor_bytes():
    header = json.dumps(
        {"feat": {"dtype": "F16", "shape": [2, 3], "data_offsets": [0, 12]}},
        separators=(",", ":"),
    ).encode()
    # Real safetensors headers are space-padded to an 8-byte boundary.
    header += b" " * ((-len(header)) % 8)
    return struct.pack("<Q", len(header)) + header + bytes(12)


def request():
    return {
        "unit_id": "spec:scene_w000_p8",
        "source_uri": "aoss://bucket/root/feature.sft",
        "sidecar_uri": "aoss://bucket/root/feature.sft.manifest.json",
        "source_index_uri": "aoss://bucket/root/prefix_index.npy",
        "source_index_record": {"window_id": 0, "tail": 7},
        "expected": {
            "vfm": "dino",
            "model_id": "facebook/dinov2-large",
            "feature_layer": 0,
            "timestep": 749,
            "feature_filename": "feature.sft",
            "mode": "streaming_prefix",
            "prompt_contract": {"mode": "not_applicable"},
            "preprocess_identity": {
                "frame_index_schema": "image_mask_intersection_v1",
                "resize": [420, 728],
                "num_frames": 64,
            },
            "frame_index_schema": "image_mask_intersection_v1",
            "tensor_key": "feat",
            "expected_shape": [2, 3],
            "expected_dtype": "float16",
            "raw_ids": [0, 1, 2, 3, 4, 5, 6, 7],
        },
    }


def fake_values(raw_ids=None):
    raw_ids = raw_ids or [0, 1, 2, 3, 4, 5, 6, 7]
    feature = safetensor_bytes()
    sidecar = json.dumps(
        {
            "file": "feature.sft",
            "size_bytes": len(feature),
            "vfm": "dino",
            "model_id": "facebook/dinov2-large",
            "mode": "streaming_prefix",
            "frame_index_schema": "image_mask_intersection_v1",
            "resize": [420, 728],
            "num_frames": 64,
            "timestep": 0,
            "prompt": "ignored",
            "tensors": {"feat": {"shape": [2, 3], "dtype": "torch.float16"}},
        },
        separators=(",", ":"),
    ).encode()
    return {
        normalize_uri("aoss://bucket/root/feature.sft"): feature,
        normalize_uri("aoss://bucket/root/feature.sft.manifest.json"): sidecar,
        normalize_uri("aoss://bucket/root/prefix_index.npy"): npy_bytes(
            {"window_id": 0, "tail": 7, "indices": raw_ids}
        ),
    }


def test_range_gate_passes_without_reading_payload_body():
    client = FakeClient(fake_values())
    receipt = check_request(RemoteReader(client, 1024 * 1024), request())
    assert receipt["status"] == PASS
    assert receipt["payload_full_read_performed"] is False
    assert [item[1] for item in client.ranges] == ["0-7", "8-71"]

    import jsonschema

    schema = json.loads(
        (Path(__file__).resolve().parents[2]
         / "protocols/legacy_exact_reference_remote_gate_receipt_v1.schema.json")
        .read_text(encoding="utf-8")
    )
    jsonschema.validate(receipt, schema)


def test_streaming_shape_and_qwen_no_pad_contracts_are_explicit():
    assert EXPECTED_STREAMING_SHAPES["wan"] == [21, 30, 52, 1536]
    assert EXPECTED_STREAMING_SHAPES["qwen2_5_vl"] is None
    assert preprocess_contract("qwen2_5_vl", 24) == {
        "resize": [336, 336],
        "num_frames": 24,
        "temporal_padding": "none",
    }
    assert preprocess_contract("dino", 8)["temporal_padding"] == "repeat_tail"


def test_catalog_qwen_shapes_are_per_prefix_and_fail_closed_without_receipt():
    unresolved = expected_shapes_by_prefix()
    assert unresolved["qwen2_5_vl"] is None
    assert unresolved["wan"] == {
        str(prefix): [21, 30, 52, 1536] for prefix in (8, 12, 16, 24)
    }

    qwen = {
        str(prefix): {"shape": [prefix // 4, 12, 12, 3584], "dtype": "float16"}
        for prefix in (8, 12, 16, 24)
    }
    resolved = expected_shapes_by_prefix(qwen)
    assert resolved["qwen2_5_vl"] == {
        str(prefix): [prefix // 4, 12, 12, 3584]
        for prefix in (8, 12, 16, 24)
    }


def _qwen_shape_receipt(tmp_path: Path, monkeypatch) -> tuple[Path, dict]:
    provenance = {
        "backend_dependencies": {},
        "model_artifacts": {},
        "required_environment": {"HF_HUB_OFFLINE": "1"},
    }
    runtime_environment = {"HF_HUB_OFFLINE": "1"}
    monkeypatch.setattr(
        catalog_module,
        "backend_provenance_binding",
        lambda vfm, root: provenance,
    )
    monkeypatch.setattr(
        catalog_module,
        "canary_runtime_environment",
        lambda vfm, value: runtime_environment,
    )
    protocol_sha = catalog_module.sha256_file(catalog_module.PROTOCOL)
    extractor = catalog_module.ROOT / "features/run_inscene15k_mllm.py"
    runner = (
        catalog_module.ROOT
        / "scripts/windowfix_v2/run_backend_determinism_canary.py"
    )
    renderer = (
        catalog_module.ROOT
        / "scripts/windowfix_v2/prepare_gap_backend_canary_jobs.py"
    )
    receipt_path = tmp_path / "qwen.json"
    spec_path = tmp_path / "qwen-spec.json"
    spec = {
        "schema_version": 2,
        "audit_type": "canonical_v2_backend_canary_execution_spec",
        "status": "READY_FOR_REVIEW_NOT_SUBMITTED",
        "submission_performed": False,
        "vfm": "qwen2_5_vl",
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "layer": -1,
        "timestep": 749,
        "protocol_file_sha256": protocol_sha,
        "extractor": str(extractor),
        "extractor_sha256": catalog_module.sha256_file(extractor),
        "canary_runner": str(runner),
        "canary_runner_sha256": catalog_module.sha256_file(runner),
        "canary_renderer": str(renderer),
        "canary_renderer_sha256": catalog_module.sha256_file(renderer),
        "backend_provenance": provenance,
        "runtime_environment": runtime_environment,
        "runtime_contract": {
            "prompt": "Describe the spatial layout of this scene.",
            "resize": [336, 336],
            "num_frames": 24,
            "python": "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python",
            "sample_index": 1,
        },
        "receipt": str(receipt_path.resolve()),
    }
    spec["content_sha256"] = catalog_module.sha256_json(spec)
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    comparison = {
        "scene": "scene-a",
        "files_compared": 8,
        "safetensors_compared": 4,
        "comparison_sha256": "a" * 64,
    }
    prefixes = {
        str(prefix): {
            "shape": [prefix // 4, 12, 12, 3584],
            "dtype": "float16",
            "single_run_tensors_checked": 4,
            "ordered_run_tensors_checked": 4,
            "single_run_header_set_sha256": "b" * 64,
            "ordered_run_header_set_sha256": "b" * 64,
        }
        for prefix in (8, 12, 16, 24)
    }
    receipt = {
        "schema_version": 2,
        "audit_type": "canonical_v2_backend_cross_process_order_canary",
        "status": "pass",
        "submission_performed": False,
        "vfm": "qwen2_5_vl",
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "layer": -1,
        "timestep": 749,
        "protocol": "inscene_streaming_canonical_v2_obs28_h124",
        "protocol_file_sha256": protocol_sha,
        "extractor": str(extractor),
        "extractor_sha256": catalog_module.sha256_file(extractor),
        "canary_script": str(runner),
        "canary_script_sha256": catalog_module.sha256_file(runner),
        "runtime_prompt": "Describe the spatial layout of this scene.",
        "runtime_resize": [336, 336],
        "runtime_num_frames": 24,
        "backend_provenance_revalidated_after_runs": True,
        "backend_provenance": provenance,
        "runtime_environment": runtime_environment,
        "canary_spec_binding": {
            "path": str(spec_path.resolve()),
            "file_sha256": catalog_module.sha256_file(spec_path),
            "content_sha256": spec["content_sha256"],
        },
        "gap_binding": {
            "canonical_source_sha256": catalog_module.CANONICAL_PARENT_SOURCE_SHA256,
            "selected_scene_layer_covered": True,
            "selected_scene": "scene-a",
        },
        "results": {
            "streaming_prefix": {"comparison": comparison},
            "target_isolated": {"comparison": comparison},
        },
        "qwen_streaming_tensor_contract": {
            "schema_version": 1,
            "contract": "QWEN_STREAMING_PREFIX_TENSOR_CONTRACT_V1",
            "tensor_key": "feat",
            "all_prefixes_present": True,
            "same_prefix_shape_dtype_consistent": True,
            "single_ordered_shape_dtype_equal": True,
            "prefixes": prefixes,
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt_path, receipt


def test_catalog_accepts_only_spec_and_provenance_bound_qwen_shape_receipt(
    tmp_path, monkeypatch
):
    receipt_path, _ = _qwen_shape_receipt(tmp_path, monkeypatch)
    contract, spec_path = qwen_contract_from_canary_receipt(
        receipt_path,
        protocol_file_sha256=catalog_module.sha256_file(catalog_module.PROTOCOL),
    )
    assert contract["8"] == {"shape": [2, 12, 12, 3584], "dtype": "float16"}
    assert spec_path.name == "qwen-spec.json"


def test_catalog_rejects_qwen_receipt_after_tensor_contract_tamper(
    tmp_path, monkeypatch
):
    receipt_path, receipt = _qwen_shape_receipt(tmp_path, monkeypatch)
    receipt["qwen_streaming_tensor_contract"]["prefixes"]["12"]["dtype"] = "bfloat16"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    with pytest.raises(catalog_module.CatalogError, match="p12 shape/dtype"):
        qwen_contract_from_canary_receipt(
            receipt_path,
            protocol_file_sha256=catalog_module.sha256_file(catalog_module.PROTOCOL),
        )


def test_range_gate_fails_closed_on_remote_index_raw_id_mismatch():
    client = FakeClient(fake_values([10, 11, 12, 13, 14, 15, 16, 17]))
    receipt = check_request(RemoteReader(client, 1024 * 1024), request())
    assert receipt["status"] == FAIL
    assert "CHECK_FAILED:source_index_raw_ids_exact" in receipt["reason_codes"]


def test_range_gate_fails_closed_on_exact_shape_mismatch():
    req = request()
    req["expected"]["expected_shape"] = [3, 2]
    receipt = check_request(RemoteReader(FakeClient(fake_values()), 1024 * 1024), req)
    assert receipt["status"] == FAIL
    assert "CHECK_FAILED:header_shape_equals_expected" in receipt["reason_codes"]


def test_range_gate_keeps_unresolved_shape_pending_without_remote_reads():
    req = request()
    req["expected"]["expected_shape"] = None
    client = FakeClient({}, error=True)
    receipt = check_request(RemoteReader(client, 1024 * 1024), req)
    assert receipt["status"] == PENDING
    assert receipt["reason_codes"] == ["BACKEND_EXACT_SHAPE_CANARY_REQUIRED"]
    assert client.ranges == []


def test_range_gate_keeps_backend_error_pending():
    client = FakeClient({}, error=True)
    receipt = check_request(RemoteReader(client, 1024 * 1024), request())
    assert receipt["status"] == PENDING
    assert receipt["reason_codes"] == ["REMOTE_GATE_ERROR:RuntimeError"]


def test_request_alarm_interrupts_a_hung_request():
    with pytest.raises(RequestTimeout):
        with request_timeout(5):
            signal.raise_signal(signal.SIGALRM)


def test_request_timeout_receipt_is_retriable_and_request_bound():
    req = request()
    receipt = request_timeout_receipt(
        req,
        seconds=7,
        error=RequestTimeout("synthetic timeout"),
    )
    assert receipt["status"] == PENDING
    assert receipt["reason_codes"] == ["REMOTE_GATE_REQUEST_TIMEOUT"]
    assert receipt["request_timeout_seconds"] == 7
    assert receipt["request_sha256"]
    assert receipt["payload_full_read_performed"] is False
