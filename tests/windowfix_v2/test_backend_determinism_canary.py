import json
import struct
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

import scripts.windowfix_v2.run_backend_determinism_canary as canary_module
from scripts.windowfix_v2.run_backend_determinism_canary import (
    canonical_json_sha,
    compare_selected_scene,
    extractor_command,
    inspect_qwen_streaming_scene,
    qwen_streaming_tensor_contract,
    target_isolated_tensor_contract,
    validate_qwen_streaming_tensor_contract,
    validate_target_isolated_tensor_contract,
)


def _extractor_command_args(*, vfm: str, resize: list[int]) -> Namespace:
    return Namespace(
        vfm=vfm,
        python=Path("/python"),
        dataset_root=Path("/dataset"),
        model_id="model-id",
        timestep=749,
        layer=1,
        prompt=".",
        resize=resize,
        num_frames=81,
    )


def test_main_extractor_canary_omits_unsupported_resize_but_binds_default():
    command = extractor_command(
        _extractor_command_args(vfm="aether", resize=[480, 720]),
        mode="streaming_prefix",
        out_root=Path("/out"),
        start=1,
        end=2,
    )
    assert "--resize" not in command


def test_main_extractor_canary_rejects_non_default_resize():
    with pytest.raises(RuntimeError, match="fixes resize"):
        extractor_command(
            _extractor_command_args(vfm="f3r", resize=[480, 720]),
            mode="streaming_prefix",
            out_root=Path("/out"),
            start=1,
            end=2,
        )


def _write_scene(
    root: Path,
    *,
    payload: bytes = b"same",
    vfm: str = "aether",
    no_noise: bool = False,
) -> None:
    scene = root / vfm / "scannetpp" / "scene-a" / "window_0000" / "prefix_000007"
    scene.mkdir(parents=True)
    feature = scene / "feature_t749_layer1.sft"
    feature.write_bytes(payload)
    sidecar = Path(f"{feature}.manifest.json")
    metadata = {"feature_generation_contract": "inscene_deterministic_noise_v1"}
    if no_noise:
        metadata.update(
            {
                "noise_seed_scheme": "deterministic_forward_no_noise_v1",
                "noise_seed_scope": "no_random_draws",
            }
        )
    else:
        metadata.update(
            {
                "noise_seed_scheme": "inscene_noise_seed_v1",
                "noise_seed_samples": [{"noise_seed": 1}],
            }
        )
    sidecar.write_text(json.dumps(metadata))


def test_compare_selected_scene_accepts_byte_identical_provenance(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_scene(left)
    _write_scene(right)
    result = compare_selected_scene(root_a=left, root_b=right, vfm="aether")
    assert result["scene"] == "scene-a"
    assert result["safetensors_compared"] == 1
    assert result["files_compared"] == 2


def test_compare_selected_scene_rejects_tensor_mismatch(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_scene(left, payload=b"left")
    _write_scene(right, payload=b"right")
    with pytest.raises(RuntimeError, match="cross-order mismatch"):
        compare_selected_scene(root_a=left, root_b=right, vfm="aether")


def test_compare_selected_scene_accepts_qwen_no_noise_contract(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_scene(left, vfm="qwen2_5_vl", no_noise=True)
    _write_scene(right, vfm="qwen2_5_vl", no_noise=True)
    result = compare_selected_scene(
        root_a=left,
        root_b=right,
        vfm="qwen2_5_vl",
    )
    assert result["safetensors_compared"] == 1


def _write_safetensor(path: Path, shape: list[int], dtype: str = "F16") -> None:
    widths = {"F16": 2, "BF16": 2, "F32": 4}
    elements = 1
    for value in shape:
        elements *= value
    payload_size = elements * widths[dtype]
    header = json.dumps(
        {
            "feat": {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [0, payload_size],
            }
        },
        separators=(",", ":"),
    ).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * payload_size)


def _write_qwen_stream_scene(root: Path, shapes: dict[int, list[int]]) -> Path:
    scene = root / "qwen2_5_vl" / "scannetpp" / "scene-a"
    records = []
    for prefix in (8, 12, 16, 24):
        tail = prefix - 1
        records.append(
            {
                "window_id": 0,
                "tail": tail,
                "valid_length": prefix,
            }
        )
        _write_safetensor(
            scene
            / "window_0000"
            / f"prefix_{tail:06d}"
            / "feature_layer-1.sft",
            shapes[prefix],
        )
    scene.mkdir(parents=True, exist_ok=True)
    with (scene / "prefix_index.npy").open("wb") as handle:
        np.save(handle, np.array(records, dtype=object))
    return scene


def test_qwen_contract_records_all_prefix_shapes_and_dtype(tmp_path):
    shapes = {
        8: [2, 12, 12, 64],
        12: [3, 12, 12, 64],
        16: [4, 12, 12, 64],
        24: [6, 12, 12, 64],
    }
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_qwen_stream_scene(left, shapes)
    _write_qwen_stream_scene(right, shapes)

    contract = qwen_streaming_tensor_contract(left, right)
    normalized = validate_qwen_streaming_tensor_contract(contract)

    assert normalized == {
        str(prefix): {"shape": shape, "dtype": "float16"}
        for prefix, shape in shapes.items()
    }
    assert contract["prefixes"]["8"]["single_run_tensors_checked"] == 1
    assert contract["prefixes"]["24"]["ordered_run_tensors_checked"] == 1


def test_qwen_contract_rejects_shape_drift_within_same_prefix(tmp_path):
    scene = _write_qwen_stream_scene(
        tmp_path,
        {prefix: [prefix, 2] for prefix in (8, 12, 16, 24)},
    )
    records = np.load(scene / "prefix_index.npy", allow_pickle=True).tolist()
    records.append({"window_id": 1, "tail": 31, "valid_length": 8})
    with (scene / "prefix_index.npy").open("wb") as handle:
        np.save(handle, np.array(records, dtype=object))
    _write_safetensor(
        scene / "window_0001" / "prefix_000031" / "feature_layer-1.sft",
        [9, 2],
    )

    with pytest.raises(RuntimeError, match="p8 shape/dtype varies"):
        inspect_qwen_streaming_scene(scene)


def test_qwen_contract_is_fail_closed_when_one_prefix_is_missing(tmp_path):
    shapes = {prefix: [prefix, 2] for prefix in (8, 12, 16, 24)}
    scene = _write_qwen_stream_scene(tmp_path, shapes)
    missing = scene / "window_0000" / "prefix_000015" / "feature_layer-1.sft"
    missing.unlink()
    with pytest.raises(RuntimeError, match="no p16"):
        inspect_qwen_streaming_scene(scene)


def _write_target_scene(root: Path, *, tail: list[int]) -> None:
    scene = root / "qwen2_5_vl" / "scannetpp" / "scene-a"
    _write_safetensor(scene / "feature_layer-1.sft", [3, *tail])


def test_target_contract_records_and_enforces_row_tail_shape_dtype(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    expected = {
        "tensor_key": "feat",
        "expected_tail_shape": [12, 12, 3584],
        "expected_dtype": "float16",
    }
    _write_target_scene(left, tail=expected["expected_tail_shape"])
    _write_target_scene(right, tail=expected["expected_tail_shape"])
    contract = target_isolated_tensor_contract(
        left, right, vfm="qwen2_5_vl", expected=expected
    )
    assert validate_target_isolated_tensor_contract(contract, expected) == {
        "tail_shape": [12, 12, 3584],
        "dtype": "float16",
    }


def test_target_contract_rejects_wrong_row_tail_shape(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_target_scene(left, tail=[12, 12, 1024])
    _write_target_scene(right, tail=[12, 12, 1024])
    with pytest.raises(RuntimeError, match="tail shape/dtype"):
        target_isolated_tensor_contract(
            left,
            right,
            vfm="qwen2_5_vl",
            expected={
                "tensor_key": "feat",
                "expected_tail_shape": [12, 12, 3584],
                "expected_dtype": "float16",
            },
        )


def _canary_spec_fixture(tmp_path: Path, monkeypatch) -> tuple[Namespace, Path, Path]:
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n")
    extractor = tmp_path / "extractor.py"
    extractor.write_text("# extractor\n")
    runner = tmp_path / "runner.py"
    runner.write_text("# runner\n")
    renderer = tmp_path / "scripts/windowfix_v2/prepare_gap_backend_canary_jobs.py"
    renderer.parent.mkdir(parents=True)
    renderer.write_text("# renderer\n")
    output = tmp_path / "receipt.json"
    spec_path = tmp_path / "spec.json"
    provenance = {
        "backend_dependencies": {
            "features/noise_seed.py": {
                "path": "/project/features/noise_seed.py",
                "sha256": "1" * 64,
            }
        },
        "model_artifacts": {},
        "required_environment": {"HF_HUB_OFFLINE": "1"},
    }
    monkeypatch.setattr(
        canary_module,
        "backend_provenance_binding",
        lambda vfm, project_root: provenance,
    )
    monkeypatch.setattr(
        canary_module,
        "canary_runtime_environment",
        lambda vfm, value: {"HF_HUB_OFFLINE": "1"},
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    args = Namespace(
        canary_spec=spec_path,
        vfm="qwen2_5_vl",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        layer=-1,
        timestep=749,
        protocol=protocol.resolve(),
        protocol_file_sha256=canary_module.sha256_file(protocol),
        extractor_sha256=canary_module.sha256_file(extractor),
        output=output,
        prompt="Describe the spatial layout of this scene.",
        resize=[336, 336],
        num_frames=24,
        python=Path("/python"),
        sample_index=1,
        project_root=tmp_path,
    )
    spec = {
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
        "canary_runner": str(runner),
        "canary_runner_sha256": canary_module.sha256_file(runner),
        "canary_renderer": str(renderer),
        "canary_renderer_sha256": canary_module.sha256_file(renderer),
        "receipt": str(output.resolve()),
        "runtime_contract": {
            "prompt": args.prompt,
            "resize": args.resize,
            "num_frames": args.num_frames,
            "python": str(args.python),
            "sample_index": args.sample_index,
        },
        "runtime_environment": {"HF_HUB_OFFLINE": "1"},
        "backend_provenance": provenance,
    }
    spec["content_sha256"] = canonical_json_sha(spec)
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    return args, extractor, runner


def test_runner_rechecks_spec_and_backend_provenance_before_execution(
    tmp_path, monkeypatch
):
    args, extractor, runner = _canary_spec_fixture(tmp_path, monkeypatch)
    _, binding = canary_module.validate_canary_spec(
        args, extractor=extractor, canary_script=runner
    )
    assert binding["path"] == str(args.canary_spec.resolve())
    assert binding["content_sha256"] == json.loads(
        args.canary_spec.read_text()
    )["content_sha256"]


def test_runner_rejects_self_consistent_spec_with_wrong_backend_provenance(
    tmp_path, monkeypatch
):
    args, extractor, runner = _canary_spec_fixture(tmp_path, monkeypatch)
    spec = json.loads(args.canary_spec.read_text())
    spec["backend_provenance"]["required_environment"] = {}
    spec.pop("content_sha256")
    spec["content_sha256"] = canonical_json_sha(spec)
    args.canary_spec.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    with pytest.raises(RuntimeError, match="backend provenance changed"):
        canary_module.validate_canary_spec(
            args, extractor=extractor, canary_script=runner
        )
