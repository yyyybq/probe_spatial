from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from features.canonical_gap_worklist import (
    CANONICAL_PROTOCOL,
    CONTRACT,
    LOCAL_READY_SCENES_CONTRACT,
    CanonicalGapWorklist,
    _local_ready_exclusions,
    canonical_json_sha,
    sha256_file,
    stream_unit_semantic_metadata,
    streaming_worklist_preprocess_identity,
)
from scripts.windowfix_v2.canonical_gap_aoss_adapter import (
    _streaming_feature_filename,
    atomic_temps_for_worklist,
    expected_scene_payloads,
    expected_extraction_unit_keys,
    load_verified_receipt,
)
from scripts.windowfix_v2.run_canonical_gap_shard import (
    _bind_completed_scenes,
    _bind_completed_units,
)

pytestmark = pytest.mark.integration


PROJECT = Path(__file__).resolve().parents[2]
SYNC = PROJECT / "scripts/run_tstep_sync_loop.sh"
ADAPTER = PROJECT / "scripts/windowfix_v2/canonical_gap_aoss_adapter.py"
NAMESPACE = "FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1"
TARGET_NAMESPACE = "FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1"
PROTOCOL_SHA = "2" * 64
PROTOCOL_FILE_SHA = "3" * 64
SOURCE_SHA = "4" * 64
MODEL_ID = "fixture/wan"


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sidecar(path: Path, extra: dict) -> None:
    value = {
        "schema_version": 1,
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **extra,
    }
    _json(Path(f"{path}.manifest.json"), value)


def _worklist(path: Path, *, layers: list[int] | None = None) -> dict:
    layers = list(layers or [5])
    value = {
        "schema_version": 1,
        "contract": CONTRACT,
        "canonical_protocol": CANONICAL_PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA,
        "protocol_file_sha256": PROTOCOL_FILE_SHA,
        "canonical_source_sha256": SOURCE_SHA,
        "mode": "streaming_prefix",
        "vfm": "wan",
        "model_id": MODEL_ID,
        "timestep": 749,
        "feature_namespace": NAMESPACE,
        "output_layers": layers,
        "source": "scannetpp",
        "prompt_contract": {"mode": "exact", "value": "."},
        "preprocess_identity": streaming_worklist_preprocess_identity(
            "wan", [480, 832], 81
        ),
        "entries": [
            {
                "scene": "scene-a",
                "window_id": 0,
                "prefix": 8,
                "layers": layers,
                "reason": "RAW_ID_MISMATCH",
            }
        ],
        "counts": {"entries": 1, "layer_items": len(layers), "scenes": 1},
    }
    value["content_sha256"] = canonical_json_sha(value)
    _json(path, value)
    return value


def test_streaming_filename_contract_matches_real_backend_extractors(tmp_path: Path) -> None:
    assert _streaming_feature_filename(
        vfm="wan", model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", timestep=749, layer=20
    ) == "feature_t749_layer20.sft"
    assert _streaming_feature_filename(
        vfm="cogvideox", model_id="THUDM/CogVideoX-5b-I2V", timestep=749, layer=20
    ) == "feature_t749_layer20.sft"
    assert _streaming_feature_filename(
        vfm="dino", model_id="facebook/dinov2-large", timestep=749, layer=0
    ) == "feature.sft"
    assert _streaming_feature_filename(
        vfm="vjepa2", model_id="facebook/vjepa2-vitl-fpc64-256", timestep=749, layer=23
    ) == "feature_layer23.sft"
    assert _streaming_feature_filename(
        vfm="videomae", model_id="MCG-NJU/videomae-large", timestep=749, layer=23
    ) == "feature_layer23.sft"
    assert _streaming_feature_filename(
        vfm="f3r", model_id="facebook/fast3r", timestep=749, layer=24
    ) == "feature_l24.sft"
    assert _streaming_feature_filename(
        vfm="minimax_h3", model_id="MiniMaxAI/MiniMax-H3", timestep=250, layer=49
    ) == "feature_t250_layer49.sft"
    assert _streaming_feature_filename(
        vfm="qwen2_5_vl", model_id="Qwen/Qwen2.5-VL-7B-Instruct", timestep=749, layer=-1
    ) == "feature_layer-1.sft"

    raw = _worklist(tmp_path / "dino-worklist.json")
    raw["vfm"] = "dino"
    raw["model_id"] = "facebook/dinov2-large"
    raw["entries"] = [{"scene": "scene-a", "window_id": 0, "prefix": 8, "layers": [0]}]
    paths = expected_scene_payloads(raw)
    assert paths["scene-a"] == [
        "scene-a/prefix_index.npy",
        "scene-a/window_0000/prefix_000007/feature.sft",
    ]


def test_atomic_temp_gate_is_scoped_to_the_positive_worklist(tmp_path: Path) -> None:
    worklist = tmp_path / "worklist.json"
    _worklist(worklist)
    root = tmp_path / "payload"
    selected = root / "scene-a/window_0000/prefix_000007/.feature_t749_layer5.sft.tmp-1"
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"partial")
    foreign = root / "scene-other/window_0000/prefix_000007/.feature_t749_layer5.sft.tmp-2"
    foreign.parent.mkdir(parents=True)
    foreign.write_bytes(b"foreign")
    assert atomic_temps_for_worklist(
        root=root, worklist_path=worklist, worklist_sha256=sha256_file(worklist)
    ) == ["scene-a/window_0000/prefix_000007/.feature_t749_layer5.sft.tmp-1"]


def test_atomic_temp_gate_can_ignore_stale_temps(tmp_path: Path, monkeypatch) -> None:
    worklist = tmp_path / "worklist.json"
    _worklist(worklist)
    root = tmp_path / "payload"
    selected = root / "scene-a/window_0000/prefix_000007/.feature_t749_layer5.sft.tmp-1"
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"partial")
    os.utime(selected, (1_700_000_000, 1_700_000_000))
    monkeypatch.setenv("GAP_SYNC_IGNORE_TEMPS_BEFORE", "1700000001")
    assert (
        atomic_temps_for_worklist(
            root=root, worklist_path=worklist, worklist_sha256=sha256_file(worklist)
        )
        == []
    )
    monkeypatch.setenv("GAP_SYNC_IGNORE_TEMPS_BEFORE", "1699999999")
    assert atomic_temps_for_worklist(
        root=root, worklist_path=worklist, worklist_sha256=sha256_file(worklist)
    ) == ["scene-a/window_0000/prefix_000007/.feature_t749_layer5.sft.tmp-1"]


def test_local_ready_scenes_are_sha_bound_but_not_remote_completion(
    tmp_path: Path, monkeypatch
) -> None:
    worklist = tmp_path / "worklist.json"
    _worklist(worklist)
    value = {
        "schema_version": 1,
        "contract": LOCAL_READY_SCENES_CONTRACT,
        "worklist_file_sha256": sha256_file(worklist),
        "local_snapshot_content_sha256": "a" * 64,
        "scenes": ["scene-a"],
    }
    value["content_sha256"] = canonical_json_sha(value)
    binding = tmp_path / "local-ready.json"
    _json(binding, value)
    monkeypatch.setenv("CANONICAL_GAP_LOCAL_READY_SCENES_FILE", str(binding))
    monkeypatch.setenv(
        "CANONICAL_GAP_LOCAL_READY_SCENES_FILE_SHA256", sha256_file(binding)
    )
    assert _local_ready_exclusions(sha256_file(worklist)) == {"scene-a"}


def _shard(path: Path, worklist: Path, worklist_value: dict) -> dict:
    value = {
        "schema_version": 1,
        "status": "READY",
        "contract": CONTRACT,
        "plan_id": "fixture-plan",
        "shard_id": "fixture-shard",
        "worklist_path": str(worklist),
        "worklist_file_sha256": sha256_file(worklist),
        "worklist_content_sha256": worklist_value["content_sha256"],
        "canonical_source_sha256": SOURCE_SHA,
        "protocol_file_sha256": PROTOCOL_FILE_SHA,
        "protocol_sha256": PROTOCOL_SHA,
        "feature_namespace": worklist_value["feature_namespace"],
        "mode": worklist_value["mode"],
        "vfm": "wan",
        "model_id": MODEL_ID,
        "timestep": 749,
        "source_bindings": {"fixture": {"sha256": "5" * 64}},
    }
    value["content_sha256"] = canonical_json_sha(value)
    _json(path, value)
    return value


def _target_worklist(path: Path) -> dict:
    value = {
        "schema_version": 1,
        "contract": CONTRACT,
        "canonical_protocol": CANONICAL_PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA,
        "protocol_file_sha256": PROTOCOL_FILE_SHA,
        "canonical_source_sha256": SOURCE_SHA,
        "mode": "target_isolated",
        "vfm": "wan",
        "model_id": MODEL_ID,
        "timestep": 749,
        "feature_namespace": TARGET_NAMESPACE,
        "output_layers": [5],
        "source": "scannetpp",
        "prompt_contract": {"mode": "exact", "value": "."},
        "preprocess_identity": {"resize": [480, 832], "num_frames": 81},
        "entries": [
            {
                "scene": "scene-a",
                "frame_index": frame,
                "layers": [5],
                "reason": "CANONICAL_TARGET_RAW_ID_UNROUTED",
            }
            for frame in (7, 9)
        ],
        "counts": {"entries": 2, "layer_items": 2, "scenes": 1},
    }
    value["content_sha256"] = canonical_json_sha(value)
    _json(path, value)
    return value


def _payloads(root: Path, worklist_path: Path, *, layer: int = 5) -> tuple[Path, Path]:
    raw_worklist = json.loads(worklist_path.read_text(encoding="utf-8"))
    worklist = CanonicalGapWorklist.load(
        worklist_path,
        expected_sha256=sha256_file(worklist_path),
        mode="streaming_prefix",
        vfm="wan",
        model_id=MODEL_ID,
        timestep=749,
        feature_namespace=NAMESPACE,
        output_layers=raw_worklist["output_layers"],
        protocol_sha256=PROTOCOL_SHA,
        actual_prompt=".",
        preprocess_identity=streaming_worklist_preprocess_identity(
            "wan", [480, 832], 81
        ),
    )
    record = {
        "window_id": 0,
        "tail": 7,
        "indices": list(range(8)),
        "window_indices": list(range(28)),
        "valid_length": 8,
        "obs_start": 0,
        "sampling": "motion_uniform",
        "motion_step": 0.35,
        "rotation_weight": 0.5,
        "canonical_protocol": CANONICAL_PROTOCOL,
        "canonical_protocol_sha256": PROTOCOL_SHA,
        "canonical_obs_len": 28,
    }
    feature = (
        root
        / f"scene-a/window_0000/prefix_000007/feature_t749_layer{layer}.sft"
    )
    feature.parent.mkdir(parents=True, exist_ok=True)
    save_file({"feat": np.zeros((1, 2, 2, 2), dtype=np.float16)}, feature)
    semantic = stream_unit_semantic_metadata(worklist, "scene-a", record, layer)
    _sidecar(
        feature,
        {
            "frame_index_schema": "image_mask_intersection_v1",
            "vfm": "wan",
            "model_id": MODEL_ID,
            "mode": "streaming_prefix",
            "feature_namespace": NAMESPACE,
            "timestep": 749,
            "prompt": ".",
            "resize": [480, 832],
            "num_frames": 81,
            "canonical_protocol": CANONICAL_PROTOCOL,
            "canonical_protocol_sha256": PROTOCOL_SHA,
            "canonical_obs_len": 28,
            "tensors": {
                "feat": {"shape": [1, 2, 2, 2], "dtype": "torch.float16"}
            },
            **semantic,
        },
    )
    index = root / "scene-a/prefix_index.npy"
    index.parent.mkdir(parents=True, exist_ok=True)
    with index.open("wb") as handle:
        np.save(handle, np.array([record], dtype=object))
    _sidecar(
        index,
        {
            "feature_namespace": NAMESPACE,
            "shape": [1],
            "dtype": "object",
            "canonical_protocol": CANONICAL_PROTOCOL,
            "canonical_protocol_sha256": PROTOCOL_SHA,
            "canonical_obs_len": 28,
            "gap_index_contract": "CANONICAL_V2_STATIC_PREFIX_INDEX_V1",
            "canonical_source_sha256": SOURCE_SHA,
        },
    )
    return feature, index


def _target_payloads(root: Path, worklist_path: Path) -> tuple[Path, Path]:
    file_sha = sha256_file(worklist_path)
    raw = json.loads(worklist_path.read_text())
    delta = root / "scene-a/.canonical_v2_target_delta" / file_sha[:16]
    index = delta / "target_indices_layer5.npy"
    index.parent.mkdir(parents=True, exist_ok=True)
    with index.open("wb") as handle:
        np.save(handle, np.array([7, 9], dtype=np.int64))
    index_sha = sha256_file(index)
    shared = {
        "gap_contract": CONTRACT,
        "gap_worklist_file_sha256": file_sha,
        "gap_worklist_content_sha256": raw["content_sha256"],
        "canonical_source_sha256": SOURCE_SHA,
        "vfm": "wan",
        "model_id": MODEL_ID,
        "timestep": 749,
        "layer": 5,
        "prompt": ".",
        "resize": [480, 832],
        "num_frames": 81,
        "target_indices_sha256": index_sha,
    }
    _sidecar(index, {"shape": [2], "dtype": "int64", **shared})
    feature = delta / "feature_t749_layer5.delta.sft"
    save_file({"feat": np.zeros((2, 2, 2, 2), dtype=np.float16)}, feature)
    _sidecar(
        feature,
        {
            "frame_index_schema": "image_mask_intersection_v1",
            "mode": "target_isolated",
            "feature_namespace": TARGET_NAMESPACE,
            "canonical_protocol": CANONICAL_PROTOCOL,
            "canonical_protocol_sha256": PROTOCOL_SHA,
            "canonical_obs_len": 28,
            "tensors": {
                "feat": {"shape": [2, 2, 2, 2], "dtype": "torch.float16"}
            },
            **shared,
        },
    )
    return feature, index


def _fake_ads(path: Path, calls: Path, *, fail: bool) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -u\n"
        "args=(\"$@\")\n"
        "src=${args[$((${#args[@]} - 2))]}\n"
        f"printf '%s\\n' \"${{src}}\" >> {calls}\n"
        "count=$(find \"${src}\" -type f | wc -l)\n"
        + (
            "printf 'Found: %s, copied: 0, failed: 1\\n' \"${count}\"\nexit 9\n"
            if fail
            else "printf 'Found: %s, copied: %s, failed: 0\\n' \"${count}\" \"${count}\"\n"
        ),
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run(
    tmp_path: Path,
    *,
    fail: bool = False,
    layers: list[int] | None = None,
    produced_layer: int = 5,
) -> tuple[subprocess.CompletedProcess[str], dict]:
    layers = list(layers or [5])
    root = tmp_path / "payload"
    log = tmp_path / "logs"
    log.mkdir()
    stop = log / "extract.finished"
    stop.touch()
    done = tmp_path / "state/done.txt"
    receipt = tmp_path / "state/remote.json"
    worklist = tmp_path / "worklist.json"
    worklist_value = _worklist(worklist, layers=layers)
    shard_path = tmp_path / "shard.json"
    shard = _shard(shard_path, worklist, worklist_value)
    feature, index = _payloads(root, worklist, layer=produced_layer)
    conf = tmp_path / "petreloss.conf"
    conf.write_text("[fj]\naccess_key = fake-ak\nsecret_key = fake-sk\n")
    calls = tmp_path / "ads.calls"
    ads = tmp_path / "ads-cli"
    _fake_ads(ads, calls, fail=fail)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(PROJECT),
            "FEATURE_KIND": "FEAT_STREAMING",
            "FEATURE_NAMESPACE": NAMESPACE,
            "FEATURE_ROOT": str(root),
            "FEATURE_SCAN_ROOT": str(root),
            "VFM": "wan",
            "T": "749",
            "LAYERS": " ".join(str(layer) for layer in layers),
            "EXPECTED_MODEL_ID": MODEL_ID,
            "EXPECTED_PROMPT": ".",
            "EXPECTED_HEIGHT": "480",
            "EXPECTED_WIDTH": "832",
            "EXPECTED_NUM_FRAMES": "81",
            "EXPECTED_PROTOCOL": CANONICAL_PROTOCOL,
            "EXPECTED_OBS_LEN": "28",
            "EXTRACT_DONE_INDEX": str(done),
            "LOG_DIR": str(log),
            "STOP_FILE": str(stop),
            "ADS_CLI": str(ads),
            "CONF": str(conf),
            "SYNC_PYTHON": sys.executable,
            "SYNC_RETRY_SEC": "0",
            "SYNC_FINAL_MAX_ATTEMPTS": "1",
            "SYNC_POLL_SEC": "0",
            "CANONICAL_GAP_WORKLIST": str(worklist),
            "CANONICAL_GAP_WORKLIST_SHA256": sha256_file(worklist),
            "GAP_SHARD_PATH": str(shard_path),
            "GAP_SHARD_FILE_SHA256": sha256_file(shard_path),
            "GAP_SHARD_CONTENT_SHA256": shard["content_sha256"],
            "GAP_REMOTE_RECEIPT": str(receipt),
            "GAP_MODE": "streaming_prefix",
            "GAP_PROTOCOL_SHA256": PROTOCOL_SHA,
            "GAP_ADAPTER": str(ADAPTER),
            "EXPECTED_GAP_ADAPTER_SHA256": sha256_file(ADAPTER),
        }
    )
    result = subprocess.run(
        ["bash", str(SYNC)],
        cwd=PROJECT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )
    return result, {
        "root": root,
        "feature": feature,
        "index": index,
        "done": done,
        "receipt": receipt,
        "calls": calls,
        "worklist": worklist,
        "shard": shard,
        "env": env,
    }


def _run_target(tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], dict]:
    root = tmp_path / "target-payload"
    log = tmp_path / "target-logs"
    log.mkdir()
    stop = log / "extract.finished"
    stop.touch()
    done = tmp_path / "target-state/done.txt"
    receipt = tmp_path / "target-state/remote.json"
    worklist = tmp_path / "target-worklist.json"
    worklist_value = _target_worklist(worklist)
    shard_path = tmp_path / "target-shard.json"
    shard = _shard(shard_path, worklist, worklist_value)
    feature, index = _target_payloads(root, worklist)
    conf = tmp_path / "target-petreloss.conf"
    conf.write_text("[fj]\naccess_key = fake-ak\nsecret_key = fake-sk\n")
    calls = tmp_path / "target-ads.calls"
    ads = tmp_path / "target-ads-cli"
    _fake_ads(ads, calls, fail=False)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(PROJECT),
            "FEATURE_KIND": "FEAT_TARGET",
            "FEATURE_NAMESPACE": TARGET_NAMESPACE,
            "FEATURE_ROOT": str(root),
            "FEATURE_SCAN_ROOT": str(root),
            "VFM": "wan",
            "T": "749",
            "LAYERS": "5",
            "EXPECTED_MODEL_ID": MODEL_ID,
            "EXPECTED_PROMPT": ".",
            "EXPECTED_HEIGHT": "480",
            "EXPECTED_WIDTH": "832",
            "EXPECTED_NUM_FRAMES": "81",
            "EXPECTED_PROTOCOL": CANONICAL_PROTOCOL,
            "EXPECTED_OBS_LEN": "28",
            "EXTRACT_DONE_INDEX": str(done),
            "LOG_DIR": str(log),
            "STOP_FILE": str(stop),
            "ADS_CLI": str(ads),
            "CONF": str(conf),
            "SYNC_PYTHON": sys.executable,
            "SYNC_RETRY_SEC": "0",
            "SYNC_FINAL_MAX_ATTEMPTS": "1",
            "SYNC_POLL_SEC": "0",
            "CANONICAL_GAP_WORKLIST": str(worklist),
            "CANONICAL_GAP_WORKLIST_SHA256": sha256_file(worklist),
            "GAP_SHARD_PATH": str(shard_path),
            "GAP_SHARD_FILE_SHA256": sha256_file(shard_path),
            "GAP_SHARD_CONTENT_SHA256": shard["content_sha256"],
            "GAP_REMOTE_RECEIPT": str(receipt),
            "GAP_MODE": "target_isolated",
            "GAP_PROTOCOL_SHA256": PROTOCOL_SHA,
            "GAP_ADAPTER": str(ADAPTER),
            "EXPECTED_GAP_ADAPTER_SHA256": sha256_file(ADAPTER),
        }
    )
    result = subprocess.run(
        ["bash", str(SYNC)],
        cwd=PROJECT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )
    return result, {
        "feature": feature,
        "index": index,
        "done": done,
        "receipt": receipt,
        "worklist": worklist,
        "shard": shard,
    }


def test_gap_sync_uploads_verifies_receipts_then_deletes_and_receipt_resumes(
    tmp_path: Path, monkeypatch
) -> None:
    result, state = _run(tmp_path)
    assert result.returncode == 0, result.stdout
    assert not state["feature"].exists()
    assert not Path(f"{state['feature']}.manifest.json").exists()
    assert state["index"].exists()
    assert Path(f"{state['index']}.manifest.json").exists()
    assert len(state["done"].read_text().splitlines()) == 2
    remote = load_verified_receipt(
        state["receipt"],
        worklist_file_sha256=sha256_file(state["worklist"]),
        shard_content_sha256=state["shard"]["content_sha256"],
    )
    assert remote["completed_scenes"] == ["scene-a"]
    assert len(remote["objects"]) == 2

    exclusion = tmp_path / "completed-scenes.json"
    completed, exclusion_sha = _bind_completed_scenes(
        state["shard"], state["receipt"], exclusion
    )
    assert completed == {"scene-a"}
    monkeypatch.setenv("CANONICAL_GAP_COMPLETED_SCENES_FILE", str(exclusion))
    monkeypatch.setenv(
        "CANONICAL_GAP_COMPLETED_SCENES_FILE_SHA256", str(exclusion_sha)
    )
    filtered = CanonicalGapWorklist.load(
        state["worklist"],
        expected_sha256=sha256_file(state["worklist"]),
        mode="streaming_prefix",
        vfm="wan",
        model_id=MODEL_ID,
        timestep=749,
        feature_namespace=NAMESPACE,
        output_layers=[5],
        protocol_sha256=PROTOCOL_SHA,
        actual_prompt=".",
        preprocess_identity=streaming_worklist_preprocess_identity(
            "wan", [480, 832], 81
        ),
    )
    assert filtered.scenes == frozenset()
    assert filtered.stream_layers("scene-a", 0, 8) == frozenset()

    monkeypatch.delenv("CANONICAL_GAP_COMPLETED_SCENES_FILE")
    monkeypatch.delenv("CANONICAL_GAP_COMPLETED_SCENES_FILE_SHA256")
    _payloads(state["root"], state["worklist"])
    second = subprocess.run(
        ["bash", str(SYNC)],
        cwd=PROJECT,
        env=state["env"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )
    assert second.returncode == 0, second.stdout
    assert len(state["calls"].read_text().splitlines()) == 1
    assert not state["feature"].exists()
    assert state["index"].exists()


def test_gap_sync_publishes_one_complete_streaming_unit_without_whole_scene(
    tmp_path: Path,
) -> None:
    result, state = _run(tmp_path, layers=[5, 6], produced_layer=5)
    assert result.returncode == 0, result.stdout
    assert "incomplete_gap_scene" not in result.stdout
    assert not state["feature"].exists()
    assert state["index"].exists()

    remote = load_verified_receipt(
        state["receipt"],
        worklist_file_sha256=sha256_file(state["worklist"]),
        shard_content_sha256=state["shard"]["content_sha256"],
    )
    completed = set(remote["completed_units"])
    assert completed == {"streaming_prefix/scene-a/window_0000/prefix_08/layer_5"}
    assert remote["completed_scenes"] == []
    raw = json.loads(state["worklist"].read_text(encoding="utf-8"))
    raw["_file_sha256"] = sha256_file(state["worklist"])
    assert expected_extraction_unit_keys(raw) == {
        "streaming_prefix/scene-a/window_0000/prefix_08/layer_5",
        "streaming_prefix/scene-a/window_0000/prefix_08/layer_6",
    }

    completed_file = tmp_path / "completed-units.json"
    units, file_sha = _bind_completed_units(state["shard"], state["receipt"], completed_file)
    assert units == completed
    assert file_sha == sha256_file(completed_file)


def test_gap_sync_upload_failure_retains_every_payload_and_has_no_receipt(
    tmp_path: Path,
) -> None:
    result, state = _run(tmp_path, fail=True)
    assert result.returncode != 0, result.stdout
    assert "upload coverage not proven" in result.stdout
    assert state["feature"].exists()
    assert Path(f"{state['feature']}.manifest.json").exists()
    assert state["index"].exists()
    assert Path(f"{state['index']}.manifest.json").exists()
    assert state["done"].read_text() == ""
    assert not state["receipt"].exists()


def test_target_gap_delta_index_feature_sidecars_share_one_verified_transaction(
    tmp_path: Path,
) -> None:
    result, state = _run_target(tmp_path)
    assert result.returncode == 0, result.stdout
    assert not state["feature"].exists()
    assert not Path(f"{state['feature']}.manifest.json").exists()
    assert not state["index"].exists()
    assert not Path(f"{state['index']}.manifest.json").exists()
    remote = load_verified_receipt(
        state["receipt"],
        worklist_file_sha256=sha256_file(state["worklist"]),
        shard_content_sha256=state["shard"]["content_sha256"],
    )
    assert remote["completed_scenes"] == ["scene-a"]
    assert len(remote["objects"]) == 2
    assert len(state["done"].read_text().splitlines()) == 2
