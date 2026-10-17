from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = PROJECT_ROOT / "scripts" / "run_tstep_extract_campaign_v2.sh"
SUBMIT = PROJECT_ROOT / "scripts" / "submit_tstep_extract_campaign_v2_sco.sh"
PROTOCOL = "inscene_streaming_canonical_v2_obs28_h124"
PROTOCOL_SHA = "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f"
FEATURE_GENERATION_CONTRACT = "inscene_deterministic_noise_v1"
NOISE_SEED_SCHEME = "inscene_noise_seed_v1"
STREAMING_NAMESPACE = "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
TARGET_NAMESPACE = "FEAT_TARGET_CANONICAL_V2_SEEDED_V1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _padded_file(path: Path, minimum: int, prefix: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    padding = "# padding\n" * ((minimum // 10) + 2)
    path.write_text(prefix + padding, encoding="utf-8")


def _fake_project(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    project = tmp_path / "project"
    dataset = tmp_path / "dataset"
    manifests = project / "manifests"
    protocol_spec = project / "protocol.json"

    _padded_file(project / "features/run_inscene15k.py", 50_000)
    _padded_file(project / "vidfm3d/utils/temporal_sampling.py", 5_000)
    _padded_file(project / "features/noise_seed.py", 5_000)
    _padded_file(project / "features/wan/wan_feature.py", 10_000)
    _padded_file(project / "features/wan/wan_feature_i2v.py", 10_000)
    _padded_file(project / "features/wan/extract_features.py", 5_000)
    _padded_file(project / "features/cogvideox/cogvideox_feature.py", 10_000)
    _padded_file(project / "features/cogvideox/cogvideox_feature_i2v.py", 10_000)
    _padded_file(project / "features/cogvideox/extract_features.py", 5_000)
    _padded_file(project / "features/minimax_h3/minimax_h3_feature.py", 10_000)
    _padded_file(project / "features/minimax_h3/extract_features.py", 1_000)
    _padded_file(
        project / "scripts/run_streaming_feat_extract_8gpu.sh",
        5_000,
        '#!/usr/bin/env bash\nmkdir -p "${LOG_DIR}"\nexit 0\n',
    )
    _padded_file(project / "scripts/run_tstep_sync_loop.sh", 5_000, "#!/usr/bin/env bash\n")
    shutil.copy2(CAMPAIGN, project / "scripts/run_tstep_extract_campaign_v2.sh")
    protocol_spec.write_text(
        json.dumps(
            {
                "canonical_protocol": PROTOCOL,
                "canonical_protocol_sha256": PROTOCOL_SHA,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifests.mkdir(parents=True)
    for prefix in (8, 12, 16, 24):
        rows: list[str] = []
        for number in range(681):
            scene = f"scene_{number:04d}"
            windows = 4 if number < 679 else (3 if number == 679 else 2)
            for window_id in range(windows):
                window = list(range(window_id * 28, (window_id + 1) * 28))
                rows.append(
                    json.dumps(
                        {
                            "scene_id": scene,
                            "source": "scannetpp",
                            "streaming_window_id": window_id,
                            "streaming_tail": prefix - 1,
                            "prefix_length": prefix,
                            "streaming_indices": window[:prefix],
                            "streaming_window_indices": window,
                            "canonical_protocol": PROTOCOL,
                            "canonical_obs_len": 28,
                            "streaming_feature_root": (
                                "aoss://baiqiao/InsScene-15K/"
                                f"{STREAMING_NAMESPACE}"
                            ),
                            "target_feature_root": (
                                "aoss://baiqiao/InsScene-15K/"
                                f"{TARGET_NAMESPACE}"
                            ),
                        },
                        separators=(",", ":"),
                    )
                )
        assert len(rows) == 2721
        (manifests / f"inscene_c_h124_prefix{prefix}.jsonl").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )

    env = os.environ.copy()
    env.update(
        {
            "PROJECT_ROOT": str(project),
            "DATASET_ROOT": str(dataset),
            "DONE_DIR": str(project / "done"),
            "CAMPAIGN_LOG_ROOT": str(project / "logs/campaign"),
            "CAMPAIGN_LOCK_FILE": str(project / "locks/tstep_canonical_v2_full_campaign.lock"),
            "PROTOCOL_SPEC": str(protocol_spec),
            "MANIFEST_DIR": str(manifests),
            "GATE_PYTHON": sys.executable,
            "CANONICAL_PROTOCOL": PROTOCOL,
            "AEC2_NAME": "h800",
            "STREAMING_WINDOW_OBS_LEN": "28",
            "STREAMING_FEATURE_NAMESPACE": STREAMING_NAMESPACE,
            "TARGET_FEATURE_NAMESPACE": TARGET_NAMESPACE,
            "SOURCE": "scannetpp",
            "STEPS": "wan_target_t499",
            "START_AT": "1",
            "DRY_RUN": "0",
            "EXPECTED_PROTOCOL_SPEC_SHA256": _sha256(protocol_spec),
            "EXPECTED_CAMPAIGN_RUNNER_SHA256": _sha256(
                project / "scripts/run_tstep_extract_campaign_v2.sh"
            ),
            "EXPECTED_EXTRACTOR_SHA256": _sha256(project / "features/run_inscene15k.py"),
            "EXPECTED_TEMPORAL_SAMPLING_SHA256": _sha256(
                project / "vidfm3d/utils/temporal_sampling.py"
            ),
            "EXPECTED_NOISE_SEED_HELPER_SHA256": _sha256(project / "features/noise_seed.py"),
            "EXPECTED_WAN_FEATURE_SHA256": _sha256(project / "features/wan/wan_feature.py"),
            "EXPECTED_WAN_I2V_FEATURE_SHA256": _sha256(project / "features/wan/wan_feature_i2v.py"),
            "EXPECTED_WAN_WRAPPER_SHA256": _sha256(project / "features/wan/extract_features.py"),
            "EXPECTED_COGVIDEOX_FEATURE_SHA256": _sha256(project / "features/cogvideox/cogvideox_feature.py"),
            "EXPECTED_COGVIDEOX_I2V_FEATURE_SHA256": _sha256(project / "features/cogvideox/cogvideox_feature_i2v.py"),
            "EXPECTED_COGVIDEOX_WRAPPER_SHA256": _sha256(project / "features/cogvideox/extract_features.py"),
            "EXPECTED_MINIMAX_H3_FEATURE_SHA256": _sha256(project / "features/minimax_h3/minimax_h3_feature.py"),
            "EXPECTED_MINIMAX_H3_WRAPPER_SHA256": _sha256(project / "features/minimax_h3/extract_features.py"),
            "EXPECTED_EXTRACT_RUNNER_SHA256": _sha256(
                project / "scripts/run_streaming_feat_extract_8gpu.sh"
            ),
            "EXPECTED_SYNC_LOOP_SHA256": _sha256(project / "scripts/run_tstep_sync_loop.sh"),
        }
    )
    for prefix in (8, 12, 16, 24):
        env[f"EXPECTED_MANIFEST_P{prefix}_SHA256"] = _sha256(
            manifests / f"inscene_c_h124_prefix{prefix}.jsonl"
        )
    return project, env


def _build_complete_target_fixture(project: Path, env: dict[str, str]) -> tuple[Path, Path]:
    done_index = (
        Path(env["DONE_DIR"]) / f"{TARGET_NAMESPACE}_wan_t499.txt"
    )
    done_index.parent.mkdir(parents=True)
    feature_root = (
        Path(env["DATASET_ROOT"])
        / TARGET_NAMESPACE
        / "wan/scannetpp"
    )
    lines: list[str] = []
    for number in range(681):
        scene = f"scene_{number:04d}"
        scene_root = feature_root / scene
        scene_root.mkdir(parents=True)
        index_path = scene_root / "target_indices.npy"
        windows = 4 if number < 679 else (3 if number == 679 else 2)
        targets = sorted(
            {
                window_id * 28 + prefix - 1 + offset
                for window_id in range(windows)
                for prefix in (8, 12, 16, 24)
                for offset in (0, 1, 2, 4)
            }
        )
        with index_path.open("wb") as handle:
            np.save(handle, np.array(targets, dtype=np.int64))
        index_hash = _sha256(index_path)
        Path(f"{index_path}.manifest.json").write_text(
            json.dumps(
                {
                    "file": index_path.name,
                    "size_bytes": index_path.stat().st_size,
                    "sha256": index_hash,
                    "canonical_protocol": PROTOCOL,
                    "canonical_protocol_sha256": PROTOCOL_SHA,
                    "canonical_obs_len": 28,
                    "feature_namespace": TARGET_NAMESPACE,
                    "feature_generation_contract": FEATURE_GENERATION_CONTRACT,
                    "noise_seed_scheme": NOISE_SEED_SCHEME,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (scene_root / "target_indices.meta.json").write_text(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "protocol_sha256": PROTOCOL_SHA,
                    "obs_len": 28,
                    "prefixes": [8, 12, 16, 24],
                    "horizons": [1, 2, 4],
                    "include_anchors": True,
                    "target_count": len(targets),
                    "target_indices_sha256": index_hash,
                    "feature_namespace": TARGET_NAMESPACE,
                    "feature_generation_contract": FEATURE_GENERATION_CONTRACT,
                    "noise_seed_scheme": NOISE_SEED_SCHEME,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        for layer in (15, 20, 25):
            lines.append(f"{scene}/feature_t499_layer{layer}.sft")
    done_index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return done_index, feature_root


def _build_complete_streaming_fixture(project: Path, env: dict[str, str]) -> Path:
    done_index = (
        Path(env["DONE_DIR"]) / f"{STREAMING_NAMESPACE}_wan_t499.txt"
    )
    done_index.parent.mkdir(parents=True, exist_ok=True)
    feature_root = (
        Path(env["DATASET_ROOT"])
        / STREAMING_NAMESPACE
        / "wan/scannetpp"
    )
    lines: list[str] = []
    for number in range(681):
        scene = f"scene_{number:04d}"
        scene_root = feature_root / scene
        scene_root.mkdir(parents=True)
        windows = 4 if number < 679 else (3 if number == 679 else 2)
        records: list[dict[str, object]] = []
        for window in range(windows):
            window_indices = list(range(window * 28, (window + 1) * 28))
            for tail in (7, 11, 15, 23):
                records.append(
                    {
                        "window_id": window,
                        "tail": tail,
                        "indices": window_indices[: tail + 1],
                        "window_indices": window_indices,
                        "canonical_protocol": PROTOCOL,
                        "canonical_protocol_sha256": PROTOCOL_SHA,
                        "canonical_obs_len": 28,
                    }
                )
                unit = f"{scene}/window_{window:04d}/prefix_{tail:06d}"
                for layer in (15, 20, 25):
                    lines.append(f"{unit}/feature_t499_layer{layer}.sft")
        index_path = scene_root / "prefix_index.npy"
        with index_path.open("wb") as handle:
            np.save(handle, np.array(records, dtype=object))
        Path(f"{index_path}.manifest.json").write_text(
            json.dumps(
                {
                    "file": index_path.name,
                    "size_bytes": index_path.stat().st_size,
                    "sha256": _sha256(index_path),
                    "canonical_protocol": PROTOCOL,
                    "canonical_protocol_sha256": PROTOCOL_SHA,
                    "canonical_obs_len": 28,
                    "feature_namespace": STREAMING_NAMESPACE,
                    "feature_generation_contract": FEATURE_GENERATION_CONTRACT,
                    "noise_seed_scheme": NOISE_SEED_SCHEME,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    assert len(lines) == 32_652
    done_index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return done_index


def _run_campaign(env: dict[str, str], log_suffix: str) -> subprocess.CompletedProcess[str]:
    run_env = dict(env)
    run_env["CAMPAIGN_LOG_ROOT"] = f'{env["CAMPAIGN_LOG_ROOT"]}_{log_suffix}'
    return subprocess.run(
        ["bash", str(CAMPAIGN)],
        cwd=PROJECT_ROOT,
        env=run_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=60,
    )


def test_target_step_requires_exact_keys_and_structural_coverage(tmp_path: Path) -> None:
    project, env = _fake_project(tmp_path)
    done_index, feature_root = _build_complete_target_fixture(project, env)

    passed = _run_campaign(env, "pass")
    assert passed.returncode == 0, passed.stdout
    assert (
        "COMPLETION_GATE PASS mode=target_isolated vfm=wan t=499 "
        "keys=2043 units=681 structural=681"
    ) in passed.stdout

    original = done_index.read_text(encoding="utf-8")
    done_index.write_text("\n".join(original.splitlines()[:-1]) + "\n", encoding="utf-8")
    partial = _run_campaign(env, "partial")
    assert partial.returncode != 0
    assert "done-index keys=2042, expected=2043" in partial.stdout

    done_index.write_text(original, encoding="utf-8")
    (feature_root / "scene_0000/target_indices.meta.json").unlink()
    missing_structure = _run_campaign(env, "missing_structure")
    assert missing_structure.returncode != 0
    assert "missing/non-regular target index metadata" in missing_structure.stdout


def test_completion_gate_rejects_old_namespace_provenance(tmp_path: Path) -> None:
    project, env = _fake_project(tmp_path)
    _done_index, feature_root = _build_complete_target_fixture(project, env)
    sidecar = feature_root / "scene_0000/target_indices.npy.manifest.json"
    value = json.loads(sidecar.read_text())
    value["feature_namespace"] = "FEAT_TARGET_CANONICAL_V2"
    sidecar.write_text(json.dumps(value) + "\n")

    result = _run_campaign(env, "old_namespace")
    assert result.returncode != 0
    assert "index sidecar generation mismatch" in result.stdout
    assert "feature_namespace" in result.stdout


def test_streaming_step_requires_all_10884_structural_units(tmp_path: Path) -> None:
    project, env = _fake_project(tmp_path)
    _build_complete_streaming_fixture(project, env)
    env["STEPS"] = "wan_stream_t499"

    passed = _run_campaign(env, "stream_pass")
    assert passed.returncode == 0, passed.stdout
    assert (
        "COMPLETION_GATE PASS mode=streaming_prefix vfm=wan t=499 "
        "keys=32652 units=10884 structural=681"
    ) in passed.stdout


def test_full_campaign_lock_is_fail_closed(tmp_path: Path) -> None:
    project, env = _fake_project(tmp_path)
    lock_path = Path(env["CAMPAIGN_LOCK_FILE"])
    lock_path.parent.mkdir(parents=True)
    with lock_path.open("a", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = _run_campaign(env, "locked")
    assert result.returncode != 0
    assert "another canonical-v2 extraction job holds" in result.stdout


def test_noise_and_model_sources_are_bound_to_canary_hashes(tmp_path: Path) -> None:
    project, env = _fake_project(tmp_path)
    path = project / "features/noise_seed.py"
    path.write_text(path.read_text(encoding="utf-8") + "# drift\n", encoding="utf-8")
    result = _run_campaign(env, "noise_drift")
    assert result.returncode != 0
    assert "noise_seed_helper changed after the accepted canary" in result.stdout


def test_8gpu_runner_exports_seeded_namespace_and_checks_visible_h800s() -> None:
    runner = PROJECT_ROOT / "scripts/run_streaming_feat_extract_8gpu.sh"
    text = runner.read_text(encoding="utf-8")
    assert "export FEATURE_NAMESPACE" in text
    assert 'REQUIRE_H800="${REQUIRE_H800:-0}"' in text
    assert "CANONICAL_PROTOCOL_SHA256" in text
    assert "torch.cuda.device_count()" in text
    assert '"H800" not in name' in text


def test_submit_pool_is_strictly_whitelisted() -> None:
    result = subprocess.run(
        ["bash", str(SUBMIT)],
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            "AEC2_NAME": "not-a-pool",
            "CANARY_RECEIPT_MANIFEST": "/does/not/matter.json",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 2
    assert "seeded-v1 full extraction must run on AEC2_NAME=h800" in result.stdout
    text = SUBMIT.read_text(encoding="utf-8")
    assert '--aec2-name="${AEC2_NAME}"' in text
    assert '--worker-spec="${WORKER_SPEC}"' in text
    assert "verify_tstep_canonical_v2_canary_receipt.py" in text
    assert "--expected-scene 033d0b9343" in text
    assert "CANARY_RECEIPT_SHA256" in text
