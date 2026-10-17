from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.verify_tstep_canonical_v2_canary_receipt import (
    HORIZONS,
    LAYERS,
    OBS_LEN,
    PREFIXES,
    PROTOCOL_ID,
    PROTOCOL_SHA256,
    DETERMINISTIC_NOISE_CONTRACT,
    SCENE_ID,
    SOURCE_PATHS,
    STREAMING_NAMESPACE,
    TARGET_NAMESPACE,
    ReceiptError,
    build_receipt,
    create_start_snapshot,
    expected_stream_keys,
    expected_target_keys,
    sha256_file,
    verify_receipt,
)


def _write(path: Path, text: str, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if mode is not None:
        path.chmod(mode)


def _protocol_spec(path: Path) -> None:
    value = {
        "canonical_protocol": PROTOCOL_ID,
        "canonical_protocol_sha256": PROTOCOL_SHA256,
        "canonical_window": {"observations": OBS_LEN},
        "streaming": {"prefix_lengths": list(PREFIXES)},
        "target": {"future_horizons": list(HORIZONS), "include_anchors": True},
        "feature_namespaces": {
            "streaming": STREAMING_NAMESPACE,
            "target": TARGET_NAMESPACE,
        },
        "feature_generation": DETERMINISTIC_NOISE_CONTRACT,
    }
    _write(path, json.dumps(value) + "\n")


def _manifest(manifest_dir: Path, prefix: int) -> Path:
    path = manifest_dir / f"inscene_c_h124_prefix{prefix}.jsonl"
    rows = []
    for window in range(4):
        indices = list(range(window * 100, window * 100 + OBS_LEN))
        rows.append(
            {
                "sample_id": f"{SCENE_ID}_w{window:03d}_p{prefix}",
                "scene_id": SCENE_ID,
                "source": "scannetpp",
                "streaming_window_id": window,
                "streaming_tail": prefix - 1,
                "streaming_indices": indices[:prefix],
                "streaming_window_indices": indices,
                "canonical_protocol": PROTOCOL_ID,
                "canonical_obs_len": OBS_LEN,
                "streaming_feature_root": (
                    "aoss://baiqiao/InsScene-15K/"
                    "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
                ),
                "target_feature_root": (
                    "aoss://baiqiao/InsScene-15K/"
                    "FEAT_TARGET_CANONICAL_V2_SEEDED_V1"
                ),
            }
        )
    _write(path, "".join(json.dumps(row) + "\n" for row in rows))
    meta = {
        "manifest_version": 2,
        "canonical_protocol": PROTOCOL_ID,
        "canonical_obs_len": OBS_LEN,
        "canonical_protocol_sha256": PROTOCOL_SHA256,
        "streaming_namespace": STREAMING_NAMESPACE,
        "target_namespace": TARGET_NAMESPACE,
        "feature_generation_contract": DETERMINISTIC_NOISE_CONTRACT["contract"],
        "prefix_length": prefix,
        "action_horizons": list(HORIZONS),
        "num_samples": 4,
        "num_scenes": 1,
        "sha256": sha256_file(path),
        "streaming_feature_root": (
            "aoss://baiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
        ),
        "target_feature_root": (
            "aoss://baiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_SEEDED_V1"
        ),
    }
    _write(Path(f"{path}.meta.json"), json.dumps(meta) + "\n")
    return path


def _audit(receipt_dir: Path, manifest: Path, prefix: int) -> Path:
    path = receipt_dir / "audit" / f"wan_scene_{SCENE_ID}_p{prefix}.json"
    report = {
        "ok": True,
        "pass": True,
        "status": "pass",
        "problem_count": 0,
        "problem_counts": {},
        "scope": "canary",
        "protocol_id": PROTOCOL_ID,
        "protocol": PROTOCOL_ID,
        "canonical_obs_len": OBS_LEN,
        "canonical_protocol_sha256": PROTOCOL_SHA256,
        "streaming_namespace": STREAMING_NAMESPACE,
        "target_namespace": TARGET_NAMESPACE,
        "feature_generation_contract": DETERMINISTIC_NOISE_CONTRACT["contract"],
        "vfm": "wan",
        "source": "scannetpp",
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "manifest_streaming_feature_root": (
            "aoss://baiqiao/InsScene-15K/"
            "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
        ),
        "manifest_target_feature_root": (
            "aoss://baiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_SEEDED_V1"
        ),
        "manifest_sample_count": 4,
        "manifest_scene_count": 1,
        "selected_scenes": [SCENE_ID],
        "samples_checked": 4,
        "scenes_checked": 1,
        "target_memberships_checked": 16,
        "prefixes": list(PREFIXES),
        "horizons": list(HORIZONS),
    }
    _write(path, json.dumps(report) + "\n", 0o644)
    return path


def _successful_logs(receipt_dir: Path) -> None:
    for mode in ("stream", "target"):
        _write(
            receipt_dir / mode / "sync_loop.log",
            "round=0001 ads_rc=0 Found: 2, copied: 2, failed: 0\n"
            "sync loop complete; no matching local pairs or temps remain\n",
        )
        _write(
            receipt_dir / mode / "supervisor.log",
            "H800 gate PASS: 1 visible CUDA device(s): ['NVIDIA H800']\n"
            "finished_at=2026-08-30T00:00:00+00:00 extract_fail=0 sync_fail=0\n"
            "local_matching_sft_left=0\nOK\n",
        )


def _setup(tmp_path: Path) -> dict[str, Path]:
    project = tmp_path / "project"
    protocol = project / "protocols/inscene_streaming_canonical_v2.json"
    _protocol_spec(protocol)
    for name, relative in SOURCE_PATHS.items():
        _write(project / relative, f"fixture source {name}\n")
    manifests = project / "artifacts/tstep_canonical_v2/manifests"
    manifest_paths = {prefix: _manifest(manifests, prefix) for prefix in PREFIXES}
    run_root = project / "logs/canary_run"
    receipt_dir = run_root / "canary"
    receipt_dir.mkdir(parents=True)
    done_dir = run_root / "aoss_done"
    done_dir.mkdir()
    _write(done_dir / f"{STREAMING_NAMESPACE}_wan_t499.txt", "")
    _write(done_dir / f"{TARGET_NAMESPACE}_wan_t499.txt", "")
    dataset = project / "dataset"
    (dataset / STREAMING_NAMESPACE / "wan/scannetpp").mkdir(parents=True)
    (dataset / TARGET_NAMESPACE / "wan/scannetpp").mkdir(parents=True)
    snapshot = receipt_dir / ".CANARY_INPUTS_START.json"
    create_start_snapshot(
        output=snapshot,
        project_root=project,
        protocol_spec=protocol,
        manifest_dir=manifests,
        done_dir=done_dir,
        dataset_root=dataset,
        scene_id=SCENE_ID,
    )
    _write(
        done_dir / f"{STREAMING_NAMESPACE}_wan_t499.txt",
        "".join(f"{key}\n" for key in sorted(expected_stream_keys(SCENE_ID))),
    )
    _write(
        done_dir / f"{TARGET_NAMESPACE}_wan_t499.txt",
        "".join(f"{key}\n" for key in sorted(expected_target_keys(SCENE_ID))),
    )
    for prefix, manifest in manifest_paths.items():
        _audit(receipt_dir, manifest, prefix)
    _successful_logs(receipt_dir)
    return {
        "project": project,
        "protocol": protocol,
        "manifests": manifests,
        "run_root": run_root,
        "receipt_dir": receipt_dir,
        "done_dir": done_dir,
        "dataset": dataset,
        "snapshot": snapshot,
        "receipt": receipt_dir / "CANARY_RECEIPT.json",
    }


def _build(paths: dict[str, Path]) -> None:
    build_receipt(
        receipt=paths["receipt"],
        start_snapshot=paths["snapshot"],
        project_root=paths["project"],
        protocol_spec=paths["protocol"],
        manifest_dir=paths["manifests"],
        done_dir=paths["done_dir"],
        dataset_root=paths["dataset"],
        scene_id=SCENE_ID,
    )


def _verify(paths: dict[str, Path]) -> dict[str, object]:
    return verify_receipt(
        receipt=paths["receipt"],
        project_root=paths["project"],
        protocol_spec=paths["protocol"],
        manifest_dir=paths["manifests"],
        done_dir=paths["done_dir"],
        dataset_root=paths["dataset"],
        expected_scene=SCENE_ID,
    )


def test_build_and_verify_machine_receipt(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    _build(paths)
    bindings = _verify(paths)
    assert bindings["stream_keys"] == 48
    assert bindings["target_keys"] == 3
    assert bindings["local_sft"] == 0
    assert paths["receipt"].stat().st_mode & 0o777 == 0o644
    raw = paths["receipt"].read_bytes()
    assert raw.endswith(b"\n")
    assert b"CANARY_PASS" not in raw


def test_audit_must_be_readable_0644(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    audit = paths["receipt_dir"] / "audit" / f"wan_scene_{SCENE_ID}_p8.json"
    audit.chmod(0o600)
    with pytest.raises(ReceiptError, match="mode 0644"):
        _build(paths)


def test_failed_ads_summary_blocks_receipt(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    _write(
        paths["receipt_dir"] / "target/sync_loop.log",
        "round=0001 ads_rc=0 Found: 2, copied: 1, failed: 1\n"
        "round=0001 sync failed; retaining all local files\n",
    )
    with pytest.raises(ReceiptError, match="sync failed"):
        _build(paths)


def test_obsolete_sync_terminal_marker_blocks_receipt(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    _write(
        paths["receipt_dir"] / "target/sync_loop.log",
        "round=0001 ads_rc=0 Found: 9, copied: 9, failed: 0\n"
        "sync loop complete; no matching local pairs remain\n",
    )
    with pytest.raises(ReceiptError, match="lacks successful terminal marker"):
        _build(paths)


def test_invalid_marker_and_source_drift_block_verifier(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    _build(paths)
    _write(paths["receipt_dir"] / "INVALID_RERUN", "INVALID\n")
    with pytest.raises(ReceiptError, match="invalidated"):
        _verify(paths)
    (paths["receipt_dir"] / "INVALID_RERUN").unlink()
    with (paths["project"] / SOURCE_PATHS["extractor"]).open("a", encoding="utf-8") as handle:
        handle.write("drift\n")
    with pytest.raises(ReceiptError, match="stale"):
        _verify(paths)


def test_receipt_symlink_is_rejected(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    _build(paths)
    link = paths["receipt_dir"] / "receipt-link.json"
    os.symlink(paths["receipt"].name, link)
    paths["receipt"] = link
    with pytest.raises(ReceiptError, match="must not be a symlink"):
        _verify(paths)


def test_deterministic_noise_contract_tamper_is_rejected(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    spec = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    spec["feature_generation"]["seed_excludes"] = ["timestep"]
    paths["protocol"].write_text(json.dumps(spec) + "\n", encoding="utf-8")
    with pytest.raises(ReceiptError, match="does not implement"):
        create_start_snapshot(
            output=paths["receipt_dir"] / "tampered-start.json",
            project_root=paths["project"],
            protocol_spec=paths["protocol"],
            manifest_dir=paths["manifests"],
            done_dir=paths["done_dir"],
            dataset_root=paths["dataset"],
            scene_id=SCENE_ID,
        )


def test_canary_and_source_bindings_are_seeded_v1_h800_only() -> None:
    project = Path(__file__).resolve().parents[1]
    canary = (project / "scripts/run_tstep_canonical_v2_canary.sh").read_text()
    submit = (project / "scripts/submit_tstep_canonical_v2_canary_sco.sh").read_text()
    assert 'SCENE_START="${SCENE_START:-9}"' in canary
    assert 'EXPECTED_SCENE="${EXPECTED_SCENE:-033d0b9343}"' in canary
    assert "REQUIRE_H800=1" in canary
    assert '[[ "${AEC2_NAME}" == "h800" ]]' in canary
    assert '[[ "${AEC2_NAME}" == "h800" ]]' in submit
    assert SOURCE_PATHS["noise_seed_helper"] == "features/noise_seed.py"
    assert SOURCE_PATHS["temporal_sampling"] == "vidfm3d/utils/temporal_sampling.py"
    assert SOURCE_PATHS["wan_i2v_feature"] == "features/wan/wan_feature_i2v.py"
    assert SOURCE_PATHS["cogvideox_feature"] == "features/cogvideox/cogvideox_feature.py"
    assert SOURCE_PATHS["cogvideox_i2v_feature"].endswith("cogvideox_feature_i2v.py")
    assert SOURCE_PATHS["minimax_h3_wrapper"].endswith("minimax_h3/extract_features.py")
