from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_tstep_canonical_alignment import (
    DEFAULT_PROTOCOL_SHA256,
    DEFAULT_PROTOCOL_ID,
    FEATURE_GENERATION_CONTRACT,
    NOISE_SEED_SCHEME,
    STREAMING_FEATURE_ROOT,
    TARGET_FEATURE_ROOT,
    audit_alignment,
    sha256_file,
    verify_audit_artifact,
    write_json_atomic,
)

pytestmark = pytest.mark.integration


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE = "scene_a"
VFM = "wan"
SOURCE = "scannetpp"
WINDOW = list(range(100, 128))
PROTOCOL_SHA = DEFAULT_PROTOCOL_SHA256
STREAMING_NAMESPACE = "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
TARGET_NAMESPACE = "FEAT_TARGET_CANONICAL_V2_SEEDED_V1"


def _write_npy_with_sidecar(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value)
    namespace = (
        STREAMING_NAMESPACE if STREAMING_NAMESPACE in path.parts else TARGET_NAMESPACE
    )
    sidecar = {
        "schema_version": 1,
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "canonical_protocol": DEFAULT_PROTOCOL_ID,
        "canonical_protocol_sha256": PROTOCOL_SHA,
        "canonical_obs_len": 28,
        "feature_namespace": namespace,
        "feature_generation_contract": FEATURE_GENERATION_CONTRACT,
        "noise_seed_scheme": NOISE_SEED_SCHEME,
        "streaming_feature_root": STREAMING_FEATURE_ROOT,
        "target_feature_root": TARGET_FEATURE_ROOT,
    }
    Path(f"{path}.manifest.json").write_text(json.dumps(sidecar) + "\n")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest = tmp_path / "manifest.jsonl"
    row = {
        "sample_id": "scene_a_w000_p8",
        "scene_id": SCENE,
        "source": SOURCE,
        "split": "train",
        "streaming_window_id": 0,
        "streaming_tail": 7,
        "streaming_tail_raw": WINDOW[7],
        "streaming_indices": WINDOW[:8],
        "window_indices": WINDOW,
        "prefix_length": 8,
        "future_horizon": 1,
        "canonical_protocol": DEFAULT_PROTOCOL_ID,
        "canonical_protocol_sha256": PROTOCOL_SHA,
        "canonical_obs_len": 28,
        "streaming_feature_root": STREAMING_FEATURE_ROOT,
        "target_feature_root": TARGET_FEATURE_ROOT,
    }
    manifest.write_text(json.dumps(row) + "\n")
    manifest_meta = {
        "manifest_version": 2,
        "canonical_protocol": DEFAULT_PROTOCOL_ID,
        "canonical_obs_len": 28,
        "prefix_length": 8,
        "action_horizons": [1, 2, 4],
        "num_samples": 1,
        "num_scenes": 1,
        "sha256": sha256_file(manifest),
        "streaming_feature_root": STREAMING_FEATURE_ROOT,
        "target_feature_root": TARGET_FEATURE_ROOT,
    }
    Path(f"{manifest}.meta.json").write_text(json.dumps(manifest_meta) + "\n")

    stream_root = tmp_path / STREAMING_NAMESPACE
    prefix_path = stream_root / VFM / SOURCE / SCENE / "prefix_index.npy"
    record = {
        "window_id": 0,
        "tail": 7,
        "prefix_length": 8,
        "indices": WINDOW[:8],
        "window_indices": WINDOW,
        "valid_length": 8,
        "canonical_protocol": DEFAULT_PROTOCOL_ID,
        "canonical_protocol_sha256": PROTOCOL_SHA,
        "canonical_obs_len": 28,
    }
    _write_npy_with_sidecar(prefix_path, np.array([record], dtype=object))

    target_root = tmp_path / TARGET_NAMESPACE
    target_path = target_root / VFM / SOURCE / SCENE / "target_indices.npy"
    targets = sorted(
        {
            WINDOW[prefix - 1 + offset]
            for prefix in (8, 12, 16, 24)
            for offset in (0, 1, 2, 4)
        }
    )
    _write_npy_with_sidecar(target_path, np.array(targets, dtype=np.int64))
    target_meta = {
        "schema_version": 2,
        "protocol": DEFAULT_PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA,
        "obs_len": 28,
        "prefixes": [8, 12, 16, 24],
        "horizons": [1, 2, 4],
        "include_anchors": True,
        "target_count": len(targets),
        "target_indices_sha256": sha256_file(target_path),
        "feature_namespace": TARGET_NAMESPACE,
        "feature_generation_contract": FEATURE_GENERATION_CONTRACT,
        "noise_seed_scheme": NOISE_SEED_SCHEME,
    }
    target_path.with_name("target_indices.meta.json").write_text(json.dumps(target_meta) + "\n")
    return manifest, stream_root, target_root


def test_full_strict_audit_passes_and_binds_manifest(tmp_path: Path) -> None:
    manifest, stream_root, target_root = _fixture(tmp_path)
    report = audit_alignment(
        manifest=manifest,
        streaming_index_root=stream_root,
        target_index_root=target_root,
        vfm=VFM,
    )
    assert report["ok"] is True
    assert report["pass"] is True
    assert report["scope"] == "full"
    assert report["samples_checked"] == report["manifest_sample_count"] == 1
    assert report["target_memberships_checked"] == 4

    artifact = tmp_path / "audit.json"
    write_json_atomic(artifact, report)
    assert artifact.stat().st_mode & 0o777 == 0o644
    assert verify_audit_artifact(
        artifact=artifact,
        manifest=manifest,
        vfm=VFM,
        source=SOURCE,
        protocol_id=DEFAULT_PROTOCOL_ID,
        canonical_obs_len=28,
        prefix=8,
        expected_artifact_sha256=sha256_file(artifact),
    ) == []

    manifest.write_text(manifest.read_text() + "\n")
    errors = verify_audit_artifact(
        artifact=artifact,
        manifest=manifest,
        vfm=VFM,
        source=SOURCE,
        protocol_id=DEFAULT_PROTOCOL_ID,
        canonical_obs_len=28,
        prefix=8,
    )
    assert "audit manifest SHA256 is stale or mismatched" in errors


def test_strict_audit_detects_prefix_and_target_mismatches(tmp_path: Path) -> None:
    manifest, stream_root, target_root = _fixture(tmp_path)
    prefix_path = stream_root / VFM / SOURCE / SCENE / "prefix_index.npy"
    records = np.load(prefix_path, allow_pickle=True).tolist()
    records[0]["indices"][-1] = 999
    _write_npy_with_sidecar(prefix_path, np.array(records, dtype=object))

    target_path = target_root / VFM / SOURCE / SCENE / "target_indices.npy"
    targets = np.load(target_path)
    targets = targets[targets != WINDOW[11]]  # p8, h4
    _write_npy_with_sidecar(target_path, targets)
    meta_path = target_path.with_name("target_indices.meta.json")
    meta = json.loads(meta_path.read_text())
    meta["target_indices_sha256"] = sha256_file(target_path)
    meta_path.write_text(json.dumps(meta) + "\n")

    report = audit_alignment(
        manifest=manifest,
        streaming_index_root=stream_root,
        target_index_root=target_root,
        vfm=VFM,
    )
    assert report["ok"] is False
    assert report["problem_counts"]["prefix_indices_mismatch"] == 1
    assert report["problem_counts"]["target_frame_missing"] == 1


def test_strict_audit_rejects_old_namespace_and_noise_contract(tmp_path: Path) -> None:
    manifest, stream_root, target_root = _fixture(tmp_path)
    prefix_sidecar = Path(
        f"{stream_root / VFM / SOURCE / SCENE / 'prefix_index.npy'}.manifest.json"
    )
    prefix_meta = json.loads(prefix_sidecar.read_text())
    prefix_meta["feature_namespace"] = "FEAT_STREAMING_CANONICAL_V2"
    prefix_sidecar.write_text(json.dumps(prefix_meta) + "\n")

    target_meta_path = (
        target_root / VFM / SOURCE / SCENE / "target_indices.meta.json"
    )
    target_meta = json.loads(target_meta_path.read_text())
    target_meta["noise_seed_scheme"] = "legacy_global_rng"
    target_meta_path.write_text(json.dumps(target_meta) + "\n")

    report = audit_alignment(
        manifest=manifest,
        streaming_index_root=stream_root,
        target_index_root=target_root,
        vfm=VFM,
    )
    assert report["ok"] is False
    assert report["problem_counts"]["prefix_index_sidecar_feature_namespace"] == 1
    assert report["problem_counts"]["target_meta_noise_seed_scheme"] == 1


def test_canary_pass_is_not_accepted_as_full_gate(tmp_path: Path) -> None:
    manifest, stream_root, target_root = _fixture(tmp_path)
    report = audit_alignment(
        manifest=manifest,
        streaming_index_root=stream_root,
        target_index_root=target_root,
        vfm=VFM,
        scene_ids=[SCENE],
    )
    assert report["ok"] is True
    assert report["scope"] == "canary"
    artifact = tmp_path / "canary.json"
    write_json_atomic(artifact, report)
    errors = verify_audit_artifact(
        artifact=artifact,
        manifest=manifest,
        vfm=VFM,
        source=SOURCE,
        protocol_id=DEFAULT_PROTOCOL_ID,
        canonical_obs_len=28,
        prefix=8,
    )
    assert any("scope" in error for error in errors)


def test_matrix_does_not_mark_scene_level_target_files_ready(tmp_path: Path) -> None:
    manifest, stream_root, target_root = _fixture(tmp_path)
    # Remove semantic provenance while retaining the scene-level target index.
    (target_root / VFM / SOURCE / SCENE / "target_indices.meta.json").unlink()
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "inscene_c_h124_prefix8.jsonl").write_bytes(manifest.read_bytes())

    done_dir = tmp_path / "done"
    done_dir.mkdir()
    for timestep in (499, 999):
        (done_dir / f"{STREAMING_NAMESPACE}_wan_t{timestep}.txt").write_text(
            f"{SCENE}/window_0000/prefix_000007/feature_t{timestep}_layer15.sft\n"
        )
        (done_dir / f"{TARGET_NAMESPACE}_wan_t{timestep}.txt").write_text(
            f"{SCENE}/feature_t{timestep}_layer15.sft\n"
        )

    output_dir = tmp_path / "matrix"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/build_tstep_probe_matrix.py"),
        "--output-dir",
        str(output_dir),
        "--done-index-dir",
        str(done_dir),
        "--manifest-dir",
        str(manifest_dir),
        "--strict-audit-dir",
        str(tmp_path / "audits"),
        "--streaming-index-root",
        str(stream_root),
        "--target-index-root",
        str(target_root),
        "--vfms",
        "wan",
        "--layers",
        "15",
        "--prefixes",
        "8",
        "--streaming-exclusions",
        str(tmp_path / "no-exclusions.txt"),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    with (output_dir / "groups.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"wait_strict_audit"}
    assert all(row["strict_audit_ok"] == "false" for row in rows)
    assert all(not row["tasks"] for row in rows)


def test_runner_requires_explicit_strict_audit_env(tmp_path: Path) -> None:
    groups = tmp_path / "groups.tsv"
    fields = [
        "group_id",
        "status",
        "vfm",
        "source",
        "timestep",
        "layer",
        "prefix",
        "manifest",
        "tasks",
        "canonical_protocol",
        "canonical_obs_len",
        "streaming_namespace",
        "target_namespace",
        "strict_audit",
        "strict_audit_sha256",
        "strict_audit_ok",
    ]
    with groups.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "group_id": "wan_t499_l15_p8",
                "status": "ready_full",
                "vfm": "wan",
                "source": SOURCE,
                "timestep": 499,
                "layer": 15,
                "prefix": 8,
                "manifest": str(tmp_path / "manifest.jsonl"),
                "tasks": "streaming_depth",
                "canonical_protocol": DEFAULT_PROTOCOL_ID,
                "canonical_obs_len": 28,
                "streaming_namespace": STREAMING_NAMESPACE,
                "target_namespace": TARGET_NAMESPACE,
                "strict_audit": str(tmp_path / "audit.json"),
                "strict_audit_sha256": "0" * 64,
                "strict_audit_ok": "true",
            }
        )
    env = os.environ.copy()
    env.update(
        {
            "GROUPS_TSV": str(groups),
            "GROUP_ID": "wan_t499_l15_p8",
            "RUN_ID": "strict_gate_test",
        }
    )
    env.pop("INSCENE_STRICT_AUDIT", None)
    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts/run_tstep_probe_group.sh")],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3
    assert "set INSCENE_STRICT_AUDIT" in result.stderr
