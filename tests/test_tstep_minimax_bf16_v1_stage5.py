"""Stage-5 gates for the MiniMax bf16 seeded-v1 namespaces (2026-10-16).

The strict audit, the matrix builder and the group runner accept the
``*_SEEDED_V1_MINIMAX_BF16_V1`` pair only for minimax_h3 layers 20/25, bind it
end to end (audit artifact -> groups.tsv -> AOSS roots), and keep the seeded-v1
behaviour unchanged.
"""

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
    DEFAULT_PROTOCOL_ID,
    DEFAULT_PROTOCOL_SHA256,
    FEATURE_GENERATION_CONTRACT,
    MINIMAX_BF16_STREAMING_NAMESPACE,
    MINIMAX_BF16_TARGET_NAMESPACE,
    MINIMAX_NOISE_SEED_SCHEME,
    STREAMING_FEATURE_ROOT,
    STREAMING_NAMESPACE,
    TARGET_FEATURE_ROOT,
    TARGET_NAMESPACE,
    audit_alignment,
    namespace_pair_errors,
    sha256_file,
    verify_audit_artifact,
    write_json_atomic,
)

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts/run_tstep_probe_group.sh"
# The bf16-aware runner is staged as scripts/run_tstep_probe_group.sh.stage5
# and installed only once no job is executing the old copy (bash re-reads a
# running script from disk); until then the runner tests are skipped.
runner_bf16 = pytest.mark.skipif(
    "seeded_v1_minimax_bf16_v1" not in RUNNER.read_text(encoding="utf-8"),
    reason="bf16-aware run_tstep_probe_group.sh not installed yet (staged as .stage5)",
)
SCENE = "scene_a"
VFM = "minimax_h3"
SOURCE = "scannetpp"
WINDOW = list(range(100, 128))
BF16 = (MINIMAX_BF16_STREAMING_NAMESPACE, MINIMAX_BF16_TARGET_NAMESPACE)
EIGHT_TASKS = (
    "streaming_depth streaming_mask view_consistency ego_belief ego_belief_v2 "
    "action_dynamics path_integration counterfactual"
)
MINIMAX_SEED_FIELDS = {
    "noise_seed_scheme": MINIMAX_NOISE_SEED_SCHEME,
    "noise_seed": 0,
    "noise_seed_scope": "reset_per_forward",
}


def _write_npy_with_sidecar(path: Path, value: np.ndarray, namespace: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value)
    sidecar = {
        "schema_version": 1,
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "canonical_protocol": DEFAULT_PROTOCOL_ID,
        "canonical_protocol_sha256": DEFAULT_PROTOCOL_SHA256,
        "canonical_obs_len": 28,
        "feature_namespace": namespace,
        "feature_generation_contract": FEATURE_GENERATION_CONTRACT,
        **MINIMAX_SEED_FIELDS,
    }
    Path(f"{path}.manifest.json").write_text(json.dumps(sidecar) + "\n")


def _fixture(tmp_path: Path, namespaces=BF16, sidecar_namespaces=None) -> tuple[Path, Path, Path]:
    """One-sample prefix-8 manifest plus local MiniMax index roots."""
    stream_ns, target_ns = namespaces
    side_stream, side_target = sidecar_namespaces or namespaces
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    manifest = manifest_dir / "inscene_c_h124_prefix8.jsonl"
    row = {
        "sample_id": "scene_a_w000_p8", "scene_id": SCENE, "source": SOURCE, "split": "train",
        "streaming_window_id": 0, "streaming_tail": 7, "streaming_tail_raw": WINDOW[7],
        "streaming_indices": WINDOW[:8], "window_indices": WINDOW, "prefix_length": 8,
        "future_horizon": 1, "canonical_protocol": DEFAULT_PROTOCOL_ID,
        "canonical_protocol_sha256": DEFAULT_PROTOCOL_SHA256, "canonical_obs_len": 28,
        # Frozen manifests always declare the seeded-v1 roots.
        "streaming_feature_root": STREAMING_FEATURE_ROOT, "target_feature_root": TARGET_FEATURE_ROOT,
    }
    manifest.write_text(json.dumps(row) + "\n")
    Path(f"{manifest}.meta.json").write_text(json.dumps({
        "manifest_version": 2, "canonical_protocol": DEFAULT_PROTOCOL_ID, "canonical_obs_len": 28,
        "prefix_length": 8, "action_horizons": [1, 2, 4], "num_samples": 1, "num_scenes": 1,
        "sha256": sha256_file(manifest), "streaming_feature_root": STREAMING_FEATURE_ROOT,
        "target_feature_root": TARGET_FEATURE_ROOT,
    }) + "\n")

    stream_root = tmp_path / stream_ns
    record = {
        "window_id": 0, "tail": 7, "prefix_length": 8, "indices": WINDOW[:8], "window_indices": WINDOW,
        "valid_length": 8, "canonical_protocol": DEFAULT_PROTOCOL_ID,
        "canonical_protocol_sha256": DEFAULT_PROTOCOL_SHA256, "canonical_obs_len": 28,
    }
    _write_npy_with_sidecar(stream_root / VFM / SOURCE / SCENE / "prefix_index.npy",
                            np.array([record], dtype=object), side_stream)

    target_root = tmp_path / target_ns
    target_path = target_root / VFM / SOURCE / SCENE / "target_indices.npy"
    targets = sorted({WINDOW[p - 1 + o] for p in (8, 12, 16, 24) for o in (0, 1, 2, 4)})
    _write_npy_with_sidecar(target_path, np.array(targets, dtype=np.int64), side_target)
    target_path.with_name("target_indices.meta.json").write_text(json.dumps({
        "schema_version": 2, "protocol": DEFAULT_PROTOCOL_ID, "protocol_sha256": DEFAULT_PROTOCOL_SHA256,
        "obs_len": 28, "prefixes": [8, 12, 16, 24], "horizons": [1, 2, 4], "include_anchors": True,
        "target_count": len(targets), "target_indices_sha256": sha256_file(target_path),
        "feature_namespace": side_target, "feature_generation_contract": FEATURE_GENERATION_CONTRACT,
        **MINIMAX_SEED_FIELDS,
    }) + "\n")
    return manifest, stream_root, target_root


def _verify(artifact: Path, manifest: Path, namespaces) -> list[str]:
    return verify_audit_artifact(
        artifact=artifact, manifest=manifest, vfm=VFM, source=SOURCE, protocol_id=DEFAULT_PROTOCOL_ID,
        canonical_obs_len=28, prefix=8, expected_artifact_sha256=sha256_file(artifact),
        streaming_namespace=namespaces[0], target_namespace=namespaces[1],
    )


def test_bf16_audit_passes_and_artifact_is_bound_to_its_namespaces(tmp_path: Path) -> None:
    manifest, stream_root, target_root = _fixture(tmp_path)
    report = audit_alignment(
        manifest=manifest, streaming_index_root=stream_root, target_index_root=target_root, vfm=VFM,
        streaming_namespace=BF16[0], target_namespace=BF16[1],
    )
    assert report["ok"] is True, report["problems"]
    assert (report["streaming_namespace"], report["target_namespace"]) == BF16
    # The manifest keeps its seeded-v1 roots; only the index namespace changes.
    assert report["manifest_streaming_feature_root"] == STREAMING_FEATURE_ROOT
    artifact = tmp_path / "audit.json"
    write_json_atomic(artifact, report)
    assert _verify(artifact, manifest, BF16) == []
    errors = _verify(artifact, manifest, (STREAMING_NAMESPACE, TARGET_NAMESPACE))
    assert any("streaming_namespace" in e for e in errors)
    assert any("target_namespace" in e for e in errors)


def test_bf16_audit_rejects_seeded_v1_sidecars_under_bf16_roots(tmp_path: Path) -> None:
    manifest, stream_root, target_root = _fixture(
        tmp_path, sidecar_namespaces=(STREAMING_NAMESPACE, TARGET_NAMESPACE)
    )
    report = audit_alignment(
        manifest=manifest, streaming_index_root=stream_root, target_index_root=target_root, vfm=VFM,
        streaming_namespace=BF16[0], target_namespace=BF16[1],
    )
    assert report["ok"] is False
    assert report["problem_counts"]["prefix_index_sidecar_feature_namespace"] == 1
    assert report["problem_counts"]["target_meta_feature_namespace"] == 1


def test_seeded_v1_default_audit_rejects_bf16_roots(tmp_path: Path) -> None:
    manifest, stream_root, target_root = _fixture(tmp_path)
    report = audit_alignment(
        manifest=manifest, streaming_index_root=stream_root, target_index_root=target_root, vfm=VFM,
    )
    assert report["ok"] is False
    assert report["problem_counts"]["streaming_index_root_namespace"] == 1
    assert report["problem_counts"]["target_index_root_namespace"] == 1


def test_namespace_pair_contract() -> None:
    assert namespace_pair_errors(STREAMING_NAMESPACE, TARGET_NAMESPACE, "wan", [15, 20, 25]) == []
    assert namespace_pair_errors(*BF16, "minimax_h3", [20, 25]) == []
    assert namespace_pair_errors(*BF16, "wan")
    assert namespace_pair_errors(*BF16, "minimax_h3", [15])
    assert namespace_pair_errors(STREAMING_NAMESPACE, BF16[1], "minimax_h3")
    assert namespace_pair_errors(BF16[0], TARGET_NAMESPACE, "minimax_h3")


def test_bf16_audit_rejects_other_vfm(tmp_path: Path) -> None:
    manifest, stream_root, target_root = _fixture(tmp_path)
    report = audit_alignment(
        manifest=manifest, streaming_index_root=stream_root, target_index_root=target_root, vfm="wan",
        streaming_namespace=BF16[0], target_namespace=BF16[1],
    )
    assert report["ok"] is False
    assert report["problem_counts"]["namespace_pair_invalid"] == 1


def _build_matrix(tmp_path: Path, *extra: str, layers: str = "20", vfms: str = VFM):
    manifest, stream_root, target_root = _fixture(tmp_path)
    done_dir = tmp_path / "done"
    done_dir.mkdir(exist_ok=True)
    for t in (500, 1000):
        for layer in (20, 25):
            with (done_dir / f"{BF16[0]}_{VFM}_t{t}.txt").open("a") as handle:
                handle.write(f"{SCENE}/window_0000/prefix_000007/feature_t{t}_layer{layer}.sft\n")
            with (done_dir / f"{BF16[1]}_{VFM}_t{t}.txt").open("a") as handle:
                handle.write(f"{SCENE}/feature_t{t}_layer{layer}.sft\n")
    output_dir = tmp_path / "matrix"
    command = [
        sys.executable, str(PROJECT_ROOT / "scripts/build_tstep_probe_matrix.py"),
        "--output-dir", str(output_dir), "--done-index-dir", str(done_dir),
        "--manifest-dir", str(manifest.parent), "--strict-audit-dir", str(tmp_path / "audits"),
        "--streaming-index-root", str(stream_root), "--target-index-root", str(target_root),
        "--streaming-namespace", BF16[0], "--target-namespace", BF16[1],
        "--vfms", vfms, "--layers", layers, "--prefixes", "8",
        "--streaming-exclusions", str(tmp_path / "no-exclusions.txt"), *extra,
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    return result, output_dir, manifest


def test_matrix_builds_bf16_groups_with_task_subset(tmp_path: Path) -> None:
    result, output_dir, _ = _build_matrix(tmp_path, "--tasks", EIGHT_TASKS)
    assert result.returncode == 0, result.stderr
    with (output_dir / "groups.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["group_id"] for row in rows] == ["minimax_h3_t500_l20_p8", "minimax_h3_t1000_l20_p8"]
    assert {row["status"] for row in rows} == {"ready_full"}
    assert {row["tasks"] for row in rows} == {EIGHT_TASKS.replace(" ", ",")}
    assert {(row["streaming_namespace"], row["target_namespace"]) for row in rows} == {BF16}
    audits = sorted(p.name for p in (tmp_path / "audits").iterdir())
    assert audits == ["minimax_h3_inscene_c_h124_prefix8.seeded_v1_minimax_bf16_v1.audit.json"]
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["namespace_pair_tag"] == "seeded_v1_minimax_bf16_v1"
    assert summary["training_runs_if_full"] == 16


@pytest.mark.parametrize("layers,vfms", [("15", VFM), ("20", "wan")])
def test_matrix_refuses_bf16_outside_minimax_l20_l25(tmp_path: Path, layers: str, vfms: str) -> None:
    result, _, _ = _build_matrix(tmp_path, layers=layers, vfms=vfms)
    assert result.returncode != 0
    assert "seeded_v1_minimax_bf16_v1" in result.stderr


def test_matrix_rejects_unknown_task(tmp_path: Path) -> None:
    result, _, _ = _build_matrix(tmp_path, "--tasks", "streaming_depth not_a_task")
    assert result.returncode != 0
    assert "--tasks" in result.stderr


def _run_group(tmp_path: Path, groups: Path, group_id: str, audit: str, **env_extra: str):
    env = os.environ.copy()
    env.pop("PROBE_FP16_PASSTHROUGH", None)
    env.update({
        "GROUPS_TSV": str(groups), "GROUP_ID": group_id, "RUN_ID": "bf16_stage5_test", "DRY_RUN": "1",
        "INSCENE_STRICT_AUDIT": audit, "PYTHON": sys.executable, "LOG_ROOT": str(tmp_path / "logroot"),
        "AOSS_CACHE_BASE": str(tmp_path / "cache"), "GPU_IDS_CSV": "0,1,2,3,4,5,6,7",
        "AOSS_CONF": str(tmp_path / "petreloss.conf"),
    })
    env.update(env_extra)
    (tmp_path / "petreloss.conf").write_text("[default]\n")
    return subprocess.run(["bash", str(RUNNER)], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)


@runner_bf16
def test_runner_dry_run_binds_bf16_roots_and_fp32_transfer(tmp_path: Path) -> None:
    result, output_dir, _ = _build_matrix(tmp_path, "--tasks", EIGHT_TASKS)
    assert result.returncode == 0, result.stderr
    groups = output_dir / "groups.tsv"
    with groups.open(newline="") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    run = _run_group(tmp_path, groups, row["group_id"], row["strict_audit"])
    assert run.returncode == 0, run.stdout + run.stderr
    assert f"stream_root=aoss://baiqiao/InsScene-15K/{BF16[0]}" in run.stdout
    assert f"target_root=aoss://baiqiao/InsScene-15K/{BF16[1]}" in run.stdout
    assert f"target_index_root=/mnt/umm/users/yinbaiqiao/InsScene-15K/{BF16[1]}" in run.stdout
    assert "namespace_family=seeded_v1_minimax_bf16_v1 probe_fp16_passthrough=0" in run.stdout
    logs = tmp_path / "logroot/groups" / row["group_id"]
    for task in EIGHT_TASKS.split():
        text = (logs / f"{task}.log").read_text()
        assert "[dry-run]" in text and "rc=0" in text
        assert f"streaming_feat_root=aoss://baiqiao/InsScene-15K/{BF16[0]}" in text
        assert "feature_layer=20" in text and "feature_timestep=500" in text
    assert f"target_feat_root=aoss://baiqiao/InsScene-15K/{BF16[1]}" in (logs / "action_dynamics.log").read_text()


@runner_bf16
def test_runner_refuses_bf16_with_fp16_passthrough(tmp_path: Path) -> None:
    result, output_dir, _ = _build_matrix(tmp_path, "--tasks", EIGHT_TASKS)
    assert result.returncode == 0, result.stderr
    groups = output_dir / "groups.tsv"
    with groups.open(newline="") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    run = _run_group(tmp_path, groups, row["group_id"], row["strict_audit"], PROBE_FP16_PASSTHROUGH="1")
    assert run.returncode == 3
    assert "require PROBE_FP16_PASSTHROUGH=0" in run.stderr


STAGED_RUNNER = PROJECT_ROOT / "scripts/run_tstep_probe_group.sh.stage5"
BF16_RUNNER = next((p for p in (RUNNER, STAGED_RUNNER)
                    if p.is_file() and "seeded_v1_minimax_bf16_v1" in p.read_text(encoding="utf-8")), None)


@pytest.mark.skipif(BF16_RUNNER is None, reason="no bf16-aware runner (installed or staged)")
def test_stage5_submitter_ready_path_plans_dry_runs_and_requires_approval(tmp_path: Path, monkeypatch) -> None:
    import shutil

    import scripts.submit_tstep_minimax_bf16_v1_retrain as stage5

    result, output_dir, _ = _build_matrix(tmp_path, "--tasks", EIGHT_TASKS)
    assert result.returncode == 0, result.stderr
    with (output_dir / "groups.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    def fake_build(out_dir: Path, audit_dir: Path) -> list[dict]:
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(output_dir / "groups.tsv", out_dir / "groups.tsv")
        return [dict(row) for row in rows]

    monkeypatch.setattr(stage5, "consolidation_ready", lambda: (True, "test: consolidation PASS"))
    monkeypatch.setattr(stage5, "config_parity", lambda: [])  # composes 128 real configs; run on its own
    monkeypatch.setattr(stage5, "build_matrix", fake_build)
    monkeypatch.setattr(stage5, "OUT_DIR", tmp_path / "campaign")
    monkeypatch.setattr(stage5, "GROUP_SCRIPT", BF16_RUNNER)
    monkeypatch.setattr(stage5, "SOURCE_MATRIX", output_dir / "groups.tsv")  # stands in for the 2026-09 matrix
    monkeypatch.setattr(stage5, "LAYERS", (20,))
    monkeypatch.setattr(stage5, "PREFIXES", (8,))
    for layout, jobs, spec in (("node8", 2, "N4lS.Iq.I80.8"), ("single", 16, "N4lS.Iq.I80.1")):
        monkeypatch.setattr(sys, "argv", ["submit", "--layout", layout])
        assert stage5.main() == 0
        plan = json.loads((tmp_path / f"campaign/launcher/plan_{layout}.json").read_text())
        assert plan["errors"] == []
        assert len(plan["jobs"]) == len(plan["dry_run"]) == jobs
        assert {job["worker_spec"] for job in plan["jobs"]} == {spec}
        assert all("PROBE_FP16_PASSTHROUGH=0" in job["command"] for job in plan["jobs"])
    monkeypatch.setattr(sys, "argv", ["submit", "--submit"])
    with pytest.raises(SystemExit, match="approval"):
        stage5.main()


def test_pilot_gate_requires_a_finite_checkpoint_for_every_probe(tmp_path: Path, monkeypatch) -> None:
    import torch

    import scripts.submit_tstep_minimax_bf16_v1_retrain as stage5

    monkeypatch.setattr(stage5, "RUNS_ROOT", tmp_path / "runs")
    jobs = {"minimax_h3_t500_l20_p8": {"tasks": ",".join(stage5.TASKS)}}

    def write(task: str, value: float) -> None:
        ckpt = tmp_path / "runs" / f"tp_minimax_h3_t500_l20_p8_{task}" / "VideoProbe3D/abc/checkpoints/epoch=0-step=1.ckpt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": 0, "state_dict": {"w": torch.full((3,), value), "step": torch.tensor(1)}}, ckpt)

    for task in stage5.TASKS[:-1]:
        write(task, 1.0)
    gate = stage5.pilot_gate(jobs)
    assert gate["status"] == "FAIL" and "counterfactual: no checkpoint yet" in gate["problems"][0]
    write(stage5.TASKS[-1], float("nan"))
    gate = stage5.pilot_gate(jobs)
    assert gate["status"] == "FAIL" and "3 non-finite of 3" in gate["problems"][0]
    write(stage5.TASKS[-1], 2.0)
    gate = stage5.pilot_gate(jobs)
    assert gate["status"] == "PASS" and len(gate["runs"]) == 8
    single = {f"minimax_h3_t500_l20_p8__{task}": {"tasks": task} for task in stage5.TASKS}
    assert stage5.pilot_gate(single)["status"] == "PASS"


@runner_bf16
def test_runner_refuses_bf16_namespaces_for_other_vfm(tmp_path: Path) -> None:
    result, output_dir, _ = _build_matrix(tmp_path, "--tasks", EIGHT_TASKS)
    assert result.returncode == 0, result.stderr
    groups = output_dir / "groups.tsv"
    with groups.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    rows[0]["vfm"] = "wan"
    with groups.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    run = _run_group(tmp_path, groups, rows[0]["group_id"], rows[0]["strict_audit"])
    assert run.returncode == 3
    assert "layers 20/25 only" in run.stderr
