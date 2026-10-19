from __future__ import annotations

import csv
import json
import os
import shlex
import stat
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = PROJECT_ROOT / "scripts/submit_tstep_probe_sco.sh"
PROTOCOL_ID = "inscene_streaming_canonical_v2_obs28_h124"
PROTOCOL_SHA256 = "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f"
STREAMING_NAMESPACE = "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
TARGET_NAMESPACE = "FEAT_TARGET_CANONICAL_V2_SEEDED_V1"


def _write(path: Path, text: str = "fixture\n", *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _fixture_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "project"
    required_scripts = (
        "build_tstep_aoss_done_index.py",
        "build_tstep_canonical_manifests.py",
        "audit_tstep_canonical_alignment.py",
        "prune_tstep_probe_cache.py",
        "run_tstep_probe_group.sh",
    )
    for name in required_scripts:
        _write(root / "scripts" / name, "# fixture\n", executable=True)
    for rel in (
        "vidfm3d/train.py",
        "vidfm3d/data/components/inscene15k_dataset.py",
        "vidfm3d/utils/aoss_fs.py",
    ):
        _write(root / rel)

    protocol = {
        "schema_version": 2,
        "canonical_protocol": PROTOCOL_ID,
        "canonical_protocol_sha256": PROTOCOL_SHA256,
        "source": "scannetpp",
        "canonical_window": {"observations": 28},
        "streaming": {"prefix_lengths": [8, 12, 16, 24]},
        "target": {
            "future_horizons": [1, 2, 4],
            "include_anchors": True,
        },
        "feature_namespaces": {
            "streaming": STREAMING_NAMESPACE,
            "target": TARGET_NAMESPACE,
        },
        "feature_generation": {
            "contract": "inscene_deterministic_noise_v1",
            "wan_cogvideox_seed_scheme": "inscene_noise_seed_v1",
            "seed_payload": "canonical_json_utf8_sorted_compact({noise_seed_scheme,identity})",
            "seed_hash": "sha256_first_8_bytes_big_endian_mask_u63",
            "seed_excludes": ["timestep", "layer", "process_rank", "sample_order"],
            "streaming_identity": [
                "canonical_protocol", "canonical_protocol_sha256", "vfm", "model_id",
                "scene_id", "sample_kind=streaming_prefix", "window_id", "prefix_len",
            ],
            "target_identity": [
                "canonical_protocol", "canonical_protocol_sha256", "vfm", "model_id",
                "scene_id", "sample_kind=target_isolated", "frame_index",
            ],
            "generator_scope": "fresh_torch_generator_per_logical_sample_on_model_device",
            "wan_cogvideox_require_explicit_generator": True,
            "minimax_h3_seed_scheme": "minimax_h3_fixed_seed_v1",
            "minimax_h3_internal_seed": 0,
            "minimax_h3_seed_scope": "reset_per_forward",
        },
        "expected_universe": {
            "streaming_prefix_units": 10884,
            "target_scene_units": 681,
        },
    }
    _write(
        root / "protocols/inscene_streaming_canonical_v2.json",
        json.dumps(protocol),
    )

    manifest_dir = root / "artifacts/tstep_canonical_v2/manifests"
    _write(manifest_dir / "canonical_summary.json", "{}\n")
    for prefix in (8, 12, 16, 24):
        manifest = manifest_dir / f"inscene_c_h124_prefix{prefix}.jsonl"
        _write(manifest, json.dumps({"fixture_prefix": prefix}) + "\n")
        _write(Path(f"{manifest}.meta.json"), "{}\n")

    _write(
        root
        / "configs/experiment/inscene15k_streaming/tstep_canonical_v2_exclusions.txt",
        "# deliberately empty\n",
    )
    _write(root / "reports/tstep_canonical_v2/INVALID_R1_20260830.md", "INVALID\n")
    _write(root / "logs/feat_extract_tstep_campaign_r7_20260828/INVALID", "INVALID\n")
    _write(
        root / "logs/tstep_probe_campaigns/tstep_probe_wan_r1_20260828/INVALID",
        "INVALID\n",
    )

    # Minimal Hydra tree for the submitter's compose-before-submit check.
    _write(
        root / "configs/train.yaml",
        "defaults:\n  - _self_\n  - experiment: null\ntrainer:\n  devices: [0]\n",
    )
    _write(
        root / "configs/experiment/inscene15k_streaming/streaming_depth_wan_v1.yaml",
        "# @package _global_\nvfm_name: wan\n",
    )

    matrix_stub = r'''#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", required=True)
parser.add_argument("--manifest-dir", required=True)
parser.add_argument("--strict-audit-dir", required=True)
parser.add_argument("--protocol-id", required=True)
parser.add_argument("--canonical-obs-len", required=True)
parser.add_argument("--streaming-namespace", required=True)
parser.add_argument("--target-namespace", required=True)
args, unknown = parser.parse_known_args()
out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
manifest = Path(args.manifest_dir).resolve() / "inscene_c_h124_prefix8.jsonl"
audit_dir = Path(args.strict_audit_dir).resolve()
audit_dir.mkdir(parents=True, exist_ok=True)
audit = audit_dir / "wan_inscene_c_h124_prefix8.audit.json"
digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
scope = os.environ.get("FIXTURE_AUDIT_SCOPE", "full")
report = {
    "ok": True,
    "pass": True,
    "status": "pass",
    "scope": scope,
    "problem_count": 0,
    "protocol_id": args.protocol_id,
    "canonical_obs_len": int(args.canonical_obs_len),
    "canonical_protocol_sha256": "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f",
    "streaming_namespace": "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1",
    "target_namespace": "FEAT_TARGET_CANONICAL_V2_SEEDED_V1",
    "feature_generation_contract": "inscene_deterministic_noise_v1",
    "vfm": "wan",
    "source": "scannetpp",
    "manifest": str(manifest),
    "manifest_sha256": digest,
    "manifest_streaming_feature_root": "aoss://baiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_SEEDED_V1",
    "manifest_target_feature_root": "aoss://baiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_SEEDED_V1",
}
audit.write_text(json.dumps(report) + "\n")
audit_sha = hashlib.sha256(audit.read_bytes()).hexdigest()
fields = [
    "group_id", "status", "vfm", "source", "timestep", "layer", "prefix",
    "manifest", "tasks", "canonical_protocol", "canonical_obs_len",
    "strict_audit", "strict_audit_sha256", "strict_audit_ok",
    "streaming_namespace", "target_namespace",
]
row = {
    "group_id": "wan_t499_l15_p8",
    "status": "ready_full",
    "vfm": "wan",
    "source": "scannetpp",
    "timestep": "499",
    "layer": "15",
    "prefix": "8",
    "manifest": str(manifest),
    "tasks": "streaming_depth",
    "canonical_protocol": args.protocol_id,
    "canonical_obs_len": args.canonical_obs_len,
    "strict_audit": str(audit),
    "strict_audit_sha256": audit_sha,
    "strict_audit_ok": "true",
    "streaming_namespace": args.streaming_namespace,
    "target_namespace": args.target_namespace,
}
with (out / "groups.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerow(row)
(out / "summary.json").write_text("{}\n")
'''
    _write(root / "scripts/build_tstep_probe_matrix.py", matrix_stub, executable=True)

    # If REFRESH_AOSS_INDEX=0 regresses, fail instead of touching AOSS.
    _write(
        root / "scripts/build_tstep_aoss_done_index.py",
        "raise SystemExit('completion-index builder must not run in this test')\n",
        executable=True,
    )

    stream_root = tmp_path / "indices" / STREAMING_NAMESPACE
    target_root = tmp_path / "indices" / TARGET_NAMESPACE
    stream_root.mkdir(parents=True)
    target_root.mkdir(parents=True)
    return root, stream_root, target_root


def _run_submitter(tmp_path: Path, *, audit_scope: str = "full") -> subprocess.CompletedProcess[str]:
    root, stream_root, target_root = _fixture_project(tmp_path)
    fake_home = tmp_path / "home"
    sco_marker = tmp_path / "sco-was-called"
    _write(
        fake_home / ".sco/bin/sco",
        f"#!/usr/bin/env bash\ntouch {shlex.quote(str(sco_marker))}\nexit 97\n",
        executable=True,
    )
    out_dir = tmp_path / "out"
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(fake_home),
            "PROJECT_ROOT": str(root),
            "PYTHON": sys.executable,
            "RUN_ID": "canonical_v2_dry_test",
            "OUT_DIR": str(out_dir),
            "REFRESH_AOSS_INDEX": "0",
            "DRY_SUBMIT": "1",
            "VFMS": "wan",
            "LAYERS": "15",
            "PREFIXES": "8",
            "MAX_GROUPS": "1",
            "STREAMING_INDEX_ROOT": str(stream_root),
            "TARGET_INDEX_ROOT": str(target_root),
            "FIXTURE_AUDIT_SCOPE": audit_scope,
        }
    )
    result = subprocess.run(
        ["bash", str(SUBMITTER), "zoetrope"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert not sco_marker.exists(), result.stdout + result.stderr
    return result


def test_canonical_v2_dry_submit_builds_safe_plan_without_sco(tmp_path: Path) -> None:
    result = _run_submitter(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "validated 1 canonical-v2 jobs; no SCO jobs created" in result.stdout

    plan_path = tmp_path / "out/launcher/submit_plan.tsv"
    with plan_path.open(newline="", encoding="utf-8") as handle:
        plans = list(csv.DictReader(handle, delimiter="\t"))
    assert len(plans) == 1
    plan = plans[0]
    # The source column sits before timestep in groups.tsv; these assertions
    # prevent the old positional Bash parsing regression.
    assert plan["source"] == "scannetpp"
    assert plan["timestep"] == "499"
    assert plan["layer"] == "15"
    assert plan["prefix"] == "8"
    assert plan["job_name"] == "bq_tp_sv1_wan_t499_l15_p8"
    remote_tokens = shlex.split(plan["command"].partition(" && ")[2])
    assert f"INSCENE_STRICT_AUDIT={plan['strict_audit']}" in remote_tokens
    assert "GROUP_ID=wan_t499_l15_p8" in remote_tokens
    assert "AEC2_NAME=zoetrope" in remote_tokens


def test_canonical_v2_submitter_rejects_canary_audit(tmp_path: Path) -> None:
    result = _run_submitter(tmp_path, audit_scope="canary")
    assert result.returncode != 0
    assert "failed, canary, stale, or mismatched audit" in result.stderr
    assert not (tmp_path / "out/launcher/submit_plan.tsv").exists()


def test_refresh_contract_is_list_only_canonical_v2() -> None:
    text = SUBMITTER.read_text(encoding="utf-8")
    for token in (
        '--streaming-namespace "${STREAMING_NAMESPACE}"',
        '--target-namespace "${TARGET_NAMESPACE}"',
        '--expected-streaming-units "${EXPECTED_STREAMING_UNITS}"',
        '--expected-target-units "${EXPECTED_TARGET_UNITS}"',
        "--require-sidecar",
    ):
        assert token in text
    assert "EXPECTED_STREAMING_UNITS=10884" in text
    assert "EXPECTED_TARGET_UNITS=681" in text
    assert "logs/tstep_canonical_v2_seeded_v1/aoss_done" in text
    assert "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1" in text
    assert "FEAT_TARGET_CANONICAL_V2_SEEDED_V1" in text
    assert "aec2_name=${1:-zoetrope}" in text
    assert "seeded-v1 probing must use the Zoetrope pool" in text
    assert "--quota-type=reserved" in text
    assert "FEAT_STREAMING_CANONICAL_V2}" not in text
    assert "FEAT_TARGET_CANONICAL_V2}" not in text
