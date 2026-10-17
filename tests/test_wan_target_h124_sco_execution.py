import hashlib
import json
import sys
from pathlib import Path

import pytest

from scripts import run_wan_target_h124_sco_canary as runner
from scripts import submit_wan_target_h124_sco_canary as submitter


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def test_sco_hostname_contract_excludes_retired_direct_host():
    assert runner.SCO_HOST_PATTERN.fullmatch("pt-abc123-worker-0")
    assert not runner.SCO_HOST_PATTERN.fullmatch("unrelated-host")
    assert runner.RETIRED_DIRECT_HOST == "pt-314e0f9747cc44dc9e91dc07705c52a4-worker-0"


def test_sco_gpu_gate_requires_exactly_one_clean_h800(monkeypatch):
    def check_output(argv, **kwargs):
        if "--query-gpu=index,name,uuid,memory.total,memory.free,memory.used,driver_version" in argv:
            return "0, NVIDIA H800, GPU-fixture, 81559, 81000, 559, 595.71.05\n"
        if "--query-compute-apps=gpu_uuid,pid,process_name,used_memory" in argv:
            return ""
        raise AssertionError(argv)

    monkeypatch.setattr(runner.subprocess, "check_output", check_output)
    state = runner.gpu_state(0)
    assert state["uuid"] == "GPU-fixture"
    assert state["memory_free_mib"] == 81000
    assert state["compute_holders"] == []


def test_sco_gpu_gate_rejects_more_than_one_visible_gpu(monkeypatch):
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda *args, **kwargs: (
            "0, NVIDIA H800, GPU-a, 81559, 81000, 559, 595.71.05\n"
            "1, NVIDIA H800, GPU-b, 81559, 81000, 559, 595.71.05\n"
        ),
    )
    with pytest.raises(runner.LaunchError, match="exactly one GPU"):
        runner.gpu_state(0)


def test_worker_command_binds_frozen_runner_and_sco_contract(tmp_path):
    bundle = tmp_path / "bundle"
    metadata = {
        "external_roots": {"repo_root": "/mnt/umm/users/yinbaiqiao/probe_spatial"},
        "sco_execution_contract": {
            "execution_tag": "fixture_sco_h800",
            "job_name": "bq_fixture",
            "aec2_name": "zoetrope",
            "worker_spec": "N4lS.Iq.I80.1",
        },
    }
    command = submitter.build_worker_command(bundle, "a" * 64, metadata)
    assert "Q00_SCO_EXECUTION_TAG=fixture_sco_h800" in command
    assert "Q00_SCO_JOB_NAME=bq_fixture" in command
    assert "CUDA_VISIBLE_DEVICES=0" in command
    assert str(bundle / "runtime_tree/scripts/run_wan_target_h124_sco_canary.py") in command
    assert "--bundle-sha256" in command and "a" * 64 in command
    assert command.endswith("--action extract --gpu 0")


def test_sco_runner_cli_rejects_nonzero_gpu(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--bundle",
            "/tmp/bundle",
            "--bundle-sha256",
            "a" * 64,
            "--repo-root",
            "/tmp/repo",
            "--action",
            "extract",
            "--gpu",
            "1",
        ],
    )
    with pytest.raises(SystemExit):
        runner.parse_args()


def test_submitter_requires_exact_cpu_validation_receipt(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "inputs").mkdir(parents=True)
    (bundle / "provenance").mkdir()
    (bundle / "launcher").mkdir()
    for name, size in (
        ("source_h1_inventory.json", 11),
        ("canary_input_image_inventory.json", 12),
        ("model_snapshot_inventory.json", 13),
    ):
        write_json(bundle / "inputs" / name, {"artifacts": [{"size_bytes": size}]})
    protected = {"experiments/registry.csv": "a" * 64}
    write_json(bundle / "inputs/protected_before.json", protected)
    environment = {
        "python_version": "fixture-python",
        "packages": {"torch": "fixture"},
        "expected_direct_host": "pt-direct-worker-0",
    }
    write_json(bundle / "provenance/environment_contract.json", environment)
    wrapper = bundle / "runtime_tree/scripts/extract_wan_target_h124_extend_exact.py"
    planned = [
        "python", "-B", str(wrapper), "--manifest", str(bundle / "inputs/manifest.jsonl"),
        "--selection-file", str(bundle / "inputs/canary_sample_ids.txt"),
        "--source-h1-audit", str(bundle / "inputs/source_h1_full_audit.json"),
        "--out-root", "/tmp/out", "--fail-on-existing",
    ]
    metadata = {
        "bundle_path": str(bundle),
        "planned_cli": planned,
        "source_hashes": {
            "runtime_tree/scripts/run_wan_target_h124_direct_canary.py": "d" * 64,
        },
    }
    expected_command = submitter.expected_validate_command(bundle, metadata)
    expected_environment = {
        "python_executable": str(submitter.EXPECTED_PYTHON.resolve()),
        "python_version": environment["python_version"],
        "packages": environment["packages"],
    }
    log = bundle / "launcher/validate.log"
    log.write_text(json.dumps({
        "status": "VALIDATION_ONLY_PASS", "scenes": 2, "rows": 8,
        "new_forwards_not_run": 16,
    }) + "\n")
    receipts = {
        "source_inventory": submitter.inventory_receipt(bundle / "inputs/source_h1_inventory.json"),
        "input_image_inventory": submitter.inventory_receipt(bundle / "inputs/canary_input_image_inventory.json"),
        "model_inventory": submitter.inventory_receipt(bundle / "inputs/model_snapshot_inventory.json"),
    }
    started = "2026-08-30T00:00:00+00:00"
    running = {
        "status": "RUNNING", "action": "validate", "started_at_utc": started,
        "host": environment["expected_direct_host"], "pid": 123,
        "bundle_sha256": "b" * 64, "launcher_sha256": "d" * 64,
        "environment_audit": expected_environment,
    }
    status = {
        "status": "PASS", "terminal_phase": "COMPLETE", "error": None,
        "action": "validate", "returncode": 0, "audit_returncode": None,
        "bundle": str(bundle), "bundle_sha256": "b" * 64,
        "launcher_sha256": "d" * 64, "host": environment["expected_direct_host"],
        "gpu": None, "audit_report": None, "audit_report_sha256": None,
        "environment_audit": expected_environment, "command": expected_command,
        "command_sha256": submitter.command_sha256(expected_command),
        "log": str(log.resolve()), "log_sha256": submitter.sha256(log),
        "started_at_utc": started, "ended_at_utc": "2026-08-30T00:01:00+00:00",
        "protected_before": protected, "protected_after": protected,
    }
    for key, value in receipts.items():
        status[f"{key}_before"] = value
        status[f"{key}_after"] = value
    write_json(bundle / "launcher/validate.running.json", running)
    write_json(bundle / "launcher/validate.status.json", status)
    assert submitter.validate_cpu_receipt(bundle, "b" * 64, metadata)[0].name == "validate.status.json"
    status["command_sha256"] = "0" * 64
    write_json(bundle / "launcher/validate.status.json.mutated", status)
    original = bundle / "launcher/validate.status.json"
    original.unlink()
    (bundle / "launcher/validate.status.json.mutated").rename(original)
    with pytest.raises(submitter.SubmitError, match="command_sha256 mismatch"):
        submitter.validate_cpu_receipt(bundle, "b" * 64, metadata)


def test_runner_binds_job_id_to_actual_worker_before_output(tmp_path):
    bundle = tmp_path / "bundle"
    launcher = bundle / "launcher"
    launcher.mkdir(parents=True)
    job_id = "pt-fixture123"
    host = "pt-fixtureworker-worker-0"
    worker_command = "fixture frozen worker command"
    (launcher / "sco_submit.log").write_text("created\n")
    (launcher / "validate.status.json").write_text("{}\n")
    claim = {
        "status": "CLAIMED", "bundle": str(bundle), "bundle_sha256": "b" * 64,
        "submitter_sha256": "s" * 64,
        "validate_status": str((launcher / "validate.status.json").resolve()),
        "validate_status_sha256": runner.sha256(launcher / "validate.status.json"),
        "container_image_digest_policy": "TAG_PIN_ONLY_ACCEPTED_FOR_THIS_CANARY",
        "worker_command": worker_command,
        "worker_command_sha256": hashlib.sha256(worker_command.encode()).hexdigest(),
        "sco_create_argv": ["sco", "create"],
    }
    describe = {
        "name": job_id, "display_name": "fixture-job", "workspace_name": "aigc",
        "framework": "PYTORCH",
        "roles": [{
            "name": "Worker", "startup_script": worker_command,
            "image_path": "fixture-image", "total_replicas": 1,
            "resource_spec": [{"name": "N4lS.Iq.I80.1", "replicas": 1}],
        }],
        "scheduling": {"priority": "HIGHEST", "quota_type": "RESERVED"},
        "resource_pool": {"name": "zoetrope"},
        "mount": [{"id": "fixture-mount", "mount_path": "/mnt/umm"}],
        "lme": {"max_retries": 0},
    }
    binding = {
        "job_id": job_id, "worker_hostname": host, "worker_status": "Running",
        "get_workers_returncode": 0,
    }
    write_json(launcher / "sco_submit.claim.json", claim)
    write_json(launcher / "sco_job_describe.json", describe)
    write_json(launcher / "sco_worker_binding.json", binding)
    status = {
        "status": "PASS", "error": None, "returncode": 0, "job_ids": [job_id],
        "job_id": job_id, "bundle": str(bundle), "bundle_sha256": "b" * 64,
        "submitter_sha256": "s" * 64,
        "claim_sha256": runner.sha256(launcher / "sco_submit.claim.json"),
        "log": str((launcher / "sco_submit.log").resolve()),
        "log_sha256": runner.sha256(launcher / "sco_submit.log"),
        "job_describe": str((launcher / "sco_job_describe.json").resolve()),
        "job_describe_sha256": runner.sha256(launcher / "sco_job_describe.json"),
        "worker_binding": str((launcher / "sco_worker_binding.json").resolve()),
        "worker_binding_sha256": runner.sha256(launcher / "sco_worker_binding.json"),
        "worker_command": worker_command, "sco_create_argv": ["sco", "create"],
        "stop_returncode": None, "stop_output": None,
    }
    write_json(launcher / "sco_submit.status.json", status)
    metadata = {
        "source_hashes": {
            "runtime_tree/scripts/submit_wan_target_h124_sco_canary.py": "s" * 64,
        },
        "sco_execution_contract": {
            "job_name": "fixture-job", "workspace": "aigc",
            "container_image": "fixture-image", "worker_spec": "N4lS.Iq.I80.1",
            "priority": "HIGHEST", "quota_type": "reserved", "aec2_name": "zoetrope",
            "storage_mount": "fixture-mount:/mnt/umm",
        },
    }
    receipt = runner.wait_submission_receipt(bundle, metadata, "b" * 64, host, timeout_seconds=1)
    assert receipt["job_id"] == job_id
    binding["worker_hostname"] = "pt-wrong-worker-0"
    write_json(launcher / "sco_worker_binding.mutated.json", binding)
    (launcher / "sco_worker_binding.json").unlink()
    (launcher / "sco_worker_binding.mutated.json").rename(launcher / "sco_worker_binding.json")
    with pytest.raises(runner.LaunchError, match="worker_binding_sha256 mismatch"):
        runner.wait_submission_receipt(bundle, metadata, "b" * 64, host, timeout_seconds=1)
