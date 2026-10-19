#!/usr/bin/env python3
"""Submit exactly one frozen Wan h124 two-scene canary to SCO H800."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_PYTHON = Path(
    "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python3.11"
)


class SubmitError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SubmitError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise SubmitError(f"Unreadable JSON {path}: {exc}") from exc


def exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    private = path.parent / f".{path.name}.incomplete-{os.getpid()}-{uuid.uuid4().hex}"
    descriptor: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(private, flags, 0o400)
        with os.fdopen(descriptor, "w") as handle:
            descriptor = None
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(private, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        private.unlink()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as exc:
        raise SubmitError(f"Refusing existing submission artifact: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def command_sha256(argv: list[str]) -> str:
    return hashlib.sha256("\0".join(argv).encode()).hexdigest()


def expected_validate_command(bundle: Path, metadata: dict[str, Any]) -> list[str]:
    planned = metadata.get("planned_cli")
    require(isinstance(planned, list) and len(planned) > 10, "Missing planned_cli")
    command = [str(value) for value in planned]
    require(command.count("--fail-on-existing") == 1, "planned_cli publication gate changed")
    require("--validate-only" not in command, "planned_cli is unexpectedly validate-only")
    command[0] = str(EXPECTED_PYTHON.resolve())
    command[2] = str(
        (bundle / "runtime_tree/scripts/extract_wan_target_h124_extend_exact.py").resolve()
    )
    frozen_bundle = Path(str(metadata["bundle_path"]))
    replacements = {
        str(frozen_bundle / "inputs/manifest.jsonl"): str(
            (bundle / "inputs/manifest.jsonl").resolve()
        ),
        str(frozen_bundle / "inputs/canary_sample_ids.txt"): str(
            (bundle / "inputs/canary_sample_ids.txt").resolve()
        ),
        str(frozen_bundle / "inputs/source_h1_full_audit.json"): str(
            (bundle / "inputs/source_h1_full_audit.json").resolve()
        ),
    }
    command = [replacements.get(value, value) for value in command]
    command.remove("--fail-on-existing")
    command.append("--validate-only")
    return command


def inventory_receipt(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    require(isinstance(payload, dict), f"Inventory is not an object: {path}")
    artifacts = payload.get("artifacts")
    require(isinstance(artifacts, list) and artifacts, f"Inventory is empty: {path}")
    return {
        "artifacts": len(artifacts),
        "bytes": sum(int(item["size_bytes"]) for item in artifacts),
        "inventory_sha256": sha256(path),
    }


def parse_utc(value: Any, label: str) -> datetime:
    require(isinstance(value, str), f"{label} is not a timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise SubmitError(f"Invalid {label}: {value}") from exc
    require(parsed.tzinfo is not None, f"{label} is not timezone-aware")
    return parsed


def validate_cpu_receipt(
    bundle: Path,
    bundle_sha: str,
    metadata: dict[str, Any],
) -> tuple[Path, str]:
    status_path = bundle / "launcher/validate.status.json"
    running_path = bundle / "launcher/validate.running.json"
    log_path = bundle / "launcher/validate.log"
    for path in (status_path, running_path, log_path):
        require(path.is_file() and not path.is_symlink(), f"Missing/unsafe CPU receipt: {path}")
    status = read_json(status_path)
    running = read_json(running_path)
    require(isinstance(status, dict) and isinstance(running, dict), "CPU receipt is malformed")
    expected_status_keys = {
        "action", "audit_report", "audit_report_sha256", "audit_returncode",
        "bundle", "bundle_sha256", "command", "command_sha256", "ended_at_utc",
        "environment_audit", "error", "gpu", "host", "input_image_inventory_after",
        "input_image_inventory_before", "launcher_sha256", "log", "log_sha256",
        "model_inventory_after", "model_inventory_before", "protected_after",
        "protected_before", "returncode", "source_inventory_after",
        "source_inventory_before", "started_at_utc", "status", "terminal_phase",
    }
    require(set(status) == expected_status_keys, "CPU validation status schema changed")
    expected_running_keys = {
        "action", "bundle_sha256", "environment_audit", "host", "launcher_sha256",
        "pid", "started_at_utc", "status",
    }
    require(set(running) == expected_running_keys, "CPU validation running schema changed")
    environment_contract = read_json(bundle / "provenance/environment_contract.json")
    require(isinstance(environment_contract, dict), "Environment contract is malformed")
    expected_environment = {
        "python_executable": str(EXPECTED_PYTHON.resolve()),
        "python_version": environment_contract.get("python_version"),
        "packages": environment_contract.get("packages"),
    }
    direct_relative = "runtime_tree/scripts/run_wan_target_h124_direct_canary.py"
    direct_sha = metadata["source_hashes"].get(direct_relative)
    expected_host = environment_contract.get("expected_direct_host")
    expected_command = expected_validate_command(bundle, metadata)
    required_status = {
        "status": "PASS", "terminal_phase": "COMPLETE", "error": None,
        "action": "validate", "returncode": 0, "audit_returncode": None,
        "bundle": str(bundle), "bundle_sha256": bundle_sha,
        "launcher_sha256": direct_sha, "host": expected_host, "gpu": None,
        "audit_report": None, "audit_report_sha256": None,
        "environment_audit": expected_environment, "command": expected_command,
        "command_sha256": command_sha256(expected_command),
        "log": str(log_path.resolve()), "log_sha256": sha256(log_path),
    }
    for key, expected in required_status.items():
        require(status.get(key) == expected, f"CPU validation {key} mismatch")
    required_running = {
        "status": "RUNNING", "action": "validate", "bundle_sha256": bundle_sha,
        "launcher_sha256": direct_sha, "host": expected_host,
        "environment_audit": expected_environment,
        "started_at_utc": status.get("started_at_utc"),
    }
    for key, expected in required_running.items():
        require(running.get(key) == expected, f"CPU validation running {key} mismatch")
    require(isinstance(running.get("pid"), int) and running["pid"] > 0, "Invalid validation PID")
    started = parse_utc(status.get("started_at_utc"), "validation start")
    ended = parse_utc(status.get("ended_at_utc"), "validation end")
    require(ended >= started, "CPU validation timestamps are reversed")
    expected_receipts = {
        "source_inventory": inventory_receipt(bundle / "inputs/source_h1_inventory.json"),
        "input_image_inventory": inventory_receipt(
            bundle / "inputs/canary_input_image_inventory.json"
        ),
        "model_inventory": inventory_receipt(bundle / "inputs/model_snapshot_inventory.json"),
    }
    for key, expected in expected_receipts.items():
        require(status.get(f"{key}_before") == expected, f"Validation {key} before mismatch")
        require(status.get(f"{key}_after") == expected, f"Validation {key} after mismatch")
    protected = read_json(bundle / "inputs/protected_before.json")
    require(status.get("protected_before") == protected, "Validation protected-before mismatch")
    require(status.get("protected_after") == protected, "Validation protected-after mismatch")
    nonempty_lines = [line for line in log_path.read_text().splitlines() if line.strip()]
    require(len(nonempty_lines) == 1, "CPU validation log must contain one JSON line")
    try:
        log_payload = json.loads(nonempty_lines[0])
    except json.JSONDecodeError as exc:
        raise SubmitError("CPU validation log is not JSON") from exc
    require(
        log_payload
        == {"status": "VALIDATION_ONLY_PASS", "scenes": 2, "rows": 8,
            "new_forwards_not_run": 16},
        "CPU validation log contract mismatch",
    )
    return status_path, sha256(status_path)


def validate_job_description(
    payload: Any,
    job_id: str,
    worker_command: str,
    sco: dict[str, Any],
) -> None:
    require(isinstance(payload, dict), "SCO describe is not an object")
    require(payload.get("name") == job_id, "SCO describe job ID mismatch")
    require(payload.get("display_name") == sco["job_name"], "SCO job name mismatch")
    require(payload.get("workspace_name") == sco["workspace"], "SCO workspace mismatch")
    require(payload.get("framework") == "PYTORCH", "SCO framework mismatch")
    scheduling = payload.get("scheduling")
    require(
        isinstance(scheduling, dict)
        and scheduling.get("priority") == sco["priority"]
        and str(scheduling.get("quota_type", "")).lower() == str(sco["quota_type"]).lower(),
        "SCO scheduling mismatch",
    )
    pool = payload.get("resource_pool")
    require(isinstance(pool, dict) and pool.get("name") == sco["aec2_name"], "SCO pool mismatch")
    roles = payload.get("roles")
    require(isinstance(roles, list) and len(roles) == 1, "SCO role count mismatch")
    role = roles[0]
    require(role.get("name") == "Worker", "SCO role mismatch")
    require(role.get("image_path") == sco["container_image"], "SCO image mismatch")
    require(role.get("startup_script") == worker_command, "SCO startup command mismatch")
    require(role.get("total_replicas") == 1, "SCO replica count mismatch")
    specs = role.get("resource_spec")
    require(
        isinstance(specs, list) and len(specs) == 1
        and specs[0].get("name") == sco["worker_spec"]
        and specs[0].get("replicas") == 1,
        "SCO worker spec mismatch",
    )
    mounts = payload.get("mount")
    mount_id, mount_path = str(sco["storage_mount"]).split(":", 1)
    require(
        isinstance(mounts, list) and len(mounts) == 1
        and mounts[0].get("id") == mount_id and mounts[0].get("mount_path") == mount_path,
        "SCO storage mount mismatch",
    )
    lme = payload.get("lme")
    require(isinstance(lme, dict) and int(lme.get("max_retries", -1)) == 0, "SCO retry policy mismatch")


def poll_worker_binding(
    sco_binary: str,
    workspace: str,
    job_id: str,
    timeout_seconds: int = 45,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_output = ""
    last_returncode: int | None = None
    while time.monotonic() < deadline:
        result = subprocess.run(
            [sco_binary, "acp", "jobs", "get-workers", f"--workspace-name={workspace}", job_id],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        last_returncode = result.returncode
        last_output = result.stdout
        rows: list[tuple[str, str]] = []
        for line in last_output.splitlines():
            parts = [part.strip() for part in line.split("|")]
            if len(parts) >= 6 and re.fullmatch(r"pt-[a-z0-9]+-worker-0", parts[1] or ""):
                rows.append((parts[1], parts[-2]))
        if result.returncode == 0 and len(rows) == 1 and rows[0][1].lower() == "running":
            return {
                "job_id": job_id,
                "worker_hostname": rows[0][0],
                "worker_status": rows[0][1],
                "get_workers_returncode": result.returncode,
                "get_workers_output": last_output,
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        time.sleep(3)
    raise SubmitError(
        f"SCO worker did not become uniquely Running within {timeout_seconds}s; "
        f"rc={last_returncode}, output={last_output!r}"
    )


def poll_job_description(
    sco_binary: str,
    workspace: str,
    job_id: str,
    worker_command: str,
    sco: dict[str, Any],
    timeout_seconds: int = 15,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error = "description unavailable"
    while time.monotonic() < deadline:
        result = subprocess.run(
            [
                sco_binary, "acp", "jobs", "describe",
                f"--workspace-name={workspace}", "-o", "json", job_id,
            ],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        try:
            require(result.returncode == 0, f"SCO describe failed rc={result.returncode}")
            payload = json.loads(result.stdout)
            validate_job_description(payload, job_id, worker_command, sco)
            return payload
        except (SubmitError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2)
    raise SubmitError(f"SCO describe did not converge within {timeout_seconds}s: {last_error}")


def verify_bundle(bundle: Path, expected_sha: str) -> tuple[dict[str, Any], str]:
    metadata_path = bundle / "bundle.json"
    require(metadata_path.is_file(), f"Missing bundle: {metadata_path}")
    require(sha256(metadata_path) == expected_sha, "Bundle SHA mismatch")
    metadata = read_json(metadata_path)
    require(isinstance(metadata, dict), "Bundle is not an object")
    require(Path(str(metadata.get("bundle_path", ""))).resolve() == bundle, "Bundle path mismatch")
    sources = metadata.get("source_hashes")
    require(isinstance(sources, dict) and sources, "Bundle source_hashes missing")
    for relative, digest in sources.items():
        path = bundle / str(relative)
        require(path.is_file() and sha256(path) == digest, f"Frozen source mismatch: {relative}")
    mutable = ("launcher/", "audit/", "logs/")
    actual = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file()
        and path != metadata_path
        and not path.relative_to(bundle).as_posix().startswith(mutable)
    }
    require(actual == set(sources), "Bundle immutable file-set mismatch")
    relative_self = "runtime_tree/scripts/submit_wan_target_h124_sco_canary.py"
    require(Path(__file__).resolve() == (bundle / relative_self).resolve(), "Submitter is not frozen")
    submitter_sha = sha256(Path(__file__).resolve())
    require(sources.get(relative_self) == submitter_sha, "Frozen submitter SHA mismatch")
    return metadata, submitter_sha


def build_worker_command(bundle: Path, bundle_sha: str, metadata: dict[str, Any]) -> str:
    sco = metadata["sco_execution_contract"]
    runner = bundle / "runtime_tree/scripts/run_wan_target_h124_sco_canary.py"
    arguments = [
        "env",
        f"Q00_SCO_EXECUTION_TAG={sco['execution_tag']}",
        f"Q00_SCO_JOB_NAME={sco['job_name']}",
        f"Q00_SCO_AEC2_NAME={sco['aec2_name']}",
        f"Q00_SCO_WORKER_SPEC={sco['worker_spec']}",
        "CUDA_VISIBLE_DEVICES=0",
        "PYTHONDONTWRITEBYTECODE=1",
        "TOKENIZERS_PARALLELISM=false",
        str(EXPECTED_PYTHON),
        "-B",
        str(runner),
        "--bundle",
        str(bundle),
        "--bundle-sha256",
        bundle_sha,
        "--repo-root",
        str(Path(metadata["external_roots"]["repo_root"])),
        "--action",
        "extract",
        "--gpu",
        "0",
    ]
    return f"cd {shlex.quote(str(Path(metadata['external_roots']['repo_root'])))} && exec {shlex.join(arguments)}"


def main(args: argparse.Namespace) -> None:
    bundle = args.bundle.resolve()
    require(re.fullmatch(r"[0-9a-f]{64}", args.bundle_sha256) is not None, "Invalid bundle SHA")
    metadata, submitter_sha = verify_bundle(bundle, args.bundle_sha256)
    sco = metadata.get("sco_execution_contract")
    require(isinstance(sco, dict), "Missing SCO execution contract")
    expected_sco = {
        "workspace": "aigc",
        "aec2_name": "zoetrope",
        "priority": "HIGHEST",
        "quota_type": "reserved",
        "worker_nodes": 1,
        "worker_spec": "N4lS.Iq.I80.1",
        "container_image": "registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617",
        "storage_mount": "019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm",
        "automatic_retry": False,
    }
    for key, expected in expected_sco.items():
        require(sco.get(key) == expected, f"SCO contract mismatch: {key}")
    require(sco.get("container_image_digest") == "UNAVAILABLE_TAG_PIN_ONLY", "Image provenance policy changed")

    validate_status_path, validate_status_sha = validate_cpu_receipt(
        bundle, args.bundle_sha256, metadata
    )

    output_root = Path(str(metadata["expected_output_root"]))
    require(not output_root.parent.exists(), f"Output namespace already exists: {output_root.parent}")
    forbidden = (
        bundle / "launcher/sco_extract.running.json",
        bundle / "launcher/sco_extract.status.json",
        bundle / "launcher/sco_extract.log",
        bundle / "audit/canary_audit.json",
    )
    require(not any(path.exists() for path in forbidden), "Existing SCO writer/audit artifact")

    sco_binary = shutil.which("sco")
    require(sco_binary is not None, "sco CLI is unavailable")
    worker_command = build_worker_command(bundle, args.bundle_sha256, metadata)
    create_argv = [
        sco_binary,
        "acp",
        "jobs",
        "create",
        "--workspace-name=aigc",
        "--aec2-name=zoetrope",
        f"--job-name={sco['job_name']}",
        "--priority=HIGHEST",
        "--quota-type=reserved",
        f"--container-image-url={sco['container_image']}",
        f"--storage-mount={sco['storage_mount']}",
        "--training-framework=pytorch",
        "--worker-nodes=1",
        "--worker-spec=N4lS.Iq.I80.1",
        f"--command={worker_command}",
    ]
    launcher = bundle / "launcher"
    claim_path = launcher / "sco_submit.claim.json"
    status_path = launcher / "sco_submit.status.json"
    log_path = launcher / "sco_submit.log"
    describe_path = launcher / "sco_job_describe.json"
    binding_path = launcher / "sco_worker_binding.json"
    require(
        not any(path.exists() for path in (status_path, log_path, describe_path, binding_path)),
        "Existing SCO submission output",
    )
    exclusive_json(
        claim_path,
        {
            "status": "CLAIMED",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "bundle": str(bundle),
            "bundle_sha256": args.bundle_sha256,
            "validate_status": str(validate_status_path.resolve()),
            "validate_status_sha256": validate_status_sha,
            "submitter_sha256": submitter_sha,
            "worker_command": worker_command,
            "worker_command_sha256": hashlib.sha256(worker_command.encode()).hexdigest(),
            "sco_create_argv": create_argv,
            "container_image_digest_policy": "TAG_PIN_ONLY_ACCEPTED_FOR_THIS_CANARY",
        },
    )

    started = datetime.now(timezone.utc).isoformat()
    returncode: int | None = None
    output = ""
    error: str | None = None
    job_ids: list[str] = []
    job_id: str | None = None
    describe_sha: str | None = None
    binding_sha: str | None = None
    stop_returncode: int | list[int] | None = None
    stop_output: str | list[str] | None = None
    try:
        result = subprocess.run(create_argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        returncode = result.returncode
        output = result.stdout
        job_ids = sorted(set(re.findall(r"pt-[a-z0-9]+", output)))
        if len(job_ids) == 1:
            job_id = job_ids[0]
        with log_path.open("x") as handle:
            handle.write(output)
            handle.flush()
            os.fsync(handle.fileno())
        require(returncode == 0, f"sco create failed rc={returncode}")
        require(len(job_ids) == 1, f"Expected exactly one SCO job ID, got {job_ids}")
        require(job_id == job_ids[0], "Reconciled SCO job ID mismatch")
        binding_payload = poll_worker_binding(
            sco_binary, str(sco["workspace"]), job_id
        )
        exclusive_json(binding_path, binding_payload)
        binding_sha = sha256(binding_path)
        describe_payload = poll_job_description(
            sco_binary, str(sco["workspace"]), job_id, worker_command, sco
        )
        exclusive_json(describe_path, describe_payload)
        describe_sha = sha256(describe_path)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if len(job_ids) == 1:
            stop_codes: list[int] = []
            stop_messages: list[str] = []
            for candidate in job_ids:
                try:
                    stopped = subprocess.run(
                        [sco_binary, "acp", "jobs", "stop", f"--workspace-name={sco['workspace']}", candidate],
                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
                    )
                    stop_codes.append(stopped.returncode)
                    stop_messages.append(stopped.stdout)
                except Exception as stop_exc:
                    stop_codes.append(-1)
                    stop_messages.append(f"{type(stop_exc).__name__}: {stop_exc}")
            stop_returncode = stop_codes[0] if len(stop_codes) == 1 else stop_codes
            stop_output = stop_messages[0] if len(stop_messages) == 1 else stop_messages
    status = {
        "status": "PASS" if error is None else "FAIL",
        "started_at_utc": started,
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "returncode": returncode,
        "error": error,
        "job_ids": job_ids,
        "job_id": job_id,
        "bundle": str(bundle),
        "bundle_sha256": args.bundle_sha256,
        "submitter_sha256": submitter_sha,
        "claim_sha256": sha256(claim_path),
        "validate_status_sha256": validate_status_sha,
        "log": str(log_path.resolve()) if log_path.exists() else None,
        "log_sha256": sha256(log_path) if log_path.exists() else None,
        "job_describe": str(describe_path.resolve()) if describe_path.exists() else None,
        "job_describe_sha256": describe_sha,
        "worker_binding": str(binding_path.resolve()) if binding_path.exists() else None,
        "worker_binding_sha256": binding_sha,
        "stop_returncode": stop_returncode,
        "stop_output": stop_output,
        "sco_create_argv": create_argv,
        "worker_command": worker_command,
    }
    try:
        exclusive_json(status_path, status)
    except Exception as publish_exc:
        # A PASS receipt is the worker's only authorization to create the
        # shared output namespace.  If that atomic publication fails after
        # create succeeded, stop every reconciled job before returning.
        if error is None and len(job_ids) == 1:
            for candidate in job_ids:
                try:
                    subprocess.run(
                        [sco_binary, "acp", "jobs", "stop", f"--workspace-name={sco['workspace']}", candidate],
                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        check=False, timeout=30,
                    )
                except Exception:
                    pass
        raise SubmitError(f"Cannot publish terminal SCO receipt: {publish_exc}") from publish_exc
    if error is not None:
        raise SubmitError(error)
    print(json.dumps(status, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--bundle-sha256", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        main(parse_args())
    except SubmitError as exc:
        raise SystemExit(f"SCO SUBMIT FAILURE: {exc}") from exc
