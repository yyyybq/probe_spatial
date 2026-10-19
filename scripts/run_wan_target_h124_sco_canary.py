#!/usr/bin/env python3
"""SCO-H800 launcher for a frozen Wan h124 target canary bundle."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import socket
import stat
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


RETIRED_DIRECT_HOST = "pt-314e0f9747cc44dc9e91dc07705c52a4-worker-0"
SCO_HOST_PATTERN = re.compile(r"pt-[a-z0-9]+-worker-0")
EXPECTED_PYTHON = Path("/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python3.11")
MIN_FREE_MIB = 75000


class LaunchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LaunchError(message)


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
        raise LaunchError(f"Unreadable JSON {path}: {exc}") from exc


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
        raise LaunchError(f"Refusing existing status: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def wait_submission_receipt(
    bundle: Path,
    metadata: dict[str, Any],
    bundle_sha: str,
    host: str,
    timeout_seconds: int = 90,
) -> dict[str, Any]:
    launcher = bundle / "launcher"
    claim_path = launcher / "sco_submit.claim.json"
    status_path = launcher / "sco_submit.status.json"
    describe_path = launcher / "sco_job_describe.json"
    binding_path = launcher / "sco_worker_binding.json"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline and not status_path.is_file():
        time.sleep(1)
    require(status_path.is_file() and not status_path.is_symlink(), "Timed out waiting for SCO submit receipt")
    for path in (claim_path, describe_path, binding_path):
        require(path.is_file() and not path.is_symlink(), f"Missing/unsafe SCO receipt artifact: {path}")
    status = read_json(status_path)
    claim = read_json(claim_path)
    describe = read_json(describe_path)
    binding = read_json(binding_path)
    require(all(isinstance(value, dict) for value in (status, claim, describe, binding)), "Malformed SCO receipt")
    sources = metadata["source_hashes"]
    submitter_sha = sources.get("runtime_tree/scripts/submit_wan_target_h124_sco_canary.py")
    require(isinstance(submitter_sha, str), "Frozen submitter SHA missing")
    job_id = status.get("job_id")
    require(isinstance(job_id, str) and re.fullmatch(r"pt-[a-z0-9]+", job_id), "Invalid submitted job ID")
    required_status = {
        "status": "PASS", "error": None, "returncode": 0,
        "job_ids": [job_id], "bundle": str(bundle), "bundle_sha256": bundle_sha,
        "submitter_sha256": submitter_sha, "claim_sha256": sha256(claim_path),
        "log": str((launcher / "sco_submit.log").resolve()),
        "log_sha256": sha256(launcher / "sco_submit.log"),
        "job_describe": str(describe_path.resolve()),
        "job_describe_sha256": sha256(describe_path),
        "worker_binding": str(binding_path.resolve()),
        "worker_binding_sha256": sha256(binding_path),
        "stop_returncode": None, "stop_output": None,
    }
    for key, expected in required_status.items():
        require(status.get(key) == expected, f"SCO submit receipt {key} mismatch")
    required_claim = {
        "status": "CLAIMED", "bundle": str(bundle), "bundle_sha256": bundle_sha,
        "submitter_sha256": submitter_sha,
        "validate_status": str((launcher / "validate.status.json").resolve()),
        "validate_status_sha256": sha256(launcher / "validate.status.json"),
        "container_image_digest_policy": "TAG_PIN_ONLY_ACCEPTED_FOR_THIS_CANARY",
    }
    for key, expected in required_claim.items():
        require(claim.get(key) == expected, f"SCO submit claim {key} mismatch")
    require(status.get("worker_command") == claim.get("worker_command"), "Worker command receipt mismatch")
    require(status.get("sco_create_argv") == claim.get("sco_create_argv"), "SCO create argv receipt mismatch")
    require(claim.get("worker_command_sha256") == hashlib.sha256(str(claim.get("worker_command")).encode()).hexdigest(), "Worker command SHA mismatch")
    require(describe.get("name") == job_id, "SCO describe job ID mismatch")
    require(describe.get("display_name") == metadata["sco_execution_contract"]["job_name"], "SCO describe job name mismatch")
    require(describe.get("workspace_name") == metadata["sco_execution_contract"]["workspace"], "SCO describe workspace mismatch")
    require(describe.get("framework") == "PYTORCH", "SCO describe framework mismatch")
    sco = metadata["sco_execution_contract"]
    roles = describe.get("roles")
    require(isinstance(roles, list) and len(roles) == 1, "SCO describe role mismatch")
    require(roles[0].get("startup_script") == claim.get("worker_command"), "SCO describe startup command mismatch")
    require(roles[0].get("name") == "Worker", "SCO describe role name mismatch")
    require(roles[0].get("image_path") == sco["container_image"], "SCO describe image mismatch")
    specs = roles[0].get("resource_spec")
    require(
        isinstance(specs, list) and len(specs) == 1
        and specs[0].get("name") == sco["worker_spec"]
        and specs[0].get("replicas") == 1
        and roles[0].get("total_replicas") == 1,
        "SCO describe worker spec mismatch",
    )
    scheduling = describe.get("scheduling")
    require(
        isinstance(scheduling, dict) and scheduling.get("priority") == sco["priority"]
        and str(scheduling.get("quota_type", "")).lower() == str(sco["quota_type"]).lower(),
        "SCO describe scheduling mismatch",
    )
    pool = describe.get("resource_pool")
    require(isinstance(pool, dict) and pool.get("name") == sco["aec2_name"], "SCO describe pool mismatch")
    mount_id, mount_path = str(sco["storage_mount"]).split(":", 1)
    mounts = describe.get("mount")
    require(
        isinstance(mounts, list) and len(mounts) == 1
        and mounts[0].get("id") == mount_id and mounts[0].get("mount_path") == mount_path,
        "SCO describe mount mismatch",
    )
    lme = describe.get("lme")
    require(isinstance(lme, dict) and int(lme.get("max_retries", -1)) == 0, "SCO retry policy mismatch")
    require(binding.get("job_id") == job_id, "SCO worker binding job ID mismatch")
    require(binding.get("worker_hostname") == host, "SCO worker binding hostname mismatch")
    require(str(binding.get("worker_status", "")).lower() == "running", "SCO worker binding is not Running")
    require(binding.get("get_workers_returncode") == 0, "SCO get-workers receipt failed")
    return {
        "job_id": job_id,
        "worker_hostname": host,
        "submit_status": str(status_path.resolve()),
        "submit_status_sha256": sha256(status_path),
        "submit_claim_sha256": sha256(claim_path),
        "job_describe_sha256": sha256(describe_path),
        "worker_binding_sha256": sha256(binding_path),
        "container_image_digest_policy": claim["container_image_digest_policy"],
    }


def verify_internal_sources(bundle: Path, metadata: dict[str, Any]) -> None:
    sources = metadata.get("source_hashes")
    require(isinstance(sources, dict) and sources, "Bundle lacks source_hashes")
    for relative, expected in sources.items():
        path = bundle / str(relative)
        require(path.is_file(), f"Missing frozen bundle source: {relative}")
        actual = sha256(path)
        require(actual == expected, f"Bundle source SHA mismatch: {relative}")
    mutable_prefixes = ("launcher/", "audit/", "logs/")
    actual_files = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file()
        and path != bundle / "bundle.json"
        and not path.relative_to(bundle).as_posix().startswith(mutable_prefixes)
    }
    require(actual_files == set(sources), "Bundle immutable file-set mismatch")


def verify_running_script(bundle: Path, metadata: dict[str, Any]) -> str:
    relative = "runtime_tree/scripts/run_wan_target_h124_sco_canary.py"
    expected = metadata["source_hashes"].get(relative)
    require(isinstance(expected, str), "Frozen launcher is absent from source_hashes")
    running = Path(__file__).resolve()
    require(running == (bundle / relative).resolve(), "Launcher is not executing from frozen runtime_tree")
    actual = sha256(running)
    require(actual == expected, "Running launcher SHA differs from frozen source hash")
    return actual


def verify_environment(bundle: Path) -> dict[str, Any]:
    contract = read_json(bundle / "provenance/environment_contract.json")
    require(isinstance(contract, dict), "Environment contract is malformed")
    require(
        Path(str(contract.get("python_executable", ""))).resolve()
        == EXPECTED_PYTHON.resolve(),
        "Frozen Python path changed",
    )
    actual_packages: dict[str, str] = {}
    packages = contract.get("packages")
    require(isinstance(packages, dict) and packages, "Frozen package inventory is missing")
    for name, expected in packages.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "NOT_INSTALLED"
        require(
            actual == expected,
            f"Python package version mismatch: {name} {actual} != {expected}",
        )
        actual_packages[name] = actual
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "packages": actual_packages,
    }


def verify_inventory(
    path: Path,
    root: Path,
    *,
    relative_paths: bool,
) -> dict[str, Any]:
    payload = read_json(path)
    require(isinstance(payload, dict), f"Inventory is not an object: {path}")
    artifacts = payload.get("artifacts")
    require(isinstance(artifacts, list) and artifacts, f"Empty inventory: {path}")
    root = root.resolve()
    require(Path(str(payload.get("root", ""))).resolve() == root, f"Inventory root mismatch: {path}")
    total = 0
    seen: set[str] = set()
    expected_relative_files: set[str] = set()
    for item in artifacts:
        require(isinstance(item, dict), f"Malformed inventory row: {path}")
        raw_path = Path(str(item["path"]))
        if relative_paths:
            require(not raw_path.is_absolute() and ".." not in raw_path.parts, f"Unsafe relative inventory path: {raw_path}")
            artifact = root / raw_path
            expected_relative_files.add(raw_path.as_posix())
        else:
            require(raw_path.is_absolute(), f"Source inventory path is not absolute: {raw_path}")
            artifact = raw_path
            require(artifact == root or root in artifact.parents, f"Source inventory escapes root: {artifact}")
        logical = str(artifact.absolute())
        require(logical not in seen, f"Duplicate inventory path: {artifact}")
        seen.add(logical)
        require(artifact.is_file(), f"Missing inventory artifact: {artifact}")
        size = artifact.stat().st_size
        require(size == int(item["size_bytes"]), f"Size mismatch: {artifact}")
        require(str(artifact.resolve()) == str(item["realpath"]), f"Realpath mismatch: {artifact}")
        require(artifact.is_symlink() is bool(item.get("is_symlink")), f"Symlink-state mismatch: {artifact}")
        require(sha256(artifact) == str(item["sha256"]), f"SHA mismatch: {artifact}")
        total += size
    if relative_paths:
        actual_relative_files = {
            candidate.relative_to(root).as_posix()
            for candidate in root.rglob("*")
            if candidate.is_file()
        }
        require(
            actual_relative_files == expected_relative_files,
            "Inventory root contains missing or extra files",
        )
    return {"artifacts": len(artifacts), "bytes": total, "inventory_sha256": sha256(path)}


def acquire_lock(path: Path):
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise LaunchError(f"Cannot safely open lock {path}: {exc}") from exc
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        os.close(fd)
        raise LaunchError(f"Lock is not a regular file: {path}")
    handle = os.fdopen(fd, "r+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise LaunchError(f"Concurrent launcher lock is held: {path}") from exc
    return handle


def verify_protected(repo_root: Path, expected: dict[str, str]) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, digest in expected.items():
        path = repo_root / relative
        require(path.is_file(), f"Missing protected file: {path}")
        actual[relative] = sha256(path)
        require(actual[relative] == digest, f"Protected pre-image mismatch: {relative}")
    return actual


def gpu_state(gpu: int) -> dict[str, Any]:
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.free,memory.used,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=30,
    )
    rows = list(csv.reader(query.splitlines(), skipinitialspace=True))
    require(len(rows) == 1, f"SCO worker must expose exactly one GPU: {rows}")
    matches = [row for row in rows if int(row[0]) == gpu]
    require(len(matches) == 1, f"GPU index {gpu} is not uniquely visible")
    row = matches[0]
    state = {
        "index": int(row[0]), "name": row[1], "uuid": row[2],
        "memory_total_mib": int(row[3]), "memory_free_mib": int(row[4]),
        "memory_used_mib": int(row[5]), "driver": row[6],
    }
    require(state["name"] == "NVIDIA H800", f"GPU is not H800: {state}")
    require(state["memory_total_mib"] >= 81000, f"Unexpected H800 memory: {state}")
    require(state["memory_free_mib"] >= MIN_FREE_MIB, f"GPU free memory gate failed: {state}")
    apps = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=30,
    )
    holders = []
    for app in csv.reader(apps.splitlines(), skipinitialspace=True):
        if app and app[0] == state["uuid"]:
            holders.append({"pid": int(app[1]), "process": app[2], "used_mib": int(app[3])})
    require(not holders, f"GPU has live compute holders: {holders}")
    state["compute_holders"] = holders
    return state


def build_child(bundle: Path, metadata: dict[str, Any], validate_only: bool) -> list[str]:
    planned = metadata.get("planned_cli")
    require(isinstance(planned, list) and len(planned) > 10, "Missing planned_cli")
    command = [str(item) for item in planned]
    command[0] = str(EXPECTED_PYTHON.resolve())
    command[2] = str((bundle / "runtime_tree/scripts/extract_wan_target_h124_extend_exact.py").resolve())
    replacements = {
        str(Path(metadata["bundle_path"]) / "inputs/manifest.jsonl"): str((bundle / "inputs/manifest.jsonl").resolve()),
        str(Path(metadata["bundle_path"]) / "inputs/canary_sample_ids.txt"): str((bundle / "inputs/canary_sample_ids.txt").resolve()),
        str(Path(metadata["bundle_path"]) / "inputs/source_h1_full_audit.json"): str((bundle / "inputs/source_h1_full_audit.json").resolve()),
    }
    command = [replacements.get(value, value) for value in command]
    if validate_only:
        command = [value for value in command if value != "--fail-on-existing"] + ["--validate-only"]
    return command


def main(args: argparse.Namespace) -> None:
    bundle = args.bundle.resolve()
    repo_root = args.repo_root.resolve()
    metadata_path = bundle / "bundle.json"
    require(metadata_path.is_file(), f"Missing bundle: {metadata_path}")
    require(sha256(metadata_path) == args.bundle_sha256, "Bundle SHA mismatch")
    metadata = read_json(metadata_path)
    require(isinstance(metadata, dict), "bundle.json is not an object")
    require(Path(str(metadata.get("bundle_path"))).resolve() == bundle, "Bundle path binding mismatch")
    require(
        Path(str(metadata.get("external_roots", {}).get("repo_root", ""))).resolve()
        == repo_root,
        "CLI repo root differs from frozen bundle repo root",
    )
    require(Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(), "Python interpreter mismatch")
    host = socket.gethostname()
    require(SCO_HOST_PATTERN.fullmatch(host) is not None, f"SCO worker hostname mismatch: {host}")
    require(host != RETIRED_DIRECT_HOST, "Refusing retired direct-node execution host")
    sco_contract = metadata.get("sco_execution_contract")
    require(isinstance(sco_contract, dict), "Bundle lacks SCO execution contract")
    expected_environment = {
        "Q00_SCO_EXECUTION_TAG": str(sco_contract.get("execution_tag", "")),
        "Q00_SCO_JOB_NAME": str(sco_contract.get("job_name", "")),
        "Q00_SCO_AEC2_NAME": str(sco_contract.get("aec2_name", "")),
        "Q00_SCO_WORKER_SPEC": str(sco_contract.get("worker_spec", "")),
    }
    for key, expected in expected_environment.items():
        require(expected and os.environ.get(key) == expected, f"SCO environment mismatch: {key}")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "SCO CUDA_VISIBLE_DEVICES must be 0")
    verify_internal_sources(bundle, metadata)
    launcher_sha = verify_running_script(bundle, metadata)
    environment_audit = verify_environment(bundle)
    submission_receipt = wait_submission_receipt(
        bundle, metadata, args.bundle_sha256, host
    )
    action_label = f"sco_{args.action}"
    action_dir = bundle / "launcher"
    action_dir.mkdir(parents=True, exist_ok=True)
    log_path = action_dir / f"{action_label}.log"
    status_path = action_dir / f"{action_label}.status.json"
    running_path = action_dir / f"{action_label}.running.json"
    require(
        not log_path.exists() and not status_path.exists() and not running_path.exists(),
        "Refusing existing launcher output",
    )
    started = datetime.now(timezone.utc).isoformat()
    exclusive_json(
        running_path,
        {
            "status": "RUNNING",
            "action": args.action,
            "started_at_utc": started,
            "host": host,
            "execution_backend": "sco_h800",
            "sco_environment": expected_environment,
            "pid": os.getpid(),
            "bundle_sha256": args.bundle_sha256,
            "launcher_sha256": launcher_sha,
            "environment_audit": environment_audit,
            "submission_receipt": submission_receipt,
        },
    )
    running_sha = sha256(running_path)
    phase = "PRE"
    child: list[str] = []
    gpu = None
    child_returncode: int | None = None
    audit_returncode: int | None = None
    source_before = image_before = model_before = protected_before = None
    source_after = image_after = model_after = protected_after = None
    audit_report_path: Path | None = None
    output_claim_path: Path | None = None
    action_lock = gpu_lock = None
    error: Exception | None = None
    try:
        protected = read_json(bundle / "inputs/protected_before.json")
        require(isinstance(protected, dict), "Protected pre-image is malformed")
        source_root = Path(str(metadata["external_roots"]["source_h1_root"]))
        data_root = Path(str(metadata["external_roots"]["data_root"]))
        model_root = Path(str(metadata["external_roots"]["model_snapshot"]))
        protected_before = verify_protected(repo_root, protected)
        source_before = verify_inventory(
            bundle / "inputs/source_h1_inventory.json",
            source_root,
            relative_paths=False,
        )
        image_before = verify_inventory(
            bundle / "inputs/canary_input_image_inventory.json",
            data_root,
            relative_paths=False,
        )
        model_before = verify_inventory(
            bundle / "inputs/model_snapshot_inventory.json",
            model_root,
            relative_paths=True,
        )
        action_lock = acquire_lock(Path("/tmp") / f"{metadata['run_tag']}.{action_label}.lock")
        output_root = Path(str(metadata["expected_output_root"]))
        if args.action == "extract":
            require(args.gpu is not None, "--gpu is required for extraction")
            preliminary_gpu = gpu_state(args.gpu)
            safe_uuid = preliminary_gpu["uuid"].replace("/", "_")
            gpu_lock = acquire_lock(Path("/tmp") / f"q00_wan_target_{safe_uuid}.lock")
            gpu = gpu_state(args.gpu)
            require(gpu["uuid"] == preliminary_gpu["uuid"], "GPU identity changed during lock")
            require(torch.cuda.is_available(), "Torch CUDA is unavailable on SCO worker")
            require(torch.cuda.device_count() == 1, "Torch must expose exactly one SCO GPU")
            require(torch.cuda.get_device_name(0) == "NVIDIA H800", "Torch device is not H800")
            gpu["torch_cuda_available"] = True
            gpu["torch_device_count"] = 1
            gpu["torch_device_name"] = torch.cuda.get_device_name(0)
            require(not output_root.parent.exists(), f"Output run namespace already exists: {output_root.parent}")
            output_root.parent.mkdir(mode=0o755)
            output_root.mkdir(mode=0o755)
            output_claim_path = output_root / ".sco_launch_claim.json"
            exclusive_json(
                output_claim_path,
                {
                    "status": "CLAIMED",
                    "run_tag": metadata["run_tag"],
                    "bundle": str(bundle),
                    "bundle_sha256": args.bundle_sha256,
                    "host": host,
                    "execution_backend": "sco_h800",
                    "sco_environment": expected_environment,
                    "submitted_job_id": submission_receipt["job_id"],
                    "submission_receipt_sha256": submission_receipt["submit_status_sha256"],
                    "submission_claim_sha256": submission_receipt["submit_claim_sha256"],
                    "job_describe_sha256": submission_receipt["job_describe_sha256"],
                    "worker_binding_sha256": submission_receipt["worker_binding_sha256"],
                    "runner_running": str(running_path.resolve()),
                    "runner_running_sha256": running_sha,
                    "gpu": gpu,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )

        child = build_child(bundle, metadata, args.action == "validate")
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = str((bundle / "runtime_tree").resolve())
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if args.action == "extract" else ""
        phase = "CHILD"
        with log_path.open("x") as log:
            result = subprocess.run(
                child,
                cwd=repo_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            log.flush()
            os.fsync(log.fileno())
        child_returncode = result.returncode
        require(result.returncode == 0, f"Frozen child failed rc={result.returncode}")

        phase = "POST"
        source_after = verify_inventory(
            bundle / "inputs/source_h1_inventory.json",
            source_root,
            relative_paths=False,
        )
        image_after = verify_inventory(
            bundle / "inputs/canary_input_image_inventory.json",
            data_root,
            relative_paths=False,
        )
        model_after = verify_inventory(
            bundle / "inputs/model_snapshot_inventory.json",
            model_root,
            relative_paths=True,
        )
        protected_after = verify_protected(repo_root, protected)
        verify_internal_sources(bundle, metadata)
        require(source_after == source_before, "Source h1 inventory changed")
        require(image_after == image_before, "Canary input-image inventory changed")
        require(model_after == model_before, "Model snapshot inventory changed")
        require(protected_after == protected_before, "Protected files changed")

        if args.action == "extract":
            phase = "AUDIT"
            audit_report_path = bundle / "audit/canary_audit.json"
            audit_log_path = action_dir / "audit.log"
            require(not audit_report_path.exists() and not audit_log_path.exists(), "Refusing existing audit output")
            audit_command = [
                str(EXPECTED_PYTHON.resolve()),
                "-B",
                str((bundle / "runtime_tree/scripts/audit_wan_target_h124_canary.py").resolve()),
                "--bundle",
                str((bundle / "bundle.json").resolve()),
                "--out-root",
                str(output_root),
                "--shard-name",
                "canary_train_val_2scenes",
                "--report",
                str(audit_report_path),
            ]
            audit_environment = environment.copy()
            audit_environment["CUDA_VISIBLE_DEVICES"] = ""
            with audit_log_path.open("x") as audit_log:
                audit_result = subprocess.run(
                    audit_command,
                    cwd=repo_root,
                    env=audit_environment,
                    stdout=audit_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                audit_log.flush()
                os.fsync(audit_log.fileno())
            audit_returncode = audit_result.returncode
            require(audit_result.returncode == 0, f"Independent audit failed rc={audit_result.returncode}")
            audit_payload = read_json(audit_report_path)
            require(isinstance(audit_payload, dict) and audit_payload.get("ok") is True, "Independent audit report is not PASS")
            require(audit_payload.get("bundle_sha256") == args.bundle_sha256, "Audit bundle SHA mismatch")
        phase = "COMPLETE"
    except Exception as exc:
        error = exc
        # A child/spawn failure must still close the read-only source/model and
        # protected-file hash chain.  Preserve the original failure and append
        # any postflight failure instead of silently losing either signal.
        if (
            source_before is not None
            and image_before is not None
            and model_before is not None
            and protected_before is not None
        ):
            failed_phase = phase
            phase = f"{failed_phase}_FAIL_POST"
            try:
                source_after = verify_inventory(
                    bundle / "inputs/source_h1_inventory.json",
                    source_root,
                    relative_paths=False,
                )
                image_after = verify_inventory(
                    bundle / "inputs/canary_input_image_inventory.json",
                    data_root,
                    relative_paths=False,
                )
                model_after = verify_inventory(
                    bundle / "inputs/model_snapshot_inventory.json",
                    model_root,
                    relative_paths=True,
                )
                protected_after = verify_protected(repo_root, protected)
                verify_internal_sources(bundle, metadata)
                require(source_after == source_before, "Source h1 inventory changed after failure")
                require(image_after == image_before, "Canary input images changed after failure")
                require(model_after == model_before, "Model snapshot inventory changed after failure")
                require(protected_after == protected_before, "Protected files changed after failure")
                phase = f"{failed_phase}_FAIL_POST_PASS"
            except Exception as post_exc:
                error = LaunchError(f"primary={exc}; failure-postflight={post_exc}")
                phase = f"{failed_phase}_FAIL_POST_FAIL"

    status = {
        "status": "PASS" if error is None else "FAIL",
        "terminal_phase": phase,
        "error": None if error is None else f"{type(error).__name__}: {error}",
        "action": args.action,
        "returncode": child_returncode,
        "audit_returncode": audit_returncode,
        "started_at_utc": started,
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": host,
        "execution_backend": "sco_h800",
        "sco_environment": expected_environment,
        "bundle": str(bundle),
        "bundle_sha256": args.bundle_sha256,
        "launcher_sha256": launcher_sha,
        "environment_audit": environment_audit,
        "submission_receipt": submission_receipt,
        "runner_running": str(running_path.resolve()),
        "runner_running_sha256": running_sha,
        "output_claim": (
            str(output_claim_path.resolve())
            if output_claim_path is not None and output_claim_path.exists()
            else None
        ),
        "output_claim_sha256": (
            sha256(output_claim_path)
            if output_claim_path is not None and output_claim_path.exists()
            else None
        ),
        "command": child,
        "command_sha256": hashlib.sha256("\0".join(child).encode()).hexdigest() if child else None,
        "gpu": gpu,
        "log": str(log_path.resolve()) if log_path.exists() else None,
        "log_sha256": sha256(log_path) if log_path.exists() else None,
        "audit_report": str(audit_report_path.resolve()) if audit_report_path and audit_report_path.exists() else None,
        "audit_report_sha256": sha256(audit_report_path) if audit_report_path and audit_report_path.exists() else None,
        "source_inventory_before": source_before,
        "source_inventory_after": source_after,
        "input_image_inventory_before": image_before,
        "input_image_inventory_after": image_after,
        "model_inventory_before": model_before,
        "model_inventory_after": model_after,
        "protected_before": protected_before,
        "protected_after": protected_after,
    }
    exclusive_json(status_path, status)
    if error is not None:
        raise LaunchError(f"{phase} failed: {error}") from error
    print(json.dumps(status, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--action", required=True, choices=("extract",))
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if not re_full_sha(args.bundle_sha256):
        parser.error("--bundle-sha256 must be a lowercase SHA256")
    if args.gpu != 0:
        parser.error("SCO canary requires the sole visible logical GPU --gpu 0")
    return args


def re_full_sha(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


if __name__ == "__main__":
    try:
        main(parse_args())
    except LaunchError as exc:
        raise SystemExit(f"LAUNCH FAILURE: {exc}") from exc
