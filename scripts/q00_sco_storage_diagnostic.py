#!/usr/bin/env python3
"""Fail-closed SCO storage diagnostic for the Wan h124 target repair.

This intentionally does not load a model, train, evaluate, or touch canonical tables.
It writes the exact per-layer target tensor payload size needed for one h124 scene and
performs fsync plus full safetensors readback before publishing a diagnostic completion.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


EXPECTED_PYTHON = Path("/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python3.11")


class DiagnosticError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        raise DiagnosticError(f"Unreadable JSON {path}: {exc}") from exc


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_json_noreplace(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    private = path.parent / f".{path.name}.incomplete-{os.getpid()}-{uuid.uuid4().hex}"
    descriptor: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(private, flags, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            descriptor = None
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(private, path, follow_symlinks=False)
        fsync_dir(path.parent)
        private.unlink()
        fsync_dir(path.parent)
    except FileExistsError as exc:
        raise DiagnosticError(f"Refusing existing diagnostic marker: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def statvfs_record(path: Path) -> dict[str, int]:
    value = os.statvfs(path)
    return {
        "block_size": int(value.f_frsize),
        "blocks_total": int(value.f_blocks),
        "blocks_available": int(value.f_bavail),
        "bytes_available": int(value.f_frsize * value.f_bavail),
        "inodes_total": int(value.f_files),
        "inodes_available": int(value.f_favail),
    }


def rlimit_record() -> dict[str, list[int]]:
    labels = {
        "as": resource.RLIMIT_AS,
        "core": resource.RLIMIT_CORE,
        "data": resource.RLIMIT_DATA,
        "fsize": resource.RLIMIT_FSIZE,
        "memlock": resource.RLIMIT_MEMLOCK,
        "nofile": resource.RLIMIT_NOFILE,
        "nproc": resource.RLIMIT_NPROC,
        "rss": resource.RLIMIT_RSS,
        "stack": resource.RLIMIT_STACK,
    }
    return {name: list(resource.getrlimit(limit)) for name, limit in labels.items()}


def mountinfo_for(path: Path) -> list[str]:
    resolved = str(path.resolve())
    lines = Path("/proc/self/mountinfo").read_text().splitlines()
    return [line for line in lines if " /mnt/umm " in line or resolved in line]


def device_record() -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    require(query.returncode == 0, f"nvidia-smi query failed: {query.stderr.strip()}")
    rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    require(len(rows) == 1, f"Expected exactly one visible GPU, got {len(rows)}")
    fields = [field.strip() for field in rows[0].split(",")]
    require(len(fields) == 6, f"Unexpected nvidia-smi schema: {rows[0]}")
    index, name, gpu_uuid, total, free, driver = fields
    require(name == "NVIDIA H800", f"Unexpected GPU: {name}")
    require(int(total) >= 81000, f"Insufficient GPU total memory: {total}")
    require(int(free) >= 75000, f"Insufficient GPU free memory: {free}")
    holders = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=False,
        text=True,
        capture_output=True,
    )
    require(holders.returncode == 0, "nvidia-smi holder query failed")
    active = [line.strip() for line in holders.stdout.splitlines() if line.strip()]
    require(not active, f"Unexpected GPU compute holders: {active}")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Torch CUDA contract failed")
    return {
        "index": int(index),
        "name": name,
        "uuid": gpu_uuid,
        "memory_total_mib": int(total),
        "memory_free_mib": int(free),
        "driver": driver,
        "torch_cuda_available": True,
        "torch_device_count": int(torch.cuda.device_count()),
        "torch_device_name": torch.cuda.get_device_name(0),
        "compute_holders": active,
    }


def environment_record(expected: dict[str, str]) -> dict[str, Any]:
    require(Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(), "Python path mismatch")
    actual = {name: importlib.metadata.version(name) for name in expected}
    require(actual == expected, f"Package contract mismatch: {actual} != {expected}")
    previous = os.umask(0)
    os.umask(previous)
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "packages": actual,
        "uid": os.getuid(),
        "gid": os.getgid(),
        "groups": os.getgroups(),
        "umask_octal": f"{previous:04o}",
        "rlimits": rlimit_record(),
    }


def validate_bundle(bundle: Path, expected_sha: str) -> dict[str, Any]:
    bundle_json = bundle / "bundle.json"
    require(bundle_json.is_file() and not bundle_json.is_symlink(), "Missing unsafe bundle.json")
    require(sha256(bundle_json) == expected_sha, "Bundle SHA mismatch")
    metadata = read_json(bundle_json)
    require(isinstance(metadata, dict), "Bundle metadata is not an object")
    require(metadata.get("kind") == "q00_wan_h124_storage_diagnostic", "Unexpected bundle kind")
    source_hashes = metadata.get("source_hashes")
    require(isinstance(source_hashes, dict) and source_hashes, "Missing source hashes")
    expected_relative = "runtime_tree/scripts/q00_sco_storage_diagnostic.py"
    expected_script = bundle / expected_relative
    require(Path(__file__).resolve() == expected_script.resolve(), "Worker is not frozen bundle script")
    require(source_hashes.get(expected_relative) == sha256(expected_script), "Frozen worker SHA mismatch")
    actual_files = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file()
        and path.relative_to(bundle).as_posix() != "bundle.json"
        and not path.relative_to(bundle).as_posix().startswith("launcher/")
    }
    require(actual_files == set(source_hashes), "Immutable bundle file-set mismatch")
    for relative, expected in source_hashes.items():
        path = bundle / str(relative)
        require(path.is_file() and not path.is_symlink(), f"Missing unsafe frozen file: {relative}")
        require(sha256(path) == expected, f"Frozen file SHA mismatch: {relative}")
    return metadata


def verify_protected(repo_root: Path, expected: dict[str, str]) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, digest in expected.items():
        path = (repo_root / relative).resolve()
        require(path.is_file() and not path.is_symlink(), f"Protected file unsafe: {relative}")
        actual[str(relative)] = sha256(path)
        require(actual[str(relative)] == digest, f"Protected SHA mismatch: {relative}")
    return actual


def write_tensor(
    directory: Path,
    layer: int,
    tensor: torch.Tensor,
    expected_shape: list[int],
    expected_bytes: int,
) -> dict[str, Any]:
    final = directory / f"feature_t749_layer{layer}.sft"
    require(not final.exists(), f"Existing target file: {final}")
    temporary = directory / f".{final.name}.tmp-{os.getpid()}"
    require(not temporary.exists(), f"Existing target temporary file: {temporary}")
    before = statvfs_record(directory)
    save_file({"feat": tensor}, str(temporary))
    descriptor = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    digest = sha256(temporary)
    with safe_open(str(temporary), framework="pt", device="cpu") as handle:
        require(handle.keys() == ["feat"], f"Unexpected safetensors keys for layer {layer}")
        value = handle.get_tensor("feat")
    require(list(value.shape) == expected_shape, f"Shape mismatch for layer {layer}: {list(value.shape)}")
    require(value.dtype == torch.float16, f"Dtype mismatch for layer {layer}: {value.dtype}")
    require(bool(torch.isfinite(value).all().item()), f"Non-finite diagnostic tensor layer {layer}")
    require(temporary.stat().st_size == expected_bytes, f"Byte mismatch for layer {layer}")
    os.link(temporary, final, follow_symlinks=False)
    fsync_dir(directory)
    temporary.unlink()
    fsync_dir(directory)
    after = statvfs_record(directory)
    return {
        "layer": layer,
        "path": str(final),
        "sha256": digest,
        "size_bytes": int(final.stat().st_size),
        "mode_octal": f"{stat_mode(final):04o}",
        "statvfs_before": before,
        "statvfs_after": after,
    }


def stat_mode(path: Path) -> int:
    return int(path.stat().st_mode & 0o777)


def run(args: argparse.Namespace) -> int:
    bundle = Path(args.bundle).resolve()
    metadata = validate_bundle(bundle, args.bundle_sha256)
    contract = metadata.get("diagnostic_contract")
    require(isinstance(contract, dict), "Missing diagnostic contract")
    expected_shape = [int(value) for value in contract["tensor_shape"]]
    expected_layers = [int(value) for value in contract["layers"]]
    expected_bytes = int(contract["file_size_bytes"])
    require(expected_shape == [12, 30, 52, 1536], "Unexpected tensor shape contract")
    require(expected_layers == [0, 5, 10, 15, 25, 29], "Unexpected layer contract")
    require(expected_bytes == 57507928, "Unexpected file size contract")
    repo_root = Path(str(metadata["external_roots"]["repo_root"])).resolve()
    output_root = Path(str(metadata["external_roots"]["output_root"])).resolve()
    require(output_root.parent.is_dir(), f"Output parent missing: {output_root.parent}")
    require(not output_root.exists(), f"Output namespace already exists: {output_root}")
    if args.validate_only:
        protected = verify_protected(repo_root, metadata["protected_before"])
        environment = environment_record(metadata["environment_contract"]["packages"])
        print(
            json.dumps(
                {
                    "status": "VALIDATION_ONLY_PASS",
                    "bundle_sha256": args.bundle_sha256,
                    "output_root_absent": True,
                    "protected": protected,
                    "environment": environment,
                },
                sort_keys=True,
            )
        )
        return 0
    launcher = bundle / "launcher"
    launcher.mkdir(exist_ok=True)
    running = launcher / "storage_diag.running.json"
    terminal = launcher / "storage_diag.status.json"
    require(not terminal.exists(), "Existing terminal diagnostic status")
    protected_before = verify_protected(repo_root, metadata["protected_before"])
    environment = environment_record(metadata["environment_contract"]["packages"])
    gpu = device_record()
    publish_json_noreplace(
        running,
        {
            "status": "RUNNING",
            "bundle": str(bundle),
            "bundle_sha256": args.bundle_sha256,
            "host": os.uname().nodename,
            "started_at_utc": utc_now(),
            "environment": environment,
            "gpu": gpu,
        },
    )
    result: dict[str, Any] = {
        "status": "FAIL",
        "bundle": str(bundle),
        "bundle_sha256": args.bundle_sha256,
        "host": os.uname().nodename,
        "started_at_utc": utc_now(),
        "environment": environment,
        "gpu": gpu,
        "output_root": str(output_root),
        "mountinfo": mountinfo_for(output_root.parent),
        "statvfs_before": statvfs_record(output_root.parent),
        "protected_before": protected_before,
        "files": [],
        "error": None,
    }
    try:
        os.mkdir(output_root, 0o755)
        fsync_dir(output_root.parent)
        publish_json_noreplace(
            output_root / ".storage_diag_claim.json",
            {
                "status": "CLAIMED",
                "bundle_sha256": args.bundle_sha256,
                "host": os.uname().nodename,
                "uid": os.getuid(),
                "gid": os.getgid(),
                "created_at_utc": utc_now(),
            },
        )
        scene = output_root / f".scene.incomplete-{uuid.uuid4().hex}"
        os.mkdir(scene, 0o700)
        fsync_dir(output_root)
        tensor = torch.zeros(expected_shape, dtype=torch.float16, device="cpu")
        for layer in expected_layers:
            result["files"].append(write_tensor(scene, layer, tensor, expected_shape, expected_bytes))
        total = sum(int(item["size_bytes"]) for item in result["files"])
        require(total == expected_bytes * len(expected_layers), "Payload byte total mismatch")
        publish_json_noreplace(
            scene / "storage_diag_scene_complete.json",
            {
                "status": "COMPLETE",
                "bundle_sha256": args.bundle_sha256,
                "shape": expected_shape,
                "layers": expected_layers,
                "files": result["files"],
                "payload_bytes": total,
            },
        )
        result["status"] = "PASS"
        result["payload_bytes"] = total
        result["statvfs_after"] = statvfs_record(output_root)
        result["protected_after"] = verify_protected(repo_root, metadata["protected_before"])
        publish_json_noreplace(output_root / "storage_diag_complete.json", result)
        result["ended_at_utc"] = utc_now()
        publish_json_noreplace(terminal, result)
        print(json.dumps({"status": "PASS", "payload_bytes": total, "output_root": str(output_root)}, sort_keys=True))
        return 0
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["ended_at_utc"] = utc_now()
        try:
            result["statvfs_after"] = statvfs_record(output_root if output_root.exists() else output_root.parent)
            result["protected_after"] = verify_protected(repo_root, metadata["protected_before"])
            publish_json_noreplace(terminal, result)
        except Exception as status_exc:
            print(f"DIAGNOSTIC_STATUS_WRITE_FAILED: {type(status_exc).__name__}: {status_exc}", file=sys.stderr)
        print(result["traceback"], file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
