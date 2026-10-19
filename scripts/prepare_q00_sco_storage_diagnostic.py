#!/usr/bin/env python3
"""Freeze a minimal, non-scientific SCO storage diagnostic bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
FEATURES_VERSIONED = Path("/mnt/umm/users/yinbaiqiao/InsScene-15K/FEATURES_VERSIONED")
EXPECTED_PYTHON = Path("/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python3.11")
PROTECTED = [
    "experiments/registry.csv",
    "results/canonical/a1_results.csv",
    "results/canonical/a2_results.csv",
    "results/canonical/all_results.csv",
    "results/canonical/b2_results.csv",
    "results/canonical/c1_results.csv",
    "results/canonical/c2_results.csv",
    "results/canonical/c3_results.csv",
]


class PrepareError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PrepareError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    private = path.parent / f".{path.name}.incomplete-{os.getpid()}-{uuid.uuid4().hex}"
    descriptor: int | None = None
    try:
        descriptor = os.open(private, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
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
    finally:
        if descriptor is not None:
            os.close(descriptor)


def git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=False, text=True, capture_output=True
    )
    require(result.returncode == 0, f"git rev-parse failed: {result.stderr.strip()}")
    commit = result.stdout.strip()
    require(len(commit) == 40, f"Invalid git commit: {commit}")
    return commit


def protected_hashes(repo: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in PROTECTED:
        path = repo / relative
        require(path.is_file() and not path.is_symlink(), f"Missing unsafe protected file: {relative}")
        result[relative] = sha256(path)
    return result


def package_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in ("torch", "safetensors", "numpy")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_tag = str(args.run_tag)
    require(run_tag.startswith("q00_wan_p8_h124_storage_diag_"), "Unexpected run-tag namespace")
    require(all(character.isalnum() or character in "_-" for character in run_tag), "Unsafe run-tag")
    repo = Path(args.repo_root).resolve()
    require(repo == REPO_ROOT.resolve(), "Unexpected repository root")
    worker_source = repo / "scripts/q00_sco_storage_diagnostic.py"
    require(worker_source.is_file() and not worker_source.is_symlink(), "Missing worker source")
    target_bundle = repo / "reports/q00_phase2_runs" / run_tag
    output_root = FEATURES_VERSIONED / run_tag
    require(not target_bundle.exists(), f"Bundle already exists: {target_bundle}")
    require(not output_root.exists(), f"Output already exists: {output_root}")
    target_bundle.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(target_bundle, 0o755)
    try:
        runtime = target_bundle / "runtime_tree/scripts"
        runtime.mkdir(parents=True)
        frozen_worker = runtime / worker_source.name
        shutil.copy2(worker_source, frozen_worker)
        os.chmod(frozen_worker, 0o555)
        worker_sha = sha256(worker_source)
        require(sha256(frozen_worker) == worker_sha, "Frozen worker SHA mismatch")
        launcher = target_bundle / "launcher"
        launcher.mkdir()
        os.chmod(launcher, 0o777)
        metadata: dict[str, Any] = {
            "kind": "q00_wan_h124_storage_diagnostic",
            "schema_version": 1,
            "run_tag": run_tag,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(repo),
            "bundle_path": str(target_bundle),
            "external_roots": {
                "repo_root": str(repo),
                "output_root": str(output_root),
                "output_parent": str(output_root.parent),
            },
            "source_hashes": {
                "runtime_tree/scripts/q00_sco_storage_diagnostic.py": worker_sha,
            },
            "environment_contract": {
                "python_executable": str(EXPECTED_PYTHON),
                "packages": package_versions(),
            },
            "diagnostic_contract": {
                "purpose": "prove or fail the exact one-scene safetensors write/fsync/readback chain before feature retry",
                "tensor_shape": [12, 30, 52, 1536],
                "dtype": "float16",
                "layers": [0, 5, 10, 15, 25, 29],
                "file_size_bytes": 57507928,
                "scene_payload_bytes": 345047568,
                "writes_new_model_forwards": False,
                "writes_features": False,
                "trains": False,
                "evaluates": False,
                "canonical_write": False,
                "no_cleanup_on_failure": True,
            },
            "sco_execution_contract": {
                "workspace": "aigc",
                "aec2_name": "zoetrope",
                "worker_spec": "N4lS.Iq.I80.1",
                "container_image": "registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617",
                "storage_mount": "019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm",
                "worker_nodes": 1,
                "priority": "HIGHEST",
                "quota_type": "reserved",
                "framework": "pytorch",
                "max_retries": 0,
                "container_image_digest_policy": "TAG_PIN_ONLY_ACCEPTED_FOR_DIAGNOSTIC",
            },
            "protected_before": protected_hashes(repo),
            "failure_being_diagnosed": {
                "job_id": "pt-xz63jp91",
                "bundle_sha256": "a48e37ddf1bed2ae24565e2c17ffc8bfd166bda76ca726ee8c8a1f619b565856",
                "log_sha256": "822e4a8cbc645fc3d1adab8184f80ed373d3b688baff687e001cecd38c73a5e2",
                "error": "safetensors ENOSPC while writing the second layer of the first target scene",
            },
        }
        atomic_json(target_bundle / "bundle.json", metadata)
        require(sha256(frozen_worker) == worker_sha, "Frozen worker changed before publication")
        require(protected_hashes(repo) == metadata["protected_before"], "Protected file changed during preparation")
        print(
            json.dumps(
                {
                    "bundle": str(target_bundle),
                    "bundle_sha256": sha256(target_bundle / "bundle.json"),
                    "output_root": str(output_root),
                    "worker_sha256": worker_sha,
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception:
        # The reserved bundle is intentionally retained as failed-preparation evidence.
        raise


if __name__ == "__main__":
    raise SystemExit(main())
