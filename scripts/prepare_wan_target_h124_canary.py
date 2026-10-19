#!/usr/bin/env python3
"""Freeze the direct-node Wan-p8 h1/h2/h4 target extraction canary.

This command only creates a versioned, immutable input/provenance bundle.  It
does not create the target output namespace and never loads a GPU model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COMMIT = "2075f3eb270cb0e8907c3927b6b079c64965b75a"
EXPECTED_MANIFEST_SHA = (
    "772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c"
)
EXPECTED_WRAPPER_SHA = (
    "ec6a13033d0e1a517a80632e5951a4fcc646740f34a267aa48627b9cb9e4002e"
)
EXPECTED_BASE_SHA = (
    "654644424e94ea8c9a1efecab5e1a65abf37321045d9b97a1f53e9bd668353d0"
)
EXPECTED_SOURCE_AUDIT_SHA = (
    "c09a11b6b60d97d162d264365e46339b9d1626d40b80665c18fa4783dffc4010"
)
EXPECTED_AUDITOR_SHA = (
    "d548a78b4ab9fbc565b14b82747c2011f4e77a9f7a0416e52347ed9dfb6f594a"
)
EXPECTED_RUNNER_SHA = (
    "ce9196ffcb7a8c4d4484e574fe6b9247c157567eca358e85c912a87fcc889efc"
)
EXPECTED_SCO_RUNNER_SHA = (
    "95fbe3119b0ab0e28b6328411ad1b2fc2620f4d18dfaa4a07de5d80d26ebbd3e"
)
EXPECTED_SCO_SUBMITTER_SHA = (
    "45bbdb08a36909c9fe162a332de8e0dd5ad262db3792c8fd21edd31dc99b785b"
)
EXPECTED_WAN_FEATURE_SHA = (
    "8af4cb15f7e9ecad09d813620609fb8f67ed3edd3853969448fec447f68d3a3a"
)
TRAIN_SCENE = "00777c41d4"
VAL_SCENE = "00a231a370"
MANIFEST = ROOT / "manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl"
SOURCE_H1_ROOT = Path(
    "/mnt/umm/users/yinbaiqiao/InsScene-15K/FEATURES_VERSIONED/"
    "wan_p8_action_dynamics_canonical_manifest_exact_20260821_200604_2075f3e/target"
)
SOURCE_H1_AUDIT = ROOT / (
    "reports/wan_p8_recovery/"
    "wan_p8_action_dynamics_canonical_manifest_exact_20260821_200604_2075f3e/"
    "full_feature_audit.json"
)
MODEL_SNAPSHOT = Path(
    "/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub/"
    "models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/"
    "0fad780a534b6463e45facd96134c9f345acfa5b"
)
DATA_ROOT = Path("/mnt/umm/users/yinbaiqiao/InsScene-15K/data")
RETIRED_PARENT_BUNDLE = ROOT / (
    "reports/q00_phase2_runs/"
    "q00_wan_p8_target_h124_canary_20260830T060728Z_2075f3eb"
)
RETIRED_PARENT_BUNDLE_SHA = (
    "34da6ba6963dd7b68fd7e66082adfe5a6ed557f541f2732e118c8148e5d3cf0a"
)
RETIRED_PARENT_VALIDATE_STATUS_SHA = (
    "7ac662783a7c8097efc6acb3e6f998607a6ccb2d08d60dbf6f9cb0fdfaea9d50"
)
RETIRED_PARENT_VALIDATE_LOG_SHA = (
    "f786d64d66a55ff1fd4442baa29bd90952a2143c0e412e3a71a05935293f94a8"
)
FAILED_SCO_BUNDLE = ROOT / (
    "reports/q00_phase2_runs/"
    "q00_wan_p8_target_h124_canary_20260830T092823Z_2075f3eb"
)
FAILED_SCO_BUNDLE_SHA = (
    "05bb70bb5a91cc353f7fd6a4cc97f984e04c9a9f675977f18126614de0f90c7c"
)
FAILED_SCO_CLAIM_SHA = (
    "492d1f1cffae0c3ac354767afee61e7631b657e1d9161681504dfa59dbfe15a7"
)
FAILED_SCO_LOG_SHA = (
    "4b419380f75cee335247e2f8bde5a3b1e069994dc87c9dfc84b2a0c3a7b9ed01"
)
FAILED_SCO_STATUS_SHA = (
    "c919253b7adae8d448ac39cb05dda303b00252f0e0d49dbf6e703518bf59b28e"
)
PROTECTED = {
    "experiments/registry.csv": "efc533bbe6286090d6c111c9d272ffb28cf7477b937f799f312eedf2a944431e",
    "results/canonical/a1_results.csv": "f7777676eafbc77dc8a8856e86a0901c98d6b5b3bdaddb568a975ef1c9f94d86",
    "results/canonical/a2_results.csv": "2613650bc7dcaa7a5bb5292c1953b9742657fbcb7967da3016b8c84c9f6ffa5f",
    "results/canonical/all_results.csv": "2a04a5e1f0165f1d7a06e7edcff38d956b845910ad95af3f7bbede3704133b17",
    "results/canonical/b2_results.csv": "0d4941709af83ba72820e4c463b96775d240096b44fca14bd4c5243e2dfa5f20",
    "results/canonical/c1_results.csv": "b88a37e239d150d297a4e97838d93888d5b6d3d0780df510e65b7f916b2ade1a",
    "results/canonical/c2_results.csv": "8c96a38500420af11427ed506bc92d4e1fc9d6b636bc34f0c4baf2d279fd9524",
    "results/canonical/c3_results.csv": "6cb2bec11abecb7cdf7549f14db84d58345e2bb5e9a0172f9435c80e5e2b7c5d",
}
SOURCE_OF_TRUTH = (
    "experiments/task_status_1096_before_expand_20260822.csv",
    "results/analysis/q00_1096/universe_1096_reconciled.csv",
    "results/analysis/q00_1096_phase1/next_training_queue.csv",
    "results/analysis/q00_1096_phase1/summary.json",
    "reports/q00_phase1_wave_b_e.md",
)
RUNTIME_FILES = (
    "scripts/extract_wan_target_h124_extend_exact.py",
    "scripts/extract_wan_manifest_exact.py",
    "scripts/audit_wan_target_h124_canary.py",
    "scripts/run_wan_target_h124_direct_canary.py",
    "scripts/run_wan_target_h124_sco_canary.py",
    "scripts/submit_wan_target_h124_sco_canary.py",
    "features/wan/wan_feature.py",
)


def runtime_source(relative: str) -> Path:
    # The live tracked featurizer is concurrently used by another campaign.
    # This canary deliberately preserves the already-audited parent runtime
    # instead of overwriting or silently adopting that unrelated dirty edit.
    if relative == "features/wan/wan_feature.py":
        return RETIRED_PARENT_BUNDLE / "runtime_tree/features/wan/wan_feature.py"
    return ROOT / relative


class FreezeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FreezeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_reserved_bundle(source: Path, destination: Path) -> None:
    """Publish without replacement on filesystems lacking renameat2 support.

    ``mkdir`` is the atomic no-replace reservation.  The destination remains
    invalid until ``bundle.json`` is copied with O_EXCL as the final validity
    marker.  A crash can therefore leave an inspectable incomplete directory,
    but can never expose a valid partial bundle or overwrite an existing one.
    """
    require(source.is_dir(), f"Missing bundle staging directory: {source}")
    bundle_source = source / "bundle.json"
    require(bundle_source.is_file(), f"Missing staged bundle.json: {bundle_source}")
    try:
        destination.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise FreezeError(f"Bundle appeared during publish: {destination}") from exc

    try:
        for child in sorted(source.iterdir(), key=lambda item: item.name):
            if child.name == "bundle.json":
                continue
            target = destination / child.name
            require(not os.path.lexists(target), f"Publish target appeared: {target}")
            os.rename(child, target)
        fsync_directory(destination)

        bundle_target = destination / "bundle.json"
        bundle_private = destination / f".bundle.json.incomplete-{os.getpid()}-{uuid.uuid4().hex}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(bundle_private, flags, 0o400)
        try:
            with bundle_source.open("rb") as reader, os.fdopen(descriptor, "wb", closefd=False) as writer:
                shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
        finally:
            os.close(descriptor)
        require(sha256(bundle_source) == sha256(bundle_private), "Private bundle.json SHA mismatch")
        try:
            os.link(bundle_private, bundle_target, follow_symlinks=False)
        except FileExistsError as exc:
            raise FreezeError(f"Bundle marker appeared during publish: {bundle_target}") from exc
        require(sha256(bundle_source) == sha256(bundle_target), "Published bundle.json SHA mismatch")
        fsync_directory(destination)
        bundle_private.unlink()
        bundle_source.unlink()
        source.rmdir()
        fsync_directory(destination.parent)
    except Exception:
        # An incomplete reserved destination intentionally has no bundle.json
        # unless the final marker copy itself completed.  Preserve all evidence.
        raise


def copy_source(source: Path, destination: Path) -> None:
    require(source.is_file(), f"Missing frozen source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    require(sha256(source) == sha256(destination), f"Copy mismatch: {source}")


def retired_parent_receipt() -> dict[str, Any]:
    bundle_path = RETIRED_PARENT_BUNDLE / "bundle.json"
    status_path = RETIRED_PARENT_BUNDLE / "launcher/validate.status.json"
    log_path = RETIRED_PARENT_BUNDLE / "launcher/validate.log"
    require(bundle_path.is_file(), "Retired parent bundle is missing")
    require(sha256(bundle_path) == RETIRED_PARENT_BUNDLE_SHA, "Retired parent bundle SHA mismatch")
    require(
        status_path.is_file() and sha256(status_path) == RETIRED_PARENT_VALIDATE_STATUS_SHA,
        "Retired parent validation receipt mismatch",
    )
    require(
        log_path.is_file() and sha256(log_path) == RETIRED_PARENT_VALIDATE_LOG_SHA,
        "Retired parent validation log mismatch",
    )
    parent = json.loads(bundle_path.read_text())
    status = json.loads(status_path.read_text())
    require(
        status.get("status") == "PASS"
        and status.get("terminal_phase") == "COMPLETE"
        and status.get("action") == "validate"
        and status.get("returncode") == 0
        and status.get("bundle_sha256") == RETIRED_PARENT_BUNDLE_SHA,
        "Retired parent validation is not exact PASS",
    )
    output_root = Path(str(parent.get("expected_output_root", "")))
    require(output_root.is_absolute(), "Retired parent output root is invalid")
    require(not output_root.parent.exists(), "Retired parent output namespace now exists")
    forbidden = [
        RETIRED_PARENT_BUNDLE / "launcher/extract.running.json",
        RETIRED_PARENT_BUNDLE / "launcher/extract.status.json",
        RETIRED_PARENT_BUNDLE / "launcher/extract.log",
        RETIRED_PARENT_BUNDLE / "audit/canary_audit.json",
    ]
    require(not any(path.exists() for path in forbidden), "Retired parent has extraction output")
    return {
        "bundle": str(RETIRED_PARENT_BUNDLE.resolve()),
        "bundle_sha256": RETIRED_PARENT_BUNDLE_SHA,
        "validate_status": str(status_path.resolve()),
        "validate_status_sha256": RETIRED_PARENT_VALIDATE_STATUS_SHA,
        "validate_log": str(log_path.resolve()),
        "validate_log_sha256": RETIRED_PARENT_VALIDATE_LOG_SHA,
        "expected_output_root": str(output_root),
        "output_namespace_absent": True,
        "extract_artifacts_absent": True,
        "disposition": "VALIDATE_ONLY_PASS_RETIRED_BEFORE_EXTRACTION",
    }


def failed_sco_attempt_receipt() -> dict[str, Any]:
    bundle_path = FAILED_SCO_BUNDLE / "bundle.json"
    claim_path = FAILED_SCO_BUNDLE / "launcher/sco_submit.claim.json"
    log_path = FAILED_SCO_BUNDLE / "launcher/sco_submit.log"
    status_path = FAILED_SCO_BUNDLE / "launcher/sco_submit.status.json"
    expected = {
        bundle_path: FAILED_SCO_BUNDLE_SHA,
        claim_path: FAILED_SCO_CLAIM_SHA,
        log_path: FAILED_SCO_LOG_SHA,
        status_path: FAILED_SCO_STATUS_SHA,
    }
    for path, digest in expected.items():
        require(path.is_file() and sha256(path) == digest, f"Failed SCO receipt mismatch: {path}")
    bundle = json.loads(bundle_path.read_text())
    status = json.loads(status_path.read_text())
    require(
        status.get("status") == "FAIL"
        and status.get("returncode") == 0
        and status.get("job_id") == "pt-67y61dvq"
        and status.get("job_ids") == ["pt-67y61dvq"]
        and status.get("stop_returncode") == 0
        and status.get("bundle_sha256") == FAILED_SCO_BUNDLE_SHA,
        "Failed SCO terminal receipt is not exact",
    )
    require("SCO worker spec mismatch" in str(status.get("error")), "Unexpected failed SCO cause")
    output_root = Path(str(bundle.get("expected_output_root", "")))
    require(output_root.is_absolute() and not output_root.parent.exists(), "Failed SCO output exists")
    forbidden = [
        FAILED_SCO_BUNDLE / "launcher/sco_extract.running.json",
        FAILED_SCO_BUNDLE / "launcher/sco_extract.status.json",
        FAILED_SCO_BUNDLE / "launcher/sco_extract.log",
        FAILED_SCO_BUNDLE / "audit/canary_audit.json",
    ]
    require(not any(path.exists() for path in forbidden), "Failed SCO worker produced output")
    return {
        "bundle": str(FAILED_SCO_BUNDLE.resolve()),
        "bundle_sha256": FAILED_SCO_BUNDLE_SHA,
        "submit_claim_sha256": FAILED_SCO_CLAIM_SHA,
        "submit_log_sha256": FAILED_SCO_LOG_SHA,
        "submit_status_sha256": FAILED_SCO_STATUS_SHA,
        "job_id": "pt-67y61dvq",
        "stop_receipt_returncode": 0,
        "scheduler_state_observed_after_stop": "SUSPENDED",
        "worker_start_time": None,
        "output_namespace_absent": True,
        "worker_output_absent": True,
        "disposition": "SUBMIT_PREFLIGHT_FAIL_STOPPED_NO_OUTPUT",
    }
def read_manifest() -> list[dict[str, Any]]:
    require(sha256(MANIFEST) == EXPECTED_MANIFEST_SHA, "Manifest SHA mismatch")
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    require(len(rows) == 2724, f"Manifest rows={len(rows)}, expected 2724")
    require(len({str(row["sample_id"]) for row in rows}) == 2724, "Duplicate sample IDs")
    require(Counter(str(row["split"]) for row in rows) == {"train": 2448, "val": 276}, "Split counts changed")
    return rows


def target_id(row: dict[str, Any], horizon: int) -> int:
    tail = int(row["streaming_tail"])
    window = [int(value) for value in row["streaming_window_indices"]]
    position = tail + horizon
    require(position < len(window), f"Target outside window: {row['sample_id']} h{horizon}")
    return window[position]


def select_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in rows if str(row["scene_id"]) in {TRAIN_SCENE, VAL_SCENE}]
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_scene[str(row["scene_id"])].append(row)
    require(set(by_scene) == {TRAIN_SCENE, VAL_SCENE}, "Canary scenes missing")
    require({str(row["split"]) for row in by_scene[TRAIN_SCENE]} == {"train"}, "Train scene split mismatch")
    require({str(row["split"]) for row in by_scene[VAL_SCENE]} == {"val"}, "Val scene split mismatch")
    for scene, scene_rows in by_scene.items():
        require(len(scene_rows) == 4, f"{scene}: expected four rows")
        ids = [target_id(row, h) for row in scene_rows for h in (1, 2, 4)]
        require(len(ids) == len(set(ids)) == 12, f"{scene}: h124 targets not disjoint")
    return sorted(selected, key=lambda row: str(row["sample_id"]))


def artifact_record(path: Path, logical_root: Path | None = None) -> dict[str, Any]:
    require(path.is_file(), f"Missing artifact: {path}")
    logical = path.relative_to(logical_root).as_posix() if logical_root else str(path)
    return {
        "path": logical,
        "realpath": str(path.resolve()),
        "is_symlink": path.is_symlink(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def source_h1_inventory(selected: list[dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for scene in sorted({str(row["scene_id"]) for row in selected}):
        scene_root = SOURCE_H1_ROOT / "wan/scannetpp" / scene
        paths = [scene_root / "target_indices.npy", Path(f"{scene_root / 'target_indices.npy'}.manifest.json")]
        for layer in (0, 5, 10, 15, 25, 29):
            feature = scene_root / f"feature_t749_layer{layer}.sft"
            paths.extend((feature, Path(f"{feature}.manifest.json")))
        for path in paths:
            record = artifact_record(path)
            record["scene"] = scene
            records.append(record)
    return {"root": str(SOURCE_H1_ROOT.resolve()), "artifacts": records}


def input_image_inventory(selected: list[dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    image_base = DATA_ROOT / "processed_scannetpp_v2/processed_scannetpp_v2"
    for row in selected:
        scene = str(row["scene_id"])
        names = row.get("valid_frames")
        require(isinstance(names, list), f"{row['sample_id']}: valid_frames missing")
        for horizon in (2, 4):
            raw_id = target_id(row, horizon)
            image = image_base / scene / "images" / str(names[raw_id])
            logical = str(image.absolute())
            require(logical not in seen, f"Duplicate canary input image: {image}")
            seen.add(logical)
            record = artifact_record(image)
            record.update(
                {
                    "scene": scene,
                    "sample_id": str(row["sample_id"]),
                    "horizon": horizon,
                    "raw_id": raw_id,
                    "frame_name": str(names[raw_id]),
                }
            )
            records.append(record)
    require(len(records) == 16, f"Canary input-image count={len(records)}, expected 16")
    return {"root": str(DATA_ROOT.resolve()), "artifacts": records}


def model_inventory() -> list[dict[str, Any]]:
    require(MODEL_SNAPSHOT.is_dir(), f"Missing model snapshot: {MODEL_SNAPSHOT}")
    paths = sorted(path for path in MODEL_SNAPSHOT.rglob("*") if path.is_file())
    require(paths, "Model snapshot contains no files")
    return [artifact_record(path, MODEL_SNAPSHOT) for path in paths]


def command_output(argv: list[str]) -> str:
    try:
        return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT, timeout=60)
    except Exception as exc:
        return f"UNAVAILABLE: {exc}\n"


def environment_record() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("torch", "diffusers", "transformers", "safetensors", "numpy", "Pillow", "accelerate"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "expected_direct_host": "pt-314e0f9747cc44dc9e91dc07705c52a4-worker-0",
        "required_gpu_name": "NVIDIA H800",
        "required_min_free_mib": 75000,
        "required_cuda_visible_device_count": 1,
    }


def build(args: argparse.Namespace) -> Path:
    require(re.fullmatch(r"q00_wan_p8_target_h124_canary_[0-9]{8}T[0-9]{6}Z_2075f3eb", args.run_tag) is not None, "Unsafe/non-versioned run tag")
    final_bundle = ROOT / "reports/q00_phase2_runs" / args.run_tag
    final_output = Path(
        "/mnt/umm/users/yinbaiqiao/InsScene-15K/FEATURES_VERSIONED"
    ) / args.run_tag / "target"
    preparer_path = Path(__file__).resolve()
    preparer_start_sha = sha256(preparer_path)
    runtime_expected = {
        "scripts/extract_wan_target_h124_extend_exact.py": EXPECTED_WRAPPER_SHA,
        "scripts/extract_wan_manifest_exact.py": EXPECTED_BASE_SHA,
        "scripts/audit_wan_target_h124_canary.py": EXPECTED_AUDITOR_SHA,
        "scripts/run_wan_target_h124_direct_canary.py": EXPECTED_RUNNER_SHA,
        "scripts/run_wan_target_h124_sco_canary.py": EXPECTED_SCO_RUNNER_SHA,
        "scripts/submit_wan_target_h124_sco_canary.py": EXPECTED_SCO_SUBMITTER_SHA,
        "features/wan/wan_feature.py": EXPECTED_WAN_FEATURE_SHA,
    }
    require(set(runtime_expected) == set(RUNTIME_FILES), "Runtime expected-hash closure mismatch")
    require(not final_bundle.exists(), f"Bundle already exists: {final_bundle}")
    require(not final_output.exists(), f"Output root already exists: {final_output}")
    require(sha256(ROOT / "scripts/extract_wan_target_h124_extend_exact.py") == EXPECTED_WRAPPER_SHA, "Extractor wrapper SHA changed")
    require(sha256(ROOT / "scripts/extract_wan_manifest_exact.py") == EXPECTED_BASE_SHA, "Base extractor SHA changed")
    require(sha256(SOURCE_H1_AUDIT) == EXPECTED_SOURCE_AUDIT_SHA, "Source h1 audit SHA changed")
    auditor = ROOT / "scripts/audit_wan_target_h124_canary.py"
    require(auditor.is_file(), "Independent canary auditor is not yet frozen")
    require(sha256(auditor) == EXPECTED_AUDITOR_SHA, "Auditor SHA changed")
    require(
        sha256(ROOT / "scripts/run_wan_target_h124_direct_canary.py")
        == EXPECTED_RUNNER_SHA,
        "Direct runner SHA changed",
    )
    require(
        sha256(ROOT / "scripts/run_wan_target_h124_sco_canary.py")
        == EXPECTED_SCO_RUNNER_SHA,
        "SCO runner SHA changed",
    )
    require(
        sha256(ROOT / "scripts/submit_wan_target_h124_sco_canary.py")
        == EXPECTED_SCO_SUBMITTER_SHA,
        "SCO submitter SHA changed",
    )
    require(
        sha256(runtime_source("features/wan/wan_feature.py")) == EXPECTED_WAN_FEATURE_SHA,
        "Frozen parent Wan featurizer SHA changed",
    )
    protected = {path: sha256(ROOT / path) for path in PROTECTED}
    require(protected == PROTECTED, "Protected pre-image mismatch")
    rows = read_manifest()
    selected = select_rows(rows)
    commit = command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).strip()
    require(commit == EXPECTED_COMMIT, f"Git commit mismatch: {commit}")
    parent_receipt = retired_parent_receipt()
    failed_sco_receipt = failed_sco_attempt_receipt()

    final_bundle.parent.mkdir(parents=True, exist_ok=True)
    staging = final_bundle.parent / f".{args.run_tag}.incomplete-{os.getpid()}-{uuid.uuid4().hex}"
    require(not staging.exists(), f"Unexpected staging path: {staging}")
    staging.mkdir()
    try:
        for relative in SOURCE_OF_TRUTH:
            copy_source(ROOT / relative, staging / "inputs/source_of_truth" / relative)
        copy_source(MANIFEST, staging / "inputs/manifest.jsonl")
        copy_source(SOURCE_H1_AUDIT, staging / "inputs/source_h1_full_audit.json")
        for relative in RUNTIME_FILES:
            copy_source(runtime_source(relative), staging / "runtime_tree" / relative)
        copy_source(preparer_path, staging / "provenance/scripts" / preparer_path.name)
        for relative, expected in runtime_expected.items():
            require(
                sha256(runtime_source(relative)) == expected
                and sha256(staging / "runtime_tree" / relative) == expected,
                f"Post-copy runtime SHA mismatch: {relative}",
            )
        require(
            sha256(preparer_path) == preparer_start_sha
            and sha256(staging / "provenance/scripts" / preparer_path.name)
            == preparer_start_sha,
            "Preparer changed during freeze copy",
        )
        test_path = ROOT / "tests/test_wan_target_h124_extend_exact.py"
        copy_source(test_path, staging / "provenance/tests" / test_path.name)
        audit_test = ROOT / "tests/test_audit_wan_target_h124_canary.py"
        copy_source(audit_test, staging / "provenance/tests" / audit_test.name)
        sco_test = ROOT / "tests/test_wan_target_h124_sco_execution.py"
        copy_source(sco_test, staging / "provenance/tests" / sco_test.name)

        selection = staging / "inputs/canary_sample_ids.txt"
        selection.write_text("\n".join(str(row["sample_id"]) for row in selected) + "\n")
        write_json(staging / "inputs/canary_rows.json", selected)
        write_json(staging / "inputs/source_h1_inventory.json", source_h1_inventory(selected))
        write_json(staging / "inputs/canary_input_image_inventory.json", input_image_inventory(selected))
        write_json(staging / "inputs/model_snapshot_inventory.json", {"root": str(MODEL_SNAPSHOT), "artifacts": model_inventory()})
        write_json(staging / "inputs/protected_before.json", protected)
        write_json(staging / "provenance/environment_contract.json", environment_record())
        (staging / "provenance/git_commit.txt").write_text(EXPECTED_COMMIT + "\n")
        (staging / "provenance/git_status_porcelain.txt").write_text(
            command_output(["git", "-C", str(ROOT), "status", "--short", "--untracked-files=all"])
        )
        (staging / "provenance/dirty.diff").write_text(
            command_output(["git", "-C", str(ROOT), "diff", "--binary"])
        )
        write_json(staging / "provenance/retired_parent_receipt.json", parent_receipt)
        write_json(staging / "provenance/failed_sco_attempt_receipt.json", failed_sco_receipt)

        # Planned paths bind the final, versioned bundle name rather than the
        # hidden construction directory.  The direct launcher independently
        # rebinds and verifies these paths against its resolved --bundle.
        frozen_wrapper = final_bundle / "runtime_tree/scripts/extract_wan_target_h124_extend_exact.py"
        frozen_manifest = final_bundle / "inputs/manifest.jsonl"
        frozen_selection = final_bundle / "inputs/canary_sample_ids.txt"
        frozen_audit = final_bundle / "inputs/source_h1_full_audit.json"
        cli = [
            "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python", "-B", str(frozen_wrapper),
            "--manifest", str(frozen_manifest), "--selection-file", str(frozen_selection),
            "--data-root", str(DATA_ROOT), "--source-h1-root", str(SOURCE_H1_ROOT),
            "--source-h1-audit", str(frozen_audit), "--out-root", str(final_output),
            "--model-id", str(MODEL_SNAPSHOT), "--shard-name", "canary_train_val_2scenes",
            "--repo-root", str(ROOT.resolve()), "--expected-git-commit", EXPECTED_COMMIT,
            "--fail-on-existing",
        ]
        (staging / "provenance/planned_cli.txt").write_text(" ".join(cli) + "\n")

        source_hashes: dict[str, str] = {}
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            source_hashes[path.relative_to(staging).as_posix()] = sha256(path)
        metadata = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_tag": args.run_tag,
            "bundle_path": str(final_bundle),
            "expected_output_root": str(final_output),
            "output_namespace_must_not_exist_before_run": True,
            "git_commit": EXPECTED_COMMIT,
            "runtime_source_overrides": {
                "features/wan/wan_feature.py": {
                    "source": str(runtime_source("features/wan/wan_feature.py").resolve()),
                    "sha256": EXPECTED_WAN_FEATURE_SHA,
                    "reason": "preserve_parent_audited_runtime_while_live_tracked_file_is_dirty",
                }
            },
            "parent_phase1_summary_sha256": sha256(ROOT / "results/analysis/q00_1096_phase1/summary.json"),
            "supersedes_execution_bundle": {
                "path": parent_receipt["bundle"],
                "bundle_sha256": parent_receipt["bundle_sha256"],
                "validate_status_sha256": parent_receipt["validate_status_sha256"],
                "validate_log_sha256": parent_receipt["validate_log_sha256"],
                "output_namespace_absent": parent_receipt["output_namespace_absent"],
                "extract_artifacts_absent": parent_receipt["extract_artifacts_absent"],
                "disposition": parent_receipt["disposition"],
                "reason": (
                    "Its frozen extractor used renameat2(RENAME_NOREPLACE), which the "
                    "mounted project filesystem rejects with EINVAL. No extraction or "
                    "target namespace was created."
                ),
            },
            "supersedes_failed_sco_attempt": failed_sco_receipt,
            "contract": {
                "campaign": "wan_cog_layer_sweep",
                "model": "wan",
                "model_variant": "multi_layer/run_dir_discovered",
                "protocol": "historical_candidate_v1_val_protocol",
                "prefix": 8,
                "timestep": 749,
                "layers": [0, 5, 10, 15, 25, 29],
                "horizons": [1, 2, 4],
                "reuse_horizons": [1],
                "extract_horizons": [2, 4],
                "feature_shape": [30, 52, 1536],
                "input_frames": 81,
                "prompt": ".",
                "seed": 42,
                "git_commit": EXPECTED_COMMIT,
                "manifest_sha256": EXPECTED_MANIFEST_SHA,
                "source_h1_audit_sha256": EXPECTED_SOURCE_AUDIT_SHA,
                "base_extractor_sha256": EXPECTED_BASE_SHA,
                "shard_name": "canary_train_val_2scenes",
                "canary_scenes": {"train": TRAIN_SCENE, "val": VAL_SCENE},
                "canary_samples": len(selected),
                "new_forward_count": 16,
                "publication_contract": "mkdir_reservation_plus_same_directory_hardlink_noreplace_v1",
            },
            "external_roots": {
                "repo_root": str(ROOT.resolve()),
                "data_root": str(DATA_ROOT.resolve()),
                "source_h1_root": str(SOURCE_H1_ROOT.resolve()),
                "model_snapshot": str(MODEL_SNAPSHOT.resolve()),
                "out_root": str(final_output),
            },
            "sco_execution_contract": {
                "execution_tag": f"{args.run_tag}_sco_h800",
                "workspace": "aigc",
                "aec2_name": "zoetrope",
                "job_name": f"bq_q00_wh124_{args.run_tag.split('_canary_', 1)[1].split('_2075f3eb', 1)[0].replace('T', '').replace('Z', '')}",
                "priority": "HIGHEST",
                "quota_type": "reserved",
                "worker_nodes": 1,
                "worker_spec": "N4lS.Iq.I80.1",
                "container_image": "registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617",
                "container_image_digest": "UNAVAILABLE_TAG_PIN_ONLY",
                "storage_mount": "019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm",
                "hostname_regex": "^pt-[a-z0-9]+-worker-0$",
                "retired_direct_host": "pt-314e0f9747cc44dc9e91dc07705c52a4-worker-0",
                "logical_gpu": 0,
                "required_gpu_name": "NVIDIA H800",
                "required_min_free_mib": 75000,
                "automatic_retry": False,
            },
            "protected_preimages": protected,
            "source_hashes": source_hashes,
            "planned_cli": cli,
            "mutable_outputs": [
                "launcher/", "audit/", "logs/", str(final_output),
            ],
            "execution_gate": "CPU validate-only PASS; frozen direct or SCO runner contract; one H800; >=75000 MiB free; no live holder; independent audit PASS",
        }
        write_json(staging / "bundle.json", metadata)
        for relative, expected in runtime_expected.items():
            require(
                sha256(runtime_source(relative)) == expected
                and sha256(staging / "runtime_tree" / relative) == expected,
                f"Pre-publish runtime SHA mismatch: {relative}",
            )
        require(
            sha256(preparer_path) == preparer_start_sha
            and sha256(staging / "provenance/scripts" / preparer_path.name)
            == preparer_start_sha,
            "Preparer changed before bundle publication",
        )
        require(
            retired_parent_receipt() == parent_receipt,
            "Retired parent state changed during bundle construction",
        )
        require(
            failed_sco_attempt_receipt() == failed_sco_receipt,
            "Failed SCO attempt state changed during bundle construction",
        )
        publish_reserved_bundle(staging, final_bundle)
    except Exception:
        # Preserve the incomplete staging directory as provenance; never reuse it.
        raise
    print(json.dumps({"status": "PASS", "bundle": str(final_bundle), "bundle_sha256": sha256(final_bundle / "bundle.json"), "output_root": str(final_output)}, sort_keys=True))
    return final_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        build(parse_args())
    except FreezeError as exc:
        raise SystemExit(f"FREEZE FAILURE: {exc}") from exc
