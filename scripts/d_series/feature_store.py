"""Fail-closed D-series feature store contracts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from safetensors.torch import load_file, save_file


MANIFEST_SHA256 = "781e25848de48fa5219aad876bd10cb8f9445328b145a19193a6f6b2ab5efcc9"


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def feature_paths(root: Path, model: str, space: str, sample_id: str):
    stem = root / model / space / sample_id
    return stem.with_suffix(".sft"), stem.with_suffix(".json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_provenance(paths):
    """Record the base commit and exact bytes of every extraction component."""
    repo = Path(__file__).resolve().parents[2]
    try:
        git = ["git", "-c", f"safe.directory={repo}", "-C", str(repo)]
        commit = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(git + ["status", "--short"], text=True).strip())
    except Exception:
        commit, dirty = "unknown", None
    resolved = [Path(path).resolve() for path in paths]
    return {
        "git_base_commit": commit,
        "git_dirty": dirty,
        "code_sha256": {str(path.relative_to(repo)): sha256_file(path) for path in resolved},
    }


def write_feature(root: Path, model: str, space: str, sample_id: str, tensors, receipt):
    payload_path, receipt_path = feature_paths(root, model, space, sample_id)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    required = {
        "schema_version": 1,
        "manifest_sha256": MANIFEST_SHA256,
        "model": model,
        "space": space,
        "sample_id": sample_id,
    }
    document = required | dict(receipt)
    declared_tensors = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype).replace("torch.", "")}
        for key, value in tensors.items()
    }
    if payload_path.exists() or receipt_path.exists():
        if not payload_path.is_file() or not receipt_path.is_file():
            raise RuntimeError(f"refusing incomplete existing feature pair: {payload_path}")
        existing = json.loads(receipt_path.read_text())
        expected_existing = document | {
            "payload_bytes": payload_path.stat().st_size,
            "payload_sha256": sha256_file(payload_path),
            "tensors": declared_tensors,
        }
        if existing != expected_existing:
            raise RuntimeError(f"refusing feature overwrite after receipt mismatch: {payload_path}")
        return
    temporary = payload_path.with_suffix(".sft.partial")
    save_file({key: value.detach().cpu().contiguous() for key, value in tensors.items()}, temporary)
    os.replace(temporary, payload_path)
    document["payload_bytes"] = payload_path.stat().st_size
    document["payload_sha256"] = sha256_file(payload_path)
    document["tensors"] = declared_tensors
    receipt_path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")


def load_verified(root: Path, model: str, space: str, sample_id: str, expected_revision: str):
    payload_path, receipt_path = feature_paths(root, model, space, sample_id)
    if not payload_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"missing feature pair: {payload_path}")
    receipt = json.loads(receipt_path.read_text())
    expected = {
        "manifest_sha256": MANIFEST_SHA256,
        "model": model,
        "space": space,
        "sample_id": sample_id,
        "model_revision": expected_revision,
        "payload_bytes": payload_path.stat().st_size,
        "payload_sha256": sha256_file(payload_path),
    }
    failures = {key: (receipt.get(key), value) for key, value in expected.items() if receipt.get(key) != value}
    if failures:
        raise ValueError(f"feature receipt mismatch for {sample_id}: {failures}")
    tensors = load_file(payload_path)
    declared = receipt.get("tensors", {})
    actual = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype).replace("torch.", "")}
        for key, value in tensors.items()
    }
    if actual != declared:
        raise ValueError(f"tensor header mismatch for {payload_path}")
    return tensors, receipt
