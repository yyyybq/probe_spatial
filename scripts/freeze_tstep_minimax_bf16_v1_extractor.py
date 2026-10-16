#!/usr/bin/env python3
"""Freeze the extractor runtime for the MiniMax-H3 bf16 seeded-v1 re-extraction.

Stage 0 of docs/MINIMAX_REEXTRACT_RETRAIN_PLAN_20261016.md.  Copies exactly the
files that scripts/run_streaming_feat_extract_8gpu.sh and its sync loop execute
into runtime/tstep_minimax_bf16_v1_extractor/source and records their identity:

  source.sha256         ``sha256sum -c`` list (relative paths)
  source_manifest.json  tree digest, per-file hashes, git HEAD, file list

The tree digest uses the same algorithm as runtime/main_v2_minimax_bf16_v3_extractor
(sorted relative paths; for each: 8-byte big-endian path length, path bytes,
raw SHA-256 of the file), so job launchers can verify the frozen tree with the
same snippet.  An existing freeze is never modified: re-running without
``--verify`` fails, ``--verify`` re-hashes the frozen tree and reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASES = {
    # r1 (frozen 2026-10-16): canary attempts 1-2 (pt-6g2hggcx, pt-fxzmatkf).
    "tstep_minimax_bf16_v1_extractor": None,
    # r2: the sync loop defers units whose scene structural index is not yet
    # written instead of aborting, and fails the run if any are still pending
    # once extraction stops.  r1 aborted on the first scene of a new namespace.
    "tstep_minimax_bf16_v1_extractor_r2": "tstep_minimax_bf16_v1_extractor",
}
CURRENT_RELEASE = "tstep_minimax_bf16_v1_extractor_r2"
RELEASE = ROOT / "runtime" / CURRENT_RELEASE
SOURCE = RELEASE / "source"


def select_release(name: str) -> None:
    global RELEASE, SOURCE
    if name not in RELEASES:
        raise SystemExit(f"unknown release {name!r}; known: {sorted(RELEASES)}")
    RELEASE = ROOT / "runtime" / name
    SOURCE = RELEASE / "source"
SCHEMA = "TSTEP_MINIMAX_BF16_V1_EXTRACTOR_IMMUTABLE_SOURCE_V1"
STORAGE_CONTRACT = "MINIMAX_H3_FINITE_BF16_V1"
NAMESPACES = [
    "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1",
    "FEAT_TARGET_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1",
]
# Everything the 8-GPU launcher, the sync loop and features.run_inscene15k
# (MiniMax-H3 branch) touch at run time.  lingbot/* and the cleanup script are
# only size-checked by the launcher's require_min_bytes guard.
FILES = [
    "features/run_inscene15k.py",
    "features/noise_seed.py",
    "features/offline_hf.py",
    "features/canonical_gap_worklist.py",
    "features/minimax_h3/extract_features.py",
    "features/minimax_h3/minimax_h3_feature.py",
    "features/lingbot/extract_features.py",
    "features/lingbot/lingbot_feature.py",
    "scripts/run_streaming_feat_extract_8gpu.sh",
    "scripts/run_tstep_sync_loop.sh",
    "scripts/delete_aoss_confirmed_local_pairs.sh",
    "vidfm3d/utils/feature_layers.py",
    "vidfm3d/utils/temporal_sampling.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(root: Path) -> tuple[str, list[Path]]:
    files = sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    digest = hashlib.sha256()
    for path in files:
        rel = path.relative_to(root).as_posix().encode()
        digest.update(len(rel).to_bytes(8, "big"))
        digest.update(rel)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest(), files


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance only
        return ""


def freeze() -> int:
    if RELEASE.exists():
        raise SystemExit(f"{RELEASE} already exists; a freeze is immutable (use --verify)")
    missing = [rel for rel in FILES if not (ROOT / rel).is_file()]
    if missing:
        raise SystemExit(f"missing source files: {missing}")
    SOURCE.mkdir(parents=True)
    records = []
    for rel in FILES:
        src = ROOT / rel
        dst = SOURCE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        shutil.copymode(src, dst)
        os.chmod(dst, 0o555 if os.access(src, os.X_OK) else 0o444)
        records.append({"path": rel, "sha256": sha256(dst), "bytes": dst.stat().st_size})
        if records[-1]["sha256"] != sha256(src):
            raise SystemExit(f"copy mismatch for {rel}")
    digest, files = tree_digest(SOURCE)
    if len(files) != len(FILES):
        raise SystemExit("frozen tree holds unexpected files")
    sha_file = RELEASE / "source.sha256"
    sha_file.write_text("".join(f"{r['sha256']}  {r['path']}\n" for r in records), encoding="utf-8")
    os.chmod(sha_file, 0o444)
    manifest = {
        "schema": SCHEMA,
        "release": RELEASE.name,
        "supersedes": RELEASES.get(RELEASE.name),
        "status": "FROZEN",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "bf16 re-extraction of MiniMax-H3 seeded-v1 timestep caches (t500/t1000, layers 20/25)",
        "plan": "docs/MINIMAX_REEXTRACT_RETRAIN_PLAN_20261016.md",
        "storage_contract": STORAGE_CONTRACT,
        "feature_namespaces": NAMESPACES,
        "source": str(SOURCE),
        "source_sha256_file": str(sha_file),
        "source_sha256_file_sha256": sha256(sha_file),
        "source_tree_sha256": digest,
        "files": len(records),
        "bytes": sum(r["bytes"] for r in records),
        "file_records": records,
        "main_tree_git_head": git("rev-parse", "HEAD"),
        "main_tree_git_dirty": bool(git("status", "--porcelain", "--", *FILES)),
        "launcher": "scripts/run_streaming_feat_extract_8gpu.sh (run from this source tree; PROJECT_ROOT resolves to it)",
    }
    (RELEASE / "source_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(RELEASE / "source_manifest.json", 0o444)
    print(json.dumps({k: manifest[k] for k in ("status", "files", "bytes", "source_tree_sha256", "source_sha256_file_sha256", "main_tree_git_head", "main_tree_git_dirty")}, indent=2))
    return 0


def verify() -> int:
    manifest = json.loads((RELEASE / "source_manifest.json").read_text(encoding="utf-8"))
    digest, files = tree_digest(SOURCE)
    ok = digest == manifest["source_tree_sha256"] and len(files) == manifest["files"]
    check = subprocess.run(["sha256sum", "--status", "-c", str(RELEASE / "source.sha256")], cwd=SOURCE)
    ok = ok and check.returncode == 0
    sha_file_ok = sha256(RELEASE / "source.sha256") == manifest["source_sha256_file_sha256"]
    drift = [
        r["path"] for r in manifest["file_records"]
        if not (ROOT / r["path"]).is_file() or sha256(ROOT / r["path"]) != r["sha256"]
    ]
    print(json.dumps({
        "status": "PASS" if ok and sha_file_ok else "FAIL",
        "source_tree_sha256": digest,
        "sha256sum_check_rc": check.returncode,
        "sha256_file_ok": sha_file_ok,
        "files": len(files),
        "main_tree_drift_since_freeze": drift,
    }, indent=2))
    return 0 if ok and sha_file_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="re-hash an existing freeze instead of creating one")
    parser.add_argument("--release", default=CURRENT_RELEASE, help=f"runtime/<release> (default {CURRENT_RELEASE})")
    args = parser.parse_args()
    select_release(args.release)
    return verify() if args.verify else freeze()


if __name__ == "__main__":
    raise SystemExit(main())
