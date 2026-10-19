#!/usr/bin/env python3
"""Create a fail-closed seven-model owner overlay with MiniMax physically absent."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path


MODELS = (
    "cogvideox", "videomae", "vjepa2", "lingbot",
    "qwen2_5_vl", "dino", "f3r",
)
EXPECTED_STREAM_ROWS = 293_868
EXPECTED_TARGET_ROWS = 565_155


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-overlay", type=Path, required=True)
    parser.add_argument("--user-scope-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError(f"output exists: {output}")
    output.mkdir(parents=True)

    base_path = args.base_overlay.resolve()
    base = json.loads(base_path.read_text())
    scope_path = args.user_scope_manifest.resolve()
    scope = json.loads(scope_path.read_text())
    if base.get("status") != "READY" or base.get("fail_closed") is not True:
        raise RuntimeError("base owner overlay is not READY")
    if scope.get("status") != "FROZEN_NOT_SUBMITTED":
        raise RuntimeError("user scope is not frozen")
    if scope.get("counts", {}).get("execution_units") != 528:
        raise RuntimeError("user scope is not the exact 528-unit scope")

    stream_path = output / "streaming_owners.sqlite3"
    target_path = output / "target_owners.sqlite3"
    shutil.copy2(base["streaming_database"]["path"], stream_path)
    shutil.copy2(base["target_database"]["path"], target_path)

    stream = sqlite3.connect(stream_path)
    before_stream = stream.execute("SELECT COUNT(*) FROM streaming_owners").fetchone()[0]
    removed_stream = stream.execute(
        "DELETE FROM streaming_owners WHERE vfm='minimax_h3'"
    ).rowcount
    stream.executemany(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
        [
            ("status", "READY"),
            ("scope", "valid7_user_scope_20260922"),
            ("user_scope_manifest_sha256", sha256(scope_path)),
            ("minimax_policy", "PHYSICALLY_ABSENT_FROM_OWNER_DB"),
        ],
    )
    stream.commit()
    after_stream = stream.execute("SELECT COUNT(*) FROM streaming_owners").fetchone()[0]
    stream_duplicates = stream.execute(
        "SELECT COUNT(*) FROM (SELECT vfm,source,scene_id,layer,timestep,"
        "window_id,prefix,COUNT(*) n FROM streaming_owners GROUP BY 1,2,3,4,5,6,7 "
        "HAVING n != 1)"
    ).fetchone()[0]
    stream_models = dict(stream.execute(
        "SELECT vfm,COUNT(*) FROM streaming_owners GROUP BY vfm ORDER BY vfm"
    ).fetchall())
    stream.close()
    if before_stream != 327_748 or removed_stream != 33_880:
        raise RuntimeError(f"unexpected streaming removal: {before_stream}/{removed_stream}")
    if after_stream != EXPECTED_STREAM_ROWS or stream_duplicates:
        raise RuntimeError("seven-model streaming closure failed")
    if set(stream_models) != set(MODELS):
        raise RuntimeError(f"unexpected streaming models: {sorted(stream_models)}")

    target = sqlite3.connect(target_path)
    target.execute("PRAGMA foreign_keys=OFF")
    before_target = target.execute("SELECT COUNT(*) FROM target_owners").fetchone()[0]
    removed_target = target.execute(
        "DELETE FROM target_owners WHERE vfm='minimax_h3'"
    ).rowcount
    removed_expected = target.execute(
        "DELETE FROM target_expected WHERE vfm='minimax_h3'"
    ).rowcount
    target.executemany(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
        [
            ("status", "READY"),
            ("scope", "valid7_user_scope_20260922"),
            ("user_scope_manifest_sha256", sha256(scope_path)),
            ("minimax_policy", "PHYSICALLY_ABSENT_FROM_OWNER_DB"),
        ],
    )
    target.commit()
    after_target = target.execute("SELECT COUNT(*) FROM target_owners").fetchone()[0]
    after_expected = target.execute("SELECT COUNT(*) FROM target_expected").fetchone()[0]
    target_duplicates = target.execute(
        "SELECT COUNT(*) FROM (SELECT vfm,source,scene_id,layer,timestep,"
        "target_raw_id,COUNT(*) n FROM target_owners GROUP BY 1,2,3,4,5,6 "
        "HAVING n != 1)"
    ).fetchone()[0]
    target_holes = target.execute(
        "SELECT COUNT(*) FROM target_expected e LEFT JOIN target_owners o USING "
        "(vfm,source,scene_id,layer,timestep,canonical_row_index) "
        "WHERE o.source_uri IS NULL"
    ).fetchone()[0]
    target_models = dict(target.execute(
        "SELECT vfm,COUNT(*) FROM target_owners GROUP BY vfm ORDER BY vfm"
    ).fetchall())
    target.close()
    if before_target != 715_863 or removed_target != 150_708 or removed_expected != 150_708:
        raise RuntimeError(
            f"unexpected target removal: {before_target}/{removed_target}/{removed_expected}"
        )
    if (
        after_target != EXPECTED_TARGET_ROWS or after_expected != EXPECTED_TARGET_ROWS
        or target_duplicates or target_holes
    ):
        raise RuntimeError("seven-model target closure failed")
    if set(target_models) != set(MODELS):
        raise RuntimeError(f"unexpected target models: {sorted(target_models)}")

    audit = {
        "schema": "PROBE816_VALID7_USER_SCOPE_OWNER_AUDIT_V1",
        "status": "READY",
        "models": list(MODELS),
        "minimax_rows": {"streaming": 0, "target": 0, "target_expected": 0},
        "zero_holes": True,
        "zero_duplicate_effective_owners": True,
        "streaming": {
            "owner_rows": after_stream,
            "duplicate_keys": stream_duplicates,
            "owners_by_model": stream_models,
            "database": {"path": str(stream_path), "sha256": sha256(stream_path)},
        },
        "target": {
            "owner_rows": after_target,
            "expected_rows": after_expected,
            "holes": target_holes,
            "duplicate_keys": target_duplicates,
            "owners_by_model": target_models,
            "database": {"path": str(target_path), "sha256": sha256(target_path)},
        },
        "user_scope": {"path": str(scope_path), "sha256": sha256(scope_path)},
        "base_overlay": {"path": str(base_path), "sha256": sha256(base_path)},
        "owner_policy": (
            "exact composite key and raw frame IDs; receipt-backed owners; "
            "MiniMax physically absent; no runtime fallback or sample dropping"
        ),
    }
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
