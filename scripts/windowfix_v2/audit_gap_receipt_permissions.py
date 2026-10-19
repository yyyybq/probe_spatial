#!/usr/bin/env python3
"""Audit canonical-v2 gap receipt readability and publication shape.

The production gap runner treats remote completion receipts as the resume
authority.  SCO workers can create receipts as another uid, so this audit keeps
the recovery decision explicit: a receipt that exists but cannot be read by the
submitter is not usable for local review or deterministic resume planning.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


DEFAULT_ROOTS = (
    Path("/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1"),
    Path("/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1"),
)


def mode_octal(path: Path) -> str:
    return f"{stat.S_IMODE(path.lstat().st_mode):04o}"


def owner(path: Path) -> dict[str, int]:
    info = path.lstat()
    return {"uid": int(info.st_uid), "gid": int(info.st_gid)}


def classify_file(path: Path, *, expected_regular_json: bool) -> dict[str, Any]:
    item: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_symlink": path.is_symlink(),
        "mode": None,
        "owner": None,
        "readable": False,
        "issues": [],
    }
    if not path.exists() and not path.is_symlink():
        item["issues"].append("missing")
        return item
    item["mode"] = mode_octal(path)
    item["owner"] = owner(path)
    if path.is_symlink():
        item["issues"].append("symlink")
    if not path.is_file():
        item["issues"].append("not_regular_file")
        return item
    item["readable"] = os.access(path, os.R_OK)
    if not item["readable"]:
        item["issues"].append("not_readable")
    if item["mode"] != "0644":
        item["issues"].append("mode_not_0644")
    if expected_regular_json and item["readable"]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                item["issues"].append("json_not_object")
        except Exception as exc:
            item["issues"].append(f"json_unreadable:{type(exc).__name__}")
    return item


def audit_root(root: Path) -> dict[str, Any]:
    log_root = root / ".windowfix_v2_gap_logs"
    remote_root = root / ".windowfix_v2_gap_remote_receipts"
    local_receipts = (
        sorted(log_root.rglob("receipt.json")) if log_root.exists() else []
    )
    remote_receipts = (
        sorted(remote_root.glob("*/remote_completion.json")) if remote_root.exists() else []
    )
    local_items = [
        classify_file(path, expected_regular_json=True) for path in local_receipts
    ]
    remote_items = [
        classify_file(path, expected_regular_json=True) for path in remote_receipts
    ]
    all_items = local_items + remote_items
    return {
        "root": str(root),
        "exists": root.exists(),
        "local_receipts": len(local_items),
        "remote_completion_receipts": len(remote_items),
        "issue_count": sum(1 for item in all_items if item["issues"]),
        "issues": [item for item in all_items if item["issues"]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        default=[],
        help="Feature root to audit. Defaults to canonical-v2 gap streaming and target roots.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    roots = args.root or list(DEFAULT_ROOTS)
    report = {
        "schema_version": 1,
        "contract": "CANONICAL_V2_GAP_RECEIPT_PERMISSION_AUDIT_V1",
        "uid": os.getuid(),
        "gid": os.getgid(),
        "roots": [audit_root(root) for root in roots],
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        os.chmod(args.output, 0o644)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
