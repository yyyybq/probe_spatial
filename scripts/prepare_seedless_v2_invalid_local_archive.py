#!/usr/bin/env python3
"""Inventory the frozen seedless-v2 local tree without mutating feature data.

This helper never changes permissions, deletes, or uploads feature data.  It
runs as the UID that owns the old SCO outputs and writes an immutable inventory
before a separate transactional AOSS archive step.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path


ROOT = Path(
    "/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2"
)
PROJECT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
INVALID_MARKER = PROJECT / "logs/tstep_canonical_v2/aoss_done/INVALID_SEEDLESS_V2"
INVENTORY = ROOT / "INVALID_LOCAL_ARCHIVE_INVENTORY.tsv"
SUMMARY = ROOT / "INVALID_LOCAL_ARCHIVE_SUMMARY.json"

FEATURE_PATTERN = (
    r"^wan/scannetpp/[^/]+/window_[0-9]{4}/prefix_[0-9]{6}/"
    r"feature_t(?:499|999)_layer(?:15|20|25)\.sft"
)
FEATURE_RE = re.compile(FEATURE_PATTERN + r"$")
SIDECAR_RE = re.compile(FEATURE_PATTERN + r"\.manifest\.json$")
TEMP_RE = re.compile(
    r"^wan/scannetpp/[^/]+/window_[0-9]{4}/prefix_[0-9]{6}/"
    r"\.feature_t(?:499|999)_layer(?:15|20|25)\.sft\.tmp-[^/]+$"
)


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def main() -> None:
    if ROOT.is_symlink() or not ROOT.is_dir() or ROOT.name != "FEAT_STREAMING_CANONICAL_V2":
        fail(f"unexpected old namespace root: {ROOT}")
    if INVALID_MARKER.is_symlink() or not INVALID_MARKER.is_file():
        fail(f"missing frozen-invalid marker: {INVALID_MARKER}")
    if (ROOT.parent / "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1").resolve() == ROOT.resolve():
        fail("seeded-v1 root unexpectedly aliases the invalid seedless root")

    rows: list[tuple[str, int, int, int, str]] = []
    counts = {"feature": 0, "sidecar": 0, "temp": 0, "structural_or_other": 0}
    bytes_by_kind = {key: 0 for key in counts}
    for path in sorted(ROOT.rglob("*")):
        if path.is_symlink():
            fail(f"symlink found in frozen invalid tree: {path}")
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if FEATURE_RE.fullmatch(rel):
            kind = "feature"
        elif SIDECAR_RE.fullmatch(rel):
            kind = "sidecar"
        elif TEMP_RE.fullmatch(rel):
            kind = "temp"
        else:
            kind = "structural_or_other"
        st = path.stat(follow_symlinks=False)
        rows.append((rel, st.st_size, st.st_dev, st.st_ino, kind))
        counts[kind] += 1
        bytes_by_kind[kind] += st.st_size

    if counts["feature"] != 1458:
        fail(f"expected exactly 1458 frozen .sft files, found {counts['feature']}")
    if counts["sidecar"] != 1457:
        fail(f"expected exactly 1457 frozen sidecars, found {counts['sidecar']}")
    if counts["temp"] != 4:
        fail(f"expected exactly 4 frozen atomic temps, found {counts['temp']}")
    if bytes_by_kind["feature"] != 146_731_382_064:
        fail(
            "frozen feature byte count changed: "
            f"{bytes_by_kind['feature']} != 146731382064"
        )

    body = "relpath\tsize_bytes\tdevice\tinode\tkind\n" + "".join(
        f"{rel}\t{size}\t{device}\t{inode}\t{kind}\n"
        for rel, size, device, inode, kind in rows
    )
    inventory_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    tmp_inventory = INVENTORY.with_name(f".{INVENTORY.name}.tmp-{os.getpid()}")
    tmp_inventory.write_text(body, encoding="utf-8")
    os.replace(tmp_inventory, INVENTORY)

    summary = {
        "schema_version": 1,
        "status": "INVALID_SEEDLESS_V2_ARCHIVE_PREPARED",
        "root": str(ROOT),
        "invalid_marker": str(INVALID_MARKER),
        "counts": counts,
        "bytes": bytes_by_kind,
        "inventory": INVENTORY.name,
        "inventory_sha256": inventory_sha,
        "deletes_performed": 0,
        "uploads_performed": 0,
    }
    tmp_summary = SUMMARY.with_name(f".{SUMMARY.name}.tmp-{os.getpid()}")
    tmp_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp_summary, SUMMARY)

    os.chmod(INVENTORY, 0o644)
    os.chmod(SUMMARY, 0o644)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
