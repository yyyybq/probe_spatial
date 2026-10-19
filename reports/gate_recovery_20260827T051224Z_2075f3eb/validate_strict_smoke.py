#!/usr/bin/env python3
"""Validate the recovery-only strict B2 manifests after all prerequisite gates."""

from __future__ import annotations

import json
import os
from pathlib import Path


REPORT_DIR = Path("/mnt/umm/users/yinbaiqiao/probe_spatial/reports/gate_recovery_20260827T051224Z_2075f3eb")
ENV_JSON = REPORT_DIR / "ENV_GATE.json"
MANIFEST_DIR = REPORT_DIR / "manifests/strict_b2_smoke"
PREFIXES = (8, 12, 16, 24)


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


gate = json.loads(ENV_JSON.read_text())
if not gate.get("pre_smoke_gate_pass"):
    raise SystemExit("refusing strict validation: prerequisite GPU/AOSS/DataLoader gate did not pass")

by_prefix = {}
for prefix in PREFIXES:
    path = MANIFEST_DIR / f"b2_strict_prefix{prefix}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"empty strict manifest: {path}")
    if any(int(row["strict_target_raw_pixels"]) != 0 for row in rows):
        raise RuntimeError(f"nonzero target pixels in {path}")
    by_prefix[prefix] = {row["strict_pair_id"]: row for row in rows}

common = set.intersection(*(set(rows) for rows in by_prefix.values()))
if not common:
    raise RuntimeError("no common strict pair across prefixes")

checked = 0
for pair_id in sorted(common):
    ref = by_prefix[PREFIXES[0]][pair_id]
    for prefix in PREFIXES[1:]:
        row = by_prefix[prefix][pair_id]
        if row["strict_centroid_world"] != ref["strict_centroid_world"]:
            raise RuntimeError(f"past-only centroid changed for {pair_id}")
        if row["strict_polar"] != ref["strict_polar"]:
            raise RuntimeError(f"strict polar label changed for {pair_id}")
        if row["streaming_tail_raw"] != ref["streaming_tail_raw"]:
            raise RuntimeError(f"fixed target changed for {pair_id}")
    checked += 1

gate["strict_b2_smoke"] = {
    "status": "PASS",
    "manifest_root": str(MANIFEST_DIR),
    "prefixes": list(PREFIXES),
    "counts": {str(prefix): len(by_prefix[prefix]) for prefix in PREFIXES},
    "common_pairs_checked": checked,
    "target_pixels_equal_zero": True,
    "past_only_centroid_invariant_across_prefixes": True,
    "fixed_target_invariant_across_prefixes": True,
}
gate["all_environment_and_smoke_gates_pass"] = True
atomic_json(ENV_JSON, gate)
print(json.dumps(gate["strict_b2_smoke"], sort_keys=True))
