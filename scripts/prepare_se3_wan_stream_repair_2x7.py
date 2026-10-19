#!/usr/bin/env python3
"""Freeze the two-context, seven-layer Wan canonical-v2 repair allow-list."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/experimental/se3_wan_vjepa2_exact_v2_layers_20260916_r1"
PARENT_DIR = ROOT / (
    "artifacts/windowfix_v2/"
    "streaming_gap_worklists_ebc496eb0587_catalog_r12_qwen_pass_20260831"
)
PARENT_AUDIT = PARENT_DIR / "streaming_gap_audit.json"
PARENT_WORKLIST = PARENT_DIR / "gap_worklists/wan-t749-0d72a7aee30c.json"
OUTPUT = ROOT / "artifacts/tstep_probe_opt_v1/se3_wan_stream_repair_2x7_20260917"
WORKLIST = OUTPUT / "source/wan_stream_repair_2x7.worklist.json"
AUDIT = OUTPUT / "source/wan_stream_repair_2x7.audit.json"
SELECTION = {
    ("90d8e25cee", 3, 12): [472, 477, 482, 488, 493, 498, 506, 510, 513, 517, 522, 528],
    ("927aacd5d1", 0, 12): [0, 11, 15, 20, 25, 29, 32, 36, 39, 42, 45, 50],
}
LAYERS = [0, 5, 10, 15, 20, 25, 29]


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        Path(temporary).replace(path)
        path.chmod(0o644)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to replace immutable repair source: {OUTPUT}")
    parent = json.loads(PARENT_WORKLIST.read_text())
    selected = []
    for key in SELECTION:
        scene, window_id, prefix = key
        matches = [
            entry for entry in parent["entries"]
            if (entry["scene"], int(entry["window_id"]), int(entry["prefix"])) == key
        ]
        if len(matches) != 1:
            raise RuntimeError(f"parent worklist key count for {key}: {len(matches)}")
        entry = dict(matches[0])
        if [int(value) for value in entry["layers"]] != LAYERS:
            raise RuntimeError(f"parent layer mismatch for {key}: {entry['layers']}")
        selected.append(entry)
    selected.sort(key=lambda value: (value["scene"], value["window_id"], value["prefix"]))

    worklist = {
        key: value for key, value in parent.items()
        if key not in {"content_sha256", "counts", "entries"}
    }
    worklist.update({
        "counts": {"entries": 2, "layer_items": 14, "scenes": 2},
        "entries": selected,
    })
    worklist["content_sha256"] = canonical_sha(worklist)
    atomic_json(WORKLIST, worklist)

    common_manifest = RESULT / "data/common_manifest.jsonl"
    parent_audit = json.loads(PARENT_AUDIT.read_text())
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "audit_type": "SE3_WAN_STREAMING_SCOPED_REPAIR_AUDIT_V1",
        "canonical_protocol": parent_audit["canonical_protocol"],
        "canonical_source": parent_audit["canonical_source"],
        "protocol_file": parent_audit["protocol_file"],
        "protocol_file_sha256": parent_audit["protocol_file_sha256"],
        "counts": {
            "hard_errors": 0,
            "gap_worklists": 1,
            "gap_entries": 2,
            "gap_layer_items": 14,
            "scenes_sum_nonunique": 2,
        },
        "hard_errors": [],
        "selection": {
            "rule": "exact unresolved Wan cells in the frozen 436-row cross-model manifest",
            "common_manifest_path": str(common_manifest),
            "common_manifest_sha256": file_sha(common_manifest),
            "contexts": [
                {
                    "scene": scene,
                    "window_id": window_id,
                    "prefix": prefix,
                    "canonical_raw_ids": raw_ids,
                    "layers": LAYERS,
                }
                for (scene, window_id, prefix), raw_ids in SELECTION.items()
            ],
        },
        "parent": {
            "audit_path": str(PARENT_AUDIT),
            "audit_file_sha256": file_sha(PARENT_AUDIT),
            "audit_content_sha256": parent_audit["content_sha256"],
            "worklist_path": str(PARENT_WORKLIST),
            "worklist_file_sha256": file_sha(PARENT_WORKLIST),
            "worklist_content_sha256": parent["content_sha256"],
            "derivation": "strict positive subset; no missing-file inference",
        },
        "outputs": {
            "gap_worklists": [{
                "path": str(WORKLIST),
                "file_sha256": file_sha(WORKLIST),
                "content_sha256": worklist["content_sha256"],
                "counts": worklist["counts"],
            }]
        },
        "notes": [
            "This allow-list contains exactly two streaming contexts and seven frozen Wan layers.",
            "It does not authorize target extraction, other prefixes, other windows, or other models.",
        ],
    }
    audit["content_sha256"] = canonical_sha(audit)
    atomic_json(AUDIT, audit)
    print(json.dumps({
        "audit": str(AUDIT),
        "audit_sha256": file_sha(AUDIT),
        "worklist": str(WORKLIST),
        "worklist_sha256": file_sha(WORKLIST),
        "counts": worklist["counts"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
