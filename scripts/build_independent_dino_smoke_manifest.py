#!/usr/bin/env python3
"""Derive the four required raw-ID-exact DINO smoke units from request v2.

This is deliberately a selection-only operation: it copies complete, immutable
v2 unit records without changing a frozen protocol or feature request.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


TAG = "independent_actionability_algebra_p8_20260831T091224Z_28bda217"
ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "artifacts" / TAG / "manifests"
SOURCE = MANIFESTS / "feature_request_manifest_v2.jsonl"
OUTPUT = MANIFESTS / "dino_l0_exact_smoke_4units_v1.jsonl"
SUMMARY = MANIFESTS / "dino_l0_exact_smoke_4units_v1.summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_create(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    source_sha = sha256(SOURCE)
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    dino = [
        row
        for row in rows
        if row.get("model") == "dino" and row.get("layer") == 0 and row.get("timestep") is None
    ]
    if len(dino) != 12358:
        raise RuntimeError(f"unexpected DINO v2 unit count: {len(dino)}")

    # A single Actionability sample provides all requested roles.  Sorting by
    # source sample ID and unit ID makes the selection independent of JSONL order.
    by_sample: dict[str, list[dict]] = {}
    for row in dino:
        for sample_id in row["source_sample_ids"]:
            if sample_id.startswith("a0_"):
                by_sample.setdefault(sample_id, []).append(row)
    required = (
        "streaming_context_p8",
        "isolated_anchor",
        "isolated_goal",
        "isolated_candidate_endpoint_1",
    )
    selected: list[dict] | None = None
    selected_sample = ""
    for sample_id in sorted(by_sample):
        candidates = by_sample[sample_id]
        chosen: list[dict] = []
        for role in required:
            matched = [row for row in candidates if role in row["requested_roles"]]
            if len(matched) != 1:
                break
            chosen.append(matched[0])
        else:
            if len({row["unit_id"] for row in chosen}) == 4:
                selected, selected_sample = chosen, sample_id
                break
    if selected is None:
        raise RuntimeError("no one-sample DINO smoke quartet in feature-request v2")
    if any(row["exact_raw_id_required"] is not True or row["fallback_forbidden"] is not True for row in selected):
        raise RuntimeError("smoke selection weakened exact identity policy")
    context = selected[0]
    if context["feature_role"] != "streaming_context_p8" or len(context["context_raw_ids"] or []) != 8:
        raise RuntimeError("selected context is not ordered p8")

    payload = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in selected)
    atomic_create(OUTPUT, payload)
    output_sha = sha256(OUTPUT)
    summary = {
        "schema_version": 1,
        "science_tag": TAG,
        "status": "COMPLETE_VALID",
        "operation": "immutable_subset_selection",
        "source_feature_request_v2": str(SOURCE),
        "source_feature_request_v2_sha256": source_sha,
        "smoke_manifest": str(OUTPUT),
        "smoke_manifest_sha256": output_sha,
        "selected_actionability_sample_id": selected_sample,
        "unit_count": len(selected),
        "roles": {
            "ordered_p8_context": selected[0]["unit_id"],
            "isolated_anchor": selected[1]["unit_id"],
            "isolated_goal": selected[2]["unit_id"],
            "isolated_candidate_endpoint_1": selected[3]["unit_id"],
        },
        "unit_ids": [row["unit_id"] for row in selected],
        "identity_contract": "all units copied verbatim from v2; exact_raw_id_required=true; fallback_forbidden=true",
    }
    atomic_create(SUMMARY, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
