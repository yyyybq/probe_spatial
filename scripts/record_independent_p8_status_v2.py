#!/usr/bin/env python3
"""Record the immutable progress/status snapshot for the independent p8 study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_create(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--science-tag", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--submit-script", type=Path, default=Path("/mnt/umm/users/yinbaiqiao/submit.sh"))
    args = parser.parse_args()
    submit = args.submit_script.resolve()
    if not args.science_tag.startswith("independent_actionability_algebra_p8_"):
        raise SystemExit("unexpected science tag")
    size = submit.stat().st_size if submit.is_file() else None
    digest = sha256(submit) if submit.is_file() else None
    if size not in (None, 0, 1):
        raise SystemExit("this snapshot is only for the observed truncated submit.sh blocker")
    artifact = args.artifact_dir.resolve()
    report = args.report_dir.resolve()
    summary = json.loads((artifact / "feature_lineage" / "exact_feature_lineage_audit_v1.json").read_text())
    blocker = artifact / "provenance" / "submit_sh_recovery_audit_v1.json"
    matrix = report / "STATUS_MATRIX_V2.csv"
    markdown = report / "SCO_ENVIRONMENT_BLOCKER.md"
    if any(path.exists() for path in (blocker, matrix, markdown)):
        raise SystemExit("status snapshot already exists; refusing overwrite")
    recovery = {
        "schema_version": 1,
        "science_tag": args.science_tag,
        "status": "BLOCKED_ENVIRONMENT",
        "submit_script": str(submit),
        "exists": submit.exists(),
        "size_bytes": size,
        "sha256": digest,
        "required_condition": "restore only a verified Cursor History copy; do not reconstruct or infer a replacement",
        "cursor_history_search": {
            "scope": "/mnt/umm/users/yinbaiqiao/.cursor",
            "result": "no complete probe-spatial submit.sh body matching the current five-argument worker-entrypoint use was recovered",
            "excluded_candidate": "a VAGEN-Lite history entry is a different repo/environment template and is not a safe restoration source",
        },
        "impact": "SCO DINO extraction/load smoke and all GPU-dependent training/evaluation remain unsubmitted; raw data, manifests, and exact request units are available.",
    }
    atomic_create(blocker, json.dumps(recovery, indent=2, sort_keys=True) + "\n")
    rows = [
        ("scope_quarantine", "COMPLETE_VALID", "old C1/C2 checkpoints and canonical outputs excluded"),
        ("raw_pose_actionability_algebra", "COMPLETE_VALID", "684 raw scenes; scene-disjoint p8 manifests and GT pose labels validated"),
        ("feature_request_v1", "FAILED_VALIDATION", "frame allow-list did not preserve ordered p8 context units; preserved unchanged"),
        ("feature_request_v2", "COMPLETE_VALID", "49,432 exact temporal/isolated units frozen"),
        ("exact_feature_lineage_audit", "COMPLETE_WITH_CAVEAT", f"0/49,432 reusable after {summary['sidecars_scanned']} sidecars"),
        ("dino_raw_id_extractor", "COMPLETE_VALID", "independent L0 extractor and local dry-run contract prepared"),
        ("dino_gpu_extraction_load_smoke", "BLOCKED_ENVIRONMENT", "submit.sh is absent/truncated; verified recovery required"),
        ("dino_direct_actionability", "NOT_RUN", "requires exact DINO coverage gate"),
        ("dino_clean_dynamics_algebra", "NOT_RUN", "requires exact DINO coverage gate"),
        ("f3r_vjepa2_wan_extraction", "NOT_RUN", "gated on validated DINO smoke"),
        ("four_model_common_set", "NOT_RUN", "requires native exact coverage for all four models"),
        ("sco_submission_adapter", "BLOCKED_ENVIRONMENT", "no verified submit.sh recovery"),
    ]
    output = ["science_tag,component,status,detail\n"]
    output.extend(",".join('"' + str(value).replace('"', '""') + '"' for value in (args.science_tag, *row)) + "\n" for row in rows)
    atomic_create(matrix, "".join(output))
    atomic_create(
        markdown,
        f"""# SCO environment blocker\n\nStatus: `BLOCKED_ENVIRONMENT`\n\n`{submit}` is {size} byte(s), SHA256 `{digest}`.  The approved submission path cannot be used honestly until an exact, verified Cursor History copy is restored.  No SCO job was created, and no GPU extraction, training, or evaluation was silently substituted with a login-node run.\n\nThe independent p8 manifests and exact DINO extractor are ready. The immediate next operation after verified recovery is the four-unit DINO raw-ID extraction/load smoke, followed by the coverage verifier.\n""",
    )
    print(json.dumps({"status": "BLOCKED_ENVIRONMENT", "matrix": str(matrix), "blocker": str(blocker)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
