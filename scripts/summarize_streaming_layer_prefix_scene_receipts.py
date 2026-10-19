#!/usr/bin/env python3
"""Summarize exact streaming receipt coverage by model/layer/prefix/scene."""
import argparse
import csv
import gzip
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/main_v2_probe_v2_816_20260917"
EVIDENCE = BASE / "probe_readiness_552_20260923"
OVERLAY = BASE / "owner_overlay_r7_receipt_union/overlay_manifest.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["dino", "videomae", "vjepa2"])
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    models = set(args.models)

    overlay = json.loads(OVERLAY.read_text())
    binding = overlay["streaming_database"]
    if sha(binding["path"]) != binding["sha256"]:
        raise RuntimeError("streaming owner database identity changed")
    database = sqlite3.connect(Path(binding["path"]).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    database.row_factory = sqlite3.Row

    specs = {}
    for row in csv.DictReader((EVIDENCE / "experiments_552.csv").open()):
        if row["model"] in models:
            specs[(row["model"], int(row["disk_layer"]), int(row["prefix"]))] = row

    scene_rows = []
    summary_rows = []
    for spec, evidence in sorted(specs.items()):
        model, layer, prefix = spec
        path = EVIDENCE / "unverified_keys" / f"{model}__l{layer}__p{prefix}__streaming.jsonl.gz"
        missing = [json.loads(line) for line in gzip.open(path, "rt")]
        missing_by_scene = Counter(row["scene_id"] for row in missing)
        owners = list(database.execute(
            "SELECT scene_id,count(*) AS n FROM streaming_owners "
            "WHERE vfm=? AND layer=? AND timestep=749 AND prefix=? GROUP BY scene_id", spec))
        required = int(evidence["streaming_required"])
        if sum(row["n"] for row in owners) != required or len(owners) != 681:
            raise RuntimeError(f"{spec}: owner universe incomplete")
        current = []
        for owner in owners:
            expected = owner["n"]
            unverified = missing_by_scene[owner["scene_id"]]
            current.append(dict(
                model=model, layer=layer, prefix=prefix, scene_id=owner["scene_id"],
                required_windows=expected, receipt_verified_windows=expected - unverified,
                receipt_unverified_windows=unverified, receipt_scene_ready=int(unverified == 0),
                owner_routed_windows=expected, owner_scene_complete=1,
            ))
        scene_rows.extend(current)
        summary_rows.append(dict(
            model=model, layer=layer, prefix=prefix, required_windows=required,
            receipt_verified_windows=int(evidence["streaming_verified"]),
            receipt_unverified_windows=int(evidence["streaming_unverified"]),
            scenes_total=len(current), receipt_ready_scenes=sum(row["receipt_scene_ready"] for row in current),
            receipt_blocked_scenes=sum(not row["receipt_scene_ready"] for row in current),
            receipt_layer_prefix_ready=int(all(row["receipt_scene_ready"] for row in current)),
            owner_routed_windows=sum(row["owner_routed_windows"] for row in current),
            owner_complete_scenes=sum(row["owner_scene_complete"] for row in current),
            owner_layer_prefix_complete=int(all(row["owner_scene_complete"] for row in current)),
        ))
    database.close()

    for name, rows in (("by_scene.csv", scene_rows), ("by_layer_prefix.csv", summary_rows)):
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    aggregate = defaultdict(lambda: dict(
        verified=0, required=0, gaps=0, ready_specs=0, ready_scenes=0, blocked_scenes=0))
    for row in summary_rows:
        current = aggregate[row["model"]]
        current["verified"] += row["receipt_verified_windows"]
        current["required"] += row["required_windows"]
        current["gaps"] += row["receipt_unverified_windows"]
        current["ready_specs"] += row["receipt_layer_prefix_ready"]
        current["ready_scenes"] += row["receipt_ready_scenes"]
        current["blocked_scenes"] += row["receipt_blocked_scenes"]
    report = dict(
        schema="STREAMING_LAYER_PREFIX_SCENE_RECEIPT_AUDIT_V1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope_models=sorted(models), specs=len(summary_rows), scene_rows=len(scene_rows),
        receipt_ready_specs=sum(row["receipt_layer_prefix_ready"] for row in summary_rows),
        owner_complete_specs=sum(row["owner_layer_prefix_complete"] for row in summary_rows),
        models=dict(aggregate),
        source_evidence={"path": str(EVIDENCE / "feature_evidence_snapshot.json"),
                         "sha256": sha(EVIDENCE / "feature_evidence_snapshot.json")},
        owner_overlay={"path": str(OVERLAY), "sha256": sha(OVERLAY)},
        by_layer_prefix=str(output / "by_layer_prefix.csv"), by_scene=str(output / "by_scene.csv"),
        acceptance=("receipt_layer_prefix_ready requires zero unverified windows in every scene; "
                    "owner completeness is routing evidence and still requires remote/load canary."),
    )
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
