#!/usr/bin/env python3
"""Join the 2026-10-16 retrain queue with the SCO submission ledgers.

Writes metadata/release_inventory/retrain_submissions_20261016.csv: one row per
queued run (176) with the job that currently re-trains it (if any), the
attempt history, the new output location and the stage it waits on.  Sources:

  metadata/release_inventory/retrain_queue_20261016.csv
  artifacts/main_v2_probe_v2_816_20260917/single_gpu_b2_fp32_20261016_r1/submissions.json
  logs/tstep_probe_campaigns/tstep_probe_b2_fp32_20261016_r1/launcher/submissions.json
  logs/tstep_minimax_bf16_v1/submissions.json   (extraction; timestep L20/L25 rows wait on it)
  logs/tstep_probe_campaigns/tstep_probe_minimax_bf16_v1_r1/launcher/submissions.json   (stage 5, once submitted)

Job *states* are not stored (they change); query them with
``sco acp jobs describe <job_id> --workspace-name=aigc``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "metadata/release_inventory/retrain_queue_20261016.csv"
OUT = ROOT / "metadata/release_inventory/retrain_submissions_20261016.csv"
CANONICAL = ROOT / "artifacts/main_v2_probe_v2_816_20260917/single_gpu_b2_fp32_20261016_r1/submissions.json"
TSTEP = ROOT / "logs/tstep_probe_campaigns/tstep_probe_b2_fp32_20261016_r1/launcher/submissions.json"
EXTRACT = ROOT / "logs/tstep_minimax_bf16_v1/submissions.json"
STAGE5_RUN_ID = "tstep_probe_minimax_bf16_v1_r1"
STAGE5 = ROOT / "logs/tstep_probe_campaigns" / STAGE5_RUN_ID / "launcher/submissions.json"
CANONICAL_OUT = "logs/inscene15k_streaming/runs/main_v2_single_gpu_b2_fp32_20261016_r1/production/main_v2_single_gpu_b2_fp32_20261016_r1"
TSTEP_OUT = "logs/inscene15k_streaming/runs/tstep_probe_seeded_v1/tstep_probe_b2_fp32_20261016_r1"
STAGE5_OUT = f"logs/inscene15k_streaming/runs/tstep_probe_seeded_v1/{STAGE5_RUN_ID}"
FIELDS = [
    "queue_group", "experiment_id", "inventory", "model", "probe", "layer", "prefix", "timestep",
    "action", "campaign", "job_id", "job_name", "submitted_utc", "previous_job_ids",
    "new_output_path", "waits_on", "old_output_path",
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"jobs": {}}


def main() -> int:
    canonical = load(CANONICAL)["jobs"]
    tstep = load(TSTEP)["jobs"]
    extract = load(EXTRACT)["jobs"]
    canary = extract.get("canary", {})
    consolidation = extract.get("consolidation", {})
    stage5 = load(STAGE5)["jobs"]
    rows = []
    with QUEUE.open(newline="", encoding="utf-8") as handle:
        for q in csv.DictReader(handle):
            row = {k: q.get(k, "") for k in FIELDS if k in q}
            row.update({"action": "", "campaign": "", "job_id": "", "job_name": "", "submitted_utc": "",
                        "previous_job_ids": "", "new_output_path": "", "waits_on": ""})
            if q["queue_group"] == "B2_NONFINITE_WEIGHTS_FP16_PASSTHROUGH" and q["inventory"] == "canonical_v2":
                rec = canonical.get(q["experiment_id"])
                row.update({"action": "retrain_fp32_transfer", "campaign": "main_v2_single_gpu_b2_fp32_20261016_r1"})
                if rec:
                    row.update({"job_id": rec.get("job_id", ""), "job_name": rec.get("job_name", ""),
                                "submitted_utc": rec.get("submitted_utc", ""),
                                "new_output_path": f"{CANONICAL_OUT}/{q['experiment_id']}"})
            elif q["queue_group"] == "B2_NONFINITE_WEIGHTS_FP16_PASSTHROUGH":
                gid = f"{q['model']}_t{q['timestep']}_l{q['layer']}_p{q['prefix']}"
                rec = tstep.get(gid)
                row.update({"action": "retrain_fp32_transfer", "campaign": "tstep_probe_b2_fp32_20261016_r1"})
                if rec:
                    prev = [a.get("job_id", "") for a in rec.get("previous_attempts", [])]
                    row.update({"job_id": rec.get("job_id", ""), "job_name": rec.get("job_name", ""),
                                "submitted_utc": rec.get("submitted_utc", ""),
                                "previous_job_ids": " ".join(p for p in prev if p),
                                "new_output_path": f"{TSTEP_OUT}/{rec.get('run_tag', '')}"})
            else:
                t = q["timestep"]
                tag = f"tp_minimax_h3_t{t}_l{q['layer']}_p{q['prefix']}_{q['probe']}"
                gid = f"minimax_h3_t{t}_l{q['layer']}_p{q['prefix']}"
                shards = {mode: " ".join(extract.get(f"production_t{t}_{mode}_s{i}of8", {}).get("job_id") or "?"
                                         for i in range(8)) for mode in ("stream", "target")}
                row.update({
                    "action": "retrain_after_bf16_reextraction", "campaign": STAGE5_RUN_ID,
                    "new_output_path": f"{STAGE5_OUT}/{tag}",
                    "waits_on": (
                        f"canary {canary.get('job_id') or 'not submitted'}; "
                        f"t{t} extraction shards stream [{shards['stream']}] target [{shards['target']}]; "
                        f"consolidation + full verification {consolidation.get('job_id') or 'not submitted'}; "
                        "strict audit; stage 5 needs explicit user approval"
                    ),
                })
                rec = stage5.get(gid) or stage5.get(f"{gid}__{q['probe']}")
                if rec:
                    row.update({"job_id": rec.get("job_id", ""), "job_name": rec.get("job_name", ""),
                                "submitted_utc": rec.get("submitted_utc", ""), "waits_on": ""})
            rows.append(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    submitted = sum(1 for r in rows if r["job_id"])
    print(f"rows={len(rows)} with_job={submitted} waiting={len(rows) - submitted} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
