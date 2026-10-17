#!/usr/bin/env python3
"""Apply the 2026-10-15 local evidence to the dated release inventories.

Inputs (all under ``artifacts/release_audit_20261015/`` unless overridden):

* ``campaign_local_verify_20261015.json`` -- per-run completion evidence for the
  LingBot C40 recovery and the MiniMax-H3 all-layer campaign (TRAIN_COMPLETE
  marker, normal-stop log, non-empty checkpoint, small-checkpoint tensor check).
* ``campaign_job_status_20261015.json`` -- scheduler states of the same jobs.
* ``ckpt_finite_sweep.jsonl`` -- optional non-finite parameter sweep over the
  checkpoints named by ``canonical_v2.csv`` and ``timestep.csv``.

Outputs: ``metadata/release_inventory/{canonical_v2,timestep}.csv`` rewritten
with a ``checkpoint_tensor_status`` column, the LingBot C rows updated, and the
MiniMax-H3 all-layer rows appended. The 2026-10-11 tables are copied unchanged
to ``metadata/release_inventory/archive_20261011/`` on first use. Run
``python scripts/summarize_release_inventory.py`` afterwards to rebuild
``all_experiments.csv`` and ``summary.json``.

This script reads local files only: no scheduler query, no training, no
evaluation. Scheduler success and local completion are recorded separately.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "metadata/release_inventory"
ARCHIVE = INVENTORY / "archive_20261011"
AUDIT = ROOT / "artifacts/release_audit_20261015"
AUDIT_DATE = "2026-10-15"
CANONICAL_SIX = {
    "streaming_depth", "streaming_mask", "view_consistency", "ego_belief_v2",
    "action_dynamics", "path_integration",
}
NON_CANONICAL_NOTE = {
    "streaming_camera": "outside the six-probe canonical V2 protocol: a separate camera head; the "
                        "protocol's A1 pose metric is the camera head trained inside streaming_depth",
    "counterfactual": "outside the six-probe canonical V2 protocol: legacy independently trained C3 head; "
                      "current C3 is an evaluation of the matched C1 checkpoint",
}
KEY_RE = re.compile(
    r"^(?P<model>[a-z0-9_]+)__scannetpp__t(?P<timestep>\d+)__l(?P<layer>\d+)__p(?P<prefix>\d+)__(?P<probe>[a-z0-9_]+)__seed(?P<seed>\d+)$"
)
REMAINING_WORK_COMPLETE = (
    "Export identity-matched evaluation metrics; audit finite values and degeneracy; "
    "freeze validation-split aggregate"
)
REMAINING_WORK_C = (
    REMAINING_WORK_COMPLETE
    + "; Confirm evaluator/runtime field compatibility and action/visual baselines; "
      "training cosine alone is insufficient"
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def archive_once() -> None:
    ARCHIVE.mkdir(exist_ok=True)
    for name in ("canonical_v2.csv", "timestep.csv", "analyses.csv", "all_experiments.csv", "summary.json", "README.md"):
        source = INVENTORY / name
        target = ARCHIVE / name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)


def load_sweep(path: Path) -> dict[str, dict]:
    """checkpoint path (relative or absolute) -> sweep record."""
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        ckpt = rec.get("checkpoint_path")
        if ckpt:
            out[ckpt] = rec
            try:
                out[os.path.relpath(ckpt, ROOT)] = rec
            except ValueError:
                pass
    return out


def tensor_status(rec: dict | None) -> str:
    if rec is None:
        return "NOT_CHECKED"
    kind = rec.get("kind")
    if kind == "MISSING":
        return "MISSING_FILE"
    if kind == "large_skipped":
        return "NOT_CHECKED_LARGE_CHECKPOINT"
    if "error" in rec:
        return "LOAD_ERROR"
    frac = rec.get("nonfinite_fraction")
    if frac is None:
        return "NOT_CHECKED"
    if frac > 0:
        return f"NONFINITE_FRACTION_{frac:.4f}"
    return "FINITE"


def manifest_identity(output_dir: Path, expected: dict) -> str:
    manifest = output_dir / "run_manifest.json"
    if not manifest.is_file() or not os.access(manifest, os.R_OK):
        return "MANIFEST_UNREADABLE"
    try:
        cfg = load_json(manifest).get("config", {})
    except Exception:
        return "MANIFEST_UNREADABLE"
    got = {
        "model": str(cfg.get("vfm_name")),
        "layer": str(cfg.get("feature_layer")),
        "prefix": str(cfg.get("prefix_len")),
        "probe": str((cfg.get("model") or {}).get("probe_type")),
        "seed": str(cfg.get("seed")),
        "timestep": str(cfg.get("feature_timestep")),
    }
    # VideoProbeLitModule configs name the family through the experiment job
    # name, not probe_type (pixalign covers depth/mask/camera).
    if got["probe"] == "pixalign":
        job = str(cfg.get("job_name", ""))
        got["probe"] = next((p for p in ("streaming_depth", "streaming_mask", "streaming_camera", "streaming_movement") if p in job), got["probe"])
    mismatches = [
        f"{key}:{got[key]}!={expected[key]}"
        for key in ("model", "prefix", "probe", "seed", "timestep")
        if str(expected[key]) != got[key]
    ]
    # ``feature_layer`` in the campaign configs is either the logical layer
    # (LingBot 48, owner catalogs map it to the disk file) or the disk layer
    # (MiniMax logical 50 trained as 49); the disk layer must appear in
    # ``feat_postfix`` (e.g. ``_t250_layer47``).
    if got["layer"] not in {str(expected["layer"]), str(expected["disk_layer"])}:
        mismatches.append(f"layer:{got['layer']}!={expected['layer']}/{expected['disk_layer']}")
    postfix = str(cfg.get("feat_postfix") or "")
    if postfix and not postfix.endswith(f"layer{expected['disk_layer']}"):
        mismatches.append(f"feat_postfix:{postfix}!=*layer{expected['disk_layer']}")
    return "MATCH" if not mismatches else "MISMATCH(" + ";".join(mismatches) + ")"


def checkpoint_paths(output_dir: Path) -> list[str]:
    paths = glob.glob(str(output_dir / "**" / "checkpoints" / "*.ckpt"), recursive=True)
    return sorted(p for p in paths if os.path.getsize(p) > 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    verify = load_json(args.audit_dir / "campaign_local_verify_20261015.json")
    jobs = load_json(args.audit_dir / "campaign_job_status_20261015.json")
    sweep = load_sweep(args.audit_dir / "ckpt_finite_sweep.jsonl")
    job_by_id = jobs["jobs"]
    job_observed = jobs["observed_utc"]
    verify_observed = verify["observed_utc"]

    archive_once()
    # Always start from the archived 2026-10-11 tables so re-running this
    # script after new evidence is idempotent (no duplicated notes).
    source_dir = ARCHIVE if (ARCHIVE / "canonical_v2.csv").exists() else INVENTORY

    # ------------------------------------------------------------ canonical V2
    fields, rows = read_csv(source_dir / "canonical_v2.csv")
    if "checkpoint_tensor_status" not in fields:
        fields.append("checkpoint_tensor_status")
    by_id = {row["experiment_id"]: row for row in rows}

    def apply_verify(row: dict[str, str], rec: dict, scope_note: str) -> None:
        key = row["experiment_id"]
        parsed = KEY_RE.match(key).groupdict()
        job = job_by_id.get(rec.get("job_id") or "", {})
        complete = rec["status"].startswith("COMPLETE")
        out_dir = ROOT / rec["output_dir"] if rec.get("output_dir") else None
        ckpts = checkpoint_paths(out_dir) if out_dir else []
        expected = {
            "model": parsed["model"], "layer": parsed["layer"],
            "disk_layer": row.get("disk_layer") or parsed["layer"],
            "prefix": parsed["prefix"], "probe": parsed["probe"], "seed": parsed["seed"],
            "timestep": parsed["timestep"],
        }
        identity = manifest_identity(out_dir, expected) if out_dir else "NO_OUTPUT_DIR"
        frac = rec.get("ckpt_nonfinite_fraction")
        if isinstance(frac, float):
            tensor = "FINITE" if frac == 0 else f"NONFINITE_FRACTION_{frac:.4f}"
        elif ckpts:
            tensor = tensor_status(sweep.get(ckpts[-1])) if sweep.get(ckpts[-1]) else "NOT_CHECKED_LARGE_CHECKPOINT"
        else:
            tensor = "NO_CHECKPOINT"
        row.update({
            "training_status": "COMPLETE" if complete else "INCOMPLETE",
            "evidence_status": (
                "NORMAL_STOP_AND_CHECKPOINT" if complete and rec.get("log_normal_stop") else
                "TRAIN_COMPLETE_MARKER_AND_CHECKPOINT" if complete else
                "RUNNING_PARTIAL_CHECKPOINT" if ckpts else "NOT_COMPLETE"
            ),
            "scheduler_status": job.get("status", "NOT_LISTED"),
            "scheduler_observed_at": job_observed,
            "training_job_id": rec.get("job_id") or "",
            "submission_state": "SUBMITTED" if rec.get("job_id") else row.get("submission_state", ""),
            "completion_receipt_status": (
                "TRAIN_COMPLETE_JSON_READ" if isinstance(rec.get("train_complete_payload"), dict) else
                "PRESENT_UNREADABLE" if rec.get("train_complete") and not rec.get("train_complete_readable") else
                "PRESENT" if rec.get("train_complete") else "NOT_PRESENT"
            ),
            "checkpoint_status": (
                "PRESENT_NONEMPTY_TENSOR_FINITE" if tensor == "FINITE" else
                "PRESENT_NONEMPTY_TENSOR_NONFINITE" if tensor.startswith("NONFINITE") else
                "PRESENT_NONEMPTY_NOT_DESERIALIZED" if ckpts else "NOT_LOCATED"
            ),
            "checkpoint_count": str(len(ckpts)),
            "checkpoint_path": os.path.relpath(ckpts[-1], ROOT) if ckpts else "",
            "checkpoint_tensor_status": tensor,
            "run_identity_status": identity,
            "output_path": rec.get("output_dir", ""),
            "observed_at": verify_observed,
            "audit_date": AUDIT_DATE,
            "task_identity_status": "MATCH" if identity == "MATCH" else "UNKNOWN",
            "timestep_identity_status": "MATCH" if identity == "MATCH" else "UNKNOWN",
        })
        canary_pass = row.get("validation_status") == "PASS"
        if complete and identity == "MATCH" and tensor == "FINITE" and canary_pass:
            row["correctness_status"] = "VERIFIED"
            row["correctness_scope"] = "TRAINING_IDENTITY_AND_PROTOCOL_GATE"
        elif complete and identity == "MATCH" and tensor == "FINITE":
            row["correctness_status"] = "NOT_VERIFIED"
            row["correctness_scope"] = "TRAINING_IDENTITY_ONLY_NO_PER_RUN_PROTOCOL_CANARY_LOCATED"
        elif tensor.startswith("NONFINITE"):
            row["correctness_status"] = "DEGENERATE"
        else:
            row["correctness_status"] = "NOT_VERIFIED"
        probe = parsed["probe"]
        if probe in NON_CANONICAL_NOTE:
            row["result_status"] = "NON_CANONICAL_PROBE_RAW_TRAINING_ARTIFACTS"
        elif tensor.startswith("NONFINITE"):
            row["result_status"] = "INVALID_NONFINITE_WEIGHTS"
        elif complete:
            row["result_status"] = "RAW_TRAINING_ARTIFACTS_NOT_METRIC_AUDITED"
        else:
            row["result_status"] = "TRAINING_IN_PROGRESS"
        row["ready_for_release"] = "NO"
        row["remaining_work"] = (
            "Finish or confirm the running job; then " + REMAINING_WORK_COMPLETE if not complete else
            "Diagnose non-finite weights (numerics/feature cache) and retrain; do not export metrics" if tensor.startswith("NONFINITE") else
            REMAINING_WORK_C if probe in {"action_dynamics", "path_integration"} else REMAINING_WORK_COMPLETE
        )
        if probe in NON_CANONICAL_NOTE:
            row["remaining_work"] = "Not part of the canonical six-probe protocol; keep as a separately named extra, not a main-table cell"
        note = f"{AUDIT_DATE}: {scope_note}"
        if probe in NON_CANONICAL_NOTE:
            note += "; " + NON_CANONICAL_NOTE[probe]
        row["notes"] = (row.get("notes", "").rstrip(". ") + ". " if row.get("notes") else "") + note
        row["evidence_path"] = ";".join(filter(None, [
            row.get("evidence_path", ""),
            "artifacts/release_audit_20261015/campaign_local_verify_20261015.json",
            "artifacts/release_audit_20261015/campaign_job_status_20261015.json",
        ]))

    lingbot_updated = 0
    for rec in verify["campaigns"]["lingbot_c40"]:
        row = by_id.get(rec["unique_key"])
        if row is None:
            raise SystemExit(f"LingBot key missing from canonical_v2.csv: {rec['unique_key']}")
        apply_verify(row, rec, "LingBot C recovery job (lingbot40_c_zoetrope_recovery_20261011_r1) finished; "
                               "TRAIN_COMPLETE.json, normal-stop log and finite small checkpoint verified locally")
        lingbot_updated += 1

    minimax_added = 0
    minimax_ledger = load_json(ROOT / "artifacts/main_v2_probe_v2_816_20260917/minimax_all_layers_20261011_r1/submissions.json")["jobs"]
    for rec in verify["campaigns"]["minimax_all"]:
        key = rec["unique_key"]
        if key in by_id:
            continue
        parsed = KEY_RE.match(key).groupdict()
        ledger = minimax_ledger.get(key, {})
        row = {field: "" for field in fields}
        row.update({
            "experiment_id": key, "entry_level": "run",
            "scope": "canonical_v2_minimax_all_layers_20261011" + ("_noncanonical_probe" if parsed["probe"] in NON_CANONICAL_NOTE else ""),
            "model": "minimax_h3", "probe": parsed["probe"], "layer": parsed["layer"], "prefix": parsed["prefix"],
            "seed": parsed["seed"], "timestep": parsed["timestep"],
            "disk_layer": str(ledger.get("disk_layer", parsed["layer"])),
            "submitted_at": ledger.get("submitted_utc", ""),
            "validation_status": "NOT_LOCATED", "validation_observed_at": "", "validation_hash_status": "NOT_LOCATED",
            "validation_result_path": "", "metric_status": "NOT_AUDITED_NO_MATCHED_OFFICIAL_EXPORT_LOCATED",
            "feature_protocol_status": "REMOTE_FEATURE_EVIDENCE_IN_PLAN_NOT_REVERIFIED",
            "unit": "experiments", "count": "1", "split": "train_with_val",
            "evaluation_acceptance_status": "NOT_ACCEPTED_METRICS_NOT_AUDITED",
            "evidence_path": "artifacts/main_v2_probe_v2_816_20260917/minimax_all_layers_20261011_r1/submissions.json",
        })
        apply_verify(row, rec, "MiniMax-H3 all-layer campaign (minimax_all_layers_20261011_r1; 95 queued jobs migrated to "
                               "H800 on 2026-10-12, same commands); logical L50 uses disk layer 49")
        rows.append(row)
        by_id[key] = row
        minimax_added += 1

    # Canonical MiniMax-H3 runs (L30 A/B16 and the all-layer campaign) read the
    # bf16 namespaces FEAT_STREAMING/FEAT_TARGET_CANONICAL_V2_MINIMAX_BF16_V2
    # (run_manifest feature_roots, START_RECEIPT streaming_root), not the fp16
    # FEAT_STREAMING/minimax_h3 files that overflow. Record the sampled
    # finiteness of those bf16 caches (job pt-ljbe3zu0) instead of a block.
    bf16_audit = args.audit_dir / "feature_finiteness/bf16_v2_feature_finiteness_20261016.jsonl"
    bf16_status = "BF16_V2_CACHE_SOURCE_NOT_SAMPLED"
    if bf16_audit.exists():
        ok = [json.loads(l) for l in bf16_audit.read_text().splitlines() if l.strip()]
        ok = [r for r in ok if r.get("status") == "OK" and "MINIMAX_BF16_V2" in r.get("path", "")]
        if ok and all(r.get("n_nonfinite", 1) == 0 for r in ok) and all(r.get("dtype") == "torch.bfloat16" for r in ok):
            bf16_status = f"BF16_V2_CACHE_SAMPLE_FINITE_{len(ok)}_FILES"
        elif ok:
            bf16_status = "BF16_V2_CACHE_SAMPLE_HAS_NONFINITE_OR_WRONG_DTYPE"
    for row in rows:
        if row.get("model") != "minimax_h3" or row.get("entry_level") != "run":
            continue
        row["feature_protocol_status"] = ";".join(filter(None, [row.get("feature_protocol_status", ""), bf16_status]))
        row["notes"] = (row.get("notes", "").rstrip(". ") + ". " if row.get("notes") else "") + (
            f"{AUDIT_DATE}: inputs were the bf16 MINIMAX_BF16_V2 caches, not the fp16 FEAT_STREAMING files; "
            f"{bf16_status} (artifacts/release_audit_20261015/feature_finiteness/)"
        )

    # Checkpoint tensor status for every other canonical row from the sweep.
    for row in rows:
        if row.get("checkpoint_tensor_status"):
            continue
        rec = sweep.get(row.get("checkpoint_path", "")) if row.get("checkpoint_path") else None
        status = tensor_status(rec) if row.get("checkpoint_path") else "NO_CHECKPOINT_PATH"
        row["checkpoint_tensor_status"] = status
        if status.startswith("NONFINITE"):
            row["correctness_status"] = "DEGENERATE"
            row["result_status"] = "INVALID_NONFINITE_WEIGHTS"
            row["remaining_work"] = "Diagnose non-finite weights (numerics/feature cache) and retrain; do not export metrics"
            row["notes"] = (row.get("notes", "").rstrip(". ") + ". " if row.get("notes") else "") + f"{AUDIT_DATE}: checkpoint tensors non-finite"
    canonical_counts = {
        "rows": len(rows),
        "lingbot_updated": lingbot_updated,
        "minimax_added": minimax_added,
        "tensor_status": {},
    }
    for row in rows:
        if row["entry_level"] != "run":
            continue
        k = row["checkpoint_tensor_status"].split("_FRACTION_")[0]
        canonical_counts["tensor_status"][k] = canonical_counts["tensor_status"].get(k, 0) + 1

    # ---------------------------------------------------------------- timestep
    tfields, trows = read_csv(source_dir / "timestep.csv")
    if "checkpoint_tensor_status" not in tfields:
        tfields.append("checkpoint_tensor_status")
    timestep_counts = {"rows": len(trows), "tensor_status": {}, "nonfinite_runs": 0}
    for row in trows:
        if row["entry_level"] != "run":
            row["checkpoint_tensor_status"] = "NOT_APPLICABLE"
            continue
        rec = sweep.get(row.get("checkpoint_path", "")) if row.get("checkpoint_path") else None
        status = tensor_status(rec) if row.get("checkpoint_path") else "NO_CHECKPOINT_PATH"
        row["checkpoint_tensor_status"] = status
        k = status.split("_FRACTION_")[0]
        timestep_counts["tensor_status"][k] = timestep_counts["tensor_status"].get(k, 0) + 1
        if status.startswith("NONFINITE"):
            timestep_counts["nonfinite_runs"] += 1
            row["correctness_status"] = "DEGENERATE"
            row["result_status"] = "INVALID_NONFINITE_WEIGHTS"
            row["checkpoint_status"] = "EXISTS_TERMINAL_FILENAME_TENSOR_NONFINITE"
            row["remaining_work"] = (
                "RETRAIN REQUIRED after diagnosis: terminal checkpoint tensors are non-finite (training diverged); "
                "any evaluation of this checkpoint is invalid, including the degenerate official rows; "
                "determine whether the feature cache or the numerics (precision/LR) caused the divergence first"
            )
            row["notes"] = (row.get("notes", "").rstrip(". ") + ". " if row.get("notes") else "") + f"{AUDIT_DATE}: checkpoint tensors non-finite ({status})"
        elif status == "FINITE" and row.get("checkpoint_status") == "EXISTS_TERMINAL_FILENAME_NOT_TENSOR_VALIDATED":
            row["checkpoint_status"] = "EXISTS_TERMINAL_FILENAME_TENSOR_FINITE"
        if row.get("model") == "minimax_h3" and str(row.get("layer")) in {"20", "25"}:
            # AOSS job pt-et0ezc14: every sampled t500/t1000 layer-20/25 cache holds inf values.
            row["result_status"] = "INVALID_FEATURE_CACHE_OVERFLOW" if not status.startswith("NONFINITE") else row["result_status"]
            row["remaining_work"] = (
                "Re-extract MiniMax-H3 layer 20/25 timestep features with non-saturating storage and retrain; "
                "existing weights/evaluations at these layers are invalid; " + row.get("remaining_work", "")
            )
            row["notes"] = (row.get("notes", "").rstrip(". ") + ". " if row.get("notes") else "") + (
                f"{AUDIT_DATE}: AOSS feature cache contains inf (artifacts/release_audit_20261015/feature_finiteness/)"
            )
            if row.get("correctness_status") == "NOT_VERIFIED":
                row["correctness_status"] = "DEGENERATE"

    print(json.dumps({"canonical_v2": canonical_counts, "timestep": timestep_counts,
                      "sweep_records": len({r.get('checkpoint_path') for r in sweep.values()}),
                      "generated_at": datetime.now(timezone.utc).isoformat()}, indent=2))
    if args.dry_run:
        return
    write_csv(INVENTORY / "canonical_v2.csv", fields, rows)
    write_csv(INVENTORY / "timestep.csv", tfields, trows)
    print("wrote", INVENTORY / "canonical_v2.csv", "and", INVENTORY / "timestep.csv")


if __name__ == "__main__":
    main()
