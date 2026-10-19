#!/usr/bin/env python3
"""Prepare exact historical-config jobs for LEGACY_AS_IS_LOOSE evaluation.

This is deliberately a small, display-oriented preparation step.  It never
uses model/layer/prefix filename similarity to identify a run: every job is
anchored by a frozen universe row and its reconciled checkpoint path.  The
output is entirely below results/analysis/q00_1096_loose/eval_runs/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/analysis/q00_1096_loose/eval_runs"
MISSING = ROOT / "results/analysis/q00_1096_loose/missing_numeric.csv"
RECON = ROOT / "results/analysis/q00_1096/universe_1096_reconciled.csv"
UNIVERSE = ROOT / "experiments/task_status_1096_before_expand_20260822.csv"

CANARY_IDS = (
    "lingbot_video_layer_benchmark_view_consistency_lingbot_multi_layer_l10_p8_hall_s42",
    "minimax_h3_layer_benchmark_streaming_depth_minimax_h3_multi_layer_l10_p24_hall_s42",
    "videomae_vjepa_layer_sweep_streaming_mask_videomae_multi_layer_l0_p8_hall_s42",
    "videomae_vjepa_layer_sweep_streaming_depth_vjepa2_multi_layer_l0_p24_hall_s42",
    "wan_cog_layer_sweep_path_integration_wan_run_dir_discovered_l0_p12_hall_s42",
)


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def regular_nonempty(path: Path | None) -> bool:
    """Return whether an exact historical local artifact is readable and non-empty."""
    try:
        return bool(path and path.is_file() and path.stat().st_size > 0)
    except OSError:
        return False


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def local_path(raw: str) -> Path | None:
    raw = clean(raw)
    if not raw or raw.startswith(("aoss://", "s3://")):
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    try:
        resolved = path.resolve(strict=False)
        resolved.relative_to(ROOT)
    except (OSError, ValueError):
        return None
    return resolved


def read_overrides(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("- ") and "=" in line:
            key, value = line[2:].split("=", 1)
            values[key] = value
    return values


def find_run_dir(checkpoint: Path) -> Path | None:
    for parent in (checkpoint.parent, *checkpoint.parents):
        if parent == ROOT.parent:
            break
        if (parent / "run_manifest.json").is_file():
            return parent
    return None


def original_manifest_from_log(path: Path) -> str:
    if not path.is_file():
        return ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if text.startswith("MANIFEST="):
            candidate = local_path(text.split("=", 1)[1])
            if candidate and candidate.is_file():
                return str(candidate.relative_to(ROOT))
    return ""


def original_launch_log(run_dir: Path | None, rec: dict[str, str]) -> Path | None:
    """Locate only the exact historical launcher log for this run identity."""
    if run_dir is None:
        return None
    campaign_run_id = run_dir.parent.name
    launch_dir = ROOT / "logs/inscene15k_streaming" / campaign_run_id / "launch"
    if not launch_dir.is_dir():
        return None
    probe = clean(rec.get("probe"))
    model = clean(rec.get("model"))
    prefix = clean(rec.get("prefix"))
    layer = clean(rec.get("layer"))
    expected = launch_dir / f"{campaign_run_id}_{probe}_{model}_p{prefix}_layer{layer}.log"
    return expected if expected.is_file() else None


def config_value(config: dict[str, Any], key: str) -> Any:
    value = config.get(key)
    return value if value is not None else ""


def json_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def terminal_receipts(out_dir: Path) -> dict[str, dict[str, Any]]:
    """Index prior loose-eval terminal receipts by their exact frozen parent ID.

    A failed as-is evaluation is evidence, not a reason to silently retry the
    same checkpoint/cache pair.  Keeping it in the inventory as HOLD makes the
    next batch deterministic and preserves the failure path for the loose
    display table.
    """
    indexed: dict[str, dict[str, Any]] = {}
    for path in sorted(out_dir.rglob("terminal_receipt.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("evaluation_mode") != "LEGACY_AS_IS_LOOSE":
            continue
        universe_id = clean(payload.get("universe_id"))
        status = clean(payload.get("status"))
        if not universe_id or status not in {"PASS", "FAIL"}:
            continue
        rel = str(path.relative_to(ROOT))
        if universe_id in indexed:
            existing = indexed[universe_id]
            existing["multiple_terminal_receipts"] = True
            existing.setdefault("receipt_paths", [existing["receipt_path"]]).append(rel)
            continue
        indexed[universe_id] = {
            "status": status,
            "receipt_path": rel,
            "error": clean(payload.get("error")),
            "run_tag": clean(payload.get("run_tag")),
            "multiple_terminal_receipts": False,
            "receipt_paths": [rel],
        }
    return indexed


def prepare_row(
    missing: dict[str, str],
    rec: dict[str, str],
    universe: dict[str, str],
    prior_receipt: dict[str, Any] | None,
) -> dict[str, Any]:
    universe_id = missing["universe_id"]
    checkpoint = local_path(rec.get("checkpoint_path", ""))
    run_dir = find_run_dir(checkpoint) if checkpoint and checkpoint.is_file() else None
    run_manifest = run_dir / "run_manifest.json" if run_dir else None
    override_path = run_dir / ".hydra/overrides.yaml" if run_dir else None
    config_path = run_dir / ".hydra/config.yaml" if run_dir else None
    overrides = read_overrides(override_path) if override_path else {}
    payload: dict[str, Any] = {}
    if run_manifest and run_manifest.is_file():
        try:
            payload = json.loads(run_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    cfg = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    experiment = clean(overrides.get("experiment"))
    train_log = local_path(universe.get("train_log_path", ""))
    launch_log = original_launch_log(run_dir, rec)
    manifest = original_manifest_from_log(launch_log) if launch_log else ""
    if not manifest and train_log:
        manifest = original_manifest_from_log(train_log)
    # A reconciled train manifest is equally admissible only when it exists.
    if not manifest:
        candidate = local_path(rec.get("train_manifest", ""))
        if candidate and candidate.is_file():
            manifest = str(candidate.relative_to(ROOT))
    family = clean(rec.get("family"))
    evaluator = "a1" if family == "A1" else "diag"
    action_horizons = config_value(cfg, "action_horizons")
    extra: dict[str, Any] = {}
    if isinstance(action_horizons, list):
        extra["action_horizons"] = action_horizons
    for key, value in overrides.items():
        if key.startswith("streaming_hidden_") or key == "counterfactual_min_overlap":
            extra[key] = value
    fields = {
        "universe_id": universe_id,
        "campaign": clean(rec.get("campaign")),
        "protocol": clean(rec.get("protocol")),
        "model": clean(rec.get("model")),
        "model_variant": clean(rec.get("model_variant")),
        "family": family,
        "probe": clean(rec.get("probe")),
        "layer": clean(rec.get("layer")),
        "prefix": clean(rec.get("prefix")),
        "horizon": clean(rec.get("horizon")),
        "seed": clean(rec.get("seed")),
        "evaluation_mode": "LEGACY_AS_IS_LOOSE",
        "evaluator": evaluator,
        "experiment": experiment,
        "vfm_name": clean(config_value(cfg, "vfm_name")),
        "video_channels": clean(config_value(cfg, "video_channels")),
        "streaming_feat_root": clean(config_value(cfg, "streaming_feat_root")),
        "target_feat_root": clean(config_value(cfg, "target_feat_root")),
        "feature_layer": clean(config_value(cfg, "feature_layer")),
        "feat_postfix": clean(config_value(cfg, "feat_postfix")),
        "feature_timestep": clean(config_value(cfg, "feature_timestep")),
        "prefix_len": clean(config_value(cfg, "prefix_len")),
        "extra_overrides_json": json_compact(extra),
        "checkpoint": str(checkpoint.relative_to(ROOT)) if checkpoint else "",
        "checkpoint_sha256": "",
        "checkpoint_size_bytes": str(checkpoint.stat().st_size) if checkpoint and checkpoint.is_file() else "",
        "original_manifest": manifest,
        "run_dir": str(run_dir.relative_to(ROOT)) if run_dir else "",
        "run_manifest": str(run_manifest.relative_to(ROOT)) if regular_nonempty(run_manifest) else "",
        "run_manifest_sha256": sha256(run_manifest) if regular_nonempty(run_manifest) else "",
        "overrides": str(override_path.relative_to(ROOT)) if regular_nonempty(override_path) else "",
        "overrides_sha256": sha256(override_path) if regular_nonempty(override_path) else "",
        "hydra_config": str(config_path.relative_to(ROOT)) if regular_nonempty(config_path) else "",
        "train_log": str(train_log.relative_to(ROOT)) if train_log and train_log.is_file() else "",
        "launcher_log": str(launch_log.relative_to(ROOT)) if launch_log and launch_log.is_file() else "",
        "known_label_feature_mismatch": missing.get("known_label_feature_mismatch", "UNKNOWN"),
        "known_manifest_mismatch": missing.get("known_manifest_mismatch", "UNKNOWN"),
        "lineage_status": missing.get("lineage_status", ""),
        "warning": missing.get("warning", ""),
        "prior_loose_eval_status": clean(prior_receipt.get("status")) if prior_receipt else "",
        "prior_loose_eval_receipt": clean(prior_receipt.get("receipt_path")) if prior_receipt else "",
        "prior_loose_eval_error": clean(prior_receipt.get("error")) if prior_receipt else "",
        "prior_loose_eval_run_tag": clean(prior_receipt.get("run_tag")) if prior_receipt else "",
        "prior_loose_eval_receipt_paths_json": json_compact(prior_receipt.get("receipt_paths", [])) if prior_receipt else "[]",
    }
    required = ["experiment", "vfm_name", "video_channels", "streaming_feat_root", "feature_layer", "feature_timestep", "prefix_len", "checkpoint", "original_manifest", "run_manifest", "overrides", "hydra_config"]
    if family in {"C1", "C2", "C3"}:
        required.append("target_feat_root")
    absent = [key for key in required if not clean(fields[key])]
    fields["preflight_status"] = "READY_LEGACY_AS_IS_LOOSE" if not absent else "NOT_READY_" + "_".join(absent)
    if prior_receipt:
        if prior_receipt.get("multiple_terminal_receipts"):
            fields["preflight_status"] = "HOLD_MULTIPLE_PRIOR_LOOSE_EVAL_RECEIPTS"
        elif prior_receipt.get("status") == "FAIL":
            fields["preflight_status"] = "HOLD_PREVIOUS_LOOSE_EVAL_FAILURE"
        elif prior_receipt.get("status") == "PASS":
            fields["preflight_status"] = "HOLD_PREVIOUS_LOOSE_EVAL_SUCCESS"
    fields["checkpoint_available"] = "true" if checkpoint and checkpoint.is_file() else "false"
    return fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if not MISSING.is_file() or not RECON.is_file() or not UNIVERSE.is_file():
        raise SystemExit("Build the loose table before preparing loose-eval jobs")
    out_dir = args.out_dir.resolve()
    missing_rows = read_csv(MISSING)
    rec_by_id = {row["universe_id"]: row for row in read_csv(RECON)}
    universe_by_id = {row["experiment_id"]: row for row in read_csv(UNIVERSE)}
    candidates = [row for row in missing_rows if row.get("checkpoint_available") == "true"]
    prior_by_id = terminal_receipts(out_dir)
    jobs = [
        prepare_row(
            row,
            rec_by_id[row["universe_id"]],
            universe_by_id[row["universe_id"]],
            prior_by_id.get(row["universe_id"]),
        )
        for row in candidates
    ]
    expected_jobs = len(candidates)
    if len(jobs) != expected_jobs or len({row["universe_id"] for row in jobs}) != expected_jobs:
        raise SystemExit(
            f"Expected {expected_jobs} exact checkpoint-available current-missing rows, got {len(jobs)}"
        )
    fields = list(jobs[0])
    write_csv(out_dir / "legacy_as_is_loose_checkpoint_inventory.csv", jobs, fields)
    canaries = [row for row in jobs if row["universe_id"] in CANARY_IDS]
    write_csv(out_dir / "legacy_as_is_loose_canaries.csv", canaries, fields)
    status_counts: dict[str, int] = {}
    for row in jobs:
        status_counts[row["preflight_status"]] = status_counts.get(row["preflight_status"], 0) + 1
    summary = {
        "evaluation_mode": "LEGACY_AS_IS_LOOSE",
        "checkpoint_available_missing_numeric": len(jobs),
        "ready_jobs": sum(row["preflight_status"] == "READY_LEGACY_AS_IS_LOOSE" for row in jobs),
        "preflight_status_counts": status_counts,
        "declared_canary_ids": list(CANARY_IDS),
        "active_canary_ids": [row["universe_id"] for row in canaries],
        "active_canary_statuses": {row["universe_id"]: row["preflight_status"] for row in canaries},
        "prior_terminal_receipts_indexed": len(prior_by_id),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (MISSING, RECON, UNIVERSE)},
        "inventory": str((out_dir / "legacy_as_is_loose_checkpoint_inventory.csv").relative_to(ROOT)),
        "canaries": str((out_dir / "legacy_as_is_loose_canaries.csv").relative_to(ROOT)),
    }
    (out_dir / "legacy_as_is_loose_prepare_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
