#!/usr/bin/env python3
"""Collect the frozen 576-experiment timestep probing matrix.

The universe is three video-generation backbones, two scheduler timesteps each,
layers {15,20,25}, prefixes {8,12,16,24}, and eight probes.  Completion follows
TRAIN_COMPLETE markers.  Headline numbers are the last logged validation metrics
from offline W&B history — training-time validation, not a later eval_diag pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "logs/inscene15k_streaming/runs"
OUT = ROOT / "results/analysis/tstep_canonical_v2_seeded_v1"
NAMESPACES = (
    "tstep_probe_seeded_v1",
    "tstep_probe_opt_v2",
    "tstep_probe_opt_v3",
    "tstep_probe_pack_v4",
    "tstep_probe_pack_v4_1",
)
LOG_NAMESPACES = (
    "tstep_probe_seeded_v1",
    "tstep_probe_opt_v2",
    "tstep_probe_opt_v3",
    "tstep_probe_pack_v4",
    "tstep_probe_pack_v4_1",
)
MODELS = {
    "wan": (499, 999),
    "cogvideox": (499, 999),
    "minimax_h3": (500, 1000),
}
LAYERS = (15, 20, 25)
PREFIXES = (8, 12, 16, 24)
PROBES = (
    "streaming_depth",
    "streaming_mask",
    "view_consistency",
    "ego_belief",
    "ego_belief_v2",
    "action_dynamics",
    "path_integration",
    "counterfactual",
)
HEAVY = {"streaming_depth", "streaming_mask"}
EXPECTED_BUDGET = {
    "heavy": {"max_epochs": 100, "global_step": 244500},
    "light": {"max_epochs": 50, "global_step": 61100},
}
PRIMARY = {
    "streaming_depth": ("val/loss_depth", "min", "A1 Depth"),
    "streaming_mask": ("val/loss_identity", "min", "A1 Mask"),
    "view_consistency": ("val/overlap_acc", "max", "A2 View"),
    "ego_belief": ("val/ego_az_err_deg", "min", "B1 Ego"),
    "ego_belief_v2": ("val/belief_ang_err_deg", "min", "B2 Persistence"),
    "action_dynamics": ("val/dyn_cos", "max", "C1 Dynamics"),
    "path_integration": ("val/path_cos", "max", "C2 Integration"),
    "counterfactual": ("val/cf_cos", "max", "C3 Counterfactual"),
}
PRIMARY_FALLBACKS = {
    "streaming_depth": ("val/loss",),
    "streaming_mask": ("val/loss",),
    "view_consistency": ("val/loss_overlap", "val/loss"),
    "ego_belief": ("val/loss_ego", "val/loss"),
    "ego_belief_v2": ("val/loss_belief", "val/loss"),
    "action_dynamics": ("val/dyn_mse", "val/loss_dyn", "val/loss"),
    "path_integration": ("val/path_mse", "val/loss_path", "val/loss"),
    "counterfactual": ("val/cf_mse", "val/loss_cf", "val/loss"),
}
NAME_RE = re.compile(r"^tp_(wan|cogvideox|minimax_h3)_t(\d+)_l(\d+)_p(\d+)_(.+)$")
CLAIM_RE = re.compile(
    r"^(wan|cogvideox|minimax_h3)__scannetpp__t(\d+)__l(\d+)__p(\d+)__(.+)__seed(\d+)$"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def expected_universe() -> list[dict[str, Any]]:
    rows = []
    for model, timesteps in MODELS.items():
        for timestep in timesteps:
            for layer in LAYERS:
                for prefix in PREFIXES:
                    for probe in PROBES:
                        rows.append({
                            "unique_key": f"{model}:scannetpp:{timestep}:{layer}:{prefix}:{probe}:seed42",
                            "model": model,
                            "source": "scannetpp",
                            "timestep": timestep,
                            "layer": layer,
                            "prefix": prefix,
                            "probe": probe,
                            "seed": 42,
                            "run_tag": f"tp_{model}_t{timestep}_l{layer}_p{prefix}_{probe}",
                        })
    return rows


def parse_tag(name: str) -> dict[str, Any] | None:
    match = NAME_RE.match(name)
    if not match:
        return None
    model, timestep, layer, prefix, probe = match.groups()
    if probe not in PROBES:
        return None
    return {
        "model": model,
        "timestep": int(timestep),
        "layer": int(layer),
        "prefix": int(prefix),
        "probe": probe,
        "unique_key": f"{model}:scannetpp:{timestep}:{layer}:{prefix}:{probe}:seed42",
        "run_tag": name,
    }


def scandir_dirs(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return [Path(entry.path) for entry in os.scandir(path) if entry.is_dir()]


def list_experiment_dirs() -> list[dict[str, Any]]:
    rows = []
    for namespace in NAMESPACES:
        ns_root = RUNS / namespace
        for batch in scandir_dirs(ns_root):
            if batch.name.startswith("tp_"):
                candidates = [batch]
                batch_name = namespace
            else:
                candidates = [child for child in scandir_dirs(batch) if child.name.startswith("tp_")]
                batch_name = batch.name
            for exp in candidates:
                parsed = parse_tag(exp.name)
                if parsed is None:
                    continue
                complete = (exp / "TRAIN_COMPLETE").is_file()
                rows.append({
                    **parsed,
                    "namespace": namespace,
                    "batch": batch_name,
                    "output_dir": str(exp),
                    "train_complete": complete,
                    "mtime": exp.stat().st_mtime,
                })
    return rows


def list_active_claims() -> set[str]:
    keys = set()
    for namespace in LOG_NAMESPACES:
        root = ROOT / "logs" / namespace
        for campaign in scandir_dirs(root):
            claims = campaign / "claims"
            if not claims.is_dir():
                continue
            for entry in os.scandir(claims):
                match = CLAIM_RE.match(entry.name)
                if match:
                    model, timestep, layer, prefix, probe, seed = match.groups()
                    keys.add(f"{model}:scannetpp:{timestep}:{layer}:{prefix}:{probe}:seed{seed}")
    return keys


def choose_primary_dir(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not attempts:
        return None
    complete = [row for row in attempts if row["train_complete"]]
    pool = complete or attempts
    return max(pool, key=lambda row: (row["train_complete"], row["mtime"]))


def wandb_paths(output_dir: Path) -> list[Path]:
    wandb_root = output_dir / "wandb"
    if not wandb_root.is_dir():
        return []
    paths = []
    for run in scandir_dirs(wandb_root):
        if not run.name.startswith("offline-run-"):
            continue
        for entry in os.scandir(run):
            if entry.is_file() and entry.name.startswith("run-") and entry.name.endswith(".wandb"):
                paths.append(Path(entry.path))
    return paths


def parse_wandb_file(path: str) -> dict[str, Any]:
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    latest: dict[str, Any] = {}
    n_hist = 0
    n_val = 0
    source = Path(path)
    with tempfile.TemporaryDirectory(prefix="tstep-wandb-") as tmp:
        copy = Path(tmp) / source.name
        shutil.copyfile(source, copy)
        store = DataStore()
        store.open_for_scan(str(copy))
        while True:
            try:
                data = store.scan_data()
            except (AssertionError, EOFError):
                break
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.WhichOneof("record_type") != "history":
                continue
            row = {}
            for item in record.history.item:
                key = item.key or "/".join(item.nested_key)
                try:
                    row[key] = json.loads(item.value_json)
                except json.JSONDecodeError:
                    row[key] = item.value_json
            if not row:
                continue
            n_hist += 1
            latest.update(row)
            if any(str(key).startswith("val/") for key in row):
                n_val += 1
    val = {key: value for key, value in latest.items() if str(key).startswith("val/")}
    return {
        "status": "parsed",
        "n_history": n_hist,
        "n_val_rows": n_val,
        "epoch": latest.get("epoch"),
        "global_step": latest.get("trainer/global_step"),
        "val": val,
        "bytes": source.stat().st_size,
        "path": path,
    }


def parse_one_run(payload: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(payload["output_dir"])
    paths = wandb_paths(output_dir)
    if not paths:
        return {**payload, "wandb_status": "missing", "val": {}, "epoch": None, "global_step": None}
    if len(paths) != 1:
        return {**payload, "wandb_status": f"multiple_{len(paths)}", "val": {}, "epoch": None, "global_step": None}
    try:
        parsed = parse_wandb_file(str(paths[0]))
    except Exception as exc:
        return {**payload, "wandb_status": f"failed:{type(exc).__name__}", "val": {}, "epoch": None, "global_step": None}
    return {
        **payload,
        "wandb_status": parsed["status"],
        "val": parsed["val"],
        "epoch": parsed.get("epoch"),
        "global_step": parsed.get("global_step"),
        "n_history": parsed.get("n_history"),
        "n_val_rows": parsed.get("n_val_rows"),
        "wandb_bytes": parsed.get("bytes"),
        "wandb_path": parsed.get("path"),
    }


def as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def select_primary(probe: str, val: dict[str, Any]) -> tuple[str | None, float | None, str]:
    name, direction, _ = PRIMARY[probe]
    if name in val:
        number = as_float(val[name])
        if number is not None:
            return name, number, direction
    for fallback in PRIMARY_FALLBACKS[probe]:
        if fallback in val:
            number = as_float(val[fallback])
            if number is not None:
                return fallback, number, direction
    return None, None, direction


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-wandb", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/tstep_mpl")

    universe = expected_universe()
    attempts = list_experiment_dirs()
    claims = list_active_claims()
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        by_key[row["unique_key"]].append(row)

    census = []
    parse_jobs = []
    for item in universe:
        key = item["unique_key"]
        chosen = choose_primary_dir(by_key.get(key, []))
        budget = EXPECTED_BUDGET["heavy" if item["probe"] in HEAVY else "light"]
        if chosen and chosen["train_complete"]:
            status = "TRAIN_COMPLETE"
        elif key in claims:
            status = "RUNNING"
        elif chosen:
            status = "PARTIAL"
        else:
            status = "MISSING"
        row = {
            **item,
            "status": status,
            "n_attempts": len(by_key.get(key, [])),
            "output_dir": chosen["output_dir"] if chosen else "",
            "namespace": chosen["namespace"] if chosen else "",
            "batch": chosen["batch"] if chosen else "",
            "train_complete": bool(chosen and chosen["train_complete"]),
            "writer_claim_active": key in claims,
            "expected_epochs": budget["max_epochs"],
            "expected_global_step": budget["global_step"],
        }
        census.append(row)
        if chosen and chosen["train_complete"] and not args.skip_wandb:
            parse_jobs.append(row)

    if args.limit:
        parse_jobs = parse_jobs[: args.limit]

    parsed_by_key: dict[str, dict[str, Any]] = {}
    if parse_jobs:
        workers = max(1, args.workers)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(parse_one_run, job): job["unique_key"] for job in parse_jobs}
            done = 0
            for future in as_completed(futures):
                result = future.result()
                parsed_by_key[result["unique_key"]] = result
                done += 1
                if done % 25 == 0 or done == len(parse_jobs):
                    print(f"wandb parsed {done}/{len(parse_jobs)}", flush=True)

    results = []
    metrics_long = []
    for row in census:
        parsed = parsed_by_key.get(row["unique_key"], {})
        val = parsed.get("val") or {}
        metric_name, metric_value, direction = select_primary(row["probe"], val) if val else (None, None, PRIMARY[row["probe"]][1])
        epoch = as_float(parsed.get("epoch"))
        step = as_float(parsed.get("global_step"))
        budget = EXPECTED_BUDGET["heavy" if row["probe"] in HEAVY else "light"]
        full_budget = (
            epoch is not None and step is not None
            and int(epoch) == budget["max_epochs"] - 1
            and int(step) + 1 >= budget["global_step"]
        )
        result = {
            **row,
            "wandb_status": parsed.get("wandb_status", ""),
            "epoch": epoch,
            "global_step": step,
            "n_history": parsed.get("n_history"),
            "n_val_rows": parsed.get("n_val_rows"),
            "full_budget": full_budget,
            "capability": PRIMARY[row["probe"]][2],
            "metric_name": metric_name or PRIMARY[row["probe"]][0],
            "metric_value": metric_value,
            "metric_direction": direction,
            "has_primary": metric_value is not None,
            "val_json": json.dumps(val, sort_keys=True, default=str) if val else "",
        }
        results.append(result)
        for name, value in val.items():
            number = as_float(value)
            if number is None:
                continue
            metrics_long.append({
                "unique_key": row["unique_key"],
                "model": row["model"],
                "timestep": row["timestep"],
                "layer": row["layer"],
                "prefix": row["prefix"],
                "probe": row["probe"],
                "metric_name": name,
                "metric_value": number,
            })

    counts = Counter(row["status"] for row in census)
    summary = {
        "generated_at_utc": utc_now(),
        "universe": len(universe),
        "attempt_dirs": len(attempts),
        "active_claims": sorted(claims),
        "status_counts": dict(counts),
        "train_complete": int(counts["TRAIN_COMPLETE"]),
        "running": int(counts["RUNNING"]),
        "partial": int(counts["PARTIAL"]),
        "missing": int(counts["MISSING"]),
        "primary_metric_rows": sum(row["has_primary"] for row in results),
        "wandb_parsed": len(parsed_by_key),
        "metric_note": (
            "Headline metrics are last-epoch training-time validation logs. "
            "C1/C2 in-batch R@1 is NaN at batch=2, so those probes use cosine similarity. "
            "This is not a substitute for eval_diag global_R@1 / intervention_validity."
        ),
    }
    (out / "collection_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(out / "census.csv", census, [
        "unique_key", "model", "source", "timestep", "layer", "prefix", "probe", "seed",
        "status", "n_attempts", "train_complete", "writer_claim_active", "namespace",
        "batch", "output_dir", "expected_epochs", "expected_global_step",
    ])
    write_csv(out / "results.csv", results, [
        "unique_key", "model", "timestep", "layer", "prefix", "probe", "seed", "status",
        "capability", "metric_name", "metric_value", "metric_direction", "has_primary",
        "train_complete", "full_budget", "epoch", "global_step", "wandb_status",
        "n_history", "n_val_rows", "namespace", "batch", "output_dir",
    ])
    write_csv(out / "data" / "metrics_long.csv", metrics_long, [
        "unique_key", "model", "timestep", "layer", "prefix", "probe", "metric_name", "metric_value",
    ])
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
