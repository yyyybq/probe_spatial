#!/usr/bin/env python3
"""Collect 20260817 eval summaries into registry and canonical CSVs."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = Path(
    os.environ.get("EVAL_REGISTRY_PATH", ROOT / "experiments/registry.csv")
).resolve()
CANONICAL = Path(
    os.environ.get("EVAL_CANONICAL_DIR", ROOT / "results/canonical")
).resolve()
C_RAW = ROOT / "logs/inscene15k_streaming/results/c_eval_with_extra_20260728/c_series_raw_metrics_with_extra.csv"
WAN_COG = ROOT / "logs/probing_jobs_20260725_ab/recovery_20260730/perf_analysis/evaluation_results_all_probes.tsv"

RUN_IDS = [
    "missing_eval_20260817",
    "missing_a1_eval_20260817",
    "full_missing_eval_20260817",
    "full_missing_a1_eval_20260817",
    "full_missing_a1_eval_retry_ckpt_20260817",
    "remaining_diag_4node_20260817",
    "remaining_a1_4node_20260817",
]
RUN_IDS.extend(
    item.strip()
    for item in os.environ.get("EXTRA_EVAL_RUN_IDS", "").split(",")
    if item.strip()
)

PROBES = (
    "visible_ego_belief_v2",
    "view_consistency",
    "streaming_depth",
    "streaming_mask",
    "ego_belief_v2",
    "ego_belief",
    "action_dynamics",
    "path_integration",
    "counterfactual",
    "abnormal",
)
MODELS = ("cogvideox", "lingbot", "minimax_h3", "videomae", "vjepa2", "wan")
JOB_RE = re.compile(
    r"(?P<probe>"
    + "|".join(re.escape(p) for p in sorted(PROBES, key=len, reverse=True))
    + r")_(?P<model>"
    + "|".join(re.escape(model) for model in sorted(MODELS, key=len, reverse=True))
    + r")_p(?P<prefix>\d+)_layer(?P<layer>[-\w.]+)"
)

REGISTRY_FIELDS = [
    "experiment_id",
    "family",
    "probe",
    "model",
    "backend",
    "model_variant",
    "layer",
    "prefix",
    "horizon",
    "seed",
    "train_split",
    "eval_split",
    "dataset_manifest",
    "config",
    "checkpoint",
    "launcher_job_id",
    "raw_artifact_path",
    "canonical_metric_path",
    "metric_name",
    "metric_value",
    "status",
    "status_reason",
    "supersedes",
    "git_commit",
    "notes",
]


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def rel(path: str | Path) -> str:
    if not path:
        return ""
    p = Path(path)
    if p.is_absolute():
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(path)
    return str(path)


def git_commit() -> str:
    override = os.environ.get("EVAL_GIT_COMMIT_OVERRIDE", "").strip().lower()
    if override:
        if not re.fullmatch(r"[0-9a-f]{7,40}", override):
            raise ValueError("invalid EVAL_GIT_COMMIT_OVERRIDE")
        return override[:7]
    proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.stdout.strip() or "UNKNOWN"


def clean(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower().replace("@", "at").replace("/", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_.+-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "na"


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"n/a", "nan", "none", "?"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def registry_id(family: str, probe: str, model: str, variant: str, layer: str, prefix: str, horizon: str, seed: str) -> str:
    return "_".join(
        [
            clean(family),
            clean(probe),
            clean(model),
            clean(variant),
            f"l{clean(layer)}",
            f"p{clean(prefix)}",
            f"h{clean(horizon)}",
            f"s{clean(seed)}",
        ]
    )


def parse_job_key(text: str) -> dict[str, str] | None:
    matches = list(JOB_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    out = match.groupdict()
    out["job_key"] = f"{out['probe']}_{out['model']}_p{out['prefix']}_layer{out['layer']}"
    return out


def manifest_lookup() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for path in ROOT.glob("logs/probing_jobs*20260817/jobs/*.tsv"):
        with path.open(newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                key = row.get("job_key")
                if key:
                    out[key] = {**out.get(key, {}), **row}
    return out


def valid_summary(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(errors="replace"))
    except Exception:
        return False
    for key in ("n_samples", "n", "num_samples"):
        if key in data:
            try:
                return int(data[key]) > 0
            except Exception:
                pass
    return True


def summaries() -> list[tuple[Path, dict[str, Any], dict[str, str]]]:
    out = []
    for run_id in RUN_IDS:
        root = ROOT / "logs/inscene15k_streaming/runs" / run_id
        if not root.exists():
            continue
        for path in root.glob("*/eval/val_summary.json"):
            if not valid_summary(path):
                continue
            data = json.loads(path.read_text(errors="replace"))
            parsed = parse_job_key(data.get("job_name", "") or path.as_posix())
            if parsed:
                parsed["run_id"] = run_id
                out.append((path, data, parsed))
    return out


def family_for_probe(probe: str) -> str:
    return {
        "streaming_depth": "A1",
        "streaming_mask": "A1",
        "view_consistency": "A2",
        "ego_belief": "B1",
        "ego_belief_v2": "B2",
        "action_dynamics": "C1",
        "path_integration": "C2",
        "counterfactual": "C3",
    }.get(probe, "UNKNOWN")


def canonical_path(family: str) -> str:
    return {
        "A1": "results/canonical/a1_results.csv",
        "A2": "results/canonical/a2_results.csv",
        "B2": "results/canonical/b2_results.csv",
        "C1": "results/canonical/c1_results.csv",
        "C2": "results/canonical/c2_results.csv",
        "C3": "results/canonical/c3_results.csv",
    }.get(family, "")


def infer_config(probe: str, model: str) -> str:
    stem = probe.split("/")[0]
    path = ROOT / "configs/experiment/inscene15k_streaming" / f"{stem}_{model}_v1.yaml"
    return rel(path) if path.exists() else ""


def c_manifest(prefix: str) -> str:
    for name in (f"inscene_c_h124_prefix{prefix}.jsonl", f"inscene_c_prefix{prefix}.jsonl"):
        path = ROOT / "manifests/c_retrain_20260722" / name
        if path.exists():
            return rel(path)
    return ""


def primary_metrics(probe: str, data: dict[str, Any]) -> list[tuple[str, str, Any]]:
    if probe == "streaming_depth":
        rows = []
        if parse_float(data.get("mean_loss_depth")) is not None:
            rows.append(("streaming_depth/depth", "mean_loss_depth", data.get("mean_loss_depth")))
        auc30 = data.get("Auc_30", data.get("mean_Auc_30"))
        if parse_float(auc30) is not None:
            rows.append(("streaming_depth/camera", "Auc_30", auc30))
        return rows
    if probe == "streaming_mask":
        return [("streaming_mask/mask", "mean_loss_identity", data.get("mean_loss_identity"))]
    if probe == "view_consistency":
        return [(probe, "balanced_acc", data.get("balanced_acc"))]
    if probe == "ego_belief_v2":
        return [(probe, "mean_ang_err_deg", data.get("mean_ang_err_deg"))]
    if probe == "ego_belief":
        return [(probe, "mean_az_err_deg", data.get("mean_az_err_deg"))]
    if probe in {"action_dynamics", "path_integration"}:
        return [(probe, "global_R@1", data.get("global_R@1"))]
    if probe == "counterfactual":
        return [(probe, "intervention_validity", data.get("intervention_validity"))]
    return []


def collect_registry_rows() -> list[dict[str, str]]:
    lookup = manifest_lookup()
    commit = git_commit()
    best: dict[tuple[str, str], tuple[float, dict[str, str]]] = {}
    for path, data, parsed in summaries():
        job_key = parsed["job_key"]
        manifest = lookup.get(job_key, {})
        probe = manifest.get("probe") or parsed["probe"]
        model = manifest.get("vfm") or parsed["model"]
        prefix = manifest.get("prefix") or str(data.get("prefix_len") or parsed["prefix"])
        layer = manifest.get("layer") or str(data.get("feature_layer") or parsed["layer"])
        family = family_for_probe(probe)
        if family == "UNKNOWN":
            continue
        for probe_label, metric_name, metric_value in primary_metrics(probe, data):
            if parse_float(metric_value) is None:
                continue
            eid = registry_id(family, probe_label, model, "eval_20260817", layer, prefix, "all", "42")
            row = {
                "experiment_id": eid,
                "family": family,
                "probe": probe_label,
                "model": model,
                "backend": model,
                "model_variant": "eval_20260817",
                "layer": layer,
                "prefix": prefix,
                "horizon": "all",
                "seed": "42",
                "train_split": "train",
                "eval_split": "val",
                "dataset_manifest": rel(manifest.get("scene_manifest") or c_manifest(prefix)),
                "config": rel(manifest.get("experiment", "").replace("inscene15k_streaming/", "configs/experiment/inscene15k_streaming/") + ".yaml")
                if manifest.get("experiment")
                else infer_config(probe_label, model),
                "checkpoint": rel(data.get("ckpt_path", "")),
                "launcher_job_id": data.get("job_name", job_key),
                "raw_artifact_path": rel(path),
                "canonical_metric_path": canonical_path(family),
                "metric_name": metric_name,
                "metric_value": str(metric_value),
                "status": "COMPLETE",
                "status_reason": "20260817 eval summary collected",
                "supersedes": "",
                "git_commit": commit,
                "notes": f"source=20260817_eval_summary; run_id={parsed['run_id']}; job_key={job_key}",
            }
            best[(eid, metric_name)] = (path.stat().st_mtime, row)
    return [row for _, row in sorted(best.values(), key=lambda item: item[1]["experiment_id"])]


def load_summary_for_registry_row(row: dict[str, str]) -> dict[str, Any]:
    path = ROOT / row.get("raw_artifact_path", "")
    if not path.exists() or path.suffix != ".json":
        return {}
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception:
        return {}


def c_lookup() -> dict[tuple[str, str, str, str], dict[str, str]]:
    out = {}
    for row in read_csv(C_RAW):
        out[(row.get("probe_type", ""), row.get("vfm_name", ""), row.get("layer", ""), row.get("prefix", ""))] = row
    return out


def wan_cog_lookup() -> dict[str, dict[str, str]]:
    return {row.get("job_id", ""): row for row in read_csv(WAN_COG, delimiter="\t")}


def source_metrics(row: dict[str, str], c_rows: dict[tuple[str, str, str, str], dict[str, str]], wan_rows: dict[str, dict[str, str]]) -> dict[str, Any]:
    summary = load_summary_for_registry_row(row)
    if summary:
        return summary
    if row.get("family") in {"C1", "C2", "C3"}:
        return c_rows.get((row["probe"], row["model"], row["layer"], row["prefix"]), {})
    if row.get("family") == "B2":
        return wan_rows.get(row.get("launcher_job_id", ""), {})
    return {}


def complete_registry() -> list[dict[str, str]]:
    return [row for row in read_csv(REGISTRY) if row.get("status") == "COMPLETE"]


def build_canonical() -> None:
    rows = complete_registry()
    c_rows = c_lookup()
    wan_rows = wan_cog_lookup()
    simple_fields = [
        "experiment_id",
        "family",
        "probe",
        "model",
        "model_variant",
        "layer",
        "prefix",
        "horizon",
        "seed",
        "eval_split",
        "metric_name",
        "metric_value",
        "status",
        "raw_artifact_path",
    ]
    for family, filename in (("A1", "a1_results.csv"), ("A2", "a2_results.csv")):
        out = []
        for row in rows:
            if row.get("family") == family and parse_float(row.get("metric_value")) is not None:
                out.append({field: row.get(field, "") for field in simple_fields})
        write_csv(CANONICAL / filename, out, simple_fields)

    b2 = []
    for row in rows:
        if row.get("family") != "B2" or row.get("metric_name") != "mean_ang_err_deg":
            continue
        metrics = source_metrics(row, c_rows, wan_rows)
        b2.append(
            {
                "experiment_id": row["experiment_id"],
                "model": row["model"],
                "model_variant": row["model_variant"],
                "layer": row["layer"],
                "prefix": row["prefix"],
                "seed": row["seed"],
                "eval_split": row["eval_split"],
                "angular_error_mean": row["metric_value"],
                "angular_error_median": metrics.get("median_ang_err_deg", ""),
                "top1": metrics.get("top1", ""),
                "top3": metrics.get("top3", ""),
                "mean_log_dist_err": metrics.get("mean_log_dist_err", ""),
                "status": row["status"],
                "raw_artifact_path": row["raw_artifact_path"],
            }
        )
    write_csv(
        CANONICAL / "b2_results.csv",
        b2,
        [
            "experiment_id",
            "model",
            "model_variant",
            "layer",
            "prefix",
            "seed",
            "eval_split",
            "angular_error_mean",
            "angular_error_median",
            "top1",
            "top3",
            "mean_log_dist_err",
            "status",
            "raw_artifact_path",
        ],
    )

    c1 = []
    for row in rows:
        if row.get("family") != "C1":
            continue
        metrics = source_metrics(row, c_rows, wan_rows)
        global_r1 = parse_float(metrics.get("global_R@1", row.get("metric_value")))
        no_action = parse_float(metrics.get("no_action_global_R@1"))
        shuffled = parse_float(metrics.get("shuffled_action_global_R@1"))
        copy = parse_float(metrics.get("last_observation_global_R@1"))
        c1.append(
            {
                "experiment_id": row["experiment_id"],
                "model": row["model"],
                "layer": row["layer"],
                "prefix": row["prefix"],
                "seed": row["seed"],
                "eval_split": row["eval_split"],
                "global_R@1": global_r1 if global_r1 is not None else "",
                "global_R@5": metrics.get("global_R@5", ""),
                "no_action_R@1": no_action if no_action is not None else "",
                "shuffled_action_R@1": shuffled if shuffled is not None else "",
                "last_observation_R@1": copy if copy is not None else "",
                "action_gain_vs_no_action": global_r1 - no_action if global_r1 is not None and no_action is not None else "",
                "action_gain_vs_shuffle": global_r1 - shuffled if global_r1 is not None and shuffled is not None else "",
                "gain_over_copy": global_r1 - copy if global_r1 is not None and copy is not None else "",
                "action_gain_cos": metrics.get("action_gain_cos", ""),
                "status": row["status"],
                "raw_artifact_path": row["raw_artifact_path"],
            }
        )
    write_csv(
        CANONICAL / "c1_results.csv",
        c1,
        [
            "experiment_id",
            "model",
            "layer",
            "prefix",
            "seed",
            "eval_split",
            "global_R@1",
            "global_R@5",
            "no_action_R@1",
            "shuffled_action_R@1",
            "last_observation_R@1",
            "action_gain_vs_no_action",
            "action_gain_vs_shuffle",
            "gain_over_copy",
            "action_gain_cos",
            "status",
            "raw_artifact_path",
        ],
    )

    c2 = []
    for row in rows:
        if row.get("family") != "C2":
            continue
        metrics = source_metrics(row, c_rows, wan_rows)
        c2.append(
            {
                "experiment_id": row["experiment_id"],
                "model": row["model"],
                "layer": row["layer"],
                "prefix": row["prefix"],
                "horizon": "all",
                "seed": row["seed"],
                "eval_split": row["eval_split"],
                "metric_name": "global_R@1",
                "metric_value": metrics.get("global_R@1", row.get("metric_value")),
                "global_R@5": metrics.get("global_R@5", ""),
                "final_pose_error": metrics.get("final_pose_error", ""),
                "drift_rate": metrics.get("drift_rate", ""),
                "loop_closure_error": metrics.get("loop_closure_error", ""),
                "status": row["status"],
                "raw_artifact_path": row["raw_artifact_path"],
            }
        )
        for horizon in ("1", "2", "4"):
            value = metrics.get(f"pose_error_h{horizon}", "")
            if value != "":
                c2.append(
                    {
                        "experiment_id": row["experiment_id"],
                        "model": row["model"],
                        "layer": row["layer"],
                        "prefix": row["prefix"],
                        "horizon": horizon,
                        "seed": row["seed"],
                        "eval_split": row["eval_split"],
                        "metric_name": "pose_error",
                        "metric_value": value,
                        "status": row["status"],
                        "raw_artifact_path": row["raw_artifact_path"],
                    }
                )
    write_csv(
        CANONICAL / "c2_results.csv",
        c2,
        [
            "experiment_id",
            "model",
            "layer",
            "prefix",
            "horizon",
            "seed",
            "eval_split",
            "metric_name",
            "metric_value",
            "global_R@5",
            "final_pose_error",
            "drift_rate",
            "loop_closure_error",
            "status",
            "raw_artifact_path",
        ],
    )

    c3 = []
    for row in rows:
        if row.get("family") != "C3":
            continue
        metrics = source_metrics(row, c_rows, wan_rows)
        for metric in ("intervention_validity", "intervention_margin", "counterfactual_consistency", "global_R@1"):
            c3.append(
                {
                    "experiment_id": row["experiment_id"],
                    "model": row["model"],
                    "layer": row["layer"],
                    "prefix": row["prefix"],
                    "horizon": "all",
                    "seed": row["seed"],
                    "eval_split": row["eval_split"],
                    "metric_name": metric,
                    "metric_value": metrics.get(metric, row.get("metric_value") if metric == row.get("metric_name") else ""),
                    "status": row["status"],
                    "raw_artifact_path": row["raw_artifact_path"],
                }
            )
        for horizon in ("1", "2", "4"):
            value = metrics.get(f"cf_cos_h{horizon}", "")
            if value != "":
                c3.append(
                    {
                        "experiment_id": row["experiment_id"],
                        "model": row["model"],
                        "layer": row["layer"],
                        "prefix": row["prefix"],
                        "horizon": horizon,
                        "seed": row["seed"],
                        "eval_split": row["eval_split"],
                        "metric_name": "cf_cos",
                        "metric_value": value,
                        "status": row["status"],
                        "raw_artifact_path": row["raw_artifact_path"],
                    }
                )
    write_csv(
        CANONICAL / "c3_results.csv",
        c3,
        [
            "experiment_id",
            "model",
            "layer",
            "prefix",
            "horizon",
            "seed",
            "eval_split",
            "metric_name",
            "metric_value",
            "status",
            "raw_artifact_path",
        ],
    )

    all_results = []
    for row in rows:
        if parse_float(row.get("metric_value")) is None:
            continue
        all_results.append(
            {
                "family": row["family"],
                "probe": row["probe"],
                "model": row["model"],
                "layer": row["layer"],
                "prefix": row["prefix"],
                "metric_name": row["metric_name"],
                "metric_value": row["metric_value"],
                "experiment_id": row["experiment_id"],
                "raw_artifact_path": row["raw_artifact_path"],
            }
        )
    write_csv(
        CANONICAL / "all_results.csv",
        sorted(all_results, key=lambda r: (r["family"], r["model"], r["probe"], int(r["prefix"]) if str(r["prefix"]).isdigit() else 0, str(r["layer"]))),
        ["family", "probe", "model", "layer", "prefix", "metric_name", "metric_value", "experiment_id", "raw_artifact_path"],
    )


def main() -> None:
    existing = read_csv(REGISTRY)
    kept = [
        row
        for row in existing
        if "source=20260817_eval_summary" not in row.get("notes", "")
        and row.get("model_variant") != "eval_20260817"
    ]
    new_rows = collect_registry_rows()
    write_csv(REGISTRY, kept + new_rows, REGISTRY_FIELDS)
    build_canonical()
    print(f"kept_registry_rows={len(kept)}")
    print(f"collected_eval_registry_rows={len(new_rows)}")
    print(f"wrote_registry_rows={len(kept) + len(new_rows)}")
    print(f"wrote_canonical_dir={CANONICAL}")


if __name__ == "__main__":
    main()
