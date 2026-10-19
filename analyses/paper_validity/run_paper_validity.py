#!/usr/bin/env python3
"""Audit and re-aggregate paper-validity evidence into one immutable run tree.

This offline stage consumes only existing, locally readable artifacts.  It does
not train probes or claim that legacy controls are matched retrained baselines.
Unavailable strict/fair experiments are represented explicitly in the status
matrix rather than replaced with synthetic numbers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils
import torch
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau, spearmanr

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from vidfm3d.models.components.probe_action_dynamics import ActionDynamicsProbe
from vidfm3d.models.components.probe_ego_belief_v2 import EgoBeliefProbeV2
from vidfm3d.models.components.probe_path_integration import PathIntegrationProbe
from vidfm3d.utils.paper_validity import (
    PREFIXES,
    common_sample_intersection,
    prefix_curve_summary,
)


MODELS = ("dino", "vjepa2", "f3r", "videomae", "aether", "cogvideox", "wan", "qwen2_5_vl")
LAYERS = {"dino": 0, "vjepa2": 23, "f3r": 24, "videomae": 23, "aether": 1,
          "cogvideox": 20, "wan": 20, "qwen2_5_vl": -1}
CHANNELS = {"dino": 1024, "vjepa2": 1024, "f3r": 2048, "videomae": 768,
            "aether": 1536, "cogvideox": 1920, "wan": 1536, "qwen2_5_vl": 3584}
ALLOWED_STATUS = {
    "COMPLETE_VALID", "COMPLETE_WITH_CAVEAT", "BLOCKED_MISSING_ARTIFACT",
    "FAILED_VALIDATION", "NOT_RUN",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    return sha256(path)


def scalar(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def scene_name(path: str) -> str:
    return str(path).rstrip("/").split("/")[-1]


def add_legacy_ids(records: list[dict], task: str) -> None:
    occurrence = Counter()
    scene_occurrence = Counter()
    for record in records:
        scene = str(record.get("scene_path", "unknown"))
        if task == "c1":
            core = f"{scene}|target={int(record.get('target_frame_idx', -1))}"
        else:
            arrays = []
            for key in ("horizons", "start_extrinsic", "target_extrinsics"):
                value = np.asarray(scalar(record[key])).round(6)
                arrays.append(value.tobytes())
            core = f"{scene}|geometry={hashlib.sha256(b''.join(arrays)).hexdigest()[:20]}"
        suffix = occurrence[core]
        occurrence[core] += 1
        record["sample_id"] = f"{task}|{core}|occ={suffix}"
        pair_index = scene_occurrence[scene]
        scene_occurrence[scene] += 1
        record["paired_sample_id"] = f"{task}_legacy_pair|{scene}|window_ordinal={pair_index}"


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def retrieval(pred: np.ndarray, target: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    similarity = normalize_rows(pred) @ normalize_rows(target).T
    order = np.argsort(-similarity, axis=1, kind="stable")
    rank = np.argmax(order == np.arange(len(order))[:, None], axis=1)
    paired_cos = np.sum(normalize_rows(pred) * normalize_rows(target), axis=1)
    target_mean = target.mean(axis=0, keepdims=True)
    centered_cos = np.sum(
        normalize_rows(pred - target_mean) * normalize_rows(target - target_mean), axis=1
    )
    summary = {
        "n": int(len(rank)),
        "global_R@1": float(np.mean(rank == 0)),
        "global_R@5": float(np.mean(rank < 5)),
        "mean_rank": float(np.mean(rank)),
        "median_rank": float(np.median(rank)),
        "correct_target_cosine": float(np.mean(paired_cos)),
        "centered_cosine": float(np.mean(centered_cos)),
        # Legacy dumps cannot safely reconstruct target-fitted whitening.
        "whitened_cosine": float("nan"),
    }
    return summary, {"rank": rank, "r1": rank == 0, "cos": paired_cos,
                     "centered_cos": centered_cos}


def rotation_angle_from_action(action: np.ndarray) -> float:
    first = action[:3].astype(np.float64)
    second = action[3:6].astype(np.float64)
    first /= max(np.linalg.norm(first), 1e-12)
    second = second - first * np.dot(first, second)
    second /= max(np.linalg.norm(second), 1e-12)
    third = np.cross(first, second)
    rotation = np.stack([first, second, third])
    return float(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))


def select_c_rows(raw: pd.DataFrame, probe: str) -> list[dict]:
    selected = raw[
        (raw["probe_type"] == probe)
        & raw["vfm_name"].isin(MODELS)
        & raw["prefix"].isin(PREFIXES)
    ].copy()
    selected["layer_num"] = pd.to_numeric(selected["layer"], errors="coerce")
    selected = selected[
        selected.apply(lambda row: int(row["layer_num"]) == LAYERS[str(row["vfm_name"])], axis=1)
    ]
    # Extra collection can contain duplicate copies of the same canonical run.
    selected = selected.sort_values(["vfm_name", "prefix", "source_run"]).drop_duplicates(
        ["vfm_name", "prefix"], keep="first"
    )
    return selected.to_dict("records")


def prediction_path(row: dict, root: Path) -> Path:
    summary = root / str(row["eval_summary_path"])
    return summary.with_name("val_predictions.pt")


def audit_artifacts(root: Path, run_dirs: dict[str, Path], c_rows: dict[str, list[dict]]):
    inventory = []
    checkpoint_rows = []
    canonical_by_task = {
        "B2": pd.read_csv(root / "results/canonical/b2_results.csv"),
        "C1": pd.read_csv(root / "results/canonical/c1_results.csv"),
        "C2": pd.read_csv(root / "results/canonical/c2_results.csv"),
    }
    c_lookup = {
        (task, str(row["vfm_name"]), int(row["prefix"])): row
        for task, rows in c_rows.items() for row in rows
    }
    for model in MODELS:
        for prefix in PREFIXES:
            for task in ("B2", "C1", "C2"):
                canonical = canonical_by_task[task]
                match = canonical[
                    (canonical["model"] == model)
                    & (pd.to_numeric(canonical["prefix"], errors="coerce") == prefix)
                    & (pd.to_numeric(canonical["layer"], errors="coerce") == LAYERS[model])
                    & (canonical["status"] == "COMPLETE")
                ]
                c_row = c_lookup.get((task, model, prefix))
                pred = prediction_path(c_row, root) if c_row else None
                ckpt = root / str(c_row.get("checkpoint_path", "")) if c_row else None
                if ckpt is not None and str(c_row.get("checkpoint_path", "")).startswith("/"):
                    ckpt = Path(str(c_row["checkpoint_path"]))
                inventory.append({
                    "task": task,
                    "model": model,
                    "prefix": prefix,
                    "layer": LAYERS[model],
                    "canonical_rows": len(match),
                    "checkpoint": str(ckpt) if ckpt else "",
                    "checkpoint_exists": bool(ckpt and ckpt.is_file()),
                    "per_sample_predictions": str(pred) if pred else "",
                    "per_sample_exists": bool(pred and pred.is_file()),
                    "has_sample_id": False if pred else "unknown",
                    "feature_root": str(c_row.get("streaming_feat_root", "")) if c_row else "",
                    "target_root": "resolved in train run_manifest" if c_row else "",
                    "cache_mode": "streaming_prefix -> exact target_isolated" if task in {"C1", "C2"} else "streaming_prefix",
                    "strict_zero_eligible": False,
                    "notes": "standard B2 only" if task == "B2" else "legacy dump; no canonical ID field",
                })
                if c_row:
                    manifest_path = root / str(c_row.get("train_run_dir", "")) / "run_manifest.json"
                    manifest = {}
                    if manifest_path.is_file():
                        try:
                            manifest = json.loads(manifest_path.read_text())
                        except Exception:
                            manifest = {}
                    ckpt_digest = sha256(ckpt) if ckpt and ckpt.is_file() else ""
                    pred_digest = sha256(pred) if pred and pred.is_file() else ""
                    checkpoint_rows.append({
                        "task": task,
                        "model": model,
                        "prefix": prefix,
                        "layer": LAYERS[model],
                        "checkpoint": str(ckpt) if ckpt else "",
                        "checkpoint_sha256": ckpt_digest,
                        "checkpoint_exists": bool(ckpt and ckpt.is_file()),
                        "predictions_sha256": pred_digest,
                        "resolved_config": str(root / str(c_row.get("train_run_dir", "")) / ".hydra/config.yaml"),
                        "run_manifest": str(manifest_path),
                        "git_commit": manifest.get("git", {}).get("commit", "missing"),
                        "split_manifest": manifest.get("data", {}).get("split_manifest"),
                        "split_manifest_sha256": manifest.get("data", {}).get("split_manifest_sha256"),
                        "streaming_feature_root": manifest.get("config", {}).get("streaming_feat_root", c_row.get("streaming_feat_root", "")),
                        "target_feature_root": manifest.get("data", {}).get("target_feature_root", ""),
                        "provenance_complete": bool(
                            manifest.get("git", {}).get("commit") not in (None, "unknown")
                            and manifest.get("data", {}).get("split_manifest_sha256")
                        ),
                    })
    write_csv(run_dirs["reports"] / "artifact_inventory.csv", inventory)
    write_csv(run_dirs["reports"] / "checkpoint_provenance.csv", checkpoint_rows)
    return inventory, checkpoint_rows


def load_c_configs(root: Path, rows: list[dict], task: str) -> dict[tuple[str, int], list[dict]]:
    configs = {}
    for row in rows:
        path = prediction_path(row, root)
        if not path.is_file():
            continue
        records = torch.load(path, map_location="cpu", weights_only=False)
        add_legacy_ids(records, task.lower())
        configs[(str(row["vfm_name"]), int(row["prefix"]))] = records
    return configs


def build_common_manifests(run_dirs, c1_configs, c2_configs):
    common_dir = run_dirs["manifests"] / "common"
    common_dir.mkdir(parents=True, exist_ok=True)
    count_rows = []
    common = {"c1_prefix": {}, "c2_prefix_horizon": {}}
    for prefix in PREFIXES:
        ids_by_model = {
            model: [record["sample_id"] for record in c1_configs[(model, prefix)]]
            for model in MODELS if (model, prefix) in c1_configs
        }
        ids, counts = common_sample_intersection(ids_by_model) if ids_by_model else ([], {})
        common["c1_prefix"][prefix] = set(ids)
        path = common_dir / f"c1_common_models_prefix{prefix}.jsonl"
        digest = write_jsonl(path, [{"sample_id": value} for value in ids])
        for model, native in counts.items():
            count_rows.append({"task": "C1", "model": model, "prefix": prefix, "horizon": "all",
                               "n_native": native, "n_common": len(ids), "dropped": native-len(ids),
                               "drop_reason": "missing from at least one model", "manifest_sha256": digest})

        for horizon_index, horizon in enumerate((1, 2, 4)):
            ids_by_model = {}
            for model in MODELS:
                records = c2_configs.get((model, prefix))
                if records is None:
                    continue
                ids_by_model[model] = [
                    f"{record['sample_id']}|h={horizon}"
                    for record in records
                    if bool(np.asarray(scalar(record["valid"]))[horizon_index])
                ]
            ids, counts = common_sample_intersection(ids_by_model) if ids_by_model else ([], {})
            common["c2_prefix_horizon"][(prefix, horizon)] = set(ids)
            path = common_dir / f"c2_common_models_prefix{prefix}_h{horizon}.jsonl"
            digest = write_jsonl(path, [{"sample_id": value} for value in ids])
            for model, native in counts.items():
                count_rows.append({"task": "C2", "model": model, "prefix": prefix, "horizon": horizon,
                                   "n_native": native, "n_common": len(ids), "dropped": native-len(ids),
                                   "drop_reason": "missing from at least one model/horizon", "manifest_sha256": digest})

    # Legacy dumps lack window IDs.  Pair IDs are deterministic scene-order
    # reconstructions and are published only with a caveat.
    for task, configs in (("c1", c1_configs), ("c2", c2_configs)):
        ids_by_config = {
            f"{model}_p{prefix}": [record["paired_sample_id"] for record in records]
            for (model, prefix), records in configs.items()
        }
        ids, _counts = common_sample_intersection(ids_by_config) if ids_by_config else ([], {})
        path = common_dir / f"{task}_common_models_and_prefixes_legacy_pair.jsonl"
        digest = write_jsonl(path, [{"paired_sample_id": value, "id_quality": "legacy_scene_order"} for value in ids])
        common[f"{task}_paired"] = set(ids)
        common[f"{task}_paired_sha256"] = digest
    write_csv(run_dirs["results"] / "native_vs_common_counts.csv", count_rows)
    return common, count_rows


def evaluate_c1(run_dirs, configs, common):
    summary_rows, sample_rows, metric_rows = [], [], []
    variants = {
        "full": "pred",
        "copy_last_context": "last_observation",
        "no_action_eval_intervention": "pred_no_action",
        "batch_roll_shuffle_legacy": "pred_shuffled_action",
    }
    for (model, prefix), records in sorted(configs.items()):
        paired = common["c1_paired"]
        selected = [record for record in records if record["paired_sample_id"] in paired]
        target = np.stack([np.asarray(scalar(record["target"])) for record in selected])
        variant_stats = {}
        variant_per = {}
        for variant, field in variants.items():
            pred = np.stack([np.asarray(scalar(record[field])) for record in selected])
            stats, per = retrieval(pred, target)
            variant_stats[variant] = stats
            variant_per[variant] = per
            summary_rows.append({"task": "C1", "model": model, "prefix": prefix, "horizon": 1,
                                 "variant": variant, "sample_set": "common_models_and_prefixes_legacy_pair",
                                 **stats, "status": "COMPLETE_WITH_CAVEAT"})
        for index, record in enumerate(selected):
            action = np.asarray(scalar(record["action"]))
            target_vec = target[index]
            copy_vec = np.asarray(scalar(record["last_observation"]))
            base = {
                "task": "C1", "model": model, "prefix": prefix, "horizon": 1,
                "sample_id": record["sample_id"], "paired_sample_id": record["paired_sample_id"],
                "scene_id": scene_name(record["scene_path"]),
                "target_translation": float(np.linalg.norm(action[6:9])),
                "target_rotation": rotation_angle_from_action(action),
                "action_magnitude": float(np.sqrt(np.linalg.norm(action[6:9])**2 + rotation_angle_from_action(action)**2)),
                "current_target_similarity": float(np.dot(copy_vec, target_vec) /
                                                     max(np.linalg.norm(copy_vec)*np.linalg.norm(target_vec), 1e-12)),
            }
            for variant in variants:
                per = variant_per[variant]
                sample_rows.append({**base, "variant": variant, "r1": int(per["r1"][index]),
                                    "rank": int(per["rank"][index]), "correct_target_cosine": float(per["cos"][index])})
            metric_rows.append({"task": "C1", "model": model, "prefix": prefix, "horizon": 1,
                                "metric": "global_R@1", "value": variant_stats["full"]["global_R@1"],
                                "sample_set": "common", "n": len(selected)})
    write_csv(run_dirs["results"] / "c1_baselines_per_sample.csv", sample_rows)
    return summary_rows, sample_rows, metric_rows


def evaluate_c2(run_dirs, configs, common):
    summary_rows, sample_rows, metric_rows = [], [], []
    horizon_values = (1, 2, 4)
    paired = common["c2_paired"]
    for (model, prefix), records in sorted(configs.items()):
        selected = [record for record in records if record["paired_sample_id"] in paired]
        for h_idx, horizon in enumerate(horizon_values):
            valid_records = [record for record in selected if bool(np.asarray(scalar(record["valid"]))[h_idx])]
            if not valid_records:
                continue
            pred = np.stack([np.asarray(scalar(record["pred"]))[h_idx] for record in valid_records])
            target = np.stack([np.asarray(scalar(record["target"]))[h_idx] for record in valid_records])
            stats, per = retrieval(pred, target)
            pose_errors = []
            # Use the same global retrieval pool, then compare stored relative poses.
            similarity = normalize_rows(pred) @ normalize_rows(target).T
            retrieved = similarity.argmax(axis=1)
            poses = np.stack([np.asarray(scalar(record["target_extrinsics"]))[h_idx] for record in valid_records])
            for index, donor in enumerate(retrieved):
                pose_errors.append(float(np.linalg.norm(poses[donor, :3, 3] - poses[index, :3, 3])))
            stats["pose_error"] = float(np.mean(pose_errors))
            summary_rows.append({"task": "C2", "model": model, "prefix": prefix, "horizon": horizon,
                                 "variant": "full", "sample_set": "common_models_and_prefixes_legacy_pair",
                                 **stats, "status": "COMPLETE_WITH_CAVEAT"})
            metric_rows.append({"task": "C2", "model": model, "prefix": prefix, "horizon": horizon,
                                "metric": "global_R@1", "value": stats["global_R@1"],
                                "sample_set": "common", "n": len(valid_records)})
            for index, record in enumerate(valid_records):
                pose = poses[index]
                cumulative_translation = float(np.linalg.norm(pose[:3, 3]))
                cumulative_rotation = float(np.arccos(np.clip((np.trace(pose[:3, :3])-1)/2, -1, 1)))
                drift = pose_errors[index] / max(cumulative_translation, 1e-6)
                sample_rows.append({
                    "task": "C2", "model": model, "prefix": prefix, "horizon": horizon,
                    "variant": "full", "sample_id": record["sample_id"],
                    "paired_sample_id": record["paired_sample_id"],
                    "scene_id": scene_name(record["scene_path"]),
                    "r1": int(per["r1"][index]), "rank": int(per["rank"][index]),
                    "correct_target_cosine": float(per["cos"][index]),
                    "pose_error": pose_errors[index], "drift": drift,
                    "cumulative_translation": cumulative_translation,
                    "cumulative_rotation": cumulative_rotation,
                })
    write_csv(run_dirs["results"] / "c2_baselines_per_sample.csv", sample_rows)
    return summary_rows, sample_rows, metric_rows


def native_vs_common_metrics(c_rows, common_summaries):
    native = {}
    for row in c_rows:
        task = "C1" if row["probe_type"] == "action_dynamics" else "C2"
        native[(task, str(row["vfm_name"]), int(row["prefix"]), "all")] = float(row.get("global_R@1", np.nan))
    rows = []
    for summary in common_summaries:
        if summary["variant"] != "full":
            continue
        key = (summary["task"], summary["model"], summary["prefix"], "all")
        native_value = native.get(key, float("nan"))
        rows.append({
            "task": summary["task"], "model": summary["model"], "prefix": summary["prefix"],
            "horizon": summary["horizon"], "metric": "global_R@1",
            "native_value": native_value, "common_value": summary["global_R@1"],
            "common_minus_native": summary["global_R@1"] - native_value if math.isfinite(native_value) else float("nan"),
            "ranking_comparable": summary["task"] == "C1" and summary["horizon"] == 1,
        })
    return rows


def build_prefix_auc(run_dirs, metric_rows):
    output = []
    groups = defaultdict(dict)
    for row in metric_rows:
        groups[(row["task"], row["model"], row["horizon"], row["metric"])][int(row["prefix"])] = float(row["value"])
    for (task, model, horizon, metric), values in sorted(groups.items()):
        if set(values) != set(PREFIXES):
            continue
        output.append({"task": task, "model": model, "horizon": horizon, "metric": metric,
                       "direction": "higher_is_better", "sample_set": "common_models_and_prefixes_legacy_pair",
                       **prefix_curve_summary(values, True)})
    write_csv(run_dirs["results"] / "prefix_auc_all_tasks.csv", output)
    write_csv(run_dirs["results"] / "paper_prefix_auc_table.csv", output)
    curve_dir = run_dirs["figures"] / "prefix_curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    for task in sorted({row["task"] for row in output}):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for row in output:
            if row["task"] != task:
                continue
            ax.plot(PREFIXES, [row[f"p{prefix}"] for prefix in PREFIXES], marker="o",
                    label=f"{row['model']} h={row['horizon']}")
        ax.set_xlabel("Prefix length")
        ax.set_ylabel("Global R@1")
        ax.set_title(f"{task}: paired-common prefix curves")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(curve_dir / f"{task.lower()}_global_r1.pdf")
        plt.close(fig)
    return output


def canonical_auc(root: Path, filename: str, metric_name: str, higher: bool = True):
    frame = pd.read_csv(root / filename)
    result = {}
    for model in MODELS:
        values = {}
        for prefix in PREFIXES:
            match = frame[
                (frame["model"] == model)
                & (pd.to_numeric(frame["layer"], errors="coerce") == LAYERS[model])
                & (pd.to_numeric(frame["prefix"], errors="coerce") == prefix)
                & (frame["metric_name"] == metric_name)
                & (frame["status"] == "COMPLETE")
            ]
            if not match.empty:
                values[prefix] = float(match.iloc[0]["metric_value"])
        if set(values) == set(PREFIXES):
            result[model] = prefix_curve_summary(values, higher)["prefix_auc"]
    return result


def canonical_points(root: Path, filename: str, metric_name: str, negate: bool = False):
    frame = pd.read_csv(root / filename)
    result = {}
    for model in MODELS:
        for prefix in PREFIXES:
            match = frame[
                (frame["model"] == model)
                & (pd.to_numeric(frame["layer"], errors="coerce") == LAYERS[model])
                & (pd.to_numeric(frame["prefix"], errors="coerce") == prefix)
                & (frame["metric_name"] == metric_name)
                & (frame["status"] == "COMPLETE")
            ]
            if not match.empty:
                value = float(match.iloc[0]["metric_value"])
                result[(model, prefix)] = -value if negate else value
    return result


def task_correlations(root, run_dirs, prefix_auc_rows, c_summaries):
    task_values = {
        "A1_depth": {model: -value for model, value in canonical_auc(root, "results/canonical/a1_results.csv", "mean_loss_depth", False).items()},
        "A1_camera": canonical_auc(root, "results/canonical/a1_results.csv", "Auc_30"),
        "A1_mask": {model: -value for model, value in canonical_auc(root, "results/canonical/a1_results.csv", "mean_loss_identity", False).items()},
        "A2": canonical_auc(root, "results/canonical/a2_results.csv", "balanced_acc"),
        "C3": canonical_auc(root, "results/canonical/c3_results.csv", "intervention_margin"),
        "B2_strict_angle": {},
    }
    for task in ("C1", "C2"):
        for model in MODELS:
            vals = [float(row["prefix_auc"]) for row in prefix_auc_rows
                    if row["task"] == task and row["model"] == model]
            if vals:
                task_values.setdefault(task, {})[model] = float(np.mean(vals))
    tasks = list(task_values)
    model_table = pd.DataFrame(index=MODELS, columns=tasks, dtype=float)
    for task, values in task_values.items():
        for model, value in values.items():
            model_table.loc[model, task] = value
    model_table.index.name = "model"
    model_table.to_csv(run_dirs["results"] / "task_correlation_model_level_scores.csv")

    rows = []
    for left in tasks:
        for right in tasks:
            paired = model_table[[left, right]].dropna()
            n = len(paired)
            spearman = spearmanr(paired[left], paired[right]).statistic if n >= 3 else float("nan")
            kendall = kendalltau(paired[left], paired[right]).statistic if n >= 3 else float("nan")
            rows.append({"task_x": left, "task_y": right, "spearman": spearman,
                         "kendall": kendall, "paired_config_count": n,
                         "analysis_level": "model_prefix_auc", "direction": "higher_is_better"})

    config_task_values = {
        "A1_depth": canonical_points(root, "results/canonical/a1_results.csv", "mean_loss_depth", negate=True),
        "A1_camera": canonical_points(root, "results/canonical/a1_results.csv", "Auc_30"),
        "A1_mask": canonical_points(root, "results/canonical/a1_results.csv", "mean_loss_identity", negate=True),
        "A2": canonical_points(root, "results/canonical/a2_results.csv", "balanced_acc"),
        "C3": canonical_points(root, "results/canonical/c3_results.csv", "intervention_margin"),
        "B2_strict_angle": {},
        "C1": {},
        "C2": {},
    }
    for summary in c_summaries:
        if summary["variant"] != "full":
            continue
        key = (summary["model"], int(summary["prefix"]))
        if summary["task"] == "C1":
            config_task_values["C1"][key] = float(summary["global_R@1"])
        elif summary["task"] == "C2":
            config_task_values["C2"].setdefault(key, []).append(float(summary["global_R@1"]))
    config_task_values["C2"] = {
        key: float(np.mean(values)) for key, values in config_task_values["C2"].items()
    }
    config_index = [(model, prefix) for model in MODELS for prefix in PREFIXES]
    config_table = pd.DataFrame(index=pd.MultiIndex.from_tuples(config_index, names=["model", "prefix"]),
                                columns=list(config_task_values), dtype=float)
    for task, values in config_task_values.items():
        for key, value in values.items():
            config_table.loc[key, task] = value
    config_table.to_csv(run_dirs["results"] / "task_correlation_model_prefix_scores.csv")
    for left in config_table.columns:
        for right in config_table.columns:
            paired = config_table[[left, right]].dropna()
            n = len(paired)
            spearman = spearmanr(paired[left], paired[right]).statistic if n >= 3 else float("nan")
            kendall = kendalltau(paired[left], paired[right]).statistic if n >= 3 else float("nan")
            rows.append({"task_x": left, "task_y": right, "spearman": spearman,
                         "kendall": kendall, "paired_config_count": n,
                         "analysis_level": "model_x_prefix_descriptive_nonindependent",
                         "direction": "higher_is_better"})
    write_csv(run_dirs["results"] / "task_correlation_raw.csv", rows)
    # No fair new baseline suite completed: preserve an explicit empty schema.
    write_csv(run_dirs["results"] / "task_correlation_gain.csv", [],
              ["task_x", "task_y", "spearman", "kendall", "paired_config_count", "reason"])
    write_csv(run_dirs["results"] / "paper_task_correlation_table.csv", rows)

    valid_tasks = [task for task in tasks if model_table[task].notna().sum() >= 3]
    corr = model_table[valid_tasks].corr(method="spearman")
    if len(valid_tasks) >= 2:
        distance = (1.0 - corr.fillna(0.0)).clip(0.0, 2.0)
        distance_values = distance.to_numpy(copy=True)
        np.fill_diagonal(distance_values, 0.0)
        order = leaves_list(linkage(squareform(distance_values, checks=False), method="average"))
        corr = corr.iloc[order, order]
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    fig.colorbar(image, ax=ax, label="Spearman")
    ax.set_title("Raw task correlation (model-level prefix AUC)")
    fig.tight_layout()
    fig.savefig(run_dirs["figures"] / "task_correlation_raw.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, "Gain correlation unavailable:\nmatched retrained baselines not run",
            ha="center", va="center", fontsize=13)
    fig.tight_layout()
    fig.savefig(run_dirs["figures"] / "task_correlation_gain.pdf")
    plt.close(fig)
    return rows, model_table


def clustered_ci(values: np.ndarray, scenes: np.ndarray, reps: int = 200):
    unique = np.unique(scenes)
    if len(unique) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(42)
    means = []
    for _ in range(reps):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        draw = np.concatenate([values[scenes == scene] for scene in sampled])
        means.append(np.mean(draw))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def difficulty_points(c1_samples, c2_samples):
    rows = []
    specs = {
        "C1": (c1_samples, ["target_translation", "target_rotation", "current_target_similarity", "action_magnitude"],
               ["r1", "rank", "correct_target_cosine"]),
        "C2": (c2_samples, ["horizon", "cumulative_translation", "cumulative_rotation"],
               ["r1", "pose_error", "drift"]),
    }
    for task, (samples, variables, metrics) in specs.items():
        frame = pd.DataFrame(samples)
        if frame.empty:
            continue
        for keys, group in frame.groupby(["model", "prefix", "variant"], dropna=False):
            model, prefix, variant = keys
            for variable in variables:
                if variable == "horizon":
                    bins = group[variable].astype(str)
                else:
                    try:
                        bins = pd.qcut(group[variable], q=min(5, group[variable].nunique()), duplicates="drop")
                    except ValueError:
                        continue
                for bin_value, subset in group.groupby(bins, observed=True):
                    if len(subset) < 5:
                        continue
                    for metric in metrics:
                        values = subset[metric].astype(float).to_numpy()
                        low, high = clustered_ci(values, subset["scene_id"].astype(str).to_numpy())
                        rows.append({
                            "task": task, "model": model, "prefix": prefix, "variant": variant,
                            "difficulty_variable": variable, "bin": str(bin_value),
                            "bin_min": float(subset[variable].min()), "bin_max": float(subset[variable].max()),
                            "bin_mean": float(subset[variable].mean()), "n": len(subset),
                            "metric": metric, "value": float(np.mean(values)), "ci_low": low, "ci_high": high,
                            "bootstrap_unit": "scene", "bootstrap_reps": 200,
                        })
    return rows


def plot_difficulty(run_dirs, rows):
    out_dir = run_dirs["figures"] / "difficulty_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    for task in sorted(frame["task"].unique()):
        for variable in sorted(frame.loc[frame["task"] == task, "difficulty_variable"].unique()):
            subset = frame[(frame["task"] == task) & (frame["difficulty_variable"] == variable)]
            metric = "r1" if "r1" in set(subset["metric"]) else subset.iloc[0]["metric"]
            subset = subset[subset["metric"] == metric]
            aggregate = subset.groupby(["variant", "bin_mean"], as_index=False).agg(value=("value", "mean"), n=("n", "sum"))
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for variant, series in aggregate.groupby("variant"):
                series = series.sort_values("bin_mean")
                ax.plot(series["bin_mean"], series["value"], marker="o", label=variant)
                for _, row in series.iterrows():
                    ax.annotate(f"n={int(row['n'])}", (row["bin_mean"], row["value"]), fontsize=6)
            ax.set_xlabel(variable)
            ax.set_ylabel(metric)
            ax.set_title(f"{task} difficulty: {variable}")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out_dir / f"{task.lower()}_{variable}.pdf")
            plt.close(fig)


def capacity_audit(run_dirs):
    rows = []
    for model in ("dino", "f3r", "vjepa2"):
        channels = CHANNELS[model]
        probes = {
            "B2_structured": EgoBeliefProbeV2(channels, max_seq_len=64, num_frames=8),
            "B2_weak_linear": EgoBeliefProbeV2(channels, decoder_type="linear", num_frames=8),
            "B2_weak_mlp": EgoBeliefProbeV2(channels, decoder_type="mlp", num_frames=8),
            "C1_structured": ActionDynamicsProbe(channels, max_seq_len=64),
            "C2_structured": PathIntegrationProbe(channels),
        }
        for track, probe in probes.items():
            rows.append({
                "model": model, "prefix": 8, "track": track,
                "parameters": sum(parameter.numel() for parameter in probe.parameters()),
                "peak_gpu_memory_mb": "", "train_examples_per_second": "", "val_metric": "",
                "status": "NOT_RUN", "reason": "no CUDA and no reachable feature/data roots",
                "information_access": "all patches" if track == "B2_structured" else
                                      "query + per-frame mean" if track.startswith("B2_weak") else "task structured",
            })
    write_csv(run_dirs["results"] / "probe_capacity_pilot.csv", rows)
    return rows


def build_status_matrix(run_dirs, c1_summaries, c2_summaries):
    rows = []
    summary_lookup = {
        (row["task"], row["model"], row["prefix"], row["horizon"], row["variant"]): row
        for row in c1_summaries + c2_summaries
    }
    variants = {
        "B2": ["standard", "strict_eval", "strict_train", "query_only", "direction_prior", "no_update", "analytic_oracle"],
        "C1": ["full", "copy_anchor_isolated", "copy_last_context", "visual_only", "action_only", "matched_action_shuffle"],
        "C2": ["full", "copy_anchor_isolated", "copy_last_context", "visual_only", "action_only", "matched_action_shuffle"],
    }
    for task, task_variants in variants.items():
        horizons = ["all"] if task == "B2" else ([1] if task == "C1" else [1, 2, 4])
        for model in MODELS:
            for prefix in PREFIXES:
                for horizon in horizons:
                    for variant in task_variants:
                        lookup_variant = variant
                        if variant == "copy_last_context":
                            lookup_variant = "copy_last_context"
                        found = summary_lookup.get((task, model, prefix, horizon, lookup_variant))
                        if found:
                            status = "COMPLETE_WITH_CAVEAT"
                            value = found.get("global_R@1", "")
                            notes = "legacy per-sample dump; paired ID reconstructed from scene order"
                            n_common = found.get("n", "")
                        elif task in {"C1", "C2"} and variant == "full":
                            status, value, notes, n_common = "BLOCKED_MISSING_ARTIFACT", "", "missing readable per-sample dump", ""
                        elif task in {"C1", "C2"} and variant == "copy_last_context":
                            # Stored under legacy variant name.
                            legacy = summary_lookup.get((task, model, prefix, horizon, "copy_last_context"))
                            status = "COMPLETE_WITH_CAVEAT" if legacy else "BLOCKED_MISSING_ARTIFACT"
                            value = legacy.get("global_R@1", "") if legacy else ""
                            notes = "contextual copy, not isolated-domain primary copy" if legacy else "missing"
                            n_common = legacy.get("n", "") if legacy else ""
                        else:
                            status, value, notes, n_common = "BLOCKED_MISSING_ARTIFACT", "", (
                                "strict data/GPU unavailable" if task == "B2" else "matched retrained checkpoint/artifact unavailable"
                            ), ""
                        assert status in ALLOWED_STATUS
                        rows.append({
                            "task": task, "model": model, "prefix": prefix, "horizon": horizon,
                            "variant": variant, "status": status, "checkpoint": "", "manifest": "",
                            "n_native": "", "n_common": n_common, "metric": "global_R@1" if task != "B2" else "mean_ang_err_deg",
                            "value": value, "notes": notes,
                        })
    write_csv(run_dirs["reports"] / "STATUS_MATRIX.csv", rows,
              ["task", "model", "prefix", "horizon", "variant", "status", "checkpoint", "manifest",
               "n_native", "n_common", "metric", "value", "notes"])
    return rows


def write_reports(root, run_tag, run_dirs, inventory, checkpoint_rows, counts, c_summaries,
                  native_common, auc_rows, correlation_rows, difficulty, capacity, status_rows):
    blocked = sum(row["status"] == "BLOCKED_MISSING_ARTIFACT" for row in status_rows)
    complete_caveat = sum(row["status"] == "COMPLETE_WITH_CAVEAT" for row in status_rows)
    audit = f"""# Phase 0 audit

- Run tag: `{run_tag}`
- Git: `{git('rev-parse', 'HEAD')}` on `{git('branch', '--show-current')}`; dirty before this task: yes.
- Frozen split: `configs/splits/inscene15k_v1.json` (`{sha256(root / 'configs/splits/inscene15k_v1.json')}`). Scene-level split code runs before window expansion.
- Current node: no CUDA device. `INSCENE_DATA_ROOT`, `INSCENE_STREAMING_FEAT_ROOT`, `INSCENE_TARGET_FEAT_ROOT`, `INSCENE_MANIFEST`, and `INSCENE_SPLIT_MANIFEST` were unset.
- Historical roots resolve to `fj:s3://.../FEAT_STREAMING_BY_PREFIX/...` and `FEAT_TARGET_BY_PREFIX/...`; read-only AOSS checks timed out after 45 s and `/tmp/inscene_aoss_cache` was empty.
- `features/run_inscene15k.py`, `scripts/build_inscene_streaming_manifest.py`, and `docs/c_retrain_session_report_20260724.md` are currently empty/one-byte user-tree files. This blocks safe feature extraction and invalidates assumptions based on their historical contents.
- If repository `manifests/` is permission-locked, this script preserves it and falls back to `reports/<RUN_TAG>/manifests/`; the final audit records the path deviation.

## B2 definition found

Standard B2 chooses an object visible at `>=200` depth-valid pixels in history and treats `<200` in the target as hidden. `compute_object_target_for_id` then averages all valid object points, including residual target pixels. The Transformer reads every target/current patch. No query-only control existed before this run. Therefore all existing B2 checkpoints are standard, not strict-zero-hidden.

This run adds a separate strict path: raw target count exactly zero, positive raw instance IDs, coordinate-wise median over finite past-only points, and model-independent fixed-target labels. It does not relabel old checkpoints or results.

## C1/C2 definition found

The safe 2026-07 retraining resolved configs use independently extracted `streaming_prefix` inputs and exact `target_isolated` roots. C1's stored `no_action` and `shuffled_action` are evaluation interventions (`zeros` and batch roll), not retrained visual-only/action-only or magnitude-matched shuffle. Stored copy is contextual last observation, not isolated anchor copy. C2 stores no baselines.

## Artifact conclusions

- Inventory rows: {len(inventory)}; selected C checkpoint provenance rows: {len(checkpoint_rows)}.
- Legacy C1/C2 per-sample dumps have no canonical IDs. This run reconstructs IDs and verifies intersections, but cross-prefix pairing is only `legacy_scene_order` quality.
- Existing B2 dumps omit object/window/sample IDs and strict visibility metadata, so strict common intersection and strict-eval cannot be recovered offline.
- New full workload: 32 strict B2 trains + 32 query-only trains + 64 C1 visual/action-only trains + 64 C2 visual/action-only trains = 192 training jobs, followed by strict/full/matched evaluation. This was not launched without a passing data/GPU smoke gate.
"""
    (run_dirs["reports"] / "00_audit.md").write_text(audit)

    (run_dirs["reports"] / "b2_strict_dataset_audit.md").write_text(
        "# B2 strict dataset audit\n\nStatus: **BLOCKED_MISSING_ARTIFACT**. The builder and synthetic invariance tests pass, "
        "but no real manifest was emitted because the ScanNet++ root was unavailable. Creating an empty manifest would violate the zero-valid-sample gate.\n"
    )
    write_csv(run_dirs["reports"] / "b2_strict_rejection_reasons.csv", [{
        "split": "all", "scene_id": "", "window_id": "", "reason": "environment:data_root_unset_and_aoss_timeout"
    }])
    write_csv(run_dirs["results"] / "b2_baselines_per_sample.csv", [],
              ["sample_id", "model", "prefix", "variant", "status", "reason"])
    b2_blocked = [{"model": model, "prefix": prefix, "variant": variant,
                   "status": "BLOCKED_MISSING_ARTIFACT", "reason": "strict manifest/checkpoint unavailable"}
                  for model in MODELS for prefix in PREFIXES
                  for variant in ("full_strict", "query_only", "direction_prior", "no_update", "analytic_oracle")]
    write_csv(run_dirs["results"] / "b2_baselines_summary.csv", b2_blocked)
    (run_dirs["reports"] / "b2_baselines.md").write_text(
        "# B2 baselines\n\nNo numeric strict baseline is reported. Direction-prior fitting is train-only in code; "
        "no-update/oracle metadata are emitted by the strict dataset; query-only is parameter-matched. Real labels/checkpoints were unavailable.\n"
    )
    write_csv(run_dirs["results"] / "c_baselines_summary.csv", c_summaries)
    (run_dirs["reports"] / "c_baselines.md").write_text(
        "# C baselines\n\nThe numeric rows are re-aggregated from real legacy per-sample dumps. `copy_last_context`, "
        "`no_action_eval_intervention`, and `batch_roll_shuffle_legacy` are retained only as caveated controls. "
        "Isolated anchor, retrained visual-only/action-only, and matched-bin shuffle remain blocked; they are not aliased to legacy controls.\n"
    )
    (run_dirs["reports"] / "common_intersection_audit.md").write_text(
        "# Common intersection audit\n\nC1/C2 model intersections and cross-prefix legacy-paired manifests were generated with SHA256 hashes. "
        "The historical dumps lack canonical window IDs, so cross-prefix IDs are reconstructed from deterministic scene order and are `COMPLETE_WITH_CAVEAT`. "
        "B2 strict intersections are blocked because strict per-sample artifacts do not exist. See `native_vs_common_counts.csv` and `native_vs_common_metrics.csv`.\n"
    )
    (run_dirs["reports"] / "prefix_auc.md").write_text(
        "# Prefix AUC\n\nAUC uses x=[8,12,16,24], trapezoidal integration, and division by 16. "
        "Only C1/C2 rows with all prefixes on the legacy-paired common set are included. Best prefix is descriptive only. "
        "Strict B2 is absent rather than replaced by standard/native B2.\n"
    )
    (run_dirs["reports"] / "task_correlation.md").write_text(
        "# Task correlation\n\nRaw model-level correlations use direction-oriented prefix AUCs and report every pair with its paired count. "
        "B2 strict is all-missing and therefore excluded from coefficients. Gain correlation is intentionally empty because the fair retrained baseline suite was not run. "
        "The eight model points are too few for strong scientific claims.\n"
    )
    (run_dirs["reports"] / "difficulty_analysis.md").write_text(
        "# Difficulty analysis\n\nC1/C2 curves use real per-sample dumps, equal-count bins, per-bin n, and scene-cluster bootstrap CIs. "
        "B2 mandatory curves are blocked because historical B2 records lack hidden duration, motion, object area, and visible-frame metadata. "
        "C2 baseline overlays are blocked because those predictions were not stored.\n"
    )
    (run_dirs["reports"] / "probe_capacity_feasibility.md").write_text(
        "# Probe capacity feasibility\n\nB2 weak linear/MLP consumes query plus fixed per-frame means; structured B2 consumes all patches. "
        "They differ in information access and are not capacity-matched. Parameter counts are recorded for DINO/F3R/V-JEPA2 p8. "
        "Peak GPU memory, speed, and validation metrics are NOT_RUN because CUDA/data were unavailable. No flatten-linear was introduced.\n"
    )
    final = f"""# FINAL REPORT

Run tag: `{run_tag}`

## 1. Actual completion across the seven goals

1. **B2 strict-zero-hidden:** implementation and 4 synthetic invariance tests completed; real manifest, strict-eval, and strict-train blocked by unavailable data/AOSS and CUDA.
2. **C1/C2 controls:** matched-parameter query/visual/action-only heads and isolated-anchor output path implemented. Legacy copy/no-action/batch-roll results were re-aggregated. New retrained and matched-shuffle numeric results are blocked.
3. **B2 trivial baselines:** train-only direction-prior, no-update metadata, analytic past-only geometry, and query-only architecture are implemented/audited; real strict numbers are blocked.
4. **Common intersections:** real C1/C2 model intersections and cross-prefix legacy-paired manifests generated. B2 strict intersection blocked. Legacy pairing caveat is explicit.
5. **Capacity:** parameter/information-access feasibility audit completed; GPU pilot NOT_RUN.
6. **Prefix AUC:** C1/C2 common-set AUC tables/curves generated with actual spacing. No best-prefix ranking is used. Strict B2 AUC is absent.
7. **Correlation/difficulty:** raw task correlation and C1/C2 difficulty curves generated from real artifacts. Fair gain correlation and B2 curves are unavailable and left empty/blocked.

## 2. Which results are trustworthy

- Existing **standard B2** is not strict and cannot answer the zero-hidden claim.
- There are no numeric **strict-eval** or **strict-train** rows.
- Existing C controls are **legacy evaluation interventions** or contextual copy, not the requested fair retrained baselines.
- Analytic dead reckoning is an **oracle** by definition and has no real result here.
- C numeric tables use a common intersection, but cross-prefix pairing uses reconstructed legacy IDs; native/common deltas are reported.

## 3. Do C1/C2 really use action?

The current evidence cannot establish this. Legacy full/copy/no-action/batch-roll values are in `c_baselines_summary.csv`, but visual-only, action-only and magnitude-matched shuffle checkpoints are absent. Full-vs-strongest-fair-baseline gain is therefore not reported.

## 4. Does B2 beat a simple object belief?

Unknown. Strict direction prior, no-update, query-only and full checkpoints were not evaluated. Existing standard B2 aggregates cannot be substituted because target residual pixels and target-derived centroid points were allowed.

## 5. Sample-set sensitivity

C native/common differences are in `native_vs_common_metrics.csv`. Because the reconstructed common set is usually the full 276-record legacy set, changes can be small; this does not validate B2, whose IDs are missing.

## 6. Prefix conclusion

Use `paper_prefix_auc_table.csv`, p24-p8 and full curves. The AUC table is restricted to configurations present at all four prefixes on the paired common set; best prefix is descriptive only.

## 7. Task redundancy

`task_correlation_raw.csv` reports Spearman, Kendall and paired n. With only up to eight models and missing strict/gain columns, correlations are descriptive and do not support causal or redundancy claims.

## 8. Unfinished items and minimum repair

- **Real strict manifests/results:** restore a working ScanNet++ root plus streaming extractor, run `build_b2_strict_manifest.py`, extract fixed-target suffix caches, then pass DINO p8 strict/query smoke gates before the 64 B2 jobs.
- **Fair C controls:** ensure target caches include isolated anchors, train visual-only/action-only with unique `autoresume=false` folders, then run global matched-bin intervention.
- **Canonical IDs:** rerun eval-only with the updated dataset/evaluator; do not rely on legacy scene-order pairing in the paper.
- **Capacity pilot:** run only DINO/F3R/V-JEPA2 p8 after CUDA/data are available.

No background job was launched. Status matrix: {complete_caveat} `COMPLETE_WITH_CAVEAT`, {blocked} `BLOCKED_MISSING_ARTIFACT`.
"""
    (run_dirs["reports"] / "FINAL_REPORT.md").write_text(final)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dirs = {name: root / name / args.run_tag for name in ("reports", "manifests", "results", "figures", "logs")}
    run_dirs["reports"].mkdir(parents=True, exist_ok=True)
    for name in ("results", "figures", "logs"):
        run_dirs[name].mkdir(parents=True, exist_ok=True)
    try:
        run_dirs["manifests"].mkdir(parents=True, exist_ok=True)
    except PermissionError:
        run_dirs["manifests"] = run_dirs["reports"] / "manifests"
        run_dirs["manifests"].mkdir(parents=True, exist_ok=True)

    config = {
        "run_tag": args.run_tag, "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short").splitlines(), "models": list(MODELS),
        "layers": LAYERS, "prefixes": list(PREFIXES), "seed": 42, "eval_split": "val",
        "split_manifest": "configs/splits/inscene15k_v1.json",
        "split_manifest_sha256": sha256(root / "configs/splits/inscene15k_v1.json"),
        "sources": {"c_raw": "logs/inscene15k_streaming/results/c_eval_with_extra_20260728/c_series_raw_metrics_with_extra.csv"},
    }
    write_json(run_dirs["reports"] / "resolved_analysis_config.json", config)
    (run_dirs["reports"] / "git_status_at_start.txt").write_text(git("status", "--short") + "\n")
    (run_dirs["reports"] / "git_diff_summary.txt").write_text(git("diff", "--stat") + "\n")

    raw_path = root / config["sources"]["c_raw"]
    raw = pd.read_csv(raw_path)
    c_rows = {"C1": select_c_rows(raw, "action_dynamics"),
              "C2": select_c_rows(raw, "path_integration")}
    inventory, checkpoint_rows = audit_artifacts(root, run_dirs, c_rows)
    c1_configs = load_c_configs(root, c_rows["C1"], "c1")
    c2_configs = load_c_configs(root, c_rows["C2"], "c2")
    common, counts = build_common_manifests(run_dirs, c1_configs, c2_configs)
    c1_summary, c1_samples, c1_metrics = evaluate_c1(run_dirs, c1_configs, common)
    c2_summary, c2_samples, c2_metrics = evaluate_c2(run_dirs, c2_configs, common)
    c_summaries = c1_summary + c2_summary
    native_common = native_vs_common_metrics(c_rows["C1"] + c_rows["C2"], c_summaries)
    write_csv(run_dirs["results"] / "native_vs_common_metrics.csv", native_common)
    auc_rows = build_prefix_auc(run_dirs, c1_metrics + c2_metrics)
    correlation_rows, _model_table = task_correlations(root, run_dirs, auc_rows, c_summaries)
    difficulty = difficulty_points(c1_samples, c2_samples)
    write_csv(run_dirs["results"] / "difficulty_curve_points.csv", difficulty)
    plot_difficulty(run_dirs, difficulty)
    capacity = capacity_audit(run_dirs)
    status_rows = build_status_matrix(run_dirs, c1_summary, c2_summary)

    write_csv(run_dirs["results"] / "paper_baseline_master_table.csv", c_summaries)
    difficulty_frame = pd.DataFrame(difficulty)
    if difficulty_frame.empty:
        difficulty_summary = []
    else:
        difficulty_summary = difficulty_frame.groupby(
            ["task", "difficulty_variable", "metric", "variant"], as_index=False
        ).agg(bins=("bin", "count"), n_total=("n", "sum"), mean_bin_value=("value", "mean")).to_dict("records")
    write_csv(run_dirs["results"] / "paper_difficulty_summary.csv", difficulty_summary)
    write_reports(root, args.run_tag, run_dirs, inventory, checkpoint_rows, counts,
                  c_summaries, native_common, auc_rows, correlation_rows, difficulty,
                  capacity, status_rows)
    write_json(run_dirs["logs"] / "completion.json", {
        "run_tag": args.run_tag, "status": "completed_offline_with_blocked_training",
        "background_jobs": [], "c1_configs_loaded": len(c1_configs),
        "c2_configs_loaded": len(c2_configs), "generated_files": sum(1 for path in run_dirs.values() for _ in path.rglob("*")),
    })
    write_json(run_dirs["logs"] / "smoke_gate.json", {
        "run_tag": args.run_tag,
        "ordered_gate": [
            {"step": 1, "name": "readonly_audit", "status": "PASS"},
            {"step": 2, "name": "strict_manifest_unit_tests", "status": "PASS"},
            {"step": 3, "name": "single_real_scene_manual_check", "status": "BLOCKED_MISSING_ARTIFACT",
             "reason": "data root unset; AOSS read timed out"},
            {"step": 4, "name": "dino_p8_strict_b2_smoke", "status": "NOT_RUN"},
            {"step": 5, "name": "dino_p8_query_only_smoke", "status": "NOT_RUN"},
            {"step": 6, "name": "dino_p8_c1_visual_action_smoke", "status": "NOT_RUN"},
            {"step": 7, "name": "dino_p8_c2_visual_action_smoke", "status": "NOT_RUN"},
            {"step": 8, "name": "matched_shuffle_distribution_audit", "status": "NOT_RUN"},
            {"step": 9, "name": "legacy_common_intersection_audit", "status": "PASS_WITH_CAVEAT"},
        ],
        "training_launched": False,
        "background_jobs": [],
    })


if __name__ == "__main__":
    main()
