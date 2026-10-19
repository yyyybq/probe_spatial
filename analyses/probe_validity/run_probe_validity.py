#!/usr/bin/env python3
"""C1/B2 construct-validity analysis.

The default mode stays inside existing aggregate/sample artifacts. Explicit
Phase-2 replay modes load fixed historical checkpoints and feature tensors to
produce sample-level diagnostics. The paper-strengthening mode also retrains
four fixed C1 control probes, but no mode launches backbone extraction. Missing
evidence is reported rather than reconstructed or guessed.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import random
import statistics
import struct
import traceback
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


EXECUTION_SOURCE_ROOT = Path(__file__).resolve().parents[2]
# Frozen evidence snapshots execute their own copy of this runner while reading
# the immutable experiment inputs from the project tree.  The override keeps
# code provenance and data provenance separate and explicit.
ROOT = Path(os.environ.get("PROBE_VALIDITY_PROJECT_ROOT", EXECUTION_SOURCE_ROOT)).resolve()
OUT = ROOT / "analyses" / "probe_validity"

PROTECTED = [
    ROOT / "experiments" / "task_status_1096.csv",
    ROOT / "results" / "paper_ready_results.csv",
    ROOT / "results" / "atomic_probe_coverage.csv",
    *sorted((ROOT / "results" / "canonical").glob("*")),
    ROOT / "experiments" / "registry.csv",
    ROOT / "metadata" / "probe_taxonomy.csv",
]

C1_CANONICAL = ROOT / "results" / "canonical" / "c1_results.csv"
C3_CANONICAL = ROOT / "results" / "canonical" / "c3_results.csv"
C_RAW = ROOT / "logs" / "inscene15k_streaming" / "results" / "c_eval_with_extra_20260728" / "c_series_raw_metrics_with_extra.csv"
B2_DEFAULT8 = ROOT / "results" / "reconstructed" / "default8_b2_angle_candidate.csv"
B2_CANONICAL = ROOT / "results" / "canonical" / "b2_results.csv"
DEFAULT8_MODELS = {
    "aether",
    "cogvideox",
    "dino",
    "f3r",
    "qwen2_5_vl",
    "videomae",
    "vjepa2",
    "wan",
}
DEFAULT_LAYERS = {
    "aether": "1",
    "cogvideox": "20",
    "dino": "0",
    "f3r": "24",
    "qwen2_5_vl": "-1",
    "videomae": "23",
    "vjepa2": "23",
    "wan": "20",
}

TMP = OUT / "tmp"
PHASE2_SEED = 42
C1_SMOKE_CKPT = ROOT / (
    "logs/inscene15k_streaming/runs/c_retrain_safe_20260722_1435/"
    "c_retrain_safe_20260722_1435_action_dynamics_f3r_p12_layer24/"
    "VideoProbe3D/qjoe9etx/checkpoints/epoch=49-step=30600.ckpt"
)
C1_SMOKE_SUMMARY = ROOT / (
    "logs/inscene15k_streaming/runs/c_eval_20260724_1903/"
    "c_eval_20260724_1903_action_dynamics_f3r_p12_layer24/eval/val_summary.json"
)
C1_SMOKE_MANIFEST = ROOT / "manifests/c_retrain_20260722/inscene_c_h124_prefix12.jsonl"
C1_MANIFESTS = {
    12: C1_SMOKE_MANIFEST,
    16: ROOT / "manifests/c_retrain_20260722/inscene_c_h124_prefix16.jsonl",
}
B2_SMOKE_CKPT = ROOT / (
    "logs/inscene15k_streaming/runs/inscene15k_streaming_b2_f3r_p12_layer24/"
    "VideoProbe3D/dfiruw4y/checkpoints/epoch=49-step=122350.ckpt"
)
B2_SMOKE_SUMMARY = ROOT / (
    "logs/inscene15k_streaming/runs/inscene15k_streaming_b2_f3r_p12_layer24/"
    "eval/val_summary.json"
)
F3R_STREAM_P12 = "fj:s3://baiqiao/InsScene-15K/FEAT_STREAMING_BY_PREFIX/f3r_prefix_12_m0p35_r0p5"
F3R_TARGET_P12 = "fj:s3://baiqiao/InsScene-15K/FEAT_TARGET_BY_PREFIX/f3r_prefix_12_m0p35_r0p5"
SCANNET_REMOTE_ROOT = "fj:s3://baiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2"
SCANNET_LOCAL_ROOT = "/mnt/umm/users/yinbaiqiao/InsScene-15K/data"

# Phase-2 contracts are pinned to the checkpoint named by the authoritative
# historical evaluator summary, not to another checkpoint in the same run.
MODEL_SPECS = {
    "f3r": {
        "display": "F3R (geometry control)",
        "objective_family": "geometry_specialist_control",
        "layer": 24,
        "feat_postfix": "_l24",
        "channels": 1024,
        "b2_max_h": 18,
        "b2_max_w": 32,
        "c1_ckpt": C1_SMOKE_CKPT,
        "c1_summary": C1_SMOKE_SUMMARY,
        "b2_ckpt": B2_SMOKE_CKPT,
        "b2_summary": B2_SMOKE_SUMMARY,
    },
    "vjepa2": {
        "display": "V-JEPA2",
        "objective_family": "predictive",
        "layer": 23,
        "feat_postfix": "_layer23",
        "channels": 1024,
        "b2_max_h": 16,
        "b2_max_w": 16,
        "c1_ckpt": ROOT / (
            "logs/inscene15k_streaming/runs/c_retrain_safe_20260722_1435/"
            "c_retrain_safe_20260722_1435_action_dynamics_vjepa2_p12_layer23/"
            "VideoProbe3D/smrebnm7/checkpoints/epoch=49-step=30600.ckpt"
        ),
        "c1_summary": ROOT / (
            "logs/inscene15k_streaming/runs/c_eval_20260724_1903/"
            "c_eval_20260724_1903_action_dynamics_vjepa2_p12_layer23/eval/val_summary.json"
        ),
        "b2_ckpt": ROOT / (
            "logs/inscene15k_streaming/runs/inscene15k_streaming_b2_vjepa2_p12_layer23/"
            "VideoProbe3D/nltwtzwh/checkpoints/epoch=49-step=122350.ckpt"
        ),
        "b2_summary": ROOT / (
            "logs/inscene15k_streaming/runs/inscene15k_streaming_b2_vjepa2_p12_layer23/"
            "eval/val_summary.json"
        ),
        "c1_p16_ckpt": ROOT / (
            "logs/inscene15k_streaming/runs/inscene15k_streaming_c1_vjepa2_p16_layer23/"
            "VideoProbe3D/90rlclo2/checkpoints/epoch=49-step=30600.ckpt"
        ),
        "c1_p16_summary": ROOT / (
            "logs/inscene15k_streaming/runs/inscene15k_streaming_c1_vjepa2_p16_layer23/"
            "eval/val_summary.json"
        ),
        "b2_p16_ckpt": ROOT / (
            "logs/inscene15k_streaming/runs/inscene15k_streaming_b2_vjepa2_p16_layer23/"
            "VideoProbe3D/vkreuvdk/checkpoints/epoch=3-step=9788.ckpt"
        ),
        "b2_p16_summary": ROOT / (
            "logs/inscene15k_streaming/runs/inscene15k_streaming_b2_vjepa2_p16_layer23/"
            "eval/val_summary.json"
        ),
    },
    "videomae": {
        "display": "VideoMAE",
        "objective_family": "masked_reconstruction",
        "layer": 23,
        "feat_postfix": "_layer23",
        "channels": 1024,
        "b2_max_h": 30,
        "b2_max_w": 52,
        "c1_ckpt": ROOT / (
            "logs/inscene15k_streaming/runs/c_retrain_extra_videomae_qwen_localfeat_20260725_1826/"
            "c_retrain_extra_videomae_qwen_localfeat_20260725_1826_action_dynamics_videomae_p12_layer23/"
            "VideoProbe3D/svhgm7jr/checkpoints/epoch=49-step=61200.ckpt"
        ),
        "c1_summary": ROOT / (
            "logs/inscene15k_streaming/runs/c_eval_extra_videomae_qwen_20260728_0256/"
            "c_eval_extra_videomae_qwen_20260728_0256_action_dynamics_videomae_p12_layer23/"
            "eval/val_summary.json"
        ),
        "b2_ckpt": ROOT / (
            "logs/inscene15k_streaming/runs/inscene15k_streaming_b2_videomae_p12_layer23/"
            "VideoProbe3D/lmoal13a/checkpoints/epoch=49-step=122350.ckpt"
        ),
        "b2_summary": ROOT / (
            "logs/inscene15k_streaming/runs/inscene15k_streaming_b2_videomae_p12_layer23/"
            "eval/val_summary.json"
        ),
        "c1_p16_ckpt": ROOT / (
            "logs/inscene15k_streaming/runs/inscene15k_streaming_c1_videomae_p16_layer23/"
            "VideoProbe3D/vtroo5e4/checkpoints/epoch=49-step=30600.ckpt"
        ),
        "c1_p16_summary": ROOT / (
            "logs/inscene15k_streaming/runs/inscene15k_streaming_c1_videomae_p16_layer23/"
            "eval/val_summary.json"
        ),
        "b2_p16_ckpt": ROOT / (
            "logs/inscene15k_streaming/runs/inscene15k_streaming_b2_videomae_p16_layer23/"
            "VideoProbe3D/q4tx2f0s/checkpoints/epoch=4-step=12235.ckpt"
        ),
        "b2_p16_summary": ROOT / (
            "logs/inscene15k_streaming/runs/inscene15k_streaming_b2_videomae_p16_layer23/"
            "eval/val_summary.json"
        ),
    },
    "wan": {
        "display": "Wan",
        "objective_family": "video_generation",
        "layer": 20,
        "feat_postfix": "_t749_layer20",
        "channels": 1536,
        "b2_max_h": 36,
        "b2_max_w": 64,
        "c1_ckpt": ROOT / (
            "logs/inscene15k_streaming/runs/c_retrain_safe_20260722_1435/"
            "c_retrain_safe_20260722_1435_action_dynamics_wan_p12_layer20/"
            "VideoProbe3D/6zf0wf3f/checkpoints/epoch=49-step=30600.ckpt"
        ),
        "c1_summary": ROOT / (
            "logs/inscene15k_streaming/runs/c_eval_20260724_1903/"
            "c_eval_20260724_1903_action_dynamics_wan_p12_layer20/eval/val_summary.json"
        ),
        "b2_ckpt": ROOT / (
            "logs/inscene15k_streaming/runs/inscene15k_streaming_b2_wan_p12_layer20_heavy64_20260716_021635/"
            "VideoProbe3D/1wcapio9/checkpoints/epoch=49-step=122350.ckpt"
        ),
        "b2_summary": ROOT / (
            "logs/inscene15k_streaming/runs/inscene15k_streaming_b2_wan_p12_layer20_heavy64_20260716_021635/"
            "eval/val_summary.json"
        ),
        "c1_p16_ckpt": ROOT / (
            "logs/inscene15k_streaming/runs/inscene15k_streaming_c1_wan_p16_layer20_heavy64_20260716_021635/"
            "VideoProbe3D/rol8llse/checkpoints/epoch=3-step=2448.ckpt"
        ),
        "c1_p16_summary": ROOT / (
            "logs/inscene15k_streaming/runs/inscene15k_streaming_c1_wan_p16_layer20_heavy64_20260716_021635/"
            "eval/val_summary.json"
        ),
        "b2_p16_ckpt": ROOT / (
            "logs/inscene15k_streaming/runs/inscene15k_streaming_b2_wan_p16_layer20_heavy64_20260716_021635/"
            "VideoProbe3D/zvu7awcy/checkpoints/epoch=0-step=2447.ckpt"
        ),
        "b2_p16_summary": ROOT / (
            "logs/inscene15k_streaming/runs/inscene15k_streaming_b2_wan_p16_layer20_heavy64_20260716_021635/"
            "eval/val_summary.json"
        ),
    },
    "cogvideox": {
        "display": "CogVideoX",
        "objective_family": "video_generation",
        "layer": 20,
        "feat_postfix": "_t749_layer20",
        "channels": 3072,
        "b2_max_h": 36,
        "b2_max_w": 64,
        "c1_ckpt": ROOT / (
            "logs/inscene15k_streaming/runs/c_retrain_safe_20260722_1435/"
            "c_retrain_safe_20260722_1435_action_dynamics_cogvideox_p12_layer20/"
            "VideoProbe3D/6kxvs5of/checkpoints/epoch=49-step=30600.ckpt"
        ),
        "c1_summary": ROOT / (
            "logs/inscene15k_streaming/runs/c_eval_20260724_1903/"
            "c_eval_20260724_1903_action_dynamics_cogvideox_p12_layer20/eval/val_summary.json"
        ),
        "b2_ckpt": ROOT / (
            "logs/inscene15k_streaming/runs/inscene15k_streaming_b2_cogvideox_p12_layer20_heavy64_20260716_021635/"
            "VideoProbe3D/47csm4l3/checkpoints/epoch=49-step=122350.ckpt"
        ),
        "b2_summary": ROOT / (
            "logs/inscene15k_streaming/runs/inscene15k_streaming_b2_cogvideox_p12_layer20_heavy64_20260716_021635/"
            "eval/val_summary.json"
        ),
        "c1_p16_ckpt": ROOT / (
            "logs/inscene15k_streaming/runs/inscene15k_streaming_c1_cogvideox_p16_layer20_heavy64_20260716_021635/"
            "VideoProbe3D/j6x1e35n/checkpoints/epoch=0-step=612.ckpt"
        ),
        "c1_p16_summary": ROOT / (
            "logs/inscene15k_streaming/runs/inscene15k_streaming_c1_cogvideox_p16_layer20_heavy64_20260716_021635/"
            "eval/val_summary.json"
        ),
        "b2_p16_ckpt": ROOT / (
            "logs/inscene15k_streaming/runs/inscene15k_streaming_b2_cogvideox_p16_layer20_heavy64_20260716_021635/"
            "VideoProbe3D/wlqpdq8j/checkpoints/epoch=49-step=122350.ckpt"
        ),
        "b2_p16_summary": ROOT / (
            "logs/inscene15k_streaming/runs/inscene15k_streaming_b2_cogvideox_p16_layer20_heavy64_20260716_021635/"
            "eval/val_summary.json"
        ),
    },
}
CORE_MODELS = ("vjepa2", "videomae", "wan", "cogvideox")


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def fval(row: Dict[str, str], key: str) -> Optional[float]:
    val = row.get(key, "")
    if val is None or val == "":
        return None
    try:
        out = float(val)
    except ValueError:
        return None
    if math.isnan(out):
        return None
    return out


def fmt(x: Optional[float], digits: int = 6) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{float(x):.{digits}g}"


def mean(xs: Iterable[float]) -> Optional[float]:
    vals = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(vals) / len(vals) if vals else None


def median(xs: Iterable[float]) -> Optional[float]:
    vals = [x for x in xs if x is not None and not math.isnan(x)]
    return statistics.median(vals) if vals else None


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0 or vy <= 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def snapshot(paths: Sequence[Path]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for path in paths:
        st = path.stat()
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        out[rel(path)] = {"size": st.st_size, "mtime": int(st.st_mtime), "sha256": h.hexdigest()}
    return out


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path_text: str) -> Dict[str, object]:
    if not path_text:
        return {}
    p = ROOT / path_text if not Path(path_text).is_absolute() else Path(path_text)
    if not p.exists():
        return {}
    with p.open() as f:
        return json.load(f)


def c3_pivot() -> Dict[Tuple[str, str, str], Dict[str, float]]:
    out: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for row in read_csv(C3_CANONICAL):
        if row.get("horizon") != "all":
            continue
        key = (row.get("model", ""), row.get("layer", ""), row.get("prefix", ""))
        out.setdefault(key, {})[row.get("metric_name", "")] = fval(row, "metric_value")
    return out


def raw_c1_index() -> Dict[Tuple[str, str, str], Dict[str, str]]:
    rows = read_csv(C_RAW) if C_RAW.exists() else []
    out = {}
    for row in rows:
        if row.get("probe_type") != "action_dynamics":
            continue
        key = (row.get("vfm_name", ""), row.get("layer", ""), row.get("prefix", ""))
        out[key] = row
    return out


def label_c1(true_r1: float, best_control: float, copy_r1: float, gain_cos: Optional[float],
             c3_margin: Optional[float], c3_validity: Optional[float]) -> str:
    delta = true_r1 - best_control
    action_cos_ok = gain_cos is not None and gain_cos > 0.01
    c3_ok = c3_margin is not None and c3_margin > 0 and c3_validity is not None and c3_validity >= 0.5
    if delta > 0.05 and action_cos_ok and c3_ok:
        return "ACTION_SPECIFIC_STRONG"
    if delta > 0.02 or (action_cos_ok and true_r1 > max(copy_r1, 0.05)):
        return "ACTION_SPECIFIC_PARTIAL"
    if true_r1 >= 0.10 and delta <= 0:
        return "PREDICTABLE_BUT_ACTION_WEAK"
    if copy_r1 >= true_r1 and copy_r1 >= 0.05:
        return "SMOOTHNESS_OR_RETRIEVAL_DOMINATED"
    return "INSUFFICIENT_EVIDENCE"


def make_c1() -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    c3 = c3_pivot()
    raw = raw_c1_index()
    rows: List[Dict[str, object]] = []
    paired_rows = []
    high_rows = []
    all_c1 = read_csv(C1_CANONICAL)
    excluded_models = sorted({r.get("model", "") for r in all_c1} - DEFAULT8_MODELS)
    excluded_layer_rows = [
        r for r in all_c1
        if r.get("model", "") in DEFAULT8_MODELS
        and r.get("layer", "") != DEFAULT_LAYERS.get(r.get("model", ""))
    ]

    for row in all_c1:
        if row.get("model", "") not in DEFAULT8_MODELS:
            continue
        if row.get("layer", "") != DEFAULT_LAYERS.get(row.get("model", "")):
            continue
        key = (row.get("model", ""), row.get("layer", ""), row.get("prefix", ""))
        raw_row = raw.get(key, {})
        true_r1 = fval(row, "global_R@1") or 0.0
        copy_r1 = fval(row, "last_observation_R@1") or 0.0
        no_r1 = fval(row, "no_action_R@1") or 0.0
        shuf_r1 = fval(row, "shuffled_action_R@1") or 0.0
        best_control = max(copy_r1, no_r1, shuf_r1)
        delta_best = true_r1 - best_control
        denom = 1.0 - best_control
        norm_gain = delta_best / denom if denom > 1e-9 else None
        gain_cos = fval(row, "action_gain_cos")
        c3_metrics = c3.get(key, {})
        c3_validity = c3_metrics.get("intervention_validity")
        c3_margin = c3_metrics.get("intervention_margin")
        js = load_json(raw_row.get("eval_summary_path", ""))
        ci = ((js.get("bootstrap_95ci") or {}).get("action_gain_cos") or {}) if js else {}
        label = label_c1(true_r1, best_control, copy_r1, gain_cos, c3_margin, c3_validity)
        if true_r1 >= 0.10:
            high_rows.append((true_r1, copy_r1, no_r1, shuf_r1, best_control, delta_best))
        if c3_margin is not None:
            paired_rows.append((delta_best, c3_margin, c3_validity))
        rows.append({
            "experiment_id": row.get("experiment_id"),
            "model": row.get("model"),
            "layer": row.get("layer"),
            "prefix": row.get("prefix"),
            "analysis_type": "existing_fixed_predictor_controls",
            "subset": "ALL_VAL",
            "candidate_pool": "GLOBAL_ALL_VAL_TARGETS",
            "action_condition": "true_vs_zero_batchroll_copy",
            "n": raw_row.get("n", ""),
            "r1": fmt(true_r1),
            "mrr_or_rank": fmt(fval(raw_row, "mean_rank")),
            "target_cosine": fmt(fval(raw_row, "mean_cos")),
            "copy_r1": fmt(copy_r1),
            "no_action_r1": fmt(no_r1),
            "shuffled_action_r1": fmt(shuf_r1),
            "delta_vs_copy": fmt(true_r1 - copy_r1),
            "delta_vs_no_action": fmt(true_r1 - no_r1),
            "delta_vs_shuffle": fmt(true_r1 - shuf_r1),
            "delta_vs_best_control": fmt(delta_best),
            "normalized_action_gain": fmt(norm_gain),
            "fixed_predictor_action_margin": fmt(gain_cos),
            "ci_low": fmt(ci.get("low")),
            "ci_high": fmt(ci.get("high")),
            "c3_intervention_validity": fmt(c3_validity),
            "c3_intervention_margin": fmt(c3_margin),
            "artifact_source": row.get("raw_artifact_path") or raw_row.get("eval_summary_path"),
            "evidence_level": "SUPPORTED_AGGREGATE_ONLY",
            "notes": f"{label}; no_action and shuffled_action are fixed-predictor inference perturbations in vidfm3d/eval_diag.py; copy is last prefix observation.",
        })

    for analysis_type, note in [
        ("motion_smoothness_audit", "Needs per-sample target/last-observation features and poses; only aggregate val_summary artifacts are present."),
        ("candidate_pool_stress_test", "Needs per-sample predictions/targets/scene trajectory metadata; raw summaries do not store ranks or candidate identities."),
        ("matched_magnitude_inverse_action_intervention", "Needs checkpoint plus torch/hydra runtime and per-batch features; current environment lacks torch and existing summaries only include zero-action and batch-roll shuffle."),
    ]:
        rows.append({
            "experiment_id": "C1_PROTOCOL_LEVEL",
            "model": "ALL_INCLUDED",
            "layer": "default",
            "prefix": "8/12/16/24",
            "analysis_type": analysis_type,
            "subset": "ALL_VAL",
            "candidate_pool": "",
            "action_condition": "",
            "artifact_source": "existing aggregate summaries audited",
            "evidence_level": "NOT_TESTABLE_MISSING_ARTIFACT",
            "notes": note,
        })

    c3_delta = [x[0] for x in paired_rows]
    c3_margin = [x[1] for x in paired_rows]
    c3_validity = [x[2] for x in paired_rows]
    summary = {
        "n_rows": len([r for r in rows if r["analysis_type"] == "existing_fixed_predictor_controls"]),
        "models": sorted({r["model"] for r in rows if r["analysis_type"] == "existing_fixed_predictor_controls"}),
        "excluded_models": excluded_models,
        "excluded_layer_rows": len(excluded_layer_rows),
        "positive_best_control_rows": sum(1 for r in rows if r["analysis_type"] == "existing_fixed_predictor_controls" and float(r["delta_vs_best_control"]) > 0),
        "strong_or_partial_rows": sum(1 for r in rows if "ACTION_SPECIFIC_STRONG" in r["notes"] or "ACTION_SPECIFIC_PARTIAL" in r["notes"]),
        "median_true_r1": median(float(r["r1"]) for r in rows if r["analysis_type"] == "existing_fixed_predictor_controls"),
        "median_delta_best": median(float(r["delta_vs_best_control"]) for r in rows if r["analysis_type"] == "existing_fixed_predictor_controls"),
        "median_copy_r1": median(float(r["copy_r1"]) for r in rows if r["analysis_type"] == "existing_fixed_predictor_controls"),
        "high_n": len(high_rows),
        "high_copy_explained": mean(min(copy, true) / true for true, copy, *_ in high_rows if true > 0),
        "high_best_control_explained": mean(min(best, true) / true for true, _copy, _no, _shuf, best, _delta in high_rows if true > 0),
        "corr_delta_c3_margin": pearson(c3_delta, c3_margin),
        "corr_delta_c3_validity": pearson(c3_delta, c3_validity),
    }
    return rows, summary


def b2_default_rows() -> List[Dict[str, str]]:
    if not B2_DEFAULT8.exists():
        return []
    return [r for r in read_csv(B2_DEFAULT8) if r.get("status", "").startswith("COMPLETE")]


def make_b2() -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    full_rows = b2_default_rows()
    for row in full_rows:
        js = load_json(row.get("summary_path", ""))
        ci = ((js.get("bootstrap_95ci") or {}).get("mean_ang_err_deg") or {}) if js else {}
        rows.append({
            "experiment_id": row.get("experiment_id"),
            "sample_subset": "ALL_VALID_EVAL",
            "model": row.get("model"),
            "layer": row.get("layer"),
            "prefix": row.get("prefix"),
            "analysis_type": "full_history_query_current_probe",
            "baseline_or_readout": "current_transformer_patch_attention",
            "perturbation": "none",
            "seed": 42,
            "n": row.get("n"),
            "mean_ang_err_deg": fmt(fval(row, "mean_ang_err_deg")),
            "median_ang_err_deg": "",
            "p25_ang_err_deg": "",
            "p75_ang_err_deg": "",
            "top1": fmt(fval(row, "top1")),
            "top3": fmt(fval(row, "top3")),
            "delta_error_vs_full": "0",
            "delta_error_vs_best_prior": "",
            "ci_low": fmt(ci.get("low")),
            "ci_high": fmt(ci.get("high")),
            "query_unique_rate": "",
            "target_seen_rate": "",
            "ambiguous_rate": "",
            "never_seen_rate": "",
            "artifact_source": row.get("summary_path"),
            "evidence_level": "SUPPORTED_AGGREGATE_ONLY",
            "notes": "Current B2 aggregate only; sample_id/category/instance/visibility labels are not stored in val_summary.json.",
        })

    chance_top1 = 1.0 / (16 * 8)
    chance_top3 = 3.0 / (16 * 8)
    rows.append({
        "experiment_id": "B2_THEORETICAL_REFERENCE",
        "sample_subset": "ALL_VALID_EVAL",
        "model": "ALL_INCLUDED",
        "layer": "default",
        "prefix": "8/12/16/24",
        "analysis_type": "theoretical_bin_chance_reference",
        "baseline_or_readout": "uniform_128_bin_chance_not_train_fit_prior",
        "perturbation": "none",
        "seed": "",
        "n": "",
        "mean_ang_err_deg": "",
        "top1": fmt(chance_top1),
        "top3": fmt(chance_top3),
        "artifact_source": "B2 uses 16 azimuth x 8 elevation bins",
        "evidence_level": "SUPPORTED_AGGREGATE_ONLY",
        "notes": "This is only a bin-chance reference, not a valid global/category/scene prior fitted on train labels.",
    })
    for baseline in [
        "global_train_angle_prior",
        "category_or_query_only_prior",
        "scene_only_prior",
        "query_plus_scene_prior",
        "trajectory_only_prior",
        "current_frame_only",
        "history_without_query",
        "query_perturbation",
        "history_perturbation",
        "identifiability_visibility_audit",
        "three_seed_no_mask_readout_audit",
    ]:
        rows.append({
            "experiment_id": "B2_PROTOCOL_LEVEL",
            "sample_subset": "ALL_VALID_EVAL",
            "model": "ALL_INCLUDED",
            "layer": "default",
            "prefix": "8/12/16/24",
            "analysis_type": baseline,
            "baseline_or_readout": baseline,
            "perturbation": baseline if "perturbation" in baseline else "none",
            "artifact_source": "existing summaries and code audit",
            "evidence_level": "NOT_TESTABLE_MISSING_ARTIFACT",
            "notes": "Requires train/eval sample-level labels and/or torch checkpoint evaluation. Existing B2 summaries lack sample_id, category, instance id, visibility counts, polar labels, and predictions; current environment also lacks torch/hydra.",
        })

    included = sorted({r.get("model", "") for r in full_rows})
    means = [fval(r, "mean_ang_err_deg") for r in full_rows]
    summary = {
        "n_rows": len(full_rows),
        "models": included,
        "median_mean_ang_err": median(x for x in means if x is not None),
        "best_full": min(((fval(r, "mean_ang_err_deg") or 999, r) for r in full_rows), key=lambda x: x[0])[1] if full_rows else {},
        "chance_top1": chance_top1,
        "chance_top3": chance_top3,
    }
    return rows, summary


FONT = {
    " ": ["000", "000", "000", "000", "000", "000", "000"],
    "0": ["111", "101", "101", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "010", "010", "111"],
    "2": ["111", "001", "001", "111", "100", "100", "111"],
    "3": ["111", "001", "001", "111", "001", "001", "111"],
    "4": ["101", "101", "101", "111", "001", "001", "001"],
    "5": ["111", "100", "100", "111", "001", "001", "111"],
    "6": ["111", "100", "100", "111", "101", "101", "111"],
    "7": ["111", "001", "001", "010", "010", "100", "100"],
    "8": ["111", "101", "101", "111", "101", "101", "111"],
    "9": ["111", "101", "101", "111", "001", "001", "111"],
    "A": ["010", "101", "101", "111", "101", "101", "101"],
    "B": ["110", "101", "101", "110", "101", "101", "110"],
    "C": ["111", "100", "100", "100", "100", "100", "111"],
    "D": ["110", "101", "101", "101", "101", "101", "110"],
    "E": ["111", "100", "100", "110", "100", "100", "111"],
    "F": ["111", "100", "100", "110", "100", "100", "100"],
    "G": ["111", "100", "100", "101", "101", "101", "111"],
    "H": ["101", "101", "101", "111", "101", "101", "101"],
    "I": ["111", "010", "010", "010", "010", "010", "111"],
    "J": ["001", "001", "001", "001", "101", "101", "111"],
    "K": ["101", "101", "110", "100", "110", "101", "101"],
    "L": ["100", "100", "100", "100", "100", "100", "111"],
    "M": ["101", "111", "111", "101", "101", "101", "101"],
    "N": ["101", "111", "111", "111", "101", "101", "101"],
    "O": ["111", "101", "101", "101", "101", "101", "111"],
    "P": ["111", "101", "101", "111", "100", "100", "100"],
    "Q": ["111", "101", "101", "101", "111", "001", "001"],
    "R": ["110", "101", "101", "110", "110", "101", "101"],
    "S": ["111", "100", "100", "111", "001", "001", "111"],
    "T": ["111", "010", "010", "010", "010", "010", "010"],
    "U": ["101", "101", "101", "101", "101", "101", "111"],
    "V": ["101", "101", "101", "101", "101", "101", "010"],
    "W": ["101", "101", "101", "101", "111", "111", "101"],
    "X": ["101", "101", "101", "010", "101", "101", "101"],
    "Y": ["101", "101", "101", "010", "010", "010", "010"],
    "Z": ["111", "001", "001", "010", "100", "100", "111"],
    ".": ["000", "000", "000", "000", "000", "110", "110"],
    "-": ["000", "000", "000", "111", "000", "000", "000"],
    ":": ["000", "110", "110", "000", "110", "110", "000"],
    "/": ["001", "001", "001", "010", "100", "100", "100"],
    "+": ["000", "010", "010", "111", "010", "010", "000"],
}


class Canvas:
    def __init__(self, w: int = 1000, h: int = 700) -> None:
        self.w = w
        self.h = h
        self.pix = bytearray([255, 255, 255] * w * h)

    def set(self, x: int, y: int, color: Tuple[int, int, int]) -> None:
        if 0 <= x < self.w and 0 <= y < self.h:
            i = (y * self.w + x) * 3
            self.pix[i:i+3] = bytes(color)

    def line(self, x0: int, y0: int, x1: int, y1: int, color=(0, 0, 0)) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            for xx in range(x0 - 1, x0 + 2):
                for yy in range(y0 - 1, y0 + 2):
                    self.set(xx, yy, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def rect(self, x0: int, y0: int, x1: int, y1: int, color, fill=True) -> None:
        if fill:
            for y in range(min(y0, y1), max(y0, y1) + 1):
                for x in range(min(x0, x1), max(x0, x1) + 1):
                    self.set(x, y, color)
        else:
            self.line(x0, y0, x1, y0, color)
            self.line(x1, y0, x1, y1, color)
            self.line(x1, y1, x0, y1, color)
            self.line(x0, y1, x0, y0, color)

    def circle(self, cx: int, cy: int, r: int, color) -> None:
        for y in range(cy - r, cy + r + 1):
            for x in range(cx - r, cx + r + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    self.set(x, y, color)

    def text(self, x: int, y: int, text: str, color=(0, 0, 0), scale: int = 2) -> None:
        cx = x
        for ch in text.upper():
            pat = FONT.get(ch, FONT.get(" ", []))
            for yy, row in enumerate(pat):
                for xx, val in enumerate(row):
                    if val == "1":
                        self.rect(cx + xx * scale, y + yy * scale,
                                  cx + (xx + 1) * scale - 1, y + (yy + 1) * scale - 1,
                                  color, fill=True)
            cx += 4 * scale

    def save_png(self, path: Path) -> None:
        raw = b"".join(b"\x00" + bytes(self.pix[y*self.w*3:(y+1)*self.w*3]) for y in range(self.h))
        def chunk(tag: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", self.w, self.h, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress(raw, 9))
        png += chunk(b"IEND", b"")
        path.write_bytes(png)


def make_c1_plot(rows: List[Dict[str, object]]) -> None:
    data = [r for r in rows if r.get("analysis_type") == "existing_fixed_predictor_controls"]
    colors = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (255, 127, 14),
              (148, 103, 189), (140, 86, 75), (23, 190, 207), (127, 127, 127)]
    models = sorted({str(r["model"]) for r in data})
    c = Canvas()
    c.text(50, 25, "C1 TRUE R1 VS ACTION GAIN OVER BEST CONTROL", scale=3)
    left, right, top, bottom = 90, 930, 80, 610
    c.line(left, bottom, right, bottom)
    c.line(left, bottom, left, top)
    c.text(410, 640, "TRUE ACTION R1", scale=2)
    c.text(15, 90, "GAIN", scale=2)
    for i in range(6):
        x = left + int((right-left) * i / 5)
        c.line(x, bottom, x, bottom + 8)
        c.text(x - 12, bottom + 15, f"{i/10:.1f}", scale=2)
    for val in [-0.2, -0.1, 0.0, 0.1, 0.2]:
        y = bottom - int((val + 0.25) / 0.55 * (bottom-top))
        c.line(left - 8, y, left, y)
        c.text(35, y - 7, f"{val:.1f}", scale=2)
        if abs(val) < 1e-9:
            c.line(left, y, right, y, color=(180, 180, 180))
    for r in data:
        xval = float(r["r1"])
        yval = float(r["delta_vs_best_control"])
        x = left + int(max(0, min(0.55, xval)) / 0.55 * (right-left))
        y = bottom - int(max(-0.25, min(0.30, yval)) / 0.55 * (bottom-top) + 0.25 / 0.55 * (bottom-top))
        idx = models.index(str(r["model"])) % len(colors)
        c.circle(x, y, 7, colors[idx])
    lx = 710
    for i, m in enumerate(models):
        y = 100 + i * 26
        c.circle(lx, y + 5, 6, colors[i % len(colors)])
        c.text(lx + 16, y, m[:12], scale=2)
    c.save_png(OUT / "c1_validity.png")


def make_b2_plot(rows: List[Dict[str, object]]) -> None:
    data = [r for r in rows if r.get("analysis_type") == "full_history_query_current_probe"]
    by_model: Dict[str, List[float]] = {}
    for r in data:
        by_model.setdefault(str(r["model"]), []).append(float(r["mean_ang_err_deg"]))
    vals = [(m, mean(vs) or 0.0) for m, vs in sorted(by_model.items())]
    c = Canvas()
    c.text(50, 25, "B2 FULL HISTORY ANGULAR ERROR AND MISSING CONTROLS", scale=3)
    left, right, top, bottom = 120, 920, 90, 570
    c.line(left, bottom, right, bottom)
    c.line(left, bottom, left, top)
    c.text(360, 640, "MODEL MEAN OVER DEFAULT PREFIXES", scale=2)
    c.text(20, 90, "DEG", scale=2)
    for val in [40, 60, 80, 100]:
        y = bottom - int((val - 30) / 80 * (bottom-top))
        c.line(left - 8, y, left, y)
        c.text(70, y - 7, str(val), scale=2)
        c.line(left, y, right, y, color=(220, 220, 220))
    bar_w = max(20, int((right-left) / max(1, len(vals) * 2)))
    for i, (m, val) in enumerate(vals):
        x = left + int((i + 0.5) * (right-left) / max(1, len(vals)))
        y = bottom - int((val - 30) / 80 * (bottom-top))
        c.rect(x - bar_w, y, x + bar_w, bottom, (31, 119, 180), fill=True)
        c.text(x - bar_w, bottom + 12, m[:8], scale=2)
        c.text(x - bar_w, y - 20, f"{val:.1f}", scale=2)
    c.text(150, 595, "PRIORS PERTURBATIONS IDENTIFIABILITY: NOT TESTABLE FROM STORED SUMMARIES", color=(160, 0, 0), scale=2)
    c.save_png(OUT / "b2_validity.png")


def report(c1_rows: List[Dict[str, object]], c1s: Dict[str, object],
           b2_rows: List[Dict[str, object]], b2s: Dict[str, object],
           before: Dict[str, Dict[str, object]], after: Dict[str, Dict[str, object]]) -> None:
    c1_models = ", ".join(c1s["models"])
    c1_excluded = ", ".join(c1s.get("excluded_models") or []) or "none"
    b2_models = ", ".join(b2s["models"])
    best_b2 = b2s.get("best_full") or {}
    protected_ok = before == after
    c1_verdict = "KEEP_BUT_REINTERPRET_AS_PREDICTABILITY"
    if (c1s.get("positive_best_control_rows") or 0) <= (c1s.get("n_rows") or 1) / 3:
        c1_verdict = "KEEP_WITH_HARD_NEGATIVE_PRIMARY_METRIC"
    b2_verdict = "KEEP_WITH_PRIOR_CONTROLS / COLLECT_MISSING_METADATA"
    hash_lines = []
    for path, meta in before.items():
        after_meta = after[path]
        status = "OK" if meta == after_meta else "CHANGED"
        hash_lines.append(
            f"- `{path}` size={meta['size']} mtime={meta['mtime']} sha256={meta['sha256']} after_sha256={after_meta['sha256']} {status}"
        )
    hash_block = "\n".join(hash_lines)

    text = f"""# Executive Summary

1. C1 high R@1 is heavily explained by copy/control shortcuts in the current stored evidence: among rows with true R@1 >= 0.10 (n={c1s['high_n']}), the copy baseline explains about {fmt(c1s['high_copy_explained'], 3)} of the true score and the best control explains about {fmt(c1s['high_best_control_explained'], 3)}.
2. The existing C1 no-action and shuffled-action baselines are fixed-predictor interventions, not separately trained models. The median action gain over the best control is {fmt(c1s['median_delta_best'], 4)} R@1; {c1s['positive_best_control_rows']}/{c1s['n_rows']} rows beat the best of copy/no-action/shuffle.
3. B2 identifiable sample rate cannot be empirically recovered from stored eval summaries. Code shows B2 intends hidden-object targets with instance IDs and masked-pooled query features, but summaries drop sample_id/category/instance/visibility metadata.
4. B2 full-history angular error is available only as aggregate evidence. No valid train-fitted query/category/scene/position prior can be computed from the stored artifacts.
5. Current B2 evidence is therefore a dataset/artifact metadata problem first, not enough to decide representation vs readout. The implemented readout is already query-conditioned patch attention, not simple global mean pooling.
6. C1 verdict: `{c1_verdict}`. B2 verdict: `{b2_verdict}`.

# Exact Probe Definitions

## C1

Code sources: `protocols/C_PROBES.md`, `vidfm3d/eval_diag.py`, `vidfm3d/models/components/probe_action_dynamics.py`, `vidfm3d/data/components/inscene15k_dataset.py`, and `vidfm3d/utils/spatial_diag.py`.

C1 input is streaming prefix `input_feat` with shape `(B,T,H,W,C)`. The action is a 9D relative pose: 6D rotation plus 3D translation from the prefix tail to target frame. The target is the spatial mean of an isolated target-frame feature. The predictor is a 2-layer transformer head with a learned query token, an action token, mean-pooled frame tokens, and cosine/MSE training loss. Retrieval uses cosine similarity against all validation targets in the summary pool; `global_R@1` is the fraction whose matching target ranks first.

No-action and shuffled-action controls in `eval_diag.py` are fixed-predictor inference perturbations: `probe(input_feat, zeros_like(action))` and `probe(input_feat, torch.roll(action, 1, batch))`. Copy is the pooled last prefix observation and uses no predictor.

## B2

Code sources: `vidfm3d/models/components/probe_ego_belief_v2.py`, `vidfm3d/models/probe_ext_module.py`, `vidfm3d/data/components/inscene15k_dataset.py`, `vidfm3d/utils/spatial_diag.py`, and `eval_b2.py`.

B2 query is a 1D masked-pooled object appearance feature from the prefix frame with the most target-object pixels. The batch stores `hidden_obj_id`, `hidden_obj_mask`, `belief_query_frame`, and `hidden_obj_polar`, but current eval summaries do not preserve those fields. GT angle is azimuth/elevation/log-distance in the last camera frame: azimuth in `[-pi,pi]`, elevation in `[-pi/2,pi/2]`, OpenCV-style +Z forward. Metrics use 16 azimuth x 8 elevation bins, top1/top3 over the joint 128-bin logits, and spherical angular distance from predicted bin center to GT direction.

The current default B2 head is not global mean pooling: it projects query and patch tokens, adds frame/spatial embeddings, runs transformer self-attention, and reads the query-initialized belief token. GT spatial masks are used only outside the head to create the query feature and target metadata, not passed as a spatial readout mask.

# Data and Models Used

C1 uses complete canonical rows for default8/default-layer models/prefixes: {c1_models}. C3 alignment uses matching canonical C3 rows. B2 uses recovered default8 per-run summaries for: {b2_models}. I did not wait for or read active evaluation campaign outputs.

Excluded/limited evidence:

- C1 models outside default8 were excluded from this validity pass: {c1_excluded}. Reason: this round is not a model-coverage expansion, and B2 default8 comparability was prioritized.
- C1 non-default layer sweep rows excluded: {c1s.get('excluded_layer_rows', 0)}. Reason: this pass uses default-layer model artifacts rather than layer selection.
- C1 motion bins, hard-negative candidate pools, matched-magnitude inverse actions: not testable from aggregate summaries and this environment lacks torch/hydra for checkpoint replay.
- B2 priors, identifiability rates, query/history perturbations: not testable because summaries lack train/eval sample labels, per-sample GT angles, instance/category IDs, and predictions.
- B2 readout retraining: not run because torch/hydra are unavailable and this would exceed offline artifact analysis.

# C1 Findings

Existing baselines show that absolute R@1 is not action evidence by itself. Median true R@1 is {fmt(c1s['median_true_r1'], 4)}, median copy R@1 is {fmt(c1s['median_copy_r1'], 4)}, and median gain over the best control is {fmt(c1s['median_delta_best'], 4)}. Only {c1s['positive_best_control_rows']}/{c1s['n_rows']} rows beat the best shortcut/control in R@1.

Fixed-predictor intervention evidence exists for zero-action and batch-rolled shuffled action. It supports weak-to-partial action use in some rows, but not a broad action-specific interpretation. Correlation between C1 gain over best control and C3 intervention margin is {fmt(c1s['corr_delta_c3_margin'], 4)}; correlation with C3 intervention validity is {fmt(c1s['corr_delta_c3_validity'], 4)}.

Motion/smoothness and candidate-pool stress tests remain `NOT_TESTABLE_MISSING_ARTIFACT` in this pass because per-sample predictions, target ranks, scene/trajectory IDs, and pose displacement bins are not stored in the audited outputs.

# B2 Findings

Full-history B2 aggregate angular errors are available. The best default8 aggregate row is `{best_b2.get('experiment_id','')}` with mean angular error {fmt(fval(best_b2, 'mean_ang_err_deg'), 4)} deg, top1 {fmt(fval(best_b2, 'top1'), 4)}, top3 {fmt(fval(best_b2, 'top3'), 4)}, n={best_b2.get('n','')}. Theoretical uniform bin chance is top1={fmt(b2s['chance_top1'], 4)} and top3={fmt(b2s['chance_top3'], 4)}, but this is not a train-fitted prior baseline.

Identifiability, prior baselines, query perturbation, history perturbation, and controlled subsets cannot be quantified from the current stored B2 summaries. The most important missing fields are sample_id, scene_id, trajectory_id, query category, instance/category multiplicity, target visibility counts, per-sample GT polar angle, and prediction logits/bin.

# Validity Verdict

C1: `{c1_verdict}` with evidence level `SUPPORTED_AGGREGATE_ONLY`. The current scores measure future-feature predictability under smoothness/retrieval shortcuts more strongly than clean action reasoning. Keep C1 only with hard-negative/control-normalized primary metrics and sample-level replay artifacts.

B2: `{b2_verdict}` with evidence level `NOT_TESTABLE_MISSING_ARTIFACT` for construct validity. Current aggregate angular error is not enough to distinguish object-specific memory from query/category/scene/layout/position priors.

# Recommended Protocol Changes

- Store per-sample C1 records: sample_id, scene/trajectory, target rank, prediction/control similarities, action, pose displacement, feature drift.
- Make C1 hard-negative retrieval and gain over best control the primary reporting metric.
- Store per-sample B2 records: query/target instance IDs, category, visibility counts, GT polar, predicted bin/logits, scene/trajectory.
- Report B2 only on identifiable subsets and always with train-fitted category/query/scene/position priors.
- Keep the current B2 transformer readout audit separate from representation claims unless no-mask readout training is actually run.

# Reproducibility

Command:

```bash
python3 analyses/probe_validity/run_probe_validity.py
```

Source paths:

- `results/canonical/c1_results.csv`
- `results/canonical/c3_results.csv`
- `logs/inscene15k_streaming/results/c_eval_with_extra_20260728/c_series_raw_metrics_with_extra.csv`
- `results/reconstructed/default8_b2_angle_candidate.csv`
- `vidfm3d/eval_diag.py`
- `eval_b2.py`
- `vidfm3d/models/components/probe_action_dynamics.py`
- `vidfm3d/models/components/probe_ego_belief_v2.py`
- `vidfm3d/data/components/inscene15k_dataset.py`
- `vidfm3d/utils/spatial_diag.py`

Files changed/generated:

- `analyses/probe_validity/run_probe_validity.py`
- `analyses/probe_validity/c1_validity.csv`
- `analyses/probe_validity/b2_validity.csv`
- `analyses/probe_validity/PROBE_VALIDITY_REPORT.md`
- `analyses/probe_validity/c1_validity.png`
- `analyses/probe_validity/b2_validity.png`

# Protected Files Check

Protected files byte-identical after the run: `{protected_ok}`.

{hash_block}

"""
    with (OUT / "PROBE_VALIDITY_REPORT.md").open("w") as f:
        f.write(text)
    with (OUT / "protected_files_before_after.json").open("w") as f:
        json.dump({"before": before, "after": after, "byte_identical": protected_ok}, f, indent=2)
    os.remove(OUT / "protected_files_before_after.json")


def _phase2_runtime():
    """Import the heavy replay stack only for an explicit Phase-2 run."""
    import torch
    import torch.nn.functional as F

    return torch, F


def _load_probe(
    checkpoint: Path, probe, device, allowed_missing: Sequence[str] = (),
):
    torch, _ = _phase2_runtime()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    probe_state = {
        key[len("probe."):]: value
        for key, value in state["state_dict"].items()
        if key.startswith("probe.")
    }
    incompatible = probe.load_state_dict(probe_state, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing != set(allowed_missing) or unexpected:
        raise RuntimeError(
            f"Checkpoint contract mismatch for {checkpoint}: "
            f"missing={sorted(missing)} allowed={sorted(allowed_missing)} "
            f"unexpected={sorted(unexpected)}"
        )
    return probe.float().to(device).eval()


def _write_gzip_csv_atomic(path: Path, rows: List[Dict[str, object]], fields: Sequence[str]) -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    # QuarkFS may reject a cross-directory rename when the project owner is at
    # quota even though the completed temporary file already exists. A stable
    # symlink keeps the public artifact path unchanged while allowing later
    # writes to replace its backing file within one directory.
    destination = path.resolve() if path.is_symlink() else path
    temporary_parent = destination.parent if path.is_symlink() else TMP
    temporary = temporary_parent / f".{destination.name}.tmp"
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text_file:
                writer = csv.DictWriter(text_file, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: row.get(key, "") for key in fields})
    try:
        os.replace(temporary, destination)
    except OSError as error:
        if error.errno != 28 or path.is_symlink():
            raise
        backing = TMP / path.name
        os.replace(temporary, backing)
        path.unlink(missing_ok=True)
        path.symlink_to(backing.relative_to(path.parent))


def _merge_gzip_model_rows(
    path: Path, model: str, rows: List[Dict[str, object]], fields: Sequence[str]
) -> None:
    existing: List[Dict[str, object]] = []
    existing_fields: List[str] = []
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fields = list(reader.fieldnames or [])
            existing = [dict(row) for row in reader if row.get("model") != model]
    for row in existing:
        row.setdefault(
            "objective_family",
            MODEL_SPECS.get(str(row.get("model")), {}).get("objective_family", ""),
        )
    merged_fields = list(dict.fromkeys([*existing_fields, *fields]))
    _write_gzip_csv_atomic(path, [*existing, *rows], merged_fields)


def _write_tmp_json(name: str, payload: Dict[str, object]) -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    path = TMP / name
    temporary = TMP / f".{name}.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _feature_root(model: str, target: bool = False, prefix: int = 12) -> str:
    kind = "FEAT_TARGET_BY_PREFIX" if target else "FEAT_STREAMING_BY_PREFIX"
    return f"fj:s3://baiqiao/InsScene-15K/{kind}/{model}_prefix_{prefix}_m0p35_r0p5"


def _dataset(
    task: str,
    split: str,
    model: str = "f3r",
    missing_features: bool = False,
    prefix: int = 12,
    action_horizon: int = 1,
):
    from vidfm3d.data.components.inscene15k_dataset import InsScene15KDataset

    spec = MODEL_SPECS[model]
    common = dict(
        sources=["processed_scannetpp_v2"],
        split=split,
        vfm_name=model,
        feat_postfix=spec["feat_postfix"],
        feature_layer=spec["layer"],
        feature_timestep=749,
        feat_pixalign=True,
        num_views=prefix,
        min_view_interval=5,
        target_h=288,
        target_w=512,
        window_size=0,
        streaming_prefix=True,
        prefix_min_len=prefix,
        prefix_max_len=prefix,
        prefix_stride=1,
    )
    if task == "c1":
        if prefix not in C1_MANIFESTS:
            raise ValueError(f"No exact canonical h124 manifest declared for prefix={prefix}")
        stream_root = _feature_root(model, prefix=prefix)
        target_root = _feature_root(model, target=True, prefix=prefix)
        os.environ["INSCENE_MANIFEST"] = str(C1_MANIFESTS[prefix])
        os.environ["INSCENE_DATA_ROOT"] = SCANNET_REMOTE_ROOT
        os.environ["INSCENE_STREAMING_FEAT_ROOT"] = stream_root
        os.environ["INSCENE_TARGET_FEAT_ROOT"] = target_root
        return InsScene15KDataset(
            root=SCANNET_REMOTE_ROOT,
            root_vfm=stream_root,
            streaming_feat_root=stream_root,
            context_len=prefix,
            query_idx_divisor=None,
            include_pmaps=False,
            diag_action=True,
            target_feat_root=target_root,
            action_horizons=[action_horizon],
            **common,
        )

    os.environ.pop("INSCENE_MANIFEST", None)
    os.environ["INSCENE_DATA_ROOT"] = SCANNET_LOCAL_ROOT
    feature_root = (
        str(TMP / "missing_features")
        if missing_features else _feature_root(model, prefix=prefix)
    )
    os.environ["INSCENE_STREAMING_FEAT_ROOT"] = feature_root
    os.environ.pop("INSCENE_TARGET_FEAT_ROOT", None)
    return InsScene15KDataset(
        root=SCANNET_LOCAL_ROOT,
        root_vfm=feature_root,
        streaming_feat_root=feature_root,
        context_len=76,
        query_idx_divisor=None,
        include_pmaps=True,
        diag_hidden_obj=True,
        streaming_shared_hidden_obj=True,
        streaming_hidden_seed_prefix_len=8,
        streaming_hidden_prefix_lengths=[8, 12, 16, 24],
        allow_missing_vfm=missing_features,
        **common,
    )


def _rotation_and_translation_magnitudes(actions):
    torch, _ = _phase2_runtime()
    row0 = torch.nn.functional.normalize(actions[:, :3].float(), dim=-1)
    row1_raw = actions[:, 3:6].float()
    row1 = row1_raw - (row1_raw * row0).sum(-1, keepdim=True) * row0
    row1 = torch.nn.functional.normalize(row1, dim=-1)
    row2 = torch.linalg.cross(row0, row1, dim=-1)
    rotations = torch.stack([row0, row1, row2], dim=1)
    trace = rotations.diagonal(dim1=-2, dim2=-1).sum(-1)
    rotation = torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))
    translation = torch.linalg.norm(actions[:, 6:9].float(), dim=-1)
    return rotations, rotation, translation


def _inverse_direction_actions(actions):
    torch, _ = _phase2_runtime()
    rotations, _, _ = _rotation_and_translation_magnitudes(actions)
    inverse_rotation = rotations.transpose(1, 2)
    inverse_translation = -torch.bmm(
        inverse_rotation, actions[:, 6:9].float().unsqueeze(-1)
    ).squeeze(-1)
    return torch.cat([inverse_rotation[:, :2].reshape(-1, 6), inverse_translation], dim=-1)


def _feature_summary(predictions, targets):
    torch, F = _phase2_runtime()
    cosine = F.cosine_similarity(predictions, targets, dim=-1)
    similarity = F.normalize(predictions, dim=-1) @ F.normalize(targets, dim=-1).T
    rank0 = (
        similarity.argsort(dim=-1, descending=True)
        == torch.arange(targets.shape[0]).unsqueeze(-1)
    ).float().argmax(dim=-1)
    return {
        "mean_cos": float(cosine.mean().item()),
        "global_R@1": float((rank0 == 0).float().mean().item()),
        "global_R@5": float((rank0 < 5).float().mean().item()),
        "mean_rank": float(rank0.float().mean().item()),
    }


def _predict_actions(probe, inputs, actions, device, batch_size: int = 16):
    torch, _ = _phase2_runtime()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            stop = min(start + batch_size, len(inputs))
            x = torch.stack(inputs[start:stop]).float().to(device)
            a = actions[start:stop].float().to(device)
            outputs.append(probe(x, a).detach().cpu())
    return torch.cat(outputs, dim=0)


def _derangement(n: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    values = list(range(n))
    while True:
        rng.shuffle(values)
        if all(i != value for i, value in enumerate(values)):
            return values


def _rank_metrics(prediction, targets, positive_index: int) -> Dict[str, float]:
    torch, F = _phase2_runtime()
    prediction = prediction.float().unsqueeze(0)
    targets = targets.float()
    similarities = (F.normalize(prediction, dim=-1) @ F.normalize(targets, dim=-1).T)[0]
    positive_similarity = float(similarities[positive_index].item())
    rank = int(
        (similarities.argsort(descending=True) == positive_index)
        .nonzero(as_tuple=False)[0, 0]
        .item()
    )
    return {
        "target_cosine": positive_similarity,
        "target_rank": rank,
        "r1": int(rank == 0),
        "reciprocal_rank": 1.0 / (rank + 1),
        "prediction_norm": float(torch.linalg.norm(prediction).item()),
    }


def _candidate_pools(index: int, metadata, targets, drift, rotation, translation, actions):
    torch, F = _phase2_runtime()
    n = len(metadata)
    all_indices = list(range(n))

    rng = random.Random(PHASE2_SEED * 1000003 + index)
    others = [value for value in all_indices if value != index]
    random_pool = [index] + rng.sample(others, min(31, len(others)))

    same_scene = [
        value for value, item in enumerate(metadata)
        if item["scene_id"] == metadata[index]["scene_id"]
    ]
    same_trajectory = [
        value for value, item in enumerate(metadata)
        if item["trajectory_id"] == metadata[index]["trajectory_id"]
    ]
    temporal = sorted(
        same_trajectory,
        key=lambda value: abs(metadata[value]["window_id"] - metadata[index]["window_id"]),
    )[:32]

    drift_pool = sorted(others, key=lambda value: abs(float(drift[value]) - float(drift[index])))[:31]
    pose_pool = sorted(
        others,
        key=lambda value: (
            (float(rotation[value]) - float(rotation[index])) ** 2
            + (float(translation[value]) - float(translation[index])) ** 2
        ),
    )[:31]
    action_similarity = F.normalize(actions.float(), dim=-1) @ F.normalize(
        actions[index:index + 1].float(), dim=-1
    ).T
    action_pool = [
        value for value in action_similarity[:, 0].argsort(descending=True).tolist()
        if value != index
    ][:31]

    pools = {
        "HISTORICAL_GLOBAL": all_indices,
        "GLOBAL_RANDOM": random_pool,
        "SAME_SCENE": same_scene,
        "SAME_TRAJECTORY": same_trajectory,
        "TEMPORAL_NEIGHBORS": temporal,
        "MATCHED_FEATURE_DRIFT": [index] + drift_pool,
        "MATCHED_POSE_DISPLACEMENT": [index] + pose_pool,
        "ACTION_CONFUSABLE_FUTURES": [index] + action_pool,
    }
    for name, values in list(pools.items()):
        unique = list(dict.fromkeys(values))
        if index not in unique:
            unique.insert(0, index)
        pools[name] = unique
    return pools


def replay_c1(model: str) -> Dict[str, object]:
    torch, F = _phase2_runtime()
    from vidfm3d.models.components.probe_action_dynamics import ActionDynamicsProbe

    spec = MODEL_SPECS[model]
    torch.manual_seed(PHASE2_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = _dataset("c1", "val", model=model)
    inputs = []
    actions_list = []
    targets_list = []
    copies = []
    metadata = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if not bool(sample["dyn_valid"].item()):
            continue
        scene = dataset.scenes[index]
        inputs.append(sample["input_feat"].detach().cpu())
        actions_list.append(sample["action"].detach().cpu().float())
        targets_list.append(sample["target_feat"].mean(dim=(0, 1)).detach().cpu().float())
        copies.append(sample["input_feat"][-1].mean(dim=(0, 1)).detach().cpu().float())
        target_frame = int(sample["target_frame_idx"].item())
        context_frames = [int(value) for value in scene.get("streaming_indices", [])]
        metadata.append({
            "manifest_sample_id": str(scene.get("sample_id", f"row{index:06d}")),
            "scene_id": str(scene.get("scene_id", Path(str(scene["scene_dir"])).name)),
            "trajectory_id": str(scene.get("scene_id", Path(str(scene["scene_dir"])).name)),
            "window_id": int(scene.get("streaming_window_id", -1)),
            "context_frames": context_frames,
            "target_frame": target_frame,
        })
        if (index + 1) % 20 == 0:
            print(f"[c1 {model} data] {index + 1}/{len(dataset)}", flush=True)

    actions = torch.stack(actions_list)
    targets = torch.stack(targets_list)
    copy_predictions = torch.stack(copies)
    n = len(inputs)
    if n != len(dataset):
        raise RuntimeError(f"C1 valid rows {n} != evaluator dataset rows {len(dataset)}")

    probe = _load_probe(
        spec["c1_ckpt"],
        ActionDynamicsProbe(
            in_channels=spec["channels"], action_dim=9, hidden_dim=512,
            num_layers=2, num_heads=8, max_seq_len=64,
        ),
        device,
        allowed_missing=("constant_context", "constant_action"),
    )
    zero_actions = torch.zeros_like(actions)
    roll_indices = []
    for start in range(0, n, 4):
        batch = list(range(start, min(start + 4, n)))
        roll_indices.extend([batch[-1], *batch[:-1]])
    random_indices = _derangement(n, PHASE2_SEED)
    _, rotation, translation = _rotation_and_translation_magnitudes(actions)
    magnitudes = torch.stack([rotation, translation], dim=-1)
    magnitude_distance = torch.cdist(magnitudes, magnitudes)
    magnitude_distance.fill_diagonal_(float("inf"))
    matched_indices = magnitude_distance.argmin(dim=-1).tolist()

    inverse_actions = _inverse_direction_actions(actions)
    rotation_flip_actions = torch.cat([inverse_actions[:, :6], actions[:, 6:9]], dim=-1)
    translation_flip_actions = torch.cat([actions[:, :6], -actions[:, 6:9]], dim=-1)
    condition_actions = {
        "true": actions,
        "zero": zero_actions,
        "batch_roll": actions[roll_indices],
        "random_valid": actions[random_indices],
        "matched_magnitude": actions[matched_indices],
        "direction_flip": inverse_actions,
        "rotation_direction_flip": rotation_flip_actions,
        "translation_direction_flip": translation_flip_actions,
    }
    predictions = {
        name: _predict_actions(
            probe, inputs, condition, device,
            batch_size=4 if int(spec["channels"]) > 1024 else 16,
        )
        for name, condition in condition_actions.items()
    }
    predictions["copy"] = copy_predictions

    history = load_json(str(spec["c1_summary"]))
    reproduced = _feature_summary(predictions["true"], targets)
    reproduced.update({
        f"no_action_{key}": value
        for key, value in _feature_summary(predictions["zero"], targets).items()
    })
    reproduced.update({
        f"shuffled_action_{key}": value
        for key, value in _feature_summary(predictions["batch_roll"], targets).items()
    })
    reproduced.update({
        f"last_observation_{key}": value
        for key, value in _feature_summary(predictions["copy"], targets).items()
    })
    comparison_keys = [
        "mean_cos", "global_R@1", "global_R@5", "mean_rank",
        "no_action_mean_cos", "no_action_global_R@1", "no_action_global_R@5",
        "no_action_mean_rank", "shuffled_action_mean_cos",
        "shuffled_action_global_R@1", "shuffled_action_global_R@5",
        "shuffled_action_mean_rank", "last_observation_mean_cos",
        "last_observation_global_R@1", "last_observation_global_R@5",
        "last_observation_mean_rank",
    ]
    differences = {key: reproduced[key] - float(history[key]) for key in comparison_keys}
    metric_tolerances = {
        key: (1.0 / n + 1e-5 if key.endswith("mean_rank") else 1e-5)
        for key in comparison_keys
    }
    gate_failures = {
        key: abs(differences[key])
        for key in comparison_keys
        if abs(differences[key]) > metric_tolerances[key]
    }
    smoke = {
        "task": "C1", "model": model, "objective_family": spec["objective_family"],
        "layer": spec["layer"], "prefix": 12,
        "n": n, "device": str(device), "reproduced": reproduced,
        "historical": {key: history[key] for key in comparison_keys},
        "absolute_differences": {key: abs(value) for key, value in differences.items()},
        "max_absolute_difference": max(abs(value) for value in differences.values()),
        "metric_tolerances": metric_tolerances,
        "gate_failures": gate_failures,
        "gate_status": "PASS" if n == int(history["n"]) and not gate_failures else "FAIL",
        "checkpoint_compatibility": (
            "Current full-conditioning class adds unused constant_context/constant_action; "
            "only those two absent historical keys are allowlisted."
        ),
        "rank_tolerance_note": (
            "Mean-rank permits one rank position over n samples because the historical "
            "GPU argsort has device/version-dependent ordering for exactly tied targets."
        ),
    }
    _write_tmp_json(f"c1_{model}_reproduction.json", smoke)
    if smoke["gate_status"] != "PASS":
        raise RuntimeError(f"C1 smoke reproduction failed: {smoke}")

    drift = 1.0 - F.cosine_similarity(copy_predictions, targets, dim=-1)
    rows = []
    condition_names = list(predictions)
    for index, item in enumerate(metadata):
        pools = _candidate_pools(index, metadata, targets, drift, rotation, translation, actions)
        stable_id = (
            f"C1|{model}|l{spec['layer']}|p12|val|{item['manifest_sample_id']}|"
            f"target{item['target_frame']:06d}"
        )
        for pool_name, candidate_indices in pools.items():
            candidate_targets = targets[candidate_indices]
            positive = candidate_indices.index(index)
            row = {
                "sample_id": stable_id,
                "manifest_sample_id": item["manifest_sample_id"],
                "model": model, "objective_family": spec["objective_family"],
                "layer": spec["layer"], "prefix": 12,
                "scene_id": item["scene_id"],
                "trajectory_id": item["trajectory_id"],
                "window_id": item["window_id"], "eval_split": "val",
                "checkpoint": rel(spec["c1_ckpt"]),
                "context_frame_ids": json.dumps(item["context_frames"], separators=(",", ":")),
                "target_frame_id": item["target_frame"],
                "candidate_target_id": f"{item['scene_id']}:{item['target_frame']:06d}",
                "candidate_pool": pool_name,
                "candidate_count": len(candidate_indices),
                "chance_r1": 1.0 / len(candidate_indices),
                "candidate_seed": PHASE2_SEED,
                "true_action_vector": json.dumps(actions[index].tolist(), separators=(",", ":")),
                "translation_magnitude": float(translation[index]),
                "rotation_magnitude_rad": float(rotation[index]),
                "total_displacement": math.hypot(
                    float(translation[index]), float(rotation[index])
                ),
                "action_type": "RELATIVE_POSE_ROT6D_TRANSLATION3D",
                "last_observation_to_target_cosine": float(1.0 - drift[index]),
                "feature_drift": float(drift[index]),
                "batch_roll_source_sample_id": metadata[roll_indices[index]]["manifest_sample_id"],
                "random_valid_source_sample_id": metadata[random_indices[index]]["manifest_sample_id"],
                "matched_magnitude_source_sample_id": metadata[matched_indices[index]]["manifest_sample_id"],
                "matched_magnitude_action_vector": json.dumps(
                    actions[matched_indices[index]].tolist(), separators=(",", ":")
                ),
            }
            for condition_name in condition_names:
                metrics = _rank_metrics(
                    predictions[condition_name][index], candidate_targets, positive
                )
                for metric_name, value in metrics.items():
                    row[f"{condition_name}_{metric_name}"] = value
            row["action_margin_vs_matched_cosine"] = (
                row["true_target_cosine"] - row["matched_magnitude_target_cosine"]
            )
            row["action_margin_vs_matched_rank"] = (
                row["matched_magnitude_target_rank"] - row["true_target_rank"]
            )
            rows.append(row)

    identity_fields = [
        "sample_id", "manifest_sample_id", "model", "objective_family", "layer", "prefix", "scene_id",
        "trajectory_id", "window_id", "eval_split", "checkpoint", "context_frame_ids",
        "target_frame_id", "candidate_target_id", "candidate_pool", "candidate_count",
        "chance_r1", "candidate_seed", "true_action_vector", "translation_magnitude",
        "rotation_magnitude_rad", "total_displacement", "action_type",
        "last_observation_to_target_cosine",
        "feature_drift", "batch_roll_source_sample_id", "random_valid_source_sample_id",
        "matched_magnitude_source_sample_id", "matched_magnitude_action_vector",
    ]
    metric_fields = [
        f"{condition}_{metric}"
        for condition in condition_names
        for metric in ["target_cosine", "target_rank", "r1", "reciprocal_rank", "prediction_norm"]
    ]
    fields = identity_fields + metric_fields + [
        "action_margin_vs_matched_cosine", "action_margin_vs_matched_rank"
    ]
    for pool_name in sorted({row["candidate_pool"] for row in rows}):
        pool_rows = [row for row in rows if row["candidate_pool"] == pool_name]
        sample_ids = [row["sample_id"] for row in pool_rows]
        if len(pool_rows) != n or len(set(sample_ids)) != n or any(not value for value in sample_ids):
            raise RuntimeError(
                f"C1 unstable sample IDs in {pool_name}: rows={len(pool_rows)} "
                f"unique={len(set(sample_ids))} expected={n}"
            )
        if any(int(row["candidate_count"]) < 1 for row in pool_rows):
            raise RuntimeError(f"C1 candidate pool {pool_name} contains an empty pool")
    _merge_gzip_model_rows(OUT / "c1_sample_diagnostics.csv.gz", model, rows, fields)
    smoke["diagnostic_rows"] = len(rows)
    smoke["unique_sample_ids"] = len({row["sample_id"] for row in rows})
    smoke["candidate_pools"] = sorted({row["candidate_pool"] for row in rows})
    _write_tmp_json(f"c1_{model}_phase2_summary.json", smoke)
    return smoke


def _b2_prediction(probe, feature, query, device):
    torch, _ = _phase2_runtime()
    with torch.no_grad():
        output = probe(
            feature.unsqueeze(0).float().to(device),
            query.unsqueeze(0).float().to(device),
        )
    logits = output["logits"].reshape(-1).detach().cpu()
    log_distance = float(output["log_dist"].reshape(-1)[0].detach().cpu().item())
    return logits, log_distance


def _b2_metrics(logits, log_distance: float, polar, topk_override=None):
    torch, _ = _phase2_runtime()
    n_el = 8
    az_bin = int(((float(polar[0]) + math.pi) / (2 * math.pi) * 16))
    el_bin = int(((float(polar[1]) + math.pi / 2) / math.pi * 8))
    az_bin = min(max(az_bin, 0), 15)
    el_bin = min(max(el_bin, 0), 7)
    target_bin = az_bin * n_el + el_bin
    if topk_override is None:
        top_indices = logits.topk(3).indices.tolist()
        predicted_bin = int(top_indices[0])
    else:
        top_indices = [int(value) for value in topk_override]
        predicted_bin = top_indices[0]
    pred_az_bin, pred_el_bin = divmod(predicted_bin, n_el)
    pred_az = (pred_az_bin + 0.5) * (2 * math.pi / 16) - math.pi
    pred_el = (pred_el_bin + 0.5) * (math.pi / 8) - math.pi / 2
    cos_angle = (
        math.sin(pred_el) * math.sin(float(polar[1]))
        + math.cos(pred_el) * math.cos(float(polar[1]))
        * math.cos(pred_az - float(polar[0]))
    )
    angle_error = math.degrees(math.acos(min(1.0, max(-1.0, cos_angle))))
    az_error = math.degrees(abs(math.atan2(
        math.sin(pred_az - float(polar[0])), math.cos(pred_az - float(polar[0]))
    )))
    el_error = math.degrees(abs(pred_el - float(polar[1])))
    return {
        "gt_joint_bin": target_bin,
        "predicted_joint_bin": predicted_bin,
        "predicted_azimuth": pred_az,
        "predicted_elevation": pred_el,
        "angular_error_deg": angle_error,
        "azimuth_error_deg": az_error,
        "elevation_error_deg": el_error,
        "log_distance_error": abs(log_distance - float(polar[2])) if math.isfinite(log_distance) else "",
        "top1": int(predicted_bin == target_bin),
        "top3": int(target_bin in top_indices[:3]),
        "topk_bins": json.dumps(top_indices[:3], separators=(",", ":")),
        "topk_logits": (
            json.dumps([float(logits[index]) for index in top_indices[:3]], separators=(",", ":"))
            if logits is not None else ""
        ),
    }


def _collect_b2_prior_targets(model: str, limit: int = 128):
    # The streaming feature root defines the exact historical window universe;
    # a dummy/missing root falls back to raw-scene windows and yields a different
    # hidden-object selection contract. Read the existing train features even
    # though only labels are retained for the prior.
    dataset = _dataset("b2", "train", model=model, missing_features=False)
    targets = []
    examined = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        examined += 1
        if bool(sample["hidden_obj_valid"].item()):
            polar = sample["hidden_obj_polar"].float()
            az_bin = min(max(int(((float(polar[0]) + math.pi) / (2 * math.pi) * 16)), 0), 15)
            el_bin = min(max(int(((float(polar[1]) + math.pi / 2) / math.pi * 8)), 0), 7)
            targets.append({
                "joint_bin": az_bin * 8 + el_bin,
                "azimuth": float(polar[0]),
                "elevation": float(polar[1]),
                "log_distance": float(polar[2]),
            })
        if len(targets) >= limit:
            break
        if (index + 1) % 20 == 0:
            print(f"[b2 {model} prior] examined={index + 1} valid={len(targets)}", flush=True)
    if not targets:
        raise RuntimeError("B2 train-only prior found no valid training labels")
    return targets, examined


def replay_b2(model: str) -> Dict[str, object]:
    torch, _ = _phase2_runtime()
    from collections import Counter, defaultdict
    from vidfm3d.models.components.probe_ego_belief_v2 import EgoBeliefProbeV2

    spec = MODEL_SPECS[model]
    torch.manual_seed(PHASE2_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = _load_probe(
        spec["b2_ckpt"],
        EgoBeliefProbeV2(
            in_channels=spec["channels"], decoder_type="transformer", num_frames=12,
            hidden_dim=512, num_layers=4, num_heads=8, max_seq_len=64,
            max_h=spec["b2_max_h"], max_w=spec["b2_max_w"],
            n_az_bins=16, n_el_bins=8, patch_pool=2,
        ),
        device,
        allowed_missing=("constant_context",),
    )
    dataset = _dataset("b2", "val", model=model)
    samples = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if not bool(sample["hidden_obj_valid"].item()):
            continue
        scene = dataset.scenes[index]
        prefix_indices = [int(value) for value in sample["prefix_indices"].tolist()]
        query_frame = int(sample["belief_query_frame"].item())
        hidden_mask = sample["hidden_obj_mask"].detach().cpu().bool()
        pixel_counts = hidden_mask.flatten(1).sum(1).tolist()
        scene_id = str(scene.get("scene_id", Path(str(scene["scene_dir"])).name))
        window_id = int(scene.get("streaming_window_id", -1))
        hidden_id = int(sample["hidden_obj_id"].item())
        samples.append({
            "dataset_index": index,
            "feature": sample["vfm_feat"].detach().cpu(),
            "query": sample["belief_query_feat"].detach().cpu(),
            "polar": sample["hidden_obj_polar"].detach().cpu().float(),
            "hidden_mask": hidden_mask,
            "pixel_counts": [int(value) for value in pixel_counts],
            "scene_id": scene_id,
            "trajectory_id": scene_id,
            "window_id": window_id,
            "manifest_sample_id": str(
                sample.get("sample_id")
                or scene.get("sample_id", f"{scene_id}_w{window_id:03d}_p12")
            ),
            "prefix_indices": prefix_indices,
            "query_frame": query_frame,
            "query_raw_frame": prefix_indices[query_frame],
            "target_raw_frame": prefix_indices[-1],
            "hidden_id": hidden_id,
        })
        if (index + 1) % 20 == 0:
            print(f"[b2 {model} data] {index + 1}/{len(dataset)} valid={len(samples)}", flush=True)

    by_scene = defaultdict(list)
    for index, item in enumerate(samples):
        by_scene[item["scene_id"]].append(index)
    same_scene_swap = {}
    for values in by_scene.values():
        if len(values) > 1:
            for position, value in enumerate(values):
                same_scene_swap[value] = values[(position + 1) % len(values)]

    # Gate the untouched historical path before fitting priors or running any
    # query/history intervention.
    gate_records = []
    for index, item in enumerate(samples):
        logits, log_distance = _b2_prediction(
            probe, item["feature"], item["query"], device
        )
        metrics = _b2_metrics(logits, log_distance, item["polar"])
        metrics["predicted_log_distance"] = log_distance
        gate_records.append(metrics)
        if (index + 1) % 20 == 0:
            print(f"[b2 {model} gate] {index + 1}/{len(samples)}", flush=True)
    gate_reproduced = {
        "n": len(gate_records),
        "mean_ang_err_deg": sum(float(row["angular_error_deg"]) for row in gate_records) / len(gate_records),
        "mean_az_err_deg": sum(float(row["azimuth_error_deg"]) for row in gate_records) / len(gate_records),
        "mean_el_err_deg": sum(float(row["elevation_error_deg"]) for row in gate_records) / len(gate_records),
        "top1": sum(int(row["top1"]) for row in gate_records) / len(gate_records),
        "top3": sum(int(row["top3"]) for row in gate_records) / len(gate_records),
        "mean_log_dist_err": sum(float(row["log_distance_error"]) for row in gate_records) / len(gate_records),
    }
    history = load_json(str(spec["b2_summary"]))
    comparison_keys = [
        "mean_ang_err_deg", "mean_az_err_deg", "mean_el_err_deg",
        "top1", "top3", "mean_log_dist_err",
    ]
    gate_differences = {
        key: gate_reproduced[key] - float(history[key]) for key in comparison_keys
    }
    gate_smoke = {
        "task": "B2", "model": model, "objective_family": spec["objective_family"],
        "layer": spec["layer"], "prefix": 12,
        "stage": "UNTOUCHED_BASE_REPLAY", "dataset_n": len(dataset),
        "n": len(gate_records), "device": str(device),
        "reproduced": gate_reproduced,
        "historical": {key: history[key] for key in comparison_keys},
        "absolute_differences": {
            key: abs(value) for key, value in gate_differences.items()
        },
        "max_absolute_difference": max(abs(value) for value in gate_differences.values()),
    }
    _write_tmp_json(f"b2_{model}_reproduction.json", gate_smoke)
    if len(gate_records) != int(history["n"]) or gate_smoke["max_absolute_difference"] > 1e-5:
        raise RuntimeError(f"B2 smoke reproduction failed before perturbations: {gate_smoke}")

    prior_targets, prior_examined = _collect_b2_prior_targets(model, limit=128)
    prior_bins = [int(target["joint_bin"]) for target in prior_targets]
    prior_counts = Counter(prior_bins)
    prior_top3 = [value for value, _ in prior_counts.most_common(3)]
    prior_logits = torch.tensor([float(prior_counts.get(value, 0)) for value in range(128)])
    mean_x = sum(
        math.cos(target["elevation"]) * math.cos(target["azimuth"])
        for target in prior_targets
    ) / len(prior_targets)
    mean_y = sum(
        math.cos(target["elevation"]) * math.sin(target["azimuth"])
        for target in prior_targets
    ) / len(prior_targets)
    mean_z = sum(math.sin(target["elevation"]) for target in prior_targets) / len(prior_targets)
    constant_azimuth = math.atan2(mean_y, mean_x)
    constant_elevation = math.atan2(mean_z, math.hypot(mean_x, mean_y))
    constant_az_bin = min(max(int(((constant_azimuth + math.pi) / (2 * math.pi) * 16)), 0), 15)
    constant_el_bin = min(max(int(((constant_elevation + math.pi / 2) / math.pi * 8)), 0), 7)
    constant_bin = constant_az_bin * 8 + constant_el_bin
    prior_log_distance = sum(target["log_distance"] for target in prior_targets) / len(prior_targets)

    rows = []
    base_records = []
    for index, item in enumerate(samples):
        generator = torch.Generator().manual_seed(PHASE2_SEED * 1000003 + index)
        permutation = torch.randperm(item["feature"].shape[0], generator=generator)
        visible_frames = [
            frame for frame, count in enumerate(item["pixel_counts"]) if count > 0
        ]
        last_only = torch.zeros_like(item["feature"])
        last_only[-1] = item["feature"][-1]
        query_only = torch.zeros_like(item["feature"])
        visible_only = torch.zeros_like(item["feature"])
        if visible_frames:
            visible_only[visible_frames] = item["feature"][visible_frames]
        remove_visible = item["feature"].clone()
        if visible_frames:
            remove_visible[visible_frames] = 0
        condition_inputs = [
            ("FULL_HISTORY_PLUS_QUERY", "CORRECT_QUERY", "FULL_HISTORY", item["feature"], item["query"], "FIXED_PREDICTOR"),
            ("HISTORY_WITHOUT_QUERY", "QUERY_MASK", "FULL_HISTORY", item["feature"], torch.zeros_like(item["query"]), "FIXED_PREDICTOR"),
            ("CURRENT_FRAME_PLUS_QUERY", "CORRECT_QUERY", "LAST_FRAME_ONLY", last_only, item["query"], "FIXED_PREDICTOR"),
            ("QUERY_FEATURE_ONLY", "CORRECT_QUERY", "NO_HISTORY", query_only, item["query"], "FIXED_PREDICTOR_DIAGNOSTIC"),
            ("FRAME_ORDER_SHUFFLE", "CORRECT_QUERY", "FRAME_ORDER_SHUFFLE", item["feature"][permutation], item["query"], "FIXED_PREDICTOR"),
            ("REVERSED_HISTORY", "CORRECT_QUERY", "REVERSED_HISTORY", item["feature"].flip(0), item["query"], "FIXED_PREDICTOR"),
            ("REMOVE_TARGET_VISIBLE_FRAMES", "CORRECT_QUERY", "REMOVE_TARGET_VISIBLE_FRAMES", remove_visible, item["query"], "FIXED_PREDICTOR"),
            ("KEEP_TARGET_VISIBLE_FRAMES_ONLY", "CORRECT_QUERY", "KEEP_TARGET_VISIBLE_FRAMES_ONLY", visible_only, item["query"], "FIXED_PREDICTOR"),
        ]
        if index in same_scene_swap:
            source = same_scene_swap[index]
            condition_inputs.append((
                "SAME_SCENE_QUERY_SWAP", "SAME_SCENE_SWAP", "FULL_HISTORY",
                item["feature"], samples[source]["query"], "FIXED_PREDICTOR",
            ))

        stable_id = (
            f"B2|{model}|l{spec['layer']}|p12|val|{item['manifest_sample_id']}|"
            f"obj{item['hidden_id']}|target{item['target_raw_frame']:06d}"
        )
        base = {
            "sample_id": stable_id,
            "manifest_sample_id": item["manifest_sample_id"],
            "scene_id": item["scene_id"], "trajectory_id": item["trajectory_id"],
            "window_id": item["window_id"], "model": model,
            "objective_family": spec["objective_family"], "layer": spec["layer"],
            "prefix": 12, "eval_split": "val", "checkpoint": rel(spec["b2_ckpt"]),
            "query_type": "MASKED_POOLED_OBJECT_APPEARANCE",
            "query_frame": item["query_frame"],
            "query_raw_frame": item["query_raw_frame"],
            "query_object_id": item["hidden_id"], "query_category": "",
            "query_feature_norm": float(torch.linalg.norm(item["query"].float()).item()),
            "query_source_reference": f"{item['scene_id']}:{item['query_raw_frame']:06d}:obj{item['hidden_id']}",
            "target_object_id": item["hidden_id"], "target_category": "",
            "gt_azimuth": float(item["polar"][0]),
            "gt_elevation": float(item["polar"][1]),
            "gt_log_distance": float(item["polar"][2]),
            "target_visible_frame_indices": json.dumps(visible_frames, separators=(",", ":")),
            "visible_pixel_count_per_frame": json.dumps(item["pixel_counts"], separators=(",", ":")),
            "target_first_visible_frame": min(visible_frames) if visible_frames else "",
            "target_last_visible_frame": max(visible_frames) if visible_frames else "",
            "target_ever_visible_in_prefix": int(bool(visible_frames)),
            "target_visible_in_final_frame": int(item["pixel_counts"][-1] > 0),
            "target_currently_hidden": int(item["pixel_counts"][-1] < 200),
            "target_visible_frame_count": len(visible_frames),
            "identifiability": (
                "UNIQUE_SEEN_THEN_HIDDEN"
                if visible_frames and item["pixel_counts"][-1] < 200
                else "METADATA_INSUFFICIENT"
            ),
            "same_category_instance_count_scene": "",
            "same_category_instance_count_history": "",
            "query_to_instance_match_count": 1,
            "perturbation_seed": PHASE2_SEED,
            "prior_fit_split": "train",
            "prior_fit_n": len(prior_bins),
        }

        for method, query_perturbation, history_perturbation, feature, query, baseline_type in condition_inputs:
            logits, log_distance = _b2_prediction(probe, feature, query, device)
            metrics = _b2_metrics(logits, log_distance, item["polar"])
            row = dict(base)
            row.update({
                "method": method, "query_perturbation": query_perturbation,
                "history_perturbation": history_perturbation,
                "baseline_type": baseline_type,
                "query_swap_source_sample_id": (
                    samples[same_scene_swap[index]]["manifest_sample_id"]
                    if method == "SAME_SCENE_QUERY_SWAP" else ""
                ),
                "history_mapping": (
                    json.dumps(permutation.tolist(), separators=(",", ":"))
                    if method == "FRAME_ORDER_SHUFFLE" else ""
                ),
                "predicted_log_distance": log_distance,
            })
            row.update(metrics)
            rows.append(row)
            if method == "FULL_HISTORY_PLUS_QUERY":
                base_records.append(row)

        for method, prior_topk in [
            ("GLOBAL_CONSTANT", [constant_bin]),
            ("GLOBAL_ANGLE_HISTOGRAM", prior_top3),
        ]:
            metrics = _b2_metrics(
                prior_logits, prior_log_distance, item["polar"], topk_override=prior_topk
            )
            row = dict(base)
            row.update({
                "method": method, "query_perturbation": "NO_QUERY",
                "history_perturbation": "NO_HISTORY", "baseline_type": "TRAIN_FITTED_PRIOR",
                "query_swap_source_sample_id": "", "history_mapping": "",
                "predicted_log_distance": prior_log_distance,
            })
            row.update(metrics)
            rows.append(row)

    reproduced = {
        "n": len(base_records),
        "mean_ang_err_deg": sum(float(row["angular_error_deg"]) for row in base_records) / len(base_records),
        "mean_az_err_deg": sum(float(row["azimuth_error_deg"]) for row in base_records) / len(base_records),
        "mean_el_err_deg": sum(float(row["elevation_error_deg"]) for row in base_records) / len(base_records),
        "top1": sum(int(row["top1"]) for row in base_records) / len(base_records),
        "top3": sum(int(row["top3"]) for row in base_records) / len(base_records),
        "mean_log_dist_err": sum(float(row["log_distance_error"]) for row in base_records) / len(base_records),
    }
    differences = {key: reproduced[key] - float(history[key]) for key in comparison_keys}
    smoke = {
        "task": "B2", "model": model, "objective_family": spec["objective_family"],
        "layer": spec["layer"], "prefix": 12,
        "dataset_n": len(dataset), "n": len(base_records), "device": str(device),
        "reproduced": reproduced,
        "historical": {key: history[key] for key in comparison_keys},
        "absolute_differences": {key: abs(value) for key, value in differences.items()},
        "max_absolute_difference": max(abs(value) for value in differences.values()),
        "prior_fit_n": len(prior_bins), "prior_examined_n": prior_examined,
        "prior_top3_bins": prior_top3, "prior_constant_bin": constant_bin,
        "prior_constant_azimuth": constant_azimuth,
        "prior_constant_elevation": constant_elevation,
        "prior_log_distance": prior_log_distance,
        "checkpoint_compatibility": (
            "Current full-context class adds unused constant_context; only that "
            "absent historical key is allowlisted."
        ),
    }
    _write_tmp_json(f"b2_{model}_reproduction.json", smoke)
    if len(base_records) != int(history["n"]) or smoke["max_absolute_difference"] > 1e-5:
        raise RuntimeError(f"B2 smoke reproduction failed: {smoke}")

    fields = [
        "sample_id", "manifest_sample_id", "scene_id", "trajectory_id", "window_id",
        "model", "objective_family", "layer", "prefix", "eval_split", "checkpoint", "query_type",
        "query_frame", "query_raw_frame", "query_object_id", "query_category",
        "query_feature_norm", "query_source_reference", "target_object_id", "target_category",
        "gt_azimuth", "gt_elevation", "gt_log_distance", "gt_joint_bin",
        "predicted_joint_bin", "topk_bins", "topk_logits", "predicted_azimuth",
        "predicted_elevation", "predicted_log_distance", "angular_error_deg",
        "azimuth_error_deg", "elevation_error_deg", "log_distance_error", "top1", "top3",
        "target_visible_frame_indices", "visible_pixel_count_per_frame",
        "target_first_visible_frame", "target_last_visible_frame",
        "target_ever_visible_in_prefix", "target_visible_in_final_frame",
        "target_currently_hidden", "target_visible_frame_count", "identifiability",
        "same_category_instance_count_scene", "same_category_instance_count_history",
        "query_to_instance_match_count", "method", "query_perturbation",
        "history_perturbation", "baseline_type", "query_swap_source_sample_id",
        "history_mapping", "perturbation_seed", "prior_fit_split", "prior_fit_n",
    ]
    base_ids = [row["sample_id"] for row in base_records]
    if len(base_ids) != len(set(base_ids)) or any(not value for value in base_ids):
        raise RuntimeError(
            f"B2 unstable base sample IDs: rows={len(base_ids)} unique={len(set(base_ids))}"
        )
    base_gt = {
        row["sample_id"]: (
            row["target_object_id"], row["gt_azimuth"],
            row["gt_elevation"], row["gt_log_distance"],
        )
        for row in base_records
    }
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        method_ids = [row["sample_id"] for row in method_rows]
        if len(method_ids) != len(set(method_ids)) or any(not value for value in method_ids):
            raise RuntimeError(f"B2 duplicate or missing sample IDs for method={method}")
        for row in method_rows:
            gt = (
                row["target_object_id"], row["gt_azimuth"],
                row["gt_elevation"], row["gt_log_distance"],
            )
            if gt != base_gt[row["sample_id"]]:
                raise RuntimeError(f"B2 perturbation changed GT target for {row['sample_id']}")
    _merge_gzip_model_rows(OUT / "b2_sample_diagnostics.csv.gz", model, rows, fields)
    smoke["diagnostic_rows"] = len(rows)
    smoke["unique_sample_ids"] = len({row["sample_id"] for row in rows})
    smoke["methods"] = sorted({row["method"] for row in rows})
    _write_tmp_json(f"b2_{model}_phase2_summary.json", smoke)
    return smoke


def run_phase2_replay(task: str, model: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    before = snapshot(PROTECTED)
    payload = {"status": "PASS", "task": task, "model": model, "seed": PHASE2_SEED}
    if task in {"c1", "all"}:
        payload["c1"] = replay_c1(model)
    if task in {"b2", "all"}:
        payload["b2"] = replay_b2(model)
    after = snapshot(PROTECTED)
    payload["protected_files_byte_identical"] = before == after
    _write_tmp_json(f"phase2_{model}_replay_summary.json", payload)
    if before != after:
        raise RuntimeError("Protected files changed during Phase-2 replay")
    print(json.dumps(payload, indent=2), flush=True)


def run_b2_prior_gate(model: str) -> None:
    targets, examined = _collect_b2_prior_targets(model, limit=1)
    print(json.dumps({
        "status": "PASS", "labels": len(targets),
        "examined": examined, "target": targets[0],
    }, indent=2), flush=True)


def _read_gzip_csv(path: Path) -> List[Dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_c1_sample_rows(rows: Sequence[Dict[str, str]], expected_n: int) -> None:
    required = {
        "sample_id", "manifest_sample_id", "scene_id", "trajectory_id",
        "candidate_pool", "candidate_count", "true_action_vector",
        "translation_magnitude", "rotation_magnitude_rad", "total_displacement",
        "target_frame_id", "true_target_cosine", "true_target_rank", "true_r1",
        "copy_target_cosine", "zero_target_cosine", "random_valid_target_cosine",
        "matched_magnitude_target_cosine", "direction_flip_target_cosine",
    }
    missing_columns = sorted(required - set(rows[0]) if rows else required)
    if missing_columns:
        raise RuntimeError(f"C1 sample diagnostics missing columns: {missing_columns}")
    expected_pools = {
        "HISTORICAL_GLOBAL", "GLOBAL_RANDOM", "SAME_SCENE", "SAME_TRAJECTORY",
        "TEMPORAL_NEIGHBORS", "MATCHED_FEATURE_DRIFT",
        "MATCHED_POSE_DISPLACEMENT", "ACTION_CONFUSABLE_FUTURES",
    }
    pools = {row["candidate_pool"] for row in rows}
    if pools != expected_pools:
        raise RuntimeError(f"C1 candidate pools mismatch: {sorted(pools)}")
    identity_by_id: Dict[str, Tuple[str, ...]] = {}
    for pool in sorted(pools):
        pool_rows = [row for row in rows if row["candidate_pool"] == pool]
        sample_ids = [row["sample_id"] for row in pool_rows]
        if len(pool_rows) != expected_n or len(set(sample_ids)) != expected_n:
            raise RuntimeError(
                f"C1 pool={pool} rows={len(pool_rows)} unique={len(set(sample_ids))} "
                f"expected={expected_n}"
            )
        for row in pool_rows:
            if not all(row[key] for key in required):
                raise RuntimeError(f"C1 missing required value for {row['sample_id']} in {pool}")
            count = int(row["candidate_count"])
            for condition in [
                "true", "copy", "zero", "batch_roll", "random_valid",
                "matched_magnitude", "direction_flip", "rotation_direction_flip",
                "translation_direction_flip",
            ]:
                rank = int(row[f"{condition}_target_rank"])
                if rank < 0 or rank >= count:
                    raise RuntimeError(
                        f"C1 rank outside pool for {row['sample_id']} {pool} {condition}"
                    )
                if int(row[f"{condition}_r1"]) != int(rank == 0):
                    raise RuntimeError(f"C1 R@1/rank conflict for {row['sample_id']} {condition}")
            identity = (
                row["manifest_sample_id"], row["scene_id"], row["trajectory_id"],
                row["target_frame_id"], row["true_action_vector"],
            )
            previous = identity_by_id.setdefault(row["sample_id"], identity)
            if previous != identity:
                raise RuntimeError(f"C1 stable ID has conflicting action/target: {row['sample_id']}")


def _validate_b2_sample_rows(rows: Sequence[Dict[str, str]], expected_n: int) -> None:
    required = {
        "sample_id", "scene_id", "trajectory_id", "query_object_id",
        "query_source_reference", "target_object_id", "gt_azimuth", "gt_elevation",
        "gt_log_distance", "predicted_joint_bin", "predicted_azimuth",
        "predicted_elevation", "angular_error_deg", "method",
        "target_visible_frame_indices", "target_currently_hidden",
    }
    missing_columns = sorted(required - set(rows[0]) if rows else required)
    if missing_columns:
        raise RuntimeError(f"B2 sample diagnostics missing columns: {missing_columns}")
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    expected_methods = {
        "FULL_HISTORY_PLUS_QUERY", "HISTORY_WITHOUT_QUERY",
        "CURRENT_FRAME_PLUS_QUERY", "QUERY_FEATURE_ONLY", "FRAME_ORDER_SHUFFLE",
        "REVERSED_HISTORY", "REMOVE_TARGET_VISIBLE_FRAMES",
        "KEEP_TARGET_VISIBLE_FRAMES_ONLY", "GLOBAL_CONSTANT",
        "GLOBAL_ANGLE_HISTOGRAM",
    }
    if not expected_methods.issubset(grouped):
        raise RuntimeError(f"B2 missing methods: {sorted(expected_methods - set(grouped))}")
    base_rows = grouped["FULL_HISTORY_PLUS_QUERY"]
    base_ids = {row["sample_id"] for row in base_rows}
    if len(base_rows) != expected_n or len(base_ids) != expected_n:
        raise RuntimeError(
            f"B2 base rows={len(base_rows)} unique={len(base_ids)} expected={expected_n}"
        )
    for method, method_rows in grouped.items():
        method_ids = [row["sample_id"] for row in method_rows]
        if len(method_ids) != len(set(method_ids)) or not set(method_ids).issubset(base_ids):
            raise RuntimeError(f"B2 unstable IDs for method={method}")
        if method != "SAME_SCENE_QUERY_SWAP" and len(method_rows) != expected_n:
            raise RuntimeError(f"B2 method={method} has {len(method_rows)} rows, expected {expected_n}")
        for row in method_rows:
            if not all(row[key] for key in required):
                raise RuntimeError(f"B2 missing required value for {row['sample_id']} {method}")
            gt_az = float(row["gt_azimuth"])
            gt_el = float(row["gt_elevation"])
            pred_az = float(row["predicted_azimuth"])
            pred_el = float(row["predicted_elevation"])
            cosine = (
                math.sin(pred_el) * math.sin(gt_el)
                + math.cos(pred_el) * math.cos(gt_el) * math.cos(pred_az - gt_az)
            )
            recomputed = math.degrees(math.acos(min(1.0, max(-1.0, cosine))))
            if abs(recomputed - float(row["angular_error_deg"])) > 1e-7:
                raise RuntimeError(f"B2 angular error is not reproducible for {row['sample_id']}")
            if row["query_object_id"] != row["target_object_id"]:
                raise RuntimeError(f"B2 query/target instance mismatch for {row['sample_id']}")
    for method in ["GLOBAL_CONSTANT", "GLOBAL_ANGLE_HISTOGRAM"]:
        if any(
            row["prior_fit_split"] != "train" or int(row["prior_fit_n"]) <= 0
            for row in grouped[method]
        ):
            raise RuntimeError(f"B2 {method} is not a train-fitted prior")


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _cluster_bootstrap_delta(
    rows: Sequence[Dict[str, str]], left: str, right: str, cluster: str,
    reps: int = 1000, seed: int = PHASE2_SEED,
) -> Tuple[float, float]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row[cluster], []).append(row)
    names = sorted(grouped)
    if not names:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    estimates = []
    for _ in range(reps):
        sampled = [rng.choice(names) for _ in names]
        values = [
            float(row[left]) - float(row[right])
            for name in sampled for row in grouped[name]
        ]
        estimates.append(sum(values) / len(values))
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _b2_method_stats(rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    errors = [float(row["angular_error_deg"]) for row in rows]
    return {
        "n": len(rows),
        "n_scenes": len({row["scene_id"] for row in rows}),
        "n_trajectories": len({row["trajectory_id"] for row in rows}),
        "mean": sum(errors) / len(errors),
        "median": _percentile(errors, 0.5),
        "p25": _percentile(errors, 0.25),
        "p75": _percentile(errors, 0.75),
        "top1": sum(int(row["top1"]) for row in rows) / len(rows),
        "top3": sum(int(row["top3"]) for row in rows) / len(rows),
    }


def _paired_b2_bootstrap(
    method_rows: Sequence[Dict[str, str]], base_by_id: Dict[str, Dict[str, str]],
    cluster: str, reps: int = 1000,
) -> Tuple[float, float]:
    paired = [row for row in method_rows if row["sample_id"] in base_by_id]
    grouped: Dict[str, List[float]] = {}
    for row in paired:
        delta = (
            float(row["angular_error_deg"])
            - float(base_by_id[row["sample_id"]]["angular_error_deg"])
        )
        grouped.setdefault(row[cluster], []).append(delta)
    names = sorted(grouped)
    if not names:
        return float("nan"), float("nan")
    rng = random.Random(PHASE2_SEED + 17)
    estimates = []
    for _ in range(reps):
        sampled = [rng.choice(names) for _ in names]
        values = [value for name in sampled for value in grouped[name]]
        estimates.append(sum(values) / len(values))
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _phase2_c1_rows(sample_rows: Sequence[Dict[str, str]]):
    model = sample_rows[0]["model"]
    spec = MODEL_SPECS[model]
    global_rows = [row for row in sample_rows if row["candidate_pool"] == "HISTORICAL_GLOBAL"]
    conditions = [
        "true", "copy", "zero", "batch_roll", "random_valid",
        "matched_magnitude", "direction_flip", "rotation_direction_flip",
        "translation_direction_flip",
    ]
    output = []
    summaries = {}
    true_r1 = sum(int(row["true_r1"]) for row in global_rows) / len(global_rows)
    copy_r1 = sum(int(row["copy_r1"]) for row in global_rows) / len(global_rows)
    zero_r1 = sum(int(row["zero_r1"]) for row in global_rows) / len(global_rows)
    roll_r1 = sum(int(row["batch_roll_r1"]) for row in global_rows) / len(global_rows)
    for pool in sorted({row["candidate_pool"] for row in sample_rows}):
        pool_rows = [row for row in sample_rows if row["candidate_pool"] == pool]
        summaries[pool] = {}
        for condition in conditions:
            r1 = sum(int(row[f"{condition}_r1"]) for row in pool_rows) / len(pool_rows)
            target_cosine = sum(float(row[f"{condition}_target_cosine"]) for row in pool_rows) / len(pool_rows)
            mean_rank = sum(float(row[f"{condition}_target_rank"]) for row in pool_rows) / len(pool_rows)
            summaries[pool][condition] = {"r1": r1, "cosine": target_cosine, "rank": mean_rank}
        pool_copy_r1 = summaries[pool]["copy"]["r1"]
        pool_zero_r1 = summaries[pool]["zero"]["r1"]
        pool_roll_r1 = summaries[pool]["batch_roll"]["r1"]
        pool_matched_r1 = summaries[pool]["matched_magnitude"]["r1"]
        for condition in conditions:
            stats = summaries[pool][condition]
            r1 = stats["r1"]
            ci_low = ci_high = trajectory_ci_low = trajectory_ci_high = ""
            if condition == "true":
                ci_low, ci_high = _cluster_bootstrap_delta(
                    pool_rows, "true_r1", "matched_magnitude_r1", "scene_id"
                )
                trajectory_ci_low, trajectory_ci_high = _cluster_bootstrap_delta(
                    pool_rows, "true_r1", "matched_magnitude_r1", "trajectory_id"
                )
            output.append({
                "experiment_id": f"phase2_c1_{model}_l{spec['layer']}_p12",
                "model": model, "objective_family": spec["objective_family"],
                "layer": spec["layer"], "prefix": 12,
                "analysis_type": "phase2_paired_candidate_pool",
                "subset": "ALL_VAL", "candidate_pool": pool,
                "action_condition": condition, "n": len(pool_rows),
                "r1": fmt(r1), "mrr_or_rank": fmt(stats["rank"]),
                "target_cosine": fmt(stats["cosine"]), "copy_r1": fmt(pool_copy_r1),
                "no_action_r1": fmt(pool_zero_r1), "shuffled_action_r1": fmt(pool_roll_r1),
                "matched_wrong_r1": fmt(pool_matched_r1),
                "true_minus_matched": fmt(
                    summaries[pool]["true"]["r1"] - pool_matched_r1
                ),
                "delta_vs_copy": fmt(r1 - pool_copy_r1),
                "delta_vs_no_action": fmt(r1 - pool_zero_r1),
                "delta_vs_shuffle": fmt(r1 - pool_roll_r1),
                "delta_vs_best_control": fmt(
                    r1 - max(pool_copy_r1, pool_zero_r1, pool_roll_r1, pool_matched_r1)
                ),
                "normalized_action_gain": "",
                "fixed_predictor_action_margin": fmt(
                    sum(float(row["action_margin_vs_matched_cosine"]) for row in pool_rows)
                    / len(pool_rows)
                ),
                "ci_low": fmt(ci_low) if ci_low != "" else "",
                "ci_high": fmt(ci_high) if ci_high != "" else "",
                "trajectory_ci_low": fmt(trajectory_ci_low) if trajectory_ci_low != "" else "",
                "trajectory_ci_high": fmt(trajectory_ci_high) if trajectory_ci_high != "" else "",
                "n_scenes": len({row["scene_id"] for row in pool_rows}),
                "n_trajectories": len({row["trajectory_id"] for row in pool_rows}),
                "c3_intervention_validity": "", "c3_intervention_margin": "",
                "artifact_source": "analyses/probe_validity/c1_sample_diagnostics.csv.gz",
                "evidence_level": (
                    "SUPPORTED_CLUSTERED_BOOTSTRAP" if condition == "true"
                    else "CONFIRMED_PAIRED_SAMPLE_EVIDENCE"
                ),
                "notes": f"{model} p12/l{spec['layer']} fixed checkpoint; all conditions share the same context, target, and candidate pool.",
            })

    boundaries = {}
    motion_summaries = {}
    def stratification_value(row: Dict[str, str], variable: str) -> float:
        if variable == "total_displacement":
            return math.hypot(
                float(row["translation_magnitude"]),
                float(row["rotation_magnitude_rad"]),
            )
        return float(row[variable])

    for variable in [
        "translation_magnitude", "rotation_magnitude_rad",
        "total_displacement", "feature_drift",
    ]:
        values = [stratification_value(row, variable) for row in global_rows]
        q1, q2 = _percentile(values, 1 / 3), _percentile(values, 2 / 3)
        boundaries[variable] = [q1, q2]
        motion_summaries[variable] = {}
        for label, predicate in [
            ("LOW", lambda value: value <= q1),
            ("MEDIUM", lambda value: q1 < value <= q2),
            ("HIGH", lambda value: value > q2),
        ]:
            selected = [
                row for row in global_rows
                if predicate(stratification_value(row, variable))
            ]
            gain = sum(int(row["true_r1"]) - int(row["matched_magnitude_r1"]) for row in selected) / len(selected)
            margin = sum(float(row["action_margin_vs_matched_cosine"]) for row in selected) / len(selected)
            positive_margin = sum(float(row["action_margin_vs_matched_cosine"]) > 0 for row in selected) / len(selected)
            ci_low, ci_high = _cluster_bootstrap_delta(
                selected, "true_r1", "matched_magnitude_r1", "scene_id"
            )
            trajectory_ci_low, trajectory_ci_high = _cluster_bootstrap_delta(
                selected, "true_r1", "matched_magnitude_r1", "trajectory_id"
            )
            matched_r1 = sum(int(row["matched_magnitude_r1"]) for row in selected) / len(selected)
            motion_summaries[variable][label] = {
                "n": len(selected), "gain": gain, "margin": margin,
                "positive_margin_rate": positive_margin,
                "scene_ci": [ci_low, ci_high],
                "trajectory_ci": [trajectory_ci_low, trajectory_ci_high],
            }
            output.append({
                "experiment_id": f"phase2_c1_{model}_l{spec['layer']}_p12",
                "model": model, "objective_family": spec["objective_family"],
                "layer": spec["layer"], "prefix": 12,
                "analysis_type": f"phase2_{variable}_tertile",
                "subset": label, "candidate_pool": "HISTORICAL_GLOBAL",
                "action_condition": "true_vs_matched_magnitude", "n": len(selected),
                "r1": fmt(sum(int(row["true_r1"]) for row in selected) / len(selected)),
                "mrr_or_rank": "",
                "target_cosine": fmt(sum(float(row["true_target_cosine"]) for row in selected) / len(selected)),
                "copy_r1": fmt(sum(int(row["copy_r1"]) for row in selected) / len(selected)),
                "no_action_r1": fmt(sum(int(row["zero_r1"]) for row in selected) / len(selected)),
                "shuffled_action_r1": fmt(sum(int(row["batch_roll_r1"]) for row in selected) / len(selected)),
                "matched_wrong_r1": fmt(matched_r1),
                "true_minus_matched": fmt(gain),
                "delta_vs_copy": fmt(sum(int(row["true_r1"]) - int(row["copy_r1"]) for row in selected) / len(selected)),
                "delta_vs_no_action": fmt(sum(int(row["true_r1"]) - int(row["zero_r1"]) for row in selected) / len(selected)),
                "delta_vs_shuffle": fmt(sum(int(row["true_r1"]) - int(row["batch_roll_r1"]) for row in selected) / len(selected)),
                "delta_vs_best_control": fmt(gain), "normalized_action_gain": "",
                "fixed_predictor_action_margin": fmt(margin),
                "ci_low": fmt(ci_low), "ci_high": fmt(ci_high),
                "trajectory_ci_low": fmt(trajectory_ci_low),
                "trajectory_ci_high": fmt(trajectory_ci_high),
                "n_scenes": len({row["scene_id"] for row in selected}),
                "n_trajectories": len({row["trajectory_id"] for row in selected}),
                "c3_intervention_validity": "", "c3_intervention_margin": "",
                "artifact_source": "analyses/probe_validity/c1_sample_diagnostics.csv.gz",
                "evidence_level": "SUPPORTED_CLUSTERED_BOOTSTRAP",
                "notes": (
                    f"Fixed tertiles q1={q1:.8g}, q2={q2:.8g}; positive cosine-margin rate={positive_margin:.6g}; "
                    f"scene CI=[{ci_low:.6g},{ci_high:.6g}]; trajectory CI=[{trajectory_ci_low:.6g},{trajectory_ci_high:.6g}]."
                ),
            })
    return output, {
        "model": model, "display": spec["display"],
        "objective_family": spec["objective_family"], "layer": spec["layer"],
        "n": len(global_rows), "true_r1": true_r1, "copy_r1": copy_r1,
        "zero_r1": zero_r1, "batch_roll_r1": roll_r1,
        "summaries": summaries, "boundaries": boundaries,
        "motion": motion_summaries,
        "mean_matched_cosine_margin": sum(
            float(row["action_margin_vs_matched_cosine"]) for row in global_rows
        ) / len(global_rows),
        "positive_matched_cosine_margin_rate": sum(
            float(row["action_margin_vs_matched_cosine"]) > 0 for row in global_rows
        ) / len(global_rows),
    }


def _phase2_b2_rows(sample_rows: Sequence[Dict[str, str]]):
    model = sample_rows[0]["model"]
    spec = MODEL_SPECS[model]
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in sample_rows:
        grouped.setdefault(row["method"], []).append(row)
    base_rows = grouped["FULL_HISTORY_PLUS_QUERY"]
    base_by_id = {row["sample_id"]: row for row in base_rows}
    base_stats = _b2_method_stats(base_rows)
    valid_prior_methods = [name for name in ["GLOBAL_CONSTANT", "GLOBAL_ANGLE_HISTOGRAM"] if name in grouped]
    prior_stats = {name: _b2_method_stats(grouped[name]) for name in valid_prior_methods}
    best_prior = min(valid_prior_methods, key=lambda name: prior_stats[name]["mean"])
    output = []
    summaries = {}
    for method in sorted(grouped):
        rows = grouped[method]
        stats = _b2_method_stats(rows)
        summaries[method] = stats
        ci_low, ci_high = _paired_b2_bootstrap(rows, base_by_id, "scene_id")
        trajectory_ci_low, trajectory_ci_high = _paired_b2_bootstrap(
            rows, base_by_id, "trajectory_id"
        )
        output.append({
            "experiment_id": f"phase2_b2_{model}_l{spec['layer']}_p12",
            "sample_subset": "ALL_VALID_HIDDEN_OBJECTS",
            "model": model, "objective_family": spec["objective_family"],
            "layer": spec["layer"], "prefix": 12,
            "analysis_type": "phase2_paired_method",
            "baseline_or_readout": method,
            "perturbation": rows[0]["query_perturbation"] + "+" + rows[0]["history_perturbation"],
            "seed": PHASE2_SEED, "n": stats["n"],
            "mean_ang_err_deg": fmt(stats["mean"]),
            "median_ang_err_deg": fmt(stats["median"]),
            "p25_ang_err_deg": fmt(stats["p25"]),
            "p75_ang_err_deg": fmt(stats["p75"]),
            "top1": fmt(stats["top1"]), "top3": fmt(stats["top3"]),
            "delta_error_vs_full": fmt(stats["mean"] - base_stats["mean"]),
            "delta_error_vs_best_prior": fmt(stats["mean"] - prior_stats[best_prior]["mean"]),
            "ci_low": fmt(ci_low), "ci_high": fmt(ci_high),
            "trajectory_ci_low": fmt(trajectory_ci_low),
            "trajectory_ci_high": fmt(trajectory_ci_high),
            "n_scenes": stats["n_scenes"], "n_trajectories": stats["n_trajectories"],
            "query_unique_rate": fmt(sum(row["identifiability"] == "UNIQUE_SEEN_THEN_HIDDEN" for row in base_rows) / len(base_rows)),
            "target_seen_rate": fmt(sum(int(row["target_ever_visible_in_prefix"]) for row in base_rows) / len(base_rows)),
            "ambiguous_rate": "", "never_seen_rate": fmt(sum(not int(row["target_ever_visible_in_prefix"]) for row in base_rows) / len(base_rows)),
            "artifact_source": "analyses/probe_validity/b2_sample_diagnostics.csv.gz",
            "evidence_level": (
                "TENTATIVE_SMALL_SUBSET" if rows[0]["baseline_type"] == "TRAIN_FITTED_PRIOR"
                else "SUPPORTED_CLUSTERED_BOOTSTRAP"
            ),
            "notes": (
                "Paired scene/trajectory-clustered CIs are for angular-error increase "
                "versus full history + correct query; trajectory_id equals scene_id in this manifest."
            ),
        })
    identifiable_rate = sum(
        row["identifiability"] == "UNIQUE_SEEN_THEN_HIDDEN" for row in base_rows
    ) / len(base_rows)
    return output, {
        "model": model, "display": spec["display"],
        "objective_family": spec["objective_family"], "layer": spec["layer"],
        "n": len(base_rows), "base": base_stats, "methods": summaries,
        "best_prior": best_prior, "best_prior_stats": prior_stats[best_prior],
        "identifiable_rate": identifiable_rate,
        "history_gain": summaries["CURRENT_FRAME_PLUS_QUERY"]["mean"] - base_stats["mean"],
        "query_gain": summaries["HISTORY_WITHOUT_QUERY"]["mean"] - base_stats["mean"],
        "prior_gain": prior_stats[best_prior]["mean"] - base_stats["mean"],
        "query_feature_only_error": summaries["QUERY_FEATURE_ONLY"]["mean"],
        "same_scene_query_swap_degradation": (
            summaries.get("SAME_SCENE_QUERY_SWAP", base_stats)["mean"] - base_stats["mean"]
        ),
        "shuffle_degradation": summaries["FRAME_ORDER_SHUFFLE"]["mean"] - base_stats["mean"],
        "reverse_degradation": summaries["REVERSED_HISTORY"]["mean"] - base_stats["mean"],
        "remove_visible_degradation": summaries["REMOVE_TARGET_VISIBLE_FRAMES"]["mean"] - base_stats["mean"],
        "visible_only_degradation": summaries["KEEP_TARGET_VISIBLE_FRAMES_ONLY"]["mean"] - base_stats["mean"],
    }


def _phase2_plots(
    c1_summaries: Dict[str, Dict[str, object]],
    b2_summaries: Dict[str, Dict[str, object]],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/probe_validity_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    order = [model for model in [*CORE_MODELS, "f3r"] if model in c1_summaries]
    labels = [MODEL_SPECS[model]["display"] for model in order]
    colors = {
        "vjepa2": "#198F8F", "videomae": "#D55E48", "wan": "#2878B5",
        "cogvideox": "#E3A72F", "f3r": "#8A8A8A",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7), constrained_layout=True)
    for model in order:
        summary = c1_summaries[model]
        same_scene = summary["summaries"]["SAME_SCENE"]
        gain = same_scene["true"]["r1"] - same_scene["matched_magnitude"]["r1"]
        axes[0].scatter(summary["true_r1"], gain, s=105, color=colors[model], zorder=3)
        axes[0].annotate(
            MODEL_SPECS[model]["display"], (summary["true_r1"], gain),
            xytext=(5, 6), textcoords="offset points", fontsize=9,
        )
    axes[0].axhline(0, color="#555555", linewidth=0.8)
    axes[0].set_xlabel("Absolute true R@1")
    axes[0].set_ylabel("Same-scene matched-action gain")
    axes[0].set_title("A. Predictability vs action specificity", loc="left", fontweight="bold")
    axes[0].grid(alpha=0.2)

    metrics = ["Matched", "High motion", "High drift", "Translation flip"]
    values = []
    for model in order:
        summary = c1_summaries[model]
        global_pool = summary["summaries"]["HISTORICAL_GLOBAL"]
        values.append([
            global_pool["true"]["r1"] - global_pool["matched_magnitude"]["r1"],
            summary["motion"]["translation_magnitude"]["HIGH"]["gain"],
            summary["motion"]["feature_drift"]["HIGH"]["gain"],
            global_pool["true"]["r1"] - global_pool["translation_direction_flip"]["r1"],
        ])
    x = np.arange(len(order))
    width = 0.18
    metric_colors = ["#198F8F", "#D55E48", "#2878B5", "#E3A72F"]
    for index, metric in enumerate(metrics):
        axes[1].bar(
            x + (index - 1.5) * width,
            [row[index] for row in values], width, label=metric,
            color=metric_colors[index],
        )
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylabel("True R@1 minus control R@1")
    axes[1].set_title("B. Intervention-sensitive gains", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].grid(axis="y", alpha=0.2)
    fig.savefig(OUT / "c1_validity.png", dpi=180)
    plt.close(fig)

    order = [model for model in [*CORE_MODELS, "f3r"] if model in b2_summaries]
    labels = [MODEL_SPECS[model]["display"] for model in order]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7), constrained_layout=True)
    absolute_methods = [
        ("Full", "FULL_HISTORY_PLUS_QUERY"),
        ("Global prior", None),
        ("Query only", "QUERY_FEATURE_ONLY"),
        ("Current frame", "CURRENT_FRAME_PLUS_QUERY"),
    ]
    x = np.arange(len(order))
    width = 0.19
    for index, (label, method) in enumerate(absolute_methods):
        values = []
        for model in order:
            summary = b2_summaries[model]
            values.append(
                summary["best_prior_stats"]["mean"]
                if method is None else summary["methods"][method]["mean"]
            )
        axes[0].bar(
            x + (index - 1.5) * width, values, width, label=label,
            color=metric_colors[index],
        )
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylabel("Mean angular error (degrees; lower is better)")
    axes[0].set_title("A. Full model and content controls", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[0].grid(axis="y", alpha=0.2)

    degradations = [
        ("Query mask", "query_gain"), ("Query swap", "same_scene_query_swap_degradation"),
        ("Remove visible", "remove_visible_degradation"),
        ("Shuffle", "shuffle_degradation"), ("Reverse", "reverse_degradation"),
    ]
    width = 0.15
    degradation_colors = ["#198F8F", "#D55E48", "#2878B5", "#E3A72F", "#7A68A6"]
    for index, (label, key) in enumerate(degradations):
        axes[1].bar(
            x + (index - 2) * width,
            [b2_summaries[model][key] for model in order],
            width, label=label, color=degradation_colors[index],
        )
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylabel("Angular-error increase vs full (degrees)")
    axes[1].set_title("B. Query, visibility, and order reliance", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].grid(axis="y", alpha=0.2)
    fig.savefig(OUT / "b2_validity.png", dpi=180)
    plt.close(fig)


def _phase2_report(c1s, b2s, c1_smoke, b2_smoke, before, after, c1_rows, b2_rows) -> None:
    protected_ok = before == after
    matched_gain = (
        c1s["summaries"]["HISTORICAL_GLOBAL"]["true"]["r1"]
        - c1s["summaries"]["HISTORICAL_GLOBAL"]["matched_magnitude"]["r1"]
    )
    same_scene_gain = (
        c1s["summaries"]["SAME_SCENE"]["true"]["r1"]
        - c1s["summaries"]["SAME_SCENE"]["matched_magnitude"]["r1"]
    )
    same_scene_translation_flip = c1s["summaries"]["SAME_SCENE"][
        "translation_direction_flip"
    ]["r1"]
    if matched_gain > 0 and same_scene_gain > 0:
        c1_verdict = "KEEP_WITH_HARD_NEGATIVE_PRIMARY"
    else:
        c1_verdict = "KEEP_BUT_REINTERPRET_AS_PREDICTABILITY"
    if b2s["prior_gain"] > 0 and b2s["query_gain"] > 0 and b2s["history_gain"] > 0:
        b2_verdict = "KEEP_WITH_PRIOR_CONTROLS"
    elif b2s["identifiable_rate"] < 1:
        b2_verdict = "KEEP_ONLY_IDENTIFIABLE_SUBSET"
    else:
        b2_verdict = "INSUFFICIENT_SAMPLE_EVIDENCE"

    high_motion = c1s["motion"]["translation_magnitude"]["HIGH"]
    high_total_motion = c1s["motion"]["total_displacement"]["HIGH"]
    high_feature_drift = c1s["motion"]["feature_drift"]["HIGH"]
    hash_lines = []
    for path, metadata in before.items():
        status = "OK" if metadata == after[path] else "CHANGED"
        hash_lines.append(
            f"- `{path}` size={metadata['size']} mtime={metadata['mtime']} "
            f"sha256={metadata['sha256']} {status}"
        )
    report_text = f"""# C1/B2 Probe Validity Study - Sample-Level Phase 2

Status: `PHASE_2_PARTIAL`

Stage 1 is complete for the fixed F3R default-layer, prefix-12 smoke cases. Stage 2 representative-model and Stage 3 default8 expansion remain incomplete, so F3R results are paired sample evidence but not a model-family conclusion. Unmet expansion gates are DINO, V-JEPA2, a fourth representative model, and the default8 x four-prefix matrix.

# A. Environment

- Environment: `/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial`
- Python: 3.11.15; Torch: 2.6.0+cu124; Hydra: 1.3.2; Lightning: 2.6.1; CUDA runtime: 12.4.
- Replay device: one isolated NVIDIA H800 allocated through SCO after the local node reported no GPU.
- Environment, AOSS, and GPU gates: PASS.

# B. Replay Reproduction

- C1 F3R/l24/p12: n={c1_smoke['n']}; cosine and recall reproduce exactly within `1e-5`. The four mean-rank values differ by at most {c1_smoke['max_absolute_difference']:.9g} (one rank position over 276 samples), accepted only for historical GPU `argsort` ordering of exact ties.
- C1 checkpoint compatibility: the concurrently updated full-conditioning class adds `constant_context` and `constant_action`; these are unused on the full path and are the only allowlisted missing historical keys. All other missing/unexpected keys remain fatal.
- B2 F3R/l24/p12: n={b2_smoke['n']}; maximum absolute historical aggregate difference={b2_smoke['max_absolute_difference']:.9g}; PASS at tolerance 1e-5.
- B2 checkpoint compatibility: the concurrently updated full-context class adds one unused `constant_context`; it is the only allowlisted missing historical key. All other missing/unexpected keys remain fatal.

# C. Sample Coverage

- C1: {c1s['n']} unique samples, {len(c1_rows)} diagnostic rows, eight candidate pools, fixed checkpoint `f3r/l24/p12`; every pool contains one row per stable sample ID.
- B2: {b2s['n']} unique full-history samples, {len(b2_rows)} condition/prior rows, fixed checkpoint `f3r/l24/p12`.
- C1 sample IDs use `model|layer|prefix|split|manifest sample/window|target frame`. B2 adds hidden object ID. Duplicate stable IDs are disallowed within each base condition.
- C1/C3 sample relation: `NOT_TESTABLE_MISSING_C3_RUNTIME_OR_ARTIFACT`; aggregate C1/C3 rows were not used as sample evidence.
- Not covered: DINO, V-JEPA2, and a fourth representative model; default8/default-layer x four-prefix expansion; B2 category/layout/trajectory labels and their fitted priors.

# D. C1 Findings

- Historical global pool true R@1={c1s['true_r1']:.6f}; copy={c1s['copy_r1']:.6f}; zero={c1s['zero_r1']:.6f}; batch-roll={c1s['batch_roll_r1']:.6f}.
- True minus magnitude-matched wrong-action R@1={matched_gain:.6f}. Mean paired cosine margin={c1s['mean_matched_cosine_margin']:.9g}; positive-margin rate={c1s['positive_matched_cosine_margin_rate']:.6f}. Evidence: `CONFIRMED_PAIRED_SAMPLE_EVIDENCE`.
- Same-scene hard-negative true-minus-matched R@1={same_scene_gain:.6f}. The pool has four candidates per scene and uses identical targets for all action conditions.
- Correct action does not dominate every intervention: same-scene translation-direction-flip R@1={same_scene_translation_flip:.6f}, versus true={c1s['summaries']['SAME_SCENE']['true']['r1']:.6f}. Directional sensitivity is therefore mixed despite the matched-wrong gain.
- High-translation tertile true-minus-matched R@1={high_motion['gain']:.6f}, scene-clustered 95% CI=[{high_motion['scene_ci'][0]:.6f}, {high_motion['scene_ci'][1]:.6f}], trajectory-clustered 95% CI=[{high_motion['trajectory_ci'][0]:.6f}, {high_motion['trajectory_ci'][1]:.6f}], n={high_motion['n']}. Scene and trajectory IDs coincide in this manifest. Evidence: `SUPPORTED_CLUSTERED_BOOTSTRAP`.
- High total-displacement gain={high_total_motion['gain']:.6f}, scene CI=[{high_total_motion['scene_ci'][0]:.6f}, {high_total_motion['scene_ci'][1]:.6f}]; high feature-drift gain={high_feature_drift['gain']:.6f}, scene CI=[{high_feature_drift['scene_ci'][0]:.6f}, {high_feature_drift['scene_ci'][1]:.6f}]. The action gain is not stable across motion/drift definitions.
- Verdict: `{c1_verdict}`. This verdict is F3R-smoke scoped.

# E. B2 Findings

- Identifiable rate under exact instance-ID visibility metadata: {b2s['identifiable_rate']:.6f}. Category-level multiplicity remains unavailable, so this is instance-ID identifiability only.
- Full history + correct query mean angular error={b2s['base']['mean']:.6f} deg, top1={b2s['base']['top1']:.6f}, top3={b2s['base']['top3']:.6f}.
- Best valid train-only prior: `{b2s['best_prior']}`, mean error={b2s['best_prior_stats']['mean']:.6f} deg; full-history gain over prior={b2s['prior_gain']:.6f} deg. Prior fit uses train labels only and is marked `TENTATIVE_SMALL_SUBSET`.
- Query-mask degradation={b2s['query_gain']:.6f} deg; same-scene real-query swap degradation={b2s['same_scene_query_swap_degradation']:.6f} deg; query-feature-only error={b2s['query_feature_only_error']:.6f} deg.
- Current-frame-only degradation={b2s['history_gain']:.6f} deg; frame-order-shuffle={b2s['shuffle_degradation']:.6f} deg; reversed-history={b2s['reverse_degradation']:.6f} deg; remove-visible-frame={b2s['remove_visible_degradation']:.6f} deg; keep-visible-only={b2s['visible_only_degradation']:.6f} deg.
- Shuffle and reversal do not degrade this checkpoint, so the sample evidence supports use of historical content and target-visible frames more clearly than temporal-order sensitivity.
- Query category and scene/layout annotations are absent from the evaluator batch, so category/layout priors and ambiguity-by-category are `NOT_TESTABLE_MISSING_ARTIFACT`; the instance-ID audit is not presented as category-level identifiability.
- Verdict: `{b2_verdict}`. This verdict is F3R-smoke scoped.

# Evidence Levels

- Replay reproduction and fixed-checkpoint per-sample comparisons: `CONFIRMED_PAIRED_SAMPLE_EVIDENCE`.
- Scene/trajectory resampling intervals: `SUPPORTED_CLUSTERED_BOOTSTRAP`.
- Train-only 128-label global prior and single-model verdicts: `TENTATIVE_SMALL_SUBSET`.
- Previous default8 table conclusions remain `SUPPORTED_AGGREGATE_ONLY`.
- C1xC3 sample matching and category/layout priors: `NOT_TESTABLE_MISSING_ARTIFACT`.

# F. Safety

- No backbone feature extraction; no C1/B2 retraining; no large multi-GPU training.
- Existing checkpoints and pre-extracted features were read-only.
- Active evaluation directories were untouched; Phase 2 temporary files stayed under `analyses/probe_validity/tmp/`.
- Protected files byte-identical: `{protected_ok}`. No canonical update and no registry update.

{chr(10).join(hash_lines)}

# G. Reproducibility

```bash
PYTHONPATH="$PWD" /mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python \\
  analyses/probe_validity/run_probe_validity.py --phase2-replay all
PYTHONPATH="$PWD" /mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python \\
  analyses/probe_validity/run_probe_validity.py --phase2-summarize
```

Random seed: {PHASE2_SEED}. C1 wrong actions are a global derangement, a nearest action-magnitude match, valid inverse relative pose, inverse rotation, and translation sign flip. B2 priors are fit only on train-split targets; query/history perturbations keep the original GT target fixed.
"""
    (OUT / "PROBE_VALIDITY_REPORT.md").write_text(report_text, encoding="utf-8")


def _cross_objective_report(
    c1_summaries: Dict[str, Dict[str, object]],
    b2_summaries: Dict[str, Dict[str, object]],
    replay_summaries: Dict[str, Dict[str, object]],
    before: Dict[str, Dict[str, object]],
    after: Dict[str, Dict[str, object]],
    c1_rows: Sequence[Dict[str, str]],
    b2_rows: Sequence[Dict[str, str]],
) -> None:
    order = [model for model in [*CORE_MODELS, "f3r"] if model in c1_summaries and model in b2_summaries]

    def c1_metrics(model: str) -> Dict[str, float]:
        summary = c1_summaries[model]
        global_pool = summary["summaries"]["HISTORICAL_GLOBAL"]
        same_scene = summary["summaries"]["SAME_SCENE"]
        return {
            "true": summary["true_r1"],
            "copy": summary["copy_r1"],
            "matched": global_pool["true"]["r1"] - global_pool["matched_magnitude"]["r1"],
            "same_scene": same_scene["true"]["r1"] - same_scene["matched_magnitude"]["r1"],
            "high_motion": summary["motion"]["translation_magnitude"]["HIGH"]["gain"],
            "high_drift": summary["motion"]["feature_drift"]["HIGH"]["gain"],
            "translation_flip": global_pool["true"]["r1"] - global_pool["translation_direction_flip"]["r1"],
        }

    c1m = {model: c1_metrics(model) for model in order}
    b2m = {
        model: {
            "full": b2_summaries[model]["base"]["mean"],
            "prior": b2_summaries[model]["best_prior_stats"]["mean"],
            "query_mask": b2_summaries[model]["query_gain"],
            "query_swap": b2_summaries[model]["same_scene_query_swap_degradation"],
            "current": b2_summaries[model]["history_gain"],
            "remove": b2_summaries[model]["remove_visible_degradation"],
            "shuffle": b2_summaries[model]["shuffle_degradation"],
            "reverse": b2_summaries[model]["reverse_degradation"],
            "query_only": b2_summaries[model]["query_feature_only_error"],
        }
        for model in order
    }

    core = [model for model in CORE_MODELS if model in c1m]
    c1_range = max(c1m[model]["same_scene"] for model in core) - min(
        c1m[model]["same_scene"] for model in core
    )
    c1_verdict = (
        "OBJECTIVE_DEPENDENT_SIGNAL" if c1_range >= 0.02
        else "KEEP_WITH_HARD_NEGATIVE_PRIMARY"
    )
    order_effects = {
        model: (b2m[model]["shuffle"] + b2m[model]["reverse"]) / 2 for model in core
    }
    max_absolute_order_effect = max(abs(value) for value in order_effects.values())
    if max_absolute_order_effect < 2.5:
        b2_verdict = "REINTERPRET_AS_HISTORY_CONTENT_RETRIEVAL"
    elif max(order_effects.values()) - min(order_effects.values()) >= 2.0:
        b2_verdict = "OBJECTIVE_DEPENDENT_TEMPORAL_MEMORY"
    else:
        b2_verdict = "KEEP_AS_REFERENCE_CONDITIONED_MEMORY"

    predictability_order = sorted(core, key=lambda model: c1m[model]["true"], reverse=True)
    action_order = sorted(core, key=lambda model: c1m[model]["same_scene"], reverse=True)
    ranks_dissociate = predictability_order != action_order
    generation_same_direction = (
        c1m["wan"]["same_scene"] * c1m["cogvideox"]["same_scene"] > 0
        if "wan" in c1m and "cogvideox" in c1m else False
    )
    generation_b2_consistent = (
        b2m["wan"]["query_mask"] * b2m["cogvideox"]["query_mask"] > 0
        and b2m["wan"]["current"] * b2m["cogvideox"]["current"] > 0
        if "wan" in b2m and "cogvideox" in b2m else False
    )
    vj_vs_vm = c1m["vjepa2"]["same_scene"] - c1m["videomae"]["same_scene"]
    vj_vs_vm_matched = c1m["vjepa2"]["matched"] - c1m["videomae"]["matched"]
    vj_vs_vm_high_motion = c1m["vjepa2"]["high_motion"] - c1m["videomae"]["high_motion"]
    vj_order_advantage = order_effects["vjepa2"] - order_effects["videomae"]

    c1_table = [
        "| Model | Objective | True R@1 | Copy | Matched gain | Same-scene gain | High-motion gain | High-drift gain | True - translation flip |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    b2_table = [
        "| Model | Objective | Full error | Global prior | Query mask delta | Query swap delta | Current-frame delta | Remove-visible delta | Shuffle delta | Reverse delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in order:
        spec = MODEL_SPECS[model]
        c = c1m[model]
        b = b2m[model]
        c1_table.append(
            f"| {spec['display']} | {spec['objective_family']} | {c['true']:.4f} | {c['copy']:.4f} | "
            f"{c['matched']:+.4f} | {c['same_scene']:+.4f} | {c['high_motion']:+.4f} | "
            f"{c['high_drift']:+.4f} | {c['translation_flip']:+.4f} |"
        )
        b2_table.append(
            f"| {spec['display']} | {spec['objective_family']} | {b['full']:.2f} | {b['prior']:.2f} | "
            f"{b['query_mask']:+.2f} | {b['query_swap']:+.2f} | {b['current']:+.2f} | "
            f"{b['remove']:+.2f} | {b['shuffle']:+.2f} | {b['reverse']:+.2f} |"
        )

    replay_lines = []
    for model in core:
        replay = replay_summaries[model]
        replay_lines.append(
            f"- {MODEL_SPECS[model]['display']}: C1 n={replay['c1']['n']}, max diff="
            f"{replay['c1']['max_absolute_difference']:.3g}; B2 n={replay['b2']['n']}, "
            f"max diff={replay['b2']['max_absolute_difference']:.3g}; `PASS`."
        )

    surprises = [
        "Absolute predictability and same-scene action specificity dissociate: Wan has the best true R@1, while V-JEPA2 has the largest same-scene hard-negative gain.",
        f"Ordered temporal updating is not supported: every core model's mean shuffle/reverse effect is below 2.5 degrees (largest={max_absolute_order_effect:.2f}), and Wan rather than V-JEPA2 is numerically largest.",
        "The generation family is directionally coherent for C1 action specificity but fractures for B2 query/history dependence: Wan and CogVideoX show opposite query-mask and current-frame effects.",
    ]
    surprise_lines = [f"{index}. {value}" for index, value in enumerate(surprises, 1)]

    hash_lines = []
    for path, metadata in before.items():
        status = "OK" if metadata == after[path] else "CHANGED"
        hash_lines.append(f"- `{path}` sha256={metadata['sha256']} {status}")

    report_text = f"""# C1/B2 Probe Validity Study - Cross-Objective Sample-Level Phase 2

Status: `PHASE_2_CROSS_OBJECTIVE_COMPLETE`

Fixed protocol: prefix 12; V-JEPA2 L23, VideoMAE L23, Wan L20, CogVideoX L20. F3R L24 is retained only as a geometry-specialist pipeline control. DINO was not analyzed. No backbone extraction, probe retraining, prefix sweep, or layer selection was performed.

# A. Replay Gates

{chr(10).join(replay_lines)}

All four core models passed before any intervention rows were admitted. C1 recall/cosine tolerance was `1e-5`; only the predeclared exact-tie mean-rank allowance was available. B2 maximum metric tolerance was `1e-5`.

# B. Cross-Objective C1

{chr(10).join(c1_table)}

- V-JEPA2 exceeds VideoMAE on same-scene hard-negative gain by {vj_vs_vm:+.4f} and on historical-global matched-magnitude gain by {vj_vs_vm_matched:+.4f}. However, its high-motion gain is {vj_vs_vm_high_motion:+.4f} lower, so the predictive-objective advantage is selective rather than universal. V-JEPA2's translation-direction-flip contrast is {c1m['vjepa2']['translation_flip']:+.4f}, versus {c1m['videomae']['translation_flip']:+.4f} for VideoMAE. Evidence: `TENTATIVE` cross-objective contrast.
- Wan and CogVideoX same-scene gains are {c1m['wan']['same_scene']:+.4f} and {c1m['cogvideox']['same_scene']:+.4f}. Generation-family direction agreement: `{generation_same_direction}`. Evidence: `OBJECTIVE_FAMILY_SUPPORTED` only when this sign agreement is true; otherwise `CONTRADICTED`.
- Absolute-R@1 order: `{', '.join(MODEL_SPECS[m]['display'] for m in predictability_order)}`. Same-scene-gain order: `{', '.join(MODEL_SPECS[m]['display'] for m in action_order)}`. Ranking dissociation: `{ranks_dissociate}`. Evidence: `CROSS_MODEL_REPLICATED` descriptive result.
- High-motion gains are compared under model-internal translation tertiles, with identical intervention construction and candidate pools. These are action-signal robustness diagnostics, not an absolute model leaderboard.
- Protocol verdict: `{c1_verdict}`.

# C. Cross-Objective B2

{chr(10).join(b2_table)}

- Correct-query dependence is measured by query mask and same-scene real-query swap; historical target-content dependence is measured by current-frame-only and remove-visible interventions.
- V-JEPA2 and VideoMAE show strong correct-query dependence: mask/swap deltas are ({b2m['vjepa2']['query_mask']:+.2f}, {b2m['vjepa2']['query_swap']:+.2f}) and ({b2m['videomae']['query_mask']:+.2f}, {b2m['videomae']['query_swap']:+.2f}) degrees. Wan is weaker/mixed ({b2m['wan']['query_mask']:+.2f}, {b2m['wan']['query_swap']:+.2f}); CogVideoX is not robust because masking improves error ({b2m['cogvideox']['query_mask']:+.2f}) while swapping degrades it ({b2m['cogvideox']['query_swap']:+.2f}). Evidence: `CROSS_MODEL_REPLICATED` descriptive pattern.
- Removing target-visible history hurts V-JEPA2, VideoMAE, and Wan by {b2m['vjepa2']['remove']:+.2f}, {b2m['videomae']['remove']:+.2f}, and {b2m['wan']['remove']:+.2f} degrees, but changes CogVideoX by only {b2m['cogvideox']['remove']:+.2f}. Full history beats the train-only prior for V-JEPA2 and VideoMAE, while Wan is slightly worse and CogVideoX substantially worse than the same prior. Evidence: `TENTATIVE_SMALL_SUBSET` for prior comparisons; perturbation results are `CROSS_MODEL_REPLICATED`.
- The generation-family B2 pattern is internally consistent: `{generation_b2_consistent}`. Wan and CogVideoX have opposite query-mask and current-frame signs, so a common generation-objective memory mechanism is `CONTRADICTED`.
- Mean shuffle/reverse degradation: V-JEPA2={order_effects['vjepa2']:+.2f}, VideoMAE={order_effects['videomae']:+.2f}, Wan={order_effects['wan']:+.2f}, CogVideoX={order_effects['cogvideox']:+.2f} degrees. All are below 2.5 degrees in absolute value; V-JEPA2 minus VideoMAE={vj_order_advantage:+.2f}, and Wan is numerically largest. The weak cross-model order effect is `CROSS_MODEL_REPLICATED`; the hypothesis that predictive learning uniquely yields ordered temporal memory is `CONTRADICTED`.
- Exact-instance visibility is audited from instance masks. Category/multiplicity annotation remains `CATEGORY_LEVEL_IDENTIFIABILITY_NOT_AVAILABLE`; no instance-level rate is presented as category-level uniqueness.
- Protocol verdict: `{b2_verdict}`.

# D. Model Role And Scope

- `F3R = GEOMETRY_SPECIALIST_PIPELINE_CONTROL`; its rows are secondary reference evidence and do not determine either verdict. Evidence: `MODEL_SPECIFIC`.
- `DINO = OUT_OF_SCOPE_FOR_PHASE2`; it does not appear in the tables or figures.
- C1 x C3 sample matching: `NOT_TESTABLE_MISSING_C3_ARTIFACT`.
- Core C1 coverage: {sum(1 for row in c1_rows if row['candidate_pool'] == 'HISTORICAL_GLOBAL' and row['model'] in CORE_MODELS)} unique-model sample rows; core B2 full-history coverage: {sum(1 for row in b2_rows if row['method'] == 'FULL_HISTORY_PLUS_QUERY' and row['model'] in CORE_MODELS)}.

# E. Scientific Surprises

{chr(10).join(surprise_lines)}

# F. Safety And Reproducibility

- Environment: `/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial`; replay used one isolated NVIDIA H800 through SCO job `pt-l59q5ufx`.
- Existing checkpoints, manifests, scene metadata, and AOSS features were read-only. Active evaluation, canonical outputs, registry, and ledgers were untouched.
- Protected files byte-identical: `{before == after}`.

{chr(10).join(hash_lines)}

```bash
PYTHONPATH="$PWD" /mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python \\
  analyses/probe_validity/run_probe_validity.py --phase2-summarize
```

Random seed: {PHASE2_SEED}. Sample diagnostics include `model` and `objective_family`; all perturbations preserve the original target and use one fixed protocol across models.
"""
    (OUT / "PROBE_VALIDITY_REPORT.md").write_text(report_text, encoding="utf-8")


def run_phase2_summarize() -> None:
    required = [OUT / "c1_sample_diagnostics.csv.gz", OUT / "b2_sample_diagnostics.csv.gz"]
    for model in CORE_MODELS:
        required.extend([
            TMP / f"c1_{model}_phase2_summary.json",
            TMP / f"b2_{model}_phase2_summary.json",
            TMP / f"phase2_{model}_replay_summary.json",
        ])
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Cannot summarize before all core replay gates pass; missing={missing}")
    before = snapshot(PROTECTED)
    c1_sample_rows = _read_gzip_csv(required[0])
    b2_sample_rows = _read_gzip_csv(required[1])
    models = [*CORE_MODELS, "f3r"]
    c1_models = {row["model"] for row in c1_sample_rows}
    b2_models = {row["model"] for row in b2_sample_rows}
    if set(models) - c1_models or set(models) - b2_models:
        raise RuntimeError(
            f"Unified diagnostics incomplete: c1={sorted(c1_models)} b2={sorted(b2_models)}"
        )

    replay_summaries = {}
    c1_summaries = {}
    b2_summaries = {}
    c1_rows = []
    b2_rows = []
    for model in models:
        model_c1 = [row for row in c1_sample_rows if row["model"] == model]
        model_b2 = [row for row in b2_sample_rows if row["model"] == model]
        expected_c1 = 276
        expected_b2 = 165
        if model in CORE_MODELS:
            c1_smoke = load_json(str(TMP / f"c1_{model}_phase2_summary.json"))
            b2_smoke = load_json(str(TMP / f"b2_{model}_phase2_summary.json"))
            replay = load_json(str(TMP / f"phase2_{model}_replay_summary.json"))
            if c1_smoke.get("gate_status") != "PASS" or c1_smoke.get("gate_failures"):
                raise RuntimeError(f"C1 replay gate is not PASS for {model}")
            if b2_smoke.get("max_absolute_difference", 1) > 1e-5:
                raise RuntimeError(f"B2 replay gate is not PASS for {model}")
            expected_c1 = int(c1_smoke["n"])
            expected_b2 = int(b2_smoke["n"])
            replay_summaries[model] = replay
        _validate_c1_sample_rows(model_c1, expected_c1)
        _validate_b2_sample_rows(model_b2, expected_b2)
        detail_c1, c1_summary = _phase2_c1_rows(model_c1)
        detail_b2, b2_summary = _phase2_b2_rows(model_b2)
        c1_summaries[model] = c1_summary
        b2_summaries[model] = b2_summary
        c1_rows.extend(detail_c1)
        b2_rows.extend(detail_b2)

        global_pool = c1_summary["summaries"]["HISTORICAL_GLOBAL"]
        same_scene = c1_summary["summaries"]["SAME_SCENE"]
        c1_rows.append({
            "experiment_id": f"phase2_c1_{model}_l{MODEL_SPECS[model]['layer']}_p12",
            "model": model, "objective_family": MODEL_SPECS[model]["objective_family"],
            "layer": MODEL_SPECS[model]["layer"], "prefix": 12,
            "analysis_type": "cross_objective_model_summary", "subset": "ALL_VAL",
            "candidate_pool": "CROSS_OBJECTIVE_SUMMARY", "action_condition": "true",
            "n": c1_summary["n"], "global_true_r1": fmt(c1_summary["true_r1"]),
            "global_copy_r1": fmt(c1_summary["copy_r1"]),
            "matched_action_gain": fmt(global_pool["true"]["r1"] - global_pool["matched_magnitude"]["r1"]),
            "same_scene_gain": fmt(same_scene["true"]["r1"] - same_scene["matched_magnitude"]["r1"]),
            "high_motion_gain": fmt(c1_summary["motion"]["translation_magnitude"]["HIGH"]["gain"]),
            "high_drift_gain": fmt(c1_summary["motion"]["feature_drift"]["HIGH"]["gain"]),
            "translation_flip_difference": fmt(global_pool["true"]["r1"] - global_pool["translation_direction_flip"]["r1"]),
            "artifact_source": "analyses/probe_validity/c1_sample_diagnostics.csv.gz",
            "evidence_level": "MODEL_SPECIFIC" if model != "f3r" else "MODEL_SPECIFIC_CONTROL",
            "notes": "Fixed P12/default-layer cross-objective summary; F3R is control only." if model == "f3r" else "Fixed P12/default-layer cross-objective summary.",
        })

        b2_rows.append({
            "experiment_id": f"phase2_b2_{model}_l{MODEL_SPECS[model]['layer']}_p12",
            "sample_subset": "ALL_VALID_HIDDEN_OBJECTS", "model": model,
            "objective_family": MODEL_SPECS[model]["objective_family"],
            "layer": MODEL_SPECS[model]["layer"], "prefix": 12,
            "analysis_type": "cross_objective_model_summary",
            "baseline_or_readout": "FULL_AND_PERTURBATION_SUMMARY", "seed": PHASE2_SEED,
            "n": b2_summary["n"], "full_error": fmt(b2_summary["base"]["mean"]),
            "global_prior_error": fmt(b2_summary["best_prior_stats"]["mean"]),
            "query_mask_delta": fmt(b2_summary["query_gain"]),
            "query_swap_delta": fmt(b2_summary["same_scene_query_swap_degradation"]),
            "current_frame_delta": fmt(b2_summary["history_gain"]),
            "remove_visible_delta": fmt(b2_summary["remove_visible_degradation"]),
            "shuffle_delta": fmt(b2_summary["shuffle_degradation"]),
            "reverse_delta": fmt(b2_summary["reverse_degradation"]),
            "query_feature_only_error": fmt(b2_summary["query_feature_only_error"]),
            "artifact_source": "analyses/probe_validity/b2_sample_diagnostics.csv.gz",
            "evidence_level": "MODEL_SPECIFIC" if model != "f3r" else "MODEL_SPECIFIC_CONTROL",
            "notes": "Category-level identifiability unavailable; exact-instance visibility only.",
        })

    c1_fields = [
        "experiment_id", "model", "objective_family", "layer", "prefix", "analysis_type", "subset",
        "candidate_pool", "action_condition", "n", "r1", "mrr_or_rank",
        "target_cosine", "copy_r1", "no_action_r1", "shuffled_action_r1",
        "matched_wrong_r1", "true_minus_matched",
        "delta_vs_copy", "delta_vs_no_action", "delta_vs_shuffle",
        "delta_vs_best_control", "normalized_action_gain",
        "fixed_predictor_action_margin", "ci_low", "ci_high",
        "trajectory_ci_low", "trajectory_ci_high", "n_scenes", "n_trajectories",
        "c3_intervention_validity", "c3_intervention_margin", "artifact_source",
        "evidence_level", "global_true_r1", "global_copy_r1", "matched_action_gain",
        "same_scene_gain", "high_motion_gain", "high_drift_gain",
        "translation_flip_difference", "notes",
    ]
    b2_fields = [
        "experiment_id", "sample_subset", "model", "objective_family", "layer", "prefix",
        "analysis_type", "baseline_or_readout", "perturbation", "seed", "n",
        "mean_ang_err_deg", "median_ang_err_deg", "p25_ang_err_deg",
        "p75_ang_err_deg", "top1", "top3", "delta_error_vs_full",
        "delta_error_vs_best_prior", "ci_low", "ci_high",
        "trajectory_ci_low", "trajectory_ci_high", "n_scenes", "n_trajectories",
        "query_unique_rate",
        "target_seen_rate", "ambiguous_rate", "never_seen_rate",
        "artifact_source", "evidence_level", "full_error", "global_prior_error",
        "query_mask_delta", "query_swap_delta", "current_frame_delta",
        "remove_visible_delta", "shuffle_delta", "reverse_delta",
        "query_feature_only_error", "notes",
    ]
    write_csv(OUT / "c1_validity.csv", c1_rows, c1_fields)
    write_csv(OUT / "b2_validity.csv", b2_rows, b2_fields)
    _phase2_plots(c1_summaries, b2_summaries)
    after = snapshot(PROTECTED)
    _cross_objective_report(
        c1_summaries, b2_summaries, replay_summaries, before, after,
        c1_sample_rows, b2_sample_rows,
    )
    final_after = snapshot(PROTECTED)
    if before != final_after:
        raise RuntimeError("Protected files changed during Phase-2 summarization")
    print(json.dumps({
        "status": "PHASE_2_CROSS_OBJECTIVE_COMPLETE",
        "protected_files_byte_identical": True,
        "models": models,
        "c1_unique_samples_per_model": {model: c1_summaries[model]["n"] for model in models},
        "b2_unique_samples_per_model": {model: b2_summaries[model]["n"] for model in models},
    }, indent=2))


def run_phase1() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    before = snapshot(PROTECTED)
    c1_rows, c1_summary = make_c1()
    b2_rows, b2_summary = make_b2()
    c1_fields = [
        "experiment_id", "model", "layer", "prefix", "analysis_type", "subset",
        "candidate_pool", "action_condition", "n", "r1", "mrr_or_rank",
        "target_cosine", "copy_r1", "no_action_r1", "shuffled_action_r1",
        "matched_wrong_r1", "true_minus_matched",
        "delta_vs_copy", "delta_vs_no_action", "delta_vs_shuffle",
        "delta_vs_best_control", "normalized_action_gain",
        "fixed_predictor_action_margin", "ci_low", "ci_high",
        "trajectory_ci_low", "trajectory_ci_high", "n_scenes", "n_trajectories",
        "c3_intervention_validity", "c3_intervention_margin", "artifact_source",
        "evidence_level", "notes",
    ]
    b2_fields = [
        "experiment_id", "sample_subset", "model", "layer", "prefix",
        "analysis_type", "baseline_or_readout", "perturbation", "seed", "n",
        "mean_ang_err_deg", "median_ang_err_deg", "p25_ang_err_deg",
        "p75_ang_err_deg", "top1", "top3", "delta_error_vs_full",
        "delta_error_vs_best_prior", "ci_low", "ci_high",
        "trajectory_ci_low", "trajectory_ci_high", "n_scenes", "n_trajectories",
        "query_unique_rate",
        "target_seen_rate", "ambiguous_rate", "never_seen_rate",
        "artifact_source", "evidence_level", "notes",
    ]
    write_csv(OUT / "c1_validity.csv", c1_rows, c1_fields)
    write_csv(OUT / "b2_validity.csv", b2_rows, b2_fields)
    make_c1_plot(c1_rows)
    make_b2_plot(b2_rows)
    after = snapshot(PROTECTED)
    report(c1_rows, c1_summary, b2_rows, b2_summary, before, after)
    print(f"Wrote {rel(OUT / 'c1_validity.csv')}")
    print(f"Wrote {rel(OUT / 'b2_validity.csv')}")
    print(f"Wrote {rel(OUT / 'PROBE_VALIDITY_REPORT.md')}")
    print(f"Protected files byte-identical: {before == after}")


# ---------------------------------------------------------------------------
# Paper-strengthening extension (P12 development protocol + focused P16).

PAPER_K = (4, 8, 16, 32)
PAPER_CANDIDATE_SEEDS = (42, 43, 44, 45, 46)
PAPER_HORIZONS = (1, 2, 4)
PAPER_PHYSICAL_THRESHOLDS = {
    "translation_magnitude": 0.25,
    "rotation_magnitude_rad": 0.15,
    "total_pose_displacement": 0.30,
}
B2_TRAJECTORY_DESCRIPTOR_CACHE: Dict[Tuple[str, Tuple[int, ...]], object] = {}


def _merge_strengthening_rows(
    path: Path,
    model: str,
    rows: List[Dict[str, object]],
) -> None:
    existing: List[Dict[str, object]] = []
    existing_fields: List[str] = []
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fields = list(reader.fieldnames or [])
            existing = [
                dict(row) for row in reader
                if not (
                    row.get("model") == model
                    and row.get("analysis_stage") == "paper_strengthening"
                )
            ]
    fields = list(dict.fromkeys([
        *existing_fields,
        *(key for row in rows for key in row),
    ]))
    _write_gzip_csv_atomic(path, [*existing, *rows], fields)


def _c1_contract(model: str, prefix: int) -> Tuple[Path, Optional[Path]]:
    spec = MODEL_SPECS[model]
    if prefix == 12:
        return Path(spec["c1_ckpt"]), Path(spec["c1_summary"])
    if prefix == 16:
        return Path(spec["c1_p16_ckpt"]), Path(spec["c1_p16_summary"])
    raise ValueError(f"Paper-strengthening C1 supports only P12/P16, got P{prefix}")


def _collect_c1_bundle(model: str, prefix: int, horizon: int) -> Dict[str, object]:
    torch, _ = _phase2_runtime()
    dataset = _dataset(
        "c1", "val", model=model, prefix=prefix, action_horizon=horizon
    )
    contexts = []
    actions = []
    targets = []
    copies = []
    metadata = []
    failures = []
    for index in range(len(dataset)):
        sample = dataset[index]
        scene = dataset.scenes[index]
        tail = int(scene["streaming_tail"])
        window = [int(value) for value in scene["streaming_window_indices"]]
        expected_target = int(window[tail + horizon])
        observed_target = int(sample["target_frame_idx"].item())
        valid = bool(sample["dyn_valid"].item())
        if not valid or observed_target != expected_target:
            failures.append({
                "index": index,
                "sample_id": str(scene.get("sample_id", "")),
                "valid": valid,
                "expected_target_raw_id": expected_target,
                "observed_target_raw_id": observed_target,
            })
            continue
        context = sample["input_feat"].mean(dim=(1, 2)).detach().cpu()
        target = sample["target_feat"].mean(dim=(0, 1)).detach().cpu().float()
        action = sample["action"].detach().cpu().float()
        scene_id = str(scene.get("scene_id", Path(str(scene["scene_dir"])).name))
        window_id = int(scene.get("streaming_window_id", -1))
        contexts.append(context)
        actions.append(action)
        targets.append(target)
        copies.append(context[-1].float())
        metadata.append({
            "manifest_sample_id": str(scene.get("sample_id", f"row{index:06d}")),
            "scene_id": scene_id,
            # The source manifests do not expose a trajectory above scene level.
            "trajectory_id": scene_id,
            "window_id": window_id,
            "context_frame_ids": [int(value) for value in scene["streaming_indices"]],
            "target_frame_id": observed_target,
            "expected_target_frame_id": expected_target,
        })
        if (index + 1) % 40 == 0:
            print(
                f"[paper C1 {model} P{prefix} h{horizon}] "
                f"{index + 1}/{len(dataset)}",
                flush=True,
            )
    coverage = len(contexts) / len(dataset) if len(dataset) else 0.0
    if failures or coverage != 1.0:
        raise RuntimeError(
            "C1_EXACT_TARGET_COVERAGE_GATE_FAILED: "
            f"model={model} prefix={prefix} horizon={horizon} "
            f"coverage={coverage:.8f} failures={failures[:5]}"
        )
    return {
        "contexts": contexts,
        "actions": torch.stack(actions),
        "targets": torch.stack(targets),
        "copies": torch.stack(copies),
        "metadata": metadata,
        "coverage": coverage,
        "manifest": rel(C1_MANIFESTS[prefix]),
        "manifest_sha256": file_sha256(C1_MANIFESTS[prefix]),
    }


def _c1_physical_index(bundle: Dict[str, object]) -> Dict[str, object]:
    torch, _ = _phase2_runtime()
    actions = bundle["actions"]
    _rotations, rotation, translation = _rotation_and_translation_magnitudes(actions)
    total = torch.sqrt(rotation.square() + translation.square())
    n = int(actions.shape[0])
    wrong = []
    distance = torch.cdist(torch.stack([translation, rotation], dim=-1),
                           torch.stack([translation, rotation], dim=-1))
    distance.fill_diagonal_(float("inf"))
    wrong = distance.argmin(dim=-1).tolist()
    metadata = bundle["metadata"]
    all_indices = list(range(n))
    candidates: Dict[str, List[List[int]]] = {
        name: [] for name in (
            "SAME_SCENE", "SAME_TRAJECTORY", "TEMPORAL_NEIGHBORS",
            "MATCHED_TRANSLATION_MAGNITUDE", "MATCHED_ROTATION_MAGNITUDE",
            "MATCHED_TOTAL_POSE_DISPLACEMENT",
        )
    }
    for index in all_indices:
        same_scene = [
            value for value in all_indices
            if metadata[value]["scene_id"] == metadata[index]["scene_id"]
        ]
        same_trajectory = [
            value for value in all_indices
            if metadata[value]["trajectory_id"] == metadata[index]["trajectory_id"]
        ]
        temporal = sorted(
            same_trajectory,
            key=lambda value: (
                abs(metadata[value]["window_id"] - metadata[index]["window_id"]),
                value,
            ),
        )
        others = [value for value in all_indices if value != index]
        by_translation = sorted(
            others, key=lambda value: (abs(float(translation[value] - translation[index])), value)
        )
        by_rotation = sorted(
            others, key=lambda value: (abs(float(rotation[value] - rotation[index])), value)
        )
        by_total = sorted(
            others, key=lambda value: (abs(float(total[value] - total[index])), value)
        )
        candidates["SAME_SCENE"].append(same_scene)
        candidates["SAME_TRAJECTORY"].append(same_trajectory)
        candidates["TEMPORAL_NEIGHBORS"].append(temporal)
        candidates["MATCHED_TRANSLATION_MAGNITUDE"].append([index, *by_translation])
        candidates["MATCHED_ROTATION_MAGNITUDE"].append([index, *by_rotation])
        candidates["MATCHED_TOTAL_POSE_DISPLACEMENT"].append([index, *by_total])
    return {
        "rotation": rotation,
        "translation": translation,
        "total": total,
        "matched_wrong_indices": wrong,
        "candidates": candidates,
    }


def _sample_physical_pool(
    values: Sequence[int],
    positive: int,
    desired_k: int,
    seed: int,
    pool_name: str,
) -> List[int]:
    unique = list(dict.fromkeys(int(value) for value in values))
    if positive not in unique:
        unique.insert(0, positive)
    negative = [value for value in unique if value != positive]
    actual_k = min(desired_k, len(negative) + 1)
    if actual_k <= 1:
        return [positive]
    # Motion-matched lists are ordered by physical distance. Sample from a
    # deterministic near-neighbour band; scene/temporal pools use all available
    # members when smaller than K.
    band = negative[: min(len(negative), max(actual_k - 1, 3 * (actual_k - 1)))]
    rng = random.Random(seed * 1000003 + positive * 97 + zlib.crc32(pool_name.encode()))
    selected = rng.sample(band, actual_k - 1) if len(band) > actual_k - 1 else band
    return [positive, *selected]


def _predict_c1_probe(probe, contexts, actions, device, batch_size: int):
    spatial = [value.unsqueeze(1).unsqueeze(1) for value in contexts]
    return _predict_actions(probe, spatial, actions, device, batch_size=batch_size)


def _train_c1_control(
    model: str,
    variant: str,
    train_bundle: Dict[str, object],
    device,
):
    torch, F = _phase2_runtime()
    from vidfm3d.models.components.probe_action_dynamics import ActionDynamicsProbe

    spec = MODEL_SPECS[model]
    conditioning = {
        "CONTEXT_ONLY_TRAINED": "full",
        "ACTION_ONLY_TRAINED": "action_only",
        "CONSTANT_ACTION_TRAINED": "visual_only",
        "CONSTANT_CONTEXT_TRAINED": "full",
    }[variant]
    torch.manual_seed(PHASE2_SEED)
    probe = ActionDynamicsProbe(
        in_channels=spec["channels"], action_dim=9, hidden_dim=512,
        num_layers=2, num_heads=8, max_seq_len=64,
        conditioning=conditioning,
    ).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.05
    )
    contexts = torch.stack(train_bundle["contexts"]).to(torch.bfloat16)
    actions = train_bundle["actions"].float()
    targets = train_bundle["targets"].float()
    constant_context = contexts.float().mean(dim=0, keepdim=True).to(torch.bfloat16)
    n = int(contexts.shape[0])
    batch_size = 4
    epochs = 50
    steps = 0
    probe.train()
    for epoch in range(epochs):
        if epoch < 5:
            lr = 3e-4 * float(epoch + 1) / 5.0
        else:
            progress = float(epoch - 5) / 45.0
            lr = 1e-6 + (3e-4 - 1e-6) * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr
        generator = torch.Generator().manual_seed(PHASE2_SEED + epoch)
        order = torch.randperm(n, generator=generator)
        running = 0.0
        for start in range(0, n, batch_size):
            index = order[start:start + batch_size]
            context = contexts[index].to(device, non_blocking=True)
            action = actions[index].to(device, non_blocking=True)
            target = targets[index].to(device, non_blocking=True)
            if variant == "CONTEXT_ONLY_TRAINED":
                action = torch.zeros_like(action)
            elif variant == "CONSTANT_CONTEXT_TRAINED":
                context = constant_context.expand(index.numel(), -1, -1).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = probe(context.unsqueeze(2).unsqueeze(2), action)
                mse = F.mse_loss(prediction.float(), target)
                cosine = 1.0 - F.cosine_similarity(prediction.float(), target, dim=-1).mean()
                loss = mse + cosine
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu()) * int(index.numel())
            steps += 1
        print(
            f"[paper C1 train {model} {variant}] epoch={epoch + 1}/50 "
            f"loss={running / n:.6f} steps={steps}",
            flush=True,
        )
    return probe.eval(), {
        "variant": variant,
        "epochs": epochs,
        "steps": steps,
        "batch_size": batch_size,
        "optimizer": "AdamW(lr=3e-4,betas=(0.9,0.95),weight_decay=0.05)",
        "scheduler": "linear_warmup_5_epochs_cosine_to_1e-6",
        "parameter_count": sum(parameter.numel() for parameter in probe.parameters()),
    }


def _c1_condition_rows(
    model: str,
    prefix: int,
    horizon: int,
    bundle: Dict[str, object],
    predictions: Dict[str, object],
    physical: Dict[str, object],
    include_k_grid: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    targets = bundle["targets"]
    metadata = bundle["metadata"]
    matched = physical["matched_wrong_indices"]
    rows: List[Dict[str, object]] = []
    aggregates: List[Dict[str, object]] = []
    pools = physical["candidates"]
    selected_pools = list(pools) if include_k_grid else ["MATCHED_TOTAL_POSE_DISPLACEMENT"]
    k_values = PAPER_K if include_k_grid else (8,)
    seeds = PAPER_CANDIDATE_SEEDS if include_k_grid else (PHASE2_SEED,)
    for pool_name in selected_pools:
        for desired_k in k_values:
            for candidate_seed in seeds:
                group = []
                for index, item in enumerate(metadata):
                    candidate_indices = _sample_physical_pool(
                        pools[pool_name][index], index, desired_k,
                        candidate_seed, pool_name,
                    )
                    candidate_targets = targets[candidate_indices]
                    positive = candidate_indices.index(index)
                    row = {
                        "analysis_stage": "paper_strengthening",
                        "task_variant": "C1",
                        "sample_id": (
                            f"C1|paper|{model}|P{prefix}|h{horizon}|"
                            f"{item['manifest_sample_id']}"
                        ),
                        "manifest_sample_id": item["manifest_sample_id"],
                        "model": model,
                        "objective_family": MODEL_SPECS[model]["objective_family"],
                        "layer": MODEL_SPECS[model]["layer"],
                        "prefix": prefix,
                        "horizon": horizon,
                        "scene_id": item["scene_id"],
                        "trajectory_id": item["trajectory_id"],
                        "window_id": item["window_id"],
                        "eval_split": "val_development",
                        "candidate_pool": pool_name,
                        "desired_candidate_count": desired_k,
                        "candidate_count": len(candidate_indices),
                        "candidate_seed": candidate_seed,
                        "chance_r1": 1.0 / len(candidate_indices),
                        "target_frame_id": item["target_frame_id"],
                        "expected_target_frame_id": item["expected_target_frame_id"],
                        "context_frame_ids": json.dumps(item["context_frame_ids"], separators=(",", ":")),
                        "translation_magnitude": float(physical["translation"][index]),
                        "rotation_magnitude_rad": float(physical["rotation"][index]),
                        "total_pose_displacement": float(physical["total"][index]),
                        "matched_wrong_source_sample_id": metadata[matched[index]]["manifest_sample_id"],
                    }
                    for condition, prediction in predictions.items():
                        metrics = _rank_metrics(prediction[index], candidate_targets, positive)
                        for key, value in metrics.items():
                            row[f"{condition}_{key}"] = value
                        chance = float(row["chance_r1"])
                        row[f"{condition}_chance_normalized_r1"] = (
                            (float(metrics["r1"]) - chance) / (1.0 - chance)
                            if chance < 1.0 else 0.0
                        )
                        row[f"{condition}_rank_percentile"] = (
                            1.0 - float(metrics["target_rank"]) / (len(candidate_indices) - 1)
                            if len(candidate_indices) > 1 else 1.0
                        )
                    row["action_cosine_margin"] = (
                        float(row["FULL_target_cosine"])
                        - float(row["MATCHED_WRONG_target_cosine"])
                    )
                    row["positive_paired_margin"] = int(row["action_cosine_margin"] > 0)
                    group.append(row)
                    rows.append(row)
                true_r1 = statistics.fmean(float(row["FULL_r1"]) for row in group)
                wrong_r1 = statistics.fmean(float(row["MATCHED_WRONG_r1"]) for row in group)
                chance = statistics.fmean(float(row["chance_r1"]) for row in group)
                ci_low, ci_high = _cluster_bootstrap_delta(
                    group, "FULL_r1", "MATCHED_WRONG_r1", "scene_id"
                )
                tr_low, tr_high = _cluster_bootstrap_delta(
                    group, "FULL_r1", "MATCHED_WRONG_r1", "trajectory_id"
                )
                aggregates.append({
                    "experiment_id": f"paper_c1_{model}_l{MODEL_SPECS[model]['layer']}_p{prefix}_h{horizon}",
                    "analysis_type": "paper_candidate_k_robustness",
                    "model": model,
                    "objective_family": MODEL_SPECS[model]["objective_family"],
                    "layer": MODEL_SPECS[model]["layer"],
                    "prefix": prefix,
                    "horizon": horizon,
                    "candidate_pool": pool_name,
                    "candidate_seed": candidate_seed,
                    "desired_candidate_count": desired_k,
                    "mean_actual_candidate_count": statistics.fmean(int(row["candidate_count"]) for row in group),
                    "k_coverage": sum(int(row["candidate_count"]) == desired_k for row in group) / len(group),
                    "n": len(group),
                    "n_scenes": len({row["scene_id"] for row in group}),
                    "n_trajectories": len({row["trajectory_id"] for row in group}),
                    "r1": true_r1,
                    "matched_wrong_r1": wrong_r1,
                    "chance_r1": chance,
                    "chance_normalized_r1": statistics.fmean(float(row["FULL_chance_normalized_r1"]) for row in group),
                    "true_minus_matched": true_r1 - wrong_r1,
                    "target_cosine": statistics.fmean(float(row["FULL_target_cosine"]) for row in group),
                    "cosine_margin": statistics.fmean(float(row["action_cosine_margin"]) for row in group),
                    "mrr": statistics.fmean(float(row["FULL_reciprocal_rank"]) for row in group),
                    "target_rank_percentile": statistics.fmean(float(row["FULL_rank_percentile"]) for row in group),
                    "positive_paired_margin_rate": statistics.fmean(float(row["positive_paired_margin"]) for row in group),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "trajectory_ci_low": tr_low,
                    "trajectory_ci_high": tr_high,
                    "artifact_source": "analyses/probe_validity/c1_sample_diagnostics.csv.gz",
                    "evidence_level": "SINGLE_PROBE_SEED_CANDIDATE_SAMPLING_ROBUSTNESS",
                })
    return rows, aggregates


def run_paper_c1_model(model: str) -> Dict[str, object]:
    torch, _ = _phase2_runtime()
    from vidfm3d.models.components.probe_action_dynamics import ActionDynamicsProbe

    spec = MODEL_SPECS[model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_diagnostics: List[Dict[str, object]] = []
    all_aggregates: List[Dict[str, object]] = []
    horizon_summary = {}
    p12_h1_bundle = None
    p12_h1_full = None
    for prefix, horizons in ((12, PAPER_HORIZONS), (16, (1,))):
        checkpoint, historical_summary = _c1_contract(model, prefix)
        probe = _load_probe(
            checkpoint,
            ActionDynamicsProbe(
                in_channels=spec["channels"], action_dim=9, hidden_dim=512,
                num_layers=2, num_heads=8, max_seq_len=64,
            ),
            device,
            allowed_missing=("constant_context", "constant_action"),
        )
        for horizon in horizons:
            bundle = _collect_c1_bundle(model, prefix, horizon)
            physical = _c1_physical_index(bundle)
            batch_size = 4 if int(spec["channels"]) > 1024 else 16
            full = _predict_c1_probe(probe, bundle["contexts"], bundle["actions"], device, batch_size)
            matched_actions = bundle["actions"][physical["matched_wrong_indices"]]
            wrong = _predict_c1_probe(probe, bundle["contexts"], matched_actions, device, batch_size)
            conditions = {"FULL": full, "MATCHED_WRONG": wrong, "COPY": bundle["copies"]}
            diagnostic, aggregate = _c1_condition_rows(
                model, prefix, horizon, bundle, conditions, physical,
                include_k_grid=(prefix == 12 and horizon == 1),
            )
            all_diagnostics.extend([
                row for row in diagnostic
                if (
                    row["candidate_pool"] == "MATCHED_TOTAL_POSE_DISPLACEMENT"
                    and int(row["desired_candidate_count"]) == 8
                    and int(row["candidate_seed"]) == PHASE2_SEED
                ) or (
                    row["candidate_pool"] == "SAME_SCENE"
                    and int(row["desired_candidate_count"]) == 4
                    and int(row["candidate_seed"]) == PHASE2_SEED
                )
            ])
            all_aggregates.extend(aggregate)
            primary_hard = [
                row for row in diagnostic
                if row["candidate_pool"] == "MATCHED_TOTAL_POSE_DISPLACEMENT"
                and int(row["desired_candidate_count"]) == 8
                and int(row["candidate_seed"]) == PHASE2_SEED
            ]
            high_motion = [
                row for row in primary_hard
                if float(row["translation_magnitude"]) >= PAPER_PHYSICAL_THRESHOLDS["translation_magnitude"]
                or float(row["rotation_magnitude_rad"]) >= PAPER_PHYSICAL_THRESHOLDS["rotation_magnitude_rad"]
                or float(row["total_pose_displacement"]) >= PAPER_PHYSICAL_THRESHOLDS["total_pose_displacement"]
            ]
            if high_motion:
                hm_low, hm_high = _cluster_bootstrap_delta(
                    high_motion, "FULL_r1", "MATCHED_WRONG_r1", "scene_id"
                )
                all_aggregates.append({
                    "experiment_id": f"paper_c1_{model}_l{spec['layer']}_p{prefix}_h{horizon}",
                    "analysis_type": "paper_high_absolute_motion",
                    "model": model,
                    "objective_family": spec["objective_family"],
                    "layer": spec["layer"],
                    "prefix": prefix,
                    "horizon": horizon,
                    "candidate_pool": "MATCHED_TOTAL_POSE_DISPLACEMENT",
                    "desired_candidate_count": 8,
                    "candidate_seed": PHASE2_SEED,
                    "subset": "HIGH_ABSOLUTE_MOTION",
                    "n": len(high_motion),
                    "r1": statistics.fmean(float(row["FULL_r1"]) for row in high_motion),
                    "matched_wrong_r1": statistics.fmean(float(row["MATCHED_WRONG_r1"]) for row in high_motion),
                    "true_minus_matched": statistics.fmean(
                        float(row["FULL_r1"]) - float(row["MATCHED_WRONG_r1"])
                        for row in high_motion
                    ),
                    "ci_low": hm_low,
                    "ci_high": hm_high,
                    "physical_thresholds": json.dumps(PAPER_PHYSICAL_THRESHOLDS, sort_keys=True),
                    "artifact_source": "analyses/probe_validity/c1_sample_diagnostics.csv.gz",
                    "evidence_level": "SINGLE_PROBE_SEED_FIXED_ABSOLUTE_THRESHOLDS",
                })
            global_stats = _feature_summary(full, bundle["targets"])
            copy_stats = _feature_summary(bundle["copies"], bundle["targets"])
            wrong_stats = _feature_summary(wrong, bundle["targets"])
            horizon_summary[f"P{prefix}_h{horizon}"] = {
                "coverage": bundle["coverage"],
                "manifest": bundle["manifest"],
                "manifest_sha256": bundle["manifest_sha256"],
                "checkpoint": rel(checkpoint),
                "historical_summary": rel(historical_summary) if historical_summary else "",
                "global_true_r1": global_stats["global_R@1"],
                "global_copy_r1": copy_stats["global_R@1"],
                "global_matched_wrong_r1": wrong_stats["global_R@1"],
                "delta_copy": global_stats["global_R@1"] - copy_stats["global_R@1"],
                "delta_action": global_stats["global_R@1"] - wrong_stats["global_R@1"],
                "target_cosine_margin": global_stats["mean_cos"] - wrong_stats["mean_cos"],
            }
            if prefix == 12 and horizon == 1:
                p12_h1_bundle = bundle
                p12_h1_full = full
        del probe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Capacity-matched controls use the exact h=1 train/eval contract. FULL is
    # the historical checkpoint replay above; only the four missing controls train.
    train_dataset = _dataset("c1", "train", model=model, prefix=12, action_horizon=1)
    train_contexts = []
    train_actions = []
    train_targets = []
    train_failures = []
    for index in range(len(train_dataset)):
        sample = train_dataset[index]
        if not bool(sample["dyn_valid"].item()):
            train_failures.append(index)
            continue
        scene = train_dataset.scenes[index]
        expected = int(scene["streaming_window_indices"][int(scene["streaming_tail"]) + 1])
        observed = int(sample["target_frame_idx"].item())
        if expected != observed:
            train_failures.append(index)
            continue
        train_contexts.append(sample["input_feat"].mean(dim=(1, 2)).detach().cpu())
        train_actions.append(sample["action"].detach().cpu().float())
        train_targets.append(sample["target_feat"].mean(dim=(0, 1)).detach().cpu().float())
        if (index + 1) % 100 == 0:
            print(f"[paper C1 train data {model}] {index + 1}/{len(train_dataset)}", flush=True)
    if train_failures or len(train_contexts) != len(train_dataset):
        raise RuntimeError(
            f"C1 train exact-target coverage failed for {model}: "
            f"valid={len(train_contexts)} total={len(train_dataset)} first={train_failures[:5]}"
        )
    train_bundle = {
        "contexts": train_contexts,
        "actions": torch.stack(train_actions),
        "targets": torch.stack(train_targets),
    }
    control_predictions = {"FULL": p12_h1_full}
    training_receipts = []
    for variant in (
        "CONTEXT_ONLY_TRAINED", "ACTION_ONLY_TRAINED",
        "CONSTANT_ACTION_TRAINED", "CONSTANT_CONTEXT_TRAINED",
    ):
        control, receipt = _train_c1_control(model, variant, train_bundle, device)
        eval_contexts = p12_h1_bundle["contexts"]
        eval_actions = p12_h1_bundle["actions"]
        if variant == "CONTEXT_ONLY_TRAINED":
            eval_actions = torch.zeros_like(eval_actions)
        if variant == "CONSTANT_CONTEXT_TRAINED":
            constant = torch.stack(train_contexts).float().mean(dim=0)
            eval_contexts = [constant for _ in eval_contexts]
        prediction = _predict_c1_probe(
            control, eval_contexts, eval_actions, device,
            4 if int(spec["channels"]) > 1024 else 16,
        )
        control_predictions[variant] = prediction
        training_receipts.append(receipt)
        del control
        if device.type == "cuda":
            torch.cuda.empty_cache()
    target = p12_h1_bundle["targets"]
    p12_physical = _c1_physical_index(p12_h1_bundle)
    ablations = {}
    for variant, prediction in control_predictions.items():
        stats = _feature_summary(prediction, target)
        hard_metrics = []
        for index in range(len(target)):
            candidates = _sample_physical_pool(
                p12_physical["candidates"]["MATCHED_TOTAL_POSE_DISPLACEMENT"][index],
                index, 8, PHASE2_SEED, "MATCHED_TOTAL_POSE_DISPLACEMENT",
            )
            hard_metrics.append(_rank_metrics(
                prediction[index], target[candidates], candidates.index(index)
            ))
        hard_r1 = statistics.fmean(float(row["r1"]) for row in hard_metrics)
        hard_mrr = statistics.fmean(float(row["reciprocal_rank"]) for row in hard_metrics)
        ablations[variant] = {**stats, "hard_k8_r1": hard_r1, "hard_k8_mrr": hard_mrr}
        all_aggregates.append({
            "experiment_id": f"paper_c1_retrained_{model}_l{spec['layer']}_p12",
            "analysis_type": "paper_retrained_ablation",
            "model": model,
            "objective_family": spec["objective_family"],
            "layer": spec["layer"],
            "prefix": 12,
            "horizon": 1,
            "action_condition": variant,
            "n": len(target),
            "r1": stats["global_R@1"],
            "hard_k8_r1": hard_r1,
            "hard_k8_mrr": hard_mrr,
            "target_cosine": stats["mean_cos"],
            "mrr_or_rank": stats["mean_rank"],
            "artifact_source": "analyses/probe_validity/c1_sample_diagnostics.csv.gz",
            "evidence_level": "RETRAINED_SINGLE_SEED42",
        })
    controls = [value["global_R@1"] for key, value in ablations.items() if key != "FULL"]
    hard_controls = [value["hard_k8_r1"] for key, value in ablations.items() if key != "FULL"]
    ablations["full_minus_strongest_control_r1"] = (
        ablations["FULL"]["global_R@1"] - max(controls)
    )
    ablations["full_minus_strongest_control_hard_k8_r1"] = (
        ablations["FULL"]["hard_k8_r1"] - max(hard_controls)
    )
    payload = {
        "status": "PASS",
        "model": model,
        "layer": spec["layer"],
        "seed": PHASE2_SEED,
        "candidate_seeds": list(PAPER_CANDIDATE_SEEDS),
        "physical_thresholds": PAPER_PHYSICAL_THRESHOLDS,
        "horizons": horizon_summary,
        "ablations": ablations,
        "training_receipts": training_receipts,
        "aggregates": all_aggregates,
        "actual_futures": "ACTION_CONFUSABLE_REAL_FUTURES_NOT_AVAILABLE",
        "actual_futures_reason": (
            "C3 reuses factual trajectory targets under alternative action labels; "
            "no branch-paired real future feature exists for one shared anchor/context."
        ),
        "matched_view_overlap": "NOT_AVAILABLE_NO_RUN_BOUND_OVERLAP_METADATA",
    }
    _write_tmp_json(f"paper_c1_{model}.json", payload)
    _merge_strengthening_rows(
        OUT / "c1_sample_diagnostics.csv.gz", model, all_diagnostics
    )
    return payload


def _b2_contract(model: str, prefix: int) -> Tuple[Path, Path]:
    spec = MODEL_SPECS[model]
    if prefix == 12:
        return Path(spec["b2_ckpt"]), Path(spec["b2_summary"])
    if prefix == 16:
        return Path(spec["b2_p16_ckpt"]), Path(spec["b2_p16_summary"])
    raise ValueError(f"Paper-strengthening B2 supports only P12/P16, got P{prefix}")


def _b2_temporal_mapping(length: int, seed: int, reverse: bool = False) -> List[int]:
    if length < 2:
        return list(range(length))
    history = list(range(length - 1))
    if reverse:
        history.reverse()
    else:
        random.Random(seed).shuffle(history)
        if history == list(range(length - 1)) and len(history) > 1:
            history = history[1:] + history[:1]
    return [*history, length - 1]


def _b2_prediction_frame_ids(probe, feature, query, frame_ids, device):
    torch, _ = _phase2_runtime()
    with torch.no_grad():
        feat = probe._patchify(feature.unsqueeze(0).float().to(device))
        _, frames, height, width, _ = feat.shape
        ids = torch.as_tensor(frame_ids, dtype=torch.long, device=device)
        if ids.numel() != frames or int(ids.min()) < 0 or int(ids.max()) >= probe.frame_embed.shape[1]:
            raise ValueError(f"Invalid B2 frame ids {frame_ids} for {frames} frames")
        z = probe.feat_proj(feat)
        row = probe.row_embed[:, :height].unsqueeze(2)
        col = probe.col_embed[:, :width].unsqueeze(1)
        z = z + (row + col).unsqueeze(1)
        z = z + probe.frame_embed[:, ids].view(1, frames, 1, 1, -1)
        z = z.reshape(1, frames * height * width, -1)
        q = (
            probe.query_proj(query.unsqueeze(0).float().to(device)).unsqueeze(1)
            + probe.belief_token
        )
        out = probe.encoder(torch.cat([q, z], dim=1))
        belief = probe.norm_out(out[:, 0])
        logits = probe.bin_head(belief).reshape(-1).detach().cpu()
        log_distance = float(probe.dist_head(belief).reshape(-1)[0].detach().cpu())
    return logits, log_distance


def _b2_order_synthetic_sanity() -> Dict[str, object]:
    # The exact mapping helper used for real samples is exercised with one
    # unordered multiset whose label depends on content-to-slot association.
    content = [2.0, 5.0, 11.0, 17.0]
    mapping = _b2_temporal_mapping(len(content), seed=PHASE2_SEED, reverse=True)
    moved = [content[index] for index in mapping]
    slots = list(range(len(content)))
    ordered_score = sum(value * (slot + 1) for slot, value in zip(slots, content))
    reversed_content_score = sum(value * (slot + 1) for slot, value in zip(slots, moved))
    token_score = sum(value * (frame_id + 1) for value, frame_id in zip(moved, mapping))
    passed = (
        sorted(content) == sorted(moved)
        and mapping[-1] == len(content) - 1
        and ordered_score != reversed_content_score
        and ordered_score == token_score
    )
    if not passed:
        raise RuntimeError("B2 synthetic order sanity failed")
    return {
        "status": "PASS",
        "same_unordered_elements": True,
        "final_frame_fixed": True,
        "ordered_score": ordered_score,
        "reverse_content_only_score": reversed_content_score,
        "reverse_frame_tokens_score": token_score,
    }


def _b2_2d_descriptor(hidden_mask, query_frame: int):
    torch, _ = _phase2_runtime()
    mask = hidden_mask.bool()
    frames, height, width = mask.shape
    counts = mask.flatten(1).sum(1).float()
    query = mask[query_frame]
    coordinates = query.nonzero(as_tuple=False).float()
    if coordinates.numel():
        cy, cx = coordinates.mean(0)
        cx = float(cx / max(width - 1, 1))
        cy = float(cy / max(height - 1, 1))
    else:
        cx = cy = 0.0
    area = counts / float(height * width)
    return torch.tensor([
        query_frame / max(frames - 1, 1), cx, cy,
        float(area[query_frame]), float(area[-1]),
        float((counts > 0).float().mean()), float(area.mean()), float(area.max()),
    ], dtype=torch.float32)


def _b2_trajectory_descriptor(scene: Dict[str, object]):
    torch, _ = _phase2_runtime()
    from vidfm3d.utils import aoss_fs

    raw = [int(value) for value in scene["streaming_indices"]]
    cache_key = (str(scene["scene_dir"]), tuple(raw))
    if cache_key in B2_TRAJECTORY_DESCRIPTOR_CACHE:
        return B2_TRAJECTORY_DESCRIPTOR_CACHE[cache_key].clone()
    meta = aoss_fs.load_npy(aoss_fs.join(scene["scene_dir"], "scene_iphone_metadata.npz"))
    valid_pose_indices = scene.get("valid_pose_indices")
    pose_indices = [
        int(valid_pose_indices[index]) if valid_pose_indices is not None else index
        for index in raw
    ]
    poses = torch.as_tensor(meta["trajectories"][pose_indices], dtype=torch.float32)
    rotation = poses[:, :3, :3]
    position = poses[:, :3, 3]
    delta = position[1:] - position[:-1]
    step = torch.linalg.norm(delta, dim=-1)
    relative_rotation = rotation[:-1].transpose(1, 2) @ rotation[1:]
    trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    angle = torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))
    net_rotation = rotation[0].T @ rotation[-1]
    net_trace = net_rotation.diagonal().sum()
    net_angle = torch.acos(((net_trace - 1.0) / 2.0).clamp(-1.0, 1.0))
    net = position[-1] - position[0]
    descriptor = torch.tensor([
        float(net[0]), float(net[1]), float(net[2]),
        float(step.sum()), float(step.mean()), float(step.std(unbiased=False)),
        float(angle.sum()), float(angle.mean()), float(angle.std(unbiased=False)),
        float(net_angle),
    ], dtype=torch.float32)
    B2_TRAJECTORY_DESCRIPTOR_CACHE[cache_key] = descriptor
    return descriptor.clone()


def _collect_b2_bundle(
    model: str, prefix: int, split: str, retain_features: bool = True
) -> Dict[str, object]:
    dataset = _dataset("b2", split, model=model, prefix=prefix)
    samples = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if not bool(sample["hidden_obj_valid"].item()):
            continue
        scene = dataset.scenes[index]
        prefix_indices = [int(value) for value in sample["prefix_indices"].tolist()]
        query_frame = int(sample["belief_query_frame"].item())
        hidden_mask = sample["hidden_obj_mask"].detach().cpu().bool()
        scene_id = str(scene.get("scene_id", Path(str(scene["scene_dir"])).name))
        window_id = int(scene.get("streaming_window_id", -1))
        hidden_id = int(sample["hidden_obj_id"].item())
        record = {
            "polar": sample["hidden_obj_polar"].detach().cpu().float(),
            "pixel_counts": [int(value) for value in hidden_mask.flatten(1).sum(1).tolist()],
            "scene_id": scene_id,
            "trajectory_id": scene_id,
            "window_id": window_id,
            "manifest_sample_id": str(
                sample.get("sample_id")
                or scene.get("sample_id", f"{scene_id}_w{window_id:03d}_p{prefix}")
            ),
            "prefix_indices": prefix_indices,
            "query_frame": query_frame,
            "query_raw_frame": prefix_indices[query_frame],
            "target_raw_frame": prefix_indices[-1],
            "hidden_id": hidden_id,
            "two_d_descriptor": _b2_2d_descriptor(hidden_mask, query_frame),
            "trajectory_descriptor": _b2_trajectory_descriptor(scene),
        }
        if retain_features:
            record["feature"] = sample["vfm_feat"].detach().cpu()
            record["query"] = sample["belief_query_feat"].detach().cpu()
        samples.append(record)
        if (index + 1) % 25 == 0:
            print(
                f"[paper B2 {model} P{prefix} {split}] "
                f"{index + 1}/{len(dataset)} valid={len(samples)}",
                flush=True,
            )
    if not samples:
        raise RuntimeError(f"B2 found no valid samples: {model} P{prefix} {split}")
    return {"samples": samples, "dataset_n": len(dataset)}


def _fit_b2_metadata_prior(train_samples, descriptor_key: str, device):
    torch, F = _phase2_runtime()
    torch.manual_seed(PHASE2_SEED)
    x = torch.stack([sample[descriptor_key] for sample in train_samples]).float()
    polar = torch.stack([sample["polar"] for sample in train_samples]).float()
    labels = (
        (((polar[:, 0] + math.pi) / (2 * math.pi) * 16).long().clamp(0, 15) * 8)
        + ((polar[:, 1] + math.pi / 2) / math.pi * 8).long().clamp(0, 7)
    )
    mean_x = x.mean(0)
    std_x = x.std(0, unbiased=False).clamp_min(1e-6)
    network = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 64), torch.nn.GELU(),
        torch.nn.Linear(64, 129),
    ).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=1e-3)
    n = len(x)
    for epoch in range(200):
        order = torch.randperm(n, generator=torch.Generator().manual_seed(PHASE2_SEED + epoch))
        for start in range(0, n, 64):
            index = order[start:start + 64]
            feature = ((x[index] - mean_x) / std_x).to(device)
            target_label = labels[index].to(device)
            target_distance = polar[index, 2].to(device)
            prediction = network(feature)
            loss = F.cross_entropy(prediction[:, :128], target_label)
            loss = loss + 0.3 * F.smooth_l1_loss(prediction[:, 128], target_distance)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return network.eval(), mean_x, std_x


def _predict_b2_metadata_prior(network, mean_x, std_x, sample, descriptor_key, device):
    torch, _ = _phase2_runtime()
    with torch.no_grad():
        x = ((sample[descriptor_key].float() - mean_x) / std_x).unsqueeze(0).to(device)
        prediction = network(x)[0].detach().cpu()
    return prediction[:128], float(prediction[128])


def run_paper_b2_model(model: str) -> Dict[str, object]:
    torch, _ = _phase2_runtime()
    from collections import Counter, defaultdict
    from vidfm3d.models.components.probe_ego_belief_v2 import EgoBeliefProbeV2

    spec = MODEL_SPECS[model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    synthetic = _b2_order_synthetic_sanity()
    all_rows: List[Dict[str, object]] = []
    all_aggregates: List[Dict[str, object]] = []
    prefix_summaries = {}
    for prefix in (12, 16):
        checkpoint, historical_summary = _b2_contract(model, prefix)
        probe = _load_probe(
            checkpoint,
            EgoBeliefProbeV2(
                in_channels=spec["channels"], decoder_type="transformer",
                num_frames=prefix, hidden_dim=512, num_layers=4, num_heads=8,
                max_seq_len=64, max_h=spec["b2_max_h"], max_w=spec["b2_max_w"],
                n_az_bins=16, n_el_bins=8, patch_pool=2,
            ),
            device,
            allowed_missing=("constant_context",),
        )
        train_bundle = _collect_b2_bundle(model, prefix, "train", retain_features=False)
        eval_bundle = _collect_b2_bundle(model, prefix, "val")
        train_samples = train_bundle["samples"]
        samples = eval_bundle["samples"]

        train_bins = []
        for sample in train_samples:
            polar = sample["polar"]
            az = min(max(int(((float(polar[0]) + math.pi) / (2 * math.pi) * 16)), 0), 15)
            el = min(max(int(((float(polar[1]) + math.pi / 2) / math.pi * 8)), 0), 7)
            train_bins.append(az * 8 + el)
        counts = Counter(train_bins)
        prior_logits = torch.tensor([float(counts.get(index, 0)) for index in range(128)])
        prior_top3 = [index for index, _count in counts.most_common(3)]
        direction = torch.stack([
            torch.tensor([
                math.cos(float(sample["polar"][1])) * math.cos(float(sample["polar"][0])),
                math.cos(float(sample["polar"][1])) * math.sin(float(sample["polar"][0])),
                math.sin(float(sample["polar"][1])),
            ]) for sample in train_samples
        ]).mean(0)
        constant_az = math.atan2(float(direction[1]), float(direction[0]))
        constant_el = math.atan2(float(direction[2]), math.hypot(float(direction[0]), float(direction[1])))
        constant_bin = min(max(int((constant_az + math.pi) / (2 * math.pi) * 16), 0), 15) * 8
        constant_bin += min(max(int((constant_el + math.pi / 2) / math.pi * 8), 0), 7)
        prior_distance = statistics.fmean(float(sample["polar"][2]) for sample in train_samples)
        trajectory_model, trajectory_mean, trajectory_std = _fit_b2_metadata_prior(
            train_samples, "trajectory_descriptor", device
        )
        leakage_model, leakage_mean, leakage_std = _fit_b2_metadata_prior(
            train_samples, "two_d_descriptor", device
        )

        by_scene = defaultdict(list)
        for index, sample in enumerate(samples):
            by_scene[sample["scene_id"]].append(index)
        swaps = {}
        for indices in by_scene.values():
            if len(indices) > 1:
                for offset, index in enumerate(indices):
                    swaps[index] = indices[(offset + 1) % len(indices)]

        method_rows = defaultdict(list)
        identity_max_diff = 0.0
        for index, sample in enumerate(samples):
            frames = int(sample["feature"].shape[0])
            identity_ids = list(range(frames))
            baseline_logits, baseline_distance = _b2_prediction(probe, sample["feature"], sample["query"], device)
            explicit_logits, explicit_distance = _b2_prediction_frame_ids(
                probe, sample["feature"], sample["query"], identity_ids, device
            )
            identity_max_diff = max(
                identity_max_diff,
                float((baseline_logits - explicit_logits).abs().max()),
                abs(baseline_distance - explicit_distance),
            )
            shuffle = _b2_temporal_mapping(frames, PHASE2_SEED * 1000003 + index)
            reverse = _b2_temporal_mapping(frames, PHASE2_SEED, reverse=True)
            visible_frames = [frame for frame, count in enumerate(sample["pixel_counts"]) if count > 0]
            current = torch.zeros_like(sample["feature"])
            current[-1] = sample["feature"][-1]
            remove_visible = sample["feature"].clone()
            if visible_frames:
                remove_visible[visible_frames] = 0
            conditions = [
                ("FULL_HISTORY_PLUS_QUERY", sample["feature"], sample["query"], identity_ids, "FIXED_PREDICTOR"),
                ("HISTORY_WITHOUT_QUERY", sample["feature"], torch.zeros_like(sample["query"]), identity_ids, "FIXED_PREDICTOR"),
                ("CURRENT_FRAME_PLUS_QUERY", current, sample["query"], identity_ids, "FIXED_PREDICTOR"),
                ("REMOVE_TARGET_VISIBLE_FRAMES", remove_visible, sample["query"], identity_ids, "FIXED_PREDICTOR"),
                ("CONTENT_ONLY_SHUFFLE", sample["feature"][shuffle], sample["query"], identity_ids, "FIXED_PREDICTOR"),
                ("FRAME_TOKEN_SHUFFLE", sample["feature"][shuffle], sample["query"], shuffle, "FIXED_PREDICTOR"),
                ("REVERSE_CONTENT_ONLY", sample["feature"][reverse], sample["query"], identity_ids, "FIXED_PREDICTOR"),
                ("REVERSE_FRAME_TOKENS", sample["feature"][reverse], sample["query"], reverse, "FIXED_PREDICTOR"),
            ]
            if index in swaps:
                conditions.append((
                    "SAME_SCENE_QUERY_SWAP", sample["feature"], samples[swaps[index]]["query"],
                    identity_ids, "FIXED_PREDICTOR",
                ))
            stable_id = (
                f"B2|paper|{model}|P{prefix}|{sample['manifest_sample_id']}|"
                f"obj{sample['hidden_id']}"
            )
            base = {
                "analysis_stage": "paper_strengthening",
                "task_variant": "B2",
                "sample_id": stable_id,
                "manifest_sample_id": sample["manifest_sample_id"],
                "model": model,
                "objective_family": spec["objective_family"],
                "layer": spec["layer"],
                "prefix": prefix,
                "scene_id": sample["scene_id"],
                "trajectory_id": sample["trajectory_id"],
                "window_id": sample["window_id"],
                "eval_split": "val_development",
                "target_object_id": sample["hidden_id"],
                "query_frame": sample["query_frame"],
                "query_raw_frame": sample["query_raw_frame"],
                "target_raw_frame": sample["target_raw_frame"],
                "gt_azimuth": float(sample["polar"][0]),
                "gt_elevation": float(sample["polar"][1]),
                "gt_log_distance": float(sample["polar"][2]),
                "target_visible_frame_indices": json.dumps(visible_frames, separators=(",", ":")),
                "perturbation_seed": PHASE2_SEED,
                "final_frame_fixed": 1,
                "query_unchanged": 1,
                "gt_unchanged": 1,
                "attention_mask": "ALL_VALID_NO_PADDING",
            }
            for method, feature, query, frame_ids, baseline_type in conditions:
                logits, distance = _b2_prediction_frame_ids(probe, feature, query, frame_ids, device)
                metrics = _b2_metrics(logits, distance, sample["polar"])
                mapping = shuffle if "SHUFFLE" in method else reverse if "REVERSE" in method else identity_ids
                row = {
                    **base,
                    "method": method,
                    "baseline_type": baseline_type,
                    "history_mapping": json.dumps(mapping, separators=(",", ":")),
                    "temporal_frame_ids": json.dumps(frame_ids, separators=(",", ":")),
                    "frame_tensor_changed": int(not torch.equal(feature, sample["feature"])),
                    "predicted_log_distance": distance,
                    **metrics,
                }
                method_rows[method].append(row)
                all_rows.append(row)

            prior_conditions = [
                ("GLOBAL_CONSTANT", prior_logits, prior_distance, [constant_bin], "TRAIN_FITTED_PRIOR"),
                ("GLOBAL_ANGLE_HISTOGRAM", prior_logits, prior_distance, prior_top3, "TRAIN_FITTED_PRIOR"),
            ]
            trajectory_logits, trajectory_distance = _predict_b2_metadata_prior(
                trajectory_model, trajectory_mean, trajectory_std, sample,
                "trajectory_descriptor", device,
            )
            leakage_logits, leakage_distance = _predict_b2_metadata_prior(
                leakage_model, leakage_mean, leakage_std, sample,
                "two_d_descriptor", device,
            )
            prior_conditions.extend([
                ("TRAJECTORY_ONLY", trajectory_logits, trajectory_distance, None, "TRAIN_FITTED_PRIOR"),
                ("TWO_D_QUERY_LEAKAGE", leakage_logits, leakage_distance, None, "LEAKAGE_DIAGNOSTIC_ONLY"),
            ])
            for method, logits, distance, topk, baseline_type in prior_conditions:
                metrics = _b2_metrics(logits, distance, sample["polar"], topk_override=topk)
                row = {
                    **base,
                    "method": method,
                    "baseline_type": baseline_type,
                    "history_mapping": "",
                    "temporal_frame_ids": "",
                    "frame_tensor_changed": "",
                    "predicted_log_distance": distance,
                    **metrics,
                }
                method_rows[method].append(row)
                all_rows.append(row)
        if identity_max_diff > 1e-5:
            raise RuntimeError(
                f"B2 explicit-frame-id baseline mismatch for {model} P{prefix}: {identity_max_diff}"
            )
        base_by_id = {row["sample_id"]: row for row in method_rows["FULL_HISTORY_PLUS_QUERY"]}
        method_summary = {}
        valid_priors = ["GLOBAL_CONSTANT", "GLOBAL_ANGLE_HISTOGRAM", "TRAJECTORY_ONLY"]
        for method, rows in sorted(method_rows.items()):
            stats = _b2_method_stats(rows)
            low, high = _paired_b2_bootstrap(rows, base_by_id, "scene_id")
            method_summary[method] = {**stats, "scene_ci": [low, high]}
        best_prior = min(valid_priors, key=lambda name: method_summary[name]["mean"])
        base_error = method_summary["FULL_HISTORY_PLUS_QUERY"]["mean"]
        for method, summary in method_summary.items():
            all_aggregates.append({
                "experiment_id": f"paper_b2_{model}_l{spec['layer']}_p{prefix}",
                "analysis_type": "paper_prior_or_order_control",
                "task_variant": "B2",
                "model": model,
                "objective_family": spec["objective_family"],
                "layer": spec["layer"],
                "prefix": prefix,
                "baseline_or_readout": method,
                "seed": PHASE2_SEED,
                "n": summary["n"],
                "mean_ang_err_deg": summary["mean"],
                "median_ang_err_deg": summary["median"],
                "top1": summary["top1"],
                "top3": summary["top3"],
                "delta_error_vs_full": summary["mean"] - base_error,
                "delta_error_vs_best_prior": summary["mean"] - method_summary[best_prior]["mean"],
                "ci_low": summary["scene_ci"][0],
                "ci_high": summary["scene_ci"][1],
                "artifact_source": "analyses/probe_validity/b2_sample_diagnostics.csv.gz",
                "evidence_level": (
                    "LEAKAGE_DIAGNOSTIC_ONLY" if method == "TWO_D_QUERY_LEAKAGE"
                    else "TRAIN_FITTED_SINGLE_SEED42" if method in valid_priors
                    else "FIXED_PREDICTOR_PAIRED"
                ),
            })
        prefix_summaries[f"P{prefix}"] = {
            "checkpoint": rel(checkpoint),
            "historical_summary": rel(historical_summary),
            "train_dataset_n": train_bundle["dataset_n"],
            "train_valid_n": len(train_samples),
            "eval_dataset_n": eval_bundle["dataset_n"],
            "eval_valid_n": len(samples),
            "identity_frame_id_max_difference": identity_max_diff,
            "best_valid_prior": best_prior,
            "methods": method_summary,
            "G_prior": method_summary[best_prior]["mean"] - base_error,
            "G_query": method_summary["HISTORY_WITHOUT_QUERY"]["mean"] - base_error,
            "G_history": method_summary["CURRENT_FRAME_PLUS_QUERY"]["mean"] - base_error,
            "unavailable": {
                "QUERY_CATEGORY_ONLY": "CATEGORY_ANNOTATION_NOT_BOUND_TO_SAMPLES",
                "SCENE_TYPE_OR_LAYOUT_ONLY": "SCENE_LAYOUT_METADATA_NOT_AVAILABLE",
                "QUERY_X_LAYOUT": "SCENE_LAYOUT_METADATA_NOT_AVAILABLE",
                "EXACT_SCENE_ID": "NO_TRAIN_EVAL_SCENE_OVERLAP; LEAKAGE_DIAGNOSTIC_NOT_APPLICABLE",
            },
        }
        del probe, trajectory_model, leakage_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    payload = {
        "status": "PASS",
        "model": model,
        "layer": spec["layer"],
        "seed": PHASE2_SEED,
        "synthetic_order_sanity": synthetic,
        "prefixes": prefix_summaries,
        "aggregates": all_aggregates,
        "b2o_status": "B2O_NOT_FEASIBLE_WITH_CURRENT_DATA",
        "b2o_reason": (
            "No pair of real, pre-extracted visual trajectories shares one object query and "
            "the same action multiset in different orders. Permuting poses would create an "
            "off-trajectory geometry-oracle task, while permuting existing frame features "
            "would not supply the corresponding real final view/label. Backbone extraction "
            "and a geometry-oracle substitute are both out of scope."
        ),
    }
    _write_tmp_json(f"paper_b2_{model}.json", payload)
    _merge_strengthening_rows(OUT / "b2_sample_diagnostics.csv.gz", model, all_rows)
    return payload


def _merge_plain_analysis_rows(
    path: Path,
    new_rows: Sequence[Dict[str, object]],
    paper_analysis_prefix: str,
) -> None:
    existing = read_csv(path) if path.exists() else []
    existing = [
        row for row in existing
        if not str(row.get("analysis_type", "")).startswith(paper_analysis_prefix)
    ]
    fields = list(dict.fromkeys([
        *(key for row in existing for key in row),
        *(key for row in new_rows for key in row),
    ]))
    write_csv(path, [*existing, *new_rows], fields)


def run_paper_strengthen(model: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    before = snapshot(PROTECTED)
    models = CORE_MODELS if model == "all" else (model,)
    for current in models:
        print(f"[paper strengthening] starting {current}", flush=True)
        run_paper_c1_model(current)
        run_paper_b2_model(current)
    after = snapshot(PROTECTED)
    if before != after:
        raise RuntimeError("Protected files changed during paper-strengthening replay")
    print(json.dumps({
        "status": "PASS",
        "models": list(models),
        "protected_files_byte_identical": True,
    }, indent=2), flush=True)


def _paper_primary_c1(payload, prefix: int, horizon: int = 1):
    rows = [
        row for row in payload["aggregates"]
        if row.get("analysis_type") == "paper_candidate_k_robustness"
        and int(row.get("prefix", 0)) == prefix
        and int(row.get("horizon", 0)) == horizon
        and row.get("candidate_pool") == "MATCHED_TOTAL_POSE_DISPLACEMENT"
        and int(row.get("desired_candidate_count", 0)) == 8
    ]
    if not rows:
        raise RuntimeError(f"Missing primary C1 P{prefix}/h{horizon} rows")
    return {
        "r1": statistics.fmean(float(row["r1"]) for row in rows),
        "matched": statistics.fmean(float(row["matched_wrong_r1"]) for row in rows),
        "gain": statistics.fmean(float(row["true_minus_matched"]) for row in rows),
        "gain_sd": statistics.pstdev(float(row["true_minus_matched"]) for row in rows),
        "normalized": statistics.fmean(float(row["chance_normalized_r1"]) for row in rows),
        "coverage": statistics.fmean(float(row["k_coverage"]) for row in rows),
    }


def _paper_report_and_plots(c1_payloads, b2_payloads, before, after) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/probe_validity_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [MODEL_SPECS[model]["display"] for model in CORE_MODELS]
    colors = ["#168C8C", "#D45B46", "#3377B5", "#6F5A9A"]
    c1_primary = {model: _paper_primary_c1(c1_payloads[model], 12) for model in CORE_MODELS}
    p16_primary = {model: _paper_primary_c1(c1_payloads[model], 16) for model in CORE_MODELS}
    c1_table = [
        "| Model | P12 true R@1 | P12 matched R@1 | P12 gain (sampling SD) | K=8 coverage | P16 gain | Full - strongest retrained control |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in CORE_MODELS:
        p12 = c1_primary[model]
        p16 = p16_primary[model]
        ablation = c1_payloads[model]["ablations"]
        c1_table.append(
            f"| {MODEL_SPECS[model]['display']} | {p12['r1']:.4f} | {p12['matched']:.4f} | "
            f"{p12['gain']:+.4f} ({p12['gain_sd']:.4f}) | {p12['coverage']:.3f} | "
            f"{p16['gain']:+.4f} | {ablation['full_minus_strongest_control_r1']:+.4f} |"
        )
    horizon_table = [
        "| Model | h1 true/copy/action gain | h2 true/copy/action gain | h4 true/copy/action gain |",
        "|---|---|---|---|",
    ]
    for model in CORE_MODELS:
        values = []
        for horizon in PAPER_HORIZONS:
            row = c1_payloads[model]["horizons"][f"P12_h{horizon}"]
            values.append(
                f"{row['global_true_r1']:.4f}/{row['delta_copy']:+.4f}/{row['delta_action']:+.4f}"
            )
        horizon_table.append(f"| {MODEL_SPECS[model]['display']} | {' | '.join(values)} |")

    b2_table = [
        "| Model | P12 full | Best valid prior | G_prior | G_query | G_history | Content shuffle | Frame-token shuffle | Reverse content | Reverse tokens |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in CORE_MODELS:
        summary = b2_payloads[model]["prefixes"]["P12"]
        methods = summary["methods"]
        full = methods["FULL_HISTORY_PLUS_QUERY"]["mean"]
        best = summary["best_valid_prior"]
        b2_table.append(
            f"| {MODEL_SPECS[model]['display']} | {full:.2f} | {best} ({methods[best]['mean']:.2f}) | "
            f"{summary['G_prior']:+.2f} | {summary['G_query']:+.2f} | {summary['G_history']:+.2f} | "
            f"{methods['CONTENT_ONLY_SHUFFLE']['mean'] - full:+.2f} | "
            f"{methods['FRAME_TOKEN_SHUFFLE']['mean'] - full:+.2f} | "
            f"{methods['REVERSE_CONTENT_ONLY']['mean'] - full:+.2f} | "
            f"{methods['REVERSE_FRAME_TOKENS']['mean'] - full:+.2f} |"
        )
    p16_lines = []
    for model in CORE_MODELS:
        p12 = b2_payloads[model]["prefixes"]["P12"]
        p16 = b2_payloads[model]["prefixes"]["P16"]
        p12_order = p12["methods"]["CONTENT_ONLY_SHUFFLE"]["mean"] - p12["methods"]["FULL_HISTORY_PLUS_QUERY"]["mean"]
        p16_order = p16["methods"]["CONTENT_ONLY_SHUFFLE"]["mean"] - p16["methods"]["FULL_HISTORY_PLUS_QUERY"]["mean"]
        p16_lines.append(
            f"- {MODEL_SPECS[model]['display']}: C1 hard-negative gain "
            f"{c1_primary[model]['gain']:+.4f} -> {p16_primary[model]['gain']:+.4f}; "
            f"B2 content-shuffle delta {p12_order:+.2f} -> {p16_order:+.2f} deg; "
            f"query gain {p12['G_query']:+.2f} -> {p16['G_query']:+.2f} deg."
        )
    safety = [
        f"- `{path}` sha256={metadata['sha256']} "
        f"{'OK' if after[path] == metadata else 'CHANGED'}"
        for path, metadata in before.items()
    ]
    report = f"""# C1/B2 Probe Validity Study - Paper-Strengthening Extension

Status: `PAPER_EXTENSION_PARTIAL`

All numerical results are development-validation results from one probe-training seed (42). Candidate seeds 42-46 vary retrieval candidates only. They are not probe-training seed robustness.

## 1. Frozen Split Audit Summary

Phase-0 classification: `NOT_INDEPENDENT`. `configs/splits/inscene15k_v1.json` has a nominal 91-scene ScanNet++ test partition, but 63 of those scenes occur in C h124 train and 9 in C h124 val; at least 10 occur in observed B2 val rows. Exact evidence and hashes are in `analyses/probe_validity/FROZEN_SPLIT_AUDIT.md`. The split was not reused for confirmation and no new split was manufactured.

## 2. C1 Candidate-K and Pool Robustness

Primary pool: representation-independent `MATCHED_TOTAL_POSE_DISPLACEMENT`, K=8, candidate seeds 42-46. Physical candidates use only scene/window IDs and translation/rotation magnitudes. Absolute high-motion thresholds were frozen at translation >= 0.25, rotation >= 0.15 rad, or total pose displacement >= 0.30.

{chr(10).join(c1_table)}

The CSV additionally records K=4/8/16/32 for same-scene, same-trajectory, temporal-neighbor, matched-translation, matched-rotation, and matched-total-pose pools, including actual K, K coverage, chance-normalized R@1, MRR, rank percentile, paired cosine-margin rate, and scene/trajectory clustered CIs. Same-scene pools contain at most four windows, so K>4 is reported with reduced actual K rather than padded. `MATCHED_VIEW_OVERLAP` is unavailable because no run-bound overlap metadata is present.

Real action-confusable futures: `ACTION_CONFUSABLE_REAL_FUTURES_NOT_AVAILABLE`. Existing C3 rows relabel actions against factual trajectory targets; they do not provide two real branch futures for one anchor/context.

### 2.1 Representation Geometry Bridge

`REPRESENTATION_GEOMETRY_BRIDGE.md` is a read-only reuse of the historical P12 common-set C1 summary (276 samples, 69 scenes), not a new replay or confirmation. It identifies Wan as the highest raw temporal-persistence core model (last-state-to-target cosine 0.9789; displacement 0.0211), while DINO is highest on centered-copy cosine and global retrieval R@1. For Wan, centering retains nonzero copy alignment and full-minus-copy R@1 remains positive but imprecise. This supports `HIGH_TEMPORAL_SMOOTHNESS + WEAK_UNRESOLVED_INCREMENTAL_ACTION_SIGNAL`; it does not change this report's `PAPER_EXTENSION_PARTIAL` status or any predeclared gate. Whitening, covariance/effective rank, pairwise target geometry, and action-matched prediction separation are unavailable in the recovered aggregate record and are not inferred.

## 3. C1 Retrained Ablations

For every core model, FULL is the historical checkpoint replay. CONTEXT_ONLY, ACTION_ONLY, CONSTANT_ACTION, and CONSTANT_CONTEXT were trained for 50 epochs with AdamW, the historical hidden size/loss/batch size, exact P12 h1 train/eval targets, and seed 42. Absolute global and K=8 hard-pool metrics are in `c1_validity.csv`; the table above reports FULL minus the strongest retrained control. A positive gap supports context-action interaction under this dataset; a weak or negative gap means a retrained shortcut remains competitive.

## 4. C1 Horizon Stress

Every P12 h=1/2/4 row passed 100% exact-feature coverage against `streaming_window_indices[tail+h]`; no generic target reconstruction or fallback root was allowed. Cells are `true R@1 / true-copy / true-matched-wrong`.

{chr(10).join(horizon_table)}

## 5. B2 Prior Controls

All priors were fit on train labels only. Valid priors are spherical global constant, train joint-angle histogram, and trajectory-only metadata readout. `TWO_D_QUERY_LEAKAGE` is retained only as `LEAKAGE_DIAGNOSTIC_ONLY`. Category, scene-layout, and query-by-layout controls are explicitly unavailable rather than inferred from embeddings.

{chr(10).join(b2_table)}

`G_prior = E_best_valid_prior - E_full`, `G_query = E_history_without_query - E_full`, and `G_history = E_current_frame_plus_query - E_full`; positive values favor the full model.

## 6. B2 Order Sanity

Synthetic order sanity passed for the exact temporal mapping helper. Every real perturbation fixes the final frame, leaves query/GT/sample ID unchanged, uses all-valid tokens with no padding, and records both content mapping and frame IDs. The explicit identity-frame-ID path matches the historical forward within 1e-5 for every model/prefix. Content-only perturbations alter content-to-slot association; frame-token perturbations carry original temporal IDs with the moved content. Because the official encoder has no causal mask, frame-token movement is theoretically a permutation of an already labeled token set; near-zero effects there are expected and are not evidence that content-slot association is irrelevant.

## 7. B2-O

`B2O_NOT_FEASIBLE_WITH_CURRENT_DATA`. No real pre-extracted trajectory pair shares one object query and action multiset under different orders. Pose permutation would create an off-trajectory geometry-oracle task, while frame permutation lacks the corresponding real final view and label. Backbone extraction and a geometry-oracle substitute were prohibited, so no weak surrogate number is reported.

## 8. P16 Replication

P16 metrics were predeclared in code before replay: C1 true/copy/matched R@1, K=8 hard gain, fixed high-motion gain; B2 full/prior/query/current/remove-visible and four order variants. No P8/P24 result was opened for protocol redesign.

{chr(10).join(p16_lines)}

## 9. Frozen/Secondary Replication

Not run. Phase 0 found no independent existing test split for these probes. Calling the nominal master test untouched would be false because its scenes were consumed by the actual legacy train/val manifests.

## 10. Final Paper-Readiness Assessment

The extension is `PAPER_EXTENSION_PARTIAL`: development-set control evidence is complete, but an independent frozen confirmation is absent; real branch-paired C1 futures and a valid visual B2-O universe are unavailable. Claims must remain protocol-specific: V-JEPA2 versus VideoMAE is one predictive/masked comparison, and Wan/CogVideoX agreement or disagreement is descriptive. No universal objective-causal claim is supported.

Scope exclusions: no probe multi-seed; no controlled-subset study; no geometry oracle; no second predictive or masked-reconstruction family model; no feature/capacity equalization; no layer or full-prefix sweep; no backbone extraction.

Safety: protected files byte-identical=`{before == after}`; no canonical/registry/ledger update; no active evaluation directory modification.

{chr(10).join(safety)}
"""
    (OUT / "PROBE_VALIDITY_REPORT.md").write_text(report, encoding="utf-8")

    x = list(range(len(CORE_MODELS)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].bar([value - 0.18 for value in x], [c1_primary[m]["gain"] for m in CORE_MODELS], 0.36, label="P12", color=colors)
    axes[0].bar([value + 0.18 for value in x], [p16_primary[m]["gain"] for m in CORE_MODELS], 0.36, label="P16", color=colors, alpha=0.55)
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylabel("True - matched-wrong R@1")
    axes[0].set_title("C1 action specificity, K=8")
    axes[0].legend(frameon=False)
    for model, color in zip(CORE_MODELS, colors):
        values = [c1_payloads[model]["horizons"][f"P12_h{h}"]["delta_action"] for h in PAPER_HORIZONS]
        axes[1].plot(PAPER_HORIZONS, values, marker="o", label=MODEL_SPECS[model]["display"], color=color)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_xticks(PAPER_HORIZONS)
    axes[1].set_xlabel("Horizon")
    axes[1].set_ylabel("Global true - matched R@1")
    axes[1].set_title("C1 horizon stress")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "c1_validity.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    full = [b2_payloads[m]["prefixes"]["P12"]["methods"]["FULL_HISTORY_PLUS_QUERY"]["mean"] for m in CORE_MODELS]
    prior = [b2_payloads[m]["prefixes"]["P12"]["methods"][b2_payloads[m]["prefixes"]["P12"]["best_valid_prior"]]["mean"] for m in CORE_MODELS]
    axes[0].bar([value - 0.18 for value in x], full, 0.36, label="Full", color=colors)
    axes[0].bar([value + 0.18 for value in x], prior, 0.36, label="Best train prior", color=colors, alpha=0.55)
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylabel("Angular error (deg, lower better)")
    axes[0].set_title("B2 full versus prior")
    axes[0].legend(frameon=False)
    methods = ["CONTENT_ONLY_SHUFFLE", "FRAME_TOKEN_SHUFFLE", "REVERSE_CONTENT_ONLY", "REVERSE_FRAME_TOKENS"]
    for model, color in zip(CORE_MODELS, colors):
        summary = b2_payloads[model]["prefixes"]["P12"]["methods"]
        baseline = summary["FULL_HISTORY_PLUS_QUERY"]["mean"]
        axes[1].plot(range(4), [summary[name]["mean"] - baseline for name in methods], marker="o", label=MODEL_SPECS[model]["display"], color=color)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_xticks(range(4), ["Content\nshuffle", "Token\nshuffle", "Content\nreverse", "Token\nreverse"])
    axes[1].set_ylabel("Error increase vs full (deg)")
    axes[1].set_title("B2 order controls")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "b2_validity.png", dpi=180)
    plt.close(fig)


def run_paper_strengthen_summarize() -> None:
    required = [
        TMP / f"paper_{task}_{model}.json"
        for model in CORE_MODELS for task in ("c1", "b2")
    ]
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Paper-strengthening summaries missing: {missing}")
    c1_payloads = {model: load_json(str(TMP / f"paper_c1_{model}.json")) for model in CORE_MODELS}
    b2_payloads = {model: load_json(str(TMP / f"paper_b2_{model}.json")) for model in CORE_MODELS}
    if any(payload.get("status") != "PASS" for payload in [*c1_payloads.values(), *b2_payloads.values()]):
        raise RuntimeError("At least one paper-strengthening model payload is not PASS")
    before = snapshot(PROTECTED)
    c1_rows = [row for payload in c1_payloads.values() for row in payload["aggregates"]]
    b2_rows = [row for payload in b2_payloads.values() for row in payload["aggregates"]]
    _merge_plain_analysis_rows(OUT / "c1_validity.csv", c1_rows, "paper_")
    _merge_plain_analysis_rows(OUT / "b2_validity.csv", b2_rows, "paper_")
    after = snapshot(PROTECTED)
    if before != after:
        raise RuntimeError("Protected files changed during paper-strengthening summarization")
    _paper_report_and_plots(c1_payloads, b2_payloads, before, after)
    final_after = snapshot(PROTECTED)
    if before != final_after:
        raise RuntimeError("Protected files changed while rendering the final report")
    for path in required:
        path.unlink()
    print(json.dumps({
        "status": "PAPER_EXTENSION_PARTIAL",
        "models": list(CORE_MODELS),
        "protected_files_byte_identical": True,
        "temporary_receipts_deleted": True,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase2-replay", choices=["c1", "b2", "all"],
        help="Replay fixed historical checkpoint(s) for one Phase-2 model.",
    )
    parser.add_argument(
        "--model", choices=list(MODEL_SPECS), default="f3r",
        help="Model contract for replay or the B2 prior gate.",
    )
    parser.add_argument(
        "--phase2-summarize", action="store_true",
        help="Summarize two PASS sample dumps into the validity tables/report/plots.",
    )
    parser.add_argument(
        "--phase2-prior-gate", action="store_true",
        help="Check that the exact B2 train universe yields one prior label.",
    )
    parser.add_argument(
        "--paper-strengthen", choices=[*CORE_MODELS, "all"],
        help="Run the frozen paper-strengthening C1/B2 protocol for one/all core models.",
    )
    parser.add_argument(
        "--paper-strengthen-summarize", action="store_true",
        help="Merge four PASS strengthening receipts into the existing outputs/report.",
    )
    parser.add_argument(
        "--p16-repair", choices=["c1", "b2"],
        help="Run one evaluation-only P16 repair cell with immutable evidence output.",
    )
    parser.add_argument("--p16-output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--reference-summary", type=Path)
    args = parser.parse_args()
    if args.p16_repair:
        required = {
            "--p16-output": args.p16_output,
            "--checkpoint": args.checkpoint,
            "--checkpoint-sha256": args.checkpoint_sha256,
            "--reference-summary": args.reference_summary,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error(f"{args.p16_repair} repair requires {', '.join(missing)}")
        from analyses.probe_validity.p16_repair import run_cell
        run_cell(
            task=args.p16_repair,
            model=args.model,
            output=args.p16_output,
            checkpoint=args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
            reference_summary=args.reference_summary,
        )
    elif args.paper_strengthen:
        try:
            run_paper_strengthen(args.paper_strengthen)
        except Exception as error:
            blocked = {
                "status": "PAPER_EXTENSION_PARTIAL",
                "model": args.paper_strengthen,
                "reason": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            _write_tmp_json(f"paper_{args.paper_strengthen}_blocked.json", blocked)
            print(json.dumps(blocked, indent=2), flush=True)
            raise
    elif args.paper_strengthen_summarize:
        run_paper_strengthen_summarize()
    elif args.phase2_replay:
        try:
            run_phase2_replay(args.phase2_replay, args.model)
        except Exception as error:
            blocked = {
                "status": "MODEL_BLOCKED",
                "model": args.model,
                "task": args.phase2_replay,
                "reason": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            _write_tmp_json(f"{args.model}_{args.phase2_replay}_blocked.json", blocked)
            print(json.dumps(blocked, indent=2), flush=True)
            raise
    elif args.phase2_prior_gate:
        run_b2_prior_gate(args.model)
    elif args.phase2_summarize:
        run_phase2_summarize()
    else:
        run_phase1()


if __name__ == "__main__":
    main()
