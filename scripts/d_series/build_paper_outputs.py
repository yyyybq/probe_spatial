#!/usr/bin/env python3
"""Rebuild D-series tables/plots/text from complete real per-sample records only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from scripts.d_series.feature_store import MANIFEST_SHA256
from vidfm3d.utils.layout_metrics import scene_cluster_paired_bootstrap, scene_macro


REQUIRED = {
    "D1": {
        "models": ["vjepa2_vitl256", "wan21_t2v_1p3b", "dinov2_vitl14"],
        "branches": ["spatial", "pooled"],
        "conditions": ["true_target", "train_layout_prior", "within_scene_wrong_target", "across_scene_shuffled_target", "last_context", "zero", "random", "spatial_token_shuffle"],
    },
    "D2": {
        "models": ["vjepa2_vitl256", "wan21_t2v_1p3b", "dinov2_vitl14"],
        "branches": ["pooled"],
        "conditions": ["true_motion", "identity_motion", "matched_horizon_shuffled_motion", "independently_trained_context_only", "last_context", "train_layout_prior", "true_pooled_target_reference"],
    },
    "D3": {
        "models": ["vjepa2_vitl256"], "branches": ["spatial", "pooled"],
        "conditions": ["native_target_reference", "frozen_readout_transfer", "tokenwise_linear_adapter", "retrained_same_capacity_readout", "last_context", "train_layout_prior", "within_scene_wrong_target", "early_tubelet_order_shuffle"],
    },
}


def load_records(run_root: Path):
    records, sources = [], []
    for path in sorted(run_root.rglob("per_sample_metrics.csv")):
        receipt_path = path.parent / "run_receipt.json"
        if not receipt_path.is_file(): raise RuntimeError(f"missing receipt for {path}")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") != "complete_development_evaluation": raise RuntimeError(f"incomplete run {receipt_path}")
        if receipt.get("manifest_sha256") != MANIFEST_SHA256: raise RuntimeError(f"manifest mismatch {receipt_path}")
        with path.open() as handle: rows = list(csv.DictReader(handle))
        if not rows: raise RuntimeError(f"empty records {path}")
        records.extend(rows); sources.append(str(path))
    if not records: raise RuntimeError("no completed per-sample records found")
    return records, sources


def gate(records):
    present = defaultdict(set)
    for row in records:
        if row["manifest_sha256"] != MANIFEST_SHA256 or row["split"] != "dev":
            raise RuntimeError(f"non-frozen record: {row.get('source_run')}")
        present[(row["experiment"], row["model"], row["branch"], int(row["seed"]))].add(row["condition"])
    missing = []
    for exp, spec in REQUIRED.items():
        for model in spec["models"]:
            for branch in spec["branches"]:
                for seed in (0, 1, 2):
                    absent = set(spec["conditions"]) - present[(exp, model, branch, seed)]
                    if absent: missing.append({"experiment": exp, "model": model, "branch": branch, "seed": seed, "conditions": sorted(absent)})
    if missing: raise RuntimeError("formal matrix incomplete; refusing paper outputs:\n" + json.dumps(missing, indent=2))


def group_rows(records, experiment):
    grouped = defaultdict(list)
    for row in records:
        if row["experiment"] == experiment:
            grouped[(row["model"], row["branch"], int(row["seed"]), row["condition"])].append(row)
    return grouped


def paired(records, condition, comparator):
    return scene_cluster_paired_bootstrap(records, condition, comparator, n_boot=10_000, seed=20270301)


def table_rows(records, experiment):
    grouped, output = group_rows(records, experiment), []
    all_grouped = defaultdict(list)
    for (model, branch, seed, condition), rows in grouped.items(): all_grouped[(model, branch, seed)].extend(rows)
    for (model, branch, seed, condition), rows in sorted(grouped.items()):
        base = all_grouped[(model, branch, seed)]
        result = {
            "model": model, "branch": branch, "condition": condition, "seed": seed,
            "scene_macro_mae": scene_macro(rows, "mae"), "scene_macro_absrel": scene_macro(rows, "absrel"),
            "scene_macro_delta1": scene_macro(rows, "delta1"), "scene_macro_coarse_mae": scene_macro(rows, "coarse_mae"),
            "scene_macro_detail_mae": scene_macro(rows, "detail_mae"),
            "scene_macro_oracle_gt_scale_pointmap_mae_m": scene_macro(rows, "oracle_gt_scale_pointmap_mae_m"),
            "n": len({r["sample_id"] for r in rows}), "scenes": len({r["scene_id"] for r in rows}),
            "split": "dev", "manifest_sha256": MANIFEST_SHA256,
            "source_run": ";".join(sorted({r["source_run"] for r in rows})),
        }
        comparator = "train_layout_prior" if experiment in {"D1", "D2"} else "native_target_reference"
        ci = paired(base, condition, comparator)
        result |= {f"delta_mae_vs_{comparator}": ci["delta"], "delta_ci_low": ci["ci_low"], "delta_ci_high": ci["ci_high"]}
        if experiment == "D1" and condition == "true_target":
            wrong = paired(base, condition, "within_scene_wrong_target")
            result |= {"delta_mae_vs_within_scene_wrong_target": wrong["delta"], "wrong_ci_low": wrong["ci_low"], "wrong_ci_high": wrong["ci_high"]}
        if experiment == "D2" and condition == "true_motion":
            for short, other in (("context_only", "independently_trained_context_only"), ("shuffled", "matched_horizon_shuffled_motion")):
                value = paired(base, condition, other)
                result |= {f"delta_mae_vs_{short}": value["delta"], f"{short}_ci_low": value["ci_low"], f"{short}_ci_high": value["ci_high"]}
        output.append(result)
    return output


def write_table(rows, csv_path, tex_path):
    fields = sorted({key for row in rows for key in row}, key=lambda x: (x not in ["model", "branch", "condition", "seed"], x))
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    columns = ["model", "branch", "condition", "seed", "scene_macro_mae", "delta_ci_low", "delta_ci_high", "n", "scenes", "source_run"]
    def esc(value): return str(value).replace("_", r"\_").replace("%", r"\%")
    lines = [r"\begin{tabular}{llllrrrrrr}", r"\toprule", " & ".join(columns) + r" \\", r"\midrule"]
    for row in rows:
        values = []
        for key in columns:
            value = row.get(key, "")
            values.append(f"{value:.4f}" if isinstance(value, float) and math.isfinite(value) else esc(value))
        lines.append(" & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n")


def plots(records, out):
    d3 = [r for r in records if r["experiment"] == "D3" and r["condition"] in {"native_target_reference", "frozen_readout_transfer", "tokenwise_linear_adapter", "retrained_same_capacity_readout"}]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, branch in zip(axes, ("spatial", "pooled")):
        subset = [r for r in d3 if r["branch"] == branch]
        conditions = sorted({r["condition"] for r in subset})
        for condition in conditions:
            ys = [scene_macro([r for r in subset if r["condition"] == condition and int(r["seed"]) == seed and int(r["horizon"]) == h], "mae") for h in (2, 4) for seed in (0, 1, 2)]
            means = [np.mean(ys[:3]), np.mean(ys[3:])]
            ax.plot([2, 4], means, marker="o", label=condition)
        ax.set(title=branch, xlabel="horizon (observation steps)", xticks=[2, 4]); ax.grid(alpha=.25)
    axes[0].set_ylabel("scene-macro normalized-depth MAE"); axes[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "d3_transfer_gap_by_horizon.pdf"); fig.savefig(out / "d3_transfer_gap_by_horizon.png", dpi=180); plt.close(fig)

    d2 = [r for r in records if r["experiment"] == "D2" and r["condition"] == "true_motion" and r.get("cosine") not in {None, "", "None"}]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, model in zip(axes, REQUIRED["D2"]["models"]):
        subset = [r for r in d2 if r["model"] == model]
        ax.scatter([float(r["cosine"]) for r in subset], [float(r["mae"]) for r in subset], s=5, alpha=.2)
        ax.set(title=model, xlabel="latent cosine"); ax.grid(alpha=.2)
    axes[0].set_ylabel("normalized-depth MAE"); fig.tight_layout()
    fig.savefig(out / "cosine_vs_depth_error.pdf"); fig.savefig(out / "cosine_vs_depth_error.png", dpi=180); plt.close(fig)


def write_text(rows_by_exp, out):
    def best(exp, condition):
        rows = [r for r in rows_by_exp[exp] if r["condition"] == condition]
        return min(rows, key=lambda r: r["scene_macro_mae"])
    d1, d2, d3 = best("D1", "true_target"), best("D2", "true_motion"), best("D3", "frozen_readout_transfer")
    paper = rf"""\subsection{{Read-Then-Predict Layout Consistency}}
We evaluate whether normalized target-view depth is readable from frozen representations and remains readable after prediction. The frozen development protocol contains 681 ScanNet++ scenes (612 train, 69 development), 2,721 windows, and horizons 2 and 4; no historically unused test scenes were available, so these results are development evidence rather than held-out benchmark results. The primary metric is scene-macro normalized-depth MAE with scene-cluster paired bootstrap intervals.

Across the complete D1 matrix, the lowest single-seed target-reference MAE was {d1['scene_macro_mae']:.3f} ({d1['model']}, {d1['branch']}, seed {d1['seed']}). In D2, the lowest true-motion MAE was {d2['scene_macro_mae']:.3f} ({d2['model']}, seed {d2['seed']}); motion dependence is judged only by paired true-vs-context-only and true-vs-shuffled intervals in Table~\ref{{tab:d2}}, not by ranking against last context alone. For V-JEPA2 D3, frozen readout transfer is separated from token-wise linear alignment and same-capacity readout retraining; the best single-seed frozen-transfer MAE was {d3['scene_macro_mae']:.3f} ({d3['branch']}, seed {d3['seed']}). A weak target reference limits any claim about dynamics, and failed recovery under these probes does not establish information-theoretic loss. Pose conditioning is derived from camera trajectories and is not an executed counterfactual action.
"""
    (out / "paper_section.tex").write_text(paper)
    slides = f"""# 第1页：科学问题与冻结协议

- 问题：目标视图深度能否从表示读出，且这种可读性是否在预测后保持？
- ScanNet++ development-only：612 train / 69 dev / 0 held-out test 场景；2,721 windows；prefix=8，horizon={{2,4}}。
- 三个冻结 backbone：V-JEPA2 ViT-L/256、Wan2.1-T2V-1.3B layer20/index749、DINOv2 ViT-L/14。
- 主指标：scene-macro normalized-depth MAE；scene-cluster paired bootstrap 95% CI。

# 第2页：三组正式主结果

- D1 最低单 seed target-reference MAE：{d1['scene_macro_mae']:.3f}（{d1['model']} / {d1['branch']} / seed {d1['seed']}）。
- D2 最低单 seed true-motion MAE：{d2['scene_macro_mae']:.3f}（{d2['model']} / seed {d2['seed']}）；结论以 true-vs-context-only、true-vs-shuffled 配对 CI 为准。
- D3 最低单 seed frozen-transfer MAE：{d3['scene_macro_mae']:.3f}（{d3['branch']} / seed {d3['seed']}）；另报 linear adapter 与同容量 retrained readout。
- 这里展示完整三 seed，不挑最好 checkpoint 替代总体结果。

# 第3页：机制与结论边界

- 4×4 粗布局、有效邻域深度梯度、within-scene wrong target、spatial-token/order shuffle 共同区分样本结构与位置先验。
- 联合检查 latent cosine、depth error、预测/目标方差，识别“高 cosine 但向平均状态收缩”。
- Frozen transfer 差而 linear/retrained 恢复，只支持接口/受限可恢复性解释；三者都差也不等于信息彻底消失。
- 相机位姿只是 pose-derived condition，不证明已执行动作下的 counterfactual causality；当前结果不是 held-out test。
"""
    (out / "slides_zh.md").write_text(slides)
    claims = """# Claims and evidence

| Claim | Required evidence | Boundary |
|---|---|---|
| Target-view depth is sample-specifically readable | D1 true target must improve over both train-layout prior and within-scene wrong target with paired scene-cluster CI | Spatial and pooled branches are judged separately; target reference is not a mathematical ceiling. |
| External prediction uses pose-derived motion | D2 true motion must improve over independently trained context-only and matched-horizon shuffled motion with paired CI | Identity-at-test is an ablation, not the unconditional baseline; this is not counterfactual action causality. |
| Native predicted latents preserve transferable layout | D3 frozen transfer relative to matched native target | Failure can be an interface mismatch. |
| A simple interface mismatch explains part of a transfer gap | D3 token-wise linear adapter recovers frozen transfer | Linear recovery does not prove that all loss is a basis shift. |
| Layout remains recoverable under the probe budget | D3 same-capacity retrained readout improves frozen transfer and prior | Non-recovery does not prove information-theoretic absence. |
| High latent similarity is geometrically meaningful | cosine/error plot together with predicted/target variance and prior | Correlation is descriptive, not causal. |

All cited values come from the three generated CSV tables and their `source_run`/manifest columns. The entire evidence set is development-only because no historically unused test scenes were available.
"""
    (out / "claims_and_evidence.md").write_text(claims)


def visual_index(records, out):
    candidates = [r for r in records if r["experiment"] == "D3" and r["condition"] == "frozen_readout_transfer"]
    grouped = defaultdict(list)
    for row in candidates: grouped[(row["branch"], row["horizon"], row["sample_id"])].append(float(row["mae"]))
    rows = []
    for (branch, horizon), values in __import__("itertools").groupby(sorted(grouped), key=lambda key: key[:2]):
        keys = list(values); ranked = sorted(((float(np.mean(grouped[key])), key[2]) for key in keys))
        picks = [("best", ranked[0]), ("median", ranked[len(ranked)//2]), ("worst", ranked[-1])]
        for rule, (mae, sample_id) in picks:
            exemplar = next(r for r in candidates if r["branch"] == branch and r["horizon"] == horizon and r["sample_id"] == sample_id)
            rows.append({"selection_rule": rule, "branch": branch, "horizon": horizon, "sample_id": sample_id,
                         "scene_id": exemplar["scene_id"], "window_id": exemplar["window_id"], "mean_across_seed_mae": mae})
    with (out / "visual_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run-root", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(); records, sources = load_records(args.run_root); gate(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = {exp: table_rows(records, exp) for exp in ("D1", "D2", "D3")}
    for exp, stem in (("D1", "table_d1_decodability"), ("D2", "table_d2_motion_conditioning"), ("D3", "table_d3_native_transfer")):
        write_table(tables[exp], args.output_dir / f"{stem}.csv", args.output_dir / f"{stem}.tex")
    plots(records, args.output_dir); visual_index(records, args.output_dir); write_text(tables, args.output_dir)
    (args.output_dir / "sources.json").write_text(json.dumps({"manifest_sha256": MANIFEST_SHA256, "per_sample_files": sources}, indent=2) + "\n")


if __name__ == "__main__": main()
