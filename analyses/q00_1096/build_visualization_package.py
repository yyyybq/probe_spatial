#!/usr/bin/env python3
"""Build the frozen-1096 global visualization package.

This script is display-first: it preserves every universe row and exposes quality
flags instead of filtering questionable historical results out of the source table.
Cross-capability plots use direction-corrected, task-wise ranks; raw metrics are
never averaged across tasks.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/analysis/q00_1096_visualization_v1_full_20260917T111025Z"
FIG = OUT / "figures"
TAB = OUT / "tables"
DATA = OUT / "data"
INV = OUT / "inventory"

SOURCE = ROOT / "results/analysis/q00_1096_loose/all_results_1096_loose.csv"
CANONICAL_ALL = ROOT / "results/canonical/all_results.csv"
C1_VALIDITY = ROOT / "analyses/probe_validity/c1_validity.csv"
WAN_VALIDITY = ROOT / "results/analysis/wan_p12_l20_action_validity/tables/condition_metrics.csv"

MODEL_META = {
    "f3r": ("Geometry-specialized", "F3R"),
    "dino": ("Image discriminative", "DINO"),
    "videomae": ("Masked-video", "VideoMAE"),
    "vjepa2": ("Predictive-video", "V-JEPA2"),
    "qwen2_5_vl": ("Multimodal/VLM", "Qwen2.5-VL"),
    "wan": ("Video-generation", "Wan"),
    "cogvideox": ("Video-generation", "CogVideoX"),
    "aether": ("Video-generation", "Aether"),
    "minimax_h3": ("Video-generation", "MiniMax-H3"),
    "lingbot": ("Video-generation", "LingBot-video"),
}
FAMILY_ORDER = [
    "Geometry-specialized", "Image discriminative", "Masked-video",
    "Predictive-video", "Multimodal/VLM", "Video-generation",
]
FAMILY_COLORS = {
    "Geometry-specialized": "#7B2CBF",
    "Image discriminative": "#0077B6",
    "Masked-video": "#2A9D8F",
    "Predictive-video": "#E9C46A",
    "Multimodal/VLM": "#E76F51",
    "Video-generation": "#D62828",
}
FAMILY_MARKERS = {
    "Geometry-specialized": "P", "Image discriminative": "o",
    "Masked-video": "s", "Predictive-video": "D",
    "Multimodal/VLM": "^", "Video-generation": "X",
}

LAYER_GRIDS = {
    "dino": [0], "f3r": [24], "vjepa2": [0, 5, 11, 17, 23],
    "videomae": [0, 5, 11, 17, 23], "qwen2_5_vl": [-1],
    "wan": [0, 5, 10, 15, 20, 25, 29],
    "cogvideox": [0, 5, 10, 15, 20, 25, 30, 35, 40],
    "aether": [1], "minimax_h3": [10, 20, 30, 40, 50],
    "lingbot": [10, 20, 30, 40, 48],
}
# Total backbone depth is independent of the sampled layer grid.  Values below
# are the last zero-based transformer block indices.  CLI aliases for the two
# 1-based terminal labels are resolved by ``actual_layer_index``.
BACKBONE_LAST_INDEX = {
    "vjepa2": 23, "videomae": 23, "wan": 29, "cogvideox": 41,
    "minimax_h3": 49, "lingbot": 47,
}
SINGLE_REPRESENTATION_MODELS = {"dino", "f3r", "qwen2_5_vl", "aether"}
REP_LAYER = {
    "dino": 0, "f3r": 24, "vjepa2": 23, "videomae": 23,
    "qwen2_5_vl": -1, "wan": 20, "cogvideox": 20, "aether": 1,
    "minimax_h3": 30, "lingbot": 30,
}

CAPABILITY_META = {
    "A1 Depth": ("A1", "streaming_depth", "mean_loss_depth", "min"),
    "A1 Mask": ("A1", "streaming_mask", "mean_loss_identity", "min"),
    "A1 Camera": ("A1", "streaming_depth/camera", "Auc_30", "max"),
    "A2 View": ("A2", "view_consistency", "balanced_acc", "max"),
    "B2 Persistence": ("B2", "ego_belief_v2", "mean_ang_err_deg", "min"),
    "C1 Dynamics": ("C1", "action_dynamics", "global_R@1", "max"),
    "C2 Integration": ("C2", "path_integration", "global_R@1", "max"),
    "C3 Counterfactual": ("C3", "counterfactual", "intervention_validity", "max"),
}
CAP_ORDER = list(CAPABILITY_META)
PANEL_ORDER = [
    "A1 Geometry", "A1 Mask", "A2 View", "B2 Persistence", "C1 Dynamics",
    "C2 Integration", "C3 Counterfactual",
]

# Stable internal keys above are retained for data joins.  All rendered paper
# labels use the final task nomenclature.
DISPLAY_TASK = {
    "A1 Geometry": "Metric geometry",
    "A1 Depth": "Metric geometry: depth",
    "A1 Camera": "Metric geometry: camera pose",
    "A1 Mask": "Instance identity",
    "A2 View": "View consistency",
    "B2 Persistence": "Out-of-view object localization",
    "C1 Dynamics": "One-step prediction",
    "C2 Integration": "Path integration",
    "C3 Counterfactual": "Counterfactual control",
}


def display_task(task: str) -> str:
    return DISPLAY_TASK.get(task, task)

QUALITY_ORDER = {
    "clean/canonical": 0, "historical/provisional": 1, "legacy": 2,
    "known mismatch": 3, "missing": 4,
}
QUALITY_MARK = {
    "clean/canonical": "", "historical/provisional": "*",
    "legacy": "L", "known mismatch": "†", "missing": "×",
}

TASK_DEFINITIONS = [
    ["L1", "Metric geometry", "Metric geometry", "Streaming visual features",
     "Metric depth; camera pose", "Depth loss ↓; AUC@30 ↑",
     "Linearly accessible scene geometry"],
    ["L1", "Instance identity", "Instance identity", "Streaming visual features",
     "Per-pixel instance embeddings", "Identity loss ↓",
     "Identity separation; not mIoU"],
    ["L1", "View consistency", "View consistency", "Pair of streaming views",
     "Whether views overlap", "Balanced accuracy ↑",
     "Viewpoint-invariant scene correspondence"],
    ["L2", "Out-of-view object localization", "Out-of-view object localization", "Object query + history tokens",
     "Object bearing in final-view frame", "Angular error (deg) ↓",
     "Persistence of an occluded object state"],
    ["L3", "One-step prediction", "One-step prediction", "Context + relative camera action",
     "Isolated next-state feature", "Global R@1 ↑",
     "Action-conditioned state update"],
    ["L3", "Path integration", "Path integration", "Context + action sequence",
     "Rolled-out target feature", "Global R@1 ↑",
     "Accumulation of egomotion over a path"],
    ["L3 control", "Counterfactual control", "Counterfactual control", "Context + alternative action",
     "Intervened target state", "Intervention validity ↑",
     "Sensitivity to a changed action"],
]


def configure_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.family": "DejaVu Sans",
        "font.size": 16, "axes.titlesize": 22, "axes.labelsize": 18,
        "xtick.labelsize": 15, "ytick.labelsize": 15,
        "legend.fontsize": 14, "figure.titlesize": 24,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save_figure(fig: plt.Figure, stem: str, directory: Path | None = None) -> None:
    directory = directory or FIG
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "yes", "1"}


def actual_layer_index(model: str, layer: int) -> int | None:
    if model == "lingbot" and layer == 48:
        return 47
    if model == "minimax_h3" and layer == 50:
        return 49
    if model in SINGLE_REPRESENTATION_MODELS or layer < 0:
        return None
    return layer


def norm_depth(model: str, layer: int) -> tuple[float, str]:
    actual = actual_layer_index(model, layer)
    if actual is None:
        return float("nan"), "single exposed output/merger representation; structural depth N/A"
    last = BACKBONE_LAST_INDEX.get(model)
    if last is None or last <= 0:
        return float("nan"), "backbone depth unavailable; structural depth N/A"
    return actual / last, f"actual_block_index / {last} (true backbone last index)"


def classify_quality(row: pd.Series) -> tuple[str, bool, bool, bool, bool, bool]:
    tier = str(row.get("result_quality_tier", ""))
    lineage = " ".join(str(row.get(c, "")) for c in [
        "lineage_status", "scientific_lineage_class", "window_evidence_grade",
    ]).upper()
    mismatch = (
        tier.startswith("T4") or truth(row.get("known_label_feature_mismatch"))
        or truth(row.get("known_manifest_mismatch"))
        or "AFFECTED_CONFIRMED" in lineage or "CONFIRMED_ERROR" in lineage
    )
    provisional = tier.startswith("T1") or tier.startswith("T2")
    legacy = "LEGACY" in tier or "LEGACY" in str(row.get("result_source", "")).upper()
    missing = not truth(row.get("selected_result_finite"))
    strict_clean = "EXACT_SAFE" in lineage or "SAFE_EXACT" in lineage
    if missing:
        quality = "missing"
    elif mismatch:
        quality = "known mismatch"
    elif provisional:
        quality = "historical/provisional"
    elif legacy:
        quality = "legacy"
    else:
        quality = "clean/canonical"
    return quality, mismatch, provisional, legacy, missing, strict_clean


def build_analysis_table() -> pd.DataFrame:
    raw = pd.read_csv(SOURCE)
    assert len(raw) == 1096 and raw.universe_id.is_unique
    raw["layer"] = pd.to_numeric(raw["layer"], errors="coerce").astype("Int64")
    raw["prefix"] = pd.to_numeric(raw["prefix"], errors="coerce").astype("Int64")
    raw["metric_value"] = pd.to_numeric(raw["selected_result"], errors="coerce")
    raw["metric_name"] = raw["selected_metric_name"].fillna(raw["primary_target_metric"])
    raw["metric_direction"] = raw["primary_target_direction"].map({"max": "higher_is_better", "min": "lower_is_better"})
    raw["task_family"] = raw["family"]
    raw["model_family"] = raw.model.map(lambda x: MODEL_META.get(x, ("Other", x))[0])
    raw["model_display"] = raw.model.map(lambda x: MODEL_META.get(x, ("Other", x))[1])
    normalized = [norm_depth(m, int(l)) for m, l in zip(raw.model, raw.layer)]
    raw["normalized_layer_depth"] = [x[0] for x in normalized]
    raw["layer_mapping_note"] = [x[1] for x in normalized]
    quality = raw.apply(classify_quality, axis=1, result_type="expand")
    quality.columns = ["result_quality", "known_mismatch", "provisional", "legacy", "missing", "strict_clean"]
    raw = pd.concat([raw, quality], axis=1)
    raw["visualization_clean"] = (raw.result_quality == "clean/canonical") & raw.selected_result_finite.astype(bool)
    raw["result_status"] = raw["result_quality_tier"]
    wanted = [
        "universe_id", "model", "model_display", "model_family", "model_variant",
        "campaign", "protocol", "task_family", "probe", "layer",
        "normalized_layer_depth", "layer_mapping_note", "prefix", "horizon",
        "default_timestep", "metric_name", "metric_value", "metric_direction",
        "result_source", "result_quality", "result_status", "canonical_valid",
        "scientifically_comparable_reconciled", "known_mismatch", "provisional",
        "legacy", "missing", "strict_clean", "visualization_clean",
        "known_label_feature_mismatch", "known_manifest_mismatch", "lineage_status",
        "scientific_lineage_class", "window_evidence_grade", "phase1_resolution",
        "result_quality_tier", "selected_result_path", "warning",
    ]
    out = raw[wanted].copy()
    out.to_csv(OUT / "analysis_table.csv", index=False)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(SOURCE.relative_to(ROOT)),
        "input_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "row_count": len(out),
        "finite_primary_results": int(np.isfinite(out.metric_value).sum()),
        "visualization_clean_definition": "T0/canonical-source with no confirmed known mismatch; lineage-unresolved rows remain",
        "strict_clean_definition": "SAFE_EXACT / EXACT_SAFE lineage evidence",
        "strict_clean_rows": int(out.strict_clean.astype(bool).sum()),
        "representative_layers": REP_LAYER,
        "layer_grids": LAYER_GRIDS,
        "backbone_last_indices": BACKBONE_LAST_INDEX,
        "special_layer_mapping": {
            "qwen2_5_vl:-1": "output/merger representation; normalized structural depth is N/A",
            "single_exposed_layer": "plotted as an isolated point in a separate output/merger band",
        },
        "prefix_aggregation": "arithmetic mean over available p8/p12/p16/p24 at representative layer; never best-prefix",
        "cross_task_normalization": "direction correction followed by within-task percentile rank",
        "protocols": sorted(out.protocol.astype(str).unique().tolist()),
        "campaigns": sorted(out.campaign.astype(str).unique().tolist()),
    }
    (DATA / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return out


def capability_for(probe: str, metric: str) -> str | None:
    if probe == "streaming_depth" and metric == "mean_loss_depth":
        return "A1 Depth"
    if probe in {"streaming_camera", "streaming_depth/camera"} and metric == "Auc_30":
        return "A1 Camera"
    for cap, (_, p, m, _) in CAPABILITY_META.items():
        if probe == p and metric == m:
            return cap
    return None


def metric_landscape(analysis: pd.DataFrame) -> pd.DataFrame:
    base = analysis[analysis.metric_value.notna()].copy()
    base["capability"] = [capability_for(p, m) for p, m in zip(base.probe, base.metric_name)]
    base = base[base.capability.notna()].copy()
    base["artifact_metric_source"] = base.selected_result_path
    base["source_universe_id"] = base.universe_id

    # A1 camera is an atomic output of streaming_depth, not a separate universe row.
    camera = pd.read_csv(CANONICAL_ALL)
    camera = camera[(camera.metric_name == "Auc_30") & (camera.probe.str.contains("camera", na=False))].copy()
    camera["layer"] = pd.to_numeric(camera.layer, errors="coerce").astype("Int64")
    camera["prefix"] = pd.to_numeric(camera.prefix, errors="coerce").astype("Int64")
    camera["metric_value"] = pd.to_numeric(camera.metric_value, errors="coerce")
    inherited = []
    parent_pool = analysis[analysis.probe.eq("streaming_depth")]
    for row in camera.itertuples(index=False):
        parents = parent_pool[(parent_pool.model == row.model) & (parent_pool.layer == row.layer) & (parent_pool.prefix == row.prefix)]
        if parents.empty:
            continue
        # Duplicate into each matching frozen campaign identity; this is an atomic
        # metric view of an existing configuration, never a new universe row.
        for parent in parents.itertuples(index=False):
            rec = parent._asdict()
            rec.update({
                "probe": "streaming_depth/camera", "metric_name": "Auc_30",
                "metric_value": float(row.metric_value), "metric_direction": "higher_is_better",
                "capability": "A1 Camera", "artifact_metric_source": "results/canonical/all_results.csv",
                "source_universe_id": parent.universe_id,
            })
            inherited.append(rec)
    if inherited:
        base = pd.concat([base, pd.DataFrame(inherited)], ignore_index=True, sort=False)
    base["oriented_value"] = np.where(base.metric_direction.eq("lower_is_better"), -base.metric_value, base.metric_value)
    base.to_csv(DATA / "metric_landscape.csv", index=False)
    return base


def model_order(models: list[str] | pd.Series) -> list[str]:
    found = set(models)
    return sorted(found, key=lambda m: (
        FAMILY_ORDER.index(MODEL_META.get(m, ("Video-generation", m))[0])
        if MODEL_META.get(m, ("Video-generation", m))[0] in FAMILY_ORDER else 99,
        MODEL_META.get(m, ("Other", m))[1],
    ))


def worst_quality(values: pd.Series) -> str:
    vals = [v for v in values.dropna() if v in QUALITY_ORDER]
    return max(vals, key=lambda x: QUALITY_ORDER[x]) if vals else "missing"


def representative_rows(metrics: pd.DataFrame, clean_only: bool = False) -> pd.DataFrame:
    d = metrics.copy()
    if clean_only:
        d = d[d.result_quality.eq("clean/canonical")]
    pieces = []
    for model, rep in REP_LAYER.items():
        z = d[(d.model == model) & (d.layer == rep)].copy()
        if z.empty:
            continue
        # Prefer default8 for core models whenever duplicate campaign identities exist.
        z["campaign_priority"] = np.where(z.campaign.eq("default8"), 0, 1)
        for cap, q in z.groupby("capability", sort=False):
            identity_cols = ["campaign", "protocol", "model_variant"]
            identity = q.groupby(identity_cols, dropna=False).size().sort_values(ascending=False)
            chosen = identity.index[0]
            if (q.campaign == "default8").any():
                default_identity = q[q.campaign == "default8"].groupby(identity_cols, dropna=False).size().sort_values(ascending=False)
                chosen = default_identity.index[0]
            q = q[(q.campaign == chosen[0]) & (q.protocol == chosen[1]) & (q.model_variant == chosen[2])]
            available_prefixes = sorted(q.loc[q.metric_value.notna(), "prefix"].dropna().astype(int).unique())
            expected_prefixes = [8, 12, 16, 24]
            pieces.append({
                "model": model, "model_display": MODEL_META[model][1],
                "model_family": MODEL_META[model][0], "representative_layer": rep,
                "model_variant": chosen[2], "campaign": chosen[0], "protocol": chosen[1],
                "capability": cap, "metric_name": q.metric_name.iloc[0],
                "metric_direction": q.metric_direction.iloc[0],
                "raw_mean": q.metric_value.mean(), "raw_sd_across_prefix": q.metric_value.std(ddof=1),
                "oriented_mean": q.oriented_value.mean(), "n_prefix": len(available_prefixes),
                "n_prefix_expected": len(expected_prefixes),
                "prefixes": ",".join(map(str, available_prefixes)),
                "expected_prefixes": ",".join(map(str, expected_prefixes)),
                "prefix_coverage": f"{len(available_prefixes)}/{len(expected_prefixes)}",
                "aggregation_status": "complete" if set(available_prefixes) == set(expected_prefixes) else "partial_available_prefix_mean",
                "quality": worst_quality(q.result_quality),
                "strict_clean_rows": int(q.strict_clean.astype(bool).sum()),
                "source_universe_ids": ";".join(q.source_universe_id.astype(str)),
            })
    out = pd.DataFrame(pieces)
    if out.empty:
        return out
    out["relative_score"] = np.nan
    for cap, idx in out.groupby("capability").groups.items():
        vals = out.loc[idx, "oriented_mean"]
        ranks = vals.rank(method="average")
        out.loc[idx, "relative_score"] = 0.5 if len(vals) == 1 else (ranks - 1) / (len(vals) - 1)
    return out


def inventory(analysis: pd.DataFrame) -> None:
    rows = []
    for model in model_order(analysis.model.tolist()):
        z = analysis[analysis.model == model]
        rows.append({
            "section": "by_model", "value": MODEL_META[model][1], "count": len(z),
            "tasks": ", ".join(sorted(z.probe.unique())),
            "prefixes": ", ".join(map(str, sorted(z.prefix.dropna().astype(int).unique()))),
            "layers": ", ".join(map(str, sorted(z.layer.dropna().astype(int).unique()))),
        })
    for col in ["task_family", "model_family", "prefix", "layer", "campaign", "protocol", "result_quality"]:
        for key, n in analysis.groupby(col, dropna=False).size().items():
            rows.append({"section": f"by_{col}", "value": key, "count": n, "tasks": "", "prefixes": "", "layers": ""})
    inv = pd.DataFrame(rows)
    inv.to_csv(INV / "result_inventory.csv", index=False)
    numeric = int(analysis.metric_value.notna().sum())
    strict = int(analysis.strict_clean.astype(bool).sum())
    lines = [
        "# 1096 Result Inventory", "", f"- Total configurations: **{len(analysis)}**",
        f"- Numeric primary results: **{numeric}**", f"- Missing primary results: **{len(analysis)-numeric}**",
        f"- Strict exact-safe rows: **{strict}**", "", "## Model × sweep inventory", "",
        "| Model | #Configs | Tasks | Prefixes | Layers |", "|---|---:|---|---|---|",
    ]
    for r in rows[:analysis.model.nunique()]:
        lines.append(f"| {r['value']} | {r['count']} | {r['tasks']} | {r['prefixes']} | {r['layers']} |")
    for col, title in [("result_quality", "Result quality"), ("campaign", "Campaign"), ("task_family", "Task family")]:
        lines += ["", f"## {title}", "", "| Value | Count |", "|---|---:|"]
        for key, n in analysis.groupby(col, dropna=False).size().sort_values(ascending=False).items():
            lines.append(f"| {key} | {n} |")
    (INV / "result_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_heatmap(ax, matrix, row_labels, col_labels, title, cmap, vmin=0, vmax=1,
                 annotations=None, cbar_label=None, center=None, add_colorbar=True):
    masked = np.ma.masked_invalid(matrix)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax) if center is not None else Normalize(vmin, vmax)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=25, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_title(title, loc="left", fontweight="bold", pad=18)
    ax.set_xticks(np.arange(-.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    if annotations is not None:
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = annotations[i][j]
                if text:
                    ax.text(j, i, text, ha="center", va="center", fontsize=14,
                            color="white" if np.isfinite(matrix[i, j]) and abs(matrix[i, j]) > .65 else "#222")
    if add_colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=.025, pad=.02)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=16)
    return im


def fig01(metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    all_rep = representative_rows(metrics, False)
    clean_rep = representative_rows(metrics, True)
    all_rep["version"] = "all"
    clean_rep["version"] = "clean"
    data = pd.concat([all_rep, clean_rep], ignore_index=True)
    data.to_csv(DATA / "fig01_capability_atlas_data.csv", index=False)
    order = model_order(metrics.model.unique().tolist())
    cmap = LinearSegmentedColormap.from_list("cap", ["#F2F0F7", "#9E9AC8", "#3F007D"])
    titles = {}
    for tag, rep in [("all", all_rep), ("clean", clean_rep)]:
        pivot = rep.pivot(index="model", columns="capability", values="relative_score").reindex(index=order, columns=CAP_ORDER)
        qp = rep.pivot(index="model", columns="capability", values="quality").reindex(index=order, columns=CAP_ORDER)
        cp = rep.pivot(index="model", columns="capability", values="prefix_coverage").reindex(index=order, columns=CAP_ORDER)
        marks = []
        for i in range(len(order)):
            row = []
            for j in range(len(CAP_ORDER)):
                if pd.isna(qp.iloc[i, j]):
                    row.append("NA")
                else:
                    mark = QUALITY_MARK.get(qp.iloc[i, j], "")
                    coverage = str(cp.iloc[i, j])
                    row.append(f"{mark}\n{coverage}" if coverage != "4/4" else mark)
            marks.append(row)
        ranges = pivot.max(axis=1) - pivot.min(axis=1)
        mean_range = float(ranges.mean(skipna=True))
        if tag == "clean" and pivot.notna().sum().sum() < 12:
            title = "Confirmed-clean coverage is too sparse for a capability atlas"
        elif mean_range >= .45:
            title = "Spatial capability profiles diverge sharply across representation families"
        else:
            title = "Spatial capability profiles vary across representation families"
        titles[tag] = title
        fig, ax = plt.subplots(figsize=(16, 9))
        draw_heatmap(ax, pivot.to_numpy(float), [MODEL_META[m][1] for m in order],
                     [display_task(task) for task in CAP_ORDER],
                     title, cmap, 0, 1, marks, "Within-task percentile (0–1)")
        ax.set_xlabel("Capability (raw metrics direction-corrected before ranking)")
        ax.set_ylabel("Model (ordered by representation family)")
        # Family separators and labels.
        fams = [MODEL_META[m][0] for m in order]
        for i in range(1, len(order)):
            if fams[i] != fams[i - 1]:
                ax.axhline(i - .5, color="#222", lw=2.2)
        footer = "Cell suffix = available/expected prefixes; † mismatch   * provisional   L legacy   NA no V1 value"
        if tag == "clean":
            footer = "Visualization-clean = canonical-source with no confirmed mismatch; lineage-unresolved rows remain."
        fig.text(.5, .015, footer, ha="center", fontsize=13, color="#555")
        fig.tight_layout(rect=(0, .035, 1, 1))
        save_figure(fig, f"fig01_capability_atlas_{tag}")
    return data, {"titles": titles, "all_rep": all_rep, "clean_rep": clean_rep}


def spearman_no_scipy(x: pd.Series, y: pd.Series) -> float:
    return float(x.rank().corr(y.rank()))


def fig02(rep: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    geom = ["A1 Depth", "A1 Mask", "A1 Camera", "A2 View"]
    dyn = ["C1 Dynamics", "C2 Integration", "C3 Counterfactual"]
    rows = []
    for model, z in rep.groupby("model"):
        g = z[z.capability.isin(geom)].relative_score
        d = z[z.capability.isin(dyn)].relative_score
        rows.append({
            "model": model, "model_display": MODEL_META[model][1], "model_family": MODEL_META[model][0],
            "geometry_score": g.mean(), "geometry_components": len(g),
            "dynamics_score": d.mean(), "dynamics_components": len(d),
            "quality": worst_quality(z.quality),
        })
    out = pd.DataFrame(rows).set_index("model").reindex(model_order(metrics_models := rep.model.unique().tolist())).reset_index()
    out["axis_status"] = np.where(out.geometry_score.isna() & out.dynamics_score.isna(), "missing both axes",
                           np.where(out.geometry_score.isna(), "missing spatial-structure axis",
                           np.where(out.dynamics_score.isna(), "missing predictive-update axis", "plotted")))
    plotted = out[out.axis_status == "plotted"].copy()
    plotted["geometry_rank"] = plotted.geometry_score.rank(ascending=False, method="min").astype(int)
    plotted["dynamics_rank"] = plotted.dynamics_score.rank(ascending=False, method="min").astype(int)
    rho = spearman_no_scipy(plotted.geometry_score, plotted.dynamics_score) if len(plotted) > 2 else float("nan")
    out = out.merge(plotted[["model", "geometry_rank", "dynamics_rank"]], on="model", how="left")
    out["spearman_rho"] = rho
    out.to_csv(DATA / "fig02_geometry_vs_dynamics_data.csv", index=False)
    if abs(rho) < .4:
        title = f"Spatial structure and predictive update are weakly coupled (Spearman ρ={rho:.2f})"
    elif rho >= 0:
        title = f"Spatial structure and predictive update covary, but remain non-identical (ρ={rho:.2f})"
    else:
        title = f"Spatial structure and predictive update trade off across models (ρ={rho:.2f})"
    fig, (ax, note_ax) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [3.8, 1.2]})
    for r in plotted.itertuples(index=False):
        ax.scatter(r.geometry_score, r.dynamics_score, s=260,
                   c=FAMILY_COLORS[r.model_family], marker=FAMILY_MARKERS[r.model_family],
                   edgecolor="#222", linewidth=1.2, zorder=3)
        label_offsets = {"qwen2_5_vl": (-100, 9), "vjepa2": (10, 9), "minimax_h3": (8, 8)}
        ax.annotate(r.model_display, (r.geometry_score, r.dynamics_score), xytext=label_offsets.get(r.model, (8, 7)),
                    textcoords="offset points", fontsize=15)
    ax.axvline(plotted.geometry_score.median(), color="#BBB", ls="--", lw=1.5)
    ax.axhline(plotted.dynamics_score.median(), color="#BBB", ls="--", lw=1.5)
    ax.set(xlim=(-.05, 1.05), ylim=(-.05, 1.05), xlabel="L1 Spatial Structure accessibility",
           ylabel="L3 Predictive Spatial Update accessibility")
    ax.set_title(title, loc="left", fontweight="bold", pad=18)
    ax.grid(color="#E8E8E8", linewidth=1)
    handles = [Line2D([0], [0], marker=FAMILY_MARKERS[f], color="w", label=f,
                      markerfacecolor=FAMILY_COLORS[f], markeredgecolor="#222", markersize=12)
               for f in FAMILY_ORDER if f in set(plotted.model_family)]
    ax.legend(handles=handles, loc="lower right", ncol=2, frameon=False)
    note_ax.axis("off")
    missing = out[out.axis_status != "plotted"]
    lines = ["Coverage", "", f"2D points: {len(plotted)}/{len(out)}", "",
             "Axes average only available", "predefined components:",
             "L1: metric geometry +", "instance identity + view consistency",
             "L3: one-step prediction +", "path integration + control", ""]
    if len(missing):
        lines += ["No coordinate (shown here):"] + [f"• {r.model_display}: {r.axis_status}" for r in missing.itertuples(index=False)]
    else:
        lines += ["All models have both axes."]
    note_ax.text(0, .98, "\n".join(lines), va="top", fontsize=13, linespacing=1.35)
    fig.tight_layout()
    save_figure(fig, "fig02_geometry_vs_dynamics")
    return out, {"rho": rho, "title": title, "plotted": plotted}


def layer_panel_data(metrics: pd.DataFrame) -> pd.DataFrame:
    d = metrics.copy()
    cap = d.groupby(["model", "model_display", "model_family", "model_variant", "campaign", "protocol",
                     "capability", "layer", "normalized_layer_depth"], as_index=False, dropna=False).agg(
        oriented_value=("oriented_value", "mean"), raw_value=("metric_value", "mean"),
        n_prefix=("prefix", "nunique"), prefixes=("prefix", lambda x: ",".join(map(str, sorted(x.dropna().astype(int).unique())))),
        quality=("result_quality", worst_quality),
    )
    cap["capability_score"] = np.nan
    for capability, idx in cap.groupby("capability").groups.items():
        ranks = cap.loc[idx, "oriented_value"].rank(method="average")
        cap.loc[idx, "capability_score"] = .5 if len(ranks) == 1 else (ranks - 1) / (len(ranks) - 1)
    cap["panel"] = cap.capability
    cap.loc[cap.capability.isin(["A1 Depth", "A1 Camera"]), "panel"] = "A1 Geometry"
    cap.loc[cap.capability == "A2 View", "panel"] = "A2 View"
    agg = cap.groupby(["model", "model_display", "model_family", "model_variant", "campaign", "protocol",
                       "panel", "layer", "normalized_layer_depth"], as_index=False, dropna=False).agg(
        normalized_performance=("capability_score", "mean"),
        component_capabilities=("capability", lambda x: ";".join(sorted(x.unique()))),
        n_prefix=("n_prefix", "max"), prefixes=("prefixes", lambda x: ";".join(sorted(set(x)))),
        quality=("quality", worst_quality),
    )
    return agg


def layer_sweep_cohort(layer: pd.DataFrame) -> pd.DataFrame:
    """Choose a campaign per model/task; retain tasks only present at default layer."""
    pieces = []
    for (model, panel), z in layer.groupby(["model", "panel"]):
        identity = ["campaign", "protocol", "model_variant"]
        coverage = z.groupby(identity).layer.nunique().sort_values(ascending=False)
        if coverage.empty:
            continue
        max_layers = coverage.iloc[0]
        candidates = coverage[coverage == max_layers].index.tolist()
        chosen = next((x for x in candidates if x[0] != "default8"), candidates[0])
        q = z[(z.campaign == chosen[0]) & (z.protocol == chosen[1]) &
              (z.model_variant == chosen[2])].copy()
        q["sweep_campaign"] = chosen[0]
        pieces.append(q)
    return pd.concat(pieces, ignore_index=True) if pieces else layer.iloc[0:0].copy()


def fig03(metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    layer = layer_sweep_cohort(layer_panel_data(metrics))
    sweep_models = [m for m, z in layer.groupby("model") if z.layer.nunique() >= 3]
    all_models = model_order(layer.model.unique().tolist())
    layer.to_csv(DATA / "fig03_layer_curves_data.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    styles = {
        model: (['-', '--', '-.', ':'][i % 4], ['o', 's', 'D', '^', 'v', 'P'][i % 6])
        for i, model in enumerate(all_models)
    }
    for ax, panel in zip(axes.flat, PANEL_ORDER):
        z = layer[layer.panel == panel]
        for model, q in z.groupby("model"):
            q = q.sort_values("normalized_layer_depth")
            ls, marker = styles[model]
            real = q[q.normalized_layer_depth.notna()]
            special = q[q.normalized_layer_depth.isna()]
            if real.layer.nunique() >= 3:
                ax.plot(real.normalized_layer_depth, real.normalized_performance, marker=marker, ls=ls, lw=2.3,
                        label=MODEL_META[model][1], color=FAMILY_COLORS[MODEL_META[model][0]], alpha=.88)
            elif len(real):
                ax.scatter(real.normalized_layer_depth, real.normalized_performance, marker=marker, s=80,
                           label=MODEL_META[model][1], color=FAMILY_COLORS[MODEL_META[model][0]], alpha=.9)
            if len(special):
                ax.scatter(np.full(len(special), 1.08), special.normalized_performance, marker=marker, s=95,
                           label=MODEL_META[model][1], color=FAMILY_COLORS[MODEL_META[model][0]],
                           edgecolor="#222", linewidth=.8, alpha=.9)
        title = display_task(panel)
        if panel == "B2 Persistence":
            title = "Out-of-view object\nlocalization"
        ax.set_title(title, fontweight="bold", fontsize=17)
        ax.grid(color="#ECECEC")
        ax.axvspan(1.035, 1.125, color="#F1F1F1", zorder=0)
        ax.set_xlim(-.03, 1.13); ax.set_ylim(-.03, 1.03)
    fig.supxlabel("True normalized backbone depth (right gray band = isolated output/merger)", y=.085, fontsize=18)
    fig.supylabel("Task-wise normalized performance", x=.012, fontsize=18)
    handles, labels = [], []
    for ax in axes.flat:
        h, lab = ax.get_legend_handles_labels(); handles.extend(h); labels.extend(lab)
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=min(5, len(by_label)), frameon=False)
    fig.suptitle("Task-dependent layer profiles (single-layer measurements shown as points)", fontweight="bold", x=.04, ha="left")
    fig.tight_layout(rect=(.025, .10, 1, .94))
    save_figure(fig, "fig03a_layer_emergence")

    panel_dir = FIG / "panels"
    for panel in PANEL_ORDER:
        fig, ax = plt.subplots(figsize=(16, 9))
        z = layer[layer.panel == panel]
        for model, q in z.groupby("model"):
            q = q.sort_values("normalized_layer_depth"); ls, marker = styles[model]
            real = q[q.normalized_layer_depth.notna()]; special = q[q.normalized_layer_depth.isna()]
            if real.layer.nunique() >= 3:
                ax.plot(real.normalized_layer_depth, real.normalized_performance, marker=marker, ls=ls, lw=3,
                        label=MODEL_META[model][1], color=FAMILY_COLORS[MODEL_META[model][0]])
            elif len(real):
                ax.scatter(real.normalized_layer_depth, real.normalized_performance, marker=marker, s=120,
                           label=MODEL_META[model][1], color=FAMILY_COLORS[MODEL_META[model][0]])
            if len(special):
                ax.scatter(np.full(len(special), 1.08), special.normalized_performance, marker=marker, s=130,
                           label=MODEL_META[model][1], color=FAMILY_COLORS[MODEL_META[model][0]], edgecolor="#222")
        ax.axvspan(1.035, 1.125, color="#F1F1F1"); ax.set_xlim(-.03, 1.13); ax.set_ylim(-.03, 1.03)
        ax.set(xlabel="True normalized backbone depth (gray band = isolated output/merger)",
               ylabel="Task-wise normalized performance")
        ax.set_title(f"{display_task(panel)}: measured-layer profile", loc="left", fontweight="bold")
        handles, labels = ax.get_legend_handles_labels(); unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), frameon=False, ncol=3); ax.grid(color="#ECECEC")
        fig.tight_layout(); save_figure(fig, f"fig03a_panel_{panel.lower().replace(' ', '_')}", panel_dir)

    peaks = []
    for (model, panel), z in layer.groupby(["model", "panel"]):
        best = z.loc[z.normalized_performance.idxmax()]
        depth = float(best.normalized_layer_depth)
        stage = "output/merger" if not np.isfinite(depth) else "early" if depth <= .33 else "middle" if depth <= .66 else "late"
        peaks.append({
            "model": model, "model_display": MODEL_META[model][1], "task": panel,
            "campaign": best.campaign,
            "peak_layer": int(best.layer), "peak_normalized_depth": depth, "peak_stage": stage,
            "peak_score": best.normalized_performance, "quality": best.quality,
            "n_layers": z.layer.nunique(), "trend_status": "layer_sweep" if z.layer.nunique() >= 3 else "single_measured_point",
        })
    peak = pd.DataFrame(peaks)
    task_stats = peak.groupby("task").agg(
        n_model_task=("model", "size"), median_peak_depth=("peak_normalized_depth", "median"),
        mean_peak_depth=("peak_normalized_depth", "mean"), min_peak_depth=("peak_normalized_depth", "min"),
        max_peak_depth=("peak_normalized_depth", "max"),
    ).reset_index()
    stage = peak.groupby(["task", "peak_stage"]).size().unstack(fill_value=0)
    stage = stage.div(stage.sum(axis=1), axis=0).add_prefix("fraction_").reset_index()
    task_stats = task_stats.merge(stage, on="task", how="left")
    peak["record_type"] = "model_task_peak"
    for c in task_stats.columns:
        if c not in peak: peak[c] = np.nan
    task_stats["record_type"] = "task_distribution"
    stats = pd.concat([peak, task_stats], ignore_index=True, sort=False)
    stats.to_csv(DATA / "fig03_layer_statistics.csv", index=False)

    order = [m for m in all_models if m in set(peak.model)]
    pp = peak.pivot(index="task", columns="model", values="peak_normalized_depth").reindex(index=PANEL_ORDER, columns=order)
    fig, ax = plt.subplots(figsize=(16, 9))
    draw_heatmap(ax, pp.to_numpy(float), PANEL_ORDER, [MODEL_META[m][1] for m in order],
                 "Best observed measured layer (not training-time emergence)",
                 LinearSegmentedColormap.from_list("depth", ["#4575B4", "#FFFFBF", "#D73027"]),
                 0, 1, None, "Peak normalized layer depth")
    for i in range(len(PANEL_ORDER)):
        for j in range(len(order)):
            v = pp.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=13)
            else:
                panel_peak = peak[(peak.task == PANEL_ORDER[i]) & (peak.model == order[j])]
                if len(panel_peak) and panel_peak.iloc[0].peak_stage == "output/merger":
                    ax.text(j, i, "OUT", ha="center", va="center", fontsize=11, color="#555")
                elif not len(panel_peak):
                    ax.text(j, i, "NA", ha="center", va="center", fontsize=11, color="#555")
    ax.set_xlabel("Model"); ax.set_ylabel("Capability")
    fig.tight_layout()
    save_figure(fig, "fig03b_peak_layer_map")
    counts = peak.peak_stage.value_counts()
    return stats, {"peak": peak, "task_stats": task_stats, "stage_counts": counts}


def bootstrap_median_ci(values: np.ndarray, reps: int = 10000, seed: int = 1096) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0: return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(values, size=(reps, len(values)), replace=True), axis=1)
    return tuple(np.quantile(meds, [.025, .975]))


def fig04(metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = metrics.copy()
    rows, curve_rows = [], []
    identity_cols = ["campaign", "protocol", "model_variant"]
    for model in model_order(d.model.unique().tolist()):
        rep_layer = REP_LAYER[model]
        z = d[(d.model == model) & (d.layer == rep_layer)]
        for cap in CAP_ORDER:
            q = z[z.capability == cap].copy()
            base = {
                "model": model, "model_display": MODEL_META[model][1], "model_family": MODEL_META[model][0],
                "task": cap, "representative_layer": rep_layer,
            }
            if q.empty:
                rows.append({**base, "pair_status": "NA: no V1 numeric value at representative layer"})
                continue
            ids = q.groupby(identity_cols, dropna=False).size().sort_values(ascending=False)
            chosen = ids.index[0]
            if (q.campaign == "default8").any():
                chosen = q[q.campaign == "default8"].groupby(identity_cols, dropna=False).size().sort_values(ascending=False).index[0]
            q = q[(q.campaign == chosen[0]) & (q.protocol == chosen[1]) & (q.model_variant == chosen[2])]
            vals = q.groupby("prefix", as_index=False).agg(
                raw=("metric_value", "mean"), oriented=("oriented_value", "mean"),
                quality=("result_quality", worst_quality), n_config=("source_universe_id", "nunique"),
            )
            prefixes = sorted(vals.prefix.astype(int).unique())
            scale = d[d.capability == cap].oriented_value.std(ddof=1)
            if not np.isfinite(scale) or scale == 0: scale = 1.0
            first = vals.sort_values("prefix").iloc[0]
            for rr in vals.sort_values("prefix").itertuples(index=False):
                curve_rows.append({**base, "metric_name": q.metric_name.iloc[0],
                    "metric_direction": q.metric_direction.iloc[0], "model_variant": chosen[2],
                    "campaign": chosen[0], "protocol": chosen[1], "prefix": int(rr.prefix),
                    "raw_value": rr.raw, "oriented_value": rr.oriented,
                    "baseline_prefix": int(first.prefix),
                    "relative_to_first_available_sd": (rr.oriented - first.oriented) / scale,
                    "quality": rr.quality, "n_config": rr.n_config})
            missing_endpoints = [str(x) for x in [8, 24] if x not in prefixes]
            record = {**base, "metric_name": q.metric_name.iloc[0], "metric_direction": q.metric_direction.iloc[0],
                "model_variant": chosen[2], "campaign": chosen[0], "protocol": chosen[1],
                "available_prefixes": ",".join(map(str, prefixes)), "n_available_prefix": len(prefixes),
                "quality": worst_quality(vals.quality)}
            if missing_endpoints:
                rows.append({**record, "pair_status": "NA: missing endpoint p" + "/p".join(missing_endpoints)})
                continue
            p8 = vals.loc[vals.prefix == 8].iloc[0]; p24 = vals.loc[vals.prefix == 24].iloc[0]
            rows.append({**record, "pair_status": "matched_same_identity", "p8": p8.raw, "p24": p24.raw,
                "raw_delta_p24_minus_p8": float(p24.raw - p8.raw),
                "oriented_delta": float(p24.oriented - p8.oriented)})
    out = pd.DataFrame(rows)
    out["standardized_delta"] = np.nan
    out["direction"] = pd.Series([None] * len(out), dtype="object")
    matched_mask = out.pair_status.eq("matched_same_identity")
    for task, idx in out[matched_mask].groupby("task").groups.items():
        scale = metrics[metrics.capability == task].oriented_value.std(ddof=1)
        if not np.isfinite(scale) or scale == 0: scale = 1.0
        out.loc[idx, "standardized_delta"] = out.loc[idx, "oriented_delta"] / scale
    out.loc[matched_mask, "direction"] = np.where(out.loc[matched_mask, "standardized_delta"] > 1e-12, "p24 > p8",
        np.where(out.loc[matched_mask, "standardized_delta"] < -1e-12, "p24 < p8", "tie"))
    matched = out[matched_mask].copy()
    task_frac = matched.groupby("task").direction.value_counts(normalize=True).unstack(fill_value=0).reset_index()
    task_frac["record_type"] = "task_fraction"
    family_frac = matched.groupby("model_family").direction.value_counts(normalize=True).unstack(fill_value=0).reset_index()
    family_frac["record_type"] = "family_fraction"
    ci = bootstrap_median_ci(matched.standardized_delta.to_numpy())
    out["record_type"] = "model_task_endpoint_status"
    out["bootstrap_median_ci_low"] = ci[0]; out["bootstrap_median_ci_high"] = ci[1]
    stats = pd.concat([out, task_frac, family_frac], ignore_index=True, sort=False)
    stats.to_csv(DATA / "fig04_prefix_statistics.csv", index=False)
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(DATA / "fig04_prefix_curves_data.csv", index=False)

    order = model_order(d.model.unique().tolist())
    pivot = out.pivot(index="model", columns="task", values="standardized_delta").reindex(index=order, columns=CAP_ORDER)
    finite_delta = np.abs(pivot.to_numpy(float)); finite_delta = finite_delta[np.isfinite(finite_delta)]
    lim = max(.1, float(np.quantile(finite_delta, .95))) if len(finite_delta) else 1.0
    annotations = [["NA" if not np.isfinite(pivot.iloc[i, j]) else "" for j in range(len(CAP_ORDER))] for i in range(len(order))]
    fig, ax = plt.subplots(figsize=(16, 9))
    draw_heatmap(ax, pivot.to_numpy(float), [MODEL_META[m][1] for m in order],
                 [display_task(task) for task in CAP_ORDER],
                 "Matched p24−p8 effects; missing endpoints remain explicit",
                 LinearSegmentedColormap.from_list("delta", ["#B2182B", "#F7F7F7", "#2166AC"]),
                 -lim, lim, annotations, "Standardized Δ (positive = longer history helps)", center=0)
    ax.set_xlabel("Capability"); ax.set_ylabel("Model")
    fig.tight_layout()
    save_figure(fig, "fig04a_prefix_delta_heatmap")

    palette = dict(zip(CAP_ORDER, plt.cm.tab10(np.linspace(0, .75, len(CAP_ORDER)))))
    for page, start in enumerate(range(0, len(order), 5), 1):
        page_models = order[start:start + 5]
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
        for ax, model in zip(axes.flat, page_models):
            z = curves[curves.model == model]
            for task, q in z.groupby("task"):
                q = q.sort_values("prefix")
                ax.plot(q.prefix, q.relative_to_first_available_sd, marker="o", lw=2.1,
                        color=palette[task], label=display_task(task))
            ax.axhline(0, color="#AAA", lw=1); ax.set_xticks([8, 12, 16, 24]); ax.grid(color="#ECECEC")
            ax.set_title(MODEL_META[model][1], fontsize=17, fontweight="bold")
        for ax in axes.flat[len(page_models):]: ax.axis("off")
        handles = [Line2D([0], [0], color=palette[t], marker="o", label=display_task(t)) for t in CAP_ORDER]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=11)
        fig.supxlabel("Prefix length"); fig.supylabel("Direction-corrected change from first available prefix (task SD)")
        fig.suptitle("Prefix curves for every model with V1 values", fontweight="bold", x=.04, ha="left")
        fig.tight_layout(rect=(.02, .09, 1, .94))
        save_figure(fig, f"fig04b_prefix_curves_all_page{page:02d}")
    panel_dir = FIG / "panels"
    for model in order:
        fig, ax = plt.subplots(figsize=(16, 9))
        z = curves[curves.model == model]
        for task, q in z.groupby("task"):
            q = q.sort_values("prefix")
            ax.plot(q.prefix, q.relative_to_first_available_sd, marker="o", lw=3,
                    color=palette[task], label=display_task(task))
        ax.axhline(0, color="#AAA", lw=1); ax.set_xticks([8, 12, 16, 24]); ax.grid(color="#ECECEC")
        ax.set(xlabel="Prefix length", ylabel="Direction-corrected change from first available prefix (task SD)")
        ax.set_title(f"{MODEL_META[model][1]} — available V1 prefix curves", loc="left", fontweight="bold")
        ax.legend(frameon=False, ncol=3)
        fig.tight_layout()
        save_figure(fig, f"fig04b_panel_{model}", panel_dir)
    return stats, {"pairs": matched, "endpoint_status": out, "ci": ci}


def matrix_case_stats(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # A tensor belongs to one campaign; never average duplicate default/sweep
    # identities into the same layer×prefix cell.
    candidates = []
    for (model, task, campaign), z in metrics.groupby(["model", "capability", "campaign"]):
        if z.layer.nunique() < 2 or z.prefix.nunique() < 2: continue
        cell = z.groupby(["layer", "normalized_layer_depth", "prefix"], as_index=False).agg(score=("oriented_value", "mean"), quality=("result_quality", worst_quality))
        pivot = cell.pivot(index="layer", columns="prefix", values="score")
        if pivot.notna().sum().sum() < 8: continue
        row_means = pivot.mean(axis=1); col_means = pivot.mean(axis=0); grand = np.nanmean(pivot.to_numpy(float))
        additive = row_means.to_numpy()[:, None] + col_means.to_numpy()[None, :] - grand
        residual = pivot.to_numpy(float) - additive
        best_by_prefix = pivot.idxmax(axis=0)
        overall_layer = int(row_means.idxmax())
        depth = norm_depth(model, overall_layer)[0]
        rows.append({
            "model": model, "model_display": MODEL_META[model][1], "model_family": MODEL_META[model][0],
            "task": task, "campaign": campaign, "n_layers": pivot.shape[0], "n_prefixes": pivot.shape[1],
            "overall_best_layer": overall_layer, "peak_depth": depth,
            "best_layer_changes": int(best_by_prefix.nunique()),
            "layer_sensitivity": float(row_means.max() - row_means.min()),
            "prefix_sensitivity": float(col_means.max() - col_means.min()),
            "interaction_rms": float(np.sqrt(np.nanmean(residual ** 2))),
            "p24_minus_p8": float(col_means.get(24, np.nan) - col_means.get(8, np.nan)),
            "quality": worst_quality(cell.quality),
        })
    all_cases = pd.DataFrame(rows)
    if all_cases.empty:
        return all_cases
    # If more than one multi-layer campaign ever exists, retain the broadest one.
    all_cases = all_cases.sort_values(["model", "task", "n_layers", "n_prefixes"], ascending=[True, True, False, False])
    return all_cases.drop_duplicates(["model", "task"], keep="first").reset_index(drop=True)


def fig05(metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cases = matrix_case_stats(metrics)
    # Complete structural export: every model × headline capability with any V1
    # value, including one-layer and partial-prefix tensors.
    complete, heatmap_rows = [], []
    identity_cols = ["campaign", "protocol", "model_variant"]
    for model in model_order(metrics.model.unique().tolist()):
        for task in CAP_ORDER:
            z = metrics[(metrics.model == model) & (metrics.capability == task)].copy()
            if z.empty:
                continue
            ids = z.groupby(identity_cols, dropna=False).agg(n_cells=("metric_value", "count"), n_layers=("layer", "nunique"), n_prefixes=("prefix", "nunique")).sort_values(["n_cells", "n_layers", "n_prefixes"], ascending=False)
            chosen = ids.index[0]
            q = z[(z.campaign == chosen[0]) & (z.protocol == chosen[1]) & (z.model_variant == chosen[2])]
            cell = q.groupby(["layer", "normalized_layer_depth", "prefix"], as_index=False, dropna=False).agg(
                oriented_value=("oriented_value", "mean"), raw_value=("metric_value", "mean"),
                quality=("result_quality", worst_quality), n_config=("source_universe_id", "nunique"),
            )
            key = f"{model}__{task.lower().replace(' ', '_')}"
            for rr in cell.itertuples(index=False):
                heatmap_rows.append({"heatmap_key": key, "model": model, "model_display": MODEL_META[model][1],
                    "task": task, "model_variant": chosen[2], "campaign": chosen[0], "protocol": chosen[1],
                    "layer": rr.layer, "normalized_layer_depth": rr.normalized_layer_depth, "prefix": rr.prefix,
                    "raw_value": rr.raw_value, "oriented_value": rr.oriented_value, "quality": rr.quality,
                    "n_config": rr.n_config})
            complete.append({"key": key, "model": model, "task": task, "cell": cell,
                             "identity": chosen, "quality": worst_quality(cell.quality)})
    pd.DataFrame(heatmap_rows).to_csv(DATA / "fig05_all_heatmaps_data.csv", index=False)

    def draw_complete_case(ax, item, show_colorbar=False):
        cell = item["cell"]
        p = cell.pivot(index="layer", columns="prefix", values="oriented_value").sort_index().reindex(columns=[8, 12, 16, 24])
        arr = p.to_numpy(float); finite = arr[np.isfinite(arr)]
        scaled = (arr - finite.min()) / (finite.max() - finite.min()) if len(finite) and finite.max() > finite.min() else np.where(np.isfinite(arr), .5, np.nan)
        labels = []
        for x in p.index:
            nd = norm_depth(item["model"], int(x))[0]
            labels.append(f"L{x}" if not np.isfinite(nd) else f"L{x} ({nd:.2f})")
        draw_heatmap(ax, scaled, labels, [f"p{x}" for x in p.columns],
                     f"{MODEL_META[item['model']][1]}\n× {display_task(item['task'])}",
                     LinearSegmentedColormap.from_list("case", ["#FFF7EC", "#FC8D59", "#7F0000"]),
                     0, 1, None, "within-panel scale" if show_colorbar else None, add_colorbar=show_colorbar)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if not np.isfinite(arr[i, j]): ax.text(j, i, "NA", ha="center", va="center", fontsize=10, color="#666")
        ax.set_xlabel("Prefix"); ax.set_ylabel("Measured layer")

    full_dir = FIG / "fig05_all_model_task"
    for item in complete:
        fig, ax = plt.subplots(figsize=(16, 9)); draw_complete_case(ax, item, True)
        fig.text(.5, .02, f"Identity: {item['identity'][0]} | {item['identity'][1]} | {item['identity'][2]}; colors are panel-local", ha="center", fontsize=11)
        fig.tight_layout(rect=(0, .04, 1, 1)); save_figure(fig, f"fig05_{item['key']}", full_dir)
    for page, start in enumerate(range(0, len(complete), 6), 1):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for ax, item in zip(axes.flat, complete[start:start + 6]): draw_complete_case(ax, item, False)
        for ax in axes.flat[len(complete[start:start + 6]):]: ax.axis("off")
        fig.suptitle("Complete V1 layer × prefix tensors (panel-local colors; NA retained)", fontweight="bold", x=.04, ha="left")
        fig.tight_layout(rect=(0, 0, 1, .94)); save_figure(fig, f"fig05b_all_heatmaps_page{page:02d}")

    selected = []
    middle = cases.assign(dist=(cases.peak_depth - .5).abs()).sort_values(["dist", "layer_sensitivity"], ascending=[True, False])
    extreme = cases.assign(extreme=(cases.peak_depth - .5).abs()).sort_values(["extreme", "layer_sensitivity"], ascending=False)
    negative = cases.sort_values("p24_minus_p8")
    interact = cases.sort_values(["interaction_rms", "best_layer_changes"], ascending=False)
    for label, frame in [("middle-layer peak", middle), ("early/late peak", extreme), ("longer prefix hurts", negative), ("strong interaction", interact)]:
        for ix in frame.index:
            if ix not in [x[1] for x in selected]:
                selected.append((label, ix)); break
    cases["selected_role"] = ""
    if globals().get("FIG05_CASES"):
        selected = []
        for model, task, role in FIG05_CASES:
            matches = cases[(cases.model == model) & (cases.task == task)]
            if len(matches) != 1:
                raise ValueError(f"Expected one representative case: {model}/{task}")
            selected.append((role, matches.index[0]))
    for label, ix in selected: cases.loc[ix, "selected_role"] = label
    cases.to_csv(DATA / "fig05_case_selection.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    cmap = LinearSegmentedColormap.from_list("case", ["#FFF7EC", "#FC8D59", "#7F0000"])
    for ax, (role, ix) in zip(axes.flat, selected):
        r = cases.loc[ix]
        z = metrics[(metrics.model == r.model) & (metrics.capability == r.task) & (metrics.campaign == r.campaign)]
        cell = z.groupby(["layer", "prefix"], as_index=False).oriented_value.mean()
        p = cell.pivot(index="layer", columns="prefix", values="oriented_value").sort_index().reindex(columns=[8, 12, 16, 24])
        arr = p.to_numpy(float)
        finite = arr[np.isfinite(arr)]
        scaled = (arr - finite.min()) / (finite.max() - finite.min()) if finite.max() > finite.min() else np.full_like(arr, .5)
        draw_heatmap(ax, scaled, [f"L{x} ({norm_depth(r.model, int(x))[0]:.2f})" if np.isfinite(norm_depth(r.model, int(x))[0]) else f"L{x}" for x in p.index],
                     [f"p{x}" for x in p.columns], f"{role}\n{r.model_display}", cmap, 0, 1, None, None)
        best = np.unravel_index(np.nanargmax(arr), arr.shape)
        for row_index, col_index in zip(*np.where(~np.isfinite(arr))):
            ax.text(col_index, row_index, "NA", ha="center", va="center", fontsize=12, color="#4E4842")
        ax.text(best[1], best[0], "BEST", ha="center", va="center", color="white", fontweight="bold", fontsize=13)
        ax.set_xlabel("Prefix")
        if ax in axes[:, 0]:
            ax.set_ylabel("Layer (normalized depth)")
        else:
            ax.set_ylabel("")
    fig.suptitle("Layer × prefix profiles: instance identity and out-of-view object localization", fontweight="bold", x=.04, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .94))
    save_figure(fig, "fig05_layer_prefix_cases")
    return cases, {"selected": selected, "complete": complete}


def fig06() -> tuple[pd.DataFrame, dict]:
    c1 = pd.read_csv(C1_VALIDITY)
    z = c1[c1.analysis_type.eq("paper_retrained_ablation")].copy()
    rows = []
    for model, q in z.groupby("model"):
        full = q[q.action_condition.eq("FULL")].iloc[0]
        controls = q[~q.action_condition.eq("FULL")]
        strongest = controls.loc[controls.r1.idxmax()]
        rows += [
            {"panel": "retrained_ablation", "model": model, "condition": "FULL", "metric": "global_R@1", "value": full.r1,
             "delta_action": full.r1 - strongest.r1, "control_type": "full", "source": "analyses/probe_validity/c1_validity.csv"},
            {"panel": "retrained_ablation", "model": model, "condition": strongest.action_condition, "metric": "global_R@1", "value": strongest.r1,
             "delta_action": full.r1 - strongest.r1, "control_type": "retrained non-interaction baseline", "source": "analyses/probe_validity/c1_validity.csv"},
        ]
    wan = pd.read_csv(WAN_VALIDITY)
    for r in wan.itertuples(index=False):
        rows.append({"panel": "wan_action_validity", "model": "wan", "condition": r.condition,
                     "metric": "global_R@1", "value": r.r1, "delta_action": np.nan,
                     "control_type": "full" if r.condition == "correct_action" else
                         "wrong/shuffled-action control" if "shuffled" in r.condition else "non-action/copy baseline",
                     "source": "results/analysis/wan_p12_l20_action_validity/tables/condition_metrics.csv"})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "fig06_construct_validity_data.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    models = [m for m in ["vjepa2", "videomae", "wan", "cogvideox"] if m in set(z.model)]
    x = np.arange(len(models)); width = .34
    fulls, controls, labels, deltas = [], [], [], []
    for model in models:
        q = out[(out.panel == "retrained_ablation") & (out.model == model)]
        f = q[q.condition == "FULL"].iloc[0]; c = q[q.condition != "FULL"].iloc[0]
        fulls.append(f.value); controls.append(c.value); labels.append(c.condition.replace("_TRAINED", "").replace("_", " "))
        deltas.append(f.delta_action)
    ax.bar(x - width/2, fulls, width, label="Full", color="#2166AC")
    ax.bar(x + width/2, controls, width, label="Strongest retrained control", color="#BDBDBD")
    for i, d in enumerate(deltas):
        ax.text(i, max(fulls[i], controls[i]) + .006, f"Δ={d:+.3f}", ha="center", fontsize=14, fontweight="bold")
    ax.set_xticks(x, [MODEL_META[m][1] for m in models]); ax.set_ylabel("Global R@1")
    ax.set_title("Full vs strongest non-interaction control", fontweight="bold", fontsize=18)
    ax.legend(frameon=False); ax.grid(axis="y", color="#ECECEC")
    ax = axes[1]
    order = ["correct_action", "shuffled_action", "eval_null_action", "copy_last_streaming_state", "copy_prefix_pooled_state"]
    wz = out[(out.panel == "wan_action_validity") & out.condition.isin(order)].set_index("condition").reindex(order).dropna(subset=["value"])
    colors = ["#2166AC"] + ["#BDBDBD"] * (len(wz) - 1)
    ax.barh(np.arange(len(wz)), wz.value, color=colors)
    ax.set_yticks(np.arange(len(wz)), [x.replace("eval_", "").replace("_", " ") for x in wz.index])
    ax.invert_yaxis(); ax.set_xlabel("Global R@1")
    ax.set_title("Wan P12/L20 action controls", fontweight="bold", fontsize=18)
    ax.grid(axis="x", color="#ECECEC")
    fig.suptitle("High probing scores can survive when the intended action signal is weak", fontweight="bold", x=.04, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .94))
    save_figure(fig, "fig06_construct_validity")

    panel_dir = FIG / "panels"
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.bar(x - width/2, fulls, width, label="Full", color="#2166AC")
    ax.bar(x + width/2, controls, width, label="Strongest retrained non-interaction baseline", color="#BDBDBD")
    for i, dlt in enumerate(deltas): ax.text(i, max(fulls[i], controls[i]) + .006, f"Δ={dlt:+.3f}", ha="center", fontsize=16, fontweight="bold")
    ax.set_xticks(x, [MODEL_META[m][1] for m in models]); ax.set_ylabel("Global R@1")
    ax.set_title("Full vs strongest retrained non-interaction baseline", loc="left", fontweight="bold")
    ax.legend(frameon=False); ax.grid(axis="y", color="#ECECEC"); fig.tight_layout()
    save_figure(fig, "fig06_panel_retrained_baselines", panel_dir)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.barh(np.arange(len(wz)), wz.value, color=colors)
    ax.set_yticks(np.arange(len(wz)), [x.replace("eval_", "").replace("_", " ") for x in wz.index])
    ax.invert_yaxis(); ax.set_xlabel("Global R@1"); ax.grid(axis="x", color="#ECECEC")
    ax.set_title("Wan P12/L20: action perturbations and non-action/copy baselines", loc="left", fontweight="bold")
    fig.tight_layout(); save_figure(fig, "fig06_panel_wan_controls", panel_dir)
    return out, {"models": models, "deltas": deltas}


def render_table(df: pd.DataFrame, stem: str, title: str, font_size: int = 14,
                 best_second: bool = False, numeric_cols: list[str] | None = None,
                 directions: dict[str, str] | None = None, footer: str = "",
                 col_widths: list[float] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(16, 9)); ax.axis("off")
    ax.set_title(title, loc="left", fontweight="bold", pad=20)
    display = df.copy()
    table = ax.table(cellText=display.values, colLabels=display.columns, cellLoc="center", colLoc="center",
                     loc="center", bbox=[0, .08, 1, .82], colWidths=col_widths)
    table.auto_set_font_size(False); table.set_fontsize(font_size)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor("white"); cell.set_linewidth(1.5)
        if i == 0:
            cell.set_facecolor("#283747"); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#F4F6F7" if i % 2 else "#EAECEE")
            if j == 0: cell.get_text().set_weight("bold")
    if best_second and numeric_cols and directions:
        for col in numeric_cols:
            if col not in df: continue
            j = df.columns.get_loc(col)
            vals = pd.to_numeric(df[col].astype(str).str.replace(r"[†*L]", "", regex=True), errors="coerce")
            vals = vals.dropna()
            if vals.empty: continue
            asc = directions[col] == "min"
            ranked = vals.sort_values(ascending=asc).index.tolist()
            if ranked:
                table[(ranked[0] + 1, j)].get_text().set_weight("bold")
                table[(ranked[0] + 1, j)].set_facecolor("#D5F5E3")
            if len(ranked) > 1:
                table[(ranked[1] + 1, j)].get_text().set_style("italic")
                table[(ranked[1] + 1, j)].set_facecolor("#FCF3CF")
    if footer: fig.text(.5, .035, footer, ha="center", fontsize=12, color="#555")
    fig.tight_layout()
    fig.savefig(TAB / f"{stem}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(TAB / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def table01() -> pd.DataFrame:
    cols = ["Level", "Probe", "Capability", "Input", "Prediction target", "Headline metric", "Interpretation"]
    df = pd.DataFrame(TASK_DEFINITIONS, columns=cols)
    df.to_csv(TAB / "table01_task_definitions.csv", index=False)
    wrapped = df.copy()
    for c in cols[1:]: wrapped[c] = wrapped[c].str.replace(" ", "\n", n=1)
    render_table(wrapped, "table01_task_definitions", "Six tasks and seven metrics, plus a counterfactual control", font_size=12,
                 footer="Definitions follow metadata/probe_taxonomy.csv and EXPERIMENT_PROTOCOL.md.")
    return df


def table02(rep_all: pd.DataFrame, rep_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rename = {"A1 Depth": "Metric depth ↓", "A1 Mask": "Instance identity ↓",
              "A1 Camera": "Camera pose ↑", "A2 View": "View consistency ↑",
              "B2 Persistence": "Out-of-view localization ↓", "C1 Dynamics": "One-step prediction ↑",
              "C2 Integration": "Path integration ↑", "C3 Counterfactual": "Counterfactual control ↑"}
    dirs = {"Metric depth ↓": "min", "Instance identity ↓": "min",
            "Camera pose ↑": "max", "View consistency ↑": "max",
            "Out-of-view localization ↓": "min", "One-step prediction ↑": "max",
            "Path integration ↑": "max", "Counterfactual control ↑": "max"}
    outputs = []
    for tag, rep in [("all", rep_all), ("clean", rep_clean)]:
        val = rep.pivot(index="model", columns="capability", values="raw_mean").reindex(index=model_order(rep.model.unique().tolist()), columns=CAP_ORDER)
        qual = rep.pivot(index="model", columns="capability", values="quality").reindex(index=val.index, columns=CAP_ORDER)
        coverage = rep.pivot(index="model", columns="capability", values="prefix_coverage").reindex(index=val.index, columns=CAP_ORDER)
        rows = []
        for model in val.index:
            row = {"Model": MODEL_META[model][1]}
            for cap in CAP_ORDER:
                v = val.loc[model, cap]
                if pd.isna(v): row[rename[cap]] = "—"
                else:
                    metric = CAPABILITY_META[cap][2]
                    fmt = f"{v:.4f}" if metric not in {"mean_ang_err_deg"} else f"{v:.2f}"
                    partial = "‡" if str(coverage.loc[model, cap]) != "4/4" else ""
                    row[rename[cap]] = fmt + QUALITY_MARK.get(qual.loc[model, cap], "") + partial
            cov_values = {cap: (str(coverage.loc[model, cap]) if pd.notna(coverage.loc[model, cap]) else "NA") for cap in CAP_ORDER}
            short = {"A1 Depth":"Depth", "A1 Mask":"Identity", "A1 Camera":"Camera", "A2 View":"View",
                     "B2 Persistence":"Out-of-view", "C1 Dynamics":"One-step",
                     "C2 Integration":"Path", "C3 Counterfactual":"Control"}
            if all(v == "4/4" for v in cov_values.values()):
                row["Prefix coverage"] = "All 4/4"
            else:
                groups = {}
                for cap, value in cov_values.items(): groups.setdefault(value, []).append(short[cap])
                row["Prefix coverage"] = "\n".join(f"{','.join(caps)}: {value}" for value, caps in groups.items())
            rows.append(row)
        df = pd.DataFrame(rows, columns=["Model"] + list(rename.values()) + ["Prefix coverage"])
        df.to_csv(TAB / f"table02_main_results_{tag}.csv", index=False)
        rep.to_csv(TAB / f"table02_main_results_{tag}_long.csv", index=False)
        footer = "Representative layer; available-prefix mean only. ‡ partial prefix coverage; † mismatch; * provisional; L legacy. Exact coverage is in final column/long CSV."
        if tag == "clean": footer = "Visualization-clean only (canonical-source, no confirmed mismatch); this is not strict exact-safe certification."
        render_table(df, f"table02_main_results_{tag}",
                     "Raw metrics reveal capability-specific leaders—there is no single winner" if tag == "all" else "Canonical non-mismatch coverage remains incomplete",
                     font_size=10, best_second=False, numeric_cols=list(rename.values()), directions=dirs, footer=footer,
                     col_widths=[.11] + [.08] * len(rename) + [.17])
        outputs.append(df)
    return outputs[0], outputs[1]


def make_takeaways(analysis, f1, f2, f3, f4, f5, f6, t2all) -> None:
    rep = f1["all_rep"]
    leaders = {}
    for cap, z in rep.groupby("capability"):
        if not z.empty:
            r = z.loc[z.relative_score.idxmax()]
            leaders[cap] = r.model_display
    peak = f3["peak"]
    stage = peak.peak_stage.value_counts().to_dict()
    pairs = f4["pairs"]
    pos = int((pairs.direction == "p24 > p8").sum()); neg = int((pairs.direction == "p24 < p8").sum()); total = len(pairs)
    median_delta = float(pairs.standardized_delta.median())
    cases = f5["selected"]
    deltas = f6["deltas"]
    numeric = int(analysis.metric_value.notna().sum())
    sections = [
        ("Fig. 1 — Spatial Capability Atlas", "空间能力呈专门化分布，不存在跨任务统一赢家。",
         [f"{len(leaders)}/{len(CAP_ORDER)} 个能力有可比较数值；领先模型来自 {len(set(leaders.values()))} 个不同模型。",
          "; ".join(f"{k}: {v}" for k, v in list(leaders.items())[:4]) + "。",
          f"1096 行中 {numeric} 行有主指标，{len(analysis)-numeric} 行缺失。"],
         "方向统一后的相对排名显示的是任务内可访问性，而非可跨任务平均的原始分数。",
         f"仅 {int(analysis.strict_clean.sum())} 行达到 strict exact-safe；clean 图仍含 lineage-unresolved canonical 行。"),
        ("Fig. 2 — Geometry vs Action-conditioned Retrieval", "几何与动作条件检索可访问性并非同一个标量能力。",
         [f"两轴 Spearman ρ={f2['rho']:.2f}（n={len(f2.get('plotted', []))} 个有双轴模型）。",
          f"Geometry #1: {f2.get('geom_top','见数据表')}；Action-conditioned retrieval #1: {f2.get('dyn_top','见数据表')}。",
          "每个轴都先做 task-wise direction correction 与 rank normalization。"],
         "轴间相关只描述模型排序共变，不等同于因果或统一能力因子。", "组成任务缺失会降低个别模型的轴覆盖。"),
        ("Fig. 3 — Measured-layer Profiles", "已测层的最佳位置横跨早中晚阶段，单层模型仅作为独立点展示。",
         [f"Peak map 含 {len(peak)} 个 model×task：early {stage.get('early',0)}、middle {stage.get('middle',0)}、late {stage.get('late',0)}、output {stage.get('output/merger',0)}。",
          f"覆盖 {peak.model.nunique()} 个模型与 {peak.task.nunique()} 类能力；仅至少三层的数据连成趋势线。",
          f"任务峰值中位深度范围 {f3['task_stats'].median_peak_depth.min():.2f}–{f3['task_stats'].median_peak_depth.max():.2f}。"],
         "不同能力在同一骨干内也可能于不同已测位置达到最佳值。", "Peak 仅表示已测层最优，不表示训练过程中的能力涌现。"),
        ("Fig. 4 — Streaming Prefix Effect", "更长历史的影响正负并存，整体并非稳定增益。",
         [f"{pos}/{total} 个配对 p24>p8，{neg}/{total} 个配对 p24<p8。",
          f"标准化 Δ 中位数 {median_delta:+.3f}，bootstrap 95% CI [{f4['ci'][0]:+.3f}, {f4['ci'][1]:+.3f}]。",
          f"共覆盖 {pairs.model.nunique()} 个模型、{pairs.task.nunique()} 类能力。"],
         "更多帧既可能补充证据，也可能引入聚合或时窗不一致带来的退化。", "Δ 仅在相同 model/task/layer/metric 内配对。"),
        ("Fig. 5 — Layer × Prefix Structure", "层与前缀的联合结构因模型和任务而异，完整图集保留部分覆盖。",
         [f"主图自动选择 {len(cases)} 个互不重复案例；完整分页图含 {len(f5['complete'])} 个有 V1 数值的 model×task tensor。",
          f"选中案例中 best-layer 变化最多为 {int(max([f5.get('changes',1)])) if cases else 0} 种。",
          "每个 panel 的 BEST 来自该 model×task 的真实二维 tensor。"],
         "LingBot-video 的 p16/p24 局部 tensor 也在完整图集中，p8/p12 明示 NA。", "panel 内颜色归一化，不能跨 panel 比绝对色深。"),
        ("Fig. 6 — Construct Validity", "高 probe 分数不保证读出了目标 action construct。",
         [f"四个核心模型 Full−strongest retrained control 为 " + ", ".join(f"{x:+.3f}" for x in deltas) + "。",
          "Wan P12/L20: correct R@1=0.152，copy=0.120，shuffled=0.100，no-action=0.054。",
          f"最小 full-control gap 为 {min(deltas):+.3f}，显示 shortcut 可接近 full。"],
         "真正的 construct evidence 是 full 相对最强非目标基线的增量，而不是绝对 R@1。", "控制结果来自 development validation、单 probe seed。"),
        ("Table 1 — Benchmark / Task Definitions", "六个 probe 将几何、持久性和状态更新明确拆分。",
         ["A1/A2 测直接几何与跨视角一致性。", "B2 测隐藏物体在末帧坐标系中的持续表征。", "C1/C2/C3 分别测一步、多步和反事实状态更新。"],
         "任务分解避免把所有空间表现压缩成一个 leaderboard。", "headline metric 仅代表当前 frozen protocol。"),
        ("Table 2 — Main Quantitative Results", "原始指标的领先者随能力变化，单榜排名会掩盖结构。",
         [f"表中包含 {len(t2all)} 个模型、{len(CAP_ORDER)} 个 headline capability。",
          "所有列使用 representative layer 与 available-prefix mean，部分覆盖以 ‡ 和覆盖列明示。", "†/*/L 直接标出质量来源，不静默删除历史结果。"],
         "表格用于读原始量纲；跨能力比较应回到 Fig.1 的任务内标准化。", "不同 campaign 身份保留在配套 CSV 中。"),
    ]
    lines = ["# PPT-ready Takeaways", ""]
    for name, takeaway, evidence, interpretation, caveat in sections:
        lines += [f"## {name}", "", "### One-sentence takeaway", "", takeaway, "", "### Evidence", ""]
        lines += [f"- {x}" for x in evidence]
        lines += ["", "### Interpretation", "", interpretation, "", "### Caveat", "", caveat, ""]
    (OUT / "ppt_takeaways.md").write_text("\n".join(lines), encoding="utf-8")


def report(analysis, f1, f2data, f2, f3, f4, f5, f6) -> None:
    numeric = int(analysis.metric_value.notna().sum())
    sweep = [m for m, z in analysis.groupby("model") if z.layer.nunique() >= 3]
    pairs = f4["pairs"]
    peak = f3["peak"]
    leaders = f1["all_rep"].sort_values("relative_score", ascending=False).groupby("capability").first().model_display.to_dict()
    posfrac = float((pairs.direction == "p24 > p8").mean()) if len(pairs) else float("nan")
    lines = [
        "# Global Visualization Report — Frozen 1096", "",
        "## Scope and aggregation contract", "",
        "This package reads the latest 1096-row display-first table and existing result/control artifacts. No experiment was run and no canonical/provenance table was modified.", "",
        "- Main summaries use predefined representative layers for every model, including L30 for LingBot-video/MiniMax-H3.",
        "- Prefix aggregation is an arithmetic mean across explicitly listed available prefixes; partial coverage is marked and never presented as 4/4.",
        "- Cross-task figures direction-correct each raw metric and rank-normalize within task before averaging.",
        "- All rows share the table's `historical_candidate_v1_val_protocol`; campaign identity remains explicit in every derived CSV.",
        "- Visualization-clean means canonical-source with no confirmed mismatch. It is not paper-clean: lineage-unresolved rows remain; strict exact-safe is reported separately.", "",
        "## Quality reality", "",
        f"The source contains {numeric} numeric primary results and {len(analysis)-numeric} missing primary results. "
        f"Only {int(analysis.strict_clean.sum())} rows are strict exact-safe. Thus the figures are a global landscape of existing evidence, not a certified final paper table.", "",
        "## Findings", "",
        f"1. Task leaders are distributed across models: " + "; ".join(f"{k}={v}" for k,v in leaders.items()) + ".",
        f"2. Geometry–action-conditioned-retrieval Spearman correlation is {f2['rho']:.3f}; these axes should not be collapsed without loss.",
        f"3. Best measured layers: early={(peak.peak_stage=='early').sum()}, middle={(peak.peak_stage=='middle').sum()}, late={(peak.peak_stage=='late').sum()}, output={(peak.peak_stage=='output/merger').sum()} across {len(peak)} model×task profiles.",
        f"4. Longer context helps in {posfrac:.1%} of {len(pairs)} p8→p24 matched comparisons; median standardized Δ={pairs.standardized_delta.median():+.3f}.",
        f"5. Construct controls show Full−strongest retrained-control gaps from {min(f6['deltas']):+.3f} to {max(f6['deltas']):+.3f} across the four core C1 models.", "",
        "## Required final summary", "",
        f"TOTAL_CONFIGS: {len(analysis)}", f"TOTAL_NUMERIC_RESULTS: {numeric}", "",
        "MODELS: " + ", ".join(MODEL_META[m][1] for m in model_order(analysis.model.unique().tolist())),
        "TASKS: " + ", ".join(sorted(analysis.probe.unique())),
        "PREFIXES: " + ", ".join(map(str, sorted(analysis.prefix.dropna().astype(int).unique()))),
        "LAYER_SWEEP_MODELS: " + ", ".join(MODEL_META[m][1] for m in model_order(sweep)), "",
        "FIGURES_GENERATED: 6 groups", "TABLES_GENERATED: 2", "",
        "TOP_GLOBAL_FINDING_1: Spatial capability leaders differ by task; there is no defensible single overall leaderboard.",
        f"TOP_GLOBAL_FINDING_2: Geometry and action-conditioned retrieval are only partially coupled (Spearman rho={f2['rho']:.3f}).",
        "TOP_GLOBAL_FINDING_3: Capability peaks occur across early, middle, and late stages rather than universally in the middle.",
        f"TOP_GLOBAL_FINDING_4: p24 beats p8 in {int((pairs.direction=='p24 > p8').sum())}/{len(pairs)} matched representative comparisons.",
        "TOP_GLOBAL_FINDING_5: Absolute C1 scores can overstate action-specific information when retrained controls remain competitive.", "",
        "MOST_SURPRISING_RESULT: Longer temporal context is not a monotonic benefit, and its sign depends on model × task.", "",
        "RESULTS_THAT_CHANGED_AFTER_LOOKING_ACROSS_ALL_1096: The apparent middle-layer and longer-context heuristics both become mixed, capability-specific patterns.", "",
        "CLAIMS_NOT_SUPPORTED_BY_CURRENT_DATA: A universal middle-layer optimum; monotonic benefit from longer history; one scalar spatial ability; a cross-task video-generation advantage; paper-final causal/action claims from raw probe score alone.",
    ]
    (OUT / "global_visualization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def handoff_documents(analysis, f1, f2data, f3, f4, f5, f6) -> None:
    source_summary = json.loads((ROOT / "results/analysis/q00_1096_loose/summary.json").read_text())
    sources = [
        (SOURCE, "Frozen 1096 V1 primary-result universe"),
        (CANONICAL_ALL, "Frozen V1 atomic A1-camera child metrics; hash is locked in the loose summary"),
        (C1_VALIDITY, "Existing V1-era C1 control results for Fig06 only"),
        (WAN_VALIDITY, "Existing V1 Wan action-control results for Fig06 only"),
    ]
    lines = ["# V1 input lock", "",
        "All main figures and Table02 are rooted in the frozen 1096-row table whose only protocol value is `historical_candidate_v1_val_protocol`.",
        "No score was read from `results/analysis/tstep_canonical_v2_seeded_v1` or any other V2 directory.", "",
        "| Path | Role | SHA256 |", "|---|---|---|",
    ]
    for path, role in sources:
        lines.append(f"| `{path.relative_to(ROOT)}` | {role} | `{hashlib.sha256(path.read_bytes()).hexdigest()}` |")
    lines += ["", f"The canonical atomic table hash matches the frozen loose-build receipt: `{source_summary['source_sha256']['canonical_all']}`.",
        "A1-camera rows are joined only to frozen streaming-depth parents with the same model/layer/prefix; campaign, protocol, model_variant and source universe ID remain explicit.",
        "`lingbot` is the sole stored model key and is changed only at display time to `LingBot-video`; no model_variant, campaign or protocol aliases are merged.", ""]
    (OUT / "V1_INPUT_SOURCES.md").write_text("\n".join(lines), encoding="utf-8")

    ling = analysis[analysis.model == "lingbot"]
    by_task = ling.groupby("probe").agg(configs=("universe_id", "size"), numeric=("metric_value", "count"))
    diag = ["# LingBot-video recovery diagnosis", "",
        f"- Frozen V1 configurations: **{len(ling)}**; finite primary results: **{int(ling.metric_value.notna().sum())}**.",
        f"- Tasks: {', '.join(sorted(ling.probe.unique()))}.",
        f"- Prefixes in universe: {', '.join(map(str, sorted(ling.prefix.dropna().astype(int).unique())))}; prefixes with numeric V1 results: 16, 24.",
        f"- Layers: {', '.join(map(str, sorted(ling.layer.dropna().astype(int).unique())))}; every layer has at least one numeric result.", "",
        "## Where it disappeared", "",
        "1. The source table and analysis table retained the stored key `lingbot`, but the display map called it `LingBot`, not the requested `LingBot-video`.",
        "2. Fig04's pre-fix `fig04` implementation required both p8 and p24 and immediately skipped a model×task when either endpoint was absent. LingBot-video has zero numeric p8 rows, so all of its delta cells vanished.",
        "3. Fig05's pre-fix `matrix_case_stats` implementation required at least three numeric prefixes. LingBot-video has only p16/p24, so every tensor was removed before case selection.",
        "4. Fig01/02/03/Table02 did contain the model under the shorter label `LingBot`; A2 remained absent because all 20 V1 A2 configurations lack a numeric value.", "",
        "The responsible locations were `MODEL_META`, `fig04`, and `matrix_case_stats` in `analyses/q00_1096/build_visualization_package.py` (pre-fix recovery snapshot: approximately lines 48, 624, and 706).",
        "The corrected functions now emit structural rows plus explicit NA reasons instead of using missing endpoints/coverage as whole-model deletion criteria.", "",
        "## V1 counts by task", "", "| Probe | Configs | Finite primary results |", "|---|---:|---:|",
    ]
    for task, r in by_task.iterrows(): diag.append(f"| {task} | {int(r.configs)} | {int(r.numeric)} |")
    diag += ["", "Other models affected by the old completeness gate: Fig04 lost no other whole model; Fig05's representative-only tensor set also omitted Aether, DINO, F3R, and Qwen2.5-VL because they expose one measured layer. The complete export now restores all five as readable one-layer or partial-prefix heatmaps.", ""]
    (OUT / "LINGBOT_VIDEO_DIAGNOSIS.md").write_text("\n".join(diag), encoding="utf-8")

    models = model_order(analysis.model.unique().tolist())
    peak = f3["peak"]
    endpoint = f4["endpoint_status"]
    control_models = set(f6["models"])
    complete_models = {x["model"] for x in f5["complete"]}
    rows = []
    for model in models:
        display = MODEL_META[model][1]
        nrep = int((f1["all_rep"].model == model).sum())
        rows.append({"model": model, "model_display": display, "chart": "Fig01 atlas", "included": True,
                     "numeric_elements": nrep, "missing_elements": len(CAP_ORDER)-nrep,
                     "reason": "full row retained; missing capabilities rendered NA"})
        r2 = f2data[f2data.model == model].iloc[0]
        rows.append({"model": model, "model_display": display, "chart": "Fig02 geometry vs action-conditioned retrieval",
                     "included": r2.axis_status == "plotted", "numeric_elements": int(r2.geometry_components + r2.dynamics_components),
                     "missing_elements": int(pd.isna(r2.geometry_score) + pd.isna(r2.dynamics_score)), "reason": r2.axis_status})
        npk = int((peak.model == model).sum())
        rows.append({"model": model, "model_display": display, "chart": "Fig03 layer curves + peak map", "included": npk > 0,
                     "numeric_elements": npk, "missing_elements": len(PANEL_ORDER)-npk,
                     "reason": "isolated output shown as point; peak means best measured layer" if model in SINGLE_REPRESENTATION_MODELS else "measured layer sweep"})
        ep = endpoint[endpoint.model == model]
        nmatch = int(ep.pair_status.eq("matched_same_identity").sum())
        rows.append({"model": model, "model_display": display, "chart": "Fig04 prefix delta + curves", "included": True,
                     "numeric_elements": nmatch, "missing_elements": len(CAP_ORDER)-nmatch,
                     "reason": f"row/curves retained; {nmatch} matched p8-p24 cells, {len(CAP_ORDER)-nmatch} NA"})
        nheat = sum(x["model"] == model for x in f5["complete"])
        rows.append({"model": model, "model_display": display, "chart": "Fig05 layer x prefix heatmaps", "included": model in complete_models,
                     "numeric_elements": nheat, "missing_elements": len(CAP_ORDER)-nheat,
                     "reason": "all model×task tensors with any V1 value exported; absent tasks have no V1 numeric value"})
        rows.append({"model": model, "model_display": display, "chart": "Fig06 existing controls", "included": model in control_models,
                     "numeric_elements": 1 if model in control_models else 0, "missing_elements": 0 if model in control_models else 1,
                     "reason": "existing V1 full/control results" if model in control_models else "no eligible existing V1 control artifact"})
        rows.append({"model": model, "model_display": display, "chart": "Table02 main results", "included": True,
                     "numeric_elements": nrep, "missing_elements": len(CAP_ORDER)-nrep,
                     "reason": "full model row retained; missing cells shown as —"})
    cov = pd.DataFrame(rows)
    cov.to_csv(INV / "model_chart_coverage.csv", index=False)
    md = ["# Model × chart coverage", "", "`included=true` means the model is visibly rendered in that chart/table; numeric and missing counts explain partial coverage.", "",
          "| Model | Chart | Included | Numeric | Missing | Reason |", "|---|---|---:|---:|---:|---|"]
    for r in cov.itertuples(index=False): md.append(f"| {r.model_display} | {r.chart} | {'yes' if r.included else 'no'} | {r.numeric_elements} | {r.missing_elements} | {r.reason} |")
    (INV / "model_chart_coverage.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def manifest() -> None:
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "output_manifest.csv":
            rows.append({"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    pd.DataFrame(rows).to_csv(OUT / "output_manifest.csv", index=False)


def main() -> None:
    for directory in [OUT, FIG, TAB, DATA, INV]: directory.mkdir(parents=True, exist_ok=True)
    configure_style()
    analysis = build_analysis_table()
    inventory(analysis)
    metrics = metric_landscape(analysis)
    _, f1 = fig01(metrics)
    f2data, f2 = fig02(f1["all_rep"])
    f2.update({"data": f2data, "geom_top": f2data.sort_values("geometry_rank").iloc[0].model_display,
               "dyn_top": f2data.sort_values("dynamics_rank").iloc[0].model_display})
    _, f3 = fig03(metrics)
    _, f4 = fig04(metrics)
    _, f5 = fig05(metrics)
    case_check = pd.read_csv(DATA / "fig05_case_selection.csv")
    case_check["selected_role"] = case_check.selected_role.fillna("")
    f5["changes"] = int(case_check.query("selected_role != ''").best_layer_changes.max())
    _, f6 = fig06()
    table01()
    t2all, _ = table02(f1["all_rep"], f1["clean_rep"])
    make_takeaways(analysis, f1, f2, f3, f4, f5, f6, t2all)
    report(analysis, f1, f2data, f2, f3, f4, f5, f6)
    handoff_documents(analysis, f1, f2data, f3, f4, f5, f6)
    summary = {
        "total_configs": len(analysis), "total_numeric_results": int(analysis.metric_value.notna().sum()),
        "strict_exact_safe_rows": int(analysis.strict_clean.sum()), "models": analysis.model.nunique(),
        "figure_groups": 6, "table_groups": 2, "output_files": sum(1 for p in OUT.rglob("*") if p.is_file()),
    }
    (OUT / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
