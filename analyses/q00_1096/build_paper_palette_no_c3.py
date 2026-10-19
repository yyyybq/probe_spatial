#!/usr/bin/env python3
"""Build the paper figure package without C3 using the manuscript palette."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import build_paper_revision as revision


base = revision.base
OUT = base.ROOT / "results/analysis/q00_1096_paper_no_c3_palette_20260923T000000Z"
base.OUT = OUT
base.FIG = OUT / "figures"
base.TAB = OUT / "tables"
base.DATA = OUT / "data"
base.INV = OUT / "inventory"

P = {
    "header": "#F1F1EE",
    "l1": "#DCE8EC",
    "l2": "#E4EAD9",
    "l3": "#EEE0D8",
    "gray": "#E8E4DE",
    "line": "#A8A096",
    "dark": "#4E4842",
    "best": "#B0C0A8",
    "second": "#CEB8A0",
    "sand": "#DCD2C2",
    "terra": "#926860",
    "link": "#5C6C70",
    "cite": "#566858",
}

TASK_COLORS = {
    "A1 Depth": P["link"],
    "A1 Mask": P["dark"],
    "A1 Camera": P["line"],
    "A2 View": P["l1"],
    "B2 Persistence": P["cite"],
    "C1 Dynamics": P["terra"],
    "C2 Integration": P["second"],
}

EXCLUDED_CAPABILITY = "C3 Counterfactual"
EXCLUDED_PROBE = "counterfactual"
base.CAP_ORDER = [x for x in base.CAP_ORDER if x != EXCLUDED_CAPABILITY]
base.PANEL_ORDER = [x for x in base.PANEL_ORDER if x != EXCLUDED_CAPABILITY]
base.TASK_DEFINITIONS = [x for x in base.TASK_DEFINITIONS if x[1] != "Counterfactual control"]
base.FIG05_CASES = [("videomae", "A1 Mask", "Instance identity"),
                    ("vjepa2", "A1 Mask", "Instance identity"),
                    ("videomae", "B2 Persistence", "Out-of-view object localization"),
                    ("vjepa2", "B2 Persistence", "Out-of-view object localization")]
base.FAMILY_COLORS.update({
    "Geometry-specialized": P["link"],
    "Image discriminative": P["cite"],
    "Masked-video": P["terra"],
    "Predictive-video": P["second"],
    "Multimodal/VLM": P["dark"],
    "Video-generation": P["line"],
})


def capability_level(capability: str) -> str:
    if capability.startswith("A1") or capability.startswith("A2"):
        return "l1"
    if capability.startswith("B2"):
        return "l2"
    return "l3"


def configure_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.family": "DejaVu Sans",
        "font.size": 16, "axes.titlesize": 22, "axes.labelsize": 18,
        "xtick.labelsize": 15, "ytick.labelsize": 15,
        "legend.fontsize": 13, "figure.titlesize": 24,
        "text.color": P["dark"], "axes.labelcolor": P["dark"],
        "axes.titlecolor": P["dark"], "xtick.color": P["dark"],
        "ytick.color": P["dark"], "axes.edgecolor": P["line"],
        "grid.color": P["gray"], "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def semantic_cmap(kind: str) -> LinearSegmentedColormap:
    if kind == "delta":
        return LinearSegmentedColormap.from_list("paper_delta", [P["terra"], P["header"], P["link"]])
    if kind == "layer":
        return LinearSegmentedColormap.from_list("paper_layer", [P["header"], P["sand"], P["terra"]])
    return LinearSegmentedColormap.from_list("paper_surface", [P["header"], P["l3"], P["second"], P["terra"]])


def draw_heatmap(ax, matrix, row_labels, col_labels, title, cmap, vmin=0, vmax=1,
                 annotations=None, cbar_label=None, center=None, add_colorbar=True):
    masked = np.ma.masked_invalid(matrix)
    kind = "delta" if center is not None else "layer" if "layer depth" in str(cbar_label).lower() else "surface"
    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax) if center is not None else Normalize(vmin, vmax)
    im = ax.imshow(masked, aspect="auto", cmap=semantic_cmap(kind), norm=norm)
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
                label = annotations[i][j]
                if label:
                    ax.text(j, i, label, ha="center", va="center", fontsize=14, color=P["dark"])
    if add_colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=.025, pad=.02)
        cbar.outline.set_edgecolor(P["line"])
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=16)
    return im


def blend(c0: str, c1: str, value: float) -> tuple[float, float, float]:
    a, b = np.array(to_rgb(c0)), np.array(to_rgb(c1))
    return tuple(a * (1 - value) + b * value)


def fig01(metrics: pd.DataFrame):
    all_rep = base.representative_rows(metrics, False)
    clean_rep = base.representative_rows(metrics, True)
    all_rep["version"] = "all"; clean_rep["version"] = "clean"
    pd.concat([all_rep, clean_rep], ignore_index=True).to_csv(base.DATA / "fig01_capability_atlas_data.csv", index=False)
    order = base.model_order(metrics.model.unique().tolist())
    titles = {}
    for tag, rep in [("all", all_rep)]:
        pivot = rep.pivot(index="model", columns="capability", values="relative_score").reindex(index=order, columns=base.CAP_ORDER)
        rgba = np.ones((len(order), len(base.CAP_ORDER), 4))
        darks = {"l1": P["link"], "l2": P["cite"], "l3": P["terra"]}
        lights = {"l1": P["l1"], "l2": P["l2"], "l3": P["l3"]}
        for j, cap in enumerate(base.CAP_ORDER):
            level = capability_level(cap)
            for i in range(len(order)):
                value = pivot.iloc[i, j]
                rgba[i, j, :3] = to_rgb(P["gray"]) if pd.isna(value) else blend(lights[level], darks[level], .82 * float(value))
        title = "Spatial capability profiles differ across representation families"
        titles[tag] = title
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.imshow(rgba, aspect="auto")
        ax.set_xticks(range(len(base.CAP_ORDER)),
                      [base.display_task(task) for task in base.CAP_ORDER],
                      rotation=25, ha="right")
        ax.set_yticks(range(len(order)), [base.MODEL_META[m][1] for m in order])
        ax.set_xticks(np.arange(-.5, len(base.CAP_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(len(order)):
            for j in range(len(base.CAP_ORDER)):
                if pd.isna(pivot.iloc[i, j]):
                    ax.text(j, i, "NA", ha="center", va="center", color=P["dark"], fontsize=13)
        ax.set_title(title, loc="left", fontweight="bold", pad=42)
        ax.set_xlabel("Capability (color intensity = within-task percentile)")
        ax.set_ylabel("Model")
        trans = ax.get_xaxis_transform()
        group_spans = [(0, 3, "L1 Spatial Structure", P["l1"]),
                       (4, 4, "L2 Object Persistence", P["l2"]),
                       (5, 6, "L3 Predictive Spatial Update", P["l3"])]
        for start, end, label, color in group_spans:
            ax.plot([start - .45, end + .45], [1.025, 1.025], transform=trans, color=color, lw=12,
                    solid_capstyle="butt", clip_on=False)
            ax.text((start + end) / 2, 1.045, label, transform=trans, ha="center", va="bottom",
                    fontsize=12, color=P["dark"])
        handles = [Patch(facecolor=P["l1"], edgecolor=P["line"], label="L1 Spatial Structure"),
                   Patch(facecolor=P["l2"], edgecolor=P["line"], label="L2 Object Persistence"),
                   Patch(facecolor=P["l3"], edgecolor=P["line"], label="L3 Predictive Spatial Update"),
                   Patch(facecolor=P["gray"], edgecolor=P["line"], label="NA")]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
        fig.tight_layout(rect=(0, .075, 1, .97))
        base.save_figure(fig, f"fig01_capability_atlas_{tag}")
    return pd.concat([all_rep, clean_rep], ignore_index=True), {"titles": titles, "all_rep": all_rep, "clean_rep": clean_rep}


def fig02(rep: pd.DataFrame):
    geometry = ["A1 Depth", "A1 Mask", "A1 Camera", "A2 View"]
    dynamics = ["C1 Dynamics", "C2 Integration"]
    rows = []
    for model in base.model_order(rep.model.unique().tolist()):
        z = rep[rep.model.eq(model)]
        g, d = z[z.capability.isin(geometry)], z[z.capability.isin(dynamics)]
        ng, nd = g.capability.nunique(), d.capability.nunique()
        rows.append({"model": model, "model_display": base.MODEL_META[model][1],
                     "model_family": base.MODEL_META[model][0],
                     "geometry_score": g.relative_score.mean() if ng == 4 else np.nan,
                     "dynamics_score": d.relative_score.mean() if nd == 2 else np.nan,
                     "geometry_components": ng, "dynamics_components": nd,
                     "geometry_component_tasks": ";".join(geometry),
                     "dynamics_component_tasks": ";".join(dynamics),
                     "aggregation": "equal-weight mean of within-metric percentiles; all components required",
                     "axis_status": "plotted" if ng == 4 and nd == 2 else
                         "Missing: " + ", ".join(x for x in geometry + dynamics if x not in set(z.capability))})
    out = pd.DataFrame(rows)
    plotted = out[out.axis_status.eq("plotted")].copy()
    plotted["geometry_rank"] = plotted.geometry_score.rank(ascending=False, method="min").astype(int)
    plotted["dynamics_rank"] = plotted.dynamics_score.rank(ascending=False, method="min").astype(int)
    rho = base.spearman_no_scipy(plotted.geometry_score, plotted.dynamics_score)
    out = out.merge(plotted[["model", "geometry_rank", "dynamics_rank"]], on="model", how="left")
    out["spearman_rho"] = rho
    out.to_csv(base.DATA / "fig02_geometry_vs_dynamics_data.csv", index=False)
    fig, (ax, note) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [4, 1]})
    ax.set_facecolor(P["header"])
    for r in plotted.itertuples(index=False):
        ax.scatter(r.geometry_score, r.dynamics_score, s=240,
                   color=base.FAMILY_COLORS[r.model_family], marker=base.FAMILY_MARKERS[r.model_family],
                   edgecolor=P["dark"], linewidth=1.2, zorder=3)
        offset = {"qwen2_5_vl": (-118, 10), "vjepa2": (10, -21), "minimax_h3": (9, 9)}.get(r.model, (8, 8))
        ax.annotate(r.model_display, (r.geometry_score, r.dynamics_score), xytext=offset,
                    textcoords="offset points", fontsize=14, color=P["dark"])
    ax.axvline(plotted.geometry_score.median(), color=P["line"], ls="--", lw=1.5)
    ax.axhline(plotted.dynamics_score.median(), color=P["line"], ls="--", lw=1.5)
    ax.set(xlim=(-.05, 1.08), ylim=(-.05, 1.08),
           xlabel="L1 Spatial Structure\n(metric geometry, instance identity, view consistency)",
           ylabel="L3 Predictive Spatial Update\n(one-step prediction, path integration)")
    ax.grid(color="white", linewidth=1.2)
    title = "Spatial structure and predictive update yield different model orderings"
    ax.set_title(title, loc="left", fontweight="bold", pad=18)
    note.axis("off")
    note.text(0, .94, "Fixed composition", fontsize=16, fontweight="bold", va="top",
              bbox={"facecolor": P["header"], "edgecolor": P["line"], "pad": 8})
    note.text(0, .79, "L1\n4 equal-weight metrics", fontsize=13, va="top",
              bbox={"facecolor": P["l1"], "edgecolor": P["line"], "pad": 8})
    note.text(0, .60, "L3\nOne-step prediction\n+ path integration", fontsize=14, va="top",
              bbox={"facecolor": P["l3"], "edgecolor": P["line"], "pad": 8})
    note.text(0, .34, "LingBot-video\nView consistency unavailable;\nshown outside the 2D plot.", fontsize=13, va="top")
    fig.tight_layout()
    base.save_figure(fig, "fig02_geometry_vs_dynamics")
    return out, {"rho": rho, "title": title, "plotted": plotted}


def render_table(df: pd.DataFrame, stem: str, title: str, font_size: int = 14,
                 best_second: bool = False, numeric_cols=None, directions=None,
                 footer: str = "", col_widths=None, eligible_rows=None):
    fig, ax = plt.subplots(figsize=(16, 9)); ax.axis("off")
    ax.set_title(title, loc="left", fontweight="bold", pad=20)
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", colLoc="center",
                     loc="center", bbox=[0, .08, 1, .82], colWidths=col_widths)
    table.auto_set_font_size(False); table.set_fontsize(font_size)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor(P["line"]); cell.set_linewidth(1.1)
        cell.get_text().set_color(P["dark"])
        if i == 0:
            cell.set_facecolor(P["header"]); cell.get_text().set_weight("bold")
        elif "Level" in df.columns:
            level = str(df.iloc[i - 1]["Level"])
            cell.set_facecolor(P["l1"] if level.startswith("L1") else P["l2"] if level.startswith("L2") else P["l3"])
        else:
            header = str(df.columns[j])
            if header.startswith(("Metric", "Instance", "Camera", "View")):
                color = P["l1"]
            elif header.startswith("Out-of-view"):
                color = P["l2"]
            elif header.startswith(("One-step", "Path")):
                color = P["l3"]
            else:
                color = P["gray"]
            cell.set_facecolor(color)
            if j == 0: cell.get_text().set_weight("bold")
    if best_second and numeric_cols and directions:
        eligible = list(range(len(df))) if eligible_rows is None else list(eligible_rows)
        for col in numeric_cols:
            if col not in df: continue
            j = df.columns.get_loc(col)
            vals = pd.to_numeric(df.loc[eligible, col].astype(str), errors="coerce").dropna()
            ranked = vals.sort_values(ascending=directions[col] == "min").index.tolist()
            if ranked:
                table[(ranked[0] + 1, j)].set_facecolor(P["best"])
                table[(ranked[0] + 1, j)].get_text().set_weight("bold")
            if len(ranked) > 1:
                table[(ranked[1] + 1, j)].set_facecolor(P["second"])
                table[(ranked[1] + 1, j)].get_text().set_weight("bold")
    if footer:
        fig.text(.5, .035, footer, ha="center", fontsize=12, color=P["dark"])
    fig.tight_layout()
    fig.savefig(base.TAB / f"{stem}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(base.TAB / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def table01():
    columns = ["Level", "Probe", "Capability", "Input", "Prediction target", "Headline metric", "Interpretation"]
    df = pd.DataFrame(base.TASK_DEFINITIONS, columns=columns)
    df.to_csv(base.TAB / "table01_task_definitions.csv", index=False)
    wrapped = df.copy()
    widths = {"Probe": 14, "Capability": 17, "Input": 19, "Prediction target": 21,
              "Headline metric": 18, "Interpretation": 20}
    for col, width in widths.items():
        wrapped[col] = wrapped[col].map(lambda x: "\n".join(textwrap.wrap(str(x), width=width)))
    render_table(wrapped, "table01_task_definitions",
                 "Six tasks and seven metrics span three spatial levels",
                 font_size=12, footer="L1 = spatial structure; L2 = object persistence; L3 = predictive spatial update.")
    return df


def table02(rep_all, rep_clean):
    rename = {"A1 Depth": "Metric depth ↓", "A1 Mask": "Instance identity ↓", "A1 Camera": "Camera pose ↑",
              "A2 View": "View consistency ↑", "B2 Persistence": "Out-of-view localization ↓",
              "C1 Dynamics": "One-step prediction ↑", "C2 Integration": "Path integration ↑"}
    directions = {"Metric depth ↓": "min", "Instance identity ↓": "min", "Camera pose ↑": "max",
                  "View consistency ↑": "max", "Out-of-view localization ↓": "min",
                  "One-step prediction ↑": "max", "Path integration ↑": "max"}
    outputs = []
    for tag, rep in [("all", rep_all)]:
        order = base.model_order(rep.model.unique().tolist())
        values = rep.pivot(index="model", columns="capability", values="raw_mean").reindex(index=order, columns=base.CAP_ORDER)
        rows = []
        for model in order:
            row = {"Model": base.MODEL_META[model][1]}
            for cap in base.CAP_ORDER:
                value = values.loc[model, cap]
                if cap == "A1 Mask" and pd.notna(value):
                    raw = pd.read_csv(base.SOURCE, dtype=str)
                    ids = rep[(rep.model == model) & (rep.capability == cap)].iloc[0].source_universe_ids.split(";")
                    vals = [Decimal(v) for v in raw[raw.universe_id.isin(ids)].selected_result]
                    row[rename[cap]] = str((sum(vals) / len(vals)).quantize(Decimal(".0001"), rounding=ROUND_HALF_UP))
                else:
                    row[rename[cap]] = "—" if pd.isna(value) else (f"{value:.2f}" if cap == "B2 Persistence" else f"{value:.4f}")
            rows.append(row)
        df = pd.DataFrame(rows, columns=["Model"] + list(rename.values()))
        df.to_csv(base.TAB / f"table02_main_results_{tag}.csv", index=False)
        rep.to_csv(base.TAB / f"table02_main_results_{tag}_long.csv", index=False)
        # Campaign names describe extraction provenance, not a benchmark tier.
        # All representations, including MiniMax-H3 and LingBot-video, are peers
        # in the main table and are eligible for the same highlighting rules.
        eligible = list(range(len(order)))
        render_table(df, f"table02_main_results_{tag}",
                     "Capability-specific leaders across the three spatial levels",
                     font_size=12, best_second=True, numeric_cols=list(rename.values()), directions=directions,
                     eligible_rows=eligible,
                     footer="Representative layer; mean over the evaluated prefixes. Green = best; sand = second best.",
                     col_widths=[.16] + [.12] * 7)
        outputs.append(df)
    return outputs[0], outputs[0]


original_exact_metric_landscape = revision.exact_metric_landscape
original_save_figure = revision.original_save_figure


def metric_landscape_no_c3(analysis):
    metrics = original_exact_metric_landscape(analysis)
    metrics = metrics[metrics.capability.ne(EXCLUDED_CAPABILITY)].copy()
    # MiniMax-H3's exact-zero A1 summaries are not valid measurements. The
    # source FP16 features contain non-finite values and the legacy loss helper
    # replaced NaN/Inf with zero, which can otherwise make these cells look like
    # perfect depth/mask scores (and a real zero camera AUC). Keep the model and
    # configuration rows, but render the affected values as unavailable rather
    # than using them for normalization, ranking, or peaks.
    invalid_minimax_a1 = (
        metrics.model.eq("minimax_h3")
        & metrics.capability.isin(["A1 Depth", "A1 Mask", "A1 Camera"])
        & metrics.metric_value.eq(0)
    )
    metrics["raw_metric_value_before_validation"] = metrics.metric_value
    metrics["measurement_status"] = "reported"
    metrics["measurement_note"] = ""
    metrics.loc[invalid_minimax_a1, "measurement_status"] = "invalid_nonfinite_zero_sanitization"
    metrics.loc[invalid_minimax_a1, "measurement_note"] = (
        "non-finite MiniMax-H3 FP16 feature/loss was replaced by zero by legacy handling"
    )
    metrics.loc[invalid_minimax_a1, ["metric_value", "oriented_value"]] = np.nan
    metrics.to_csv(base.DATA / "metric_landscape.csv", index=False)
    return metrics


def save_figure(fig, stem, directory=None):
    for ax in fig.axes:
        ax.tick_params(colors=P["dark"])
        for spine in ax.spines.values(): spine.set_edgecolor(P["line"])
        for line in ax.get_xgridlines() + ax.get_ygridlines(): line.set_color(P["gray"])
        for patch in ax.patches:
            color = patch.get_facecolor()[:3]
            if np.linalg.norm(np.array(color) - np.array(to_rgb("#2166AC"))) < .08:
                patch.set_facecolor(P["terra"])
            elif np.linalg.norm(np.array(color) - np.array(to_rgb("#BDBDBD"))) < .08:
                patch.set_facecolor(P["sand"])
        if stem.startswith("fig04b_"):
            for line in ax.lines:
                if line.get_label() in TASK_COLORS:
                    line.set_color(TASK_COLORS[line.get_label()])
            legend = ax.get_legend()
            if legend:
                for line, label in zip(legend.get_lines(), [t.get_text() for t in legend.get_texts()]):
                    if label in TASK_COLORS: line.set_color(TASK_COLORS[label])
        if stem.startswith("fig06"):
            legend = ax.get_legend()
            if legend:
                labels = [t.get_text() for t in legend.get_texts()]
                for handle, label in zip(legend.legend_handles, labels):
                    handle.set_facecolor(P["terra"] if label == "Full" else P["sand"])
                    handle.set_edgecolor("none")
    if stem.startswith("fig04b_"):
        for legend in fig.legends:
            for line, label in zip(legend.get_lines(), [t.get_text() for t in legend.get_texts()]):
                if label in TASK_COLORS: line.set_color(TASK_COLORS[label])
    if stem == "fig03a_layer_emergence" and len(fig.axes) > len(base.PANEL_ORDER):
        for ax in fig.axes[len(base.PANEL_ORDER):]: ax.axis("off")
    if stem == "fig04a_prefix_delta_heatmap":
        fig.axes[0].set_title("Longer history has capability-dependent effects", loc="left", fontweight="bold", pad=18)
    directory = directory or base.FIG
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=180, facecolor="white")
    fig.savefig(directory / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def export_plot_csvs():
    """Place the actual plotted subset next to every PNG/PDF, including panels."""
    atlas = pd.read_csv(base.DATA / "fig01_capability_atlas_data.csv")
    atlas = atlas[atlas.version.eq("all")]
    layer = pd.read_csv(base.DATA / "fig03_layer_curves_data.csv")
    peaks = pd.read_csv(base.DATA / "fig03_layer_statistics.csv")
    peaks = peaks[peaks.record_type.eq("model_task_peak")]
    endpoints = pd.read_csv(base.DATA / "fig04_prefix_statistics.csv")
    endpoints = endpoints[endpoints.record_type.eq("model_task_endpoint_status")]
    curves = pd.read_csv(base.DATA / "fig04_prefix_curves_data.csv")
    heat = pd.read_csv(base.DATA / "fig05_all_heatmaps_data.csv")
    cases = pd.read_csv(base.DATA / "fig05_case_selection.csv").fillna("")
    cases = cases[cases.selected_role.ne("")]
    controls = pd.read_csv(base.DATA / "fig06_construct_validity_data.csv")
    order = base.model_order(atlas.model.unique().tolist())
    assert atlas[atlas.capability.eq("A1 Mask")].model.nunique() == 10
    assert layer[layer.panel.eq("A1 Mask")].model.nunique() == 10
    assert heat[heat.task.eq("A1 Mask")].model.nunique() == 10
    assert not atlas.capability.str.contains("C3").any()
    mappings = []
    for path in sorted(base.FIG.rglob("*.png")):
        stem = path.stem
        if path.parent == base.TAB:
            continue  # table CSVs are already colocated
        if stem.startswith("fig01"):
            data = atlas
        elif stem.startswith("fig02"):
            data = pd.read_csv(base.DATA / "fig02_geometry_vs_dynamics_data.csv")
        elif stem.startswith("fig03a_panel_"):
            task = next(x for x in base.PANEL_ORDER if stem.endswith(x.lower().replace(" ", "_")))
            data = layer[layer.panel.eq(task)]
        elif stem.startswith("fig03a"):
            data = layer
        elif stem.startswith("fig03b"):
            data = peaks
        elif stem.startswith("fig04a"):
            data = endpoints
        elif stem.startswith("fig04b_panel_"):
            data = curves[curves.model.eq(stem.removeprefix("fig04b_panel_"))]
        elif stem.startswith("fig04b"):
            page = int(stem[-2:]) - 1
            data = curves[curves.model.isin(order[page * 5:page * 5 + 5])]
        elif stem == "fig05_layer_prefix_cases":
            data = heat.merge(cases[["model", "task", "campaign", "selected_role"]], on=["model", "task", "campaign"])
        elif stem.startswith("fig05b_all_heatmaps_page"):
            page = int(stem[-2:]) - 1
            keys = heat.heatmap_key.drop_duplicates().tolist()[page * 6:page * 6 + 6]
            data = heat[heat.heatmap_key.isin(keys)]
        elif stem.startswith("fig05_"):
            data = heat[heat.heatmap_key.eq(stem.removeprefix("fig05_"))]
        elif stem == "fig06_panel_retrained_baselines":
            data = controls[controls.panel.eq("retrained_ablation")]
        elif stem == "fig06_panel_wan_controls":
            data = controls[controls.panel.eq("wan_action_validity")]
        elif stem.startswith("fig06"):
            data = controls
        else:
            raise ValueError(f"Unmapped figure: {path}")
        assert len(data), path
        data.to_csv(path.with_suffix(".csv"), index=False)
        mappings.append({"figure": str(path.relative_to(base.OUT)), "csv": str(path.with_suffix(".csv").relative_to(base.OUT)),
                         "rows": len(data)})
    pd.DataFrame(mappings).to_csv(base.OUT / "figure_data_index.csv", index=False)
    coverage = []
    for model in order:
        for fig in ("Fig01", "Fig02", "Fig03", "Fig04", "Fig05", "Fig06"):
            status, reason = "included", ""
            if fig == "Fig02" and model == "lingbot":
                status, reason = "side note", "A2 unavailable; no fabricated 2D coordinate"
            if fig == "Fig06" and model not in set(controls.model):
                status, reason = "no numeric panel", "No existing V1 control result in the source table"
            if fig == "Fig04" and model == "lingbot":
                reason = "Row retained; endpoint deltas NA where p8 is unavailable"
            coverage.append({"model": base.MODEL_META[model][1], "figure": fig, "status": status, "reason": reason})
    pd.DataFrame(coverage).to_csv(base.OUT / "model_figure_coverage.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=base.ROOT / "figures/exp_figures" /
                        ("v1_mask_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    parser.add_argument(
        "--render-existing-data",
        type=Path,
        help="Render Fig. 3 and Fig. 5 from an existing package's data/metric_landscape.csv.",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite existing package: {args.out}")
    base.OUT = args.out.resolve()
    base.FIG = base.OUT
    base.DATA = base.OUT / "data"
    base.TAB = base.OUT / "tables"
    base.INV = base.OUT / "inventory"
    for directory in [base.OUT, base.FIG, base.TAB, base.DATA, base.INV]: directory.mkdir(parents=True, exist_ok=True)
    configure_style()
    if args.render_existing_data is not None:
        source = args.render_existing_data.resolve()
        metric_path = source / "metric_landscape.csv" if source.is_dir() else source
        metrics = pd.read_csv(metric_path)
        base.fig03(metrics)
        base.fig05(metrics)
        print(json.dumps({"source": str(metric_path), "rendered": ["fig03a_layer_emergence", "fig05_layer_prefix_cases"]}, indent=2))
        return
    analysis = base.build_analysis_table()
    analysis = analysis[analysis.probe.ne(EXCLUDED_PROBE)].copy()
    analysis.to_csv(base.OUT / "analysis_table.csv", index=False)
    metadata_path = base.DATA / "analysis_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update({"figure_scope": "A1/A2/B2/C1/C2 only", "excluded_probe": EXCLUDED_PROBE,
                     "scoped_row_count": len(analysis)})
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    base.inventory(analysis)
    metrics = metric_landscape_no_c3(analysis)
    _, f1 = fig01(metrics)
    f2data, f2 = fig02(f1["all_rep"])
    _, f3 = base.fig03(metrics)
    _, f4 = base.fig04(metrics)
    _, f5 = base.fig05(metrics)
    _, f6 = base.fig06()
    table01()
    t2all, _ = table02(f1["all_rep"], f1["clean_rep"])
    scope = [
        "# Paper figure package", "",
        "- Included equally: metric geometry (depth and camera), instance identity, view consistency, out-of-view object localization, one-step prediction, and path integration.",
        "- Fig02 spatial-structure axis: equal-weight mean of metric-depth, instance-identity, camera-pose, and view-consistency percentiles.",
        "- Fig03 selects a separate campaign per model/task; singleton instance-identity results remain points.",
        "- Fig05 main cases: VideoMAE and V-JEPA2, each with instance identity and out-of-view object localization; full atlas includes every task.",
        "- Excluded from every figure and graphical table: C3 counterfactual.",
        "- Palette: the 13 manuscript colors supplied on 2026-09-23.",
        "- Fig02 predictive-update axis: one-step prediction and path integration only.",
        "- Table highlighting: best #B0C0A8; second best #CEB8A0.", "",
    ]
    (base.OUT / "FIGURE_SCOPE_AND_PALETTE.md").write_text("\n".join(scope))
    summary = {"source_configs": 1096, "scoped_configs": len(analysis), "excluded_probe": EXCLUDED_PROBE,
               "models": analysis.model.nunique(), "capabilities": base.CAP_ORDER,
               "figure_groups": 6, "table_groups": 2}
    (base.OUT / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    export_plot_csvs()
    base.manifest()
    print(json.dumps(summary, indent=2))


base.configure_style = configure_style
base.draw_heatmap = draw_heatmap
base.save_figure = save_figure
base.metric_landscape = metric_landscape_no_c3
base.fig01 = fig01
base.fig02 = fig02
base.render_table = render_table
base.table01 = table01
base.table02 = table02

if __name__ == "__main__":
    main()
