#!/usr/bin/env python3
"""Paper figure for early vs late generation coordinates (ICLR).

The plot uses the frozen canonical-v2 seeded-v1 summaries and retains stable
internal probe keys while displaying the final paper task nomenclature.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/analysis/tstep_canonical_v2_seeded_v1/results.csv"
PAPER = ROOT / "figures/exp_figures/timestep_mask_20260923"
ANALYSIS = ROOT / "results/analysis/tstep_canonical_v2_seeded_v1/figures"

P = {
    "header": "#F1F1EE",
    "gray": "#E8E4DE",
    "line": "#A8A096",
    "dark": "#4E4842",
    "early": "#5C6C70",
    "late": "#CEB8A0",
    "terra": "#926860",
    "link": "#5C6C70",
}

MODELS = ["wan", "cogvideox", "minimax_h3"]
MODEL_LABEL = {"wan": "Wan", "cogvideox": "CogVideoX", "minimax_h3": "MiniMax-H3"}
TSTEP = {"wan": (499, 999), "cogvideox": (499, 999), "minimax_h3": (500, 1000)}
LAYERS = [15, 20, 25]
PREFIXES = [8, 12, 16, 24]
PANELS = [
    ("streaming_depth", "Metric geometry: depth", "Depth loss $\\downarrow$", "min"),
    ("streaming_mask", "Instance identity", "Identity loss $\\downarrow$", "min"),
    ("view_consistency", "View consistency", "View accuracy $\\uparrow$", "max"),
    ("ego_belief_v2", "Out-of-view object localization", "Localization error ($^\\circ$) $\\downarrow$", "min"),
]

GRID_LABEL = {
    "streaming_depth": "Metric depth",
    "streaming_mask": "Instance identity",
    "view_consistency": "View consistency",
    "ego_belief_v2": "Out-of-view loc.",
}

PANEL_STEM = {
    "streaming_depth": "fig06_panel_streaming_depth",
    "streaming_mask": "fig06_panel_streaming_mask",
    "view_consistency": "fig06_panel_view_consistency",
    "ego_belief_v2": "fig06_panel_ego_belief_v2",
}


def configure_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "text.color": P["dark"],
        "axes.labelcolor": P["dark"],
        "axes.titlecolor": P["dark"],
        "xtick.color": P["dark"],
        "ytick.color": P["dark"],
        "axes.edgecolor": P["line"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_frame() -> pd.DataFrame:
    frame = pd.read_csv(SRC)
    frame["metric_value"] = pd.to_numeric(frame["metric_value"], errors="coerce")
    frame["timestep"] = pd.to_numeric(frame["timestep"], errors="coerce").astype("Int64")
    frame["layer"] = pd.to_numeric(frame["layer"], errors="coerce").astype("Int64")
    frame["prefix"] = pd.to_numeric(frame["prefix"], errors="coerce").astype("Int64")
    zero_a1 = (
        frame.probe.isin(["streaming_depth", "streaming_mask"])
        & frame.metric_value.notna()
        & (frame.metric_value.abs() < 1e-12)
    )
    frame.loc[zero_a1, "metric_value"] = np.nan
    return frame


def layer20_mean(frame: pd.DataFrame, model: str, probe: str, timestep: int) -> float | None:
    q = frame[
        (frame.model == model)
        & (frame.probe == probe)
        & (frame.layer == 20)
        & (frame.timestep == timestep)
        & frame.metric_value.notna()
    ]
    if q.empty:
        return None
    return float(q.metric_value.mean())


def pair_delta(frame: pd.DataFrame, model: str, probe: str, layer: int, prefix: int, direction: str) -> float:
    early, late = TSTEP[model]
    q = frame[(frame.model == model) & (frame.probe == probe) & (frame.layer == layer) & (frame.prefix == prefix)]
    a = q[q.timestep == early]
    b = q[q.timestep == late]
    if a.empty or b.empty or a.metric_value.isna().all() or b.metric_value.isna().all():
        return np.nan
    av = float(a.metric_value.iloc[0])
    bv = float(b.metric_value.iloc[0])
    if direction == "min":
        return av - bv
    return bv - av


def draw_bars(ax, frame: pd.DataFrame, probe: str, ylabel: str, title: str) -> None:
    x = np.arange(len(MODELS))
    width = 0.34
    early_vals, late_vals = [], []
    for model in MODELS:
        early_t, late_t = TSTEP[model]
        early_vals.append(layer20_mean(frame, model, probe, early_t))
        late_vals.append(layer20_mean(frame, model, probe, late_t))

    finite = [v for v in early_vals + late_vals if v is not None]
    ymax = max(finite) * 1.22 if finite else 1.0
    for i, (ev, lv) in enumerate(zip(early_vals, late_vals)):
        if ev is None and lv is None:
            ax.text(x[i], ymax * 0.48, "n.a.", ha="center", va="center",
                    fontsize=10, color=P["line"], style="italic")
            continue
        if ev is not None:
            ax.bar(x[i] - width / 2, ev, width, color=P["early"], edgecolor="none", zorder=2)
            label = f"{ev:.3f}" if ev < 10 else f"{ev:.1f}"
            ax.text(x[i] - width / 2, ev, label, ha="center", va="bottom",
                    fontsize=8, color=P["dark"])
        if lv is not None:
            ax.bar(x[i] + width / 2, lv, width, color=P["late"], edgecolor="none", zorder=2)
            label = f"{lv:.3f}" if lv < 10 else f"{lv:.1f}"
            ax.text(x[i] + width / 2, lv, label, ha="center", va="bottom",
                    fontsize=8, color=P["dark"])

    ax.set_ylim(0, ymax)
    ax.set_xticks(x, [MODEL_LABEL[m] for m in MODELS])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=8)
    ax.grid(axis="y", color=P["gray"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def draw_grid(ax, frame: pd.DataFrame) -> None:
    signs = []
    labels = []
    for model in MODELS:
        for probe, short, _, direction in PANELS:
            row = []
            for layer in LAYERS:
                for prefix in PREFIXES:
                    delta = pair_delta(frame, model, probe, layer, prefix, direction)
                    if not np.isfinite(delta):
                        row.append(np.nan)
                    elif delta > 1e-12:
                        row.append(1.0)
                    elif delta < -1e-12:
                        row.append(-1.0)
                    else:
                        row.append(0.0)
            signs.append(row)
            labels.append(f"{MODEL_LABEL[model]}  {GRID_LABEL[probe]}")
    matrix = np.array(signs, dtype=float)
    cols = [f"L{layer} p{prefix}" for layer in LAYERS for prefix in PREFIXES]
    cmap = LinearSegmentedColormap.from_list("sign", [P["terra"], P["header"], P["link"]])
    ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, vmin=-1.2, vmax=1.2)
    ax.set_xticks(range(len(cols)), cols, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Layer–prefix pairs (E = early better, L = late better)",
                 loc="left", fontweight="bold", pad=8)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    for boundary in range(len(PANELS), len(MODELS) * len(PANELS), len(PANELS)):
        ax.axhline(boundary - 0.5, color="white", linewidth=3.0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isfinite(val):
                ax.text(j, i, "n.a.", ha="center", va="center", fontsize=7, color=P["line"])
                continue
            mark = "E" if val < 0 else ("L" if val > 0 else "=")
            ax.text(j, i, mark, ha="center", va="center", fontsize=8,
                    color="white" if val != 0 else P["dark"], fontweight="bold")


def main() -> int:
    configure_style()
    frame = load_frame()
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.35], hspace=0.42, wspace=0.30,
                          left=0.12, right=0.98, top=0.90, bottom=0.11)
    for i, (probe, short, ylabel, _) in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, i])
        draw_bars(ax, frame, probe, ylabel, short)
    ax_grid = fig.add_subplot(gs[1, :])
    draw_grid(ax_grid, frame)
    fig.legend(
        handles=[
            Patch(facecolor=P["early"], edgecolor="none", label="Early"),
            Patch(facecolor=P["late"], edgecolor="none", label="Late"),
            Patch(facecolor=P["terra"], edgecolor="none", label="Early better"),
            Patch(facecolor=P["link"], edgecolor="none", label="Late better"),
            Patch(facecolor=P["header"], edgecolor=P["line"], label="No change"),
        ],
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
    )
    fig.text(
        0.5, 0.012,
        "Canonical-v2 seeded-v1 validation summaries. "
        "Each pair of bars is the same model at the early vs late generation coordinate "
        "(layer 20, mean of prefixes 8/12/16/24).\n"
        "Lower is better for metric depth, instance identity, and out-of-view object localization; higher is better for view consistency. "
        "MiniMax-H3 zero-valued depth and instance identity summaries at layers 20/25 are excluded.\n"
        "Bottom grid: E/L/= for each layer–prefix pair. CogVideoX and MiniMax-H3 localization do not change.",
        ha="center", fontsize=8.0, color=P["line"],
    )
    PAPER.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    for dest in (PAPER, ANALYSIS):
        fig.savefig(dest / "fig06_timestep_analysis.pdf", bbox_inches="tight", facecolor="white")
        fig.savefig(dest / "fig06_timestep_analysis.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    for probe, title, ylabel, _ in PANELS:
        panel, ax = plt.subplots(figsize=(4.8, 3.4))
        draw_bars(ax, frame, probe, ylabel, title)
        panel.tight_layout()
        panel.savefig(PAPER / f"{PANEL_STEM[probe]}.pdf", bbox_inches="tight", facecolor="white")
        panel.savefig(PAPER / f"{PANEL_STEM[probe]}.png", dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(panel)
    print(f"wrote {PAPER / 'fig06_timestep_analysis.pdf'}")
    print(f"wrote {PAPER / 'fig06_timestep_analysis.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
