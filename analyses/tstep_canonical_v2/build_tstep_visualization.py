#!/usr/bin/env python3
"""Visualize the canonical-v2 timestep probing matrix.

The layout follows analyses/q00_1096/build_visualization_package.py: keep every
universe row, never average raw metrics across tasks, and rank within a task
after direction correction.  The extra axis here is scheduler timestep.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/analysis/tstep_canonical_v2_seeded_v1"
FIG = OUT / "figures"
TAB = OUT / "tables"
DATA = OUT / "data"
INV = OUT / "inventory"

MODEL_META = {
    "wan": ("Wan", "#D62828"),
    "cogvideox": ("CogVideoX", "#F77F00"),
    "minimax_h3": ("MiniMax-H3", "#7B2CBF"),
}
MODEL_ORDER = ["wan", "cogvideox", "minimax_h3"]
TSTEP_PAIR = {
    "wan": (499, 999),
    "cogvideox": (499, 999),
    "minimax_h3": (500, 1000),
}
CAP_ORDER = [
    "A1 Depth", "A1 Mask", "A2 View", "B1 Ego",
    "B2 Persistence", "C1 Dynamics", "C2 Integration", "C3 Counterfactual",
]
REP_LAYER = 20
PREFIXES = [8, 12, 16, 24]


def configure_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.family": "DejaVu Sans",
        "font.size": 15, "axes.titlesize": 20, "axes.labelsize": 16,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "legend.fontsize": 13, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def tstep_label(model: str, timestep: int) -> str:
    low, high = TSTEP_PAIR[model]
    tag = "early" if int(timestep) == low else "late" if int(timestep) == high else str(timestep)
    return f"{MODEL_META[model][0]} t={int(timestep)} ({tag})"


def load_results() -> pd.DataFrame:
    frame = pd.read_csv(OUT / "results.csv")
    frame["metric_value"] = pd.to_numeric(frame["metric_value"], errors="coerce")
    frame["timestep"] = pd.to_numeric(frame["timestep"], errors="coerce").astype("Int64")
    frame["layer"] = pd.to_numeric(frame["layer"], errors="coerce").astype("Int64")
    frame["prefix"] = pd.to_numeric(frame["prefix"], errors="coerce").astype("Int64")
    zero_a1 = (
        frame.probe.isin(["streaming_depth", "streaming_mask"])
        & frame.metric_value.notna()
        & (frame.metric_value.abs() < 1e-12)
    )
    frame["a1_zero_masked"] = zero_a1
    frame.loc[zero_a1, "metric_value"] = np.nan
    frame["oriented"] = np.where(
        frame.metric_direction.eq("min"), -frame.metric_value, frame.metric_value
    )
    frame["model_display"] = frame.model.map(lambda m: MODEL_META.get(m, (m, "#333"))[0])
    frame["tstep_role"] = [
        "early" if pd.notna(t) and int(t) == TSTEP_PAIR.get(m, (None, None))[0]
        else "late" if pd.notna(t) and int(t) == TSTEP_PAIR.get(m, (None, None))[1]
        else "other"
        for m, t in zip(frame.model, frame.timestep)
    ]
    return frame


def rank_within(frame: pd.DataFrame, value: str, group: str) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, idx in frame.groupby(group).groups.items():
        vals = frame.loc[idx, value]
        if vals.notna().sum() == 0:
            continue
        ranks = vals.rank(method="average")
        n = vals.notna().sum()
        out.loc[idx] = 0.5 if n <= 1 else (ranks - 1) / (n - 1)
    return out


def draw_heatmap(ax, matrix, row_labels, col_labels, title, cmap, vmin, vmax, annotations=None, center=None):
    masked = np.ma.masked_invalid(matrix)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax) if center is not None else Normalize(vmin, vmax)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=28, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_title(title, loc="left", fontweight="bold", pad=14)
    ax.set_xticks(np.arange(-.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    if annotations is not None:
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = annotations[i][j]
                if text:
                    ax.text(j, i, text, ha="center", va="center", fontsize=11,
                            color="white" if np.isfinite(matrix[i, j]) and abs(matrix[i, j]) > .55 else "#222")
    ax.figure.colorbar(im, ax=ax, fraction=.025, pad=.02)
    return im


def fig_completion(frame: pd.DataFrame) -> None:
    status_score = {"TRAIN_COMPLETE": 1.0, "RUNNING": 0.5, "PARTIAL": 0.25, "MISSING": np.nan}
    rows, labels = [], []
    for model in MODEL_ORDER:
        for timestep in TSTEP_PAIR[model]:
            rows.append((model, timestep))
            labels.append(tstep_label(model, timestep))
    matrix = np.full((len(rows), len(CAP_ORDER)), np.nan)
    marks = [["" for _ in CAP_ORDER] for _ in rows]
    for i, (model, timestep) in enumerate(rows):
        z = frame[(frame.model == model) & (frame.timestep == timestep)]
        for j, cap in enumerate(CAP_ORDER):
            q = z[z.capability == cap]
            complete = int(q.status.eq("TRAIN_COMPLETE").sum())
            total = len(q)
            running = int(q.status.eq("RUNNING").sum())
            matrix[i, j] = complete / total if total else np.nan
            if running:
                marks[i][j] = f"{complete}/{total}*"
            elif complete == total:
                marks[i][j] = f"{complete}"
            else:
                marks[i][j] = f"{complete}/{total}"
    fig, ax = plt.subplots(figsize=(16, 7))
    cmap = LinearSegmentedColormap.from_list("done", ["#F7F7F7", "#6A51A3"])
    draw_heatmap(ax, matrix, labels, CAP_ORDER,
                 "Timestep probing completion: 3 models × 2 scheduler indices × 8 probes",
                 cmap, 0, 1, marks)
    fig.text(.5, .02, "* includes currently running writers. Denominator is 12 cells (3 layers × 4 prefixes).",
             ha="center", fontsize=12, color="#555")
    fig.tight_layout(rect=(0, .05, 1, 1))
    save_figure(fig, "fig00_completion")


def representative(frame: pd.DataFrame) -> pd.DataFrame:
    z = frame[(frame.layer == REP_LAYER) & frame.metric_value.notna()].copy()
    pieces = []
    for keys, q in z.groupby(["model", "timestep", "capability", "probe", "metric_name", "metric_direction"]):
        model, timestep, cap, probe, metric, direction = keys
        pieces.append({
            "model": model, "model_display": MODEL_META[model][0],
            "timestep": int(timestep), "tstep_role": q.tstep_role.iloc[0],
            "capability": cap, "probe": probe, "metric_name": metric,
            "metric_direction": direction,
            "raw_mean": q.metric_value.mean(),
            "oriented_mean": q.oriented.mean(),
            "n_prefix": q.prefix.nunique(),
            "prefixes": ",".join(map(str, sorted(q.prefix.dropna().astype(int).unique()))),
            "n_complete": int(q.status.eq("TRAIN_COMPLETE").sum()),
        })
    out = pd.DataFrame(pieces)
    if out.empty:
        return out
    out["relative_score"] = rank_within(out, "oriented_mean", "capability")
    return out


def fig_atlas(rep: pd.DataFrame) -> None:
    if rep.empty:
        return
    labels = [tstep_label(m, t) for m in MODEL_ORDER for t in TSTEP_PAIR[m]]
    index = [(m, t) for m in MODEL_ORDER for t in TSTEP_PAIR[m]]
    pivot = pd.DataFrame(np.nan, index=pd.MultiIndex.from_tuples(index), columns=CAP_ORDER)
    for row in rep.itertuples(index=False):
        pivot.loc[(row.model, row.timestep), row.capability] = row.relative_score
    matrix = pivot.to_numpy(float)
    fig, ax = plt.subplots(figsize=(16, 7))
    cmap = LinearSegmentedColormap.from_list("cap", ["#F2F0F7", "#9E9AC8", "#3F007D"])
    draw_heatmap(ax, matrix, labels, CAP_ORDER,
                 "Within-task accessibility ranks at layer 20, averaged over prefixes",
                 cmap, 0, 1)
    ax.set_xlabel("Capability (training-time validation; direction-corrected then ranked)")
    fig.text(.5, .02, "C1–C3 use cosine similarity because in-batch R@1 is undefined at batch=2. This is not eval_diag global_R@1.",
             ha="center", fontsize=12, color="#555")
    fig.tight_layout(rect=(0, .05, 1, 1))
    save_figure(fig, "fig01_capability_atlas")


def paired_timestep(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    work = frame[frame.metric_value.notna() & frame.status.eq("TRAIN_COMPLETE")]
    for (model, layer, prefix, probe), q in work.groupby(["model", "layer", "prefix", "probe"]):
        low, high = TSTEP_PAIR[model]
        a = q[q.timestep == low]
        b = q[q.timestep == high]
        if a.empty or b.empty:
            continue
        a_val = float(a.metric_value.iloc[0])
        b_val = float(b.metric_value.iloc[0])
        direction = a.metric_direction.iloc[0]
        oriented_delta = (b.oriented.iloc[0] - a.oriented.iloc[0])
        rows.append({
            "model": model, "model_display": MODEL_META[model][0],
            "layer": int(layer), "prefix": int(prefix), "probe": probe,
            "capability": a.capability.iloc[0],
            "metric_name": a.metric_name.iloc[0],
            "metric_direction": direction,
            "early_timestep": low, "late_timestep": high,
            "early_value": a_val, "late_value": b_val,
            "raw_delta_late_minus_early": b_val - a_val,
            "oriented_delta_late_minus_early": float(oriented_delta),
            "late_better": bool(oriented_delta > 0),
        })
    return pd.DataFrame(rows)


def fig_timestep_delta(pairs: pd.DataFrame) -> None:
    if pairs.empty:
        return
    agg = pairs.groupby(["model", "capability"], as_index=False).oriented_delta_late_minus_early.mean()
    labels = [MODEL_META[m][0] for m in MODEL_ORDER]
    matrix = np.full((len(MODEL_ORDER), len(CAP_ORDER)), np.nan)
    marks = [["" for _ in CAP_ORDER] for _ in MODEL_ORDER]
    for i, model in enumerate(MODEL_ORDER):
        for j, cap in enumerate(CAP_ORDER):
            q = agg[(agg.model == model) & (agg.capability == cap)]
            if q.empty:
                continue
            value = float(q.oriented_delta_late_minus_early.iloc[0])
            matrix[i, j] = value
            marks[i][j] = f"{value:+.3f}"
    finite = matrix[np.isfinite(matrix)]
    vmax = float(np.nanmax(np.abs(finite))) if len(finite) else 1.0
    vmax = max(vmax, 1e-6)
    fig, ax = plt.subplots(figsize=(16, 5.5))
    cmap = LinearSegmentedColormap.from_list("delta", ["#B2182B", "#F7F7F7", "#2166AC"])
    draw_heatmap(ax, matrix, labels, CAP_ORDER,
                 "Late minus early timestep (oriented so positive = late scheduler index is better)",
                 cmap, -vmax, vmax, marks, center=0)
    ax.set_xlabel("Mean over completed layer/prefix pairs with both timesteps")
    fig.tight_layout()
    save_figure(fig, "fig02_timestep_delta")


def fig_layer(frame: pd.DataFrame) -> None:
    z = frame[frame.metric_value.notna()].copy()
    if z.empty:
        return
    z["relative"] = rank_within(z, "oriented", "capability")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    for ax, cap in zip(axes.flat, CAP_ORDER):
        q = z[z.capability == cap]
        for model in MODEL_ORDER:
            for role, ls in (("early", "-"), ("late", "--")):
                r = q[(q.model == model) & (q.tstep_role == role)]
                if r.empty:
                    continue
                means = r.groupby("layer").relative.mean().sort_index()
                ax.plot(means.index, means.values, marker="o", ls=ls, lw=2.0,
                        color=MODEL_META[model][1], label=f"{MODEL_META[model][0]} {role}")
        ax.set_title(cap, fontweight="bold", fontsize=14)
        ax.grid(color="#ECECEC")
        ax.set_xticks([15, 20, 25])
        ax.set_ylim(-.05, 1.05)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
    fig.suptitle("Layer trends by scheduler timestep (within-task ranks, all prefixes)", fontweight="bold")
    fig.tight_layout(rect=(0, .08, 1, .96))
    save_figure(fig, "fig03_layer_by_timestep")


def fig_prefix(frame: pd.DataFrame) -> None:
    z = frame[(frame.layer == REP_LAYER) & frame.metric_value.notna()].copy()
    if z.empty:
        return
    z["relative"] = rank_within(z, "oriented", "capability")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    for ax, cap in zip(axes.flat, CAP_ORDER):
        q = z[z.capability == cap]
        for model in MODEL_ORDER:
            for role, ls in (("early", "-"), ("late", "--")):
                r = q[(q.model == model) & (q.tstep_role == role)]
                if r.empty:
                    continue
                means = r.groupby("prefix").relative.mean().sort_index()
                ax.plot(means.index, means.values, marker="o", ls=ls, lw=2.0,
                        color=MODEL_META[model][1], label=f"{MODEL_META[model][0]} {role}")
        ax.set_title(cap, fontweight="bold", fontsize=14)
        ax.grid(color="#ECECEC")
        ax.set_xticks(PREFIXES)
        ax.set_ylim(-.05, 1.05)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
    fig.suptitle("Prefix trends at layer 20 (within-task ranks)", fontweight="bold")
    fig.tight_layout(rect=(0, .08, 1, .96))
    save_figure(fig, "fig04_prefix_by_timestep")


def table_main(rep: pd.DataFrame) -> pd.DataFrame:
    TAB.mkdir(parents=True, exist_ok=True)
    if rep.empty:
        return rep
    rows = []
    for model in MODEL_ORDER:
        for timestep in TSTEP_PAIR[model]:
            row = {"Model": tstep_label(model, timestep)}
            q = rep[(rep.model == model) & (rep.timestep == timestep)]
            for cap in CAP_ORDER:
                r = q[q.capability == cap]
                if r.empty or not np.isfinite(r.raw_mean.iloc[0]):
                    row[cap] = "—"
                else:
                    value = r.raw_mean.iloc[0]
                    row[cap] = f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(TAB / "table02_layer20_prefix_mean.csv", index=False)
    return table


def table_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pairs
    summary = pairs.groupby(["model", "capability"], as_index=False).agg(
        n=("oriented_delta_late_minus_early", "size"),
        mean_oriented_delta=("oriented_delta_late_minus_early", "mean"),
        frac_late_better=("late_better", "mean"),
        mean_early=("early_value", "mean"),
        mean_late=("late_value", "mean"),
    )
    summary["model_display"] = summary.model.map(lambda m: MODEL_META[m][0])
    summary.to_csv(TAB / "table03_timestep_paired_deltas.csv", index=False)
    return summary


def _cell_label(row) -> str:
    return (
        f"{row.model} t={int(row.timestep)} L{int(row.layer)} p{int(row.prefix)} {row.probe}"
    )


def write_findings(frame: pd.DataFrame, rep: pd.DataFrame, pairs: pd.DataFrame, summary: dict) -> None:
    counts = frame.status.value_counts().to_dict()
    n_complete = int(counts.get("TRAIN_COMPLETE", 0))
    n_running = int(counts.get("RUNNING", 0))
    n_partial = int(counts.get("PARTIAL", 0))
    n_missing = int(counts.get("MISSING", 0))
    n_primary = int(frame.has_primary.map(lambda v: str(v).strip().lower() in {"true", "1", "1.0"}).sum())
    n_plot = int(frame.metric_value.notna().sum())
    zero = frame[frame.a1_zero_masked].sort_values(["model", "probe", "timestep", "layer", "prefix"])
    b2 = frame[(frame.probe.eq("ego_belief_v2")) & frame.metric_value.notna()].copy()
    chance_b2 = b2[b2.metric_value.sub(90.15).abs() <= 0.25].sort_values(
        ["model", "timestep", "layer", "prefix"]
    )
    lines = [
        "# Timestep probing analysis (canonical-v2 seeded-v1)",
        "",
        f"Generated **{datetime.now(timezone.utc).isoformat()}**.",
        "",
        "This package is a **preliminary training-time validation read** of the frozen matrix. It is not a paper table. Headline numbers are the last logged `val/*` metrics, not `eval_diag`.",
        "",
        "## Scope",
        "",
        "Frozen universe: Wan / CogVideoX / MiniMax-H3 × 2 scheduler timesteps × layers `{15,20,25}` × prefixes `{8,12,16,24}` × 8 probes = **576** experiments, seed 42, ScanNet++ canonical-v2 seeded-v1.",
        "",
        "Timestep labels are **early vs late scheduler index** (Wan/CogVideoX 499 vs 999; MiniMax-H3 500 vs 1000). They are not a claim about noise level.",
        "",
        "Primary attempt for each cell is the `TRAIN_COMPLETE` run with the latest mtime. Historical non-canonical timestep runs are not used to fill gaps.",
        "",
        "## Completion",
        "",
        f"- TRAIN_COMPLETE: **{n_complete} / 576**",
        f"- RUNNING: **{n_running}**",
        f"- PARTIAL: **{n_partial}**",
        f"- MISSING: **{n_missing}**",
        f"- Rows with a finite training-time primary metric (includes exact 0.0 before the A1 mask): **{n_primary}**",
        f"- Rows kept in plots after masking `|A1 loss| < 1e-12`: **{n_plot}**",
        "",
    ]
    primary_ok = frame.has_primary.map(lambda v: str(v).strip().lower() in {"true", "1", "1.0"})
    no_primary = frame[~primary_ok]
    if not no_primary.empty:
        lines.append(
            f"Finite-primary gaps (logged val is missing or non-finite; not backfilled): **{len(no_primary)}**."
        )
        grouped = (
            no_primary.groupby(["model", "probe", "layer"], dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values(["model", "probe", "layer"])
        )
        for row in grouped.itertuples(index=False):
            lines.append(f"- `{row.model}` {row.probe} L{int(row.layer)}: {int(row.n)}")
        lines.append("")
    running = frame[frame.status.eq("RUNNING")].sort_values(["model", "timestep", "layer", "prefix", "probe"])
    partial = frame[frame.status.eq("PARTIAL")].sort_values(["model", "timestep", "layer", "prefix", "probe"])
    missing = frame[frame.status.eq("MISSING")].sort_values(["model", "timestep", "layer", "prefix", "probe"])
    if running.empty and partial.empty and missing.empty:
        n_claims = len(summary.get("active_claims") or [])
        lines.append("No RUNNING, PARTIAL, or MISSING cells. Empty cells were not invented.")
        if n_claims:
            lines.append(
                f"Writer-claim files still on disk: **{n_claims}**. "
                "They do not override a `TRAIN_COMPLETE` marker, so those cells stay complete."
            )
    else:
        lines.append("Incomplete cells (left empty; not backfilled):")
        lines.append("")
        for label, block in (("RUNNING", running), ("PARTIAL", partial), ("MISSING", missing)):
            if block.empty:
                continue
            lines.append(f"### {label}")
            lines.append("")
            for row in block.itertuples(index=False):
                lines.append(f"- `{row.unique_key}`")
            lines.append("")
    lines += [
        "",
        "## Metric contract",
        "",
        "These numbers are **last-epoch training validation logs**, not a later `eval_diag` pass. In particular:",
        "",
        "- A1 Depth / Mask use `val/loss_depth` and `val/loss_identity` (or `val/loss`).",
        "- A2 uses `val/overlap_acc`, not the paper balanced accuracy table metric.",
        "- B1 / B2 use azimuth / spherical angular error.",
        "- C1–C3 use cosine similarity. In-batch R@1 is NaN at batch=2, so it is not reported.",
        "",
        "Do not paste these C-probe values into a paper table that expects `global_R@1` or `intervention_validity`.",
        "",
        "## Timestep effect",
        "",
        "Comparisons are late scheduler index minus early scheduler index. Positive oriented Δ means the late index is better. This is not a noise-schedule interpretation.",
        "",
    ]
    if pairs.empty:
        lines.append("Not enough paired complete cells to estimate late-minus-early effects.")
    else:
        better = pairs.groupby("capability").late_better.mean()
        delta = pairs.groupby("capability").oriented_delta_late_minus_early.mean()
        lines.append("Fraction of completed layer/prefix pairs where the **late** scheduler index is better (oriented). A1 pairs already drop exact-zero losses. B2 pairs that sit on chance are still in this count and are flagged below; they are not a timestep effect.")
        lines.append("")
        for cap in CAP_ORDER:
            if cap not in better.index:
                continue
            lines.append(f"- {cap}: {100*better[cap]:.0f}% late-better, mean oriented Δ={delta[cap]:+.4f} (n={int((pairs.capability==cap).sum())})")
        by_model = pairs.groupby("model").late_better.mean()
        lines += ["", "By model, pooling all capabilities (same caveats):", ""]
        for model in MODEL_ORDER:
            if model in by_model.index:
                lines.append(f"- {MODEL_META[model][0]}: {100*by_model[model]:.0f}% late-better")
    if not rep.empty:
        lines += ["", "## Layer-20 snapshot (prefix mean)", ""]
        lines.append("Best cell is the highest direction-corrected prefix mean at layer 20. Exact-zero A1 losses are already excluded. A B2 value near 90.15° is chance-level, not a win.")
        lines.append("")
        for cap in CAP_ORDER:
            q = rep[rep.capability == cap].dropna(subset=["raw_mean"])
            if q.empty:
                continue
            best = q.loc[q.oriented_mean.idxmax()]
            lines.append(
                f"- {cap}: best is {best.model_display} t={int(best.timestep)} "
                f"({best.metric_name}={best.raw_mean:.4f}, n_prefix={int(best.n_prefix)})"
            )
    lines += [
        "",
        "## Validity flags",
        "",
        f"- Exact-zero A1 losses (`|val/loss| < 1e-12` on depth or mask) are masked to NaN before ranking and pairing. Count: **{len(zero)}**. These match non-finite feature / loss-zeroing and are **not** a geometry win.",
    ]
    if zero.empty:
        lines.append("- No A1 cells were masked.")
    else:
        lines.append("- Masked A1 cells:")
        for row in zero.itertuples(index=False):
            lines.append(f"  - `{_cell_label(row)}`")
    lines += [
        f"- B2 (`val/belief_ang_err_deg`) within 0.25° of 90.15° is treated as chance-level, not a timestep effect. Count among finite B2 cells: **{len(chance_b2)} / {len(b2)}**.",
    ]
    if not chance_b2.empty:
        by_model_b2 = chance_b2.groupby("model").size()
        bits = [f"{MODEL_META[m][0]} {int(n)}" for m, n in by_model_b2.items()]
        lines.append("- Chance-level B2 by model: " + ", ".join(bits) + ".")
        lines.append("- Chance-level B2 cells (prefix 8, every layer and both scheduler indices):")
        for row in chance_b2.itertuples(index=False):
            lines.append(f"  - `{_cell_label(row)}` = {row.metric_value:.4f}°")
    frozen = frame[
        frame.probe.eq("ego_belief_v2")
        & frame.model.isin(["cogvideox", "minimax_h3"])
        & frame.metric_value.notna()
    ]
    if not frozen.empty:
        n_distinct = int(frozen.metric_value.round(6).nunique())
        lines.append(
            f"- CogVideoX and MiniMax-H3 B2 together use **{n_distinct}** distinct values. "
            "The value is constant across layer and across early vs late scheduler index, and the two models log the same constants. "
            "Prefix 8 is 90.146° (chance). Prefixes 12 / 16 / 24 are also frozen (about 80.20° / 73.40° / 70.52°). "
            "None of these cells is a timestep effect."
        )
    c_caps = frame[frame.capability.isin(["C1 Dynamics", "C2 Integration", "C3 Counterfactual"]) & frame.metric_value.notna()]
    if not c_caps.empty:
        lines.append(
            f"- C1–C3 cosine on finite cells spans {c_caps.metric_value.min():.4f}–{c_caps.metric_value.max():.4f}. "
            "Late-minus-early deltas near 1e-4 are saturation noise, not a retrieval result."
        )
    else:
        lines.append("- C1–C3 cosine is expected near 0.995–0.999; late-minus-early deltas near 1e-4 would be saturation noise.")
    lines += [
        "",
        "## Next",
        "",
    ]
    if n_complete == 576 and n_running == 0 and n_partial == 0 and n_missing == 0:
        lines.append("1. The training matrix is closed at 576/576 `TRAIN_COMPLETE`. Do not reopen it with historical non-canonical timestep runs.")
    else:
        lines.append("1. Incomplete cells stay incomplete. Rerun collection only after new `TRAIN_COMPLETE` markers exist; do not backfill from non-canonical runs.")
    lines += [
        "2. A paper C table still needs `eval_diag` `global_R@1` and `intervention_validity`. Training-time cosine is not that table. In-batch R@1 is undefined at batch=2.",
        "3. A paper A1/A2 table that expects `Auc_30` or balanced accuracy still needs a separate eval. Training `val/loss_*` and `val/overlap_acc` are not those metrics.",
        "",
    ]
    (OUT / "FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    configure_style()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    INV.mkdir(parents=True, exist_ok=True)
    summary = json.loads((OUT / "collection_summary.json").read_text())
    frame = load_results()
    frame.to_csv(DATA / "analysis_table.csv", index=False)
    fig_completion(frame)
    rep = representative(frame)
    if not rep.empty:
        rep.to_csv(DATA / "fig01_capability_atlas_data.csv", index=False)
    fig_atlas(rep)
    pairs = paired_timestep(frame)
    if not pairs.empty:
        pairs.to_csv(DATA / "fig02_timestep_delta_data.csv", index=False)
    fig_timestep_delta(pairs)
    fig_layer(frame)
    fig_prefix(frame)
    table_main(rep)
    pair_summary = table_pairs(pairs)
    if not pair_summary.empty:
        pair_summary.to_csv(DATA / "timestep_pair_summary.csv", index=False)
    write_findings(frame, rep, pairs, summary)
    print(f"wrote analysis under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
