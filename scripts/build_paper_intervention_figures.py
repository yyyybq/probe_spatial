#!/usr/bin/env python3
"""Build the three paper figures from existing results; no model execution.

Run from the project root: python scripts/build_paper_intervention_figures.py
Requires matplotlib and numpy. All manuscript and evidence paths are relative.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/probe-paper-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D3_SOURCE = Path("experiments/d_series_scannetpp_v1/diagnostics/") / (
    "dseries_vjepa2_d1_d3_three_seed_20260917_zoetrope_r1/across_seed_metrics.csv"
)
ACTION_SOURCE = Path("results/independent_goal_conditioned_continuous_action_p8_v1_20260921/") / (
    "continuous_action_paired_effects.csv"
)
INK = "#483E3A"
MUTED = "#817975"
RULE = "#DAD5D1"
ORANGE = "#D96232"
PALE_ORANGE = "#FBEEE7"
BLUE = "#497F94"
PALE_BLUE = "#EAF2F5"
GREEN = "#648468"
PALE_GREEN = "#EEF3EC"
GRAY = "#96918D"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": INK,
    "axes.edgecolor": RULE, "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white", "mathtext.default": "regular",
})


def load_rows(path):
    with (ROOT / path).open(newline="") as f:
        return list(csv.DictReader(f))


def unique(rows, **query):
    matches = [r for r in rows if all(r[k] == str(v) for k, v in query.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {query}, found {len(matches)}")
    return matches[0]


def text(ax, x, y, value, size=8, **kwargs):
    return ax.text(x, y, value, fontsize=size, va="center", **kwargs)


def box(ax, x, y, w, h, label, fill="white", edge=RULE, size=8, dashed=False):
    patch = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.018", linewidth=0.8,
        facecolor=fill, edgecolor=edge, linestyle="--" if dashed else "-")
    ax.add_patch(patch)
    text(ax, x+w/2, y+h/2, label, size=size, ha="center", linespacing=1.3)
    return patch


def arrow(ax, start, end, color=MUTED, dashed=False, **kwargs):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
        mutation_scale=7, linewidth=0.85, color=color,
        linestyle="--" if dashed else "-", shrinkA=2, shrinkB=2, **kwargs))


def diagram_axis(fig, rect, title, subtitle):
    ax = fig.add_axes(rect)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    text(ax, 0, 0.96, title, size=9.3, weight="bold")
    text(ax, 0, 0.875, subtitle, size=7.5, color=MUTED)
    return ax


def save(fig, out, stem):
    # Fixed canvas dimensions make the three formats interchangeable.
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(out / f"{stem}.{suffix}", dpi=300)
    plt.close(fig)


def intervention_figure(out):
    fig = plt.figure(figsize=(7.0, 2.75))
    a = diagram_axis(fig, [.025, .06, .283, .91],
                     "a  Action use", "Fixed context, predictor and target")
    for y, label, color in [(.70, "Correct action", GREEN),
                            (.52, "Shuffled action", ORANGE),
                            (.34, "Zero action", ORANGE)]:
        box(a, .02, y-.06, .53, .12, label, fill=PALE_GREEN if color == GREEN else PALE_ORANGE)
        arrow(a, (.56, y), (.68, y))
        box(a, .69, y-.06, .12, .12, "$F$", fill=PALE_BLUE)
        arrow(a, (.82, y), (.96, y))
        text(a, .985, y, "$\\hat z$", ha="center", size=9)
    box(a, .02, .10, .53, .12, "Copy last state", fill="#F2F0EE")
    arrow(a, (.56, .16), (.96, .16))
    text(a, .985, .16, "$z_t$", ha="center", size=9)
    text(a, .51, .025, "All outputs scored against the same future", size=7.0, ha="center")

    b = diagram_axis(fig, [.335, .06, .335, .91],
                     "b  History content and order", "Fixed object query; illustrative token slots")
    xs = np.linspace(.48, .94, 5)
    for j, x in enumerate(xs):
        text(b, x, .78, "$t$" if j == 4 else f"$t-{4-j}$", size=7, ha="center")
    variants = [
        ("Full", [0, 1, 2, 3, 4]),
        ("Current only", [None, None, None, None, 4]),
        ("Remove visible", [None, None, 2, 3, 4]),
        ("Zero visual", [None]*5),
        ("Shuffle", [2, 0, 3, 1, 4]),
        ("Reverse", [3, 2, 1, 0, 4]),
    ]
    for y, (name, contents) in zip(np.linspace(.68, .13, 6), variants):
        text(b, 0, y, name, size=7.5)
        for x, token in zip(xs, contents):
            visible = token in (0, 1)
            face = "white" if token is None else PALE_ORANGE if visible else PALE_BLUE
            edge = RULE if token is None else ORANGE if visible else BLUE
            b.add_patch(Rectangle((x-.042, y-.035), .084, .07,
                facecolor=face, edgecolor=edge, linewidth=.7,
                linestyle="--" if token is None else "-"))
            text(b, x, y, "–" if token is None else str(token+1), size=7, ha="center")
    text(b, .5, .025, "Orange tokens: target previously visible", size=7.0, color=ORANGE, ha="center")

    c = diagram_axis(fig, [.70, .06, .28, .91],
                     "c  Transformation order", "Same start state and real endpoint C")
    for y, name, expression, color in [
        (.69, "Sequential", "$T_1\\;\\rightarrow\\;T_2$", BLUE),
        (.46, "Composed", "$T_2 T_1$", BLUE),
        (.23, "Swapped", "$T_2\\;\\rightarrow\\;T_1$", ORANGE),
    ]:
        text(c, .03, y+.09, name, size=7.5, color=color)
        text(c, .04, y, "$s_A$", size=9)
        arrow(c, (.16, y), (.28, y), color=color)
        box(c, .29, y-.055, .46, .11, expression,
            fill=PALE_ORANGE if color == ORANGE else PALE_BLUE, size=9)
        arrow(c, (.76, y), (.94, y), color=color)
        text(c, .97, y, "$\\hat z$", size=9, ha="center")
    text(c, .5, .025, "Compare composition and endpoint errors", size=7, ha="center")
    for x in [.32, .685]:
        fig.add_artist(plt.Line2D([x, x], [.12, .83], transform=fig.transFigure,
                                  color=RULE, linewidth=.65))
    save(fig, out, "fig07_probe_interventions")


def predictive_interface_figure(out):
    rows = load_rows(D3_SOURCE)
    conditions = ["native_target_reference", "frozen_readout_transfer",
                  "retrained_same_capacity_readout", "last_context"]
    colors = [GREEN, ORANGE, BLUE, GRAY]
    labels = ["Real / original", "Pred. / frozen", "Pred. / retrained", "Copy / frozen"]
    fig = plt.figure(figsize=(7.0, 2.85))
    a = diagram_axis(fig, [.025, .16, .315, .81],
                     "a  Shared spatial reader", "Train on real states, then freeze")
    for y, label, fill in [(.72, "Real future state", PALE_GREEN),
                            (.51, "Native prediction\nfrom context", PALE_ORANGE),
                            (.30, "Copied last state", "#F2F0EE")]:
        box(a, .01, y-.067, .51, .134, label, fill=fill, size=7.5)
        arrow(a, (.53, y), (.73, y))
    box(a, .74, .24, .24, .55, "Same\nfrozen\n$g_{\\rm real}$", fill=PALE_BLUE, edge=BLUE, size=8)
    text(a, .5, .16, "Separate, equal-capacity control", size=7, ha="center", color=BLUE)
    box(a, .01, .015, .51, .105, "Native prediction", fill=PALE_ORANGE, size=7.5)
    arrow(a, (.53, .067), (.73, .067), color=BLUE, dashed=True)
    box(a, .74, .015, .24, .105, "$g_{\\rm pred}$", fill="white", edge=BLUE, dashed=True, size=8.5)
    fig.text(.36, .938, "b  Readout error after native prediction", fontsize=9.3, weight="bold")
    data = []
    for i, h in enumerate([2, 4]):
        ax = fig.add_axes([.565+i*.225, .235, .195, .57])
        ax.set_xlim(.17, .315)
        ax.set_ylim(-.55, 3.55)
        ax.set_xticks([.18, .24, .30])
        ax.tick_params(axis="both", labelsize=7.7, length=0)
        ax.tick_params(axis="y", pad=7)
        ax.set_yticks(range(4), labels[::-1] if i == 0 else [""]*4)
        ax.set_title(f"$h={h}$", fontsize=9, pad=8)
        ax.grid(axis="x", color=RULE, linewidth=.55, alpha=.7)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        prior = unique(rows, experiment="D3", reference="native_target", branch="spatial",
                       horizon=h, condition="train_layout_prior")
        prior_value = float(prior["mae_mean_across_seeds"])
        ax.axvline(prior_value, color=MUTED, linestyle=(0, (3, 2)), linewidth=1, zorder=1)
        for j, condition in enumerate(conditions):
            r = unique(rows, experiment="D3", reference="native_target", branch="spatial",
                       horizon=h, condition=condition)
            mean, sd = float(r["mae_mean_across_seeds"]), float(r["mae_sd_across_seeds"])
            assert sd >= 0 and int(r["n_seeds"]) == 3
            ax.errorbar(mean, 3-j, xerr=sd, fmt="o", markersize=5,
                        color=colors[j], elinewidth=1.3, capsize=2.5, zorder=3)
            ax.text(mean, 3-j+.24, f"{mean:.3f}", color=colors[j], ha="center", fontsize=7.2)
            data.append({"horizon": h, "condition": condition, "mean": mean, "seed_sd": sd})
        data.append({"horizon": h, "condition": "train_layout_prior", "mean": prior_value})
    fig.text(.745, .10, "Normalized-depth MAE  ↓", ha="center", fontsize=8)
    fig.add_artist(plt.Line2D([.48, .515], [.037, .037], transform=fig.transFigure,
                              color=MUTED, linestyle=(0, (3, 2)), linewidth=1))
    fig.text(.526, .037, "Train-layout prior", fontsize=7.3, va="center")
    fig.text(.965, .037, "Bars: ±1 seed SD", fontsize=7.3, va="center", ha="right", color=MUTED)
    save(fig, out, "fig08_predictive_interface")
    return data


def goal_conditioning_figure(out):
    rows = load_rows(ACTION_SOURCE)
    models = [("dino", "DINOv2-L"), ("f3r", "Fast3R"), ("vjepa2", "V-JEPA2"), ("wan", "Wan")]
    fig = plt.figure(figsize=(6.8, 2.8))
    panels = [
        ("full_minus_goal_only", "positive_progress", 100., BLUE,
         "a  Current-state contribution", "FULL − GOAL-ONLY", "Positive-progress gain (percentage points)",
         (-3, 24), [0, 5, 10, 15, 20]),
        ("full_minus_current_only", "goal_distance_reduction", 1., ORANGE,
         "b  Goal contribution", "FULL − CURRENT-ONLY", "Goal-distance reduction difference",
         (-.034, .008), [-.03, -.02, -.01, 0]),
    ]
    data = []
    for i, (comparison, metric, factor, color, title, subtitle, xlabel, lim, ticks) in enumerate(panels):
        left = .14 + i*.47
        fig.text(left, .94, title, fontsize=9.3, weight="bold")
        fig.text(left, .855, subtitle, fontsize=8, color=MUTED)
        ax = fig.add_axes([left, .23, .34, .55])
        ax.set(xlim=lim, ylim=(-.5, 3.6), xticks=ticks)
        ax.set_yticks(range(4), [name for _, name in models][::-1] if i == 0 else [""]*4)
        ax.tick_params(axis="both", length=0, labelsize=8, pad=7)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.axvspan(0, lim[1], color=PALE_BLUE, alpha=.65, zorder=0)
        ax.axvline(0, color=MUTED, linewidth=.95, linestyle=(0, (3, 2)), zorder=2)
        ax.grid(axis="x", color=RULE, linewidth=.5, zorder=0)
        for j, (model, name) in enumerate(models):
            r = unique(rows, model=model, comparison=comparison, metric=metric)
            value, low, high = [float(r[key])*factor for key in ("estimate", "ci_low", "ci_high")]
            assert low <= value <= high and int(r["n"]) == 256
            ax.errorbar(value, 3-j, xerr=[[value-low], [high-value]], fmt="o",
                        color=color, markersize=5, elinewidth=1.5, capsize=3, zorder=3)
            ax.text(value, 3-j+.22, f"{value:+.1f}" if factor == 100 else f"{value:+.3f}",
                    ha="center", color=color, fontsize=7.8)
            data.append({"model": model, "comparison": comparison, "metric": metric,
                         "estimate": value, "ci_low": low, "ci_high": high,
                         "display_scale": factor})
        ax.set_xlabel(xlabel, fontsize=7.6, labelpad=9)
    fig.text(.55, .035, "Positive values favor FULL  ·  95% paired scene-bootstrap intervals", ha="center",
             fontsize=7.5, color=MUTED)
    save(fig, out, "fig09_goal_conditioning")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/exp_figures"))
    args = parser.parse_args()
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    intervention_figure(out)
    d3 = predictive_interface_figure(out)
    action = goal_conditioning_figure(out)
    provenance = {
        "sources": [{"path": str(p), "sha256": hashlib.sha256((ROOT/p).read_bytes()).hexdigest()}
                    for p in [D3_SOURCE, ACTION_SOURCE]],
        "fig07": "Conceptual feature-token schematic; five illustrative slots, not a sampled trajectory.",
        "fig08": {"data": d3, "interval": "Three-training-seed sample SD; not a confidence interval."},
        "fig09": {"data": action, "interval": "Paired 95% scene-bootstrap CI, 2000 replicates."},
    }
    (out / "fig07_09_plot_data.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Wrote fig07/08/09 as PDF, SVG and PNG, plus plotted data and source hashes.")


if __name__ == "__main__":
    main()
