#!/usr/bin/env python3
"""Summarize C-series results with prefix-curve robustness metrics.

Reports, for every probe × VFM:
  - prefix curve (p8/p12/p16/p24)
  - prefix AUC (trapezoid over prefix length)
  - p24 − p8 improvement
  - best prefix / best value
  - prefix sensitivity (max−min and std across prefixes)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


PRIMARY_METRIC = {
    "action_dynamics": "global_R@1",
    "path_integration": "global_R@1",
    "counterfactual": "intervention_validity",
}
PREFIXES = [8, 12, 16, 24]


def trapz_auc(xs: list[int], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    area = 0.0
    for i in range(1, len(xs)):
        area += (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2.0
    # Normalize by prefix span so AUC is comparable to metric scale.
    span = xs[-1] - xs[0]
    return area / span if span else float("nan")


def fmt(x: float, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:.{digits}f}"


def load_rows(raw_csv: Path) -> list[dict[str, str]]:
    with raw_csv.open(newline="") as f:
        return list(csv.DictReader(f))


def group_curves(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[int, dict]]:
    grouped: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for row in rows:
        probe = row["probe_type"]
        vfm = row["vfm_name"]
        prefix = int(row["prefix"])
        metric = PRIMARY_METRIC[probe]
        value_s = row.get("primary_value") or row.get(metric) or ""
        if value_s == "":
            continue
        value = float(value_s)
        prev = grouped[(probe, vfm)].get(prefix)
        # If duplicates exist, keep the higher primary value.
        if prev is None or value > prev["value"]:
            grouped[(probe, vfm)][prefix] = {
                "value": value,
                "layer": row.get("layer", ""),
                "n": row.get("n", ""),
                "metric": metric,
                "key": row.get("key", ""),
            }
    return grouped


def summarize(grouped: dict[tuple[str, str], dict[int, dict]]) -> list[dict]:
    out: list[dict] = []
    for (probe, vfm), by_prefix in sorted(grouped.items()):
        xs = [p for p in PREFIXES if p in by_prefix]
        if not xs:
            continue
        ys = [by_prefix[p]["value"] for p in xs]
        best_i = max(range(len(xs)), key=lambda i: ys[i])
        best_prefix = xs[best_i]
        best_value = ys[best_i]
        p8 = by_prefix.get(8, {}).get("value")
        p24 = by_prefix.get(24, {}).get("value")
        improvement = (
            (p24 - p8) if (p8 is not None and p24 is not None) else float("nan")
        )
        sens_range = max(ys) - min(ys)
        sens_std = statistics.pstdev(ys) if len(ys) > 1 else 0.0
        auc = trapz_auc(xs, ys)
        row = {
            "probe": probe,
            "vfm": vfm,
            "metric": PRIMARY_METRIC[probe],
            "layer": by_prefix[best_prefix]["layer"],
            "n": by_prefix[best_prefix]["n"],
            "p8": by_prefix.get(8, {}).get("value"),
            "p12": by_prefix.get(12, {}).get("value"),
            "p16": by_prefix.get(16, {}).get("value"),
            "p24": by_prefix.get(24, {}).get("value"),
            "prefix_auc": auc,
            "p24_minus_p8": improvement,
            "best_prefix": best_prefix,
            "best_value": best_value,
            "prefix_range": sens_range,
            "prefix_std": sens_std,
            "num_prefixes": len(xs),
        }
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    cols = [
        "probe",
        "vfm",
        "metric",
        "layer",
        "n",
        "p8",
        "p12",
        "p16",
        "p24",
        "prefix_auc",
        "p24_minus_p8",
        "best_prefix",
        "best_value",
        "prefix_range",
        "prefix_std",
        "num_prefixes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in cols})


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# C-series prefix robustness report",
        "",
        "Best-prefix selection alone introduces optimistic selection bias. "
        "This report also reports the full prefix curve, AUC, p24−p8 "
        "improvement, and prefix sensitivity.",
        "",
        "## Definitions",
        "",
        "- **prefix curve**: primary metric at prefixes `{8,12,16,24}`",
        "- **prefix AUC**: trapezoidal integral over prefix length, normalized by `(24-8)`",
        "- **p24−p8**: improvement from shortest to longest prefix (positive = longer context helps)",
        "- **best prefix / best value**: argmax over available prefixes (kept for completeness)",
        "- **prefix sensitivity**: `range = max−min`, `std` across the four prefixes",
        "",
        "Primary metrics: C1/C2 = `global_R@1` (↑), C3 = `intervention_validity` (↑).",
        "",
    ]

    for probe in ["action_dynamics", "path_integration", "counterfactual"]:
        subset = [r for r in rows if r["probe"] == probe]
        if not subset:
            continue
        metric = PRIMARY_METRIC[probe]
        lines.extend(
            [
                f"## {probe} (`{metric}` ↑)",
                "",
                "| VFM | p8 | p12 | p16 | p24 | AUC | p24−p8 | Best p | Best | Range | Std |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for r in sorted(subset, key=lambda x: (-(x["prefix_auc"] or -1e9), x["vfm"])):
            lines.append(
                "| "
                + " | ".join(
                    [
                        r["vfm"],
                        fmt(r["p8"]),
                        fmt(r["p12"]),
                        fmt(r["p16"]),
                        fmt(r["p24"]),
                        fmt(r["prefix_auc"]),
                        fmt(r["p24_minus_p8"]),
                        str(r["best_prefix"]),
                        fmt(r["best_value"]),
                        fmt(r["prefix_range"]),
                        fmt(r["prefix_std"]),
                    ]
                )
                + " |"
            )
        lines.append("")

        # Compact ranking notes
        by_auc = sorted(subset, key=lambda x: x["prefix_auc"], reverse=True)
        by_imp = sorted(
            [r for r in subset if r["p24_minus_p8"] == r["p24_minus_p8"]],
            key=lambda x: x["p24_minus_p8"],
            reverse=True,
        )
        by_sens = sorted(subset, key=lambda x: x["prefix_range"])
        lines.extend(
            [
                f"**AUC ranking:** "
                + ", ".join(f"{r['vfm']}={fmt(r['prefix_auc'])}" for r in by_auc),
                "",
                f"**p24−p8 ranking:** "
                + ", ".join(f"{r['vfm']}={fmt(r['p24_minus_p8'])}" for r in by_imp),
                "",
                f"**Most stable (lowest range):** "
                + ", ".join(f"{r['vfm']}={fmt(r['prefix_range'])}" for r in by_sens[:3]),
                "",
            ]
        )

    lines.extend(
        [
            "## Caveats",
            "",
            "- Best-prefix remains useful as an upper envelope, but AUC and p24−p8 are preferred for cross-model comparison.",
            "- High `prefix_range`/`prefix_std` means the model is sensitive to context length; a single best-prefix number is less trustworthy.",
            "- Negative `p24−p8` means longer prefixes hurt this probe.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-csv",
        default="logs/inscene15k_streaming/results/c_eval_with_extra_20260728/c_series_raw_metrics_with_extra.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="logs/inscene15k_streaming/results/c_eval_with_extra_20260728",
    )
    args = parser.parse_args()

    raw_csv = Path(args.raw_csv)
    out_dir = Path(args.out_dir)
    rows = load_rows(raw_csv)
    summary = summarize(group_curves(rows))

    csv_path = out_dir / "c_series_prefix_robustness.csv"
    md_path = out_dir / "c_series_prefix_robustness.md"
    meta_path = out_dir / "c_series_prefix_robustness.json"

    write_csv(csv_path, summary)
    write_markdown(md_path, summary)
    meta_path.write_text(
        json.dumps(
            {
                "source_raw_csv": str(raw_csv),
                "num_groups": len(summary),
                "prefixes": PREFIXES,
                "primary_metric": PRIMARY_METRIC,
                "outputs": {
                    "csv": str(csv_path),
                    "markdown": str(md_path),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps({"csv": str(csv_path), "markdown": str(md_path), "groups": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
