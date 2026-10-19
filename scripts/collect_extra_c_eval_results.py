#!/usr/bin/env python3
"""Merge extra VideoMAE/Qwen C eval summaries into the final C metrics table."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_RESULT_DIR = ROOT / "logs/inscene15k_streaming/results/c_eval_20260724_1903"
BASE_RAW = BASE_RESULT_DIR / "c_series_raw_metrics.csv"
EXTRA_RUN_TAG = "c_eval_extra_videomae_qwen_20260728_0256"
EXTRA_EVAL_ROOT = ROOT / "logs/inscene15k_streaming/runs" / EXTRA_RUN_TAG
OUT_DIR = ROOT / "logs/inscene15k_streaming/results/c_eval_with_extra_20260728"

PROBE_LABEL = {
    "action_dynamics": "C1 action_dynamics",
    "path_integration": "C2 path_integration",
    "counterfactual": "C3 counterfactual",
}
PRIMARY_METRIC = {
    "action_dynamics": "global_R@1",
    "path_integration": "global_R@1",
    "counterfactual": "intervention_validity",
}


def parse_key(key: str) -> tuple[str, str, int, str]:
    for probe in PROBE_LABEL:
        prefix = probe + "_"
        if key.startswith(prefix):
            rem = key[len(prefix) :]
            match = re.match(r"(.+)_p(\d+)_layer(.+)$", rem)
            if not match:
                raise ValueError(f"Cannot parse key: {key}")
            return probe, match.group(1), int(match.group(2)), match.group(3)
    raise ValueError(f"Unknown probe in key: {key}")


def load_base_rows() -> tuple[list[dict[str, str]], list[str]]:
    with BASE_RAW.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def extra_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    summaries = sorted(EXTRA_EVAL_ROOT.rglob("eval/val_summary.json"))
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text())
        job_name = str(summary["job_name"])
        prefix = EXTRA_RUN_TAG + "_"
        key = job_name[len(prefix) :] if job_name.startswith(prefix) else job_name
        probe, vfm, prefix_len, layer = parse_key(key)
        primary_metric = PRIMARY_METRIC[probe]
        row: dict[str, str] = {
            "task": PROBE_LABEL[probe],
            "probe_type": probe,
            "vfm_name": vfm,
            "prefix": str(prefix_len),
            "layer": str(layer),
            "primary_metric": primary_metric,
            "primary_value": str(summary.get(primary_metric, "")),
            "n": str(summary.get("n", "")),
            "source_run": "extra_videomae_qwen",
            "checkpoint_path": str(summary.get("ckpt_path", "")),
            "eval_summary_path": str(summary_path.relative_to(ROOT)),
            "eval_run_dir": str(summary_path.parents[1].relative_to(ROOT)),
            "key": key,
            "train_run_dir": "",
        }
        for key_name, value in summary.items():
            if isinstance(value, (str, int, float)) or value is None:
                row[key_name] = "" if value is None else str(value)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def best_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row.get("probe_type", ""), row.get("vfm_name", ""))
        try:
            value = float(row.get("primary_value", "nan"))
        except ValueError:
            continue
        if key not in best or value > float(best[key].get("primary_value", "nan")):
            best[key] = row
    return [best[key] for key in sorted(best)]


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    headers = ["Probe", "VFM", "Best prefix", "Layer", "Metric", "Value", "n"]
    lines = [
        "# C-series final metrics with VideoMAE/Qwen",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        try:
            value = f"{float(row.get('primary_value', 'nan')):.4f}"
        except ValueError:
            value = row.get("primary_value", "")
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("probe_type", ""),
                    row.get("vfm_name", ""),
                    row.get("prefix", ""),
                    row.get("layer", ""),
                    row.get("primary_metric", ""),
                    value,
                    row.get("n", ""),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    base_rows, base_columns = load_base_rows()
    extras = extra_rows()
    if len(extras) != 24:
        raise SystemExit(f"Expected 24 extra eval summaries, found {len(extras)}")
    rows = base_rows + extras
    columns = list(base_columns)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / "c_series_raw_metrics_with_extra.csv"
    best_path = OUT_DIR / "c_series_best_by_probe_vfm_with_extra.csv"
    md_path = OUT_DIR / "c_series_final_metrics_with_extra.md"
    validation_path = OUT_DIR / "c_series_validation_with_extra.json"

    write_csv(raw_path, rows, columns)
    best = best_rows(rows)
    write_csv(best_path, best, columns)
    write_markdown(md_path, best)
    validation_path.write_text(
        json.dumps(
            {
                "base_rows": len(base_rows),
                "extra_rows": len(extras),
                "total_rows": len(rows),
                "best_rows": len(best),
                "extra_run_tag": EXTRA_RUN_TAG,
                "raw_metrics": str(raw_path),
                "best_metrics": str(best_path),
                "markdown": str(md_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(validation_path.read_text())


if __name__ == "__main__":
    main()
