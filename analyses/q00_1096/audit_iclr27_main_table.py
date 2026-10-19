#!/usr/bin/env python3
"""Recover the omitted V1 mask column and check the manuscript, without inference.

The six-metric figure package is left unchanged. Outputs document the extended
seven-metric manuscript table and the exact mask configurations behind it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/analysis/q00_1096_loose/all_results_1096_loose.csv"
PACKAGE = ROOT / "results/analysis/q00_1096_paper_no_c3_palette_20260923T000000Z"
SOURCE_SHA = "a1b8958671aeee31a9c17ab58ec0963c79d389bf7e8600198039651746c62b8e"
MODELS = {
    "dino": ("DINO", "DINOv2-L", 0),
    "f3r": ("F3R", "Fast3R", 24),
    "videomae": ("VideoMAE", "VideoMAE-L", 23),
    "vjepa2": ("V-JEPA2", "V-JEPA2-L", 23),
    "qwen2_5_vl": ("Qwen2.5-VL", "Qwen2.5-VL", -1),
    "wan": ("Wan", "Wan", 20),
    "cogvideox": ("CogVideoX", "CogVideoX", 20),
    "aether": ("Aether", "Aether", 1),
    "minimax_h3": ("MiniMax-H3", "MiniMax-H3", 30),
    "lingbot": ("LingBot-video", "LingBot-video", 30),
}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="New directory; existing destinations are rejected")
    args = parser.parse_args()
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    raw = read_csv(SOURCE)
    masks = [r for r in raw if r["probe"] == "streaming_mask"]
    old = {r["Model"]: r for r in read_csv(PACKAGE / "tables/table02_main_results_all.csv")}
    old_fields = list(next(iter(old.values())))[1:]
    mask_fields = ["universe_id", "model", "model_variant", "campaign", "protocol",
                   "probe", "layer", "prefix", "horizon", "seed", "default_timestep",
                   "selected_metric_name", "selected_result", "selected_result_finite",
                   "selected_result_path", "selected_result_candidate_json"]
    records, main_rows, summaries = [], [], []
    default_deltas = []
    for model, (display, tex_name, layer) in MODELS.items():
        campaign = {"lingbot": "lingbot_video_layer_benchmark",
                    "minimax_h3": "minimax_h3_layer_benchmark"}.get(model, "default8")
        selected = sorted([r for r in masks if r["model"] == model
                           and int(r["layer"]) == layer and r["campaign"] == campaign],
                          key=lambda r: int(r["prefix"]))
        assert len(selected) == 4 and {r["prefix"] for r in selected} == {"8", "12", "16", "24"}
        # Never average campaigns, variants, protocols, seeds, horizons or timesteps together.
        identities = {tuple(r[k] for k in ["campaign", "model_variant", "protocol", "seed",
                                           "horizon", "default_timestep"]) for r in selected}
        assert len(identities) == 1
        assert selected[0]["protocol"] == "historical_candidate_v1_val_protocol"
        finite = [r for r in selected if r["selected_result_finite"] == "true"]
        assert finite and all(r["selected_metric_name"] == "mean_loss_identity" for r in finite)
        values = [Decimal(r["selected_result"]) for r in finite]
        assert all(v.is_finite() for v in values)
        mean = sum(values) / len(values)
        mask_text = str(mean.quantize(Decimal(".0001"), rounding=ROUND_HALF_UP))
        row = {"Model": tex_name, "Layer": layer,
               old_fields[0]: old[display][old_fields[0]], "A1 Mask loss ↓": mask_text}
        row.update({key: old[display][key] for key in old_fields[1:]})
        main_rows.append(row)
        records.extend({key: r[key] for key in mask_fields} for r in selected)
        summaries.append({"model": model, "layer": layer, "campaign": campaign,
                          "metric": "mean_loss_identity", "direction": "lower_is_better",
                          "prefixes": ",".join(r["prefix"] for r in finite),
                          "n_numeric": len(finite), "n_registered": len(selected),
                          "raw_mean": str(mean), "display_value": mask_text,
                          "source_universe_ids": ";".join(r["universe_id"] for r in finite)})
        if campaign == "default8":
            by_prefix = {r["prefix"]: Decimal(r["selected_result"]) for r in finite}
            default_deltas.append(by_prefix["24"] - by_prefix["8"])

    text = (ROOT / "iclr27.tex").read_text()
    active = re.sub(r"(?<!\\)%[^\n]*", "", text)
    block = active[active.index(r"\label{tab:main-results}"):active.index(r"\paragraph{Spatial structure separates")]
    tex_rows = {}
    for match in re.finditer(r"\\model\{([^}]+)\}\s*\([^)]*\)(.*?)(?=\\\\)", block, re.S):
        tex_rows[match[1]] = [next(iter(re.findall(r"\d+\.\d+", cell)), "—")
                              for cell in match[2].split("&")[1:]]
    assert len(tex_rows) == 10
    for row in main_rows:
        assert tex_rows[row["Model"]] == list(row.values())[2:], (row, tex_rows[row["Model"]])
    labels = Counter(re.findall(r"\\label\{([^}]+)\}", active))
    refs = {x.strip() for group in re.findall(r"\\(?:ref|cref|Cref|eqref|autoref)\{([^}]+)\}", active)
            for x in group.split(",")}
    assert not refs - labels.keys(), refs - labels.keys()
    assert all(count == 1 for count in labels.values())
    stack = []
    for kind, env in re.findall(r"\\(begin|end)\{([^}]+)\}", active):
        if kind == "begin":
            stack.append(env)
        else:
            assert stack and stack.pop() == env, (kind, env, stack)
    assert not stack
    six = [r for r in read_csv(PACKAGE / "data/fig04_prefix_statistics.csv")
           if r["record_type"] == "model_task_endpoint_status" and r["campaign"] == "default8"]
    p8 = sum(r["direction"] == "p24 < p8" for r in six)
    p24 = sum(r["direction"] == "p24 > p8" for r in six)
    counts = {"six_metric_figure": {"p8_better": p8, "p24_better": p24},
              "seven_metric_table": {"p8_better": p8 + sum(x > 0 for x in default_deltas),
                                      "p24_better": p24 + sum(x < 0 for x in default_deltas)}}
    assert counts["seven_metric_table"] == {"p8_better": 45, "p24_better": 10}
    metadata = {"input": str(SOURCE.relative_to(ROOT)), "input_sha256": SOURCE_SHA,
                "result_version": "V1", "mask_configurations": len(masks),
                "mask_numeric_configurations": sum(r["selected_result_finite"] == "true" for r in masks),
                "selection": "same representative layer/campaign as existing main table; mean of finite prefixes",
                "mask_rounding": "Decimal ROUND_HALF_UP to four places",
                "missing_fill": None, "C3_included": False, "B1_at_main_operating_points": False,
                "main_table_cells_checked": 70, "prefix_endpoint_counts": counts,
                "latex_static_checks": "unique labels, resolved refs, balanced environments",
                "latex_compile": "not performed; compiler/template dependencies unavailable",
                "figure_scope": "Existing six-metric figures and their references unchanged"}
    args.out.mkdir(parents=True, exist_ok=False)
    write_csv(args.out / "main_results_seven_metrics.csv", main_rows)
    write_csv(args.out / "mask_source_configurations.csv", records)
    write_csv(args.out / "mask_aggregation.csv", summaries)
    (args.out / "audit.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
