#!/usr/bin/env python3
"""Check C3-from-C1 predictions against the C1 evaluation of the same checkpoint.

C3 re-runs the frozen C1 predictor from the main tree.  For every candidate in every
C3 action set, the prediction and target must equal the C1 evaluation record (frozen
training pack, v2 evaluator) with the same manifest_sample_id and action horizon.  For
h1-trained checkpoints only the h1 candidate has a C1 counterpart.

Writes results/c3_from_c1_eval/c3/<mode>_c1_parity.json; prints counts.  Read-only on
the evaluation outputs.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch

ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
CAMPAIGN = ROOT / "artifacts/c3_from_c1_eval_20261018_c3"
RUNS = ROOT / "logs/inscene15k_streaming/runs"
SOURCE_RUNS = {"r2": RUNS / "main_v2_eval_v2_canonical_20261017_r2/production",
               "r3": RUNS / "main_v2_eval_v2_canonical_20261017_r3/production",
               "r4": RUNS / "main_v2_eval_v2_canonical_20261017_r4/production",
               "t1": RUNS / "tstep_eval_v2_20261018_t1/production"}


def one(path: Path, name: str) -> Path:
    found = sorted(path.glob(f"**/{name}"))
    if len(found) != 1:
        raise RuntimeError(f"expected one {name} under {path}, found {len(found)}")
    return found[0]


def bits(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    view = {torch.float16: torch.int16, torch.bfloat16: torch.int16, torch.float32: torch.int32,
            torch.float64: torch.int64}.get(t.dtype)
    return t.view(view) if view is not None else t


def compare(c3_records: list, c1_records: list, trained: list[int]) -> dict:
    by_key = {}
    per_sample = Counter(str(r.get("manifest_sample_id")) for r in c1_records)
    for r in c1_records:
        horizon = r.get("action_horizon")
        # Older packs (timestep, MiniMax) do not record the horizon; with one
        # training horizon there is exactly one C1 record per sample.
        if horizon is None and len(trained) == 1 and per_sample[str(r.get("manifest_sample_id"))] == 1:
            horizon = trained[0]
        by_key[(str(r.get("manifest_sample_id")), horizon)] = r
    counts, worst, worst_rel = Counter(), 0.0, 0.0
    for group in c3_records:
        for i, candidate in enumerate(group["candidate_ids"]):
            ref = by_key.get((str(group["manifest_sample_id"]), candidate))
            if ref is None:
                counts["no_c1_counterpart"] += 1
                continue
            counts["matched"] += 1
            pred, target = group["pred"][i], group["target"][i]
            if torch.equal(bits(pred), bits(ref["pred"])):
                counts["pred_identical"] += 1
            else:
                delta = float((pred.double() - ref["pred"].double()).abs().max())
                worst = max(worst, delta)
                # Relative to the prediction scale: MiniMax C1 outputs reach ~1e5.
                worst_rel = max(worst_rel, delta / max(float(ref["pred"].double().abs().max()), 1e-12))
            if torch.equal(bits(target), bits(ref["target"])):
                counts["target_identical"] += 1
    return {**counts, "max_abs_pred_diff": worst, "max_rel_pred_diff": worst_rel,
            "c1_records": len(c1_records), "c3_sets": len(c3_records)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("canary", "production"), required=True)
    args = parser.parse_args()
    plan = json.loads((CAMPAIGN / "plan.json").read_text())["items"]
    rows, totals = {}, Counter()
    for key, entry in sorted(plan.items()):
        out = RUNS / "c3_from_c1_eval_20261018_c3" / args.mode / key
        if not (out / "EVAL_COMPLETE.json").is_file():
            continue
        item = json.loads(Path(entry["path"]).read_text().splitlines()[0])
        source = SOURCE_RUNS[item["source_c1_eval"]["revision"]] / key
        c3 = torch.load(one(out, "val_c3_action_sensitivity_predictions.pt"), map_location="cpu", weights_only=False)
        c1 = torch.load(one(source, "val_predictions.pt"), map_location="cpu", weights_only=False)
        row = compare(c3, c1, item["trained_action_horizons"])
        row["variant"] = item["c3_variant"]
        matched = row.get("matched", 0)
        row["status"] = ("IDENTICAL" if matched and row.get("pred_identical", 0) == matched
                         and row.get("target_identical", 0) == matched else
                         "WITHIN_FLOAT_ROUNDING" if matched and row["max_rel_pred_diff"] <= 1e-6
                         and row.get("target_identical", 0) == matched else "DIFFERENT")
        rows[key] = row
        totals[(item["c3_variant"], row["status"])] += 1
    out_dir = ROOT / "results/c3_from_c1_eval/c3"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.mode}_c1_parity.json").write_text(json.dumps(
        {"mode": args.mode, "totals": {f"{v}|{s}": n for (v, s), n in totals.items()}, "items": rows},
        indent=1, sort_keys=True) + "\n")
    print(json.dumps({f"{v}|{s}": n for (v, s), n in sorted(totals.items())}))
    for key, row in rows.items():
        if row["status"] != "IDENTICAL":
            print(row["status"], key, {k: v for k, v in row.items() if k not in ("status",)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
