#!/usr/bin/env python3
"""Summarize reusable P12 feature-geometry evidence without replaying a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    ROOT
    / "results"
    / "paper_validity_20260821T200224Z_2075f3e"
    / "c_baselines_summary.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "analyses" / "probe_validity"
MODELS = ("vjepa2", "videomae", "wan", "cogvideox", "dino")
CORE_MODELS = ("vjepa2", "videomae", "wan", "cogvideox")
LAYER_BY_MODEL = {
    "vjepa2": 23,
    "videomae": 23,
    "wan": 20,
    "cogvideox": 20,
    "dino": 0,
}
REQUIRED_VARIANTS = ("full", "copy_last_context", "batch_roll_shuffle_legacy")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def one(rows: Iterable[Mapping[str, str]], **criteria: str) -> Mapping[str, str]:
    matches = [
        row
        for row in rows
        if all(str(row.get(key, "")) == str(value) for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def fvalue(row: Mapping[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in ("", "nan", "NaN"):
        raise RuntimeError(f"missing numeric {key} in {row}")
    return float(value)


def collect(source: Path = DEFAULT_SOURCE) -> list[dict[str, float | int | str]]:
    rows = read_csv(source)
    output: list[dict[str, float | int | str]] = []
    for model in MODELS:
        selected = {
            variant: one(
                rows,
                task="C1",
                model=model,
                prefix="12",
                variant=variant,
                sample_set="common_models_and_prefixes_legacy_pair",
            )
            for variant in REQUIRED_VARIANTS
        }
        full = selected["full"]
        copy = selected["copy_last_context"]
        shuffle = selected["batch_roll_shuffle_legacy"]
        n = int(fvalue(full, "n"))
        if n != 276 or any(int(fvalue(row, "n")) != n for row in selected.values()):
            raise RuntimeError(f"unexpected P12 sample count for {model}")
        raw_copy_cosine = fvalue(copy, "correct_target_cosine")
        full_r1 = fvalue(full, "global_R@1")
        copy_r1 = fvalue(copy, "global_R@1")
        output.append(
            {
                "model": model,
                "layer": LAYER_BY_MODEL[model],
                "prefix": 12,
                "n": n,
                "raw_copy_target_cosine": raw_copy_cosine,
                "raw_full_target_cosine": fvalue(full, "correct_target_cosine"),
                "temporal_displacement": 1.0 - raw_copy_cosine,
                "centered_copy_target_cosine": fvalue(copy, "centered_cosine"),
                "centered_full_target_cosine": fvalue(full, "centered_cosine"),
                "full_r1": full_r1,
                "copy_r1": copy_r1,
                "shuffle_r1": fvalue(shuffle, "global_R@1"),
                "action_gain_vs_shuffle": full_r1 - fvalue(shuffle, "global_R@1"),
                "full_minus_copy_r1": full_r1 - copy_r1,
            }
        )
    return output


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start
        while end + 1 < len(ordered) and ordered[end + 1][1] == ordered[start][1]:
            end += 1
        rank = (start + end + 2) / 2.0
        for index in range(start, end + 1):
            output[ordered[index][0]] = rank
        start = end + 1
    return output


def pearson(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("pearson requires equally sized vectors with at least two values")
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in x_values))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in y_values))
    return numerator / (x_scale * y_scale)


def correlations(rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows]

    action_gain = values("action_gain_vs_shuffle")
    full_r1 = values("full_r1")
    centered_copy = values("centered_copy_target_cosine")
    return {
        "n_models": float(len(rows)),
        "full_r1_action_gain_pearson": pearson(full_r1, action_gain),
        "full_r1_action_gain_spearman": pearson(ranks(full_r1), ranks(action_gain)),
        "centered_copy_action_gain_pearson": pearson(centered_copy, action_gain),
        "centered_copy_action_gain_spearman": pearson(
            ranks(centered_copy), ranks(action_gain)
        ),
    }


def fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(
    rows: list[dict[str, float | int | str]], source: Path, source_hash: str
) -> str:
    by_model = {str(row["model"]): row for row in rows}
    wan = by_model["wan"]
    dino = by_model["dino"]
    core = [by_model[model] for model in CORE_MODELS]
    corr = correlations(rows)
    table = "\n".join(
        "| {model} L{layer} | {raw_copy} | {raw_full} | {disp} | {copy_centered} | {full_centered} | {copy_r1} | {full_r1} | {shuffle_gain} | {copy_gain} |".format(
            model=row["model"],
            layer=row["layer"],
            raw_copy=fmt(float(row["raw_copy_target_cosine"])),
            raw_full=fmt(float(row["raw_full_target_cosine"])),
            disp=fmt(float(row["temporal_displacement"])),
            copy_centered=fmt(float(row["centered_copy_target_cosine"])),
            full_centered=fmt(float(row["centered_full_target_cosine"])),
            copy_r1=fmt(float(row["copy_r1"])),
            full_r1=fmt(float(row["full_r1"])),
            shuffle_gain=f"{float(row['action_gain_vs_shuffle']):+.4f}",
            copy_gain=f"{float(row['full_minus_copy_r1']):+.4f}",
        )
        for row in rows
    )
    max_core_raw = max(core, key=lambda row: float(row["raw_copy_target_cosine"]))
    return f"""# Representation Geometry / Cosine Illusion Bridge

Status: `DESCRIPTIVE_REUSE_NOT_NEW_CONFIRMATION`

## Scope

This bridge reuses the historical P12 C1 common-set summary, not a new model
run: 276 validation samples from 69 scenes, prefix 12, and the selected default
layer for each model. Its source is `{source.relative_to(ROOT)}`
(`sha256={source_hash}`). The historical summary was produced by
`analyses/paper_validity/run_paper_validity.py`; it uses the same Wan P12/L20
checkpoint and C1 aggregate values as the strict construct-validity audit
(Wan full R@1 0.1522; copy R@1 0.1196). This is development-set, post-hoc
mechanistic context and does not alter any predeclared gate or paper status.

`raw copy cosine` is the mean cosine from the last observed feature to the
future target. `temporal displacement = 1 - raw copy cosine`. `centered`
means the global target mean was subtracted before L2 normalization and cosine
retrieval; it removes a shared feature direction but is not whitening.

| Model | Raw copy cosine | Raw full cosine | Temporal displacement | Centered copy cosine | Centered full cosine | Copy R@1 | Full R@1 | Full - shuffle R@1 | Full - copy R@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

## What The Recovered Record Says

- Wan, not DINO, has the highest P12 raw temporal cosine among the four current
  core models: {fmt(float(max_core_raw['raw_copy_target_cosine']))} for Wan,
  corresponding to only {fmt(float(wan['temporal_displacement']))} mean raw
  cosine displacement. Its centered copy cosine remains {fmt(float(wan['centered_copy_target_cosine']))},
  so this is not solely a global-mean artifact.
- DINO is likely the remembered "highest" result for two different quantities:
  it has the highest P12 full retrieval R@1 ({fmt(float(dino['full_r1']))}) and
  centered copy cosine ({fmt(float(dino['centered_copy_target_cosine']))}). It
  does *not* have the largest raw temporal cosine; Wan does. Centered full
  prediction alignment is instead highest for Wan ({fmt(float(wan['centered_full_target_cosine']))}).
- Wan's raw cosine is close to saturation (copy {fmt(float(wan['raw_copy_target_cosine']))},
  full {fmt(float(wan['raw_full_target_cosine']))}), making raw cosine changes
  intrinsically hard to interpret.
  Yet its retrieval comparison is not a pure-copy tie: full minus copy is
  {float(wan['full_minus_copy_r1']):+.4f} R@1, exactly the strict audit's
  point estimate. The confidence interval crosses zero, so the valid conclusion
  remains weak action-sensitive evidence, not an established non-copy transition.

## Discriminability And Actionability

Using global full R@1 only as a model-level *retrieval-discriminability proxy*
and legacy full-minus-shuffle R@1 as an *actionability proxy*, their correlation
over these five models is Pearson {corr['full_r1_action_gain_pearson']:.4f} and
Spearman {corr['full_r1_action_gain_spearman']:.4f}. Centered copy cosine versus
the same actionability proxy is Pearson {corr['centered_copy_action_gain_pearson']:.4f}
and Spearman {corr['centered_copy_action_gain_spearman']:.4f}. These are only
five non-independent model-level points, share the same readout and candidate
pool, and are visibly influenced by DINO; they are descriptive covariation, not
evidence that feature geometry causes action sensitivity.

## Consequence For Wan's Copy Gate

The recovered record supports a mixed explanation. Wan's feature space is
unusually temporally smooth in raw cosine, which can make an MSE/cosine-trained
predictor look close to the target even when it mostly persists state. However,
centering leaves nonzero state persistence and the action-conditioned predictor
still has a positive, albeit imprecise, retrieval advantage over copy. Thus the
copy-gate failure cannot be reduced either to "no dynamics" or to a cosmetic
cosine artifact. The current evidence supports: `HIGH_TEMPORAL_SMOOTHNESS +
WEAK_UNRESOLVED_INCREMENTAL_ACTION_SIGNAL`.

## Not Available In The Recovered Record

The old result deliberately records `whitened_cosine = NaN`: the raw target
tensors required to fit a train-only whitening transform were not retained in
that packaged analysis. It also does not contain random-pair or same-scene
target-pair cosine distributions, covariance spectra/effective rank, pairwise
prediction separation under matched actions, or matched-action ground-truth
target separation. Those quantities should not be reconstructed from aggregate
cosines. A full geometry study would require the frozen raw target/prediction
tensors, a train-fitted centering/whitening transform, and scene-clustered
uncertainty; it would remain development evidence until an independent split
exists.
"""


def write_outputs(source: Path, output_dir: Path) -> tuple[Path, Path]:
    rows = collect(source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "representation_geometry_legacy_p12.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report_path = output_dir / "REPRESENTATION_GEOMETRY_BRIDGE.md"
    report_path.write_text(render_report(rows, source, source_hash))
    return csv_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    csv_path, report_path = write_outputs(args.source, args.output_dir)
    print(f"wrote {csv_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
