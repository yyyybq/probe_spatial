#!/usr/bin/env python3
"""Build SSH eval manifests from the full missing-eval audit."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "logs/eval_gap_audit_20260817"

DIAG_FIELDS = [
    "run_id",
    "job_key",
    "probe",
    "vfm",
    "prefix",
    "layer",
    "train_run_dir",
    "ckpt_path",
    "experiment",
    "vfm_name",
    "video_channels",
    "streaming_feat_root",
    "target_feat_root",
    "feature_layer",
    "feat_postfix",
    "feature_timestep",
    "prefix_len",
    "scene_manifest",
    "extra_overrides_json",
]

A1_FIELDS = DIAG_FIELDS + ["eval_outputs"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def shard_rows(rows: list[dict[str, str]], n_shards: int) -> list[list[dict[str, str]]]:
    shards = [[] for _ in range(n_shards)]
    for index, row in enumerate(rows):
        shards[index % n_shards].append(row)
    return shards


def eval_outputs(row: dict[str, str]) -> str:
    if row["probe"] == "streaming_depth":
        return "depth:mean_loss_depth,camera:Auc_30"
    if row["probe"] == "streaming_mask":
        return "mask:mean_loss_identity"
    return ""


def render_summary(diag_rows: list[dict[str, str]], a1_rows: list[dict[str, str]]) -> str:
    rows = diag_rows + a1_rows
    lines = [
        "# Full Missing Eval Manifest",
        "",
        f"total_jobs: {len(rows)}",
        f"diag_jobs: {len(diag_rows)}",
        f"a1_jobs: {len(a1_rows)}",
        "",
        "| vfm | probe | family | jobs | prefixes | layers |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    counts = Counter((r["vfm"], r["probe"], r["eval_family"]) for r in rows)
    for (vfm, probe, family), count in sorted(counts.items()):
        subset = [r for r in rows if (r["vfm"], r["probe"], r["eval_family"]) == (vfm, probe, family)]
        prefixes = ",".join(sorted({r["prefix"] for r in subset}, key=lambda x: int(x)))
        layers = ",".join(sorted({r["layer"] for r in subset}, key=lambda x: int(x) if x.isdigit() else x))
        lines.append(f"| {vfm} | {probe} | {family} | {count} | {prefixes} | {layers} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-csv", type=Path, default=AUDIT_DIR / "missing_eval.csv")
    parser.add_argument("--diag-out", type=Path, default=ROOT / "logs/probing_jobs_full_missing_eval_20260817")
    parser.add_argument("--a1-out", type=Path, default=ROOT / "logs/probing_jobs_full_missing_a1_eval_20260817")
    parser.add_argument("--diag-shards", type=int, default=2)
    parser.add_argument("--a1-shards", type=int, default=2)
    args = parser.parse_args()

    rows = read_rows(args.audit_csv)
    diag_rows = [r for r in rows if r["eval_family"] == "diag"]
    a1_rows = [dict(r, eval_outputs=eval_outputs(r)) for r in rows if r["eval_family"] == "a1"]

    for row in rows:
        missing = [
            field
            for field in ("job_key", "probe", "vfm", "prefix", "layer", "ckpt_path", "experiment", "streaming_feat_root")
            if not row.get(field)
        ]
        if missing:
            raise SystemExit(f"{row.get('job_key', '<unknown>')} is missing required fields: {missing}")

    write_tsv(args.diag_out / "jobs/eval_jobs.tsv", diag_rows, DIAG_FIELDS)
    write_tsv(args.a1_out / "jobs/eval_jobs.tsv", a1_rows, A1_FIELDS)

    for index, shard in enumerate(shard_rows(diag_rows, args.diag_shards)):
        write_tsv(args.diag_out / f"jobs/shard_{index:02d}.tsv", shard, DIAG_FIELDS)
    for index, shard in enumerate(shard_rows(a1_rows, args.a1_shards)):
        write_tsv(args.a1_out / f"jobs/shard_{index:02d}.tsv", shard, A1_FIELDS)

    summary = render_summary(diag_rows, a1_rows)
    (args.diag_out / "summary.md").write_text(summary)
    (args.a1_out / "summary.md").write_text(summary)
    print(f"wrote {len(diag_rows)} diag jobs to {args.diag_out}")
    print(f"wrote {len(a1_rows)} a1 jobs to {args.a1_out}")


if __name__ == "__main__":
    main()
