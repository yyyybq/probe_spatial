#!/usr/bin/env python3
"""Build a manifest for trained A1 streaming probe jobs that still need eval."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "logs/inscene15k_streaming/runs"

SUPPORTED_PROBES = ("streaming_depth", "streaming_mask")
PROBE_RE = re.compile(
    r"(?P<probe>"
    + "|".join(re.escape(p) for p in sorted(SUPPORTED_PROBES, key=len, reverse=True))
    + r")_(?P<vfm>.+)_p(?P<prefix>\d+)_layer(?P<layer>[-\w.]+)\.log$"
)
STEP_RE = re.compile(r"step=(\d+)")

DEFAULT_CAMPAIGNS = (
    "minimax_h3_abc_retry_20260812_1848",
    "minimax_h3_a1_pose_mask_20260812_1905",
    "vjepa2_videomae_a1_mask_20260812_1909",
    "vjepa2_videomae_abc_retry6_20260813_0249",
)


def tail_text(path: Path, n: int = 4096) -> str:
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - n))
        return f.read().decode("utf-8", errors="replace")


def parse_log_name(path: Path) -> dict[str, str] | None:
    match = PROBE_RE.search(path.name)
    if not match:
        return None
    return match.groupdict()


def parse_begin_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("STREAM_ROOT="):
                out["streaming_feat_root"] = line.split("=", 1)[1]
            elif line.startswith("TARGET_ROOT="):
                out["target_feat_root"] = line.split("=", 1)[1]
            elif line.startswith("MANIFEST="):
                out["scene_manifest"] = line.split("=", 1)[1]
            if len(out) >= 3:
                break
    return out


def read_overrides(run_dir: Path) -> dict[str, str]:
    path = run_dir / ".hydra/overrides.yaml"
    overrides: dict[str, str] = {}
    if not path.exists():
        return overrides
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("- ") and "=" in line:
            key, value = line[2:].split("=", 1)
            overrides[key] = value
    return overrides


def checkpoint_step(path: Path) -> tuple[int, float]:
    match = STEP_RE.search(path.name)
    step = int(match.group(1)) if match else -1
    return step, path.stat().st_mtime


def latest_ckpt(run_dir: Path) -> Path | None:
    ckpts = sorted(run_dir.glob("VideoProbe3D/*/checkpoints/*.ckpt"))
    if not ckpts:
        return None
    return max(ckpts, key=checkpoint_step)


def eval_exists(job_key: str, run_id: str, out_run_id: str) -> bool:
    exact_paths = [
        RUNS_ROOT / out_run_id / f"{out_run_id}_{job_key}" / "eval/val_summary.json",
        RUNS_ROOT / run_id / f"{run_id}_{job_key}" / "eval/val_summary.json",
    ]
    for path in exact_paths:
        if not path.exists():
            continue
        try:
            summary = json.loads(path.read_text())
            if int(summary.get("n_samples", summary.get("n", 0))) > 0:
                return True
        except Exception:
            return True
    return False


def eval_outputs(probe: str) -> str:
    if probe == "streaming_depth":
        return "depth:mean_loss_depth,camera:Auc_30"
    if probe == "streaming_mask":
        return "mask:mean_loss_identity"
    return ""


def resolve_scene_manifest(raw_path: str, prefix: str) -> str:
    if not raw_path:
        return ""
    raw = Path(raw_path)
    candidate = raw if raw.is_absolute() else ROOT / raw
    return str(candidate) if candidate.exists() else ""


def build_jobs(campaigns: list[str], out_run_id: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for run_id in campaigns:
        launch_dir = ROOT / "logs/inscene15k_streaming" / run_id / "launch"
        if not launch_dir.exists():
            skipped.append({"run_id": run_id, "reason": "missing_launch_dir"})
            continue
        for log_path in sorted(launch_dir.glob("*.log")):
            parsed = parse_log_name(log_path)
            if not parsed:
                continue
            tail = tail_text(log_path)
            job_key = f"{parsed['probe']}_{parsed['vfm']}_p{parsed['prefix']}_layer{parsed['layer']}"
            if "END " not in tail or "rc=0" not in tail:
                skipped.append({"run_id": run_id, "job_key": job_key, "reason": "not_successful_training"})
                continue
            if eval_exists(job_key, run_id, out_run_id):
                skipped.append({"run_id": run_id, "job_key": job_key, "reason": "eval_summary_exists"})
                continue

            train_run_dir = RUNS_ROOT / run_id / f"{run_id}_{job_key}"
            if not train_run_dir.exists():
                skipped.append({"run_id": run_id, "job_key": job_key, "reason": "missing_train_run_dir"})
                continue
            ckpt = latest_ckpt(train_run_dir)
            if ckpt is None:
                skipped.append({"run_id": run_id, "job_key": job_key, "reason": "missing_checkpoint"})
                continue

            overrides = read_overrides(train_run_dir)
            begin_env = parse_begin_env(log_path)
            scene_manifest = resolve_scene_manifest(
                begin_env.get("scene_manifest", ""),
                parsed["prefix"],
            )
            if not scene_manifest:
                skipped.append(
                    {
                        "run_id": run_id,
                        "job_key": job_key,
                        "reason": "missing_original_manifest",
                        "scene_manifest": begin_env.get("scene_manifest", ""),
                    }
                )
                continue
            experiment = overrides.get(
                "experiment",
                f"inscene15k_streaming/{parsed['probe']}_{parsed['vfm']}_v1",
            )
            row = {
                "run_id": run_id,
                "job_key": job_key,
                "probe": parsed["probe"],
                "eval_outputs": eval_outputs(parsed["probe"]),
                "vfm": parsed["vfm"],
                "prefix": parsed["prefix"],
                "layer": parsed["layer"],
                "train_run_dir": str(train_run_dir),
                "ckpt_path": str(ckpt.resolve()),
                "experiment": experiment,
                "vfm_name": overrides.get("vfm_name", parsed["vfm"]),
                "video_channels": overrides.get("video_channels", ""),
                "streaming_feat_root": overrides.get(
                    "streaming_feat_root", begin_env.get("streaming_feat_root", "")
                ),
                "feature_layer": overrides.get("feature_layer", parsed["layer"]),
                "feat_postfix": overrides.get("feat_postfix", ""),
                "feature_timestep": overrides.get("feature_timestep", ""),
                "prefix_len": overrides.get("prefix_len", parsed["prefix"]),
                "scene_manifest": scene_manifest,
                "extra_overrides_json": json.dumps(
                    {
                        key: value
                        for key, value in overrides.items()
                        if key.startswith("streaming_hidden_")
                    },
                    sort_keys=True,
                ),
            }
            missing_required = [
                key
                for key in ("video_channels", "streaming_feat_root", "feature_timestep")
                if not row[key]
            ]
            if missing_required:
                skipped.append(
                    {
                        "run_id": run_id,
                        "job_key": job_key,
                        "reason": "missing_required_fields",
                        "fields": ",".join(missing_required),
                    }
                )
                continue
            rows.append(row)
    rows.sort(key=lambda r: (r["vfm"], r["probe"], int(r["prefix"]), r["layer"], r["run_id"]))
    return rows, skipped


def write_tsv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(ROOT / "logs/probing_jobs_missing_a1_eval_20260817"))
    parser.add_argument("--run-id", default="missing_a1_eval_20260817")
    parser.add_argument("--campaign", action="append", dest="campaigns")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    campaigns = args.campaigns or list(DEFAULT_CAMPAIGNS)
    rows, skipped = build_jobs(campaigns, args.run_id)
    columns = [
        "run_id",
        "job_key",
        "probe",
        "eval_outputs",
        "vfm",
        "prefix",
        "layer",
        "train_run_dir",
        "ckpt_path",
        "experiment",
        "vfm_name",
        "video_channels",
        "streaming_feat_root",
        "feature_layer",
        "feat_postfix",
        "feature_timestep",
        "prefix_len",
        "scene_manifest",
        "extra_overrides_json",
    ]
    write_tsv(out_dir / "jobs/eval_jobs.tsv", rows, columns)
    write_tsv(
        out_dir / "jobs/skipped.tsv",
        skipped,
        ["run_id", "job_key", "reason", "fields"],
    )

    counts = Counter((row["vfm"], row["probe"]) for row in rows)
    lines = [
        "# Missing A1 Eval Jobs",
        "",
        f"run_id: `{args.run_id}`",
        f"jobs: {len(rows)}",
        "",
        "| vfm | probe | eval outputs | jobs |",
        "| --- | --- | --- | ---: |",
    ]
    for (vfm, probe), count in sorted(counts.items()):
        lines.append(f"| {vfm} | {probe} | {eval_outputs(probe)} | {count} |")
    lines.extend(["", "Manifest: `jobs/eval_jobs.tsv`", "Skipped: `jobs/skipped.tsv`", ""])
    (out_dir / "README.md").write_text("\n".join(lines))
    print(f"wrote {len(rows)} A1 eval jobs to {out_dir / 'jobs/eval_jobs.tsv'}")


if __name__ == "__main__":
    main()
