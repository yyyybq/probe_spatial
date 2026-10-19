#!/usr/bin/env python3
"""Audit trained probe artifacts and find jobs that still need evaluation."""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "logs/inscene15k_streaming"
RUNS_ROOT = LOG_ROOT / "runs"
EVAL_ROOT = LOG_ROOT / "evaluations"
PROTECTED_ROOT = LOG_ROOT / "checkpoints_protected"

PROBES = (
    "visible_ego_belief_v2",
    "view_consistency",
    "streaming_depth",
    "streaming_mask",
    "ego_belief_v2",
    "ego_belief",
    "action_dynamics",
    "path_integration",
    "counterfactual",
    "abnormal",
)
JOB_RE = re.compile(
    r"(?P<probe>"
    + "|".join(re.escape(p) for p in sorted(PROBES, key=len, reverse=True))
    + r")_(?P<vfm>.+)_p(?P<prefix>\d+)_layer(?P<layer>[-\w.]+)"
)
JOB_END_RE = re.compile(
    r"(?P<probe>"
    + "|".join(re.escape(p) for p in sorted(PROBES, key=len, reverse=True))
    + r")_(?P<vfm>[^/]+?)_p(?P<prefix>\d+)_layer(?P<layer>[-\w.]+)(?:\.log)?$"
)
PROBE_START_RE = re.compile("|".join(re.escape(p) for p in sorted(PROBES, key=len, reverse=True)))
STEP_RE = re.compile(r"step[=_-](\d+)")
MANIFEST_MISSING_PREFIX = "MISSING_ORIGINAL_MANIFEST:"


def tail_text(path: Path, n: int = 4096) -> str:
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - n))
        return f.read().decode("utf-8", errors="replace")


def parse_job_key(text: str) -> dict[str, str] | None:
    parts = re.split(r"[\\/]", str(text))
    match = None
    for part in reversed(parts):
        for start in reversed([m.start() for m in PROBE_START_RE.finditer(part)]):
            maybe = JOB_END_RE.match(part[start:])
            if maybe:
                match = maybe
                break
        if match:
            break
    if match is None:
        matches = list(JOB_RE.finditer(str(text)))
        if not matches:
            return None
        match = matches[-1]
    d = match.groupdict()
    d["job_key"] = f"{d['probe']}_{d['vfm']}_p{d['prefix']}_layer{d['layer']}"
    return d


def read_overrides(run_dir: Path) -> dict[str, str]:
    path = run_dir / ".hydra/overrides.yaml"
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("- ") and "=" in line:
            key, value = line[2:].split("=", 1)
            out[key] = value.strip("'\"")
    return out


def parse_begin_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
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
    except OSError:
        pass
    return out


def parse_training_manifest(run_dir: Path) -> str:
    log_path = run_dir / "inscene15k_streaming.log"
    if not log_path.exists():
        return ""
    pattern = re.compile(r"scene manifest (.+?) for split=(?:val|train)")
    for line in log_path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            return match.group(1)
    return ""


def ckpt_rank(path: Path) -> tuple[float, int, float]:
    metric_match = re.search(r"metric=([0-9.]+)", path.name)
    metric = -float(metric_match.group(1)) if metric_match else 0.0
    step_match = STEP_RE.search(path.name)
    step = int(step_match.group(1)) if step_match else -1
    return metric, step, path.stat().st_mtime


def latest_ckpt(paths: list[Path]) -> Path | None:
    paths = [p for p in paths if p.exists()]
    if not paths:
        return None
    return max(paths, key=ckpt_rank)


def valid_summary(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(errors="replace"))
    except Exception:
        return True
    for key in ("n_samples", "n", "num_samples"):
        if key in data:
            try:
                return int(data[key]) > 0
            except Exception:
                pass
    return True


def eval_paths_by_job_key() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for summary in RUNS_ROOT.glob("*/*/eval/val_summary.json"):
        parsed = parse_job_key(str(summary.parent.parent))
        if parsed and valid_summary(summary):
            out.setdefault(parsed["job_key"], []).append(str(summary))
    for summary in EVAL_ROOT.glob("*/**/val_summary.json"):
        parsed = parse_job_key(str(summary))
        if parsed and valid_summary(summary):
            out.setdefault(parsed["job_key"], []).append(str(summary))
    return out


def status_ok_or_running_by_job_key() -> set[str]:
    out: set[str] = set()
    for status in ROOT.glob("logs/probing_jobs_*eval_20260817/launcher/*_status.jsonl"):
        lines = status.read_text(errors="replace").splitlines()
        events = []
        for line in lines:
            try:
                events.append(json.loads(line))
            except Exception:
                pass
        last_start = max((i for i, e in enumerate(events) if e.get("event") == "shard_start"), default=-1)
        for e in events[last_start + 1 :]:
            if e.get("event") == "job_done" and e.get("rc") == 0 and e.get("key"):
                out.add(e["key"])
    for manifest in ROOT.glob("logs/probing_jobs_*eval_20260817/jobs/*.tsv"):
        if not manifest.exists():
            continue
        with manifest.open(newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row.get("job_key"):
                    out.add(row["job_key"])
    return out


def resolve_scene_manifest(raw_path: str, prefix: str) -> str:
    if not raw_path:
        return ""
    raw = Path(raw_path)
    candidate = raw if raw.is_absolute() else ROOT / raw
    if candidate.exists():
        return str(candidate)
    return f"{MANIFEST_MISSING_PREFIX}{raw_path}"


def manifest_status(scene_manifest: str) -> str:
    if scene_manifest.startswith(MANIFEST_MISSING_PREFIX):
        return "missing_original"
    if scene_manifest:
        return "ok"
    return "missing_unspecified"


def infer_experiment(probe: str, vfm: str) -> str:
    path = ROOT / "configs/experiment/inscene15k_streaming" / f"{probe}_{vfm}_v1.yaml"
    if path.exists():
        return f"inscene15k_streaming/{probe}_{vfm}_v1"
    return ""


def candidate_from_run_dir(run_dir: Path) -> dict[str, str] | None:
    parsed = parse_job_key(run_dir.name)
    if not parsed:
        return None
    ckpt = latest_ckpt(list(run_dir.glob("VideoProbe3D/*/checkpoints/*.ckpt")))
    if not ckpt:
        return None
    overrides = read_overrides(run_dir)
    row = {
        **parsed,
        "source_kind": "run_dir",
        "run_id": run_dir.parent.name,
        "train_run_dir": str(run_dir),
        "ckpt_path": str(ckpt),
        "experiment": overrides.get("experiment", infer_experiment(parsed["probe"], parsed["vfm"])),
        "vfm_name": overrides.get("vfm_name", parsed["vfm"]),
        "video_channels": overrides.get("video_channels", ""),
        "streaming_feat_root": overrides.get("streaming_feat_root", ""),
        "target_feat_root": overrides.get("target_feat_root", ""),
        "feature_layer": overrides.get("feature_layer", parsed["layer"]),
        "feat_postfix": overrides.get("feat_postfix", ""),
        "feature_timestep": overrides.get("feature_timestep", ""),
        "prefix_len": overrides.get("prefix_len", parsed["prefix"]),
        "scene_manifest": resolve_scene_manifest(parse_training_manifest(run_dir), parsed["prefix"]),
        "extra_overrides_json": json.dumps(
            {
                key: value
                for key, value in overrides.items()
                if key.startswith("streaming_hidden_")
                or key in {"action_horizons", "counterfactual_min_overlap"}
            },
            sort_keys=True,
        ),
    }
    return row


def candidates_from_launch_logs() -> list[dict[str, str]]:
    out = []
    for log_path in LOG_ROOT.glob("*/launch/*.log"):
        parsed = parse_job_key(log_path.name)
        if not parsed:
            continue
        try:
            tail = tail_text(log_path)
        except OSError:
            continue
        if "END " not in tail or "rc=0" not in tail:
            continue
        run_id = log_path.parent.parent.name
        run_dir = RUNS_ROOT / run_id / f"{run_id}_{parsed['job_key']}"
        row = candidate_from_run_dir(run_dir)
        if not row:
            continue
        begin_env = parse_begin_env(log_path)
        row["source_kind"] = "launch_success"
        row["launch_log"] = str(log_path)
        for key in ("streaming_feat_root", "target_feat_root"):
            if not row.get(key):
                row[key] = begin_env.get(key, "")
        row["scene_manifest"] = resolve_scene_manifest(
            begin_env.get("scene_manifest", "") or parse_training_manifest(run_dir),
            parsed["prefix"],
        )
        out.append(row)
    return out


def candidates_from_run_dirs() -> list[dict[str, str]]:
    out = []
    for overrides in RUNS_ROOT.glob("*/*/.hydra/overrides.yaml"):
        row = candidate_from_run_dir(overrides.parent.parent)
        if row:
            out.append(row)
    return out


def protected_best_ckpt(job_dir: Path) -> Path | None:
    manifest = job_dir / "checkpoint_manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(errors="replace"))
            best = data.get("global_best", {}).get("path")
            if best and Path(best).exists():
                return Path(best)
        except Exception:
            pass
    return latest_ckpt(list(job_dir.glob("*.ckpt")))


def candidates_from_protected() -> list[dict[str, str]]:
    out = []
    for job_dir in PROTECTED_ROOT.iterdir() if PROTECTED_ROOT.exists() else []:
        if not job_dir.is_dir():
            continue
        parsed = parse_job_key(job_dir.name)
        if not parsed:
            continue
        ckpt = protected_best_ckpt(job_dir)
        if not ckpt:
            continue
        probe = parsed["probe"]
        vfm = parsed["vfm"]
        row = {
            **parsed,
            "source_kind": "checkpoints_protected",
            "run_id": "",
            "train_run_dir": "",
            "ckpt_path": str(ckpt),
            "experiment": infer_experiment(probe, vfm),
            "vfm_name": vfm,
            "video_channels": "3072" if vfm in {"wan", "cogvideox"} else "",
            "streaming_feat_root": "aoss://baiqiao/InsScene-15K/FEAT_STREAMING",
            "target_feat_root": "aoss://baiqiao/InsScene-15K/FEAT_TARGET",
            "feature_layer": parsed["layer"],
            "feat_postfix": f"_layer{parsed['layer']}",
            "feature_timestep": "21" if vfm == "wan" else "49" if vfm == "cogvideox" else "",
            "prefix_len": parsed["prefix"],
            "scene_manifest": "",
            "extra_overrides_json": "{}",
            "launch_log": "",
        }
        out.append(row)
    return out


def dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    priority = {"launch_success": 3, "run_dir": 2, "checkpoints_protected": 1}
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row["job_key"]
        cur = best.get(key)
        if cur is None or priority.get(row["source_kind"], 0) > priority.get(cur["source_kind"], 0):
            best[key] = row
    return sorted(best.values(), key=lambda r: (r["vfm"], r["probe"], int(r["prefix"]), r["layer"]))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    out_dir = ROOT / "logs/eval_gap_audit_20260817"
    eval_map = eval_paths_by_job_key()
    scheduled = status_ok_or_running_by_job_key()
    rows = dedupe(candidates_from_launch_logs() + candidates_from_run_dirs() + candidates_from_protected())
    for row in rows:
        job_key = row["job_key"]
        row["has_eval"] = "1" if job_key in eval_map else "0"
        row["eval_paths"] = ";".join(eval_map.get(job_key, []))
        row["scheduled_or_done_20260817"] = "1" if job_key in scheduled else "0"
        row["manifest_status"] = manifest_status(row.get("scene_manifest", ""))
        row["can_eval"] = "1" if row["experiment"] and row["manifest_status"] == "ok" else "0"
        row["eval_family"] = "a1" if row["probe"] in {"streaming_depth", "streaming_mask"} else "diag"
    missing = [
        row
        for row in rows
        if row["has_eval"] == "0" and row["scheduled_or_done_20260817"] == "0" and row["can_eval"] == "1"
    ]
    fields = [
        "job_key",
        "probe",
        "vfm",
        "prefix",
        "layer",
        "source_kind",
        "run_id",
        "has_eval",
        "scheduled_or_done_20260817",
        "can_eval",
        "manifest_status",
        "eval_family",
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
        "train_run_dir",
        "launch_log",
        "eval_paths",
    ]
    write_csv(out_dir / "all_trained.csv", rows, fields)
    write_csv(out_dir / "missing_eval.csv", missing, fields)
    counts = Counter((row["vfm"], row["probe"], row["eval_family"]) for row in missing)
    lines = [
        "# Full Missing Eval Audit",
        "",
        f"trained_candidates: {len(rows)}",
        f"missing_unscheduled_eval: {len(missing)}",
        "",
        "| vfm | probe | family | missing |",
        "| --- | --- | --- | ---: |",
    ]
    for (vfm, probe, family), count in sorted(counts.items()):
        lines.append(f"| {vfm} | {probe} | {family} | {count} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(rows)} candidates; {len(missing)} missing eval rows to {out_dir}")


if __name__ == "__main__":
    main()
