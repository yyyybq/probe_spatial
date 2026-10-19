#!/usr/bin/env python3
"""Stage, audit, and atomically promote six Wan p8 clean eval summaries."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LAYERS = [0, 5, 10, 15, 25, 29]
CANONICAL_NAMES = [
    "a1_results.csv",
    "a2_results.csv",
    "all_results.csv",
    "b2_results.csv",
    "c1_results.csv",
    "c2_results.csv",
    "c3_results.csv",
]
UNCHANGED_CANONICAL = {
    "a1_results.csv",
    "a2_results.csv",
    "b2_results.csv",
    "c2_results.csv",
    "c3_results.csv",
}
RUN_IDS = [
    "missing_eval_20260817",
    "missing_a1_eval_20260817",
    "full_missing_eval_20260817",
    "full_missing_a1_eval_20260817",
    "full_missing_a1_eval_retry_ckpt_20260817",
    "remaining_diag_4node_20260817",
    "remaining_a1_4node_20260817",
]
MODELS = ("cogvideox", "lingbot", "minimax_h3", "videomae", "vjepa2", "wan")
JOB_RE = re.compile(
    r"(?P<probe>action_dynamics)_(?P<model>wan)_p(?P<prefix>8)_layer(?P<layer>0|5|10|15|25|29)"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def metadata(path: Path) -> dict[str, Any]:
    return {"path": str(path), "rows": len(read_csv(path)), "sha256": sha256(path)}


def valid_summary(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(errors="replace"))
        n = int(data.get("n", data.get("n_samples", 0)))
        return n > 0
    except Exception:
        return False


def parse_job_key(path: Path, data: dict[str, Any]) -> str:
    text = str(data.get("job_name") or path)
    probes = (
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
    pattern = re.compile(
        r"(?P<probe>" + "|".join(sorted(probes, key=len, reverse=True))
        + r")_(?P<model>" + "|".join(sorted(MODELS, key=len, reverse=True))
        + r")_p(?P<prefix>\d+)_layer(?P<layer>[-\w.]+)"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError(f"cannot parse job key: {path}")
    item = matches[-1].groupdict()
    return f"{item['probe']}_{item['model']}_p{item['prefix']}_layer{item['layer']}"


def eval_inventory(eval_run_tag: str) -> dict[str, Any]:
    entries = []
    for run_id in [*RUN_IDS, eval_run_tag]:
        run_root = ROOT / "logs/inscene15k_streaming/runs" / run_id
        for path in run_root.glob("*/eval/val_summary.json"):
            if not valid_summary(path):
                continue
            data = json.loads(path.read_text(errors="replace"))
            entries.append(
                {
                    "run_id": run_id,
                    "job_key": parse_job_key(path, data),
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                    "n": int(data.get("n", data.get("n_samples", 0))),
                }
            )
    counts = Counter(item["job_key"] for item in entries)
    duplicates = {key: count for key, count in counts.items() if count > 1}
    expected_new = {f"action_dynamics_wan_p8_layer{layer}" for layer in LAYERS}
    actual_new = {item["job_key"] for item in entries if item["run_id"] == eval_run_tag}
    return {
        "valid_summaries": len(entries),
        "unique_job_keys": len(counts),
        "duplicate_job_keys": duplicates,
        "expected_new_job_keys": sorted(expected_new),
        "actual_new_job_keys": sorted(actual_new),
        "entries": entries,
        "ok": len(entries) == 428 and len(counts) == 428 and not duplicates and actual_new == expected_new,
    }


def expected_registry_ids() -> set[str]:
    return {
        f"c1_action_dynamics_wan_eval_20260817_l{layer}_p8_hall_s42"
        for layer in LAYERS
    }


def rows_by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    ids = [row.get("experiment_id", "") for row in rows]
    duplicates = [key for key, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate experiment IDs: {duplicates[:20]}")
    return {row["experiment_id"]: row for row in rows}


def complete_metric_duplicates(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    keys = []
    for row in rows:
        if row.get("status") != "COMPLETE":
            continue
        keys.append(
            (
                row.get("family", ""), row.get("probe", ""), row.get("model", ""),
                row.get("model_variant", ""), row.get("layer", ""), row.get("prefix", ""),
                row.get("horizon", ""), row.get("seed", ""), row.get("eval_split", ""),
                row.get("metric_name", ""),
            )
        )
    return [
        {"key": list(key), "count": count}
        for key, count in Counter(keys).items()
        if count > 1
    ]


def validate_staging(pre_registry: Path, stage_registry: Path, stage_canonical: Path) -> dict[str, Any]:
    pre_rows = read_csv(pre_registry)
    stage_rows = read_csv(stage_registry)
    pre_by_id = rows_by_id(pre_rows)
    stage_by_id = rows_by_id(stage_rows)
    added = set(stage_by_id) - set(pre_by_id)
    removed = set(pre_by_id) - set(stage_by_id)
    changed_old = [key for key in pre_by_id if stage_by_id.get(key) != pre_by_id[key]]
    metric_duplicates = complete_metric_duplicates(stage_rows)
    expected = expected_registry_ids()
    canonical_checks: dict[str, Any] = {}
    for name in CANONICAL_NAMES:
        before = ROOT / "results/canonical" / name
        after = stage_canonical / name
        before_rows = read_csv(before)
        after_rows = read_csv(after)
        byte_unchanged = sha256(before) == sha256(after)
        if name in UNCHANGED_CANONICAL:
            added_ids: set[str] = set()
            removed_ids: set[str] = set()
            changed_ids: list[str] = []
            ok = byte_unchanged and len(before_rows) == len(after_rows)
        else:
            before_by_id = rows_by_id(before_rows)
            after_by_id = rows_by_id(after_rows)
            added_ids = set(after_by_id) - set(before_by_id)
            removed_ids = set(before_by_id) - set(after_by_id)
            changed_ids = [key for key in before_by_id if after_by_id.get(key) != before_by_id[key]]
            ok = added_ids == expected and not removed_ids and not changed_ids
        canonical_checks[name] = {
            "ok": ok,
            "before_rows": len(before_rows),
            "after_rows": len(after_rows),
            "before_sha256": sha256(before),
            "after_sha256": sha256(after),
            "added_ids": sorted(added_ids),
            "removed_ids": sorted(removed_ids),
            "changed_existing_ids": sorted(changed_ids),
            "byte_unchanged": byte_unchanged,
        }
    registry_ok = (
        added == expected
        and not removed
        and not changed_old
        and len(stage_rows) == len(pre_rows) + 6
        and not metric_duplicates
    )
    return {
        "ok": registry_ok and all(item["ok"] for item in canonical_checks.values()),
        "registry": {
            "ok": registry_ok,
            "before_rows": len(pre_rows),
            "after_rows": len(stage_rows),
            "added_ids": sorted(added),
            "removed_ids": sorted(removed),
            "changed_existing_ids": sorted(changed_old),
            "duplicate_metric_rows": metric_duplicates,
        },
        "canonical": canonical_checks,
    }


def write_diff(before: Path, after: Path, out: Path) -> None:
    diff = difflib.unified_diff(
        before.read_text(errors="replace").splitlines(keepends=True),
        after.read_text(errors="replace").splitlines(keepends=True),
        fromfile=str(before),
        tofile=str(after),
    )
    out.write_text("".join(diff))


def atomic_copy(source: Path, destination: Path) -> None:
    tmp = destination.with_name(f".{destination.name}.wan-p8-promote-{os.getpid()}")
    shutil.copy2(source, tmp)
    os.replace(tmp, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--eval-run-tag", required=True)
    parser.add_argument("--eval-audit", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--update-root")
    args = parser.parse_args()

    eval_audit = json.loads(Path(args.eval_audit).read_text())
    if eval_audit.get("ok") is not True:
        raise SystemExit("eval audit is not PASS")
    report_dir = Path(args.report_dir).resolve()
    frozen_commit_path = report_dir / "provenance/git_commit.txt"
    frozen_commit = frozen_commit_path.read_text().strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", frozen_commit):
        raise SystemExit(f"invalid frozen git commit: {frozen_commit_path}")
    update_root = (
        Path(args.update_root).resolve()
        if args.update_root
        else report_dir / "registry_update"
    )
    try:
        update_root.relative_to(report_dir)
    except ValueError as exc:
        raise SystemExit("update root must be inside report dir") from exc
    if update_root.exists():
        raise SystemExit(f"refusing existing registry update staging: {update_root}")
    pre_dir = update_root / "pre"
    stage_dir = update_root / "staging"
    post_dir = update_root / "post"
    pre_dir.mkdir(parents=True)
    stage_dir.mkdir(parents=True)
    post_dir.mkdir(parents=True)

    registry = ROOT / "experiments/registry.csv"
    canonical = ROOT / "results/canonical"
    pre_registry = pre_dir / "registry.csv"
    shutil.copy2(registry, pre_registry)
    for name in CANONICAL_NAMES:
        shutil.copy2(canonical / name, pre_dir / name)
    before_meta = {
        "registry": metadata(registry),
        "canonical": {name: metadata(canonical / name) for name in CANONICAL_NAMES},
    }
    (pre_dir / "metadata.json").write_text(json.dumps(before_meta, indent=2, sort_keys=True) + "\n")

    stage_registry = stage_dir / "registry.csv"
    stage_canonical = stage_dir / "canonical"
    shutil.copy2(registry, stage_registry)
    env = os.environ.copy()
    env.update(
        {
            "EXTRA_EVAL_RUN_IDS": args.eval_run_tag,
            "EVAL_REGISTRY_PATH": str(stage_registry),
            "EVAL_CANONICAL_DIR": str(stage_canonical),
            "EVAL_GIT_COMMIT_OVERRIDE": frozen_commit,
        }
    )
    proc = subprocess.run(
        [sys.executable, "scripts/update_registry_with_eval_summaries.py"],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (stage_dir / "official_updater.log").write_text(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit(f"official staging updater failed rc={proc.returncode}")

    inventory = eval_inventory(args.eval_run_tag)
    audit = validate_staging(pre_registry, stage_registry, stage_canonical)
    audit["eval_inventory"] = inventory
    audit["ok"] = audit["ok"] and inventory["ok"]
    (stage_dir / "staging_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    write_diff(pre_registry, stage_registry, stage_dir / "registry.diff")
    for name in CANONICAL_NAMES:
        write_diff(pre_dir / name, stage_canonical / name, stage_dir / f"{name}.diff")
    if not audit["ok"]:
        raise SystemExit("staging audit failed; canonical files were not modified")

    atomic_copy(stage_registry, registry)
    for name in CANONICAL_NAMES:
        atomic_copy(stage_canonical / name, canonical / name)

    consistency = subprocess.run(
        [sys.executable, "scripts/check_project_consistency.py"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (post_dir / "project_consistency.log").write_text(consistency.stdout)
    if consistency.returncode != 0:
        raise SystemExit("post-promotion project consistency failed")

    post_meta = {
        "registry": metadata(registry),
        "canonical": {name: metadata(canonical / name) for name in CANONICAL_NAMES},
        "valid_eval_summaries": inventory["valid_summaries"],
        "unique_eval_job_keys": inventory["unique_job_keys"],
        "missing_wan_p8_evals": 0,
        "duplicate_experiment_ids": 0,
        "duplicate_metric_rows": 0,
        "canonical_results_valid": True,
    }
    (post_dir / "metadata.json").write_text(json.dumps(post_meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "valid_eval_summaries": 428, "registry_rows": post_meta["registry"]["rows"]}))


if __name__ == "__main__":
    main()
