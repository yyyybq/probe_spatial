#!/usr/bin/env python3
"""Evaluate one frozen parent as-is without canonical promotion.

This runner is intentionally narrow: it consumes a row from the CPU-produced
``legacy_as_is_loose_checkpoint_inventory.csv`` and evaluates the exact
historical checkpoint against the exact historical manifest/config fields in
that row.  It never discovers a sibling checkpoint, never writes under
``logs/`` or ``results/canonical/``, and writes its only receipt below the
display-only loose namespace.

Known cache/label/window lineage issues are reported as warnings, rather than
being a gate in this explicitly non-comparable evaluation mode.  Broken input,
failed evaluator, an empty evaluation, or a non-finite primary metric remain
hard failures.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "results/analysis/q00_1096_loose/eval_runs/legacy_as_is_loose_checkpoint_inventory.csv"
DEFAULT_OUT = ROOT / "results/analysis/q00_1096_loose/eval_runs"
PYTHON_DEFAULT = Path("/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python")
DATA_ROOT = "aoss://baiqiao/InsScene-15K/data"
PRIMARY_BY_PROBE = {
    "streaming_depth": "mean_loss_depth",
    "streaming_camera": "Auc_30",
    "streaming_mask": "mean_loss_identity",
    "view_consistency": "balanced_acc",
    "ego_belief": "mean_az_err_deg",
    "ego_belief_v2": "mean_ang_err_deg",
    "action_dynamics": "global_R@1",
    "path_integration": "global_R@1",
    "counterfactual": "intervention_validity",
}
PROTECTED = (
    "experiments/registry.csv",
    "results/canonical/a1_results.csv",
    "results/canonical/a2_results.csv",
    "results/canonical/all_results.csv",
    "results/canonical/b2_results.csv",
    "results/canonical/c1_results.csv",
    "results/canonical/c2_results.csv",
    "results/canonical/c3_results.csv",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def under_root(raw: str, *, required: bool = True) -> Path | None:
    if not raw:
        if required:
            raise FileNotFoundError("empty repository-relative path")
        return None
    path = Path(raw)
    if path.is_absolute():
        candidate = path.resolve(strict=False)
    else:
        candidate = (ROOT / path).resolve(strict=False)
    try:
        candidate.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"path escapes repository root: {raw}") from exc
    if required and not candidate.is_file():
        raise FileNotFoundError(f"required file is unavailable: {candidate}")
    return candidate


def read_inventory(
    path: Path,
    universe_id: str,
    *,
    allow_legacy_constant_compat_retry: bool = False,
) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("universe_id") == universe_id]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one inventory row for {universe_id}, got {len(rows)}")
    row = rows[0]
    if row.get("evaluation_mode") != "LEGACY_AS_IS_LOOSE":
        raise ValueError("inventory row has a different evaluation mode")
    status = row.get("preflight_status")
    permitted_compat_retry = (
        allow_legacy_constant_compat_retry
        and status == "HOLD_PREVIOUS_LOOSE_EVAL_FAILURE"
        and row.get("probe") in {"action_dynamics", "path_integration"}
    )
    if status != "READY_LEGACY_AS_IS_LOOSE" and not permitted_compat_retry:
        raise ValueError(f"row is not CPU-ready: {row.get('preflight_status')}")
    if row.get("checkpoint_available", "").lower() != "true":
        raise ValueError("row is not checkpoint-available")
    return row


def validate_legacy_constant_compat_retry(row: dict[str, str]) -> dict[str, str]:
    """Authorize one explicit retry only for the logged two-parameter drift."""
    receipt_path = under_root(row.get("prior_loose_eval_receipt", ""))
    assert receipt_path
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"prior loose receipt is not JSON: {receipt_path}") from exc
    required = {
        "universe_id": row["universe_id"],
        "evaluation_mode": "LEGACY_AS_IS_LOOSE",
        "status": "FAIL",
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("prior loose receipt is not the exact failed legacy evaluation for this row")
    evaluator_log = under_root(str(receipt.get("evaluator_log", "")))
    assert evaluator_log
    traceback_text = evaluator_log.read_text(encoding="utf-8", errors="replace")
    if (
        "Missing key(s) in state_dict" not in traceback_text
        or "probe.constant_context" not in traceback_text
        or "probe.constant_action" not in traceback_text
        or "Unexpected key(s) in state_dict" in traceback_text
    ):
        raise ValueError("prior failure is not the exact two-constant strict-load compatibility case")
    return {
        "prior_receipt": str(receipt_path.relative_to(ROOT)),
        "prior_receipt_sha256": sha256(receipt_path),
    }


def protected_hashes() -> dict[str, str]:
    return {relative: sha256(ROOT / relative) for relative in PROTECTED}


def regular_nonempty(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise RuntimeError(f"{label} is not a readable regular non-empty file: {path}")


def checkpoint_snapshot(path: Path) -> dict[str, Any]:
    regular_nonempty(path, "checkpoint")
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def hydra_escape(value: str) -> str:
    return value.replace("=", r"\=")


def command_for(row: dict[str, str], run_root: Path, python: Path) -> tuple[list[str], dict[str, str], Path, str]:
    family = row["family"]
    evaluator = "vidfm3d/eval_a1_streaming.py" if family == "A1" else "vidfm3d/eval_diag.py"
    if row.get("evaluator") != ("a1" if family == "A1" else "diag"):
        raise ValueError("inventory evaluator/family disagreement")
    probe = row["probe"]
    metric = PRIMARY_BY_PROBE.get(probe)
    if not metric:
        raise ValueError(f"no loose primary metric contract for probe={probe!r}")
    checkpoint = under_root(row["checkpoint"])
    manifest = under_root(row["original_manifest"])
    run_manifest = under_root(row["run_manifest"])
    overrides = under_root(row["overrides"])
    config = under_root(row["hydra_config"])
    assert checkpoint and manifest and run_manifest and overrides and config
    for path, label in ((checkpoint, "checkpoint"), (manifest, "manifest"), (run_manifest, "run_manifest"), (overrides, "overrides"), (config, "hydra config")):
        regular_nonempty(path, label)
    required = ("experiment", "vfm_name", "video_channels", "streaming_feat_root", "feature_layer", "feature_timestep", "prefix_len")
    missing = [name for name in required if not row.get(name)]
    if missing:
        raise ValueError(f"inventory row lacks historical evaluator fields: {missing}")
    if family in {"C1", "C2", "C3"} and not row.get("target_feat_root"):
        raise ValueError("diagnostic action/counterfactual row lacks target feature root")
    try:
        extras = json.loads(row.get("extra_overrides_json") or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("invalid extra_overrides_json") from exc
    if not isinstance(extras, dict):
        raise ValueError("extra_overrides_json must be an object")
    hydra_dir = run_root / "hydra_output"
    loose_name = f"legacy_as_is_loose_{row['universe_id']}"
    cmd = [
        str(python),
        evaluator,
        f"experiment={row['experiment']}",
        f"vfm_name={row['vfm_name']}",
        f"video_channels={row['video_channels']}",
        f"streaming_feat_root={row['streaming_feat_root']}",
        f"feature_layer={row['feature_layer']}",
        f"feature_timestep={row['feature_timestep']}",
        f"prefix_len={row['prefix_len']}",
        f"job_name={loose_name}",
        f"paths.log_dir={run_root / 'logs'}",
        f"paths.run_folder_name={run_root.name}",
        f"hydra.run.dir={hydra_dir}",
        "hydra.output_subdir=.hydra",
        "hydra.job.chdir=false",
        "ckpt_path=" + hydra_escape(str(checkpoint)),
        "+eval_split=val",
        "train=false",
        "test=false",
        "data.data_module.num_workers_val=2",
        "data.data_module.batch_size_per_device_val=4",
        "logger.wandb.offline=true",
        f"logger.wandb.save_dir={run_root / 'wandb'}",
    ]
    if row.get("feat_postfix"):
        cmd.append(f"feat_postfix={row['feat_postfix']}")
    if row.get("target_feat_root"):
        cmd.append(f"++target_feat_root={row['target_feat_root']}")
    if family != "A1":
        cmd.append("++bootstrap_reps=0")
    for key, value in sorted(extras.items()):
        if isinstance(value, (str, int, float, bool)):
            cmd.append(f"{key}={value}")
        else:
            cmd.append(f"{key}={compact(value)}")
    env = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(ROOT),
        "PROJECT_ROOT": str(ROOT),
        "Q00_EVALUATION_MODE": "LEGACY_AS_IS_LOOSE",
        "CUDA_VISIBLE_DEVICES": os.environ["LOOSE_EVAL_GPU"],
        "Q00_REQUIRE_CUDA": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "INSCENE_DATA_ROOT": DATA_ROOT,
        "INSCENE_MANIFEST": str(manifest),
        "INSCENE_STREAMING_FEAT_ROOT": row["streaming_feat_root"],
        "AOSS_CACHE_DIR": str(Path("/mnt/local_cache/inscene/aoss_cache") / "q00_1096_loose" / run_root.parent.name / row["universe_id"]),
    }
    if row.get("target_feat_root"):
        env["INSCENE_TARGET_FEAT_ROOT"] = row["target_feat_root"]
    # Historical action/path checkpoints can predate two parameters added later
    # for unused conditioning ablations.  The evaluator still admits only the
    # exact full-conditioning two-key transition and otherwise keeps strict
    # loading; model name is deliberately not a proxy for that guarded proof.
    if row.get("probe") in {"action_dynamics", "path_integration"}:
        env["Q00_LEGACY_AS_IS_STRICT_CONSTANTS_COMPAT"] = "1"
    return cmd, env, hydra_dir / "eval" / "val_summary.json", metric


def write_terminal(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"terminal receipt already exists: {path}")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def hand_back_to_workspace_owner(path: Path, uid: int, gid: int) -> None:
    """Make direct-node root output readable by the shared workspace client.

    SSH execution on the supplied node is necessarily root, whereas the
    workspace client maps the repository owner to UID/GID 20325.  This touches
    only the just-created loose-evaluation namespace after the evaluator has
    stopped; it does not follow symlinks or touch any source/canonical path.
    """
    if os.geteuid() != 0:
        return
    for current, dirs, files in os.walk(path, topdown=False, followlinks=False):
        base = Path(current)
        for name in files:
            os.lchown(base / name, uid, gid)
        for name in dirs:
            os.lchown(base / name, uid, gid)
    os.lchown(path, uid, gid)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--universe-id", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--gpu", required=True, help="one physical GPU ID exposed to this child")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--python", type=Path, default=PYTHON_DEFAULT)
    parser.add_argument("--owner-uid", type=int, default=20325)
    parser.add_argument("--owner-gid", type=int, default=20325)
    parser.add_argument(
        "--retry-legacy-constant-compat",
        action="store_true",
        help="retry only a prior action/path failure proven to be the two unused legacy constant parameters",
    )
    args = parser.parse_args()
    if not args.run_tag or "/" in args.run_tag or ".." in args.run_tag:
        raise SystemExit("run-tag must be a simple non-empty namespace component")
    if not args.gpu.isascii() or not args.gpu.isdecimal():
        raise SystemExit("gpu must be a non-negative decimal physical GPU ID")
    inventory = args.inventory.resolve(strict=True)
    out_dir = args.out_dir.resolve()
    try:
        out_dir.relative_to(ROOT / "results/analysis/q00_1096_loose/eval_runs")
    except ValueError as exc:
        raise SystemExit("out-dir must remain under results/analysis/q00_1096_loose/eval_runs") from exc
    python = args.python.resolve(strict=True)
    if not python.is_file():
        raise SystemExit(f"Python interpreter is unavailable: {python}")
    row = read_inventory(
        inventory,
        args.universe_id,
        allow_legacy_constant_compat_retry=args.retry_legacy_constant_compat,
    )
    compatibility_retry = (
        validate_legacy_constant_compat_retry(row)
        if args.retry_legacy_constant_compat
        else None
    )
    run_root = out_dir / args.run_tag / args.universe_id
    run_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_root.mkdir(mode=0o770)
    except FileExistsError as exc:
        raise SystemExit(f"refusing to reuse loose-eval namespace: {run_root}") from exc

    receipt_path = run_root / "terminal_receipt.json"
    log_path = run_root / "evaluator.log"
    started = utc_now()
    before_protected = protected_hashes()
    payload: dict[str, Any] = {
        "evaluation_mode": "LEGACY_AS_IS_LOOSE",
        "universe_id": args.universe_id,
        "status": "FAIL",
        "started_at": started,
        "run_tag": args.run_tag,
        "run_root": str(run_root.relative_to(ROOT)),
        "inventory": str(inventory.relative_to(ROOT)),
        "inventory_sha256": sha256(inventory),
        "warning": ";".join(filter(None, [row.get("warning", ""), "LEGACY_AS_IS_LOOSE_NOT_SCIENTIFICALLY_COMPARABLE"])),
        "known_label_feature_mismatch": row.get("known_label_feature_mismatch", "UNKNOWN"),
        "known_manifest_mismatch": row.get("known_manifest_mismatch", "UNKNOWN"),
        "lineage_status": row.get("lineage_status", ""),
        "scientifically_comparable": False,
        "canonical_promotion": False,
        "protected_before": before_protected,
        "metric_name": PRIMARY_BY_PROBE.get(row.get("probe", ""), ""),
        "metric_value": None,
    }
    if compatibility_retry is not None:
        payload["legacy_state_dict_compatibility_retry"] = compatibility_retry
    try:
        os.environ["LOOSE_EVAL_GPU"] = args.gpu
        cmd, child_env, summary_path, target_metric = command_for(row, run_root, python)
        checkpoint = under_root(row["checkpoint"])
        assert checkpoint
        checkpoint_before = checkpoint_snapshot(checkpoint)
        payload.update({
            "evaluator": row["evaluator"],
            "command": cmd,
            "command_sha256": hashlib.sha256("\0".join(cmd).encode()).hexdigest(),
            "checkpoint": checkpoint_before,
            "historical_manifest": str(under_root(row["original_manifest"])),
            "historical_run_manifest": str(under_root(row["run_manifest"])),
            "historical_hydra_config": str(under_root(row["hydra_config"])),
            "historical_overrides": str(under_root(row["overrides"])),
            "feature_roots": {"streaming": row["streaming_feat_root"], "target": row.get("target_feat_root", "")},
            "environment": child_env,
        })
        env = os.environ.copy()
        env.update(child_env)
        with log_path.open("x", encoding="utf-8") as log:
            log.write("COMMAND " + compact(cmd) + "\n")
            log.flush()
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        payload["returncode"] = result.returncode
        payload["evaluator_log"] = str(log_path.relative_to(ROOT))
        if result.returncode != 0:
            raise RuntimeError(f"evaluator returned {result.returncode}")
        regular_nonempty(summary_path, "evaluation summary")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        n = finite(summary.get("n", summary.get("n_samples")))
        if n is None or n <= 0:
            raise RuntimeError(f"evaluator summary has no positive sample count: {summary_path}")
        value = finite(summary.get(target_metric))
        if value is None:
            raise RuntimeError(f"primary metric {target_metric!r} is missing/non-finite in {summary_path}")
        predictions = summary_path.parent / "val_predictions.pt"
        regular_nonempty(predictions, "evaluation predictions")
        checkpoint_after = checkpoint_snapshot(checkpoint)
        if checkpoint_after != checkpoint_before:
            raise RuntimeError("checkpoint changed during loose evaluation")
        after_protected = protected_hashes()
        if after_protected != before_protected:
            raise RuntimeError("protected registry/canonical files changed during loose evaluation")
        payload.update({
            "status": "PASS",
            "metric_name": target_metric,
            "metric_value": value,
            "n": int(n),
            "summary_path": str(summary_path.relative_to(ROOT)),
            "summary_sha256": sha256(summary_path),
            "predictions_path": str(predictions.relative_to(ROOT)),
            "predictions_size_bytes": predictions.stat().st_size,
            "checkpoint_after": checkpoint_after,
            "protected_after": after_protected,
        })
        compatibility = summary.get("legacy_state_dict_compatibility")
        if compatibility is not None:
            payload["legacy_state_dict_compatibility"] = compatibility
            payload["warning"] = ";".join(filter(None, [
                payload.get("warning", ""),
                "LEGACY_STRICT_LOAD_COMPAT_UNUSED_CONSTANT_PARAMETERS",
            ]))
    except Exception as exc:  # receipt is evidence, not a fallback path
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
        payload["protected_after"] = protected_hashes()
        if log_path.exists():
            payload["evaluator_log"] = str(log_path.relative_to(ROOT))
    payload["ended_at"] = utc_now()
    # The receipt must itself be readable by the workspace-side refresher.
    # Hand back all evaluator artifacts first, publish the terminal receipt,
    # then hand back that final file and its parent directory as well.
    hand_back_to_workspace_owner(run_root, args.owner_uid, args.owner_gid)
    write_terminal(receipt_path, payload)
    hand_back_to_workspace_owner(run_root, args.owner_uid, args.owner_gid)
    print(json.dumps({"universe_id": args.universe_id, "status": payload["status"], "receipt": str(receipt_path.relative_to(ROOT))}, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
