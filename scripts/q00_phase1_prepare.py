#!/usr/bin/env python3
"""Freeze exact Q00 phase-1 inputs and build the 99-row Wave-B eval manifest.

This script is intentionally preparation-only.  It never starts evaluation and
never writes registry/canonical files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = ROOT / "experiments/task_status_1096_before_expand_20260822.csv"
RECONCILIATION = ROOT / "results/analysis/q00_1096/universe_1096_reconciled.csv"
WAVES = ROOT / "results/analysis/q00_1096/execution_waves.csv"
Q00_REPORT = ROOT / "reports/q00_universe_reconciliation_1096.md"
PROTOCOL = "historical_candidate_v1_val_protocol"
EXPECTED_UNIVERSE_SHA = "9c7d6807b701de4b052819971a5c011e29c717e1c3e958ec9ae23116cdcfe779"
STREAM_ROOT = "aoss://baiqiao/InsScene-15K/FEAT_STREAMING"
TARGET_ROOT = "aoss://baiqiao/InsScene-15K/FEAT_TARGET"
PYTHON_BIN = "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"
CANONICAL_NAMES = (
    "all_results.csv",
    "a1_results.csv",
    "a2_results.csv",
    "b2_results.csv",
    "c1_results.csv",
    "c2_results.csv",
    "c3_results.csv",
)

CANARY_IDS = {
    "lingbot_video_layer_benchmark_ego_belief_v2_lingbot_multi_layer_l48_p8_hall_s42",
    "minimax_h3_layer_benchmark_streaming_mask_minimax_h3_multi_layer_l10_p16_hall_s42",
    "wan_cog_layer_sweep_action_dynamics_wan_run_dir_discovered_l0_p12_hall_s42",
}

SCRIPT_INPUTS = (
    "scripts/q00_phase1_prepare.py",
    "scripts/q00_phase1_run_eval_shard.py",
    "scripts/q00_phase1_audit_eval.py",
    "scripts/q00_phase1_promote.py",
    "scripts/check_project_consistency.py",
    "scripts/q00_phase1_finalize.py",
    "vidfm3d/eval_diag.py",
    "vidfm3d/eval_a1_streaming.py",
    "vidfm3d/data/components/inscene15k_dataset.py",
    "vidfm3d/utils/aoss_fs.py",
    "vidfm3d/utils/provenance.py",
)

RUNTIME_GLOBS = (
    "configs/**/*.yaml",
    "vidfm3d/**/*.py",
    "pyproject.toml",
)

WAVE_E_RESOLUTION_INPUTS = {
    "b2_minimax.csv": (Path("/tmp/q00_phase1_wave_e_b2_minimax.csv"), "0bdb3dbac731ee3793166949e28289bf26077db4ed70a5fcfca14e555871b5a3"),
    "missing42.csv": (Path("/tmp/q00_phase1_wave_e_missing42.csv"), "34ea498faae438b7e1ec7fc56e1a16240bdd365082af278f58b84e055e21fbd6"),
    "wancog.csv": (Path("/tmp/q00_phase1_wave_e_wancog.csv"), "44d81655c694c24d55a7f5536ba3d473ed9da6dbfe91186ed709caea6e819094"),
}

B2_VALID_SUBSET_CONTRACT = {
    "manifest_sha256": "772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c",
    "prefix": "8",
    "expected_valid_n": 47,
    "ordered_signature_sha256": "c393f1a090525a565034bebad33b906a4e5a6efe59df88b25638e20c23ac376c",
    "signature_fields": ["scene_path", "target_bin", "polar_gt_rounded_1e-6"],
    "evidence": {
        "logs/inscene15k_streaming/runs/full_missing_eval_20260817/full_missing_eval_20260817_ego_belief_v2_minimax_h3_p8_layer10/eval/val_predictions.pt": "69204192d6876019b439b9c9b5e0b04d9b14a3e818a598975ebe716c592f47f7",
        "logs/inscene15k_streaming/runs/full_missing_eval_20260817/full_missing_eval_20260817_ego_belief_v2_videomae_p8_layer0/eval/val_predictions.pt": "833468fe80e68d3159654f30cb3c908de23fac3cd93dce91cfdc2199ff045f0a",
        "logs/inscene15k_streaming/runs/remaining_diag_4node_20260817/remaining_diag_4node_20260817_ego_belief_v2_vjepa2_p8_layer11/eval/val_predictions.pt": "0593aeab353b1d1e826cab49d233dcc7b5e5f6c30f336c283e8614c71b972833",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str], delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def rel(path: Path | str) -> str:
    item = Path(path)
    if item.is_absolute():
        try:
            return str(item.relative_to(ROOT))
        except ValueError:
            return str(item)
    return str(item)


def clean(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower().replace("@", "at").replace("/", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_.+-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "na"


def registry_id(row: dict[str, str], probe_label: str) -> str:
    """Build an ID whose machine-readable namespace preserves full identity."""
    return "_".join(
        [
            "q00",
            clean(row["family"]),
            clean(probe_label),
            clean(row["model"]),
            clean(row["campaign"]),
            clean(row["model_variant"]),
            clean(row["protocol"]),
            f"l{clean(row['layer'])}",
            f"p{clean(row['prefix'])}",
            f"h{clean(row['horizon'])}",
            f"s{clean(row['seed'])}",
        ]
    )


def summary_contract(row: dict[str, str]) -> dict[str, Any]:
    family, probe = row["family"], row["probe"]
    if family == "A1" and probe == "streaming_depth":
        required = ["mean_loss_depth", "n_loss_depth", "Auc_30", "n_Auc_30"]
        counts = ["n_loss_depth", "n_Auc_30"]
        heads = ["depth", "camera"]
    elif family == "A1" and probe == "streaming_mask":
        required = ["mean_loss_identity", "n_loss_identity"]
        counts = ["n_loss_identity"]
        heads = ["identity"]
    elif family == "A2":
        required = [
            "n", "n_pairs", "mean_overlap_mae", "hard_acc", "pos_acc", "neg_acc",
            "balanced_acc", "roc_auc", "pr_auc", "n_pos", "n_neg",
        ]
        counts, heads = [], []
    elif family == "B2":
        required = [
            "n", "mean_ang_err_deg", "mean_az_err_deg", "mean_el_err_deg",
            "top1", "top3", "mean_log_dist_err",
        ]
        counts, heads = [], []
    elif family == "C1":
        required = [
            "n", "global_R@1", "global_R@5", "mean_cos",
            "no_action_global_R@1", "shuffled_action_global_R@1",
            "last_observation_global_R@1", "no_action_mean_cos", "action_gain_cos",
        ]
        counts, heads = [], []
    elif family == "C2":
        required = [
            "n", "n_valid_records", "n_valid_steps", "global_R@1", "global_R@5",
            "final_pose_error", "drift_rate", "mean_step_pose_error",
            "n_loop", "pose_error_h1", "pose_error_h2", "pose_error_h4",
        ]
        counts, heads = [], []
    elif family == "C3":
        required = [
            "n", "n_valid_interventions", "global_R@1", "global_R@5",
            "counterfactual_consistency", "intervention_validity", "intervention_margin",
            "cf_cos_h1", "cf_cos_h2", "cf_cos_h4",
        ]
        counts, heads = [], []
    else:
        raise ValueError(f"unsupported family/probe contract: {family}/{probe}")
    return {"required_numeric_keys": required, "count_equals_expected_n": counts, "active_heads": heads}


def expected_outputs(row: dict[str, str]) -> list[dict[str, str]]:
    family, probe = row["family"], row["probe"]
    if family == "A1" and probe == "streaming_depth":
        contracts = [("streaming_depth/depth", "mean_loss_depth"), ("streaming_depth/camera", "Auc_30")]
    elif family == "A1" and probe == "streaming_mask":
        contracts = [("streaming_mask/mask", "mean_loss_identity")]
    else:
        metric = {
            "A2": "balanced_acc",
            "B2": "mean_ang_err_deg",
            "C1": "global_R@1",
            "C2": "global_R@1",
            "C3": "intervention_validity",
        }[family]
        contracts = [(probe, metric)]
    return [
        {
            "registry_id": registry_id(row, label),
            "probe_label": label,
            "metric_name": metric,
        }
        for label, metric in contracts
    ]


def overrides(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        item = line.strip()
        if item.startswith("- ") and "=" in item:
            key, value = item[2:].split("=", 1)
            result[key] = value
    return result


def parsed_override(value: str) -> Any:
    """Parse one Hydra override value without resolving or executing it."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid frozen Hydra override value: {value!r}") from exc


def resolved_training_config(
    run_manifest_path: Path,
    hydra_config_path: Path,
    config_tree_path: Path,
) -> tuple[dict[str, Any], str]:
    """Recover the exact resolved training config from the run provenance.

    Hydra's ``.hydra/config.yaml`` retains interpolation tokens and one legacy
    LingBot run has a zero-byte copy.  ``run_manifest.json`` was emitted from
    the resolved runtime config for every Wave-B run, so it is the authoritative
    machine-readable training-config artifact.  The Hydra/config-tree files are
    still required and hashed as independent lineage evidence.
    """
    if not run_manifest_path.is_file() or run_manifest_path.stat().st_size <= 0:
        raise ValueError(f"missing/nonempty run_manifest: {run_manifest_path}")
    if not hydra_config_path.is_file():
        raise ValueError(f"missing Hydra config artifact: {hydra_config_path}")
    if not config_tree_path.is_file() or config_tree_path.stat().st_size <= 0:
        raise ValueError(f"missing/nonempty config tree: {config_tree_path}")
    payload = json.loads(run_manifest_path.read_text())
    config = payload.get("config")
    if not isinstance(config, dict) or not config:
        raise ValueError(f"run_manifest lacks resolved config: {run_manifest_path}")
    canonical = json.dumps(config, indent=2, sort_keys=True) + "\n"
    return config, canonical


def validate_resolved_training_contract(
    row: dict[str, str],
    ov: dict[str, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed unless the resolved training config matches full identity."""
    expected_probe = row["probe"]
    expected_training_probe = "pixalign" if row["family"] == "A1" else expected_probe
    logical_layer = int(row["layer"])
    expected_feature_layer = (
        logical_layer - 1
        if (row["model"], logical_layer) in {("lingbot", 48), ("minimax_h3", 50)}
        else logical_layer
    )
    expected_timestep = int(row["default_timestep"])
    expected_postfix = f"_t{expected_timestep}_layer{expected_feature_layer}"
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    trainer = config.get("trainer") if isinstance(config.get("trainer"), dict) else {}
    checks = {
        "seed": str(config.get("seed")) == row["seed"],
        "probe": str(model.get("probe_type")) == expected_training_probe,
        "vfm_name": str(config.get("vfm_name")) == ov["vfm_name"] == row["model"],
        "video_channels": str(config.get("video_channels")) == ov["video_channels"],
        "prefix_len": str(config.get("prefix_len")) == ov["prefix_len"] == row["prefix"],
        "feature_layer": str(config.get("feature_layer")) == ov["feature_layer"],
        "feature_timestep": str(config.get("feature_timestep")) == ov["feature_timestep"],
        "feat_postfix": str(config.get("feat_postfix")) == ov["feat_postfix"],
        "logical_to_feature_layer": int(config.get("feature_layer")) == expected_feature_layer,
        "default_timestep": int(config.get("feature_timestep")) == expected_timestep,
        "feature_postfix_identity": str(config.get("feat_postfix")) == expected_postfix,
        "streaming_feat_root": config.get("streaming_feat_root") == ov["streaming_feat_root"] == STREAM_ROOT,
        "max_epochs": str(trainer.get("max_epochs")) == row["expected_max_epochs"],
    }
    target_required = expected_probe in {"action_dynamics", "path_integration", "counterfactual"}
    checks["target_feat_root"] = (
        config.get("target_feat_root") == ov.get("target_feat_root") == TARGET_ROOT
        if target_required
        else config.get("target_feat_root") in (None, "")
    )
    critical_keys = (
        "action_horizons",
        "counterfactual_min_overlap",
        "streaming_hidden_seed_prefix_len",
        "streaming_hidden_prefix_lengths",
    )
    critical: dict[str, Any] = {}
    for key in critical_keys:
        if config.get(key) is not None:
            critical[key] = config[key]
        if key in ov:
            checks[f"override_{key}"] = parsed_override(ov[key]) == config.get(key)
    if expected_probe == "path_integration":
        checks["path_action_horizons"] = critical.get("action_horizons") == [1, 2, 4]
    if expected_probe == "counterfactual":
        checks["counterfactual_action_horizons"] = critical.get("action_horizons") == [1, 2, 4]
        checks["counterfactual_min_overlap"] = critical.get("counterfactual_min_overlap") == 0.05
    if expected_probe == "action_dynamics":
        checks["action_dynamics_horizon"] = critical.get("action_horizons") == [1]
    if expected_probe == "ego_belief_v2":
        checks["hidden_seed_prefix"] = critical.get("streaming_hidden_seed_prefix_len") == int(row["prefix"])
        checks["hidden_prefix_lengths"] = critical.get("streaming_hidden_prefix_lengths") == [int(row["prefix"])]
    if not all(checks.values()):
        raise ValueError(f"resolved training-config identity mismatch {row['universe_id']}: {checks}")
    return {"checks": checks, "critical_eval_contract": critical}


def manifest_info(path: Path) -> tuple[int, dict[str, Any], str, dict[str, int]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    train = [row for row in rows if str(row.get("split")) == "train"]
    val = [row for row in rows if str(row.get("split")) == "val"]
    if not train or not val:
        raise ValueError(f"manifest must contain nonempty train and val splits: {path}")
    sample = val[0]
    required = {"sample_id", "scene_id", "source", "streaming_tail", "streaming_indices", "streaming_window_id"}
    missing = sorted(required - set(sample))
    if missing:
        raise ValueError(f"manifest sample missing {missing}: {path}")
    ordered_ids = [str(row["sample_id"]) for row in val]
    if any(not item for item in ordered_ids) or len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError(f"manifest val sample IDs empty/duplicate: {path}")
    train_ids = {str(row.get("sample_id", "")) for row in train}
    val_ids = set(ordered_ids)
    train_scenes = {str(row.get("scene_id", "")) for row in train}
    val_scenes = {str(row.get("scene_id", "")) for row in val}
    all_ids = [str(row.get("sample_id", "")) for row in rows]
    if (
        any(not item for item in all_ids)
        or len(set(all_ids)) != len(all_ids)
        or train_ids & val_ids
        or train_scenes & val_scenes
    ):
        raise ValueError(f"manifest uniqueness/leakage gate failed: {path}")
    ordered_sha = hashlib.sha256(("\n".join(ordered_ids) + "\n").encode()).hexdigest()
    audit = {
        "train_n": len(train),
        "val_n": len(val),
        "total_n": len(rows),
        "sample_id_overlap_n": len(train_ids & val_ids),
        "scene_id_overlap_n": len(train_scenes & val_scenes),
    }
    return len(val), sample, ordered_sha, audit


def run(args: list[str]) -> str:
    proc = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if proc.returncode:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(args)}\n{proc.stdout}")
    return proc.stdout


def file_meta(path: Path, hash_value: str | None = None) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": rel(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": hash_value or sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--bundle-root", default=str(ROOT / "reports/q00_phase1_runs"))
    parser.add_argument("--hash-workers", type=int, default=4)
    args = parser.parse_args()

    bundle = Path(args.bundle_root).resolve() / args.run_tag
    if bundle.exists():
        raise SystemExit(f"refusing existing bundle: {bundle}")
    bundle.mkdir(parents=True)
    inputs = bundle / "inputs"

    protected = [ROOT / "experiments/registry.csv"] + [ROOT / "results/canonical" / name for name in CANONICAL_NAMES]
    protected_before = {rel(path): file_meta(path) for path in protected}
    if sha256(UNIVERSE) != EXPECTED_UNIVERSE_SHA:
        raise SystemExit("frozen universe SHA mismatch")

    universe = read_csv(UNIVERSE)
    reconciliation = read_csv(RECONCILIATION)
    waves = read_csv(WAVES)
    if len(universe) != 1096 or len(reconciliation) != 1096:
        raise SystemExit("universe/reconciliation row count mismatch")
    by_id = {row["universe_id"]: row for row in reconciliation}
    wave_b = [row for row in waves if row["wave"] == "Wave B"]
    if len(wave_b) != 99 or len({row["universe_id"] for row in wave_b}) != 99:
        raise SystemExit("Wave B is not exactly 99 unique rows")
    if {row["universe_id"] for row in wave_b} - set(by_id):
        raise SystemExit("Wave B contains IDs absent from reconciliation")
    identity_fields = ("campaign", "family", "probe", "model", "model_variant", "layer", "prefix")
    for wave_row in wave_b:
        reconciled = by_id[wave_row["universe_id"]]
        mismatches = [field for field in identity_fields if wave_row.get(field, "") != reconciled.get(field, "")]
        if mismatches:
            raise SystemExit(f"Wave-B/reconciliation identity mismatch {wave_row['universe_id']}: {mismatches}")
    if not CANARY_IDS <= {row["universe_id"] for row in wave_b}:
        raise SystemExit("canary IDs are not a subset of Wave B")

    ckpt_paths = [ROOT / by_id[row["universe_id"]]["checkpoint_path"] for row in wave_b]
    if any(not path.is_file() for path in ckpt_paths):
        raise SystemExit("one or more exact checkpoints are missing")
    with ThreadPoolExecutor(max_workers=max(1, args.hash_workers)) as pool:
        ckpt_hashes = dict(zip(ckpt_paths, pool.map(sha256, ckpt_paths)))

    jobs: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    artifact_inventory: dict[str, dict[str, Any]] = {}
    feature_inventory: list[dict[str, Any]] = []
    for wave_row in wave_b:
        row = by_id[wave_row["universe_id"]]
        if row["protocol"] != PROTOCOL:
            raise SystemExit(f"protocol mismatch: {row['universe_id']}")
        if row["checkpoint_status"] != "STRICT_CLEAN_FINAL_READY":
            raise SystemExit(f"checkpoint not strict-final: {row['universe_id']}")
        if row["clean_completion_status"] != "strict_pass=true;clean_marker=true;bad_marker=false;epoch_complete=true":
            raise SystemExit(f"clean-completion mismatch: {row['universe_id']}")
        manifest = ROOT / row["manifest"]
        ckpt = ROOT / row["checkpoint_path"]
        hydra_dir = ROOT / row["run_directory"] / ".hydra"
        config_path = ROOT / row["config"]
        hydra_config_path = hydra_dir / "config.yaml"
        overrides_path = hydra_dir / "overrides.yaml"
        run_manifest_path = ROOT / row["run_directory"] / "run_manifest.json"
        config_tree_path = ROOT / row["run_directory"] / "config_tree.log"
        train_log_path = ROOT / row["train_log_path"]
        launcher_manifest_path = ROOT / row["launcher_manifest"]
        required_local = (
            manifest,
            config_path,
            hydra_config_path,
            overrides_path,
            run_manifest_path,
            config_tree_path,
            train_log_path,
            launcher_manifest_path,
        )
        if not all(path.is_file() for path in required_local):
            missing_paths = [rel(path) for path in required_local if not path.is_file()]
            raise SystemExit(f"missing exact local artifacts {missing_paths}: {row['universe_id']}")
        manifest_hash = sha256(manifest)
        if manifest_hash != row["manifest_sha256"]:
            raise SystemExit(f"manifest SHA mismatch: {row['universe_id']}")
        ov = overrides(overrides_path)
        required = ("experiment", "vfm_name", "video_channels", "streaming_feat_root", "feature_layer",
                    "feat_postfix", "feature_timestep", "prefix_len")
        missing = [key for key in required if not ov.get(key)]
        if missing:
            raise SystemExit(f"missing exact overrides {missing}: {row['universe_id']}")
        if ov["vfm_name"] != row["model"] or ov["prefix_len"] != row["prefix"]:
            raise SystemExit(f"override identity mismatch: {row['universe_id']}")
        if ov["streaming_feat_root"] != STREAM_ROOT:
            raise SystemExit(f"unexpected streaming root: {row['universe_id']}")
        target_required = row["probe"] in {"action_dynamics", "path_integration", "counterfactual"}
        if target_required and ov.get("target_feat_root") != TARGET_ROOT:
            raise SystemExit(f"missing/unexpected target root: {row['universe_id']}")
        resolved_config, resolved_config_text = resolved_training_config(
            run_manifest_path, hydra_config_path, config_tree_path
        )
        resolved_contract = validate_resolved_training_contract(row, ov, resolved_config)
        resolved_config_path = inputs / "resolved_training_configs" / f"{row['universe_id']}.json"
        resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_config_path.write_text(resolved_config_text)
        resolved_config_hash = sha256(resolved_config_path)
        expected_n, sample, manifest_val_ids_sha, manifest_audit = manifest_info(manifest)
        outputs = expected_outputs(row)
        contract = summary_contract(row)
        expected_result_n = expected_n
        valid_subset_signature = ""
        valid_subset_evidence = ""
        if row["family"] == "B2":
            if manifest_hash != B2_VALID_SUBSET_CONTRACT["manifest_sha256"] or row["prefix"] != B2_VALID_SUBSET_CONTRACT["prefix"]:
                raise SystemExit(f"no frozen B2 valid-subset contract for {row['universe_id']}")
            expected_result_n = int(B2_VALID_SUBSET_CONTRACT["expected_valid_n"])
            valid_subset_signature = str(B2_VALID_SUBSET_CONTRACT["ordered_signature_sha256"])
            valid_subset_evidence = ";".join(B2_VALID_SUBSET_CONTRACT["evidence"])
        expected_epoch = int(row["expected_final_epoch"])
        epoch_match = re.search(r"epoch=(\d+)-step=(\d+)\.ckpt$", ckpt.name)
        if not epoch_match or int(epoch_match.group(1)) != expected_epoch:
            raise SystemExit(f"checkpoint filename is not expected final epoch: {row['universe_id']}")
        extra = resolved_contract["critical_eval_contract"]
        sample_target = ""
        window = [int(value) for value in sample.get("streaming_window_indices", [])]
        prefix_len = len(sample["streaming_indices"])
        if target_required and len(window) > prefix_len:
            sample_target = str(window[prefix_len])
        output_dir = ROOT / "logs/inscene15k_streaming/runs" / args.run_tag / f"{args.run_tag}_{row['probe']}_{row['model']}_p{row['prefix']}_layer{row['layer']}"
        job = {
            "run_tag": args.run_tag,
            "universe_id": row["universe_id"],
            "campaign": row["campaign"],
            "model_variant": row["model_variant"],
            "protocol": row["protocol"],
            "family": row["family"],
            "probe": row["probe"],
            "model": row["model"],
            "layer": row["layer"],
            "prefix": row["prefix"],
            "horizon": row["horizon"],
            "seed": row["seed"],
            "is_canary": "true" if row["universe_id"] in CANARY_IDS else "false",
            "evaluator": "a1" if row["family"] == "A1" else "diag",
            "experiment": ov["experiment"],
            "vfm_name": ov["vfm_name"],
            "video_channels": ov["video_channels"],
            "streaming_feat_root": ov["streaming_feat_root"],
            "target_feat_root": ov.get("target_feat_root", ""),
            "feature_layer": ov["feature_layer"],
            "feat_postfix": ov["feat_postfix"],
            "feature_timestep": ov["feature_timestep"],
            "prefix_len": ov["prefix_len"],
            "extra_overrides_json": json.dumps(extra, sort_keys=True, separators=(",", ":")),
            "manifest": rel(manifest),
            "frozen_manifest": rel(bundle / "inputs/manifests" / manifest.name),
            "manifest_sha256": manifest_hash,
            "expected_n": str(expected_n),
            "manifest_train_n": str(manifest_audit["train_n"]),
            "manifest_total_n": str(manifest_audit["total_n"]),
            "manifest_train_val_sample_overlap_n": str(manifest_audit["sample_id_overlap_n"]),
            "manifest_train_val_scene_overlap_n": str(manifest_audit["scene_id_overlap_n"]),
            "manifest_val_ordered_ids_sha256": manifest_val_ids_sha,
            "expected_result_n": str(expected_result_n),
            "expected_valid_subset_signature_sha256": valid_subset_signature,
            "valid_subset_contract_evidence": valid_subset_evidence,
            "checkpoint": rel(ckpt),
            "checkpoint_sha256": ckpt_hashes[ckpt],
            "checkpoint_size_bytes": str(ckpt.stat().st_size),
            "checkpoint_status": row["checkpoint_status"],
            "clean_completion_status": row["clean_completion_status"],
            "expected_max_epochs": row["expected_max_epochs"],
            "expected_final_epoch": row["expected_final_epoch"],
            "expected_final_step": row["checkpoint_step"],
            "expected_outputs_json": json.dumps(outputs, sort_keys=True, separators=(",", ":")),
            "required_summary_metrics_json": json.dumps(contract["required_numeric_keys"], separators=(",", ":")),
            "count_equals_expected_n_json": json.dumps(contract["count_equals_expected_n"], separators=(",", ":")),
            "expected_active_heads_json": json.dumps(contract["active_heads"], separators=(",", ":")),
            "prediction_required": "true",
            "sample_id": sample["sample_id"],
            "sample_scene_id": sample["scene_id"],
            "sample_source": sample["source"],
            "sample_window_id": str(sample["streaming_window_id"]),
            "sample_tail": str(sample["streaming_tail"]),
            "sample_streaming_indices_json": json.dumps(sample["streaming_indices"], separators=(",", ":")),
            "sample_target_raw_id": sample_target,
            "hydra_config": row["hydra_config_path"],
            "hydra_config_sha256": sha256(hydra_config_path),
            "hydra_overrides": rel(overrides_path),
            "hydra_overrides_sha256": sha256(overrides_path),
            "run_manifest": rel(run_manifest_path),
            "run_manifest_sha256": sha256(run_manifest_path),
            "config_tree": rel(config_tree_path),
            "config_tree_sha256": sha256(config_tree_path),
            "resolved_config_source_kind": "run_manifest.config_resolved",
            "resolved_config_frozen": rel(resolved_config_path),
            "resolved_config_sha256": resolved_config_hash,
            "resolved_eval_contract_json": json.dumps(
                resolved_contract, sort_keys=True, separators=(",", ":")
            ),
            "experiment_config": rel(config_path),
            "experiment_config_sha256": sha256(config_path),
            "train_log": row["train_log_path"],
            "train_log_sha256": sha256(train_log_path),
            "train_run_directory": row["run_directory"],
            "launcher_manifest": row["launcher_manifest"],
            "launcher_manifest_sha256": sha256(launcher_manifest_path),
            "output_directory": rel(output_dir),
            "summary_path": rel(output_dir / "eval/val_summary.json"),
            "prediction_path": rel(output_dir / "eval/val_predictions.pt"),
            "preflight_path": rel(output_dir / "eval/q00_preflight.json"),
            "command_path": rel(output_dir / "eval/q00_command.json"),
        }
        jobs.append(job)
        checkpoints.append({
            "universe_id": row["universe_id"],
            **file_meta(ckpt, ckpt_hashes[ckpt]),
            "checkpoint_status": row["checkpoint_status"],
            "clean_completion_status": row["clean_completion_status"],
            "expected_final_epoch": row["expected_final_epoch"],
            "expected_final_step": row["checkpoint_step"],
        })
        for artifact in required_local:
            if artifact.is_file() and str(artifact) not in artifact_inventory:
                artifact_inventory[str(artifact)] = file_meta(artifact)
        feature_inventory.append({
            "universe_id": row["universe_id"],
            "streaming_feat_root": ov["streaming_feat_root"],
            "target_feat_root": ov.get("target_feat_root", ""),
            "feature_layer": ov["feature_layer"],
            "feature_timestep": ov["feature_timestep"],
            "feat_postfix": ov["feat_postfix"],
            "sample_id": sample["sample_id"],
            "remote_status": "PENDING_CANARY_OR_EVAL_PREFLIGHT",
        })

    jobs.sort(key=lambda item: (item["is_canary"] != "true", item["model"], item["family"], item["probe"], int(item["prefix"]), int(item["layer"])))
    if len(jobs) != 99 or sum(item["is_canary"] == "true" for item in jobs) != 3:
        raise SystemExit("job/canary count mismatch")
    expected_registry_ids = [entry["registry_id"] for job in jobs for entry in json.loads(job["expected_outputs_json"])]
    if len(expected_registry_ids) != 105 or len(set(expected_registry_ids)) != 105:
        raise SystemExit("expected atomic registry ID count mismatch")
    current_registry_ids = {row["experiment_id"] for row in read_csv(ROOT / "experiments/registry.csv")}
    current_canonical_ids = {
        row["experiment_id"]
        for name in CANONICAL_NAMES
        for row in read_csv(ROOT / "results/canonical" / name)
        if row.get("experiment_id")
    }
    overlap = set(expected_registry_ids) & (current_registry_ids | current_canonical_ids)
    if overlap:
        raise SystemExit(f"new full-identity registry IDs already exist: {sorted(overlap)}")

    fields = list(jobs[0])
    write_csv(inputs / "wave_b_99.tsv", jobs, fields, delimiter="\t")
    canary_jobs = [item for item in jobs if item["is_canary"] == "true"]
    write_csv(inputs / "wave_b_canary_3.tsv", canary_jobs, fields, delimiter="\t")
    write_csv(inputs / "wave_b_remaining_96.tsv", [item for item in jobs if item["is_canary"] == "false"], fields, delimiter="\t")
    canary_shards: list[Path] = []
    for item in canary_jobs:
        shard = inputs / "canary_shards" / f"canary_{item['model']}.tsv"
        write_csv(shard, [item], fields, delimiter="\t")
        canary_shards.append(shard)
    write_csv(inputs / "checkpoint_inventory.csv", checkpoints, list(checkpoints[0]))
    write_csv(inputs / "feature_root_inventory.csv", feature_inventory, list(feature_inventory[0]))
    write_json(inputs / "artifact_inventory.json", sorted(artifact_inventory.values(), key=lambda item: item["path"]))
    write_json(inputs / "expected_registry_ids.json", expected_registry_ids)
    write_json(inputs / "protected_before.json", protected_before)

    frozen_source_copies: list[Path] = []
    for source in (UNIVERSE, RECONCILIATION, WAVES, Q00_REPORT):
        destination = inputs / "source_of_truth" / rel(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        frozen_source_copies.append(destination)
    frozen_resolved_config_copies = sorted(
        (inputs / "resolved_training_configs").glob("*.json")
    )
    if len(frozen_resolved_config_copies) != 99:
        raise SystemExit("resolved training-config freeze is not exactly 99 files")
    frozen_manifest_copies: list[Path] = []
    for manifest in sorted({ROOT / item["manifest"] for item in jobs}):
        destination = inputs / "manifests" / manifest.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest, destination)
        frozen_manifest_copies.append(destination)
    frozen_config_copies: list[Path] = []
    for config in sorted({ROOT / by_id[item["universe_id"]]["config"] for item in jobs if by_id[item["universe_id"]]["config"]}):
        if config.is_file():
            destination = inputs / "configs" / rel(config)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config, destination)
            frozen_config_copies.append(destination)
    for script_name in SCRIPT_INPUTS:
        source = ROOT / script_name
        if not source.is_file():
            raise SystemExit(f"missing phase-1 script input: {script_name}")
        destination = bundle / "provenance/scripts" / script_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    commit = run(["git", "rev-parse", "HEAD"]).strip()
    branch = run(["git", "branch", "--show-current"]).strip()
    status = run(["git", "status", "--porcelain=v1", "-uall"])
    diff = run(["git", "diff", "--binary"])
    provenance = bundle / "provenance"
    (provenance / "git_commit.txt").write_text(commit + "\n")
    (provenance / "git_branch.txt").write_text(branch + "\n")
    (provenance / "git_status_porcelain.txt").write_text(status)
    (provenance / "dirty.diff").write_text(diff)
    script_inventory = [file_meta(bundle / "provenance/scripts" / name) for name in SCRIPT_INPUTS]
    write_json(provenance / "script_inventory.json", script_inventory)

    runtime_sources: set[Path] = set()
    for pattern in RUNTIME_GLOBS:
        runtime_sources.update(path for path in ROOT.glob(pattern) if path.is_file())
    runtime_inventory = []
    for source in sorted(runtime_sources):
        destination = provenance / "runtime_tree" / rel(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        runtime_inventory.append({
            "path": rel(source),
            "sha256": sha256(source),
            "frozen_path": rel(destination),
            "frozen_sha256": sha256(destination),
            "size_bytes": source.stat().st_size,
        })
    write_json(provenance / "runtime_inventory.json", runtime_inventory)

    wave_e_frozen_hashes = {}
    for name, (source, expected_hash) in WAVE_E_RESOLUTION_INPUTS.items():
        if not source.is_file() or sha256(source) != expected_hash:
            raise SystemExit(f"Wave-E resolution input missing/hash mismatch: {source}")
        destination = provenance / "wave_e_resolution_inputs" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256(destination) != expected_hash:
            raise SystemExit(f"Wave-E frozen copy hash mismatch: {destination}")
        wave_e_frozen_hashes[f"provenance/wave_e_resolution_inputs/{name}"] = expected_hash

    b2_contract_copy = json.loads(json.dumps(B2_VALID_SUBSET_CONTRACT))
    b2_evidence_frozen_hashes = {}
    for index, (source_name, expected_hash) in enumerate(B2_VALID_SUBSET_CONTRACT["evidence"].items(), 1):
        source = ROOT / source_name
        if not source.is_file() or sha256(source) != expected_hash:
            raise SystemExit(f"B2 valid-subset evidence missing/hash mismatch: {source}")
        destination = provenance / "b2_valid_subset_evidence" / f"evidence_{index:02d}_val_predictions.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256(destination) != expected_hash:
            raise SystemExit(f"B2 valid-subset frozen evidence hash mismatch: {destination}")
        b2_evidence_frozen_hashes[f"provenance/b2_valid_subset_evidence/{destination.name}"] = expected_hash
    write_json(provenance / "b2_valid_subset_contract.json", b2_contract_copy)

    cli = [
        f"{PYTHON_BIN} -B scripts/q00_phase1_prepare.py --run-tag {args.run_tag}",
        *[
            f"RUN_TAG={args.run_tag} BUNDLE={bundle} SHARD_TSV={shard} SLOTS=1 GPU_IDS_CSV=0 PYTHON_BIN={PYTHON_BIN} {PYTHON_BIN} -B scripts/q00_phase1_run_eval_shard.py"
            for shard in canary_shards
        ],
        f"{PYTHON_BIN} -B scripts/q00_phase1_audit_eval.py --bundle {bundle} --scope canary",
        "# Submit the 96-row manifest only after audit reports all three canaries PASS.",
        f"RUN_TAG={args.run_tag} BUNDLE={bundle} SHARD_TSV={inputs / 'wave_b_remaining_96.tsv'} SLOTS=8 GPU_IDS_CSV=0,1,2,3,4,5,6,7 PYTHON_BIN={PYTHON_BIN} {PYTHON_BIN} -B scripts/q00_phase1_run_eval_shard.py",
        f"{PYTHON_BIN} -B scripts/q00_phase1_audit_eval.py --bundle {bundle} --scope full",
        f"{PYTHON_BIN} -B scripts/q00_phase1_promote.py --bundle {bundle}",
        f"{PYTHON_BIN} -B scripts/q00_phase1_finalize.py --bundle {bundle}",
    ]
    (provenance / "planned_cli.txt").write_text("\n".join(cli) + "\n")
    frozen_input_hashes = {
        "experiments/task_status_1096_before_expand_20260822.csv": sha256(UNIVERSE),
        "results/analysis/q00_1096/universe_1096_reconciled.csv": sha256(RECONCILIATION),
        "results/analysis/q00_1096/execution_waves.csv": sha256(WAVES),
        "reports/q00_universe_reconciliation_1096.md": sha256(Q00_REPORT),
        "inputs/wave_b_99.tsv": sha256(inputs / "wave_b_99.tsv"),
        "inputs/wave_b_canary_3.tsv": sha256(inputs / "wave_b_canary_3.tsv"),
        "inputs/wave_b_remaining_96.tsv": sha256(inputs / "wave_b_remaining_96.tsv"),
        "inputs/expected_registry_ids.json": sha256(inputs / "expected_registry_ids.json"),
        "inputs/protected_before.json": sha256(inputs / "protected_before.json"),
        "inputs/checkpoint_inventory.csv": sha256(inputs / "checkpoint_inventory.csv"),
        "inputs/feature_root_inventory.csv": sha256(inputs / "feature_root_inventory.csv"),
        "inputs/artifact_inventory.json": sha256(inputs / "artifact_inventory.json"),
        "provenance/script_inventory.json": sha256(provenance / "script_inventory.json"),
        "provenance/runtime_inventory.json": sha256(provenance / "runtime_inventory.json"),
        "provenance/git_commit.txt": sha256(provenance / "git_commit.txt"),
        "provenance/git_branch.txt": sha256(provenance / "git_branch.txt"),
        "provenance/git_status_porcelain.txt": sha256(provenance / "git_status_porcelain.txt"),
        "provenance/dirty.diff": sha256(provenance / "dirty.diff"),
        "provenance/planned_cli.txt": sha256(provenance / "planned_cli.txt"),
        "provenance/b2_valid_subset_contract.json": sha256(provenance / "b2_valid_subset_contract.json"),
        **b2_evidence_frozen_hashes,
        **wave_e_frozen_hashes,
    }
    for shard in canary_shards:
        frozen_input_hashes[rel(shard.relative_to(bundle))] = sha256(shard)
    for frozen_path in (
        frozen_source_copies
        + frozen_manifest_copies
        + frozen_config_copies
        + frozen_resolved_config_copies
    ):
        frozen_input_hashes[rel(frozen_path.relative_to(bundle))] = sha256(frozen_path)
    for metadata_item in artifact_inventory.values():
        artifact_path = Path(metadata_item["path"])
        if artifact_path.is_absolute():
            try:
                artifact_key = str(artifact_path.relative_to(ROOT))
            except ValueError as exc:
                raise SystemExit(f"artifact inventory path escapes repository: {artifact_path}") from exc
        else:
            artifact_key = str(artifact_path)
        frozen_input_hashes[artifact_key] = metadata_item["sha256"]
    metadata = {
        "run_tag": args.run_tag,
        "root": str(ROOT),
        "bundle": str(bundle),
        "git_commit": commit,
        "git_branch": branch,
        "universe_rows": len(universe),
        "universe_sha256": sha256(UNIVERSE),
        "reconciliation_sha256": sha256(RECONCILIATION),
        "execution_waves_sha256": sha256(WAVES),
        "source_hashes": frozen_input_hashes,
        "wave_b_parent_rows": len(jobs),
        "wave_b_atomic_registry_ids": len(expected_registry_ids),
        "canary_rows": 3,
        "remaining_rows": 96,
        "protocol": PROTOCOL,
        "protected_before": protected_before,
        "preparation_only": True,
        "registry_identity_policy": "experiment_id encodes campaign+real model_variant+protocol; model_variant column is real frozen variant",
        "runtime_inventory_files": len(runtime_inventory),
        "resolved_training_config_files": len(frozen_resolved_config_copies),
        "resolved_training_config_policy": "run_manifest.config (resolved) is authoritative; .hydra/config.yaml and config_tree.log are independently hashed evidence",
    }
    write_json(bundle / "bundle.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
