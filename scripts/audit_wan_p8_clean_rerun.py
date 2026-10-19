#!/usr/bin/env python3
"""Audit Wan p8 canonical-clean training/eval artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
LAYERS = [0, 5, 10, 15, 25, 29]
MANIFEST_SHA = "772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c"
BAD_LOG_RE = re.compile(
    r"(Killed|CUDA out of memory|DataLoader worker.*Terminated|Traceback|RuntimeError|Exception)",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_name(run_tag: str, layer: int) -> str:
    return f"{run_tag}_canonical_clean_action_dynamics_wan_p8_layer{layer}"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception:
        return {}


def log_bad_patterns(paths: list[Path]) -> list[dict[str, Any]]:
    hits = []
    for path in paths:
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if BAD_LOG_RE.search(line):
                hits.append({"path": str(path), "line": lineno, "text": line[:500]})
                if len(hits) >= 20:
                    return hits
    return hits


def latest_checkpoint(run_dir: Path) -> Path | None:
    ckpts = sorted(run_dir.glob("VideoProbe3D/*/checkpoints/*.ckpt"))
    ckpts += sorted(run_dir.glob("checkpoints/*.ckpt"))
    if not ckpts:
        return None
    last = [p for p in ckpts if p.name == "last.ckpt"]
    return last[-1] if last else ckpts[-1]


def checkpoint_state(path: Path) -> dict[str, Any]:
    try:
        import torch

        state = torch.load(str(path), map_location="cpu", weights_only=False)
        return {
            "readable": True,
            "epoch": int(state.get("epoch", -1)),
            "global_step": int(state.get("global_step", -1)),
            "error": "",
        }
    except Exception as exc:
        return {"readable": False, "epoch": -1, "global_step": -1, "error": repr(exc)}


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(errors="replace")) or {}
    except Exception:
        return {}


def root_contract(prov: dict[str, Any], expected_streaming: Path, expected_target: Path) -> tuple[bool, list[dict[str, Any]]]:
    roots = prov.get("data", {}).get("feature_roots", [])
    by_name = {item.get("name"): item for item in roots if isinstance(item, dict)}
    checks = []
    for name, expected in (("streaming", expected_streaming), ("target", expected_target)):
        item = by_name.get(name, {})
        actual = item.get("realpath")
        checks.append(
            {
                "name": name,
                "expected": str(expected),
                "actual": actual,
                "exists": item.get("exists"),
                "is_symlink": item.get("is_symlink"),
                "ok": bool(
                    actual
                    and Path(actual).resolve() == expected
                    and item.get("exists") is True
                    and item.get("is_symlink") is False
                ),
            }
        )
    return all(item["ok"] for item in checks), checks


def audit_train(args: argparse.Namespace) -> dict[str, Any]:
    report_dir = Path(args.report_dir)
    out = {"run_tag": args.run_tag, "layers": {}, "ok": True}
    expected_feature_root = Path(
        "/mnt/umm/users/yinbaiqiao/InsScene-15K/FEATURES_VERSIONED"
    ) / args.run_tag
    expected_streaming = (expected_feature_root / "streaming").resolve()
    expected_target = (expected_feature_root / "target").resolve()
    required_provenance = [
        "full_gate_summary.json",
        "historical_training_contract.json",
        "launcher_feature_roots.txt",
        "launcher_git_commit.txt",
        "launcher_git_dirty.diff",
        "launcher_git_status_short.txt",
        "launcher_manifest_snapshot.jsonl",
        "launcher_train_command.txt",
        "manifest_snapshot.jsonl",
        "run_provenance.json",
    ]
    for layer in LAYERS:
        name = run_name(args.run_tag, layer)
        run_dir = ROOT / "logs/inscene15k_streaming/runs" / args.run_tag / name
        external_log = report_dir / "logs" / f"train_layer{layer}.log"
        status = read_json(report_dir / "logs" / f"train_layer{layer}.status.json")
        hydra_log = run_dir / "inscene15k_streaming.log"
        provenance = run_dir / "provenance" / "run_provenance.json"
        manifest_snapshot = run_dir / "provenance" / "manifest_snapshot.jsonl"
        config_path = run_dir / ".hydra" / "config.yaml"
        ckpt = latest_checkpoint(run_dir)
        bad = log_bad_patterns([external_log, hydra_log])
        external_text = external_log.read_text(errors="replace") if external_log.exists() else ""
        manifest_ok = (
            manifest_snapshot.exists()
            and sha256(manifest_snapshot) == MANIFEST_SHA
        )
        prov = read_json(provenance)
        roots_ok, root_checks = root_contract(prov, expected_streaming, expected_target)
        cfg = load_yaml(config_path)
        config_ok = bool(
            cfg
            and cfg.get("ckpt_path") is None
            and cfg.get("autoresume") is False
            and int(cfg.get("seed", -1)) == 42
            and int(cfg.get("feature_layer", -1)) == layer
            and int(cfg.get("feature_timestep", -1)) == 749
            and int(cfg.get("prefix_len", -1)) == 8
            and cfg.get("action_horizons") == [1]
            and int(cfg.get("trainer", {}).get("max_epochs", -1)) == 50
            and Path(cfg.get("streaming_feat_root", "")).resolve() == expected_streaming
            and Path(cfg.get("target_feat_root", "")).resolve() == expected_target
        )
        provenance_missing = [
            name for name in required_provenance if not (run_dir / "provenance" / name).is_file()
        ]
        data_prov = prov.get("data", {})
        provenance_ok = bool(
            provenance.exists()
            and data_prov.get("scene_manifest_sha256") == MANIFEST_SHA
            and not provenance_missing
        )
        state = checkpoint_state(ckpt) if ckpt and ckpt.exists() else {
            "readable": False,
            "epoch": -1,
            "global_step": -1,
            "error": "checkpoint missing",
        }
        checkpoint_resolved = ckpt.resolve() if ckpt and ckpt.exists() else None
        checkpoint_in_run = bool(
            checkpoint_resolved
            and checkpoint_resolved.is_relative_to(run_dir.resolve())
            and checkpoint_resolved.name == "epoch=49-step=15300.ckpt"
        )
        clean_exit_marker = f"TRAIN CLEAN EXIT layer={layer}" in external_text
        trainer_stop_marker = "Trainer.fit stopped: `max_epochs=50` reached." in external_text
        clean = (
            int(status.get("rc", -1)) == 0
            and not bad
            and ckpt is not None
            and ckpt.exists()
            and state["readable"]
            and state["epoch"] >= 49
            and state["global_step"] >= 15300
            and checkpoint_in_run
            and clean_exit_marker
            and provenance_ok
            and manifest_ok
            and roots_ok
            and config_ok
        )
        out["layers"][str(layer)] = {
            "status": "CLEAN_COMPLETE" if clean else "INVALID",
            "run_dir": str(run_dir),
            "return_code": status.get("rc"),
            "checkpoint": str(ckpt) if ckpt else "",
            "checkpoint_realpath": str(checkpoint_resolved) if checkpoint_resolved else "",
            "checkpoint_sha256": sha256(ckpt) if ckpt and ckpt.exists() else "",
            "checkpoint_state": state,
            "checkpoint_in_current_run": checkpoint_in_run,
            "provenance": str(provenance),
            "provenance_ok": provenance_ok,
            "provenance_missing": provenance_missing,
            "manifest_snapshot": str(manifest_snapshot),
            "manifest_snapshot_sha256": sha256(manifest_snapshot) if manifest_snapshot.exists() else "",
            "feature_root_checks": root_checks,
            "config": str(config_path),
            "config_ok": config_ok,
            "clean_exit_marker": clean_exit_marker,
            "trainer_stop_marker": trainer_stop_marker,
            "bad_log_hits": bad,
        }
        out["ok"] = out["ok"] and clean
    return out


def finite_number(value: Any) -> bool:
    try:
        x = float(value)
    except Exception:
        return False
    return math.isfinite(x)


def numeric_values_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(numeric_values_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(numeric_values_finite(item) for item in value)
    return True


def audit_eval(args: argparse.Namespace) -> dict[str, Any]:
    report_dir = Path(args.report_dir)
    train_audit = read_json(Path(args.train_audit))
    out = {"eval_run_tag": args.eval_run_tag, "layers": {}, "ok": True}
    expected_feature_root = Path(
        "/mnt/umm/users/yinbaiqiao/InsScene-15K/FEATURES_VERSIONED"
    ) / args.run_tag
    expected_streaming = (expected_feature_root / "streaming").resolve()
    expected_target = (expected_feature_root / "target").resolve()
    required = [
        "mean_cos",
        "global_R@1",
        "global_R@5",
        "mean_rank",
        "no_action_global_R@1",
        "shuffled_action_global_R@1",
        "last_observation_global_R@1",
        "action_gain_cos",
    ]
    for layer in LAYERS:
        name = run_name(args.run_tag, layer)
        eval_dir = ROOT / "logs/inscene15k_streaming/runs" / args.eval_run_tag / name
        summary_path = eval_dir / "eval" / "val_summary.json"
        external_log = report_dir / "logs" / f"eval_layer{layer}.log"
        status = read_json(report_dir / "logs" / f"eval_layer{layer}.status.json")
        summary = read_json(summary_path)
        provenance_path = eval_dir / "provenance" / "eval_provenance.json"
        provenance = read_json(provenance_path)
        config_path = eval_dir / ".hydra" / "config.yaml"
        cfg = load_yaml(config_path)
        n = int(summary.get("n", summary.get("n_samples", 0)) or 0)
        metrics_ok = (
            all(finite_number(summary.get(key)) for key in required)
            and numeric_values_finite(summary)
        )
        bad = log_bad_patterns([external_log])
        external_text = external_log.read_text(errors="replace") if external_log.exists() else ""
        train_entry = train_audit.get("layers", {}).get(str(layer), {})
        ckpt = Path(train_entry.get("checkpoint", ""))
        ckpt_hash = sha256(ckpt) if ckpt.exists() else ""
        summary_hash = sha256(summary_path) if summary_path.exists() else ""
        provenance_ok = bool(
            provenance.get("ok") is True
            and int(provenance.get("layer", -1)) == layer
            and provenance.get("manifest_sha256") == MANIFEST_SHA
            and Path(provenance.get("streaming_root", "")).resolve() == expected_streaming
            and Path(provenance.get("target_root", "")).resolve() == expected_target
            and Path(provenance.get("checkpoint", "")).resolve() == ckpt.resolve()
            and provenance.get("checkpoint_sha256") == ckpt_hash
            and provenance.get("summary_sha256") == summary_hash
            and int(provenance.get("n", 0)) == 276
        )
        config_ok = bool(
            cfg
            and cfg.get("train") is False
            and cfg.get("test") is False
            and cfg.get("autoresume") is False
            and int(cfg.get("seed", -1)) == 42
            and int(cfg.get("feature_layer", -1)) == layer
            and int(cfg.get("feature_timestep", -1)) == 749
            and int(cfg.get("prefix_len", -1)) == 8
            and Path(cfg.get("streaming_feat_root", "")).resolve() == expected_streaming
            and Path(cfg.get("target_feat_root", "")).resolve() == expected_target
            and Path(cfg.get("ckpt_path", "")).resolve() == ckpt.resolve()
        )
        summary_contract = bool(
            summary.get("probe_type") == "action_dynamics"
            and int(summary.get("feature_layer", -1)) == layer
            and int(summary.get("feature_timestep", -1)) == 749
            and summary.get("vfm_name") == "wan"
            and Path(summary.get("ckpt_path", "")).resolve() == ckpt.resolve()
            and Path(summary.get("streaming_feat_root", "")).resolve() == expected_streaming
        )
        ok = bool(
            train_audit.get("ok") is True
            and train_entry.get("status") == "CLEAN_COMPLETE"
            and int(status.get("rc", -1)) == 0
            and n == 276
            and metrics_ok
            and not bad
            and ckpt_hash
            and ckpt_hash == train_entry.get("checkpoint_sha256")
            and summary_contract
            and provenance_ok
            and config_ok
            and f"EVAL STAGING PASS layer={layer}" in external_text
        )
        out["layers"][str(layer)] = {
            "status": "EVAL_COMPLETE" if ok else "INVALID",
            "summary": str(summary_path),
            "return_code": status.get("rc"),
            "n": n,
            "metrics": {key: summary.get(key) for key in required},
            "metrics_finite": metrics_ok,
            "ckpt_path": str(ckpt),
            "ckpt_sha256": ckpt_hash,
            "summary_sha256": summary_hash,
            "summary_contract_ok": summary_contract,
            "provenance": str(provenance_path),
            "provenance_ok": provenance_ok,
            "config": str(config_path),
            "config_ok": config_ok,
            "bad_log_hits": bad,
        }
        out["ok"] = out["ok"] and ok
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("train")
    t.add_argument("--run-tag", required=True)
    t.add_argument("--report-dir", required=True)
    t.add_argument("--out-json", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--run-tag", required=True)
    e.add_argument("--eval-run-tag", required=True)
    e.add_argument("--report-dir", required=True)
    e.add_argument("--train-audit", required=True)
    e.add_argument("--out-json", required=True)
    args = ap.parse_args()

    result = audit_train(args) if args.mode == "train" else audit_eval(args)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.replace(out)
    print(json.dumps({"ok": result["ok"]}, indent=2))
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
