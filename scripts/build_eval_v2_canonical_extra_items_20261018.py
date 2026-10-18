#!/usr/bin/env python3
"""Build v2 evaluation items (revision r4) for the canonical-V2 runs that r2 did not cover.

Groups (each item copies its training run's own item, launcher environment and
code pack, and is checked against what the training recorded):

  b2_fp32_retrain   16 B2 (ego_belief_v2) fp32 re-trains of 2026-10-16
                    (single_gpu_b2_fp32_20261016_r1; packs r7_single_v3 / aetherfix_r1,
                    per-model overlay); they replace the 16 non-finite canonical B2 runs.
  minimax_all       96 MiniMax-H3 all-layer runs of the six-probe protocol
                    (main_v2_probe_minimax_all_20261011_r1; pack minimax_bf16_v2, bf16 V2
                    feature roots).  The campaign's streaming_camera and counterfactual runs
                    are outside the protocol and are not selected.
  minimax_l30_ab16  16 MiniMax-H3 layer-30 A/B runs (minimax_l30_ab16_20260930_r2), whose
                    formal training command (environment + overrides) is in
                    formal_submission_ledger.json.

Checks: training receipts (TRAIN_COMPLETE) agree on state, item SHA, code pack and
overlay where they exist; the run's run_manifest.json agrees with the rebuilt scene
manifest and feature roots; the final checkpoint is the last epoch of the run's own
Hydra config.  Evaluation packs: runtime/main_v2_eval_v2r2_from_<training pack>.

Items: artifacts/main_v2_probe_v2_816_20260917/eval_v2_canonical_20261017_r4/.
Default: report only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shlex
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
A = ROOT / "artifacts/main_v2_probe_v2_816_20260917"
RUNS = ROOT / "logs/inscene15k_streaming/runs"
CAMPAIGN = A / "eval_v2_canonical_20261017_r4"
AUDIT = ROOT / "artifacts/release_audit_20261018/overlay_binding_audit"
INVENTORY = ROOT / "metadata/release_inventory/canonical_v2.csv"
EVALUATOR = {
    "streaming_depth": ("a1", "mean_loss_depth"),
    "streaming_mask": ("a1", "mean_loss_identity"),
    "view_consistency": ("diag", "balanced_acc"),
    "ego_belief_v2": ("diag", "mean_ang_err_deg"),
    "action_dynamics": ("diag", "global_R@1"),
    "path_integration": ("diag", "global_R@1"),
}
DATA_ROOT = "/mnt/umm/users/yinbaiqiao/InsScene-15K/data"
SCENE_ROOT = DATA_ROOT + "/processed_scannetpp_v2/processed_scannetpp_v2"
MIRROR = "/mnt/umm/users/yinbaiqiao/InsScene-15K"
OVERLAY_ROOT = "aoss://baiqiao/PROBE816_OWNER_OVERLAY_ONLY"
EXECUTION_ONLY = re.compile(r"^(paths\.|logger|autoresume|callbacks\.|job_name|data\.data_module\.num_workers)")
ENV_EXECUTION_ONLY = {"PYTHONPATH", "AOSS_CACHE_DIR", "AOSS_FS_METRICS", "AOSS_FS_METRICS_MODE",
                      "AOSS_LOG_LOCAL_HITS", "OMP_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF", "WANDB_MODE",
                      "MPLCONFIGDIR", "PROJECT_ROOT", "PATH", "CUDA_VISIBLE_DEVICES"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"runner_{sha256(path)[:8]}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # constants and functions only; main() is guarded
    return module


def receipts() -> dict[str, dict]:
    """run directory -> TRAIN_COMPLETE fields, from both receipt dumps."""
    found = {}
    for name in ("training_receipts.jsonl", "training_receipts_extra.jsonl"):
        path = AUDIT / name
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line) if line.strip() else {}
            if str(row.get("path", "")).endswith("/TRAIN_COMPLETE.json") and "fields" in row:
                found[str(Path(row["path"]).parent)] = {**row["fields"], "_path": row["path"]}
    return found


def final_checkpoint(run_dir: Path) -> Path:
    found = sorted(run_dir.glob("VideoProbe3D/*/checkpoints/epoch=*.ckpt"))
    if len(found) != 1:
        raise ValueError(f"expected one checkpoint under {run_dir}, found {len(found)}")
    config = (run_dir / ".hydra/config.yaml").read_text()
    max_epochs = re.search(r"^\s+max_epochs:\s*(\d+)", config, re.M)
    epoch = int(re.search(r"epoch=(\d+)", found[0].name).group(1))
    if not max_epochs or epoch != int(max_epochs.group(1)) - 1:
        raise ValueError(f"checkpoint epoch {epoch} is not the last of max_epochs in {run_dir}")
    return found[0]


def items_by_key(item_root: Path) -> dict[str, tuple[Path, dict]]:
    out = {}
    for path in sorted(item_root.glob("*.jsonl")):
        item = json.loads(path.read_text().splitlines()[0])
        out[item["unique_key"]] = (path, item)
    return out


def check_manifest(run_dir: Path, env: dict) -> list[str]:
    data = json.loads((run_dir / "run_manifest.json").read_text())
    problems = []
    if data["data"].get("scene_manifest") != env["INSCENE_MANIFEST"]:
        problems.append("scene manifest path")
    if data["data"].get("scene_manifest_sha256") != env["INSCENE_MANIFEST_SHA256"]:
        problems.append("scene manifest sha")
    if data["extractor"].get("streaming_feat_root") not in (env["INSCENE_STREAMING_FEAT_ROOT"], None):
        problems.append("streaming root")
    return problems


def group_b2(packs: dict, rec: dict, problems: Counter) -> list[dict]:
    campaign = A / "single_gpu_b2_fp32_20261016_r1"
    runner_path = campaign / "run_single.py"
    runner = load_module(runner_path)
    items = items_by_key(runner.ITEM_ROOT)
    out = []
    for key, (item_path, titem) in items.items():
        dirs = sorted((RUNS / "main_v2_single_gpu_b2_fp32_20261016_r1/production").glob(f"*/{key}"))
        if len(dirs) != 1:
            problems["b2: run dir not unique"] += 1
            continue
        run_dir = dirs[0]
        info = runner.MODEL_PACKS[titem["model"]]
        receipt = rec.get(str(run_dir))
        if receipt is None:
            problems["b2: no TRAIN_COMPLETE receipt"] += 1
            continue
        bad = [n for n, ok in (("state", receipt.get("state") == "SUCCEEDED"),
                               ("item", receipt.get("item_sha256") == sha256(item_path)),
                               ("pack", receipt.get("source_sha_file_sha256") == info["source_sha_file_sha256"]),
                               ("overlay", receipt.get("overlay_sha256") == info["overlay_sha256"])) if not ok]
        if bad:
            problems["b2: receipt mismatch " + "/".join(bad)] += 1
            continue
        manifest = titem["execution_manifest"]
        env = {
            "PROBE_OWNER_OVERLAY_MANIFEST": str(info["overlay"]), "PROBE_OWNER_LOCAL_MIRROR_ROOT": MIRROR,
            "INSCENE_DATA_ROOT": DATA_ROOT, "INSCENE_LOCAL_SCENE_ROOT": SCENE_ROOT,
            "INSCENE_MANIFEST": manifest["path"], "INSCENE_MANIFEST_SHA256": manifest["sha256"],
            "INSCENE_STREAMING_FEAT_ROOT": OVERLAY_ROOT, "INSCENE_TARGET_FEAT_ROOT": OVERLAY_ROOT,
            "INSCENE_TARGET_INDEX_ROOT": f"{MIRROR}/PROBE816_OWNER_OVERLAY_ONLY",
            "INSCENE_TARGET_LAYOUT": "flat", "INSCENE_TRUST_EXPANDED_MANIFEST": "1",
            "INSCENE_PROBE_TASK": titem["probe"], "INSCENE_METADATA_CACHE_MAX_SCENES": "32",
            "PROBE_FP16_PASSTHROUGH": "0",
        }
        out.append(dict(group="b2_fp32_retrain", key=key, titem=titem, item_path=item_path, run_dir=run_dir,
                        env=env, env_family="overlay", pack=Path(info["pack"]).name,
                        overrides=list(titem["scientific_overrides"]),
                        overlay={"path": str(info["overlay"]), "sha256": info["overlay_sha256"]},
                        runner=runner_path, receipt=receipt, scope="canonical_v2_b2_fp32_retrain_20261016"))
    return out


def group_minimax_all(packs: dict, rec: dict, problems: Counter, inventory: list[dict]) -> list[dict]:
    runner_path = ROOT / "runtime/main_v2_probe_minimax_all_20261011_r1/run_single.py"
    runner = load_module(runner_path)
    items = items_by_key(runner.ITEM_ROOT)
    pack_sha = sha256(Path(runner.PACK) / "source.sha256")
    out = []
    for row in inventory:
        if row["scope"] != "canonical_v2_minimax_all_layers_20261011" or row["probe"] not in EVALUATOR:
            continue
        key = row["experiment_id"]
        if key not in items:
            problems["minimax_all: no training item"] += 1
            continue
        item_path, titem = items[key]
        run_dir = RUNS / "main_v2_probe_minimax_all_20261011_r1" / key
        receipt = rec.get(str(run_dir))
        if receipt is None:
            problems["minimax_all: no TRAIN_COMPLETE receipt"] += 1
            continue
        bad = [n for n, ok in (("state", receipt.get("state") == "SUCCEEDED"),
                               ("item", receipt.get("item_sha256") in (None, sha256(item_path))),
                               ("pack", receipt.get("source_sha_file_sha256") in (None, pack_sha))) if not ok]
        if bad:
            problems["minimax_all: receipt mismatch " + "/".join(bad)] += 1
            continue
        manifest = titem["execution_manifest"]
        env = {
            "INSCENE_DATA_ROOT": DATA_ROOT, "INSCENE_LOCAL_SCENE_ROOT": SCENE_ROOT,
            "INSCENE_MANIFEST": manifest["path"], "INSCENE_MANIFEST_SHA256": manifest["sha256"],
            "INSCENE_STREAMING_FEAT_ROOT": runner.STREAM_ROOT, "INSCENE_TARGET_FEAT_ROOT": runner.TARGET_ROOT,
            "INSCENE_TARGET_INDEX_ROOT": runner.TARGET_ROOT, "INSCENE_TARGET_LAYOUT": "flat",
            "INSCENE_TRUST_EXPANDED_MANIFEST": "1", "INSCENE_PROBE_TASK": titem["probe"],
            "INSCENE_METADATA_CACHE_MAX_SCENES": "32", "PROBE_FP16_PASSTHROUGH": "0",
            "PROBE_FAIL_ON_NONFINITE_LOSS": "1",
        }
        out.append(dict(group="minimax_all", key=key, titem=titem, item_path=item_path, run_dir=run_dir,
                        env=env, env_family="direct", pack=Path(runner.PACK).name,
                        overrides=list(titem["scientific_overrides"]), overlay=None, runner=runner_path,
                        receipt=receipt, scope=row["scope"], inventory=row))
    return out


def group_l30(packs: dict, problems: Counter, inventory: list[dict]) -> list[dict]:
    campaign = A / "minimax_l30_ab16_20260930_r2"
    ledger_path = campaign / "formal_submission_ledger.json"
    jobs = json.loads(ledger_path.read_text())["jobs"]
    by_case = {str(Path(job["case_path"]).parent): job for job in jobs.values()}
    out = []
    for row in inventory:
        if row["scope"] != "canonical_v2_minimax_l30_ab16":
            continue
        key = row["experiment_id"]
        ckpt = ROOT / row["checkpoint_path"]
        case_dir = str(ckpt.parents[7])  # cases/p<P>/<probe>/formal_output/inscene15k_streaming/runs/run/VideoProbe3D/<id>/checkpoints
        job = by_case.get(case_dir)
        if job is None:
            problems["l30: no formal ledger entry"] += 1
            continue
        words = shlex.split(job["command"])
        pack_source = Path(words[words.index("cd") + 1])
        env_start = words.index("env") + 1
        train = words.index("vidfm3d/train.py")
        env_raw = dict(w.split("=", 1) for w in words[env_start:train - 1])
        overrides = [o for o in words[train + 1:] if not EXECUTION_ONLY.match(o)]
        env = {k: v for k, v in env_raw.items() if k not in ENV_EXECUTION_ONLY}
        # The training command did not set PROBE_FP16_PASSTHROUGH; this pack's default is 0.
        env.setdefault("PROBE_FP16_PASSTHROUGH", "0")
        manifest = Path(env["INSCENE_MANIFEST"])
        if sha256(manifest) != env["INSCENE_MANIFEST_SHA256"]:
            problems["l30: scene manifest changed"] += 1
            continue
        case_path = Path(job["case_path"])
        case = json.loads(case_path.read_text())
        out.append(dict(group="minimax_l30_ab16", key=key, titem={
            "experiment_config": dict(o.split("=", 1) for o in overrides if "=" in o)["experiment"],
            "execution_manifest": {"path": str(manifest), "sha256": env["INSCENE_MANIFEST_SHA256"]},
            "model": row["model"], "probe": row["probe"], "layer": row["layer"], "prefix": row["prefix"],
            "case_config_sha256": case.get("config_sha256")},
            item_path=case_path, run_dir=ckpt.parents[3], env=env, env_family="direct_l30",
            pack=pack_source.parent.name,
            overrides=[o for o in overrides if not o.startswith("experiment=")], overlay=None,
            runner=ledger_path, receipt={"formal_job_ids": job.get("job_ids"), "state": job.get("state"),
                                         "returncode": job.get("returncode")},
            scope=row["scope"], inventory=row))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    inventory = list(csv.DictReader(INVENTORY.open(newline="", encoding="utf-8")))
    packs = {}
    for d in (ROOT / "runtime").glob("main_v2_eval_v2r2_from_*"):
        manifest = json.loads((d / "EVAL_PACK.json").read_text())
        packs[manifest["base_pack"].split("/", 1)[1]] = {
            "name": d.name, "source_sha256_file_sha256": sha256(d / "source.sha256")}
    rec, problems = receipts(), Counter()
    selected = group_b2(packs, rec, problems) + group_minimax_all(packs, rec, problems, inventory) + \
        group_l30(packs, problems, inventory)
    items = []
    for sel in selected:
        key, titem, run_dir = sel["key"], sel["titem"], sel["run_dir"]
        try:
            ckpt = final_checkpoint(run_dir)
        except (OSError, ValueError) as exc:
            problems[f"{sel['group']}: {type(exc).__name__}"] += 1
            continue
        inventory_note = None
        if sel.get("inventory") and Path(sel["inventory"]["checkpoint_path"]).name != ckpt.name:
            recorded = ROOT / sel["inventory"]["checkpoint_path"]
            if recorded.parent != ckpt.parent:
                problems[f"{sel['group']}: checkpoint differs from inventory"] += 1
                continue
            # Same run and checkpoint folder: the inventory row was taken while the
            # run was still training (2026-10-15); the receipt says it completed.
            inventory_note = f"inventory checkpoint {recorded.name} superseded by final {ckpt.name}"
        bad = check_manifest(run_dir, sel["env"])
        if bad:
            problems[f"{sel['group']}: run_manifest disagrees: {'/'.join(bad)}"] += 1
            continue
        pack = packs.get(sel["pack"])
        if pack is None:
            problems[f"{sel['group']}: no eval pack for {sel['pack']}"] += 1
            continue
        config = ROOT / "runtime" / pack["name"] / "source/configs/experiment" / f"{titem['experiment_config']}.yaml"
        config_sha = sha256(config)
        expected = titem.get("experiment_config_sha256") or titem.get("case_config_sha256")
        if expected and expected != config_sha:
            problems[f"{sel['group']}: experiment config differs"] += 1
            continue
        probe = titem["probe"]
        evaluator, metric = EVALUATOR[probe]
        st = ckpt.stat()
        items.append({
            "schema": "MAIN_V2_EVAL_V2_ITEM_V1", "unique_key": key, "model": titem["model"], "probe": probe,
            "layer": str(titem["layer"]), "prefix": str(titem["prefix"]), "scope": sel["scope"],
            "group": sel["group"], "evaluator": evaluator, "primary_metric": metric,
            "inventory_correctness_status": (sel.get("inventory") or {}).get("correctness_status", "RETRAIN"),
            "inventory_checkpoint_tensor_status": (sel.get("inventory") or {}).get("checkpoint_tensor_status", ""),
            "experiment_config": titem["experiment_config"], "experiment_config_sha256": config_sha,
            "scientific_overrides": sel["overrides"],
            "execution_manifest": {"path": titem["execution_manifest"]["path"],
                                   "sha256": titem["execution_manifest"]["sha256"]},
            "overlay": sel["overlay"], "env_family": sel["env_family"], "env": sel["env"],
            "training_item": {"path": str(sel["item_path"]), "sha256": sha256(sel["item_path"])},
            "training_runner": {"path": str(sel["runner"]), "sha256": sha256(sel["runner"])},
            "training_receipt": {k: v for k, v in sel["receipt"].items()
                                 if k in ("_path", "state", "overlay_sha256", "item_sha256", "source_sha_file_sha256",
                                          "completed_utc", "formal_job_ids", "returncode")},
            "training_pack": sel["pack"], "training_code_tag": f"code/{sel['pack']}",
            "eval_pack": pack,
            "checkpoint": {"path": str(ckpt), "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns},
            **({"inventory_note": inventory_note} if inventory_note else {}),
        })
    dup =[k for k, n in Counter(i["unique_key"] for i in items).items() if n > 1]
    if dup:
        raise SystemExit(f"duplicate keys: {dup[:5]}")
    print(json.dumps({"items": len(items), "problems": dict(problems),
                      "by_group": dict(Counter(i["group"] for i in items)),
                      "by_probe": dict(Counter(i["probe"] for i in items)),
                      "by_eval_pack": dict(Counter(i["eval_pack"]["name"] for i in items))}, indent=1, sort_keys=True))
    if not args.write:
        return 0
    item_dir = CAMPAIGN / "items"
    item_dir.mkdir(parents=True, exist_ok=True)
    plan_path = CAMPAIGN / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {"schema": "MAIN_V2_EVAL_V2_PLAN_V1", "items": {}}
    for item in sorted(items, key=lambda i: i["unique_key"]):
        key = item["unique_key"]
        text = json.dumps(item, sort_keys=True) + "\n"
        existing = plan["items"].get(key)
        path = Path(existing["path"]) if existing else item_dir / f"{len(plan['items']):04d}_{key}.jsonl"
        if path.exists():
            if path.read_text() != text:
                raise SystemExit(f"{path} exists with different content; items are immutable")
        else:
            path.write_text(text, encoding="utf-8")
            os.chmod(path, 0o444)
        plan["items"][key] = {"path": str(path), "sha256": sha256(path), "probe": item["probe"],
                              "model": item["model"], "eval_pack": item["eval_pack"]["name"]}
    plan["updated_utc"] = datetime.now(timezone.utc).isoformat()
    plan["runner"] = str(CAMPAIGN / "run_eval_single.py")
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(items)} items; plan has {len(plan['items'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
