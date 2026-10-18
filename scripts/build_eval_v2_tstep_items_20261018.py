#!/usr/bin/env python3
"""Build SHA-bound v2 evaluation items for the timestep (t-step) probe checkpoints.

Selection (metadata/release_inventory/timestep.csv and the retrain table):

  * runs trained by the frozen packs tstep_probe_opt_v3 / tstep_probe_pack_v4_1
    whose correctness status is NOT_VERIFIED (the DEGENERATE rows are the
    non-finite B2 runs and the MiniMax L20/L25 overflow runs, which are being
    replaced), for the trained probes A1 depth, A1 mask, A2, B1, B2, C1, C2;
  * the 2026-10-16 B2 fp32 retrains (tstep_probe_b2_fp32_20261016_r1), which ran
    from the main tree at e4e4886, in place of the non-finite B2 runs.

Not selected: the separately trained ``counterfactual`` runs (old C3 protocol;
the canonical C3 is an evaluation of C1 checkpoints that needs the main tree's
action-set dataset) and the 2026-09-07 seeded_v1 runs (no recoverable code).

Per run the item copies the run's own training overrides (.hydra/overrides.yaml)
minus execution-only keys, rebuilds the launcher's environment, and checks it
against the run's run_manifest.json (data root, scene manifest and its SHA-256,
streaming and target feature roots).  The evaluation pack is
runtime/tstep_eval_v2_from_<training pack> (training code + v2 evaluators).

Items: artifacts/tstep_eval_v2_20261018_t1/items/, plan: .../plan.json.
Items are immutable once written.  Default: report only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
CAMPAIGN = ROOT / "artifacts/tstep_eval_v2_20261018_t1"
INVENTORY = ROOT / "metadata/release_inventory/timestep.csv"
RETRAINS = ROOT / "metadata/release_inventory/retrain_submissions_20261016.csv"
EVALUATOR = {
    "streaming_depth": ("a1", "mean_loss_depth"),
    "streaming_mask": ("a1", "mean_loss_identity"),
    "view_consistency": ("diag", "balanced_acc"),
    "ego_belief": ("diag", "mean_az_err_deg"),
    "ego_belief_v2": ("diag", "mean_ang_err_deg"),
    "action_dynamics": ("diag", "global_R@1"),
    "path_integration": ("diag", "global_R@1"),
}
PACKS = {"tstep_probe_opt_v3": "tstep_eval_v2_from_tstep_probe_opt_v3",
         "tstep_probe_pack_v4_1": "tstep_eval_v2_from_tstep_probe_pack_v4_1",
         "main_e4e4886": "tstep_eval_v2_from_main_e4e4886"}
LAUNCHERS = {"tstep_probe_opt_v3": "runtime/tstep_probe_opt_v3/source/scripts/run_tstep_probe_single.sh",
             "tstep_probe_pack_v4_1": "runtime/tstep_probe_pack_v4_1/source/scripts/run_tstep_probe_single.sh"}
DROP = re.compile(r"^(job_name|paths\.|logger\.|autoresume|data\.data_module\.num_workers|ckpt_path)")
EXCLUSIONS = "configs/experiment/inscene15k_streaming/tstep_canonical_v2_exclusions.txt"
PROTOCOL, OBS_LEN = "inscene_streaming_canonical_v2_obs28_h124", "28"
MIRROR = "/mnt/umm/users/yinbaiqiao/InsScene-15K"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def overrides_of(run_dir: Path) -> list[str]:
    lines = (run_dir / ".hydra/overrides.yaml").read_text().splitlines()
    if not all(line.startswith("- ") for line in lines if line.strip()):
        raise ValueError(f"unexpected overrides.yaml layout in {run_dir}")
    return [line[2:] for line in lines if line.strip()]


def final_checkpoint(run_dir: Path) -> Path:
    found = sorted(run_dir.glob("VideoProbe3D/*/checkpoints/epoch=*.ckpt"))
    if len(found) != 1:
        raise ValueError(f"expected one final checkpoint under {run_dir}, found {len(found)}")
    return found[0]


def selection() -> list[dict]:
    rows = list(csv.DictReader(INVENTORY.open(newline="", encoding="utf-8")))
    chosen = []
    for row in rows:
        m = re.search(r"runs/(tstep_probe_opt_v3|tstep_probe_pack_v4_1)/([^/]+)/", row["checkpoint_path"] or "")
        if not m or row["correctness_status"] != "NOT_VERIFIED" or row["probe"] not in EVALUATOR:
            continue
        if row["checkpoint_tensor_status"].startswith("NONFINITE"):
            continue
        chosen.append({"row": row, "pack": m.group(1), "run_id": m.group(2),
                       "run_dir": ROOT / str(Path(row["checkpoint_path"]).parents[3]), "scope": m.group(1)})
    for row in csv.DictReader(RETRAINS.open(newline="", encoding="utf-8")):
        if row["campaign"] != "tstep_probe_b2_fp32_20261016_r1":
            continue
        chosen.append({"row": {**row, "correctness_status": "RETRAINED_FP32", "checkpoint_tensor_status": ""},
                       "pack": "main_e4e4886", "run_id": "tstep_probe_b2_fp32_20261016_r1",
                       "run_dir": ROOT / row["new_output_path"], "scope": "tstep_b2_fp32_retrain_20261016"})
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    eval_packs = {}
    for base, name in PACKS.items():
        pack = ROOT / "runtime" / name
        manifest = json.loads((pack / "EVAL_PACK.json").read_text())
        if manifest.get("status") != "FROZEN":
            raise SystemExit(f"{name} is not FROZEN")
        eval_packs[base] = {"name": name, "source_sha256_file_sha256": sha256(pack / "source.sha256"),
                            "base": manifest.get("base_git_tag") or manifest.get("base_pack")}
    items, problems = [], Counter()
    for sel in selection():
        row, run_dir, pack = sel["row"], sel["run_dir"], sel["pack"]
        model, probe = row["model"], row["probe"]
        layer, prefix, timestep = str(row["layer"]), str(row["prefix"]), str(row["timestep"])
        key = f"{model}__scannetpp__t{timestep}__l{layer}__p{prefix}__{probe}__seed42"
        try:
            overrides = overrides_of(run_dir)
            ckpt = final_checkpoint(run_dir)
            manifest = json.loads((run_dir / "run_manifest.json").read_text())
        except (OSError, ValueError) as exc:
            problems[f"run_dir_unreadable: {type(exc).__name__}"] += 1
            continue
        if not (run_dir / "TRAIN_COMPLETE").exists():
            problems["no_TRAIN_COMPLETE"] += 1
            continue
        scientific = [o for o in overrides if not DROP.match(o)]
        ov = dict(o.split("=", 1) for o in scientific if "=" in o)
        expected = {"vfm_name": model, "feature_layer": layer, "feature_timestep": timestep, "prefix_len": prefix,
                    "experiment": f"inscene15k_streaming/{probe}_{model}_v1"}
        if any(ov.get(k) != v for k, v in expected.items()):
            problems["overrides_disagree_with_inventory"] += 1
            continue
        data, extractor = manifest["data"], manifest["extractor"]
        scene_manifest = Path(data["scene_manifest"])
        if not scene_manifest.is_file() or sha256(scene_manifest) != data["scene_manifest_sha256"]:
            problems["scene_manifest_changed"] += 1
            continue
        stream_root = extractor["streaming_feat_root"]
        target_root = data.get("target_feature_root")
        if ov.get("streaming_feat_root") != stream_root or not target_root:
            problems["feature_roots_inconsistent"] += 1
            continue
        if "target_feat_root" in ov and ov["target_feat_root"] != target_root:
            problems["target_root_inconsistent"] += 1
            continue
        epack = eval_packs[pack]
        src = ROOT / "runtime" / epack["name"] / "source"
        exclusions = src / EXCLUSIONS
        config = src / "configs/experiment" / f"{ov['experiment']}.yaml"
        if not exclusions.is_file() or not config.is_file():
            problems["eval_pack_lacks_config_or_exclusions"] += 1
            continue
        env = {
            "INSCENE_DATA_ROOT": data["data_root"],
            "INSCENE_LOCAL_SCENE_ROOT": data["data_root"] + "/processed_scannetpp_v2/processed_scannetpp_v2",
            "INSCENE_MANIFEST": str(scene_manifest),
            "INSCENE_MANIFEST_SHA256": data["scene_manifest_sha256"],
            "INSCENE_CANONICAL_PROTOCOL": PROTOCOL, "INSCENE_CANONICAL_OBS_LEN": OBS_LEN,
            "INSCENE_STREAMING_EXCLUSIONS": str(exclusions),
            "INSCENE_STREAMING_FEAT_ROOT": stream_root, "INSCENE_TARGET_FEAT_ROOT": target_root,
            "INSCENE_TARGET_INDEX_ROOT": f"{MIRROR}/{target_root.rstrip('/').rsplit('/', 1)[1]}",
            "INSCENE_TARGET_LAYOUT": "flat", "INSCENE_TRUST_EXPANDED_MANIFEST": "1",
            "INSCENE_METADATA_CACHE_MAX_SCENES": "32",
            # Packs ran with fp16 pass-through and the per-probe fast paths; the
            # B2 retrains ran the main tree with features widened to fp32.
            "PROBE_FP16_PASSTHROUGH": "0" if pack == "main_e4e4886" else "1",
        }
        if pack != "main_e4e4886":
            env["INSCENE_PROBE_TASK"] = probe
        evaluator, metric = EVALUATOR[probe]
        launcher = ROOT / LAUNCHERS[pack] if pack in LAUNCHERS else ROOT / "scripts/run_tstep_probe_group.sh"
        st = ckpt.stat()
        items.append({
            "schema": "TSTEP_EVAL_V2_ITEM_V1", "unique_key": key, "model": model, "probe": probe,
            "layer": layer, "prefix": prefix, "timestep": timestep, "scope": sel["scope"],
            "training_run_id": sel["run_id"],
            "inventory_correctness_status": row["correctness_status"],
            "inventory_checkpoint_tensor_status": row.get("checkpoint_tensor_status", ""),
            "evaluator": evaluator, "primary_metric": metric,
            "experiment_config": ov["experiment"], "experiment_config_sha256": sha256(config),
            "scientific_overrides": scientific,
            "execution_manifest": {"path": str(scene_manifest), "sha256": data["scene_manifest_sha256"]},
            "overlay": None, "env_family": "tstep_direct", "env": env,
            # No training item JSON exists for timestep runs; the run's own Hydra
            # override record is the training-time binding.
            "training_item": {"path": str(run_dir / ".hydra/overrides.yaml"),
                              "sha256": sha256(run_dir / ".hydra/overrides.yaml")},
            "training_runner": {"path": str(launcher), "sha256": sha256(launcher)},
            "training_run_manifest": {"path": str(run_dir / "run_manifest.json"),
                                      "sha256": sha256(run_dir / "run_manifest.json")},
            "training_pack": pack,
            "training_code_tag": f"code/{pack}" if pack in LAUNCHERS else "commit e4e4886",
            "eval_pack": {"name": epack["name"], "source_sha256_file_sha256": epack["source_sha256_file_sha256"]},
            "checkpoint": {"path": str(ckpt), "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns},
        })
    keys = Counter(i["unique_key"] for i in items)
    duplicates = [k for k, n in keys.items() if n > 1]
    if duplicates:
        raise SystemExit(f"duplicate unique keys: {duplicates[:5]}")
    report = {"items": len(items), "problems": dict(problems),
              "by_scope": dict(Counter(i["scope"] for i in items)),
              "by_probe": dict(Counter(i["probe"] for i in items)),
              "by_model": dict(Counter(i["model"] for i in items)),
              "by_eval_pack": dict(Counter(i["eval_pack"]["name"] for i in items))}
    print(json.dumps(report, indent=1, sort_keys=True))
    if not args.write:
        return 0
    item_dir = CAMPAIGN / "items"
    item_dir.mkdir(parents=True, exist_ok=True)
    plan_path = CAMPAIGN / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {"schema": "TSTEP_EVAL_V2_PLAN_V1", "items": {}}
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
