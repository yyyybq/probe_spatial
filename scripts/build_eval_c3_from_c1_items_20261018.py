#!/usr/bin/env python3
"""Build C3-from-C1 evaluation items for every evaluated C1 checkpoint (main tree).

C3 (protocols/C_PROBES.md) is an evaluation of a frozen C1 one-step predictor: the
candidate actions of one anchor are the h1/h2/h4 targets of the same trajectory, and
eval_diag scores them with ``++evaluation_protocol=c3_action_sensitivity``.  These are
same-trajectory endpoints, a multi-action sensitivity diagnostic, not restored
counterfactual branches.

Sources (the C1 items already evaluated with v2):
  canonical  eval_v2_canonical_20261017_r2 (minus the 4 superseded by r3), r3, r4
  timestep   tstep_eval_v2_20261018_t1

Variants, from the horizons each checkpoint was trained with:
  c3_from_c1                 trained on h1/h2/h4; its own validation loader already
                             emits 3-candidate action sets (the protocol C3).
  h1_trained_diagnostic      trained on h1 only; evaluated with action_horizons=[1,2,4]
                             at evaluation time, so h2/h4 actions were never seen in
                             training.  Not the protocol C3; reported separately.

Code: the release main tree, frozen as runtime/main_eval_v2_from_main_206c5de.  Each item
keeps its source item's environment, overrides, training references and checkpoint, and
records the source item so the C3 predictions can be checked against the C1 run.

Items: artifacts/c3_from_c1_eval_20261018_c3/.  Default: report only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
CAMPAIGN = ROOT / "artifacts/c3_from_c1_eval_20261018_c3"
MAIN_PACK = "main_eval_v2_from_main_206c5de"
SOURCES = [
    ("r2", ROOT / "artifacts/main_v2_probe_v2_816_20260917/eval_v2_canonical_20261017_r2/plan.json", "canonical"),
    ("r3", ROOT / "artifacts/main_v2_probe_v2_816_20260917/eval_v2_canonical_20261017_r3/plan.json", "canonical"),
    ("r4", ROOT / "artifacts/main_v2_probe_v2_816_20260917/eval_v2_canonical_20261017_r4/plan.json", "canonical"),
    ("t1", ROOT / "artifacts/tstep_eval_v2_20261018_t1/plan.json", "timestep"),
]
SUMMARY = "val_c3_action_sensitivity_summary.json"
PROTOCOL = "++evaluation_protocol=c3_action_sensitivity"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trained_horizons(item: dict) -> list[int]:
    config = ROOT / "runtime" / item["eval_pack"]["name"] / "source/configs/experiment" / f"{item['experiment_config']}.yaml"
    value = re.search(r"^action_horizons:\s*\[([^\]]*)\]", config.read_text(), re.M)
    horizons = [int(v) for v in value.group(1).split(",") if v.strip()] if value else [1]
    for override in item["scientific_overrides"]:
        if override.startswith("action_horizons="):
            horizons = [int(v) for v in re.findall(r"\d+", override.split("=", 1)[1])]
    return horizons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    pack = ROOT / "runtime" / MAIN_PACK
    if json.loads((pack / "EVAL_PACK.json").read_text()).get("status") != "FROZEN":
        raise SystemExit(f"{MAIN_PACK} is not FROZEN")
    eval_pack = {"name": MAIN_PACK, "source_sha256_file_sha256": sha256(pack / "source.sha256")}
    r3_keys = set(json.loads(SOURCES[1][1].read_text())["items"])
    items, problems = [], Counter()
    for revision, plan_path, family in SOURCES:
        plan = json.loads(plan_path.read_text())["items"]
        for key, entry in sorted(plan.items()):
            if entry["probe"] != "action_dynamics" or (revision == "r2" and key in r3_keys):
                continue
            source_path = Path(entry["path"])
            if sha256(source_path) != entry["sha256"]:
                problems["source item changed"] += 1
                continue
            source = json.loads(source_path.read_text().splitlines()[0])
            ckpt = Path(source["checkpoint"]["path"])
            if not ckpt.is_file() or ckpt.stat().st_size != source["checkpoint"]["size_bytes"]:
                problems["checkpoint missing or changed"] += 1
                continue
            config = pack / "source/configs/experiment" / f"{source['experiment_config']}.yaml"
            if not config.is_file():
                problems["main tree lacks experiment config"] += 1
                continue
            main_config_sha = sha256(config)
            horizons = trained_horizons(source)
            variant = "c3_from_c1" if len(horizons) >= 2 else "h1_trained_diagnostic"
            overrides = [PROTOCOL] + ([] if variant == "c3_from_c1" else ["action_horizons=[1,2,4]"])
            item = {key_: value for key_, value in source.items()
                    if key_ not in ("schema", "probe", "primary_metric", "evaluator", "eval_pack",
                                    "experiment_config_sha256", "group")}
            item.update({
                "schema": "C3_FROM_C1_EVAL_ITEM_V1", "probe": "c3_from_c1", "c1_probe": "action_dynamics",
                "family": family, "c3_variant": variant, "trained_action_horizons": horizons,
                "evaluator": "diag", "primary_metric": "intervention_validity", "summary_name": SUMMARY,
                "evaluation_overrides": overrides, "eval_pack": eval_pack,
                "experiment_config_sha256": main_config_sha,
                "training_experiment_config_sha256": source["experiment_config_sha256"],
                "experiment_config_identical_to_training": main_config_sha == source["experiment_config_sha256"],
                "source_c1_eval": {"revision": revision, "path": str(source_path), "sha256": entry["sha256"]},
            })
            if not item["experiment_config_identical_to_training"]:
                problems[f"{family}: main-tree config differs from training config (recorded)"] += 1
            items.append(item)
    keys = Counter(i["unique_key"] for i in items)
    if any(n > 1 for n in keys.values()):
        raise SystemExit(f"duplicate keys: {[k for k, n in keys.items() if n > 1][:5]}")
    print(json.dumps({"items": len(items), "notes_and_problems": dict(problems),
                      "by_family_variant": dict(Counter(f"{i['family']}|{i['c3_variant']}" for i in items)),
                      "by_source": dict(Counter(i["source_c1_eval"]["revision"] for i in items)),
                      "by_model": dict(Counter(i["model"] for i in items))}, indent=1, sort_keys=True))
    if not args.write:
        return 0
    item_dir = CAMPAIGN / "items"
    item_dir.mkdir(parents=True, exist_ok=True)
    plan_path = CAMPAIGN / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {"schema": "C3_FROM_C1_EVAL_PLAN_V1", "items": {}}
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
                              "model": item["model"], "eval_pack": MAIN_PACK, "c3_variant": item["c3_variant"]}
    plan["updated_utc"] = datetime.now(timezone.utc).isoformat()
    plan["runner"] = str(CAMPAIGN / "run_eval_single.py")
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(items)} items; plan has {len(plan['items'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
