#!/usr/bin/env python3
"""Stage-5 config parity: new MiniMax L20/L25 runs vs the 128 runs they replace.

For every TSTEP_MINIMAX_L20_L25_FEATURE_OVERFLOW row of
metadata/release_inventory/retrain_queue_20261016.csv, compose the Hydra
config exactly as scripts/run_tstep_probe_group.sh would for stage 5 (bf16
namespaces, run id tstep_probe_minimax_bf16_v1_r1) and compare it, unresolved
and flattened, with the old run's saved .hydra/config.yaml.  Keys that must
differ (run naming, paths, loggers, tags, autoresume, feature roots, hydra
internals) are ignored; ``ckpt_path`` is reported separately because some old
runs were resumed from an earlier partial run.  The feature-transfer setting
(PROBE_FP16_PASSTHROUGH) is an environment variable, not part of the config.

Exit 0 when no cell differs outside the ignored keys and ckpt_path.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "metadata/release_inventory/retrain_queue_20261016.csv"
QUEUE_GROUP = "TSTEP_MINIMAX_L20_L25_FEATURE_OVERFLOW"
STREAMING_NS = "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1"
TARGET_NS = "FEAT_TARGET_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1"
RUN_ID = "tstep_probe_minimax_bf16_v1_r1"
IGNORED = ("paths.", "logger.", "job_name", "tags", "autoresume", "streaming_feat_root", "target_feat_root",
           "callbacks.model_checkpoint.dirpath", "hydra")
BATCH1_TASKS = ("streaming_depth", "streaming_movement", "streaming_mask")  # as run_tstep_probe_group.sh
TARGET_TASKS = ("action_dynamics", "path_integration", "counterfactual")


def flat(value, prefix: str = "") -> dict:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flat(item, f"{prefix}{key}."))
        return out
    if isinstance(value, list):
        return {prefix[:-1]: json.dumps(value, sort_keys=True, default=str)}
    return {prefix[:-1]: value}


def overrides(task: str, t: str, layer: str, prefix: str) -> list[str]:
    bs = "1" if task in BATCH1_TASKS else "2"
    tag = f"tp_minimax_h3_t{t}_l{layer}_p{prefix}_{task}"
    ov = [
        f"experiment=inscene15k_streaming/{task}_minimax_h3_v1", "vfm_name=minimax_h3", "video_channels=5376",
        f"streaming_feat_root=aoss://baiqiao/InsScene-15K/{STREAMING_NS}", f"feature_layer={layer}",
        f"feature_timestep={t}", f"feat_postfix=_t{t}_layer{layer}", f"prefix_len={prefix}", f"job_name={tag}",
        f"paths.run_folder_name=tstep_probe_seeded_v1/{RUN_ID}/{tag}", f"logger.wandb.name={RUN_ID}_{tag}",
        f"logger.wandb.group={RUN_ID}", "logger.wandb.offline=true", "autoresume=true", "trainer.devices=[0]",
        f"data.data_module.batch_size_per_device={bs}", f"data.data_module.batch_size_per_device_val={bs}",
        "data.data_module.num_workers=4", "data.data_module.num_workers_val=2",
    ]
    if task in TARGET_TASKS:
        ov.append(f"target_feat_root=aoss://baiqiao/InsScene-15K/{TARGET_NS}")
    return ov


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = [r for r in csv.DictReader(QUEUE.open(newline="", encoding="utf-8")) if r["queue_group"] == QUEUE_GROUP]
    cells, resumed, missing = {}, {}, []
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        for row in rows:
            task, t, layer, prefix = row["probe"], row["timestep"], row["layer"], row["prefix"]
            tag = f"tp_minimax_h3_t{t}_l{layer}_p{prefix}_{task}"
            old_cfg = ROOT / row["old_output_path"] / ".hydra/config.yaml"
            if not old_cfg.is_file():
                missing.append(str(old_cfg))
                continue
            new = flat(OmegaConf.to_container(compose(config_name="train", overrides=overrides(task, t, layer, prefix)),
                                              resolve=False))
            old = flat(OmegaConf.to_container(OmegaConf.load(old_cfg), resolve=False))
            diff = {k: [old.get(k, "<absent>"), new.get(k, "<absent>")] for k in sorted(set(new) | set(old))
                    if not k.startswith(IGNORED) and str(old.get(k, "<absent>")) != str(new.get(k, "<absent>"))}
            if "ckpt_path" in diff:
                resumed[tag] = diff.pop("ckpt_path")[0]
            if diff:
                cells[tag] = diff
    summary = {
        "schema": "TSTEP_MINIMAX_BF16_V1_CONFIG_PARITY_V1", "queue": str(QUEUE), "cells": len(rows),
        "cells_with_config_diffs": len(cells), "old_runs_resumed_from_checkpoint": len(resumed),
        "missing_old_config": missing, "ignored_key_prefixes": list(IGNORED),
        "old_campaigns": dict(Counter(r["old_output_path"].split("/")[4] for r in rows)),
        "diffs": cells, "resumed_old_runs": resumed,
        "status": "PASS" if not cells and not missing and rows else "FAIL",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("status", "cells", "cells_with_config_diffs",
                                              "old_runs_resumed_from_checkpoint", "old_campaigns")}, indent=1))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
