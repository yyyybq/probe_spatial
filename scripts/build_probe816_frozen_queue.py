#!/usr/bin/env python3
"""Freeze the exact 816-experiment scientific protocol into a runnable queue."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_EXPERIMENTS_SHA = "e0c035595fd7c11af7dbb928717de2518ddd2e52d97424658964aeab19ae4470"
CHANNELS = {
    "cogvideox": 3072,
    "dino": 1024,
    "f3r": 1024,
    "lingbot": 2048,
    "minimax_h3": 5376,
    "qwen2_5_vl": 3584,
    "videomae": 1024,
    "vjepa2": 1024,
}
TIMESTEP = {model: (250 if model in {"lingbot", "minimax_h3"} else 749) for model in CHANNELS}
DISK_LAYER = {("lingbot", 48): 47, ("minimax_h3", 50): 49}
TARGET_PROBES = {"action_dynamics", "path_integration", "counterfactual"}
RANK = {
    "streaming_depth": 0,
    "streaming_mask": 1,
    "streaming_camera": 2,
    "view_consistency": 3,
    "ego_belief": 4,
    "ego_belief_v2": 5,
    "action_dynamics": 6,
    "path_integration": 7,
    "counterfactual": 8,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def config_for(source: Path, row: dict[str, str]) -> tuple[str, list[str], str]:
    model, probe = row["model"], row["probe_config"]
    extra: list[str] = []
    provenance = "campaign_train_config"
    if model == "qwen2_5_vl":
        if probe == "streaming_mask":
            stem = "direct_vlm_streaming_depth_v1"
            extra = [
                "model.probe.active_heads=[identity]",
                "model.loss_weights.dmap=0.0",
                "model.loss_weights.camera=0.0",
                "model.loss_weights.identity=1.0",
            ]
            provenance = "historical_hydra_override_reconstruction"
        else:
            stem = f"direct_vlm_{probe}_v1"
            provenance = "historical_direct_vlm_mapping"
    elif not row["train_config"] and probe == "streaming_mask" and model in {"dino", "f3r"}:
        stem = f"streaming_depth_{model}_v1"
        extra = [
            "model.probe.active_heads=[identity]",
            "model.loss_weights.dmap=0.0",
            "model.loss_weights.camera=0.0",
            "model.loss_weights.identity=1.0",
        ]
        provenance = "historical_hydra_override_reconstruction"
    else:
        stem = Path(row["train_config"]).stem
    relative = f"configs/experiment/inscene15k_streaming/{stem}.yaml"
    path = source / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing regular experiment config for {row['experiment_id']}: {path}")
    return stem, extra, provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError(f"output exists: {args.output_dir}")
    if sha256(args.experiments) != EXPECTED_EXPERIMENTS_SHA:
        raise ValueError("experiments SHA mismatch")
    rows = list(csv.DictReader(args.experiments.open(newline="", encoding="utf-8")))
    if len(rows) != 816 or len({row["experiment_id"] for row in rows}) != 816:
        raise ValueError("expected exactly 816 unique experiments")
    manifest_sha = sha256(args.canonical_manifest)
    if manifest_sha != "ebc496eb0587cda372ccf29a7c5244ef454096151f73c8eed9511967ded1902d":
        raise ValueError("canonical manifest SHA mismatch")

    queue: list[dict[str, Any]] = []
    config_bindings: dict[str, dict[str, Any]] = {}
    for row in rows:
        model = row["model"]
        probe = row["probe_config"]
        layer = int(row["layer"])
        prefix = int(row["prefix"])
        if model not in CHANNELS or probe not in RANK or row["seed"] != "42":
            raise ValueError(f"unsupported experiment row: {row['experiment_id']}")
        stem, extra, provenance = config_for(args.source, row)
        config_path = args.source / "configs" / "experiment" / "inscene15k_streaming" / f"{stem}.yaml"
        config_bindings.setdefault(stem, {
            "path": str(config_path.resolve()),
            "sha256": sha256(config_path),
        })
        disk_layer = DISK_LAYER.get((model, layer), layer)
        key = f"{model}__scannetpp__t{TIMESTEP[model]}__l{layer}__p{prefix}__{probe}__seed42"
        scientific_overrides = [
            "seed=42",
            f"vfm_name={model}",
            f"video_channels={CHANNELS[model]}",
            f"feature_layer={layer}",
            f"feature_timestep={TIMESTEP[model]}",
            f"feat_postfix=_t{TIMESTEP[model]}_layer{disk_layer}",
            f"prefix_len={prefix}",
            "data.data_module.batch_size_per_device=2",
            "data.data_module.batch_size_per_device_val=2",
            "trainer.devices=[0]",
            "trainer.num_nodes=1",
            "autoresume=false",
        ] + extra
        queue.append({
            "schema": "CANONICAL_V2_PROBE816_FROZEN_QUEUE_ITEM_V1",
            "experiment_id": row["experiment_id"],
            "unique_key": key,
            "model": model,
            "layer": layer,
            "disk_layer": disk_layer,
            "timestep": TIMESTEP[model],
            "prefix": prefix,
            "probe": probe,
            "probe_family": row["probe_family"],
            "atomic_probe_outputs": row["atomic_probe_outputs"].split(","),
            "seed": 42,
            "requires_target": probe in TARGET_PROBES,
            "experiment_config": f"inscene15k_streaming/{stem}",
            "experiment_config_sha256": config_bindings[stem]["sha256"],
            "config_provenance": provenance,
            "scientific_overrides": scientific_overrides,
            "protocol": {
                "batch_size_per_device": 2,
                "batch_size_per_device_val": 2,
                "gradient_accumulation": "inherit_frozen_config",
                "optimizer_lr_budget_precision_validation_model_loss": "inherit_sha_bound_config",
                "canonical_manifest_sha256": manifest_sha,
                "sample_order_rule": "canonical_manifest_then_seeded_loader",
            },
            "old_run_artifact_path": row["run_artifact_path"],
            "old_checkpoint_path": row["checkpoint_path"],
            "old_training_status": row["training_status"],
            "sort_rank": RANK[probe],
        })

    source_rows = queue
    by_key: dict[str, list[dict[str, Any]]] = {}
    for row in source_rows:
        by_key.setdefault(row["unique_key"], []).append(row)
    queue = []
    aliases: list[dict[str, Any]] = []
    for key, group in by_key.items():
        signatures = {
            json.dumps(
                {
                    "model": row["model"], "layer": row["layer"], "disk_layer": row["disk_layer"],
                    "timestep": row["timestep"], "prefix": row["prefix"], "probe": row["probe"],
                    "seed": row["seed"], "config": row["experiment_config"],
                    "config_sha": row["experiment_config_sha256"],
                    "overrides": row["scientific_overrides"], "protocol": row["protocol"],
                },
                sort_keys=True,
            )
            for row in group
        }
        if len(signatures) != 1:
            raise ValueError(f"duplicate execution key has conflicting science: {key}")
        # A single atomic writer owns a scientifically identical execution.
        # Every source registry row remains explicitly accounted for below.
        owner = sorted(group, key=lambda row: row["experiment_id"])[0]
        owner["experiment_ids"] = sorted(row["experiment_id"] for row in group)
        owner["source_registry_rows"] = len(group)
        queue.append(owner)
        for row in group:
            aliases.append({
                "experiment_id": row["experiment_id"],
                "unique_key": key,
                "owner_experiment_id": owner["experiment_id"],
                "role": "owner" if row is owner else "duplicate_alias",
            })
    if len(queue) != 760 or len(aliases) != 816:
        raise ValueError(f"unexpected dedupe cardinality: units={len(queue)} aliases={len(aliases)}")
    queue.sort(key=lambda row: (row["sort_rank"], -CHANNELS[row["model"]], row["model"], row["prefix"], row["layer"], row["probe"]))
    for ordinal, row in enumerate(queue):
        row["queue_ordinal"] = ordinal

    args.output_dir.mkdir(parents=True)
    queue_path = args.output_dir / "queue_816.jsonl"
    with queue_path.open("w", encoding="utf-8") as handle:
        for row in queue:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    aliases_path = args.output_dir / "experiment_owner_map_816.jsonl"
    with aliases_path.open("w", encoding="utf-8") as handle:
        for row in sorted(aliases, key=lambda item: item["experiment_id"]):
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    manifest = {
        "schema": "CANONICAL_V2_PROBE816_FROZEN_QUEUE_V1",
        "status": "FROZEN_NOT_SUBMITTED",
        "experiments": {"path": str(args.experiments.resolve()), "sha256": sha256(args.experiments), "rows": 816},
        "canonical_manifest": {"path": str(args.canonical_manifest.resolve()), "sha256": manifest_sha},
        "source_root": str(args.source.resolve()),
        "queue": {"path": str(queue_path.resolve()), "sha256": sha256(queue_path), "rows": 760},
        "experiment_owner_map": {"path": str(aliases_path.resolve()), "sha256": sha256(aliases_path), "rows": 816},
        "config_bindings": dict(sorted(config_bindings.items())),
        "counts": {
            "models": dict(sorted(Counter(row["model"] for row in queue).items())),
            "probes": dict(sorted(Counter(row["probe"] for row in queue).items())),
            "target_required": sum(row["requires_target"] for row in queue),
            "source_registry_rows": 816,
            "unique_execution_units": 760,
            "duplicate_alias_rows": 56,
        },
        "policy": {
            "scientific_batch_size": 2,
            "order": "historically_longest_first_depth_mask_then_other_probes",
            "formal_submission_gate": "overlay_ready_and_equivalence_and_performance_canary_pass",
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
