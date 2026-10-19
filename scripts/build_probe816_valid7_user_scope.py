#!/usr/bin/env python3
"""Build the user-corrected seven-model probing queue for 2026-09-22."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


MODELS = (
    "cogvideox", "videomae", "vjepa2", "lingbot",
    "qwen2_5_vl", "dino", "f3r",
)
LAYERS = {
    "cogvideox": (0, 5, 10, 30, 35, 40),
    "videomae": (5, 11, 17, 23),
    "vjepa2": (5, 11, 17, 23),
    "lingbot": (10, 20, 30, 40, 48),
    "qwen2_5_vl": (-1,),
    "dino": (0,),
    "f3r": (24,),
}
TIMESTEPS = {
    "cogvideox": 749, "videomae": 749, "vjepa2": 749,
    "lingbot": 250, "qwen2_5_vl": 749, "dino": 749, "f3r": 749,
}
PREFIXES = (8, 12, 16, 24)
PROBES = (
    "streaming_depth", "streaming_mask", "view_consistency",
    "ego_belief_v2", "action_dynamics", "path_integration",
)
PROBE_META = {
    # The current A1 protocol is the established depth + camera-pose task.
    # The newer depth + camera-moment variant is deliberately out of scope.
    "streaming_depth": ("A1", ["A1-depth", "A1-camera-pose"], False),
    "streaming_mask": ("A1", ["A1-mask"], False),
    "view_consistency": ("A2", ["A2"], False),
    "ego_belief_v2": ("B2", ["B2"], False),
    "action_dynamics": ("C1", ["C1"], True),
    "path_integration": ("C2", ["C2"], True),
}
CONFIG_ROOT: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def config_name(model: str, probe: str) -> str:
    if probe == "action_dynamics":
        return f"inscene15k_streaming/action_dynamics_{model}_h124_v1"
    return f"inscene15k_streaming/{probe}_{model}_v1"


def config_sha(model: str, probe: str) -> str:
    path = CONFIG_ROOT / f"{config_name(model, probe)}.yaml"
    if not path.is_file():
        raise RuntimeError(f"missing formal experiment config: {path}")
    return sha256(path)


def replace_feature_overrides(values: list[str], layer: int, timestep: int) -> list[str]:
    result = []
    found_layer = found_postfix = False
    for value in values:
        if value.startswith("feature_layer="):
            result.append(f"feature_layer={layer}")
            found_layer = True
        elif value.startswith("feat_postfix="):
            result.append(f"feat_postfix=_t{timestep}_layer{layer}")
            found_postfix = True
        elif value.startswith("model.probe.active_heads=") or value.startswith("model.loss_weights."):
            # These are streaming-mask-only additions on its existing rows;
            # synthetic rows use their formal experiment config unchanged.
            continue
        else:
            result.append(value)
    if not found_layer or not found_postfix:
        raise RuntimeError("template lacks feature-layer overrides")
    return result


def bind_protocol_config(row: dict[str, Any]) -> None:
    """Bind protocol variants that must never inherit a legacy queue entry."""
    family, outputs, requires_target = PROBE_META[row["probe"]]
    row["probe_family"] = family
    row["atomic_probe_outputs"] = outputs
    row["requires_target"] = requires_target
    if row["probe"] != "action_dynamics":
        return
    row["experiment_config"] = config_name(row["model"], row["probe"])
    row["experiment_config_sha256"] = config_sha(row["model"], row["probe"])
    row["config_provenance"] = "c1_balanced_h124_v1_user_protocol_20260922"
    overrides = [
        value for value in row["scientific_overrides"]
        if not value.startswith("action_horizons=")
    ]
    overrides.append("action_horizons=[1,2,4]")
    row["scientific_overrides"] = overrides
    row["c1_train_sampling"] = {
        "total_samples": "N",
        "horizons": [1, 2, 4],
        "allocation": "seeded balanced partition; counts differ by at most one",
    }
    row["c1_validation_sampling"] = {
        "horizons": [1, 2, 4],
        "allocation": "all three horizons per canonical prefix",
    }


def synthetic_row(
    templates: dict[tuple[str, str, int, int], dict[str, Any]],
    model: str,
    layer: int,
    prefix: int,
    probe: str,
) -> dict[str, Any]:
    timestep = TIMESTEPS[model]
    if model == "cogvideox":
        template = templates[(model, probe, 20, prefix)]
    elif model in {"dino", "f3r"} and probe in {"action_dynamics", "path_integration"}:
        template = templates[(model, "streaming_depth", layer, prefix)]
    else:
        raise RuntimeError(f"no approved synthesis rule: {model}/{layer}/{prefix}/{probe}")
    row = copy.deepcopy(template)
    family, outputs, requires_target = PROBE_META[probe]
    unique_key = (
        f"{model}__scannetpp__t{timestep}__l{layer}__p{prefix}__{probe}__seed42"
    )
    experiment_id = f"user0922_{probe}_{model}_l{layer}_p{prefix}_s42"
    row.update({
        "schema": "CANONICAL_V2_PROBE816_USER_SCOPE_QUEUE_ITEM_V1",
        "unique_key": unique_key,
        "experiment_id": experiment_id,
        "experiment_ids": [experiment_id],
        "source_registry_rows": 1,
        "config_provenance": "user_scope_20260922_existing_formal_config",
        "model": model,
        "layer": layer,
        "disk_layer": layer,
        "timestep": timestep,
        "prefix": prefix,
        "probe": probe,
        "probe_family": family,
        "atomic_probe_outputs": outputs,
        "requires_target": requires_target,
        "experiment_config": config_name(model, probe),
        "experiment_config_sha256": config_sha(model, probe),
        "scientific_overrides": replace_feature_overrides(
            template["scientific_overrides"], layer, timestep
        ),
        "old_checkpoint_path": None,
        "old_run_artifact_path": None,
        "old_training_status": "NOT_PREVIOUSLY_REGISTERED",
    })
    return row


def main() -> None:
    global CONFIG_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-queue", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config-root", type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "experiment",
    )
    args = parser.parse_args()
    CONFIG_ROOT = args.config_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError(f"output exists: {output}")

    base_rows = [
        json.loads(line) for line in args.base_queue.read_text().splitlines()
        if line.strip()
    ]
    templates = {
        (row["model"], row["probe"], int(row["layer"]), int(row["prefix"])): row
        for row in base_rows
    }
    desired = [
        (model, layer, prefix, probe)
        for model in MODELS
        for layer in LAYERS[model]
        for prefix in PREFIXES
        for probe in PROBES
    ]
    rows: list[dict[str, Any]] = []
    synthetic = []
    for model, layer, prefix, probe in desired:
        existing = templates.get((model, probe, layer, prefix))
        if existing is None:
            row = synthetic_row(templates, model, layer, prefix, probe)
            synthetic.append(row["unique_key"])
        else:
            row = copy.deepcopy(existing)
            row["schema"] = "CANONICAL_V2_PROBE816_USER_SCOPE_QUEUE_ITEM_V1"
        bind_protocol_config(row)
        rows.append(row)

    keys = [row["unique_key"] for row in rows]
    if len(rows) != 528 or len(set(keys)) != 528:
        raise RuntimeError(f"expected 528 unique execution units, got {len(rows)}/{len(set(keys))}")
    if len(synthetic) != 88:
        raise RuntimeError(f"expected 88 scope-extension units, got {len(synthetic)}")
    if any(row["probe"] in {"ego_belief", "counterfactual"} for row in rows):
        raise RuntimeError("excluded probe leaked into queue")
    if any(
        row["model"] == "cogvideox" and row["layer"] in {15, 20, 25}
        for row in rows
    ):
        raise RuntimeError("excluded CogVideoX layer leaked into queue")
    if any(
        row["model"] in {"videomae", "vjepa2"} and row["layer"] == 0
        for row in rows
    ):
        raise RuntimeError("excluded layer zero leaked into queue")
    for row in rows:
        if row["probe"] == "action_dynamics":
            expected = config_name(row["model"], row["probe"])
            if row["experiment_config"] != expected:
                raise RuntimeError(f"legacy C1 config leaked into queue: {row['unique_key']}")
            if row["scientific_overrides"].count("action_horizons=[1,2,4]") != 1:
                raise RuntimeError(f"C1 h124 override is not exact: {row['unique_key']}")
        if "A1-camera" in row["atomic_probe_outputs"]:
            raise RuntimeError(f"ambiguous legacy A1 camera label leaked: {row['unique_key']}")

    for ordinal, row in enumerate(rows):
        row["queue_ordinal"] = ordinal
    owner_map = []
    for row in rows:
        owner_id = row["experiment_ids"][0]
        for index, experiment_id in enumerate(row["experiment_ids"]):
            owner_map.append({
                "experiment_id": experiment_id,
                "owner_experiment_id": owner_id,
                "role": "owner" if index == 0 else "alias",
                "unique_key": row["unique_key"],
            })
    if len(owner_map) != 576:
        raise RuntimeError(f"expected 576 source requests including aliases, got {len(owner_map)}")

    queue_path = output / "queue_528.jsonl"
    map_path = output / "experiment_owner_map_576.jsonl"
    keys_path = output / "unique_keys_528.txt"
    experiments_path = output / "experiments_528.csv"
    write_exclusive(
        queue_path,
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
    )
    write_exclusive(
        map_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in owner_map),
    )
    write_exclusive(keys_path, "".join(key + "\n" for key in keys))
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=("experiment_id", "probe_config", "model", "layer", "prefix", "seed"),
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "experiment_id": row["experiment_id"],
            "probe_config": row["probe"],
            "model": row["model"],
            "layer": row["layer"],
            "prefix": row["prefix"],
            "seed": row["seed"],
        })
    write_exclusive(experiments_path, csv_buffer.getvalue())
    by_model = Counter(row["model"] for row in rows)
    by_probe = Counter(row["probe"] for row in rows)
    stream_depth_units = by_probe["streaming_depth"]
    manifest = {
        "schema": "PROBE816_VALID7_USER_SCOPE_20260922_V2",
        "status": "FROZEN_NOT_SUBMITTED",
        "scope_sha256": canonical_sha({
            "models": MODELS, "layers": LAYERS, "prefixes": PREFIXES,
            "probes": PROBES, "seed": 42,
        }),
        "rules": {
            "excluded_probes": ["ego_belief", "counterfactual"],
            "excluded_cogvideox_layers": [15, 20, 25],
            "excluded_videomae_layers": [0],
            "excluded_vjepa2_layers": [0],
            "all_retained_layers_receive_all_six_execution_probes": True,
            "seven_scientific_outputs": [
                "A1-depth", "A1-camera-pose", "A1-mask", "A2", "B2", "C1", "C2"
            ],
            "excluded_a1_variant": "depth+camera-moment",
            "included_a1_variant": "depth+camera-pose",
            "c1_config_pattern": "action_dynamics_<model>_h124_v1",
            "c1_train_sampling": "total N; h1/h2/h4 seeded balanced partition",
        },
        "counts": {
            "execution_units": len(rows),
            "source_requests_including_aliases": len(owner_map),
            "duplicate_aliases": len(owner_map) - len(rows),
            "synthetic_units_from_existing_formal_configs": len(synthetic),
            "scientific_outputs": len(rows) + stream_depth_units,
            "by_model": dict(by_model),
            "by_probe": dict(by_probe),
        },
        "queue": {"path": str(queue_path), "sha256": sha256(queue_path)},
        "owner_map": {"path": str(map_path), "sha256": sha256(map_path)},
        "unique_keys": {"path": str(keys_path), "sha256": sha256(keys_path)},
        "experiments": {
            "path": str(experiments_path), "sha256": sha256(experiments_path),
            "rows": len(rows),
        },
        "base_queue": {
            "path": str(args.base_queue.resolve()), "sha256": sha256(args.base_queue)
        },
        "config_root": str(CONFIG_ROOT),
        "synthetic_unique_keys": synthetic,
    }
    write_exclusive(
        output / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
