#!/usr/bin/env python3
"""Find real manifest samples with invalid B supervision or an empty identity frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]


def dataset_for(source_root: pathlib.Path, task: str, prefix: int):
    sys.path.insert(0, str(source_root))
    manifest = ROOT / f"artifacts/tstep_canonical_v2/manifests/inscene_c_h124_prefix{prefix}.jsonl"
    os.environ.update(
        PROJECT_ROOT=str(source_root),
        INSCENE_DATA_ROOT="/mnt/umm/users/yinbaiqiao/InsScene-15K/data",
        INSCENE_LOCAL_SCENE_ROOT="/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2",
        INSCENE_MANIFEST=str(manifest),
        INSCENE_MANIFEST_SHA256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        INSCENE_CANONICAL_PROTOCOL="inscene_streaming_canonical_v2_obs28_h124",
        INSCENE_CANONICAL_OBS_LEN="28",
        INSCENE_STREAMING_EXCLUSIONS=str(source_root / "configs/experiment/inscene15k_streaming/tstep_canonical_v2_exclusions.txt"),
        INSCENE_STREAMING_FEAT_ROOT="aoss://baiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_SEEDED_V1",
        INSCENE_TARGET_FEAT_ROOT="aoss://baiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_SEEDED_V1",
        INSCENE_TARGET_INDEX_ROOT="/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_SEEDED_V1",
        INSCENE_TARGET_LAYOUT="flat",
        INSCENE_TRUST_EXPANDED_MANIFEST="1",
        INSCENE_PROBE_TASK=task,
        INSCENE_METADATA_CACHE_MAX_SCENES="32",
        INSCENE_LABEL_CACHE_ROOT=str(ROOT / "artifacts/tstep_probe_opt_v1/edge_label_sidecar"),
    )
    from hydra import compose, initialize_config_dir
    from vidfm3d.data.components.inscene15k_dataset import InsScene15KDataset
    overrides = [
        f"experiment=inscene15k_streaming/{task}_wan_v1", "vfm_name=wan",
        "video_channels=1536", "feature_layer=15", "feature_timestep=999",
        "feat_postfix=_t999_layer15", f"prefix_len={prefix}", "job_name=edge_scan",
        "paths.run_folder_name=edge_scan", "logger.wandb.offline=true", "trainer.devices=[0]",
        f"data.data_module.batch_size_per_device={1 if task == 'streaming_mask' else 2}",
        f"data.data_module.batch_size_per_device_val={1 if task == 'streaming_mask' else 2}",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=str(source_root / "configs")):
        cfg = compose(config_name="train.yaml", overrides=overrides)
    expression = cfg.data.data_module.train_datasets[0]
    return eval(str(expression), {"InsScene15KDataset": InsScene15KDataset})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=pathlib.Path)
    parser.add_argument("--prefix", type=int, default=8)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    import torch

    hidden = dataset_for(args.source_root.resolve(), "ego_belief", args.prefix)
    invalid_hidden = None
    for index in range(min(args.limit, len(hidden))):
        scene = hidden.scenes[index]
        if hidden._shared_streaming_hidden_obj_id(scene) is None:
            invalid_hidden = {"index": index, "sample_id": scene.get("manifest_sample_id", scene.get("sample_id"))}
            break

    masks = dataset_for(args.source_root.resolve(), "streaming_mask", args.prefix)
    empty_mask = None
    for index in range(min(args.limit, len(masks))):
        scene = masks.scenes[index]
        selected = torch.as_tensor(scene["streaming_indices"], dtype=torch.long)
        identity = masks._load_identity_masks_scene(scene, selected)
        empty_frames = torch.nonzero((identity <= 0).all(dim=(1, 2))).flatten().tolist()
        if empty_frames:
            empty_mask = {
                "index": index, "empty_frames": empty_frames,
                "sample_id": scene.get("manifest_sample_id", scene.get("sample_id")),
            }
            break
    report = {
        "schema": "tstep-probe-edge-scan-v1", "prefix": args.prefix,
        "limit": args.limit, "invalid_hidden": invalid_hidden, "empty_mask": empty_mask,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if invalid_hidden is not None and empty_mask is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
