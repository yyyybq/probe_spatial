#!/usr/bin/env python3
"""One-process old/new canary worker for a single frozen probe experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
CHANNELS = {"wan": 1536, "cogvideox": 3072, "minimax_h3": 5376}
REQUIRED = {
    "streaming_depth": {"vfm_feat", "vfm_idx", "image", "cmaps", "dmaps", "intrinsics", "extrinsics"},
    "streaming_mask": {"vfm_feat", "vfm_idx", "image", "cmaps", "identity_ids"},
    "view_consistency": {"vfm_feat", "overlap_gt"},
    "ego_belief": {"vfm_feat", "hidden_obj_valid", "hidden_obj_mask", "hidden_obj_polar"},
    "ego_belief_v2": {"vfm_feat", "hidden_obj_valid", "belief_query_feat", "hidden_obj_polar"},
    "action_dynamics": {"input_feat", "action", "target_feat", "dyn_valid"},
    "path_integration": {"input_feat_seq", "path_actions", "target_feat_seq", "path_horizon_valid"},
    "counterfactual": {"input_feat_seq", "counterfactual_actions", "target_feat_seq", "counterfactual_valid"},
}


def tensor_digest(tensor) -> dict:
    value = tensor.detach().cpu().contiguous()
    import torch
    raw_view = value.reshape(-1).view(torch.uint8).numpy()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(raw_view).hexdigest(),
        "bytes": value.numel() * value.element_size(),
    }


def parameter_digest(module) -> str:
    """Hash trainable model state only; exclude mutable metric/logging buffers."""
    digest = hashlib.sha256()
    for name, value in sorted(module.named_parameters()):
        if not value.requires_grad:
            continue
        digest.update(name.encode())
        import torch
        byte_view = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(byte_view)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-root", required=True, type=pathlib.Path)
    parser.add_argument("--implementation-name", required=True)
    parser.add_argument("--task", required=True, choices=sorted(REQUIRED))
    parser.add_argument("--backend", required=True, choices=sorted(CHANNELS))
    parser.add_argument("--timestep", required=True, type=int)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--prefix", required=True, type=int)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--parameter-output", type=pathlib.Path)
    parser.add_argument("--train-steps", type=int, default=3)
    parser.add_argument("--empty-mask-fixture", action="store_true")
    args = parser.parse_args()

    impl = args.implementation_root.resolve()
    sys.path.insert(0, str(impl))
    os.environ["PROJECT_ROOT"] = str(impl)
    manifest = ROOT / f"artifacts/tstep_canonical_v2/manifests/inscene_c_h124_prefix{args.prefix}.jsonl"
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    os.environ.update(
        INSCENE_DATA_ROOT="/mnt/umm/users/yinbaiqiao/InsScene-15K/data",
        INSCENE_LOCAL_SCENE_ROOT="/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2",
        INSCENE_MANIFEST=str(manifest),
        INSCENE_MANIFEST_SHA256=manifest_sha,
        INSCENE_CANONICAL_PROTOCOL="inscene_streaming_canonical_v2_obs28_h124",
        INSCENE_CANONICAL_OBS_LEN="28",
        INSCENE_STREAMING_EXCLUSIONS=str(impl / "configs/experiment/inscene15k_streaming/tstep_canonical_v2_exclusions.txt"),
        INSCENE_STREAMING_FEAT_ROOT="aoss://baiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_SEEDED_V1",
        INSCENE_TARGET_FEAT_ROOT="aoss://baiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_SEEDED_V1",
        INSCENE_TARGET_INDEX_ROOT="/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_SEEDED_V1",
        INSCENE_TARGET_LAYOUT="flat",
        INSCENE_TRUST_EXPANDED_MANIFEST="1",
        INSCENE_PROBE_TASK=args.task,
        INSCENE_METADATA_CACHE_MAX_SCENES="32",
        PROBE_FP16_PASSTHROUGH="1",
    )
    if args.implementation_name == "optimized":
        os.environ.setdefault(
            "INSCENE_LABEL_CACHE_ROOT",
            str(ROOT / "artifacts/tstep_probe_opt_v1/canary_label_sidecar"),
        )
        os.environ["AOSS_FS_METRICS_MODE"] = "aggregate"
    else:
        os.environ.pop("INSCENE_LABEL_CACHE_ROOT", None)
        os.environ.pop("AOSS_FS_METRICS_MODE", None)

    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data._utils.collate import default_collate
    from vidfm3d.data.components.inscene15k_dataset import InsScene15KDataset

    overrides = [
        f"experiment=inscene15k_streaming/{args.task}_{args.backend}_v1",
        f"vfm_name={args.backend}", f"video_channels={CHANNELS[args.backend]}",
        "streaming_feat_root=aoss://baiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_SEEDED_V1",
        f"feature_layer={args.layer}", f"feature_timestep={args.timestep}",
        f"feat_postfix=_t{args.timestep}_layer{args.layer}", f"prefix_len={args.prefix}",
        "job_name=canary", "paths.run_folder_name=canary", "logger.wandb.offline=true",
        "model.output_path=/tmp/tstep-probe-canary-viz",
        "autoresume=true", "trainer.devices=[0]",
        f"data.data_module.batch_size_per_device={1 if args.task in {'streaming_depth','streaming_mask'} else 2}",
        f"data.data_module.batch_size_per_device_val={1 if args.task in {'streaming_depth','streaming_mask'} else 2}",
        "data.data_module.num_workers=4", "data.data_module.num_workers_val=2",
    ]
    if args.task in {"action_dynamics", "path_integration", "counterfactual"}:
        overrides.append("target_feat_root=aoss://baiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_SEEDED_V1")
    with initialize_config_dir(version_base="1.3", config_dir=str(impl / "configs")):
        cfg = compose(config_name="train.yaml", overrides=overrides)
    expression = cfg.data.data_module[
        "train_datasets" if args.split == "train" else "validation_datasets"
    ][0]
    dataset = eval(str(expression), {"InsScene15KDataset": InsScene15KDataset})

    samples, data_seconds = [], []
    # First call captures cold label build; subsequent calls are steady state.
    for _ in range(3):
        begin = time.perf_counter()
        sample = dataset[args.index]
        if args.empty_mask_fixture:
            if args.task != "streaming_mask":
                raise ValueError("--empty-mask-fixture is only valid for streaming_mask")
            sample["identity_ids"] = torch.full_like(sample["identity_ids"], -1)
        data_seconds.append(time.perf_counter() - begin)
        samples.append(sample)
    missing = REQUIRED[args.task] - set(samples[-1])
    if missing:
        raise RuntimeError(f"sample lacks required fields: {sorted(missing)}")
    required_digest = {
        key: tensor_digest(samples[-1][key])
        for key in sorted(REQUIRED[args.task])
        if isinstance(samples[-1][key], torch.Tensor)
    }
    required_values = {
        key: samples[-1][key].detach().cpu().tolist()
        for key in sorted(REQUIRED[args.task])
        if isinstance(samples[-1][key], torch.Tensor) and samples[-1][key].numel() <= 64
    }
    sample_bytes = sum(
        value.numel() * value.element_size()
        for value in samples[-1].values() if isinstance(value, torch.Tensor)
    )

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    module = instantiate(cfg.model).cuda()
    module.train()
    initial_parameter_sha256 = parameter_digest(module)
    optimizer = module.hparams.optimizer(params=module.parameters())
    h2d_ms, compute_ms, losses = [], [], []
    for step in range(args.train_steps):
        sample = dataset[args.index + step]
        if args.empty_mask_fixture:
            sample["identity_ids"] = torch.full_like(sample["identity_ids"], -1)
        batch = default_collate([sample])
        start_h2d, end_h2d = torch.cuda.Event(True), torch.cuda.Event(True)
        start_compute, end_compute = torch.cuda.Event(True), torch.cuda.Event(True)
        start_h2d.record()
        batch = module.transfer_batch_to_device(batch, torch.device("cuda", 0), 0)
        end_h2d.record()
        start_compute.record()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if args.task in {"streaming_depth", "streaming_mask"}:
                _, _, loss, _ = module.model_step(batch, return_metrics=False)
            else:
                loss, _ = module.model_step(batch, train=True)
        if loss.requires_grad:
            loss.backward()
            optimizer.step()
        end_compute.record()
        torch.cuda.synchronize()
        h2d_ms.append(start_h2d.elapsed_time(end_h2d))
        compute_ms.append(start_compute.elapsed_time(end_compute))
        losses.append(float(loss.detach().cpu()))

    result = {
        "implementation": args.implementation_name,
        "implementation_root": str(impl),
        "task": args.task,
        "backend": args.backend,
        "timestep": args.timestep,
        "layer": args.layer,
        "prefix": args.prefix,
        "split": args.split,
        "index": args.index,
        "manifest_sha256": manifest_sha,
        "manifest_sample_id": str(samples[-1].get("manifest_sample_id", "")),
        "sample_keys": sorted(samples[-1]),
        "sample_tensor_bytes": sample_bytes,
        "required_fields": required_digest,
        "required_values": required_values,
        "data_seconds_cold_then_steady": data_seconds,
        "h2d_ms": h2d_ms,
        "gpu_compute_ms": compute_ms,
        "losses": losses,
        "initial_parameter_sha256": initial_parameter_sha256,
        "final_parameter_sha256": parameter_digest(module),
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "effective_batch_size": 1 if args.task in {"streaming_depth", "streaming_mask"} else 2,
        "num_workers": 4,
        "num_workers_val": 2,
        "precision": "bf16-mixed",
        "seed": 42,
        "empty_mask_fixture": args.empty_mask_fixture,
    }
    if args.parameter_output is not None:
        args.parameter_output.parent.mkdir(parents=True, exist_ok=True)
        trainable = {
            name: value.detach().cpu().clone()
            for name, value in module.named_parameters() if value.requires_grad
        }
        torch.save(trainable, args.parameter_output)
        result["parameter_output"] = str(args.parameter_output)
        result["trainable_parameter_count"] = sum(value.numel() for value in trainable.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
