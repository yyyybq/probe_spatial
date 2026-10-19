#!/usr/bin/env python3
"""Freeze and compare the six historical Wan p8 probe training configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


LAYERS = [0, 5, 10, 15, 25, 29]
HISTORICAL_GROUP = "wan_cog_layer_probe_20260724_2248"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pick(cfg: Any) -> dict[str, Any]:
    return OmegaConf.to_container(
        OmegaConf.create(
            {
                "seed": cfg.seed,
                "train": cfg.train,
                "test": cfg.test,
                "ckpt_path": cfg.ckpt_path,
                "autoresume": cfg.autoresume,
                "data": {
                    "batch_size_per_device": cfg.data.data_module.batch_size_per_device,
                    "batch_size_per_device_val": cfg.data.data_module.batch_size_per_device_val,
                    "num_workers": cfg.data.data_module.num_workers,
                    "num_workers_val": cfg.data.data_module.num_workers_val,
                    "pin_memory": cfg.data.data_module.pin_memory,
                },
                "optimizer": {
                    "target": cfg.model.optimizer._target_,
                    "lr": cfg.model.optimizer.lr,
                    "betas": cfg.model.optimizer.betas,
                    "weight_decay": cfg.model.optimizer.weight_decay,
                },
                "scheduler": {
                    "target": cfg.model.scheduler._target_,
                    "warmup_epochs": cfg.model.scheduler.warmup_epochs,
                    "eta_min": cfg.model.scheduler.eta_min,
                },
                "probe": {
                    "target": cfg.model.probe._target_,
                    "in_channels": cfg.model.probe.in_channels,
                    "action_dim": cfg.model.probe.action_dim,
                    "hidden_dim": cfg.model.probe.hidden_dim,
                    "num_layers": cfg.model.probe.num_layers,
                    "num_heads": cfg.model.probe.num_heads,
                    "max_seq_len": cfg.model.probe.max_seq_len,
                },
                "loss": {
                    "mse_loss_weight": cfg.model.mse_loss_weight,
                    "cosine_loss_weight": cfg.model.cosine_loss_weight,
                },
                "checkpoint": {
                    "every_n_epochs": cfg.callbacks.model_checkpoint.every_n_epochs,
                    "save_last": cfg.callbacks.model_checkpoint.save_last,
                    "save_top_k": cfg.callbacks.model_checkpoint.save_top_k,
                    "early_stopping": cfg.callbacks.early_stopping,
                },
                "trainer": {
                    "min_epochs": cfg.trainer.min_epochs,
                    "max_epochs": cfg.trainer.max_epochs,
                    "accelerator": cfg.trainer.accelerator,
                    "devices": cfg.trainer.devices,
                    "strategy": cfg.trainer.strategy,
                    "num_nodes": cfg.trainer.num_nodes,
                    "check_val_every_n_epoch": cfg.trainer.check_val_every_n_epoch,
                    "precision": cfg.trainer.precision,
                    "sync_batchnorm": cfg.trainer.sync_batchnorm,
                    "use_distributed_sampler": cfg.trainer.use_distributed_sampler,
                    "deterministic": cfg.trainer.deterministic,
                },
                "feature": {
                    "vfm_name": cfg.vfm_name,
                    "video_channels": cfg.video_channels,
                    "feature_timestep": cfg.feature_timestep,
                    "prefix_len": cfg.prefix_len,
                    "action_horizons": cfg.action_horizons,
                    "feat_postfix": cfg.feat_postfix,
                },
            }
        ),
        resolve=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    records = {}
    common = None
    for layer in LAYERS:
        run = f"{HISTORICAL_GROUP}_action_dynamics_wan_p8_layer{layer}"
        path = root / "logs" / "inscene15k_streaming" / "runs" / HISTORICAL_GROUP / run / ".hydra" / "config.yaml"
        if not path.is_file():
            raise SystemExit(f"Missing historical config: {path}")
        cfg = OmegaConf.load(path)
        if int(cfg.feature_layer) != layer:
            raise SystemExit(f"Historical layer mismatch: {path}")
        selected = pick(cfg)
        if common is None:
            common = selected
        elif selected != common:
            raise SystemExit(f"Historical hyperparameter drift at layer {layer}")
        records[str(layer)] = {"path": str(path), "sha256": sha256(path)}

    expected = {
        "seed": 42,
        "train": True,
        "test": False,
        "ckpt_path": None,
        "autoresume": False,
        "data": {"batch_size_per_device": 8, "batch_size_per_device_val": 8, "num_workers": 4, "num_workers_val": 2, "pin_memory": True},
        "optimizer": {"target": "torch.optim.AdamW", "lr": 0.0003, "betas": [0.9, 0.95], "weight_decay": 0.05},
        "scheduler": {"target": "pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR", "warmup_epochs": 5, "eta_min": 0.000001},
        "probe": {"target": "vidfm3d.models.components.probe_action_dynamics.ActionDynamicsProbe", "in_channels": 1536, "action_dim": 9, "hidden_dim": 512, "num_layers": 2, "num_heads": 8, "max_seq_len": 64},
        "loss": {"mse_loss_weight": 1.0, "cosine_loss_weight": 1.0},
        "checkpoint": {"every_n_epochs": 5, "save_last": True, "save_top_k": -1, "early_stopping": None},
        "trainer": {"min_epochs": 1, "max_epochs": 50, "accelerator": "gpu", "devices": [0], "strategy": "auto", "num_nodes": 1, "check_val_every_n_epoch": 1, "precision": "bf16-mixed", "sync_batchnorm": True, "use_distributed_sampler": False, "deterministic": False},
        "feature": {"vfm_name": "wan", "video_channels": 1536, "feature_timestep": 749, "prefix_len": 8, "action_horizons": [1], "feat_postfix": "_t749_layer20"},
    }
    if common != expected:
        raise SystemExit("Historical config contract does not match the audited expected values")

    payload = {
        "ok": True,
        "historical_group": HISTORICAL_GROUP,
        "layers": LAYERS,
        "historical_configs": records,
        "common_hyperparameters": common,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out = Path(args.out_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    tmp.write_text(encoded)
    os.replace(tmp, out)
    print(json.dumps({"ok": True, "layers": LAYERS, "contract_sha256": sha256(out)}))


if __name__ == "__main__":
    main()
