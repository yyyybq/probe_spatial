"""Streaming A1 evaluation.

A1 is extended with a protocol-matched relative-movement job:
  streaming_depth  -> depth MAE + pose (camera head, AUC-30)
  streaming_movement -> depth MAE + adjacent relative camera movement
  streaming_mask   -> identity / mask loss

Pose is the camera head inside streaming_depth. There is no separate
streaming_camera experiment.

Loads a VideoProbeLitModule from a checkpoint and runs the streaming
validation dataloader, computing and saving records/metrics to:
  {output_dir}/eval/val_predictions.pt
  {output_dir}/eval/val_summary.json

Usage (Hydra):
  python vidfm3d/eval_a1_streaming.py \
      experiment=inscene15k_streaming/streaming_depth_dino_v1 \
      "+ckpt_path='/path/to/epoch=99-step=xxx.ckpt'" \
      prefix_len=24 vfm_name=dino video_channels=1024 \
      streaming_feat_root=${INSCENE15K_ROOT}/FEAT_STREAMING_BY_PREFIX/dino_prefix_24 \
      eval_split=val
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from vidfm3d.utils import pylogger
from vidfm3d.utils.loss import (
    camera_loss,
    depth_loss,
    identity_loss,
    relative_movement_loss,
)
from vidfm3d.utils.eval_utils import align_pmaps, build_mask

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

CAMERA_METRICS = [
    "Auc_3",
    "Auc_5",
    "Auc_10",
    "Auc_30",
    "Rac_5",
    "Rac_15",
    "Tac_5",
    "Tac_15",
]
ADJACENT_MOVEMENT_METRICS = [
    "adjacent_movement_RotErr_deg",
    "adjacent_movement_TransAngle_deg",
    "adjacent_movement_TransL2",
    "adjacent_movement_TransMagErr",
    "adjacent_movement_Rac_5",
    "adjacent_movement_Rac_15",
    "adjacent_movement_Tac_5",
    "adjacent_movement_Tac_15",
    "adjacent_movement_Auc_3",
    "adjacent_movement_Auc_5",
    "adjacent_movement_Auc_10",
    "adjacent_movement_Auc_30",
]

# v2: a non-finite input or prediction is recorded as an invalid sample (loss
# NaN, ``prediction_finite=False``) instead of a zero loss. The training loss
# helpers replace NaN/inf by zero, which previously turned diverged weights or
# corrupt features into ``mean_loss_depth = 0`` and ``Auc_30 = 0``.
EVALUATOR_VERSION = "a1_streaming_eval_v2_nonfinite_aware"


def _sample_is_finite(tensors, index: int) -> bool:
    """True when every tensor's ``index``-th batch row is finite."""
    for value in tensors:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            if not _sample_is_finite(value, index):
                return False
            continue
        if torch.is_tensor(value) and value.is_floating_point():
            if value.shape[0] <= index or not bool(torch.isfinite(value[index]).all()):
                return False
    return True


def _execution_device() -> tuple[torch.device, Dict[str, Any]]:
    """Select the eval device and capture the exact CUDA execution context."""
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count())
    if os.environ.get("Q00_REQUIRE_CUDA") == "1" and (
        not cuda_available or cuda_device_count < 1
    ):
        raise RuntimeError(
            "Q00_REQUIRE_CUDA=1 but CUDA is unavailable or no CUDA device is visible"
        )

    device = torch.device("cuda:0" if cuda_available else "cpu")
    cuda_device_name = None
    cuda_device_uuid = None
    if device.type == "cuda":
        cuda_device_name = str(torch.cuda.get_device_name(device))
        device_uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
        if device_uuid is not None:
            cuda_device_uuid = str(device_uuid)

    return device, {
        "execution_device": str(device),
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_name": cuda_device_name,
        "cuda_device_uuid": cuda_device_uuid,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _infer_active_heads(cfg) -> List[str]:
    """Infer which heads are active from the model config."""
    try:
        heads = list(cfg.model.probe.active_heads)
        return heads
    except Exception:
        pass
    # Fallback: infer from experiment / job name.
    # streaming_depth contains "depth" and owns A1 pose (camera head).
    job = str(cfg.get("job_name", ""))
    if "mask" in job or "identity" in job:
        return ["identity"]
    if "movement" in job:
        return ["depth", "movement"]
    if "streaming_depth" in job or "depth" in job:
        return ["depth", "camera"]
    if "camera" in job:
        return ["camera"]
    return ["depth", "camera"]


def _source_from_path(path: str) -> str:
    if "processed_infinigen" in path:
        return "infinigen"
    if "processed_scannetpp" in path:
        return "scannetpp"
    return "unknown"


@torch.no_grad()
def evaluate(model, loader, active_heads: List[str], device: torch.device):
    """Run full validation and return per-sample records."""
    all_metrics: List[Dict[str, Any]] = []

    for i, batch in enumerate(loader):
        batch_cpu = batch[0] if isinstance(batch, (list, tuple)) else batch
        batch_gpu = {}
        for k, v in batch_cpu.items():
            if isinstance(v, torch.Tensor):
                batch_gpu[k] = (
                    v.to(device).float() if v.is_floating_point() else v.to(device)
                )
            else:
                batch_gpu[k] = v

        with torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            video_tokens = batch_gpu["vfm_feat"]
            preds = model.probe(
                video_tokens.permute(0, 1, 4, 2, 3),
                batch_gpu["image"].shape,
                fg_mask=batch_gpu.get("masks"),
            )

        out = {}
        if "depth" in preds:
            out["dmaps"] = preds["depth"].permute(0, 1, 4, 2, 3)
        if "pose_enc_list" in preds:
            out["pose_list"] = preds["pose_enc_list"]
            out["pose"] = preds["pose_enc"]
        if "movement_list" in preds:
            out["movement_list"] = preds["movement_list"]
            out["movement"] = preds["movement"]
        if "identity" in preds:
            out["identity"] = preds["identity"]

        B = batch_gpu["image"].shape[0]
        scene_paths = batch_gpu.get("scene_path", ["?"] * B)
        manifest_sample_ids = batch_gpu.get("manifest_sample_id", [None] * B)
        window_ids = batch_gpu.get("streaming_window_id")
        tail_raw = batch_gpu.get("streaming_tail_raw")

        for b in range(B):
            record: Dict[str, Any] = {
                "scene_path": (
                    scene_paths[b]
                    if isinstance(scene_paths, list)
                    else str(scene_paths[b])
                ),
                "manifest_sample_id": (
                    manifest_sample_ids[b]
                    if isinstance(manifest_sample_ids, list)
                    else str(manifest_sample_ids[b])
                ),
                "streaming_window_id": (
                    int(window_ids[b].item())
                    if isinstance(window_ids, torch.Tensor)
                    else -1
                ),
                "streaming_tail_raw": (
                    int(tail_raw[b].item())
                    if isinstance(tail_raw, torch.Tensor)
                    else -1
                ),
                "input_finite": _sample_is_finite([video_tokens], b),
                "prediction_finite": _sample_is_finite(list(out.values()), b),
            }

            if not record["prediction_finite"]:
                # Do not route NaN/inf through the loss helpers: they clamp
                # non-finite values to zero and would report a perfect score.
                for name in ("loss_depth", "loss_camera", "loss_movement", "loss_identity"):
                    head = {"loss_depth": "depth", "loss_camera": "camera",
                            "loss_movement": "movement", "loss_identity": "identity"}[name]
                    if head in active_heads:
                        record[name] = float("nan")
                all_metrics.append(record)
                continue

            if "dmaps" in out and "depth" in active_heads:
                try:
                    d_dict = depth_loss(
                        out["dmaps"].float().permute(0, 1, 3, 4, 2)[b : b + 1],
                        batch_gpu["cmaps"].squeeze(2).float()[b : b + 1],
                        batch_gpu["dmaps"]
                        .float()
                        .permute(0, 1, 3, 4, 2)[b : b + 1],
                        mask=None,
                        gradient_loss="grad",
                    )
                    record["loss_depth"] = float(d_dict["loss_depth"].item())
                except Exception:
                    record["loss_depth"] = float("nan")

            if "pose" in out and "camera" in active_heads:
                try:
                    c_dict = camera_loss(
                        [pl[b : b + 1] for pl in out["pose_list"]],
                        batch_gpu["intrinsics"].float()[b : b + 1],
                        batch_gpu["extrinsics"].float()[b : b + 1],
                        batch_gpu["image"].shape[-2:],
                        loss_type="huber",
                        return_metrics=True,
                    )
                    record["loss_camera"] = float(c_dict["loss_camera"].item())
                    for mk in CAMERA_METRICS + ADJACENT_MOVEMENT_METRICS:
                        if mk in c_dict:
                            v = c_dict[mk]
                            record[mk] = float(v.item() if torch.is_tensor(v) else v)
                except Exception:
                    record["loss_camera"] = float("nan")

            if "movement" in out and "movement" in active_heads:
                try:
                    movement_dict = relative_movement_loss(
                        [value[b : b + 1] for value in out["movement_list"]],
                        batch_gpu["extrinsics"].float()[b : b + 1],
                        loss_type="huber",
                        return_metrics=True,
                    )
                    record["loss_movement"] = float(
                        movement_dict["loss_movement"].item()
                    )
                    for metric_name in ADJACENT_MOVEMENT_METRICS:
                        value = movement_dict.get(metric_name)
                        if value is not None:
                            record[metric_name] = float(
                                value.item() if torch.is_tensor(value) else value
                            )
                except Exception:
                    record["loss_movement"] = float("nan")

            if "identity" in out and "identity" in active_heads:
                try:
                    id_dict = identity_loss(
                        out["identity"].float()[b:b+1],
                        batch_gpu["identity_ids"][b:b+1],
                        mask=None,
                    )
                    record["loss_identity"] = float(id_dict["loss_identity"].item())
                except Exception:
                    record["loss_identity"] = float("nan")

            all_metrics.append(record)

        if (i + 1) % 20 == 0:
            log.info(f"[{i+1}/{len(loader)}] processed {len(all_metrics)} samples")

    return all_metrics


_FLAG_KEYS = ("input_finite", "prediction_finite")


def _summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = set()
    for r in records:
        keys.update(
            k for k, v in r.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in _FLAG_KEYS
        )
    keys.discard("scene_path")

    summary: Dict[str, Any] = {}
    for k in sorted(keys):
        vals = [r[k] for r in records if isinstance(r.get(k), (int, float)) and math.isfinite(r.get(k, float("nan")))]
        if vals:
            summary[f"mean_{k}"] = float(sum(vals) / len(vals))
            summary[f"n_{k}"] = len(vals)

    # Special: camera AUC averaging
    for mk in CAMERA_METRICS + ADJACENT_MOVEMENT_METRICS:
        vals = [r[mk] for r in records if isinstance(r.get(mk), (int, float)) and math.isfinite(r.get(mk, float("nan")))]
        if vals:
            summary[mk] = float(sum(vals) / len(vals))

    # Invalid samples are counted, never averaged as zero. Records written by
    # the v1 evaluator carry no flags and are treated as finite.
    summary["n_records"] = len(records)
    summary["n_nonfinite_input"] = sum(1 for r in records if r.get("input_finite") is False)
    summary["n_nonfinite_prediction"] = sum(1 for r in records if r.get("prediction_finite") is False)
    summary["n_valid_prediction"] = len(records) - summary["n_nonfinite_prediction"]
    return summary


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def _require_val_split(cfg) -> str:
    """This evaluator always reads the validation loader and writes val_* files.

    Refuse any other eval_split instead of silently evaluating validation data
    under a test request (scripts/run_streaming_probe_sweep.sh forwards
    +eval_split to it).
    """
    split = cfg.get("eval_split", "val")
    if split != "val":
        raise ValueError(
            f"eval_a1_streaming evaluates the validation split only; got eval_split={split!r}"
        )
    return split


def main(cfg: DictConfig) -> None:
    if not cfg.get("ckpt_path"):
        raise ValueError("Must provide ckpt_path=<path>")
    _require_val_split(cfg)
    device, execution_metadata = _execution_device()

    log.info(OmegaConf.to_yaml(cfg))

    datamodule = hydra.utils.instantiate(cfg.data.data_module)
    datamodule.setup(stage="validate")
    from types import SimpleNamespace
    datamodule.trainer = SimpleNamespace(
        strategy=None, world_size=1, global_rank=0, local_rank=0, num_nodes=1
    )

    loader = datamodule.val_dataloader()
    if isinstance(loader, list):
        loader = loader[0]

    model = hydra.utils.instantiate(cfg.model)
    state = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    model.float().to(device).eval()

    active_heads = _infer_active_heads(cfg)
    log.info(f"Active heads for eval: {active_heads}")

    records = evaluate(model, loader, active_heads, device)

    # Overall summary
    summary = _summarize(records)

    # By source
    for src in ["infinigen", "scannetpp", "unknown"]:
        sub = [r for r in records if _source_from_path(str(r.get("scene_path", ""))) == src]
        if sub:
            summary[f"by_source.{src}"] = _summarize(sub)
            summary[f"by_source.{src}"]["n"] = len(sub)

    if active_heads == ["identity"]:
        summary["probe_type"] = "streaming_mask"
    elif "movement" in active_heads:
        summary["probe_type"] = "streaming_movement"
    else:
        summary["probe_type"] = "streaming_depth"
    summary["active_heads"] = active_heads
    summary["ckpt_path"] = str(cfg.ckpt_path)
    summary["vfm_name"] = cfg.get("vfm_name")
    summary["prefix_len"] = cfg.get("prefix_len")
    summary["feature_layer"] = cfg.get("feature_layer")
    summary["feat_postfix"] = cfg.get("feat_postfix")
    summary["feature_timestep"] = cfg.get("feature_timestep")
    summary["streaming_feat_root"] = cfg.get("streaming_feat_root")
    summary["target_feat_root"] = cfg.get("target_feat_root")
    summary["n_samples"] = len(records)
    summary["evaluator_version"] = EVALUATOR_VERSION
    summary["loader_manifest_ids"] = [str(record["manifest_sample_id"]) for record in records]
    summary["valid_manifest_ids"] = [
        str(record["manifest_sample_id"]) for record in records if record.get("prediction_finite", True)
    ]
    summary["invalid_manifest_ids"] = [
        str(record["manifest_sample_id"]) for record in records if not record.get("prediction_finite", True)
    ]
    summary.update(execution_metadata)
    if cfg.get("job_name"):
        summary["job_name"] = str(cfg.job_name)
    for key in ("Q00_UNIVERSE_ID", "Q00_CAMPAIGN", "Q00_MODEL_VARIANT", "Q00_PROTOCOL",
                "Q00_MANIFEST_SHA256", "Q00_CONFIG_SHA256",
                "Q00_RESOLVED_CONFIG_SHA256", "Q00_RESOLVED_CONFIG_SOURCE_KIND"):
        if os.environ.get(key):
            summary[key.lower()] = os.environ[key]

    out_dir = Path(cfg.paths.output_dir) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "val_predictions.pt"
    sum_path = out_dir / "val_summary.json"
    torch.save(records, pred_path)
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Saved eval records to {pred_path}")
    log.info(f"Saved eval summary to {sum_path}")
    log.info(json.dumps({k: v for k, v in summary.items() if isinstance(v, (int, float, str))}, indent=2))


if __name__ == "__main__":
    main()
