"""Per-task evaluation for v3 checkpoints with stratified reporting by source.

Protocol changes:
1) Auto-resolve latest epoch checkpoint for each model run.
2) Report metrics both overall and by source (infinigen/scannetpp).
3) Use sample-weighted aggregation (instead of plain batch mean).
4) Save a JSON artifact for reproducibility.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import datetime

import torch

from vidfm3d.data.components.inscene15k_dataset import InsScene15KDataset
from vidfm3d.models.components.probe_pixalign import ProbeModelPA
from vidfm3d.utils.loss import camera_loss, depth_loss, identity_loss


CONFIGS = {
    "wan": {
        "video_channels": 1536,
        "vfm_name": "wan",
        "feat_postfix": "_t749_layer20",
        "batch_size": 4,
        "run_dir": "logs/inscene15k/runs/inscene15k_wan_probe_v3",
    },
    "cogvideox": {
        "video_channels": 3072,
        "vfm_name": "cogvideox",
        "feat_postfix": "_t749_layer20",
        "batch_size": 2,
        "run_dir": "logs/inscene15k/runs/inscene15k_cogvideox_probe_v3",
    },
    "vjepa2": {
        "video_channels": 1024,
        "vfm_name": "vjepa2",
        "feat_postfix": "_layer23",
        "batch_size": 4,
        "run_dir": "logs/inscene15k/runs/inscene15k_vjepa2_probe_v3",
    },
}

CAMERA_METRICS = ["Auc_3", "Auc_5", "Auc_10", "Auc_30", "Rac_5", "Rac_15", "Tac_5", "Tac_15"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--models",
        type=str,
        default="wan,cogvideox,vjepa2",
        help="Comma-separated model names from CONFIGS",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-root", type=str, default="/nas/baiqiao/InsScene-15K/data")
    parser.add_argument("--feat-root", type=str, default="/nas/baiqiao/InsScene-15K/FEAT")
    parser.add_argument("--out-json", type=str, default="logs/eval_pertask_v3_stratified.json")
    return parser.parse_args()


def extract_epoch(path: str) -> int:
    base = os.path.basename(path)
    # expected pattern epoch=79-step=xxxxx.ckpt
    if "epoch=" in base:
        try:
            return int(base.split("epoch=")[1].split("-")[0])
        except Exception:
            return -1
    return -1


def resolve_latest_ckpt(run_dir: str) -> str:
    pattern = os.path.join(run_dir, "VideoProbe3D", "*", "checkpoints", "epoch=*-step=*.ckpt")
    cands = glob.glob(pattern)
    if not cands:
        # fallback to legacy checkpoint naming if needed
        pattern_legacy = os.path.join(run_dir, "checkpoints", "epoch_*.ckpt")
        cands = glob.glob(pattern_legacy)
    if not cands:
        raise FileNotFoundError(f"No checkpoint found under {run_dir}")

    cands = sorted(cands, key=lambda p: (extract_epoch(p), os.path.getmtime(p)))
    return cands[-1]


def infer_source(scene_path: str) -> str:
    p = scene_path.lower()
    if "infinigen" in p:
        return "infinigen"
    if "scannetpp" in p:
        return "scannetpp"
    return "unknown"


def build_model(cfg: dict, ckpt_path: str, device: torch.device):
    probe = ProbeModelPA(
        video_channels=cfg["video_channels"],
        embed_dim=1024,
        backbone_depth=4,
        dpt_dim=256,
        dpt_stage_channels=[256, 512, 1024, 1024],
        dpt_intermediate_layer_idx=[0, 1, 2, 3],
        gradient_checkpointing=False,
        active_heads=["depth", "camera", "identity"],
        identity_dim=64,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    probe_state = {k.replace("probe.", ""): v for k, v in state.items() if k.startswith("probe.")}
    probe.load_state_dict(probe_state)
    return probe.to(device).eval()


def build_dataset(cfg: dict, data_root: str, feat_root: str):
    return InsScene15KDataset(
        root=data_root,
        root_vfm=feat_root,
        sources=["processed_infinigen", "processed_scannetpp_v2"],
        split="val",
        vfm_name=cfg["vfm_name"],
        feat_postfix=cfg["feat_postfix"],
        feat_pixalign=True,
        num_views=4,
        min_view_interval=5,
        context_len=76,
        query_idx_divisor=4,
        target_h=288,
        target_w=512,
        window_size=200,
        include_pmaps=False,
        seed=42,
    )


def update_group(acc: dict, group: str, metrics: dict, n: int):
    acc[group]["count"] += n
    for k, v in metrics.items():
        acc[group][k] += float(v) * n


def finalize_group(acc: dict):
    out = {}
    for group, vals in acc.items():
        n = vals.pop("count")
        out[group] = {"samples": int(n)}
        if n == 0:
            continue
        for k, v in vals.items():
            out[group][k] = v / n
    return out


@torch.no_grad()
def evaluate_one(model_name: str, cfg: dict, args, device: torch.device):
    ckpt_path = resolve_latest_ckpt(cfg["run_dir"])
    print("=" * 72)
    print(f"Evaluating {model_name} with {ckpt_path}")
    print("=" * 72)

    model = build_model(cfg, ckpt_path, device)
    ds = build_dataset(cfg, args.data_root, args.feat_root)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
    )

    # weighted aggregation: overall + by source
    agg = defaultdict(lambda: defaultdict(float))
    agg["overall"]["count"] = 0.0
    agg["infinigen"]["count"] = 0.0
    agg["scannetpp"]["count"] = 0.0
    agg["unknown"]["count"] = 0.0

    for i, batch in enumerate(loader):
        tensor_keys = [k for k, v in batch.items() if isinstance(v, torch.Tensor)]
        for k in tensor_keys:
            batch[k] = batch[k].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            video_tokens = batch["vfm_feat"]
            preds = model(video_tokens.permute(0, 1, 4, 2, 3), batch["image"].shape, fg_mask=batch.get("masks"))

        out = {}
        if "depth" in preds:
            out["dmaps"] = preds["depth"].permute(0, 1, 4, 2, 3)
        if "pose_enc_list" in preds:
            out["pose_list"] = preds["pose_enc_list"]
            out["pose"] = preds["pose_enc"]
        if "identity" in preds:
            out["identity"] = preds["identity"]

        conf_map = batch["cmaps"].squeeze(2)
        fg_mask = None
        metrics = {}

        if "dmaps" in out:
            dmap_dict = depth_loss(
                out["dmaps"].float().permute(0, 1, 3, 4, 2),
                conf_map.float(),
                batch["dmaps"].float().permute(0, 1, 3, 4, 2),
                mask=fg_mask,
                gradient_loss="grad",
            )
            metrics["depth"] = dmap_dict["loss_depth"].item()

        if "pose" in out:
            cam_dict = camera_loss(
                out["pose_list"],
                batch["intrinsics"].float(),
                batch["extrinsics"].float(),
                batch["image"].shape[-2:],
                loss_type="huber",
                return_metrics=True,
            )
            metrics["camera"] = cam_dict["loss_camera"].item()
            for mk in CAMERA_METRICS:
                if mk in cam_dict:
                    mv = cam_dict[mk]
                    metrics[mk] = mv.item() if torch.is_tensor(mv) else float(mv)

        if "identity" in out:
            id_dict = identity_loss(out["identity"].float(), batch["identity_ids"], mask=fg_mask)
            metrics["identity"] = id_dict["loss_identity"].item()

        metrics["total"] = sum(metrics.get(k, 0.0) for k in ["depth", "camera", "identity"])

        batch_size = batch["image"].shape[0]
        update_group(agg, "overall", metrics, batch_size)

        scene_paths = batch.get("scene_path", [""] * batch_size)
        source_count = defaultdict(int)
        for p in scene_paths:
            source_count[infer_source(str(p))] += 1
        for src, n in source_count.items():
            update_group(agg, src, metrics, n)

        if (i + 1) % 20 == 0:
            print(f"  processed {i + 1}/{len(loader)} batches")

    grouped = finalize_group(agg)
    grouped["checkpoint"] = ckpt_path
    grouped["checkpoint_epoch"] = extract_epoch(ckpt_path)
    return grouped


def print_table(results: dict):
    print("\n" + "=" * 72)
    print("OVERALL COMPARISON")
    print("=" * 72)
    metrics = ["depth", "camera", "identity", "total", "Auc_30", "Rac_15", "Tac_15"]
    header = f"{'metric':14s} {'wan':>12s} {'cogvideox':>12s} {'vjepa2':>12s}"
    print(header)
    print("-" * len(header))
    for m in metrics:
        vals = [results[k]["overall"].get(m, float("nan")) for k in ["wan", "cogvideox", "vjepa2"] if k in results]
        row = f"{m:14s}"
        for model_name in ["wan", "cogvideox", "vjepa2"]:
            v = results.get(model_name, {}).get("overall", {}).get(m, float("nan"))
            row += f" {v:12.4f}"
        print(row)

    print("\n" + "=" * 72)
    print("SOURCE-STRATIFIED SUMMARY")
    print("=" * 72)
    for model_name, r in results.items():
        print(f"[{model_name}] ckpt_epoch={r.get('checkpoint_epoch', -1)}")
        for src in ["infinigen", "scannetpp", "unknown"]:
            if src not in r:
                continue
            print(
                f"  {src:10s} samples={r[src].get('samples', 0):5d} "
                f"depth={r[src].get('depth', float('nan')):8.4f} "
                f"camera={r[src].get('camera', float('nan')):8.4f} "
                f"identity={r[src].get('identity', float('nan')):8.4f} "
                f"Auc_30={r[src].get('Auc_30', float('nan')):8.4f}"
            )


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    results = {}
    for m in model_names:
        if m not in CONFIGS:
            raise ValueError(f"Unknown model '{m}'. Available: {list(CONFIGS.keys())}")
        results[m] = evaluate_one(m, CONFIGS[m], args, device)

    print_table(results)

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "protocol": {
            "stratified_by_source": True,
            "aggregation": "sample_weighted",
            "split": "val",
            "window_size": 200,
            "heads": ["depth", "camera", "identity"],
        },
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved report to {args.out_json}")


if __name__ == "__main__":
    main()
