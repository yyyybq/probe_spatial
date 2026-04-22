"""Per-task evaluation for v3 checkpoints (no point head, windowed dataset)."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))

from vidfm3d.models.components.probe_pixalign import ProbeModelPA
from vidfm3d.data.components.inscene15k_dataset import InsScene15KDataset
from vidfm3d.utils.loss import camera_loss, depth_loss, identity_loss

DEVICE = torch.device(f"cuda:{sys.argv[1]}" if len(sys.argv) > 1 else "cuda:0")

CONFIGS = {
    "wan": {
        "ckpt": "logs/inscene15k/runs/inscene15k_wan_probe_v3/checkpoints/epoch_023.ckpt",
        "video_channels": 1536,
        "vfm_name": "wan",
        "feat_postfix": "_t749_layer20",
        "batch_size": 4,
    },
    "cogvideox": {
        "ckpt": "logs/inscene15k/runs/inscene15k_cogvideox_probe_v3/checkpoints/epoch_025.ckpt",
        "video_channels": 3072,
        "vfm_name": "cogvideox",
        "feat_postfix": "_t749_layer20",
        "batch_size": 2,
    },
    "vjepa2": {
        "ckpt": "logs/inscene15k/runs/inscene15k_vjepa2_probe_v3/checkpoints/epoch_023.ckpt",
        "video_channels": 1024,
        "vfm_name": "vjepa2",
        "feat_postfix": "_layer23",
        "batch_size": 4,
    },
}

def load_model(cfg):
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
    ckpt = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    probe_state = {k.replace("probe.", ""): v for k, v in state.items() if k.startswith("probe.")}
    probe.load_state_dict(probe_state)
    probe = probe.to(DEVICE).eval()
    return probe

def build_dataset(cfg):
    return InsScene15KDataset(
        root="/nas/baiqiao/InsScene-15K/data",
        root_vfm="/nas/baiqiao/InsScene-15K/FEAT",
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
        seed=42,
    )

@torch.no_grad()
def evaluate(model_name, cfg):
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} (ckpt: {os.path.basename(cfg['ckpt'])})")
    print(f"{'='*60}")

    probe = load_model(cfg)
    ds = build_dataset(cfg)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)

    loss_accum = {}
    n_batches = 0

    for i, batch in enumerate(loader):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            video_tokens = batch["vfm_feat"]
            preds = probe(
                video_tokens.permute(0, 1, 4, 2, 3),
                batch["image"].shape,
                fg_mask=batch.get("masks"),
            )

        output = {}
        if "depth" in preds:
            output["dmaps"] = preds["depth"].permute(0, 1, 4, 2, 3)
        if "pose_enc_list" in preds:
            output["pose_list"] = preds["pose_enc_list"]
            output["pose"] = preds["pose_enc"]
        if "identity" in preds:
            output["identity"] = preds["identity"]
            output["identity_conf"] = preds["identity_conf"]

        fg_mask = None
        conf_map = batch["cmaps"].squeeze(2)
        losses = {}

        if "dmaps" in output:
            dmap_dict = depth_loss(
                output["dmaps"].float().permute(0, 1, 3, 4, 2),
                conf_map.float(),
                batch["dmaps"].float().permute(0, 1, 3, 4, 2),
                mask=fg_mask,
                gradient_loss="grad",
            )
            losses["depth"] = dmap_dict["loss_depth"].item()

        if "pose" in output:
            cam_dict = camera_loss(
                output["pose_list"],
                batch["intrinsics"].float(),
                batch["extrinsics"].float(),
                batch["image"].shape[-2:],
                loss_type="huber",
                return_metrics=True,
            )
            losses["camera"] = cam_dict["loss_camera"].item()
            for mk in ["Auc_3", "Auc_5", "Auc_10", "Auc_30", "Rac_5", "Rac_15", "Tac_5", "Tac_15"]:
                if mk in cam_dict:
                    v = cam_dict[mk]
                    losses[mk] = v.item() if torch.is_tensor(v) else v

        if "identity" in output:
            id_dict = identity_loss(
                output["identity"].float(),
                batch["identity_ids"],
                mask=fg_mask,
            )
            losses["identity"] = id_dict["loss_identity"].item()

        total = sum(losses.get(k, 0) for k in ["depth", "camera", "identity"])
        losses["total"] = total

        for k, v in losses.items():
            loss_accum.setdefault(k, []).append(v)

        n_batches += 1
        if (i + 1) % 10 == 0:
            print(f"  Batch {i+1}/{len(loader)}", flush=True)

    print(f"\n--- {model_name} Results ({n_batches} batches, {n_batches * cfg['batch_size']} samples) ---")
    results = {}
    for k, vals in sorted(loss_accum.items()):
        mean_v = sum(vals) / len(vals)
        results[k] = mean_v
        print(f"  {k:15s}: {mean_v:.4f}")
    return results


if __name__ == "__main__":
    all_results = {}
    for name in ["wan", "cogvideox", "vjepa2"]:
        all_results[name] = evaluate(name, CONFIGS[name])

    print(f"\n{'='*60}")
    print("COMPARISON TABLE (v3: no point head, window_size=200)")
    print(f"{'='*60}")
    keys = ["depth", "camera", "identity", "total", "Auc_3", "Auc_5", "Auc_10", "Auc_30", "Rac_5", "Rac_15", "Tac_5", "Tac_15"]
    header = f"{'Metric':15s} {'WAN':>10s} {'CogVideoX':>10s} {'V-JEPA2':>10s}"
    print(header)
    print("-" * len(header))
    for k in keys:
        vals = [all_results[m].get(k, float('nan')) for m in ["wan", "cogvideox", "vjepa2"]]
        best = min(vals) if k in ["depth", "camera", "identity", "total"] else max(vals)
        row = f"{k:15s}"
        for v in vals:
            marker = " *" if v == best else "  "
            row += f" {v:>8.4f}{marker}"
        print(row)
