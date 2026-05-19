"""
Evaluate C1 (action_dynamics) checkpoints — compute dyn_R@1, dyn_cos, val/loss.
Bypasses Hydra entirely. Builds datamodule from the existing YAML strings directly.

Usage (from probe_spatial/ with vidfm3d env):
    CUDA_VISIBLE_DEVICES=2 python eval_c1_r1.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── patch path ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath("."))

from vidfm3d.models.probe_ext_module import ProbeExtensionLitModule
from vidfm3d.data.components.inscene15k_dataset import InsScene15KDataset

DEVICE = "cuda:0"
BATCH  = 16     # larger batch → better R@1 signal
N_WORKERS = 4

# ── shared dataset kwargs ────────────────────────────────────────────────────
BASE_KW = dict(
    root        = "/nas/baiqiao/InsScene-15K/data",
    root_vfm    = "/nas/baiqiao/InsScene-15K/FEAT",
    sources     = ["processed_infinigen", "processed_scannetpp_v2"],
    split       = "val",
    vfm_name    = "wan",
    feat_postfix= "_t749_layer20",
    feat_pixalign=True,
    num_views   = 4,
    min_view_interval=5,
    context_len = 76,
    query_idx_divisor=4,
    target_h    = 288,
    target_w    = 512,
    window_size = 200,
    include_pmaps=False,
    diag_action = True,
    target_feat_root="/nas/baiqiao/InsScene-15K/FEAT_TARGET",
)

RUNS = {
    "wan_VFM (real feat)": {
        "ckpt": "logs/inscene15k_ext/runs/inscene15k_ext_action_dynamics_wan_v1/checkpoints/epoch=49-step=104850.ckpt",
        "extra_kw": {},
    },
    "wan_CTRL (randn feat)": {
        "ckpt": "logs/inscene15k_ext/runs/inscene15k_ext_action_dynamics_wan_ctrl/checkpoints/epoch=49-step=104850.ckpt",
        "extra_kw": {"scramble_feat": True},
    },
}


@torch.no_grad()
def evaluate(run_name, ckpt_path, extra_kw):
    print(f"\n>>> {run_name}")

    ds = InsScene15KDataset(**BASE_KW, **extra_kw)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                        num_workers=N_WORKERS, pin_memory=True, drop_last=False)

    # load_from_checkpoint can't reconstruct 'probe' (not in hyper_parameters),
    # so manually build the probe and load weights from state_dict.
    from vidfm3d.models.components.probe_action_dynamics import ActionDynamicsProbe
    probe = ActionDynamicsProbe(
        in_channels=1536, action_dim=9, hidden_dim=512,
        num_layers=2, num_heads=8
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # state_dict keys are prefixed with "probe."
    probe_sd = {k[len("probe."):]: v for k, v in ckpt["state_dict"].items()
                if k.startswith("probe.")}
    probe.load_state_dict(probe_sd, strict=True)
    probe = probe.to(DEVICE).to(torch.bfloat16).eval()

    losses, mses, cos_sims, r1s, mean_ranks, n_vals = [], [], [], [], [], []

    for batch in loader:
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        valid = batch.get("dyn_valid")
        if valid is None:
            valid = torch.ones(batch["input_feat"].shape[0], dtype=torch.bool, device=DEVICE)
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue

        input_feat  = batch["input_feat"][valid].to(torch.bfloat16)
        action      = batch["action"][valid].to(torch.bfloat16)
        target_feat = batch["target_feat"][valid].to(torch.bfloat16)

        target_pooled = target_feat.mean(dim=(1, 2)).float()        # (B', C)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = probe(input_feat, action).float()                # (B', C)

        mse = F.mse_loss(pred, target_pooled).float()
        cos_sim = F.cosine_similarity(pred.float(), target_pooled.float(), dim=-1).mean()
        loss = mse + (1.0 - cos_sim)

        B = pred.shape[0]
        if B >= 2:
            sim = F.normalize(pred.float(), dim=-1) @ F.normalize(target_pooled.float(), dim=-1).T
            rank = (sim.argsort(dim=-1, descending=True) ==
                    torch.arange(B, device=sim.device).unsqueeze(-1)
                    ).float().argmax(dim=-1)
            r1 = (rank == 0).float().mean().item()
            mr = rank.float().mean().item()
        else:
            r1 = float("nan")
            mr = float("nan")

        losses.append(loss.item())
        mses.append(mse.item())
        cos_sims.append(cos_sim.item())
        r1s.append(r1)
        mean_ranks.append(mr)
        n_vals.append(n_valid)

    import statistics
    def wavg(vals, weights):
        return sum(v * w for v, w in zip(vals, weights)) / sum(weights)

    w = n_vals
    print(f"  val/loss      = {wavg(losses, w):.4f}")
    print(f"  dyn_mse       = {wavg(mses, w):.4f}")
    print(f"  dyn_cos_sim   = {wavg(cos_sims, w):.4f}")
    r1_clean = [v for v in r1s if v == v]  # drop NaN
    mr_clean = [v for v in mean_ranks if v == v]
    print(f"  dyn_R@1       = {statistics.mean(r1_clean):.4f}  (batch={BATCH}, random baseline={1/BATCH:.4f})")
    print(f"  dyn_mean_rank = {statistics.mean(mr_clean):.2f}   (random baseline={(BATCH-1)/2:.1f})")


print(f"\n{'='*65}")
print(f"  C1 Action-Dynamics Probe  —  Validation Metrics")
print(f"  batch_size={BATCH}  |  R@1 random baseline = {1/BATCH:.3f}")
print(f"{'='*65}")

for run_name, info in RUNS.items():
    evaluate(run_name, info["ckpt"], info["extra_kw"])

print("\nDone.")
