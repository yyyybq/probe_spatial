"""
Quick eval script: load C1 (action_dynamics) checkpoints and print val metrics.
Usage:
    python eval_c1_metrics.py
"""
import sys
import os
import torch
import lightning as L
from omegaconf import OmegaConf

# ── paths ───────────────────────────────────────────────────────────────────
RUNS = {
    "wan_VFM": {
        "ckpt": "logs/inscene15k_ext/runs/inscene15k_ext_action_dynamics_wan_v1/checkpoints/epoch=49-step=104850.ckpt",
        "experiment": "inscene15k_ext/action_dynamics_wan_v1",
    },
    "wan_CTRL": {
        "ckpt": "logs/inscene15k_ext/runs/inscene15k_ext_action_dynamics_wan_ctrl/checkpoints/epoch=49-step=104850.ckpt",
        "experiment": "inscene15k_ext/action_dynamics_wan_ctrl",
    },
}

# ── run validation via Hydra+Lightning ──────────────────────────────────────
import subprocess, json, re

results = {}
for name, info in RUNS.items():
    ckpt = os.path.abspath(info["ckpt"])
    exp  = info["experiment"]
    log_path = f"/tmp/eval_c1_{name}.log"

    cmd = [
        "python", "vidfm3d/train.py",
        f"experiment={exp}",
        "~logger",
        f"ckpt_path={ckpt}",
        "trainer.limit_train_batches=0",
        "trainer.max_epochs=1",
        "trainer.num_sanity_val_steps=0",
    ]
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    sys.stdout.flush()

    ret = subprocess.run(cmd, capture_output=False)
    print(f"\n  Exit code: {ret.returncode}")
