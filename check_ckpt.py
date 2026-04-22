import torch
import sys
import os

paths = [
    "logs/inscene15k/runs/inscene15k_wan_probe_v3/checkpoints/last-v1.ckpt",
    "logs/inscene15k/runs/inscene15k_cogvideox_probe_v3/checkpoints/last-v2.ckpt",
    "logs/inscene15k/runs/inscene15k_vjepa2_probe_v3/checkpoints/last-v1.ckpt"
]

for path in paths:
    if os.path.exists(path):
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            print(f"{path}: epoch={ckpt.get('epoch')}, global_step={ckpt.get('global_step')}")
        except Exception as e:
            print(f"{path}: Error loading: {e}")
    else:
        print(f"{path}: Not found")
