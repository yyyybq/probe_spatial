#!/usr/bin/env python3
"""One-device forward/backward smoke test for the primary diagnostic heads."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vidfm3d.models.components.probe_action_dynamics import ActionDynamicsProbe
from vidfm3d.models.components.probe_ego_belief import EgoBeliefProbe
from vidfm3d.models.components.probe_ego_belief_v2 import EgoBeliefProbeV2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    feat = torch.randn(2, 4, 4, 6, 32, device=device)
    mask = torch.zeros(2, 4, 4, 6, dtype=torch.bool, device=device)
    mask[:, :3, 1:3, 2:4] = True
    query = torch.randn(2, 32, device=device)
    action = torch.randn(2, 9, device=device)
    heads = [
        (EgoBeliefProbe(32, 64, 1, 4).to(device), lambda m: m(feat, mask).square().mean()),
        (EgoBeliefProbeV2(32, 64, 1, 4, 4, 8, 8, 4, 2).to(device),
         lambda m: m(feat, query)["logits"].square().mean()),
        (ActionDynamicsProbe(32, 9, 64, 1, 4).to(device),
         lambda m: m(feat[:, :-1], action).square().mean()),
    ]
    for module, loss_fn in heads:
        loss = loss_fn(module)
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)
        assert any(grad.abs().sum() > 0 for grad in grads)
    print(f"smoke test passed on {device}")


if __name__ == "__main__":
    main()
