#!/usr/bin/env python3
"""Wan L29 specialization of the verified independent P8 final extractor."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import extract_independent_p8_final_features as base  # noqa: E402


base.MODEL_CONTRACTS = {
    "wan": {"layer": 29, "timestep": 749, "input_frames": 81, "width": 1536},
}


class WanL29Backend(base.Backend):
    """Keep the accepted independent Wan input contract; change only layer 20 to 29."""

    def extract(self, paths: list[Path], seed: int) -> tuple[Any, list[int], dict[str, Any]]:
        from features.wan.extract_features import reshape_to_t_h_w_c

        torch = self.torch
        images = [base.load_pil(path, (832, 480)) for path in paths]
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            output = self.backend.forward(
                video=images,
                prompt="",
                t=749,
                output_layer_indices=[29],
                ensemble_size=1,
                noise_generator=generator,
            )
        raw = reshape_to_t_h_w_c(output[29]).float().cpu()
        return raw.mean(dim=(0, 1, 2)), list(raw.shape), {
            "noise_seed": seed,
            "prompt": "",
            "ensemble": 1,
            "layer_override": 29,
        }


base.Backend = WanL29Backend
# Make the sidecar bind to this specialization rather than the imported base.
base.__file__ = __file__


if __name__ == "__main__":
    raise SystemExit(base.main())
