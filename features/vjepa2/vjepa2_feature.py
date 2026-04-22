"""
V-JEPA 2 feature extractor using HuggingFace transformers.

Models (all use 64 frames, 256×256 input):
  - facebook/vjepa2-vitl-fpc64-256   ViT-L  300M  hidden=1024  24 layers
  - facebook/vjepa2-vith-fpc64-256   ViT-H  600M  hidden=1280  32 layers
  - facebook/vjepa2-vitg-fpc64-256   ViT-g  1.0B  hidden=1408  40 layers

Output per layer: (T_patches, H_patches, W_patches, hidden_size)
  ViT-L 256: (32, 16, 16, 1024)   ~32 MB FP16

Requires: pip install -U git+https://github.com/huggingface/transformers
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image


class VJEPA2Featurizer:
    """Wraps a HuggingFace V-JEPA 2 model for patch-level feature extraction."""

    def __init__(
        self,
        model_id: str = "facebook/vjepa2-vitl-fpc64-256",
        device: str = "cuda",
    ) -> None:
        from transformers import AutoModel, AutoVideoProcessor

        self.processor = AutoVideoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        ).to(device).eval()
        self.device = device

        cfg = self.model.config
        self.patch_size = (
            cfg.patch_size
            if isinstance(cfg.patch_size, int)
            else cfg.patch_size[0]
        )
        self.tubelet_size = cfg.tubelet_size
        self.crop_size = cfg.crop_size
        self.frames_per_clip = cfg.frames_per_clip
        self.hidden_size = cfg.hidden_size
        self.num_hidden_layers = cfg.num_hidden_layers

    @torch.no_grad()
    def __call__(
        self,
        frames: List[Image.Image],
        output_layers: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract features from V-JEPA 2 encoder.

        Args:
            frames: List of PIL RGB images (frames_per_clip images).
            output_layers: Encoder layer indices (0-based) to extract.
                           None → last encoder layer only.

        Returns:
            Dict[layer_idx → Tensor(T, H, W, C)]  in model dtype (fp16).
        """
        # Processor handles resize, crop, and normalisation
        video_np = np.stack([np.array(f) for f in frames])  # (T, H, W, 3)
        inputs = self.processor(video_np, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        need_hidden = output_layers is not None
        outputs = self.model(
            **inputs,
            output_hidden_states=need_hidden,
            skip_predictor=True,
        )

        T = self.frames_per_clip // self.tubelet_size
        H = self.crop_size // self.patch_size
        W = self.crop_size // self.patch_size

        result: Dict[int, torch.Tensor] = {}
        if output_layers is None:
            # Last encoder layer
            feat = outputs.last_hidden_state[0]  # (seq_len, hidden)
            feat = feat.reshape(T, H, W, self.hidden_size)
            result[self.num_hidden_layers - 1] = feat
        else:
            for idx in output_layers:
                # hidden_states: [embedding, layer-0, layer-1, …, layer-(N-1)]
                feat = outputs.hidden_states[idx + 1][0]
                feat = feat.reshape(T, H, W, self.hidden_size)
                result[idx] = feat

        return result


@lru_cache(maxsize=None)
def get_vjepa2_featurizer(
    model_id: str = "facebook/vjepa2-vitl-fpc64-256",
) -> VJEPA2Featurizer:
    """Build exactly once per model_id (cached)."""
    return VJEPA2Featurizer(model_id=model_id)
