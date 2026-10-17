"""One-step LingBot-Video DiT feature extraction (Wan-style protocol).

Architecture (MoE-30B-A3B local checkpoint)
-------------------------------------------
- ``LingBotVideoTransformer3DModel``: 48 blocks, hidden 2048, patch (1,2,2)
- VAE: ``AutoencoderKLWan`` — spatial /8, temporal /4 (first frame standalone)
- Flow matching: ``x_t = (1-σ)·x + σ·ε``, transformer timestep = ``σ*1000``
- Joint sequence ``[video | text]``; features keep video tokens only

CLI / cache naming uses thousandths of σ: ``t=250`` ⇒ σ=0.250,
filename ``feature_t250_layer{L}.sft``.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

# Local paths are deployment-specific: set them in the environment or pass them
# explicitly (the 2026 cluster used .../Consistency/lingbot-video).
DEFAULT_MODEL_PATH = os.environ.get("LINGBOT_MODEL_PATH", "")
DEFAULT_LINGBOT_ROOT = os.environ.get("LINGBOT_ROOT", "")
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832
DEFAULT_NUM_FRAMES = 49  # must be 4n+1
DEFAULT_T_MILLI = 250  # σ = 0.250
DEFAULT_LAYER = 30


def _ensure_lingbot_on_path(lingbot_root: str) -> None:
    root = os.path.abspath(lingbot_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def t_milli_to_sigma(t: Union[int, float]) -> float:
    """Map cache/CLI timestep to flow-matching σ ∈ (0, 1)."""
    t = float(t)
    if 0.0 < t <= 1.0:
        return t
    if 0.0 <= t <= 1000.0:
        return max(min(t / 1000.0, 0.999), 1e-4)
    raise ValueError(f"Unsupported timestep value: {t}")


def align_num_frames(num_frames: int) -> int:
    """Snap up to the next ``4n+1`` required by LingBotVideoPipeline."""
    n = int(num_frames)
    if n <= 1:
        return 1
    if (n - 1) % 4 == 0:
        return n
    return ((n - 1 + 3) // 4) * 4 + 1


class LingBotFeaturizer:
    """Extract multi-layer video-token features from LingBot-Video DiT."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_PATH,
        null_prompt: str = ".",
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        num_frames: int = DEFAULT_NUM_FRAMES,
        lingbot_root: str = DEFAULT_LINGBOT_ROOT,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        seed: int = 0,
        enable_cpu_offload: bool = True,
    ) -> None:
        if not model_id:
            raise ValueError("LingBot model path is not set: pass model_id or set LINGBOT_MODEL_PATH")
        if not lingbot_root:
            raise ValueError("LingBot source root is not set: pass lingbot_root or set LINGBOT_ROOT")
        _ensure_lingbot_on_path(lingbot_root)
        from lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline

        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"height/width must be multiples of 16, got {height}x{width}")
        self.num_frames = align_num_frames(num_frames)
        if self.num_frames != int(num_frames):
            # Keep explicit; callers should pad/crop to this length.
            pass

        self.model_id = model_id
        self.null_prompt = null_prompt if str(null_prompt).strip() else "."
        self.height = int(height)
        self.width = int(width)
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.seed = int(seed)

        self.pipe = LingBotVideoPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        )
        # Keep VAE in fp32 for stable encodes (matches inference scripts).
        if getattr(self.pipe, "vae", None) is not None:
            self.pipe.vae.to(dtype=torch.float32)

        self.enable_cpu_offload = bool(enable_cpu_offload)
        if self.enable_cpu_offload and hasattr(self.pipe, "enable_model_cpu_offload"):
            self.pipe.enable_model_cpu_offload(device=str(self.device))
        else:
            self.pipe.to(self.device)

        cfg = self.pipe.transformer.config
        self.patch_size = tuple(cfg.patch_size)
        self.hidden_size = int(cfg.hidden_size)
        self.num_layers = int(cfg.depth)
        self.in_channels = int(cfg.in_channels)
        self.vae_scale_spatial = int(getattr(self.pipe, "vae_scale_factor_spatial", 8))
        self.vae_scale_temporal = int(getattr(self.pipe, "vae_scale_factor_temporal", 4))

        self.latent_height = self.height // self.vae_scale_spatial
        self.latent_width = self.width // self.vae_scale_spatial
        self.num_latent_frames = (self.num_frames - 1) // self.vae_scale_temporal + 1
        self.feat_t = self.num_latent_frames // self.patch_size[0]
        self.feat_h = self.latent_height // self.patch_size[1]
        self.feat_w = self.latent_width // self.patch_size[2]

        with torch.no_grad():
            embeds, mask = self._encode_prompt(self.null_prompt)
            self.null_prompt_embeds = embeds
            self.null_prompt_mask = mask

    def _encode_prompt(self, prompt: str):
        """Encode text; keep inputs on the text-encoder device.

        Pipeline ``encode_prompt`` calls ``text_encoder.model(...)`` and bypasses
        accelerate cpu-offload hooks on ``text_encoder``. Under offload the
        weights stay on CPU while ``device=cuda`` would put ``input_ids`` on GPU.
        """
        te = self.pipe.text_encoder
        try:
            te_device = next(te.parameters()).device
        except StopIteration:
            te_device = torch.device("cpu")
        return self.pipe.encode_prompt(prompt, device=te_device)

    def _encode_video_latent(
        self,
        pixel: torch.Tensor,
        noise_generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """VAE-encode video in fp32 without CUDA autocast.

        ``LingBotVideoPipeline.encode_video_latent`` enables
        ``autocast(cuda, bf16)`` when the VAE is on GPU. Wan CausalConv3d then
        dispatches to ``aten::slow_conv3d_forward``, which has no CUDA kernel in
        this torch build.
        """
        from torchvision.transforms.functional import normalize as normalize_image_tensor

        vae = self.pipe.vae
        vae.to(device=self.device, dtype=torch.float32)
        video = pixel.to(device=self.device, dtype=torch.float32)
        bsz, channels, frames, height, width = video.shape
        flat_video = video.permute(0, 2, 1, 3, 4).reshape(
            bsz * frames, channels, height, width
        )
        norm_flat = normalize_image_tensor(
            flat_video, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=False
        )
        norm_video = (
            norm_flat.reshape(bsz, frames, channels, height, width)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        with torch.autocast("cuda", enabled=False):
            encoded = vae.encode(norm_video)
        if hasattr(encoded, "latent_dist"):
            latents = encoded.latent_dist.sample(generator=noise_generator)
        else:
            latents = encoded[0] if isinstance(encoded, tuple) else encoded
        latents = self.pipe._vae_latent_to_dit(latents).float()
        if self.enable_cpu_offload:
            vae.to("cpu")
            torch.cuda.empty_cache()
        return latents

    def _frames_to_tensor(self, frames: Sequence[Image.Image]) -> torch.Tensor:
        if len(frames) != self.num_frames:
            raise ValueError(
                f"Expected {self.num_frames} frames (4n+1), got {len(frames)}"
            )
        arrs = []
        for im in frames:
            if im.size != (self.width, self.height):
                im = im.resize((self.width, self.height), Image.LANCZOS)
            arrs.append(np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0)
        video = torch.from_numpy(np.stack(arrs, axis=0))  # (T,H,W,3)
        video = video.permute(3, 0, 1, 2).contiguous()  # (3,T,H,W)
        return video.unsqueeze(0)  # (1,3,T,H,W)

    @torch.no_grad()
    def forward(
        self,
        video: List[Image.Image],
        prompt: str = ".",
        t: Union[int, float] = DEFAULT_T_MILLI,
        output_layer_indices: Optional[List[int]] = None,
        ensemble_size: int = 1,
        noise_generator: Optional[torch.Generator] = None,
    ) -> Dict[int, torch.FloatTensor]:
        """Return ``{layer: (1, T*H*W, C)}`` video-token features."""
        if ensemble_size != 1:
            raise NotImplementedError("LingBot featurizer supports ensemble_size=1 only")
        if output_layer_indices is None:
            output_layer_indices = [DEFAULT_LAYER]
        output_layer_indices = [int(x) for x in output_layer_indices]
        for idx in output_layer_indices:
            if idx < 0 or idx >= self.num_layers:
                raise ValueError(
                    f"layer {idx} out of range for depth={self.num_layers} (valid 0..{self.num_layers - 1})"
                )

        sigma = t_milli_to_sigma(t)
        frames = list(video)
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        if len(frames) > self.num_frames:
            frames = frames[: self.num_frames]

        pixel = self._frames_to_tensor(frames)
        latents = self._encode_video_latent(
            pixel, noise_generator=noise_generator
        )

        generator = noise_generator
        if generator is None:
            gen_device = "cpu" if latents.device.type == "cpu" else str(latents.device)
            generator = torch.Generator(device=gen_device)
            generator.manual_seed(self.seed)
        noise = torch.randn(
            latents.shape, generator=generator, device=latents.device, dtype=torch.float32
        )
        # Flow matching: x_t = (1-σ) x + σ ε
        latents_noisy = (1.0 - sigma) * latents + sigma * noise
        timestep = torch.tensor(
            [sigma * 1000.0], device=latents.device, dtype=torch.float32
        )

        if (not str(prompt).strip()) or prompt == self.null_prompt:
            prompt_embeds = self.null_prompt_embeds
            prompt_mask = self.null_prompt_mask
        else:
            prompt_embeds, prompt_mask = self._encode_prompt(prompt)

        # Match latent / conditioning device to transformer execution device.
        transformer = self.pipe.transformer
        hook = getattr(transformer, "_hf_hook", None)
        if hook is not None and hasattr(hook, "pre_forward"):
            hook.pre_forward(transformer)
        exec_device = next(transformer.parameters()).device

        latents_noisy = latents_noisy.to(device=exec_device, dtype=self.torch_dtype)
        timestep = timestep.to(device=exec_device)
        prompt_embeds = prompt_embeds.to(device=exec_device, dtype=self.torch_dtype)
        prompt_mask = prompt_mask.to(device=exec_device)

        n_video = self.feat_t * self.feat_h * self.feat_w
        wanted = set(output_layer_indices)
        captured: Dict[int, torch.Tensor] = {}
        handles = []

        def _make_hook(layer_idx: int):
            def _hook(_module, _inp, out):
                # out: (B, n_video + L_text, C)
                feats = out[:, :n_video].detach()
                captured[layer_idx] = feats

            return _hook

        for i, block in enumerate(transformer.blocks):
            if i in wanted:
                handles.append(block.register_forward_hook(_make_hook(i)))

        try:
            _ = transformer(
                hidden_states=latents_noisy,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_mask,
                return_dict=True,
            )
        finally:
            for h in handles:
                h.remove()

        out: Dict[int, torch.FloatTensor] = {}
        for idx in output_layer_indices:
            if idx not in captured:
                continue
            feat = captured[idx].float()
            if feat.shape[1] != n_video:
                raise RuntimeError(
                    f"Layer {idx}: expected {n_video} video tokens, got {feat.shape[1]}"
                )
            out[idx] = feat.cpu()
        return out


@lru_cache(maxsize=None)
def get_lingbot_featurizer(
    model_id: str = DEFAULT_MODEL_PATH,
    null_prompt: str = ".",
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    num_frames: int = DEFAULT_NUM_FRAMES,
    lingbot_root: str = DEFAULT_LINGBOT_ROOT,
    enable_cpu_offload: bool = True,
):
    return LingBotFeaturizer(
        model_id=model_id,
        null_prompt=null_prompt if str(null_prompt).strip() else ".",
        height=height,
        width=width,
        num_frames=num_frames,
        lingbot_root=lingbot_root,
        enable_cpu_offload=enable_cpu_offload,
    )
