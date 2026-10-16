"""One-step MiniMax-H3 DiT feature extraction (Wan-style protocol).

Protocol
--------
1. Resize / pad RGB frames to an aligned clip length ``17*n+5``.
2. ImageNet-normalize and VAE-encode to video latents.
3. Rectified-flow noise at a fixed continuous timestep ``t ∈ (0, 1)``
   (``x_t = t * x_0 + (1 - t) * noise``; smaller ``t`` ⇒ noisier).
4. Pack ``[text | target audio | target video]`` (no keyframes) and run a
   single transformer forward.
5. Collect hidden states after selected ``transformer_blocks``, keep only
   video rows, and reshape to ``(T, H, W, C)``.

CLI / cache naming uses an integer thousandths code: ``t=250`` means ``0.250``,
filename ``feature_t250_layer{L}.sft``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL_PATH = "/mnt/umm/users/yinbaiqiao/Consistency/minimax-h3/models/MiniMax-H3"
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832
DEFAULT_NUM_FRAMES = 73  # 17 * 4 + 5
DEFAULT_T_MILLI = 250  # continuous t = 0.250
DEFAULT_LAYER = 30


def t_milli_to_continuous(t: Union[int, float]) -> float:
    """Map cache/CLI timestep to MiniMax continuous noise level in (0, 1)."""
    t = float(t)
    if 0.0 < t <= 1.0:
        return t
    if 0.0 <= t <= 1000.0:
        return max(min(t / 1000.0, 0.999), 1e-4)
    raise ValueError(f"Unsupported timestep value: {t}")


def t_to_milli(t: Union[int, float]) -> int:
    """Canonical integer code used in feature filenames."""
    cont = t_milli_to_continuous(t)
    return int(round(cont * 1000.0))


class MiniMaxH3Featurizer:
    """Extract multi-layer video-token features from MiniMax-H3."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_PATH,
        null_prompt: str = ".",
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        num_frames: int = DEFAULT_NUM_FRAMES,
        memory_reserve_margin: str = "12GB",
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        local_files_only: bool = True,
        seed: int = 0,
    ) -> None:
        from diffusers import ComponentsManager, ModularPipeline
        from diffusers.modular_pipelines.minimax_h3.encoders import (
            MiniMaxH3TextEncoderStep,
        )
        from diffusers.modular_pipelines.minimax_h3.packing import (
            align_num_frames,
            audio_latent_num_frames,
            video_latent_num_frames,
        )

        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(f"height/width must be multiples of 32, got {height}x{width}")

        self.model_id = model_id
        # Empty prompts yield zero text tokens and break the packed layout.
        self.null_prompt = null_prompt if str(null_prompt).strip() else "."
        self.height = int(height)
        self.width = int(width)
        self.num_frames = int(align_num_frames(num_frames))
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.seed = int(seed)
        self.latent_height = self.height // 16
        self.latent_width = self.width // 16
        self.num_latent_frames = int(video_latent_num_frames(self.num_frames))
        self.num_audio_latents = int(audio_latent_num_frames(self.num_frames))

        manager = ComponentsManager()
        pipe = ModularPipeline.from_pretrained(
            model_id,
            components_manager=manager,
            local_files_only=local_files_only,
        )
        pipe.load_components(
            dtype=torch_dtype,
            pretrained_model_name_or_path=model_id,
            local_files_only=local_files_only,
        )
        manager.enable_auto_cpu_offload(
            device=str(self.device),
            memory_reserve_margin=memory_reserve_margin,
        )
        self.pipe = pipe
        self.manager = manager

        self.patch_size = tuple(getattr(pipe, "patch_size", (1, 2, 2)))
        if hasattr(pipe, "transformer") and hasattr(pipe.transformer, "config"):
            self.patch_size = tuple(pipe.transformer.config.patch_size)
            self.hidden_size = int(pipe.transformer.config.hidden_size)
            self.audio_in_channels = int(pipe.transformer.config.audio_in_channels)
            self.in_channels = int(pipe.transformer.config.in_channels)
        else:
            self.hidden_size = 5376
            self.audio_in_channels = 32
            self.in_channels = 24

        self._text_step = MiniMaxH3TextEncoderStep
        with torch.no_grad():
            self.null_prompt_embeds, self.null_text_token_tags = self._encode_prompt(
                self.null_prompt
            )

        ph, pw = self.patch_size[1], self.patch_size[2]
        self.feat_t = self.num_latent_frames // self.patch_size[0]
        self.feat_h = self.latent_height // ph
        self.feat_w = self.latent_width // pw

    def _components(self):
        return self.pipe

    def _encode_prompt(self, prompt: str):
        embeds, tags = self._text_step.encode_prompt(
            self._components(),
            prompt=prompt,
            images=None,
            device=self.device,
            dtype=self.torch_dtype,
        )
        return embeds, tags.to(dtype=torch.long)

    def _frames_to_tensor(self, frames: Sequence[Image.Image]) -> torch.Tensor:
        from diffusers.modular_pipelines.minimax_h3.packing import (
            MINIMAX_H3_PIXEL_MEAN,
            MINIMAX_H3_PIXEL_STD,
        )

        if len(frames) != self.num_frames:
            raise ValueError(
                f"Expected {self.num_frames} frames after padding, got {len(frames)}"
            )
        arrs = []
        for im in frames:
            if im.size != (self.width, self.height):
                im = im.resize((self.width, self.height), Image.LANCZOS)
            arrs.append(np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0)
        video = torch.from_numpy(np.stack(arrs, axis=0))  # (T, H, W, 3)
        video = video.permute(3, 0, 1, 2).contiguous()  # (3, T, H, W)
        mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, dtype=torch.float32).view(3, 1, 1, 1)
        std = torch.tensor(MINIMAX_H3_PIXEL_STD, dtype=torch.float32).view(3, 1, 1, 1)
        video = (video - mean) / std
        return video.unsqueeze(0)  # (1, 3, T, H, W)

    def _encode_video_latents(self, video: torch.Tensor) -> torch.Tensor:
        vae = self.pipe.vae
        # Ensure offload hook moves VAE in if needed.
        hook = getattr(vae, "_hf_hook", None)
        if hook is not None and hasattr(hook, "pre_forward"):
            hook.pre_forward(vae)
        posterior = vae.encode(video.to(device=self.device, dtype=torch.float32)).latent_dist
        latents = posterior.mean
        latents_mean = torch.tensor(
            vae.config.latents_mean, device=latents.device, dtype=torch.float32
        ).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(
            vae.config.latents_std, device=latents.device, dtype=torch.float32
        ).view(1, -1, 1, 1, 1)
        latents = (latents.float() - latents_mean) / latents_std
        if latents.shape[2] != self.num_latent_frames:
            raise RuntimeError(
                f"VAE returned {latents.shape[2]} latent frames, expected {self.num_latent_frames}"
            )
        return latents

    @torch.no_grad()
    def forward(
        self,
        video: List[Image.Image],
        prompt: str = "",
        t: Union[int, float] = DEFAULT_T_MILLI,
        output_layer_indices: Optional[List[int]] = None,
        ensemble_size: int = 1,
    ) -> Dict[int, torch.FloatTensor]:
        """Return ``{layer: (1, T*H*W, C)}`` video-token features."""
        from diffusers.modular_pipelines.minimax_h3.packing import (
            MINIMAX_H3_AUDIO_CHANNELS,
            build_packed_sequence,
            build_row_timesteps,
            patchify_video_latents,
        )

        if ensemble_size != 1:
            raise NotImplementedError("MiniMax-H3 featurizer currently supports ensemble_size=1 only")
        if output_layer_indices is None:
            output_layer_indices = [DEFAULT_LAYER]
        output_layer_indices = [int(x) for x in output_layer_indices]
        t_cont = t_milli_to_continuous(t)

        if prompt == self.null_prompt:
            prompt_embeds = self.null_prompt_embeds
            text_token_tags = self.null_text_token_tags
        else:
            prompt_embeds, text_token_tags = self._encode_prompt(prompt)

        frames = list(video)
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        if len(frames) > self.num_frames:
            frames = frames[: self.num_frames]

        pixel = self._frames_to_tensor(frames)
        latents = self._encode_video_latents(pixel).to(device=self.device, dtype=torch.float32)

        gen_device = self.device if self.device.type == "cpu" else torch.device("cuda")
        generator = torch.Generator(device=gen_device)
        generator.manual_seed(self.seed)
        noise = torch.randn(
            latents.shape, generator=generator, device=latents.device, dtype=torch.float32
        )
        latents_noisy = self.pipe.scheduler.scale_noise(latents, t_cont, noise)
        video_rows = patchify_video_latents(latents_noisy, self.patch_size).unsqueeze(0)

        audio_rows = torch.randn(
            1,
            self.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS,
            self.audio_in_channels,
            generator=generator,
            device=latents.device,
            dtype=torch.float32,
        )

        layout = build_packed_sequence(
            text_token_tags=text_token_tags.cpu(),
            num_latent_frames=self.num_latent_frames,
            latent_height=self.latent_height,
            latent_width=self.latent_width,
            num_audio_latents=self.num_audio_latents,
            patch_size=self.patch_size,
            keyframe_anchors=(),
        )
        unique_timesteps, timestep_indices = build_row_timesteps(
            layout,
            video_timestep=t_cont,
            audio_timestep=t_cont,
            condition_video_timestep=t_cont,
            condition_audio_timestep=t_cont,
        )

        # Move structural tensors to device.
        position_ids = layout.position_ids.to(self.device)
        token_tags = layout.token_tags.to(self.device)
        video_indices = layout.video_indices.to(self.device)
        audio_indices = layout.audio_indices.to(self.device)
        text_indices = layout.text_indices.to(self.device)
        unique_timesteps = unique_timesteps.to(self.device)
        timestep_indices = timestep_indices.to(self.device)
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.torch_dtype)
        video_rows = video_rows.to(device=self.device, dtype=torch.float32)
        audio_rows = audio_rows.to(device=self.device, dtype=torch.float32)

        transformer = self.pipe.transformer
        hook = getattr(transformer, "_hf_hook", None)
        if hook is not None and hasattr(hook, "pre_forward"):
            hook.pre_forward(transformer)

        wanted = set(output_layer_indices)
        captured: Dict[int, torch.Tensor] = {}
        handles = []

        def _make_hook(layer_idx: int):
            def _hook(_module, _inp, out):
                # out: (B, S, C)
                feats = out.index_select(1, video_indices).detach()
                captured[layer_idx] = feats

            return _hook

        for i, block in enumerate(transformer.transformer_blocks):
            if i in wanted:
                handles.append(block.register_forward_hook(_make_hook(i)))

        try:
            _ = transformer(
                hidden_states=video_rows,
                audio_hidden_states=audio_rows,
                encoder_hidden_states=prompt_embeds,
                timestep=unique_timesteps,
                timestep_indices=timestep_indices,
                token_tags=token_tags,
                position_ids=position_ids,
                video_indices=video_indices,
                audio_indices=audio_indices,
                text_indices=text_indices,
                return_dict=True,
            )
        finally:
            for h in handles:
                h.remove()

        expected = self.feat_t * self.feat_h * self.feat_w
        out: Dict[int, torch.FloatTensor] = {}
        for idx in output_layer_indices:
            if idx not in captured:
                continue
            feat = captured[idx].float()
            if feat.shape[1] != expected:
                raise RuntimeError(
                    f"Layer {idx}: expected {expected} video tokens, got {feat.shape[1]}"
                )
            # Fail closed on a non-finite forward: deep MiniMax blocks reach
            # |x| ~ 4e6, so a NaN/inf here is a model-side fault, not a storage
            # rounding issue, and must never be written to a cache.
            finite = torch.isfinite(feat)
            if not bool(finite.all().item()):
                raise FloatingPointError(
                    f"MiniMax layer {idx} forward produced non-finite values: "
                    f"total={int((~finite).sum().item())} "
                    f"nan={int(torch.isnan(feat).sum().item())} "
                    f"posinf={int(torch.isposinf(feat).sum().item())} "
                    f"neginf={int(torch.isneginf(feat).sum().item())}"
                )
            out[idx] = feat.cpu()
        return out


@lru_cache(maxsize=None)
def get_minimax_h3_featurizer(
    model_id: str = DEFAULT_MODEL_PATH,
    null_prompt: str = ".",
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    num_frames: int = DEFAULT_NUM_FRAMES,
    memory_reserve_margin: str = "12GB",
):
    """Cached loader; one MiniMax-H3 stack per process."""
    return MiniMaxH3Featurizer(
        model_id=model_id,
        null_prompt=null_prompt if str(null_prompt).strip() else ".",
        height=height,
        width=width,
        num_frames=num_frames,
        memory_reserve_margin=memory_reserve_margin,
    )
