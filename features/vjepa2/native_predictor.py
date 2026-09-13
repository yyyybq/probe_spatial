"""Strict V-JEPA2 native-predictor contract used by D3.

The Hugging Face convenience ``forward`` encodes every supplied patch before
gathering ``context_mask``.  For D3 we instead gather patch embeddings before
encoder self-attention, preserve their absolute [time,height,width] indices,
and pass only those context tokens to the frozen encoder and predictor.
"""

from __future__ import annotations

import torch


def tubelet_token_indices(
    tubelet_indices: list[int] | tuple[int, ...], spatial_size=(16, 16), *, device=None,
) -> torch.Tensor:
    """Flatten [time,height,width] positions in the order used by Conv3d.flatten."""
    height, width = (int(v) for v in spatial_size)
    spatial = height * width
    rows = [torch.arange(t * spatial, (t + 1) * spatial, device=device) for t in tubelet_indices]
    return torch.cat(rows) if rows else torch.empty(0, dtype=torch.long, device=device)


def observation_to_tubelet(observation_index: int, tubelet_size: int = 2) -> int:
    return int(observation_index) // int(tubelet_size)


def make_prefix_safe_clip(
    pixel_values: torch.Tensor,
    prefix_observations: int,
    total_frames: int = 64,
    pad_mode: str = "repeat_tail",
) -> torch.Tensor:
    """Remove future pixels before any encoder operation.

    ``pixel_values`` is ``B,T,C,H,W``.  Supplying two clips with the same prefix
    and arbitrary future content therefore produces bit-identical safe clips.
    """
    if pixel_values.ndim != 5:
        raise ValueError(f"expected B,T,C,H,W; got {tuple(pixel_values.shape)}")
    prefix = pixel_values[:, : int(prefix_observations)]
    if prefix.shape[1] == 0:
        raise ValueError("prefix must contain at least one observation")
    if prefix.shape[1] > total_frames:
        raise ValueError("prefix exceeds total_frames")
    missing = total_frames - prefix.shape[1]
    if pad_mode == "repeat_tail":
        padding = prefix[:, -1:].expand(-1, missing, -1, -1, -1)
    elif pad_mode == "zeros":
        padding = prefix.new_zeros(prefix.shape[0], missing, *prefix.shape[2:])
    else:
        raise ValueError("pad_mode must be repeat_tail or zeros")
    return torch.cat((prefix, padding), dim=1)


def _masked_encoder(model, safe_pixel_values: torch.Tensor, context_mask: torch.Tensor):
    """Run the encoder with masking before attention and absolute RoPE ids."""
    encoder = model.encoder
    embeddings = encoder.embeddings(safe_pixel_values)
    batch = embeddings.shape[0]
    mask = context_mask.to(embeddings.device).unsqueeze(0).expand(batch, -1)
    gather = mask.unsqueeze(-1).expand(-1, -1, embeddings.shape[-1])
    hidden = torch.gather(embeddings, 1, gather)
    for layer in encoder.layer:
        hidden = layer(hidden, mask)[0]
    return encoder.layernorm(hidden), mask


@torch.no_grad()
def predict_future_tubelets(
    model,
    pixel_values: torch.Tensor,
    *,
    prefix_observations: int = 8,
    target_observation_indices=(9, 11),
    pad_mode: str = "repeat_tail",
):
    """Predict final-space target tokens without exposing future RGB.

    The fixed D-series label rule uses the final observation of each two-frame
    tubelet: observation 9 for h2 and observation 11 for h4 when prefix=8.
    """
    config = model.config
    tubelet_size = int(config.tubelet_size)
    spatial = (int(config.crop_size // config.patch_size),) * 2
    context_tubelets = list(range(prefix_observations // tubelet_size))
    target_tubelets = [observation_to_tubelet(i, tubelet_size) for i in target_observation_indices]
    if len(set(target_tubelets)) != len(target_tubelets):
        raise ValueError("target observations must map to distinct future tubelets")
    safe = make_prefix_safe_clip(pixel_values, prefix_observations, int(config.frames_per_clip), pad_mode)
    context_mask = tubelet_token_indices(context_tubelets, spatial, device=safe.device)
    context, context_positions = _masked_encoder(model, safe, context_mask)
    outputs = {}
    for obs, tubelet in zip(target_observation_indices, target_tubelets):
        target = tubelet_token_indices([tubelet], spatial, device=safe.device)
        target = target.unsqueeze(0).expand(pixel_values.shape[0], -1)
        pred = model.predictor(context, [context_positions], [target]).last_hidden_state
        outputs[int(obs)] = pred.reshape(pixel_values.shape[0], *spatial, int(config.hidden_size))
    return outputs


@torch.no_grad()
def native_target_reference(model, teacher_pixel_values: torch.Tensor, target_observation_indices=(9, 11)):
    """Oracle native teacher reference: final normalized full-clip encoder space."""
    config = model.config
    spatial = (int(config.crop_size // config.patch_size),) * 2
    encoded = model.encoder(teacher_pixel_values).last_hidden_state
    outputs = {}
    for obs in target_observation_indices:
        tubelet = observation_to_tubelet(int(obs), int(config.tubelet_size))
        mask = tubelet_token_indices([tubelet], spatial, device=encoded.device)
        gather = mask[None, :, None].expand(encoded.shape[0], -1, encoded.shape[-1])
        outputs[int(obs)] = torch.gather(encoded, 1, gather).reshape(
            encoded.shape[0], *spatial, int(config.hidden_size)
        )
    return outputs
