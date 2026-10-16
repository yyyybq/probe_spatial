"""Layer naming and defaults for cached feature files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FeatureLayerSpec:
    """Static layer metadata for one feature backend."""

    default_layer: int
    num_layers: int | None
    feat_postfix: str
    in_channels: int | None
    note: str = ""

    @property
    def last_layer(self) -> int | None:
        return self.num_layers - 1 if self.num_layers is not None else None

    @property
    def default_timestep(self) -> int | None:
        """Scheduler timestep encoded in the default cache name, if any."""
        match = _POSTFIX_TIMESTEP.match(self.feat_postfix)
        return int(match.group(1)) if match else None


_POSTFIX_TIMESTEP = re.compile(r"^_t(\d+)_layer")

# Diffusion backbones whose cache names carry the scheduler timestep:
# ``feature_t{t}_layer{L}.sft``. Encoder backbones use ``feature_layer{L}.sft``.
_TIMESTEP_POSTFIX_VFMS = frozenset(
    {"wan", "cogvideox", "minimax_h3", "lingbot", "aether", "opensora"}
)

FEATURE_LAYER_SPECS: dict[str, FeatureLayerSpec] = {
    # Current diagnostic-suite defaults.
    "wan": FeatureLayerSpec(
        default_layer=20,
        num_layers=30,
        feat_postfix="_t749_layer20",
        in_channels=1536,
        note="Wan2.1-T2V-1.3B diffusion transformer block 20 at t=749.",
    ),
    "cogvideox": FeatureLayerSpec(
        default_layer=20,
        num_layers=42,
        feat_postfix="_t749_layer20",
        in_channels=3072,
        note="CogVideoX-5B transformer block 20 at t=749.",
    ),
    "vjepa2": FeatureLayerSpec(
        default_layer=23,
        num_layers=24,
        feat_postfix="_layer23",
        in_channels=1024,
        note="V-JEPA2 ViT-L last encoder block, 0-based layer 23.",
    ),
    "vjepa2-vitl": FeatureLayerSpec(
        default_layer=23,
        num_layers=24,
        feat_postfix="_layer23",
        in_channels=1024,
    ),
    "vjepa2-vith": FeatureLayerSpec(
        default_layer=31,
        num_layers=32,
        feat_postfix="_layer31",
        in_channels=1280,
    ),
    "vjepa2-vitg": FeatureLayerSpec(
        default_layer=39,
        num_layers=40,
        feat_postfix="_layer39",
        in_channels=1408,
    ),
    "qwen2_5_vl": FeatureLayerSpec(
        default_layer=-1,
        num_layers=32,
        feat_postfix="_layer-1",
        in_channels=3584,
        note="Default cache is Qwen2.5-VL visual-merger output; layers 0-31 are vision-tower blocks.",
    ),
    "qwen2_5_vl_3b": FeatureLayerSpec(
        default_layer=-1,
        num_layers=32,
        feat_postfix="_layer-1",
        in_channels=2048,
        note="Default cache is Qwen2.5-VL-3B visual-merger output; layers 0-31 are vision-tower blocks.",
    ),
    "bagel": FeatureLayerSpec(
        default_layer=-1,
        num_layers=None,
        feat_postfix="_layer-1",
        in_channels=3584,
        note="Default cache is the backend's last hidden/visual-token layer.",
    ),
    # ---- New VFMs --------------------------------------------------------- #
    "dino": FeatureLayerSpec(
        default_layer=0,
        num_layers=1,
        feat_postfix="",
        in_channels=1024,
        note="DINOv2-Large last hidden state; 420×728 input → 30×52 patch grid.",
    ),
    "aether": FeatureLayerSpec(
        default_layer=1,
        num_layers=42,
        feat_postfix="_t749_layer1",
        in_channels=3072,
        note="Aether (CogVideoX-5B backbone) transformer block at t=749.",
    ),
    "opensora": FeatureLayerSpec(
        default_layer=10,
        num_layers=None,
        feat_postfix="_t1_layer10",
        in_channels=3072,
        note="Open-Sora MMDiT single_block; default timestep=0.25 (t index 1).",
    ),
    "f3r": FeatureLayerSpec(
        default_layer=24,
        num_layers=25,
        feat_postfix="_l24",
        in_channels=1024,
        note="Fast3R ViT-L transformer block 24; 288×512 input → 18×32 patch grid.",
    ),
    # Canonical-V2 diffusion backbones extracted at scheduler timestep 250.
    # Both caches are named feature_t250_layer{L}.sft (features/minimax_h3,
    # features/lingbot). Campaign ledgers call the last block by its 1-based
    # count (MiniMax "L50" = disk layer 49, LingBot "L48" = disk layer 47);
    # the registry only knows 0-based disk layers.
    "minimax_h3": FeatureLayerSpec(
        default_layer=30,
        num_layers=50,
        feat_postfix="_t250_layer30",
        in_channels=5376,
        note="MiniMax-H3 video DiT block 30 at t=250; 480×832 input → 15×26 token grid (fp16 caches saturate at layers ≥20).",
    ),
    "lingbot": FeatureLayerSpec(
        default_layer=30,
        num_layers=48,
        feat_postfix="_t250_layer30",
        in_channels=2048,
        note="LingBot-Video DiT block 30 at t=250; 480×832 input → 30×52 token grid.",
    ),
}


def canonical_vfm_name(vfm_name: str, model_id: str | None = None) -> str:
    """Return the layer-spec key for a VFM/backend plus optional model id."""

    if vfm_name == "vjepa2" and model_id:
        model_id_l = model_id.lower()
        if "vith" in model_id_l:
            return "vjepa2-vith"
        if "vitg" in model_id_l:
            return "vjepa2-vitg"
        return "vjepa2-vitl"
    return vfm_name


def get_feature_layer_spec(vfm_name: str, model_id: str | None = None) -> FeatureLayerSpec | None:
    return FEATURE_LAYER_SPECS.get(canonical_vfm_name(vfm_name, model_id))


def default_output_layers(vfm_name: str, model_id: str | None = None) -> list[int]:
    spec = get_feature_layer_spec(vfm_name, model_id)
    if spec is None:
        return [0]
    return [spec.default_layer]


def all_output_layers(vfm_name: str, model_id: str | None = None) -> list[int]:
    spec = get_feature_layer_spec(vfm_name, model_id)
    if spec is None or spec.num_layers is None:
        raise ValueError(
            f"No static layer count is registered for {vfm_name}; pass --output-layers explicitly."
        )
    return list(range(spec.num_layers))


def default_feature_timestep(vfm_name: str, model_id: str | None = None) -> int:
    """Scheduler timestep used by a backend's default cache name (749 if unknown)."""
    spec = get_feature_layer_spec(vfm_name, model_id)
    if spec is not None and spec.default_timestep is not None:
        return spec.default_timestep
    return 749


def feat_postfix_for_layer(vfm_name: str, layer: int, t: int | None = None) -> str:
    if vfm_name in _TIMESTEP_POSTFIX_VFMS:
        t = default_feature_timestep(vfm_name) if t is None else int(t)
        return f"_t{t}_layer{layer}"
    if vfm_name == "f3r":
        return f"_l{layer}"
    if vfm_name == "dino":
        return ""
    return f"_layer{layer}"


def default_feature_postfix(vfm_name: str, model_id: str | None = None) -> str:
    spec = get_feature_layer_spec(vfm_name, model_id)
    if spec is None:
        return "_t749_layer20"
    return spec.feat_postfix


def default_feature_channels(vfm_name: str, fallback: int = 1536) -> int:
    spec = get_feature_layer_spec(vfm_name)
    if spec is None or spec.in_channels is None:
        return fallback
    return spec.in_channels


# Spatial token grid (H, W) used by each VFM at its standard input resolution.
# wan: 480×832 → 30×52 patches (patch_size=16)
# cogvideox: 288×512 → 18×32 patches (patch_size=16)
# vjepa2*: 256×256 → 16×16 patches (patch_size=16)
_DEFAULT_FEATURE_HW: dict[str, tuple[int, int]] = {
    "wan": (30, 52),
    "cogvideox": (18, 32),
    "vjepa2": (16, 16),
    "vjepa2-vitl": (16, 16),
    "vjepa2-vith": (16, 16),
    "vjepa2-vitg": (16, 16),
    "dino": (30, 52),    # 420×728, patch_size=14
    "aether": (30, 45),  # 480×720, patch_size=16
    "opensora": (30, 53),  # 480×848, patch_size=16
    "f3r": (18, 32),     # 288×512, patch_size=16
    "minimax_h3": (15, 26),  # 480×832 → 30×52 latent → (1,2,2) patches; cached files are (T,15,26,5376)
    "lingbot": (30, 52),     # 480×832; cached files are (T,30,52,2048)
}


def default_feature_hw(vfm_name: str, fallback: tuple[int, int] = (18, 32)) -> tuple[int, int]:
    """Return the default (H, W) spatial token grid for a given VFM."""
    name = canonical_vfm_name(vfm_name)
    return _DEFAULT_FEATURE_HW.get(name, fallback)


def feature_filename(
    vfm_name: str,
    *,
    feat_postfix: str | None = None,
    feature_layer: int | str | None = None,
    feature_timestep: int | str | None = None,
    feature_prefix: str = "feature",
) -> str:
    """Return the safetensors filename for one cached feature layer.

    ``feat_postfix`` keeps old configs working.  ``feature_layer`` is the new
    sweep-friendly path and maps to the extractor filenames:
    ``feature_t{t}_layer{L}.sft`` for diffusion VFMs and
    ``feature_layer{L}.sft`` for V-JEPA2 / MLLM caches. When ``feature_layer``
    is given without ``feature_timestep``, the backend's registered default
    timestep is used (749 for Wan/CogVideoX/Aether, 250 for MiniMax/LingBot).
    """

    if vfm_name in {"vjepa", "dino"}:
        return "feature.sft"

    if isinstance(feature_layer, str):
        feature_layer = None if feature_layer in {"", "None", "none", "null"} else int(feature_layer)
    if isinstance(feature_timestep, str):
        feature_timestep = None if feature_timestep in {"", "None", "none", "null"} else int(feature_timestep)

    if feature_layer is not None:
        t = default_feature_timestep(vfm_name) if feature_timestep is None else int(feature_timestep)
        return f"{feature_prefix}{feat_postfix_for_layer(vfm_name, int(feature_layer), t)}.sft"

    postfix = feat_postfix if feat_postfix is not None else default_feature_postfix(vfm_name)
    return f"{feature_prefix}{postfix}.sft"


def parse_layers_arg(
    values: Iterable[str] | None,
    *,
    vfm_name: str,
    model_id: str | None = None,
    all_layers: bool = False,
) -> list[int]:
    """Parse CLI layer values, supporting `default`, `last`, and `all`."""

    if all_layers:
        return all_output_layers(vfm_name, model_id)
    if not values:
        return default_output_layers(vfm_name, model_id)

    parsed: list[int] = []
    for value in values:
        value_l = str(value).lower()
        if value_l == "default":
            parsed.extend(default_output_layers(vfm_name, model_id))
        elif value_l == "last":
            spec = get_feature_layer_spec(vfm_name, model_id)
            if spec is None or spec.last_layer is None:
                parsed.append(-1)
            else:
                parsed.append(spec.last_layer)
        elif value_l == "all":
            parsed.extend(all_output_layers(vfm_name, model_id))
        else:
            parsed.append(int(value))

    out = []
    seen = set()
    for layer in parsed:
        if layer not in seen:
            out.append(layer)
            seen.add(layer)
    return out
