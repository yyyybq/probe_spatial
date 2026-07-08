"""Layer naming and defaults for cached feature files."""

from __future__ import annotations

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


def feat_postfix_for_layer(vfm_name: str, layer: int, t: int = 749) -> str:
    if vfm_name in {"wan", "cogvideox", "aether", "opensora"}:
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
    ``feature_layer{L}.sft`` for V-JEPA2 / MLLM caches.
    """

    if vfm_name in {"vjepa", "dino"}:
        return "feature.sft"

    if isinstance(feature_layer, str):
        feature_layer = None if feature_layer in {"", "None", "none", "null"} else int(feature_layer)
    if isinstance(feature_timestep, str):
        feature_timestep = None if feature_timestep in {"", "None", "none", "null"} else int(feature_timestep)

    if feature_layer is not None:
        t = 749 if feature_timestep is None else int(feature_timestep)
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
