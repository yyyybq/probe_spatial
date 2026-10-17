import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from features.noise_seed import (
    DETERMINISTIC_NOISE_CONTRACT,
    NO_NOISE_SEED_SCHEME,
    deterministic_no_noise_metadata,
)
from features.run_inscene15k import (
    backend_generation_contract,
    validate_seeded_cache_policy,
)


ROOT = Path(__file__).resolve().parents[2]


def _args(vfm):
    return SimpleNamespace(
        mode="streaming_prefix",
        target_from_streaming_windows=False,
        vfm=vfm,
        no_cache_checksum=False,
        temporal_sampling="motion_uniform",
        streaming_window_obs_len=28,
        streaming_motion_step=0.35,
        streaming_rotation_weight=0.5,
        streaming_window_stride=8,
        streaming_max_windows_per_scene=4,
        prefix_lengths=[8, 12, 16, 24],
        target_prefix_lengths=[8, 12, 16, 24],
        target_horizons=[1, 2, 4],
    )


def _calls(relative_path, function_name):
    tree = ast.parse((ROOT / relative_path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return [child for child in ast.walk(node) if isinstance(child, ast.Call)]
    raise AssertionError(f"{function_name} not found in {relative_path}")


def test_no_noise_backends_have_explicit_provenance_dict():
    expected = {
        "feature_generation_contract": DETERMINISTIC_NOISE_CONTRACT,
        "noise_seed_scheme": NO_NOISE_SEED_SCHEME,
        "noise_seed_scope": "no_random_draws",
    }
    assert deterministic_no_noise_metadata() == expected
    for vfm in ("dino", "vjepa2", "videomae"):
        assert backend_generation_contract(_args(vfm)) == expected
        validate_seeded_cache_policy(
            _args(vfm), "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
        )


def test_seeded_policy_accepts_hardened_backends_and_rejects_unknown():
    for vfm in ("wan", "cogvideox", "minimax_h3", "aether", "f3r", "lingbot"):
        validate_seeded_cache_policy(
            _args(vfm), "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
        )
    with pytest.raises(ValueError, match="does not support"):
        validate_seeded_cache_policy(
            _args("qwen2_5_vl"), "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
        )


def test_aether_streaming_save_file_writes_noise_metadata():
    source = (ROOT / "features/run_inscene15k.py").read_text()
    start = source.index('elif args.vfm == "aether":\n                        feats = forward_aether(')
    end = source.index('elif args.vfm == "opensora":', start)
    assert "noise_metadata=record_noise_metadata" in source[start:end]


def test_aether_noise_draw_uses_generator_device_helper():
    calls = _calls("features/aether/aether_feature.py", "__call__")
    draw = next(
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "randn_like_on_generator_device"
    )
    assert isinstance(draw.args[1], ast.Name)
    assert draw.args[1].id == "generator"


def test_aether_reconstruct_forces_scheduler_generator():
    source = (ROOT / "features/aether/aether_feature.py").read_text()
    start = source.index("def reconstruct(")
    end = source.index("def __call__(", start)
    body = source[start:end]
    assert 'extra_step_kwargs["generator"] = generator' in body
    assert "randn_like_local" not in body


def test_aether_featurizer_evals_modules_and_does_not_use_inplace_normal():
    source = (ROOT / "features/aether/aether_feature.py").read_text()
    assert "module.eval()" in source
    assert "from features.noise_seed import randn_like_on_generator_device" in source
    assert "randn_like_local" not in source


def test_aether_canonical_generator_is_cpu_local():
    source = (ROOT / "features/run_inscene15k.py").read_text()
    start = source.index("def generator_for_noise_sample")
    end = source.index("# Process scenes", start)
    body = source[start:end]
    assert '{"f3r", "aether"}' in body
    assert 'device="cpu"' in body


def test_lingbot_vae_sample_uses_sample_local_generator():
    calls = _calls("features/lingbot/lingbot_feature.py", "_encode_video_latent")
    sample = next(
        call
        for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "sample"
    )
    keyword = next(item for item in sample.keywords if item.arg == "generator")
    assert isinstance(keyword.value, ast.Name)
    assert keyword.value.id == "noise_generator"


def test_fast3r_random_image_ids_accept_caller_generator():
    source = (ROOT / "features/f3r/f3r/models/fast3r.py").read_text()
    tree = ast.parse(source)
    helpers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_get_random_image_pos", "_get_random_freqs_cis"}
    ]
    assert len(helpers) == 2
    for helper in helpers:
        assert "random_generator" in [arg.arg for arg in helper.args.args]
        assert any(
            isinstance(node, ast.Name) and node.id == "random_generator"
            for node in ast.walk(helper)
        )
