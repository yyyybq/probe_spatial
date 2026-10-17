import ast
import hashlib
from pathlib import Path

import torch

from features.noise_seed import (
    DETERMINISTIC_NOISE_CONTRACT,
    NOISE_SEED_SCHEME,
    canonical_noise_identity,
    canonical_noise_metadata,
    canonical_noise_sample,
    make_noise_generator,
    noise_metadata_matches,
    randn_like_local,
    randn_like_on_generator_device,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {
    "protocol": "inscene_streaming_canonical_v2_obs28_h124",
    "protocol_sha256": "a" * 64,
}


def _stream_sample(**overrides):
    kwargs = {
        "protocol": PROTOCOL,
        "vfm": "wan",
        "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "scene": "01ce24e652",
        "sample_kind": "streaming_prefix",
        "window_id": 1,
        "prefix_len": 12,
    }
    kwargs.update(overrides)
    return canonical_noise_sample(**kwargs)


def _tensor_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def test_stream_identity_exact_fields_exclude_timestep_and_layer():
    identity = canonical_noise_identity(
        protocol=PROTOCOL,
        vfm="cogvideox",
        model_id="THUDM/CogVideoX-5b-I2V",
        scene="scene-a",
        sample_kind="streaming_prefix",
        window_id=3,
        prefix_len=24,
    )
    assert identity == {
        "canonical_protocol": PROTOCOL["protocol"],
        "canonical_protocol_sha256": PROTOCOL["protocol_sha256"],
        "vfm": "cogvideox",
        "model_id": "THUDM/CogVideoX-5b-I2V",
        "scene_id": "scene-a",
        "sample_kind": "streaming_prefix",
        "window_id": 3,
        "prefix_len": 24,
    }
    assert "timestep" not in identity
    assert "layer" not in identity


def test_same_sample_rerun_has_identical_tensor_and_sha():
    sample_a = _stream_sample()
    sample_b = _stream_sample()
    assert sample_a == sample_b
    # Published test vector freezes JSON canonicalization, hash byte order, and
    # the signed-safe u63 mask for inscene_noise_seed_v1.
    assert sample_a["noise_seed"] == 3263381780902667276

    def fake_feature(sample):
        generator = make_noise_generator(sample["noise_seed"], "cpu")
        base_noise = randn_like_local(torch.empty(3, 4), generator)
        return (base_noise * 1.75 + 0.25).half()

    feature_a = fake_feature(sample_a)
    feature_b = fake_feature(sample_b)
    assert torch.equal(feature_a, feature_b)
    assert _tensor_sha(feature_a) == _tensor_sha(feature_b)


def test_different_timestep_uses_identical_base_noise_mock():
    sample = _stream_sample()

    def mocked_forward(timestep):
        generator = make_noise_generator(sample["noise_seed"], "cpu")
        base_noise = randn_like_local(torch.empty(2, 5), generator)
        # The scheduler result may depend on timestep; expose noise separately.
        return base_noise, base_noise * float(timestep)

    noise_499, noisy_499 = mocked_forward(499)
    noise_999, noisy_999 = mocked_forward(999)
    assert torch.equal(noise_499, noise_999)
    assert not torch.equal(noisy_499, noisy_999)


def test_different_logical_samples_have_different_seeds():
    samples = [
        _stream_sample(),
        _stream_sample(prefix_len=16),
        _stream_sample(window_id=2),
        _stream_sample(scene="another-scene"),
        canonical_noise_sample(
            protocol=PROTOCOL,
            vfm="wan",
            model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            scene="01ce24e652",
            sample_kind="target_isolated",
            frame_index=100,
        ),
        canonical_noise_sample(
            protocol=PROTOCOL,
            vfm="wan",
            model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            scene="01ce24e652",
            sample_kind="target_isolated",
            frame_index=101,
        ),
    ]
    seeds = [sample["noise_seed"] for sample in samples]
    assert len(set(seeds)) == len(seeds)


def test_local_generator_does_not_advance_global_rng():
    torch.manual_seed(12345)
    expected_global = torch.randn(8)
    torch.manual_seed(12345)
    local = make_noise_generator(_stream_sample()["noise_seed"], "cpu")
    randn_like_local(torch.empty(32), local)
    actual_global = torch.randn(8)
    assert torch.equal(expected_global, actual_global)


def test_generator_device_draw_is_stable_across_global_rng_and_other_samples():
    tensor = torch.empty(4, 8, dtype=torch.bfloat16)
    seed = 424242
    torch.manual_seed(1)
    first = randn_like_on_generator_device(tensor, make_noise_generator(seed, "cpu"))
    torch.manual_seed(999)
    torch.randn(10000)
    second = randn_like_on_generator_device(tensor, make_noise_generator(seed, "cpu"))
    other = make_noise_generator(7, "cpu")
    randn_like_on_generator_device(torch.empty(32, 32, dtype=torch.bfloat16), other)
    third = randn_like_on_generator_device(tensor, make_noise_generator(seed, "cpu"))
    assert torch.equal(first.float(), second.float())
    assert torch.equal(first.float(), third.float())


def test_sidecar_metadata_is_strict_and_auditable():
    metadata = canonical_noise_metadata([_stream_sample()])
    assert metadata["feature_generation_contract"] == DETERMINISTIC_NOISE_CONTRACT
    assert metadata["noise_seed_scheme"] == NOISE_SEED_SCHEME
    assert metadata["noise_seed"] == metadata["noise_seed_samples"][0]["noise_seed"]
    assert noise_metadata_matches(metadata, dict(metadata))

    corrupted = dict(metadata)
    corrupted["noise_seed"] += 1
    assert not noise_metadata_matches(corrupted, metadata)


def _pipeline_call_nodes(path: Path, class_name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == "__call__":
                    return [nested for nested in ast.walk(child) if isinstance(nested, ast.Call)]
    raise AssertionError(f"{class_name}.__call__ not found in {path}")


def test_i2v_prepare_latents_and_explicit_noise_share_local_generator():
    for relative, class_name in [
        ("features/wan/wan_feature_i2v.py", "OneStepWanPipeline"),
        ("features/cogvideox/cogvideox_feature_i2v.py", "OneStepCogVideoXPipeline"),
    ]:
        calls = _pipeline_call_nodes(ROOT / relative, class_name)
        prepare_calls = [
            call
            for call in calls
            if isinstance(call.func, ast.Attribute)
            and call.func.attr == "prepare_latents"
        ]
        noise_calls = [
            call
            for call in calls
            if isinstance(call.func, ast.Name)
            and call.func.id == "randn_like_local"
        ]
        assert len(prepare_calls) == 1
        assert len(noise_calls) == 1
        assert any(
            isinstance(arg, ast.Name) and arg.id == "sample_generator"
            for arg in prepare_calls[0].args
        )
        generator_kw = next(
            keyword for keyword in noise_calls[0].keywords if keyword.arg == "generator"
        )
        assert isinstance(generator_kw.value, ast.Name)
        assert generator_kw.value.id == "sample_generator"
