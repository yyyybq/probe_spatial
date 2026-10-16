"""MiniMax-H3 caches must be stored as finite bf16; other backends keep fp16.

The contract (``MINIMAX_H3_FINITE_BF16_V1``) was first applied by the frozen
extractor behind the canonical ``*_MINIMAX_BF16_V2`` namespaces and is merged
into the main tree for the 2026-10 timestep re-extraction. fp16 caches of the
same layers overflowed to inf (audit 2026-10-15/16).
"""

import pytest
import torch

from features.run_inscene15k import (
    MINIMAX_BF16_FEATURE_NAMESPACES,
    MINIMAX_STORAGE_CONTRACT,
    SEEDED_FEATURE_NAMESPACES,
    is_seeded_feature_namespace,
    prepare_feature_for_storage,
    validate_finite_feature,
)


def test_minimax_features_are_stored_as_finite_bf16():
    big = torch.tensor([[1.0e5, -3.7e6, 0.5]], dtype=torch.float32)  # beyond fp16's 65,504
    stored = prepare_feature_for_storage(big, vfm="minimax_h3", context="unit")
    assert stored.dtype == torch.bfloat16
    assert torch.isfinite(stored).all()
    assert stored.is_contiguous()
    # bf16 keeps the magnitude; fp16 would have produced inf here.
    assert torch.allclose(stored.float(), big, rtol=1e-2)
    assert big.half().isinf().any()


def test_other_backends_keep_fp16_storage():
    stored = prepare_feature_for_storage(torch.ones(2, 3), vfm="wan", context="unit")
    assert stored.dtype == torch.float16


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_minimax_feature_is_refused_before_publish(bad):
    feat = torch.ones(4, 4, dtype=torch.float32)
    feat[1, 2] = bad
    with pytest.raises(FloatingPointError, match="refusing non-finite feature"):
        prepare_feature_for_storage(feat, vfm="minimax_h3", context="unit")
    with pytest.raises(FloatingPointError, match="total=1"):
        validate_finite_feature(feat, context="unit")


def test_validate_finite_feature_reports_shape_and_magnitude():
    info = validate_finite_feature(torch.tensor([[2.0, -8.0]]), context="unit")
    assert info == {"dtype": "torch.float32", "shape": [1, 2], "max_abs": 8.0}
    with pytest.raises(TypeError):
        validate_finite_feature(torch.tensor([1, 2]), context="unit")


def test_bf16_seeded_namespaces_are_registered():
    assert MINIMAX_STORAGE_CONTRACT == "MINIMAX_H3_FINITE_BF16_V1"
    for name in (
        "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1",
        "FEAT_TARGET_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1",
    ):
        assert name in SEEDED_FEATURE_NAMESPACES
        assert name in MINIMAX_BF16_FEATURE_NAMESPACES
        assert is_seeded_feature_namespace(name)
    assert "FEAT_STREAMING_CANONICAL_V2_MINIMAX_BF16_V2" in MINIMAX_BF16_FEATURE_NAMESPACES
    assert not is_seeded_feature_namespace("FEAT_STREAMING_CANONICAL_V2_MINIMAX_BF16_V2")
    assert not is_seeded_feature_namespace("FEAT_STREAMING")
