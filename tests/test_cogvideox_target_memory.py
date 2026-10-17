from __future__ import annotations

from collections import defaultdict

import torch

from features.run_inscene15k import collect_cogvideox_target_rows_cpu


def test_cogvideox_target_rows_are_independent_cpu_tensors() -> None:
    feats = {
        15: torch.arange(2 * 3 * 2 * 2 * 4, dtype=torch.float32).reshape(2, 3, 2, 2, 4),
        20: torch.arange(2 * 3 * 1 * 2 * 4, dtype=torch.float32).reshape(2, 3, 1, 2, 4),
    }
    expected = {
        layer: feat.reshape(-1, *feat.shape[2:])[3].clone()
        for layer, feat in feats.items()
    }
    collected = defaultdict(list)

    collect_cogvideox_target_rows_cpu(feats, collected)

    assert set(collected) == {15, 20}
    for layer, source in feats.items():
        row = collected[layer][0]
        assert row.device.type == "cpu"
        assert row.is_contiguous()
        assert torch.equal(row, expected[layer])
        assert row.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()

