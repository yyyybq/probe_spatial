"""B2 temporal-order validity helpers of analyses/probe_validity (internal analysis code).

Moved out of test_probe_contracts.py on 2026-10-18: analyses/ is not part of the
release, so these tests run only in the internal checkout."""

import csv
import gzip
from pathlib import Path

import torch

from analyses.probe_validity import run_probe_validity as probe_validity
from vidfm3d.models.components.probe_ego_belief_v2 import EgoBeliefProbeV2


def test_b2_order_mapping_changes_history_and_fixes_final_frame():
    shuffled = probe_validity._b2_temporal_mapping(12, seed=42)
    reversed_mapping = probe_validity._b2_temporal_mapping(12, seed=42, reverse=True)
    assert shuffled != list(range(12))
    assert reversed_mapping != list(range(12))
    assert shuffled[-1] == reversed_mapping[-1] == 11
    assert sorted(shuffled) == sorted(reversed_mapping) == list(range(12))


def test_b2_synthetic_order_sanity_detects_content_slot_change():
    receipt = probe_validity._b2_order_synthetic_sanity()
    assert receipt["status"] == "PASS"
    assert receipt["ordered_score"] != receipt["reverse_content_only_score"]
    assert receipt["ordered_score"] == receipt["reverse_frame_tokens_score"]


def test_b2_explicit_identity_frame_ids_match_standard_forward():
    torch.manual_seed(42)
    probe = EgoBeliefProbeV2(
        in_channels=8, hidden_dim=16, num_layers=1, num_heads=4,
        max_seq_len=8, max_h=4, max_w=4, patch_pool=1,
    ).eval()
    feature = torch.randn(4, 2, 2, 8)
    query = torch.randn(8)
    standard_logits, standard_distance = probe_validity._b2_prediction(
        probe, feature, query, torch.device("cpu")
    )
    explicit_logits, explicit_distance = probe_validity._b2_prediction_frame_ids(
        probe, feature, query, [0, 1, 2, 3], torch.device("cpu")
    )
    assert torch.allclose(standard_logits, explicit_logits, atol=1e-6, rtol=1e-6)
    assert abs(standard_distance - explicit_distance) < 1e-6


def test_probe_diagnostics_writer_preserves_symlink_backing(tmp_path, monkeypatch):
    backing_dir = tmp_path / "tmp"
    backing_dir.mkdir()
    backing = backing_dir / "diagnostics.csv.gz"
    public = tmp_path / "diagnostics.csv.gz"
    public.symlink_to(Path("tmp") / backing.name)
    monkeypatch.setattr(probe_validity, "TMP", backing_dir)

    probe_validity._write_gzip_csv_atomic(public, [{"value": "first"}], ["value"])
    probe_validity._write_gzip_csv_atomic(public, [{"value": "second"}], ["value"])

    assert public.is_symlink()
    with gzip.open(public, "rt", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == [{"value": "second"}]
