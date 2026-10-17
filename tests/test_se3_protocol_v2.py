from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from analyses.se3_latent_mvp.geometry import se3_exp, se3_log
from analyses.se3_protocol_v2.geometry import bch_second_log, branch_poses, group_commutator, is_commuting, translation_norm
from analyses.se3_protocol_v2.metrics import composite_aware_poincare, grounded_order_metrics, sequential_additive_poincare, tie_aware_order_summary
from analyses.se3_protocol_v2.observed_order import derangement, one_step_metrics, pose_error
from analyses.se3_protocol_v2.cross_model_pilot import manifest_keys, parse_seeds, tuple_key
from analyses.se3_protocol_v2.seese3_replication import temporal_pairs
from analyses.se3_protocol_v2.see_se3_smoke import PoincareAdapter
from analyses.se3_protocol_v2.incremental_v2 import matched_cross_scene_donor, prediction_arrays_independent
from analyses.se3_protocol_v2.paper_v1 import AdditivePredictor


def test_branch_pose_order_matches_frozen_w2c_contract() -> None:
    e_a = se3_exp(np.asarray([0.1, 0.0, 0.0, 0.2, -0.1, 0.3]))
    t1 = se3_exp(np.asarray([0.0, 0.2, 0.0, 0.1, 0.0, 0.0]))
    t2 = se3_exp(np.asarray([0.0, 0.0, -0.3, 0.0, 0.2, 0.0]))
    poses = branch_poses(e_a, t1, t2)
    np.testing.assert_allclose(poses["E_12"], t2 @ t1 @ e_a)
    np.testing.assert_allclose(poses["E_21"], t1 @ t2 @ e_a)
    assert not np.allclose(poses["E_12"], poses["E_21"])


def test_commuting_translation_negative_control() -> None:
    t1 = se3_exp(np.asarray([0.0, 0.0, 0.0, 0.2, -0.1, 0.3]))
    t2 = se3_exp(np.asarray([0.0, 0.0, 0.0, -0.4, 0.5, 0.1]))
    assert is_commuting(t1, t2)
    assert translation_norm(group_commutator(t1, t2)) < 1e-12


def test_rotation_translation_positive_and_bch_sanity() -> None:
    t1 = se3_exp(np.asarray([0.0, 0.0, 0.08, 0.0, 0.0, 0.0]))
    t2 = se3_exp(np.asarray([0.0, 0.0, 0.0, 0.12, 0.0, 0.0]))
    assert not is_commuting(t1, t2)
    exact = se3_log(t2 @ t1)
    first_order = se3_log(t1) + se3_log(t2)
    second_order = bch_second_log(t1, t2)
    assert np.linalg.norm(exact - second_order) < np.linalg.norm(exact - first_order)


def test_assignment_requires_grounded_targets_and_detects_copy_collapse() -> None:
    target_12 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target_21 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    full = grounded_order_metrics(target_12, target_21, target_12, target_21, target_12, target_21, torch.ones(2))
    copy_state = torch.ones_like(target_12)
    copy = grounded_order_metrics(copy_state, copy_state, copy_state, copy_state, target_12, target_21, torch.ones(2))
    assert full["pair_assignment_accuracy"] == 1.0
    assert copy["collapse_rate_ratio_lt_0p10"] == 1.0
    assert copy["recursive_12_endpoint_cosine"] > 0.2


def test_additive_poincare_is_order_blind_but_composite_aware_is_not() -> None:
    g = torch.zeros(1, 3)
    w_pinv = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    xi1 = torch.tensor([[0.0, 0.0, 0.2, 0.0, 0.0, 0.0]])
    xi2 = torch.tensor([[0.0, 0.1, 0.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(sequential_additive_poincare(g, w_pinv, xi1, xi2), sequential_additive_poincare(g, w_pinv, xi2, xi1))
    t1 = se3_exp(xi1.numpy()[0])
    t2 = se3_exp(xi2.numpy()[0])
    first = composite_aware_poincare(g, w_pinv, np.asarray([[t1, t2]]))
    second = composite_aware_poincare(g, w_pinv, np.asarray([[t2, t1]]))
    assert not torch.allclose(first, second)


def test_poincare_adapter_matches_requested_siamese_architecture() -> None:
    adapter = PoincareAdapter(13)
    assert adapter.phi[0].in_features == 13
    assert adapter.phi[0].out_features == 64
    assert adapter.phi[3].in_features == 64
    assert adapter.phi[3].out_features == 20
    assert adapter.readout.in_features == 20
    assert adapter.readout.out_features == 6
    assert adapter.readout.bias is None


def test_observed_one_step_grounding_and_pose_derangement_controls() -> None:
    z_b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    z_c = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    perfect = one_step_metrics(z_b, z_c, z_b, z_c)
    assert perfect["one_step_grounding_cosine"] < 1e-6
    permuted = derangement(11, 7)
    assert not np.any(permuted == np.arange(11))


def test_temporal_split_has_disjoint_raw_frame_support() -> None:
    transform = se3_exp(np.asarray([0.0, 0.0, 0.02, 0.05, 0.0, 0.0]))
    features = torch.zeros(3, 5)
    records = [
        SimpleNamespace(scene_id="scene", raw_frame_indices=(0, 1, 2), features=features, transforms=(transform, transform, transform)),
        SimpleNamespace(scene_id="scene", raw_frame_indices=(70, 71, 72), features=features, transforms=(transform, transform, transform)),
        SimpleNamespace(scene_id="scene", raw_frame_indices=(95, 96, 100), features=features, transforms=(transform, transform, transform)),
    ]
    pairs, audit = temporal_pairs(records, temporal_gap=1)
    assert audit["raw_frame_support_disjoint"]
    assert all(pairs[name] for name in ("train", "val", "test"))


def test_pose_error_is_an_se3_residual_not_only_a_twist_coordinate_difference() -> None:
    target = np.asarray([[0.2, 0.0, 0.0, 0.1, -0.2, 0.3]])
    error = pose_error(target, target, np.ones(6))
    np.testing.assert_allclose(error["rotation_deg"], 0.0, atol=1e-9)
    np.testing.assert_allclose(error["translation_m"], 0.0, atol=1e-9)


def test_cross_model_tuple_keys_freeze_identity_and_do_not_accept_seed_duplicates() -> None:
    record = SimpleNamespace(scene_id="scene", sample_id="tuple", raw_frame_indices=(3, 5, 8))
    assert tuple_key(record) == ("scene", "tuple", (3, 5, 8))
    assert manifest_keys([{"scene_id": "scene", "sample_id": "tuple", "frame_A": "3", "frame_B": "5", "frame_C": "8"}]) == [("scene", "tuple", 3, 5, 8)]
    assert parse_seeds("42,43,44") == (42, 43, 44)
    try:
        parse_seeds("42,42")
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate training seeds must be rejected")


def test_tie_aware_order_summary_treats_copy_as_unbiased_ties() -> None:
    summary = tie_aware_order_summary(
        np.asarray([0.2, 0.3]),
        np.asarray([0.2, 0.3]),
        np.asarray([0.2, 0.3]),
    )
    assert summary["wins"] == 0
    assert summary["losses"] == 0
    assert summary["ties"] == 2
    assert summary["tie_aware_accuracy"] == 0.5
    assert summary["strict_accuracy"] == 0.0
    assert summary["normalized_margin"] == 0.0


def test_tie_aware_normalization_is_ratio_of_aggregates() -> None:
    summary = tie_aware_order_summary(
        np.asarray([0.1, 0.5]),
        np.asarray([0.2, 0.7]),
        np.asarray([0.2, 0.6]),
        epsilon=0.0,
    )
    assert abs(summary["normalized_margin"] - 0.375) < 1e-12


def test_additive_paths_are_computed_independently_and_differ_only_by_roundoff() -> None:
    torch.manual_seed(7)
    model = AdditivePredictor(feature_dim=8, hidden_dim=5).eval()
    z_a, z_b, z_c = (torch.randn(3, 8) for _ in range(3))
    t1 = np.repeat(np.eye(4)[None], 3, axis=0)
    t2 = np.repeat(np.eye(4)[None], 3, axis=0)
    t1[:, 0, 3] = np.asarray([0.1, 0.2, 0.3])
    t2[:, 1, 3] = np.asarray([0.4, 0.5, 0.6])
    arrays, predictions, audit = prediction_arrays_independent(
        "additive", {"additive": model}, z_a, z_b, z_c, t1, t2,
        torch.zeros(6), torch.ones(6),
    )
    assert predictions["recursive"].data_ptr() != predictions["reversed"].data_ptr()
    assert audit["paths_independently_evaluated"]
    assert audit["within_roundoff_bound"]
    assert audit["max_abs_recursive_reverse_delta"] <= audit["roundoff_bound"]
    assert np.max(np.abs(arrays["margin"])) < 1e-12
    assert np.all(arrays["tie_score"] == 0.5)


def test_incremental_matched_donor_is_exact_and_cross_scene_with_duplicate_scenes() -> None:
    records = []
    for index, scene in enumerate(("a", "a", "b", "c", "d", "e")):
        first = se3_exp(np.asarray([0, 0, 0, 0.01 * (index + 1), 0, 0], dtype=float))
        second = se3_exp(np.asarray([0, 0, 0, 0, 0.02 * (index + 1), 0], dtype=float))
        records.append(SimpleNamespace(scene_id=scene, transforms=(first, second, second @ first)))
    donors = matched_cross_scene_donor(records)
    assert sorted(donors.tolist()) == list(range(len(records)))
    assert not np.any(donors == np.arange(len(records)))
    assert all(records[i].scene_id != records[j].scene_id for i, j in enumerate(donors))
