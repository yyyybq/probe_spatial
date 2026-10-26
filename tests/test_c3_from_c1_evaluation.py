"""C3 is an evaluation protocol over the frozen C1 one-step predictor."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vidfm3d.data.components.c1_action_set_dataset import C1ActionSetDataset
from vidfm3d.eval_diag import (
    _bootstrap_c1_action_sensitivity,
    _group_c1_records_for_action_sensitivity,
    _record_c1_action_sensitivity,
    _summarize_c1_action_sensitivity,
)


def _artifact_record() -> dict:
    return {
        "action_set_id": "anchor-1",
        "manifest_sample_id": "manifest-1",
        "scene_path": "/scene/a",
        "split": "val",
        "input_feat": torch.zeros(3, 1, 1, 2),
        "candidate_actions": torch.zeros(2, 9),
        "candidate_target_feat": torch.zeros(2, 1, 1, 2),
        "candidate_ids": torch.tensor([10, 11]),
        "candidate_valid": torch.tensor([True, True]),
        "action_equivalence_group": torch.tensor([0, 1]),
    }


class _OneStepProbe(torch.nn.Module):
    def forward(self, context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # The context contribution is shared inside an action set. Candidate
        # differences can only come from the one action supplied to C1.
        return context.mean(dim=(1, 2, 3)) + action[:, :2]


def test_multi_candidate_eval_calls_c1_once_per_action() -> None:
    module = SimpleNamespace(probe=_OneStepProbe())
    context = torch.zeros(1, 3, 1, 1, 2)
    actions = torch.tensor([[[1.0, 0.0] + [0.0] * 7, [0.0, 1.0] + [0.0] * 7]])
    targets = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]]).reshape(1, 2, 1, 1, 2)
    batch = {
        "input_feat": context,
        "candidate_actions": actions,
        "candidate_target_feat": targets,
        "candidate_valid": torch.ones(1, 2, dtype=torch.bool),
        "action_set_id": ["anchor-1"],
        "manifest_sample_id": ["sample-1"],
        "candidate_ids": torch.tensor([[10, 11]]),
        "action_equivalence_group": torch.tensor([[0, 1]]),
        "scene_path": ["/scene/a"],
    }
    records = _record_c1_action_sensitivity(module, batch)
    torch.testing.assert_close(records[0]["pred"], torch.eye(2))
    summary = _summarize_c1_action_sensitivity(records)
    assert summary["c3_head_source"] == "c1_action_dynamics"
    assert summary["intervention_validity"] == 1.0
    assert summary["intervention_chance"] == 0.5
    assert summary["intervention_validity_chance_adjusted"] == 1.0
    assert summary["n_valid_action_sets"] == 1
    assert summary["n_valid_candidates"] == 2


def test_frozen_action_set_dataset_validates_and_filters_split(tmp_path) -> None:
    artifact = tmp_path / "action_sets.pt"
    torch.save({"records": [_artifact_record()]}, artifact)
    dataset = C1ActionSetDataset(str(artifact), split="val")
    assert len(dataset) == 1
    assert dataset[0]["action_set_id"] == "anchor-1"


def test_frozen_action_set_dataset_rejects_single_candidate(tmp_path) -> None:
    record = _artifact_record()
    record["candidate_actions"] = record["candidate_actions"][:1]
    record["candidate_target_feat"] = record["candidate_target_feat"][:1]
    record["candidate_ids"] = record["candidate_ids"][:1]
    record["candidate_valid"] = record["candidate_valid"][:1]
    record["action_equivalence_group"] = record["action_equivalence_group"][:1]
    artifact = tmp_path / "bad_action_sets.pt"
    torch.save([record], artifact)
    with pytest.raises(ValueError, match="at least two"):
        C1ActionSetDataset(str(artifact))


def test_flat_c1_rows_group_by_anchor_and_report_equivalence() -> None:
    rows = []
    predictions = ([1.0, 0.0], [0.99, 0.01], [0.0, 1.0])
    targets = ([1.0, 0.0], [1.0, 0.0], [0.0, 1.0])
    for candidate, (prediction, target, equivalence) in enumerate(
        zip(predictions, targets, ("forward", "forward", "turn"))
    ):
        rows.append({
            "scene_path": "/scene/a",
            "manifest_sample_id": "anchor-1",
            "action_set_id": "anchor-1",
            "candidate_id": candidate,
            "action_equivalence_group": equivalence,
            "action": torch.tensor([float(candidate)] + [0.0] * 8),
            "pred": torch.tensor(prediction),
            "target": torch.tensor(target),
        })
    grouped = _group_c1_records_for_action_sensitivity(rows)
    assert len(grouped) == 1
    summary = _summarize_c1_action_sensitivity(grouped)
    assert summary["n_equivalent_action_pairs"] == 1
    assert summary["n_distinct_action_pairs"] == 2
    assert summary["intervention_validity"] == 1.0
    assert summary["exact_candidate_validity"] < 1.0
    assert summary["equivalent_action_prediction_distance"] < summary["distinct_action_prediction_distance"]
    assert summary["equivalence_separation_gap"] > 0


def _flat_set(set_id: str, predictions, finite=None) -> list[dict]:
    rows = []
    for candidate, prediction in enumerate(predictions):
        row = {
            "scene_path": f"/scene/{set_id}",
            "manifest_sample_id": set_id,
            "action_set_id": set_id,
            "candidate_id": candidate,
            "action_equivalence_group": None,
            "action": torch.tensor([float(candidate)] + [0.0] * 8),
            "pred": torch.tensor(prediction),
            "target": torch.eye(3)[candidate],
        }
        if finite is not None:
            row["prediction_finite"] = finite[candidate]
        rows.append(row)
    return rows


def test_nonfinite_candidate_excludes_its_action_set_and_is_counted() -> None:
    """Regression (2026-10-19): grouping used to drop prediction_finite.

    argmax over a NaN similarity column picks that column, so a set with a NaN
    candidate was scored as one hit and the non-finite count stayed 0.
    """
    from vidfm3d.eval_diag import _valid_evaluation_record

    good = _flat_set("good", ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]))
    bad = _flat_set(
        "bad", ([1.0, 0.0, 0.0], [float("nan")] * 3, [0.0, 0.0, 1.0]), finite=[True, False, True]
    )
    grouped = _group_c1_records_for_action_sensitivity(good + bad)
    by_id = {record["action_set_id"]: record for record in grouped}
    assert by_id["good"]["prediction_finite"] is True
    assert by_id["bad"]["prediction_finite"] is False
    assert by_id["bad"]["n_nonfinite_candidates"] == 1
    assert not by_id["bad"]["valid"].any()
    assert not _valid_evaluation_record(by_id["bad"], "c3_action_sensitivity")
    assert sum(1 for r in grouped if r.get("prediction_finite", True) is False) == 1

    summary = _summarize_c1_action_sensitivity(grouped)
    assert summary["n_valid_action_sets"] == 1
    assert summary["intervention_validity"] == 1.0
    interval = _bootstrap_c1_action_sensitivity(grouped, reps=20, seed=0)["intervention_validity"]
    assert interval == [1.0, 1.0]


def test_nan_prediction_without_flag_is_still_not_scored() -> None:
    rows = _flat_set("legacy", ([1.0, 0.0, 0.0], [float("nan")] * 3, [0.0, 0.0, 1.0]))
    grouped = _group_c1_records_for_action_sensitivity(rows)
    assert grouped[0]["prediction_finite"] is False
    assert _summarize_c1_action_sensitivity(grouped)["n_valid_action_sets"] == 0


def test_batched_recorder_flags_nonfinite_valid_candidates() -> None:
    class _NanForSecondAction(torch.nn.Module):
        def forward(self, context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            out = action[:, :2].clone()
            out[action[:, 1] > 0] = float("nan")
            return out

    module = SimpleNamespace(probe=_NanForSecondAction())
    actions = torch.tensor([[[1.0, 0.0] + [0.0] * 7, [0.0, 1.0] + [0.0] * 7]]).repeat(2, 1, 1)
    batch = {
        "input_feat": torch.zeros(2, 3, 1, 1, 2),
        "candidate_actions": actions,
        "candidate_target_feat": torch.eye(2).reshape(1, 2, 2).repeat(2, 1, 1),
        # In the second set the NaN candidate is already invalid, so the set stays usable.
        "candidate_valid": torch.tensor([[True, True], [True, False]]),
        "action_set_id": ["nan-valid", "nan-invalid"],
        "scene_path": ["/scene/a", "/scene/b"],
    }
    first, second = _record_c1_action_sensitivity(module, batch)
    assert first["prediction_finite"] is False and not first["valid"].any()
    assert second["prediction_finite"] is True
    assert second["valid"].tolist() == [True, False]


def test_singleton_anchor_is_not_misreported_as_action_sensitivity() -> None:
    rows = [{
        "manifest_sample_id": "only-one",
        "action_set_id": "only-one",
        "candidate_id": 0,
        "action_equivalence_group": None,
        "action": torch.zeros(9),
        "pred": torch.zeros(2),
        "target": torch.zeros(2),
    }]
    with pytest.raises(ValueError, match="at least two"):
        _group_c1_records_for_action_sensitivity(rows)


def test_action_set_bootstrap_is_deterministic_and_set_level() -> None:
    module = SimpleNamespace(probe=_OneStepProbe())
    context = torch.zeros(1, 3, 1, 1, 2)
    batch = {
        "input_feat": context,
        "candidate_actions": torch.tensor([[[1.0, 0.0] + [0.0] * 7, [0.0, 1.0] + [0.0] * 7]]),
        "candidate_target_feat": torch.eye(2).reshape(1, 2, 1, 1, 2),
        "candidate_valid": torch.ones(1, 2, dtype=torch.bool),
        "action_set_id": ["anchor-1"],
        "manifest_sample_id": ["sample-1"],
        "candidate_ids": torch.tensor([[10, 11]]),
        "action_equivalence_group": torch.tensor([[0, 1]]),
        "scene_path": ["/scene/a"],
    }
    records = _record_c1_action_sensitivity(module, batch)
    first = _bootstrap_c1_action_sensitivity(records, reps=20, seed=7)
    second = _bootstrap_c1_action_sensitivity(records, reps=20, seed=7)
    assert first == second
    assert first["intervention_validity"] == [1.0, 1.0]


def test_action_set_dataset_rejects_non_integer_equivalence(tmp_path) -> None:
    record = _artifact_record()
    record["action_equivalence_group"] = torch.tensor([0.0, 1.0])
    artifact = tmp_path / "bad_equivalence.pt"
    torch.save([record], artifact)
    with pytest.raises(ValueError, match="integer class IDs"):
        C1ActionSetDataset(str(artifact))
