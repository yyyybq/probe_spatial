"""Non-finite features/predictions must be rejected or counted, never scored as zero.

Background (2026-10-15 audit): timestep MiniMax-H3 layer-20/25 probes trained
on a corrupt cache to 100% NaN weights while the loss helpers reported zero
loss, and diverged CogVideoX/MiniMax B2 heads produced ``pred_bin == 0`` for
every sample because ``argmax`` over NaN logits returns 0. These tests pin the
training-time rejection, the evaluator flags, and the cache-name registry for
the t=250 MiniMax-H3/LingBot backends.
"""

import functools
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import vidfm3d.eval_a1_streaming as a1_eval
import vidfm3d.eval_diag as diag_eval
from vidfm3d.models.probe_ext_module import ProbeExtensionLitModule
from vidfm3d.utils import feature_layers
from vidfm3d.utils.loss import NonFiniteInputError, require_finite_input

REPO = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------- registry
@pytest.mark.parametrize(
    "vfm,layer,timestep,expected",
    [
        ("minimax_h3", 10, 250, "feature_t250_layer10.sft"),
        ("minimax_h3", 49, None, "feature_t250_layer49.sft"),
        ("lingbot", 47, None, "feature_t250_layer47.sft"),
        ("lingbot", 30, 250, "feature_t250_layer30.sft"),
        ("wan", 20, None, "feature_t749_layer20.sft"),
        ("cogvideox", 5, 499, "feature_t499_layer5.sft"),
        ("vjepa2", 23, None, "feature_layer23.sft"),
        ("f3r", 24, None, "feature_l24.sft"),
    ],
)
def test_cache_filename_matches_extractor_convention(vfm, layer, timestep, expected):
    assert feature_layers.feature_filename(vfm, feature_layer=layer, feature_timestep=timestep) == expected


def test_t250_backends_are_registered_with_disk_layer_counts():
    minimax = feature_layers.get_feature_layer_spec("minimax_h3")
    lingbot = feature_layers.get_feature_layer_spec("lingbot")
    assert (minimax.in_channels, minimax.last_layer, minimax.default_timestep) == (5376, 49, 250)
    assert (lingbot.in_channels, lingbot.last_layer, lingbot.default_timestep) == (2048, 47, 250)
    assert feature_layers.default_feature_timestep("wan") == 749
    assert feature_layers.default_feature_timestep("vjepa2") == 749
    # Grids read from cached files on disk: MiniMax (T,15,26,5376), LingBot (T,30,52,2048).
    assert feature_layers.default_feature_hw("minimax_h3") == (15, 26)
    assert feature_layers.default_feature_hw("lingbot") == (30, 52)
    assert feature_layers.default_feature_channels("lingbot") == 2048


def test_layer_resolver_exposes_default_timestep():
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts/resolve_feature_layers.py"), "--vfm", "lingbot",
         "--field", "default_timestep"],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "250"


# ---------------------------------------------------------------- training
def test_require_finite_input_names_key_and_scene():
    batch = {
        "vfm_feat": torch.ones(2, 3, 2, 2, 4),
        "scene_path": ["scene-a", "scene-b"],
    }
    require_finite_input(batch, "vfm_feat", "absent_key")
    batch["vfm_feat"][1, 0, 0, 0, 0] = float("nan")
    with pytest.raises(NonFiniteInputError, match=r"vfm_feat.*1 of 2 samples.*scene-b"):
        require_finite_input(batch, "vfm_feat")


class _ConstantProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, features, condition=None):
        return self.value.expand(features.shape[0], features.shape[1], features.shape[1])


def test_probe_module_rejects_nonfinite_features_before_forward():
    module = ProbeExtensionLitModule(
        probe=_ConstantProbe(), probe_type="view_consistency",
        optimizer=functools.partial(torch.optim.SGD, lr=0.1), scheduler=None,
    )
    feat = torch.ones(2, 3, 2, 2, 4)
    batch = {"vfm_feat": feat, "overlap_gt": torch.eye(3).expand(2, 3, 3).clone()}
    loss, _ = module.model_step(batch, train=True)
    assert torch.isfinite(loss)
    feat[0, 1] = float("inf")
    with pytest.raises(NonFiniteInputError):
        module.model_step(batch, train=True)


# ------------------------------------------------------------ A1 evaluator
class _A1Stub(torch.nn.Module):
    """Depth head whose second sample diverged to NaN."""

    def __init__(self):
        super().__init__()
        self.probe = self

    def forward(self, tokens, image_shape, fg_mask=None):
        B, S = tokens.shape[:2]
        H, W = image_shape[-2:]
        depth = torch.ones(B, S, H, W, 1)
        depth[1] = float("nan")
        return {"depth": depth}


def test_a1_evaluate_flags_nonfinite_prediction_instead_of_zero_loss():
    B, S, H, W = 2, 2, 4, 4
    batch = {
        "vfm_feat": torch.ones(B, S, 2, 2, 3),
        "image": torch.zeros(B, S, 3, H, W),
        "cmaps": torch.ones(B, S, 1, H, W),
        "dmaps": torch.full((B, S, 1, H, W), 2.0),
        "scene_path": ["scene-a", "scene-b"],
        "manifest_sample_id": ["a", "b"],
    }
    records = a1_eval.evaluate(_A1Stub(), [batch], ["depth"], torch.device("cpu"))
    assert [r["prediction_finite"] for r in records] == [True, False]
    assert math.isfinite(records[0]["loss_depth"])
    assert math.isnan(records[1]["loss_depth"])
    summary = a1_eval._summarize(records)
    assert summary["n_loss_depth"] == 1
    assert summary["n_nonfinite_prediction"] == 1
    assert summary["n_valid_prediction"] == 1
    assert "mean_prediction_finite" not in summary


def test_a1_summary_treats_v1_records_as_finite():
    summary = a1_eval._summarize([{"loss_depth": 0.5}, {"loss_depth": 0.25}])
    assert summary["mean_loss_depth"] == pytest.approx(0.375)
    assert summary["n_nonfinite_prediction"] == 0
    assert summary["n_valid_prediction"] == 2


# ----------------------------------------------------------- diag evaluator
class _B2Stub(torch.nn.Module):
    n_az_bins = 4
    n_el_bins = 2

    def __init__(self):
        super().__init__()
        self.probe = self

    def forward(self, feat, query):
        B = feat.shape[0]
        logits = torch.zeros(B, 4, 2)
        logits[:, 3, 1] = 1.0            # a deliberate, finite argmax (bin 7)
        logits[1] = float("nan")         # diverged sample -> argmax would be 0
        return {"logits": logits, "log_dist": torch.zeros(B)}


def test_b2_records_flag_nan_logits_and_summary_excludes_them():
    batch = {
        "vfm_feat": torch.ones(2, 3, 2, 2, 4),
        "belief_query_feat": torch.ones(2, 4),
        "hidden_obj_valid": torch.ones(2, dtype=torch.bool),
        "hidden_obj_polar": torch.zeros(2, 3),
        "scene_path": ["scene-a", "scene-b"],
        "manifest_sample_id": ["a", "b"],
    }
    records = diag_eval._record_ego_belief_v2(_B2Stub(), batch)
    assert [r["prediction_finite"] for r in records] == [True, False]
    assert records[1]["pred_bin"] == 0  # argmax over NaN: the value that was previously trusted
    summary = diag_eval._summarize(records, "ego_belief_v2")
    assert summary["n"] == 2 and summary["n_scored"] == 1 and summary["n_nonfinite_pred"] == 1
    assert summary["mean_ang_err_deg"] == pytest.approx(records[0]["ang_err_deg"])
    assert diag_eval._valid_evaluation_record(records[0], "ego_belief_v2")
    assert not diag_eval._valid_evaluation_record(records[1], "ego_belief_v2")
    only_nan = diag_eval._summarize([records[1]], "ego_belief_v2")
    assert only_nan == {"n": 1, "n_scored": 0, "n_nonfinite_pred": 1}


def test_a2_summary_uses_module_thresholds_and_skips_nan_predictions():
    gt = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    finite = {"overlap_gt": gt, "overlap_pred": torch.tensor([[1.0, 0.9], [0.9, 1.0]]), "prediction_finite": True}
    nan = {"overlap_gt": gt, "overlap_pred": torch.full((2, 2), float("nan")), "prediction_finite": False}
    loose = diag_eval._summarize([finite, nan], "view_consistency", overlap_pos_threshold=0.4)
    strict = diag_eval._summarize([finite, nan], "view_consistency", overlap_pos_threshold=0.6)
    assert loose["n_pos"] == 2 and loose["pos_acc"] == 1.0
    assert strict["n_pos"] == 0 and math.isnan(strict["pos_acc"])
    assert loose["n_nonfinite_pred"] == 1 and loose["n_pairs"] == 2
    assert loose["overlap_pos_threshold"] == 0.4


def test_a2_records_flag_nonfinite_logits():
    class _A2Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.probe = self

        def forward(self, feat):
            logits = torch.zeros(feat.shape[0], feat.shape[1], feat.shape[1])
            logits[0] = float("nan")
            return logits

    batch = {"vfm_feat": torch.ones(2, 3, 2, 2, 4), "overlap_gt": torch.ones(2, 3, 3),
             "scene_path": ["scene-a", "scene-b"], "manifest_sample_id": ["a", "b"]}
    records = diag_eval._record_view_consistency(_A2Stub(), batch)
    assert [r["prediction_finite"] for r in records] == [False, True]
    assert not diag_eval._valid_evaluation_record(records[0], "view_consistency")


def _c2_batch(pred_nan_sample: int | None):
    """Two C2 samples with three horizons; optionally make one sample's predictions NaN."""

    class _PathProbe(torch.nn.Module):
        def forward(self, input_feat_seq, path_actions):
            out = torch.eye(3).expand(path_actions.shape[0], 3, 3).clone()
            if pred_nan_sample is not None:
                out[pred_nan_sample] = float("nan")
            return out

    eye = torch.eye(4)
    target_extrinsics = eye.repeat(2, 3, 1, 1)
    target_extrinsics[:, :, 0, 3] = torch.tensor([1.0, 2.0, 3.0])
    batch = {
        "input_feat_seq": torch.zeros(2, 4, 1, 1, 3),
        "path_actions": torch.zeros(2, 3, 9),
        "target_feat_seq": torch.eye(3).reshape(1, 3, 1, 1, 3).repeat(2, 1, 1, 1, 1),
        "path_horizon_valid": torch.ones(2, 3, dtype=torch.bool),
        "action_horizons": torch.tensor([[1, 2, 4], [1, 2, 4]]),
        "start_extrinsic": eye.repeat(2, 1, 1),
        "target_extrinsics": target_extrinsics,
        "scene_path": ["/scene/a", "/scene/b"],
    }
    return SimpleNamespace(probe=_PathProbe()), batch


def test_c2_records_flag_nan_predictions_and_summary_excludes_them():
    """Regression (2026-10-19): C2 records had no prediction_finite flag.

    A NaN prediction still produced a finite global_R@1, and the export gate on
    non-finite predictions could not fire for path_integration.
    """
    module, batch = _c2_batch(pred_nan_sample=1)
    records = diag_eval._record_path_integration(module, batch)
    assert [r["prediction_finite"] for r in records] == [True, False]
    assert not diag_eval._valid_evaluation_record(records[1], "path_integration")

    summary = diag_eval._summarize(records, "path_integration")
    assert summary["n_nonfinite_pred"] == 1
    assert summary["n_valid_records"] == 1
    assert summary["global_R@1"] == 1.0

    clean = diag_eval._summarize(diag_eval._record_path_integration(*_c2_batch(None)), "path_integration")
    assert clean["n_nonfinite_pred"] == 0 and clean["n_valid_records"] == 2


def test_c2_invalid_horizons_do_not_decide_finiteness():
    module, batch = _c2_batch(pred_nan_sample=None)
    batch["path_horizon_valid"] = torch.tensor([[True, True, True], [True, True, False]])

    class _NanOnInvalidHorizon(torch.nn.Module):
        def forward(self, input_feat_seq, path_actions):
            out = torch.eye(3).expand(2, 3, 3).clone()
            out[1, 2] = float("nan")
            return out

    records = diag_eval._record_path_integration(SimpleNamespace(probe=_NanOnInvalidHorizon()), batch)
    assert [r["prediction_finite"] for r in records] == [True, True]


@pytest.mark.parametrize("split, ok", [(None, True), ("val", True), ("test", False), ("train", False)])
def test_a1_evaluator_refuses_splits_it_cannot_evaluate(split, ok):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({} if split is None else {"eval_split": split})
    if ok:
        assert a1_eval._require_val_split(cfg) == "val"
    else:
        with pytest.raises(ValueError, match="validation split only"):
            a1_eval._require_val_split(cfg)
