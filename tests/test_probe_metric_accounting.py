"""Invalid targets must not improve losses or change distributed metric schemas."""

import functools
import json
import math
from types import SimpleNamespace

import pytest
import torch
from lightning import Trainer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from vidfm3d.eval_diag import (
    _record_abnormal, _record_ego_belief, _valid_evaluation_record,
)
from vidfm3d.models.probe_ext_module import ProbeExtensionLitModule
from vidfm3d.models.video_probe_module import VideoProbeLitModule
from vidfm3d.data.components.inscene15k_dataset import InsScene15KDataset
import vidfm3d.eval_diag as eval_entrypoint


class FixtureProbe(torch.nn.Module):
    n_az_bins = 4
    n_el_bins = 2

    def __init__(self, kind):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.0))
        self.kind = kind

    def forward(self, features, condition=None):
        batch_size = features.shape[0]
        if self.kind == "abnormal":
            return self.value.expand(batch_size)
        if self.kind == "ego_belief":
            return self.value.expand(batch_size, 3)
        if self.kind == "ego_belief_v2":
            return {"logits": self.value.expand(batch_size, 4, 2),
                    "log_dist": self.value.expand(batch_size)}
        if self.kind == "action_dynamics":
            return self.value.expand(batch_size, 8)
        return self.value.expand(batch_size, condition.shape[1], 8)


def _module(kind):
    return ProbeExtensionLitModule(
        probe=FixtureProbe(kind), probe_type=kind,
        optimizer=functools.partial(torch.optim.SGD, lr=0.1), scheduler=None,
    )


def _batch(valid=True):
    return {
        "vfm_feat": torch.ones(2, 3, 2, 2, 8),
        "vfm_feat_shuffled": torch.zeros(2, 3, 2, 2, 8),
        "abnormal_feat_valid": torch.full((2,), valid),
        "hidden_obj_valid": torch.full((2,), valid),
        "hidden_obj_mask": torch.ones(2, 3, 4, 4),
        "hidden_obj_polar": torch.ones(2, 3),
        "belief_query_feat": torch.ones(2, 8),
        "input_feat": torch.ones(2, 3, 2, 2, 8),
        "action": torch.ones(2, 9),
        "target_feat": torch.ones(2, 2, 2, 8),
        "dyn_valid": torch.full((2,), valid),
        "input_feat_seq": torch.ones(2, 3, 2, 2, 8),
        "path_actions": torch.ones(2, 3, 9),
        "counterfactual_actions": torch.ones(2, 3, 9),
        "target_feat_seq": torch.ones(2, 3, 2, 2, 8),
        "path_horizon_valid": torch.full((2, 3), valid),
        "counterfactual_valid": torch.full((2, 3), valid),
        "scene_path": ["scene-a", "scene-b"],
        "manifest_sample_id": ["sample-a", "sample-b"],
    }


@pytest.mark.parametrize("kind", [
    "abnormal", "ego_belief", "ego_belief_v2", "action_dynamics",
    "path_integration", "counterfactual",
])
def test_ranks_with_valid_and_invalid_targets_have_identical_metric_schema(kind):
    module = _module(kind)
    _, valid_metrics = module.model_step(_batch(True), train=False)
    empty_loss, empty_metrics = module.model_step(_batch(False), train=False)
    # Reduction order matters as well as metric names when ranks synchronize.
    assert list(empty_metrics) == list(valid_metrics)
    assert empty_loss.item() == 0
    assert empty_loss.requires_grad


class FixedLossModule(ProbeExtensionLitModule):
    def __init__(self):
        super().__init__(
            probe=torch.nn.Linear(1, 1), probe_type="action_dynamics",
            optimizer=functools.partial(torch.optim.SGD, lr=0.1), scheduler=None,
        )

    def model_step(self, batch, train):
        loss = batch["loss"].float().mean()
        return loss, {"loss_dyn": loss, "dyn_n_valid": batch["n_valid"].float().sum()}


@pytest.mark.parametrize("all_invalid", [False, True])
def test_lightning_validation_does_not_count_empty_batches(tmp_path, all_invalid):
    rows = [{"loss": 0.0, "n_valid": 0}]
    if not all_invalid:
        rows += [{"loss": 4.0, "n_valid": 1}, {"loss": 2.0, "n_valid": 3}]
    trainer = Trainer(
        accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, default_root_dir=tmp_path,
    )
    result = trainer.validate(FixedLossModule(), dataloaders=[DataLoader(rows)], verbose=False)[0]
    if all_invalid:
        assert math.isnan(result["val/loss"])
        assert result["val/dyn_n_valid"] == 0
    else:
        assert result["val/loss"] == pytest.approx(2.5)
        assert result["val/loss_dyn"] == pytest.approx(2.5)
        assert result["val/dyn_n_valid"] == 4


def test_abnormal_records_preserve_manifest_lineage_and_invalidity():
    batch = _batch()
    batch["abnormal_feat_valid"] = torch.tensor([True, False])
    records = _record_abnormal(_module("abnormal"), batch)
    assert [r["manifest_sample_id"] for r in records] == ["sample-a", "sample-b"]
    assert [_valid_evaluation_record(r, "abnormal") for r in records] == [True, False]


def test_ego_records_keep_correct_lineage_after_invalid_samples_are_filtered():
    batch = _batch()
    batch["hidden_obj_valid"] = torch.tensor([False, True])
    records = _record_ego_belief(_module("ego_belief"), batch)
    assert len(records) == 1
    assert records[0]["manifest_sample_id"] == "sample-b"


def test_record_validity_respects_horizon_and_action_set_cardinality():
    record = {"valid": torch.tensor([False, True, False])}
    assert _valid_evaluation_record(record, "path_integration")
    assert not _valid_evaluation_record(record, "c3_action_sensitivity")
    assert not _valid_evaluation_record({"valid": torch.zeros(3, dtype=torch.bool)}, "path_integration")
    assert _valid_evaluation_record({"valid": torch.tensor([True, True])}, "c3_action_sensitivity")


def test_abnormal_eval_summary_counts_only_valid_records(monkeypatch, tmp_path):
    module = _module("abnormal")
    checkpoint = tmp_path / "probe.ckpt"
    torch.save({"state_dict": module.state_dict()}, checkpoint)
    batch = _batch()
    batch["abnormal_feat_valid"] = torch.tensor([True, False])
    rows = [{key: value[i] for key, value in batch.items()} for i in range(2)]
    datamodule = SimpleNamespace(
        setup=lambda stage: None, val_dataloader=lambda: DataLoader(rows, batch_size=2),
    )
    cfg = OmegaConf.create({
        "ckpt_path": str(checkpoint), "seed": 0, "bootstrap_reps": 0,
        "paths": {"output_dir": str(tmp_path)},
        "data": {"data_module": {"_target_": "DataModule", "validation_datasets": ["fixture"]}},
        "model": {"_target_": "Model", "probe_type": "abnormal"},
    })
    monkeypatch.setattr(eval_entrypoint, "_execution_device", lambda: (torch.device("cpu"), {}))
    monkeypatch.setattr(
        eval_entrypoint.hydra.utils, "instantiate",
        lambda config: datamodule if config._target_ == "DataModule" else module,
    )

    eval_entrypoint.main.__wrapped__(cfg)

    summary = json.loads((tmp_path / "eval" / "val_summary.json").read_text())
    assert summary["loader_n_samples"] == 2
    assert summary["valid_record_n"] == summary["invalid_record_n"] == 1
    assert summary["valid_record_fraction"] == 0.5
    assert summary["valid_manifest_ids"] == ["sample-a"]
    assert summary["invalid_manifest_ids"] == ["sample-b"]


class FixtureSAE(FixtureProbe):
    use_overlap_readout = False

    def forward(self, features):
        return {"sampled_tokens": features, "recon": features,
                "sparse": features * 0, "frame_sparse": features.mean((2, 3))}

    def predict_ego(self, features, query):
        return features.new_tensor([math.pi - 0.01, 0, 0]) + self.value.expand(features.shape[0], 3)


def test_sae_ego_azimuth_loss_and_metric_wrap_at_pi():
    module = ProbeExtensionLitModule(
        probe=FixtureSAE("sae_spatial"), probe_type="sae_spatial",
        optimizer=functools.partial(torch.optim.SGD, lr=0.1), scheduler=None,
    )
    batch = _batch()
    batch["hidden_obj_polar"] = torch.tensor([[-math.pi + 0.01, 0, 0]]).repeat(2, 1)
    _, metrics = module.model_step(batch, train=False)
    assert metrics["sae_ego_az_err_deg"] < 2
    assert metrics["sae_ego_loss"] < 0.001
    _, empty_metrics = module.model_step(_batch(False), train=False)
    assert list(empty_metrics) == list(metrics)


def test_unknown_dataset_split_cannot_silently_become_validation(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset split"):
        InsScene15KDataset(root=str(tmp_path), split="trian")


def test_unknown_eval_split_cannot_silently_use_test_data():
    cfg = OmegaConf.create({"ckpt_path": "unused.ckpt", "eval_split": "validation"})
    with pytest.raises(ValueError, match="Unknown eval_split"):
        eval_entrypoint.main.__wrapped__(cfg)


class FixtureOverlapSAE(FixtureSAE):
    use_overlap_readout = True

    def forward(self, features):
        output = super().forward(features)
        output["recon"] = features + self.value + 1
        return output

    def predict_overlap(self, features):
        return self.value.expand(features.shape[0], features.shape[1], features.shape[1])


def _overlap_sae_module():
    return ProbeExtensionLitModule(
        probe=FixtureOverlapSAE("sae_spatial"), probe_type="sae_spatial",
        optimizer=functools.partial(torch.optim.SGD, lr=0.1), scheduler=None,
    )


def _overlap_sae_batch(frames):
    return {"vfm_feat": torch.ones(2, frames, 2, 2, 8),
            "overlap_gt": torch.zeros(2, frames, frames)}


@pytest.mark.parametrize("frames", [1, 2])
def test_sae_overlap_empty_pairs_preserve_reconstruction_loss(frames):
    module = _overlap_sae_module()
    loss, metrics = module.model_step(_overlap_sae_batch(frames), train=True)
    assert torch.isfinite(loss)
    assert metrics["sae_overlap_n_pairs"] == 2 * frames * (frames - 1)
    if frames == 1:
        assert loss.item() == pytest.approx(1)
        assert math.isnan(metrics["sae_overlap_bce"])
        assert math.isnan(metrics["sae_overlap_mae"])
    else:
        assert loss.item() == pytest.approx(1 + 0.2 * math.log(2))
        assert metrics["sae_overlap_bce"].item() == pytest.approx(math.log(2))
        assert metrics["sae_overlap_mae"].item() == pytest.approx(0.5)
    loss.backward()
    assert torch.isfinite(module.probe.value.grad)


def test_sae_overlap_empty_batch_does_not_pollute_valid_pairs(tmp_path):
    trainer = Trainer(
        accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, default_root_dir=tmp_path,
    )
    rows = [_overlap_sae_batch(1), _overlap_sae_batch(2)]
    result = trainer.validate(
        _overlap_sae_module(), dataloaders=[DataLoader(rows, batch_size=None)], verbose=False,
    )[0]
    assert result["val/loss"] == pytest.approx(1 + 0.1 * math.log(2))
    assert result["val/sae_overlap_bce"] == pytest.approx(math.log(2))
    assert result["val/sae_overlap_mae"] == pytest.approx(0.5)
    assert result["val/sae_overlap_n_pairs"] == 4


def test_compile_is_rejected_instead_of_silently_ignored():
    with pytest.raises(ValueError, match="does not support compile=True"):
        ProbeExtensionLitModule(
            probe=FixtureProbe("abnormal"), probe_type="abnormal",
            optimizer=functools.partial(torch.optim.SGD, lr=0.1), scheduler=None, compile=True,
        )


class FixedA1LossModule(VideoProbeLitModule):
    def __init__(self, output_path):
        super().__init__(
            probe=torch.nn.Linear(1, 1), loss_weights={},
            optimizer=functools.partial(torch.optim.SGD, lr=0.1), scheduler=None,
            compile=False, output_path=str(output_path),
        )
        self.saved_visualizations = []

    def model_step(self, batch, return_metrics=False):
        loss = batch["sample_loss"].float().mean()
        return batch, {}, loss, {"fixture_loss": loss}

    def _make_pca_tensor(self, *args, **kwargs):
        return None

    def visualize_image_grid(self, **kwargs):
        self.saved_visualizations.append((kwargs["save_path"], kwargs["name"]))


def test_a1_loss_weights_partial_batches_and_resets_between_eval_stages(tmp_path):
    def row(loss):
        return {"sample_loss": loss, "image": torch.zeros(1, 3, 2, 2),
                "cmaps": torch.ones(1, 1, 2, 2), "vfm_feat": torch.zeros(1, 1, 1, 1),
                "vfm_idx": torch.tensor([0]), "vfm_name": "dino"}

    trainer = Trainer(
        accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, default_root_dir=tmp_path,
    )
    module = FixedA1LossModule(tmp_path)
    # A full batch has loss 4 and the last singleton has loss 0: sample mean 3.
    validation = DataLoader([row(4), row(4), row(4), row(0)], batch_size=3)
    result = trainer.validate(module, dataloaders=[validation], verbose=False)[0]
    assert result["val/loss"] == pytest.approx(3)
    assert result["val/fixture_loss"] == pytest.approx(3)
    assert module.saved_visualizations
    assert all("/viz/val/" in path and name.startswith("val-viz/")
               for path, name in module.saved_visualizations)
    module.saved_visualizations.clear()
    # Different test losses also prove validation state does not leak into test.
    testing = DataLoader([row(8), row(8), row(8), row(0)], batch_size=3)
    result = trainer.test(module, dataloaders=[testing], verbose=False)[0]
    assert result["test/loss"] == pytest.approx(6)
    assert result["test/fixture_loss"] == pytest.approx(6)
    assert not any(name.startswith("val/") for name in result)
    assert module.saved_visualizations
    assert all("/viz/test/" in path and name.startswith("test-viz/")
               for path, name in module.saved_visualizations)
