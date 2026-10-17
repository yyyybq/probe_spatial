"""Regression coverage for reproducible and configurable training entrypoints."""

import functools
import io
import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

import vidfm3d.train as train_entrypoint
from vidfm3d.models.probe_ext_module import ProbeExtensionLitModule


@pytest.mark.parametrize("seed", [0, 42, None])
def test_training_seeds_zero_and_only_skips_none(monkeypatch, tmp_path, seed):
    cfg = OmegaConf.create({
        "seed": seed,
        "train": False,
        "test": False,
        "trainer": {"_target_": "Trainer", "devices": 1},
        "data": {"data_module": {"_target_": "DataModule"}},
        "model": {"_target_": "Model"},
        "paths": {"output_dir": str(tmp_path)},
    })
    seed_everything = Mock()
    monkeypatch.setattr(train_entrypoint.L, "seed_everything", seed_everything)
    monkeypatch.setattr(train_entrypoint, "write_run_manifest", Mock())
    monkeypatch.setattr(train_entrypoint, "instantiate_callbacks", lambda _: [])
    monkeypatch.setattr(train_entrypoint, "instantiate_loggers", lambda _: [])
    monkeypatch.setattr(
        train_entrypoint.hydra.utils, "instantiate",
        lambda *args, **kwargs: SimpleNamespace(callback_metrics={}),
    )

    train_entrypoint.train(cfg)

    if seed is None:
        seed_everything.assert_not_called()
    else:
        seed_everything.assert_called_once_with(seed, workers=True)


@pytest.mark.parametrize("load_options", [{}, {"weights_only": True}])
def test_training_import_leaves_torch_load_unmodified(load_options):
    assert torch.load is torch.serialization.load
    payload = io.BytesIO()
    torch.save({"custom": SimpleNamespace(value=1)}, payload)
    payload.seek(0)
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        torch.load(payload, **load_options)


def test_probe_training_without_scheduler_logs_optimizer_lr(monkeypatch):
    module = ProbeExtensionLitModule(
        probe=torch.nn.Linear(2, 1), probe_type="action_dynamics",
        optimizer=functools.partial(torch.optim.SGD, lr=0.025), scheduler=None,
    )
    optimizer = torch.optim.SGD(module.parameters(), lr=0.025)
    module.trainer = SimpleNamespace(optimizers=[optimizer], lr_scheduler_configs=[])
    loss = module.probe.weight.sum()
    monkeypatch.setattr(module, "model_step", lambda *args, **kwargs: (loss, {}))
    logged = {}
    monkeypatch.setattr(module, "log", lambda name, value, **kwargs: logged.update({name: value}))

    assert module.training_step({}, 0) is loss
    assert logged["trainer/lr"] == 0.025


def test_autoresume_without_logger_loads_checkpoint_on_cpu(monkeypatch, tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    checkpoint = checkpoints / "last.ckpt"
    checkpoint.touch()
    cfg = OmegaConf.create({
        "paths": {"output_dir": str(tmp_path)}, "ckpt_path": None,
        "logger": None, "train": True,
    })
    load = Mock(return_value={})
    monkeypatch.setattr(train_entrypoint.torch, "load", load)
    monkeypatch.setattr(train_entrypoint, "extras", lambda _: None)
    monkeypatch.setattr(train_entrypoint, "train", lambda _: ({}, {}))
    monkeypatch.setattr(train_entrypoint, "get_metric_value", lambda **kwargs: None)

    train_entrypoint.main.__wrapped__(cfg)

    assert cfg.ckpt_path == str(checkpoint)
    load.assert_called_once_with(str(checkpoint), map_location="cpu", weights_only=False)


@pytest.mark.parametrize(
    "fit,best_path,expected",
    [(True, "best.ckpt", "best.ckpt"), (True, "", None),
     (False, "best.ckpt", "resume.ckpt")],
)
def test_post_training_test_uses_trained_weights(monkeypatch, tmp_path, fit, best_path, expected):
    cfg = OmegaConf.create({
        "seed": None, "train": fit, "test": True, "ckpt_path": "resume.ckpt",
        "trainer": {"_target_": "Trainer", "devices": 1},
        "data": {"data_module": {"_target_": "DataModule"}},
        "model": {"_target_": "Model"},
        "paths": {"output_dir": str(tmp_path)},
    })
    trainer = SimpleNamespace(
        callback_metrics={"train/loss": 1.0},
        checkpoint_callback=SimpleNamespace(best_model_path=best_path),
        fit=Mock(), test=Mock(),
    )
    monkeypatch.setattr(train_entrypoint, "write_run_manifest", Mock())
    monkeypatch.setattr(train_entrypoint, "instantiate_callbacks", lambda _: [])
    monkeypatch.setattr(train_entrypoint, "instantiate_loggers", lambda _: [])
    monkeypatch.setattr(train_entrypoint.hydra.utils, "instantiate", lambda *args, **kwargs: trainer)

    train_entrypoint.train(cfg)

    assert trainer.test.call_args.kwargs["ckpt_path"] == expected
    assert trainer.test.call_args.kwargs["weights_only"] is False
    if fit:
        assert trainer.fit.call_args.kwargs["weights_only"] is False
