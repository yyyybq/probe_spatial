"""Offline extraction must use the caller's cache on a clean machine."""

from pathlib import Path

import pytest

from features.offline_hf import resolve_offline_snapshot


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def make_snapshot(cache: Path) -> Path:
    root = cache / "models--org--model"
    (root / "refs").mkdir(parents=True)
    (root / "refs" / "main").write_text("a" * 40)
    snapshot = root / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    return snapshot


@pytest.mark.parametrize("variable,suffix", [
    ("HF_HUB_CACHE", ""), ("HUGGINGFACE_HUB_CACHE", ""),
    ("HF_HOME", "hub"), ("XDG_CACHE_HOME", "huggingface/hub"),
    (None, ".cache/huggingface/hub"),
])
def test_cache_environment_and_default_home(monkeypatch, tmp_path, variable, suffix):
    base = tmp_path / "configured" if variable else tmp_path
    if variable:
        monkeypatch.setenv(variable, str(base))
    snapshot = make_snapshot(base / suffix)
    load_path, provenance = resolve_offline_snapshot(
        "org/model", required_paths=("config.json",),
    )
    assert load_path == str(snapshot)
    assert provenance["resolved_revision"] == "a" * 40


def test_explicit_hub_cache_does_not_fall_back(monkeypatch, tmp_path):
    make_snapshot(tmp_path / "home" / "hub")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="no complete revision-pinned"):
        resolve_offline_snapshot("org/model", required_paths=("config.json",))


def test_generator_requirements_are_preserved_in_local_provenance(tmp_path):
    model = tmp_path / "local-model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    _, provenance = resolve_offline_snapshot(
        str(model), required_paths=(name for name in ["config.json"]),
    )
    assert provenance["required_paths"] == ["config.json"]
