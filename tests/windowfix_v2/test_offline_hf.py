from pathlib import Path

import pytest

from features.offline_hf import resolve_offline_snapshot


@pytest.fixture(autouse=True)
def isolate_hub_cache_environment(monkeypatch):
    # These tests intentionally select a cache via HF_HOME.
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)


def _snapshot(home: Path, model_id: str, revision: str) -> Path:
    root = home / "hub" / ("models--" + model_id.replace("/", "--"))
    (root / "refs").mkdir(parents=True)
    (root / "refs" / "main").write_text(revision + "\n", encoding="utf-8")
    snapshot = root / "snapshots" / revision
    (snapshot / "transformer").mkdir(parents=True)
    (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
    (snapshot / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    return snapshot


def test_offline_hub_id_resolves_revision_pinned_snapshot(monkeypatch, tmp_path):
    model_id = "org/model"
    revision = "a" * 40
    snapshot = _snapshot(tmp_path, model_id, revision)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    load_id, provenance = resolve_offline_snapshot(
        model_id,
        required_paths=("model_index.json", "transformer/config.json"),
    )

    assert load_id == str(snapshot)
    assert provenance["logical_model_id"] == model_id
    assert provenance["runtime_model_id"] == str(snapshot)
    assert provenance["resolved_revision"] == revision


def test_offline_hub_id_fails_when_snapshot_is_incomplete(monkeypatch, tmp_path):
    _snapshot(tmp_path, "org/model", "b" * 40)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    with pytest.raises(RuntimeError, match="no complete revision-pinned local snapshot"):
        resolve_offline_snapshot("org/model", required_paths=("vae/config.json",))


def test_direct_local_model_path_keeps_logical_identity(tmp_path):
    model = tmp_path / "model"
    (model / "config.json").parent.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")

    load_id, provenance = resolve_offline_snapshot(
        str(model), required_paths=("config.json",)
    )

    assert load_id == str(model)
    assert provenance["load_kind"] == "direct_local_path"
