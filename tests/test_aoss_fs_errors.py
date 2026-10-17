"""Remote outages must not become permanently cached missing inputs."""

import pytest

from vidfm3d.utils import aoss_fs


class UnavailableClient:
    def contains(self, path):
        raise ConnectionError("offline")

    def isdir(self, path):
        raise ConnectionError("offline")

    def list(self, path):
        raise ConnectionError("offline")


@pytest.mark.parametrize("operation", [aoss_fs.exists, aoss_fs.isdir])
def test_remote_failure_is_reported_and_not_cached(monkeypatch, tmp_path, operation):
    monkeypatch.setenv("AOSS_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(aoss_fs, "_EXISTS_RESULT_CACHE", {})
    monkeypatch.setattr(aoss_fs, "get_client", lambda: UnavailableClient())
    path = "s3://example/features/sample.sft"
    with pytest.raises(OSError, match="Unable to query remote"):
        operation(path)
    assert path not in aoss_fs._EXISTS_RESULT_CACHE

    class RecoveredClient:
        def contains(self, path):
            return True

        def isdir(self, path):
            return True

    monkeypatch.setattr(aoss_fs, "get_client", lambda: RecoveredClient())
    assert operation(path)


def test_remote_missing_result_can_still_be_cached(monkeypatch, tmp_path):
    class MissingClient:
        def contains(self, path):
            return False

        def isdir(self, path):
            return False

    monkeypatch.setenv("AOSS_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(aoss_fs, "_EXISTS_RESULT_CACHE", {})
    monkeypatch.setattr(aoss_fs, "get_client", lambda: MissingClient())
    path = "s3://example/missing.sft"
    assert not aoss_fs.exists(path)
    assert aoss_fs._EXISTS_RESULT_CACHE[path] is False


@pytest.mark.parametrize("operation", [aoss_fs.exists, aoss_fs.isdir])
@pytest.mark.parametrize("children", [[], ["child.sft"]])
def test_listing_fallback_only_proves_presence(monkeypatch, tmp_path, operation, children):
    class ListingClient(UnavailableClient):
        def list(self, path):
            return iter(children)

    monkeypatch.setenv("AOSS_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(aoss_fs, "_EXISTS_RESULT_CACHE", {})
    monkeypatch.setattr(aoss_fs, "get_client", lambda: ListingClient())
    path = "s3://example/features/sample.sft"
    if children:
        assert operation(path)
    else:
        with pytest.raises(OSError, match="Unable to query remote"):
            operation(path)
        assert path not in aoss_fs._EXISTS_RESULT_CACHE
