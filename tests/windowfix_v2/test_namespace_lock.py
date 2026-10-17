from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "analyses/windowfix_v2/build_namespace_lock.py"
SPEC = importlib.util.spec_from_file_location("build_namespace_lock", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_build_binds_every_manifest_hash(tmp_path: Path):
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps({
        "canonical_protocol": "v2",
        "streaming": {"prefix_lengths": [8, 12]},
        "feature_namespaces": {"streaming": "STREAM_V2", "target": "TARGET_V2"},
    }))
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifests = {}
    for prefix in (8, 12):
        path = manifest_dir / f"inscene_c_h124_prefix{prefix}.jsonl"
        path.write_text(f'{{"prefix":{prefix}}}\n')
        digest = module.sha256_file(path)
        Path(f"{path}.meta.json").write_text(json.dumps({"sha256": digest}))
        manifests[str(prefix)] = {"sha256": digest}
    (manifest_dir / "canonical_summary.json").write_text(json.dumps({
        "universe_sha256": "a" * 64,
        "shared_unit_count": 2,
        "manifests": manifests,
    }))
    lock = module.build(protocol_path, manifest_dir, "aoss://bucket/base/")
    assert lock["manifest_sha256"] == {
        key: value["sha256"] for key, value in manifests.items()
    }
    assert lock["feature_roots"]["streaming"] == "aoss://bucket/base/STREAM_V2"
    assert lock["feature_roots"]["target"] == "aoss://bucket/base/TARGET_V2"
