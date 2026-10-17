from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "analyses/windowfix_v2/build_reextract_plan.py"
SPEC = importlib.util.spec_from_file_location("build_reextract_plan", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_stale_protocol_is_rejected(tmp_path: Path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"acceptance": {"legacy_features_valid": False}}))
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"source": {"protocol_file_sha256": "0" * 64}}))
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"status": "pass", "problem_count": 0}))
    try:
        module.build(
            inventory_path=inventory,
            protocol_path=protocol,
            label_receipt_path=receipt,
        )
    except ValueError as exc:
        assert "inventory is stale" in str(exc)
    else:
        raise AssertionError("stale protocol must fail closed")


def test_backend_matrix_is_exhaustive():
    assert set(module.STRICT_BACKEND_STATUS) == {
        "wan",
        "cogvideox",
        "minimax_h3",
        "dino",
        "vjepa2",
        "videomae",
        "aether",
        "f3r",
        "lingbot",
        "qwen2_5_vl",
    }
    assert module.STRICT_BACKEND_STATUS["wan"][0] == "READY"
    assert module.STRICT_BACKEND_STATUS["qwen2_5_vl"][0].startswith("BLOCKED")
