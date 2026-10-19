#!/usr/bin/env python3
"""One-model, one-real-state v50@150 prefill/tap verification worker."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .extract_features import QwenExtractor, _load_images, _requirements, _task_text
from .io_utils import read_jsonl, sha256_file, stable_hash, write_json
from .protocol import validate_state_bank


V50 = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/unified_eval_runs/frozen_ckpts/v50_step150")


def _code_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: sha256_file(root / name) for name in ("sco_v50_smoke_worker.py", "extract_features.py", "io_utils.py", "protocol.py")}


def _checkpoint_identity(path: Path) -> dict[str, Any]:
    config = path / "config.json"
    index = path / "model.safetensors.index.json"
    if not config.is_file() or not index.is_file():
        raise RuntimeError("v50 exact HF config/index missing")
    mapping = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
    shards = sorted(set(mapping.values()))
    absent = [name for name in shards if not (path / name).is_file() or (path / name).stat().st_size == 0]
    if absent:
        raise RuntimeError(f"v50 index references absent/empty shards: {absent[:3]}")
    config_value = json.loads(config.read_text(encoding="utf-8"))
    return {"identity": "v50_7b_w3_format001_stable global_step_150 frozen HF export", "path": str(path), "model_type": config_value.get("model_type"), "architectures": config_value.get("architectures"), "config_sha256": sha256_file(config), "index_sha256": sha256_file(index), "indexed_tensor_count": len(mapping), "shards": [{"name": name, "bytes": (path / name).stat().st_size, "sha256": sha256_file(path / name)} for name in shards]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=V50)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    check = validate_state_bank(args.state_bank)
    if not check["valid"] or check["rows"] < 1:
        raise RuntimeError(f"input is not a valid real state bank: {check}")
    identity = _checkpoint_identity(args.checkpoint)
    row = next(read_jsonl(args.state_bank))
    torch, image_cls, _ = _requirements()
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device in worker; no forward attempted")
    extractor = QwenExtractor(args.checkpoint, "cuda:0", "bfloat16", "multimodal")
    values = extractor.extract(_load_images(row, args.state_bank, image_cls), _task_text(row))
    if set(values) != {"vision_terminal", "llm_mid_last_prompt", "llm_final_norm_last_prompt"}:
        raise RuntimeError(f"three required taps were not returned: {sorted(values)}")
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError("NaN/Inf in real v50 prefill tap")
    payload = {"sample_ids": [row["sample_id"]], "taps": {name: value.unsqueeze(0) for name, value in values.items()}}
    feature = args.output_dir / "v50_step150.real_one_state.pt"
    torch.save(payload, feature)
    attention_position = getattr(extractor, "last_prompt_token_position", None)
    manifest = {
        "status": "FORWARD_PASSED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_sha256": _code_identity(),
        "checkpoint": identity,
        "state_bank": {"path": str(args.state_bank), "validation": check, "sample_id": row["sample_id"], "sample_id_hash": stable_hash([row["sample_id"]]), "history_image_sha256": [item["image_sha256"] for item in row["history"]]},
        "execution": {"generation": "disabled", "use_cache": False, "gradients": "disabled", "device": str(torch.cuda.get_device_name(0)), "dtype": "bfloat16"},
        "actual_taps": {"vision_terminal": {"module": "model.visual", "pooling": "mean over terminal visual tokens"}, "llm_mid_last_prompt": {"module": extractor.mid_module_path, "token_position": attention_position}, "llm_final_norm_last_prompt": {"module": extractor.norm_module_path, "token_position": attention_position}},
        "tap_shapes": {name: list(value.shape) for name, value in values.items()},
        "feature": str(feature),
    }
    write_json(args.output_dir / "v50_smoke_manifest.json", manifest)
    print(json.dumps({"status": "FORWARD_PASSED", "sample_id": row["sample_id"], "tap_shapes": manifest["tap_shapes"]}))


if __name__ == "__main__":
    main()
