from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

import scripts.windowfix_v2.prepare_gap_backend_canary_jobs as renderer
import scripts.windowfix_v2.prepare_sco_gap_jobs as gap_planner
from features.canonical_gap_worklist import canonical_json_sha
from scripts.windowfix_v2.run_backend_determinism_canary import validate_gap_binding


PROJECT = Path(__file__).resolve().parents[2]
PROTOCOL = PROJECT / "protocols/inscene_streaming_canonical_v2.json"
BACKENDS = {
    "aether": {
        "model_id": "AetherWorldModel/AetherV1",
        "layer": 1,
        "preprocess": {"resize": [480, 720], "num_frames": 81},
        "prompt": "",
        "tail": [30, 45, 3072],
    },
    "qwen2_5_vl": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "layer": -1,
        "preprocess": {"resize": [336, 336], "num_frames": 8},
        "prompt": "Describe the spatial layout of this scene.",
        "tail": [12, 12, 3584],
    },
}


def _sha(path: Path) -> str:
    return renderer.sha256_file(path)


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _hashed_json(path: Path, value: dict) -> dict:
    value["content_sha256"] = canonical_json_sha(value)
    _json(path, value)
    return value


def _fixture(
    tmp_path: Path,
    *,
    namespace_override: str | None = None,
) -> tuple[Path, Path, list[dict]]:
    protocol = json.loads(PROTOCOL.read_text())
    source = tmp_path / "canonical.jsonl"
    source.write_text("frozen\n")
    source_sha = _sha(source)

    merge_units = []
    worklist_bindings = []
    blocked = []
    for vfm, config in BACKENDS.items():
        prompt_contract = {"mode": "exact", "value": config["prompt"]}
        merge_units.append(
            {
                "vfm": vfm,
                "model_id": config["model_id"],
                "timestep": 749,
                "layer": config["layer"],
                "accepted_modes": ["target_isolated"],
                "prompt_contract": prompt_contract,
                "preprocess_identity": config["preprocess"],
                "tensor_key": "feat",
                "expected_tail_shape": config["tail"],
                "expected_dtype": "torch.float16",
            }
        )
        worklist_path = tmp_path / "worklists" / f"{vfm}.json"
        worklist = _hashed_json(
            worklist_path,
            {
                "schema_version": 1,
                "contract": "CANONICAL_V2_GAP_WORKLIST_V1",
                "canonical_protocol": "inscene_streaming_canonical_v2_obs28_h124",
                "protocol_sha256": protocol["canonical_protocol_sha256"],
                "protocol_file_sha256": _sha(PROTOCOL),
                "canonical_source_sha256": source_sha,
                "mode": "target_isolated",
                "vfm": vfm,
                "model_id": config["model_id"],
                "timestep": 749,
                "feature_namespace": namespace_override
                or "FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1",
                "output_layers": [config["layer"]],
                "source": "scannetpp",
                "prompt_contract": prompt_contract,
                "preprocess_identity": config["preprocess"],
                "entries": [
                    {
                        "scene": "scene-a",
                        "frame_index": 7,
                        "layers": [config["layer"]],
                        "reason": "CANONICAL_TARGET_RAW_ID_UNROUTED",
                    }
                ],
                "counts": {"entries": 1, "layer_items": 1, "scenes": 1},
            },
        )
        worklist_bindings.append(
            {
                "path": str(worklist_path.resolve()),
                "file_sha256": _sha(worklist_path),
                "content_sha256": worklist["content_sha256"],
                "counts": worklist["counts"],
            }
        )
        blocked.append(
            {
                "vfm": vfm,
                "model_id": config["model_id"],
                "timestep": 749,
                "mode": "target_isolated",
                "layers": [config["layer"]],
                "worklist_path": str(worklist_path.resolve()),
                "worklist_file_sha256": _sha(worklist_path),
                "layer_items": 1,
                "reason": "BLOCKED_BACKEND_CANARY_NOT_BOUND",
            }
        )

    merge_path = tmp_path / "target_merge_plan.json"
    merge = _hashed_json(
        merge_path,
        {
            "schema_version": 1,
            "contract": "CANONICAL_V2_TARGET_ROW_MERGE_V1",
            "protocol_file_sha256": _sha(PROTOCOL),
            "canonical_source": {"path": str(source), "sha256": source_sha},
            "units": merge_units,
        },
    )
    audit_path = tmp_path / "target_audit.json"
    audit = _hashed_json(
        audit_path,
        {
            "schema_version": 1,
            "status": "INCOMPLETE",
            "counts": {"hard_errors": 0},
            "hard_errors": [],
            "protocol_file_sha256": _sha(PROTOCOL),
            "canonical_source": {"path": str(source), "sha256": source_sha},
            "plan": str(merge_path.resolve()),
            "plan_sha256": _sha(merge_path),
            "outputs": {"gap_worklists": worklist_bindings},
        },
    )
    assert merge["content_sha256"]
    plan_dir = tmp_path / "gap_plan"
    summary_path = plan_dir / "summary.json"
    blocked_path = plan_dir / "blocked.json"
    _json(
        summary_path,
        {
            "schema_version": 1,
            "status": "BLOCKED_PARTIAL",
            "submission_performed": False,
            "protocol_file_sha256": _sha(PROTOCOL),
            "canonical_source_sha256": source_sha,
            "source_audits": [
                {"path": str(audit_path.resolve()), "sha256": _sha(audit_path)}
            ],
            "blocked": blocked,
        },
    )
    _json(blocked_path, {"schema_version": 1, "blocked": blocked})
    _json(
        plan_dir / "PLAN.lock.json",
        {
            "schema_version": 1,
            "summary_sha256": _sha(summary_path),
            "blocked_sha256": _sha(blocked_path),
        },
    )
    bindings = [
        {
            "audit_path": str(audit_path.resolve()),
            "audit_sha256": _sha(audit_path),
            "audit_content_sha256": audit["content_sha256"],
            "worklist_path": item["path"],
            "worklist_file_sha256": item["file_sha256"],
            "worklist_content_sha256": item["content_sha256"],
            "worklist": json.loads(Path(item["path"]).read_text()),
        }
        for item in worklist_bindings
    ]
    return plan_dir, source, bindings


def _patch_provenance(monkeypatch) -> None:
    provenance = {"fixture": True}
    monkeypatch.setattr(
        renderer, "backend_provenance_binding", lambda vfm, root: provenance
    )
    monkeypatch.setattr(renderer, "canary_runtime_environment", lambda vfm, p: {})
    monkeypatch.setattr(
        gap_planner,
        "canary_backend_provenance_binding",
        lambda vfm, root: provenance,
    )
    monkeypatch.setattr(
        gap_planner, "canary_runtime_environment", lambda vfm, p: {}
    )


def test_target_subset_renderer_and_post_validator_accept_only_current_gated_vfms(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_provenance(monkeypatch)
    plan_dir, _source, bindings = _fixture(tmp_path)
    output = tmp_path / "canaries"
    assert renderer.main(
        [
            "--plan-dir",
            str(plan_dir),
            "--output-dir",
            str(output),
            "--protocol",
            str(PROTOCOL),
            "--project-root",
            str(PROJECT),
            "--dataset-root",
            str(tmp_path / "dataset"),
        ]
    ) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["backends"] == ["aether", "qwen2_5_vl"]
    assert summary["worklist_modes"] == {
        "aether": "target_isolated",
        "qwen2_5_vl": "target_isolated",
    }
    for vfm, config in BACKENDS.items():
        spec = json.loads((output / "specs" / f"{vfm}.json").read_text())
        assert spec["blocked_mode"] == "target_isolated"
        assert spec["target_tensor_contract"]["expected_tail_shape"] == config["tail"]
        assert spec["target_tensor_contract"]["expected_dtype"] == "float16"

    # Missing receipts remain fail-closed but do not make the validator demand
    # absent F3R/Lingbot specs that are not present in these source audits.
    assert gap_planner.valid_canary_bindings(
        output, _sha(PROTOCOL), bindings, PROJECT
    ) == {}

    qwen_spec = json.loads((output / "specs/qwen2_5_vl.json").read_text())
    runtime = qwen_spec["runtime_contract"]
    args = Namespace(
        blocked_worklist=Path(qwen_spec["blocked_worklist"]),
        blocked_worklist_file_sha256=qwen_spec["blocked_worklist_file_sha256"],
        blocked_worklist_content_sha256=qwen_spec[
            "blocked_worklist_content_sha256"
        ],
        source_audit=Path(qwen_spec["source_audit"]),
        source_audit_sha256=qwen_spec["source_audit_sha256"],
        source_plan_summary=Path(qwen_spec["source_plan_summary"]),
        source_plan_summary_sha256=qwen_spec["source_plan_summary_sha256"],
        resize=runtime["resize"],
        num_frames=runtime["num_frames"],
        prompt=runtime["prompt"],
        vfm=qwen_spec["vfm"],
        model_id=qwen_spec["model_id"],
        timestep=qwen_spec["timestep"],
        layer=qwen_spec["layer"],
        protocol_file_sha256=qwen_spec["protocol_file_sha256"],
    )
    binding = validate_gap_binding(args, qwen_spec)
    assert binding["worklist_mode"] == "target_isolated"
    assert binding["target_tensor_contract"]["expected_tail_shape"] == [12, 12, 3584]


def test_target_subset_renderer_rejects_mode_namespace_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_provenance(monkeypatch)
    plan_dir, _source, _bindings = _fixture(
        tmp_path,
        namespace_override="FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1",
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        renderer.main(
            [
                "--plan-dir",
                str(plan_dir),
                "--output-dir",
                str(tmp_path / "bad-canaries"),
                "--protocol",
                str(PROTOCOL),
                "--project-root",
                str(PROJECT),
            ]
        )
