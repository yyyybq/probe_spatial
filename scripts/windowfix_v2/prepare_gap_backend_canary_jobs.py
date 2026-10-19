#!/usr/bin/env python3
"""Render review-only backend determinism canary SCO specs for gap plans.

This tool does not submit SCO jobs.  It consumes a SHA-locked
``prepare_sco_gap_jobs.py`` plan with canary-blocked worklists, writes one
immutable spec per requested backend, and wraps the worker command in
``bash`` so startup diagnostics are persisted even when the default
SCO shell is ``/bin/sh``.  The create payload must not contain ``/bin/bash``;
the ACP gateway currently rejects those commands with HTTP 403.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.canonical_gap_worklist import (  # noqa: E402
    CanonicalGapWorklist,
    canonical_json_sha,
    sha256_file,
    streaming_worklist_preprocess_identity,
)
from scripts.windowfix_v2.run_backend_determinism_canary import (  # noqa: E402
    backend_provenance_binding,
    canary_runtime_environment,
)


BACKENDS: dict[str, dict[str, Any]] = {
    "aether": {
        "model_id": "AetherWorldModel/AetherV1",
        "layer": 1,
        "timestep": 749,
        "python": "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python",
        "env": {},
    },
    "f3r": {
        "model_id": "jedyang97/Fast3R_ViT_Large_512",
        "layer": 24,
        "timestep": 749,
        "python": "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python",
        "env": {},
    },
    "lingbot": {
        "model_id": "/mnt/umm/users/yinbaiqiao/Consistency/lingbot-video/models/lingbot-video-moe-30b-a3b",
        "layer": 10,
        "timestep": 250,
        "python": "/mnt/umm/users/yinbaiqiao/.conda/envs/lingbot-video/bin/python",
        "env": {
            "DIFFUSERS_ATTN_BACKEND": "_native_flash",
            "HF_HOME": "/mnt/umm/shared_model/huggingface",
            "LINGBOT_MOE_EXPERT_BACKEND": "grouped_mm",
            "LINGBOT_MOE_PAD_BACKEND": "vectorized",
            "PYTHONPATH": "/mnt/umm/users/yinbaiqiao/Consistency/lingbot-video:/mnt/umm/users/yinbaiqiao/probe_spatial",
        },
    },
    "qwen2_5_vl": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "layer": -1,
        "timestep": 749,
        "python": "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python",
        "env": {},
    },
}
GAP_NAMESPACES = {
    "streaming_prefix": "FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1",
    "target_isolated": "FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1",
}
REQUIRED_TARGET_TAIL_SHAPES = {
    "aether": [30, 45, 3072],
    "qwen2_5_vl": [12, 12, 3584],
}


def read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"{label} must be a regular non-symlink file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must contain one JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def require_sha(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise SystemExit(f"{label} SHA mismatch: {path}")


def render(argv: Sequence[Any]) -> str:
    return shlex.join([str(item) for item in argv])


def safe_slug(value: str) -> str:
    clean = re.sub(r"[^a-z0-9._-]+", "-", value.strip().lower()).strip("-._")
    if not clean or clean != value:
        raise SystemExit("job-name-suffix must already be a safe lowercase slug")
    return clean


def prompt_value(contract: Any) -> str:
    if not isinstance(contract, dict):
        raise SystemExit("invalid prompt_contract")
    if contract.get("mode") == "exact" and isinstance(contract.get("value"), str):
        return str(contract["value"])
    if contract == {"mode": "not_applicable", "value": None}:
        return ""
    raise SystemExit("invalid prompt_contract")


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = read_json(path, "canonical protocol")
    file_sha = sha256_file(path)
    if protocol.get("canonical_protocol") != "inscene_streaming_canonical_v2_obs28_h124":
        raise SystemExit("wrong canonical protocol")
    return protocol, file_sha


def validate_parent(plan_dir: Path, protocol_file_sha: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    summary_path = plan_dir / "summary.json"
    blocked_path = plan_dir / "blocked.json"
    lock_path = plan_dir / "PLAN.lock.json"
    summary = read_json(summary_path, "gap plan summary")
    blocked = read_json(blocked_path, "gap blocked inventory")
    lock = read_json(lock_path, "gap plan lock")
    require_sha(summary_path, lock.get("summary_sha256"), "gap summary")
    require_sha(blocked_path, lock.get("blocked_sha256"), "gap blocked inventory")
    if (
        summary.get("status") != "BLOCKED_PARTIAL"
        or summary.get("submission_performed") is not False
        or summary.get("blocked") != blocked.get("blocked")
        or summary.get("protocol_file_sha256") != protocol_file_sha
    ):
        raise SystemExit("gap plan is not a consistent BLOCKED_PARTIAL review artifact")
    audits = summary.get("source_audits")
    if not isinstance(audits, list) or len(audits) != 1:
        raise SystemExit("canary rendering requires exactly one source audit")
    return summary, blocked, lock


def target_tensor_contract(
    audit: dict[str, Any],
    *,
    vfm: str,
    model_id: str,
    timestep: int,
    layer: int,
    prompt_contract: dict[str, Any],
    preprocess_identity: dict[str, Any],
    protocol_file_sha256: str,
    canonical_source_sha256: str,
) -> dict[str, Any]:
    plan_path = Path(str(audit.get("plan", ""))).resolve()
    require_sha(plan_path, audit.get("plan_sha256"), "target merge plan")
    plan = read_json(plan_path, "target merge plan")
    raw = dict(plan)
    content = raw.pop("content_sha256", None)
    if content != canonical_json_sha(raw):
        raise SystemExit("target merge plan content SHA mismatch")
    if (
        plan.get("contract") != "CANONICAL_V2_TARGET_ROW_MERGE_V1"
        or plan.get("protocol_file_sha256") != protocol_file_sha256
        or plan.get("canonical_source", {}).get("sha256") != canonical_source_sha256
    ):
        raise SystemExit("target merge plan canonical identity mismatch")
    matches = []
    for unit in plan.get("units", []):
        if not isinstance(unit, dict):
            continue
        if (
            unit.get("vfm") == vfm
            and unit.get("model_id") == model_id
            and int(unit.get("timestep", -1)) == timestep
            and int(unit.get("layer", 10**9)) == layer
            and "target_isolated" in unit.get("accepted_modes", [])
            and unit.get("prompt_contract") == prompt_contract
            and unit.get("preprocess_identity") == preprocess_identity
        ):
            matches.append(unit)
    identities = {
        (
            str(unit.get("tensor_key")),
            tuple(int(x) for x in unit.get("expected_tail_shape", [])),
            str(unit.get("expected_dtype")).removeprefix("torch.").lower(),
        )
        for unit in matches
    }
    if len(identities) != 1:
        raise SystemExit(f"target merge plan has no unique tensor contract: {vfm}")
    tensor_key, tail_shape, dtype = next(iter(identities))
    if (
        tensor_key != "feat"
        or dtype != "float16"
        or not tail_shape
        or list(tail_shape) != REQUIRED_TARGET_TAIL_SHAPES.get(vfm, list(tail_shape))
    ):
        raise SystemExit(f"unsupported target tensor contract: {vfm}")
    return {
        "schema_version": 1,
        "contract": "CANONICAL_V2_TARGET_ROW_TENSOR_CONTRACT_V1",
        "source_target_merge_plan": str(plan_path),
        "source_target_merge_plan_file_sha256": str(audit.get("plan_sha256")),
        "source_target_merge_plan_content_sha256": str(content),
        "tensor_key": tensor_key,
        "expected_tail_shape": list(tail_shape),
        "expected_dtype": dtype,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--only-vfm", action="append", default=[])
    parser.add_argument("--sample-index", type=int, default=1)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=Path("/mnt/umm/users/yinbaiqiao/InsScene-15K"))
    parser.add_argument("--workspace", default="aigc")
    parser.add_argument("--aec2-name", default="h800")
    parser.add_argument("--worker-spec", default="N4lS.Iq.I80.8")
    parser.add_argument("--priority", default="HIGHEST")
    parser.add_argument("--quota-type", choices=["reserved", "spot"], default="reserved")
    parser.add_argument("--container-image", default="registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617")
    parser.add_argument("--storage-mount", default="019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm")
    parser.add_argument(
        "--job-name-suffix",
        default="",
        help="Optional safe lowercase suffix appended only to the SCO display name.",
    )
    args = parser.parse_args(argv)
    if args.sample_index < 0:
        raise SystemExit("--sample-index must be >=0")

    plan_dir = args.plan_dir.resolve()
    output_dir = args.output_dir.resolve()
    project_root = args.project_root.resolve()
    protocol_path = args.protocol.resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to replace immutable canary plan: {output_dir}")
    job_name_suffix = safe_slug(args.job_name_suffix) if args.job_name_suffix else ""
    protocol, protocol_file_sha = load_protocol(protocol_path)
    summary, blocked_file, lock = validate_parent(plan_dir, protocol_file_sha)

    audit_binding = summary["source_audits"][0]
    audit_path = Path(str(audit_binding["path"])).resolve()
    require_sha(audit_path, audit_binding.get("sha256"), "gap source audit")
    audit = read_json(audit_path, "gap source audit")
    audit_raw = dict(audit)
    audit_content_sha = audit_raw.pop("content_sha256", None)
    if audit_content_sha != canonical_json_sha(audit_raw):
        raise SystemExit("gap source audit content SHA mismatch")
    canonical_source = audit.get("canonical_source", {})
    canonical_source_path = Path(str(canonical_source.get("path", ""))).resolve()
    require_sha(canonical_source_path, canonical_source.get("sha256"), "canonical source")
    if canonical_source.get("sha256") != summary.get("canonical_source_sha256"):
        raise SystemExit("gap plan/source audit canonical source mismatch")

    blocked = list(summary.get("blocked") or [])
    requested = set(args.only_vfm)
    if requested:
        blocked = [item for item in blocked if str(item.get("vfm")) in requested]
    if not blocked:
        raise SystemExit("canary subset selects zero blocked backends")
    unknown = requested - {str(item.get("vfm")) for item in summary.get("blocked", [])}
    if unknown:
        raise SystemExit("--only-vfm is not blocked by this parent plan: " + ",".join(sorted(unknown)))
    seen: set[str] = set()
    for item in blocked:
        vfm = str(item.get("vfm"))
        if vfm in seen or vfm not in BACKENDS:
            raise SystemExit("blocked inventory must be a unique reviewed backend subset")
        seen.add(vfm)
        if item.get("reason") != "BLOCKED_BACKEND_CANARY_NOT_BOUND":
            raise SystemExit("blocked item is not canary-gated")
        if item.get("mode") not in GAP_NAMESPACES:
            raise SystemExit("blocked item has unsupported mode")

    audit_worklists = {
        str(Path(str(item.get("path", ""))).resolve()): item
        for item in audit.get("outputs", {}).get("gap_worklists", [])
        if isinstance(item, dict)
    }
    main_extractor = project_root / "features/run_inscene15k.py"
    mllm_extractor = project_root / "features/run_inscene15k_mllm.py"
    canary_runner = project_root / "scripts/windowfix_v2/run_backend_determinism_canary.py"
    gap_renderer = project_root / "scripts/windowfix_v2/prepare_sco_gap_jobs.py"
    canary_renderer = Path(__file__).resolve()

    staging = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    (staging / "specs").mkdir(parents=True)
    (staging / "scripts").mkdir()
    (staging / "receipts").mkdir()
    (staging / "logs").mkdir()
    commands = [
        "# REVIEW ARTIFACT ONLY. The preparer never invokes SCO.",
        "# Submit via record_backend_canary_submissions.py --execute.",
    ]
    modes: dict[str, str] = {}

    for item in sorted(blocked, key=lambda row: str(row["vfm"])):
        vfm = str(item["vfm"])
        mode = str(item["mode"])
        config = BACKENDS[vfm]
        if (
            item.get("model_id") != config["model_id"]
            or int(item.get("timestep", -1)) != int(config["timestep"])
            or int(config["layer"]) not in [int(x) for x in item.get("layers", [])]
        ):
            raise SystemExit(f"blocked backend identity differs from canary contract: {vfm}")
        worklist_path = Path(str(item["worklist_path"])).resolve()
        audit_worklist = audit_worklists.get(str(worklist_path))
        if not isinstance(audit_worklist, dict):
            raise SystemExit(f"source audit does not bind blocked worklist: {worklist_path}")
        require_sha(worklist_path, item.get("worklist_file_sha256"), "blocked worklist")
        raw_worklist = read_json(worklist_path, "blocked worklist")
        prompt = prompt_value(raw_worklist.get("prompt_contract"))
        preprocess = raw_worklist.get("preprocess_identity")
        if not isinstance(preprocess, dict):
            raise SystemExit(f"invalid preprocess identity: {vfm}")
        if mode == "streaming_prefix":
            n_frames = int(preprocess.get("num_frames", 1))
            expected_preprocess = streaming_worklist_preprocess_identity(vfm, preprocess.get("resize", ()), n_frames)
            if preprocess != expected_preprocess:
                raise SystemExit(f"non-canonical streaming preprocess: {vfm}")
            if "num_frames_by_prefix" in preprocess:
                n_frames = max(int(x) for x in preprocess["num_frames_by_prefix"].values())
        else:
            if set(preprocess) != {"resize", "num_frames"}:
                raise SystemExit(f"non-canonical target preprocess: {vfm}")
            n_frames = int(preprocess["num_frames"])

        worklist = CanonicalGapWorklist.load(
            worklist_path,
            expected_sha256=str(item["worklist_file_sha256"]),
            mode=mode,
            vfm=vfm,
            model_id=str(config["model_id"]),
            timestep=int(config["timestep"]),
            feature_namespace=GAP_NAMESPACES[mode],
            output_layers=[int(x) for x in raw_worklist["output_layers"]],
            protocol_sha256=str(protocol["canonical_protocol_sha256"]),
            actual_prompt=prompt,
            preprocess_identity=preprocess,
        )
        if worklist.content_sha256 != audit_worklist.get("content_sha256"):
            raise SystemExit(f"blocked worklist content SHA mismatch: {vfm}")

        extractor = mllm_extractor if vfm == "qwen2_5_vl" else main_extractor
        receipt = output_dir / "receipts" / f"{vfm}.json"
        log_path = output_dir / "logs" / f"{vfm}.log"
        spec_path = output_dir / "specs" / f"{vfm}.json"
        worker_script_path = output_dir / "scripts" / f"{vfm}.sh"
        backend_provenance = backend_provenance_binding(vfm, project_root)
        runtime_environment = canary_runtime_environment(vfm, backend_provenance)
        if any(runtime_environment.get(k) != v for k, v in config["env"].items()):
            raise SystemExit(f"canary environment table drifted for {vfm}")
        target_contract = (
            target_tensor_contract(
                audit,
                vfm=vfm,
                model_id=str(config["model_id"]),
                timestep=int(config["timestep"]),
                layer=int(config["layer"]),
                prompt_contract=worklist.prompt_contract,
                preprocess_identity=worklist.preprocess_identity,
                protocol_file_sha256=protocol_file_sha,
                canonical_source_sha256=worklist.canonical_source_sha256,
            )
            if mode == "target_isolated"
            else None
        )

        inner = [
            str(config["python"]),
            str(canary_runner),
            "--vfm", vfm,
            "--sample-index", str(args.sample_index),
            "--model-id", str(config["model_id"]),
            "--layer", str(config["layer"]),
            "--timestep", str(config["timestep"]),
            "--python", str(config["python"]),
            "--project-root", str(project_root),
            "--dataset-root", str(args.dataset_root.resolve()),
            "--protocol", str(protocol_path),
            "--protocol-file-sha256", protocol_file_sha,
            "--extractor-sha256", sha256_file(extractor),
            "--canary-spec", str(spec_path),
            "--prompt", prompt,
            "--resize", *[str(x) for x in preprocess["resize"]],
            "--num-frames", str(n_frames),
            "--blocked-worklist", str(worklist_path),
            "--blocked-worklist-file-sha256", worklist.file_sha256,
            "--blocked-worklist-content-sha256", worklist.content_sha256,
            "--source-audit", str(audit_path),
            "--source-audit-sha256", sha256_file(audit_path),
            "--source-plan-summary", str(plan_dir / "summary.json"),
            "--source-plan-summary-sha256", sha256_file(plan_dir / "summary.json"),
            "--output", str(receipt),
        ]
        environment = [f"{k}={v}" for k, v in sorted(runtime_environment.items())]
        inner_command = f"cd {shlex.quote(str(project_root))} && "
        if environment:
            inner_command += render(["env", *environment]) + " "
        inner_command += render(inner)
        worker_script = (
            "#!/usr/bin/env bash\n"
            "set -o pipefail; "
            f"mkdir -p {shlex.quote(str(log_path.parent))} {shlex.quote(str(receipt.parent))}; "
            f"({inner_command}) > {shlex.quote(str(log_path))} 2>&1; "
            "canary_rc=$?; "
            f"if [ -f {shlex.quote(str(receipt))} ]; then chmod 0644 {shlex.quote(str(receipt))}; fi; "
            f"chmod 0644 {shlex.quote(str(log_path))} || true; "
            "exit ${canary_rc}"
        )
        staged_worker_script = staging / "scripts" / f"{vfm}.sh"
        staged_worker_script.write_text(worker_script + "\n", encoding="utf-8")
        staged_worker_script.chmod(0o755)
        worker_command = (
            f"cd {shlex.quote(str(project_root))} && "
            f"{render(['bash', str(worker_script_path)])}"
        )
        job_name = f"bq-wf2-gap-canary-{'stream' if mode == 'streaming_prefix' else 'target'}-{vfm}"
        if job_name_suffix:
            job_name = f"{job_name}-{job_name_suffix}"
        sco_argv = [
            "sco", "acp", "jobs", "create",
            f"--workspace-name={args.workspace}",
            f"--aec2-name={args.aec2_name}",
            f"--job-name={job_name}",
            f"--priority={args.priority}",
            f"--quota-type={args.quota_type}",
            f"--container-image-url={args.container_image}",
            f"--storage-mount={args.storage_mount}",
            "--training-framework=pytorch",
            "--worker-nodes=1",
            f"--worker-spec={args.worker_spec}",
            f"--command={worker_command}",
        ]
        spec = {
            "schema_version": 2,
            "audit_type": "canonical_v2_backend_canary_execution_spec",
            "status": "READY_FOR_REVIEW_NOT_SUBMITTED",
            "submission_performed": False,
            "source_gap_plan": str(plan_dir),
            "source_plan_summary": str(plan_dir / "summary.json"),
            "source_plan_summary_sha256": sha256_file(plan_dir / "summary.json"),
            "source_plan_lock_sha256": sha256_file(plan_dir / "PLAN.lock.json"),
            "source_audit": str(audit_path),
            "source_audit_sha256": sha256_file(audit_path),
            "source_audit_content_sha256": audit_content_sha,
            "blocked_worklist": str(worklist_path),
            "blocked_worklist_file_sha256": worklist.file_sha256,
            "blocked_worklist_content_sha256": worklist.content_sha256,
            "canonical_source_sha256": worklist.canonical_source_sha256,
            "vfm": vfm,
            "model_id": config["model_id"],
            "timestep": config["timestep"],
            "layer": config["layer"],
            "blocked_layers": [int(x) for x in item["layers"]],
            "blocked_mode": mode,
            "feature_namespace": GAP_NAMESPACES[mode],
            "prompt_contract": worklist.prompt_contract,
            "preprocess_identity": worklist.preprocess_identity,
            "protocol_file": str(protocol_path),
            "protocol_file_sha256": protocol_file_sha,
            "extractor": str(extractor),
            "extractor_sha256": sha256_file(extractor),
            "canary_runner": str(canary_runner),
            "canary_runner_sha256": sha256_file(canary_runner),
            "canary_renderer": str(canary_renderer),
            "canary_renderer_sha256": sha256_file(canary_renderer),
            "runtime_contract": {
                "prompt": prompt,
                "resize": [int(x) for x in preprocess["resize"]],
                "num_frames": n_frames,
                "python": str(config["python"]),
                "sample_index": args.sample_index,
            },
            "runtime_environment": runtime_environment,
            "backend_provenance": backend_provenance,
            "target_tensor_contract": target_contract,
            "receipt": str(receipt),
            "log": str(log_path),
            "worker_script": str(worker_script_path),
            "worker_script_sha256": sha256_file(staged_worker_script),
            "worker_command": worker_command,
            "sco_argv": sco_argv,
            "sco_command": render(sco_argv),
        }
        spec["content_sha256"] = canonical_json_sha(spec)
        atomic_json(staging / "specs" / f"{vfm}.json", spec)
        commands.append(spec["sco_command"])
        modes[vfm] = mode

    (staging / "sco_create_commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    post_output = output_dir.with_name(f"{output_dir.name}_postcanary_gap_plan")
    post_argv = [
        "python3", str(gap_renderer),
        "--source-audit", str(audit_path),
        "--protocol", str(protocol_path),
        "--canonical-source", str(canonical_source_path),
        "--backend-canary-dir", str(output_dir),
        "--output-dir", str(post_output),
    ]
    post_path = staging / "post_canary_render_command.txt"
    post_path.write_text(render(post_argv) + "\n", encoding="utf-8")
    specs = sorted((staging / "specs").glob("*.json"))
    summary_out = {
        "schema_version": 1,
        "status": "READY_FOR_REVIEW_NOT_SUBMITTED",
        "submission_performed": False,
        "source_gap_plan": str(plan_dir),
        "source_plan_summary_sha256": sha256_file(plan_dir / "summary.json"),
        "source_plan_lock_sha256": sha256_file(plan_dir / "PLAN.lock.json"),
        "source_audit_sha256": sha256_file(audit_path),
        "canonical_source_sha256": canonical_source.get("sha256"),
        "protocol_file_sha256": protocol_file_sha,
        "main_extractor_sha256": sha256_file(main_extractor),
        "mllm_extractor_sha256": sha256_file(mllm_extractor),
        "canary_runner_sha256": sha256_file(canary_runner),
        "canary_renderer_sha256": sha256_file(canary_renderer),
        "backends": [path.stem for path in specs],
        "worklist_modes": modes,
        "job_count": len(specs),
        "commands": str(output_dir / "sco_create_commands.txt"),
        "post_canary_render_command": str(output_dir / "post_canary_render_command.txt"),
        "post_canary_output": str(post_output),
    }
    atomic_json(staging / "summary.json", summary_out)
    lock_out = {
        "schema_version": 1,
        "summary_sha256": sha256_file(staging / "summary.json"),
        "specs": {path.name: sha256_file(path) for path in specs},
        "worker_scripts": {path.name: sha256_file(path) for path in sorted((staging / "scripts").glob("*.sh"))},
        "commands_sha256": sha256_file(staging / "sco_create_commands.txt"),
        "post_canary_render_command_sha256": sha256_file(post_path),
    }
    atomic_json(staging / "PLAN.lock.json", lock_out)
    os.replace(staging, output_dir)
    print(json.dumps(summary_out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
