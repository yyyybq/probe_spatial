#!/usr/bin/env python3
"""Render SCO review artifacts for blocked backend determinism canaries.

This renderer never imports or invokes SCO.  It writes JSON specs and one text
file containing reviewed `sco acp jobs create` commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import tempfile
from pathlib import Path
from typing import Any


BACKENDS = {
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
            "HF_HOME": "/mnt/umm/shared_model/huggingface",
            "DIFFUSERS_ATTN_BACKEND": "_native_flash",
            "LINGBOT_MOE_PAD_BACKEND": "vectorized",
            "LINGBOT_MOE_EXPERT_BACKEND": "grouped_mm",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dataset-root", type=Path, default=Path("/mnt/umm/users/yinbaiqiao/InsScene-15K"))
    parser.add_argument("--workspace", default="aigc")
    parser.add_argument("--aec2-name", default="h800")
    parser.add_argument("--worker-spec", default="N4lS.Iq.I80.8")
    parser.add_argument("--priority", default="HIGHEST")
    parser.add_argument(
        "--container-image",
        default="registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617",
    )
    parser.add_argument(
        "--storage-mount",
        default="019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan_dir = args.plan_dir.resolve()
    project_root = args.project_root.resolve()
    protocol = args.protocol.resolve()
    main_extractor = project_root / "features/run_inscene15k.py"
    mllm_extractor = project_root / "features/run_inscene15k_mllm.py"
    canary = project_root / "scripts/windowfix_v2/run_backend_determinism_canary.py"
    summary_path = plan_dir / "summary.json"
    for path in (protocol, main_extractor, mllm_extractor, canary, summary_path):
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"missing regular bound input: {path}")
    plan = json.loads(summary_path.read_text())
    protocol_sha = sha256_file(protocol)
    main_extractor_sha = sha256_file(main_extractor)
    mllm_extractor_sha = sha256_file(mllm_extractor)
    if plan.get("protocol", {}).get("file_sha256") != protocol_sha:
        raise SystemExit("plan protocol SHA does not match current protocol")
    if plan.get("source_bindings", {}).get("main_extractor", {}).get("sha256") != main_extractor_sha:
        raise SystemExit("plan extractor SHA does not match current extractor")
    if plan.get("source_bindings", {}).get("mllm_extractor", {}).get("sha256") != mllm_extractor_sha:
        raise SystemExit("plan MLLM extractor SHA does not match current extractor")

    output_dir = plan_dir / "backend_canaries"
    output_dir.mkdir(parents=True, exist_ok=True)
    specs_dir = output_dir / "specs"
    specs_dir.mkdir(exist_ok=True)
    commands = [
        "# REVIEW ARTIFACT ONLY. This file is not executed by the preparer.",
        "# Submit these canaries before unblocking Aether/F3R/LingBot/Qwen full extraction.",
    ]
    specs: list[dict[str, Any]] = []
    for vfm, config in BACKENDS.items():
        extractor = mllm_extractor if vfm == "qwen2_5_vl" else main_extractor
        extractor_sha = mllm_extractor_sha if vfm == "qwen2_5_vl" else main_extractor_sha
        receipt = output_dir / "receipts" / f"{vfm}.json"
        inner = [
            str(config["python"]),
            str(canary),
            "--vfm",
            vfm,
            "--model-id",
            str(config["model_id"]),
            "--layer",
            str(config["layer"]),
            "--timestep",
            str(config["timestep"]),
            "--project-root",
            str(project_root),
            "--dataset-root",
            str(args.dataset_root.resolve()),
            "--protocol",
            str(protocol),
            "--protocol-file-sha256",
            protocol_sha,
            "--extractor-sha256",
            extractor_sha,
            "--output",
            str(receipt),
        ]
        environment = [f"{key}={value}" for key, value in sorted(config["env"].items())]
        worker_command = f"cd {shlex.quote(str(project_root))} && "
        if environment:
            worker_command += shlex.join(["env", *environment]) + " "
        worker_command += shlex.join(inner)
        sco = [
            "sco",
            "acp",
            "jobs",
            "create",
            f"--workspace-name={args.workspace}",
            f"--aec2-name={args.aec2_name}",
            f"--job-name=bq-wf2-canary-{vfm}",
            f"--priority={args.priority}",
            f"--container-image-url={args.container_image}",
            f"--storage-mount={args.storage_mount}",
            "--training-framework=pytorch",
            "--worker-nodes=1",
            f"--worker-spec={args.worker_spec}",
            f"--command={worker_command}",
        ]
        spec = {
            "schema_version": 1,
            "status": "READY_FOR_REVIEW_NOT_SUBMITTED",
            "submission_performed": False,
            "vfm": vfm,
            **config,
            "protocol_file": str(protocol),
            "protocol_file_sha256": protocol_sha,
            "extractor": str(extractor),
            "extractor_sha256": extractor_sha,
            "canary_script": str(canary),
            "canary_script_sha256": sha256_file(canary),
            "receipt": str(receipt),
            "worker_command": worker_command,
            "sco_argv": sco,
        }
        atomic_json(specs_dir / f"{vfm}.json", spec)
        specs.append(spec)
        commands.append(shlex.join(sco))

    command_path = output_dir / "sco_create_commands.txt"
    command_path.write_text("\n".join(commands) + "\n")
    post_canary_output = plan_dir.with_name(f"{plan_dir.name}_postcanary")
    post_canary_argv = [
        "env",
        f"SPECS={plan['spec_inventory']['path']}",
        f"CANONICAL_SOURCE={plan['canonical_binding']['canonical_source']['path']}",
        f"NAMESPACE_LOCK={plan['canonical_binding']['namespace_lock']['path']}",
        f"LABEL_AUDIT={plan['canonical_binding']['label_lineage_audit']['path']}",
        f"BACKEND_CANARY_DIR={output_dir}",
        f"OUTPUT_DIR={post_canary_output}",
        "bash",
        str(project_root / "scripts/windowfix_v2/render_sco_plan.sh"),
    ]
    post_canary_path = output_dir / "post_canary_render_command.txt"
    post_canary_path.write_text(
        "# REVIEW ARTIFACT ONLY. Run only after all four canary receipts exist.\n"
        + shlex.join(post_canary_argv)
        + "\n"
    )
    summary = {
        "schema_version": 1,
        "status": "READY_FOR_REVIEW_NOT_SUBMITTED",
        "submission_performed": False,
        "plan_id": plan.get("plan_id"),
        "protocol_file_sha256": protocol_sha,
        "main_extractor_sha256": main_extractor_sha,
        "mllm_extractor_sha256": mllm_extractor_sha,
        "canary_script_sha256": sha256_file(canary),
        "backends": sorted(BACKENDS),
        "job_count": len(specs),
        "commands": str(command_path),
        "post_canary_render_command": str(post_canary_path),
        "post_canary_output": str(post_canary_output),
    }
    atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
