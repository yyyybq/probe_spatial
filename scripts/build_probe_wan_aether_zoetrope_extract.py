#!/usr/bin/env python3
"""Render zoetrope gap jobs for official Wan/Aether hole worklists.

Each VFM is one SCO job (the gap runner holds an exclusive per-VFM lock).
The job packs ``--processes-per-gpu`` extract processes onto each visible GPU
and runs ``run_tstep_sync_loop.sh`` for extract-upload-delete.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.canonical_gap_worklist import canonical_json_sha, sha256_file
from scripts.windowfix_v2.prepare_sco_gap_jobs import (
    backend_dependency_bindings,
    environment_for,
)
CAMPAIGN = ROOT / "artifacts/wan_aether_streaming_official_r1"
WORKLISTS = {
    "wan": CAMPAIGN / "extract_worklists/wan_official_key_holes.worklist.json",
    "aether": CAMPAIGN / "extract_worklists/aether_official_key_holes.worklist.json",
}
PROTOCOL = ROOT / "protocols/inscene_streaming_canonical_v2.json"
SOURCE = ROOT / "manifests/c_retrain_20260722/inscene_c_h124_prefix24.jsonl"
PYTHON = "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"


def binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-id", required=True)
    parser.add_argument("--worker-spec", default="N4lS.Iq.I80.8")
    parser.add_argument("--n-cards", type=int, default=8)
    parser.add_argument("--processes-per-gpu", type=int, default=2)
    parser.add_argument("--job-name-suffix", default="8g-0921")
    args = parser.parse_args()
    if args.n_cards < 1 or args.processes_per_gpu < 1:
        raise SystemExit("n-cards and processes-per-gpu must be positive")
    return args


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    n_processes = args.n_cards * args.processes_per_gpu
    cuda_visible = ",".join(str(index) for index in range(args.n_cards))
    protocol_sha = sha256_file(PROTOCOL)
    source_sha = sha256_file(SOURCE)
    audit_worklists = []
    loaded = {}
    for vfm, path in WORKLISTS.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        loaded[vfm] = payload
        audit_worklists.append(
            {
                "content_sha256": payload["content_sha256"],
                "counts": payload["counts"],
                "file_sha256": sha256_file(path),
                "path": str(path.resolve()),
                "vfm": vfm,
            }
        )
    audit = {
        "audit_type": "CANONICAL_V2_OFFICIAL_KEY_HOLE_AUDIT_V1",
        "canonical_protocol": "inscene_streaming_canonical_v2_obs28_h124",
        "canonical_source": {"path": str(SOURCE), "sha256": source_sha},
        "counts": {
            "gap_worklists": len(audit_worklists),
            "hard_errors": 0,
            "layer_items": sum(int(item["counts"]["layer_items"]) for item in audit_worklists),
        },
        "hard_errors": [],
        "notes": [
            "Positive allow-list of official-key gate FileNotFound holes.",
            "Disjoint from frozen 816 request SHA 5e0854c11f5141780ae797da3f1368cffec270a42b9ec7ab2d4cf31908597cd5.",
        ],
        "outputs": {"gap_worklists": audit_worklists},
        "protocol_file_sha256": protocol_sha,
        "status": "INCOMPLETE",
    }
    audit["content_sha256"] = canonical_json_sha(audit)
    audit_path = output / "official_key_hole_audit.json"
    write_json(audit_path, audit)
    audit_sha = sha256_file(audit_path)

    plan_bindings = {
        "runner": binding(ROOT / "scripts/windowfix_v2/run_canonical_gap_shard.py"),
        "record_filter": binding(ROOT / "features/canonical_gap_worklist.py"),
        "aoss_sync_loop": binding(ROOT / "scripts/run_tstep_sync_loop.sh"),
        "aoss_gap_adapter": binding(ROOT / "scripts/windowfix_v2/canonical_gap_aoss_adapter.py"),
    }
    jobs = []
    for number, vfm in enumerate(("aether", "wan"), 1):
        worklist_path = WORKLISTS[vfm]
        worklist = loaded[vfm]
        worklist_file_sha = sha256_file(worklist_path)
        shard_id = f"wa-hole-{number:03d}-{vfm}-749-streaming_prefix"
        env = environment_for(vfm, ROOT)
        env["FEATURE_NAMESPACE"] = "FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1"
        shard = {
            "backend_canary_binding": None,
            "canonical_protocol": worklist["canonical_protocol"],
            "canonical_source_path": str(SOURCE),
            "canonical_source_sha256": source_sha,
            "contract": "CANONICAL_V2_GAP_WORKLIST_V1",
            "counts": worklist["counts"],
            "dataset_data_root": "/mnt/umm/users/yinbaiqiao/InsScene-15K/data",
            "environment": env,
            "feature_namespace": worklist["feature_namespace"],
            "feature_root": "/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1",
            "layers": worklist["output_layers"],
            "mode": worklist["mode"],
            "model_id": worklist["model_id"],
            "n_gpus": n_processes,
            "plan_id": args.plan_id,
            "preprocess_identity": worklist["preprocess_identity"],
            "project_root": str(ROOT),
            "prompt_contract": worklist["prompt_contract"],
            "protocol_file_sha256": protocol_sha,
            "protocol_path": str(PROTOCOL),
            "protocol_sha256": worklist["protocol_sha256"],
            "python": PYTHON,
            "runtime_prompt": (
                worklist["prompt_contract"]["value"]
                if worklist["prompt_contract"]["mode"] == "exact"
                else ""
            ),
            "scene_end": 682,
            "scene_start": 0,
            "schema_version": 1,
            "shard_id": shard_id,
            "source": "scannetpp",
            "source_audit_content_sha256": audit["content_sha256"],
            "source_audit_path": str(audit_path.resolve()),
            "source_audit_sha256": audit_sha,
            "source_bindings": {
                **plan_bindings,
                "extractor": binding(ROOT / "features/run_inscene15k.py"),
                "backend_dependencies": backend_dependency_bindings(vfm, ROOT),
                "model_configs": {},
            },
            "status": "READY",
            "timestep": worklist["timestep"],
            "vfm": vfm,
            "worklist_content_sha256": worklist["content_sha256"],
            "worklist_file_sha256": worklist_file_sha,
            "worklist_path": str(worklist_path.resolve()),
        }
        shard["content_sha256"] = canonical_json_sha(shard)
        shard_path = output / "shards" / f"{shard_id}.json"
        write_json(shard_path, shard)
        batch = {
            "batch_id": f"wa-hole-batch-{vfm}",
            "plan_id": args.plan_id,
            "schema_version": 1,
            "serialization_reason": (
                f"one {args.n_cards}-GPU SCO job per VFM; "
                f"{args.processes_per_gpu} extract processes per card"
            ),
            "serialization_scope": f"streaming_prefix only for {vfm}",
            "shards": [
                {
                    "mode": "streaming_prefix",
                    "path": str(shard_path.resolve()),
                    "sha256": sha256_file(shard_path),
                    "shard_id": shard_id,
                    "timestep": 749,
                }
            ],
            "status": "READY",
            "vfm": vfm,
        }
        batch["content_sha256"] = canonical_json_sha(batch)
        batch_path = output / "batches" / f"wa-hole-batch-{vfm}.json"
        write_json(batch_path, batch)
        log_dir = output / "logs"
        script_path = output / "scripts" / f"wa-hole-batch-{vfm}.sh"
        command = (
            "set -o pipefail; "
            f"mkdir -p {log_dir}; "
            f"(cd {ROOT} && CUDA_VISIBLE_DEVICES={cuda_visible} {PYTHON} "
            f"{ROOT}/scripts/windowfix_v2/run_canonical_gap_shard.py "
            f"--batch {batch_path} --execute) "
            f"> {log_dir}/wa-hole-batch-{vfm}.log 2>&1; "
            "gap_rc=$?; "
            f"chmod 0644 {log_dir}/wa-hole-batch-{vfm}.log || true; "
            "exit ${gap_rc}"
        )
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("#!/usr/bin/env bash\n" + command + "\n", encoding="utf-8")
        script_path.chmod(0o755)
        job_name = f"bq-wa-hole-{vfm[:2]}-{args.job_name_suffix}"
        jobs.append(
            {
                "vfm": vfm,
                "job_name": job_name,
                "batch": str(batch_path),
                "shard": str(shard_path),
                "script": str(script_path),
                "n_cards": args.n_cards,
                "n_gpus_processes": n_processes,
                "processes_per_gpu": args.processes_per_gpu,
                "layer_items": worklist["counts"]["layer_items"],
            }
        )

    submit = output / "submit_zoetrope.sh"
    submit.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
export PATH="${{HOME}}/.sco/bin:${{PATH}}"
ROOT={ROOT}
PLAN={output}
LAUNCH=${{PLAN}}/launcher
mkdir -p "${{LAUNCH}}"

submit_one() {{
  local vfm="$1"
  local job_name="$2"
  local script="$3"
  local command="cd ${{ROOT}} && bash ${{script}}"
  printf '%s\\n' "${{command}}" > "${{LAUNCH}}/${{vfm}}.command.txt"
  printf '%s\\n' "${{job_name}}" > "${{LAUNCH}}/${{vfm}}.job_name.txt"
  sco acp jobs create \\
    --workspace-name=aigc \\
    --aec2-name=zoetrope \\
    --job-name="${{job_name}}" \\
    --priority=HIGHEST \\
    --quota-type=reserved \\
    --container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617 \\
    --storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm \\
    --training-framework=pytorch \\
    --worker-nodes=1 \\
    --worker-spec={args.worker_spec} \\
    --command="${{command}}" | tee "${{LAUNCH}}/${{vfm}}.sco_submit.log"
  grep -oE 'pt-[a-z0-9]+' "${{LAUNCH}}/${{vfm}}.sco_submit.log" | awk '!seen[$0]++' > "${{LAUNCH}}/${{vfm}}.job_ids.txt"
  local job_id
  job_id=$(head -1 "${{LAUNCH}}/${{vfm}}.job_ids.txt")
  test -n "${{job_id}}"
  printf '%s %s %s\\n' "${{vfm}}" "${{job_name}}" "${{job_id}}"
}}

"""
        + "\n".join(
            f'submit_one {job["vfm"]} {job["job_name"]} {job["script"]}'
            for job in jobs
        )
        + "\n",
        encoding="utf-8",
    )
    submit.chmod(0o755)
    write_json(
        output / "plan.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "plan_id": args.plan_id,
            "worker_spec": args.worker_spec,
            "aec2_name": "zoetrope",
            "n_cards": args.n_cards,
            "processes_per_gpu": args.processes_per_gpu,
            "n_extract_processes": n_processes,
            "cuda_visible_devices": cuda_visible,
            "sync_loop": str(ROOT / "scripts/run_tstep_sync_loop.sh"),
            "jobs": jobs,
            "audit": {"path": str(audit_path), "sha256": audit_sha},
        },
    )
    print(json.dumps({"output": str(output), "jobs": jobs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
