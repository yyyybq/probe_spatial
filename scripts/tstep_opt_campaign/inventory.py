#!/usr/bin/env python3
"""Freeze a recoverable, experiment-level inventory for the seeded-v1 campaigns.

The campaign scope is deliberately defined by the two checked-in job-id files.  No
job-name prefix matching is used here or by the cancellation companion script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import shlex
import subprocess
import sys
from datetime import datetime, timezone

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON = "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
TASKS = (
    "streaming_depth",
    "streaming_mask",
    "view_consistency",
    "ego_belief",
    "ego_belief_v2",
    "action_dynamics",
    "path_integration",
    "counterfactual",
)
CHANNELS = {"wan": 1536, "cogvideox": 3072, "minimax_h3": 5376}
CAMPAIGNS = (
    ROOT / "logs/tstep_probe_campaigns/tstep_probe_v2_sv1_wan_cgx_20260907_zoetrope_r1",
    ROOT / "logs/tstep_probe_campaigns/tstep_probe_v2_sv1_minimax_h3_20260909_zoetrope_r1",
)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_frame_identity(path: pathlib.Path) -> dict[str, object]:
    digest = hashlib.sha256()
    records = 0
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            # Keep the source order.  This intentionally rejects a manifest schema
            # without an explicit raw-frame list rather than guessing an identity.
            frames = None
            for key in ("raw_frame_ids", "frame_ids", "frames", "image_ids"):
                value = item.get(key)
                if isinstance(value, list):
                    frames = value
                    break
            if frames is None:
                # Expanded manifests keep the complete ordered record; hashing it is
                # stronger than path similarity and remains schema-independent.
                payload = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            else:
                payload = json.dumps(frames, ensure_ascii=False, separators=(",", ":"))
            digest.update(payload.encode("utf-8"))
            digest.update(b"\n")
            records += 1
    return {"records": records, "sha256": digest.hexdigest()}


def scheduler_list() -> list[dict]:
    proc = subprocess.run(
        [SCO, "acp", "jobs", "list", "--workspace-name=aigc", "--page-size", "500", "-o", "json"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    value = json.loads(proc.stdout)
    if not isinstance(value, list):
        raise RuntimeError(f"unexpected scheduler response: {type(value).__name__}")
    return value


def read_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def output_evidence(run_dir: pathlib.Path) -> dict[str, object]:
    ckpts = sorted(run_dir.glob("**/*.ckpt")) if run_dir.exists() else []
    marker = run_dir / "TRAIN_COMPLETE"
    hydra_cfgs = sorted(run_dir.glob("**/.hydra/config.yaml")) if run_dir.exists() else []
    return {
        "run_dir": str(run_dir),
        "train_complete": marker.is_file(),
        "train_complete_path": str(marker),
        "checkpoints": [
            {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in ckpts
        ],
        "existing_hydra_configs": [str(p) for p in hydra_cfgs],
    }


def task_overrides(group: dict[str, str], campaign_id: str, task: str) -> list[str]:
    backend = group["vfm"]
    timestep, layer, prefix = group["timestep"], group["layer"], group["prefix"]
    tag = f"tp_{backend}_t{timestep}_l{layer}_p{prefix}_{task}"
    run_folder = f"tstep_probe_seeded_v1/{campaign_id}/{tag}"
    batch = 1 if task in {"streaming_depth", "streaming_mask"} else 2
    values = [
        f"experiment=inscene15k_streaming/{task}_{backend}_v1",
        f"vfm_name={backend}",
        f"video_channels={CHANNELS[backend]}",
        "streaming_feat_root=aoss://baiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_SEEDED_V1",
        f"feature_layer={layer}",
        f"feature_timestep={timestep}",
        f"feat_postfix=_t{timestep}_layer{layer}",
        f"prefix_len={prefix}",
        f"job_name={tag}",
        f"paths.run_folder_name={run_folder}",
        f"logger.wandb.name={campaign_id}_{tag}",
        f"logger.wandb.group={campaign_id}",
        "logger.wandb.offline=true",
        "autoresume=true",
        "trainer.devices=[0]",
        f"data.data_module.batch_size_per_device={batch}",
        f"data.data_module.batch_size_per_device_val={batch}",
        "data.data_module.num_workers=4",
        "data.data_module.num_workers_val=2",
    ]
    if task in {"action_dynamics", "path_integration", "counterfactual"}:
        values.append("target_feat_root=aoss://baiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_SEEDED_V1")
    return values


def set_compose_env(group: dict[str, str]) -> None:
    os.environ.update(
        INSCENE_DATA_ROOT="/mnt/umm/users/yinbaiqiao/InsScene-15K/data",
        INSCENE_LOCAL_SCENE_ROOT="/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2",
        INSCENE_MANIFEST=group["manifest"],
        INSCENE_CANONICAL_PROTOCOL=group["canonical_protocol"],
        INSCENE_CANONICAL_OBS_LEN=group["canonical_obs_len"],
        INSCENE_STREAMING_EXCLUSIONS=str(
            ROOT / "configs/experiment/inscene15k_streaming/tstep_canonical_v2_exclusions.txt"
        ),
        INSCENE_STREAMING_FEAT_ROOT="aoss://baiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_SEEDED_V1",
        INSCENE_TARGET_FEAT_ROOT="aoss://baiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_SEEDED_V1",
        INSCENE_TARGET_INDEX_ROOT="/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_SEEDED_V1",
        INSCENE_TARGET_LAYOUT="flat",
        INSCENE_TRUST_EXPANDED_MANIFEST="1",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--skip-resolved-configs", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    configs_dir = out / "resolved_configs"
    configs_dir.mkdir()

    all_jobs = scheduler_list()
    by_id = {job["name"]: job for job in all_jobs}
    (out / "scheduler_list_raw.json").write_text(json.dumps(all_jobs, indent=2, ensure_ascii=False) + "\n")

    inventory: dict[str, object] = {
        "schema": "tstep-probe-cancel-inventory-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope_rule": "exact membership in the two launcher/sco_job_ids.txt files",
        "campaigns": [],
        "jobs": [],
        "experiments": [],
    }
    manifest_identities: dict[str, dict[str, object]] = {}

    config_ctx = initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs"))
    with config_ctx:
        for campaign_dir in CAMPAIGNS:
            campaign_id = campaign_dir.name
            ids = [x.strip() for x in (campaign_dir / "launcher/sco_job_ids.txt").read_text().splitlines() if x.strip()]
            submit_rows = read_rows(campaign_dir / "launcher/submit_plan.tsv")
            groups = {r["group_id"]: r for r in read_rows(campaign_dir / "matrix/groups.tsv")}
            expected_names = {r["job_name"]: r for r in submit_rows}
            campaign_jobs = []
            for job_id in ids:
                job = by_id.get(job_id)
                if job is None:
                    raise RuntimeError(f"scheduler omitted scoped job {job_id}")
                role = job["roles"][0]
                command = role["startup_script"]
                tokens = shlex.split(command)
                group_tokens = [x for x in tokens if x.startswith("GROUP_ID=")]
                if len(group_tokens) != 1:
                    raise RuntimeError(f"cannot recover GROUP_ID from {job_id}: {command}")
                group_id = group_tokens[0].split("=", 1)[1]
                group = groups[group_id]
                if job["display_name"] not in expected_names:
                    raise RuntimeError(f"job {job_id} display name is outside frozen submit plan")
                job_record = {
                    "job_id": job_id,
                    "campaign_id": campaign_id,
                    "group_id": group_id,
                    "scheduler": job,
                    "submitted_command": command,
                    "submit_plan": expected_names[job["display_name"]],
                    "experiment_keys": [],
                }
                manifest = pathlib.Path(group["manifest"])
                if str(manifest) not in manifest_identities:
                    manifest_identities[str(manifest)] = {
                        "path": str(manifest),
                        "sha256": sha256_file(manifest),
                        "ordered_raw_frame_identity": ordered_frame_identity(manifest),
                    }
                for task in group["tasks"].split(","):
                    if task not in TASKS:
                        raise RuntimeError(f"unexpected task {task!r}")
                    key = ":".join(
                        [group["vfm"], group["source"], group["timestep"], group["layer"], group["prefix"], task, "seed42"]
                    )
                    tag = f"tp_{group['vfm']}_t{group['timestep']}_l{group['layer']}_p{group['prefix']}_{task}"
                    run_dir = ROOT / "logs/inscene15k_streaming/runs/tstep_probe_seeded_v1" / campaign_id / tag
                    overrides = task_overrides(group, campaign_id, task)
                    resolved_path = configs_dir / f"{key.replace(':', '__')}.yaml"
                    if not args.skip_resolved_configs:
                        set_compose_env(group)
                        cfg = compose(config_name="train.yaml", overrides=overrides)
                        # Hydra's ``hydra:runtime.*`` values exist only after the
                        # launcher changes into the run directory.  The composed
                        # config is nevertheless final with every CLI override
                        # applied; preserve those runtime interpolations verbatim.
                        OmegaConf.save(config=cfg, f=resolved_path, resolve=False)
                    experiment = {
                        "unique_key": key,
                        "old_job_id": job_id,
                        "campaign_id": campaign_id,
                        "group_id": group_id,
                        "backend": group["vfm"],
                        "source": group["source"],
                        "timestep": int(group["timestep"]),
                        "layer": int(group["layer"]),
                        "prefix": int(group["prefix"]),
                        "horizon": 124,
                        "probe": task,
                        "seed": 42,
                        "manifest_identity": manifest_identities[str(manifest)],
                        "feature_identity": {
                            "streaming_namespace": group["streaming_namespace"],
                            "target_namespace": group["target_namespace"],
                            "feature_layer": int(group["layer"]),
                            "feature_timestep": int(group["timestep"]),
                            "feature_postfix": f"_t{group['timestep']}_layer{group['layer']}",
                            "strict_audit_path": group["strict_audit"],
                            "strict_audit_sha256": group["strict_audit_sha256"],
                        },
                        "effective_overrides": overrides,
                        "resolved_config": str(resolved_path) if not args.skip_resolved_configs else None,
                        "output_evidence": output_evidence(run_dir),
                        "task_log": str(campaign_dir.parent.parent / "tstep_probe_seeded_v1" / campaign_id / "groups" / group_id / f"{task}.log"),
                        "resume_policy": "autoresume=true; only project-validated same-run last.ckpt resume is allowed",
                    }
                    inventory["experiments"].append(experiment)
                    job_record["experiment_keys"].append(key)
                inventory["jobs"].append(job_record)
                campaign_jobs.append(job_id)
            inventory["campaigns"].append(
                {"campaign_id": campaign_id, "directory": str(campaign_dir), "job_ids": campaign_jobs}
            )

    inventory["manifest_identities"] = list(manifest_identities.values())
    states: dict[str, int] = {}
    for record in inventory["jobs"]:
        state = record["scheduler"]["state"]
        states[state] = states.get(state, 0) + 1
    inventory["state_counts"] = states
    path = out / "inventory.json"
    path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n")
    hashes = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            hashes.append(f"{sha256_file(p)}  {p.relative_to(out)}")
    (out / "SHA256SUMS").write_text("\n".join(hashes) + "\n")
    print(json.dumps({"output": str(out), "state_counts": states, "jobs": len(inventory["jobs"]), "experiments": len(inventory["experiments"])}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
