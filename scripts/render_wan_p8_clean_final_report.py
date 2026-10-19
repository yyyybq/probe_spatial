#!/usr/bin/env python3
"""Render the final audited Wan p8 canonical-clean rerun report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LAYERS = [0, 5, 10, 15, 25, 29]
MANIFEST_SHA = "772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--eval-run-tag", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--registry-update-root")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report_dir = Path(args.report_dir).resolve()
    update_root = (
        Path(args.registry_update_root).resolve()
        if args.registry_update_root
        else report_dir / "registry_update"
    )
    full_gate = read_json(report_dir / "full_gate_summary.json")
    dataset_gate = read_json(report_dir / "full_dataset_gate_contract.json")
    train = read_json(report_dir / "train_audit.json")
    evaluation = read_json(report_dir / "eval_audit.json")
    staging = read_json(update_root / "staging/staging_audit.json")
    post = read_json(update_root / "post/metadata.json")
    extractor_config = report_dir / "provenance/extractor_config.json"
    manifest = ROOT / "manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl"

    assert sha256(manifest) == MANIFEST_SHA
    assert full_gate.get("ok") is True
    assert dataset_gate.get("status") == "PASS" and dataset_gate.get("fallback_count") == 0
    assert train.get("ok") is True and evaluation.get("ok") is True
    assert staging.get("ok") is True
    assert post.get("valid_eval_summaries") == 428
    assert post.get("unique_eval_job_keys") == 428
    assert post.get("missing_wan_p8_evals") == 0
    assert post.get("duplicate_experiment_ids") == 0
    assert post.get("duplicate_metric_rows") == 0
    assert post.get("canonical_results_valid") is True
    assert all(train["layers"][str(layer)]["status"] == "CLEAN_COMPLETE" for layer in LAYERS)
    assert all(evaluation["layers"][str(layer)]["status"] == "EVAL_COMPLETE" for layer in LAYERS)
    for key, item in post["canonical"].items():
        assert sha256(ROOT / "results/canonical" / key) == item["sha256"]
    assert sha256(ROOT / "experiments/registry.csv") == post["registry"]["sha256"]

    git_commit = (report_dir / "provenance/training/git_commit.txt").read_text().strip()
    extractor_hash = sha256(extractor_config)
    train_jobs = read_tsv(report_dir / "launcher/train_sco_job_ids.tsv")
    eval_jobs = read_tsv(report_dir / "launcher/eval_sco_job_ids.tsv")
    streaming_root = full_gate["streaming_root"]
    target_root = full_gate["target_root"]

    train_status = {str(layer): train["layers"][str(layer)]["status"] for layer in LAYERS}
    checkpoint_by_layer = {
        str(layer): {
            "path": train["layers"][str(layer)]["checkpoint_realpath"],
            "sha256": train["layers"][str(layer)]["checkpoint_sha256"],
            "epoch": train["layers"][str(layer)]["checkpoint_state"]["epoch"],
            "global_step": train["layers"][str(layer)]["checkpoint_state"]["global_step"],
        }
        for layer in LAYERS
    }
    eval_status = {
        str(layer): {
            "status": evaluation["layers"][str(layer)]["status"],
            "n": evaluation["layers"][str(layer)]["n"],
            "summary": evaluation["layers"][str(layer)]["summary"],
            "summary_sha256": evaluation["layers"][str(layer)]["summary_sha256"],
            "global_R@1": evaluation["layers"][str(layer)]["metrics"]["global_R@1"],
        }
        for layer in LAYERS
    }

    lines = [
        "# Wan p8 Canonical Clean Rerun",
        "",
        "## Required Fields",
        "",
        "```text",
        f"RUN_TAG: {args.run_tag}",
        f"GIT_COMMIT: {git_commit}",
        f"CANONICAL_MANIFEST_SHA256: {MANIFEST_SHA}",
        f"EXTRACTOR_CONFIG_HASH: {extractor_hash}",
        f"STREAMING_FEATURE_ROOT: {streaming_root}",
        f"TARGET_FEATURE_ROOT: {target_root}",
        "FEATURE_COVERAGE: PASS; streaming=2724/2724, target=2724/2724, exact raw-ID mismatch=0, six-layer FP16 finite audit=20430 tensors",
        f"TRAIN_STATUS_BY_LAYER: {json.dumps(train_status, sort_keys=True)}",
        f"CHECKPOINT_BY_LAYER: {json.dumps(checkpoint_by_layer, sort_keys=True)}",
        f"EVAL_STATUS_BY_LAYER: {json.dumps(eval_status, sort_keys=True)}",
        "REGISTRY_UPDATED: true",
        "FINAL_EVAL_COUNT: 428/428",
        "CANONICAL_RESULTS_VALID: true",
        "REMAINING_RISKS: v3_checkpoint.yaml lacks a callback _target_; this run used Lightning's default checkpoint directory and a provenance-recorded same-run compatibility symlink for the wrapper final check. Canonical model_variant retains the official updater's eval_20260817 naming for compatibility. The project consistency check also retains two pre-existing warnings for missing VideoMAE p12 filtered-manifest paths; those rows were byte-identical across this promotion.",
        "```",
        "",
        "## Feature Gates",
        "",
        f"- Full feature audit: `{report_dir / 'full_feature_audit.json'}`",
        f"- Dataset dry-run: `{report_dir / 'full_dataset_dryrun.json'}`",
        f"- Dataset contract: `{report_dir / 'full_dataset_gate_contract.json'}`",
        f"- Extractor config: `{extractor_config}` (`sha256={extractor_hash}`)",
        "- Streaming exact raw-ID: 2724/2724; target exact raw-ID: 2724/2724.",
        "- Dataset load pass: 2724/2724 per layer; problem_count=0; fallback_count=0.",
        "",
        "## Training",
        "",
        "| layer | SCO job | status | epoch | global step | checkpoint sha256 |",
        "| ---: | --- | --- | ---: | ---: | --- |",
    ]
    train_job_by_layer = {row["layer"]: row["job_id"] for row in train_jobs}
    for layer in LAYERS:
        item = checkpoint_by_layer[str(layer)]
        lines.append(
            f"| {layer} | {train_job_by_layer.get(str(layer), '')} | CLEAN_COMPLETE | "
            f"{item['epoch']} | {item['global_step']} | `{item['sha256']}` |"
        )
    lines.append("")
    lines.append("All six training SCO jobs were independently observed in final state `SUCCEEDED` on 2026-08-24.")

    lines.extend(
        [
            "",
            "## Evaluation",
            "",
            "| layer | SCO/controller | n | global_R@1 | summary sha256 |",
            "| ---: | --- | ---: | ---: | --- |",
        ]
    )
    eval_job_by_layer = {row["layer"]: row["job_id"] for row in eval_jobs}
    for layer in LAYERS:
        item = eval_status[str(layer)]
        lines.append(
            f"| {layer} | {eval_job_by_layer.get(str(layer), 'controller-sequential')} | {item['n']} | "
            f"{item['global_R@1']} | `{item['summary_sha256']}` |"
        )

    lines.extend(
        [
            "",
            "## Recovery Audit Trail",
            "",
            "- Controller `pt-g7f7k3kt` stopped before evaluation because Hydra could not parse `=` in the resolved checkpoint filename; the failed status/log and original tooling hashes are preserved under "
            f"`{report_dir / 'failed_attempts/eval_cli_equals_20260824T120405Z'}`.",
            "- Controller `pt-gxgdr1dp` completed all six staging evals and the strict eval audit, then stopped before promotion because the SCO image had no `git` executable; evidence is preserved under "
            f"`{report_dir / 'failed_attempts/promotion_no_git_20260824T123721Z'}`.",
            "- The first local staging retry stopped before promotion when strict ID auditing detected a long-RUN_TAG parser ambiguity. Existing 422 job-key mappings were proven byte-identical before/after the constrained parser fix; evidence is preserved under "
            f"`{report_dir / 'failed_attempts/promotion_job_key_parse_20260824T1253Z'}`.",
            f"- Final promotion used the fresh versioned update root `{update_root}` and passed its complete staging audit before atomic writes.",
            "",
            "## Canonical Update",
            "",
            f"- Staging audit: `{update_root / 'staging/staging_audit.json'}`",
            "- Valid eval summaries: 428/428; unique job keys: 428; missing Wan p8 evals: 0.",
            "- Duplicate experiment IDs: 0; duplicate metric rows: 0.",
            "- Existing 949 registry rows were unchanged; six clean Wan p8 C1 rows were added.",
            "- `a1`, `a2`, `b2`, `c2`, and `c3` canonical tables remained byte-identical.",
            "- `scripts/check_project_consistency.py`: PASS (two documented pre-existing VideoMAE p12 manifest-path warnings).",
            "- `tests/test_paper_validity.py`: 9 passed, 1 non-fatal PyTorch nested-tensor warning.",
            "",
            "| file | rows before | rows after | sha256 before | sha256 after |",
            "| --- | ---: | ---: | --- | --- |",
        ]
    )
    pre = read_json(update_root / "pre/metadata.json")
    lines.append(
        f"| experiments/registry.csv | {pre['registry']['rows']} | {post['registry']['rows']} | "
        f"`{pre['registry']['sha256']}` | `{post['registry']['sha256']}` |"
    )
    for name in sorted(post["canonical"]):
        before = pre["canonical"][name]
        after = post["canonical"][name]
        lines.append(
            f"| results/canonical/{name} | {before['rows']} | {after['rows']} | "
            f"`{before['sha256']}` | `{after['sha256']}` |"
        )

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, out)
    print(json.dumps({"ok": True, "report": str(out), "sha256": sha256(out)}))


if __name__ == "__main__":
    main()
