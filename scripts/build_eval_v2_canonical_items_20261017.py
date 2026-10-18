#!/usr/bin/env python3
"""Build SHA-bound v2 evaluation items for canonical-V2 checkpoints.

For each selected inventory row (metadata/release_inventory/canonical_v2.csv):

  1. find the training runner whose OUTPUT_ROOT holds the row's output and the
     training item (same unique_key) in that runner's ITEM_ROOT;
  2. copy the training item's experiment config, scientific overrides and
     execution manifest, and rebuild the training runner's environment
     (owner-overlay family or direct MiniMax bf16 family; any runner whose
     environment block differs from the known templates is refused);
  3. bind the evaluation pack runtime/main_v2_eval_v2_from_<training pack>
     and the git tag code/<training pack>;
  4. record the checkpoint path, size and mtime (its SHA-256 is computed by the
     runner on the GPU node).

Items: artifacts/main_v2_probe_v2_816_20260917/eval_v2_canonical_20261017_r1/items/
Plan:  .../plan.json (item hashes, selection, counts).  Items are never rewritten
with different content.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "artifacts/main_v2_probe_v2_816_20260917/eval_v2_canonical_20261017_r1"
INVENTORY = ROOT / "metadata/release_inventory/canonical_v2.csv"
EVALUATOR = {
    "streaming_depth": ("a1", "mean_loss_depth"),
    "streaming_mask": ("a1", "mean_loss_identity"),
    "view_consistency": ("diag", "balanced_acc"),
    "ego_belief_v2": ("diag", "mean_ang_err_deg"),
    "action_dynamics": ("diag", "global_R@1"),
    "path_integration": ("diag", "global_R@1"),
}
OVERLAY_TEMPLATE = {
    '"PROBE_OWNER_LOCAL_MIRROR_ROOT": "/mnt/umm/users/yinbaiqiao/InsScene-15K"',
    '"INSCENE_STREAMING_FEAT_ROOT": "aoss://baiqiao/PROBE816_OWNER_OVERLAY_ONLY"',
    '"INSCENE_TARGET_FEAT_ROOT": "aoss://baiqiao/PROBE816_OWNER_OVERLAY_ONLY"',
    '"INSCENE_TARGET_INDEX_ROOT": "/mnt/umm/users/yinbaiqiao/InsScene-15K/PROBE816_OWNER_OVERLAY_ONLY"',
    '"INSCENE_TARGET_LAYOUT": "flat"', '"INSCENE_TRUST_EXPANDED_MANIFEST": "1"',
    '"INSCENE_METADATA_CACHE_MAX_SCENES": "32"', '"PROBE_OWNER_OVERLAY_MANIFEST": str(OVERLAY)',
}
DIRECT_TEMPLATE = {
    '"INSCENE_STREAMING_FEAT_ROOT": STREAM_ROOT', '"INSCENE_TARGET_FEAT_ROOT": TARGET_ROOT',
    '"INSCENE_TARGET_INDEX_ROOT": TARGET_ROOT', '"INSCENE_TARGET_LAYOUT": "flat"',
    '"INSCENE_TRUST_EXPANDED_MANIFEST": "1"', '"INSCENE_METADATA_CACHE_MAX_SCENES": "32"',
}
DATA_ROOT = "/mnt/umm/users/yinbaiqiao/InsScene-15K/data"
SCENE_ROOT = DATA_ROOT + "/processed_scannetpp_v2/processed_scannetpp_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_var(text: str, name: str) -> Path | None:
    absolute = re.search(rf'^{name} = Path\("(/[^"]+)"\)', text, re.M)
    if absolute:
        return Path(absolute.group(1))
    m = re.search(rf'^{name} = (\w+) / "([^"]+)"', text, re.M)
    if not m:
        return None
    base, rel = m.groups()
    if base == "ROOT":
        return ROOT / rel
    parent = resolve_var(text, base)
    return parent / rel if parent else None


def parse_runner(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    out = re.search(r'OUTPUT_ROOT = ROOT / "logs/inscene15k_streaming/runs/([^"]+)"', text)
    pack = re.search(r'^PACK = ROOT / "runtime/([^"]+)"', text, re.M)
    item_root = resolve_var(text, "ITEM_ROOT")
    if not (out and pack and item_root):
        return None
    info = {"runner": path, "campaign": out.group(1), "pack": pack.group(1), "item_root": item_root,
            "fp16": (re.search(r'"PROBE_FP16_PASSTHROUGH": "([01])"', text) or [None, "0"])[1]}
    if all(lit in text for lit in OVERLAY_TEMPLATE):
        overlay = resolve_var(text, "OVERLAY")
        osha = re.search(r'EXPECTED_OVERLAY_SHA256 = "([0-9a-f]{64})"', text)
        if not (overlay and osha):
            return None
        info.update(family="overlay", overlay=overlay, overlay_sha256=osha.group(1))
    elif all(lit in text for lit in DIRECT_TEMPLATE):
        info.update(family="direct",
                    stream_root=re.search(r'^STREAM_ROOT = "([^"]+)"', text, re.M).group(1),
                    target_root=re.search(r'^TARGET_ROOT = "([^"]+)"', text, re.M).group(1))
    else:
        info["family"] = "UNKNOWN_TEMPLATE"
    return info


def campaign_of(output_path: str) -> str:
    m = re.match(r"(?:\./)?logs/inscene15k_streaming/runs/([^/]+)", output_path or "")
    return m.group(1) if m else ""


def ledger_runners() -> dict[tuple[str, str], set[str]]:
    """(unique_key, run_id) -> runner paths that training submission ledgers launched.

    A retry runner can share its Hydra output folder with the original campaign
    while binding a different overlay (e.g. main_v2_probe_v2_vm24_targetfix_r1),
    so the run id of the checkpoint, not the output folder, decides the runner.
    """
    found: dict[tuple[str, str], set[str]] = defaultdict(set)
    for path in glob(str(ROOT / "artifacts/main_v2_probe_v2_816_20260917/*/submissions.json")):
        try:
            jobs = json.loads(Path(path).read_text()).get("jobs", {})
        except Exception:  # noqa: BLE001 - unreadable ledgers just give no evidence
            continue
        for name, job in (jobs.items() if isinstance(jobs, dict) else enumerate(jobs)):
            if not isinstance(job, dict):
                continue
            command = str(job.get("startup_command") or job.get("command") or "")
            run_id = re.search(r"--run-id\s+(\S+)", command)
            runner = re.search(r"(\S+/run_single\.py)", command)
            key = job.get("unique_key") or (name if isinstance(name, str) else None)
            if run_id and runner and key:
                found[(key, run_id.group(1))].add(runner.group(1))
    return found


def load_receipts(path: Path | None) -> dict[str, dict]:
    """<mode>/<run_id>/<key> -> TRAIN_COMPLETE fields, from dump_training_receipts.py output."""
    if path is None:
        return {}
    receipts = {}
    for line in path.read_text().splitlines():
        row = json.loads(line) if line.strip() else {}
        if str(row.get("path", "")).endswith("/TRAIN_COMPLETE.json") and "fields" in row:
            tail = "/".join(Path(row["path"]).parent.parts[-3:])
            receipts.setdefault(tail, []).append({**row["fields"], "_path": row["path"]})
    return {tail: rows[0] for tail, rows in receipts.items() if len(rows) == 1}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keys", nargs="*", default=[], help="explicit unique keys")
    parser.add_argument("--all-verified", action="store_true", help="every VERIFIED row whose weights are not known non-finite")
    parser.add_argument("--write", action="store_true", help="write items and plan (default: report only)")
    parser.add_argument("--revision", choices=("r1", "r2", "r3"), default="r1",
                        help="r1: campaign/packs main_v2_eval_v2_from_*; r2: patched C-probe packs "
                             "main_v2_eval_v2r2_from_*; r3: r2 packs, runner chosen by the checkpoint's "
                             "run id and checked against its training receipt")
    parser.add_argument("--receipts", type=Path, default=None,
                        help="training_receipts.jsonl from dump_training_receipts.py; required for r3")
    parser.add_argument("--campaign-dir", type=Path, default=None,
                        help="write items and plan here instead of the revision's campaign folder (dry comparisons)")
    args = parser.parse_args()
    global CAMPAIGN
    CAMPAIGN = args.campaign_dir or ROOT / f"artifacts/main_v2_probe_v2_816_20260917/eval_v2_canonical_20261017_{args.revision}"
    pack_glob = "runtime/main_v2_eval_v2_from_*" if args.revision == "r1" else "runtime/main_v2_eval_v2r2_from_*"
    if args.revision == "r3" and args.receipts is None:
        raise SystemExit("--revision r3 needs --receipts")
    ledger = ledger_runners() if args.revision == "r3" else {}
    receipts = load_receipts(args.receipts)

    runners = [r for r in (parse_runner(Path(p)) for p in
               glob(str(ROOT / "artifacts/**/run_single.py"), recursive=True) + glob(str(ROOT / "runtime/*/run_single.py"))) if r]
    by_campaign = defaultdict(list)
    for r in runners:
        by_campaign[r["campaign"]].append(r)
    eval_packs = {}
    for d in glob(str(ROOT / pack_glob)):
        manifest = json.loads(Path(d, "EVAL_PACK.json").read_text())
        eval_packs[manifest["base_pack"].split("/", 1)[1]] = {
            "name": Path(d).name, "source_sha256_file_sha256": sha256(Path(d, "source.sha256"))}

    rows = list(csv.DictReader(INVENTORY.open(newline="", encoding="utf-8")))
    if args.all_verified:
        selected = [r for r in rows if r["correctness_status"] == "VERIFIED"
                    and not r["checkpoint_tensor_status"].startswith("NONFINITE")]
    else:
        wanted = set(args.keys)
        selected = [r for r in rows if r["experiment_id"] in wanted]
        missing = wanted - {r["experiment_id"] for r in selected}
        if missing:
            raise SystemExit(f"keys not in inventory: {sorted(missing)}")

    items, problems = [], Counter()
    for row in selected:
        key = row["experiment_id"]
        cands = by_campaign.get(campaign_of(row["output_path"]), [])
        run_parts = Path(row["checkpoint_path"]).parts
        run_tail = ("/".join(run_parts[run_parts.index("VideoProbe3D") - 3:run_parts.index("VideoProbe3D")])
                    if "VideoProbe3D" in run_parts else "")
        run_id = run_tail.split("/")[1] if run_tail else ""
        named = ledger.get((key, run_id))
        if named:
            cands = [r for r in (parse_runner(Path(p)) for p in sorted(named)) if r]
        found = []
        for r in cands:
            for f in glob(str(r["item_root"] / "*.jsonl")):
                try:
                    it = json.loads(Path(f).read_text().strip().splitlines()[0])
                except Exception:  # noqa: BLE001 - malformed item files are skipped
                    continue
                if it.get("unique_key") == key:
                    found.append((r, Path(f), it))
        if len(found) != 1:
            problems[f"training_item_{'missing' if not found else 'ambiguous'}"] += 1
            continue
        runner, item_path, titem = found[0]
        if runner["family"] not in ("overlay", "direct"):
            problems["unknown_runner_template"] += 1
            continue
        if runner["pack"] not in eval_packs:
            problems[f"no_eval_pack_for_{runner['pack']}"] += 1
            continue
        if row["probe"] not in EVALUATOR:
            problems[f"no_evaluator_for_{row['probe']}"] += 1
            continue
        ckpt = Path(row["checkpoint_path"])
        ckpt = ckpt if ckpt.is_absolute() else ROOT / ckpt
        if not ckpt.is_file():
            problems["checkpoint_missing"] += 1
            continue
        receipt = None
        if args.revision == "r3":
            receipt = receipts.get(run_tail)
            if receipt is None:
                problems["training_receipt_missing"] += 1
                continue
            mismatched = [label for label, ok in (
                ("state", receipt.get("state") == "SUCCEEDED"),
                ("item", receipt.get("item_sha256") == sha256(item_path)),
                ("pack", receipt.get("source_sha_file_sha256") == sha256(ROOT / "runtime" / runner["pack"] / "source.sha256")),
                ("overlay", runner["family"] != "overlay" or receipt.get("overlay_sha256") == runner["overlay_sha256"]),
            ) if not ok]
            if mismatched:
                problems["training_receipt_mismatch_" + "_".join(mismatched)] += 1
                continue
        manifest = titem["execution_manifest"]
        env = {
            "INSCENE_DATA_ROOT": DATA_ROOT, "INSCENE_LOCAL_SCENE_ROOT": SCENE_ROOT,
            "INSCENE_MANIFEST": manifest["path"], "INSCENE_MANIFEST_SHA256": manifest["sha256"],
            "INSCENE_TARGET_LAYOUT": "flat", "INSCENE_TRUST_EXPANDED_MANIFEST": "1",
            "INSCENE_PROBE_TASK": titem["probe"], "INSCENE_METADATA_CACHE_MAX_SCENES": "32",
            "PROBE_FP16_PASSTHROUGH": runner["fp16"],
        }
        if runner["family"] == "overlay":
            env.update({
                "PROBE_OWNER_OVERLAY_MANIFEST": str(runner["overlay"]),
                "PROBE_OWNER_LOCAL_MIRROR_ROOT": "/mnt/umm/users/yinbaiqiao/InsScene-15K",
                "INSCENE_STREAMING_FEAT_ROOT": "aoss://baiqiao/PROBE816_OWNER_OVERLAY_ONLY",
                "INSCENE_TARGET_FEAT_ROOT": "aoss://baiqiao/PROBE816_OWNER_OVERLAY_ONLY",
                "INSCENE_TARGET_INDEX_ROOT": "/mnt/umm/users/yinbaiqiao/InsScene-15K/PROBE816_OWNER_OVERLAY_ONLY",
            })
        else:
            env.update({"INSCENE_STREAMING_FEAT_ROOT": runner["stream_root"],
                        "INSCENE_TARGET_FEAT_ROOT": runner["target_root"],
                        "INSCENE_TARGET_INDEX_ROOT": runner["target_root"]})
        evaluator, metric = EVALUATOR[row["probe"]]
        st = ckpt.stat()
        items.append({
            "schema": "MAIN_V2_EVAL_V2_ITEM_V1", "unique_key": key, "model": row["model"], "probe": row["probe"],
            "layer": row["layer"], "prefix": row["prefix"], "scope": row["scope"],
            "inventory_correctness_status": row["correctness_status"],
            "inventory_checkpoint_tensor_status": row["checkpoint_tensor_status"],
            "evaluator": evaluator, "primary_metric": metric,
            "experiment_config": titem["experiment_config"],
            "experiment_config_sha256": titem["experiment_config_sha256"],
            "scientific_overrides": titem["scientific_overrides"],
            "execution_manifest": {"path": manifest["path"], "sha256": manifest["sha256"]},
            "overlay": ({"path": str(runner["overlay"]), "sha256": runner["overlay_sha256"]}
                        if runner["family"] == "overlay" else None),
            "env_family": runner["family"], "env": env,
            "training_item": {"path": str(item_path), "sha256": sha256(item_path)},
            "training_runner": {"path": str(runner["runner"]), "sha256": sha256(runner["runner"])},
            "training_pack": runner["pack"], "training_code_tag": f"code/{runner['pack']}",
            "eval_pack": eval_packs[runner["pack"]],
            "checkpoint": {"path": str(ckpt), "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns},
            **({"training_receipt": {k: receipt.get(k) for k in (
                "_path", "state", "overlay_sha256", "item_sha256", "source_sha_file_sha256", "completed_utc")}}
               if receipt else {}),
        })
    report = {"selected_rows": len(selected), "items": len(items), "problems": dict(problems),
              "by_probe": dict(Counter(i["probe"] for i in items)),
              "by_family": dict(Counter(i["env_family"] for i in items)),
              "by_eval_pack": dict(Counter(i["eval_pack"]["name"] for i in items))}
    print(json.dumps(report, indent=1, sort_keys=True))
    if not args.write:
        return 0
    item_dir = CAMPAIGN / "items"
    item_dir.mkdir(parents=True, exist_ok=True)
    plan_path = CAMPAIGN / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {"schema": "MAIN_V2_EVAL_V2_PLAN_V1", "items": {}}
    for item in items:
        key = item["unique_key"]
        text = json.dumps(item, sort_keys=True) + "\n"
        existing = plan["items"].get(key)
        path = Path(existing["path"]) if existing else item_dir / f"{len(plan['items']):04d}_{key}.jsonl"
        if path.exists():
            if path.read_text() != text:
                raise SystemExit(f"{path} exists with different content; items are immutable")
        else:
            path.write_text(text, encoding="utf-8")
            os.chmod(path, 0o444)
        plan["items"][key] = {"path": str(path), "sha256": sha256(path), "probe": item["probe"],
                              "model": item["model"], "eval_pack": item["eval_pack"]["name"]}
    plan["updated_utc"] = datetime.now(timezone.utc).isoformat()
    plan["runner"] = str(CAMPAIGN / "run_eval_single.py")
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(items)} items; plan has {len(plan['items'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
