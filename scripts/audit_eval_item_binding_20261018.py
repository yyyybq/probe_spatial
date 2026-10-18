#!/usr/bin/env python3
"""Check that every canonical-V2 evaluation item is bound to what its checkpoint was trained with.

Ground truth is the training receipt of the run that wrote the checkpoint
(TRAIN_COMPLETE.json in the run directory, else a completed/<key>.json state file
whose output_dir is that directory).  Receipts are root-only, so they are read
from a dump made on a job node (artifacts/release_audit_20261018/overlay_binding_audit/
training_receipts.jsonl, written by dump_training_receipts.py).

Per item the receipt must agree with the evaluation item on
  overlay_sha256            vs item.overlay.sha256
  item_sha256               vs item.training_item.sha256
  source_sha_file_sha256    vs sha256(runtime/<item.training_pack>/source.sha256)
and the receipt must say SUCCEEDED.  The training runner is identified from the
campaign submission ledger whose job for this key used the checkpoint's run id.

Writes <audit dir>/BINDING_AUDIT.json and .tsv; prints counts.  Read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from glob import glob
from pathlib import Path

ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
AUDIT = ROOT / "artifacts/release_audit_20261018/overlay_binding_audit"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_dir_of(ckpt: str) -> Path:
    path = Path(ckpt)
    parts = path.parts
    if "VideoProbe3D" not in parts:
        raise ValueError(f"unexpected checkpoint layout: {ckpt}")
    return Path(*parts[: parts.index("VideoProbe3D")])


def ledger_runners() -> dict[tuple[str, str], list[str]]:
    """(unique_key, run_id) -> runner paths named in submission ledgers."""
    found: dict[tuple[str, str], list[str]] = {}
    for path in glob(str(ROOT / "artifacts/main_v2_probe_v2_816_20260917/*/submissions.json")):
        try:
            jobs = json.loads(Path(path).read_text()).get("jobs", {})
        except Exception:  # noqa: BLE001 - unreadable ledgers are reported by absence
            continue
        for name, job in (jobs.items() if isinstance(jobs, dict) else enumerate(jobs)):
            if not isinstance(job, dict):
                continue
            command = str(job.get("startup_command") or job.get("command") or "")
            run_id = re.search(r"--run-id\s+(\S+)", command)
            runner = re.search(r"(\S+/run_single\.py)", command)
            key = job.get("unique_key") or (name if isinstance(name, str) else None)
            if run_id and runner and key:
                found.setdefault((key, run_id.group(1)), []).append(runner.group(1))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--revision", default="r2")
    args = parser.parse_args()
    campaign = ROOT / f"artifacts/main_v2_probe_v2_816_20260917/eval_v2_canonical_20261017_{args.revision}"
    plan = json.loads((campaign / "plan.json").read_text())["items"]
    receipts = [json.loads(line) for line in (AUDIT / "training_receipts.jsonl").read_text().splitlines() if line]
    by_complete = {str(Path(r["path"]).parent): r["fields"] for r in receipts
                   if r["path"].endswith("/TRAIN_COMPLETE.json") and "fields" in r}
    by_state = {}
    for r in receipts:
        if "/completed/" in r["path"] and "fields" in r:
            by_state.setdefault(str(r["fields"].get("output_dir", "")), []).append(r["fields"])
    runners = ledger_runners()
    pack_sha = {}
    rows, counts = [], Counter()
    for key, entry in sorted(plan.items()):
        item = json.loads(Path(entry["path"]).read_text().splitlines()[0])
        run_dir = run_dir_of(item["checkpoint"]["path"])
        receipt = by_complete.get(str(run_dir))
        source = "TRAIN_COMPLETE"
        if receipt is None and by_state.get(str(run_dir)):
            receipt, source = by_state[str(run_dir)][-1], "state/completed"
        if receipt is None:
            # Some retry runners write receipts under their own OUTPUT_ROOT but send
            # Hydra output (checkpoints) to another campaign folder; match the
            # same <mode>/<run_id>/<key> tail instead of the full directory.
            tail = "/".join(run_dir.parts[-3:])
            same = [f for d, f in by_complete.items() if d.endswith("/" + tail)]
            if len(same) == 1:
                receipt, source = same[0], "TRAIN_COMPLETE (same run id and key, other folder)"
        pack = item["training_pack"]
        if pack not in pack_sha:
            pack_sha[pack] = sha256(ROOT / "runtime" / pack / "source.sha256")
        run_id = run_dir.parent.name
        row = {"unique_key": key, "run_dir": str(run_dir), "run_id": run_id, "receipt_source": source if receipt else "",
               "item_overlay_sha256": (item.get("overlay") or {}).get("sha256", ""),
               "receipt_overlay_sha256": (receipt or {}).get("overlay_sha256", ""),
               "item_training_item_sha256": item["training_item"]["sha256"],
               "receipt_item_sha256": (receipt or {}).get("item_sha256", ""),
               "item_pack_source_sha256": pack_sha[pack],
               "receipt_source_sha_file_sha256": (receipt or {}).get("source_sha_file_sha256", ""),
               "receipt_state": (receipt or {}).get("state", ""),
               "item_runner": item["training_runner"]["path"],
               "ledger_runners": ";".join(sorted(set(runners.get((key, run_id), [])))),
               "eval_status": ""}
        problems = []
        if receipt is None:
            problems.append("no training receipt for the checkpoint's run")
        else:
            if row["receipt_state"] != "SUCCEEDED":
                problems.append(f"receipt state {row['receipt_state']}")
            for field, label in (("overlay_sha256", "overlay"), ("item_sha256", "training item"),
                                 ("source_sha_file_sha256", "code pack")):
                expected = {"overlay_sha256": row["item_overlay_sha256"], "item_sha256": row["item_training_item_sha256"],
                            "source_sha_file_sha256": row["item_pack_source_sha256"]}[field]
                if receipt.get(field) is not None and receipt.get(field) != expected:
                    problems.append(f"{label} differs")
                if receipt.get(field) is None and field == "overlay_sha256" and row["item_overlay_sha256"]:
                    problems.append("receipt has no overlay but item binds one")
        if row["ledger_runners"] and row["item_runner"] not in row["ledger_runners"].split(";"):
            problems.append("runner differs from the submission ledger")
        row["status"] = "MATCH" if not problems else "MISMATCH"
        row["problems"] = "; ".join(problems)
        counts[row["status"]] += 1
        rows.append(row)
    (AUDIT / "BINDING_AUDIT.json").write_text(json.dumps(
        {"revision": args.revision, "counts": dict(counts), "rows": rows}, indent=1, sort_keys=True) + "\n")
    with (AUDIT / "BINDING_AUDIT.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(counts)))
    for row in rows:
        if row["status"] != "MATCH":
            print(row["unique_key"], "|", row["problems"], "|", row["ledger_runners"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
