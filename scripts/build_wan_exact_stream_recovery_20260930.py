#!/usr/bin/env python3
"""Materialize the 2026-09-28 exact Wan positive allow-list as one V2 shard."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/main_v2_probe_v2_816_20260917"
OUT = BASE / "wan_exact_stream_recovery_20260930_r1"
PARENT = ROOT / "artifacts/wan_aether_streaming_official_r1/extract_worklists/wan_official_key_holes.worklist.json"
ALLOWLIST = BASE / "wan_streaming_exact_audit_20260928_r1/missing_allowlist.jsonl"
PROTOCOL = ROOT / "protocols/inscene_streaming_canonical_v2.json"
SOURCE = ROOT / "manifests/c_retrain_20260722/inscene_c_h124_prefix24.jsonl"

def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=False) if not path.parent.exists() else None
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

def main():
    if OUT.exists():
        raise SystemExit(f"immutable output exists: {OUT}")
    OUT.mkdir(parents=True)
    wanted = set()
    for line in ALLOWLIST.read_text().splitlines():
        r = json.loads(line)
        wanted.add((int(r["layer"]), str(r["scene_id"]), int(r["streaming_window_id"]), int(r["prefix"])))
    parent = json.loads(PARENT.read_text())
    entries = []
    seen = set()
    for entry in parent["entries"]:
        scene, window, prefix = str(entry["scene"]), int(entry["window_id"]), int(entry["prefix"])
        layers = [int(layer) for layer in entry["layers"] if (int(layer), scene, window, prefix) in wanted]
        if layers:
            copied = dict(entry); copied["layers"] = layers; entries.append(copied)
            seen.update((layer, scene, window, prefix) for layer in layers)
    if seen != wanted:
        raise SystemExit(f"parent/allow-list mismatch: found={len(seen)}, wanted={len(wanted)}")
    work = {key: value for key, value in parent.items() if key not in {"entries", "counts", "content_sha256"}}
    work["entries"] = entries
    work["output_layers"] = sorted({key[0] for key in wanted})
    work["counts"] = {"entries": len(entries), "layer_items": len(wanted),
                      "scenes": len({key[1] for key in wanted})}
    work["content_sha256"] = canonical(work)
    work_path = OUT / "wan_streaming_exact_missing.worklist.json"
    write(work_path, work)
    audit = {"audit_type": "WAN_STREAMING_EXACT_POSITIVE_ALLOWLIST_AUDIT_V1",
             "canonical_protocol": work["canonical_protocol"],
             "canonical_source": {"path": str(SOURCE), "sha256": sha(SOURCE)},
             "protocol_file_sha256": sha(PROTOCOL), "status": "INCOMPLETE",
             "hard_errors": [],
             "counts": {"gap_worklists": 1, "hard_errors": 0, "layer_items": len(wanted)},
             "notes": ["Exact positive keys derived from 2026-09-28 remote-receipt audit.",
                       "No key outside missing_allowlist.jsonl is scheduled."],
             "outputs": {"gap_worklists": [{"path": str(work_path), "file_sha256": sha(work_path),
                                                "content_sha256": work["content_sha256"], "counts": work["counts"], "vfm": "wan"}]}}
    audit["content_sha256"] = canonical(audit)
    audit_path = OUT / "wan_exact_source_audit.json"
    write(audit_path, audit)
    command = ["/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python",
               str(ROOT / "scripts/windowfix_v2/prepare_sco_gap_jobs.py"),
               "--source-audit", str(audit_path), "--protocol", str(PROTOCOL),
               "--canonical-source", str(SOURCE), "--output-dir", str(OUT / "workers"),
               "--include-vfm", "wan", "--n-gpus", "1", "--aec2-name", "zoetrope",
               "--worker-spec", "N4lS.Iq.I80.1", "--priority", "HIGHEST", "--quota-type", "reserved",
               "--job-name-suffix", "exact-0930"]
    subprocess.run(command, cwd=ROOT, check=True)
    report = {"status": "READY_NOT_SUBMITTED", "worklist": str(work_path),
              "worklist_sha256": sha(work_path), "exact_atomic_keys": len(wanted),
              "source_allowlist": str(ALLOWLIST), "workers": str(OUT / "workers")}
    write(OUT / "submission_plan.json", report)
    print(json.dumps(report, sort_keys=True))

if __name__ == "__main__":
    main()
