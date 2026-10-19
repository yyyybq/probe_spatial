#!/usr/bin/env python3
"""Poll probe-fill launch logs and surface early failure storms.

Usage:
  python scripts/monitor_probe_fill.py --run-id probe_fill_fill_20260814_1800 --once
  python scripts/monitor_probe_fill.py --out-dir logs/probing_jobs_...
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

PROBE_RE = re.compile(
    r"_(streaming_depth|streaming_mask|view_consistency|"
    r"ego_belief_v2|action_dynamics|path_integration|counterfactual)"
    r"_(vjepa2|videomae|minimax_h3|wan|cogvideox|lingbot)_p(\d+)_layer(\d+)\.log$"
)
END_RE = re.compile(r"^END .* rc=(-?\d+)\s*$", re.M)
ERR_PATTERNS = [
    ("missing_train.py", re.compile(r"can't open file .*train\.py")),
    ("missing_v3_checkpoint", re.compile(r"Could not find 'callbacks/v3_checkpoint'")),
    ("MissingConfig", re.compile(r"MissingConfigException|Could not find '")),
    ("OOM", re.compile(r"CUDA out of memory")),
    ("FileNotFound", re.compile(r"FileNotFoundError")),
    ("ModuleNotFound", re.compile(r"ModuleNotFoundError")),
    ("RuntimeError", re.compile(r"RuntimeError:")),
    ("AssertionError", re.compile(r"AssertionError")),
    ("Traceback", re.compile(r"Traceback \(most recent call last\)")),
]


def _tail_text(path: Path, nbytes: int = 8192) -> str:
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - nbytes))
        return f.read().decode("utf-8", "replace")


def _head_text(path: Path, nbytes: int = 4096) -> str:
    with path.open("rb") as f:
        return f.read(nbytes).decode("utf-8", "replace")


def classify_log(path: Path) -> dict:
    name = path.name
    m = PROBE_RE.search(name)
    probe, vfm = (m.group(1), m.group(2)) if m else ("?", "?")
    st = path.stat()
    if st.st_size == 0:
        return {
            "file": name,
            "vfm": vfm,
            "probe": probe,
            "status": "empty",
            "rc": None,
            "reason": "empty",
            "size": 0,
            "mtime": st.st_mtime,
        }
    head = _head_text(path)
    tail = _tail_text(path)
    txt = head + "\n" + tail
    rc_m = END_RE.search(tail)
    if rc_m is None:
        status, rc, reason = "running", None, "running"
    else:
        rc = int(rc_m.group(1))
        status = "ok" if rc == 0 else "fail"
        reason = f"rc={rc}"
        if status == "fail":
            for lab, rx in ERR_PATTERNS:
                if rx.search(txt):
                    reason = lab
                    break
    return {
        "file": name,
        "vfm": vfm,
        "probe": probe,
        "status": status,
        "rc": rc,
        "reason": reason,
        "size": st.st_size,
        "mtime": st.st_mtime,
    }


def snapshot(log_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(log_dir.glob("*.log")):
        try:
            rows.append(classify_log(p))
        except OSError as e:
            rows.append({"file": p.name, "status": "error", "reason": str(e)})
    return rows


def write_progress(out_dir: Path, rows: list[dict], expected: int) -> None:
    launcher = out_dir / "launcher"
    launcher.mkdir(parents=True, exist_ok=True)
    counts = Counter(r.get("status", "?") for r in rows)
    by_reason = Counter(r.get("reason", "?") for r in rows if r.get("status") == "fail")
    line = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_logs": len(rows),
        "expected": expected,
        "counts": dict(counts),
        "fail_reasons": dict(by_reason),
    }
    with (launcher / "progress.jsonl").open("a") as f:
        f.write(json.dumps(line) + "\n")
    txt = [
        f"ts={line['ts']}",
        f"logs={len(rows)} expected={expected}",
        f"counts={dict(counts)}",
        f"fail_reasons={dict(by_reason)}",
        "",
        "## recent fails",
    ]
    for r in rows:
        if r.get("status") == "fail":
            txt.append(f"  {r['file']}: {r.get('reason')}")
    (launcher / "PROGRESS.txt").write_text("\n".join(txt) + "\n")
    n_fail = counts.get("fail", 0)
    n_ok = counts.get("ok", 0)
    if n_fail >= 8 and n_fail > max(4, n_ok // 2):
        (launcher / "ALERT.txt").write_text(
            f"failure storm: fail={n_fail} ok={n_ok} reasons={dict(by_reason)}\n"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--expected", type=int, default=0)
    ap.add_argument("--interval-sec", type=int, default=60)
    ap.add_argument("--max-hours", type=float, default=48.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    root = args.project_root
    out_dir = args.out_dir
    run_id = args.run_id
    if out_dir is None:
        if not run_id:
            print("need --run-id or --out-dir", file=sys.stderr)
            sys.exit(2)
        out_dir = root / f"logs/probing_jobs_{run_id}"
    if not run_id:
        run_id = out_dir.name.replace("probing_jobs_", "", 1)
    log_dir = root / "logs/inscene15k_streaming" / run_id / "launch"
    expected = args.expected
    if expected <= 0:
        tsv = out_dir / "jobs/all.tsv"
        if tsv.is_file():
            expected = max(0, sum(1 for _ in tsv.open()) - 1)

    deadline = time.time() + args.max_hours * 3600
    while True:
        rows = snapshot(log_dir) if log_dir.is_dir() else []
        write_progress(out_dir, rows, expected)
        counts = Counter(r.get("status") for r in rows)
        print(
            f"[{time.strftime('%H:%M:%S')}] logs={len(rows)}/{expected} "
            f"ok={counts.get('ok',0)} fail={counts.get('fail',0)} "
            f"running={counts.get('running',0)}"
        )
        if args.once:
            break
        if expected > 0 and counts.get("ok", 0) + counts.get("fail", 0) >= expected:
            print("all jobs reached END")
            break
        if time.time() >= deadline:
            print("max-hours reached")
            break
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
