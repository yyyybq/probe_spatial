#!/usr/bin/env python3
"""Poll MiniMax-H3 ABC probe launch logs and surface early failure storms.

Usage:
  python scripts/monitor_minimax_h3_abc_probe.py \
    --run-id minimax_h3_abc_20260810_1230

  python scripts/monitor_minimax_h3_abc_probe.py \
    --out-dir logs/probing_jobs_minimax_h3_abc_20260810_1230

Writes under OUT_DIR/launcher/:
  progress.jsonl   one snapshot per poll
  PROGRESS.txt     human-readable latest status
  ALERT.txt        created on early failure storm / high fail rate
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
    r"ego_belief_v2|ego_belief|action_dynamics|path_integration|counterfactual)"
    r"_(minimax_h3)_p(\d+)_layer(\d+)\.log$"
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
        status, rc = "running", None
        reason = "running"
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


def summarize(rows: list[dict], expected: int) -> dict:
    c = Counter(r["status"] for r in rows)
    reasons = Counter(r["reason"] for r in rows if r["status"] == "fail")
    by_vfm = Counter()
    by_probe = Counter()
    for r in rows:
        by_vfm[f"{r['vfm']}:{r['status']}"] += 1
        by_probe[f"{r['vfm']}/{r['probe']}:{r['status']}"] += 1
    newest = max((r["mtime"] for r in rows), default=None)
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_logs": len(rows),
        "expected": expected,
        "ok": c.get("ok", 0),
        "fail": c.get("fail", 0),
        "running": c.get("running", 0),
        "empty": c.get("empty", 0),
        "missing": max(0, expected - len(rows)),
        "fail_reasons": reasons.most_common(12),
        "by_vfm": dict(sorted(by_vfm.items())),
        "by_probe": dict(sorted(by_probe.items())),
        "newest_mtime": newest,
        "newest_age_sec": (time.time() - newest) if newest else None,
        "fail_examples": [r["file"] for r in rows if r["status"] == "fail"][:8],
        "ok_examples": [r["file"] for r in rows if r["status"] == "ok"][:5],
    }


def render_text(summary: dict, alert: str | None) -> str:
    lines = [
        f"ts_utc={summary['ts']}",
        f"logs={summary['n_logs']}/{summary['expected']} "
        f"ok={summary['ok']} fail={summary['fail']} "
        f"running={summary['running']} empty={summary['empty']} "
        f"missing={summary['missing']}",
    ]
    if summary["newest_age_sec"] is not None:
        lines.append(f"newest_log_age_sec={summary['newest_age_sec']:.0f}")
    if summary["fail_reasons"]:
        lines.append("fail_reasons:")
        for reason, n in summary["fail_reasons"]:
            lines.append(f"  {n:4d}  {reason}")
    if summary["by_vfm"]:
        lines.append("by_vfm:")
        for k, v in summary["by_vfm"].items():
            lines.append(f"  {k}: {v}")
    if alert:
        lines.append("")
        lines.append(f"ALERT: {alert}")
    return "\n".join(lines) + "\n"


def decide_alert(summary: dict, *, min_ended: int, fail_rate: float, early_same: int) -> str | None:
    ended = summary["ok"] + summary["fail"]
    if ended < min_ended and summary["fail"] < early_same:
        return None
    if summary["fail"] >= early_same and summary["ok"] == 0:
        top = summary["fail_reasons"][0][0] if summary["fail_reasons"] else "unknown"
        return (
            f"early failure storm: {summary['fail']} fails / 0 ok "
            f"(top reason={top}). Check launch logs immediately."
        )
    if ended >= min_ended:
        rate = summary["fail"] / max(1, ended)
        if rate >= fail_rate and summary["fail"] >= min_ended:
            top = summary["fail_reasons"][0][0] if summary["fail_reasons"] else "unknown"
            return (
                f"high fail rate {rate:.0%} "
                f"({summary['fail']}/{ended}), top reason={top}"
            )
    return None


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, str]:
    project = Path(args.project_root).resolve()
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
        run_id = args.run_id or out_dir.name.replace("probing_jobs_", "", 1)
    else:
        if not args.run_id:
            raise SystemExit("set --run-id or --out-dir")
        run_id = args.run_id
        out_dir = project / "logs" / f"probing_jobs_{run_id}"
    launch_dir = project / "logs" / "inscene15k_streaming" / run_id / "launch"
    return out_dir, launch_dir, run_id


def once(out_dir: Path, launch_dir: Path, expected: int, args: argparse.Namespace) -> tuple[dict, str | None]:
    launcher = out_dir / "launcher"
    launcher.mkdir(parents=True, exist_ok=True)
    rows = [classify_log(p) for p in sorted(launch_dir.glob("*.log"))] if launch_dir.is_dir() else []
    summary = summarize(rows, expected)
    alert = decide_alert(
        summary,
        min_ended=args.min_ended,
        fail_rate=args.fail_rate,
        early_same=args.early_same,
    )
    summary["alert"] = alert
    with (launcher / "progress.jsonl").open("a") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    text = render_text(summary, alert)
    (launcher / "PROGRESS.txt").write_text(text)
    if alert:
        (launcher / "ALERT.txt").write_text(
            text
            + "\n"
            + f"launch_dir={launch_dir}\n"
            + "tip: rg -n \"Could not find|Traceback|Error\" "
            + f"{launch_dir} | head\n"
        )
        print(f"[ALERT] {alert}", file=sys.stderr)
    print(text, end="")
    return summary, alert


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default="/mnt/umm/users/yinbaiqiao/probe_spatial")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--expected", type=int, default=120)
    ap.add_argument("--interval-sec", type=int, default=60)
    ap.add_argument("--max-hours", type=float, default=36.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--min-ended", type=int, default=8, help="ended jobs before fail-rate alert")
    ap.add_argument("--fail-rate", type=float, default=0.5)
    ap.add_argument("--early-same", type=int, default=8, help="fails with 0 ok to trigger storm alert")
    ap.add_argument(
        "--exit-on-alert",
        action="store_true",
        help="exit nonzero when an alert fires (useful for CI/watchdogs)",
    )
    ap.add_argument(
        "--no-stop-when-done",
        action="store_true",
        help="keep polling until max-hours even after all jobs finish",
    )
    args = ap.parse_args()

    out_dir, launch_dir, run_id = resolve_paths(args)
    print(f"[monitor] run_id={run_id}", flush=True)
    print(f"[monitor] out_dir={out_dir}", flush=True)
    print(f"[monitor] launch_dir={launch_dir}", flush=True)

    deadline = time.time() + args.max_hours * 3600
    while True:
        summary, alert = once(out_dir, launch_dir, args.expected, args)
        if alert and args.exit_on_alert:
            return 2
        done = (
            summary["ok"] + summary["fail"] >= args.expected
            and summary["running"] == 0
            and summary["n_logs"] >= args.expected
        )
        if args.once or ((not args.no_stop_when_done) and done):
            return 1 if summary["fail"] and summary["ok"] == 0 else 0
        if time.time() >= deadline:
            print("[monitor] max-hours reached", file=sys.stderr)
            return 3
        time.sleep(max(5, args.interval_sec))


if __name__ == "__main__":
    raise SystemExit(main())
