#!/usr/bin/env python3
"""Numerically compare trainable-parameter snapshots from short canary runs."""

from __future__ import annotations

import argparse
import json
import pathlib

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=pathlib.Path)
    parser.add_argument("--candidate", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=True)
    failures = []
    rows = []
    squared_diff = 0.0
    squared_reference = 0.0
    absolute_diff = 0.0
    total_numel = 0
    if set(reference) != set(candidate):
        failures.append("parameter name sets differ")
    for name in sorted(set(reference) & set(candidate)):
        left = reference[name].to(torch.float64)
        right = candidate[name].to(torch.float64)
        if left.shape != right.shape:
            failures.append(f"{name}: shapes differ")
            continue
        abs_diff = (left - right).abs()
        denom = torch.maximum(left.abs(), right.abs()).clamp_min(1e-30)
        max_abs = float(abs_diff.max()) if abs_diff.numel() else 0.0
        max_rel = float((abs_diff / denom).max()) if abs_diff.numel() else 0.0
        close = bool(torch.allclose(left, right, atol=args.atol, rtol=args.rtol, equal_nan=True))
        if not close:
            failures.append(f"{name}: outside atol={args.atol:g}, rtol={args.rtol:g}")
        squared_diff += float((abs_diff * abs_diff).sum())
        squared_reference += float((left * left).sum())
        absolute_diff += float(abs_diff.sum())
        total_numel += left.numel()
        rows.append({"name": name, "numel": left.numel(), "max_abs": max_abs,
                     "max_rel": max_rel, "within_tolerance": close})
    report = {
        "schema": "tstep-probe-parameter-comparison-v1",
        "reference": str(args.reference), "candidate": str(args.candidate),
        "atol": args.atol, "rtol": args.rtol, "passed": not failures,
        "failures": failures, "max_abs": max((r["max_abs"] for r in rows), default=0.0),
        "max_rel": max((r["max_rel"] for r in rows), default=0.0), "parameters": rows,
        "l2": squared_diff ** 0.5,
        "relative_l2": (squared_diff / squared_reference) ** 0.5 if squared_reference else 0.0,
        "mean_abs": absolute_diff / total_numel if total_numel else 0.0,
        "total_numel": total_numel,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not args.quiet:
        print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
