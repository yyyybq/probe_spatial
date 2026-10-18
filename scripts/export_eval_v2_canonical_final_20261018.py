#!/usr/bin/env python3
"""Combine the canonical-V2 v2-evaluation exports into one final table.

Base: results/canonical_v2_eval_v2/r2/production_metrics.csv (676 items).
Supersede: every key that revision r3 re-evaluated (the 8 VideoMAE p24 C1/C2 items
whose r2 items bound the wrong owner overlay; see
artifacts/release_audit_20261018/overlay_binding_audit/BINDING_AUDIT.json).
A superseding r3 row must itself be ACCEPTED, else the export refuses.

Writes results/canonical_v2_eval_v2/final/production_metrics.{csv,json} with an
added eval_revision column.  Run the per-revision exports first.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/canonical_v2_eval_v2/r2/production_metrics.csv"
SUPERSEDE = ROOT / "results/canonical_v2_eval_v2/r3/production_metrics.csv"
OUT = ROOT / "results/canonical_v2_eval_v2/final"


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    base, newer = read(BASE), {r["unique_key"]: r for r in read(SUPERSEDE)}
    bad = [k for k, r in newer.items() if r["status"] != "ACCEPTED"]
    if bad:
        raise SystemExit(f"superseding r3 rows not ACCEPTED: {bad}")
    unknown = set(newer) - {r["unique_key"] for r in base}
    if unknown:
        raise SystemExit(f"r3 rows not in the r2 plan: {sorted(unknown)}")
    rows = []
    for row in base:
        replacement = newer.get(row["unique_key"])
        rows.append({**(replacement or row), "eval_revision": "r3" if replacement else "r2"})
    fields = list(rows[0])
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "production_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = dict(Counter(r["status"] for r in rows))
    (OUT / "production_metrics.json").write_text(json.dumps(
        {"base": str(BASE), "supersede": str(SUPERSEDE), "superseded_keys": sorted(newer),
         "counts": counts, "by_revision": dict(Counter(r["eval_revision"] for r in rows)), "rows": rows},
        indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts, "superseded": len(newer)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
