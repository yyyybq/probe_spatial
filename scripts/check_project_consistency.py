#!/usr/bin/env python3
"""Validate registry/canonical consistency for probing_spatial."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALID_STATUSES = {
    "PLANNED",
    "RUNNING",
    "COMPLETE",
    "FAILED",
    "INCOMPLETE",
    "INVALID",
    "SUPERSEDED",
    "ARCHIVED",
}
CANONICAL_FILES = [
    "results/canonical/a1_results.csv",
    "results/canonical/a2_results.csv",
    "results/canonical/b2_results.csv",
    "results/canonical/c1_results.csv",
    "results/canonical/c2_results.csv",
    "results/canonical/c3_results.csv",
]
STALE_REFERENCE_TOKENS = [
    "c_series_best_by_probe_vfm",
    "latest_provisional",
    "formal_best",
    "provisional_best",
    "results_table.csv",
    "evaluation_results_all_probes.tsv",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"n/a", "nan", "none", "?"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def exists(path: str) -> bool:
    if not path:
        return True
    p = Path(path)
    if p.is_absolute():
        return p.exists()
    return (ROOT / p).exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="experiments/registry.csv")
    parser.add_argument("--strict-config", action="store_true", help="Treat missing config as an error instead of warning")
    args = parser.parse_args()
    registry_path = ROOT / args.registry
    rows = read_csv(registry_path)
    errors: list[str] = []
    warnings: list[str] = []
    if not rows:
        errors.append(f"registry missing or empty: {registry_path}")

    ids = [row.get("experiment_id", "") for row in rows]
    for experiment_id, count in Counter(ids).items():
        if count > 1:
            errors.append(f"duplicate experiment_id: {experiment_id} count={count}")

    by_id = {row.get("experiment_id", ""): row for row in rows}
    for row in rows:
        eid = row.get("experiment_id", "")
        status = row.get("status", "")
        if status not in VALID_STATUSES:
            errors.append(f"{eid}: invalid status {status}")
        if status == "COMPLETE":
            if not exists(row.get("raw_artifact_path", "")):
                errors.append(f"{eid}: COMPLETE raw_artifact_path missing: {row.get('raw_artifact_path')}")
            if parse_float(row.get("metric_value")) is None:
                errors.append(f"{eid}: COMPLETE metric_value is not numeric: {row.get('metric_value')}")
            config = row.get("config", "")
            if config and not exists(config):
                message = f"{eid}: config path missing: {config}"
                if args.strict_config:
                    errors.append(message)
                else:
                    warnings.append(message)
        manifest = row.get("dataset_manifest", "")
        if manifest and not exists(manifest):
            warnings.append(f"{eid}: dataset_manifest missing: {manifest}")

    config_keys: defaultdict[tuple[str, ...], list[str]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "COMPLETE":
            continue
        key = (
            row.get("family", ""),
            row.get("probe", ""),
            row.get("model", ""),
            row.get("model_variant", ""),
            row.get("layer", ""),
            row.get("prefix", ""),
            row.get("horizon", ""),
            row.get("seed", ""),
            row.get("eval_split", ""),
            row.get("metric_name", ""),
        )
        config_keys[key].append(row.get("experiment_id", ""))
    for key, members in config_keys.items():
        if len(members) > 1:
            errors.append(f"conflicting COMPLETE rows for {key}: {members}")

    for rel_path in CANONICAL_FILES:
        path = ROOT / rel_path
        if not path.exists():
            errors.append(f"missing canonical table: {rel_path}")
            continue
        for i, row in enumerate(read_csv(path), start=2):
            eid = row.get("experiment_id", "")
            reg = by_id.get(eid)
            if not reg:
                errors.append(f"{rel_path}:{i}: experiment_id not in registry: {eid}")
                continue
            if reg.get("status") != "COMPLETE":
                errors.append(f"{rel_path}:{i}: canonical row references non-COMPLETE experiment: {eid}")
            if row.get("status") and row.get("status") != "COMPLETE":
                errors.append(f"{rel_path}:{i}: canonical row status is not COMPLETE: {eid}")

    for rel_path in ["README.md"]:
        path = ROOT / rel_path
        if not path.exists():
            continue
        text = path.read_text(errors="replace")
        for token in STALE_REFERENCE_TOKENS:
            if token in text:
                warnings.append(
                    f"{rel_path}: references historical token `{token}`; verify it points readers to registry/canonical replacements"
                )

    status_counts = Counter(row.get("status", "") for row in rows)
    print("Registry status counts:")
    for status in sorted(status_counts):
        print(f"  {status}: {status_counts[status]}")
    if warnings:
        print("\nWARNINGS:")
        for warning in warnings[:100]:
            print(f"  - {warning}")
        if len(warnings) > 100:
            print(f"  ... {len(warnings) - 100} more warnings")
    if errors:
        print("\nFAIL:")
        for error in errors[:100]:
            print(f"  - {error}")
        if len(errors) > 100:
            print(f"  ... {len(errors) - 100} more errors")
        sys.exit(1)
    print("\nPASS: registry and canonical tables are internally consistent")


if __name__ == "__main__":
    main()
