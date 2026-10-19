#!/usr/bin/env python3
"""Build the read-only Q00 reconciliation of the frozen historical 1096 universe.

This program reads repository artifacts and writes only the requested analysis
outputs.  It deliberately does not launch training, feature extraction, or
evaluation, and it asserts that registry/canonical source hashes are unchanged.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_SOURCE = ROOT / "experiments/task_status_1096_before_expand_20260822.csv"
EXPECTED_SOURCE = ROOT / "experiments/expected_experiments.csv"
RECON_SOURCE = ROOT / "experiments/registry_reconstructed_candidate.csv"
EXPANDED_LEDGER = ROOT / "experiments/task_status_1096.csv"
REGISTRY = ROOT / "experiments/registry.csv"
CANONICAL_DIR = ROOT / "results/canonical"
CANONICAL_ALL = CANONICAL_DIR / "all_results.csv"
B2_RECON = ROOT / "results/reconstructed/default8_b2_angle_candidate.csv"
LINEAGE = ROOT / "experiments/wan_cog_lineage_resolution.csv"
PARTIAL_REVIEW = ROOT / "experiments/work_queue/training_partial_review.csv"
HISTORICAL_REPORT = ROOT / "archive/consolidation_history/20260817/historical_reconstruction_report.md"
VIDEOMAE_SUPERVISOR = ROOT / "logs/videomae_streaming_extract_20260807_161245/supervisor.log"
VIDEOMAE_REPAIR_SYNC = ROOT / "logs/videomae_6115_repair_sco_20260809_163604/aoss_sync.log"
RECON_SCRIPT = ROOT / "scripts/reconstruct_historical_experiments.py"
LEDGER_SCRIPT = ROOT / "scripts/build_task_status_ledger.py"
FILE_MAP = ROOT / "docs/FILE_MAP.md"
OUT_DIR = ROOT / "results/analysis/q00_1096"
REPORT = ROOT / "reports/q00_universe_reconciliation_1096.md"

PROTOCOL = "historical_candidate_v1_val_protocol"
FULL_ID_FIELDS = (
    "campaign",
    "model",
    "model_variant",
    "layer",
    "prefix",
    "probe_config",
    "horizon",
    "seed",
)

PRIMARY_STATUSES = (
    "CANONICAL_COMPLETE",
    "RAW_RESULT_READY_TO_PACKAGE",
    "CHECKPOINT_READY_EVAL_ONLY",
    "FEATURE_READY_TRAIN_AND_EVAL",
    "FEATURE_PARTIAL_OR_MISSING",
    "CHECKPOINT_INVALID_OR_INTERRUPTED",
    "MANIFEST_OR_PROVENANCE_BLOCKED",
    "NOT_IN_REGISTRY",
    "EXACT_DUPLICATE_ALIAS",
    "DEPRECATED_OR_PROTOCOL_INVALID",
    "UNRESOLVED",
)

BOTTLENECKS = (
    "NONE",
    "PACKAGING",
    "EVAL",
    "TRAINING",
    "FEATURE_EXTRACTION",
    "CHECKPOINT",
    "MANIFEST",
    "CONFIG_OR_CODE",
    "PROVENANCE",
    "DUPLICATE_OR_COLLISION",
    "PROTOCOL_INVALID",
    "UNKNOWN",
)

FAMILY_TABLE = {
    "A1": "results/canonical/a1_results.csv",
    "A2": "results/canonical/a2_results.csv",
    "B1": "",  # B1 is represented only in all_results.csv.
    "B2": "results/canonical/b2_results.csv",
    "C1": "results/canonical/c1_results.csv",
    "C2": "results/canonical/c2_results.csv",
    "C3": "results/canonical/c3_results.csv",
}

EXPECTED_METRICS = {
    "streaming_depth": {"mean_loss_depth", "Auc_30"},
    "streaming_camera": {"Auc_30"},
    "streaming_mask": {"mean_loss_identity"},
    "view_consistency": {"balanced_acc"},
    "ego_belief": {"mean_az_err_deg"},
    "ego_belief_v2": {"mean_ang_err_deg"},
    "action_dynamics": {"global_R@1"},
    "path_integration": {"global_R@1"},
    "counterfactual": {"intervention_validity"},
}

# Historical successful-run medians supplied by the resource audit.  These are
# planning estimates, not promises; the report gives an explicit uncertainty band.
TRAIN_MEDIAN_GPU_H = {
    "counterfactual": 14.94,
    "path_integration": 14.22,
    "action_dynamics": 9.68,
    "view_consistency": 20.95,
    "ego_belief_v2": 20.34,
    "streaming_mask": 31.22,
    "streaming_depth": 22.85,
    "streaming_camera": 22.85,
    "ego_belief": 18.09,
}

SCRIPT_SHA_EXPECTED = {
    "scripts/reconstruct_historical_experiments.py": "9aafc1ebb6a0e07bf94d5ea81e3cdf67c79b2776e637e611fbba12438fb5de05",
    "scripts/build_task_status_ledger.py": "d847603ea590a9b329506962345af654603f4cde14955c3c6b688e9ee5f18748",
    "scripts/build_probe_fill_jobs.py": "5505ebe7bd725b53efc98c831f24ec00b7fa17ba07f88d9a3f2a6a711bc373a1",
    "scripts/build_missing_eval_jobs.py": "a44789bba903b9f9816aa49b62d726a766c7e7306b0b48ac7003161bad24252c",
    "scripts/update_registry_with_eval_summaries.py": "aefb6033e66bf7983767e98ebcca8795fa94156309395aecb759e04126eb889b",
    "scripts/promote_wan_p8_clean_eval.py": "b8b389324d402caf9c4f5995930e28060287741b554d9199330fed267ed04d9f",
    "scripts/audit_wan_p8_clean_rerun.py": "11c56a4a108d74c71e4727799fe07a401b12396c8434504115dc29aaad69db60",
    "scripts/extract_wan_manifest_exact.py": "654644424e94ea8c9a1efecab5e1a65abf37321045d9b97a1f53e9bd668353d0",
    "features/run_inscene15k.py": "ee95d1d4d2ba554a919d2668d7be555222dfd1a0ba440a5a83d7a67d1eba9486",
    "scripts/run_vjepa2_videomae_streaming_8gpu_extract.sh": "6d0e49795fce53f62c054e0203d6a9a632506fbab8ded7fcaf18ec605eb98167",
}


def rel(path: Path | str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if p.is_absolute():
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)
    return str(p)


def abs_path(path: str | Path | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


_SHA_CACHE: dict[str, str] = {}


def sha256(path: Path | str | None, *, refresh: bool = False) -> str:
    p = abs_path(path)
    if p is None or not p.is_file():
        return ""
    key = str(p)
    if refresh:
        _SHA_CACHE.pop(key, None)
    if key in _SHA_CACHE:
        return _SHA_CACHE[key]
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def file_state(path: str | Path | None) -> str:
    p = abs_path(path)
    if p is None:
        return "UNSPECIFIED"
    if p.is_file():
        return "PRESENT"
    if p.exists():
        return "PRESENT_NONFILE"
    return "MISSING"


def git_status(path: Path) -> str:
    proc = subprocess.run(
        ["git", "status", "--short", "--", rel(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    value = proc.stdout.strip()
    return value if value else "CLEAN_TRACKED"


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


def normalized_probe(probe: str) -> str:
    if probe in {"streaming_depth", "streaming_depth/depth", "streaming_depth/camera"}:
        return "streaming_depth"
    if probe in {"streaming_mask", "streaming_mask/mask", "streaming_depth/mask"}:
        return "streaming_mask"
    return probe


def registry_campaign(row: dict[str, str]) -> str:
    variant = row["model_variant"]
    model = row["model"]
    if variant in {"default_streaming", "c_retrain_with_extra"}:
        return "default8"
    if variant == "wan_cog_layer_sweep":
        return "wan_cog_layer_sweep"
    if variant == "eval_20260817":
        return {
            "lingbot": "lingbot_video_layer_benchmark",
            "minimax_h3": "minimax_h3_layer_benchmark",
            "vjepa2": "videomae_vjepa_layer_sweep",
            "videomae": "videomae_vjepa_layer_sweep",
            "wan": "wan_cog_layer_sweep",
            "cogvideox": "wan_cog_layer_sweep",
        }.get(model, "")
    return ""


def parent_lookup_key(row: dict[str, str], *, registry_row: bool = False) -> tuple[str, ...]:
    if registry_row:
        return (
            registry_campaign(row),
            row["model"],
            row["layer"],
            row["prefix"],
            normalized_probe(row["probe"]),
            row["horizon"],
            row["seed"],
        )
    return (
        row["campaign"],
        row["model"],
        row["layer"],
        row["prefix"],
        row["probe_config"],
        row["horizon"],
        row["seed"],
    )


def full_identity(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[f] for f in FULL_ID_FIELDS)


def expected_identity(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["campaign"],
        row["model"],
        row["model_variant"],
        row["layer"],
        row["prefix"],
        row["probe"],
        row["horizon"],
        row["seed"],
    )


def identity_set_sha(rows: Iterable[dict[str, str]], fields: tuple[str, ...]) -> str:
    encoded = [json.dumps([r[f] for f in fields], ensure_ascii=False, separators=(",", ":")) for r in rows]
    return hashlib.sha256(("\n".join(sorted(encoded)) + "\n").encode()).hexdigest()


def semijoin(values: Iterable[str]) -> str:
    return ";".join(dict.fromkeys(v for v in values if v))


def yaml_values(path: str | Path | None, key: str) -> list[str]:
    p = abs_path(path)
    if p is None or not p.is_file():
        return []
    text = p.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(r"^\s*" + re.escape(key) + r":\s*([^\n#]+?)\s*$", re.MULTILINE)
    return [m.group(1).strip().strip("\"'") for m in pattern.finditer(text)]


def last_int_value(paths: Iterable[str | Path | None], key: str) -> int | None:
    for path in paths:
        vals = yaml_values(path, key)
        numeric = [int(v) for v in vals if re.fullmatch(r"-?\d+", v)]
        if numeric:
            return numeric[-1]
    return None


def last_value(paths: Iterable[str | Path | None], key: str) -> str:
    for path in paths:
        vals = yaml_values(path, key)
        if vals:
            return vals[-1]
    return ""


def parse_checkpoint(path: str) -> tuple[int | None, int | None]:
    epoch_match = re.search(r"epoch[=_-](\d+)", path or "")
    step_match = re.search(r"step[=_-](\d+)", path or "")
    return (
        int(epoch_match.group(1)) if epoch_match else None,
        int(step_match.group(1)) if step_match else None,
    )


def strict_checkpoint_audit(
    checkpoint: str,
    log_path: str,
    config_paths: Iterable[str | Path | None],
) -> dict[str, Any]:
    epoch, step = parse_checkpoint(checkpoint)
    max_epochs = last_int_value(config_paths, "max_epochs")
    log = abs_path(log_path)
    clean_marker = False
    bad_marker = False
    if log is not None and log.is_file():
        text = log.read_text(encoding="utf-8", errors="replace")
        starts = [m.start() for m in re.finditer(r"Starting training", text, re.IGNORECASE)]
        segment = text[starts[-1] :] if starts else text
        clean_marker = bool(
            re.search(
                r"Trainer\.fit.*stopped.*max_epochs|max_epochs[^\n]*(?:reached|stopped)|"
                r"Training completed|FIT_END[^\n]*(?:success|rc=0)|END[^\n]*rc=0",
                segment,
                re.IGNORECASE,
            )
        )
        bad_marker = bool(
            re.search(
                r"Traceback \(most recent call last\)|RuntimeError:|FileNotFoundError|"
                r"CUDA out of memory|OutOfMemoryError|\bKilled\b|EADDRINUSE",
                segment,
                re.IGNORECASE,
            )
        )
    checkpoint_exists = bool(checkpoint and abs_path(checkpoint) and abs_path(checkpoint).is_file())
    epoch_complete = epoch is not None and max_epochs is not None and epoch + 1 >= max_epochs
    strict_pass = checkpoint_exists and clean_marker and not bad_marker and epoch_complete
    return {
        "checkpoint_exists": checkpoint_exists,
        "epoch": epoch,
        "step": step,
        "max_epochs": max_epochs,
        "expected_final_epoch": max_epochs - 1 if max_epochs is not None else None,
        "clean_marker": clean_marker,
        "bad_marker": bad_marker,
        "epoch_complete": epoch_complete,
        "strict_pass": strict_pass,
    }


def prediction_candidates(summary_paths: Iterable[str]) -> list[str]:
    found: list[str] = []
    for summary in summary_paths:
        p = abs_path(summary)
        if p is None or not p.is_file():
            continue
        for name in ("val_predictions.pt", "predictions.pt", "val_prediction.pt"):
            candidate = p.with_name(name)
            if candidate.is_file():
                found.append(rel(candidate))
    return list(dict.fromkeys(found))


def md_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in fields})


def bool_s(value: bool) -> str:
    return "true" if value else "false"


def finite_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main() -> None:
    required = [
        UNIVERSE_SOURCE,
        EXPECTED_SOURCE,
        RECON_SOURCE,
        REGISTRY,
        CANONICAL_ALL,
        B2_RECON,
        LINEAGE,
    ]
    missing_required = [rel(p) for p in required if not p.is_file()]
    if missing_required:
        raise SystemExit(f"required sources missing: {missing_required}")

    protected_paths = [REGISTRY] + sorted(CANONICAL_DIR.glob("*.csv"))
    protected_before = {rel(p): sha256(p) for p in protected_paths}

    universe = read_csv(UNIVERSE_SOURCE)
    expected = read_csv(EXPECTED_SOURCE)
    reconstructed = read_csv(RECON_SOURCE)
    registry = read_csv(REGISTRY)
    canonical_all = read_csv(CANONICAL_ALL)
    b2_rows = read_csv(B2_RECON)
    lineage_rows = read_csv(LINEAGE)
    partial_rows = read_csv(PARTIAL_REVIEW) if PARTIAL_REVIEW.is_file() else []

    assert len(universe) == 1096, len(universe)
    assert len(expected) == 1096, len(expected)
    assert len(reconstructed) == 1096, len(reconstructed)
    assert len({r["experiment_id"] for r in universe}) == 1096
    assert len({full_identity(r) for r in universe}) == 1096
    universe_identity = {full_identity(r) for r in universe}
    expected_identity_set = {expected_identity(r) for r in expected}
    recon_identity_set = {
        (
            r["campaign"],
            r["model"],
            r["model_variant"],
            r["layer"],
            r["prefix"],
            r["probe"],
            r["horizon"],
            r["seed"],
        )
        for r in reconstructed
    }
    assert universe_identity == expected_identity_set == recon_identity_set
    assert all(u["experiment_id"] == r["experiment_id"] for u, r in zip(universe, reconstructed))

    expanded = read_csv(EXPANDED_LEDGER) if EXPANDED_LEDGER.is_file() else []
    expanded_ids = {r["experiment_id"] for r in expanded}
    universe_ids = {r["experiment_id"] for r in universe}
    expanded_additions = sorted(expanded_ids - universe_ids)
    assert not (universe_ids - expanded_ids) if expanded else True
    if expanded:
        assert len(expanded_additions) == 164

    tokens = ("t499", "t999", "t500", "t1000", "diffusion_step", "tstep")
    diffusion_hits = {
        token: sum(token.lower() in " ".join(r.values()).lower() for r in universe) for token in tokens
    }
    assert sum(diffusion_hits.values()) == 0

    universe_by_id = {r["experiment_id"]: r for r in universe}
    reconstructed_by_id = {r["experiment_id"]: r for r in reconstructed}
    parent_index: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in universe:
        parent_index[parent_lookup_key(row)].append(row)
    assert all(len(v) == 1 for v in parent_index.values())

    registry_matches: dict[str, list[dict[str, str]]] = defaultdict(list)
    registry_to_parent: dict[str, str] = {}
    registry_extra: list[dict[str, str]] = []
    registry_ambiguous: list[dict[str, str]] = []
    for row in registry:
        matches = parent_index.get(parent_lookup_key(row, registry_row=True), [])
        if len(matches) == 1:
            uid = matches[0]["experiment_id"]
            registry_matches[uid].append(row)
            registry_to_parent[row["experiment_id"]] = uid
        elif not matches:
            registry_extra.append(row)
        else:
            registry_ambiguous.append(row)
    assert not registry_ambiguous
    assert len(registry_to_parent) == 928, len(registry_to_parent)
    assert len(registry_extra) == 27, len(registry_extra)
    assert len(registry_matches) == 834, len(registry_matches)

    # Two Wan parent rows have two COMPLETE registry variants with different
    # values/provenance.  They must not be pooled.  Select the historical
    # wan_cog_layer_sweep artifact as the primary mapping for the frozen target
    # and retain eval_20260817 as an explicit alternate artifact mapping.
    primary_registry_matches: dict[str, list[dict[str, str]]] = defaultdict(list)
    alternate_registry_matches: dict[str, list[dict[str, str]]] = defaultdict(list)
    for uid, matches in registry_matches.items():
        grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
        for row in matches:
            grouped[(row["family"], row["probe"], row["metric_name"], row["status"])].append(row)
        for same_atomic in grouped.values():
            variants = {r["model_variant"] for r in same_atomic}
            if (
                len(variants) > 1
                and universe_by_id[uid]["campaign"] == "wan_cog_layer_sweep"
                and "wan_cog_layer_sweep" in variants
            ):
                chosen = [r for r in same_atomic if r["model_variant"] == "wan_cog_layer_sweep"]
                alternates = [r for r in same_atomic if r["model_variant"] != "wan_cog_layer_sweep"]
                assert len(chosen) == 1
                primary_registry_matches[uid].extend(chosen)
                alternate_registry_matches[uid].extend(alternates)
            else:
                primary_registry_matches[uid].extend(same_atomic)
    assert sum(len(v) for v in alternate_registry_matches.values()) == 2
    assert set(alternate_registry_matches) == {
        "wan_cog_layer_sweep_ego_belief_wan_multi_layer_l29_p24_hall_s42",
        "wan_cog_layer_sweep_ego_belief_v2_wan_multi_layer_l0_p24_hall_s42",
    }

    canonical_ids_all = {r["experiment_id"] for r in canonical_all}
    registry_complete_ids = {r["experiment_id"] for r in registry if r["status"] == "COMPLETE"}
    assert canonical_ids_all == registry_complete_ids
    canonical_in_universe_ids = canonical_ids_all & set(registry_to_parent)
    canonical_extra_ids = canonical_ids_all - set(registry_to_parent)
    assert len(canonical_in_universe_ids) == 734
    assert len(canonical_extra_ids) == 27
    primary_canonical_in_universe_ids = {
        r["experiment_id"]
        for rows in primary_registry_matches.values()
        for r in rows
        if r["status"] == "COMPLETE"
    }
    alternate_canonical_in_universe_ids = {
        r["experiment_id"]
        for rows in alternate_registry_matches.values()
        for r in rows
        if r["status"] == "COMPLETE"
    }
    assert len(primary_canonical_in_universe_ids) == 732
    assert len(alternate_canonical_in_universe_ids) == 2
    assert primary_canonical_in_universe_ids | alternate_canonical_in_universe_ids == canonical_in_universe_ids

    family_id_sets: dict[str, set[str]] = {}
    family_physical_counts: dict[str, Counter[str]] = {}
    for family, table in FAMILY_TABLE.items():
        if not table:
            family_id_sets[family] = set()
            family_physical_counts[family] = Counter()
            continue
        rows = read_csv(ROOT / table)
        family_id_sets[family] = {r["experiment_id"] for r in rows}
        family_physical_counts[family] = Counter(r["experiment_id"] for r in rows)

    canonical_parent_ids: set[str] = set()
    expected_metric_missing: dict[str, set[str]] = {}
    for uid, matches in primary_registry_matches.items():
        complete = [r for r in matches if r["status"] == "COMPLETE" and r["experiment_id"] in canonical_ids_all]
        metrics = {r["metric_name"] for r in complete}
        needed = EXPECTED_METRICS[universe_by_id[uid]["probe_config"]]
        if needed <= metrics:
            canonical_parent_ids.add(uid)
        elif complete:
            expected_metric_missing[uid] = needed - metrics
    assert len(canonical_parent_ids) == 642, len(canonical_parent_ids)
    assert not expected_metric_missing, expected_metric_missing

    incomplete_registry = [r for r in registry if r["status"] == "INCOMPLETE"]
    assert len(incomplete_registry) == 194
    assert all(r["experiment_id"] in registry_to_parent for r in incomplete_registry)
    incomplete_parent_ids = {registry_to_parent[r["experiment_id"]] for r in incomplete_registry}
    incomplete_only = incomplete_parent_ids - canonical_parent_ids
    incomplete_overlap = incomplete_parent_ids & canonical_parent_ids
    assert len(incomplete_only) == 192
    assert len(incomplete_overlap) == 2

    b2_by_id = {r["experiment_id"]: r for r in b2_rows}
    b2_angle_ids = {
        uid for uid, row in b2_by_id.items() if row["status"] == "COMPLETE_RAW_ONLY" and finite_number(row["mean_ang_err_deg"])
    }
    assert len(b2_angle_ids) == 31
    b2_partial_ids = {uid for uid, row in b2_by_id.items() if row["status"] != "COMPLETE_RAW_ONLY"}
    assert len(b2_partial_ids) == 1
    b2_partial_id = next(iter(b2_partial_ids))

    provisional_by_id = {
        r["old_experiment_id"]: r
        for r in lineage_rows
        if r["resolution_status"] == "PROVISIONAL_ONLY_NO_REPLACEMENT"
    }
    assert len(provisional_by_id) == 162
    provisional_numeric_ids = {
        uid for uid, row in provisional_by_id.items() if "=" in row["old_metric_value"]
    }
    assert len(provisional_numeric_ids) == 162
    found_nearby_ids = {
        uid for uid, row in provisional_by_id.items() if "FOUND_NEARBY_ONLY" in row["old_historical_label"]
    }
    assert len(found_nearby_ids) == 5

    partial_by_id = {r["experiment_id"]: r for r in partial_rows}
    no_registry_ids = universe_ids - set(registry_matches)
    assert len(no_registry_ids) == 262

    # Canonical atomic registry collision audit: dropping model_variant creates 21
    # groups.  They are retained as separate IDs and never collapsed.
    collision_index: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in registry:
        if row["status"] != "COMPLETE":
            continue
        collision_key = (
            row["family"],
            row["probe"],
            row["model"],
            row["layer"],
            row["prefix"],
            row["horizon"],
            row["seed"],
        )
        collision_index[collision_key].append(row)
    collision_groups = [
        rows
        for rows in collision_index.values()
        if len({r["model_variant"] for r in rows}) > 1
    ]
    assert len(collision_groups) == 21
    assert sum(len(g) for g in collision_groups) == 42
    collision_id_to_group: dict[str, str] = {}
    collision_group_json: list[dict[str, Any]] = []
    for idx, rows in enumerate(sorted(collision_groups, key=lambda g: g[0]["experiment_id"]), 1):
        gid = f"variant_collision_{idx:02d}"
        ids = sorted(r["experiment_id"] for r in rows)
        for rid in ids:
            collision_id_to_group[rid] = gid
        collision_group_json.append(
            {
                "collision_group": gid,
                "identity_without_model_variant": {
                    k: rows[0][k] for k in ("family", "probe", "model", "layer", "prefix", "horizon", "seed")
                },
                "registry_ids": ids,
                "model_variants": sorted({r["model_variant"] for r in rows}),
                "mapped_universe_ids": sorted({registry_to_parent[rid] for rid in ids}),
                "same_universe_parent": len({registry_to_parent[rid] for rid in ids}) == 1,
                "primary_registry_ids": sorted(
                    rid
                    for rid in ids
                    if any(
                        p["experiment_id"] == rid
                        for p in primary_registry_matches[registry_to_parent[rid]]
                    )
                ),
                "alternate_registry_ids": sorted(
                    rid
                    for rid in ids
                    if any(
                        p["experiment_id"] == rid
                        for p in alternate_registry_matches[registry_to_parent[rid]]
                    )
                ),
                "disposition": "RETAIN_SEPARATELY_NOT_EXACT_DUPLICATES",
            }
        )

    # Checkpoint selection and row-level reconciliation.
    reconciled: list[dict[str, Any]] = []
    for source in universe:
        uid = source["experiment_id"]
        reg_rows = registry_matches.get(uid, [])
        primary_reg_rows = primary_registry_matches.get(uid, [])
        alternate_reg_rows = alternate_registry_matches.get(uid, [])
        complete_reg = [r for r in primary_reg_rows if r["status"] == "COMPLETE"]
        alternate_complete_reg = [r for r in alternate_reg_rows if r["status"] == "COMPLETE"]
        incomplete_reg = [r for r in reg_rows if r["status"] == "INCOMPLETE"]
        b2 = b2_by_id.get(uid)
        lineage = provisional_by_id.get(uid)

        summary_paths: list[str] = []
        for candidate in (
            b2["summary_path"] if b2 else "",
            source["eval_output_path"],
            source["raw_metric_path"] if source["raw_metric_path"].endswith(".json") else "",
        ):
            if candidate and abs_path(candidate) and abs_path(candidate).is_file():
                summary_paths.append(rel(candidate))
        for row in complete_reg:
            raw = row["raw_artifact_path"]
            if raw.endswith(".json") and abs_path(raw) and abs_path(raw).is_file():
                summary_paths.append(rel(raw))
        summary_paths = list(dict.fromkeys(summary_paths))

        checkpoint_candidates: list[str] = []
        for candidate in (
            *(r["checkpoint"] for r in complete_reg),
            b2["ckpt_path"] if b2 else "",
            lineage["old_checkpoint_path"] if lineage else "",
            source["checkpoint_path"],
        ):
            if candidate:
                checkpoint_candidates.append(rel(candidate))
        checkpoint_candidates = list(dict.fromkeys(checkpoint_candidates))
        existing_cp = [p for p in checkpoint_candidates if abs_path(p) and abs_path(p).is_file()]
        checkpoint = existing_cp[0] if existing_cp else (checkpoint_candidates[0] if checkpoint_candidates else "")

        run_dir = source["run_artifact_path"]
        if b2 and b2["run_artifact_path"]:
            run_dir = b2["run_artifact_path"]
        if lineage and lineage["old_run_path"]:
            run_dir = lineage["old_run_path"]
        hydra_config = rel(Path(run_dir) / ".hydra/config.yaml") if run_dir else ""
        run_manifest = rel(Path(run_dir) / "run_manifest.json") if run_dir else ""
        config_paths = [hydra_config, source["train_config"]]
        cp_audit = strict_checkpoint_audit(checkpoint, source["train_log_path"], config_paths)

        # A B2 summary explicitly binds to a final epoch-49 checkpoint.  Use that
        # stronger evidence even when the legacy aggregate ledger chose another ckpt.
        if uid in b2_angle_ids and b2:
            b2_epoch, b2_step = parse_checkpoint(b2["ckpt_path"])
            cp_audit["epoch"] = b2_epoch
            cp_audit["step"] = b2_step
            cp_audit["checkpoint_exists"] = bool(abs_path(b2["ckpt_path"]) and abs_path(b2["ckpt_path"]).is_file())
            cp_audit["max_epochs"] = 50
            cp_audit["expected_final_epoch"] = 49
            cp_audit["epoch_complete"] = b2_epoch == 49
            cp_audit["strict_pass"] = bool(cp_audit["checkpoint_exists"] and cp_audit["epoch_complete"])

        train_manifest = source["train_manifest"]
        eval_manifest = source["eval_manifest"]
        recon_row = reconstructed_by_id[uid]
        launcher_manifest = recon_row["job_manifest"]
        registry_dataset_manifest = semijoin(r["dataset_manifest"] for r in reg_rows)
        alternate_registry_dataset_manifest = semijoin(r["dataset_manifest"] for r in alternate_reg_rows)
        selected_manifest = train_manifest or eval_manifest

        timestep = last_int_value(config_paths, "feature_timestep")
        timestep_provenance = "RESOLVED_RUN_OR_CONFIG"
        if timestep is None:
            timestep = 250 if source["model"] in {"lingbot", "minimax_h3"} else 749
            timestep_provenance = "MODEL_DEFAULT_INFERRED_CONFIG_MISSING"

        streaming_root = last_value(config_paths, "streaming_feat_root")
        target_root = last_value(config_paths, "target_feat_root")

        has_target_numeric = uid in canonical_parent_ids or uid in b2_angle_ids or uid in provisional_numeric_ids
        # The frozen ledger's metric_evidence was populated with a loose key and
        # can borrow results across campaigns.  Count numeric evidence only when
        # an exact finite artifact exists.  The interrupted default B2 F3R row is
        # top1=n/a, so it has no auxiliary numeric result either.
        auxiliary_numeric = has_target_numeric
        numeric_kind: list[str] = []
        numeric_evidence: list[str] = []
        if uid in canonical_parent_ids:
            numeric_kind.append("CANONICAL_TARGET_METRIC")
            numeric_evidence.extend(f"{r['metric_name']}={r['metric_value']}" for r in complete_reg)
        if uid in b2_angle_ids and b2:
            numeric_kind.append("LEGACY_B2_TARGET_METRIC_PROTOCOL_UNFROZEN")
            numeric_evidence.append(f"mean_ang_err_deg={b2['mean_ang_err_deg']};n={b2['n']}")
        if uid in provisional_numeric_ids and lineage:
            numeric_kind.append("WAN_COG_PROVISIONAL_TARGET_METRIC")
            numeric_evidence.append(lineage["old_metric_value"])

        manifest_exists = bool(train_manifest and abs_path(train_manifest) and abs_path(train_manifest).is_file())
        config_exists = bool(source["train_config"] and abs_path(source["train_config"]) and abs_path(source["train_config"]).is_file())
        is_video_feature_hole = (
            uid in no_registry_ids
            and source["campaign"] == "videomae_vjepa_layer_sweep"
            and source["model"] == "videomae"
            and source["prefix"] == "24"
            and source["training_status"] == "FAILED"
        )
        is_minimax_camera = uid in no_registry_ids and source["probe_config"] == "streaming_camera"
        is_f3r_b2_partial = uid == b2_partial_id

        # Exactly one primary status per frozen universe row.  NOT_IN_REGISTRY is
        # retained as an orthogonal property when richer lifecycle evidence exists.
        if uid in canonical_parent_ids:
            status = "CANONICAL_COMPLETE"
            bottleneck = "NONE"
            wave = "DONE"
            next_action = "none; preserve exact campaign/model_variant/protocol identity"
        elif uid in b2_angle_ids:
            status = "MANIFEST_OR_PROVENANCE_BLOCKED"
            bottleneck = "MANIFEST"
            wave = "Wave E"
            next_action = "audit legacy deterministic split/cardinality against formal B2 protocol; then Wave A or exact re-eval"
        elif uid in provisional_numeric_ids:
            status = "MANIFEST_OR_PROVENANCE_BLOCKED"
            bottleneck = "PROVENANCE"
            wave = "Wave F" if uid in found_nearby_ids else "Wave E"
            next_action = (
                "human tolerance/protocol decision, then clean reconstruction and formal replacement"
                if uid in found_nearby_ids
                else "reconstruct exact manifest/run provenance, then clean retrain and formal eval"
            )
        elif is_video_feature_hole:
            status = "FEATURE_PARTIAL_OR_MISSING"
            bottleneck = "FEATURE_EXTRACTION"
            wave = "Wave D"
            next_action = "extract the five missing VideoMAE prefix-24 layer caches after manifest-exact canary, then clean train/eval"
        elif is_minimax_camera:
            if source["training_status"] == "PARTIAL_OR_INTERRUPTED":
                status = "CHECKPOINT_INVALID_OR_INTERRUPTED"
            else:
                status = "MANIFEST_OR_PROVENANCE_BLOCKED"
            bottleneck = "CONFIG_OR_CODE"
            wave = "Wave E"
            next_action = "freeze/reconstruct camera-only config and missing filtered manifest; audit construct/protocol; then clean train/eval"
        elif is_f3r_b2_partial:
            status = "CHECKPOINT_INVALID_OR_INTERRUPTED"
            bottleneck = "CHECKPOINT"
            wave = "Wave E"
            next_action = "freeze formal B2 split/protocol, discard interrupted epoch-27 checkpoint for promotion, then clean train/eval"
        elif uid in no_registry_ids:
            if not manifest_exists:
                status = "MANIFEST_OR_PROVENANCE_BLOCKED"
                bottleneck = "MANIFEST"
                wave = "Wave E"
                next_action = (
                    "reconstruct and hash exact filtered manifest; if checkpoint strict gate passes, eval only, otherwise clean train/eval"
                )
            elif cp_audit["strict_pass"]:
                status = "CHECKPOINT_READY_EVAL_ONLY"
                bottleneck = "EVAL"
                wave = "Wave B"
                next_action = "run exact-manifest eval after live feature-root canary and strict checkpoint audit"
            else:
                status = "CHECKPOINT_INVALID_OR_INTERRUPTED"
                bottleneck = "CHECKPOINT"
                wave = "Wave C"
                next_action = "clean no-resume train to expected final epoch, strict audit, then exact-manifest eval"
        else:
            status = "UNRESOLVED"
            bottleneck = "UNKNOWN"
            wave = "Wave F"
            next_action = "manual identity and provenance resolution"

        if status not in PRIMARY_STATUSES or bottleneck not in BOTTLENECKS:
            raise AssertionError((uid, status, bottleneck))

        if is_video_feature_hole:
            feature_status = "PARTIAL_PREFIX_000023_MISSING"
            feature_coverage = "0/required prefix-24 cache for this layer; explicit FileNotFoundError"
            feature_provenance = "historical failure log and frozen ledger"
        elif is_minimax_camera and source["training_status"] == "JOB_DECLARED":
            feature_status = "SHARED_CACHE_LIKELY_NOT_ROW_VERIFIED"
            feature_coverage = "UNKNOWN"
            feature_provenance = "job declaration only; no run"
        elif cp_audit["checkpoint_exists"] or source["training_status"] == "COMPLETE":
            feature_status = "HISTORICAL_FEATURES_CONSUMED"
            feature_coverage = "training/validation consumption evidence; live remote inventory not re-extracted"
            feature_provenance = semijoin([hydra_config, run_manifest, source["train_log_path"]])
        else:
            feature_status = "PARTIAL_OR_UNVERIFIED"
            feature_coverage = "UNKNOWN"
            feature_provenance = semijoin([hydra_config, source["train_log_path"]])

        if alternate_complete_reg:
            scientifically_comparable = "YES_PRIMARY_VARIANT_ONLY;ALTERNATE_VARIANT_SEPARATED_NOT_POOLED"
        elif uid in canonical_parent_ids:
            scientifically_comparable = "YES_WITHIN_EXACT_CAMPAIGN_PROTOCOL;DO_NOT_POOL_COLLISIONS"
        elif uid in b2_angle_ids:
            scientifically_comparable = "NO_PENDING_SPLIT_PROTOCOL_AUDIT"
        elif uid in provisional_numeric_ids:
            scientifically_comparable = "NO_TRUE_BEST_PROVENANCE_MISSING"
        else:
            scientifically_comparable = "NOT_YET_EVALUATED"

        if uid in canonical_parent_ids:
            checkpoint_status = "CANONICAL_ACCEPTED;" + ("STRICT_LOCAL_PASS" if cp_audit["strict_pass"] else "LOCAL_STRICT_AUDIT_NOT_PROVEN")
        elif uid in provisional_numeric_ids:
            checkpoint_status = "EXISTS_BUT_TRUE_BEST_NOT_RETAINED" if cp_audit["checkpoint_exists"] else "TRUE_BEST_MISSING"
        elif uid in b2_angle_ids:
            checkpoint_status = "SUMMARY_BOUND_FINAL_EPOCH49" if cp_audit["strict_pass"] else "SUMMARY_CHECKPOINT_INVALID"
        elif cp_audit["strict_pass"]:
            checkpoint_status = "STRICT_CLEAN_FINAL_READY"
        elif cp_audit["checkpoint_exists"]:
            checkpoint_status = "INVALID_OR_INTERRUPTED_NOT_FINAL"
        else:
            checkpoint_status = "MISSING"

        complete_ids = [r["experiment_id"] for r in complete_reg]
        alternate_complete_ids = [r["experiment_id"] for r in alternate_complete_reg]
        all_complete_ids = complete_ids + alternate_complete_ids
        all_reg_ids = [r["experiment_id"] for r in reg_rows]
        family_table_path = FAMILY_TABLE[source["probe_family"]]
        family_hits = [rid for rid in complete_ids if rid in family_id_sets[source["probe_family"]]]
        family_physical_row_count = sum(
            family_physical_counts[source["probe_family"]][rid] for rid in family_hits
        )
        collision_ids = [rid for rid in all_complete_ids if rid in collision_id_to_group]
        collision_groups_for_row = sorted({collision_id_to_group[rid] for rid in collision_ids})

        raw_paths = [
            *(r["raw_artifact_path"] for r in primary_reg_rows),
            source["raw_metric_path"],
            lineage["old_metric_path"] if lineage else "",
            b2["summary_path"] if b2 else "",
        ]
        alternate_raw_paths = [r["raw_artifact_path"] for r in alternate_reg_rows]
        evidence = [
            rel(UNIVERSE_SOURCE),
            rel(REGISTRY) if reg_rows else "",
            rel(CANONICAL_ALL) if complete_reg else "",
            family_table_path if family_hits else "",
            source["train_config"],
            train_manifest,
            eval_manifest,
            launcher_manifest,
            registry_dataset_manifest,
            hydra_config if file_state(hydra_config) == "PRESENT" else "",
            run_manifest if file_state(run_manifest) == "PRESENT" else "",
            run_dir,
            source["train_log_path"],
            checkpoint,
            *summary_paths,
            lineage["old_metric_path"] if lineage else "",
            rel(B2_RECON) if b2 else "",
            rel(LINEAGE) if lineage else "",
            rel(PARTIAL_REVIEW) if uid in partial_by_id else "",
        ]

        predictions = prediction_candidates(summary_paths)
        row = {
            "universe_id": uid,
            "model": source["model"],
            "model_variant": source["model_variant"],
            "campaign": source["campaign"],
            "protocol": PROTOCOL,
            "family": source["probe_family"],
            "probe": source["probe_config"],
            "layer": source["layer"],
            "prefix": source["prefix"],
            "default_timestep": timestep,
            "default_timestep_provenance": timestep_provenance,
            "horizon": source["horizon"],
            "seed": source["seed"],
            "manifest": selected_manifest,
            "manifest_sha256": sha256(selected_manifest),
            "train_manifest": train_manifest,
            "train_manifest_sha256": sha256(train_manifest),
            "train_manifest_status": file_state(train_manifest),
            "eval_manifest": eval_manifest,
            "eval_manifest_sha256": sha256(eval_manifest),
            "eval_manifest_status": file_state(eval_manifest),
            "launcher_manifest": launcher_manifest,
            "launcher_manifest_sha256": sha256(launcher_manifest),
            "launcher_manifest_status": file_state(launcher_manifest),
            "registry_dataset_manifest": registry_dataset_manifest,
            "registry_dataset_manifest_sha256": semijoin(sha256(p) for p in registry_dataset_manifest.split(";") if p),
            "alternate_registry_dataset_manifest": alternate_registry_dataset_manifest,
            "registry_id": semijoin(all_reg_ids),
            "registry_status": semijoin(f"{r['experiment_id']}={r['status']}" for r in reg_rows),
            "primary_registry_id": semijoin(r["experiment_id"] for r in primary_reg_rows),
            "alternate_registry_id": semijoin(r["experiment_id"] for r in alternate_reg_rows),
            "registry_representation_present": bool_s(bool(reg_rows)),
            "registry_id_count": len(reg_rows),
            "canonical_id": semijoin(complete_ids),
            "alternate_canonical_id": semijoin(alternate_complete_ids),
            "mapped_canonical_id_count_total": len(all_complete_ids),
            "canonical_all_results_row_count": len(complete_ids),
            "family_canonical_table": family_table_path or "NO_DEDICATED_B1_TABLE",
            "family_canonical_id": semijoin(family_hits),
            "family_canonical_id_count": len(family_hits),
            "family_canonical_row_count": family_physical_row_count,
            "canonical_metric_names": semijoin(r["metric_name"] for r in complete_reg),
            "alternate_canonical_metric_names": semijoin(r["metric_name"] for r in alternate_complete_reg),
            "alternate_canonical_metric_evidence": semijoin(
                f"{r['experiment_id']}:{r['metric_name']}={r['metric_value']}" for r in alternate_complete_reg
            ),
            "raw_summary_path": semijoin(summary_paths),
            "raw_summary_sha256": semijoin(sha256(p) for p in summary_paths),
            "raw_artifact_paths": semijoin(rel(p) for p in raw_paths),
            "alternate_raw_artifact_paths": semijoin(rel(p) for p in alternate_raw_paths),
            "prediction_artifact_path": semijoin(predictions),
            "prediction_artifact_status": "PRESENT" if predictions else "MISSING_OR_NOT_APPLICABLE",
            "checkpoint_path": checkpoint,
            "checkpoint_candidates": semijoin(checkpoint_candidates),
            "alternate_checkpoint_paths": semijoin(r["checkpoint"] for r in alternate_complete_reg),
            "checkpoint_status": checkpoint_status,
            "checkpoint_epoch": cp_audit["epoch"] if cp_audit["epoch"] is not None else "",
            "checkpoint_step": cp_audit["step"] if cp_audit["step"] is not None else "",
            "expected_final_epoch": cp_audit["expected_final_epoch"] if cp_audit["expected_final_epoch"] is not None else "",
            "expected_max_epochs": cp_audit["max_epochs"] if cp_audit["max_epochs"] is not None else "",
            "clean_completion_status": (
                f"strict_pass={bool_s(cp_audit['strict_pass'])};clean_marker={bool_s(cp_audit['clean_marker'])};"
                f"bad_marker={bool_s(cp_audit['bad_marker'])};epoch_complete={bool_s(cp_audit['epoch_complete'])}"
            ),
            "config": source["train_config"],
            "config_status": file_state(source["train_config"]),
            "hydra_config_path": hydra_config,
            "hydra_config_status": file_state(hydra_config),
            "run_directory": run_dir,
            "run_directory_status": file_state(run_dir),
            "train_log_path": source["train_log_path"],
            "train_log_status": file_state(source["train_log_path"]),
            "streaming_feature_root": streaming_root,
            "target_feature_root": target_root,
            "feature_coverage": feature_coverage,
            "feature_provenance": feature_provenance,
            "feature_status": feature_status,
            "numeric_result_present": bool_s(has_target_numeric),
            "auxiliary_numeric_present": bool_s(auxiliary_numeric),
            "numeric_result_kind": semijoin(numeric_kind),
            "numeric_metric_evidence": semijoin(numeric_evidence),
            "canonical_valid": bool_s(uid in canonical_parent_ids),
            "scientifically_comparable": scientifically_comparable,
            "status": status,
            "bottleneck": bottleneck,
            "execution_wave": wave,
            "next_action": next_action,
            "not_in_registry": bool_s(uid in no_registry_ids),
            "registry_variant_collision_group": semijoin(collision_groups_for_row),
            "registry_variant_collision_ids": semijoin(collision_ids),
            "artifact_mapping_disposition": (
                "PRIMARY_FROZEN_WAN_COG_VARIANT_SELECTED;ALTERNATE_EVAL_20260817_SEPARATED_NOT_POOLED"
                if alternate_reg_rows
                else "NO_SAME_PARENT_VARIANT_ALTERNATE"
            ),
            "full_identity": json.dumps(
                dict([(f, source[f]) for f in FULL_ID_FIELDS] + [("protocol", PROTOCOL)]),
                sort_keys=True,
            ),
            "evidence_paths": semijoin(rel(p) for p in evidence),
        }
        reconciled.append(row)

    assert len(reconciled) == 1096
    assert len({r["universe_id"] for r in reconciled}) == 1096
    status_counts = Counter(r["status"] for r in reconciled)
    for status in PRIMARY_STATUSES:
        status_counts.setdefault(status, 0)
    assert sum(status_counts.values()) == 1096
    assert status_counts["UNRESOLVED"] == 0
    assert {k: v for k, v in status_counts.items() if v} == {
        "CANONICAL_COMPLETE": 642,
        "MANIFEST_OR_PROVENANCE_BLOCKED": 237,
        "CHECKPOINT_READY_EVAL_ONLY": 99,
        "CHECKPOINT_INVALID_OR_INTERRUPTED": 83,
        "FEATURE_PARTIAL_OR_MISSING": 35,
    }, status_counts

    bottleneck_counts = Counter(r["bottleneck"] for r in reconciled)
    assert sum(bottleneck_counts.values()) == 1096
    wave_counts = Counter(r["execution_wave"] for r in reconciled if r["execution_wave"] != "DONE")
    assert wave_counts == Counter({"Wave E": 249, "Wave B": 99, "Wave C": 66, "Wave D": 35, "Wave F": 5})

    numeric_target_count = sum(r["numeric_result_present"] == "true" for r in reconciled)
    auxiliary_any_count = sum(r["auxiliary_numeric_present"] == "true" for r in reconciled)
    assert numeric_target_count == 833
    assert auxiliary_any_count == 833

    fields = [
        "universe_id", "model", "model_variant", "campaign", "protocol", "family", "probe", "layer", "prefix",
        "default_timestep", "default_timestep_provenance", "horizon", "seed", "manifest", "manifest_sha256",
        "train_manifest", "train_manifest_sha256", "train_manifest_status", "eval_manifest", "eval_manifest_sha256",
        "eval_manifest_status", "launcher_manifest", "launcher_manifest_sha256", "launcher_manifest_status",
        "registry_dataset_manifest", "registry_dataset_manifest_sha256", "alternate_registry_dataset_manifest",
        "registry_id", "registry_status", "primary_registry_id", "alternate_registry_id",
        "registry_representation_present", "registry_id_count", "canonical_id", "alternate_canonical_id",
        "mapped_canonical_id_count_total", "canonical_all_results_row_count",
        "family_canonical_table", "family_canonical_id", "family_canonical_id_count", "family_canonical_row_count", "canonical_metric_names",
        "alternate_canonical_metric_names", "alternate_canonical_metric_evidence",
        "raw_summary_path", "raw_summary_sha256", "raw_artifact_paths", "alternate_raw_artifact_paths", "prediction_artifact_path",
        "prediction_artifact_status", "checkpoint_path", "checkpoint_candidates", "checkpoint_status", "checkpoint_epoch",
        "alternate_checkpoint_paths",
        "checkpoint_step", "expected_final_epoch", "expected_max_epochs", "clean_completion_status", "config", "config_status",
        "hydra_config_path", "hydra_config_status", "run_directory", "run_directory_status", "train_log_path",
        "train_log_status", "streaming_feature_root", "target_feature_root", "feature_coverage", "feature_provenance",
        "feature_status", "numeric_result_present", "auxiliary_numeric_present", "numeric_result_kind",
        "numeric_metric_evidence", "canonical_valid", "scientifically_comparable", "status", "bottleneck",
        "execution_wave", "next_action", "not_in_registry", "registry_variant_collision_group",
        "registry_variant_collision_ids", "artifact_mapping_disposition", "full_identity", "evidence_paths",
    ]
    write_csv(OUT_DIR / "universe_1096_reconciled.csv", reconciled, fields)
    missing_rows = [r for r in reconciled if r["status"] != "CANONICAL_COMPLETE"]
    assert len(missing_rows) == 454
    write_csv(OUT_DIR / "missing_or_noncanonical.csv", missing_rows, fields)

    # Detailed bottleneck distribution, preserving campaign/family/model/prefix.
    bottleneck_groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in reconciled:
        key = (
            row["status"], row["bottleneck"], row["execution_wave"], row["campaign"],
            row["family"], row["probe"], row["model"], str(row["prefix"]),
        )
        bottleneck_groups[key].append(row)
    bottleneck_summary = []
    for key, rows in sorted(bottleneck_groups.items()):
        bottleneck_summary.append(
            {
                "status": key[0], "bottleneck": key[1], "wave": key[2], "campaign": key[3],
                "family": key[4], "probe": key[5], "model": key[6], "prefix": key[7],
                "count": len(rows),
                "numeric_result_present": sum(r["numeric_result_present"] == "true" for r in rows),
                "canonical_valid": sum(r["canonical_valid"] == "true" for r in rows),
                "not_in_registry": sum(r["not_in_registry"] == "true" for r in rows),
                "universe_ids": semijoin(r["universe_id"] for r in rows),
            }
        )
    write_csv(
        OUT_DIR / "bottleneck_summary.csv",
        bottleneck_summary,
        ["status", "bottleneck", "wave", "campaign", "family", "probe", "model", "prefix", "count",
         "numeric_result_present", "canonical_valid", "not_in_registry", "universe_ids"],
    )

    # One row per unfinished configuration is the exact Wave A--F execution list.
    wave_rows: list[dict[str, Any]] = []
    feature_cache_groups: set[tuple[str, ...]] = set()
    feature_extraction_jobs_assigned = False
    for row in missing_rows:
        wave = row["execution_wave"]
        needs_train = (
            row["status"] in {"CHECKPOINT_INVALID_OR_INTERRUPTED", "FEATURE_PARTIAL_OR_MISSING"}
            or row["numeric_result_kind"].find("WAN_COG_PROVISIONAL") >= 0
            or row["probe"] == "streaming_camera"
        )
        # Missing-manifest strict-ready rows can go directly to eval after Wave E;
        # strict-fail rows require retraining.  Parse the stored audit flags.
        if wave == "Wave E" and row["status"] == "MANIFEST_OR_PROVENANCE_BLOCKED" and not row["numeric_result_present"] == "true":
            needs_train = "strict_pass=false" in row["clean_completion_status"]
        if row["numeric_result_kind"].find("LEGACY_B2_TARGET") >= 0:
            needs_train = False
        if wave == "Wave F":
            needs_train = True

        train_h = TRAIN_MEDIAN_GPU_H.get(row["probe"], 16.32) if needs_train else 0.0
        needs_eval = not (row["numeric_result_kind"].find("LEGACY_B2_TARGET") >= 0)
        eval_h = 0.05 if needs_eval else 0.0
        feature_group = (
            row["model"], str(row["default_timestep"]), str(row["layer"]),
            str(row["prefix"]), row["protocol"],
        ) if wave == "Wave D" else ()
        feature_jobs = 0
        feature_job_group = ""
        feature_h_point = 0.0
        feature_h_low = 0.0
        feature_h_high = 0.0
        feature_disk_point = 0.0
        feature_disk_budget = 0.0
        if feature_group:
            feature_cache_groups.add(feature_group)
            feature_job_group = "videomae_t749_p24_layers0_5_11_17_23_manifest_exact"
            if not feature_extraction_jobs_assigned:
                # One 1-scene canary plus one shared 4xH800 full extraction job.
                feature_jobs = 2
                feature_h_point = 0.75
                feature_h_low = 0.60
                feature_h_high = 1.05
                feature_disk_point = 40.69
                feature_disk_budget = 45.0
                feature_extraction_jobs_assigned = True
        if wave == "Wave B":
            scripts = "scripts/build_missing_eval_jobs.py;scripts/submit_missing_eval_sco.sh;scripts/run_missing_eval_shard.py"
            gate = "strict final checkpoint; exact manifest SHA; live streaming/target root canary; finite exact-n summary"
        elif wave == "Wave C":
            scripts = "scripts/build_probe_fill_jobs.py;scripts/submit_probe_fill_sco.sh;scripts/run_probe_fill_shard.sh;strict clean audit;missing-eval chain"
            gate = "frozen row input; no-resume; final epoch/step; no fatal markers; protected best+last only; exact eval identity"
        elif wave == "Wave D":
            scripts = "features/run_inscene15k.py (direct entry with --prefix-lengths 24 --num-frames 24);scripts/run_vjepa2_videomae_streaming_8gpu_extract.sh (BLOCKED until NUM_FRAMES=24 passthrough is added and frozen);scripts/validate_feature_cache.py;probe-fill chain;strict clean audit;missing-eval chain"
            gate = "first reconstruct/freeze exact filtered manifest; do not run the current shell runner unchanged because it does not pass --num-frames; either use the direct extractor entry with --prefix-lengths 24 --num-frames 24 or first add/freeze a NUM_FRAMES=24 runner passthrough; 1-scene canary must produce prefix_000023 at layers 0/5/11/17/23; full coverage must equal signed manifest (historical expectation 2721 windows); shape/dtype/finite/AOSS done-index gate; then Wave C gates"
        elif wave == "Wave E":
            scripts = "scripts/reconstruct_historical_experiments.py;lineage/config/log audit;then conditional Wave A/B/C"
            gate = "exact manifest/config/protocol reconstruction with hashes; fail closed if identity or split cardinality is unprovable"
        else:
            scripts = "signed human protocol/tolerance decision;then conditional reconstruction/replacement"
            gate = "do not merge model_variant/campaign collisions; signed disposition required"
        wave_rows.append(
            {
                "wave": wave,
                "universe_id": row["universe_id"],
                "campaign": row["campaign"], "family": row["family"], "probe": row["probe"], "model": row["model"],
                "model_variant": row["model_variant"], "layer": row["layer"], "prefix": row["prefix"],
                "current_status": row["status"], "bottleneck": row["bottleneck"], "next_action": row["next_action"],
                "entry_scripts": scripts,
                "gpu_feature_jobs": feature_jobs,
                "feature_cache_group": "|".join(feature_group),
                "feature_extraction_job_group": feature_job_group,
                "gpu_train_jobs": 1 if needs_train else 0,
                "gpu_eval_jobs": 1 if needs_eval else 0,
                "estimated_feature_gpu_h_point": f"{feature_h_point:.2f}",
                "estimated_feature_gpu_h_low": f"{feature_h_low:.2f}",
                "estimated_feature_gpu_h_high": f"{feature_h_high:.2f}",
                "estimated_train_gpu_h_point": f"{train_h:.2f}",
                "estimated_train_gpu_h_low": f"{train_h * 0.5:.2f}",
                "estimated_train_gpu_h_high": f"{train_h * 2.0:.2f}",
                "estimated_eval_gpu_h_point": f"{eval_h:.2f}",
                "estimated_eval_gpu_h_p90": f"{0.13 if needs_eval else 0.0:.2f}",
                "new_checkpoint_disk_gib_low": f"{0.4 if needs_train else 0.0:.2f}",
                "new_checkpoint_disk_gib_high": f"{1.86 if needs_train else 0.0:.2f}",
                "new_feature_disk_gib_point": f"{feature_disk_point:.2f}",
                "new_feature_disk_gib_budget": f"{feature_disk_budget:.2f}",
                "gate_and_stop_condition": gate + "; script SHA must equal frozen provenance copy and launcher record",
                "canonical_eligibility": "YES only after all gates and staging-only promotion diff pass; otherwise NO",
                "resource_cost_condition": (
                    "CONDITIONAL_AFTER_RECONSTRUCTION_OR_HUMAN_GATE"
                    if wave in {"Wave E", "Wave F"}
                    else "DIRECT_AFTER_ROW_GATE"
                ),
                "evidence_paths": row["evidence_paths"],
            }
        )
    assert len(wave_rows) == 454
    wave_fields = [
        "wave", "universe_id", "campaign", "family", "probe", "model", "model_variant", "layer", "prefix",
        "current_status", "bottleneck", "next_action", "entry_scripts", "feature_cache_group", "feature_extraction_job_group",
        "gpu_feature_jobs", "gpu_train_jobs", "gpu_eval_jobs", "estimated_feature_gpu_h_point", "estimated_feature_gpu_h_low",
        "estimated_feature_gpu_h_high", "estimated_train_gpu_h_point", "estimated_train_gpu_h_low",
        "estimated_train_gpu_h_high", "estimated_eval_gpu_h_point", "estimated_eval_gpu_h_p90",
        "new_checkpoint_disk_gib_low", "new_checkpoint_disk_gib_high", "new_feature_disk_gib_point", "new_feature_disk_gib_budget",
        "gate_and_stop_condition", "canonical_eligibility", "resource_cost_condition", "evidence_paths",
    ]
    write_csv(OUT_DIR / "execution_waves.csv", wave_rows, wave_fields)

    family_coverage: dict[str, dict[str, Any]] = {}
    for family in sorted({r["family"] for r in reconciled}):
        rows = [r for r in reconciled if r["family"] == family]
        family_coverage[family] = {
            "universe": len(rows),
            "canonical_complete": sum(r["canonical_valid"] == "true" for r in rows),
            "numeric_target_present": sum(r["numeric_result_present"] == "true" for r in rows),
            "noncanonical": sum(r["canonical_valid"] != "true" for r in rows),
            "models": dict(sorted(Counter(r["model"] for r in rows).items())),
            "campaigns": dict(sorted(Counter(r["campaign"] for r in rows).items())),
            "layers": dict(sorted(Counter(str(r["layer"]) for r in rows).items(), key=lambda x: int(x[0]))),
            "prefixes": dict(sorted(Counter(str(r["prefix"]) for r in rows).items(), key=lambda x: int(x[0]))),
            "status_counts": dict(sorted(Counter(r["status"] for r in rows).items())),
        }

    wave_aggregates: dict[str, dict[str, Any]] = {}
    for wave in ("Wave A", "Wave B", "Wave C", "Wave D", "Wave E", "Wave F"):
        rows = [r for r in wave_rows if r["wave"] == wave]
        wave_aggregates[wave] = {
            "count": len(rows),
            "gpu_feature_jobs": sum(int(r["gpu_feature_jobs"]) for r in rows),
            "gpu_train_jobs": sum(int(r["gpu_train_jobs"]) for r in rows),
            "gpu_eval_jobs": sum(int(r["gpu_eval_jobs"]) for r in rows),
            "feature_gpu_h_point": round(sum(float(r["estimated_feature_gpu_h_point"]) for r in rows), 2),
            "feature_gpu_h_low": round(sum(float(r["estimated_feature_gpu_h_low"]) for r in rows), 2),
            "feature_gpu_h_high": round(sum(float(r["estimated_feature_gpu_h_high"]) for r in rows), 2),
            "train_gpu_h_point": round(sum(float(r["estimated_train_gpu_h_point"]) for r in rows), 2),
            "train_gpu_h_low": round(sum(float(r["estimated_train_gpu_h_low"]) for r in rows), 2),
            "train_gpu_h_high": round(sum(float(r["estimated_train_gpu_h_high"]) for r in rows), 2),
            "eval_gpu_h_point": round(sum(float(r["estimated_eval_gpu_h_point"]) for r in rows), 2),
            "eval_gpu_h_p90": round(sum(float(r["estimated_eval_gpu_h_p90"]) for r in rows), 2),
            "checkpoint_disk_gib_low": round(sum(float(r["new_checkpoint_disk_gib_low"]) for r in rows), 2),
            "checkpoint_disk_gib_high": round(sum(float(r["new_checkpoint_disk_gib_high"]) for r in rows), 2),
            "feature_disk_gib_point": round(sum(float(r["new_feature_disk_gib_point"]) for r in rows), 2),
            "feature_disk_gib_budget": round(sum(float(r["new_feature_disk_gib_budget"]) for r in rows), 2),
            "models": dict(sorted(Counter(r["model"] for r in rows).items())),
            "families": dict(sorted(Counter(r["family"] for r in rows).items())),
            "campaigns": dict(sorted(Counter(r["campaign"] for r in rows).items())),
            "layers": dict(sorted(Counter(str(r["layer"]) for r in rows).items(), key=lambda x: int(x[0]))),
            "prefixes": dict(sorted(Counter(str(r["prefix"]) for r in rows).items(), key=lambda x: int(x[0]))),
        }

    source_records = []
    for path in (
        UNIVERSE_SOURCE, EXPECTED_SOURCE, RECON_SOURCE, EXPANDED_LEDGER,
        RECON_SCRIPT, LEDGER_SCRIPT, HISTORICAL_REPORT, FILE_MAP,
        VIDEOMAE_SUPERVISOR, VIDEOMAE_REPAIR_SYNC,
    ):
        source_records.append(
            {
                "path": rel(path),
                "sha256": sha256(path),
                "git_status": git_status(path),
                "row_count": len(read_csv(path)) if path.suffix == ".csv" and path.is_file() else None,
            }
        )

    script_records = {}
    for path, expected_sha in SCRIPT_SHA_EXPECTED.items():
        actual = sha256(path)
        script_records[path] = {
            "sha256": actual,
            "expected_audit_sha256": expected_sha,
            "matches": actual == expected_sha,
            "git_status": git_status(ROOT / path) if (ROOT / path).exists() else "MISSING",
        }

    eventual_train_ids = [r["universe_id"] for r in wave_rows if int(r["gpu_train_jobs"]) == 1]
    eventual_eval_ids = [r["universe_id"] for r in wave_rows if int(r["gpu_eval_jobs"]) == 1]
    assert len(eventual_train_ids) == 287
    assert len(eventual_eval_ids) == 423
    assert len(feature_cache_groups) == 5
    assert sum(int(r["gpu_feature_jobs"]) for r in wave_rows) == 2

    acceptance = {
        "UNIVERSE_SIZE": 1096,
        "UNIVERSE_SOURCE": rel(UNIVERSE_SOURCE),
        "UNIVERSE_SHA256": sha256(UNIVERSE_SOURCE),
        "DIFFUSION_STEP_SWEEP_EXCLUDED": True,
        "CANONICAL_COMPLETE": status_counts["CANONICAL_COMPLETE"],
        "NUMERIC_RESULT_PRESENT": numeric_target_count,
        "REGISTRY_INCOMPLETE": len(incomplete_registry),
        "NOT_IN_REGISTRY": len(no_registry_ids),
        "READY_TO_PACKAGE": status_counts["RAW_RESULT_READY_TO_PACKAGE"],
        "EVAL_ONLY": status_counts["CHECKPOINT_READY_EVAL_ONLY"],
        "NEEDS_TRAINING": len(eventual_train_ids),
        "NEEDS_FEATURE_EXTRACTION": status_counts["FEATURE_PARTIAL_OR_MISSING"],
        "PROVENANCE_BLOCKED": status_counts["MANIFEST_OR_PROVENANCE_BLOCKED"],
        "DUPLICATE_OR_COLLISION": "21 groups / 42 registry IDs; 0 exact duplicate aliases",
        "DEPRECATED_OR_INVALID": status_counts["DEPRECATED_OR_PROTOCOL_INVALID"],
        "UNRESOLVED": status_counts["UNRESOLVED"],
        "RESOLVED_TOTAL": sum(status_counts.values()),
        "COUNT_SUM_EQUALS_1096": sum(status_counts.values()) == 1096,
    }

    summary: dict[str, Any] = {
        "generated_at_utc": "2026-08-28",
        "mode": "STRICT_READ_ONLY_RECONCILIATION_NO_EXPERIMENT_EXECUTION",
        "git_head": git_head(),
        "protocol_namespace": PROTOCOL,
        "universe": {
            "source": rel(UNIVERSE_SOURCE),
            "sha256": sha256(UNIVERSE_SOURCE),
            "git_status": git_status(UNIVERSE_SOURCE),
            "size": 1096,
            "unique_universe_id": len({r["universe_id"] for r in reconciled}),
            "unique_full_identity": len(universe_identity),
            "identity_set_sha256_documented_algorithm": identity_set_sha(universe, FULL_ID_FIELDS),
            "alternate_1096_sources": source_records[1:3],
            "alternate_identity_sets_equal": True,
            "expanded_ledger_rows": len(expanded),
            "expanded_additions_excluded": len(expanded_additions),
            "expanded_addition_ids": expanded_additions,
            "expanded_addition_campaigns": dict(sorted(Counter(r["campaign"] for r in expanded if r["experiment_id"] in set(expanded_additions)).items())),
            "expanded_addition_models": dict(sorted(Counter(r["model"] for r in expanded if r["experiment_id"] in set(expanded_additions)).items())),
            "diffusion_token_hits": diffusion_hits,
            "default_timesteps_retained": dict(sorted(Counter(str(r["default_timestep"]) for r in reconciled).items())),
        },
        "source_provenance": source_records,
        "script_provenance": script_records,
        "registry_reconciliation": {
            "registry_ids": len(registry),
            "mapped_to_universe_ids": len(registry_to_parent),
            "mapped_universe_parent_rows": len(registry_matches),
            "registry_ids_outside_universe": len(registry_extra),
            "registry_ids_outside_universe_list": [r["experiment_id"] for r in registry_extra],
            "atomic_or_alias_surplus_inside_universe": len(registry_to_parent) - len(registry_matches),
            "universe_rows_without_registry_representation": len(no_registry_ids),
            "why_1096_minus_955_is_141_but_not_141_configs": "262 no-registry parent rows - 94 in-universe atomic/alias surplus - 27 post-freeze IDs = 141",
            "incomplete_registry_ids": len(incomplete_registry),
            "incomplete_parent_rows_only": len(incomplete_only),
            "incomplete_parent_rows_also_canonical": sorted(incomplete_overlap),
        },
        "canonical_reconciliation": {
            "canonical_ids": len(canonical_ids_all),
            "canonical_ids_mapped_to_universe": len(canonical_in_universe_ids),
            "primary_canonical_ids_mapped_to_universe": len(primary_canonical_in_universe_ids),
            "alternate_same_parent_canonical_ids_separated": len(alternate_canonical_in_universe_ids),
            "canonical_universe_parent_rows": len(canonical_parent_ids),
            "canonical_ids_outside_universe": len(canonical_extra_ids),
            "canonical_ids_outside_universe_list": sorted(canonical_extra_ids),
            "all_expected_atomic_metrics_present_for_canonical_parents": True,
        },
        "numeric_results": {
            "target_contract_numeric_present": numeric_target_count,
            "any_exact_finite_numeric": auxiliary_any_count,
            "no_numeric_at_all": 1096 - auxiliary_any_count,
            "canonical_target_numeric_parents": len(canonical_parent_ids),
            "default_b2_noncanonical_target_numeric": len(b2_angle_ids),
            "wan_cog_provisional_numeric_rows": len(provisional_numeric_ids),
            "wan_cog_provisional_overlap_with_canonical_parents": len(provisional_numeric_ids & canonical_parent_ids),
            "auxiliary_only_rows": [],
            "target_metric_missing_ids": [r["universe_id"] for r in reconciled if r["numeric_result_present"] != "true"],
            "no_numeric_at_all_ids": [r["universe_id"] for r in reconciled if r["auxiliary_numeric_present"] != "true"],
        },
        "status_counts": {s: status_counts[s] for s in PRIMARY_STATUSES},
        "bottleneck_counts": {b: bottleneck_counts.get(b, 0) for b in BOTTLENECKS},
        "wave_counts": {w: wave_counts.get(w, 0) for w in ("Wave A", "Wave B", "Wave C", "Wave D", "Wave E", "Wave F")},
        "family_coverage": family_coverage,
        "wave_resources": wave_aggregates,
        "eventual_shortest_path": {
            "packaging_candidates_after_protocol_gate": 31,
            "train_jobs": len(eventual_train_ids),
            "eval_jobs": len(eventual_eval_ids),
            "feature_cache_groups": 5,
            "feature_jobs_including_canary": 2,
            "total_gpu_jobs_including_feature_canary": len(eventual_train_ids) + len(eventual_eval_ids) + 2,
            "train_gpu_h_point": round(sum(float(r["estimated_train_gpu_h_point"]) for r in wave_rows), 2),
            "train_gpu_h_low": round(sum(float(r["estimated_train_gpu_h_low"]) for r in wave_rows), 2),
            "train_gpu_h_high": round(sum(float(r["estimated_train_gpu_h_high"]) for r in wave_rows), 2),
            "eval_gpu_h_point": round(sum(float(r["estimated_eval_gpu_h_point"]) for r in wave_rows), 2),
            "eval_gpu_h_p90": round(sum(float(r["estimated_eval_gpu_h_p90"]) for r in wave_rows), 2),
            "feature_gpu_h_point": 0.75,
            "feature_gpu_h_low": 0.60,
            "feature_gpu_h_high": 1.05,
            "total_gpu_h_point": round(
                sum(float(r["estimated_train_gpu_h_point"]) for r in wave_rows)
                + sum(float(r["estimated_eval_gpu_h_point"]) for r in wave_rows)
                + sum(float(r["estimated_feature_gpu_h_point"]) for r in wave_rows),
                2,
            ),
            "total_gpu_h_planning_low": round(
                sum(float(r["estimated_train_gpu_h_low"]) for r in wave_rows)
                + sum(float(r["estimated_eval_gpu_h_point"]) for r in wave_rows)
                + sum(float(r["estimated_feature_gpu_h_low"]) for r in wave_rows),
                2,
            ),
            "total_gpu_h_planning_high": round(
                sum(float(r["estimated_train_gpu_h_high"]) for r in wave_rows)
                + sum(float(r["estimated_eval_gpu_h_p90"]) for r in wave_rows)
                + sum(float(r["estimated_feature_gpu_h_high"]) for r in wave_rows),
                2,
            ),
            "checkpoint_disk_gib_low": round(sum(float(r["new_checkpoint_disk_gib_low"]) for r in wave_rows), 2),
            "checkpoint_disk_gib_high": round(sum(float(r["new_checkpoint_disk_gib_high"]) for r in wave_rows), 2),
            "feature_disk_gib_point": 40.69,
            "feature_disk_gib_budget": 45.0,
            "train_universe_ids": eventual_train_ids,
            "eval_universe_ids": eventual_eval_ids,
        },
        "default_b2_audit": {
            "registry_incomplete": 32,
            "target_summaries_present": 31,
            "prediction_artifacts_present": sum(bool(r["prediction_artifact_path"]) for r in reconciled if r["universe_id"] in b2_by_id),
            "protocol_blocked": 31,
            "interrupted_f3r_l24_p24": b2_partial_id,
        },
        "wan_cog_provisional_audit": {
            "rows": 162,
            "numeric": 162,
            "true_best_or_provenance_blocked": 162,
            "truly_missing_best": 157,
            "found_nearby_only": 5,
            "found_nearby_ids": sorted(found_nearby_ids),
        },
        "manifest_audit": {
            "frozen_rows_referencing_missing_filtered_manifest": sum(
                bool(r["train_manifest"]) and file_state(r["train_manifest"]) == "MISSING" for r in reconciled
            ),
            "known_missing_filtered_paths": dict(
                sorted(Counter(r["train_manifest"] for r in reconciled if r["train_manifest_status"] == "MISSING").items())
            ),
        },
        "variant_collision_groups": collision_group_json,
        "same_parent_variant_alternates": {
            uid: {
                "primary_registry_ids": [r["experiment_id"] for r in primary_registry_matches[uid]],
                "alternate_registry_ids": [r["experiment_id"] for r in alternate_registry_matches[uid]],
                "disposition": "SEPARATED_NOT_POOLED",
            }
            for uid in sorted(alternate_registry_matches)
        },
        "execution_sets": {
            "ready_to_package_now": [r["universe_id"] for r in reconciled if r["status"] == "RAW_RESULT_READY_TO_PACKAGE"],
            "eval_only": [r["universe_id"] for r in reconciled if r["status"] == "CHECKPOINT_READY_EVAL_ONLY"],
            "feature_extraction": [r["universe_id"] for r in reconciled if r["status"] == "FEATURE_PARTIAL_OR_MISSING"],
            "provenance_blocked": [r["universe_id"] for r in reconciled if r["status"] == "MANIFEST_OR_PROVENANCE_BLOCKED"],
        },
        "protected_source_hashes_before": protected_before,
        "acceptance": acceptance,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "universe_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Force physical rereads; do not reuse the SHA cache for the after check.
    protected_verified_after_outputs = {rel(p): sha256(p, refresh=True) for p in protected_paths}
    if protected_verified_after_outputs != protected_before:
        raise AssertionError("protected registry/canonical source changed while writing analysis outputs")

    # Report generation.
    family_rows = [
        (
            fam,
            data["universe"],
            data["canonical_complete"],
            data["numeric_target_present"],
            data["noncanonical"],
            ", ".join(f"{k}:{v}" for k, v in data["models"].items()),
            ", ".join(f"l{k}:{v}" for k, v in data["layers"].items()),
            ", ".join(f"p{k}:{v}" for k, v in data["prefixes"].items()),
            ", ".join(f"{k}:{v}" for k, v in data["campaigns"].items()),
            ", ".join(f"{k}:{v}" for k, v in data["status_counts"].items()),
        )
        for fam, data in family_coverage.items()
    ]
    status_rows = [(s, status_counts[s]) for s in PRIMARY_STATUSES]
    bottleneck_rows = [(b, bottleneck_counts.get(b, 0)) for b in BOTTLENECKS]
    wave_table_rows = []
    for wave, data in wave_aggregates.items():
        wave_table_rows.append(
            (
                wave,
                data["count"],
                f"{data['gpu_feature_jobs']}/{data['gpu_train_jobs']}/{data['gpu_eval_jobs']}",
                f"feature {data['feature_gpu_h_point']:.2f} ({data['feature_gpu_h_low']:.2f}–{data['feature_gpu_h_high']:.2f}); "
                f"train {data['train_gpu_h_point']:.2f} ({data['train_gpu_h_low']:.2f}–{data['train_gpu_h_high']:.2f}); "
                f"eval {data['eval_gpu_h_point']:.2f} (p90 {data['eval_gpu_h_p90']:.2f})",
                f"ckpt {data['checkpoint_disk_gib_low']:.1f}–{data['checkpoint_disk_gib_high']:.1f} GiB; "
                f"feature {data['feature_disk_gib_point']:.1f} GiB (budget {data['feature_disk_gib_budget']:.1f})",
            )
        )

    wave_distribution_lines = []
    for wave, data in wave_aggregates.items():
        if not data["count"]:
            wave_distribution_lines.append(f"- {wave}: 0")
            continue
        wave_distribution_lines.append(
            f"- {wave}: models=" + ", ".join(f"{k}:{v}" for k, v in data["models"].items())
            + "; families=" + ", ".join(f"{k}:{v}" for k, v in data["families"].items())
            + "; campaigns=" + ", ".join(f"{k}:{v}" for k, v in data["campaigns"].items())
            + "; layers=" + ", ".join(f"l{k}:{v}" for k, v in data["layers"].items())
            + "; prefixes=" + ", ".join(f"p{k}:{v}" for k, v in data["prefixes"].items())
        )
    wave_distribution_md = "\n".join(wave_distribution_lines)
    protected_hash_table = md_table(
        ["Protected source", "SHA256 before", "SHA256 after", "unchanged"],
        [(p, h, protected_verified_after_outputs[p], "YES") for p, h in protected_before.items()],
    )

    report = f"""# Q00：历史 1096 个非 diffusion-step 配置逐行 reconciliation

生成日期：2026-08-28（UTC）  
模式：严格只读对账；本轮没有启动训练、feature extraction、eval 或提交任务，也没有修改 registry/canonical。

## 结论先行

1096 universe 可以唯一冻结为 `{rel(UNIVERSE_SOURCE)}`（SHA256 `{sha256(UNIVERSE_SOURCE)}`）。三份 1096 行机器表的 full identity 集合完全相等；当前 1260 行 ledger 的 164 个 `expand20260822_` 新增项已全部排除。1096 中没有 t499/t999/t500/t1000 或 diffusion-step sweep；历史默认 t749/t250 保留在逐行 `default_timestep` 字段中。

按目标 metric contract 计，已有数值结果 **{numeric_target_count}/1096**：642 个 canonical-valid parent configurations、31 个 legacy B2 angle summaries、以及 162 个 Wan/Cog provisional rows（其中 2 个 parent 已有后续 canonical，故并集为 833）。F3R B2 l24/p24 的旧 `top1` 实际为 n/a，不是辅助数值，因此“任何 exact finite numeric”同样是 **{auxiliary_any_count}/1096**；完全没有任何数值的是 **{1096 - auxiliary_any_count}** 个配置（262 个无 registry representation + 该 F3R 行）。canonical-valid 是 **642/1096**，不是 761；761 是 atomic canonical ID 数量。

当前可直接 packaging 的严格数量是 **0**。31 个 default B2 虽有目标数值，但 split manifest 未绑定、样本量不一致且协议未冻结，必须先过 Wave E 的 split/protocol audit；通过后才转 Wave A。严格 checkpoint + manifest 条件下可 eval-only 的是 **99** 个；83 个 checkpoint 中断/无效，35 个 VideoMAE p24 缺 feature，237 个以 manifest/provenance/config 为首要阻塞。

## 1. Universe 冻结、来源和版本比较

{md_table(
    ["路径", "行数", "SHA256", "git 状态", "判定"],
    [
        (rel(UNIVERSE_SOURCE), 1096, sha256(UNIVERSE_SOURCE), git_status(UNIVERSE_SOURCE), "冻结 source of truth"),
        (rel(EXPECTED_SOURCE), 1096, sha256(EXPECTED_SOURCE), git_status(EXPECTED_SOURCE), "同一 full-identity 集；较少生命周期字段"),
        (rel(RECON_SOURCE), 1096, sha256(RECON_SOURCE), git_status(RECON_SOURCE), "同一 full-identity 集；reconstruction candidate"),
        (rel(EXPANDED_LEDGER), len(expanded), sha256(EXPANDED_LEDGER), git_status(EXPANDED_LEDGER), "含 164 个 post-freeze 扩展，不属于本轮"),
    ],
)}

冻结 full identity 是 `(campaign, model, model_variant, layer, prefix, probe, horizon, seed, protocol)`；`universe_id` 直接使用源表确定性的 `experiment_id`。1096 个 ID 和 1096 个 full identities 均唯一。源表没有原生 protocol 列，因此本报告显式加 namespace `{PROTOCOL}`；它是历史协议标签，不等于自动科学可比。

选择 pre-expand task ledger 而不是 reconstruction candidate 的依据是 `docs/FILE_MAP.md` 将 task ledger 标为 authoritative、candidate 标为 non-authoritative；pre-expand 文件又保留 48 个生命周期/provenance 字段。当前 1260 ledger 中原 1096 行的 48 个字段逐项相同，只有 164 个 `expand20260822_` additions（default8 156、wan_cog 8；Cog 132、Wan 32）。

生成 provenance：`{rel(RECON_SCRIPT)}`（`{sha256(RECON_SCRIPT)}`）生成 expected/reconstruction candidate；`{rel(LEDGER_SCRIPT)}`（`{sha256(LEDGER_SCRIPT)}`）生成生命周期 ledger；历史说明位于 `{rel(HISTORICAL_REPORT)}`（`{sha256(HISTORICAL_REPORT)}`）。这些关键文件当前均为 untracked 工作树文件，不能仅以 git commit 声称可复现；后续每次执行必须复制入口到 run provenance 并冻结 SHA。

历史脚本不能直接重用其状态：ledger 的 canonical/eval lookup 只用 `(model, layer, prefix, probe)`，忽略 campaign/model_variant/protocol；本报告完全重做 full-identity crosswalk。

## 2. Registry 与 canonical 精确对账

- Registry 有 955 IDs；928 个精确映射到 frozen universe 的 834 个 parent rows，27 个是 post-freeze Wan/Cog IDs，不属于 1096。
- 因为 A1 depth/camera atomic children 和少量 variant/collision IDs，928 个 in-universe registry IDs 比 parent rows 多 94 个。
- 所以真正没有任何 registry representation 的是 **262 个配置**。`1096 - 955 = 141` 只是不同粒度相减；正确恒等式是 `262 - 94 - 27 = 141`，不存在可列出的“141 个 missing configurations”。
- 761 个 canonical IDs 中，734 个映射回 642 个 frozen parent rows（732 个 primary mappings + 2 个 same-parent alternate mappings）；27 个是同一批 post-freeze extras。642 个 parent 的 expected atomic metrics 均齐全。
- 194 个 registry INCOMPLETE IDs 全部映射回 universe：162 个 Wan/Cog provisional + 32 个 default B2。它们对应 192 个 only-incomplete parents；另 2 个 parent 同时有后续 exact canonical A1 depth/camera IDs。

忽略 registry `model_variant` 后有 **21 collision groups / 42 COMPLETE IDs**。本报告保留每个 ID、campaign、variant 和 provenance，0 个被标成 exact duplicate alias。完整列表在 `universe_summary.json.variant_collision_groups` 和逐行 CSV 的 collision 字段中。

其中两组 Wan collision 映射到同一 frozen parent，但没有合并 artifact：B1 Wan l29/p24 选择原 `wan_cog_layer_sweep` 的 56.21144057 为 primary，`eval_20260817` 的 57.74463957 单列 alternate；B2 Wan l0/p24 同理选择 80.49305772，77.51404363 单列 alternate。逐行 CSV 的 `primary_registry_id` / `alternate_registry_id`、metric/raw/checkpoint alternate 字段完全分离；科学可比性只针对 primary variant。

## 3. 每个 task family 的覆盖矩阵

{md_table(["Family", "Universe", "Canonical-valid", "目标数值", "Noncanonical", "Models", "Layers", "Prefixes", "Campaigns", "状态分布"], family_rows)}

Family/probe 的目标 metric contract：A1 depth=`mean_loss_depth`+`Auc_30`，A1 mask=`mean_loss_identity`，A1 camera=`Auc_30`，A2=`balanced_acc`，B1=`mean_az_err_deg`，B2=`mean_ang_err_deg`，C1/C2=`global_R@1`，C3=`intervention_validity`。B1 没有独立 family canonical CSV，只在 `all_results.csv`；逐行表显式记录这一点。

## 4. 唯一主状态与瓶颈

每一行恰好一个主状态；以下数量之和严格为 1096：

{md_table(["主状态", "数量"], status_rows)}

`NOT_IN_REGISTRY=262` 是正交属性，不会覆盖更有信息量的 checkpoint/feature/provenance 主状态。瓶颈分布：

{md_table(["Bottleneck", "数量"], bottleneck_rows)}

逐行证据含 registry/canonical IDs、summary/prediction、checkpoint epoch/step、期望 final epoch、clean marker、manifest/config/log/run、feature roots 与 provenance。路径存在只证明 artifact existence，不自动证明 protocol validity。

## 5. 三个重点审计

### Default B2 的 32 个 INCOMPLETE

31/32 有 exact `val_summary.json`、`mean_ang_err_deg` 和 `val_predictions.pt`，summary 绑定的 checkpoint 存在且为 epoch 49 final。它们的数值可信到“legacy run 产生了这个数”，但 run manifest 的 split manifest 为 null，n 随 prefix/model 从 67 到 198 不等，而 registry 声称 `configs/splits/inscene15k_v1.json`。因此当前是 `MANIFEST_OR_PROVENANCE_BLOCKED`，不是 Wave A。

唯一例外 `{b2_partial_id}`：只有 epoch 27 checkpoint，日志进入 epoch 28 后中断，resume 含 stall/EADDRINUSE，无 summary/prediction/clean stop；状态为 `CHECKPOINT_INVALID_OR_INTERRUPTED`，先做 B2 protocol reconstruction，再 clean train/eval，绝不是 eval-only。

### Wan/Cog 162 provisional

162/162 属于 1096，162/162 有可解析 target numeric、历史 run/checkpoint 记录，但 0 个 formal summary，lineage 均为 `PROVISIONAL_ONLY_NO_REPLACEMENT`。旧 callback/retention bug 意味真正 best checkpoint 多数从未保存；“TRULY_MISSING”是 true-best 缺失，不是完全没有 checkpoint。它们不能 packaging。157 个先进入 Wave E；5 个 `FOUND_NEARBY_ONLY` 进入 Wave F 人工容差/协议决策。两个 provisional parent 已有后续 canonical，主状态仍为 canonical，旧数值只保留为 lineage evidence。

### 262 个没有 registry representation 的配置

- 99：现有 manifest + strict final checkpoint，Wave B eval-only；eval 前仍需 live AOSS feature canary。
- 66：manifest/feature 可用，但 checkpoint 未到 final 或无 clean completion，Wave C clean train+eval。
- 42：V-JEPA/VideoMAE checkpoint 存在但历史 filtered manifest 路径已丢失；37 个 checkpoint strict-pass、5 个 strict-fail，统一先 Wave E reconstruct。
- 35：VideoMAE p24 × 5 layers × 7 probes 的 `prefix_000023` feature hole，Wave D。
- 20：MiniMax camera-only config 当前缺失且 filtered manifests 丢失；16 个历史 run 中断、4 个只声明 job。不能假设等同 joint depth+camera，先 Wave E 做 config/protocol audit。

## 6. 执行 Waves（本轮未执行）

`execution_waves.csv` 是逐配置精确列表；以下是聚合。Jobs 列为 feature/train/eval。训练点估使用历史成功运行的 probe median，低/高区间为 0.5×/2× 规划带；eval 点估 0.05 GPU-h/row、p90 0.13。feature 成本在 canary 前不伪造。

{md_table(["Wave", "配置", "GPU jobs F/T/E", "GPU-hours", "新增 checkpoint 盘"], wave_table_rows)}

逐 wave 分布（完整 layer × 配置列表以 `execution_waves.csv` 为准）：

{wave_distribution_md}

- **Wave A：0 当前可直接 promotion。** 31 个 B2 只有在 Wave E 的 split/protocol/cardinality audit 通过后才转入 A。必须 staging copy + expected-ID-only diff + generalized atomic promotion + project consistency；不得直接写 protected tables。
- **Wave B：99。** 入口 `build_missing_eval_jobs.py` / A1 专用 builder；gate 是 strict final checkpoint、manifest SHA、feature root live canary、exact n/finite/identity、staging audit。任一不符即停止。
- **Wave C：66。** 复用 probe-fill chain 但必须改为只读入 frozen reconciliation、no-resume、strict clean-completion；现有宽松 done scanner 不可原样使用。
- **Wave D：35。** 5 个 layer cache groups 由一次共享五层 extraction 产生：1 个单 scene canary + 1 个 4×H800 full job，合计点估 0.75 H800-GPUh（0.60–1.05），新增 40.69 GiB（预留 45 GiB）。历史 runner 未传 `--num-frames 24`，而 extractor 默认 16 会静默过滤 p24；必须显式 `PREFIX_LENGTHS=24 --num-frames 24`，canary 看到 `prefix_000023` 五层后才可全量。历史成本证据为 `{rel(VIDEOMAE_SUPERVISOR)}`（`{sha256(VIDEOMAE_SUPERVISOR)}`）和 `{rel(VIDEOMAE_REPAIR_SYNC)}`（`{sha256(VIDEOMAE_REPAIR_SYNC)}`）。
- **Wave E：249。** provenance reconstruction 本身 0 GPU。包括 155 provisional、31 B2 summaries、42 missing-manifest checkpoints、20 MiniMax camera、1 F3R B2。重建失败则保持 blocked，不启动下游；成功后转 A/B/C。
- **Wave F：5。** `FOUND_NEARBY_ONLY` 必须签名的人为 protocol/tolerance disposition；不得以相似文件名或忽略 variant 的方式合并。

每个 wave 的入口脚本目前多为 untracked。硬 gate：`plan 中脚本 SHA == frozen provenance copy SHA == launcher 记录 SHA`；否则停止。canonical eligibility 还要求 full identity、protocol、manifest/root/checkpoint hashes、finite metric、expected n 以及 staging-only promotion diff 全部通过。

## 7. 最短补齐路径与成本

在“31 个 legacy B2 经审计可 packaging”的最短路径假设下：

- 31 packaging promotions（0 GPU）；
- 287 clean train jobs；
- 423 eval jobs；
- 5 个 VideoMAE layer cache groups，由 2 个 feature jobs（1 canary + 1 shared full extraction）产生；
- 合计 **712 GPU jobs**（287 train + 423 eval + 2 feature）；
- 训练点估 **{summary['eventual_shortest_path']['train_gpu_h_point']:.2f} GPU-h**，规划带 **{summary['eventual_shortest_path']['train_gpu_h_low']:.2f}–{summary['eventual_shortest_path']['train_gpu_h_high']:.2f} GPU-h**；
- eval 点估 **{summary['eventual_shortest_path']['eval_gpu_h_point']:.2f} GPU-h**，按历史 p90 **{summary['eventual_shortest_path']['eval_gpu_h_p90']:.2f} GPU-h**；
- feature extraction 点估 **0.75 H800-GPUh**（0.60–1.05），新增 **40.69 GiB**、预算 45 GiB；canary 不通过则停止，不能授权 full job；
- 已知总点估 **{summary['eventual_shortest_path']['total_gpu_h_point']:.2f} GPU-h**，规划 envelope **{summary['eventual_shortest_path']['total_gpu_h_planning_low']:.2f}–{summary['eventual_shortest_path']['total_gpu_h_planning_high']:.2f} GPU-h**（训练 + eval + feature；硬件代际混合，需按实际队列换算）；
- protected best+last checkpoint 预计新增 **{summary['eventual_shortest_path']['checkpoint_disk_gib_low']:.1f}–{summary['eventual_shortest_path']['checkpoint_disk_gib_high']:.1f} GiB**；加 feature 点估后合计 **155.5–574.5 GiB**（按 45 GiB feature budget 则预留 159.8–578.8 GiB）。

如果 31 个 B2 legacy protocol 不能证明等价，最短路径再增加 31 个 exact-manifest eval jobs（点估 1.55 GPU-h，p90 4.03），但通常不需要重训其 epoch-49 checkpoint。

## 8. 哪些不能直接跑、哪些不应静默删除

- 不应先扩 sweep；先补齐这 1096。
- 不应直接运行旧 probe-fill done scanner；它会把“有 checkpoint/eval 或 END rc=0”误算为完成。
- 不应把 162 provisional 或 31 B2 summary 当 packaging-ready。
- 不应把 F3R B2 p24 当 eval-only。
- 不应把 MiniMax camera 静默视作 joint depth/camera alias。
- 不应合并 21 个 model_variant collision groups。
- 没有任何配置因“价值低”被删；deprecated/invalid 目前为 0，所有 1096 行均保留。

## 9. 保护文件完整性

生成脚本在写分析产物前后重新计算 registry 和全部 canonical CSV 的 SHA256，并要求完全一致。执行后哈希见 `universe_summary.json.protected_source_hashes_before/after`。本报告及五个分析表之外没有实验状态写入。

{protected_hash_table}

## 10. 对九个执行问题的直接回答

1. **已有目标数值 833；任何 exact finite numeric 也是 833。** F3R B2 l24/p24 的 top1 是 n/a，不构成数值结果。
2. **Canonical-valid 642 个 parent configs。** 734 个 in-universe atomic IDs 映射到这 642 行，其中 732 个 primary、2 个 same-parent alternate 已分栏；另有 27 个 canonical IDs 在 frozen universe 外。
3. **完全没有任何数值 263 个。** 包括 262 个无 registry representation rows，加上有 INCOMPLETE registry 但没有任何 finite metric 的 F3R B2 l24/p24；逐项见 CSV 过滤 `auxiliary_numeric_present=false`。
4. **现在只差 packaging 的严格数量 0。** 31 个 B2 是“通过 protocol audit 后条件性 packaging”。
5. **只差 eval 99。** 精确列表为 `execution_waves.csv` 的 Wave B。
6. **formal 补齐计划需要 clean training 287。** 包括条件性通过 E/F gate 后的 replacement jobs；逐项看 `gpu_train_jobs=1`。
7. **需要重新抽 feature 35 个配置 / 5 个共享 VideoMAE cache groups。** 全部是 p24 的 5 layers × 7 probes。
8. **主状态 manifest/provenance blocked 237。** 另有 17 个以 checkpoint/config 为主状态但也必须先经过 Wave E/F gate，因此 Wave E/F 合计 254。
9. **最短路径：31 条件性 packaging + 287 train + 423 eval + 2 feature jobs（产出 5 cache groups）。** 共 712 GPU jobs；训练点估 6168.22 GPU-h、规划带 3083.96–12336.44，eval 21.15（p90 54.99），feature 0.75（0.60–1.05），总点估 {summary['eventual_shortest_path']['total_gpu_h_point']:.2f}、总规划 envelope {summary['eventual_shortest_path']['total_gpu_h_planning_low']:.2f}–{summary['eventual_shortest_path']['total_gpu_h_planning_high']:.2f} GPU-h。

## 11. 硬性验收

下面的 `NUMERIC_RESULT_PRESENT` 指“存在该 family 的目标 metric”；F3R B2 的 top1=n/a。`NEEDS_TRAINING`、`NOT_IN_REGISTRY`、collision 是正交执行/审计计数；唯一主状态的 partition 见第 4 节。

```text
UNIVERSE_SIZE: {acceptance['UNIVERSE_SIZE']}
UNIVERSE_SOURCE: {acceptance['UNIVERSE_SOURCE']}
UNIVERSE_SHA256: {acceptance['UNIVERSE_SHA256']}
DIFFUSION_STEP_SWEEP_EXCLUDED: {str(acceptance['DIFFUSION_STEP_SWEEP_EXCLUDED']).lower()}
CANONICAL_COMPLETE: {acceptance['CANONICAL_COMPLETE']}
NUMERIC_RESULT_PRESENT: {acceptance['NUMERIC_RESULT_PRESENT']}
REGISTRY_INCOMPLETE: {acceptance['REGISTRY_INCOMPLETE']}
NOT_IN_REGISTRY: {acceptance['NOT_IN_REGISTRY']}
READY_TO_PACKAGE: {acceptance['READY_TO_PACKAGE']}
EVAL_ONLY: {acceptance['EVAL_ONLY']}
NEEDS_TRAINING: {acceptance['NEEDS_TRAINING']}
NEEDS_FEATURE_EXTRACTION: {acceptance['NEEDS_FEATURE_EXTRACTION']}
PROVENANCE_BLOCKED: {acceptance['PROVENANCE_BLOCKED']}
DUPLICATE_OR_COLLISION: {acceptance['DUPLICATE_OR_COLLISION']}
DEPRECATED_OR_INVALID: {acceptance['DEPRECATED_OR_INVALID']}
UNRESOLVED: {acceptance['UNRESOLVED']}
RESOLVED_TOTAL: {acceptance['RESOLVED_TOTAL']}
COUNT_SUM_EQUALS_1096: {str(acceptance['COUNT_SUM_EQUALS_1096']).lower()}
```

逐行答案位于 `results/analysis/q00_1096/universe_1096_reconciled.csv`；454 个缺失/非 canonical 配置位于 `missing_or_noncanonical.csv`；精确 waves 与成本位于 `execution_waves.csv`。
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")

    protected_after = {rel(p): sha256(p, refresh=True) for p in protected_paths}
    if protected_after != protected_before:
        raise AssertionError("protected registry/canonical source changed during reconciliation")
    summary["protected_source_hashes_after"] = protected_after
    summary["protected_source_hashes_unchanged"] = True
    (OUT_DIR / "universe_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "universe": len(reconciled),
        "canonical": status_counts["CANONICAL_COMPLETE"],
        "numeric_target": numeric_target_count,
        "numeric_any": auxiliary_any_count,
        "missing_or_noncanonical": len(missing_rows),
        "status_counts": dict(status_counts),
        "wave_counts": dict(wave_counts),
        "protected_unchanged": protected_after == protected_before,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
