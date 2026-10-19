#!/usr/bin/env python3
"""Build the prefix x probe x timestep training matrix from completion indexes.

This is a metadata-only, fail-closed gate: it reads frozen JSONL manifests,
local done-index text files, and tiny canonical index/provenance files. It
never downloads an AOSS feature object and never uses AOSS HEAD on the
training hot path.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

try:
    from scripts.audit_tstep_canonical_alignment import (
        AUDIT_SCHEMA_VERSION,
        AUDIT_TYPE,
        DEFAULT_CANONICAL_OBS_LEN,
        DEFAULT_HORIZONS,
        DEFAULT_PREFIXES,
        DEFAULT_PROTOCOL_ID,
        DEFAULT_PROTOCOL_SHA256,
        NAMESPACE_PAIRS,
        audit_alignment,
        namespace_pair_errors,
        sha256_file,
        write_json_atomic,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from audit_tstep_canonical_alignment import (  # type: ignore[no-redef]
        AUDIT_SCHEMA_VERSION,
        AUDIT_TYPE,
        DEFAULT_CANONICAL_OBS_LEN,
        DEFAULT_HORIZONS,
        DEFAULT_PREFIXES,
        DEFAULT_PROTOCOL_ID,
        DEFAULT_PROTOCOL_SHA256,
        NAMESPACE_PAIRS,
        audit_alignment,
        namespace_pair_errors,
        sha256_file,
        write_json_atomic,
    )


MODEL_TIMESTEPS = {
    "wan": (499, 999),
    "cogvideox": (499, 999),
    "minimax_h3": (500, 1000),
}
STREAM_TASKS = (
    "streaming_depth",
    "streaming_movement",
    "streaming_mask",
    "view_consistency",
    "ego_belief",
    "ego_belief_v2",
)
TARGET_TASKS = ("action_dynamics", "path_integration", "counterfactual")
ALL_TASKS = STREAM_TASKS + TARGET_TASKS
STREAMING_NAMESPACE = "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
TARGET_NAMESPACE = "FEAT_TARGET_CANONICAL_V2_SEEDED_V1"


def words(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def ints(value: str) -> list[int]:
    return [int(item) for item in words(value)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--done-index-dir", default="logs/tstep_canonical_v2_seeded_v1/aoss_done"
    )
    parser.add_argument("--streaming-namespace", default=STREAMING_NAMESPACE)
    parser.add_argument("--target-namespace", default=TARGET_NAMESPACE)
    parser.add_argument(
        "--manifest-dir",
        default="artifacts/tstep_canonical_v2/manifests",
    )
    parser.add_argument(
        "--strict-audit-dir",
        default="artifacts/tstep_canonical_v2/audits",
    )
    parser.add_argument(
        "--streaming-index-root",
        default=(
            "/mnt/umm/users/yinbaiqiao/InsScene-15K/"
            "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
        ),
    )
    parser.add_argument(
        "--target-index-root",
        default=(
            "/mnt/umm/users/yinbaiqiao/InsScene-15K/"
            "FEAT_TARGET_CANONICAL_V2_SEEDED_V1"
        ),
    )
    parser.add_argument("--source", default="scannetpp")
    parser.add_argument("--protocol-id", default=DEFAULT_PROTOCOL_ID)
    parser.add_argument("--canonical-obs-len", type=int, default=DEFAULT_CANONICAL_OBS_LEN)
    parser.add_argument("--target-horizons", default=" ".join(map(str, DEFAULT_HORIZONS)))
    parser.add_argument("--vfms", default="wan cogvideox minimax_h3")
    parser.add_argument("--layers", default="15 20 25")
    parser.add_argument("--prefixes", default="8 12 16 24")
    parser.add_argument(
        "--tasks",
        default=" ".join(ALL_TASKS),
        help="probe subset to schedule (kept in canonical order); the 2026-09 "
        "timestep campaigns trained the 8 tasks without streaming_movement",
    )
    parser.add_argument(
        "--streaming-exclusions",
        default="configs/experiment/inscene15k_streaming/tstep_canonical_v2_exclusions.txt",
        help="v2-only exclusion list; do not point this at the legacy tstep exclusions",
    )
    return parser.parse_args()


def load_expected(manifest: Path, expected_source: str = "scannetpp") -> tuple[set[str], set[str]]:
    streaming_units: set[str] = set()
    scenes: set[str] = set()
    with manifest.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            scene = str(row.get("scene_id") or Path(str(row["scene_dir"])).name)
            source = str(row.get("source", "scannetpp"))
            if source != expected_source:
                raise ValueError(
                    f"{manifest}:{lineno}: source={source!r}, expected {expected_source!r}"
                )
            window = int(row.get("streaming_window_id", row.get("window_id")))
            tail_value = row.get("streaming_tail", row.get("tail"))
            if tail_value is None:
                tail_value = int(row["prefix_length"]) - 1
            tail = int(tail_value)
            streaming_units.add(
                f"{scene}/window_{window:04d}/prefix_{tail:06d}"
            )
            scenes.add(scene)
    if not streaming_units:
        raise ValueError(f"empty manifest: {manifest}")
    return streaming_units, scenes


def load_feature_units(index: Path, timestep: int, layers: list[int]) -> dict[int, set[str]]:
    by_layer: dict[int, set[str]] = defaultdict(set)
    if not index.is_file():
        return by_layer
    wanted = set(layers)
    pattern = re.compile(rf"feature_t{timestep}_layer(\d+)\.sft$")
    with index.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            rel = line.strip()
            match = pattern.search(rel.rsplit("/", 1)[-1])
            if not match:
                continue
            layer = int(match.group(1))
            if layer in wanted and "/" in rel:
                by_layer[layer].add(rel.rsplit("/", 1)[0])
    return by_layer


def local_target_index_scenes(
    root: Path, vfm: str, source: str, expected: set[str]
) -> set[str]:
    base = root / vfm / source
    return {scene for scene in expected if (base / scene / "target_indices.npy").is_file()}


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir).resolve()
    done_dir = (project_root / args.done_index_dir).resolve()
    manifest_dir = (project_root / args.manifest_dir).resolve()
    strict_audit_dir = (project_root / args.strict_audit_dir).resolve()
    streaming_index_root = Path(args.streaming_index_root).resolve()
    target_index_root = Path(args.target_index_root).resolve()
    exclusions_path = (
        (project_root / args.streaming_exclusions).resolve()
        if args.streaming_exclusions
        else None
    )
    vfms = words(args.vfms)
    layers = ints(args.layers)
    prefixes = ints(args.prefixes)
    target_horizons = ints(args.target_horizons)

    namespace_pattern = re.compile(r"^[A-Z0-9_]+$")
    for label, namespace in (
        ("--streaming-namespace", args.streaming_namespace),
        ("--target-namespace", args.target_namespace),
    ):
        if not namespace_pattern.fullmatch(namespace):
            raise SystemExit(f"{label} must contain only A-Z, 0-9, and underscore")
    namespace_pair = (args.streaming_namespace, args.target_namespace)
    if namespace_pair not in NAMESPACE_PAIRS:
        raise SystemExit(
            f"--streaming-namespace/--target-namespace must be one of the seeded-v1 "
            f"pairs {sorted(NAMESPACE_PAIRS)}; got {namespace_pair}"
        )
    namespace_tag = NAMESPACE_PAIRS[namespace_pair]["tag"]
    if streaming_index_root.name != args.streaming_namespace:
        raise SystemExit(
            f"--streaming-index-root must end in {args.streaming_namespace}"
        )
    if target_index_root.name != args.target_namespace:
        raise SystemExit(f"--target-index-root must end in {args.target_namespace}")
    if args.protocol_id != DEFAULT_PROTOCOL_ID:
        raise SystemExit(f"--protocol-id must be {DEFAULT_PROTOCOL_ID}")
    if args.canonical_obs_len != DEFAULT_CANONICAL_OBS_LEN:
        raise SystemExit(
            f"--canonical-obs-len must be {DEFAULT_CANONICAL_OBS_LEN}"
        )
    if target_horizons != list(DEFAULT_HORIZONS):
        raise SystemExit(
            f"--target-horizons must be {' '.join(map(str, DEFAULT_HORIZONS))}"
        )

    unknown = sorted(set(vfms) - set(MODEL_TIMESTEPS))
    if unknown:
        raise SystemExit(f"unsupported VFM(s): {', '.join(unknown)}")
    if not layers or not prefixes:
        raise SystemExit("--layers and --prefixes must be non-empty")
    for vfm in vfms:
        errors = namespace_pair_errors(args.streaming_namespace, args.target_namespace, vfm, layers)
        if errors:
            raise SystemExit("; ".join(errors))
    requested_tasks = words(args.tasks)
    unknown_tasks = sorted(set(requested_tasks) - set(ALL_TASKS))
    if unknown_tasks or not requested_tasks or len(set(requested_tasks)) != len(requested_tasks):
        raise SystemExit(f"--tasks must be distinct names from {list(ALL_TASKS)}; got {requested_tasks}")
    selected_tasks = tuple(task for task in ALL_TASKS if task in requested_tasks)
    selected_stream_tasks = tuple(task for task in selected_tasks if task in STREAM_TASKS)

    exclusions: set[str] = set()
    if exclusions_path is not None and exclusions_path.is_file():
        exclusions = {
            line.strip()
            for line in exclusions_path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

    expected_by_prefix: dict[int, tuple[Path, set[str], set[str]]] = {}
    for prefix in prefixes:
        manifest = manifest_dir / f"inscene_c_h124_prefix{prefix}.jsonl"
        if not manifest.is_file():
            raise SystemExit(f"missing frozen manifest: {manifest}")
        units, scenes = load_expected(manifest, args.source)
        units -= exclusions
        expected_by_prefix[prefix] = (manifest, units, scenes)

    # One full semantic audit per (VFM, prefix) is enough for all layers and
    # timesteps.  These reports are regenerated atomically with the matrix and
    # their hashes are embedded in every trainable row.
    strict_by_vfm_prefix: dict[tuple[str, int], tuple[Path, dict[str, object], str]] = {}
    for vfm in vfms:
        for prefix in prefixes:
            manifest = expected_by_prefix[prefix][0]
            # Non-default namespace pairs get their own file names so that a
            # mis-pointed --strict-audit-dir can never overwrite (and thereby
            # invalidate the recorded SHA-256 of) a seeded-v1 audit.
            suffix = "" if namespace_tag == "seeded_v1" else f".{namespace_tag}"
            audit_path = strict_audit_dir / f"{vfm}_{manifest.stem}{suffix}.audit.json"
            try:
                report = audit_alignment(
                    manifest=manifest,
                    streaming_index_root=streaming_index_root,
                    target_index_root=target_index_root,
                    vfm=vfm,
                    source=args.source,
                    protocol_id=args.protocol_id,
                    canonical_obs_len=args.canonical_obs_len,
                    prefixes=DEFAULT_PREFIXES,
                    horizons=target_horizons,
                    streaming_namespace=args.streaming_namespace,
                    target_namespace=args.target_namespace,
                )
            except Exception as exc:
                report = {
                    "schema_version": AUDIT_SCHEMA_VERSION,
                    "audit_type": AUDIT_TYPE,
                    "ok": False,
                    "pass": False,
                    "status": "fail",
                    "scope": "full",
                    "protocol_id": args.protocol_id,
                    "protocol": args.protocol_id,
                    "canonical_obs_len": args.canonical_obs_len,
                    "prefixes": list(DEFAULT_PREFIXES),
                    "horizons": target_horizons,
                    "include_anchors": True,
                    "vfm": vfm,
                    "source": args.source,
                    "manifest": str(manifest.resolve()),
                    "manifest_sha256": sha256_file(manifest),
                    "manifest_sample_count": 0,
                    "manifest_scene_count": 0,
                    "samples_checked": 0,
                    "scenes_checked": 0,
                    "streaming_namespace": args.streaming_namespace,
                    "target_namespace": args.target_namespace,
                    "problem_count": 1,
                    "problem_counts": {"audit_internal_error": 1},
                    "problems": [{"kind": "audit_internal_error", "detail": repr(exc)}],
                }
            write_json_atomic(audit_path, report)
            strict_by_vfm_prefix[(vfm, prefix)] = (
                audit_path.resolve(),
                report,
                sha256_file(audit_path),
            )

    rows: list[dict[str, object]] = []
    summaries: dict[str, object] = {}
    for vfm in vfms:
        for task in ALL_TASKS:
            config = (
                project_root
                / "configs/experiment/inscene15k_streaming"
                / f"{task}_{vfm}_v1.yaml"
            )
            if not config.is_file():
                raise SystemExit(f"missing experiment config: {config}")

        all_expected_scenes = set().union(
            *(expected_by_prefix[prefix][2] for prefix in prefixes)
        )
        # A listed remote key proves only that a file exists.  It cannot prove
        # sample-level semantics, so only locally audited tiny indexes count.
        target_index_scenes = local_target_index_scenes(
            target_index_root, vfm, args.source, all_expected_scenes
        )

        for timestep in MODEL_TIMESTEPS[vfm]:
            stream_file = done_dir / f"{args.streaming_namespace}_{vfm}_t{timestep}.txt"
            target_file = done_dir / f"{args.target_namespace}_{vfm}_t{timestep}.txt"
            stream_by_layer = load_feature_units(stream_file, timestep, layers)
            target_by_layer = load_feature_units(target_file, timestep, layers)

            for layer in layers:
                for prefix in prefixes:
                    manifest, expected_units, expected_scenes = expected_by_prefix[prefix]
                    stream_found = len(expected_units & stream_by_layer[layer])
                    target_found = len(expected_scenes & target_by_layer[layer])
                    target_indices_found = len(expected_scenes & target_index_scenes)
                    strict_audit, strict_report, strict_audit_sha256 = strict_by_vfm_prefix[
                        (vfm, prefix)
                    ]
                    strict_ok = strict_report.get("ok") is True
                    stream_ready = stream_found == len(expected_units)
                    target_ready = (
                        target_found == len(expected_scenes)
                        and target_indices_found == len(expected_scenes)
                        and strict_ok
                    )

                    if not stream_ready:
                        status = "wait_streaming"
                        tasks: tuple[str, ...] = ()
                        reason = f"streaming {stream_found}/{len(expected_units)}"
                    elif not strict_ok:
                        status = "wait_strict_audit"
                        tasks = ()
                        problem_counts = strict_report.get("problem_counts", {})
                        reason = (
                            f"strict canonical audit failed: "
                            f"{strict_report.get('problem_count', 'unknown')} problem(s); "
                            f"{json.dumps(problem_counts, sort_keys=True)}"
                        )
                    elif target_ready:
                        status = "ready_full"
                        tasks = selected_tasks
                        reason = ""
                    else:
                        status = "ready_streaming_only"
                        tasks = selected_stream_tasks
                        reason = (
                            f"target_features {target_found}/{len(expected_scenes)}; "
                            f"target_indices {target_indices_found}/{len(expected_scenes)}"
                        )

                    group_id = f"{vfm}_t{timestep}_l{layer}_p{prefix}"
                    rows.append(
                        {
                            "group_id": group_id,
                            "status": status,
                            "vfm": vfm,
                            "source": args.source,
                            "timestep": timestep,
                            "layer": layer,
                            "prefix": prefix,
                            "manifest": str(manifest),
                            "tasks": ",".join(tasks),
                            "stream_expected": len(expected_units),
                            "stream_found": stream_found,
                            "target_expected": len(expected_scenes),
                            "target_found": target_found,
                            "target_indices_found": target_indices_found,
                            "canonical_protocol": args.protocol_id,
                            "canonical_obs_len": args.canonical_obs_len,
                            "streaming_namespace": args.streaming_namespace,
                            "target_namespace": args.target_namespace,
                            "strict_audit": str(strict_audit),
                            "strict_audit_sha256": strict_audit_sha256,
                            "strict_audit_ok": str(strict_ok).lower(),
                            "strict_audit_problem_count": strict_report.get("problem_count", ""),
                            "reason": reason,
                        }
                    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    all_path = output_dir / "groups.tsv"
    tmp = all_path.with_name(f".{all_path.name}.tmp-{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, all_path)

    for status in (
        "ready_full",
        "ready_streaming_only",
        "wait_streaming",
        "wait_strict_audit",
    ):
        status_path = output_dir / f"{status}.tsv"
        with status_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(row for row in rows if row["status"] == status)

    status_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        status_counts[str(row["status"])] += 1
    summaries = {
        "groups": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "training_runs_if_full": len(rows) * len(selected_tasks),
        "tasks": list(selected_tasks),
        "namespace_pair_tag": namespace_tag,
        "vfms": vfms,
        "layers": layers,
        "prefixes": prefixes,
        "done_index_dir": str(done_dir),
        "streaming_namespace": args.streaming_namespace,
        "target_namespace": args.target_namespace,
        "target_index_root": str(target_index_root),
        "streaming_index_root": str(streaming_index_root),
        "canonical_protocol": args.protocol_id,
        "canonical_protocol_sha256": DEFAULT_PROTOCOL_SHA256,
        "canonical_obs_len": args.canonical_obs_len,
        "strict_audit_dir": str(strict_audit_dir.resolve()),
        "target_horizons": target_horizons,
        "streaming_exclusions": str(exclusions_path) if exclusions_path is not None else "",
        "streaming_exclusion_units": sorted(exclusions),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summaries, indent=2, sort_keys=True))
    print(f"wrote {all_path}")


if __name__ == "__main__":
    main()
