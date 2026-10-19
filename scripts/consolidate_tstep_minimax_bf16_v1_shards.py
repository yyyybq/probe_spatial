#!/usr/bin/env python3
"""Consolidate the 32 production shards of the MiniMax-H3 bf16 seeded-v1 re-extraction.

Stage-3 exit step of docs/MINIMAX_REEXTRACT_RETRAIN_PLAN_20261016.md.  Each of
the 32 one-GPU shard jobs (2 timesteps x 2 modes x 8 scene shards, see
scripts/run_tstep_minimax_bf16_v1_extract_shard.sh) owns a local root, a
done-index and a step lock.  This script

  1. checks every shard: its supervisor log ends in OK with extract_fail=0,
     sync_fail=0 and local_matching_sft_left=0, and its done-index holds
     exactly the layer-20/25 units that the fp16 seeded-v1 index holds for the
     scenes of the shard's span (nothing missing, nothing extra);
  2. merges the 8 shard indices of each (namespace, timestep) with the
     stage-1 canary's scene-0 entries into
     logs/tstep_canonical_v2_seeded_v1/aoss_done/<NS>_minimax_h3_t<T>.txt;
  3. copies the keep_local structural files (streaming: prefix_index.npy and
     its .manifest.json; target: target_indices.npy, its .manifest.json and
     target_indices.meta.json) from the shard roots into
     InsScene-15K/<NS>/minimax_h3/scannetpp/<scene>/.  The t500 and t1000
     shards each wrote one copy per scene; the two must be byte-identical or
     the script refuses.  Every copy is also compared with the fp16 seeded-v1
     original (.npy byte-identical, JSON equal up to feature_namespace).

--check (default) only reads and writes the report; it is safe on the login
node and doubles as a progress view while shards run.  --apply performs steps
2-3 only if the check found no problem, and must run as root inside an SCO job
(the shard trees and the canonical scene directories were created by root-owned
jobs).  Structural files are written before the indices; every write is atomic
(temp file + rename, mode 0644); an existing destination is never replaced by
different bytes; re-running --apply after a successful run changes nothing.
The previous canonical index contents are saved next to the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/mnt/umm/users/yinbaiqiao/InsScene-15K")
LEDGER = ROOT / "logs/tstep_minimax_bf16_v1/submissions.json"
DONE_DIR = ROOT / "logs/tstep_canonical_v2_seeded_v1/aoss_done"
SHARD_DIRNAME = "tstep_minimax_bf16_v1_shards"
N_SCENES = 682
SHARD_COUNT = 8
CANARY_SCENE = "00777c41d4"  # scene 0, published by the stage-1 canary
TIMESTEPS = (500, 1000)
LAYERS = (20, 25)
NAMESPACES = {  # mode -> (new bf16 namespace, fp16 seeded-v1 original)
    "stream": ("FEAT_STREAMING_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1", "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"),
    "target": ("FEAT_TARGET_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1", "FEAT_TARGET_CANONICAL_V2_SEEDED_V1"),
}
STRUCTURAL = {
    "stream": ("prefix_index.npy", "prefix_index.npy.manifest.json"),
    "target": ("target_indices.npy", "target_indices.npy.manifest.json", "target_indices.meta.json"),
}
LAYER_RE = re.compile(r"feature_t(\d+)_layer(\d+)\.sft$")
MAX_EXAMPLES = 5


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_index(path: Path) -> set[str]:
    if not path.is_file() or path.is_symlink():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def unit_layer(rel: str, timestep: int) -> int | None:
    match = LAYER_RE.search(rel)
    if not match or int(match.group(1)) != timestep:
        return None
    return int(match.group(2))


def strip_namespace(value):
    if isinstance(value, dict):
        return {k: strip_namespace(v) for k, v in value.items() if k != "feature_namespace"}
    if isinstance(value, list):
        return [strip_namespace(v) for v in value]
    return value


def collect_scenes(dataset_root: Path) -> list[str]:
    """Scene order of features/run_inscene15k.py:collect_scannetpp_scenes.

    The extractor sorts os.listdir and drops directories without images/,
    refined_ins_ids/ or scene_iphone_metadata.npz (plus finer per-frame checks).
    If this cheaper filter yields the extractor's count, the orders agree.
    """
    source = dataset_root / "data/processed_scannetpp_v2"
    if (source / "processed_scannetpp_v2").is_dir():
        source = source / "processed_scannetpp_v2"
    scenes = []
    for name in sorted(os.listdir(source)):
        scene = source / name
        if (scene / "images").is_dir() and (scene / "refined_ins_ids").is_dir() and (
            scene / "scene_iphone_metadata.npz"
        ).exists():
            scenes.append(name)
    return scenes


def shard_span(index: int, count: int, n_scenes: int) -> tuple[int, int]:
    start, end = n_scenes * index // count, n_scenes * (index + 1) // count
    return (1 if start == 0 else start), end


def supervisor_status(log_dir: Path) -> dict:
    """Parse the frozen launcher's closing lines (finished_at ... / OK)."""
    path = log_dir / "supervisor.log"
    status = {"supervisor_log": str(path), "finished": False, "ok": False}
    if not path.is_file():
        status["problem"] = "supervisor.log missing (job not started?)"
        return status
    text = path.read_bytes().replace(b"\0", b"").decode("utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    finished = [line for line in lines if line.startswith("finished_at=")]
    if not finished:
        status["problem"] = "no finished_at line (still running or killed)"
        return status
    status["finished"] = True
    last = finished[-1]
    status["finished_line"] = last
    fields = dict(item.split("=", 1) for item in last.split() if "=" in item)
    for key, pattern in (("done_index_lines", r"^done_index_lines=(\d+)$"),
                         ("local_matching_sft_left", r"^local_matching_sft_left=(\d+)$")):
        values = [int(m.group(1)) for line in lines for m in [re.match(pattern, line)] if m]
        status[key] = values[-1] if values else None
    status["extract_fail"] = fields.get("extract_fail")
    status["sync_fail"] = fields.get("sync_fail")
    status["last_line"] = lines[-1]
    status["extract_finished_marker"] = (log_dir / "extract.finished").exists()
    status["ok"] = (
        status["extract_fail"] == "0" and status["sync_fail"] == "0"
        and status["local_matching_sft_left"] == 0 and lines[-1] == "OK"
    )
    if not status["ok"]:
        status["problem"] = f"supervisor did not finish cleanly: {last!r}, last line {lines[-1]!r}"
    return status


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    if sha256_bytes(path.read_bytes()) != sha256_bytes(data):
        raise RuntimeError(f"read-back mismatch after writing {path}")


def compare_with_original(name: str, data: bytes, original: Path, ns_new: str) -> str | None:
    if not original.is_file():
        return f"fp16 original missing: {original}"
    old = original.read_bytes()
    if name.endswith(".json"):
        new_json, old_json = json.loads(data), json.loads(old)
        if new_json.get("feature_namespace") != ns_new:
            return f"feature_namespace {new_json.get('feature_namespace')!r} != {ns_new!r}"
        a, b = strip_namespace(new_json), strip_namespace(old_json)
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        return f"differs from fp16 original in {diff}" if diff else None
    return None if data == old else "not byte-identical to the fp16 original"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="write indices and copy structural files (root, SCO job)")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--done-dir", type=Path, default=DONE_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--scene-list", type=Path, help="one scene id per line (default: scan the dataset)")
    parser.add_argument("--n-scenes", type=int, default=N_SCENES)
    parser.add_argument("--shard-count", type=int, default=SHARD_COUNT)
    parser.add_argument("--canary-scene", default=CANARY_SCENE)
    args = parser.parse_args()

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report: dict = {
        "schema": "TSTEP_MINIMAX_BF16_V1_CONSOLIDATION_V1",
        "mode": "apply" if args.apply else "check",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "started_utc": started, "uid": os.getuid(), "host": os.uname().nodename,
        "ledger": str(args.ledger), "done_dir": str(args.done_dir), "dataset_root": str(args.dataset_root),
        "problems": [], "shards": {}, "steps": {}, "structural": {},
    }
    problems: list[str] = report["problems"]

    scenes = (
        [s.strip() for s in args.scene_list.read_text(encoding="utf-8").splitlines() if s.strip()]
        if args.scene_list else collect_scenes(args.dataset_root)
    )
    report["scene_list_count"] = len(scenes)
    if len(scenes) != args.n_scenes:
        problems.append(f"scene list has {len(scenes)} scenes, extractor saw {args.n_scenes}; shard spans unknown")
    if scenes and scenes[0] != args.canary_scene:
        problems.append(f"scene 0 is {scenes[0]!r}, expected canary scene {args.canary_scene!r}")
    shard_of_scene: dict[str, int] = {}
    for index in range(args.shard_count):
        start, end = shard_span(index, args.shard_count, args.n_scenes)
        for scene in scenes[start:end]:
            shard_of_scene[scene] = index

    jobs = json.loads(args.ledger.read_text(encoding="utf-8"))["jobs"]
    copies: list[tuple[Path, bytes]] = []
    index_writes: list[tuple[Path, set[str]]] = []

    for mode, (ns_new, ns_old) in NAMESPACES.items():
        old_by_t: dict[int, dict[str, set[str]]] = {}
        for t in TIMESTEPS:
            key = f"t{t}_{mode}"
            old = {rel for rel in read_index(args.done_dir / f"{ns_old}_minimax_h3_t{t}.txt")
                   if unit_layer(rel, t) in LAYERS}
            old_by_scene: dict[str, set[str]] = defaultdict(set)
            for rel in old:
                old_by_scene[rel.split("/", 1)[0]].add(rel)
            old_by_t[t] = old_by_scene
            canon_path = args.done_dir / f"{ns_new}_minimax_h3_t{t}.txt"
            canon = read_index(canon_path)
            canary = {rel for rel in canon if rel.startswith(args.canary_scene + "/")}
            step = {"fp16_units": len(old), "fp16_scenes": len(old_by_scene),
                    "canonical_before": len(canon), "canary_units": len(canary), "shards": {}}
            if canary != old_by_scene.get(args.canary_scene, set()):
                problems.append(f"{key}: canary scene-0 entries ({len(canary)}) differ from fp16 "
                                f"({len(old_by_scene.get(args.canary_scene, set()))})")
            merged = set(canary)
            for index in range(args.shard_count):
                shard_id = f"t{t}_{mode}_s{index}of{args.shard_count}"
                rec = jobs.get(f"production_{shard_id}")
                info: dict = {}
                report["shards"][shard_id] = info
                if not rec:
                    problems.append(f"{shard_id}: not in the ledger")
                    continue
                info.update({"job_id": rec.get("job_id"), "log_dir": rec.get("log_dir"),
                             "done_index": rec.get("done_index")})
                status = supervisor_status(Path(rec["log_dir"]))
                info["supervisor"] = status
                if not status["ok"]:
                    problems.append(f"{shard_id}: {status.get('problem')}")
                lines = read_index(Path(rec["done_index"]))
                start, end = shard_span(index, args.shard_count, args.n_scenes)
                span = scenes[start:end]
                expected = set().union(*(old_by_scene.get(s, set()) for s in span)) if span else set()
                got_scenes = {rel.split("/", 1)[0] for rel in lines}
                info.update({"span": f"{start}:{end}", "span_scenes": len(span),
                             "expected_units": len(expected), "units": len(lines),
                             "scenes_with_units": len(got_scenes)})
                if status.get("done_index_lines") is not None and status["done_index_lines"] != len(lines):
                    problems.append(f"{shard_id}: supervisor counted {status['done_index_lines']} index lines, "
                                    f"file has {len(lines)}")
                missing, extra = expected - lines, lines - expected
                if missing or extra:
                    problems.append(f"{shard_id}: unit set differs from fp16 for its span "
                                    f"(missing={len(missing)} e.g. {sorted(missing)[:MAX_EXAMPLES]}, "
                                    f"extra={len(extra)} e.g. {sorted(extra)[:MAX_EXAMPLES]})")
                overlap = merged & lines
                if overlap:
                    problems.append(f"{shard_id}: {len(overlap)} units already claimed by another shard/canary")
                merged |= lines
                step["shards"][shard_id] = len(lines)
            all_old = set().union(*old_by_scene.values()) if old_by_scene else set()
            stray = canon - merged
            if stray:
                problems.append(f"{key}: canonical index holds {len(stray)} entries not from the shards/canary, "
                                f"e.g. {sorted(stray)[:MAX_EXAMPLES]}")
            if merged != all_old:
                problems.append(f"{key}: merged {len(merged)} units != fp16 layer-20/25 units {len(all_old)} "
                                f"(missing={len(all_old - merged)}, extra={len(merged - all_old)})")
            step.update({"merged_units": len(merged), "merged_scenes": len({r.split('/', 1)[0] for r in merged}),
                         "canonical_already_complete": canon == merged})
            report["steps"][key] = step
            index_writes.append((canon_path, merged))

        # Structural files: one copy per scene from the t500 shard and one from the t1000 shard.
        dest_root = args.dataset_root / ns_new / "minimax_h3/scannetpp"
        orig_root = args.dataset_root / ns_old / "minimax_h3/scannetpp"
        scenes_needed = sorted(set(old_by_t[TIMESTEPS[0]]) | set(old_by_t[TIMESTEPS[1]]))
        summary = {"scenes": len(scenes_needed), "files_to_copy": 0, "files_already_present": 0,
                   "t500_t1000_mismatches": 0, "problems": 0}
        for scene in scenes_needed:
            for name in STRUCTURAL[mode]:
                dest = dest_root / scene / name
                if scene == args.canary_scene:
                    sources = [dest]  # published by the canary into the canonical tree
                else:
                    index = shard_of_scene.get(scene)
                    if index is None:
                        problems.append(f"{mode}/{scene}: scene not in any shard span")
                        summary["problems"] += 1
                        continue
                    sources = [
                        args.dataset_root / SHARD_DIRNAME / f"t{t}_{mode}_s{index}of{args.shard_count}"
                        / ns_new / "minimax_h3/scannetpp" / scene / name
                        for t in TIMESTEPS
                    ]
                blobs = []
                for src in sources:
                    if not src.is_file() or src.is_symlink():
                        problems.append(f"{mode}/{scene}/{name}: missing or not a regular file: {src}")
                        summary["problems"] += 1
                        break
                    blobs.append(src.read_bytes())
                if len(blobs) != len(sources):
                    continue
                if any(blob != blobs[0] for blob in blobs[1:]):
                    problems.append(f"{mode}/{scene}/{name}: t500 and t1000 shard copies differ; refusing")
                    summary["t500_t1000_mismatches"] += 1
                    summary["problems"] += 1
                    continue
                try:
                    issue = compare_with_original(name, blobs[0], orig_root / scene / name, ns_new)
                except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
                    issue = f"comparison raised {exc!r}"
                if issue:
                    problems.append(f"{mode}/{scene}/{name}: {issue}")
                    summary["problems"] += 1
                    continue
                if dest.exists():
                    if not dest.is_file() or dest.is_symlink() or dest.read_bytes() != blobs[0]:
                        problems.append(f"{mode}/{scene}/{name}: destination exists with different content: {dest}")
                        summary["problems"] += 1
                    else:
                        summary["files_already_present"] += 1
                    continue
                copies.append((dest, blobs[0]))
                summary["files_to_copy"] += 1
        report["structural"][mode] = summary

    report["status"] = "PASS" if not problems else "FAIL"
    report["n_problems"] = len(problems)
    if len(problems) > 200:
        report["problems"] = problems[:200] + [f"... {len(problems) - 200} more"]

    if args.apply and not problems:
        backup_dir = args.report.parent / "previous_canonical_indices"
        written = {"structural_files": 0, "indices": []}
        for dest, data in copies:
            atomic_write(dest, data)
            written["structural_files"] += 1
        for path, merged in index_writes:
            current = read_index(path)
            if current == merged:
                continue
            if path.exists():
                backup_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup_dir / f"{path.name}.{started}")
            atomic_write(path, "".join(f"{rel}\n" for rel in sorted(merged)).encode("utf-8"))
            if read_index(path) != merged:
                raise RuntimeError(f"index read-back mismatch: {path}")
            written["indices"].append({"path": str(path), "lines": len(merged),
                                       "sha256": sha256_bytes(path.read_bytes())})
        report["applied"] = written
    elif args.apply:
        report["applied"] = None
        report["apply_refused"] = "check found problems; nothing was written"

    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{args.report.name}.tmp-", dir=args.report.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(tmp, 0o644)
    os.replace(tmp, args.report)
    print(json.dumps({"status": report["status"], "mode": report["mode"], "n_problems": len(problems),
                      "steps": {k: {kk: v[kk] for kk in ("merged_units", "fp16_units", "canonical_before")}
                                for k, v in report["steps"].items()},
                      "structural": report["structural"], "applied": report.get("applied"),
                      "first_problems": problems[:10]}, indent=1))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
