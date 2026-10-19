"""scripts/consolidate_tstep_minimax_bf16_v1_shards.py on a synthetic 16-scene tree."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/consolidate_tstep_minimax_bf16_v1_shards.py"
NS = {
    "stream": ("FEAT_STREAMING_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1", "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"),
    "target": ("FEAT_TARGET_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1", "FEAT_TARGET_CANONICAL_V2_SEEDED_V1"),
}
STRUCT = {
    "stream": ("prefix_index.npy", "prefix_index.npy.manifest.json"),
    "target": ("target_indices.npy", "target_indices.npy.manifest.json", "target_indices.meta.json"),
}
N_SCENES = 16
SHARDS = 8
SCENES = [f"s{i:02d}" for i in range(N_SCENES)]
EXCLUDED = "s05"  # collected by the extractor but absent from the fp16 index


def units(scene: str, mode: str, t: int, layers=(20, 25)) -> list[str]:
    if mode == "target":
        return [f"{scene}/feature_t{t}_layer{layer}.sft" for layer in layers]
    return [f"{scene}/window_0000/prefix_{tail:06d}/feature_t{t}_layer{layer}.sft"
            for tail in (7, 11) for layer in layers]


def struct_bytes(scene: str, name: str, namespace: str) -> bytes:
    if name.endswith(".json"):
        return json.dumps({"file": name, "scene": scene, "feature_namespace": namespace}, sort_keys=True).encode()
    return f"npy:{scene}:{name}".encode()


def span(i: int) -> range:
    start, end = N_SCENES * i // SHARDS, N_SCENES * (i + 1) // SHARDS
    return range(1 if start == 0 else start, end)


def build(tmp_path: Path) -> dict:
    data = tmp_path / "InsScene-15K"
    done = tmp_path / "aoss_done"
    done.mkdir()
    jobs = {}
    indexed = [s for s in SCENES if s != EXCLUDED]
    for mode, (ns_new, ns_old) in NS.items():
        for t in (500, 1000):
            lines = [u for s in indexed for u in units(s, mode, t, (15, 20, 25))]
            (done / f"{ns_old}_minimax_h3_t{t}.txt").write_text("\n".join(lines) + "\n")
            (done / f"{ns_new}_minimax_h3_t{t}.txt").write_text("\n".join(units(SCENES[0], mode, t)) + "\n")
            for i in range(SHARDS):
                shard = f"t{t}_{mode}_s{i}of{SHARDS}"
                state = tmp_path / "shards" / shard
                log_dir = tmp_path / "logs" / shard
                (state / "aoss_done").mkdir(parents=True)
                log_dir.mkdir(parents=True)
                index = state / "aoss_done" / f"{ns_new}_minimax_h3_t{t}.txt"
                shard_lines = [u for k in span(i) if SCENES[k] != EXCLUDED for u in units(SCENES[k], mode, t)]
                index.write_text("".join(f"{u}\n" for u in shard_lines))
                (log_dir / "supervisor.log").write_text(
                    "banner\nfinished_at=2026-10-17T00:00:00+00:00 extract_fail=0 sync_fail=0\n"
                    f"done_index_lines={len(shard_lines)}\nlocal_matching_sft_left=0\nOK\n"
                )
                (log_dir / "extract.finished").write_text("")
                jobs[f"production_{shard}"] = {"job_id": f"pt-{shard}", "log_dir": str(log_dir),
                                               "done_index": str(index)}
                for k in span(i):
                    scene = SCENES[k]
                    if scene == EXCLUDED:
                        continue
                    for name in STRUCT[mode]:
                        p = data / "tstep_minimax_bf16_v1_shards" / shard / ns_new / "minimax_h3/scannetpp" / scene / name
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_bytes(struct_bytes(scene, name, ns_new))
        for scene in indexed:
            for name in STRUCT[mode]:
                p = data / ns_old / "minimax_h3/scannetpp" / scene / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(struct_bytes(scene, name, ns_old))
        for name in STRUCT[mode]:  # canary scene already in the canonical tree
            p = data / ns_new / "minimax_h3/scannetpp" / SCENES[0] / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(struct_bytes(SCENES[0], name, ns_new))
    ledger = tmp_path / "submissions.json"
    ledger.write_text(json.dumps({"jobs": jobs}))
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text("\n".join(SCENES) + "\n")
    return {"data": data, "done": done, "ledger": ledger, "scene_list": scene_list}


def run(tmp_path: Path, tree: dict, *extra: str):
    report = tmp_path / "out" / "report.json"
    cmd = [sys.executable, str(SCRIPT), "--report", str(report), "--ledger", str(tree["ledger"]),
           "--done-dir", str(tree["done"]), "--dataset-root", str(tree["data"]),
           "--scene-list", str(tree["scene_list"]), "--n-scenes", str(N_SCENES),
           "--shard-count", str(SHARDS), "--canary-scene", SCENES[0], *extra]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result, json.loads(report.read_text())


def test_check_then_apply_merges_indices_and_copies_structure(tmp_path: Path) -> None:
    tree = build(tmp_path)
    result, report = run(tmp_path, tree)
    assert result.returncode == 0, report["problems"]
    assert report["status"] == "PASS"
    # check mode wrote nothing
    stream_ns = NS["stream"][0]
    assert len((tree["done"] / f"{stream_ns}_minimax_h3_t500.txt").read_text().split()) == 4
    assert report["structural"]["stream"]["files_to_copy"] == 2 * (N_SCENES - 2)

    result, report = run(tmp_path, tree, "--apply")
    assert result.returncode == 0, report["problems"]
    for mode, (ns_new, _) in NS.items():
        for t in (500, 1000):
            got = (tree["done"] / f"{ns_new}_minimax_h3_t{t}.txt").read_text().split()
            want = sorted(u for s in SCENES if s != EXCLUDED for u in units(s, mode, t))
            assert got == want
        for scene in SCENES[1:]:
            for name in STRUCT[mode]:
                dest = tree["data"] / ns_new / "minimax_h3/scannetpp" / scene / name
                assert dest.exists() == (scene != EXCLUDED)
                if scene != EXCLUDED:
                    assert dest.stat().st_mode & 0o777 == 0o644
    backups = list((tmp_path / "out/previous_canonical_indices").iterdir())
    assert len(backups) == 4

    result, report = run(tmp_path, tree, "--apply")  # idempotent
    assert result.returncode == 0
    assert report["applied"] == {"structural_files": 0, "indices": []}


def test_refuses_when_t500_and_t1000_structural_copies_differ(tmp_path: Path) -> None:
    tree = build(tmp_path)
    ns_new = NS["stream"][0]
    victim = (tree["data"] / "tstep_minimax_bf16_v1_shards" / "t1000_stream_s3of8" / ns_new
              / "minimax_h3/scannetpp" / SCENES[6] / "prefix_index.npy")
    victim.write_bytes(b"different")
    result, report = run(tmp_path, tree, "--apply")
    assert result.returncode == 1
    assert report["structural"]["stream"]["t500_t1000_mismatches"] == 1
    assert report["applied"] is None
    assert not (tree["data"] / ns_new / "minimax_h3/scannetpp" / SCENES[1]).exists()
    assert len((tree["done"] / f"{ns_new}_minimax_h3_t500.txt").read_text().split()) == 4


def test_refuses_incomplete_shard_and_unclean_supervisor(tmp_path: Path) -> None:
    tree = build(tmp_path)
    ledger = json.loads(tree["ledger"].read_text())
    rec = ledger["jobs"]["production_t500_target_s2of8"]
    index = Path(rec["done_index"])
    index.write_text("".join(index.read_text().splitlines(keepends=True)[:-1]))
    log = Path(ledger["jobs"]["production_t1000_stream_s7of8"]["log_dir"]) / "supervisor.log"
    log.write_text(log.read_text().replace("sync_fail=0", "sync_fail=1"))
    result, report = run(tmp_path, tree)
    assert result.returncode == 1
    text = "\n".join(report["problems"])
    assert "t500_target_s2of8: unit set differs" in text and "missing=1" in text
    assert "t500_target_s2of8: supervisor counted" in text
    assert "t1000_stream_s7of8: supervisor did not finish cleanly" in text


def test_refuses_structural_drift_from_fp16_original(tmp_path: Path) -> None:
    tree = build(tmp_path)
    ns_new, ns_old = NS["target"]
    for t in (500, 1000):
        p = (tree["data"] / "tstep_minimax_bf16_v1_shards" / f"t{t}_target_s4of8" / ns_new
             / "minimax_h3/scannetpp" / SCENES[9] / "target_indices.meta.json")
        meta = json.loads(p.read_text())
        meta["extra_key"] = 1
        p.write_text(json.dumps(meta))
    result, report = run(tmp_path, tree)
    assert result.returncode == 1
    assert any("target_indices.meta.json: differs from fp16 original in ['extra_key']" in p
               for p in report["problems"])
