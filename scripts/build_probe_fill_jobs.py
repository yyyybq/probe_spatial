#!/usr/bin/env python3
"""Build missing multilayer ABC probe jobs for the 2026-08-14 fill campaign.

Pose (A1 camera) is trained inside streaming_depth — no separate streaming_camera.

Sources of "done":
  - launch/*.log with END rc=0
  - checkpoints_protected/*/ *.ckpt
  - evaluations/* directory names
  - canonical_best_checkpoint_registry.json (non-TRULY_MISSING)

Usage:
  python scripts/build_probe_fill_jobs.py --out-dir logs/probing_jobs_fill_...
  python scripts/build_probe_fill_jobs.py --campaign lingbot --out-dir ...
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

JOB_RE = re.compile(
    r"(?P<probe>streaming_depth|streaming_mask|view_consistency|ego_belief_v2|ego_belief|"
    r"action_dynamics|path_integration|counterfactual)"
    r"_(?P<vfm>[a-z0-9_]+)_p(?P<prefix>\d+)_layer(?P<layer>-?\d+)",
    re.I,
)
END_RE = re.compile(r"END .* rc=(-?\d+)\s*$", re.M)

EXPECTED = {
    "vjepa2": {
        "layers": [0, 5, 11, 17, 23],
        "prefixes": [8, 12, 16, 24],
        "probes": [
            "streaming_depth",
            "streaming_mask",
            "view_consistency",
            "ego_belief_v2",
            "action_dynamics",
            "path_integration",
            "counterfactual",
        ],
        "ch": 1024,
        "t": 749,
        "disk": {},
        "bs_default": 8,
    },
    "videomae": {
        "layers": [0, 5, 11, 17, 23],
        "prefixes": [8, 12, 16, 24],
        "probes": [
            "streaming_depth",
            "streaming_mask",
            "view_consistency",
            "ego_belief_v2",
            "action_dynamics",
            "path_integration",
            "counterfactual",
        ],
        "ch": 1024,
        "t": 749,
        "disk": {},
        "bs_default": 8,
    },
    "minimax_h3": {
        "layers": [10, 20, 30, 40, 50],
        "prefixes": [8, 12, 16, 24],
        "probes": [
            "streaming_depth",
            "streaming_mask",
            "view_consistency",
            "ego_belief_v2",
            "action_dynamics",
            "path_integration",
            "counterfactual",
        ],
        "ch": 5376,
        "t": 250,
        "disk": {50: 49},
        "bs_default": 2,
    },
    "wan": {
        "layers": [0, 5, 10, 15, 25, 29],
        "prefixes": [8, 12, 16, 24],
        "probes": [
            "streaming_depth",
            "streaming_mask",
            "view_consistency",
            "ego_belief_v2",
            "action_dynamics",
            "path_integration",
            "counterfactual",
        ],
        "ch": 1536,
        "t": 749,
        "disk": {},
        "bs_default": 4,
    },
    "cogvideox": {
        "layers": [0, 5, 10, 15, 25, 30, 35, 40],
        "prefixes": [8, 12, 16, 24],
        "probes": [
            "streaming_depth",
            "streaming_mask",
            "view_consistency",
            "ego_belief_v2",
            "action_dynamics",
            "path_integration",
            "counterfactual",
        ],
        "ch": 3072,
        "t": 749,
        "disk": {},
        "bs_default": 4,
    },
    "lingbot": {
        "layers": [10, 20, 30, 40, 48],
        "prefixes": [8, 12, 16, 24],
        "probes": [
            "streaming_depth",
            "streaming_mask",
            "view_consistency",
            "ego_belief_v2",
            "action_dynamics",
            "path_integration",
            "counterfactual",
        ],
        "ch": 2048,
        "t": 250,
        "disk": {48: 47},
        "bs_default": 2,
    },
}

CAMPAIGN_VFMS = {
    "all": list(EXPECTED),
    "fill": ["vjepa2", "videomae", "minimax_h3", "wan", "cogvideox"],
    "a1": ["vjepa2", "videomae", "minimax_h3", "wan", "cogvideox"],
    "wan_cog": ["wan", "cogvideox"],
    "lingbot": ["lingbot"],
    "p24": ["vjepa2", "videomae", "minimax_h3"],
}


def _tail_decode(path: Path, nbytes: int = 800_000) -> str:
    data = path.read_bytes()
    if len(data) > nbytes:
        data = data[-nbytes:]
    return data.decode("utf-8", "ignore")


def collect_ok(root: Path) -> set[tuple[str, str, int, int]]:
    ok: set[tuple[str, str, int, int]] = set()

    for p in (root / "logs/inscene15k_streaming").glob("*/launch/*.log"):
        m = JOB_RE.search(p.name)
        if not m:
            continue
        try:
            text = _tail_decode(p)
        except OSError:
            continue
        ms = list(END_RE.finditer(text))
        if ms and int(ms[-1].group(1)) == 0:
            ok.add((m["vfm"], m["probe"], int(m["prefix"]), int(m["layer"])))

    prot = root / "logs/inscene15k_streaming/checkpoints_protected"
    if prot.is_dir():
        for d in prot.iterdir():
            m = JOB_RE.search(d.name)
            if not m:
                continue
            if any(d.glob("*.ckpt")):
                ok.add((m["vfm"], m["probe"], int(m["prefix"]), int(m["layer"])))

    ev = root / "logs/inscene15k_streaming/evaluations"
    if ev.is_dir():
        for d in ev.iterdir():
            m = JOB_RE.search(d.name)
            if m:
                ok.add((m["vfm"], m["probe"], int(m["prefix"]), int(m["layer"])))

    reg = (
        root
        / "logs/probing_jobs_20260725_ab/recovery_20260730/perf_analysis/"
        / "canonical_best_checkpoint_registry.json"
    )
    if reg.is_file():
        data = json.loads(reg.read_text())
        for row in data.get("rows", []):
            job = row.get("job") or row.get("job_name") or ""
            m = JOB_RE.search(job)
            if not m:
                continue
            cls = row.get("ckpt_class") or row.get("class") or ""
            status = row.get("eval_status") or ""
            if cls != "TRULY_MISSING" or status in ("ready_formal", "diagnostic_only"):
                ok.add((m["vfm"], m["probe"], int(m["prefix"]), int(m["layer"])))
    return ok


def manifest_for_prefix(root: Path, prefix: int) -> Path:
    filtered = (
        root
        / "logs/probing_jobs_20260724_1808/manifests_filtered"
        / f"inscene_c_h124_prefix{prefix}_no48da_w003.jsonl"
    )
    if filtered.is_file():
        return filtered
    return root / "manifests/c_retrain_20260722" / f"inscene_c_h124_prefix{prefix}.jsonl"


def campaign_filter(campaign: str, vfm: str, probe: str, prefix: int) -> bool:
    if campaign in ("all", "fill", "lingbot", "wan_cog"):
        return True
    if campaign == "a1":
        return probe in ("streaming_depth", "streaming_mask")
    if campaign == "p24":
        return prefix == 24
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--campaign",
        default="fill",
        choices=sorted(CAMPAIGN_VFMS),
        help="fill=all non-lingbot missing; all includes lingbot",
    )
    ap.add_argument("--skip-scan", action="store_true", help="emit full matrix (ignore done)")
    args = ap.parse_args()
    root = args.project_root.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "jobs").mkdir(exist_ok=True)

    ok = set() if args.skip_scan else collect_ok(root)
    vfms = CAMPAIGN_VFMS[args.campaign]

    rows = []
    summary = []
    for vfm in vfms:
        exp = EXPECTED[vfm]
        for probe in exp["probes"]:
            need = {(p, l) for p in exp["prefixes"] for l in exp["layers"]}
            have = {(p, l) for (v, pr, p, l) in ok if v == vfm and pr == probe}
            if probe == "ego_belief_v2":
                have |= {(p, l) for (v, pr, p, l) in ok if v == vfm and pr == "ego_belief"}
            miss = sorted(need - have)
            miss = [
                (p, l)
                for p, l in miss
                if campaign_filter(args.campaign, vfm, probe, p)
            ]
            if miss:
                summary.append(
                    f"{vfm:12s} {probe:22s} miss={len(miss):3d} "
                    f"by_p={dict(sorted(Counter(p for p, _ in miss).items()))}"
                )
            cfg = f"configs/experiment/inscene15k_streaming/{probe}_{vfm}_v1.yaml"
            if not (root / cfg).is_file():
                summary.append(f"MISS_CFG {cfg}")
                continue
            for p, l in miss:
                disk = exp["disk"].get(l, l)
                man = manifest_for_prefix(root, p)
                rows.append(
                    (
                        vfm,
                        probe,
                        str(p),
                        str(l),
                        str(disk),
                        str(exp["ch"]),
                        str(exp["t"]),
                        str(exp["bs_default"]),
                        str(man),
                        cfg,
                    )
                )

    tsv = out / "jobs/all.tsv"
    header = (
        "vfm\tprobe\tprefix\tlayer\tdisk_layer\tvideo_channels\t"
        "feature_timestep\tbatch_size\tmanifest\tcfg\n"
    )
    tsv.write_text(header + "\n".join("\t".join(r) for r in rows) + ("\n" if rows else ""))

    # bucket breakdown
    buckets: dict[str, int] = defaultdict(int)
    for vfm, probe, prefix, *_ in rows:
        p = int(prefix)
        if vfm == "lingbot":
            buckets["lingbot_abc"] += 1
        elif probe == "streaming_mask":
            buckets["a1_mask"] += 1
        elif probe == "streaming_depth":
            buckets["a1_depth_pose"] += 1
        elif vfm in ("wan", "cogvideox") and probe in (
            "action_dynamics",
            "path_integration",
            "counterfactual",
        ):
            buckets["wan_cog_C"] += 1
        elif vfm in ("wan", "cogvideox"):
            buckets["wan_cog_AB_fill"] += 1
        else:
            buckets[f"other_{probe}_p{p}"] += 1

    plan = out / "CHECKLIST.txt"
    plan.write_text(
        "\n".join(
            [
                f"campaign={args.campaign}",
                f"ok_unique_scanned={len(ok)}",
                f"missing_jobs={len(rows)}",
                "",
                "NOTE: A1 pose == camera head inside streaming_depth",
                "      (streaming_camera was cancelled; do not resubmit).",
                "NOTE: Classic single-layer A1 mask/pose for dino/f3r/aether/qwen",
                "      already covered in logs/eval_results/results_summary.md.",
                "",
                "## missing by vfm/probe",
                *summary,
                "",
                "## buckets",
                *[f"{k}: {v}" for k, v in sorted(buckets.items())],
                "",
                f"jobs_tsv: {tsv}",
            ]
        )
        + "\n"
    )
    print(plan.read_text())
    print(f"Wrote {len(rows)} jobs -> {tsv}")


if __name__ == "__main__":
    main()
