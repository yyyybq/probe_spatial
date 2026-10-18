#!/usr/bin/env python3
"""Build frozen evaluation packs: a training pack plus the v2 evaluators.

Canonical V2 checkpoints were trained from frozen packs under runtime/ whose data
path (owner overlay, label generation) the main tree no longer has; see
docs/CODE_ALIGNMENT_20261017.md.  To score them on the samples they were trained
for, but with the non-finite-aware v2 evaluators, each evaluation pack is

  the base pack's frozen file set (exactly the files in its source.sha256)
  + vidfm3d/eval_diag.py, vidfm3d/eval_a1_streaming.py, vidfm3d/utils/loss.py
    taken from a git commit of the main tree.

loss.py is replaced because eval_a1_streaming v2 imports relative_movement_loss.
For r7_single_v3 and aetherfix_r1 the main loss.py only adds code; for
minimax_bf16_v2 it also drops an opt-in fail-fast on non-finite *training*
loss (PROBE_FAIL_ON_NONFINITE_LOSS, default off), which evaluation never uses.

Output: runtime/main_v2_eval_v2_from_<base>/{source/, source.sha256, EVAL_PACK.json}.
An existing pack is never modified.  ``--check`` imports both evaluators from
each pack in a subprocess (CPU only) and verifies source.sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPLACED = ["vidfm3d/eval_diag.py", "vidfm3d/eval_a1_streaming.py", "vidfm3d/utils/loss.py"]
PY = "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"
DATASET = "vidfm3d/data/components/inscene15k_dataset.py"
# r2: the training packs' C-probe fast path (_getitem_feature_action_diag) emits
# only training inputs; eval_diag v2 also needs the camera poses and horizon
# offsets for C2 (final pose error, drift, per-horizon errors).  The patch
# always recomputes the geometry (the code a label-cache miss runs) and adds
# target_extrinsics / start_extrinsic / action_horizons exactly as the main
# tree's fast path does.  Training inputs are produced by unchanged code.
R2_PATCHES = [
    (
        "        geometry = self._load_label_cache(cache_identity, cache_fields)\n        if geometry is None:\n",
        "        # Evaluation pack r2: always recompute the geometry (the label-cache miss\n"
        "        # path) so the camera poses exist for the evaluation-only fields below.\n"
        "        geometry = None\n        if geometry is None:\n",
    ),
    (
        '        if self.diag_path_integration:\n            output["path_actions"] = geometry["path_actions"]\n'
        '            output["path_horizon_valid"] = horizon_valid\n',
        '        if self.diag_path_integration:\n            output["path_actions"] = geometry["path_actions"]\n'
        '            output["path_horizon_valid"] = horizon_valid\n'
        "            # Evaluation-only fields (evaluation pack r2), computed as in the main\n"
        "            # tree: poses in the reference frame and scale that produced path_actions.\n"
        '            output["target_extrinsics"] = extrinsics[1:].cpu()\n'
        '            output["start_extrinsic"] = extrinsics[0].cpu()\n'
        '            output["action_horizons"] = (\n'
        "                (target_global - sel_global[0]).to(torch.long).cpu()\n"
        "                if K > 0 else torch.zeros(0, dtype=torch.long)\n"
        "            )\n",
    ),
]
REVISIONS = {
    "r1": {"prefix": "main_v2_eval_v2_from_",
           "bases": ["main_v2_probe_v2_r7_single_v3", "main_v2_probe_v2_aetherfix_r1", "main_v2_probe_minimax_bf16_v2"],
           "patches": {}},
    "r2": {"prefix": "main_v2_eval_v2r2_from_",
           "bases": ["main_v2_probe_v2_r7_single_v3", "main_v2_probe_v2_aetherfix_r1"],
           "patches": {DATASET: R2_PATCHES}},
    # 2026-10-18: the MiniMax bf16 pack's C-probe fast path has the same anchors;
    # same r2 patch so C2 gets its evaluation-only pose fields.
    "r2m": {"prefix": "main_v2_eval_v2r2_from_",
            "bases": ["main_v2_probe_minimax_bf16_v2"],
            "patches": {DATASET: R2_PATCHES}},
    # Timestep training packs have no source.sha256; their identity is the
    # snapshot recorded as git tag code/<pack> on 2026-10-17.  The base file set
    # is that tag's tree, and every file must still equal runtime/<pack>/source.
    "t1": {"prefix": "tstep_eval_v2_from_",
           "bases": ["tstep_probe_opt_v3", "tstep_probe_pack_v4_1"],
           "base_kind": "tag", "patches": {DATASET: R2_PATCHES}},
    # The 2026-10-16 timestep B2 fp32 retrains ran from the main tree at e4e4886
    # (no pack).  Base = that commit's vidfm3d/, configs/ and .project-root.
    "t1m": {"prefix": "tstep_eval_v2_from_",
            "bases": ["main_e4e4886"],
            "base_kind": "commit", "commits": {"main_e4e4886": "e4e4886"},
            "paths": ["vidfm3d", "configs", ".project-root"], "patches": {}},
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def base_files(base: str, spec: dict) -> tuple[dict[str, bytes], dict]:
    """rel path -> bytes of the base pack's frozen file set, plus identity facts."""
    kind = spec.get("base_kind", "runtime")
    base_dir = ROOT / "runtime" / base
    files: dict[str, bytes] = {}
    if kind == "runtime":
        for line in (base_dir / "source.sha256").read_text(encoding="utf-8").splitlines():
            sha, rel = line.split("  ", 1)
            rel = rel[2:] if rel.startswith("./") else rel
            data = (base_dir / "source" / rel).read_bytes()
            if sha256_bytes(data) != sha:
                raise SystemExit(f"base pack {base} file changed since freeze: {rel}")
            files[rel] = data
        return files, {"base_pack": f"runtime/{base}", "base_git_tag": f"code/{base}",
                       "base_source_sha256_file_sha256": sha256_bytes((base_dir / "source.sha256").read_bytes())}
    rev = f"code/{base}" if kind == "tag" else spec["commits"][base]
    commit = git("rev-parse", f"{rev}^{{commit}}").decode().strip()
    listing = git("ls-tree", "-r", "-z", commit, "--", *spec.get("paths", [])).decode().split("\0")
    changed = []
    for entry in filter(None, listing):
        meta, rel = entry.split("\t", 1)
        mode, kind_, _ = meta.split()
        if kind_ != "blob" or mode == "120000":
            raise SystemExit(f"{rev}: unsupported tree entry {mode} {kind_} {rel}")
        data = git("show", f"{commit}:{rel}")
        files[rel] = data
        if kind == "tag":
            live = base_dir / "source" / rel
            if not live.is_file() or sha256_bytes(live.read_bytes()) != sha256_bytes(data):
                changed.append(rel)
    if changed:
        raise SystemExit(f"runtime/{base}/source differs from {rev} in {len(changed)} files, e.g. {changed[:3]}")
    listing_text = "".join(f"{sha256_bytes(files[r])}  ./{r}\n" for r in sorted(files))
    return files, {"base_pack": f"runtime/{base}" if kind == "tag" else f"git {commit}",
                   "base_git_tag": rev if kind == "tag" else None, "base_commit": commit,
                   "base_source_sha256_file_sha256": sha256_bytes(listing_text.encode()),
                   "base_identity_note": ("snapshot of runtime/<pack>/source recorded 2026-10-17; "
                                          "no freeze-time hash list exists" if kind == "tag" else
                                          "main-tree commit the training ran from")}


def build(base: str, commit: str, revision: str = "r1") -> Path:
    spec = REVISIONS[revision]
    out = ROOT / "runtime" / f"{spec['prefix']}{base}"
    if out.exists():
        raise SystemExit(f"{out} exists; evaluation packs are immutable")
    files, identity = base_files(base, spec)
    src = out / "source"
    src.mkdir(parents=True)
    records = {}
    for rel, data in files.items():
        records[rel] = {"sha256": sha256_bytes(data), "origin": identity["base_pack"]}
    replaced = {}
    for rel in REPLACED:
        data = git("show", f"{commit}:{rel}")
        replaced[rel] = {
            "sha256": sha256_bytes(data),
            "origin": f"git {commit}",
            "base_sha256": records.get(rel, {}).get("sha256"),
        }
        records[rel] = {"sha256": replaced[rel]["sha256"], "origin": f"git {commit}"}
    patched = {}
    for rel, patches in spec["patches"].items():
        text = files[rel].decode("utf-8")
        for old, new in patches:
            if text.count(old) != 1:
                raise SystemExit(f"{base}:{rel}: patch anchor found {text.count(old)} times, expected 1")
            text = text.replace(old, new)
        data = text.encode("utf-8")
        patched[rel] = {"sha256": sha256_bytes(data), "base_sha256": records[rel]["sha256"],
                        "patches": len(patches), "data": data}
        records[rel] = {"sha256": patched[rel]["sha256"], "origin": f"{identity['base_pack']} + {revision} patch"}
    for rel in sorted(records):
        dest = src / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if rel in replaced:
            dest.write_bytes(git("show", f"{commit}:{rel}"))
        elif rel in patched:
            dest.write_bytes(patched[rel]["data"])
        else:
            dest.write_bytes(files[rel])
        os.chmod(dest, 0o444)
        if sha256_bytes(dest.read_bytes()) != records[rel]["sha256"]:
            raise SystemExit(f"copy mismatch {rel}")
    sha_file = out / "source.sha256"
    sha_file.write_text("".join(f"{records[rel]['sha256']}  ./{rel}\n" for rel in sorted(records)), encoding="utf-8")
    os.chmod(sha_file, 0o444)
    manifest = {
        "schema": "MAIN_V2_EVAL_V2_PACK_V1",
        "status": "FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **identity,
        "revision": revision,
        "replaced_from_commit": commit,
        "replaced_files": replaced,
        "patched_files": {k: {kk: vv for kk, vv in v.items() if kk != "data"} for k, v in patched.items()},
        "patch_reason": ("C-probe fast path: always recompute geometry and emit target_extrinsics, "
                         "start_extrinsic, action_horizons for eval_diag v2 (C2)") if patched else None,
        "files": len(records),
        "source_sha256_file_sha256": sha256_bytes(sha_file.read_bytes()),
        "reason": "score checkpoints on the training pack's data path with the non-finite-aware v2 evaluators",
        "doc": "docs/CODE_ALIGNMENT_20261017.md",
    }
    (out / "EVAL_PACK.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(out / "EVAL_PACK.json", 0o444)
    return out


def check(base: str, revision: str = "r1") -> dict:
    out = ROOT / "runtime" / f"{REVISIONS[revision]['prefix']}{base}"
    src = out / "source"
    rc = subprocess.run(["sha256sum", "--status", "-c", str(out / "source.sha256")], cwd=src).returncode
    probe = subprocess.run(
        [PY, "-c", "import vidfm3d.eval_diag as d, vidfm3d.eval_a1_streaming as a; "
                   "print(d.EVALUATOR_VERSION, a.EVALUATOR_VERSION)"],
        cwd=src, env={**os.environ, "PYTHONPATH": str(src), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True,
    )
    return {"pack": out.name, "sha256sum_rc": rc, "import_rc": probe.returncode,
            "versions": probe.stdout.strip().splitlines()[-1:] , "stderr_tail": probe.stderr.strip().splitlines()[-2:]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--commit", help="main-tree commit providing the replaced files")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--revision", choices=sorted(REVISIONS), default="r1")
    args = parser.parse_args()
    bases = REVISIONS[args.revision]["bases"]
    if not args.check:
        if not args.commit:
            raise SystemExit("--commit is required to build")
        for base in bases:
            print("built", build(base, args.commit, args.revision))
    for base in bases:
        print(json.dumps(check(base, args.revision)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
