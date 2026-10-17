import argparse
import json
import shlex
from pathlib import Path

import torch


DEFAULT_MODELS = {"dino", "vjepa2", "f3r", "videomae"}


def parse_overrides(command: str) -> dict[str, str]:
    overrides = {}
    for token in shlex.split(command):
        if "=" in token and not token.startswith("python"):
            key, value = token.split("=", 1)
            overrides[key] = value
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check probe checkpoints from a task_manifest.tsv file."
    )
    parser.add_argument(
        "--manifest",
        default="logs/probe_train_24gpu_20260715_194202/task_manifest.tsv",
    )
    parser.add_argument(
        "--runs-dir",
        default="logs/inscene15k_streaming/runs",
    )
    parser.add_argument(
        "--models",
        default=",".join(sorted(DEFAULT_MODELS)),
        help="Comma-separated vfm_name allowlist.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON records.")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    runs_dir = Path(args.runs_dir)
    models = {x.strip() for x in args.models.split(",") if x.strip()}
    rows = []

    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        task_id, task_name, command = line.split("\t", 2)
        overrides = parse_overrides(command)
        vfm_name = overrides.get("vfm_name", "")
        if vfm_name not in models:
            continue
        run_folder = overrides.get(
            "paths.run_folder_name", f"inscene15k_streaming_{task_name}"
        )
        ckpt_dir = runs_dir / run_folder / "checkpoints"
        ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        row = {
            "task_id": task_id,
            "task_name": task_name,
            "vfm_name": vfm_name,
            "run_folder": run_folder,
            "ckpt_dir": str(ckpt_dir),
            "ckpt_count": len(ckpts),
            "latest_ckpt": str(ckpts[-1]) if ckpts else None,
            "epoch": None,
            "global_step": None,
            "error": None,
        }
        if ckpts:
            try:
                ckpt = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
                row["epoch"] = ckpt.get("epoch")
                row["global_step"] = ckpt.get("global_step")
            except Exception as exc:
                row["error"] = repr(exc)
        rows.append(row)

    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return

    total = len(rows)
    with_ckpt = sum(1 for row in rows if row["ckpt_count"])
    print(f"total={total} with_ckpt={with_ckpt} no_ckpt={total - with_ckpt}")
    for row in rows:
        status = "CKPT" if row["ckpt_count"] else "NO_CKPT"
        print(
            f"{status}\t{row['task_name']}\tckpts={row['ckpt_count']}"
            f"\tepoch={row['epoch']}\tglobal_step={row['global_step']}"
            f"\tlatest={row['latest_ckpt']}"
        )


if __name__ == "__main__":
    main()
