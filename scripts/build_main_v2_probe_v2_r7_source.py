#!/usr/bin/env python3
"""Build the isolated r7 probe runtime from the validated r5 runtime."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    project = args.project.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    output.mkdir(parents=True)
    for tree in ("configs", "scripts", "vidfm3d"):
        shutil.copytree(
            base / tree,
            output / tree,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    shutil.copy2(base / ".project-root", output / ".project-root")
    config_dir = project / "configs/experiment/inscene15k_streaming"
    os.chmod(output / "configs/experiment/inscene15k_streaming", 0o755)
    for source in sorted(config_dir.glob("action_dynamics_*_h124_v1.yaml")):
        shutil.copy2(source, output / "configs/experiment/inscene15k_streaming" / source.name)
    # Horizon-aware evaluation is the only eval change relative to r5.
    os.chmod(output / "vidfm3d", 0o755)
    os.chmod(output / "vidfm3d/eval_diag.py", 0o644)
    shutil.copy2(project / "vidfm3d/eval_diag.py", output / "vidfm3d/eval_diag.py")
    print(output)


if __name__ == "__main__":
    main()
