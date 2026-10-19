#!/usr/bin/env python3
"""Plot only geometry/probe evidence that is valid under the protocol gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    branches = [json.loads(line) for line in (output / "branches.jsonl").read_text().splitlines()]
    smoke = json.loads((output / "seese3_m3_m4_primitive_smoke.json").read_text())
    roles = ("T2", "21")
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for role in roles:
        axes[0].hist([row["endpoint_observations"][role]["pose_residual"]["translation_m"] for row in branches], bins=28, alpha=0.55, label=role)
        axes[1].hist([row["endpoint_observations"][role]["pose_residual"]["rotation_deg"] for row in branches], bins=28, alpha=0.55, label=role)
    axes[0].set_xlabel("nearest-observed translation residual (m)")
    axes[1].set_xlabel("nearest-observed rotation residual (deg)")
    for axis in axes:
        axis.set_ylabel("branches")
        axis.legend(frameon=False)
        axis.grid(alpha=0.2)
    figure.suptitle("Approx-grounded endpoint residuals — not exact render ground truth")
    figure.tight_layout()
    figure.savefig(output / "approx_endpoint_residuals.png", dpi=180)
    figure.savefig(output / "approx_endpoint_residuals.pdf")
    plt.close(figure)

    metrics = ("overall_zscore_r2", "rotation_zscore_r2", "translation_zscore_r2", "translation_direction_r2", "translation_magnitude_r2")
    labels = ("overall", "rotation", "translation", "direction", "magnitude")
    x = np.arange(len(metrics))
    figure, axis = plt.subplots(figsize=(7.4, 3.8))
    axis.bar(x - 0.18, [smoke["M3"][key] for key in metrics], 0.36, label="M3 raw linear")
    axis.bar(x + 0.18, [smoke["M4"][key] for key in metrics], 0.36, label="M4 Poincaré adapter")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, labels)
    axis.set_ylabel("held-out R²")
    axis.set_title("Legacy observed primitive-frame SeeSE3 smoke (not a replication)")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "seese3_primitive_smoke.png", dpi=180)
    figure.savefig(output / "seese3_primitive_smoke.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
