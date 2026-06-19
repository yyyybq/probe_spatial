#!/usr/bin/env python3
"""Create a deterministic, source-stratified scene split manifest."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vidfm3d.data.components.inscene15k_dataset import InsScene15KDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to < 1")

    dataset = InsScene15KDataset(
        root=args.data_root,
        split="all",
        root_vfm=None,
        include_pmaps=False,
        window_size=0,
    )
    by_source = {}
    for scene in dataset.scenes:
        by_source.setdefault(scene["source"], []).append(dataset._scene_key(scene))

    rng = np.random.default_rng(args.seed)
    splits = {"train": [], "val": [], "test": []}
    source_counts = {}
    for source, keys in sorted(by_source.items()):
        keys = sorted(keys)
        keys = [keys[i] for i in rng.permutation(len(keys))]
        n_test = round(len(keys) * args.test_ratio)
        n_val = round(len(keys) * args.val_ratio)
        splits["test"].extend(keys[:n_test])
        splits["val"].extend(keys[n_test:n_test + n_val])
        splits["train"].extend(keys[n_test + n_val:])
        source_counts[source] = len(keys)

    for values in splits.values():
        values.sort()
    payload = {
        "schema_version": 1,
        "dataset": "InsScene-15K",
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "source_counts": source_counts,
        "splits": splits,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print({name: len(values) for name, values in splits.items()})


if __name__ == "__main__":
    main()
