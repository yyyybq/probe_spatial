#!/usr/bin/env python3
"""Print registered feature-layer defaults for shell scripts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_FEATURE_LAYERS_PATH = Path(__file__).resolve().parents[1] / "vidfm3d" / "utils" / "feature_layers.py"
_SPEC = importlib.util.spec_from_file_location("feature_layers", _FEATURE_LAYERS_PATH)
feature_layers = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = feature_layers
_SPEC.loader.exec_module(feature_layers)

all_output_layers = feature_layers.all_output_layers
get_feature_layer_spec = feature_layers.get_feature_layer_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vfm", required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--field", choices=["default_layer", "last_layer", "feat_postfix", "in_channels"])
    parser.add_argument("--format", choices=["json", "list"], default="json")
    args = parser.parse_args()

    spec = get_feature_layer_spec(args.vfm, args.model_id)
    if spec is None:
        raise SystemExit(f"No feature-layer spec registered for {args.vfm}")

    if args.field:
        value = getattr(spec, args.field)
        if value is None:
            raise SystemExit(f"{args.field} is unknown for {args.vfm}")
        print(value)
        return

    if args.format == "list":
        layers = all_output_layers(args.vfm, args.model_id)
        print(" ".join(str(layer) for layer in layers))
        return

    print(json.dumps(spec.__dict__, indent=2))


if __name__ == "__main__":
    main()
