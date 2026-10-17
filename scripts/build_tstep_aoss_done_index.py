#!/usr/bin/env python3
"""Build local completion indexes from AOSS LIST responses only.

No object body is fetched. Each output line is a feature path relative to
``<FEATURE_NAMESPACE>/<vfm>/<source>`` so ``features.run_inscene15k`` can use it as
``EXTRACT_DONE_INDEX``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from aoss_client.ceph.ceph import Ceph
from aoss_client.client import Client


DEFAULT_TIMESTEPS = {
    "wan": (499, 999),
    "cogvideox": (499, 999),
    "minimax_h3": (500, 1000),
}
EXPECTED_UNITS = {"FEAT_STREAMING": 10_884, "FEAT_TARGET": 681}


def recursive_object_keys(client: Client, uri: str, page_size: int):
    """Yield full keys via the same S3 LIST API used by ``Client.list``.

    ``Client.list`` fixes ``Delimiter=/`` and therefore exposes one directory
    level at a time. Omitting the delimiter keeps this a metadata-only LIST but
    avoids tens of thousands of directory round trips for streaming trees.
    """
    mixed = client._get_local_client()  # aoss_client has no recursive public LIST
    cluster, bucket, prefix = Ceph.parse_uri(
        uri, mixed._ceph_dict, mixed._default_cluster
    )
    backend = mixed._ceph_dict[cluster]
    s3_client = backend._s3_resource.meta.client
    paginator = s3_client.get_paginator("list_objects")
    pages = paginator.paginate(
        Bucket=bucket,
        Prefix=prefix or "",
        PaginationConfig={"PageSize": page_size},
    )
    for page in pages:
        for item in page.get("Contents", []):
            key = str(item.get("Key", ""))
            if key:
                yield prefix or "", key


def atomic_write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text("".join(f"{line}\n" for line in lines))
    os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="logs/tstep_canonical_v2_seeded_v1/aoss_done",
        help="Directory receiving one completion index per kind/VFM/timestep.",
    )
    parser.add_argument(
        "--conf",
        default=os.environ.get(
            "AOSS_CONF", "/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf"
        ),
    )
    parser.add_argument("--bucket", default="baiqiao")
    parser.add_argument("--dataset-prefix", default="InsScene-15K")
    parser.add_argument(
        "--aoss-api-host",
        default="aoss-internal.cn-fz-01.fjscmsapi-oss.com",
        help="Internal endpoint added to NO_PROXY before metadata-only LIST.",
    )
    parser.add_argument(
        "--kinds", nargs="+", default=["FEAT_STREAMING", "FEAT_TARGET"]
    )
    parser.add_argument(
        "--streaming-namespace",
        default="FEAT_STREAMING_CANONICAL_V2_SEEDED_V1",
        help="AOSS tree used for semantic kind FEAT_STREAMING.",
    )
    parser.add_argument(
        "--target-namespace",
        default="FEAT_TARGET_CANONICAL_V2_SEEDED_V1",
        help="AOSS tree used for semantic kind FEAT_TARGET.",
    )
    parser.add_argument(
        "--expected-streaming-units",
        type=int,
        default=EXPECTED_UNITS["FEAT_STREAMING"],
        help="Expected scene/window/prefix units per streaming timestep.",
    )
    parser.add_argument(
        "--expected-target-units",
        type=int,
        default=EXPECTED_UNITS["FEAT_TARGET"],
        help="Expected scene units per target timestep.",
    )
    parser.add_argument(
        "--vfms", nargs="+", default=["wan", "cogvideox", "minimax_h3"]
    )
    parser.add_argument("--source", default="scannetpp")
    parser.add_argument("--layers", nargs="+", type=int, default=[15, 20, 25])
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument(
        "--require-sidecar",
        action="store_true",
        help=(
            "Write only feature keys with a matching .manifest.json sidecar. "
            "Use when the extractor cache must regenerate incomplete remote pairs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown_vfms = sorted(set(args.vfms) - set(DEFAULT_TIMESTEPS))
    if unknown_vfms:
        raise SystemExit(f"unsupported VFM(s): {', '.join(unknown_vfms)}")
    unknown_kinds = sorted(set(args.kinds) - set(EXPECTED_UNITS))
    if unknown_kinds:
        raise SystemExit(f"unsupported feature kind(s): {', '.join(unknown_kinds)}")

    output_dir = Path(args.output_dir).resolve()
    no_proxy_items = [
        item.strip()
        for item in os.environ.get("NO_PROXY", os.environ.get("no_proxy", "")).split(",")
        if item.strip()
    ]
    for item in (args.aoss_api_host, f".{args.aoss_api_host}"):
        if item not in no_proxy_items:
            no_proxy_items.append(item)
    os.environ["NO_PROXY"] = ",".join(no_proxy_items)
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
    client = Client(args.conf)
    wanted_layers = set(args.layers)
    namespaces = {
        "FEAT_STREAMING": args.streaming_namespace,
        "FEAT_TARGET": args.target_namespace,
    }
    expected_units = {
        "FEAT_STREAMING": args.expected_streaming_units,
        "FEAT_TARGET": args.expected_target_units,
    }
    for kind, namespace in namespaces.items():
        if not re.fullmatch(r"FEAT_[A-Z0-9_]+", namespace):
            raise SystemExit(f"invalid {kind} namespace: {namespace!r}")
        if not namespace.startswith(kind):
            raise SystemExit(
                f"namespace {namespace!r} does not match semantic kind {kind}"
            )
    summary: dict[str, dict[str, object]] = {}

    for kind in args.kinds:
        namespace = namespaces[kind]
        for vfm in args.vfms:
            timesteps = DEFAULT_TIMESTEPS[vfm]
            wanted_names = {
                f"feature_t{t}_layer{layer}.sft": (t, layer)
                for t in timesteps
                for layer in wanted_layers
            }
            wanted_sidecars = {
                f"{name}.manifest.json": match
                for name, match in wanted_names.items()
            }
            by_timestep: dict[int, list[str]] = {t: [] for t in timesteps}
            sidecars_by_timestep: dict[int, set[str]] = {
                t: set() for t in timesteps
            }
            target_index_lines: list[str] = []
            target_meta_lines: list[str] = []
            prefix_index_lines: list[str] = []
            scanned = 0
            uri = (
                f"fj:s3://{args.bucket}/{args.dataset_prefix}/{namespace}/"
                f"{vfm}/{args.source}/"
            )
            print(
                f"LIST {namespace}/{vfm}/{args.source} "
                f"(semantic kind {kind})",
                flush=True,
            )
            for prefix, key in recursive_object_keys(client, uri, args.page_size):
                scanned += 1
                name = key.rsplit("/", 1)[-1]
                match = wanted_names.get(name)
                sidecar_match = wanted_sidecars.get(name)
                is_target_index = kind == "FEAT_TARGET" and name == "target_indices.npy"
                is_target_meta = (
                    kind == "FEAT_TARGET" and name == "target_indices.meta.json"
                )
                is_prefix_index = (
                    kind == "FEAT_STREAMING" and name == "prefix_index.npy"
                )
                if (
                    match is None
                    and sidecar_match is None
                    and not is_target_index
                    and not is_target_meta
                    and not is_prefix_index
                ):
                    continue
                if not key.startswith(prefix):
                    raise RuntimeError(f"listed key escaped prefix: {key}")
                rel = key[len(prefix) :].lstrip("/")
                if not rel or ".." in Path(rel).parts:
                    continue
                if is_target_index:
                    target_index_lines.append(rel)
                elif is_target_meta:
                    target_meta_lines.append(rel)
                elif is_prefix_index:
                    prefix_index_lines.append(rel)
                elif match is not None:
                    timestep, _layer = match
                    by_timestep[timestep].append(rel)
                else:
                    timestep, _layer = sidecar_match
                    sidecars_by_timestep[timestep].add(
                        rel.removesuffix(".manifest.json")
                    )

            target_index_count = 0
            target_meta_count = 0
            prefix_index_count = 0
            if kind == "FEAT_TARGET":
                target_index_lines = sorted(set(target_index_lines))
                target_meta_lines = sorted(set(target_meta_lines))
                target_index_output = output_dir / f"{namespace}_{vfm}_target_indices.txt"
                target_meta_output = output_dir / f"{namespace}_{vfm}_target_indices_meta.txt"
                atomic_write_lines(target_index_output, target_index_lines)
                atomic_write_lines(target_meta_output, target_meta_lines)
                target_index_count = len(target_index_lines)
                target_meta_count = len(target_meta_lines)
                print(
                    f"  {namespace}_{vfm}_target_indices: keys={target_index_count} "
                    f"meta={target_meta_count} -> {target_index_output}",
                    flush=True,
                )
            else:
                prefix_index_lines = sorted(set(prefix_index_lines))
                prefix_index_output = output_dir / f"{namespace}_{vfm}_prefix_indices.txt"
                atomic_write_lines(prefix_index_output, prefix_index_lines)
                prefix_index_count = len(prefix_index_lines)
                print(
                    f"  {namespace}_{vfm}_prefix_indices: "
                    f"keys={prefix_index_count} -> {prefix_index_output}",
                    flush=True,
                )

            for timestep in timesteps:
                listed_lines = sorted(set(by_timestep[timestep]))
                if args.require_sidecar:
                    lines = [
                        rel
                        for rel in listed_lines
                        if rel in sidecars_by_timestep[timestep]
                    ]
                else:
                    lines = listed_lines
                output = output_dir / f"{namespace}_{vfm}_t{timestep}.txt"
                atomic_write_lines(output, lines)

                unit_layers: dict[str, set[int]] = defaultdict(set)
                pattern = re.compile(
                    rf"^feature_t{timestep}_layer({'|'.join(map(str, args.layers))})\.sft$"
                )
                for rel in lines:
                    name = rel.rsplit("/", 1)[-1]
                    match = pattern.fullmatch(name)
                    if match:
                        unit = rel.rsplit("/", 1)[0]
                        unit_layers[unit].add(int(match.group(1)))
                complete_units = sum(
                    layers == wanted_layers for layers in unit_layers.values()
                )
                partial_units = sum(
                    bool(layers) and layers != wanted_layers
                    for layers in unit_layers.values()
                )
                expected = expected_units[kind]
                key_name = f"{namespace}_{vfm}_t{timestep}"
                summary[key_name] = {
                    "index": str(output),
                    "semantic_kind": kind,
                    "aoss_namespace": namespace,
                    "listed_objects_under_tree": scanned,
                    "listed_feature_keys": len(listed_lines),
                    "feature_keys": len(lines),
                    "sidecar_keys": len(sidecars_by_timestep[timestep]),
                    "feature_keys_with_sidecar": sum(
                        rel in sidecars_by_timestep[timestep] for rel in lines
                    ),
                    "complete_units": complete_units,
                    "partial_units": partial_units,
                    "target_index_keys": target_index_count,
                    "target_index_meta_keys": target_meta_count,
                    "prefix_index_keys": prefix_index_count,
                    "expected_units": expected,
                    "completion_fraction": complete_units / expected,
                }
                print(
                    f"  {key_name}: keys={len(lines)} complete={complete_units}/"
                    f"{expected} partial={partial_units} -> {output}",
                    flush=True,
                )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"WROTE {summary_path}", flush=True)


if __name__ == "__main__":
    main()
