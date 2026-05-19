#!/usr/bin/env python3
"""
Download and extract InsScene-15K dataset (infinigen + scannetpp_v2).

Usage:
    python download_and_extract.py [--step download|extract|all]
"""
import argparse
import glob
import os
import subprocess
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "lifuguan/InsScene-15K"
# Override with --base-dir argument or set BASE_DIR here
BASE_DIR = Path(os.environ.get("INSCENE_DIR", "/data/InsScene-15K"))
RAW_DIR = BASE_DIR / "raw"  # HF cache / raw downloads
DATA_DIR = BASE_DIR / "data"  # Extracted data


def download_infinigen():
    """Download processed_infinigen (individual zip files per subscene)."""
    print("=" * 60)
    print("Downloading processed_infinigen ...")
    print("=" * 60)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(RAW_DIR),
        allow_patterns=["processed_infinigen/**"],
    )
    print("processed_infinigen download complete.")


def download_scannetpp():
    """Download processed_scannetpp_v2 (multi-part zip files)."""
    print("=" * 60)
    print("Downloading processed_scannetpp_v2 ...")
    print("=" * 60)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(RAW_DIR),
        allow_patterns=["processed_scannetpp_v2/**"],
    )
    print("processed_scannetpp_v2 download complete.")


def extract_infinigen():
    """Extract each subscene zip in processed_infinigen."""
    print("=" * 60)
    print("Extracting processed_infinigen ...")
    print("=" * 60)
    src_root = RAW_DIR / "processed_infinigen"
    dst_root = DATA_DIR / "processed_infinigen"
    dst_root.mkdir(parents=True, exist_ok=True)

    # Also copy metadata files
    for meta in ["dataset_metadata.json", "README.md"]:
        meta_src = src_root / meta
        meta_dst = dst_root / meta
        if meta_src.exists() and not meta_dst.exists():
            import shutil
            shutil.copy2(meta_src, meta_dst)

    scene_dirs = sorted(
        [d for d in src_root.iterdir() if d.is_dir() and d.name.startswith("scene_")]
    )
    total_scenes = len(scene_dirs)
    extracted = 0

    for si, scene_dir in enumerate(scene_dirs):
        zips = sorted(scene_dir.glob("*.zip"))
        for zi, zf in enumerate(zips):
            subscene_name = zf.stem  # e.g. "16255241"
            out_dir = dst_root / scene_dir.name / subscene_name
            # Skip if already extracted
            if out_dir.exists() and any(out_dir.iterdir()):
                extracted += 1
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(zf, "r") as z:
                    z.extractall(out_dir)
                extracted += 1
                if extracted % 50 == 0:
                    print(f"  Extracted {extracted} subscenes ...")
            except Exception as e:
                print(f"  ERROR extracting {zf}: {e}")

        if (si + 1) % 10 == 0:
            print(f"  Scenes processed: {si+1}/{total_scenes}")

    print(f"Infinigen extraction done: {extracted} subscenes total.")


def extract_scannetpp():
    """Reassemble multi-part zip and extract processed_scannetpp_v2."""
    print("=" * 60)
    print("Extracting processed_scannetpp_v2 ...")
    print("=" * 60)
    src_root = RAW_DIR / "processed_scannetpp_v2"
    dst_root = DATA_DIR / "processed_scannetpp_v2"
    dst_root.mkdir(parents=True, exist_ok=True)

    combined_zip = BASE_DIR / "processed_scannetpp_v2_combined.zip"

    if not combined_zip.exists():
        # Find all parts and sort them
        parts = sorted(glob.glob(str(src_root / "processed_scannetpp_v2.zip.*")))
        if not parts:
            print("  No zip parts found! Skipping.")
            return

        print(f"  Reassembling {len(parts)} parts into {combined_zip} ...")
        # Use shell `cat` for efficiency on large files
        cmd = f"cat {' '.join(parts)} > {combined_zip}"
        subprocess.run(cmd, shell=True, check=True)
        print(f"  Combined zip size: {combined_zip.stat().st_size / (1024**3):.1f} GB")
    else:
        print(f"  Combined zip already exists: {combined_zip.stat().st_size / (1024**3):.1f} GB")

    # Extract
    print(f"  Extracting to {dst_root} ...")
    subprocess.run(
        ["unzip", "-o", "-q", str(combined_zip), "-d", str(dst_root)],
        check=True,
    )
    print("ScanNet++ extraction done.")

    # Clean up combined zip to save space
    print(f"  Removing combined zip to save space ...")
    combined_zip.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        choices=["download", "extract", "all", "download_infinigen",
                 "download_scannetpp", "extract_infinigen", "extract_scannetpp"],
        default="all",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Root directory for the dataset (overrides INSCENE_DIR env var). "
             "E.g. --base-dir /data/InsScene-15K",
    )
    args = parser.parse_args()

    global BASE_DIR, RAW_DIR, DATA_DIR
    if args.base_dir:
        BASE_DIR = Path(args.base_dir)
        RAW_DIR = BASE_DIR / "raw"
        DATA_DIR = BASE_DIR / "data"

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.step in ("download", "all", "download_infinigen"):
        download_infinigen()
    if args.step in ("download", "all", "download_scannetpp"):
        download_scannetpp()
    if args.step in ("extract", "all", "extract_infinigen"):
        extract_infinigen()
    if args.step in ("extract", "all", "extract_scannetpp"):
        extract_scannetpp()

    if args.step in ("extract", "all"):
        # Print final stats
        print("\n" + "=" * 60)
        print("Final dataset statistics:")
        for source in ["processed_infinigen", "processed_scannetpp_v2"]:
            src_path = DATA_DIR / source
            if src_path.exists():
                if source == "processed_infinigen":
                    scenes = [d for d in src_path.iterdir() if d.is_dir() and d.name.startswith("scene_")]
                    subscenes = sum(1 for s in scenes for d in s.iterdir() if d.is_dir())
                    print(f"  {source}: {len(scenes)} scenes, {subscenes} subscenes")
                else:
                    scenes = [d for d in src_path.iterdir() if d.is_dir()]
                    print(f"  {source}: {len(scenes)} scenes")
        print("=" * 60)


if __name__ == "__main__":
    main()
