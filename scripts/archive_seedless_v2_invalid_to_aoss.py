#!/usr/bin/env python3
"""Archive the frozen seedless-v2 local tree to AOSS, then free feature bytes.

The archive is transaction-like and fail closed:

1. Revalidate the immutable inventory made by the H800 inventory canary.
2. Exclusively lock the frozen source tree and validate its exact file set.
3. Run ads-cli directly on that immutable tree without --delete-src.
4. Require rc=0, failed=0, exact Found count, and an authoritative AOSS LIST
   with the exact relative keys and sizes.
5. Write a local receipt, then unlink only frozen feature/sidecar/temp files.

Canonical structural indices and all audit artifacts remain local.
"""

from __future__ import annotations

import configparser
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import traceback
from pathlib import Path

from aoss_client.ceph.ceph import Ceph
from aoss_client.client import Client

ROOT = Path(
    "/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2"
)
INVENTORY = ROOT / "INVALID_LOCAL_ARCHIVE_INVENTORY.tsv"
SUMMARY = ROOT / "INVALID_LOCAL_ARCHIVE_SUMMARY.json"
ARCHIVE_NAME = "FEAT_STREAMING_CANONICAL_V2_SEEDLESS_V2_20260830"
REMOTE_URI = f"fj:s3://baiqiao/InsScene-15K/INVALID_ARCHIVE/{ARCHIVE_NAME}/"
REMOTE_PREFIX = f"InsScene-15K/INVALID_ARCHIVE/{ARCHIVE_NAME}/"
STALE_STAGING = ROOT / ".INVALID_AOSS_ARCHIVE_STAGING"
RECEIPT = ROOT / "INVALID_AOSS_ARCHIVE_RECEIPT.json"
ERROR_REPORT = ROOT / "INVALID_AOSS_ARCHIVE_ERROR.json"
ADS_LOG = Path("/tmp/INVALID_AOSS_ARCHIVE_ADS.log")
LOCK = ROOT / "wan/scannetpp/.tstep_sync.lock"
STALE_ARCHIVE_LOCK = ROOT / ".INVALID_AOSS_ARCHIVE.lock"
CONF = Path("/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf")
ADS_CLI = Path("/mnt/umm/users/yinbaiqiao/aoss/ads-cli")
API_HOST = "aoss-internal.cn-fz-01.fjscmsapi-oss.com"
EXPECTED_INVENTORY_SHA256 = (
    "6f80a51fb730fc98d83851b0850fe62a8f153988495ce19275edb0e6efc7d6e8"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(path: Path) -> tuple[int, int, int]:
    st = path.stat(follow_symlinks=False)
    return st.st_dev, st.st_ino, st.st_size


def load_inventory() -> tuple[dict[str, tuple[int, int, int, str]], dict]:
    if INVENTORY.is_symlink() or not INVENTORY.is_file():
        fail(f"missing/non-regular inventory: {INVENTORY}")
    if file_sha256(INVENTORY) != EXPECTED_INVENTORY_SHA256:
        fail("inventory SHA256 changed")
    if SUMMARY.is_symlink() or not SUMMARY.is_file():
        fail(f"missing/non-regular summary: {SUMMARY}")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if summary.get("status") != "INVALID_SEEDLESS_V2_ARCHIVE_PREPARED":
        fail("invalid inventory status")
    if summary.get("inventory_sha256") != EXPECTED_INVENTORY_SHA256:
        fail("summary/inventory SHA mismatch")

    lines = INVENTORY.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "relpath\tsize_bytes\tdevice\tinode\tkind":
        fail("inventory header mismatch")
    rows: dict[str, tuple[int, int, int, str]] = {}
    for line in lines[1:]:
        rel, size, device, inode, kind = line.split("\t")
        path = Path(rel)
        if path.is_absolute() or ".." in path.parts or not rel:
            fail(f"unsafe inventory relpath: {rel!r}")
        if kind not in {"feature", "sidecar", "temp", "structural_or_other"}:
            fail(f"invalid inventory kind: {kind}")
        if rel in rows:
            fail(f"duplicate inventory relpath: {rel}")
        rows[rel] = (int(device), int(inode), int(size), kind)
    if len(rows) != 2992:
        fail(f"expected 2992 inventory rows, found {len(rows)}")
    return rows, summary


def validate_source(rows: dict[str, tuple[int, int, int, str]]) -> None:
    for rel, (device, inode, size, _kind) in rows.items():
        path = ROOT / rel
        if path.is_symlink() or not path.is_file():
            fail(f"inventory source missing/non-regular: {rel}")
        current = signature(path)
        # QuarkFS exposes a mount-namespace-specific st_dev inside different
        # SCO pods.  Inode+size are stable across jobs; the same-job hardlink
        # staging check below still requires the full dev/inode/size tuple.
        if current[1:] != (inode, size):
            fail(f"inventory source changed: {rel}")


def exact_source_files(rows: dict[str, tuple[int, int, int, str]]) -> dict[str, int]:
    expected: dict[str, int] = {}
    sources = {rel: ROOT / rel for rel in rows}
    sources[INVENTORY.name] = INVENTORY
    sources[SUMMARY.name] = SUMMARY
    for rel, source in sorted(sources.items()):
        if source.is_symlink() or not source.is_file():
            fail(f"expected archive source missing/non-regular: {rel}")
        expected[rel] = source.stat(follow_symlinks=False).st_size
    actual = {
        path.relative_to(ROOT).as_posix(): path.stat(follow_symlinks=False).st_size
        for path in ROOT.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        wrong_size = sorted(
            key for key in set(actual) & set(expected) if actual[key] != expected[key]
        )
        fail(
            "source tree exact-file gate failed: "
            f"missing={len(missing)} extra={len(extra)} wrong_size={len(wrong_size)}"
        )
    return expected


def aoss_objects() -> dict[str, int]:
    no_proxy = f"localhost,127.0.0.1,::1,baiqiao.{API_HOST},.{API_HOST},{API_HOST}"
    os.environ["NO_PROXY"] = no_proxy
    os.environ["no_proxy"] = no_proxy
    client = Client(str(CONF))
    mixed = client._get_local_client()
    cluster, bucket, prefix = Ceph.parse_uri(
        REMOTE_URI, mixed._ceph_dict, mixed._default_cluster
    )
    backend = mixed._ceph_dict[cluster]
    s3_client = backend._s3_resource.meta.client
    paginator = s3_client.get_paginator("list_objects")
    result: dict[str, int] = {}
    for page in paginator.paginate(
        Bucket=bucket,
        Prefix=prefix or "",
        PaginationConfig={"PageSize": 1000},
    ):
        for item in page.get("Contents", []):
            key = str(item.get("Key", ""))
            if not key.startswith(prefix or ""):
                fail(f"AOSS LIST escaped archive prefix: {key}")
            rel = key[len(prefix or "") :].lstrip("/")
            if not rel or ".." in Path(rel).parts or rel in result:
                fail(f"invalid/duplicate AOSS archive key: {key}")
            result[rel] = int(item.get("Size", -1))
    return result


def ads_destination() -> str:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(CONF)
    if not parser.has_section("fj"):
        fail("missing [fj] AOSS configuration")
    access_key = parser.get("fj", "access_key", fallback="").strip()
    secret_key = parser.get("fj", "secret_key", fallback="").strip()
    if not access_key or not secret_key:
        fail("missing AOSS credentials")
    return (
        f"s3://{access_key}:{secret_key}@baiqiao.{API_HOST}/"
        f"{REMOTE_PREFIX}"
    )


def run_ads(expected_count: int) -> dict[str, int | str]:
    if not ADS_CLI.is_file() or not os.access(ADS_CLI, os.X_OK):
        fail(f"ads-cli missing/non-executable: {ADS_CLI}")
    env = os.environ.copy()
    no_proxy = f"localhost,127.0.0.1,::1,baiqiao.{API_HOST},.{API_HOST},{API_HOST}"
    env["NO_PROXY"] = no_proxy
    env["no_proxy"] = no_proxy
    command = [
        str(ADS_CLI),
        "-p", "32",
        "-l", "8",
        "--conntimeout", "120",
        "--timeout", "600",
        "sync",
        f"{ROOT}/",
        ads_destination(),
    ]
    with ADS_LOG.open("wb") as output:
        completed = subprocess.run(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    text = ADS_LOG.read_bytes().replace(b"\x00", b"\n").replace(b"\r", b"\n").decode(
        "utf-8", errors="replace"
    )
    summaries = [line for line in text.splitlines() if "Found:" in line and "failed:" in line]
    summary = summaries[-1] if summaries else ""
    found_match = re.search(r"Found:\s*([0-9][0-9,]*)", summary)
    failed_match = re.search(r"failed:\s*([0-9][0-9,]*)", summary)
    found = int(found_match.group(1).replace(",", "")) if found_match else -1
    failed = int(failed_match.group(1).replace(",", "")) if failed_match else -1
    if completed.returncode != 0 or failed != 0 or found != expected_count:
        fail(
            f"ads coverage failed rc={completed.returncode} found={found} "
            f"expected={expected_count} failed={failed}"
        )
    return {"rc": completed.returncode, "found": found, "failed": failed, "summary": summary}


def write_receipt(expected: dict[str, int], ads: dict, summary: dict) -> None:
    receipt = {
        "schema_version": 1,
        "status": "AOSS_ARCHIVE_VERIFIED",
        "source_root": str(ROOT),
        "remote_uri": REMOTE_URI,
        "inventory_sha256": EXPECTED_INVENTORY_SHA256,
        "source_object_count": len(expected),
        "source_total_bytes": sum(expected.values()),
        "ads": ads,
        "authoritative_list": {
            "object_count": len(expected),
            "total_bytes": sum(expected.values()),
            "exact_keys_and_sizes": True,
        },
        "frozen_counts": summary["counts"],
        "frozen_bytes": summary["bytes"],
    }
    temp = RECEIPT.with_name(f".{RECEIPT.name}.tmp-{os.getpid()}")
    temp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, RECEIPT)
    os.chmod(RECEIPT, 0o644)


def delete_archived_payload(rows: dict[str, tuple[int, int, int, str]]) -> dict[str, int]:
    deleted = {"feature": 0, "sidecar": 0, "temp": 0}
    for rel, (device, inode, size, kind) in rows.items():
        if kind not in deleted:
            continue
        path = ROOT / rel
        if path.is_symlink() or not path.is_file():
            fail(f"source disappeared before verified delete: {rel}")
        if signature(path)[1:] != (inode, size):
            fail(f"source changed before verified delete: {rel}")
    for rel, (_device, _inode, _size, kind) in rows.items():
        if kind in deleted:
            (ROOT / rel).unlink()
            deleted[kind] += 1
    return deleted


def main() -> None:
    if ROOT.is_symlink() or not ROOT.is_dir():
        fail(f"invalid source root: {ROOT}")
    if ERROR_REPORT.exists():
        if ERROR_REPORT.is_symlink() or not ERROR_REPORT.is_file():
            fail(f"unsafe stale error report: {ERROR_REPORT}")
        ERROR_REPORT.unlink()
    if STALE_STAGING.exists():
        if (
            STALE_STAGING.is_symlink()
            or not STALE_STAGING.is_dir()
            or STALE_STAGING.parent != ROOT
        ):
            fail(f"unsafe stale staging path: {STALE_STAGING}")
        shutil.rmtree(STALE_STAGING)
    if STALE_ARCHIVE_LOCK.exists():
        if STALE_ARCHIVE_LOCK.is_symlink() or not STALE_ARCHIVE_LOCK.is_file():
            fail(f"unsafe stale archive lock: {STALE_ARCHIVE_LOCK}")
        STALE_ARCHIVE_LOCK.unlink()
    if LOCK.is_symlink() or not LOCK.is_file():
        fail(f"missing/non-regular extraction lock: {LOCK}")
    with LOCK.open("r+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows, summary = load_inventory()
        validate_source(rows)
        expected = exact_source_files(rows)

        before = aoss_objects()
        unexpected = {
            key: size for key, size in before.items()
            if key not in expected or expected[key] != size
        }
        if unexpected:
            fail(f"remote archive contains unexpected keys/sizes: {len(unexpected)}")

        ads = run_ads(len(expected))
        validate_source(rows)
        if exact_source_files(rows) != expected:
            fail("source tree changed during ads-cli")
        remote = aoss_objects()
        if remote != expected:
            missing = sorted(set(expected) - set(remote))
            extra = sorted(set(remote) - set(expected))
            wrong_size = sorted(
                key for key in set(remote) & set(expected) if remote[key] != expected[key]
            )
            fail(
                "authoritative AOSS LIST mismatch: "
                f"missing={len(missing)} extra={len(extra)} wrong_size={len(wrong_size)}"
            )

        write_receipt(expected, ads, summary)
        deleted = delete_archived_payload(rows)
        remaining = {kind: 0 for kind in deleted}
        for rel, (_device, _inode, _size, kind) in rows.items():
            if kind in remaining and (ROOT / rel).exists():
                remaining[kind] += 1
        if any(remaining.values()):
            fail(f"verified local deletion incomplete: {remaining}")
        print(json.dumps({"status": "COMPLETE", "deleted": deleted, "remote": len(remote)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        report = {
            "schema_version": 1,
            "status": "FAILED_BEFORE_VERIFIED_DELETE",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        temp = ERROR_REPORT.with_name(f".{ERROR_REPORT.name}.tmp-{os.getpid()}")
        temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, ERROR_REPORT)
        os.chmod(ERROR_REPORT, 0o644)
        raise
