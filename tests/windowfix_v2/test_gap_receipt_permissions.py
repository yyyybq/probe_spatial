from __future__ import annotations

import os
from pathlib import Path

from scripts.windowfix_v2.audit_gap_receipt_permissions import audit_root


def test_audit_reports_non_0644_local_receipt(tmp_path: Path) -> None:
    receipt = (
        tmp_path
        / ".windowfix_v2_gap_logs"
        / "wan"
        / "shard"
        / "attempt"
        / "receipt.json"
    )
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"status":"PASS"}\n', encoding="utf-8")
    os.chmod(receipt, 0o600)

    report = audit_root(tmp_path)

    assert report["local_receipts"] == 1
    assert report["issue_count"] == 1
    assert report["issues"][0]["path"] == str(receipt)
    assert "mode_not_0644" in report["issues"][0]["issues"]


def test_audit_reports_symlink_remote_completion(tmp_path: Path) -> None:
    target = tmp_path / "real_remote_completion.json"
    target.write_text('{"status":"PASS"}\n', encoding="utf-8")
    remote = (
        tmp_path
        / ".windowfix_v2_gap_remote_receipts"
        / ("a" * 64)
        / "remote_completion.json"
    )
    remote.parent.mkdir(parents=True)
    remote.symlink_to(target)

    report = audit_root(tmp_path)

    assert report["remote_completion_receipts"] == 1
    assert report["issue_count"] == 1
    assert report["issues"][0]["path"] == str(remote)
    assert "symlink" in report["issues"][0]["issues"]
