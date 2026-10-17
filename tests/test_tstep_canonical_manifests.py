from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_tstep_canonical_manifests import (
    CANONICAL_OBS_LEN,
    DEFAULT_PREFIXES,
    MANIFEST_NAME,
    PROTOCOL_ID,
    STREAMING_FEATURE_ROOT,
    TARGET_FEATURE_ROOT,
    ManifestError,
    audit_manifests,
    build_manifests,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def source_row(scene: str, window_id: int, split: str, offset: int) -> dict:
    window = list(range(offset, offset + CANONICAL_OBS_LEN))
    return {
        "sample_id": f"{scene}_w{window_id:03d}_p24",
        "split": split,
        "source": "scannetpp",
        "scene_id": scene,
        "scene_dir": f"/dataset/{scene}",
        "num_frames": 200,
        "valid_frames": [f"frame_{index:06d}.jpg" for index in range(200)],
        "valid_pose_indices": list(range(200)),
        "streaming_tail": 23,
        "streaming_tail_raw": window[23],
        "streaming_indices": window[:24],
        "streaming_window_indices": window,
        "streaming_window_id": window_id,
        "streaming_obs_start": window_id * 8,
        "streaming_sampling": "motion_uniform",
        "streaming_feature_root": "/legacy/prefix24",
        "target_feature_root": "/legacy/target24",
    }


class CanonicalManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source_prefix24.jsonl"
        self.rows = [
            source_row("scene_a", 0, "train", 0),
            source_row("scene_a", 1, "train", 40),
            source_row("scene_b", 0, "val", 80),
        ]
        write_jsonl(self.source, self.rows)
        self.frozen = self.root / "frozen"
        for prefix in DEFAULT_PREFIXES:
            split_rows = [
                {"scene_id": row["scene_id"], "split": row["split"]}
                for row in self.rows
            ]
            write_jsonl(self.frozen / MANIFEST_NAME.format(prefix=prefix), split_rows)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def build(self) -> Path:
        output = self.root / "output"
        build_manifests(
            canonical_manifest=self.source,
            split_manifests=[
                self.frozen / MANIFEST_NAME.format(prefix=prefix)
                for prefix in DEFAULT_PREFIXES
            ],
            output_dir=output,
            expected_shared_units=3,
        )
        return output

    def test_builds_one_shared_window_universe_and_audits_hashes(self) -> None:
        output = self.build()
        result = audit_manifests(manifest_dir=output, expected_shared_units=3)
        self.assertTrue(result["ok"])
        self.assertEqual(result["shared_unit_count"], 3)
        self.assertEqual(result["logical_sample_count"], 12)

        windows_by_prefix = {}
        for prefix in DEFAULT_PREFIXES:
            path = output / MANIFEST_NAME.format(prefix=prefix)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            windows_by_prefix[prefix] = [row["streaming_window_indices"] for row in rows]
            for row in rows:
                self.assertEqual(row["canonical_protocol"], PROTOCOL_ID)
                self.assertEqual(row["canonical_obs_len"], CANONICAL_OBS_LEN)
                self.assertEqual(row["prefix_length"], prefix)
                self.assertEqual(row["target_horizons"], [1, 2, 4])
                self.assertIs(row["include_anchors"], True)
                self.assertEqual(row["streaming_indices"], row["streaming_window_indices"][:prefix])
                self.assertEqual(row["streaming_feature_root"], STREAMING_FEATURE_ROOT)
                self.assertEqual(row["target_feature_root"], TARGET_FEATURE_ROOT)
        for prefix in DEFAULT_PREFIXES[1:]:
            self.assertEqual(windows_by_prefix[prefix], windows_by_prefix[8])

    def test_duplicate_source_unit_fails_before_generation(self) -> None:
        duplicate_rows = self.rows + [copy.deepcopy(self.rows[0])]
        duplicate_rows[-1]["sample_id"] = "different_id"
        write_jsonl(self.source, duplicate_rows)
        with self.assertRaisesRegex(ManifestError, "duplicate canonical unit"):
            build_manifests(
                canonical_manifest=self.source,
                split_manifests=[self.frozen / MANIFEST_NAME.format(prefix=24)],
                output_dir=self.root / "duplicate-output",
                expected_shared_units=0,
            )

    def test_cross_prefix_misalignment_is_rejected(self) -> None:
        output = self.build()
        path = output / MANIFEST_NAME.format(prefix=12)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["streaming_indices"][0] += 1
        write_jsonl(path, rows)
        with self.assertRaisesRegex(ManifestError, "streaming_indices do not match"):
            audit_manifests(manifest_dir=output, expected_shared_units=3)

    def test_old_v2_feature_root_is_rejected(self) -> None:
        output = self.build()
        path = output / MANIFEST_NAME.format(prefix=8)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["streaming_feature_root"] = (
            "aoss://baiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2"
        )
        write_jsonl(path, rows)
        with self.assertRaisesRegex(ManifestError, "streaming_feature_root"):
            audit_manifests(manifest_dir=output, expected_shared_units=3)


if __name__ == "__main__":
    unittest.main()
