from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import pytest

from analyses.windowfix_v2.build_feature_inventory import (
    ROOT,
    InventoryError,
    build_inventory,
    write_inventory,
)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_run_manifest(
    root: Path, run_name: str, *, vfm: str, layer: int, timestep: int
) -> str:
    run_dir = root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config": {
                    "vfm_name": vfm,
                    "feature_layer": layer,
                    "feature_timestep": timestep,
                },
            }
        ),
        encoding="utf-8",
    )
    return str(run_dir.relative_to(root))


def fixture_protocol(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "canonical_protocol": "inscene_streaming_canonical_v2_obs28_h124",
                "canonical_window": {
                    "observations": 28,
                    "sampling": "motion_uniform",
                    "motion_step": 0.35,
                    "rotation_weight": 0.5,
                    "window_stride": 8,
                    "max_windows_per_scene": 4,
                },
                "streaming": {"prefix_lengths": [8, 12, 16, 24]},
                "target": {
                    "prefix_lengths": [8, 12, 16, 24],
                    "future_horizons": [1, 2, 4],
                    "include_anchors": True,
                },
                "expected_universe": {
                    "canonical_windows": 2721,
                    "target_scene_units": 681,
                },
            }
        ),
        encoding="utf-8",
    )


class FeatureInventoryFixtureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.protocol = self.root / "protocol.json"
        fixture_protocol(self.protocol)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_deduplicates_tasks_and_resolves_missing_manifest_from_siblings(self) -> None:
        manifest = write_run_manifest(
            self.root, "minimax_l50_a", vfm="minimax_h3", layer=49, timestep=250
        )
        rows = [
            {
                "experiment_id": "a1",
                "campaign": "minimax_h3_layer_benchmark",
                "model": "minimax_h3",
                "model_variant": "multi_layer",
                "layer": "50",
                "prefix": "24",
                "probe_config": "streaming_depth",
                "probe_family": "A1",
                "run_artifact_path": manifest,
            },
            {
                "experiment_id": "c1",
                "campaign": "minimax_h3_layer_benchmark",
                "model": "minimax_h3",
                "model_variant": "multi_layer",
                "layer": "50",
                "prefix": "24",
                "probe_config": "action_dynamics",
                "probe_family": "C1",
                "run_artifact_path": "runs/not_started",
            },
        ]
        ledger = self.root / "ledger.csv"
        write_csv(ledger, rows)
        document = build_inventory(
            ledger_path=ledger,
            protocol_path=self.protocol,
            project_root=self.root,
            require_frozen_ledger=False,
            expected_rows=2,
        )
        self.assertEqual(document["counts"]["feature_specs"], 1)
        self.assertEqual(document["counts"]["sibling_consensus_resolved_experiments"], 1)
        self.assertEqual(
            document["physical_layer_aliases"],
            [{"model": "minimax_h3", "ledger_layer": 50, "feature_layer": 49}],
        )
        spec = document["specs"][0]
        self.assertEqual(spec["ledger_layers"], "50")
        self.assertEqual(spec["feature_layer"], 49)
        self.assertEqual(spec["timestep"], 250)
        self.assertEqual(spec["feature_filename"], "feature_t250_layer49.sft")
        self.assertEqual(spec["tasks"], "action_dynamics;streaming_depth")
        self.assertEqual(spec["requires_target"], "true")
        self.assertEqual(spec["experiment_count"], 2)
        self.assertEqual(spec["missing_run_manifest_count"], 1)

    def test_conflicting_physical_resolution_fails_closed(self) -> None:
        first = write_run_manifest(
            self.root, "lingbot_a", vfm="lingbot", layer=47, timestep=250
        )
        second = write_run_manifest(
            self.root, "lingbot_b", vfm="lingbot", layer=48, timestep=250
        )
        rows = []
        for index, manifest in enumerate((first, second)):
            rows.append(
                {
                    "experiment_id": f"row{index}",
                    "campaign": "lingbot_video_layer_benchmark",
                    "model": "lingbot",
                    "model_variant": "multi_layer",
                    "layer": "48",
                    "prefix": "8",
                    "probe_config": "streaming_depth",
                    "probe_family": "A1",
                    "run_artifact_path": manifest,
                }
            )
        ledger = self.root / "ledger.csv"
        write_csv(ledger, rows)
        with self.assertRaisesRegex(InventoryError, "conflicting run-manifest"):
            build_inventory(
                ledger_path=ledger,
                protocol_path=self.protocol,
                project_root=self.root,
                require_frozen_ledger=False,
                expected_rows=2,
            )

    def test_writes_deterministic_versioned_csv_and_json(self) -> None:
        manifest = write_run_manifest(
            self.root, "dino", vfm="dino", layer=0, timestep=749
        )
        ledger = self.root / "ledger.csv"
        write_csv(
            ledger,
            [
                {
                    "experiment_id": "dino-a2",
                    "campaign": "default8",
                    "model": "dino",
                    "model_variant": "default_layer_benchmark",
                    "layer": "0",
                    "prefix": "8",
                    "probe_config": "view_consistency",
                    "probe_family": "A2",
                    "run_artifact_path": manifest,
                }
            ],
        )
        document = build_inventory(
            ledger_path=ledger,
            protocol_path=self.protocol,
            project_root=self.root,
            require_frozen_ledger=False,
            expected_rows=1,
        )
        csv_path, json_path = write_inventory(document, self.root / "output")
        self.assertIn(document["inventory_version"], csv_path.name)
        self.assertIn(document["inventory_version"], json_path.name)
        with csv_path.open(newline="", encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertEqual(len(csv_rows), 1)
        self.assertEqual(csv_rows[0]["action"], "PENDING_CACHE_AUDIT")
        self.assertEqual(csv_rows[0]["requires_target"], "false")
        self.assertEqual(json.loads(json_path.read_text())["inventory_sha256"], document["inventory_sha256"])


@pytest.mark.artifact
class FrozenFeatureInventoryIntegrationTest(unittest.TestCase):
    def test_frozen_1096_resolves_to_expected_physical_inventory(self) -> None:
        document = build_inventory()
        counts = document["counts"]
        self.assertEqual(counts["experiments"], 1096)
        self.assertEqual(counts["feature_specs"], 160)
        self.assertEqual(counts["target_required_specs"], 128)
        self.assertEqual(counts["streaming_extraction_groups"], 40)
        self.assertEqual(counts["target_extraction_groups"], 32)
        self.assertEqual(counts["run_manifest_resolved_experiments"], 1092)
        self.assertEqual(counts["sibling_consensus_resolved_experiments"], 4)
        self.assertEqual(
            counts["feature_specs_by_vfm"],
            {
                "aether": 4,
                "cogvideox": 36,
                "dino": 4,
                "f3r": 4,
                "lingbot": 20,
                "minimax_h3": 20,
                "qwen2_5_vl": 4,
                "videomae": 20,
                "vjepa2": 20,
                "wan": 28,
            },
        )
        self.assertEqual(
            document["physical_layer_aliases"],
            [
                {"model": "lingbot", "ledger_layer": 48, "feature_layer": 47},
                {"model": "minimax_h3", "ledger_layer": 50, "feature_layer": 49},
            ],
        )
        qwen_specs = [row for row in document["specs"] if row["vfm"] == "qwen2_5_vl"]
        self.assertEqual(len(qwen_specs), 4)
        self.assertTrue(
            all(
                row["extractor_entrypoint"] == "features/run_inscene15k_mllm.py"
                for row in qwen_specs
            )
        )
        self.assertTrue(
            all(
                row["extractor_status"] in {"PRESENT", "MISSING_IN_WORKTREE"}
                for row in qwen_specs
            )
        )


if __name__ == "__main__":
    unittest.main()
