from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


ROOT = Path(__file__).resolve().parents[2]
PREPARE = ROOT / "scripts/windowfix_v2/prepare_sco_feature_jobs.py"
RUNNER = ROOT / "scripts/windowfix_v2/run_canonical_feature_shard.py"
COMPACT_AUDIT = ROOT / "scripts/windowfix_v2/audit_compact_alignment.py"


def load_prepare_module():
    spec = importlib.util.spec_from_file_location("windowfix_prepare_sco", PREPARE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_compact_audit_module():
    spec = importlib.util.spec_from_file_location("windowfix_compact_audit", COMPACT_AUDIT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ScoFeatureJobPreparationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="windowfix-v2-sco-test-")
        self.base = Path(self.temp.name)
        self.protocol = self.base / "protocol.json"
        payload = {
            "schema_version": 2,
            "canonical_protocol": "inscene_streaming_canonical_v2_obs28_h124",
            "source": "scannetpp",
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
            "feature_namespaces": {
                "streaming": "FEAT_STREAMING_CANONICAL_V2_TEST",
                "target": "FEAT_TARGET_CANONICAL_V2_TEST",
            },
            "expected_universe": {"canonical_windows": 1, "eligible_scenes": 1},
            "acceptance": {"legacy_features_valid": False, "audit_gate": "fail_closed"},
        }
        self.protocol.write_text(json.dumps(payload, indent=2) + "\n")
        module = load_prepare_module()
        self.protocol_file_sha = hashlib.sha256(self.protocol.read_bytes()).hexdigest()
        self.feature_protocol_sha = module.extractor_protocol_sha(payload)
        self.canonical_source = self.base / "canonical_obs28.jsonl"
        source_row = {
            "scene_id": "scene-a",
            "scene_dir": "/raw/scene-a",
            "split": "train",
            "source": "scannetpp",
            "streaming_sampling": "motion_uniform",
            "streaming_window_id": 0,
            "streaming_obs_start": 0,
            "streaming_window_indices": list(range(28)),
            "streaming_indices": list(range(24)),
            "streaming_tail": 23,
            "streaming_tail_raw": 23,
            "valid_frames": [f"frame_{index:06d}.jpg" for index in range(28)],
            "valid_pose_indices": list(range(28)),
            "canonical_protocol": "inscene_streaming_canonical_v2_obs28_h124",
        }
        self.canonical_source.write_text(json.dumps(source_row) + "\n")
        self.canonical_source_sha = hashlib.sha256(self.canonical_source.read_bytes()).hexdigest()
        self.source_sidecar = Path(f"{self.canonical_source}.meta.json")
        self.source_sidecar.write_text(
            json.dumps(
                {
                    "sha256": self.canonical_source_sha,
                    "prefix_lengths": [24],
                    "action_horizons": [1, 2, 4],
                    "num_samples": 1,
                    "num_included_scenes": 1,
                }
            )
        )
        self.source_sidecar_sha = hashlib.sha256(self.source_sidecar.read_bytes()).hexdigest()
        canonical_universe_payload = {
            "scene_id": "scene-a",
            "split": "train",
            "streaming_window_id": 0,
            "streaming_window_indices": list(range(28)),
        }
        universe_sha = hashlib.sha256(
            (
                json.dumps(canonical_universe_payload, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode()
        ).hexdigest()
        label_lineage_universe_payload = {
            "scene": "scene-a",
            "split": "train",
            "window_id": 0,
            "window": list(range(28)),
        }
        label_lineage_universe_sha = hashlib.sha256(
            (
                json.dumps(
                    label_lineage_universe_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
        ).hexdigest()
        self.namespace_lock = self.base / "namespace_lock.json"
        lock = {
            "schema_version": 2,
            "canonical_protocol": "inscene_streaming_canonical_v2_obs28_h124",
            "protocol_file_sha256": self.protocol_file_sha,
            "canonical_source": {
                "path": str(self.canonical_source),
                "sha256": self.canonical_source_sha,
                "sidecar_path": str(self.source_sidecar),
                "sidecar_sha256": self.source_sidecar_sha,
                "rows": 1,
                "scenes": 1,
                "canonical_obs_len": 28,
                "source_prefix": 24,
            },
            "canonical_universe_sha256": universe_sha,
            "canonical_windows": 1,
            "feature_roots": {
                "streaming": "aoss://baiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_TEST",
                "target": "aoss://baiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_TEST",
            },
            "override_policy": {
                "fail_closed": True,
                "row_roots_are_not_authoritative": True,
            },
        }
        self.namespace_lock.write_text(json.dumps(lock, indent=2) + "\n")
        self.namespace_lock_sha = hashlib.sha256(self.namespace_lock.read_bytes()).hexdigest()
        mapping_digest = hashlib.sha256()
        per_prefix_mapping_sha = {}
        for prefix in (8, 12, 16, 24):
            mapping = {
                "scene": "scene-a",
                "window_id": 0,
                "prefix": prefix,
                "input": list(range(prefix)),
                "anchor": prefix - 1,
                "future": {str(horizon): prefix - 1 + horizon for horizon in (1, 2, 4)},
            }
            encoded = (json.dumps(mapping, sort_keys=True, separators=(",", ":")) + "\n").encode()
            mapping_digest.update(encoded)
            per_prefix_mapping_sha[str(prefix)] = hashlib.sha256(encoded).hexdigest()
        self.label_audit = self.base / "label_audit.json"
        receipt = {
            "audit_schema_version": 2,
            "audit_type": "canonical_v2_compact_source_label_lineage",
            "status": "pass",
            "fail_closed": True,
            "problem_count": 0,
            "protocol": "inscene_streaming_canonical_v2_obs28_h124",
            "protocol_file": str(self.protocol),
            "protocol_file_sha256": self.protocol_file_sha,
            "source_manifest": str(self.canonical_source),
            "source_manifest_sha256": self.canonical_source_sha,
            "source_sidecar_sha256": self.source_sidecar_sha,
            "source_sidecar_declared_samples": 1,
            "namespace_binding": {
                "path": str(self.namespace_lock),
                "sha256": self.namespace_lock_sha,
                "feature_roots": lock["feature_roots"],
            },
            "feature_roots": {
                "streaming": "aoss://baiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_TEST",
                "target": "aoss://baiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_TEST",
            },
            "derivation": {
                "canonical_obs_len": 28,
                "prefixes": [8, 12, 16, 24],
                "input_rule": "streaming_window_indices[:prefix]",
                "anchor_rule": "streaming_window_indices[prefix-1]",
                "future_rule": "streaming_window_indices[prefix-1+horizon]",
                "horizons": [1, 2, 4],
                "row_level_legacy_feature_roots_ignored": True,
            },
            "counts": {
                "canonical_windows": 1,
                "derived_prefix_rows": 4,
                "input_frame_references": 60,
                "anchor_references": 4,
                "future_target_references": 12,
                "raw_scenes_checked": 1,
                "raw_frame_identities_checked": 28,
                "image_paths_checked": 28,
                "mask_paths_checked": 28,
                "depth_paths_checked": 28,
                "pose_rows_checked": 28,
            },
            "scene_count": 1,
            "canonical_universe_sha256": universe_sha,
            "label_lineage_universe_sha256": label_lineage_universe_sha,
            "input_anchor_future_mapping_sha256": mapping_digest.hexdigest(),
            "per_prefix_mapping_sha256": per_prefix_mapping_sha,
        }
        self.label_audit.write_text(json.dumps(receipt, indent=2) + "\n")

    def tearDown(self):
        self.temp.cleanup()

    def write_specs(self, actions=("REEXTRACT_MISSING", "REEXTRACT_MISMATCH")) -> Path:
        path = self.base / "specs.csv"
        fields = [
            "spec_id",
            "canonical_protocol",
            "protocol_sha256",
            "protocol_file_sha256",
            "vfm",
            "model",
            "model_variant",
            "model_id",
            "feature_layer",
            "timestep",
            "prefix",
            "requires_streaming",
            "requires_target",
            "action",
        ]
        rows = [
            {
                "spec_id": "wan-p8-l15",
                "canonical_protocol": "inscene_streaming_canonical_v2_obs28_h124",
                "protocol_sha256": self.feature_protocol_sha,
                "protocol_file_sha256": self.protocol_file_sha,
                "vfm": "wan",
                "model": "wan",
                "model_variant": "multi_layer",
                "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "feature_layer": 15,
                "timestep": 749,
                "prefix": 8,
                "requires_streaming": "true",
                "requires_target": "true",
                "action": actions[0],
            },
            {
                "spec_id": "wan-p12-l20",
                "canonical_protocol": "inscene_streaming_canonical_v2_obs28_h124",
                "protocol_sha256": self.feature_protocol_sha,
                "protocol_file_sha256": self.protocol_file_sha,
                "vfm": "wan",
                "model": "wan",
                "model_variant": "multi_layer",
                "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "feature_layer": 20,
                "timestep": 749,
                "prefix": 12,
                "requires_streaming": "true",
                "requires_target": "false",
                "action": actions[1],
            },
            {
                "spec_id": "videomae-reuse",
                "canonical_protocol": "inscene_streaming_canonical_v2_obs28_h124",
                "protocol_sha256": self.feature_protocol_sha,
                "protocol_file_sha256": self.protocol_file_sha,
                "vfm": "videomae",
                "model": "videomae",
                "model_variant": "default",
                "model_id": "MCG-NJU/videomae-large",
                "feature_layer": 23,
                "timestep": 0,
                "prefix": 24,
                "requires_streaming": "true",
                "requires_target": "false",
                "action": "REEXTRACT_MISSING",
            },
            {
                "spec_id": "qwen-blocked",
                "canonical_protocol": "inscene_streaming_canonical_v2_obs28_h124",
                "protocol_sha256": self.feature_protocol_sha,
                "protocol_file_sha256": self.protocol_file_sha,
                "vfm": "qwen2_5_vl",
                "model": "qwen2_5_vl",
                "model_variant": "default",
                "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                "feature_layer": -1,
                "timestep": 749,
                "prefix": 8,
                "requires_streaming": "true",
                "requires_target": "true",
                "action": "REEXTRACT_MISSING",
            },
            {
                "spec_id": "aether-blocked",
                "canonical_protocol": "inscene_streaming_canonical_v2_obs28_h124",
                "protocol_sha256": self.feature_protocol_sha,
                "protocol_file_sha256": self.protocol_file_sha,
                "vfm": "aether",
                "model": "aether",
                "model_variant": "default",
                "model_id": "AetherWorldModel/AetherV1",
                "feature_layer": 1,
                "timestep": 749,
                "prefix": 8,
                "requires_streaming": "true",
                "requires_target": "true",
                "action": "REEXTRACT_MISSING",
            },
        ]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def command(self, specs: Path, output: Path, *extra: str) -> list[str]:
        return [
            sys.executable,
            str(PREPARE),
            "--specs",
            str(specs),
            "--protocol",
            str(self.protocol),
            "--canonical-source",
            str(self.canonical_source),
            "--namespace-lock",
            str(self.namespace_lock),
            "--label-audit",
            str(self.label_audit),
            "--output-dir",
            str(output),
            "--dataset-root",
            str(self.base / "dataset"),
            "--project-root",
            str(ROOT),
            "--scene-count",
            "16",
            "--n-gpus",
            "4",
            *extra,
        ]

    def test_render_groups_all_prefixes_and_blocks_uncertified_backends(self):
        output = self.base / "plan"
        result = subprocess.run(self.command(self.write_specs(), output), text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["status"], "BLOCKED")
        self.assertEqual(summary["counts"]["ready_shards"], 3)
        self.assertEqual(summary["counts"]["streaming_shards"], 2)
        self.assertEqual(summary["counts"]["target_shards"], 1)
        self.assertEqual(summary["counts"]["ready_sco_batches"], 2)
        self.assertEqual(summary["counts"]["blocked_groups"], 2)
        blocked = {item["vfm"]: item for item in summary["blocked"]}
        self.assertIn("BLOCKED_GPU_CANARY_QWEN", blocked["qwen2_5_vl"]["reasons"][0])
        self.assertIn("BLOCKED_GPU_CANARY_AETHER", blocked["aether"]["reasons"][0])
        self.assertNotIn("videomae", blocked)

        shards = [json.loads(path.read_text()) for path in sorted((output / "shards").glob("*.json"))]
        for shard in shards:
            self.assertEqual(shard["prefixes"], [8, 12, 16, 24])
            self.assertEqual(shard["canonical_obs_len"], 28)
            self.assertEqual(Path(shard["feature_root"]).name, shard["feature_namespace"])
            self.assertEqual(shard["protocol_file_sha256"], self.protocol_file_sha)
            self.assertNotIn("FEAT_STREAMING", Path(shard["feature_root"]).parts)
        stream_layer_set = next(
            item["layers"]
            for item in shards
            if item["mode"] == "streaming_prefix" and item["vfm"] == "wan"
        )
        target_layer_set = next(
            item["layers"]
            for item in shards
            if item["mode"] == "target_isolated" and item["vfm"] == "wan"
        )
        self.assertEqual(stream_layer_set, [15, 20])
        self.assertEqual(target_layer_set, [15])
        commands = (output / "sco_create_commands.txt").read_text()
        self.assertNotIn("qwen", commands)
        self.assertNotIn("aether", commands)
        self.assertIn("videomae", commands)
        self.assertEqual(summary["submission_performed"], False)

        stream_shard = next(
            item for item in shards
            if item["mode"] == "streaming_prefix" and item["vfm"] == "wan"
        )
        stream_path = output / "shards" / f"{stream_shard['shard_id']}.json"
        dry = subprocess.run(
            [sys.executable, str(RUNNER), "--shard", str(stream_path)],
            text=True,
            capture_output=True,
        )
        self.assertEqual(dry.returncode, 0, dry.stderr)
        preview = json.loads(dry.stdout)
        self.assertTrue(preview["dry_run"])
        self.assertEqual(len(preview["commands"]), 4)
        for command in preview["commands"]:
            self.assertIn("--streaming-window-obs-len", command["argv"])
            self.assertIn("8,12,16,24", command["argv"])

        batch_path = next(
            path
            for path in (output / "batches").glob("*.json")
            if json.loads(path.read_text())["vfm"] == "wan"
        )
        batch_dry = subprocess.run(
            [sys.executable, str(RUNNER), "--batch", str(batch_path)],
            text=True,
            capture_output=True,
        )
        self.assertEqual(batch_dry.returncode, 0, batch_dry.stderr)
        batch_preview = json.loads(batch_dry.stdout)
        self.assertEqual(batch_preview["serialization"], "sequential")
        self.assertEqual(len(batch_preview["shards"]), 2)

    def test_full_pass_canary_receipt_unblocks_only_matching_backend(self):
        canary_root = self.base / "backend_canaries"
        receipt_dir = canary_root / "receipts"
        receipt_dir.mkdir(parents=True)
        extractor = ROOT / "features/run_inscene15k.py"
        canary_script = ROOT / "scripts/windowfix_v2/run_backend_determinism_canary.py"
        comparison = {
            "scene": "scene-a",
            "files_compared": 3,
            "safetensors_compared": 1,
            "comparison_sha256": "a" * 64,
        }
        receipt = {
            "schema_version": 1,
            "audit_type": "canonical_v2_backend_cross_process_order_canary",
            "status": "pass",
            "vfm": "aether",
            "model_id": "AetherWorldModel/AetherV1",
            "layer": 1,
            "timestep": 749,
            "protocol": "inscene_streaming_canonical_v2_obs28_h124",
            "protocol_file_sha256": self.protocol_file_sha,
            "extractor": str(extractor),
            "extractor_sha256": hashlib.sha256(extractor.read_bytes()).hexdigest(),
            "canary_script": str(canary_script),
            "canary_script_sha256": hashlib.sha256(canary_script.read_bytes()).hexdigest(),
            "results": {
                "streaming_prefix": {"comparison": comparison},
                "target_isolated": {"comparison": comparison},
            },
        }
        (receipt_dir / "aether.json").write_text(json.dumps(receipt, indent=2) + "\n")

        output = self.base / "canary-plan"
        result = subprocess.run(
            self.command(
                self.write_specs(),
                output,
                "--backend-canary-dir",
                str(canary_root),
            ),
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["counts"]["ready_shards"], 5)
        self.assertEqual(summary["counts"]["blocked_groups"], 1)
        self.assertEqual(summary["blocked"][0]["vfm"], "qwen2_5_vl")
        self.assertIn("aether", summary["backend_canary_bindings"])

        aether_shard = next(
            path
            for path in (output / "shards").glob("*.json")
            if json.loads(path.read_text())["vfm"] == "aether"
        )
        dry = subprocess.run(
            [sys.executable, str(RUNNER), "--shard", str(aether_shard)],
            text=True,
            capture_output=True,
        )
        self.assertEqual(dry.returncode, 0, dry.stderr)

    def test_pending_is_fail_closed_unless_explicitly_allowed(self):
        specs = self.write_specs(actions=("PENDING_CACHE_AUDIT", "PENDING_CACHE_AUDIT"))
        refused = subprocess.run(
            self.command(specs, self.base / "refused"), text=True, capture_output=True
        )
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("await cache audit", refused.stderr)
        allowed = subprocess.run(
            self.command(specs, self.base / "allowed", "--allow-pending"),
            text=True,
            capture_output=True,
        )
        self.assertEqual(allowed.returncode, 0, allowed.stderr)

    def test_protocol_file_hash_mismatch_is_rejected(self):
        specs = self.write_specs()
        self.protocol.write_text(self.protocol.read_text() + "\n")
        result = subprocess.run(
            self.command(specs, self.base / "stale"), text=True, capture_output=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("protocol file SHA mismatch", result.stderr)

    def test_existing_output_is_immutable(self):
        specs = self.write_specs()
        output = self.base / "existing"
        output.mkdir()
        result = subprocess.run(self.command(specs, output), text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to replace immutable output directory", result.stderr)

    def test_compact_manifest_derives_prefix_from_one_obs28_source(self):
        module = load_compact_audit_module()
        derived = self.base / "derived-p8.jsonl"
        rows, scenes, digest = module.derive_manifest(
            source=self.canonical_source,
            output=derived,
            prefix=8,
            streaming_root="aoss://baiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_SEEDED_V1",
            target_root="aoss://baiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_SEEDED_V1",
        )
        self.assertEqual((rows, scenes), (1, 1))
        row = json.loads(derived.read_text())
        self.assertEqual(row["streaming_indices"], list(range(8)))
        self.assertEqual(row["streaming_window_indices"], list(range(28)))
        self.assertEqual(row["streaming_tail_raw"], 7)
        meta = json.loads(Path(f"{derived}.meta.json").read_text())
        self.assertEqual(meta["sha256"], digest)
        self.assertEqual(meta["prefix_length"], 8)


if __name__ == "__main__":
    unittest.main()
