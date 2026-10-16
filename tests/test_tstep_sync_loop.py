from __future__ import annotations

import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

pytestmark = pytest.mark.integration


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNC_SCRIPT = PROJECT_ROOT / "scripts" / "run_tstep_sync_loop.sh"
PROTOCOL = "inscene_streaming_canonical_v2_obs28_h124"
PROTOCOL_SHA = "a" * 64
SEEDED_PROTOCOL_SHA = "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f"
STREAM_SEEDED_NAMESPACE = "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
TARGET_SEEDED_NAMESPACE = "FEAT_TARGET_CANONICAL_V2_SEEDED_V1"
GENERATION_CONTRACT = "inscene_deterministic_noise_v1"
NOISE_SEED_SCHEME = "inscene_noise_seed_v1"
MINIMAX_SEED_SCHEME = "minimax_h3_fixed_seed_v1"
MODEL_ID = "/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b"
MINIMAX_MODEL_ID = "/mnt/umm/users/yinbaiqiao/Consistency/minimax-h3/models/MiniMax-H3"
LAYERS = (15, 20, 25)

VFM_INPUT_CONTRACT = {
    "wan": {"prompt": ".", "resize": [480, 832], "num_frames": 81},
    "cogvideox": {"prompt": "", "resize": [480, 720], "num_frames": 97},
    "minimax_h3": {"prompt": ".", "resize": [480, 832], "num_frames": 73},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def noise_sample(identity: dict) -> dict:
    payload = {"noise_seed_scheme": NOISE_SEED_SCHEME, "identity": identity}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)
    return {"noise_seed": seed, "noise_seed_identity": identity}


def seeded_generation(namespace: str, vfm: str) -> dict:
    common = {
        "feature_namespace": namespace,
        "feature_generation_contract": GENERATION_CONTRACT,
    }
    if vfm in {"wan", "cogvideox"}:
        return {**common, "noise_seed_scheme": NOISE_SEED_SCHEME}
    if vfm == "minimax_h3":
        return {
            **common,
            "noise_seed_scheme": MINIMAX_SEED_SCHEME,
            "noise_seed": 0,
            "noise_seed_scope": "reset_per_forward",
        }
    raise ValueError(vfm)


class TimestepSyncLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        # Use the platform temporary directory by default. To exercise a
        # production filesystem's hardlink/mtime behavior, explicitly set
        # PROBE_TEST_TMPDIR to a writable directory on that filesystem.
        self.temporary = tempfile.TemporaryDirectory(
            prefix="test-tstep-sync-", dir=os.environ.get("PROBE_TEST_TMPDIR")
        )
        self.root = Path(self.temporary.name)
        self.feature_root = self.root / "features"
        self.log_dir = self.root / "logs"
        self.done_index = self.root / "done" / "index.txt"
        self.stop_file = self.log_dir / "extract.finished"
        self.conf = self.root / "petreloss.conf"
        self.calls = self.root / "ads.calls"
        self.staged_listing = self.root / "ads.staged-files"
        self.conf.write_text(
            "[fj]\naccess_key = fake-ak\nsecret_key = fake-sk\n",
            encoding="utf-8",
        )
        self.log_dir.mkdir(parents=True)
        self.stop_file.touch()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _feature_sidecar(
        self,
        feature: Path,
        *,
        mode: str,
        timestep: int = 499,
        vfm: str = "wan",
        model_id: str = MODEL_ID,
        protocol_sha: str = PROTOCOL_SHA,
        extra: dict | None = None,
    ) -> Path:
        sidecar = Path(f"{feature}.manifest.json")
        # The shape is the shape written by make_streaming_unit/make_target_unit.
        from safetensors import safe_open

        with safe_open(str(feature), framework="np", device="cpu") as handle:
            shape = list(handle.get_slice("feat").get_shape())
        input_contract = VFM_INPUT_CONTRACT[vfm]
        metadata = {
            "schema_version": 1,
            "frame_index_schema": "image_mask_intersection_v1",
            "vfm": vfm,
            "model_id": model_id,
            "mode": mode,
            "timestep": timestep,
            "prompt": input_contract["prompt"],
            "resize": input_contract["resize"],
            "num_frames": input_contract["num_frames"],
            "file": feature.name,
            "size_bytes": feature.stat().st_size,
            "sha256": sha256(feature),
            "tensors": {"feat": {"shape": shape, "dtype": "torch.float16"}},
            "canonical_protocol": PROTOCOL,
            "canonical_protocol_sha256": protocol_sha,
            "canonical_obs_len": 28,
        }
        if extra:
            metadata.update(extra)
        write_json(sidecar, metadata)
        return sidecar

    def _write_npy_with_sidecar(
        self,
        path: Path,
        array: np.ndarray,
        *,
        protocol_sha: str = PROTOCOL_SHA,
        extra: dict | None = None,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.save(handle, array)
        sidecar = Path(f"{path}.manifest.json")
        metadata = {
            "schema_version": 1,
            "file": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "canonical_protocol": PROTOCOL,
            "canonical_protocol_sha256": protocol_sha,
            "canonical_obs_len": 28,
        }
        if extra:
            metadata.update(extra)
        write_json(sidecar, metadata)
        return sidecar

    def make_streaming_unit(
        self,
        *,
        scene: str = "scene",
        window: int = 0,
        tail: int = 7,
        layers: tuple[int, ...] = LAYERS,
        structure: bool = True,
        seeded: bool = False,
        vfm: str = "wan",
        timestep: int = 499,
        model_id: str = MODEL_ID,
    ) -> dict[int, tuple[Path, Path]]:
        unit = self.feature_root / scene / f"window_{window:04d}" / f"prefix_{tail:06d}"
        namespace = STREAM_SEEDED_NAMESPACE
        protocol_sha = SEEDED_PROTOCOL_SHA if seeded else PROTOCOL_SHA
        feature_extra = seeded_generation(namespace, vfm) if seeded else {}
        if seeded and vfm in {"wan", "cogvideox"}:
            sample = noise_sample(
                {
                    "canonical_protocol": PROTOCOL,
                    "canonical_protocol_sha256": SEEDED_PROTOCOL_SHA,
                    "vfm": vfm,
                    "model_id": model_id,
                    "scene_id": scene,
                    "sample_kind": "streaming_prefix",
                    "window_id": window,
                    "prefix_len": tail + 1,
                }
            )
            feature_extra = {
                **feature_extra,
                "noise_seed_samples": [sample],
                **sample,
            }
        pairs = {}
        for layer in layers:
            feature = unit / f"feature_t{timestep}_layer{layer}.sft"
            feature.parent.mkdir(parents=True, exist_ok=True)
            save_file({"feat": np.zeros((1, 2, 2, 2), dtype=np.float16)}, feature)
            pairs[layer] = (
                feature,
                self._feature_sidecar(
                    feature,
                    mode="streaming_prefix",
                    timestep=timestep,
                    vfm=vfm,
                    model_id=model_id,
                    protocol_sha=protocol_sha,
                    extra=feature_extra,
                ),
            )
        if structure:
            window_indices = list(range(100, 128))
            record = {
                "window_id": window,
                "tail": tail,
                "indices": window_indices[: tail + 1],
                "window_indices": window_indices,
                "valid_length": tail + 1,
                "canonical_protocol": PROTOCOL,
                "canonical_protocol_sha256": protocol_sha,
                "canonical_obs_len": 28,
            }
            self._write_npy_with_sidecar(
                self.feature_root / scene / "prefix_index.npy",
                np.array([record], dtype=object),
                protocol_sha=protocol_sha,
                extra=seeded_generation(namespace, vfm) if seeded else None,
            )
        return pairs

    def make_target_unit(
        self,
        *,
        scene: str = "scene",
        seeded: bool = False,
        vfm: str = "wan",
        timestep: int = 499,
        model_id: str = MODEL_ID,
    ) -> dict[int, tuple[Path, Path]]:
        base = self.feature_root / scene
        namespace = TARGET_SEEDED_NAMESPACE
        protocol_sha = SEEDED_PROTOCOL_SHA if seeded else PROTOCOL_SHA
        window = list(range(100, 128))
        provenance = []
        targets = set()
        for prefix in (8, 12, 16, 24):
            positions = [("anchor", 0, prefix - 1)] + [
                ("future", horizon, prefix - 1 + horizon)
                for horizon in (1, 2, 4)
            ]
            for role, horizon, position in positions:
                frame = window[position]
                targets.add(frame)
                provenance.append(
                    {
                        "window_id": 0,
                        "prefix_length": prefix,
                        "role": role,
                        "horizon": horizon,
                        "position": position,
                        "frame_index": frame,
                        "canonical_protocol": PROTOCOL,
                        "canonical_obs_len": 28,
                    }
                )
        target_array = np.array(sorted(targets), dtype=np.int64)
        target_path = base / "target_indices.npy"
        generation = seeded_generation(namespace, vfm) if seeded else {}
        self._write_npy_with_sidecar(
            target_path,
            target_array,
            protocol_sha=protocol_sha,
            extra=generation,
        )
        target_meta = {
            "schema_version": 2,
            "protocol": PROTOCOL,
            "protocol_sha256": protocol_sha,
            "obs_len": 28,
            "prefixes": [8, 12, 16, 24],
            "horizons": [1, 2, 4],
            "include_anchors": True,
            "target_count": len(target_array),
            "target_indices_sha256": sha256(target_path),
            "provenance": provenance,
            **generation,
        }
        write_json(base / "target_indices.meta.json", target_meta)
        feature_extra = dict(generation)
        if seeded and vfm in {"wan", "cogvideox"}:
            feature_extra["noise_seed_samples"] = [
                noise_sample(
                    {
                        "canonical_protocol": PROTOCOL,
                        "canonical_protocol_sha256": SEEDED_PROTOCOL_SHA,
                        "vfm": vfm,
                        "model_id": model_id,
                        "scene_id": scene,
                        "sample_kind": "target_isolated",
                        "frame_index": int(frame_index),
                    }
                )
                for frame_index in target_array.tolist()
            ]
        pairs = {}
        for layer in LAYERS:
            feature = base / f"feature_t{timestep}_layer{layer}.sft"
            feature.parent.mkdir(parents=True, exist_ok=True)
            save_file(
                {"feat": np.zeros((len(target_array), 2, 2, 2), dtype=np.float16)},
                feature,
            )
            pairs[layer] = (
                feature,
                self._feature_sidecar(
                    feature,
                    mode="target_isolated",
                    timestep=timestep,
                    vfm=vfm,
                    model_id=model_id,
                    protocol_sha=protocol_sha,
                    extra=feature_extra,
                ),
            )
        return pairs

    def make_old_pair(self) -> tuple[Path, Path]:
        feature = (
            self.feature_root
            / "scene/window_0000/prefix_000007/feature_t749_layer15.sft"
        )
        feature.parent.mkdir(parents=True, exist_ok=True)
        feature.write_bytes(b"legacy feature")
        sidecar = Path(f"{feature}.manifest.json")
        sidecar.write_text("{}\n", encoding="utf-8")
        return feature, sidecar

    def fake_ads_cli(
        self,
        *,
        failed: int = 0,
        exit_code: int = 0,
        found_delta: int = 0,
        before_summary: str = "",
    ) -> Path:
        path = self.root / f"ads-cli-{len(list(self.root.glob('ads-cli-*')))}"
        script = f"""#!/usr/bin/env bash
set -u
args=(\"$@\")
src=${{args[$((${{#args[@]}} - 2))]}}
printf '%s\\n' \"${{src}}\" >> {shlex.quote(str(self.calls))}
find \"${{src}}\" -type f -printf '%P\\n' | LC_ALL=C sort > {shlex.quote(str(self.staged_listing))}
count=$(wc -l < {shlex.quote(str(self.staged_listing))})
{before_summary}
found=$((count + {found_delta}))
printf 'Found: %s, copied: %s, failed: {failed}\\000\\r\\n' \"${{found}}\" \"${{found}}\"
exit {exit_code}
"""
        path.write_text(script, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def mutating_ads_cli(self, feature: Path) -> Path:
        helper = self.root / "mutate_feature.py"
        helper.write_text(
            """import hashlib, json, os, sys
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file
feature = Path(sys.argv[1])
sidecar = Path(f"{feature}.manifest.json")
tmp = feature.with_name(feature.name + ".replacement")
save_file({"feat": np.ones((2, 2, 2, 2), dtype=np.float16)}, tmp)
os.replace(tmp, feature)
meta = json.loads(sidecar.read_text())
meta["size_bytes"] = feature.stat().st_size
meta["sha256"] = hashlib.sha256(feature.read_bytes()).hexdigest()
meta["tensors"]["feat"]["shape"] = [2, 2, 2, 2]
side_tmp = sidecar.with_name(sidecar.name + ".replacement")
side_tmp.write_text(json.dumps(meta) + "\\n")
os.replace(side_tmp, sidecar)
""",
            encoding="utf-8",
        )
        flag = self.root / "mutated-once"
        before = (
            f"if [[ ! -e {shlex.quote(str(flag))} ]]; then "
            f"{shlex.quote(sys.executable)} {shlex.quote(str(helper))} "
            f"{shlex.quote(str(feature))}; touch {shlex.quote(str(flag))}; fi"
        )
        return self.fake_ads_cli(before_summary=before)

    def temp_release_sleep(self, temp: Path, *, after_calls: int) -> tuple[Path, Path]:
        """Fake sleep that releases an active writer and then declares STOP."""
        fake_bin = self.root / "temp-release-bin"
        counter = self.root / "sleep.calls"
        sleep = fake_bin / "sleep"
        sleep.parent.mkdir(parents=True)
        sleep.write_text(
            f"""#!/usr/bin/env bash
set -u
count=0
if [[ -f {shlex.quote(str(counter))} ]]; then read -r count < {shlex.quote(str(counter))}; fi
count=$((count + 1))
printf '%s\n' "${{count}}" > {shlex.quote(str(counter))}
if [[ "${{count}}" -eq {after_calls} ]]; then
  rm -f -- {shlex.quote(str(temp))}
  touch -- {shlex.quote(str(self.stop_file))}
fi
""",
            encoding="utf-8",
        )
        sleep.chmod(0o755)
        return fake_bin, counter

    def hardlink_racing_python(self, feature: Path) -> Path:
        """Replace a source inode exactly once between snapshot and hardlink."""
        wrapper = self.root / "hardlink-racing-python"
        replacement = feature.with_name(f".{feature.name}.replacement")
        flag = self.root / "hardlink-race-injected"
        wrapper.write_text(
            f"""#!/usr/bin/env bash
set -u
if [[ "$#" -eq 5 && "$1" == "-" && "$3" == *round_*.snapshot.jsonl && ! -e {shlex.quote(str(flag))} ]]; then
  cp -- {shlex.quote(str(feature))} {shlex.quote(str(replacement))}
  mv -- {shlex.quote(str(replacement))} {shlex.quote(str(feature))}
  touch -- {shlex.quote(str(flag))}
fi
exec {shlex.quote(sys.executable)} "$@"
""",
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
        return wrapper

    def run_loop(
        self,
        ads_cli: Path,
        *,
        kind: str = "FEAT_STREAMING",
        namespace: str | None = None,
        vfm: str = "wan",
        timestep: int = 499,
        model_id: str = MODEL_ID,
        extra_env: dict[str, str] | None = None,
        background: bool = False,
    ) -> subprocess.CompletedProcess[str] | subprocess.Popen[str]:
        namespace = namespace or f"{kind}_CANONICAL_V2"
        env = os.environ.copy()
        env.update(
            {
                "FEATURE_KIND": kind,
                "FEATURE_NAMESPACE": namespace,
                "FEATURE_ROOT": str(self.feature_root),
                "VFM": vfm,
                "T": str(timestep),
                "LAYERS": "15 20 25",
                "EXPECTED_MODEL_ID": model_id,
                "EXTRACT_DONE_INDEX": str(self.done_index),
                "LOG_DIR": str(self.log_dir),
                "STOP_FILE": str(self.stop_file),
                "ADS_CLI": str(ads_cli),
                "CONF": str(self.conf),
                "SYNC_PYTHON": sys.executable,
                "SYNC_RETRY_SEC": "0",
                "SYNC_FINAL_MAX_ATTEMPTS": "1",
                "SYNC_POLL_SEC": "0",
                # Historical contract tests must keep the open/hash validator.
                "SYNC_TRUST_EXTRACT": "0",
            }
        )
        if extra_env:
            env.update(extra_env)
        if background:
            return subprocess.Popen(
                ["bash", str(SYNC_SCRIPT)],
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        return subprocess.run(
            ["bash", str(SYNC_SCRIPT)],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # Shared UMM load can make repeated cold NumPy/safetensors helper
            # startups slow; STOP_FILE retries themselves are bounded at one.
            timeout=300,
            check=False,
        )

    def assert_pairs_exist(self, pairs: dict[int, tuple[Path, Path]], expected: bool) -> None:
        for feature, sidecar in pairs.values():
            self.assertEqual(feature.exists(), expected)
            self.assertEqual(sidecar.exists(), expected)

    def test_success_uses_exact_staging_then_commits_index_before_deleting_triplet(self) -> None:
        pairs = self.make_streaming_unit()
        old_feature, old_sidecar = self.make_old_pair()

        result = self.run_loop(self.fake_ads_cli())

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assert_pairs_exist(pairs, False)
        self.assertTrue(old_feature.exists())
        self.assertTrue(old_sidecar.exists())
        self.assertEqual(
            self.done_index.read_text().splitlines(),
            [
                f"scene/window_0000/prefix_000007/feature_t499_layer{layer}.sft"
                for layer in LAYERS
            ],
        )
        staged = set(self.staged_listing.read_text().splitlines())
        ads_source = Path(self.calls.read_text().splitlines()[-1].rstrip("/"))
        self.assertEqual(ads_source.parent, self.feature_root)
        self.assertTrue(ads_source.name.startswith(".tstep_sync_staging_"))
        self.assertNotEqual(ads_source, self.feature_root)
        self.assertIn("scene/prefix_index.npy", staged)
        self.assertIn("scene/prefix_index.npy.manifest.json", staged)
        self.assertEqual(len(staged), 8)  # three feature pairs + structural pair
        self.assertFalse(list(self.feature_root.glob(".tstep_sync_staging_*")))

    def test_scan_root_isolates_one_scene_but_preserves_global_relative_keys(self) -> None:
        selected = self.make_streaming_unit(scene="selected")
        other = self.make_streaming_unit(scene="other")
        other_temp = other[15][0].with_name(f".{other[15][0].name}.tmp-writer")
        other_temp.write_bytes(b"unrelated scene writer")

        result = self.run_loop(
            self.fake_ads_cli(),
            extra_env={"FEATURE_SCAN_ROOT": str(self.feature_root / "selected")},
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(
            f"feature_scan_root={self.feature_root / 'selected'}", result.stdout
        )
        self.assert_pairs_exist(selected, False)
        self.assert_pairs_exist(other, True)
        self.assertTrue(other_temp.exists())
        self.assertEqual(
            self.done_index.read_text().splitlines(),
            [
                f"selected/window_0000/prefix_000007/feature_t499_layer{layer}.sft"
                for layer in LAYERS
            ],
        )
        staged = set(self.staged_listing.read_text().splitlines())
        self.assertEqual(
            staged,
            {
                "selected/prefix_index.npy",
                "selected/prefix_index.npy.manifest.json",
                *{
                    f"selected/window_0000/prefix_000007/feature_t499_layer{layer}.sft"
                    for layer in LAYERS
                },
                *{
                    f"selected/window_0000/prefix_000007/feature_t499_layer{layer}.sft.manifest.json"
                    for layer in LAYERS
                },
            },
        )

    def test_final_atomic_temp_blocks_ads_for_exact_final_budget(self) -> None:
        pairs = self.make_streaming_unit()
        temp = pairs[15][0].with_name(f".{pairs[15][0].name}.tmp-123")
        temp.write_bytes(b"writer in progress")

        result = self.run_loop(
            self.fake_ads_cli(),
            extra_env={"SYNC_FINAL_MAX_ATTEMPTS": "4"},
        )

        self.assertNotEqual(result.returncode, 0, result.stdout)
        marker = "before snapshot final atomic temp files remain; safely deferring"
        self.assertIn(marker, result.stdout)
        self.assertEqual(
            result.stdout.count(marker),
            4,
            result.stdout,
        )
        self.assertIn("final sync failed 4 times; giving up", result.stdout)
        self.assertFalse(self.calls.exists(), "ads-cli must not run while writer temp exists")
        self.assert_pairs_exist(pairs, True)
        self.assertEqual(self.done_index.read_text(), "")

    def test_active_only_atomic_temp_waits_beyond_failure_budget(self) -> None:
        temp = (
            self.feature_root
            / "writer/window_0000/prefix_000007/.feature_t499_layer15.sft.tmp-writer"
        )
        temp.parent.mkdir(parents=True)
        temp.write_bytes(b"writer in progress")
        self.stop_file.unlink()
        fake_bin, sleep_counter = self.temp_release_sleep(temp, after_calls=5)

        result = self.run_loop(
            self.fake_ads_cli(),
            extra_env={
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "SYNC_EVERY_SEC": "0",
                "SYNC_MAX_FAILURES": "2",
                "SYNC_FINAL_MAX_ATTEMPTS": "2",
            },
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(sleep_counter.read_text().strip(), "5")
        self.assertNotIn("sync failed 2 consecutive times", result.stdout)
        self.assertFalse(self.calls.exists(), "a temp alone is never an ads candidate")
        self.assertFalse(temp.exists())
        self.assertEqual(self.done_index.read_text(), "")

    def test_active_other_scene_temp_does_not_starve_stable_snapshot(self) -> None:
        pairs = self.make_streaming_unit(scene="stable")
        temp = (
            self.feature_root
            / "writer/window_0000/prefix_000007/.feature_t499_layer15.sft.tmp-writer"
        )
        temp.parent.mkdir(parents=True)
        temp.write_bytes(b"writer in progress")
        self.stop_file.unlink()
        fake_bin, sleep_counter = self.temp_release_sleep(temp, after_calls=5)

        result = self.run_loop(
            self.fake_ads_cli(),
            extra_env={
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "SYNC_EVERY_SEC": "0",
                "SYNC_MAX_FAILURES": "2",
                "SYNC_FINAL_MAX_ATTEMPTS": "2",
            },
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(sleep_counter.read_text().strip(), "5")
        self.assertEqual(len(self.calls.read_text().splitlines()), 1)
        self.assertNotIn("final atomic temp files remain", result.stdout)
        self.assert_pairs_exist(pairs, False)
        self.assertFalse(temp.exists())
        self.assertEqual(
            self.done_index.read_text().splitlines(),
            [
                f"stable/window_0000/prefix_000007/feature_t499_layer{layer}.sft"
                for layer in LAYERS
            ],
        )

    def test_hardlink_source_race_safely_defers_without_ads_or_delete(self) -> None:
        pairs = self.make_streaming_unit()
        racing_python = self.hardlink_racing_python(pairs[15][0])

        result = self.run_loop(
            self.fake_ads_cli(),
            extra_env={
                "SYNC_PYTHON": str(racing_python),
                "SYNC_FINAL_MAX_ATTEMPTS": "2",
            },
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(
            "source changed during hardlink staging; safely deferring",
            result.stdout,
        )
        self.assertIn("round=0002", result.stdout)
        self.assertEqual(len(self.calls.read_text().splitlines()), 1)
        self.assert_pairs_exist(pairs, False)
        self.assertEqual(len(self.done_index.read_text().splitlines()), 3)

    def test_real_sidecar_contract_mismatch_never_stages_or_deletes(self) -> None:
        pairs = self.make_streaming_unit()
        sidecar = pairs[20][1]
        meta = json.loads(sidecar.read_text())
        meta["model_id"] = "wrong/model"
        write_json(sidecar, meta)

        result = self.run_loop(self.fake_ads_cli())

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("feature sidecar model_id", result.stdout)
        self.assertFalse(self.calls.exists())
        self.assert_pairs_exist(pairs, True)
        self.assertEqual(self.done_index.read_text(), "")

    def test_failed_zero_with_insufficient_found_coverage_retains_everything(self) -> None:
        pairs = self.make_streaming_unit()

        result = self.run_loop(self.fake_ads_cli(found_delta=-1))

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("upload coverage not proven", result.stdout)
        self.assertIn("found=7 expected=8 failed=0", result.stdout)
        self.assert_pairs_exist(pairs, True)
        self.assertEqual(self.done_index.read_text(), "")

    def test_done_index_merge_failure_happens_before_any_source_delete(self) -> None:
        pairs = self.make_streaming_unit()
        fake_bin = self.root / "fake-bin"
        observed = self.root / "all-pairs-existed-at-merge"
        paths = [path for pair in pairs.values() for path in pair]
        conditions = " && ".join(f"[[ -e {shlex.quote(str(path))} ]]" for path in paths)
        _mv = fake_bin / "mv"
        _mv.parent.mkdir(parents=True)
        _mv.write_text(
            f"#!/usr/bin/env bash\nif {conditions}; then touch {shlex.quote(str(observed))}; fi\nexit 42\n",
            encoding="utf-8",
        )
        _mv.chmod(0o755)

        result = self.run_loop(
            self.fake_ads_cli(),
            extra_env={"PATH": f"{fake_bin}:{os.environ['PATH']}"},
        )

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("done-index merge failed; retaining all source pairs", result.stdout)
        self.assertTrue(observed.exists(), "all sources must exist when atomic merge is attempted")
        self.assert_pairs_exist(pairs, True)
        self.assertEqual(self.done_index.read_text(), "")

    def test_missing_structural_index_blocks_entire_three_layer_unit(self) -> None:
        pairs = self.make_streaming_unit(structure=False)

        result = self.run_loop(self.fake_ads_cli())

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("missing/non-regular structural index", result.stdout)
        self.assertFalse(self.calls.exists())
        self.assert_pairs_exist(pairs, True)
        self.assertEqual(self.done_index.read_text(), "")

    def test_unit_awaiting_prefix_index_is_deferred_until_index_appears(self) -> None:
        # A brand-new namespace: the extractor writes a scene's features before
        # its prefix_index.npy.  While extraction runs that unit must wait, not
        # kill the loop (2026-10-16 canary job pt-fxzmatkf died exactly here).
        self.stop_file.unlink()
        pairs = self.make_streaming_unit(structure=False)
        loop = self.run_loop(self.fake_ads_cli(), background=True)
        try:
            deadline = time.monotonic() + 240
            pending_seen = False
            while time.monotonic() < deadline and loop.poll() is None:
                for path in self.log_dir.glob("round_*.pending.txt"):
                    if path.is_file() and "awaiting scene/prefix_index.npy" in path.read_text():
                        pending_seen = True
                if pending_seen:
                    break
                time.sleep(0.2)
            self.assertTrue(pending_seen, "loop never reported the unit as pending")
            self.assertIsNone(loop.poll(), "loop exited while the unit was only pending")
            self.assertFalse(self.calls.exists(), "nothing may be uploaded before the index exists")
            self.assert_pairs_exist(pairs, True)

            window_indices = list(range(100, 128))
            record = {
                "window_id": 0,
                "tail": 7,
                "indices": window_indices[:8],
                "window_indices": window_indices,
                "valid_length": 8,
                "canonical_protocol": PROTOCOL,
                "canonical_protocol_sha256": PROTOCOL_SHA,
                "canonical_obs_len": 28,
            }
            self._write_npy_with_sidecar(
                self.feature_root / "scene" / "prefix_index.npy",
                np.array([record], dtype=object),
                protocol_sha=PROTOCOL_SHA,
            )
            self.stop_file.touch()
            stdout, _ = loop.communicate(timeout=300)
        finally:
            if loop.poll() is None:
                loop.kill()
                loop.communicate()

        self.assertEqual(loop.returncode, 0, stdout)
        self.assert_pairs_exist(pairs, False)
        self.assertEqual(len(self.done_index.read_text().splitlines()), 3)
        staged = set(self.staged_listing.read_text().splitlines())
        self.assertIn("scene/prefix_index.npy", staged)

    def test_target_unit_without_structural_files_fails_once_extraction_stops(self) -> None:
        pairs = self.make_target_unit()
        for name in ("target_indices.npy", "target_indices.npy.manifest.json", "target_indices.meta.json"):
            (self.feature_root / "scene" / name).unlink()

        result = self.run_loop(self.fake_ads_cli(), kind="FEAT_TARGET")

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("never got their structural index", result.stdout)
        self.assertIn("scene: awaiting target_indices", result.stdout)
        self.assertFalse(self.calls.exists())
        self.assert_pairs_exist(pairs, True)
        self.assertEqual(self.done_index.read_text(), "")

    def test_prefix_index_must_contain_the_exact_unit_record(self) -> None:
        pairs = self.make_streaming_unit()
        prefix_index = self.feature_root / "scene/prefix_index.npy"
        record = np.load(prefix_index, allow_pickle=True).tolist()[0]
        record["tail"] = 11
        self._write_npy_with_sidecar(prefix_index, np.array([record], dtype=object))

        result = self.run_loop(self.fake_ads_cli())

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("prefix record missing for window=0 tail=7", result.stdout)
        self.assertFalse(self.calls.exists())
        self.assert_pairs_exist(pairs, True)
        self.assertEqual(self.done_index.read_text(), "")

    def test_source_change_during_ads_defers_whole_unit_until_next_round(self) -> None:
        pairs = self.make_streaming_unit()

        result = self.run_loop(
            self.mutating_ads_cli(pairs[15][0]),
            extra_env={"SYNC_FINAL_MAX_ATTEMPTS": "2"},
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("all snapshotted units changed; retaining sources", result.stdout)
        self.assertIn("round=0002", result.stdout)
        self.assert_pairs_exist(pairs, False)
        self.assertEqual(len(self.done_index.read_text().splitlines()), 3)

    def test_target_unit_requires_and_stages_all_three_structural_files(self) -> None:
        pairs = self.make_target_unit()

        result = self.run_loop(self.fake_ads_cli(), kind="FEAT_TARGET")

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assert_pairs_exist(pairs, False)
        self.assertEqual(len(self.done_index.read_text().splitlines()), 3)
        staged = set(self.staged_listing.read_text().splitlines())
        self.assertIn("scene/target_indices.npy", staged)
        self.assertIn("scene/target_indices.npy.manifest.json", staged)
        self.assertIn("scene/target_indices.meta.json", staged)
        self.assertEqual(len(staged), 9)  # three feature pairs + three structural files

    def test_seeded_streaming_validates_exact_seed_contract_then_publishes(self) -> None:
        pairs = self.make_streaming_unit(seeded=True)

        result = self.run_loop(
            self.fake_ads_cli(), namespace=STREAM_SEEDED_NAMESPACE
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assert_pairs_exist(pairs, False)
        self.assertEqual(len(self.calls.read_text().splitlines()), 1)
        self.assertEqual(len(self.done_index.read_text().splitlines()), 3)

    def test_seeded_corrupt_streaming_seed_never_uploads_or_deletes(self) -> None:
        pairs = self.make_streaming_unit(seeded=True)
        sidecar = pairs[20][1]
        meta = json.loads(sidecar.read_text())
        meta["noise_seed_samples"][0]["noise_seed"] += 1
        write_json(sidecar, meta)

        result = self.run_loop(
            self.fake_ads_cli(), namespace=STREAM_SEEDED_NAMESPACE
        )

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("feature sidecar noise_seed_samples mismatch", result.stdout)
        self.assertFalse(self.calls.exists())
        self.assert_pairs_exist(pairs, True)
        self.assertEqual(self.done_index.read_text(), "")

    def test_seeded_structural_namespace_mismatch_never_uploads_or_deletes(self) -> None:
        pairs = self.make_streaming_unit(seeded=True)
        sidecar = self.feature_root / "scene/prefix_index.npy.manifest.json"
        meta = json.loads(sidecar.read_text())
        meta["feature_namespace"] = "FEAT_STREAMING_CANONICAL_V2"
        write_json(sidecar, meta)

        result = self.run_loop(
            self.fake_ads_cli(), namespace=STREAM_SEEDED_NAMESPACE
        )

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("structural sidecar feature_namespace", result.stdout)
        self.assertFalse(self.calls.exists())
        self.assert_pairs_exist(pairs, True)
        self.assertEqual(self.done_index.read_text(), "")

    def test_seeded_feature_namespace_mismatch_never_uploads_or_deletes(self) -> None:
        pairs = self.make_streaming_unit(seeded=True)
        sidecar = pairs[25][1]
        meta = json.loads(sidecar.read_text())
        meta["feature_namespace"] = "FEAT_STREAMING_CANONICAL_V2"
        write_json(sidecar, meta)

        result = self.run_loop(
            self.fake_ads_cli(), namespace=STREAM_SEEDED_NAMESPACE
        )

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("feature sidecar feature_namespace", result.stdout)
        self.assertFalse(self.calls.exists())
        self.assert_pairs_exist(pairs, True)
        self.assertEqual(self.done_index.read_text(), "")

    def test_seeded_cross_consistent_but_unfrozen_protocol_sha_is_rejected(self) -> None:
        pairs = self.make_streaming_unit(seeded=True)
        wrong_sha = "b" * 64
        prefix_index = self.feature_root / "scene/prefix_index.npy"
        record = np.load(prefix_index, allow_pickle=True).tolist()[0]
        record["canonical_protocol_sha256"] = wrong_sha
        self._write_npy_with_sidecar(
            prefix_index,
            np.array([record], dtype=object),
            protocol_sha=wrong_sha,
            extra=seeded_generation(STREAM_SEEDED_NAMESPACE, "wan"),
        )
        for _feature, sidecar in pairs.values():
            meta = json.loads(sidecar.read_text())
            identity = dict(meta["noise_seed_identity"])
            identity["canonical_protocol_sha256"] = wrong_sha
            sample = noise_sample(identity)
            meta.update(
                {
                    "canonical_protocol_sha256": wrong_sha,
                    "noise_seed_samples": [sample],
                    **sample,
                }
            )
            write_json(sidecar, meta)

        result = self.run_loop(
            self.fake_ads_cli(), namespace=STREAM_SEEDED_NAMESPACE
        )

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("seeded-v1 canonical protocol SHA256 mismatch", result.stdout)
        self.assertFalse(self.calls.exists())
        self.assert_pairs_exist(pairs, True)
        self.assertEqual(self.done_index.read_text(), "")

    def test_trust_extract_publishes_without_rehashing_sidecar(self) -> None:
        pairs = self.make_streaming_unit(seeded=True)
        sidecar = pairs[15][1]
        meta = json.loads(sidecar.read_text())
        meta["sha256"] = "0" * 64
        write_json(sidecar, meta)

        result = self.run_loop(
            self.fake_ads_cli(),
            namespace=STREAM_SEEDED_NAMESPACE,
            extra_env={"SYNC_TRUST_EXTRACT": "1"},
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("trust_extract=1", result.stdout)
        self.assert_pairs_exist(pairs, False)
        self.assertEqual(len(self.done_index.read_text().splitlines()), 3)

    def test_feature_content_sha_is_recomputed_before_any_upload(self) -> None:
        pairs = self.make_streaming_unit(seeded=True)
        sidecar = pairs[15][1]
        meta = json.loads(sidecar.read_text())
        meta["sha256"] = "0" * 64
        write_json(sidecar, meta)

        result = self.run_loop(
            self.fake_ads_cli(), namespace=STREAM_SEEDED_NAMESPACE
        )

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("feature sidecar SHA256 mismatch", result.stdout)
        self.assertFalse(self.calls.exists())
        self.assert_pairs_exist(pairs, True)
        self.assertEqual(self.done_index.read_text(), "")

    def test_seeded_target_validates_ordered_frame_seeds_then_publishes(self) -> None:
        pairs = self.make_target_unit(seeded=True)

        result = self.run_loop(
            self.fake_ads_cli(),
            kind="FEAT_TARGET",
            namespace=TARGET_SEEDED_NAMESPACE,
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assert_pairs_exist(pairs, False)
        self.assertEqual(len(self.done_index.read_text().splitlines()), 3)
        self.assertEqual(len(self.staged_listing.read_text().splitlines()), 9)

    def test_seeded_target_reordered_seed_samples_never_uploads_or_deletes(self) -> None:
        pairs = self.make_target_unit(seeded=True)
        sidecar = pairs[25][1]
        meta = json.loads(sidecar.read_text())
        meta["noise_seed_samples"] = list(reversed(meta["noise_seed_samples"]))
        write_json(sidecar, meta)

        result = self.run_loop(
            self.fake_ads_cli(),
            kind="FEAT_TARGET",
            namespace=TARGET_SEEDED_NAMESPACE,
        )

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("feature sidecar noise_seed_samples mismatch", result.stdout)
        self.assertFalse(self.calls.exists())
        self.assert_pairs_exist(pairs, True)
        self.assertEqual(self.done_index.read_text(), "")

    def test_seeded_minimax_fixed_seed_contract_publishes(self) -> None:
        pairs = self.make_streaming_unit(
            seeded=True,
            vfm="minimax_h3",
            timestep=500,
            model_id=MINIMAX_MODEL_ID,
        )

        result = self.run_loop(
            self.fake_ads_cli(),
            namespace=STREAM_SEEDED_NAMESPACE,
            vfm="minimax_h3",
            timestep=500,
            model_id=MINIMAX_MODEL_ID,
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assert_pairs_exist(pairs, False)
        self.assertEqual(len(self.done_index.read_text().splitlines()), 3)

    def test_seeded_minimax_noninteger_fixed_seed_never_uploads_or_deletes(self) -> None:
        pairs = self.make_streaming_unit(
            seeded=True,
            vfm="minimax_h3",
            timestep=500,
            model_id=MINIMAX_MODEL_ID,
        )
        sidecar = pairs[15][1]
        meta = json.loads(sidecar.read_text())
        meta["noise_seed"] = False
        write_json(sidecar, meta)

        result = self.run_loop(
            self.fake_ads_cli(),
            namespace=STREAM_SEEDED_NAMESPACE,
            vfm="minimax_h3",
            timestep=500,
            model_id=MINIMAX_MODEL_ID,
        )

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("expected integer 0", result.stdout)
        self.assertFalse(self.calls.exists())
        self.assert_pairs_exist(pairs, True)
        self.assertEqual(self.done_index.read_text(), "")

    def test_seeded_model_override_is_rejected_before_upload(self) -> None:
        pairs = self.make_streaming_unit(seeded=True)

        result = self.run_loop(
            self.fake_ads_cli(),
            namespace=STREAM_SEEDED_NAMESPACE,
            extra_env={"EXPECTED_MODEL_ID": "unexpected/model"},
        )

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("model_id may not be overridden", result.stdout)
        self.assertFalse(self.calls.exists())
        self.assert_pairs_exist(pairs, True)

    def test_seeded_prompt_override_is_rejected_before_upload(self) -> None:
        pairs = self.make_streaming_unit(seeded=True)

        result = self.run_loop(
            self.fake_ads_cli(),
            namespace=STREAM_SEEDED_NAMESPACE,
            extra_env={"EXPECTED_PROMPT": "unexpected prompt"},
        )

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("prompt may not be overridden", result.stdout)
        self.assertFalse(self.calls.exists())
        self.assert_pairs_exist(pairs, True)


if __name__ == "__main__":
    unittest.main()
