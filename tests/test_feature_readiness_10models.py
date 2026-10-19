"""Dashboard state must not turn missing historical evidence into missing data."""
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "readiness", Path(__file__).resolve().parents[1] / "scripts/update_feature_readiness_10models.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReadinessTests(unittest.TestCase):
    def test_old_unreadable_receipt_does_not_override_complete_coverage(self):
        stats = {m: {"remaining_without_verified_evidence": 0} for m in ("streaming", "target")}
        self.assertEqual(module.feature_status(stats, True, "dino", [{"error": "PermissionError"}]),
                         ("READY", True))
        self.assertEqual(module.feature_status(stats, True, "dino", [{"blocking": True}]),
                         ("PENDING", False))

    def test_target_alone_does_not_make_whole_model_ready(self):
        stats = {"streaming": {"remaining_without_verified_evidence": 1},
                 "target": {"remaining_without_verified_evidence": 0}}
        self.assertEqual(module.feature_status(stats, True, "dino", []), ("PENDING", False))

    def test_additional_models_are_not_implicitly_added_to_queue(self):
        stats = {"streaming": {"remaining_without_verified_evidence": 0},
                 "target": {"remaining_without_verified_evidence": 0}}
        self.assertEqual(module.feature_status(stats, False, "aether", []), ("READY", False))

    def test_latest_attempt_wins_regardless_of_order(self):
        old = dict(model="vjepa2", owner="stream-01", category="production",
                   created_utc="2026-09-22", state="RUNNING")
        new = dict(old, created_utc="2026-09-23", state="FAILED")
        for jobs in ([old, new], [new, old]):
            self.assertEqual(module.latest_owner_jobs(jobs, "vjepa2"), [new])

    def test_distinct_canary_paths_do_not_collapse(self):
        first = dict(model="f3r", owner="/stream/f3r.sh", category="canary",
                     created_utc="2026-09-23", state="FAILED")
        second = dict(first, owner="/target/f3r.sh", state="SUCCEEDED")
        self.assertEqual(len(module.latest_owner_jobs([first, second], "f3r")), 2)

    def test_postcanary_is_production_and_shards_are_separate(self):
        owner, category = module.job_owner('bash /dino_postcanary/scripts/stream.sh', 'dino-postcanary')
        self.assertEqual((owner, category), ('stream.sh', 'production'))
        a = module.job_owner('SHARD_INDEX=0 SHARD_COUNT=8 bash /mini/worker.sh', 'mini-s00')
        b = module.job_owner('SHARD_INDEX=1 SHARD_COUNT=8 bash /mini/worker.sh', 'mini-s01')
        self.assertNotEqual(a[0], b[0])


if __name__ == "__main__":
    unittest.main()
