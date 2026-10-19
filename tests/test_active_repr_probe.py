import unittest

from active_repr_probe.geometry import direction_bin, distance_bin, distance_m, relative_camera_vector, rotation_delta_deg
from active_repr_probe.protocol import validate_state_bank


IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


class GeometryTests(unittest.TestCase):
    def test_camera_axes_match_active_spatial_plus_z_forward(self):
        self.assertEqual(relative_camera_vector(IDENTITY, [0, 0, 2]), (0.0, 0.0, 2.0))
        self.assertEqual(direction_bin(IDENTITY, [0, 0, 2])["azimuth_bin"], 0)
        self.assertEqual(direction_bin(IDENTITY, [2, 0, 0])["azimuth_bin"], 2)
        self.assertEqual(direction_bin(IDENTITY, [-2, 0, 0])["azimuth_bin"], 6)

    def test_distance_edges_are_frozen(self):
        self.assertEqual(distance_m(IDENTITY, [0, 0, 3]), 3.0)
        self.assertEqual([distance_bin(x) for x in (0.5, 1.0, 2.0, 4.0, 6.0)], [0, 1, 2, 3, 4])

    def test_rotation_delta(self):
        quarter_turn = [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]
        self.assertAlmostEqual(rotation_delta_deg(IDENTITY, quarter_turn), 90.0, places=5)


class ProtocolTests(unittest.TestCase):
    def test_validator_requires_contiguous_w3_history(self):
        import json
        import tempfile
        from pathlib import Path
        row = {"sample_id": "one", "scene_id": "s", "episode_id": "e", "split": "train", "history": [{"env_step": 0}, {"env_step": 1}, {"env_step": 2}], "task": {}, "labels": {}, "observability": {}, "counterfactual_actions": []}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"; path.write_text(json.dumps(row) + "\n")
            self.assertTrue(validate_state_bank(path)["valid"])


if __name__ == "__main__":
    unittest.main()
