"""Measured impact corrections are applied at the actual shooting distance."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from robo_app.maze.shot_calibration import load_shot_calibration, shot_offset


class ShotCalibrationTests(unittest.TestCase):
    def test_load_and_interpolate_measured_impact_bias(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "shot_calibration.json"
            path.write_text(json.dumps({"samples": [
                {"distance_m": 0.8, "miss_right_cm": 0, "miss_up_cm": -2},
                {"distance_m": 0.4, "miss_right_cm": 2, "miss_up_cm": 0},
            ]}), encoding="utf-8")
            points = load_shot_calibration(path)
        x, y = shot_offset(points, 0.6)
        self.assertAlmostEqual(x, math.degrees(math.atan2(0.02, 0.4)) / 2)
        self.assertAlmostEqual(y, math.degrees(math.atan2(-0.02, 0.8)) / 2)
        self.assertEqual(shot_offset(points, 0.1), points[0][1:])
        self.assertEqual(shot_offset(points, 1.0), points[-1][1:])

    def test_invalid_calibration_is_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "shot_calibration.json"
            path.write_text('{"samples":[{"distance_m":0.4,"miss_right_cm":500,"miss_up_cm":0}]}',
                            encoding="utf-8")
            self.assertEqual(load_shot_calibration(path), [])


if __name__ == "__main__":
    unittest.main()
