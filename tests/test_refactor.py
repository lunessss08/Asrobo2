"""Checks that the split modules still cooperate without connecting a robot."""
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import gimbalshoot as gs
import maze_shooter as maze
from robo_app.aim import settings as aim_settings
from robo_app.maze import settings as maze_settings


class ModuleIntegrationTests(unittest.TestCase):
    def test_runtime_files_stay_in_project_root(self):
        root = Path(__file__).resolve().parents[1]
        for path in (gs.CONFIG_PATH, maze.MAP_FILE, maze.LOOK_LOG_ROOT, maze.SENSOR_MAP_FILE):
            self.assertEqual(path.parent, root)

    def test_map_routes_and_serialization(self):
        m = maze.MazeMap(3, 2)
        m.set_edge((0, 0), (1, 0), 1)
        m.set_edge((1, 0), (2, 0), 1)
        m.visited.add((0, 0))
        self.assertTrue(m.add_target((2, 0), 'red', 'square', 1, (0.87, 0.0)))
        self.assertEqual(m.path_to((0, 0), 1, (2, 0)), [(0, 0), (1, 0), (2, 0)])
        self.assertIsNone(m.path_to((0, 0), 1, (0, 1)))
        restored = maze.MazeMap.from_json(m.to_json())
        self.assertEqual(restored.to_json(), m.to_json())

    def test_simulated_camera_feeds_vision(self):
        robot = maze.SimRobot((0, 0), 0, 4, 4)
        frame = robot.read_frame()
        fd = gs.prepare_frame(frame)
        targets = gs.detect_all_colors(fd, gs.default_color_ranges(), gs.BlindZones(gs.AUTO_BLIND_ZONE))
        self.assertEqual(frame.shape[2], 3)
        self.assertIsInstance(targets, list)
        self.assertGreater(robot.read_tof_mm(), 0)
        robot.close()

    def test_maze_threshold_override_reaches_vision(self):
        # This plate exceeds auto-aim's original maximum area, but must be
        # accepted when the maze app relaxes the thresholds for nearby plates.
        self.assertGreaterEqual(aim_settings.MAX_AREA_RATIO, maze_settings.PLATE_MAX_AREA_RATIO)
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        cv2.rectangle(frame, (350, 180), (510, 300), (0, 0, 255), -1)
        fd = gs.prepare_frame(frame)
        rng = gs.default_color_ranges()['red']
        zones = gs.BlindZones(gs.AUTO_BLIND_ZONE)
        previous = aim_settings.MAX_AREA_RATIO
        try:
            aim_settings.MAX_AREA_RATIO = 0.001
            fd = gs.prepare_frame(frame)
            self.assertEqual(gs.find_targets(fd, gs.build_mask(fd.hsv, rng, zones), 'red'), [])
            aim_settings.MAX_AREA_RATIO = maze_settings.PLATE_MAX_AREA_RATIO
            fd = gs.prepare_frame(frame)
            self.assertTrue(gs.find_targets(fd, gs.build_mask(fd.hsv, rng, zones), 'red'))
        finally:
            aim_settings.MAX_AREA_RATIO = previous

    def test_simulation_mission_renders_without_hardware(self):
        robot = maze.SimRobot((0, 0), 0, 4, 4)
        grabber = maze.FrameGrabber(robot)
        grabber.frame = robot.read_frame()
        mission = maze.MazeMission(robot, grabber, maze.MazeMap(4, 4), (0, 0), 0)
        app = maze.App(mission, grabber)
        with patch('cv2.imshow') as show:
            app.render()
        canvas = show.call_args[0][1]
        self.assertEqual(canvas.shape, (720, 1280, 3))
        mission.stop()
        robot.close()

    def test_controller_defaults_remain_safe(self):
        controller = gs.FireController()
        self.assertFalse(controller.armed)
        self.assertFalse(controller.evaluate(0, 0, False)[1])
        pid = gs.PIDController(2.5, 0, 0, output_limit=25)
        self.assertEqual(pid.compute(100), 25)


if __name__ == '__main__':
    unittest.main()
