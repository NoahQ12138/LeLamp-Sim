"""Pointing geometry and priority checks; physics coverage also lives in verify_interactions."""
from types import SimpleNamespace
import unittest
import numpy as np
from gestures import pointing_direction
from verify_gestures import hand
from verify_vision import confirm
from detection_events import DetectionEvents


def landmarks(points):
    return [SimpleNamespace(x=x, y=y, z=z) for x,y,z in points]


class PointTests(unittest.TestCase):
    def test_four_directions_with_aspect_ratio_and_mirroring(self):
        for angle, expected in ((0,"DOWN"), (90,"LEFT"), (180,"UP"), (270,"RIGHT")):
            theta = np.deg2rad(angle)
            rotation = np.array([[np.cos(theta),-np.sin(theta),0],
                                 [np.sin(theta),np.cos(theta),0],[0,0,1]])
            world = hand((5,)) @ rotation.T
            pixels = world * 35 + [320,240,0]
            normalized = pixels / [640,480,1]
            self.assertEqual(pointing_direction(landmarks(world), landmarks(normalized),640,480), expected)

    def test_rejects_other_hand_shapes_diagonal_and_foreshortened(self):
        for shape in ((), (5,9), (5,9,13,17)):
            points = hand(shape)
            self.assertIsNone(pointing_direction(landmarks(points), landmarks(points),640,480))
        world = hand((5,))
        pixels = world.copy()
        pixels[8,:2] = pixels[5,:2] + [.01,.01]
        self.assertIsNone(pointing_direction(landmarks(world),landmarks(pixels),640,480))
        pixels[8,:2] = pixels[5,:2] + [3/640,3/480]
        pixels[17,:2] = pixels[5,:2] + [.001,0]
        self.assertIsNone(pointing_direction(landmarks(world),landmarks(pixels),640,480))
        self.assertIsNone(pointing_direction([],[],640,480))

    def test_pointing_wins_over_hand_but_not_phone_or_peace(self):
        events = DetectionEvents()
        confirm(events, POINT_LEFT=True,HAND=True,FACE=True)
        self.assertEqual(events.trigger(0), "POINT_LEFT")
        self.assertIsNone(events.trigger(10))
        confirm(events, PEACE=True)
        self.assertEqual(events.trigger(11,"POINT_LEFT"), "PEACE")
        confirm(events, PHONE=True)
        self.assertEqual(events.trigger(12,"PEACE"), "PHONE")


if __name__ == "__main__":
    unittest.main()
