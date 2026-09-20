"""Check peace-sign geometry, rejection cases, and invariance to hand orientation."""
from types import SimpleNamespace
import unittest
import numpy as np
from gestures import is_peace_sign


def hand(extended=(5, 9)):
    points = np.zeros((21, 3))
    for base, x in ((5,-0.6),(9,0),(13,0.6),(17,1.2)):
        if base in extended:
            points[base:base+4] = [[x,1,0],[x,2,0],[x,3,0],[x,4,0]]
        else:
            points[base:base+4] = [[x,1,0],[x,2,0],[x,1.6,0.5],[x,1.0,0.5]]
    return points


def detect(points):
    return is_peace_sign([SimpleNamespace(x=x,y=y,z=z) for x,y,z in points])


class PeaceTests(unittest.TestCase):
    def test_peace_with_rotation_mirroring_and_scale(self):
        points = hand()
        # Separate fingertips enough for a clear V.
        points[5:9,0] -= np.arange(4)*0.1
        points[9:13,0] += np.arange(4)*0.1
        self.assertTrue(detect(points))
        self.assertTrue(detect(points * [-1,1,1]))
        self.assertTrue(detect(points @ np.array([[0,0,1],[1,0,0],[0,1,0]]) * 0.03 + 2))

    def test_open_palm_fist_and_pointing_do_not_trigger(self):
        for extended in ((5,9,13,17), (), (5,), (9,), (5,9,13)):
            with self.subTest(extended=extended):
                self.assertFalse(detect(hand(extended)))

    def test_closed_v_and_invalid_landmarks_do_not_trigger(self):
        points = hand()
        points[9:13,0] = points[5:9,0]
        self.assertFalse(detect(points))
        self.assertFalse(detect(np.zeros((21,3))))
        self.assertFalse(detect(np.full((21,3), np.nan)))
        self.assertFalse(is_peace_sign([]))


if __name__ == "__main__":
    unittest.main()
