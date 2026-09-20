"""Deterministic priority/preemption and detection-latch regression checks."""
import unittest
import numpy as np
from detection_events import DetectionEvents, CONFIRM, COOLDOWN, LEAVE_FRAMES


def confirm(events, **detections):
    for _ in range(max(CONFIRM.values())):
        events.observe(detections)


def context_for(name):
    events = DetectionEvents()
    if name in ("HAPPY", "ANGRY"):
        confirm(events, FACE=True)
        assert events.trigger(-10) == "FACE"
    return events


class PriorityTests(unittest.TestCase):
    def test_simultaneous_phone_wins_and_blocks_lower_classes(self):
        events = DetectionEvents()
        confirm(events, PHONE=True, PEACE=True, FACE=True)
        self.assertEqual(events.trigger(0), "PHONE")
        self.assertIsNone(events.trigger(100, "IDLE"))
        for _ in range(LEAVE_FRAMES):
            events.observe({"PHONE": False, "PEACE": True, "FACE": True})
        self.assertEqual(events.trigger(101), "PEACE")

    def test_higher_priority_preempts_but_lower_does_not(self):
        events = DetectionEvents()
        confirm(events, FACE=True)
        self.assertEqual(events.trigger(0), "FACE")
        confirm(events, PEACE=True)
        self.assertEqual(events.trigger(0.1, "FACE"), "PEACE")
        confirm(events, PHONE=True)
        self.assertEqual(events.trigger(0.2, "PEACE"), "PHONE")
        other = DetectionEvents()
        confirm(other, FACE=True, PEACE=True)
        self.assertIsNone(other.trigger(0, "PHONE"))

    def test_phone_preemption_is_continuous_and_finishes_safely(self):
        from behaviors import BehaviorManager
        from lelamp_controller import LeLampController
        controller = LeLampController()
        manager = BehaviorManager(controller)
        manager.start("FACE", controller.data.time)
        for _ in range(65):
            manager.update(controller.data.time)
            controller.step()
        before = controller.data.ctrl.copy()
        manager.start("PHONE", controller.data.time)
        manager.update(controller.data.time)
        np.testing.assert_allclose(controller.data.ctrl, before)
        for _ in range(400):
            manager.update(controller.data.time)
            controller.step()
        self.assertEqual(manager.state, "IDLE")
        np.testing.assert_allclose(controller.data.ctrl, controller.neutral)
        self.assertTrue(controller.lamp.enabled)


class LatchTests(unittest.TestCase):
    def test_confirmation_and_negative_frame_reset(self):
        for name, required in CONFIRM.items():
            with self.subTest(name=name):
                events = context_for(name)
                for _ in range(required-1):
                    events.observe({name: True})
                self.assertIsNone(events.trigger(0))
                events.observe({name: False})
                for _ in range(required-1):
                    events.observe({name: True})
                self.assertIsNone(events.trigger(1))
                events.observe({name: True})
                self.assertEqual(events.trigger(2), name)

    def test_continuous_visibility_never_retriggers(self):
        for name in CONFIRM:
            events = context_for(name)
            confirm(events, **{name: True})
            self.assertEqual(events.trigger(0), name)
            for tick in range(1, 101):
                events.observe({name: True})
                self.assertIsNone(events.trigger(tick))

    def test_both_disappearance_and_cooldown_required(self):
        for name in CONFIRM:
            with self.subTest(name=name):
                events = context_for(name)
                confirm(events, **{name: True})
                self.assertEqual(events.trigger(0), name)
                for _ in range(LEAVE_FRAMES-1):
                    events.observe({name: False})
                confirm(events, **{name: True})
                self.assertIsNone(events.trigger(100))
                for _ in range(LEAVE_FRAMES):
                    events.observe({name: False})
                confirm(events, **{name: True})
                self.assertIsNone(events.trigger(COOLDOWN[name]-0.01))
                self.assertEqual(events.trigger(COOLDOWN[name]), name)

    def test_skipped_yolo_results_neither_confirm_nor_rearm(self):
        events = DetectionEvents()
        events.observe({"PHONE": True})
        for _ in range(10):
            events.observe({"PHONE": None})
        self.assertIsNone(events.trigger(0))
        events.observe({"PHONE": True})
        self.assertEqual(events.trigger(0), "PHONE")
        for _ in range(100):
            events.observe({"PHONE": None})
        self.assertTrue(events.latched["PHONE"])
        self.assertIsNone(events.trigger(100))


if __name__ == "__main__":
    unittest.main()
