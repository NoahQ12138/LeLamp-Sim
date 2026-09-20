"""Phone exclusivity, delayed detector evidence, audio dispatch, and preemption."""
import unittest
from unittest.mock import Mock
import numpy as np
from detection_events import DetectionEvents, LEAVE_FRAMES
from verify_vision import confirm
from vision_behavior import start_reaction
from speech import PHONE_VOICE_LINE


class PhoneTests(unittest.TestCase):
    def test_all_requested_combinations_dispatch_phone_only_once(self):
        for lower in ({"FACE":True,"PEACE":True}, {"PEACE":True}, {"FACE":True},
                      {"HAND":True,"FACE":True,"PEACE":True,"HAPPY":True}):
            events, speaker, manager = DetectionEvents(), Mock(), Mock()
            confirm(events, PHONE=True, **lower)
            event = events.trigger(0)
            self.assertEqual(event, "PHONE")
            start_reaction(manager, event, speaker)
            speaker.speak.assert_called_once_with(PHONE_VOICE_LINE, interrupt=True)
            for tick in range(1, 100):
                events.observe({"PHONE":True, **lower})
                self.assertIsNone(events.trigger(tick))
            self.assertFalse(events.latched["FACE"])
            self.assertFalse(events.latched["PEACE"])

    def test_first_phone_evidence_blocks_faster_face_and_peace(self):
        events = DetectionEvents()
        confirm(events, FACE=True, PEACE=True)
        events.observe({"PHONE":True})
        self.assertIsNone(events.trigger(0))
        events.observe({"PHONE":None})
        self.assertIsNone(events.trigger(.1))
        events.observe({"PHONE":True})
        self.assertEqual(events.trigger(.2), "PHONE")

    def test_hand_priority_and_phone_rearm(self):
        events = DetectionEvents()
        confirm(events, HAND=True, FACE=True)
        self.assertEqual(events.trigger(0), "HAND")
        confirm(events, PEACE=True)
        self.assertEqual(events.trigger(1, "HAND"), "PEACE")
        confirm(events, PHONE=True)
        self.assertEqual(events.trigger(2, "PEACE"), "PHONE")
        for _ in range(LEAVE_FRAMES):
            events.observe({"PHONE":False})
        confirm(events, PHONE=True)
        self.assertIsNone(events.trigger(6.99))
        self.assertEqual(events.trigger(7), "PHONE")

    def test_phone_preempts_dance_continuously_and_flashes(self):
        from behaviors import BehaviorManager
        from lelamp_controller import LeLampController
        controller = LeLampController()
        manager = BehaviorManager(controller)
        manager.start("PEACE", controller.data.time)
        for _ in range(70):
            manager.update(controller.data.time)
            controller.step()
        before = controller.data.ctrl.copy()
        start_reaction(manager, "PHONE", Mock())
        manager.update(controller.data.time)
        np.testing.assert_allclose(controller.data.ctrl, before)
        light_states = set()
        for _ in range(450):
            manager.update(controller.data.time)
            controller.step()
            light_states.add(controller.lamp.enabled)
        self.assertEqual(light_states, {True, False})
        self.assertEqual(manager.state, "IDLE")
        np.testing.assert_allclose(controller.data.ctrl, controller.neutral)


if __name__ == "__main__":
    unittest.main()
