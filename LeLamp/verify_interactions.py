"""Regression checks for idle/wake, expression gating, and authoritative voice OFF."""
import unittest
import numpy as np
from activity import ActivityLight
from detection_events import DetectionEvents
from expressions import classify_scores
from speech import recognized_command
from verify_vision import confirm


class InteractionTests(unittest.TestCase):
    def test_accepted_expression_dispatches_after_wake_with_face_still_visible(self):
        from unittest.mock import Mock
        from vision_behavior import start_reaction
        for expression in ("ANGRY", "HAPPY"):
            events, light = DetectionEvents(), ActivityLight(0)
            confirm(events, FACE=True)
            self.assertEqual(events.trigger(0), "FACE")
            light.update(10)
            light.update(12)
            confirm(events, FACE=True, **{expression:True})
            pending = events.trigger(13)
            self.assertEqual(pending, expression)
            light.interact(13)
            self.assertFalse(light.awake)
            light.update(13.7)
            self.assertTrue(light.awake)
            self.assertFalse(events.blocks_pending(pending))
            manager, speaker = Mock(), Mock()
            if not events.blocks_pending(pending) and light.awake:
                start_reaction(manager, pending, speaker)
            manager.start.assert_called_once()
            if expression == "ANGRY":
                speaker.speak_angry.assert_called_once_with()
            self.assertIsNone(events.trigger(20))

    def test_pending_expression_still_yields_to_phone_and_hands(self):
        for higher in ("PHONE", "PEACE", "HAND"):
            events = DetectionEvents()
            confirm(events, FACE=True)
            events.trigger(0)
            confirm(events, ANGRY=True)
            self.assertEqual(events.trigger(1), "ANGRY")
            events.observe({higher:True})
            self.assertTrue(events.blocks_pending("ANGRY"))
            confirm(events, **{higher:True})
            self.assertEqual(events.trigger(2, "ANGRY"), higher)

    def test_idle_and_wake_are_smooth_and_discrete(self):
        light = ActivityLight(0)
        self.assertEqual(light.update(9.99), 1)
        self.assertEqual(light.update(10), 1)
        self.assertAlmostEqual(light.update(11), .6)
        self.assertAlmostEqual(light.update(12), .2)
        self.assertAlmostEqual(light.update(20), .2)
        light.interact(20)
        self.assertFalse(light.awake)
        self.assertAlmostEqual(light.update(20.3), .6)
        self.assertAlmostEqual(light.update(20.6), 1)
        self.assertTrue(light.awake)
        self.assertEqual(light.last_activity, 20)

    def test_face_presence_does_not_keep_lamp_awake(self):
        light, events = ActivityLight(0), DetectionEvents()
        confirm(events, FACE=True)
        self.assertEqual(events.trigger(0), "FACE")
        light.interact(0)
        for tick in range(1, 14):
            events.observe({"FACE": True})
            self.assertIsNone(events.trigger(tick))
            light.update(tick)
        self.assertAlmostEqual(light.value, .2)

    def test_expressions_wait_for_greeting_and_need_face(self):
        for expression in ("HAPPY", "ANGRY"):
            events = DetectionEvents()
            confirm(events, **{expression: True})
            self.assertIsNone(events.trigger(0))
            confirm(events, FACE=True)
            self.assertEqual(events.trigger(1), "FACE")
            self.assertIsNone(events.trigger(2, "FACE"))
            self.assertEqual(events.trigger(4), expression)
            self.assertIsNone(events.trigger(20))

    def test_expression_model_class_order_and_uncertainty(self):
        for index, expected in ((1,"HAPPY"), (4,"ANGRY"), (3,"UNKNOWN"), (0,"UNKNOWN")):
            logits = np.zeros(8)
            logits[index] = 8
            self.assertEqual(classify_scores(logits)[0], expected)
        self.assertEqual(classify_scores(np.zeros(8))[0], "UNKNOWN")
        with self.assertRaises(ValueError):
            classify_scores([float("nan")]*8)

    def test_voice_requires_exact_final_high_confidence_command(self):
        for text, expected in (("  LAMP ON  ","ON"), ("on","ON"), ("turn on","ON"),
                               ("off","OFF"), ("turn off","OFF"), ("lamp off","OFF"),
                               ("hello there",None), ("please turn off now",None)):
            self.assertEqual(recognized_command({"text":text,"result":[{"conf":.95}]})[1], expected)
        self.assertIsNone(recognized_command({"partial":"on"})[1])
        self.assertIsNone(recognized_command({"text":"on","result":[{"conf":.5}]})[1])

    def test_sensitive_anger_accepts_neutral_split_but_rejects_other_winners(self):
        for probabilities, expected in (
            ([.25,.05,.05,.05,.45,.05,.05,.05], "ANGRY"),
            ([.60,.03,.03,.03,.22,.03,.03,.03], "UNKNOWN"),
            ([.10,.40,.04,.04,.30,.04,.04,.04], "UNKNOWN"),
            ([.12,.12,.12,.12,.16,.12,.12,.12], "UNKNOWN"),
            ([.25,.05,.05,.45,.05,.05,.05,.05], "UNKNOWN"),
            ([.40,.04,.04,.04,.36,.04,.04,.04], "ANGRY"),
            ([.39,.03,.03,.03,.40,.04,.04,.04], "ANGRY"),
            ([.15,.15,.10,.10,.30,.10,.05,.05], "ANGRY"),
        ):
            self.assertEqual(classify_scores(np.log(probabilities))[0], expected)

    def test_manual_off_overrides_every_reaction_and_idle(self):
        from lelamp_controller import LeLampController
        from behaviors import BehaviorManager, REACTIONS
        controller = LeLampController()
        base = controller.activity.last_activity
        controller.voice_command("OFF", base)
        for name in REACTIONS:
            manager = BehaviorManager(controller)
            manager.start(name, controller.data.time)
            for _ in range(550):
                manager.update(controller.data.time)
                controller.update_lighting(base+100)
                controller.step()
                self.assertFalse(controller.lamp.enabled)
            self.assertEqual(manager.state, "IDLE")
            np.testing.assert_allclose(controller.data.ctrl, controller.neutral)
        controller.voice_command("ON", base+101)
        controller.update_lighting(base+102)
        self.assertTrue(controller.lamp.enabled)
        self.assertAlmostEqual(controller.lamp.brightness, 1)


if __name__ == "__main__":
    unittest.main()
