"""Single-thread-owned simulation controller, reusing existing model/light safety code."""
import mujoco
import numpy as np
import time
from activity import ActivityLight
from behavior_test import clamp_command
from joint_test import prepare, check_state
from lamp_light import LampLight, load_lit_model

YAW_ACTUATOR = "1"
PITCH_ACTUATOR = "3"  # Positive tilts the measured lamp beam downward.
HEAD_ACTUATOR = "5"  # Joint on the actual diffuser/head body.


class LeLampController:
    def __init__(self):
        self.model = load_lit_model()
        self.data = mujoco.MjData(self.model)
        self.neutral, _ = prepare(self.model, self.data)
        self.yaw = self.model.actuator(YAW_ACTUATOR).id
        self.pitch = self.model.actuator(PITCH_ACTUATOR).id
        self.lamp = LampLight(self.model)  # Caller owns viewer lock; no second thread writes.
        self.lamp.lamp_on()
        self.manual_lamp_off = False
        self.activity = ActivityLight()
        self.reaction_brightness = 1.0
        self.model.vis.headlight.diffuse[:] = 0.15
        self.model.vis.headlight.ambient[:] = 0.12
        for i in range(self.model.nlight):
            if i != self.lamp.light_id:
                self.model.light_diffuse[i] *= 0.15
        for name in (YAW_ACTUATOR, PITCH_ACTUATOR, HEAD_ACTUATOR):
            a = self.model.actuator(name).id
            j = int(self.model.actuator_trnid[a, 0])
            print(f"Actuator {name!r} -> joint {self.model.joint(j).name!r}: "
                  f"ctrl={self.model.actuator_ctrlrange[a]}, joint={self.model.jnt_range[j]} rad", flush=True)

    def pose(self, values):
        result = self.neutral.copy()
        for name, value in values.items():
            a = self.model.actuator(name).id
            j = int(self.model.actuator_trnid[a, 0])
            low, high = self.model.jnt_range[j]
            # Keep a five-degree buffer, in addition to both actual model limits.
            value = float(np.clip(value, low + np.deg2rad(5), high - np.deg2rad(5)))
            result[a] = clamp_command(self.model, a, value)
        return result

    def step(self):
        check_state(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        check_state(self.model, self.data)

    def set_light(self, enabled):
        enabled = bool(enabled) and not self.manual_lamp_off
        if bool(enabled) != self.lamp.enabled:
            (self.lamp.lamp_on if enabled else self.lamp.lamp_off)()

    def voice_command(self, command, now=None):
        if command not in ("ON", "OFF"):
            return
        self.manual_lamp_off = command == "OFF"
        self.note_activity(time.monotonic() if now is None else now)
        self.reaction_brightness = 1.0
        self.update_lighting(time.monotonic() if now is None else now)
        self.set_light(command == "ON")
        print(f"Lamp command: {command}; manual-off={self.manual_lamp_off}", flush=True)

    def set_reaction_brightness(self, value):
        self.reaction_brightness = float(np.clip(value, 0, 1))
        self._apply_brightness()

    def _apply_brightness(self):
        value = self.activity.value * self.reaction_brightness
        if abs(self.lamp.brightness-value) > 1e-5:
            self.lamp.set_lamp_brightness(value)

    def update_lighting(self, now):
        if not self.manual_lamp_off:
            self.activity.update(now)
        self._apply_brightness()

    def note_activity(self, now):
        if self.manual_lamp_off:
            self.activity.last_activity = now
        else:
            self.activity.interact(now)
