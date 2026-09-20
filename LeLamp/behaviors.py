"""Non-blocking cosine-eased reaction sequences, updated at every physics step."""
from dataclasses import dataclass
import math
from lelamp_controller import PITCH_ACTUATOR, YAW_ACTUATOR, HEAD_ACTUATOR

FACE_AMPLITUDE = 0.12
FACE_FLASH_INTERVAL = 0.55
DANCE_YAW_AMPLITUDE = 0.12
DANCE_PITCH_AMPLITUDE = 0.10
DANCE_HEAD_AMPLITUDE = 0.12
DANCE_BEAT_SECONDS = 0.40
PHONE_SHAKE_AMPLITUDE = 0.18
PHONE_SHAKE_CYCLES = 4
PHONE_FLASH_INTERVAL = 0.25
PHONE_SPEECH_LEAD_SECONDS = 0.35
HAPPY_AMPLITUDE = 0.06
ANGRY_TILT_AMPLITUDE = 0.08
ANGRY_BRIGHTNESS = 0.65
POINT_YAW = 0.15
POINT_PITCH = 0.10


def point_reaction(direction):
    poses = {"LEFT": {YAW_ACTUATOR: POINT_YAW}, "RIGHT": {YAW_ACTUATOR: -POINT_YAW},
             "UP": {PITCH_ACTUATOR: -POINT_PITCH}, "DOWN": {PITCH_ACTUATOR: POINT_PITCH}}
    pose = poses[direction]
    return [Segment(0.55, pose), Segment(1.0, pose), Segment(0.6, {})]


@dataclass
class Segment:
    duration: float
    pose: dict
    light: bool = True
    brightness: float = 1.0


def face_reaction():
    return [Segment(0.3, {}),
            Segment(FACE_FLASH_INTERVAL, {PITCH_ACTUATOR: FACE_AMPLITUDE}, True),
            Segment(FACE_FLASH_INTERVAL, {PITCH_ACTUATOR: -FACE_AMPLITUDE}, False),
            Segment(FACE_FLASH_INTERVAL, {PITCH_ACTUATOR: FACE_AMPLITUDE/2}, True),
            Segment(FACE_FLASH_INTERVAL, {}, True)]


def peace_reaction():
    return [Segment(0.3, {}),
            Segment(DANCE_BEAT_SECONDS, {YAW_ACTUATOR: DANCE_YAW_AMPLITUDE}),
            Segment(DANCE_BEAT_SECONDS, {YAW_ACTUATOR: -DANCE_YAW_AMPLITUDE}),
            Segment(DANCE_BEAT_SECONDS, {PITCH_ACTUATOR: -DANCE_PITCH_AMPLITUDE}),
            Segment(DANCE_BEAT_SECONDS, {PITCH_ACTUATOR: DANCE_PITCH_AMPLITUDE}),
            Segment(DANCE_BEAT_SECONDS, {HEAD_ACTUATOR: DANCE_HEAD_AMPLITUDE}),
            Segment(DANCE_BEAT_SECONDS, {HEAD_ACTUATOR: -DANCE_HEAD_AMPLITUDE}),
            Segment(0.6, {})]


def phone_reaction():
    sequence = [Segment(PHONE_SPEECH_LEAD_SECONDS, {}, True)]
    for _ in range(PHONE_SHAKE_CYCLES):
        sequence.extend([Segment(PHONE_FLASH_INTERVAL, {YAW_ACTUATOR: PHONE_SHAKE_AMPLITUDE}, True),
                         Segment(PHONE_FLASH_INTERVAL, {YAW_ACTUATOR: -PHONE_SHAKE_AMPLITUDE}, False)])
    return sequence + [Segment(0.5, {})]


def hand_reaction():
    return [Segment(0.3, {}), Segment(0.45, {YAW_ACTUATOR: 0.08}),
            Segment(0.45, {YAW_ACTUATOR: -0.08}), Segment(0.5, {})]


def happy_reaction():
    return [Segment(0.3, {}),
            Segment(0.5, {YAW_ACTUATOR: HAPPY_AMPLITUDE, PITCH_ACTUATOR: -HAPPY_AMPLITUDE}),
            Segment(0.5, {YAW_ACTUATOR: -HAPPY_AMPLITUDE, PITCH_ACTUATOR: -HAPPY_AMPLITUDE}),
            Segment(0.6, {})]


def angry_reaction():
    return [Segment(0.3, {}),
            Segment(0.8, {HEAD_ACTUATOR: ANGRY_TILT_AMPLITUDE, PITCH_ACTUATOR: 0.06}, brightness=ANGRY_BRIGHTNESS),
            Segment(0.5, {HEAD_ACTUATOR: ANGRY_TILT_AMPLITUDE, PITCH_ACTUATOR: 0.06}, brightness=ANGRY_BRIGHTNESS),
            Segment(0.8, {})]


REACTIONS = {"FACE": face_reaction, "PEACE": peace_reaction, "PHONE": phone_reaction,
             "HAPPY": happy_reaction, "ANGRY": angry_reaction, "HAND": hand_reaction}
for _direction in ("LEFT", "RIGHT", "UP", "DOWN"):
    REACTIONS["POINT_"+_direction] = lambda direction=_direction: point_reaction(direction)


class BehaviorManager:
    def __init__(self, controller):
        self.controller = controller
        self.state = "IDLE"
        self.sequence = []
        self.index = 0

    def start(self, name, now):
        self.state = name
        self.sequence = REACTIONS[name]()
        self.index = 0
        self.segment_start = now
        self.start_pose = self.controller.data.ctrl.copy()
        self.target = self.controller.pose(self.sequence[0].pose)
        print(f"Starting {name} reaction", flush=True)

    def update(self, now):
        if self.state == "IDLE":
            return
        while now - self.segment_start >= self.sequence[self.index].duration:
            self.segment_start += self.sequence[self.index].duration
            self.start_pose = self.target.copy()
            self.index += 1
            if self.index == len(self.sequence):
                self.controller.data.ctrl[:] = self.controller.neutral
                self.controller.set_light(True)
                self.controller.set_reaction_brightness(1.0)
                print(f"Finished {self.state}; neutral targets and lamp {'ON' if self.controller.lamp.enabled else 'OFF'}", flush=True)
                self.state = "IDLE"
                return
            self.target = self.controller.pose(self.sequence[self.index].pose)
        segment = self.sequence[self.index]
        t = max(0.0, min(1.0, (now-self.segment_start)/segment.duration))
        blend = (1-math.cos(math.pi*t))/2
        self.controller.data.ctrl[:] = self.start_pose + blend*(self.target-self.start_pose)
        self.controller.set_light(segment.light)
        self.controller.set_reaction_brightness(segment.brightness)
