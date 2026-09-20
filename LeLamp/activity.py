"""Wall-clock brightness envelope; only discrete interactions reset inactivity."""
import math
import time

IDLE_TIMEOUT = 10.0
IDLE_BRIGHTNESS = 0.20
DIM_SECONDS = 2.0
WAKE_SECONDS = 0.6


class ActivityLight:
    def __init__(self, now=None):
        self.last_activity = time.monotonic() if now is None else now
        self.started = self.last_activity
        self.value = self.source = self.target = 1.0
        self.duration = WAKE_SECONDS

    def _sample(self, now):
        t = min(1.0, max(0.0, (now-self.started)/self.duration))
        self.value = self.source + (self.target-self.source)*(1-math.cos(math.pi*t))/2
        return self.value

    def interact(self, now):
        self.update(now)
        self.last_activity = now
        if self.target != 1.0:
            self.source, self.target, self.started = self.value, 1.0, now
            self.duration = WAKE_SECONDS
            print("New interaction: smoothly waking lamp", flush=True)

    def update(self, now):
        self._sample(now)
        if now-self.last_activity >= IDLE_TIMEOUT and self.target != IDLE_BRIGHTNESS:
            self.source, self.target = self.value, IDLE_BRIGHTNESS
            self.started, self.duration = now, DIM_SECONDS
            print("No new interaction for 10s: dimming to 20%", flush=True)
        return self.value

    @property
    def awake(self):
        return self.target == 1.0 and self.value >= 1.0-1e-6
