"""Confirmation, disappearance latch, cooldown, and strict detection priority."""
FACE_COOLDOWN = 4.0
PEACE_COOLDOWN = 4.0
PHONE_COOLDOWN = 5.0
FACE_CONFIRM_FRAMES = 3
PEACE_CONFIRM_FRAMES = 3
PHONE_CONFIRM_FRAMES = 2  # Actual YOLO runs; skipped frames do not count.
EXPRESSION_CONFIRM_FRAMES = 4
ANGRY_CONFIRM_FRAMES = 3
HAPPY_COOLDOWN = 5.0
ANGRY_COOLDOWN = 5.0
LEAVE_FRAMES = 5
POINT_EVENTS = ("POINT_LEFT", "POINT_RIGHT", "POINT_UP", "POINT_DOWN")
PRIORITY = {"IDLE": 0, "HAPPY": 1, "ANGRY": 1, "FACE": 2, "HAND": 3, "PEACE": 4, "PHONE": 5}
CONFIRM = {"FACE": FACE_CONFIRM_FRAMES, "PEACE": PEACE_CONFIRM_FRAMES, "PHONE": PHONE_CONFIRM_FRAMES,
           "HAPPY": EXPRESSION_CONFIRM_FRAMES, "ANGRY": ANGRY_CONFIRM_FRAMES, "HAND": 3}
COOLDOWN = {"FACE": FACE_COOLDOWN, "PEACE": PEACE_COOLDOWN, "PHONE": PHONE_COOLDOWN,
            "HAPPY": HAPPY_COOLDOWN, "ANGRY": ANGRY_COOLDOWN, "HAND": 4.0}
PRIORITY.update({name: 3.5 for name in POINT_EVENTS})
CONFIRM.update({name: 3 for name in POINT_EVENTS})
COOLDOWN.update({name: 1.0 for name in POINT_EVENTS})
VISUAL_ORDER = ("PHONE", "PEACE", *POINT_EVENTS, "HAND", "FACE")


class DetectionEvents:
    def __init__(self):
        self.hits = {name: 0 for name in CONFIRM}
        self.misses = self.hits.copy()
        self.present = {name: False for name in CONFIRM}
        self.latched = self.present.copy()
        self.last_trigger = {name: float("-inf") for name in CONFIRM}

    def observe(self, detections):
        for name, value in detections.items():
            if value is None:
                continue
            if value:
                self.hits[name] += 1
                self.misses[name] = 0
                if self.hits[name] >= CONFIRM[name]:
                    self.present[name] = True
            else:
                self.hits[name] = 0
                self.misses[name] += 1
                if self.misses[name] >= LEAVE_FRAMES:
                    self.present[name] = False
                    self.latched[name] = False

    @property
    def detected(self):
        return next(iter(self.visible), "IDLE")

    @property
    def visible(self):
        return [name for name in VISUAL_ORDER if self.present[name]]

    @property
    def candidate(self):
        # Fresh positive evidence blocks lower classes while confirmation is
        # pending. A skipped YOLO inference retains that evidence.
        return next((name for name in VISUAL_ORDER
                     if self.present[name] or self.hits[name] > 0), "IDLE")

    def trigger(self, now, active="IDLE"):
        name = self.candidate
        if name == "FACE" and self.latched["FACE"]:
            # A greeted face may express something new. Its presence must not
            # permanently block expressions as phone/peace presence intentionally does.
            if self.hits["FACE"] == 0:
                return None
            name = next((kind for kind in ("HAPPY", "ANGRY")
                         if self.hits[kind] >= CONFIRM[kind] and not self.latched[kind]), "IDLE")
        if name == "IDLE" or self.hits[name] < CONFIRM[name]:
            return None
        if self.latched[name] or now-self.last_trigger[name] < COOLDOWN[name]:
            return None  # A present high-priority object also suppresses lower reactions.
        if PRIORITY[name] <= PRIORITY[active]:
            return None
        self.latched[name] = True
        self.last_trigger[name] = now
        return name

    def blocks_pending(self, name):
        candidate = self.candidate
        # A greeted face is the context for an expression, not a new greeting.
        # Do not re-apply raw FACE priority after trigger() has accepted it.
        if candidate == "FACE" and self.latched["FACE"] and name in ("HAPPY", "ANGRY"):
            return False
        return PRIORITY[candidate] > PRIORITY[name]
