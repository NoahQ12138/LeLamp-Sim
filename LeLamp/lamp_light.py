"""Runtime control of the derived scene's actual spotlight (no material changes)."""
from contextlib import nullcontext
from pathlib import Path
import math
import mujoco
import numpy as np


def load_lit_model():
    root = Path(__file__).resolve().parent / "simulation"
    assets = {p.name: p.read_bytes() for p in (root / "assets").glob("*.stl")}
    return mujoco.MjModel.from_xml_string(
        (root / "scene_with_light.xml").read_text(encoding="utf-8"), assets=assets
    )


def unit_value(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Light values must be finite.")
    return max(0.0, min(1.0, value))


class LampLight:
    def __init__(self, model, viewer=None):
        self.model, self.viewer = model, viewer
        self.light_id = model.light("lamp_light").id
        for field in ("light_active", "light_diffuse", "light_specular"):
            if not hasattr(model, field):
                raise RuntimeError(f"Installed MuJoCo lacks {field}")
        self.brightness = 1.0
        self.full_diffuse = model.light_diffuse[self.light_id].copy()
        self.full_specular = model.light_specular[self.light_id].copy()
        self.color = np.ones(3)
        self.enabled = True
        self._apply()

    def _apply(self):
        with self.viewer.lock() if self.viewer else nullcontext():
            self.model.light_active[self.light_id] = self.enabled and self.brightness > 0
            self.model.light_diffuse[self.light_id] = self.full_diffuse * self.color * self.brightness
            self.model.light_specular[self.light_id] = self.full_specular * self.color * self.brightness
        if self.viewer:
            self.viewer.sync()  # Full sync propagates runtime light changes.

    def lamp_on(self):
        self.enabled = True
        self._apply()

    def lamp_off(self):
        self.enabled = False
        self._apply()

    def set_lamp_brightness(self, value):
        self.brightness = unit_value(value)
        self._apply()

    def set_lamp_color(self, r, g, b):
        self.color = np.array([unit_value(v) for v in (r, g, b)])
        self._apply()
