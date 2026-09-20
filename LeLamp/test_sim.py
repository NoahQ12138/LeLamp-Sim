from pathlib import Path

import mujoco
import mujoco.viewer

def load_model():
    """Load unchanged repository files, including Unicode mesh names on Windows."""
    simulation = Path(__file__).resolve().parent / "simulation"
    # Python reads Unicode paths reliably; MuJoCo receives the original bytes
    # through its virtual filesystem instead of opening mesh paths itself.
    assets = {p.name: p.read_bytes() for p in (simulation / "assets").glob("*.stl")}
    assets.update({p.name: p.read_bytes() for p in simulation.glob("*.xml")})
    return mujoco.MjModel.from_xml_string(
        (simulation / "scene.xml").read_text(encoding="utf-8"), assets=assets
    )


if __name__ == "__main__":
    model = load_model()
    data = mujoco.MjData(model)
    print(f"Loaded LeLamp: {model.njnt} joints, {model.nu} actuators, {model.nsensor} sensors.", flush=True)
    mujoco.viewer.launch(model, data)
