"""Verify unchanged physics, runtime light control, attachment, and rendered light."""
from pathlib import Path
import struct
import zlib
import mujoco
import numpy as np
from test_sim import load_model
from lamp_light import LampLight, load_lit_model


def save_png(path, pixels):
    def chunk(kind, data):
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))
    height, width, _ = pixels.shape
    raw = b"".join(b"\0" + row.tobytes() for row in pixels)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def main():
    original, model = load_model(), load_lit_model()
    fields = [name for name in dir(original)
              if name.startswith(("body_", "jnt_", "geom_", "mesh_", "actuator_", "dof_", "sensor_", "mat_"))
              and isinstance(getattr(original, name), np.ndarray)]
    for field in fields:
        np.testing.assert_array_equal(getattr(original, field), getattr(model, field), err_msg=field)
    assert original.opt.timestep == model.opt.timestep
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lamp = LampLight(model)
    lamp.set_lamp_brightness(2)
    assert lamp.brightness == 1
    lamp.set_lamp_color(-1, 0.5, 2)
    np.testing.assert_allclose(model.light_diffuse[lamp.light_id], lamp.full_diffuse * [0, 0.5, 1])
    lamp.set_lamp_brightness(0.5)
    np.testing.assert_allclose(model.light_diffuse[lamp.light_id], lamp.full_diffuse * [0, 0.25, 0.5])
    lamp.set_lamp_color(1, 1, 1)
    lamp.set_lamp_brightness(1)
    # Rotation of the actual head joint must carry both light position/direction.
    before = data.light_xdir[lamp.light_id].copy()
    data.qpos[model.jnt_qposadr[model.joint("5").id]] += 0.1
    mujoco.mj_forward(model, data)
    assert np.linalg.norm(data.light_xdir[lamp.light_id] - before) > 0.01
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    model.vis.headlight.diffuse[:] = 0.15
    model.vis.headlight.ambient[:] = 0.12
    model.light_diffuse[0] *= 0.15
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = [-0.15, 0.1, 0.13]
    camera.distance = 1.05
    camera.azimuth = 145
    camera.elevation = -25
    root = Path(__file__).resolve().parent
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        lamp.lamp_off()
        renderer.update_scene(data, camera=camera)
        off = renderer.render().copy()
        lamp.lamp_on()
        renderer.update_scene(data, camera=camera)
        on = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=camera)
        segmentation = renderer.render().copy()
    floor = (segmentation[:, :, 0] == model.geom("floor").id) & (segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
    difference = on.astype(float).mean(2) - off.astype(float).mean(2)
    lit_floor = int(np.count_nonzero(floor & (difference > 15)))
    assert lit_floor > 100, f"Spotlight did not visibly illuminate floor: {lit_floor} pixels"
    save_png(root / "lighting_off.png", off)
    save_png(root / "lighting_on.png", on)
    print(f"PASS: {len(fields)} physics/material arrays unchanged; attachment, brightness/color clamps passed.")
    print(f"PASS: {lit_floor} floor pixels brightened by >15/255; actual scene illumination verified.")


if __name__ == "__main__":
    main()
