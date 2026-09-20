"""Show actual lamp illumination: OFF / ON / OFF / ON, then hold the viewer open."""
import time
import mujoco
import mujoco.viewer
from lamp_light import LampLight, load_lit_model
from joint_test import prepare, check_state


def main():
    model = load_lit_model()
    data = mujoco.MjData(model)
    neutral, _ = prepare(model, data)
    light_id = model.light("lamp_light").id
    print(f"Light {light_id}: lamp_light, attached to {model.body(int(model.light_bodyid[light_id])).name}", flush=True)
    # Dim the existing studio lights only in this demo so the floor spot is clear.
    model.vis.headlight.diffuse[:] = 0.15
    model.vis.headlight.ambient[:] = 0.12
    for index in range(model.nlight):
        if index != light_id:
            model.light_diffuse[index] *= 0.15
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [-0.15, 0.1, 0.13]
            viewer.cam.distance = 1.05
            viewer.cam.azimuth = 145
            viewer.cam.elevation = -25
        lamp = LampLight(model, viewer)
        print("Demo studio lights dimmed; lamp on/off changes actual spotlight illumination.", flush=True)

        def wait(seconds):
            until = time.perf_counter() + seconds
            while viewer.is_running() and time.perf_counter() < until:
                start = time.perf_counter()
                with viewer.lock():
                    data.ctrl[:] = neutral
                    mujoco.mj_step(model, data)
                    check_state(model, data)
                viewer.sync()
                time.sleep(max(0, model.opt.timestep - (time.perf_counter() - start)))
            return viewer.is_running()

        for enabled in (False, True, False, True):
            if not viewer.is_running():
                return
            (lamp.lamp_on if enabled else lamp.lamp_off)()
            print("ON" if enabled else "OFF", flush=True)
            if not wait(1):
                return
        print("Test finished; lamp ON. Close the viewer to exit.", flush=True)
        while viewer.is_running():
            wait(0.1)


if __name__ == "__main__":
    main()
