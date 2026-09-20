"""Look around using a yaw-named actuator or an explicitly selected actuator name."""
import argparse
import math
import time

import mujoco
import mujoco.viewer

from joint_test import check_state, prepare
from test_sim import load_model


def clamp_command(model, actuator_id, target):
    """Clamp a finite command to the actual, enabled actuator control range."""
    if not math.isfinite(target):
        raise ValueError("Control targets must be finite.")
    if not model.actuator_ctrllimited[actuator_id]:
        raise ValueError("An explicit actuator control range is required.")
    low, high = model.actuator_ctrlrange[actuator_id]
    return float(max(low, min(high, target)))


def print_actuators(model, data):
    print("Available actuators:", flush=True)
    for index in range(model.nu):
        print(f"  index={index}, name={model.actuator(index).name!r}, "
              f"ctrlrange={model.actuator_ctrlrange[index].tolist()}, "
              f"limited={bool(model.actuator_ctrllimited[index])}", flush=True)
    print("Joint mappings at the reference pose:", flush=True)
    for index in range(model.nu):
        if model.actuator_trntype[index] == mujoco.mjtTrn.mjTRN_JOINT:
            joint_id = int(model.actuator_trnid[index, 0])
            print(f"  {model.actuator(index).name!r} -> joint {model.joint(joint_id).name!r}, "
                  f"world axis={data.xaxis[joint_id].round(6).tolist()}, "
                  f"joint range={model.jnt_range[joint_id].tolist()}", flush=True)


def explain_missing_yaw(model, data):
    print("No actuator name contains 'yaw'. Stopped safely without stepping physics or commanding motion.")
    candidates = []
    for index in range(model.nu):
        if model.actuator_trntype[index] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        joint_id = int(model.actuator_trnid[index, 0])
        if (model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
                and abs(data.xaxis[joint_id, 2]) > 0.95):
            candidates.append((index, joint_id))
    for index, joint_id in candidates:
        body_id = int(model.jnt_bodyid[joint_id])
        print(f"Likely yaw candidate: actuator {model.actuator(index).name!r}, "
              f"joint {model.joint(joint_id).name!r}, body {model.body(body_id).name!r}.")
        print(f"  Its hinge axis is nearly vertical; anchor={data.xanchor[joint_id].round(6).tolist()} m.")
        print(f"  Control range: {model.actuator_ctrlrange[index].tolist()}.")
    print("The inspected LeLamp XML defines position servos with unit hinge gear: targets are radians.")
    print("Candidate geometry does not establish visual left/right sign. No numeric-name fallback was selected.")


class LookAround:
    def __init__(self, model, data, viewer, neutral):
        self.model, self.data, self.viewer = model, data, viewer
        self.commands = neutral.copy()

    def step(self):
        if not self.viewer.is_running():
            raise InterruptedError("Viewer closed; behavior stopped.")
        tick = time.perf_counter()
        with self.viewer.lock():
            self.data.ctrl[:] = self.commands
            check_state(self.model, self.data)
            mujoco.mj_step(self.model, self.data)
            check_state(self.model, self.data)
        self.viewer.sync()
        time.sleep(max(0, self.model.opt.timestep - (time.perf_counter() - tick)))

    def move_actuator(self, actuator_id, target, duration):
        """Cosine-eased position target, with physics and real-time rendering."""
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Duration must be positive and finite.")
        target = clamp_command(self.model, actuator_id, target)
        joint_id = int(self.model.actuator_trnid[actuator_id, 0])
        low, high = self.model.jnt_range[joint_id]
        target = float(max(low, min(high, target)))
        with self.viewer.lock():
            start = float(self.data.ctrl[actuator_id])
        steps = math.ceil(duration / self.model.opt.timestep)
        for index in range(steps):
            blend = (1 - math.cos(math.pi * (index + 1) / steps)) / 2
            self.commands[actuator_id] = clamp_command(
                self.model, actuator_id, start + blend * (target - start)
            )
            self.step()

    def wait(self, duration):
        for _ in range(math.ceil(duration / self.model.opt.timestep)):
            self.step()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actuator", help="Explicit model actuator name, e.g. 1 for LeLamp base yaw.")
    parser.add_argument("--wiggle", action="store_true", help="Four quick, smooth side-to-side wiggles, then center.")
    args = parser.parse_args()
    # Loads simulation/scene.xml through the existing Windows Unicode asset fix.
    model = load_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # Compute axes; does not advance simulation time.
    print_actuators(model, data)
    yaw_names = [model.actuator(i).name for i in range(model.nu)
                 if "yaw" in model.actuator(i).name.lower()]
    if args.actuator is not None:
        names = [model.actuator(i).name for i in range(model.nu)]
        if args.actuator not in names:
            parser.error(f"Unknown actuator name {args.actuator!r}; available names: {names}")
        yaw_names = [args.actuator]
        print(f"Explicit actuator selection: {args.actuator!r}", flush=True)
    if not yaw_names:
        explain_missing_yaw(model, data)
        return
    if len(yaw_names) != 1:
        print(f"Ambiguous yaw actuators: {yaw_names}. Stopped without commanding motion.")
        return
    actuator_id = model.actuator(yaw_names[0]).id
    # Validate direct, unit-gear position servos, explicit limits, and neutral poses.
    neutral, _ = prepare(model, data)
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    print(f"Using actuator: {yaw_names[0]!r}; joint {model.joint(joint_id).name!r}; units: radians.")
    if args.wiggle:
        print("Starting wiggle: four cycles at +/-0.12 rad (~6.9 degrees), 0.35 s per sweep.", flush=True)
        sequence = [("Center", 0.0, 0.6, 0.5)]
        for cycle in range(1, 5):
            sequence.extend([(f"Wiggle {cycle}: positive", 0.12, 0.35, 0.0),
                             (f"Wiggle {cycle}: negative", -0.12, 0.35, 0.0)])
        sequence.append(("Return to center", 0.0, 0.6, 0.5))
    else:
        print("Starting look-around behavior: 0 -> +0.15 -> -0.15 -> 0 rad (about +/-8.6 degrees).", flush=True)
        sequence = [(label, target, 3.0, 0.5) for label, target in
                    (("Center", 0.0), ("Look left (positive target)", 0.15),
                     ("Look right (negative target)", -0.15), ("Return to center", 0.0))]
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            behavior = LookAround(model, data, viewer, neutral)
            for label, target, duration, pause in sequence:
                print(f"{label}: requested target {target:+.3f} rad", flush=True)
                behavior.move_actuator(actuator_id, target, duration)
                behavior.wait(pause)
                with viewer.lock():
                    position = float(data.qpos[model.jnt_qposadr[joint_id]])
                print(f"  Measured joint position: {position:+.5f} rad "
                      f"({math.degrees(position):+.2f} degrees); limit checks passed.", flush=True)
            print("Behavior finished. Holding center; close the viewer to exit.", flush=True)
            while viewer.is_running():
                behavior.step()
    except (InterruptedError, KeyboardInterrupt):
        print("Behavior stopped.")


if __name__ == "__main__":
    main()
