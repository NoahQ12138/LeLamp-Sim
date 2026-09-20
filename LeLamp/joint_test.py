"""Gently exercise named LeLamp position actuators in simulation only."""
import argparse
from contextlib import nullcontext
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from test_sim import load_model


def prepare(model, data):
    neutral = np.zeros(model.nu)
    tests = {}
    for index in range(model.nu):
        name = model.actuator(index).name
        if not name:
            raise ValueError("Every tested actuator must have a name.")
        # Resolve by name, then read the transmission; names are not indices.
        actuator_id = model.actuator(name).id
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if (model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT
                or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE
                or not np.allclose(model.actuator_gear[actuator_id], [1, 0, 0, 0, 0, 0])
                or model.actuator_dyntype[actuator_id] != mujoco.mjtDyn.mjDYN_NONE
                or model.actuator_gaintype[actuator_id] != mujoco.mjtGain.mjGAIN_FIXED
                or model.actuator_biastype[actuator_id] != mujoco.mjtBias.mjBIAS_AFFINE
                or model.actuator_gainprm[actuator_id, 0] <= 0
                or not np.isclose(model.actuator_biasprm[actuator_id, 0], 0)
                or not np.isclose(model.actuator_biasprm[actuator_id, 1],
                                  -model.actuator_gainprm[actuator_id, 0])):
            raise ValueError(f"{name!r}: expected a direct, unit-gear hinge position actuator.")
        if not (model.jnt_limited[joint_id] and model.actuator_ctrllimited[actuator_id]):
            raise ValueError(f"{name!r}: explicit joint and control limits are required.")
        low = max(model.jnt_range[joint_id, 0], model.actuator_ctrlrange[actuator_id, 0])
        high = min(model.jnt_range[joint_id, 1], model.actuator_ctrlrange[actuator_id, 1])
        center = float(model.qpos0[model.jnt_qposadr[joint_id]])
        margin = math.radians(1)
        amplitude = min(math.radians(2), (center - low - margin) / 2,
                        (high - center - margin) / 2)
        if amplitude <= 0:
            raise ValueError(f"{name!r}: reference pose is too close to a limit.")
        neutral[actuator_id] = center
        tests[name] = (actuator_id, center, amplitude)
    data.ctrl[:] = neutral
    mujoco.mj_forward(model, data)
    return neutral, tests


def check_state(model, data):
    if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
        raise RuntimeError("Non-finite simulation state; stopping.")
    if np.any(data.warning.number):
        raise RuntimeError(f"MuJoCo reported a physics warning: {data.warning.number}.")
    for index in range(model.njnt):
        if model.jnt_limited[index] and model.jnt_type[index] == mujoco.mjtJoint.mjJNT_HINGE:
            position = data.qpos[model.jnt_qposadr[index]]
            low, high = model.jnt_range[index]
            # MuJoCo limits are soft constraints; stop if actual motion crosses one.
            if not low <= position <= high:
                raise RuntimeError(f"Joint {model.joint(index).name!r} crossed its limit; stopping.")
    limited = model.actuator_ctrllimited.astype(bool)
    if (np.any(data.ctrl[limited] < model.actuator_ctrlrange[limited, 0])
            or np.any(data.ctrl[limited] > model.actuator_ctrlrange[limited, 1])):
        raise RuntimeError("A control target crossed its range; stopping.")


def run_phase(model, data, viewer, neutral, actuator_id, start, end, seconds):
    steps = math.ceil(seconds / model.opt.timestep)
    for step in range(steps):
        if viewer is not None and not viewer.is_running():
            return False
        tick = time.perf_counter()
        fraction = (step + 1) / steps
        blend = (1 - math.cos(math.pi * fraction)) / 2
        with viewer.lock() if viewer is not None else nullcontext():
            data.ctrl[:] = neutral
            if actuator_id is not None:
                data.ctrl[actuator_id] = start + (end - start) * blend
            check_state(model, data)
            mujoco.mj_step(model, data)
            check_state(model, data)
        if viewer is not None:
            viewer.sync()
            time.sleep(max(0, model.opt.timestep - (time.perf_counter() - tick)))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run the same physics checks without a window.")
    parser.add_argument("--actuator", help="Test only this actuator name (for example: 3).")
    args = parser.parse_args()
    model = load_model()
    data = mujoco.MjData(model)
    neutral, tests = prepare(model, data)
    if args.actuator is not None:
        if args.actuator not in tests:
            parser.error(f"Unknown actuator {args.actuator!r}; choose from {list(tests)}")
        tests = {args.actuator: tests[args.actuator]}
    context = nullcontext(None) if args.headless else mujoco.viewer.launch_passive(model, data)
    with context as viewer:
        print("Settling at reference targets. Close the viewer to stop.", flush=True)
        if not run_phase(model, data, viewer, neutral, None, 0, 0, 2):
            return
        for name, (actuator_id, center, amplitude) in tests.items():
            print(f"Testing actuator {name!r}: +/- {math.degrees(amplitude):.2f} degrees", flush=True)
            for target in (center + amplitude, center - amplitude):
                for start, end, duration in (
                    (center, target, 1), (target, target, 0.5),
                    (target, center, 1), (center, center, 0.5),
                ):
                    if not run_phase(model, data, viewer, neutral, actuator_id, start, end, duration):
                        return
        if not run_phase(model, data, viewer, neutral, None, 0, 0, 2):
            return
        print("Completed: all targets returned to neutral; joint/control limits and physics checks passed.", flush=True)


if __name__ == "__main__":
    main()
