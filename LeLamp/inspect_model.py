"""Print the compiled LeLamp model's names, limits, and actuator mappings."""
import mujoco

from test_sim import load_model


def main():
    model = load_model()
    print(f"MuJoCo {mujoco.__version__}; joint angles and position controls are radians.")
    print("\nJoints:")
    for index in range(model.njnt):
        joint = model.joint(index)
        kind = mujoco.mjtJoint(int(model.jnt_type[index])).name.removeprefix("mjJNT_")
        limits = model.jnt_range[index].tolist() if model.jnt_limited[index] else "unlimited"
        print(f"  {joint.name!r}: type={kind}, range={limits}")
    print("\nActuators:")
    for index in range(model.nu):
        actuator = model.actuator(index)
        transmission = mujoco.mjtTrn(int(model.actuator_trntype[index]))
        target = ""
        if transmission == mujoco.mjtTrn.mjTRN_JOINT:
            target = f", joint={model.joint(int(model.actuator_trnid[index, 0])).name!r}"
        limits = model.actuator_ctrlrange[index].tolist() if model.actuator_ctrllimited[index] else "unlimited"
        print(f"  {actuator.name!r}: control range={limits}{target}")
    print("\nSensors:")
    for index in range(model.nsensor):
        kind = mujoco.mjtSensor(int(model.sensor_type[index])).name.removeprefix("mjSENS_")
        print(f"  {model.sensor(index).name!r}: type={kind}, dimension={model.sensor_dim[index]}")


if __name__ == "__main__":
    main()
