# Windows simulation setup

Repository: `C:\Users\noahq\OneDrive\Documents\HTN\LeLamp-Sim\LeLamp`

Open this **LeLamp** folder in VS Code (File > Open Folder). If needed, run
**Python: Select Interpreter** and select `.venv\Scripts\python.exe`.
The workspace settings default to that interpreter.

In a PowerShell terminal:

```powershell
cd C:\Users\noahq\OneDrive\Documents\HTN\LeLamp-Sim\LeLamp
.\.venv\Scripts\Activate.ps1
python --version
where.exe python
python -c "import sys; print(sys.executable)"
python -c "import mujoco; print(mujoco.__version__)"
python test_sim.py
```

If activation is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
and activate again. Activation succeeded during setup without that change.
Activation applies to the current terminal; it does not persist into a new one.
You can also use `.\.venv\Scripts\python.exe` instead of `python` without activation.

Close the viewer, then run either:

```powershell
python inspect_model.py
python joint_test.py
```

`inspect_model.py` prints compiled joint types/ranges, actuator names/control
ranges and joint mappings, and sensors. Hinge angles and position controls are radians.

`joint_test.py` tests actuator names in model order, one target at a time, up to
2 degrees either side of the reference target. It ramps for one second, holds for
half a second, returns to neutral, and pauses before the next movement. Other
actuators retain their neutral targets. Gravity and contacts can still cause
passive motion. The free base joint is unactuated. The script checks actual hinge
positions, control ranges, finite state, and physics warnings on every step,
and stops on a violation. The normal sequence lasts about 34 seconds and closes
its viewer when complete. Closing the viewer early stops the test.

Optional commands:

```powershell
python joint_test.py --actuator 3
python joint_test.py --headless
```

The numeric strings `"1"` through `"5"` are model **names**, not array indices.
The compiled order is `"2", "1", "3", "4", "5"`.
Sensors are `accel_sensor` and `gyro_sensor`.

## Setup performed

From the existing HTN folder:

```powershell
git clone https://github.com/humancomputerlab/LeLamp.git
cd LeLamp
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
where.exe python
python -c "import sys; print(sys.executable); assert sys.prefix != sys.base_prefix"
python -m pip install --upgrade pip
python -m pip install mujoco
python -c "import mujoco; print(mujoco.__version__)"
python test_sim.py
python inspect_model.py
python joint_test.py --headless
python -m pip check
python joint_test.py
```

Installed Python 3.12.5, pip 26.2.1, and MuJoCo 3.13.0. Setup used the explicit
`.venv\Scripts\python.exe` for commands executed in separate tool shells.
Git/network and Python temporary-file access required elevated tool access.

## Windows asset loading fix

The initial `from_xml_path("simulation/scene.xml")` failed opening a Chinese-named
STL. An absolute scene path also failed. Python could read all 30 STL files.
`test_sim.load_model()` therefore reads the original XML and mesh bytes using
Python and supplies them to MuJoCo's virtual filesystem. It resolves paths relative
to the script, so all three scripts work from other working directories too.
No original XML or assets were modified.

MuJoCo viewer API: https://mujoco.readthedocs.io/en/stable/python.html
