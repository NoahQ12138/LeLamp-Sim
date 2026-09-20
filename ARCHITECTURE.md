# LeLamp-Sim code and simulation architecture

This repository contains the Human Computer Lab LeLamp model and the Windows
simulation application under [`LeLamp/`](LeLamp/). The app turns webcam detections
and microphone commands into smooth MuJoCo control inputs, illumination changes,
and speech. See [ACTIONS_AND_STATES.txt](ACTIONS_AND_STATES.txt) for the full demo
reference and [INTERACTIONS.md](LeLamp/INTERACTIONS.md) for tuning details.

## Start the application

From the repository root in PowerShell:

```powershell
cd LeLamp
.\.venv\Scripts\python.exe vision_behavior.py
```

The virtual environment and downloaded weights are intentionally not in Git.
For a fresh checkout, install Python, create `LeLamp/.venv`, install MuJoCo and
`LeLamp/requirements-vision.txt`, and download the weights described in
[VISION_SETUP.md](LeLamp/VISION_SETUP.md) and [INTERACTIONS.md](LeLamp/INTERACTIONS.md).
Model hashes are recorded in `LeLamp/models/manifest.json`. The existing setup
was tested with Python 3.12 and MuJoCo 3.13.0.

## Code map

| File in `LeLamp/` | Responsibility |
| --- | --- |
| `vision_behavior.py` | Entry point, viewer, event dispatch, wake-before-action, voice consumption, overlay, cleanup |
| `webcam.py` | Shared DirectShow camera startup and resolution configuration |
| `vision.py` | Camera worker; MediaPipe face/hands, YOLO phone, expression integration; publishes bounded result queue |
| `gestures.py` | Peace and directional-pointing geometry from 21 hand landmarks |
| `expressions.py` | OpenCV DNN FER+ inference on largest visible face crop; HAPPY/ANGRY/UNKNOWN |
| `detection_events.py` | Confirmation counts, disappearance latches, cooldowns, priority and pending-action guard |
| `behaviors.py` | Time-segmented pose/light sequences and nonblocking cosine interpolation |
| `lelamp_controller.py` | Model/data ownership, named actuator mapping, control/joint clamping, physics checks, manual-off policy |
| `activity.py` | Monotonic inactivity timer and smooth idle/wake brightness envelope |
| `lamp_light.py` | Derived model loader and actual MuJoCo spotlight state/intensity/color |
| `speech.py` | Windows SAPI speech queue, cancellable speech, Vosk/sounddevice microphone worker |
| `jokes.py` | Background download of online jokes, random selection and local fallback |
| `joint_test.py` | Servo/limit validation, neutral initialization and simulation safety checks |
| `behavior_test.py` | Standalone earlier movement demo and reusable command clamping |
| `test_sim.py` | Original unlit scene loader and minimal viewer launch |
| `build_light_scene.py` | Builds the separate lighting scene without altering original XML |
| `inspect_model.py`, `inspect_lighting.py` | Metadata diagnostics |
| `verify_*.py` | Regression tests and hardware/audio diagnostics |

The upstream `3D/` and `docs/` folders contain hardware and assembly material;
they are not imported by the simulation application. Upstream attribution and
license remain in [`LeLamp/README.md`](LeLamp/README.md) and
[`LeLamp/LICENSE`](LeLamp/LICENSE).

## How information flows

```text
Webcam -> VisionWorker -> bounded frame/detection queue
                              |
                              v
                    DetectionEvents (main thread)
                              |
Microphone -> Vosk -> command queue -> main loop
                              |
                    activity / priority / pending action
                              |
                       start_reaction()
                         /           \
                 BehaviorManager     Speaker queue -> SAPI -> laptop speakers
                         |                         ^
                 LeLampController             cached random joke
                    /          \                   ^
              data.ctrl     LampLight       background JokeBank download
                    \          /
                     MuJoCo step -> viewer.sync()
```

Only the main thread mutates MuJoCo model/data and lighting. It uses the passive
viewer's lock when updating state, steps physics, then synchronizes rendering.
Vision, microphone recognition, speech and joke fetching have separate workers.
Slow recognition, speaking, or network access does not run inside the physics loop.
Main-loop sleep targets the model timestep, though actual real-time speed depends
on rendering and available CPU.

## From a detection to movement

1. `Vision.process()` draws boxes/landmarks and returns boolean detections.
   Skipped phone/expression inferences return `None`, not a fabricated detection.
2. `DetectionEvents.observe()` counts fresh evidence. `trigger()` chooses one
   eligible event and latches it. Phone > peace > pointing > generic hand > new
   face > expression. Early phone evidence also blocks lower-priority dispatch.
3. A new event resets activity. When dimmed, the event waits for the 0.6-second
   brightness ramp. Higher-priority events can replace the pending action.
   A face already greeted does not block the expression associated with it.
4. `start_reaction()` queues any speech and starts `BehaviorManager` at simulation
   time `data.time`. Phone speech invalidates queued/current lower-priority speech.
5. Each `Segment` defines duration, target pose, on/off flag and brightness factor.
   The manager interpolates from current control values using cosine easing.
   Preemption begins at current controls to avoid an instantaneous command jump.
6. `LeLampController.pose()` resolves actuator names, intersects actual limits,
   and keeps a five-degree joint buffer. `step()` checks state before and after
   `mujoco.mj_step()`. Completion returns targets to neutral.

## Connection to the MuJoCo model

`LeLampController` calls `lamp_light.load_lit_model()` to construct an `MjModel`
from `simulation/scene_with_light.xml`, then creates `MjData(model)` for dynamic
state. The lit scene is a separate derivative of the original robot scene.
`test_sim.py` instead loads the original `simulation/scene.xml` and XML includes.
Python reads mesh/XML bytes into MuJoCo's virtual filesystem to handle Unicode
mesh filenames on Windows; model paths resolve relative to the Python files.

`joint_test.prepare()` verifies the position-servo transmission setup and finds
neutral controls. Behavior code uses these model actuator names:

| Purpose | Actuator name | Joint name | Control/joint range, radians |
| --- | --- | --- | --- |
| Left/right yaw | `1` | `1` | approximately -5.021 to 1.262 |
| Up/down pitch | `3` | `3` | approximately -2.820 to 0.322 |
| Head twist | `5` | `5` | approximately -0.853 to 2.288 |

These are names resolved via MuJoCo metadata, not hard-coded array indices.
Positive pitch moves the beam downward. All other actuator targets start from
the validated neutral pose. Use `inspect_model.py` for the complete model listing.

`data.ctrl` holds desired servo positions; physics computes actual `qpos/qvel`.
`LampLight` modifies the named spotlight's `light_active`, `light_diffuse` and
`light_specular`, so brightness affects actual rendered illumination. It does
not merely recolor the mesh. The light follows the head/diffuser body.

## State separation and speech

Motion state, pending event, detector latch, manual-off flag, idle envelope and
reaction brightness are independent. Effective brightness is the idle envelope
times the reaction brightness. Voice OFF takes precedence over all attempts to
enable the light; voice ON clears that flag. Continuous face presence is not
activity. Fresh accepted events and voice commands reset the ten-second timer.

SAPI is owned by one COM-initialized worker. Asynchronous Speak plus short
worker-only waits allow interruption; external joke text is spoken as literal
text. The microphone suppresses recognition during speech and briefly afterward.
Only finalized exact ON/OFF phrases with sufficient word confidence become events.

Jokes come from the Official Joke API via a generic request containing no camera,
microphone, or expression data. A background thread fetches ten; selection avoids
an immediate repeat. Failed/unfinished downloads use local jokes. Vision and
speech recognition run locally, and captured audio/video are not saved/uploaded.

## Tests and limitations

```powershell
.\.venv\Scripts\python.exe -m unittest verify_gestures verify_vision verify_interactions verify_phone verify_jokes verify_pointing -v
.\.venv\Scripts\python.exe vision_behavior.py --manual point_up
.\.venv\Scripts\python.exe vision_behavior.py --manual angry
```

The 30-test regression suite covers geometry, confirmation/rearming, priority,
expression wake dispatch, joke handling and model physics/control checks.
Manual and webcam tests are separate: automated passes do not establish webcam
accuracy. FER+ labels visible expressions, not internal emotions; sensitive anger
thresholds can produce false positives. Hand pointing is a geometric heuristic,
and camera orientation/occlusion can affect it. Actual viewer orientation also
changes how model yaw appears on screen.

Normal shutdown returns toward neutral, disables the spotlight and closes workers
and windows. Camera/model/physics failures are reported explicitly; physics errors
stop further stepping. A camera driver blocked inside a read can delay release.
