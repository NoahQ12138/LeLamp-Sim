# Webcam-driven LeLamp reactions

Expression recognition, offline speech, voice commands, and idle dimming are now
included. See [INTERACTIONS.md](INTERACTIONS.md) for their settings and test status.

Run in PowerShell, from any initial directory:

```powershell
cd C:\Users\noahq\OneDrive\Documents\HTN\LeLamp-Sim\LeLamp
.\.venv\Scripts\python.exe vision_behavior.py
```

Or activate `.\.venv\Scripts\Activate.ps1`, then run `python vision_behavior.py`.
Keep both the MuJoCo viewer and **LeLamp Vision** window visible. Press **q** in
the camera window to exit. Closing either window also exits. `webcam.py` now uses
Windows DirectShow: camera opening took 0.46-0.58 seconds in the repair tests,
instead of roughly 50 seconds with the previous default backend. The webcam window
shows startup status immediately, then a Camera LIVE frame counter. Close other
camera apps and check Windows desktop-app camera permissions if opening fails.
For a different camera, edit `CAMERA_INDEX` in `webcam.py`.
Frames are processed locally and are not saved or uploaded.

## Models and detection tuning

`vision.py` contains detector constants:

| Detection | Implementation | Threshold |
| --- | --- | --- |
| Face | MediaPipe Tasks FaceDetector, BlazeFace short-range | 0.50 |
| Peace sign | MediaPipe HandLandmarker + world-landmark finger geometry, up to two hands | detection/presence/tracking all 0.50 |
| Phone | Ultralytics YOLO11n, pretrained COCO weights | `PHONE_CONFIDENCE = 0.50` |

The installed MediaPipe uses the Tasks API rather than the old `mp.solutions.hands`
API. The hand model provides landmarks; `gestures.py` identifies a peace sign:
index and middle extended and separated, ring and pinky curled. Thumb position is
unrestricted. Other visible hands now trigger a small generic-hand wiggle below
phone and peace priority.
The phone class is **67**, resolved from the model's `"cell phone"` label at runtime.
YOLO runs on CPU at input size 416, every second processed camera frame. Cached
phone boxes can be drawn between inferences, but only fresh YOLO results count
toward confirmation or disappearance. All weights live under `models/`.

Packages were installed only after checking `.venv`: OpenCV 5.0.0.93, MediaPipe
1.0.1, and Ultralytics 8.4.155. MediaPipe also requires the OpenCV contrib wheel.
MuJoCo remains 3.13.0. `requirements-vision.txt` records the tested direct packages;
`models/manifest.json` records SHA-256 hashes of the downloaded models.

## Reactions and motion tuning

`behaviors.py` contains all amplitudes, cycle counts, and durations. Values are
**radians**, not degrees. Smooth cosine interpolation runs per physics step.

| Reaction | Actuator name -> joint | Amplitude | Motion | Light |
| --- | --- | --- | --- | --- |
| Face | `"3"` -> `"3"` | +/-0.12 rad (~6.9 degrees), final down-nod +0.06 | down / up / slight down / neutral; 0.55 s per segment | alternates with the 0.55 s nod segments |
| Peace sign | yaw `"1"`, pitch `"3"`, head `"5"` | yaw/head +/-0.12 rad; pitch +/-0.10 rad | left / right / up / down / head twist both ways / neutral; 0.40 s per beat | stays on |
| Phone | `"1"` -> `"1"` | +/-0.18 rad (~10.3 degrees) | 4 left/right cycles, 0.25 s per sweep | alternates every sweep, 0.25 s after initial centering |

Yaw actuator `"1"` has joint/control range `[-5.021034, 1.262151]`.
Pitch actuator `"3"` has joint/control range `[-2.820024, 0.321569]`. Model
inspection showed a +0.1-radian pitch change tilts the light direction downward
(world Z component changes from -0.508 to -0.582). Both are direct position servos.
The controller resolves names through model metadata, validates actuator types,
clamps against actual control and joint limits, and retains a five-degree buffer.
It also checks actual joint positions, finite state, and MuJoCo warnings each step.

Phone motion remains the faster alarm shake; the peace-sign dance uses three axes. Head actuator `"5"` drives the actual diffuser body, with range `[-0.853337, 2.288255]` rad. Dance targets stay small and finish at neutral.
Tuning occurs in `behaviors.py`; keep changes small and rerun manual tests.
The lamp starts on and stays on while idle or dancing. Face and phone reactions
still flash, then finish with neutral targets and the lamp on.

## Priority and repeated detections

`detection_events.py` contains confirmation and cooldown settings:

| Detection | Confirmation | Cooldown |
| --- | --- | --- |
| Face | 3 consecutive processed frames | 4 seconds |
| Peace sign | 3 consecutive processed frames | 4 seconds |
| Phone | 2 consecutive actual YOLO runs | 5 seconds |

**PHONE > PEACE > HAND > NEW FACE > HAPPY/ANGRY > IDLE.** A higher-priority reaction preempts a lower-priority
reaction, interpolating from its current control targets through neutral. Thus a
phone can interrupt either face or dance, and a peace sign can interrupt face.
A confirmed high-priority object continues to suppress lower-priority reactions
while it remains present, even when its own reaction is latched or on cooldown.

After a trigger, that class must be absent for **5 fresh detector results** before
it can rearm; its cooldown must also have expired. A continuously visible object
does not repeat forever. Phone skipped frames do not count as disappearance.
New reactions are suppressed when camera results are over one second old.

## Architecture and reuse

`vision.py`: detector classes and a camera/inference worker with a bounded result
queue. The worker never accesses MuJoCo. `detection_events.py`: event filtering.
`behaviors.py`: non-blocking motion/light sequences. `lelamp_controller.py`:
model mappings and safety. `vision_behavior.py`: main simulation loop, webcam UI,
priority dispatch, and cleanup. MuJoCo and the viewer have a single writer.
Inference and animations run independently, so an animation does not freeze video.

Reused unchanged: `lamp_light.load_lit_model()`, `LampLight`,
`joint_test.prepare()`, `joint_test.check_state()`, and
`behavior_test.clamp_command()`. The new state machine uses the same cosine easing
as the existing smooth helper, without its blocking loop. The scene remains
`simulation/scene_with_light.xml`; no XML modifications were needed.

Normal exit gently returns targets toward neutral, switches the lamp off, stops
the vision worker, releases the camera, and closes the windows. If a physics error
occurs, it stops stepping instead of attempting further animation. A camera driver
stuck inside a read can delay resource release; the worker reports a slow shutdown.

## Individual test commands

```powershell
.\.venv\Scripts\python.exe check_webcam.py
.\.venv\Scripts\python.exe vision.py --stage face
.\.venv\Scripts\python.exe vision.py --stage peace
.\.venv\Scripts\python.exe vision.py --stage phone
.\.venv\Scripts\python.exe vision_behavior.py --manual face
.\.venv\Scripts\python.exe vision_behavior.py --manual peace
.\.venv\Scripts\python.exe vision_behavior.py --manual phone
.\.venv\Scripts\python.exe vision_behavior.py --stage face --seconds 15
.\.venv\Scripts\python.exe vision_behavior.py --stage peace --seconds 20
.\.venv\Scripts\python.exe vision_behavior.py --stage phone --seconds 25
```

Stage peace enables face and hand/peace detection; stage phone/all also enables
phone detection. `--stage hand` now tests generic hands, and `--manual hand` runs
the generic-hand wiggle. Use `--manual peace` for the dance.
Manual tests open only MuJoCo and finish automatically. A timed live stage raises
an error if its requested reaction never triggers; hold the relevant object in view.

## Validation and tuning

The original face/phone detector and reaction tests passed on the laptop webcam.
For the peace-sign update, geometry tests cover both handedness orientations,
rotation/scale, open palms, fists, pointing, and invalid landmarks. The manual
dance passed the physics and joint/control checks. Priority and latch regression
tests now use PEACE in place of HAND.
After repairing camera startup, the full live app recognized a peace sign and
completed the dance successfully (one FACE and one PEACE reaction in 25 seconds).
The standalone camera check delivered 289 frames in ten seconds. Both tests
released the camera and closed their windows afterward.

`gestures.py` contains the finger-angle and V-spread thresholds. It uses 3D world
landmarks, avoiding a simple screen-up/screen-down finger rule. It is still a
heuristic: occlusion or an edge-on hand can cause missed detections. Show a clear V
with ring/little fingers folded toward the camera. An ordinary hand triggers the
generic-hand wiggle instead of the dance, and takes priority over a face greeting.

`behaviors.py`: `DANCE_YAW_AMPLITUDE`, `DANCE_PITCH_AMPLITUDE`,
`DANCE_HEAD_AMPLITUDE`, and `DANCE_BEAT_SECONDS` tune the dance.
`detection_events.py`: `PEACE_CONFIRM_FRAMES` and `PEACE_COOLDOWN` tune triggering.
The webcam status shows PEACE and DANCING. Lower the hand or relax the gesture
for five detector frames to rearm; keep the four-second cooldown in mind.

```powershell
.\.venv\Scripts\python.exe -m unittest verify_gestures verify_vision -v
```

Primary references: [MediaPipe FaceDetector](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector/python),
[MediaPipe HandLandmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python),
[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/).
