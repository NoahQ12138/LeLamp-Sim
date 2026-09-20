# Expressions, voice, and idle lighting

Launch the complete app in PowerShell:

```powershell
cd C:\Users\noahq\OneDrive\Documents\HTN\LeLamp-Sim\LeLamp
.\.venv\Scripts\python.exe vision_behavior.py
```

Keep the camera and MuJoCo windows visible. Press q in the camera window or close
either window to exit. The light starts on. Original robot XML files are unchanged.

## Pointing directions

Extend your index finger and curl the middle, ring, and little fingers. Point
left/right/up/down as shown in the unmirrored camera preview. Three consecutive
frames confirm the direction. The lamp moves smoothly for 0.55 seconds, holds
for one second, then returns to neutral over 0.6 seconds. Left/right use yaw
actuator `1` at +/-0.15 rad; up/down use pitch actuator `3` at -/+0.10 rad.
All commands use the existing model-based control and joint limits.

Priority is PHONE > PEACE > POINTING > GENERIC HAND > NEW FACE > EXPRESSIONS.
The overlay shows POINT_LEFT, POINT_RIGHT, POINT_UP, or POINT_DOWN. Holding the
same point does not repeat the action; relax or change direction for five frames
to rearm, with a one-second per-direction cooldown. A new direction waits for the
current pointing reaction to finish. Diagonals and fingers aimed toward the lens
do not select a direction. Conflicting directions from two hands are suppressed.

Manual tests: `python vision_behavior.py --manual point_left` (also `point_right`,
`point_up`, `point_down`). All 30 regression tests passed, including pointing
geometry, priority, and all four motions under physics/joint/control checks.
Live webcam pointing accuracy has not yet been verified.

## Expressions and existing reactions

The pretrained ONNX Model Zoo **FER+** classifier runs through existing OpenCV DNN,
using the largest detected face's grayscale crop at 64x64. It classifies visible
expressions, not a person's internal emotional state. HAPPY is its happiness class;
ANGRY uses its anger class (index 4); sadness is no longer a trigger. Other
classes and uncertain results display UNKNOWN. No face identity tracking is performed.

`expressions.py` uses HAPPY confidence 0.65 and winning margin 0.15. ANGRY needs
a score of at least 0.25 and must rank first, or second to neutral with a gap
no greater than 0.15. This can increase false triggers on neutral faces. Inference runs every third camera frame. Confirmation requires
four fresh positives for HAPPY and three for ANGRY. These initial thresholds
have not yet been calibrated against a live angry expression.
Five negative results rearm an expression, with a five-second cooldown as well.
Continuous smiling does not continuously trigger reactions.

HAPPY gives a small 0.06-radian friendly movement. ANGRY reuses the gentle head tilt
and downward movement, with reaction brightness at 65%, then returns to neutral.
The existing face nod/flash, peace-sign dance, and phone shake/flash are preserved.
Priority is PHONE > PEACE > generic HAND > new FACE greeting > HAPPY/ANGRY > IDLE. A held phone or peace
sign suppresses lower-priority reactions. Expressions wait for the greeting to
finish. Generic hands now trigger a small +/-0.08-radian yaw wiggle, while peace
signs keep their dance. A face must disappear for five detection frames and re-enter before
another greeting, subject to the existing four-second cooldown.

## Offline voice

Windows SAPI via **pywin32** provides speech output. One dedicated worker owns the
engine and processes a bounded queue, so messages do not overlap or block physics.
The reusable `speech.speak(text)` function queues speech. A new face queues
"Hello there" after wake-up. Each newly triggered ANGRY reaction also says:
"You seem angry. Let me lighten up your mood with a joke." followed by a random
joke and "Hahaha". This is a scripted response to a visible expression label,
not a claim that the classifier knows the person's internal emotional state.

`jokes.py` prefetches ten general jokes from the
[Official Joke API](https://github.com/15Dkatz/official_joke_api) on a background
thread when speech starts. It selects one randomly per ANGRY event and avoids
an immediate repeat. While loading or if the request fails, it uses local backup
jokes. No camera, microphone, or expression data is sent to the API. There are
no new dependencies. TTS remains local, and phone events interrupt the joke.
The existing ANGRY latch/cooldown prevents continual repetition. Test with
`python vision_behavior.py --manual angry`; the normal launch command is unchanged.

Each new PHONE event instead says **"Stop scrolling, please lock in twin."**
The phone latch and five-second cooldown prevent repetition while held in view.
After wake-up, speech is queued and the shake has a 0.35-second lead-in; flashing
runs during the shake, then targets return to neutral. Phone audio cancels active
and queued greetings through the existing speech worker. Async SAPI and short
worker-only waits keep physics responsive. No disco music playback exists in
this checkout, so there is no music stream to stop.

Phone evidence blocks lower-priority dispatch even while waiting for its second
fresh confirmation. PHONE + FACE + PEACE selects only PHONE. The overlay lists
all confirmed detections separately from the winning priority and active behavior.
When a phone stays visible after its reaction, lower reactions remain suppressed.

**Vosk small English 0.15** recognizes microphone input locally using **sounddevice**.
The laptop Realtek microphone array is preferred; otherwise the OS default is used.
Change `MICROPHONE_DEVICE` in `speech.py` to select another device. The microphone
worker queues commands; only the main thread changes the lamp or MuJoCo state.

| Action | Exact accepted phrases |
| --- | --- |
| ON | on; turn on; lamp on |
| OFF | off; turn off; lamp off |

Recognition must be finalized, with each word confidence at least 0.80. Case and
extra whitespace are normalized. Other phrases and partial recognition are ignored.
The recognizer uses a restricted command grammar plus an unknown-word alternative;
background speech can still be misrecognized. The terminal prints heard phrases.
Input is muted during TTS and briefly afterward to reduce speaker feedback.
Camera frames and microphone audio are processed locally, not saved or uploaded.

OFF sets `manual_lamp_off`: every reaction, flash, and idle update respects it.
Only ON clears it. Motion can still react while the light is manually off.
`TTS_ENABLED` and `VOICE_INPUT_ENABLED` in `speech.py` toggle speech features;
`--no-voice` disables microphone input for a single run.

## Idle and wake-up

`activity.py` uses a monotonic clock. Ten seconds without a newly accepted visual
reaction or voice ON/OFF starts a two-second cosine fade to 20% brightness.
Continuous face presence does not reset the timer. Idle does not switch on a
manually disabled light or run its dimming transition while manually off.

A new accepted interaction resets the timer and fades brightness back to 100%
over 0.6 seconds. Visual reactions and their greeting wait for this fade before
starting. Higher-priority events can replace a pending reaction. When manually
off, visual motion proceeds without switching the light on. Reaction brightness
multiplies the idle brightness envelope. The overlay shows brightness, dimming,
manual OFF, expressions, active behavior, and microphone status.

## Dependencies and model files

The environment already had OpenCV, MediaPipe, Ultralytics, MuJoCo, and sounddevice.
This extension installed Vosk and pywin32. Initial TTS testing also installed
pyttsx3, comtypes, and pypiwin32; the final implementation uses SAPI directly.
`requirements-vision.txt` records the direct runtime dependencies.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-vision.txt
```

Downloaded models are ignored by Git. A fresh checkout needs these two additional
downloads alongside the existing face, hand, and phone weights:

- [FER+ ONNX model](https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx): save as `models/emotion-ferplus-8.onnx`.
- [Vosk small English ZIP](https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip): extract so `models/vosk-model-small-en-us-0.15/am/final.mdl` exists.

`models/manifest.json` includes SHA-256 hashes of the downloaded ONNX and ZIP.
Model preprocessing and class order follow the [official FER+ documentation](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus).

## Validation

```powershell
.\.venv\Scripts\python.exe -m unittest verify_gestures verify_vision verify_interactions verify_phone -v
.\.venv\Scripts\python.exe speech.py
.\.venv\Scripts\python.exe verify_speech.py
.\.venv\Scripts\python.exe speech.py --listen --seconds 35
.\.venv\Scripts\python.exe expressions.py --seconds 30
.\.venv\Scripts\python.exe vision_behavior.py --manual happy
.\.venv\Scripts\python.exe vision_behavior.py --manual angry
.\.venv\Scripts\python.exe vision_behavior.py --manual phone
.\.venv\Scripts\python.exe vision_behavior.py --seconds 35
```

All 20 automated regression tests passed, including expression confirmation,
priority, face latching, smooth idle/wake, and manual OFF throughout all five
original reactions and the generic-hand reaction while stepping MuJoCo with
joint/control/physics checks. Phone exclusivity tests cover all requested
combinations, delayed confirmation, and dance preemption. The manual phone viewer
test played its line and completed the shake/flash sequence successfully. An
actual SAPI test confirmed interruption of an active greeting, removal of a queued
greeting, and exactly one completed phone utterance. Earlier live
tests confirmed spoken face greeting and HAPPY detection/reaction. Manual SAD
motion passed. Generated speech verified recognition of turn on/turn off and
rejection of hello there.

Live ANGRY classification and live microphone ON/OFF commands remain unverified.
The ANGRY replacement passed all 21 regression tests, including checking that
anger triggers the reaction and sadness does not.
After reconnecting the camera, the combined 45-second test opened webcam 0 in
0.66 seconds, processed live frames, initialized the Realtek microphone, entered
idle dimming, and exited cleanly. No visual reactions or accepted voice commands
occurred during that run, so those remaining live interaction checks are still
unverified.
