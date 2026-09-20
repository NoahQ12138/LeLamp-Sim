"""LeLamp reactions; simulation updates are owned by the main thread."""
import argparse
import time
import math
import queue
import mujoco.viewer
from behaviors import BehaviorManager, REACTIONS
from lelamp_controller import LeLampController
from detection_events import DetectionEvents

VISION_STALE_SECONDS = 1.0


def start_reaction(manager, name, speaker=None):
    if speaker is not None:
        if name == "PHONE":
            from speech import PHONE_VOICE_LINE
            speaker.speak(PHONE_VOICE_LINE, interrupt=True)
        elif name == "FACE":
            speaker.speak("Hello there")
        elif name == "ANGRY":
            speaker.speak_angry()
    manager.start(name, manager.controller.data.time)


def return_to_neutral(controller, viewer):
    if not viewer.is_running():
        controller.lamp.lamp_off()
        return
    with viewer.lock():
        start_pose = controller.data.ctrl.copy()
        controller.set_light(False)
    steps = math.ceil(0.5/controller.model.opt.timestep)
    for index in range(steps):
        if not viewer.is_running():
            break
        tick = time.perf_counter()
        with viewer.lock():
            blend = (1-math.cos(math.pi*(index+1)/steps))/2
            controller.data.ctrl[:] = start_pose + blend*(controller.neutral-start_pose)
            controller.step()
        viewer.sync()
        time.sleep(max(0, controller.model.opt.timestep-(time.perf_counter()-tick)))


def manual(name):
    from speech import get_speaker, TTS_ENABLED
    controller = LeLampController()
    manager = BehaviorManager(controller)
    speaker = get_speaker() if TTS_ENABLED and name in ("phone", "angry") else None
    try:
        with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
            start_reaction(manager, name.upper(), speaker)
            deadline = time.monotonic()+60
            while viewer.is_running():
                start = time.perf_counter()
                with viewer.lock():
                    manager.update(controller.data.time)
                    controller.step()
                viewer.sync()
                if manager.state == "IDLE" and (speaker is None or speaker.error or
                        speaker.messages.unfinished_tasks == 0 or time.monotonic() > deadline):
                    print("PASS: manual reaction completed with physics/joint/control checks.", flush=True)
                    break
                time.sleep(max(0, controller.model.opt.timestep-(time.perf_counter()-start)))
    finally:
        if speaker is not None:
            speaker.close()


def live(stage="all", seconds=None, voice_input=True):
    import cv2
    import numpy as np
    from vision import VisionWorker
    from speech import get_speaker, TTS_ENABLED, Microphone, VOICE_INPUT_ENABLED
    controller = LeLampController()
    manager = BehaviorManager(controller)
    events = DetectionEvents()
    worker = VisionWorker(stage)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    last_capture = 0.0
    started = None
    triggers = {name: 0 for name in REACTIONS}
    healthy = True
    speaker = get_speaker() if TTS_ENABLED else None
    microphone = Microphone(speaker) if voice_input and VOICE_INPUT_ENABLED else None
    voice_counts = {"ON": 0, "OFF": 0}
    pending_event = None
    try:
        with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [-0.15, 0.1, 0.13]
                viewer.cam.distance = 1.05
                viewer.cam.azimuth = 145
                viewer.cam.elevation = -25
            worker.start()
            if microphone is not None:
                microphone.start()
            try:
                while viewer.is_running():
                    tick = time.perf_counter()
                    now = time.monotonic()
                    if worker.error is not None:
                        raise RuntimeError(f"Vision failed: {worker.error}") from worker.error
                    while True:
                        try:
                            captured, frame, detections = worker.results.get_nowait()
                        except queue.Empty:
                            break
                        if now-captured <= VISION_STALE_SECONDS:
                            events.observe(detections)
                            last_capture = captured
                            if started is None:
                                started = now
                    event = events.trigger(now, pending_event or manager.state) if now-last_capture <= VISION_STALE_SECONDS else None
                    with viewer.lock():
                        if microphone is not None:
                            while True:
                                try:
                                    command, recognized_at = microphone.events.get_nowait()
                                except queue.Empty:
                                    break
                                if now-recognized_at <= 2:
                                    controller.voice_command(command, now)
                                    voice_counts[command] += 1
                        if event:
                            controller.note_activity(now)
                            pending_event = event
                            if event == "PHONE" and speaker is not None:
                                speaker.cancel_pending()
                        controller.update_lighting(now)
                        blocked = pending_event and events.blocks_pending(pending_event)
                        if pending_event and not blocked and (controller.activity.awake or controller.manual_lamp_off):
                            triggers[pending_event] += 1
                            start_reaction(manager, pending_event, speaker)
                            pending_event = None
                        manager.update(controller.data.time)
                        controller.step()
                    viewer.sync()
                    if frame is not None:
                        output = frame.copy()
                        detected = (worker.status if last_capture == 0 else
                                    " + ".join(events.visible) or "IDLE" if now-last_capture <= VISION_STALE_SECONDS else "CAMERA STALE")
                        status = {"FACE": "GREETING", "HAND": "HAND WIGGLE", "PEACE": "PEACE DANCE", "PHONE": "PHONE REACTION",
                                  "HAPPY": "HAPPY REACTION", "ANGRY": "ANGRY REACTION"}.get(manager.state)
                        if status is None:
                            status = (manager.state.replace("_", " ") if manager.state.startswith("POINT_") else
                                      "COOLDOWN / WAIT FOR LEAVE" if events.detected != "IDLE" else "IDLE")
                        if pending_event:
                            status = ("WAIT FOR " + events.candidate if blocked else "WAKING -> " + pending_event)
                        cv2.rectangle(output, (0,0), (output.shape[1],165), (25,25,25), -1)
                        cv2.putText(output, f"Detected: {detected}", (10,25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                        priority = (events.candidate if blocked else pending_event or
                                    (manager.state if manager.state != "IDLE" else events.candidate))
                        if now-last_capture > VISION_STALE_SECONDS:
                            priority = "UNKNOWN"
                        cv2.putText(output, f"Priority: {priority} | Behavior: {status}", (10,55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                        cv2.putText(output, worker.status, (10,85),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                        voice_status = microphone.status if microphone is not None else "Disabled"
                        lamp_status = "OFF (manual)" if controller.manual_lamp_off else "ON" if controller.lamp.enabled else "FLASH OFF"
                        if controller.lamp.enabled and controller.activity.target < 1:
                            lamp_status = "DIMMED" if controller.activity.value <= 0.201 else "DIMMING"
                        cv2.putText(output, f"Lamp: {lamp_status} {controller.lamp.brightness:.0%} | Voice: {voice_status}", (10,115),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
                        expression = worker.expression_label if now-last_capture <= VISION_STALE_SECONDS else "UNKNOWN"
                        cv2.putText(output, f"Expression: {expression} ({worker.expression_confidence:.2f}) - visible cues only",
                                    (10,145), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
                        cv2.imshow("LeLamp Vision", output)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                        if cv2.getWindowProperty("LeLamp Vision", cv2.WND_PROP_VISIBLE) < 1:
                            break
                    if seconds is not None and started is not None and now-started >= seconds:
                        break
                    time.sleep(max(0, controller.model.opt.timestep-(time.perf_counter()-tick)))
            except KeyboardInterrupt:
                pass
            except Exception:
                healthy = False
                raise
            finally:
                worker.stop_requested.set()
                if healthy:
                    return_to_neutral(controller, viewer)
                else:
                    with viewer.lock():
                        controller.set_light(False)
    finally:
        if worker.ident is not None:
            worker.close()
        if microphone is not None:
            microphone.close()
        if speaker is not None:
            speaker.close()
        cv2.destroyAllWindows()
        print(f"Reaction triggers: {triggers}; camera/viewer cleanup complete.", flush=True)
        print(f"Voice commands: {voice_counts}", flush=True)
    return triggers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", choices=tuple(name.lower() for name in REACTIONS))
    parser.add_argument("--stage", choices=("face", "peace", "hand", "phone", "all"), default="all")
    parser.add_argument("--seconds", type=float, help="Stop a live stage test after this many seconds of camera results.")
    parser.add_argument("--no-voice", action="store_true", help="Disable microphone input for this run.")
    args = parser.parse_args()
    if args.manual:
        manual(args.manual)
    else:
        counts = live(args.stage, args.seconds, voice_input=not args.no_voice)
        key = args.stage.upper()
        if args.seconds and args.stage != "all" and not counts[key]:
            raise RuntimeError(f"No {args.stage} reaction triggered; stage not yet verified.")


if __name__ == "__main__":
    main()
