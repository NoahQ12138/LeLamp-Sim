"""Local Windows speech output; the worker never accesses MuJoCo."""
import queue
import threading
import time
import json
from pathlib import Path

TTS_ENABLED = True
TTS_RATE = 0  # Windows SAPI rate, from -10 to +10.
VOICE_INPUT_ENABLED = True
MICROPHONE_DEVICE = None  # Auto: prefer laptop microphone array; otherwise OS default.
VOICE_CONFIDENCE = 0.80
ON_PHRASES = {"on", "turn on", "lamp on"}
OFF_PHRASES = {"off", "turn off", "lamp off"}
VOICE_MODEL = Path(__file__).resolve().parent / "models" / "vosk-model-small-en-us-0.15"
PHONE_VOICE_LINE = "Stop scrolling, please lock in twin."


class Speaker(threading.Thread):
    def __init__(self):
        super().__init__(name="LeLamp speech output", daemon=True)
        self.messages = queue.Queue(maxsize=8)
        self.stop_requested = threading.Event()
        self.speaking = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.completed = 0
        self.muted_until = 0.0
        self.generation = 0
        self.message_lock = threading.Lock()
        from jokes import JokeBank
        self.jokes = JokeBank()
        self.start()

    def speak_angry(self):
        self.speak(self.jokes.angry_line())

    def cancel_pending(self):
        """Invalidate queued/current speech; SAPI cancellation stays on its worker."""
        with self.message_lock:
            self.generation += 1
            while True:
                try:
                    self.messages.get_nowait()
                    self.messages.task_done()
                except queue.Empty:
                    break

    def speak(self, text, interrupt=False):
        if not TTS_ENABLED or self.stop_requested.is_set():
            return
        if interrupt:
            self.cancel_pending()
        with self.message_lock:
            try:
                self.messages.put_nowait((self.generation, time.monotonic(), str(text)))
            except queue.Full:
                print("Speech queue full; skipping interaction message.", flush=True)

    def run(self):
        engine = None
        initialized = False
        try:
            import pythoncom
            pythoncom.CoInitialize()
            initialized = True
            import win32com.client
            engine = win32com.client.Dispatch("SAPI.SpVoice")
            engine.Rate = TTS_RATE
            self.ready.set()
            while not self.stop_requested.is_set():
                try:
                    generation, created, text = self.messages.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if time.monotonic()-created > 8 or generation != self.generation:
                        continue
                    self.speaking.set()
                    print(f'Speaking: "{text}"', flush=True)
                    # SAPI async + purge, followed by short worker-only waits.
                    engine.Speak(text, 19)  # Async + purge + literal text (not XML).
                    while not engine.WaitUntilDone(20):
                        pythoncom.PumpWaitingMessages()
                        if self.stop_requested.is_set() or generation != self.generation:
                            engine.Speak("", 3)
                            break
                    else:
                        self.completed += 1
                finally:
                    self.muted_until = time.monotonic() + 0.6
                    self.speaking.clear()
                    self.messages.task_done()
        except Exception as error:
            self.error = error
            print(f"Speech output unavailable: {error}", flush=True)
        finally:
            self.ready.set()
            engine = None
            if initialized:
                pythoncom.CoUninitialize()

    def close(self):
        self.stop_requested.set()
        self.join(timeout=5)


_speaker = None


def get_speaker():
    global _speaker
    if _speaker is None or not _speaker.is_alive():
        _speaker = Speaker()
    return _speaker


def speak(text):
    get_speaker().speak(text)


def recognized_command(result):
    text = " ".join(result.get("text", "").lower().strip().split())
    words = result.get("result", [])
    if not words or min(word.get("conf", 0) for word in words) < VOICE_CONFIDENCE:
        return text, None
    return text, "ON" if text in ON_PHRASES else "OFF" if text in OFF_PHRASES else None


class Microphone(threading.Thread):
    """Only finalized recognition produces commands; no MuJoCo writes here."""
    def __init__(self, speaker=None):
        super().__init__(name="LeLamp microphone", daemon=True)
        self.speaker = speaker
        self.stop_requested = threading.Event()
        self.events = queue.Queue(maxsize=8)
        self.audio = queue.Queue(maxsize=32)
        self.ready = threading.Event()
        self.error = None
        self.status = "Loading offline speech model..."
        self.heard = 0
        self.last_text = ""
        self.rms = 0.0
        self.last_command = (None, float("-inf"))

    def run(self):
        try:
            import sounddevice as sd
            import numpy as np
            from vosk import Model, KaldiRecognizer, SetLogLevel
            SetLogLevel(-1)
            model = Model(str(VOICE_MODEL))
            device = MICROPHONE_DEVICE
            if device is None:
                device = next((i for i, d in enumerate(sd.query_devices())
                               if d["max_input_channels"] > 0 and "microphone array" in d["name"].lower()
                               and "realtek" in d["name"].lower()), None)
            info = sd.query_devices(device, "input")
            sample_rate = int(info["default_samplerate"])
            recognizer = KaldiRecognizer(model, sample_rate,
                json.dumps(sorted(ON_PHRASES | OFF_PHRASES) + ["[unk]"]))
            recognizer.SetWords(True)

            def callback(data, frames, clock, status):
                try:
                    self.audio.put_nowait((time.monotonic(), bytes(data), bool(status)))
                except queue.Full:
                    # Dropped audio must never be spliced into a command.
                    self.overflow = True

            self.overflow = False
            with sd.RawInputStream(device=device, samplerate=sample_rate,
                                   blocksize=sample_rate//10, dtype="int16", channels=1, callback=callback):
                print(f"Microphone: {info['name']} at {sample_rate} Hz; listening locally.", flush=True)
                self.ready.set()
                while not self.stop_requested.is_set():
                    try:
                        captured, data, invalid = self.audio.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    samples = np.frombuffer(data, dtype=np.int16).astype(float)
                    self.rms = float(np.sqrt(np.mean(samples*samples))/32768)
                    muted = self.speaker is not None and (self.speaker.speaking.is_set()
                                                         or captured < self.speaker.muted_until)
                    if muted or invalid or self.overflow or time.monotonic()-captured > 1:
                        recognizer.Reset()
                        self.overflow = False
                        self.status = "Paused for speech output" if muted else "Listening"
                        continue
                    self.status = "Listening"
                    if not recognizer.AcceptWaveform(data):
                        continue  # Never trigger from PartialResult().
                    text, command = recognized_command(json.loads(recognizer.Result()))
                    if not text:
                        continue
                    self.heard += 1
                    self.last_text = text
                    print(f'Heard: "{text}" | Command: {command or "ignored"}', flush=True)
                    now = time.monotonic()
                    previous, last_time = self.last_command
                    if command and (command != previous or now-last_time >= 1.5):
                        self.last_command = command, now
                        try:
                            self.events.put_nowait((command, now))
                        except queue.Full:
                            pass
        except Exception as error:
            self.error = error
            self.status = f"Microphone unavailable: {error}"
            print(self.status, flush=True)
        finally:
            self.ready.set()

    def close(self):
        self.stop_requested.set()
        if self.ident is not None:
            self.join(timeout=5)


def test_tts():
    speaker = get_speaker()
    try:
        speak("Hello there")
        deadline = time.monotonic()+30
        while speaker.completed == 0 and speaker.error is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if speaker.error:
            raise RuntimeError("TTS failed") from speaker.error
        if speaker.completed == 0:
            raise RuntimeError("TTS did not complete within 30 seconds")
        print("PASS: Windows speech output completed.")
    finally:
        speaker.close()


def test_microphone(seconds):
    import cv2
    import numpy as np
    microphone = Microphone()
    microphone.start()
    commands = []
    try:
        if not microphone.ready.wait(30):
            raise RuntimeError("Microphone initialization timed out")
        if microphone.error:
            raise RuntimeError("Microphone initialization failed") from microphone.error
        print(f"Say 'lamp on', pause, then 'lamp off'. Listening for {seconds:g} seconds.", flush=True)
        until = time.monotonic()+seconds
        while time.monotonic() < until and microphone.error is None:
            try:
                command, _ = microphone.events.get(timeout=0.02)
                commands.append(command)
            except queue.Empty:
                pass
            screen = np.zeros((280, 800, 3), dtype=np.uint8)
            lines = ["LeLamp voice test - microphone is LIVE",
                     "Say TURN ON, pause, then TURN OFF",
                     f"Heard: {microphone.last_text or '(waiting for speech)'}",
                     f"Accepted: {commands} | {max(0, int(until-time.monotonic()))} seconds left | q: quit"]
            for i, line in enumerate(lines):
                cv2.putText(screen, line, (15, 35+i*45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.rectangle(screen, (15,225), (15+int(min(1,microphone.rms*15)*750),250), (0,255,0), -1)
            cv2.imshow("LeLamp Voice Test", screen)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        print(f"Microphone test: {microphone.heard} final utterances; commands: {commands}", flush=True)
        if microphone.error:
            raise RuntimeError("Microphone failed") from microphone.error
        if not {"ON", "OFF"}.issubset(commands):
            raise RuntimeError("Both ON and OFF must be heard to verify this stage.")
    finally:
        microphone.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", action="store_true")
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()
    test_microphone(args.seconds) if args.listen else test_tts()
