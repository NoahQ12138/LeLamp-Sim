"""Diagnose local recognition with generated speech, without using the microphone."""
import json
from pathlib import Path
import tempfile
import wave
import numpy as np
from speech import recognized_command, VOICE_MODEL, ON_PHRASES, OFF_PHRASES


def main():
    import win32com.client
    from vosk import Model, KaldiRecognizer, SetLogLevel
    SetLogLevel(-1)
    model = Model(str(VOICE_MODEL))
    engine = win32com.client.Dispatch("SAPI.SpVoice")
    try:
        with tempfile.TemporaryDirectory(prefix="lelamp-speech-test-") as directory:
            for phrase, expected in (("turn on", "ON"), ("turn off", "OFF"), ("hello there", None)):
                path = Path(directory) / "sample.wav"
                stream = win32com.client.Dispatch("SAPI.SpFileStream")
                stream.Open(str(path), 3)  # SSFMCreateForWrite.
                engine.AudioOutputStream = stream
                try:
                    engine.Speak(phrase)
                finally:
                    stream.Close()
                with wave.open(str(path), "rb") as source:
                    assert source.getsampwidth() == 2
                    samples = np.frombuffer(source.readframes(source.getnframes()), np.int16)
                    if source.getnchannels() > 1:
                        samples = samples.reshape(-1, source.getnchannels()).mean(axis=1).astype(np.int16)
                    recognizer = KaldiRecognizer(model, source.getframerate(),
                        json.dumps(sorted(ON_PHRASES | OFF_PHRASES)+["[unk]"]))
                    recognizer.SetWords(True)
                    recognizer.AcceptWaveform(samples.tobytes())
                    result = json.loads(recognizer.FinalResult())
                text, command = recognized_command(result)
                print(f"Generated {phrase!r} -> {text!r} -> {command}", flush=True)
                assert command == expected, result
        print("PASS: synthesized ON/OFF recognized, unrelated speech ignored.")
    finally:
        engine = None


if __name__ == "__main__":
    main()
