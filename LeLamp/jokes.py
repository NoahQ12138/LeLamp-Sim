"""Prefetch online jokes without blocking speech, vision, or physics."""
import json
import random
import threading
from urllib.request import Request, urlopen

JOKE_URL = "https://official-joke-api.appspot.com/jokes/general/ten"
ANGRY_INTRO = "You seem angry. Let me lighten up your mood with a joke."
FALLBACK_JOKES = (
    "Why did the lamp get promoted? It had a bright idea.",
    "Why did the robot bring a ladder? To take its programming to the next level.",
)


def fetch_jokes():
    request = Request(JOKE_URL, headers={"User-Agent": "LeLamp-Sim/1.0"})
    with urlopen(request, timeout=4) as response:
        payload = json.loads(response.read(65536))
    if not isinstance(payload, list):
        raise ValueError("Expected a list of jokes")
    jokes = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        setup, punchline = item.get("setup"), item.get("punchline")
        if not isinstance(setup, str) or not isinstance(punchline, str):
            continue
        text = " ".join((setup + " " + punchline).split())
        if setup.strip() and punchline.strip() and len(text) <= 500:
            jokes.append(text)
    if not jokes:
        raise ValueError("No usable jokes returned")
    return tuple(dict.fromkeys(jokes))


class JokeBank:
    def __init__(self):
        self.jokes = FALLBACK_JOKES
        self.previous = None
        self.ready = threading.Event()
        self.online = False
        threading.Thread(target=self._load, name="LeLamp joke download", daemon=True).start()

    def _load(self):
        try:
            self.jokes = fetch_jokes()
            self.online = True
            print(f"Loaded {len(self.jokes)} online jokes.", flush=True)
        except Exception as error:
            print(f"Online jokes unavailable; using local backup jokes: {error}", flush=True)
        finally:
            self.ready.set()

    def angry_line(self):
        choices = [joke for joke in self.jokes if joke != self.previous] or list(self.jokes)
        self.previous = random.choice(choices)
        return f"{ANGRY_INTRO} {self.previous} Hahaha"
