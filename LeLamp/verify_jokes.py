"""Joke download validation, offline fallback, and event-only speech dispatch."""
import json
import unittest
from unittest.mock import Mock, patch
from jokes import JokeBank, fetch_jokes, ANGRY_INTRO
from vision_behavior import start_reaction


class JokeTests(unittest.TestCase):
    def test_online_line_ends_in_laughter_and_avoids_immediate_repeat(self):
        with patch("jokes.fetch_jokes", return_value=("Joke one.", "Joke two.")):
            bank = JokeBank()
            self.assertTrue(bank.ready.wait(2))
        first, second = bank.angry_line(), bank.angry_line()
        self.assertTrue(bank.online)
        self.assertTrue(first.startswith(ANGRY_INTRO))
        self.assertTrue(first.endswith("Hahaha"))
        self.assertNotEqual(first, second)

    def test_network_failure_still_provides_joke(self):
        with patch("jokes.fetch_jokes", side_effect=TimeoutError("test timeout")):
            bank = JokeBank()
            self.assertTrue(bank.ready.wait(2))
        self.assertFalse(bank.online)
        self.assertTrue(bank.angry_line().endswith("Hahaha"))

    def test_validates_online_payload(self):
        response = Mock()
        response.read.return_value = json.dumps([
            {"setup":"Setup?", "punchline":"Punchline."},
            {"setup":None, "punchline":"bad"},
        ]).encode()
        context = Mock()
        context.__enter__ = Mock(return_value=response)
        context.__exit__ = Mock(return_value=False)
        with patch("jokes.urlopen", return_value=context):
            self.assertEqual(fetch_jokes(), ("Setup? Punchline.",))

    def test_angry_dispatch_queues_once(self):
        manager, speaker = Mock(), Mock()
        start_reaction(manager, "ANGRY", speaker)
        speaker.speak_angry.assert_called_once_with()
        manager.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
