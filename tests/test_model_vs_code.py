"""elephant — tests: the model-vs-code dial (the 8th sense)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elephant.dial import Dial, DialBank
from elephant.dials import DEFAULT_DIALS, ModelVsCodeDial
from elephant.room import Message, Room


def _code_room():
    return Room("ci", [
        Message("bot", "fix: handle null pointer in parser", ts=0),
        Message("bot", "def process(x): return x * 2", ts=1),
        Message("bot", "Traceback (most recent call last): KeyError: 'x'", ts=2),
        Message("dev", "impl SignalNode for Gain { fn process(&mut self, input: f64) -> f64 { input * self.amount } }", ts=3),
    ])


def _prose_room():
    return Room("salon", [
        Message("writer", "I think the room holds something warm — we built it together and it remembers us.", ts=0),
        Message("writer", "Perhaps the elephant is not something you see, but something you feel when you walk in.", ts=1),
        Message("critic", "However, I wonder whether any of us noticed it before tonight. It seems, in a sense, alive.", ts=2),
    ])


def test_code_heavy_room_reads_negative():
    r = ModelVsCodeDial().read(_code_room())
    assert isinstance(r, float)
    assert r < 0.0, r


def test_prose_heavy_room_reads_positive():
    r = ModelVsCodeDial().read(_prose_room())
    assert isinstance(r, float)
    assert r > 0.0, r


def test_code_room_is_more_negative_than_prose_room():
    code = ModelVsCodeDial().read(_code_room())
    prose = ModelVsCodeDial().read(_prose_room())
    assert code < prose, (code, prose)


def test_empty_room_is_neutral():
    assert ModelVsCodeDial().read(Room("empty")) == 0.0


def test_reading_is_bounded():
    for room in (_code_room(), _prose_room(), Room("empty")):
        r = ModelVsCodeDial().read(room)
        assert -1.0 <= r <= 1.0, r


def test_registered_as_eighth_dial_and_satisfies_abc():
    assert len(DEFAULT_DIALS) == 9, [d.name for d in DEFAULT_DIALS]  # + vision (9th)
    names = [d.name for d in DEFAULT_DIALS]
    assert names.index("model_vs_code") == 7
    mv = [d for d in DEFAULT_DIALS if d.name == "model_vs_code"][0]
    assert isinstance(mv, Dial)
    assert isinstance(mv, ModelVsCodeDial)
    # It reads cleanly through the bank.
    readings = DialBank(DEFAULT_DIALS).readings(_prose_room())
    assert "model_vs_code" in readings
    assert readings["model_vs_code"] > 0.0


if __name__ == "__main__":
    for fn in [test_code_heavy_room_reads_negative, test_prose_heavy_room_reads_positive,
               test_code_room_is_more_negative_than_prose_room,
               test_empty_room_is_neutral, test_reading_is_bounded,
               test_registered_as_eighth_dial_and_satisfies_abc]:
        fn()
        print(f"PASS {fn.__name__}")
    print("\nAll model-vs-code tests passed.")
