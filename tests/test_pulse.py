"""elephant — tests: the internal monologue pulse.

The captain's directive: agents run internal monologues on constant
pulses even when they aren't talking; each pulse takes a perception
check — the macro read of the table as a whole hand. Two numbers show
direction; more than two show rate of change.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from elephant.pulse import (
    DEFAULT_NOISE_FLOOR,
    PerceptionReport,
    PulseLoop,
    compose_monologue,
    compose_whole_hand,
    direction,
    rate_of_change,
)
from elephant.room import Message, Room
from elephant.space import ChatSpace
from elephant.tapnight import Participant, TapNightSession


# ---------------------------------------------------------------------- #
# The perception-check math                                              #
# ---------------------------------------------------------------------- #
def test_direction_from_last_two():
    # one number is nothing; two numbers show direction
    assert direction([{"mood": 0.1}, {"mood": 0.3}]) == pytest.approx({"mood": 0.2})
    assert direction([{"mood": 0.3}, {"mood": 0.1}]) == pytest.approx({"mood": -0.2})
    # only the last two readings matter — the older ones are the past
    assert direction([{"mood": -1.0}, {"mood": 0.1}, {"mood": 0.3}]) == pytest.approx(
        {"mood": 0.2})


def test_rate_of_change_from_three():
    # constant climb: velocity, but no acceleration — the rate is 0
    assert rate_of_change([{"mood": 0.1}, {"mood": 0.2}, {"mood": 0.3}]) == pytest.approx(
        {"mood": 0.0})
    # accelerating climb: the second difference is positive
    assert rate_of_change([{"mood": 0.1}, {"mood": 0.2}, {"mood": 0.4}]) == pytest.approx(
        {"mood": 0.1})
    # decelerating climb: the second difference is negative
    assert rate_of_change([{"mood": 0.1}, {"mood": 0.4}, {"mood": 0.5}]) == pytest.approx(
        {"mood": -0.2})
    # only the last three readings matter
    assert rate_of_change([{"mood": 0.0}, {"mood": 1.0}, {"mood": 0.1},
                           {"mood": 0.2}, {"mood": 0.4}]) == pytest.approx(
        {"mood": 0.1})


def test_ts_normalized_math():
    # with real timestamps the movement is per-second, not per-sample
    s = [{"mood": 0.0}, {"mood": 0.2}, {"mood": 0.4}]
    assert direction(s, ts=[0, 5, 10]) == pytest.approx({"mood": 0.04})   # 0.2 / 5s
    assert rate_of_change(s, ts=[0, 5, 10]) == pytest.approx({"mood": 0.0})
    # non-uniform spacing: the exact quadratic-interpolant acceleration
    # v12 = 0.2/1 = 0.2, v23 = 0.2/2 = 0.1, a = 2(0.1 - 0.2)/3 = -1/15
    assert rate_of_change(s, ts=[0, 1, 3]) == pytest.approx({"mood": -1.0 / 15.0})
    # uniform spacing (no ts): per-pulse units, second difference
    assert rate_of_change([{"mood": 0.0}, {"mood": 0.2}, {"mood": 0.4}]) == pytest.approx(
        {"mood": 0.0})


def test_short_series_read_zero():
    assert direction([]) == {}
    assert direction([{"mood": 0.2}]) == {"mood": 0.0}
    assert rate_of_change([]) == {}
    assert rate_of_change([{"mood": 0.1}, {"mood": 0.2}]) == {"mood": 0.0}


def test_nan_carried_forward_not_a_movement():
    # a NaN reading is not a movement; the last valid value holds
    s = [{"mood": 0.2}, {"mood": float("nan")}, {"mood": 0.3}]
    assert direction(s) == pytest.approx({"mood": 0.1})
    assert rate_of_change(s) == pytest.approx({"mood": 0.1})


def test_noise_floor_reads_small_moves_as_zero():
    # small moves below the floor read as 0 — the number doesn't matter
    s = [{"mood": 0.0}, {"mood": 0.005}, {"mood": 0.012}]
    assert direction(s, noise_floor=0.01) == {"mood": 0.0}
    assert rate_of_change(s, noise_floor=0.01) == {"mood": 0.0}
    # the same series above a finer floor is real movement
    assert direction(s, noise_floor=0.001) == pytest.approx({"mood": 0.007})
    assert rate_of_change(s, noise_floor=0.001) == pytest.approx({"mood": 0.002})
    assert DEFAULT_NOISE_FLOOR == 0.02


def test_ts_floor_is_in_output_units():
    # a slow real drift over long timestamps stays visible per-second:
    # the floor converts with the units instead of zeroing the signal
    s = [{"mood": 0.0}, {"mood": 0.3}]
    assert direction(s, ts=[0, 30]) == pytest.approx({"mood": 0.01})   # 0.3 / 30s


def test_constant_room_is_still():
    s = [{"mood": 0.4, "cynicism": 0.2}] * 5
    assert direction(s) == {"mood": 0.0, "cynicism": 0.0}
    assert rate_of_change(s) == {"mood": 0.0, "cynicism": 0.0}


def test_cooling_then_stabilizing_direction_flips_then_rate_decays():
    # cooling hard: direction negative, rate negative (accelerating down)
    cool = [{"mood": 0.6}, {"mood": 0.3}, {"mood": -0.2}]
    assert direction(cool) == pytest.approx({"mood": -0.5})
    assert rate_of_change(cool) == pytest.approx({"mood": -0.2})
    # then the room stabilizes: the flip is over, the rate decays toward 0
    settle = [{"mood": -0.2}, {"mood": -0.25}, {"mood": -0.26}]
    assert direction(settle) == {"mood": 0.0}          # below the noise floor
    assert rate_of_change(settle) == pytest.approx({"mood": 0.04})  # easing
    # above a finer floor the raw numbers are visible
    assert direction(settle, noise_floor=0.001) == pytest.approx({"mood": -0.01})
    assert rate_of_change(settle, noise_floor=0.001) == pytest.approx({"mood": 0.04})


# ---------------------------------------------------------------------- #
# The PulseLoop — the constant heartbeat                                 #
# ---------------------------------------------------------------------- #
WARMING_TEXTS = [
    "cold dull stale dead flat",                                   # mood -1.00
    "cold and dull, but kind — warm and glad, flat at first",      # mood -0.91
    "no fear, no panic — just great, good, and home",              # mood -0.67
    "never tired, never lost — happy, nice, and together",         # mood -0.56
    ("love this beautiful wonderful bright room — cheers and thanks, we're "
     "alive, laughing, glowing, at peace, soft and gentle, glad and happy, "
     "warm and good"),                                             # mood +0.44
]


def _drive(loop, texts, ts_step=10.0, tick_step=5.0):
    """Append one message per tick, interleaved, so every pulse sees a
    room that has MOVED since the last pulse."""
    for i, text in enumerate(texts):
        loop.room.messages.append(
            Message(author="host", text=text, ts=i * ts_step))
        loop.tick(now=(i + 1) * tick_step)


def test_warming_room_reports_warming_direction_and_positive_rate():
    room = Room("The Warm Up")
    loop = PulseLoop("observer", room, period=5.0)
    _drive(loop, WARMING_TEXTS)

    r = loop.perception_check()
    assert isinstance(r, PerceptionReport)
    assert r.n_readings == 5
    assert r.warmth_direction > 0          # the room is warming
    assert r.warmth_rate > 0               # and the warming is accelerating
    assert r.direction["mood"] > 0         # mood is climbing
    assert r.rate_of_change["mood"] > 0    # the climb itself is speeding up
    assert "warming" in r.whole_hand
    assert r.board()


def test_cooling_room_direction_flips_then_rate_decays():
    # --- the cooling leg ---
    room = Room("The Cold Plunge")
    loop = PulseLoop("observer", room, period=5.0)
    _drive(loop, [
        "love this beautiful warm room — glad, happy, cheers",
        "love warm kind glad, but cold dull",
        "cold and dead, flat and stale — no, never, no",
        "cold, dull, dead — empty, stale, lost, broke",
        "flat and tired, cold — wrong, bad, afraid",
    ])
    r = loop.perception_check()
    assert r.n_readings == 5
    assert r.warmth_direction < 0          # the room is cooling
    assert r.direction["mood"] < 0         # mood is falling
    assert "cooling" in r.whole_hand

    # --- the stabilizing leg: mood goes flat, its rate decays toward 0 ---
    room2 = Room("Still Water")
    loop2 = PulseLoop("observer", room2, period=5.0)
    _drive(loop2, ["kind warm cold dull"] * 5)   # balanced: mood holds at 0
    r2 = loop2.perception_check()
    assert abs(r2.direction["mood"]) < 1e-9
    assert abs(r2.rate_of_change["mood"]) < 1e-9
    assert "holding steady" in r2.whole_hand


def test_constant_room_reports_no_direction_no_rate():
    room = Room("Still")
    loop = PulseLoop("observer", room, period=5.0)
    for i in range(5):
        loop.tick(now=i * 5.0)              # nothing happens — never moves
    r = loop.perception_check()
    assert r.n_readings == 5
    assert r.warmth_direction == 0.0
    assert all(abs(v) < 1e-12 for v in r.direction.values())
    assert all(abs(v) < 1e-12 for v in r.rate_of_change.values())
    assert "holding steady" in r.whole_hand


def test_internal_monologue_returns_string_even_when_silent():
    room = Room("The Tap")
    loop = PulseLoop("silent_agent", room, period=5.0)
    # the table talks; the agent says NOTHING
    for i in range(6):
        room.messages.append(Message(author="host", text="cold dull stale",
                                     ts=i * 5.0))
        loop.tick(now=i * 5.0)
    r = loop.perception_check()
    assert r.agent_said is False

    mono = loop.internal_monologue()
    assert isinstance(mono, str)
    assert len(mono) > 20
    assert 1 <= mono.count(".") <= 3        # 1-3 sentences, never a wall

    # prompt-aware monologue stays short too
    mono2 = loop.internal_monologue(prompt="is the room warming?")
    assert isinstance(mono2, str)
    assert 1 <= mono2.count(".") <= 3
    assert "is the room warming?" in mono2

    # before any pulse: the silence is still not empty
    fresh = PulseLoop("fresh", Room("New"), period=5.0)
    assert isinstance(fresh.internal_monologue(), str)
    assert len(fresh.internal_monologue()) > 0


def test_pulse_ticks_without_speaking_and_still_perceives():
    space = ChatSpace("The Tap")
    agent = PulseLoop("poet", space, period=5.0, history=10)
    # the table talks; the poet never posts
    for i, text in enumerate([
        "love this warm room",
        "great and kind, cheers",
        "beautiful, glad, thank you",
        "wonderful, happy, together",
        "bright and alive, good",
    ]):
        space.post("welder", text)
        r = agent.tick(now=(i + 1) * 5.0)
        assert r.n_readings == i + 1
    r = agent.perception_check()
    assert r.n_readings == 5
    assert r.agent_said is False
    assert r.whole_hand
    assert len(agent.last_readings()) == 5   # the raw series is kept
    # the last tick's own report carries the table's traffic
    last = agent.last_report()
    assert last is not None and last.traffic == 1


def test_history_is_bounded():
    room = Room("R")
    loop = PulseLoop("a", room, period=1.0, history=3)
    for i in range(6):
        loop.tick(now=i * 1.0)
    assert len(loop.last_readings()) == 3
    assert loop.perception_check().n_readings == 3


def test_stale_tick_is_ignored_and_due_reports():
    room = Room("R")
    loop = PulseLoop("a", room, period=5.0)
    assert loop.due(0.0)                     # no tick yet: a pulse is due
    loop.tick(now=0.0)
    assert not loop.due(4.9)
    assert loop.due(5.0)
    loop.tick(now=0.0)                       # stale — the heart doesn't double-beat
    assert loop.perception_check().n_readings == 1
    assert loop.last_ts == 0.0


def test_irregular_ticks_still_read_movement():
    # a missed pulse (long gap) must not zero the read: the pulse is
    # per-pulse, not per-wall-clock-second — the movement is the perception
    room = Room("R")
    loop = PulseLoop("a", room, period=5.0)
    texts = ["cold dull stale", "kind warm glad",
             "love beautiful wonderful bright"]
    for i, text in enumerate(texts):
        room.messages.append(Message(author="host", text=text,
                                     ts=float(i) * 100.0))
        loop.tick(now=[0.0, 5.0, 600.0][i])   # a 595-second gap between pulses
    r = loop.perception_check()
    assert r.direction["mood"] > 0


def test_room_reset_does_not_fabricate_a_traffic_spike():
    room = Room("R")
    loop = PulseLoop("a", room, period=1.0)
    for i in range(3):
        room.messages.append(Message(author="host", text="hello",
                                     ts=float(i)))
        loop.tick(now=float(i + 1))
    assert loop.last_report().traffic == 1
    room.messages.clear()                      # the room resets between pulses
    r = loop.tick(now=5.0)
    assert r.traffic == 0                      # a reset is not a traffic spike
    assert r.n_readings == 4
    room.messages.append(Message(author="a", text="back", ts=10.0))
    r2 = loop.tick(now=6.0)
    assert r2.traffic == 1                     # the new session's first message
    assert r2.agent_said is True               # ...and it was the agent speaking


# ---------------------------------------------------------------------- #
# Integration — spaces and the Tap                                        #
# ---------------------------------------------------------------------- #
def test_loop_reads_spaces():
    space = ChatSpace("The Tap")
    loop = PulseLoop("critic", space)
    loop.tick(now=0.0)                       # empty room: dials at rest
    space.post("welder", "sure, sure, obviously great. 🙄")
    r = loop.tick(now=5.0)
    assert r.n_readings == 2
    assert r.direction["cynicism"] > 0       # the sneer arrived
    assert r.agent_said is False             # the critic said nothing


def test_loop_uses_tapnight_session_bank_and_room():
    s = TapNightSession("The Tap", participants=[
        Participant("writer", dial_weights={"mood": 1.0}, vibe={"mood": 0.6}),
    ])
    s.start_session()
    loop = PulseLoop("writer", s)            # bank + room come from the session
    assert loop.bank is s.bank
    s.speak("writer", "I love this warm room, truly. haha", reactions={"❤️": 2})
    loop.tick(now=60.0)
    s.speak("writer", "Beautiful, wonderful, glad to be here. cheers")
    loop.tick(now=120.0)
    r = loop.perception_check()
    assert r.n_readings == 2
    assert r.agent_id == "writer"
    assert isinstance(loop.internal_monologue(), str)
    assert len(loop.internal_monologue()) > 20


def test_compose_helpers_are_deterministic():
    room = Room("Warm")
    loop = PulseLoop("observer", room, period=5.0)
    _drive(loop, WARMING_TEXTS)
    r = loop.perception_check()
    assert compose_whole_hand(r) == r.whole_hand
    assert compose_monologue(r) == loop.internal_monologue()
    assert compose_monologue(r).startswith("I haven't said a word")
