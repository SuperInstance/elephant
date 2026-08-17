"""elephant — tests: the field, the dials, the dynamics."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elephant.dial import DialBank
from elephant.dials import DEFAULT_DIALS
from elephant.field import (
    RoomField, acclimation_curve, acclimation_rate_from,
    charisma_pull, read_field,
)
from elephant.room import Message, Room


def make_rooms():
    warm = Room("warm", [
        Message("a", "I love this place, it's warm and kind.", ts=0),
        Message("b", "Haha, truly! 😂 cheers everyone", ts=2),
        Message("a", "we built it together, honestly", ts=4),
    ])
    cold = Room("cold", [
        Message("x", "whatever. sure. fine.", ts=0),
        Message("y", "fire! flood! evacuate now!!!", ts=1),
        Message("x", "great.", ts=2),
    ])
    return warm, cold


def test_dials_read():
    bank = DialBank(DEFAULT_DIALS)
    warm, cold = make_rooms()
    fw = read_field(warm, bank)
    fc = read_field(cold, bank)
    assert set(fw.readings) == {d.name for d in DEFAULT_DIALS}
    assert fw.warmth() > fc.warmth(), (fw.warmth(), fc.warmth())


def test_elephant_gap():
    bank = DialBank(DEFAULT_DIALS)
    warm, cold = make_rooms()
    fw = read_field(warm, bank)
    fc = read_field(cold, bank)
    assert fw.distance(fc) > 0.1
    assert fw.sauna_plunge_gap(fc) > 0


def test_acclimation_converges():
    room = np.array([1.0, 0.5, -0.2])
    agent = np.array([-1.0, 0.0, 0.5])
    for t in (0, 1, 5, 20, 100):
        c = acclimation_curve(agent, room, rate=0.1, t=t)
        assert np.linalg.norm(c - room) <= np.linalg.norm(agent - room) + 1e-9
    c100 = acclimation_curve(agent, room, 0.1, 100)
    assert np.linalg.norm(c100 - room) < 1e-3


def test_acclimation_rate_inversion():
    room = np.array([1.0, 0.5, -0.2])
    agent = np.array([-1.0, 0.0, 0.5])
    obs = acclimation_curve(agent, room, rate=0.12, t=10)
    inferred = acclimation_rate_from(agent, obs, room, t=10)
    assert abs(inferred - 0.12) < 0.02


def test_acclimation_rate_overshoot_is_finite():
    # An agent that has already overshot the room (projection <= 0) must
    # yield a large *finite* rate, never inf.
    room = np.array([1.0, 0.5, -0.2])
    agent = np.array([-1.0, 0.0, 0.5])
    overshot = room + (agent - room) * -0.5   # past the room, on the far side
    rate = acclimation_rate_from(agent, overshot, room, t=10)
    assert math.isfinite(rate), rate
    assert rate > 0.5


def test_density_is_windowed():
    # A stale message far in the past must not dilute a recent burst.
    room = Room("mixed", [
        Message("a", "stale", ts=0),
        Message("a", "now1", ts=1000),
        Message("b", "now2", ts=1010),
        Message("c", "now3", ts=1020),
    ])
    d30 = room.density(window=30.0)     # only the 3 recent messages
    dall = room.density(window=2000.0)  # everything
    assert d30 > dall, (d30, dall)
    assert room.density(window=0.5) == 0.0  # no two messages within 0.5s


def test_charisma_pulls_room():
    room = np.array([0.0, 0.0, 0.0])
    agent = np.array([1.0, 1.0, 1.0])
    moved = charisma_pull(room, agent, charisma=0.2, interactions=30)
    assert np.linalg.norm(moved - room) > 0.1
    assert np.dot(moved, agent) > 0


def test_concentration_cold_tighter():
    # A one-note room is tighter (higher κ) than a scattered one.
    tight = RoomField({"mood": 0.9, "volume": 0.9, "earnestness": 0.9,
                       "cynicism": 0.1, "joke_landing": 0.8,
                       "panic": 0.0, "presence": 0.9})
    loose = RoomField({"mood": 0.3, "volume": 0.5, "earnestness": 0.5,
                       "cynicism": 0.5, "joke_landing": 0.1,
                       "panic": 0.5, "presence": 0.5})
    assert tight.concentration() > loose.concentration()


if __name__ == "__main__":
    for fn in [test_dials_read, test_elephant_gap, test_acclimation_converges,
               test_acclimation_rate_inversion, test_acclimation_rate_overshoot_is_finite,
               test_charisma_pulls_room, test_density_is_windowed,
               test_concentration_cold_tighter]:
        fn()
        print(f"PASS {fn.__name__}")
    print("\nAll elephant tests passed.")
