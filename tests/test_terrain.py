"""elephant — tests: the terrain, the shadows, and the deadband.

The captain's founding reframing (docs/terrain-2026-08-17.md): the true
vectorized state of a room is the Terrain — beyond human reading. What
we see are Shadows (lossy witness marks). The point is never fidelity,
it is enough information to agree on the action. And a deadband rings
up the chain of command: below significance nothing rings, the room
breathes; when the terrain crosses — a real warming, a real panic, a
real anomaly — the witness mark rings the host, and keeps ringing up
the chain while the room keeps crossing, and descends when the room
goes quiet.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import pytest

from elephant.terrain import (
    ChainOfCommand,
    Deadband,
    Ring,
    Shadow,
    StateVector,
    Terrain,
    read_state,
)
from elephant.space import ChatSpace

WARM = [
    "good talk tonight, though the week was long, dull, and tired — but kind and warm",
    "glad you came — the day was cold and flat, but the hearth is bright and home",
    "cheers — the walk was empty and stale, but this table is good and warm",
    "peace at last — the storm was cold and dead, but the room is soft and kind",
]


def _warm_room() -> ChatSpace:
    """A ChatSpace warmed by the four WARM texts, 5s apart."""
    space = ChatSpace("The Tap")
    for i, text in enumerate(WARM):
        space.post("welder", text, ts=float(i) * 5.0)
    return space


# ---------------------------------------------------------------------- #
# Terrain — the true state                                               #
# ---------------------------------------------------------------------- #
def test_terrain_records_and_retrieves_state():
    t = Terrain("The Tap")
    t.record({"mood": 0.3, "panic": 0.1}, ts=10.0)
    t.record({"mood": 0.5, "panic": 0.1}, ts=20.0)
    t.record({"mood": 0.4, "panic": 0.05}, ts=30.0)

    assert len(t) == 3
    assert t.state_at(25.0) is t.records[1]      # nearest at-or-before
    assert t.state_at(10.0) is t.records[0]
    assert t.state_at(5.0) is None               # before the first record
    assert t.state_at(99.0) is t.records[-1]     # after the last: the last
    assert t.recent(2) == [t.records[1], t.records[2]]   # newest last
    assert t.last is t.records[-1]

    # warmth is computed from the readings the vector claims
    from elephant.field import RoomField
    assert t.last.warmth == pytest.approx(
        RoomField({"mood": 0.4, "panic": 0.05}).warmth())

    # capacity bounds the buffer (the old ground is forgotten)
    small = Terrain("x", capacity=3)
    for i in range(10):
        small.record({"mood": i / 10.0}, ts=float(i))
    assert len(small) == 3
    assert small.records[0].ts == 7.0

    # record_room reads a space through the dial bank
    space = _warm_room()
    terr = Terrain("The Tap")
    sv = terr.record_room(space, ts=2.0)
    assert sv.space_id == "The Tap"
    assert "mood" in sv.dials and "model_vs_code" in sv.dials
    assert sv.warmth > 0.0
    assert len(sv.salience()) == len(terr.salience_labels())


def test_state_vector_salience_and_movement():
    t = Terrain("The Tap")
    t.record({"mood": 0.4, "panic": 0.05}, ts=0.0)
    t.record({"mood": 0.5, "panic": 0.45}, ts=1.0)
    sv = t.last
    sal = sv.salience()
    assert sal.shape[0] == len(t.salience_labels()) == len(t.names) + 3
    assert sal[0] == pytest.approx(0.5)                    # mood
    assert sal[-3] == pytest.approx(sv.warmth)             # warmth
    assert sal[-2] == pytest.approx(sv.trend)              # trend
    assert sal[-1] == pytest.approx(sv.anomaly)            # anomaly

    db = Deadband(threshold=0.10)
    t.record({"mood": 0.4, "panic": 0.05}, ts=10.0)
    db.check(t)                                            # anchors
    t.record({"mood": 0.5, "panic": 0.45}, ts=11.0)
    assert db.movement(t) == pytest.approx(0.4)            # panic 0.05 -> 0.45
    a, b = t.records[-2].salience(), t.records[-1].salience()
    assert db.movement(t) == pytest.approx(float(np.abs(b - a).max()))

    # the trail of words is witness material
    t2 = Terrain("The Tap")
    t2.hear("welder", "good evening", 1.0)
    t2.hear("mara", "cheers", 2.0)
    assert len(t2.transcript) == 2


# ---------------------------------------------------------------------- #
# Shadow — the witness mark                                              #
# ---------------------------------------------------------------------- #
def test_shadow_is_lossy_but_truthful():
    t = Terrain("The Tap")
    t.record({
        "mood": 0.35, "joke_landing": 0.8, "panic": 0.05, "volume": 0.3,
        "earnestness": 0.6, "cynicism": 0.2, "presence": 0.5,
        "model_vs_code": 0.3, "vision": 0.5,
    }, ts=60.0)

    line = Shadow(t).project()
    assert line.startswith("shadow")
    assert "The Tap" in line
    assert "\n" not in line                      # ONE line of witness

    # every number the line claims IS a number in the state it witnesses
    last = t.last
    claimed = {round(last.warmth, 2)}
    claimed.update(round(float(v), 2) for v in last.dials.values())
    for tok in re.findall(r"[+-]\d+\.\d\d", line):
        assert float(tok) in claimed, f"line number {tok} not in the terrain state"

    # the loudest dials make the line
    assert f"{last.dials['joke_landing']:+.2f}" in line
    assert "the room is laughing" in line        # joke_landing 0.8 >= 0.4

    # empty terrain: an honest shadow
    assert Shadow(Terrain("nowhere")).project() == "shadow · nowhere · no state yet"
    assert json.loads(Shadow(Terrain("nowhere")).project(format="json"))["state"] \
        == "no state yet"

    # json projection carries the same truth
    js = json.loads(Shadow(t).project(format="json"))
    assert js["label"] == "shadow" and js["space_id"] == "The Tap"
    assert js["warmth"] == round(last.warmth, 2)
    assert js["dials"]["joke_landing"] == 0.8

    # the transcript is a trail of labeled witness marks
    t2 = Terrain("The Tap")
    t2.hear("welder", "good evening", 1.0)
    t2.hear("mara", "cheers", 2.0)
    tr = Shadow(t2).render_transcript()
    assert tr.count("\n") == 1
    assert "[shadow]" in tr and "welder" in tr and "mara" in tr
    assert Shadow(t2).render_transcript(since=2.0).count("\n") == 0
    assert Shadow(t2).render_transcript(limit=1).count("\n") == 0


# ---------------------------------------------------------------------- #
# Deadband — the discipline                                              #
# ---------------------------------------------------------------------- #
def test_deadband_silent_under_the_band():
    t = Terrain("The Tap")
    db = Deadband(threshold=0.10)
    t.record({"mood": 0.4, "panic": 0.05}, ts=0.0)
    assert db.check(t) is None and not db.crossed     # first check anchors
    for i in range(10):
        t.record({"mood": 0.4, "panic": 0.05}, ts=1.0 + i)
        assert db.check(t) is None, f"ring on a still pulse {i}"
        assert not db.crossed

    # a room breathing under the band never disturbs anyone
    t2 = Terrain("The Tap")
    db2 = Deadband(threshold=0.10)
    t2.record({"mood": 0.3}, ts=0.0)
    db2.check(t2)
    for i in range(30):
        t2.record({"mood": 0.3 + 0.01 * (i % 3 - 1)}, ts=1.0 + i)
        assert db2.check(t2) is None
        assert not db2.crossed


def test_deadband_rings_over_the_band():
    t = Terrain("The Tap")
    db = Deadband(threshold=0.10)
    t.record({"mood": 0.4, "panic": 0.05}, ts=0.0)
    assert db.check(t) is None                        # anchors the quiet state

    t.record({"mood": -0.6, "panic": 0.75}, ts=5.0)   # a real panic
    ring = db.check(t)
    assert ring is not None
    assert ring.magnitude > 0.10
    assert "panic" in ring.what_crossed or "mood" in ring.what_crossed
    assert ring.shadow.startswith("shadow")
    assert ring.who == "the elephant"
    assert db.crossed

    # while the terrain stays away from the quiet state it KEEPS crossing
    for i in range(3):
        t.record({"mood": -0.5, "panic": 0.7}, ts=6.0 + i)
        assert db.check(t) is not None
        assert db.crossed

    # the room returns to the quiet state -> the deadband quiets and re-anchors
    t.record({"mood": 0.4, "panic": 0.05}, ts=9.0)
    assert db.check(t) is None and not db.crossed


# ---------------------------------------------------------------------- #
# ChainOfCommand — a deadband ringing up the chain                       #
# ---------------------------------------------------------------------- #
def test_chain_escalates_on_repeated_crossings_and_descends_on_quiet():
    chain = ChainOfCommand()                     # host -> foreman -> captain
    ring = Ring(ts=1.0, space_id="The Tap", magnitude=0.8, threshold=0.1,
                what_crossed="panic +0.71", shadow="shadow · The Tap · t=1.0")

    assert chain.ring() is None
    assert chain.escalate(ring) == "host"        # lowest step hears it first
    assert chain.escalate(ring) == "foreman"     # keeps crossing -> rises
    assert chain.escalate(ring) == "captain"
    assert chain.escalate(ring) == "captain"     # at the top, it holds
    assert chain.ring() == "captain"
    assert len(chain.history) == 4
    assert chain.last_ring is ring

    # a quiet room descends: quiet_after=2, then one level per quiet check
    assert chain.quiet() == "captain"            # streak 1 — not yet
    assert chain.quiet() == "foreman"            # streak 2 — descends
    assert chain.quiet() == "host"               # streak 3
    assert chain.quiet() is None                 # streak 4 — no one ringing

    # a fresh ring after quiet starts at the lowest step again
    assert chain.escalate(ring) == "host"

    # report() is the driver's one call: ring in, quiet in
    chain2 = ChainOfCommand(quiet_after=1)
    assert chain2.report(ring) == "host"
    assert chain2.report(None) is None           # one quiet check, one level down


# ---------------------------------------------------------------------- #
# End to end — the cave in miniature                                     #
# ---------------------------------------------------------------------- #
def test_end_to_end_warming_room_rings_up_to_foreman():
    space = _warm_room()
    t = Terrain("The Tap")
    db = Deadband(threshold=0.10)
    chain = ChainOfCommand()

    for i in range(3):
        t.record_room(space, ts=float(i) * 5.0)
        assert db.check(t) is None               # anchoring + a still warm room

    # the warming jump — the room climbs hard and stays warm
    space.post("welder", "warm bright good glad beautiful love", ts=20.0)
    t.record_room(space, ts=20.0)
    ring = db.check(t)
    assert ring is not None
    assert chain.report(ring) == "host"

    # the room stays warm -> the terrain keeps crossing -> the ring rises
    space.post("mara", "good great wonderful love", ts=25.0)
    t.record_room(space, ts=25.0)
    assert chain.report(db.check(t)) == "foreman"

    # the next evening: the same warm room, quiet again -> the chain descends
    fresh = ChatSpace("The Tap — next evening")
    for i, text in enumerate(WARM[:3]):
        fresh.post("welder", text, ts=float(i) * 5.0)
    for i in range(4):
        t.record_room(fresh, ts=200.0 + i)
        chain.report(db.check(t))
    assert chain.ring() is None


def test_end_to_end_stable_room_never_rings():
    space = _warm_room()
    t = Terrain("The Tap")
    db = Deadband(threshold=0.10)
    chain = ChainOfCommand()
    for i in range(12):
        space.post("welder", WARM[i % len(WARM)], ts=20.0 + i * 5.0)
        t.record_room(space, ts=20.0 + i * 5.0)
        ring = db.check(t)
        assert ring is None
        assert chain.report(ring) is None
    assert chain.ring() is None
    assert chain.history == []
    assert not db.crossed


# ---------------------------------------------------------------------- #
# Hardening — the edge cases the design must not break on                #
# ---------------------------------------------------------------------- #
def test_deadband_handles_empty_and_single_record_terrain():
    db = Deadband(threshold=0.10)
    assert db.check(Terrain("empty")) is None          # no state: no ring
    assert not db.crossed
    t = Terrain("The Tap")
    t.record({"mood": 0.4}, ts=0.0)
    assert db.check(t) is None                          # anchors, no ring
    assert not db.crossed
    assert db.movement(t) == 0.0                        # anchored at itself


def test_deadband_release_uses_the_threshold_that_tripped_it():
    t = Terrain("The Tap")
    db = Deadband(threshold=0.10, hysteresis=0.5)
    t.record({"mood": 0.4, "panic": 0.05}, ts=0.0)
    db.check(t)                                         # anchors
    t.record({"mood": -0.6, "panic": 0.75}, ts=5.0)   # crosses at 0.10
    assert db.check(t) is not None
    db.threshold = 1.0                                  # raised AFTER tripping
    # still crossed: the release is 0.05 (0.10 * 0.5), not 0.50
    t.record({"mood": -0.5, "panic": 0.7}, ts=6.0)
    assert db.check(t) is not None
    # returning within the ORIGINAL release quiets it
    t.record({"mood": 0.4, "panic": 0.05}, ts=7.0)
    assert db.check(t) is None and not db.crossed


def test_slow_drift_below_the_band_never_rings():
    # a room drifting 0.01 per check never crosses the 0.10 band: each
    # quiet check re-anchors, so only a move PAST the band in one check
    # disturbs anyone (the level-gate semantics, per the spec's table).
    t = Terrain("The Tap")
    db = Deadband(threshold=0.10)
    t.record({"mood": 0.0}, ts=0.0)
    db.check(t)
    for i in range(60):
        t.record({"mood": 0.01 * (i + 1)}, ts=1.0 + i)
        assert db.check(t) is None
        assert not db.crossed


def test_shadow_stays_truthful_with_many_salient_dials():
    # seven dials all off their resting value: the one-line shadow shows
    # only three (lossy by design) but every number it prints is real,
    # and the json projection carries the full picture.
    t = Terrain("The Tap")
    t.record({
        "mood": -0.7, "joke_landing": -0.5, "panic": 0.9, "volume": 0.8,
        "earnestness": 0.9, "cynicism": 0.7, "presence": 0.2,
        "model_vs_code": -0.6, "vision": 0.1,
    }, ts=60.0)
    line = Shadow(t).project()
    assert line.count(" · ") <= 6                       # label, id, t, warmth + 3 dials
    last = t.last
    claimed = {round(last.warmth, 2)}
    claimed.update(round(float(v), 2) for v in last.dials.values())
    for tok in re.findall(r"[+-]\d+\.\d\d", line):
        assert float(tok) in claimed
    js = json.loads(Shadow(t).project(format="json"))
    assert len(js["dials"]) == 3                         # the top three
    assert js["dials"]["panic"] == 0.9                   # the loudest is there
    assert "a fight is breaking out" in line


def test_record_never_mutates_a_caller_state_vector():
    sv = StateVector(ts=5.0, space_id="The Tap", dials={"mood": 0.4},
                     field=np.array([0.4]), warmth=0.12, kappa=0.5)
    t = Terrain("The Tap")
    stored = t.record(sv, ts=9.0, meta={"phase": "test"})
    assert sv.ts == 5.0 and sv.space_id == "The Tap"   # untouched
    assert stored.ts == 9.0 and stored.meta == {"phase": "test"}
    assert stored is not sv


def test_state_vector_keeps_kappa_in_the_true_state():
    # κ is part of the terrain (the true state holds it) even though the
    # deadband does not ring on it — the discipline reads the senses.
    t = Terrain("The Tap")
    sv = t.record({"mood": 0.4, "panic": 0.05}, ts=0.0)
    assert sv.kappa > 0.0
    assert "kappa" not in t.salience_labels()
