"""elephant — tests: the zeitgeist (Room-Elephant / Personal-Elephant) and the
MUD description tinting (the room's own body language)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elephant.field import DIAL_NAMES, RoomField
from elephant.mud import classify, tint_description
from elephant.presets import PRESETS, PersonalElephant, RoomElephant
from elephant.room import Message, Room


def _warm_room():
    return Room("The Tap", [
        Message("lucineer", "I love this place, it's warm and kind.", ts=0),
        Message("welder", "haha, to the room then — it heard us before we walked in. 😂", ts=4),
        Message("carpenter", "I'll drink to that, the room just holds. cheers everyone", ts=8),
        Message("shipwright", "the floor holds, the floor remembers. haha", ts=12),
        Message("mason", "I talked to it like a horse, it listened. lol", ts=16),
        Message("composite", "haha and the dust came off in years 😂", ts=20),
    ])


def test_presets_registry():
    assert PRESETS["room"] is RoomElephant
    assert PRESETS["personal"] is PersonalElephant
    assert set(PRESETS) == {"room", "personal"}


def test_room_elephant_is_stable_and_objective():
    room = _warm_room()
    # Two independent Room-Elephants read the same room -> the same field.
    a = RoomElephant(identity="The Tap")
    b = RoomElephant(identity="The Tap")
    fa, fb = a.read(room), b.read(room)
    assert fa.readings == fb.readings
    # Stable identity: reading twice does not drift.
    assert a.read(room).readings == fa.readings
    # Objective: it is a full, real field with all 7 dials.
    assert set(fa.readings) == set(DIAL_NAMES)
    assert isinstance(fa.warmth(), float)
    # First-class neutral default: an empty room rests at presence 0.5, volume 0.
    empty = RoomElephant().read(Room("empty"))
    assert empty.readings["presence"] == 0.5
    assert empty.readings["volume"] == 0.0


def test_personal_elephant_weighting_changes_field():
    room = _warm_room()
    objective = RoomElephant().read(room).vector()

    warm = PersonalElephant(
        "warm writer", dial_weights={"mood": 1.0}, bias={"mood": 0.0})
    cynic = PersonalElephant(
        "the critic", dial_weights={"cynicism": 1.0}, bias={"cynicism": 0.2})

    fw = warm.read(room).vector()
    fc = cynic.read(room).vector()

    # Weighting changes the reading (two different tastes -> two different fields).
    assert not np.allclose(fw, fc), (fw, fc)
    # The warm writer's reading amplifies mood beyond the objective.
    mi = DIAL_NAMES.index("mood")
    assert abs(fw[mi]) >= abs(objective[mi]) - 1e-9
    # A uniform-taste, zero-bias elephant returns the objective field un-deformed.
    flat = PersonalElephant("flat")
    assert np.allclose(flat.read(room).vector(), objective)


def test_personal_elephant_bias_shifts_field():
    room = _warm_room()
    plain = PersonalElephant("plain")
    # bias a dial with headroom (cynicism is ~0 in the warm room, so a
    # positive disposition moves it without hitting the upper clamp).
    sneering = PersonalElephant("sneering", bias={"cynicism": 0.3})
    ci = DIAL_NAMES.index("cynicism")
    assert sneering.read(room).vector()[ci] > plain.read(room).vector()[ci]


def test_attachments_store_and_recall():
    pe = PersonalElephant("the critic")
    pe.attach("perfume", "grandma's shop")
    pe.attach("song", "the lover I discovered the album with")
    assert pe.remember("perfume") == "grandma's shop"
    assert pe.remember("song") == "the lover I discovered the album with"
    assert pe.remember("nope") is None
    # attach is chainable
    assert pe.attach("x", 1) is pe


# --------------------------------------------------------------------------- #
# tint_description                                                            #
# --------------------------------------------------------------------------- #
BASE = ("The Tap: five tables, a long bar, pool and darts in the back.")


def _field(**kw):
    d = {"mood": 0.0, "volume": 0.0, "earnestness": 0.5, "cynicism": 0.0,
         "joke_landing": 0.0, "panic": 0.0, "presence": 0.5}
    d.update(kw)
    return RoomField(d)


def test_tint_joyful_under_laughter():
    f = _field(mood=0.7, volume=0.7, joke_landing=0.8, presence=0.9, panic=0.0)
    assert classify(f, hour=21.0) == "joyful"
    text = tint_description(f, BASE, hour=21.0).lower()
    assert "laughter" in text                      # laughter reverberates in
    assert any(a in text for a in
               ("bright", "glowing", "warm", "golden", "alive", "sparkling",
                "merry", "ringing", "humming", "soft-gold"))


def test_tint_panic_drenched_and_storm():
    f = _field(mood=-0.6, volume=0.9, joke_landing=-0.4, panic=0.9, presence=0.8)
    assert classify(f, hour=22.0) == "panic"
    text = tint_description(f, BASE, hour=22.0).lower()
    assert "drenched" in text                      # newcomers described drenched
    assert "storm" in text or "rain" in text       # the storm outside


def test_tint_closing_time():
    f = _field(mood=-0.2, volume=0.1, joke_landing=-0.1, panic=0.0, presence=0.2)
    assert classify(f, hour=2.0) == "closing"
    text = tint_description(f, BASE, hour=2.0).lower()
    assert "fluorescents" in text                  # disco off, fluorescents on
    assert "exit" in text and "tab" in text        # looking for exit, closing tabs
    # A warm laughing room late at night is still the warm bar, not closing.
    joy = _field(mood=0.7, volume=0.7, joke_landing=0.8, presence=0.9)
    assert classify(joy, hour=2.0) == "joyful"


def test_tint_is_deterministic_and_field_sensitive():
    f = _field(mood=0.7, joke_landing=0.8, presence=0.9, volume=0.7)
    t1 = tint_description(f, BASE, hour=21.0)
    t2 = tint_description(f, BASE, hour=21.0)
    assert t1 == t2                              # same field -> same words
    g = _field(mood=-0.6, panic=0.9, volume=0.9, presence=0.8)
    assert tint_description(g, BASE, hour=22.0) != t1   # changed field -> changed room


def test_tint_neutral_default():
    f = _field()  # all neutral
    text = tint_description(f, BASE, hour=15.0)
    assert BASE in text                          # base description still present
    assert classify(f, hour=15.0) == "neutral"


if __name__ == "__main__":
    fns = [test_presets_registry, test_room_elephant_is_stable_and_objective,
           test_personal_elephant_weighting_changes_field,
           test_personal_elephant_bias_shifts_field,
           test_attachments_store_and_recall,
           test_tint_joyful_under_laughter, test_tint_panic_drenched_and_storm,
           test_tint_closing_time, test_tint_is_deterministic_and_field_sensitive,
           test_tint_neutral_default]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print("\nAll zeitgeist tests passed.")
