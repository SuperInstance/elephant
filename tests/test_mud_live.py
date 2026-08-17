"""tests — the MUD live integration: the elephant's light in a real room.

These tests exercise the bridge's pure core (no network): the tint
round-trips through `MudSpace`, the field is read from fed messages, the
three states yield three *different* descriptions, and the adapter registry
resolves `mud` to `MudSpace`. The live Tap fetch is intentionally *not*
exercised here — tests must pass offline, and the bridge falls back to
transcripts on its own.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "examples"))

from elephant.field import RoomField
from elephant.mud import classify, tint_description
from elephant.space import AdapterRegistry, MudSpace

# Reuse the bridge's own pure helpers (synthetic scenes + build/read).
from mud_live_integration import (  # noqa: E402
    BASE_DESCRIPTION,
    build_mud_space,
    demo_three_states,
    read_and_tint,
    synthetic_scenes,
)


# --------------------------------------------------------------------------- #
# registry lookup                                                             #
# --------------------------------------------------------------------------- #
def test_registry_resolves_mud_to_mudspace():
    space = AdapterRegistry.get("mud", "The Tap")
    assert isinstance(space, MudSpace)
    assert space.kind == "mud"
    assert "mud" in AdapterRegistry.kinds()
    assert AdapterRegistry.has("mud")
    assert space.tint_target() == "the room description"


# --------------------------------------------------------------------------- #
# tint round-trips through MudSpace                                           #
# --------------------------------------------------------------------------- #
def test_tint_round_trips_through_mudspace():
    space = build_mud_space("The Tap", BASE_DESCRIPTION, synthetic_scenes()[0][1])
    field = space.read()
    tinted = space.send_back(field)
    # The tinted description is what agents actually read now.
    assert space.description == tinted
    # It is tint_description's output: the base description is still inside.
    assert BASE_DESCRIPTION in tinted
    # And the MudSpace.tint seam prefers elephant/mud.py's tint_description.
    assert tinted == tint_description(field, BASE_DESCRIPTION)


def test_send_back_defaults_to_tint():
    space = build_mud_space("The Tap", BASE_DESCRIPTION, synthetic_scenes()[0][1])
    field = space.read()
    # send_back without a pre-computed text derives it from the field.
    text = space.send_back(field)
    assert text == space.tint(field)
    assert space.description == text


# --------------------------------------------------------------------------- #
# field reads from fed messages                                               #
# --------------------------------------------------------------------------- #
def test_field_reads_from_fed_messages():
    label, events, _hour = synthetic_scenes()[0]  # warm laughter
    space = build_mud_space("The Tap", BASE_DESCRIPTION, events)
    field = space.read()
    assert isinstance(field, RoomField)
    # All seven dials are present.
    for name in ("mood", "volume", "earnestness", "cynicism",
                 "joke_landing", "panic", "presence"):
        assert name in field.readings
    # The warm-laughter room is warm, and the jokes landed.
    assert field.warmth() > 0.0
    assert field.readings["joke_landing"] > 0.0


def test_field_reads_panic_from_fed_messages():
    _label, events, _hour = synthetic_scenes()[1]  # a fight breaking out
    space = build_mud_space("The Tap", BASE_DESCRIPTION, events)
    field = space.read()
    assert field.readings["panic"] >= 0.5  # crosses the panic threshold
    assert classify(field, hour=23.0) == "panic"


# --------------------------------------------------------------------------- #
# three states -> three DIFFERENT descriptions                                #
# --------------------------------------------------------------------------- #
def test_three_states_three_different_descriptions():
    results = demo_three_states()
    assert len(results) == 3
    modes = [r["mode"] for r in results]
    # The three states land in the three expected body-language modes.
    assert modes == ["joyful", "panic", "closing"], modes
    # Same base description, three different spoken descriptions.
    tinted = [r["tinted"] for r in results]
    assert len(set(tinted)) == 3
    for r in results:
        assert r["base_description"] == BASE_DESCRIPTION
        assert BASE_DESCRIPTION in r["tinted"]  # mutation, not replacement


def test_three_states_are_the_light():
    results = demo_three_states()
    joyful, panic, closing = results

    # Joyful: laughter reverberates into the words.
    assert "laughter" in joyful["tinted"].lower()
    # Panic: newcomers arrive drenched, storm outside.
    assert "drenched" in panic["tinted"].lower()
    assert "storm" in panic["tinted"].lower() or "rain" in panic["tinted"].lower()
    # Closing: disco off, fluorescents on, people drift toward the exit.
    low = closing["tinted"].lower()
    assert "fluorescents" in low
    assert "exit" in low and "tab" in low


# --------------------------------------------------------------------------- #
# synthetic scene shapes (sanity for the demo feed)                           #
# --------------------------------------------------------------------------- #
def test_synthetic_scenes_shape():
    scenes = synthetic_scenes()
    assert len(scenes) == 3
    for _label, events, hour in scenes:
        assert events and all(len(e) == 2 for e in events)
        assert isinstance(hour, float)


if __name__ == "__main__":
    fns = [test_registry_resolves_mud_to_mudspace,
           test_tint_round_trips_through_mudspace,
           test_send_back_defaults_to_tint,
           test_field_reads_from_fed_messages,
           test_field_reads_panic_from_fed_messages,
           test_three_states_three_different_descriptions,
           test_three_states_are_the_light,
           test_synthetic_scenes_shape]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print("\nAll mud-live tests passed.")
