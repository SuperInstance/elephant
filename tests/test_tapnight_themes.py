"""elephant — tests: the themed Tap nights (four rooms, four elephants)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elephant.field import DIAL_NAMES, RoomField
from elephant.tapnight import Participant
from elephant.tapnight_themes import THEMES, Theme


def _cast(theme: Theme):
    return theme.cast()


def _seeded_session(theme: Theme):
    """A session started and seeded with the theme's opening tone."""
    s = theme.make_session()
    s.start_session()
    theme.seed(s)
    return s


def _raw_profile(theme: Theme) -> dict:
    """The starter tones' own reading — the dial bank read straight off the
    seed messages, before charisma/acclimation bends the field. This is what
    the theme's *content* produces; the effective field (charisma + cast
    warming to it) is exercised separately in the end-to-end test."""
    s = _seeded_session(theme)
    return dict(zip(DIAL_NAMES, s.raw_field().vector()))


# ---------------------------------------------------------------------- #
# Registry + cast                                                        #
# ---------------------------------------------------------------------- #
def test_theme_registry_has_four_rooms():
    assert set(THEMES) == {"improv", "open_mic", "singles", "speed_dating", "trivia", "ttrpg"}
    for key, theme in THEMES.items():
        assert isinstance(theme, Theme)
        assert theme.key == key
        assert isinstance(theme.description, str) and theme.description
        assert isinstance(theme.room_tone, list) and theme.room_tone
        assert isinstance(theme.prompts, dict) and theme.prompts


def test_each_theme_builds_a_valid_cast():
    for key, theme in THEMES.items():
        cast = _cast(theme)
        assert cast, f"{key}: empty cast"
        names = set()
        for p in cast:
            assert isinstance(p, Participant), f"{key}: {p!r}"
            assert p.name not in names, f"{key}: duplicate {p.name}"
            names.add(p.name)
            w = p.dial_weights
            assert w.shape == (len(DIAL_NAMES),), f"{key}: {p.name} weight shape"
            assert (w >= 0.0).all(), f"{key}: {p.name} negative weights"
            assert abs(float(w.sum()) - 1.0) < 1e-6, f"{key}: {p.name} not normalized"


def test_each_cast_has_distinct_archetype_weights():
    # The point of a themed cast is DIFFERENT guitarists — no two archetypes
    # in a room should carry identical dial_weights.
    for key, theme in THEMES.items():
        cast = _cast(theme)
        ws = np.stack([p.dial_weights for p in cast])
        # pairwise spread within the cast
        spread = 0.0
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                spread = max(spread, float(np.linalg.norm(ws[i] - ws[j])))
        assert spread > 0.05, f"{key}: archetype weights too similar ({spread:.3f})"


# ---------------------------------------------------------------------- #
# End-to-end: speak -> field -> tune                                     #
# ---------------------------------------------------------------------- #
def test_themed_session_runs_end_to_end():
    for key, theme in THEMES.items():
        s = theme.make_session()
        assert len(s.participants) == len(_cast(theme))
        s.start_session()
        theme.seed(s)

        # field is a real RoomField over all 7 dials, within bounds
        f = s.room_field()
        assert isinstance(f, RoomField)
        assert set(f.readings) == set(DIAL_NAMES)
        assert -1.0 <= f.warmth() <= 1.0
        assert f.concentration() >= 0.0

        # tune every participant: weights stay valid (non-negative, sum to 1)
        for name in s.participants:
            s.tune_participant(name)
            w = s.participants[name].dial_weights
            assert (w >= 0.0).all(), f"{key}: {name} negative after tune"
            assert abs(float(w.sum()) - 1.0) < 1e-6, f"{key}: {name} drift after tune"

        line = s.end_session()
        assert isinstance(line, str) and line


# ---------------------------------------------------------------------- #
# The four rooms read DIFFERENTLY on their starter tones                 #
# ---------------------------------------------------------------------- #
def test_starter_tones_produce_distinct_field_profiles():
    prof = {key: _raw_profile(theme) for key, theme in THEMES.items()}

    # Trivia is the earnest room; open mic is the laughter room. Trivia's
    # earnestness exceeds open mic's joke_landing — the intended direction.
    assert prof["trivia"]["earnestness"] > prof["improv"]["joke_landing"], (
        prof["trivia"]["earnestness"], prof["improv"]["joke_landing"])

    # Singles is the thrumming room (everyone present, watching); TTRPG is
    # the panicked room. Singles' presence exceeds TTRPG's panic.
    assert prof["singles"]["presence"] > prof["ttrpg"]["panic"], (
        prof["singles"]["presence"], prof["ttrpg"]["panic"])


def test_each_theme_reads_as_its_intended_room():
    prof = {key: _raw_profile(theme) for key, theme in THEMES.items()}

    # TTRPG is the panic room — a tense roll spikes panic above every other night.
    assert prof["ttrpg"]["panic"] > prof["trivia"]["panic"]
    assert prof["ttrpg"]["panic"] > prof["improv"]["panic"]
    assert prof["ttrpg"]["panic"] > prof["singles"]["panic"]

    # Open mic is the laughter room — jokes land (positive) and beat every other night.
    assert prof["improv"]["joke_landing"] > 0.0
    assert prof["improv"]["joke_landing"] > prof["trivia"]["joke_landing"]
    assert prof["improv"]["joke_landing"] > prof["ttrpg"]["joke_landing"]
    assert prof["improv"]["joke_landing"] > prof["singles"]["joke_landing"]

    # Trivia is suspicious of wrong answers — cynicism leads the field there.
    assert prof["trivia"]["cynicism"] > prof["improv"]["cynicism"]
    assert prof["trivia"]["cynicism"] > prof["ttrpg"]["cynicism"]

    # Singles is warm-but-nervous: warmer than the suspicious trivia room,
    # with tentative (not roaring) laughter.
    assert prof["singles"]["mood"] > prof["trivia"]["mood"]
    assert prof["singles"]["joke_landing"] < prof["improv"]["joke_landing"]


def test_singles_chemistry_is_observable():
    # The observable in a singles room is chemistry: two agents reading the
    # same warm room through different dials. maya leans mood; rowan leans
    # presence — their priors genuinely differ.
    cast = {p.name: p for p in _cast(THEMES["singles"])}
    assert set(cast) >= {"maya", "rowan"}
    maya = cast["maya"].dial_weights
    rowan = cast["rowan"].dial_weights
    assert int(np.argmax(maya)) != int(np.argmax(rowan)), (
        "maya and rowan should read the room through different dials")
    assert np.linalg.norm(maya - rowan) > 0.05


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} tap-night theme tests passed.")
    sys.exit(1 if failed else 0)
