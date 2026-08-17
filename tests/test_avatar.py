"""elephant — tests: the round-character avatar.

An `Avatar` starts FLAT (a persona seed + a preset dial prior) and becomes
ROUND by attending themed Tap nights: it speaks lines, its PersonalElephant
senses the room, it self-tunes toward felt engagement (the guitarist
principle — tastes diverge under the same room), it binds attachments (the
moments that meant something — the perfume that takes you to grandma's
shop), and it enriches its persona one note per night.

These tests prove the learning is real: profiles move, presets diverge,
speech carries memory, the sheet tells the arc, and the monologue pulses
even in silence.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elephant.avatar import PRESETS, Avatar
from elephant.field import DIAL_NAMES
from elephant.tapnight_themes import THEMES

COMEDIAN_SEED = ("I'm Marty — I make the room laugh, and I need the laugh "
                 "to land.")
BROODER_SEED = ("I'm Ira — I sit with the heavy things and I don't trust "
                "a room that's too warm.")
WALLFLOWER_SEED = ("I'm Wren — I'd rather be seen than heard, and I watch "
                   "the door.")

ATTACHMENT_FIELDS = {"event_key", "night", "line", "room", "moved", "felt",
                     "memory"}


def _session(key: str):
    """A started, seeded themed night ready for avatars to walk into."""
    theme = THEMES[key]
    s = theme.make_session()
    s.start_session()
    theme.seed(s)
    return s


# ---------------------------------------------------------------------- #
# The learning is real                                                   #
# ---------------------------------------------------------------------- #
def test_avatar_learns_across_themed_nights():
    """An avatar attending open mic -> singles -> ttrpg ends with a
    DIFFERENT dial profile than it started: it learned, it's round."""
    bro = Avatar("Ira", BROODER_SEED, preset="brooder")
    start = bro.elephant.dial_weights.copy()

    s = _session("open_mic")
    bro.attend(s, [
        ("Cold water, warm light — the room hums between them like a caught "
         "breath.", {"❤️": 2}),
        ("The lamplight keeps nothing; it just glows. Bright and good and "
         "gone.", {"❤️": 2}),
    ], night_key="open_mic")
    s.end_session()

    s = _session("singles")
    bro.attend(s, [
        ("I pause just inside the door, take a deep breath, and let my eyes "
         "adjust to the energy of the space.", {"👍": 1}),
        ("A faint trace of cedarwood, dry and warm, mingled with the soft "
         "tang of rain-soaked earth — comfort and anticipation.", {"😏": 1}),
    ], night_key="singles")
    s.end_session()

    s = _session("ttrpg")
    bro.attend(s, [
        ("I don't know where we are. The depth marks on the chart don't "
         "match.", {"❤️": 1}),
        ("You saw the hull breathe when it laughed — please tell me I "
         "imagined that.", {"❤️": 2}),
    ], night_key="ttrpg")
    s.end_session()

    final = bro.elephant.dial_weights
    # It learned: the profile moved, and stays a probability distribution.
    assert np.linalg.norm(final - start) > 0.05
    assert abs(float(final.sum()) - 1.0) < 1e-9
    # It has a history: three nights, three notes, attachments bound.
    assert len(bro.nights) == 3
    assert len(bro.character_notes) == 3
    assert len(bro.elephant.attachments) > 0
    # The brooder became MORE sensitive to the fear it carries.
    assert final[DIAL_NAMES.index("panic")] > 0.20


def test_two_avatars_same_night_diverge():
    """Two avatars with different presets attending the SAME night end
    with DIFFERENT profiles — the guitarist principle: same room, two
    guitars, two tastes."""
    com = Avatar("Marty", COMEDIAN_SEED, preset="comedian")
    bro = Avatar("Ira", BROODER_SEED, preset="brooder")
    com_start = com.elephant.dial_weights.copy()
    bro_start = bro.elephant.dial_weights.copy()

    s = _session("open_mic")
    com.attend(s, [
        ("I walk into the Tap and the elephant is already at the bar. "
         "'Buddy, that's a lot of trunk.' The whole room goes HAHA.",
         {"😂": 3}),
        ("The room's so warm tonight the elephant took its coat off. See? "
         "Even the elephant came for the jokes.", {"😂": 2, "❤️": 1}),
    ], night_key="open_mic")
    bro.attend(s, [
        ("I wrote this for the room, honestly — every chair keeps a story "
         "it's glad to tell, and I meant every word.", {"❤️": 2}),
        ("Cold water, warm light — the room hums between them like a caught "
         "breath.", {"❤️": 2}),
    ], night_key="open_mic")
    s.end_session()

    final = np.linalg.norm(com.elephant.dial_weights - bro.elephant.dial_weights)
    initial = np.linalg.norm(com_start - bro_start)
    # Tastes diverged under the SAME room — not converged, not identical.
    assert final > initial
    assert final > 0.2
    # Each sharpened toward its OWN guitar — and away from the other's.
    assert com.elephant.dial_weights[DIAL_NAMES.index("joke_landing")] > 0.45
    assert bro.elephant.dial_weights[DIAL_NAMES.index("panic")] > 0.20
    assert com.elephant.dial_weights[DIAL_NAMES.index("panic")] < 0.20
    assert bro.elephant.dial_weights[DIAL_NAMES.index("joke_landing")] < 0.45


def test_learning_is_deterministic():
    """Same script, same presets -> identical profiles AND identical
    attachments. Roundness is learned, not rolled."""
    a = Avatar("M", COMEDIAN_SEED, preset="comedian")
    b = Avatar("M", COMEDIAN_SEED, preset="comedian")
    lines = [
        ("I walk into the Tap and the elephant is already at the bar. "
         "'Buddy, that's a lot of trunk.' HAHA.", {"😂": 3}),
        ("The room's so warm the elephant took its coat off. See? Even the "
         "elephant came for the jokes.", {"😂": 2, "❤️": 1}),
    ]
    sa, sb = _session("open_mic"), _session("open_mic")
    a.attend(sa, lines, night_key="open_mic")
    b.attend(sb, lines, night_key="open_mic")
    assert np.allclose(a.elephant.dial_weights, b.elephant.dial_weights)
    # the learning is identical all the way down: same moments, same memories
    assert [x["event_key"] for x in a.elephant.attachments.values()] == \
           [x["event_key"] for x in b.elephant.attachments.values()]
    assert [x["memory"] for x in a.elephant.attachments.values()] == \
           [x["memory"] for x in b.elephant.attachments.values()]


# ---------------------------------------------------------------------- #
# Speech carries memory                                                  #
# ---------------------------------------------------------------------- #
def test_speak_reflects_attachments():
    """A remembered moment appears in what the avatar says — the flat seed
    is no longer the whole voice."""
    com = Avatar("Marty", COMEDIAN_SEED, preset="comedian")
    flat = com.speak()
    assert COMEDIAN_SEED.rstrip(".") in flat
    assert "I keep remembering" not in flat

    s = _session("open_mic")
    com.attend(s, [
        ("I walk into the Tap and the elephant is already at the bar. "
         "'Buddy, that's a lot of trunk.' The whole room goes HAHA.",
         {"😂": 3}),
    ], night_key="open_mic")
    s.end_session()

    memory = list(com.elephant.attachments.values())[0]["memory"]
    round_text = com.speak("the open mic")
    # The drift is real: the round voice is not the flat voice.
    assert round_text != flat
    # And the memory is IN the voice — the moment speaks through it.
    assert memory.rstrip(".") in round_text


def test_attend_accepts_aligned_reactions():
    """Bare string lines pair with an aligned reactions list."""
    wall = Avatar("Wren", WALLFLOWER_SEED, preset="wallflower")
    s = _session("ttrpg")
    wall.attend(s,
                ["Blimey, did you *talk* just now?! I'm Pip, cabin boy!",
                 "WE DID IT! The monster's gone and it sang like it was "
                 "happy about it!"],
                reactions=[{"😄": 1}, {"😂": 2, "👏": 2}],
                night_key="ttrpg")
    s.end_session()
    assert len(wall.elephant.attachments) > 0
    assert len(wall.nights) == 1
    # reactions reached the session (heat tracked on the avatar's messages)
    heat = sum(m.reaction_heat for m in s.room.messages
               if m.author == "Wren")
    assert heat == 5  # 1 + (2 + 2)


# ---------------------------------------------------------------------- #
# The sheet tells the arc                                                #
# ---------------------------------------------------------------------- #
def test_character_sheet_coherent():
    """The sheet is the round character: persona, dial drift, attachments
    with their moments, nights, and a readable through-line."""
    wall = Avatar("Wren", WALLFLOWER_SEED, preset="wallflower")
    s = _session("ttrpg")
    wall.attend(s, [
        ("Blimey, did you *talk* just now?! I'm Pip, cabin boy, and I "
         "promise I'll be the best swabber you've ever had!", {"😄": 1}),
        ("WE DID IT! The monster's gone and it sang like it was happy "
         "about it!", {"😂": 2, "👏": 2}),
    ], night_key="ttrpg")
    s.end_session()

    sheet = wall.character_sheet()
    assert set(sheet) == {"name", "persona", "dial_profile", "sensitive_to",
                          "attachments", "nights_attended", "through_line"}
    assert sheet["name"] == "Wren"
    # persona: the seed is intact, the notes are the enrichment
    assert sheet["persona"]["seed"] == WALLFLOWER_SEED
    assert len(sheet["persona"]["notes"]) == 1
    assert sheet["persona"]["current"] == wall.persona
    assert WALLFLOWER_SEED in sheet["persona"]["current"]
    # dial profile: started, now, and the drift — the learning is visible
    for key in ("started_with", "now", "drift"):
        assert set(sheet["dial_profile"][key]) == set(DIAL_NAMES)
    assert any(abs(v) > 1e-6 for v in sheet["dial_profile"]["drift"].values())
    # sensitive_to: top dials, in order, weights descending
    top = sheet["sensitive_to"]
    assert len(top) == 3
    assert top[0]["weight"] >= top[1]["weight"] >= top[2]["weight"]
    # attachments carry the full moment
    assert sheet["attachments"]
    for att in sheet["attachments"]:
        assert set(att) == ATTACHMENT_FIELDS
        assert att["line"] and att["memory"]
    # nights: the arc has chapters
    assert len(sheet["nights_attended"]) == 1
    assert sheet["nights_attended"][0]["attachments"]
    assert set(sheet["nights_attended"][0]["attachments"]) == {
        a["event_key"] for a in sheet["attachments"]}
    # the through-line is readable: who walked in, what they lean, what they keep
    tl = sheet["through_line"]
    assert "Wren" in tl and "at The Tap" in tl
    assert "walked in as" in tl and "keeps" in tl
    assert sheet["dial_profile"]["now"][top[0]["dial"]] > 0


def test_fresh_avatar_sheet_is_flat():
    """Before any nights: no notes, no attachments, a 'still flat'
    through-line — the seed is the whole character."""
    com = Avatar("Marty", COMEDIAN_SEED, preset="comedian")
    sheet = com.character_sheet()
    assert sheet["persona"]["current"] == COMEDIAN_SEED
    assert sheet["persona"]["notes"] == []
    assert sheet["attachments"] == []
    assert sheet["nights_attended"] == []
    assert "still flat" in sheet["through_line"]
    assert np.allclose(np.asarray(list(sheet["dial_profile"]["drift"].values())),
                       np.zeros(len(DIAL_NAMES)))


# ---------------------------------------------------------------------- #
# Silence is not empty                                                   #
# ---------------------------------------------------------------------- #
def test_monologue_runs_on_pulses_in_silence():
    """The avatar's internal monologue pulses even when it says nothing —
    the room moves, the avatar senses, the log grows."""
    wall = Avatar("Wren", WALLFLOWER_SEED, preset="wallflower")
    s = _session("open_mic")
    wall.attend(s, [
        ("Quiet — the room is full, thrumming like a struck string.",
         {"❤️": 1}),
    ], night_key="open_mic")
    spoken_before = len([m for m in s.room.messages if m.author == "Wren"])

    first = wall.monologue(s)          # pulse 1 — the ear warms
    assert isinstance(first, str) and first
    # the night goes on; the avatar says nothing
    s.speak("poet", "The lamplight keeps nothing; it just glows. Bright "
                    "and good and gone.", reactions={"❤️": 2})
    second = wall.monologue(s)         # pulse 2 — direction
    s.speak("comic", "Haha — even the elephant came for the jokes!",
            reactions={"😂": 3})
    third = wall.monologue(s)          # pulse 3 — rate of change

    assert len(wall.monologue_log) == 3
    assert all(m["text"] for m in wall.monologue_log)
    assert first != second and second != third   # the sensing moved
    # it never opened its mouth in the room — silence, but not emptiness
    spoken_after = len([m for m in s.room.messages if m.author == "Wren"])
    assert spoken_after == spoken_before == 1
    # and the monologue carries the avatar's own ear
    assert "my ear leans" in third


def test_persona_is_enriched_night_by_night():
    """The seed stays, the notes accumulate — the character rounds."""
    bro = Avatar("Ira", BROODER_SEED, preset="brooder")
    s = _session("singles")
    bro.attend(s, [("I pause just inside the door and let my eyes adjust.",
                    {"👍": 1})], night_key="singles")
    s.end_session()
    s = _session("open_mic")
    bro.attend(s, [("Cold water, warm light — the room hums between them.",
                    {"❤️": 2})], night_key="open_mic")
    s.end_session()

    assert bro.persona_prompt == BROODER_SEED          # the flat seed is intact
    assert len(bro.character_notes) == 2               # one note per night
    assert "singles" in bro.persona and "open_mic" in bro.persona
    assert len(bro.character_sheet()["persona"]["notes"]) == 2
    # each preset is a distinct first guitar
    assert set(PRESETS) == {"comedian", "brooder", "wallflower"}


if __name__ == "__main__":
    for fn in [test_avatar_learns_across_themed_nights,
               test_two_avatars_same_night_diverge,
               test_learning_is_deterministic,
               test_speak_reflects_attachments,
               test_attend_accepts_aligned_reactions,
               test_character_sheet_coherent,
               test_fresh_avatar_sheet_is_flat,
               test_monologue_runs_on_pulses_in_silence,
               test_persona_is_enriched_night_by_night]:
        fn()
        print(f"PASS {fn.__name__}")
    print("\nAll avatar tests passed.")
