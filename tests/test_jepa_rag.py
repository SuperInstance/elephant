"""elephant — tests: JEPA-RAG, where JEPA readings are first-class
citizens alongside time and space stamps.

The captain's directive: "Think about a RAG with Jepa readings as
first-class citizens along side time and space stamps."

Moments are built from REAL fleet data — The Tap's trade-night
transcripts (ai-writings/tap-trades/2026-08-16), the captain's
speeches (ai-writings/speeches) — chunked into moments with readings
computed by the dial bank, plus a few boat moments (a storm watch, a
dawn watch, an empty wheelhouse, a galley fight) so the feeling
queries have something unmistakable to find. If the fleet writings
are not on this machine, the fleet tests skip; the synthetic tests
always run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from elephant.field import RoomField
from elephant.jepa_rag import (
    JEPA_DIAL_NAMES,
    JepaMemory,
    MomentHit,
    moment_from_room,
    moment_from_text,
    moments_from_markdown,
)
from elephant.room import Message, Room

TAP_DIR = "/home/eileen/projects/ai-writings/tap-trades/2026-08-16"
SPEECH_DIR = "/home/eileen/projects/ai-writings/speeches"
HAVE_FLEET = os.path.isdir(TAP_DIR) and os.path.isdir(SPEECH_DIR)

requires_fleet = pytest.mark.skipif(
    not HAVE_FLEET, reason="fleet writings not present on this machine")


# ---------------------------------------------------------------------- #
# The boats — real fleet rooms the transcripts never saw                  #
# ---------------------------------------------------------------------- #
def _fight_room() -> Room:
    """The galley fight: several voices, alarm words, a crowd reaction —
    the dials feel density, ripple, and panic that one message can't."""
    room = Room("galley")
    room.messages = [
        Message("welder", "You ran her aground and you know it!!!", ts=0.0),
        Message("carpenter", "Say that again and see what happens!", ts=2.0),
        Message("deck", "ALL HANDS — panic in the galley, somebody get the "
                        "wheelhouse — MAYDAY watch, NOW!!!", ts=4.0,
                reactions={"😱": 5}),
    ]
    room.messages.sort(key=lambda m: m.ts)
    return room


def _storm_room() -> Room:
    """The storm watch: the sea reminding everyone who's boss."""
    room = Room("wheelhouse")
    room.messages = [
        Message("[room]", "The wheelhouse window is white with spray — the "
                          "squall line three miles out and closing.", ts=0.0),
        Message("deck", "All hands on deck: batten the hatches, secure the "
                        "gear, NOW!!!", ts=3.0),
        Message("[room]", "The bow digs and the whole hull groans. MAYDAY "
                          "watch is up.", ts=6.0),
    ]
    room.messages.sort(key=lambda m: m.ts)
    return room


def _dawn_room() -> Room:
    """The dawn watch: the warmest moment the boats ever have."""
    room = Room("wheelhouse")
    room.messages = [
        Message("[room]", "Four a.m. and the wheelhouse is warm and quiet.",
                ts=0.0),
        Message("[room]", "Coffee steam, the compass glow, nothing on the "
                          "radar but the sounder's heartbeat.", ts=2.0),
        Message("watch", "A good night to be a boat — calm, held, at peace.",
                ts=4.0),
    ]
    room.messages.sort(key=lambda m: m.ts)
    return room


def _empty_room() -> Room:
    """The empty wheelhouse: nobody on watch but the instruments."""
    room = Room("wheelhouse")
    room.messages = [
        Message("[room]", "Three a.m., nobody on watch but the instruments. "
                          "Cold coffee in the cup, the heater ticking, the "
                          "sea flat and black.", ts=0.0),
        Message("[room]", "The radio hisses static. Nothing to see, nothing "
                          "to say — the room gone still and sharp.", ts=2.0),
    ]
    room.messages.sort(key=lambda m: m.ts)
    return room


DAY = 86400.0
ANCHOR = 1786924800.0          # 2026-08-16T00:00:00Z


def boat_moments() -> list:
    """The boats' moments: storm, dawn, empty wheelhouse, and the fight."""
    return [
        moment_from_room(_storm_room(), "wheelhouse", ts=ANCHOR - 3 * DAY,
                         meta={"source": "boat", "name": "storm watch"}),
        moment_from_room(_dawn_room(), "wheelhouse", ts=ANCHOR - 0.5 * DAY,
                         meta={"source": "boat", "name": "dawn watch"}),
        moment_from_room(_empty_room(), "wheelhouse", ts=ANCHOR - 1 * DAY,
                         meta={"source": "boat", "name": "empty wheelhouse"}),
        moment_from_room(_fight_room(), "wheelhouse", ts=ANCHOR - 7 * DAY,
                         meta={"source": "boat", "name": "the fight"}),
    ]


@requires_fleet
def _fleet_memory() -> JepaMemory:
    """The elephant's memory of the fleet: The Tap's trade nights, the
    speeches, and the boats."""
    mem = JepaMemory()
    tap_files = sorted(f for f in os.listdir(TAP_DIR) if f.endswith(".md"))
    for i, fn in enumerate(tap_files):
        for m in moments_from_markdown(os.path.join(TAP_DIR, fn), "the-tap",
                                       base_ts=ANCHOR + i * 3600.0):
            mem.ingest(m)
    speech_files = sorted(f for f in os.listdir(SPEECH_DIR)
                          if f.endswith(".md") and f[0].isdigit())
    for i, fn in enumerate(speech_files):
        for m in moments_from_markdown(os.path.join(SPEECH_DIR, fn), "speeches",
                                       base_ts=ANCHOR - 200 * DAY + i * 3600.0):
            mem.ingest(m)
    for m in boat_moments():
        mem.ingest(m)
    mem.index()
    return mem


# ---------------------------------------------------------------------- #
# Ingest + index                                                         #
# ---------------------------------------------------------------------- #
def test_ingest_and_index_builds_the_retrieval_structures():
    mem = JepaMemory()
    mem.ingest(moment_from_text("warm hello at the bar", "a", ts=0.0))
    mem.ingest(moment_from_text("cold goodbye in the rain", "b", ts=10.0))
    mem.index()
    assert len(mem) == 2
    assert mem.reading_vectors.shape == (2, len(JEPA_DIAL_NAMES))
    assert mem.field_matrix.shape == (2, 2)          # warmth + κ per moment
    assert mem.timestamps.shape == (2,)
    assert mem.space_stamps.shape == (2,)
    assert mem.reading_vectors.dtype == float
    # the field matrix really is the warmth + concentration of the moments
    assert mem.field_matrix[0, 0] > mem.field_matrix[1, 0]


def test_index_rebuilds_lazily_after_ingest():
    mem = JepaMemory()
    mem.ingest(moment_from_text("hello", "a"))
    assert len(mem.query_text("hello")) == 1         # auto-indexed
    mem.ingest(moment_from_text("the world says hello too", "a"))
    assert len(mem.query_text("world")) == 1         # stale index rebuilt
    assert len(mem) == 2


def test_ingest_requires_a_shadow():
    mem = JepaMemory()
    with pytest.raises(ValueError):
        mem.ingest({"readings": {"mood": 0.5}})


# ---------------------------------------------------------------------- #
# query_readings — the first-class-citizen query                         #
# ---------------------------------------------------------------------- #
@requires_fleet
def test_query_readings_finds_the_mood_high_panic_low_moment():
    mem = _fleet_memory()
    hits = mem.query_readings({"mood": 0.9, "joke_landing": 0.7, "panic": 0.0},
                              top_k=5)
    assert hits
    top = hits[0]
    # the intended reading profile — this is what a reading query is for
    assert top.readings["mood"] > 0.5
    assert top.readings["panic"] < 0.1
    # ranked best-first, honestly
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


@requires_fleet
def test_query_readings_finds_the_panic_moment():
    mem = _fleet_memory()
    hits = mem.query_readings({"panic": 0.9, "mood": -0.6, "volume": 0.8},
                              top_k=3)
    assert hits
    # the galley fight (or the storm) is the panickiest thing the
    # elephant remembers — a reading query surfaces it over the warm
    # tap nights
    assert hits[0].readings["panic"] > 0.3
    assert hits[0].readings["mood"] < hits[-1].readings["mood"] + 0.01 or \
        hits[0].readings["panic"] > hits[-1].readings["panic"]


def test_query_readings_accepts_partial_dicts_fields_and_vectors():
    mem = JepaMemory()
    mem.ingest(moment_from_text("a warm and glad and good night", "a"))
    mem.ingest(moment_from_text("a cold and dead and flat night", "a"))
    by_dict = mem.query_readings({"mood": 0.9}, top_k=1)[0]
    by_field = mem.query_readings(RoomField({"mood": 0.9}), top_k=1)[0]
    vec = np.zeros(len(JEPA_DIAL_NAMES))
    vec[0] = 0.9                                        # mood is dim 0
    by_vec = mem.query_readings(vec, top_k=1)[0]
    assert by_dict.index == by_field.index == by_vec.index
    assert by_dict.readings["mood"] > 0.5


def test_query_readings_range_constraints_are_literal_thresholds():
    mem = JepaMemory()
    mem.ingest(moment_from_text("warm calm night", "a", ts=0.0))
    mem.ingest(moment_from_text("loud panicky storm", "b", ts=1.0))
    hits = mem.query_readings({"mood": (0.4, 1.0), "panic": (0.0, 0.3)})
    # the warm calm night satisfies both bounds; the storm fails both
    # (mood 0.0, panic 0.4) — constraints are literal, no partial credit
    assert hits[0].text.startswith("warm calm")
    assert hits[0].score == 1.0
    assert hits[1].score == 0.0


@requires_fleet
def test_query_readings_constraints_never_let_panic_sneak_in():
    mem = _fleet_memory()
    hits = mem.query_readings({"panic": (0.0, 0.2)}, top_k=10)
    assert hits
    # a literal "panic < 0.2" — every top hit satisfies the bound,
    # even though a cosine closeness query might rank a panicky
    # moment nearby
    assert all(h.readings["panic"] <= 0.2 for h in hits)
    # and the galley fight (panic 0.82) cannot satisfy it at all
    assert all(h.readings["panic"] < 0.5 for h in hits)


# ---------------------------------------------------------------------- #
# query_field — the perfume query                                        #
# ---------------------------------------------------------------------- #
@requires_fleet
def test_query_field_retrieves_the_nearest_feeling():
    mem = _fleet_memory()
    warm_night = RoomField({
        "mood": 0.8, "joke_landing": 0.7, "panic": 0.0, "volume": 0.6,
        "presence": 0.7, "earnestness": 0.7, "cynicism": 0.2,
        "model_vs_code": 0.7, "vision": 0.6,
    })
    cold_watch = RoomField({
        "mood": -0.8, "panic": 0.0, "volume": 0.1, "cynicism": 0.8,
    })
    warm_hits = mem.query_field(warm_night, top_k=3)
    cold_hits = mem.query_field(cold_watch, top_k=3)
    assert warm_hits and cold_hits
    # the moment that feels like a warm night is warmer-feeling than
    # the moment that feels like a cold watch
    assert warm_hits[0].readings["mood"] > cold_hits[0].readings["mood"]
    assert warm_hits[0].readings["panic"] < 0.3
    # ... and the cold-feeling retrieval does not surface the panic moments
    assert all(h.readings["panic"] < 0.5 for h in cold_hits) or \
        cold_hits[0].readings["panic"] < warm_hits[-1].readings["panic"] + 0.3


# ---------------------------------------------------------------------- #
# query_time / query_space — the stamps as dimensions                    #
# ---------------------------------------------------------------------- #
def test_query_time_is_a_stamp_dimension():
    mem = JepaMemory()
    mem.ingest(moment_from_text("first watch", "a", ts=160.0))
    mem.ingest(moment_from_text("second watch", "a", ts=200.0))
    mem.ingest(moment_from_text("old fight", "a", ts=900.0))
    hits = mem.query_time((150.0, 250.0))
    # the hard filter: only moments inside the window
    assert all(150.0 <= h.ts <= 250.0 for h in hits)
    # ranked by proximity to the window's center (200.0)
    assert [h.text for h in hits] == ["second watch", "first watch"]
    # a single instant is an exact-match time stamp
    assert mem.query_time(200.0)[0].text == "second watch"
    assert mem.query_time((0.0, 50.0)) == []


def test_query_space_is_a_stamp_dimension():
    mem = JepaMemory()
    mem.ingest(moment_from_text("tap warm night", "the-tap", ts=10.0))
    mem.ingest(moment_from_text("wheelhouse cold", "wheelhouse", ts=5.0))
    mem.ingest(moment_from_text("wheelhouse storm", "wheelhouse", ts=50.0))
    hits = mem.query_space("wheelhouse")
    assert len(hits) == 2
    assert [h.text for h in hits] == ["wheelhouse storm", "wheelhouse cold"]
    assert all(h.space_id == "wheelhouse" for h in hits)
    assert mem.query_space("nowhere") == []


@requires_fleet
def test_query_space_returns_the_boat_moments():
    mem = _fleet_memory()
    hits = mem.query_space("wheelhouse")
    assert len(hits) == 4
    assert {h.meta.get("name") for h in hits} == {
        "storm watch", "dawn watch", "empty wheelhouse", "the fight"}
    # every hit is from the wheelhouse — the stamp did the filtering
    assert all(h.space_id == "wheelhouse" for h in hits)


# ---------------------------------------------------------------------- #
# query_combined — the captain's "alongside" made concrete               #
# ---------------------------------------------------------------------- #
def test_combined_renormalizes_weights_over_present_dimensions():
    mem = JepaMemory()
    mem.ingest(moment_from_text("warm toast at the tap", "tap", ts=0.0))
    mem.ingest(moment_from_text("cold storm in the wheelhouse", "boat", ts=100.0))
    hits = mem.query_combined({"readings": {"mood": 0.9}})
    assert hits[0].text.startswith("warm toast")     # readings weight = 1.0


def test_combined_zero_weight_dim_is_excluded():
    mem = JepaMemory()
    mem.ingest(moment_from_text("warm toast", "tap", ts=0.0))
    mem.ingest(moment_from_text("cold storm", "boat", ts=100.0))
    hits = mem.query_combined(
        {"text": "storm", "readings": {"mood": 0.9}},
        weights={"text": 0.0, "readings": 1.0})
    # text contributes nothing: the warm moment wins on readings alone,
    # even though the words say "storm"
    assert hits[0].readings["mood"] > 0.5
    assert hits[0].score > 0.0


def test_combined_uses_all_four_dimensions():
    mem = JepaMemory()
    mem.ingest(moment_from_text("warm toast at the tap", "tap", ts=25.0))
    mem.ingest(moment_from_text("cold storm in the wheelhouse", "boat", ts=100.0))
    mem.ingest(moment_from_text("another warm toast", "tap", ts=1000.0))
    hits = mem.query_combined(
        {"text": "toast", "space": "tap", "ts": (0.0, 50.0),
         "readings": {"mood": 0.9}},
        top_k=3)
    # the first moment sits at the window's center and matches every
    # dimension — it wins the weighted combination
    assert hits[0].text.startswith("warm toast")
    assert hits[0].space_id == "tap" and hits[0].ts == 25.0
    # the second tap moment loses on time; the boat moment loses on
    # space, readings, and time
    assert hits[0].score > hits[1].score >= hits[2].score


@requires_fleet
def test_combined_reading_heavy_beats_text_only_for_a_feeling_query():
    mem = _fleet_memory()
    feeling = "where did things get loud and scary"
    text_only = mem.query_text(feeling, top_k=5)
    combined = mem.query_combined(
        {"text": feeling, "readings": {"panic": 0.9, "volume": 0.8,
                                       "mood": -0.5}},
        weights={"readings": 0.8, "text": 0.2}, top_k=5)
    assert text_only and combined
    # the reading-heavy query surfaces the intended profile — panic —
    # better than the words alone ever could
    assert combined[0].readings["panic"] > 0.3
    assert combined[0].readings["panic"] >= text_only[0].readings["panic"]


@requires_fleet
def test_combined_space_and_time_stamps_answer_where_and_when():
    mem = _fleet_memory()
    hits = mem.query_combined(
        {"space": "wheelhouse", "ts": (ANCHOR - 8 * DAY, ANCHOR - 2 * DAY)},
        top_k=3)
    # the wheelhouse moments inside last week's window rank first:
    # space matched AND time matched
    assert hits[0].space_id == "wheelhouse"
    assert ANCHOR - 8 * DAY <= hits[0].ts <= ANCHOR - 2 * DAY


# ---------------------------------------------------------------------- #
# Honesty: the first-class citizens ride along                           #
# ---------------------------------------------------------------------- #
@requires_fleet
def test_hits_carry_their_terrain_context():
    mem = _fleet_memory()
    queries = [
        lambda: mem.query_text("fight"),
        lambda: mem.query_readings({"panic": 0.9}),
        lambda: mem.query_field({"mood": 0.8, "panic": 0.0}),
        lambda: mem.query_time((0.0, 1e18)),
        lambda: mem.query_space("the-tap"),
        lambda: mem.query_combined({"space": "wheelhouse",
                                    "readings": {"panic": 0.9}}),
    ]
    for q in queries:
        hits = q()
        assert hits, "every query type returns hits on the fleet memory"
        for h in hits:
            assert isinstance(h, MomentHit)
            assert h.text                                  # the shadow
            assert h.readings                               # the readings
            assert h.space_id                               # the space stamp
            assert h.meta is not None
            # the full reading vector rides along, every dial of it
            assert set(JEPA_DIAL_NAMES) <= set(h.readings)
            assert h.vector.shape == (len(JEPA_DIAL_NAMES),)
            assert h.vector.dtype == float


# ---------------------------------------------------------------------- #
# Chunking fleet transcripts into moments                                #
# ---------------------------------------------------------------------- #
@requires_fleet
def test_moments_from_markdown_chunks_a_real_transcript():
    path = os.path.join(TAP_DIR, "evening-at-the-tap.md")
    moments = moments_from_markdown(path, "the-tap", base_ts=1000.0, step=300.0)
    assert len(moments) >= 3
    for m in moments:
        assert m["space_id"] == "the-tap"
        assert m["text"].strip()
        assert set(JEPA_DIAL_NAMES) <= set(m["readings"])
        assert m["ts"] >= 1000.0
    # timestamps keep the chunk order — a stamp dimension, not noise
    tss = [m["ts"] for m in moments]
    assert tss == sorted(tss)


@requires_fleet
def test_fleet_memory_is_built_from_real_fleet_data():
    mem = _fleet_memory()
    spaces = set(mem.spaces())
    assert "the-tap" in spaces
    assert "speeches" in spaces
    assert "wheelhouse" in spaces
    assert len(mem) > 30                       # many moments, all read


def test_boat_readings_are_what_the_dials_feel():
    # the fight is panicky and loud; the dawn watch is warm and calm —
    # computed by the dial bank, not hand-set
    fight = moment_from_room(_fight_room(), "wheelhouse", ts=0.0)
    dawn = moment_from_room(_dawn_room(), "wheelhouse", ts=1.0)
    assert fight["readings"]["panic"] > 0.3
    assert fight["readings"]["panic"] > dawn["readings"]["panic"]
    assert dawn["readings"]["mood"] > 0.5
    # the dawn watch is calm — its only panic is the message-rate floor
    # of the stampede sense (three messages in four seconds)
    assert dawn["readings"]["panic"] < 0.2
