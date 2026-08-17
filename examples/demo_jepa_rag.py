#!/usr/bin/env python3
"""JEPA-RAG — the elephant remembers every room by FEELING.

The captain's directive (exact words):

    "Think about a RAG with Jepa readings as first-class citizens
    along side time and space stamps."

A normal RAG indexes text and retrieves similar text. This demo builds
a memory where each MOMENT is a shadow with its terrain context — the
JEPA reading vector, the time stamp, the space stamp — and then asks
it four ways:

  1. "Find the moment that felt like The Tap on a good night"
     — query_field with a warm field (the perfume query)
  2. "Find the fight" — query_readings with panic high (the
     first-class-citizen query)
  3. "What was the wheelhouse like?" — query_space (the stamp as a
     dimension)
  4. "Find a moment like now" — nearest neighbor in JEPA space to a
     current reading (the perfume that takes you to grandma's shop)
  5. "The wheelhouse, last week" — query_combined: space + time
     stamps alongside the feeling, weighted

Every hit prints the witness (the shadow) WITH its reading vector —
the first-class citizens ride along, enough to agree on the action
(docs/terrain-2026-08-17.md).

The memory is built from the fleet's real rooms: The Tap's trade-night
transcripts (2026-08-16), the captain's speeches, and the boats'
wheelhouse. If the writings are missing on this machine the demo falls
back to the boats alone, so it always runs.

Run:  python3 examples/demo_jepa_rag.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elephant.field import RoomField
from elephant.jepa_rag import (
    JEPA_DIAL_NAMES,
    JepaMemory,
    moment_from_room,
    moment_from_text,
    moments_from_markdown,
)
from elephant.room import Message, Room

TAP_DIR = "/home/eileen/projects/ai-writings/tap-trades/2026-08-16"
SPEECH_DIR = "/home/eileen/projects/ai-writings/speeches"

DAY = 86400.0
ANCHOR = 1786924800.0          # 2026-08-16T00:00:00Z — the trade nights


# ---------------------------------------------------------------------- #
# The boats — rooms the transcripts never saw                             #
# ---------------------------------------------------------------------- #
def _fight_room() -> Room:
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


def boat_moments() -> list:
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


# ---------------------------------------------------------------------- #
# Building the memory from the fleet's real rooms                        #
# ---------------------------------------------------------------------- #
def build_fleet_memory() -> JepaMemory:
    mem = JepaMemory()

    # The Tap's trade nights — every transcript, chunked into moments,
    # each chunk read by the dial bank into its own JEPA vector.
    if os.path.isdir(TAP_DIR):
        files = sorted(f for f in os.listdir(TAP_DIR) if f.endswith(".md"))
        for i, fn in enumerate(files):
            for m in moments_from_markdown(os.path.join(TAP_DIR, fn),
                                           "the-tap",
                                           base_ts=ANCHOR + i * 3600.0):
                mem.ingest(m)

    # The captain's speeches — the fleet's public voice.
    if os.path.isdir(SPEECH_DIR):
        files = sorted(f for f in os.listdir(SPEECH_DIR)
                       if f.endswith(".md") and f[0].isdigit())
        for i, fn in enumerate(files):
            for m in moments_from_markdown(os.path.join(SPEECH_DIR, fn),
                                           "speeches",
                                           base_ts=ANCHOR - 200 * DAY + i * 3600.0):
                mem.ingest(m)

    # The boats — the wheelhouse remembers its nights too.
    for m in boat_moments():
        mem.ingest(m)

    mem.index()
    return mem


# ---------------------------------------------------------------------- #
# Printing — the witness with its terrain context                        #
# ---------------------------------------------------------------------- #
def _show(title: str, hits, excerpt: int = 300) -> None:
    print(f"\n  {title}")
    for h in hits:
        print(f"    {h.board()}")
        print(f"      readings: {h.reading_line()}")
        shadow = h.text if len(h.text) <= excerpt else h.text[:excerpt] + " …"
        print(f"      shadow: {shadow!r}")


def _current_reading() -> dict:
    """'Now' — a mid-session read: slightly warm, low panic, some
    earnestness in the air."""
    return {
        "mood": 0.35, "volume": 0.45, "earnestness": 0.65,
        "cynicism": 0.25, "joke_landing": 0.4, "panic": 0.05,
        "presence": 0.5, "model_vs_code": 0.6, "vision": 0.5,
    }


def main() -> None:
    memory = build_fleet_memory()
    print("=" * 78)
    print("JEPA-RAG — JEPA readings as first-class citizens,")
    print("beside time stamps and space stamps.")
    print("=" * 78)
    print(memory.summary())

    # 1 — the perfume query: the felt-like query, by field
    print("\n" + "─" * 78)
    print("Q1: 'Find the moment that felt like The Tap on a good night'")
    print("    query_field(warm field) — nearest neighbors in JEPA space")
    warm_night = RoomField({
        "mood": 0.8, "joke_landing": 0.7, "panic": 0.0, "volume": 0.6,
        "presence": 0.7, "earnestness": 0.7, "cynicism": 0.2,
        "model_vs_code": 0.7, "vision": 0.6,
    })
    _show("the warm moments, with their shadows:",
          memory.query_field(warm_night, top_k=3))

    # 2 — the first-class-citizen query: by reading profile
    print("\n" + "─" * 78)
    print("Q2: 'Find the fight'")
    print("    query_readings(panic high, mood low) — the reading IS the query")
    _show("the panic moments:",
          memory.query_readings(
              {"panic": 0.9, "volume": 0.8, "mood": -0.6, "joke_landing": -0.4},
              top_k=3))

    # 3 — the space stamp as a dimension
    print("\n" + "─" * 78)
    print("Q3: 'What was the wheelhouse like?'")
    print("    query_space('wheelhouse') — the stamp is the query")
    _show("the wheelhouse's moments, newest first:",
          memory.query_space("wheelhouse"), excerpt=220)

    # 4 — the moment like now: nearest neighbor in JEPA space
    print("\n" + "─" * 78)
    print("Q4: 'Find a moment like now'")
    print("    query_field(current reading) — the perfume that takes you")
    print("    to grandma's shop")
    _show("the moments that feel most like right now:",
          memory.query_field(_current_reading(), top_k=3))

    # 5 — the full RAG query: everything alongside
    print("\n" + "─" * 78)
    print("Q5: 'The wheelhouse, last week'")
    print("    query_combined(space + time + readings) — text 0.3, readings")
    print("    0.5, time 0.1, space 0.1 — the captain's 'alongside'")
    hits = memory.query_combined(
        {"space": "wheelhouse",
         "ts": (ANCHOR - 8 * DAY, ANCHOR - 2 * DAY),
         "readings": {"panic": 0.9, "mood": -0.6}},
        top_k=3)
    _show("last week in the wheelhouse:",
          hits, excerpt=220)

    print("\n" + "=" * 78)
    print("Every shadow came back with its terrain context — the reading")
    print("vector, the time stamp, the space stamp. The elephant remembers")
    print("every room it has ever been in, and retrieves by FEELING.")
    print("Enough to agree on the action.")
    print("=" * 78)


if __name__ == "__main__":
    main()
