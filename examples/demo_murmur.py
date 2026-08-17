#!/usr/bin/env python3
"""THE CAVE WALL DEMO — murmur, the shadow writer.

The captain's Terrain reframing (docs/terrain-2026-08-17.md):

    The true state is the Terrain. What a human (or an agent) actually
    sees — the trail of words, the motions, the outputs, the internal
    monologues — are the Shadows on the cave wall: witness marks of
    the terrain's activities. Each shadow is a lossy projection of the
    terrain — enough to recognize, never enough to be complete. The
    shadow is not the thinking. The shadow is the witness.

Murmur-agent is the fleet's all-night thinker: every thought becomes a
git commit. This demo is the cave wall writing itself — murmur given
the elephant's senses:

  1. THE TERRAIN — one evening at The Tap: a warm room, a fire, a
     quiet deck. The dials feel it; the PulseLoop reads it as a whole
     hand (two numbers show direction, three+ show rate of change).
  2. THE WALL — a MurmurJournal in /tmp (plain dir — no git needed).
  3. THE OVERNIGHT WATCH — every pulse's internal monologue becomes a
     witness mark: the silent thinking committed WITH that pulse's
     perception readings as terrain front-matter. The wall writes
     itself, informed by the terrain.
  4. THE ELEPHANT READS THE WALL — the journal as a room: field +
     dials from the shadow-trail.
  5. RETRIEVAL BY FEELING — "when did murmur last feel like this?"
     A panic reading finds the panic witness. The deadband rings.
  6. THE WITNESSES — the shadow-trail with its terrain front-matter.

Run:  python3 examples/demo_murmur.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elephant.dial import DialBank
from elephant.dials import DEFAULT_DIALS
from elephant.murmur import MurmurJournal, MurmurSpace, murmurize_pulse
from elephant.pulse import PulseLoop
from elephant.space import ChatSpace

BANK = DialBank(DEFAULT_DIALS)
SEP = "-" * 72


def line() -> None:
    print(SEP)


def main() -> None:
    print()
    print("  THE CAVE WALL DEMO — murmur, the shadow writer")
    print("  the wall writes itself, informed by the terrain")
    line()

    # ------------------------------------------------------------------ #
    # 1. THE TERRAIN — one evening at The Tap                            #
    # ------------------------------------------------------------------ #
    print("\n[1] THE TERRAIN — one evening at The Tap")
    tap = ChatSpace("the-tap-overnight")
    loop = PulseLoop("murmur", tap, period=2.0)

    # ------------------------------------------------------------------ #
    # 2. THE WALL — a plain-dir journal in /tmp                          #
    # ------------------------------------------------------------------ #
    wall = tempfile.mkdtemp(prefix="murmur-cave-wall-")
    journal = MurmurJournal(wall, space_id="the-tap")
    print(f"    the wall: {journal.path}")
    print(f"    git: {journal.git_enabled} (plain dir — the wall needs no git)")

    # ------------------------------------------------------------------ #
    # 3. THE OVERNIGHT WATCH — pulses become witness marks               #
    # ------------------------------------------------------------------ #
    print("\n[2] THE OVERNIGHT WATCH — every pulse's silence, committed")
    print("    (t = wall-clock of the night; panic/mood are the pulse's")
    print("     JEPA readings — the terrain context each shadow carries)")
    print()
    print(f"    {'t':>4}  {'phase':<10}  {'panic':>6} {'mood':>6}   witness")
    print(f"    {'-'*4}  {'-'*10}  {'-'*6} {'-'*6}   {'-'*52}")

    # WARM — the room climbs out of the cold.
    tap.post("marlo", "the room is warm and bright, everyone is laughing, "
                      "good vibes all night", ts=0.0)
    e = murmurize_pulse(loop, journal, topic="warm night", ts=0.0)
    _board("0.0", "warm", e)
    tap.post("pincher", "cheers and thanks, this place glows — love this "
                        "beautiful room", ts=1.0, reactions={"😂": 2})
    e = murmurize_pulse(loop, journal, topic="warm night", ts=2.0)
    _board("2.0", "warm", e)
    tap.post("comic", "what a wonderful warm night, glad you all came home",
             ts=2.0)
    e = murmurize_pulse(loop, journal, topic="warm night", ts=4.0)
    _board("4.0", "warm", e)

    # PANIC — the fire. The stampede sense feels it before anyone says
    # the word twice; the pulse that felt it commits it.
    tap.post("deck", "FIRE in the galley! evacuate now!! mayday mayday "
                     "all hands", ts=12.0)
    e = murmurize_pulse(loop, journal, topic="panic spike", ts=12.0)
    _board("12.0", "panic", e)
    tap.post("deck", "this is bad, we are trapped, everything is wrong, "
                     "fear is everywhere", ts=13.0)
    e = murmurize_pulse(loop, journal, topic="panic spike", ts=14.0)
    _board("14.0", "panic", e)

    # QUIET — the agent walks to the quiet deck. New room, fresh ear,
    # same wall: the journal holds witnesses from every room it sits in.
    deck = ChatSpace("the-quiet-deck")
    quiet = PulseLoop("murmur", deck, period=2.0)
    deck.post("quiet", "the darts are quiet tonight, the counter is dry, "
                       "the light is low", ts=0.0)
    e = murmurize_pulse(quiet, journal, topic="quiet hour", ts=30.0)
    _board("30.0", "quiet", e)
    deck.post("quiet", "the door opens, someone comes in, nods, and sits "
                       "down slow", ts=2.0)
    e = murmurize_pulse(quiet, journal, topic="quiet hour", ts=32.0)
    _board("32.0", "quiet", e)
    e = murmurize_pulse(quiet, journal, topic="quiet hour", ts=34.0)
    _board("34.0", "quiet", e)

    print(f"\n    {len(journal)} witness marks on the wall. The agent said "
          f"nothing all night; the wall wrote everything.")

    # ------------------------------------------------------------------ #
    # 4. THE ELEPHANT READS THE WALL — the shadow-trail as a room        #
    # ------------------------------------------------------------------ #
    print("\n[3] THE ELEPHANT READS THE WALL — the shadow-trail as a room")
    room = journal.read_room()
    field = journal.read_field()
    print(f"    room: {room.name} — {len(room)} messages "
          f"(author='murmur', ts=pulse time, channel=topic)")
    print()
    print(f"    field: warmth {field.warmth():+.2f}, κ "
          f"{field.concentration():.2f}")
    movers = sorted(field.readings.items(), key=lambda kv: abs(kv[1]),
                    reverse=True)[:6]
    print("    dials: " + "  ".join(f"{k} {v:+.2f}" for k, v in movers))

    space = MurmurSpace(journal)
    print(f"\n    the trail as a doc room ({space.kind} adapter):")
    print(f"      {space.tint(field)}")
    print(f"      send_back -> status: {space.send_back(field)}")

    # ------------------------------------------------------------------ #
    # 5. RETRIEVAL BY FEELING — the deadband rings                       #
    # ------------------------------------------------------------------ #
    print("\n[4] RETRIEVAL BY FEELING — 'when did murmur last feel like this?'")
    hits = journal.retrieve({"panic": 0.7}, k=2)
    hit = hits[0]
    print(f"\n    query  {{'panic': 0.7}}  ->  {hit['topic']!r} "
          f"(distance {hit['_distance']:.2f})")
    for h in hits:
        print(f"      witness at t={h['ts']:.1f}: {h['text']}")

    hit = journal.retrieve({"mood": 1.0})[0]
    print(f"\n    query  {{'mood': 1.0}}   ->  {hit['topic']!r} "
          f"(distance {hit['_distance']:.2f})")
    print(f"      witness at t={hit['ts']:.1f}: {hit['text']}")

    # The deadband: only the fire crossed the band — it rings; the
    # quiet hours never ring.
    ring = journal.retrieve({"panic": (0.5, 1.0)}, k=5)
    print(f"\n    query  {{'panic': (0.5, 1.0)}}  (the deadband ring) -> "
          f"{len(ring)} witness(es) crossed the band:")
    for r in ring:
        print(f"      t={r['ts']:5.1f}  panic {r['readings']['panic']:.2f}  "
              f"'{r['topic']}'")

    # ------------------------------------------------------------------ #
    # 6. THE WITNESSES — the shadow-trail with its terrain front-matter  #
    # ------------------------------------------------------------------ #
    print("\n[5] THE SHADOW-TRAIL — every witness mark with its terrain")
    for e in journal.entries():
        p = e["readings"]
        conf = e.get("confidence")
        conf_txt = f"  confidence {conf:.2f}" if conf is not None else ""
        print(f"\n    {e['index']:02d}. t={e['ts']:5.1f}  "
              f"[{e['space_id']}]  topic={e['topic']!r}{conf_txt}")
        print(f"       panic {p['panic']:.2f}  mood {p['mood']:.2f}  "
              f"warmth {e.get('warmth', 0.0):+.2f}  "
              f"Δpanic {e.get('direction', {}).get('panic', 0.0):+.2f}")
        print(f"       {e['text']}")

    print(f"\n    one witness file, raw (the shadow with its terrain "
          f"front-matter):")
    first = sorted(os.listdir(journal.path))[0]
    with open(os.path.join(journal.path, first)) as f:
        print("\n" + "".join(f"      {ln}\n" for ln in f.read().rstrip().split("\n")))

    line()
    print("\n  The wall wrote itself. The elephant gave it senses; the")
    print("  terrain gave it context; the shadows remember what the room")
    print("  felt like. Enough to agree on the action — the panic rang.")
    print()


def _board(t: str, phase: str, entry: dict) -> None:
    r = entry["readings"]
    text = entry["text"].replace("\n", " ")
    if len(text) > 52:
        text = text[:49] + "..."
    print(f"    {t:>4}  {phase:<10}  {r['panic']:6.2f} {r['mood']:6.2f}   "
          f"{text}")


if __name__ == "__main__":
    main()
