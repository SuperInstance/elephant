#!/usr/bin/env python3
"""The cave demo — the terrain, the shadows, and the deadband.

The captain's reframing (docs/terrain-2026-08-17.md), run in one room
across one evening:

  The terrain is the true state — the full vectorized state of the
  room over time, beyond any human reading. The shadows are what we
  can bear to see — one lossy line per pulse, labeled as a shadow,
  never claiming to be the terrain. The deadband is the discipline —
  below significance, nothing rings; the room breathes, no one is
  disturbed. And a deadband rings up the chain of command: a real
  panic crosses the band, the ring reaches the host, and while the
  room keeps crossing the ring rises — host, foreman, captain. When
  the fight blows over and the room is quiet again, the chain
  descends.

  ACT I   the quiet hour — the terrain accumulates, the shadows
          flicker, the deadband stays silent.
  ACT II  the fight — panic crosses the band; the chain rings up
          host -> foreman -> captain.
  ACT III the blow-over — a new evening, the same warm room; the
          deadband quiets, the chain descends.

Run:  python3 examples/demo_terrain.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elephant.dualdb import DualDBRoom
from elephant.pulse import PulseLoop
from elephant.space import ChatSpace
from elephant.terrain import ChainOfCommand, Deadband, Shadow, Terrain

# The quiet hour's talk — every line the same 3 warm / 2 cool word
# balance, so the room's field is genuinely still: mood pins at +0.40,
# panic near zero, the elephant dozing.
WARM = [
    "good talk tonight, though the week was long, dull, and tired — but kind and warm",
    "glad you came — the day was cold and flat, but the hearth is bright and home",
    "cheers — the walk was empty and stale, but this table is good and warm",
    "peace at last — the storm was cold and dead, but the room is soft and kind",
]

# The fight — alarm words feed the panic dial, the mood turns, and the
# room's accumulated warmth cushions the fall while the panic spike
# carries the ring.
FIGHT = [
    ("welder", "this is wrong, I hate it — and I smell smoke. is that a FIRE?"),
    ("mara", "PANIC — everyone get out NOW, FIRE in the back, go go go!!!"),
    ("tobin", "help, it's a FLOOD — the hull is breached, abandon ship, EVERYONE now!"),
    ("welder", "I told you this would sink us — you never listen, you never care"),
    ("mara", "no. just no. this is bad and wrong — I hate it, honestly"),
    ("tobin", "we're done here. the room is dead to me — cold, empty, stale"),
]


def main() -> None:
    space = ChatSpace("The Tap")
    terrain = Terrain("The Tap")
    dual = DualDBRoom(space.room)
    deadband = Deadband(threshold=0.10, hysteresis=0.6)   # 5x the noise floor
    chain = ChainOfCommand()
    shadow = Shadow(terrain)
    watcher = PulseLoop("watcher", space, period=5.0)

    print("=" * 78)
    print("THE CAVE — one room, its terrain, its shadows, and the deadband")
    print("The elephant feels. The deadband decides. The chain acts.")
    print("=" * 78)

    # ------------------------------------------------------------------ #
    # ACT I — the quiet hour                                             #
    # ------------------------------------------------------------------ #
    print("\n— ACT I · THE QUIET HOUR — the room breathes, no one is disturbed —")
    # The bar has been open an hour; the first round is already on the
    # table before the watcher checks in (a room's first words are not
    # a movement — they are the room).
    space.post("welder", WARM[0], ts=0.0)
    terrain.hear("welder", WARM[0], ts=0.0)

    for i in range(10):
        now = 5.0 + i * 5.0
        text = WARM[i % 4]
        space.post("welder", text, ts=now)
        terrain.hear("welder", text, ts=now)
        terrain.record_room(space, dual=dual, ts=now, meta={"phase": "quiet"})
        m = deadband.movement(terrain)      # the movement the deadband will judge
        ring = deadband.check(terrain)
        level = chain.report(ring)
        watcher.tick(now)
        print(f"t={now:5.1f}  {shadow.project()}")
        print(f"          movement {m:.3f} (band 0.10)  "
              f"chain: {level or '— the room breathes'}")
        if i == 9:
            print(f"          [silent watcher] {watcher.internal_monologue()}")

    quiet_rings = len(chain.history)
    print(f"          → the quiet hour rang {quiet_rings} times. "
          f"the shadows flickered. no one was disturbed.")

    # ------------------------------------------------------------------ #
    # ACT II — the fight                                                 #
    # ------------------------------------------------------------------ #
    print("\n— ACT II · THE FIGHT — a real panic crosses the band —")
    for i, (author, text) in enumerate(FIGHT):
        now = 60.0 + i * 5.0
        space.post(author, text, ts=now)
        terrain.hear(author, text, ts=now)
        terrain.record_room(space, dual=dual, ts=now, meta={"phase": "fight"})
        watcher.tick(now)
        m = deadband.movement(terrain)
        ring = deadband.check(terrain)
        level = chain.report(ring)
        print(f"t={now:5.1f}  {shadow.project()}")
        if ring is not None:
            print(f"          🔔 {ring}")
        print(f"          [chain] {level or '—'}")
        print(f"          [watcher] {watcher.internal_monologue()}")

    # ------------------------------------------------------------------ #
    # ACT III — the blow-over                                            #
    # ------------------------------------------------------------------ #
    print("\n— ACT III · THE BLOW-OVER — a new evening, the same warm room —")
    fresh = ChatSpace("The Tap — next evening")
    fresh_posts = [
        (210.0, WARM[0]), (215.0, WARM[1]), (220.0, WARM[2]),
        (225.0, WARM[3]), (230.0, WARM[0]), (235.0, WARM[1]),
    ]
    for ts, text in fresh_posts:
        fresh.post("welder", text, ts=ts)
        terrain.hear("welder", text, ts=ts)
    dual2 = DualDBRoom(fresh.room)
    # The new evening is already talking by the time the watcher checks
    # in; the terrain records the settled room, not its first words.
    for now in (220.0, 225.0, 230.0, 235.0):
        terrain.record_room(fresh, dual=dual2, ts=now, meta={"phase": "next evening"})
        m = deadband.movement(terrain)
        ring = deadband.check(terrain)
        level = chain.report(ring)
        print(f"t={now:5.1f}  {shadow.project()}")
        print(f"          movement {m:.3f} (band 0.10)  "
              f"chain: {level or '— the room breathes'}")

    # ------------------------------------------------------------------ #
    # The witness marks, and the captain's word                          #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 78)
    print("THE TRAIL OF WORDS — the shadows of the evening (the last few)")
    print(Shadow(terrain).render_transcript(limit=6))
    print("=" * 78)
    print("The elephant is the room's temperature. The terrain is the")
    print("room's truth. The shadow is what we can bear to see. The")
    print("deadband is the discipline that decides when the truth must")
    print("ring — and it rings up the chain of command.")
    print(f"Rings this evening: {len(chain.history)}. "
          f"Final chain position: {chain.ring() or 'no one ringing — the room breathes'}.")
    print("=" * 78)


if __name__ == "__main__":
    main()
