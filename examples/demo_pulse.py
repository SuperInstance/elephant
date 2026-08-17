#!/usr/bin/env python3
"""The internal-monologue pulse — the macro read, like a currency pair.

The captain's insight (the design core of `elephant/pulse.py`):

    Agents run internal monologues on CONSTANT PULSES even when they
    aren't talking. Each pulse takes a perception check — reads the
    table's conversation as a WHOLE HAND, the way a trader reads a
    currency pair. The number doesn't matter. TWO numbers show
    DIRECTION. MORE THAN TWO show RATE OF CHANGE.

This demo feeds one room through three phases — WARM, FLAT, COOL — and
shows a silent agent's pulse-by-pulse perception checks: direction
(last two readings) and rate of change (last three+, the second
difference) per dial, like a trader's board, plus its internal
monologues during the silence.

Watch for the macro read emerging from the numbers:

  WARM — the room climbs out of the cold; direction turns positive,
         then the rate spikes as the mood surges (and eases as the
         dial saturates).
  FLAT — the table goes quiet; direction flattens to zero, rate to
         zero. The trader holds.
  COOL — the room turns; direction flips negative, and the fall
         decelerates — the rate decays back toward zero.

Run:  python3 examples/demo_pulse.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elephant.pulse import PulseLoop
from elephant.space import ChatSpace


# ---------------------------------------------------------------------- #
# The room script — three phases of one evening at The Tap               #
# ---------------------------------------------------------------------- #
WARM = [
    "cold dull stale dead flat",
    "cold and dull, but kind — warm and glad, flat at first",
    "no fear, no panic — just great, good, and home",
    "never tired, never lost — happy, nice, and together",
    ("love this beautiful wonderful bright room — cheers and thanks, we're "
     "alive, laughing, glowing, at peace, soft and gentle, glad and happy, "
     "warm and good 😂"),
]
WARM_REACTIONS = [None, None, None, None, {"😂": 2, "❤️": 2}]

FLAT = [
    "the darts are quiet tonight, the counter is dry, and the light is low",
    "someone pulls a glass off the shelf; it clinks once, then the room settles",
    "the door opens, someone comes in, nods, and sits",
]

COOL = [
    "cold. sure, sure, whatever. the room has gone flat and dead — empty, stale, lost",
    "no. just no. this is bad and wrong — I hate it, honestly",
    "crickets. groan. ugh — it fails, it's failing, it's dead",
    "stale and tired, cold and flat — the fire is out, the room is empty",
]

PHASES = [
    ("WARM", WARM, WARM_REACTIONS,
     "the room climbs out of the cold — direction goes positive, the rate spikes"),
    ("FLAT", FLAT, [None] * 3,
     "the table goes quiet — direction flattens to zero, the trader holds"),
    ("COOL", COOL, [None] * 4,
     "the room turns — direction flips negative, then the fall eases"),
]


def main() -> None:
    space = ChatSpace("The Tap")
    trader = PulseLoop("trader", space, period=5.0, history=12)

    print("=" * 76)
    print("THE INTERNAL MONOLOGUE PULSE — a silent agent reading the table")
    print("as a whole hand, the way a trader reads a currency pair.")
    print("The number doesn't matter: TWO numbers show DIRECTION,")
    print("MORE THAN TWO show RATE OF CHANGE.")
    print("=" * 76)

    pulse_no = 0
    for name, posts, reactions, blurb in PHASES:
        print(f"\n--- PHASE {name}: {blurb} ---")
        for text, rx in zip(posts, reactions):
            space.post("welder", text, reactions=rx)
            r = trader.tick(now=(pulse_no + 1) * 5.0)
            pulse_no += 1
            print(f"pulse {pulse_no:02d}  {r.board()}")
            print(f"         [silent] {trader.internal_monologue()}")
            if pulse_no in (5, 8):           # phase boundaries: the whole hand
                print(f"         {r.whole_hand}")

        # the raw series — the numbers that don't matter individually
        raw = " -> ".join(f"{d['mood']:+.2f}" for d in trader.last_readings())
        last = trader.last_report()
        print(f"  raw mood: {raw}")
        if last is not None and last.n_readings >= 3:
            print(f"  macro:   mood {last.direction['mood']:+.2f}/pulse, "
                  f"rate {last.rate_of_change['mood']:+.2f}/pulse² — "
                  f"the movement is the perception.")

    print("\n" + "=" * 76)
    print("The trader never said a word all evening. The pulses never stopped.")
    print("The whole hand, one last time:")
    print("  " + trader.internal_monologue(prompt="what is the room telling you?"))
    print("=" * 76)


if __name__ == "__main__":
    main()
