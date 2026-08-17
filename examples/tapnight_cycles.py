"""Tap-night cycles — run the elephant at The Tap over many evenings.

The captain's brief: build the elephant INTO the Tap (after-work gatherings
where the crew reads each other's creative works) so the agents can LEARN to
use it through many cycles, and so the settings can be *discovered* by players
of different tastes. Settings can't be designed top-down — different agents
desire different settings and self-fine-tune to the moment they're in.

This script runs 14 evening cycles. Each evening the cast arrives, reads
pieces, the elephant reads the room, and each participant self-tunes. At the
end we print the DIVERGED taste table — who became which guitarist.

Run:  python3 examples/tapnight_cycles.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elephant.tapnight import DIAL_NAMES, Participant, TapNightSession


# ---------------------------------------------------------------------- #
# The cast — six personalities with DIFFERENT initial priors.            #
# Each works a different dial, like guitarists who favor different       #
# guitars: one for looks, one for sound, one for the neck.               #
# ---------------------------------------------------------------------- #
def _cast():
    # Each personality claims ONE signature dial (their guitar); the rest of
    # their vibe rests at that dial's neutral center. This is what lets tastes
    # DIVERGE: felt engagement = vibe - room-without-them, so only the dials a
    # participant is genuinely distinctive on register as "engaged".
    return [
        # fiction writer — warm + jokey; cares about mood & whether the joke lands.
        Participant("writer",
                    dial_weights={"mood": 0.40, "joke_landing": 0.30,
                                  "earnestness": 0.15, "presence": 0.10, "volume": 0.05},
                    acclimation_rate=0.35, charisma=0.20,
                    vibe={"mood": 0.70, "joke_landing": 0.50,
                          "earnestness": 0.55, "presence": 0.55}),
        # poet — volume + presence; reads the room's pulse.
        Participant("poet",
                    dial_weights={"mood": 0.30, "volume": 0.30, "presence": 0.20,
                                  "joke_landing": 0.10, "earnestness": 0.10},
                    acclimation_rate=0.25, charisma=0.15,
                    vibe={"volume": 0.70, "presence": 0.60, "mood": 0.50}),
        # essayist — earnest above all.
        Participant("essayist",
                    dial_weights={"earnestness": 0.40, "mood": 0.20, "cynicism": 0.10,
                                  "presence": 0.10, "volume": 0.10,
                                  "joke_landing": 0.05, "panic": 0.05},
                    acclimation_rate=0.30, charisma=0.10,
                    vibe={"earnestness": 0.80, "mood": 0.40}),
        # engineer — earnest but dry; precise, a little cynical.
        Participant("engineer",
                    dial_weights={"earnestness": 0.35, "cynicism": 0.15, "volume": 0.15,
                                  "mood": 0.10, "joke_landing": 0.10,
                                  "presence": 0.10, "panic": 0.05},
                    acclimation_rate=0.15, charisma=0.25,
                    vibe={"earnestness": 0.65, "cynicism": 0.45}),
        # critic — cynicism + the zinger; rolls their eyes so the rest can glow.
        Participant("critic",
                    dial_weights={"cynicism": 0.40, "joke_landing": 0.15,
                                  "earnestness": 0.15, "mood": 0.10,
                                  "volume": 0.10, "presence": 0.05, "panic": 0.05},
                    acclimation_rate=0.20, charisma=0.18,
                    vibe={"cynicism": 0.70, "joke_landing": 0.40}),
        # captain — presence + mood + earnest; holds the room.
        Participant("captain",
                    dial_weights={"presence": 0.35, "mood": 0.20, "earnestness": 0.20,
                                  "volume": 0.10, "joke_landing": 0.05,
                                  "cynicism": 0.05, "panic": 0.05},
                    acclimation_rate=0.40, charisma=0.30,
                    vibe={"presence": 0.75, "mood": 0.60, "earnestness": 0.60}),
    ]


# Works read aloud. Each is (text, reactions): reactions are the crowd's hands —
# the laugh, the heart, the sneer. They feed joke_landing and felt engagement.
WORKS = {
    "writer": [
        ("The old house stays warm, and its walls stay kind — every room keeps a story it's glad to tell.", {"❤️": 2}),
        ("A shed gone up in a day, and by noon the whole yard was laughing, happy as a joke that lands.", {"😂": 2, "❤️": 1}),
        ("Some true things land like a punchline — the whole table goes haha at once, and no one planned it.", {"😂": 3}),
        ("The light comes soft and gentle, and the coffee smells like a good morning kept warm.", {"❤️": 2}),
        ("The sea holds a boat the way a good room holds a good night — easy, and alive.", {"❤️": 2, "👍": 1}),
    ],
    "poet": [
        ("Cold water, warm light — the room hums between them like a caught breath.", {"❤️": 1}),
        ("Loud at the door, then a hush! A wave rolls in and the whole room leans.", {"👏": 2}),
        ("The lamplight keeps nothing; it just glows. Bright and good and gone.", {"❤️": 2}),
        ("Quiet — the room is full, thrumming like a struck string.", {"❤️": 1, "👏": 1}),
    ],
    "essayist": [
        ("I mean it truly: the work we do together means something, and I felt it again tonight.", {"❤️": 2}),
        ("Honestly, I think we learned more failing than we did succeeding, and that's worth saying.", {"👍": 2}),
        ("We remember what we built, not what we lost — and I am glad of it.", {"❤️": 2}),
        ("Actually, the room holds more than the work; it holds the wanting to do it.", {"❤️": 1, "👍": 1}),
    ],
    "engineer": [
        ("Right. The numbers add up, obviously — the seam fits, sure, if you stop pretending.", {"😂": 2}),
        ("It holds. Not pretty, but the seam is true — honestly, that's the part I trust.", {"👍": 2}),
        ("Clearly we overbuilt it. Sure thing, ship it and watch it drift.", {"😂": 2}),
        ("The spec was wrong. I mean that sincerely — we built the wrong thing.", {"👍": 1}),
    ],
    "critic": [
        ("Sure, sure — another masterpiece. Obviously the glass is half empty. 🙄", {"🙄": 2}),
        ("Whatever. Another evening of everyone being wrong, as if that matters. 🙄", {"🙄": 2, "😂": 1}),
        ("Oh, more feelings. Whatever — someone has to roll their eyes so the rest of you can sit there. 🙄", {"😂": 2}),
        ("A joke? Sure. Here's the punchline: we all came back tomorrow anyway. 😏", {"😂": 3}),
    ],
    "captain": [
        ("To the room, then. It heard us before we walked in, and it holds what we bring.", {"❤️": 2}),
        ("Together, tonight. We held the mark and we'll hold each other — cheers to that.", {"❤️": 2, "👍": 1}),
        ("The boat's home, the table's home — I'd rather be here than anywhere, honestly.", {"❤️": 2}),
        ("Good work, crew. The room's warm because you're in it. Yes.", {"❤️": 3, "👍": 1}),
    ],
}

ORDER = ["writer", "poet", "essayist", "engineer", "critic", "captain"]


def _who_held_the_room(session, raw):
    """Which participant's charisma pulled the field hardest tonight."""
    best, best_mag = None, -1.0
    for name, p in session.participants.items():
        n = session._interactions.get(name, 0)
        if n <= 0:
            continue
        s = 1.0 - np.exp(-p.charisma * n)
        mag = s * np.linalg.norm(session._vibe_start.get(name, p.vibe) - raw)
        if mag > best_mag:
            best, best_mag = name, mag
    return best


def _narrative(session, night, prev_warmth):
    """One line of narrative per evening, driven by the measured numbers."""
    f = session.room_field()
    raw = session.raw_field()
    held = _who_held_the_room(session, raw.vector())
    top = session._top_dials(1)[0]
    trend = ("warmed" if f.warmth() > prev_warmth + 1e-3
             else "cooled" if f.warmth() < prev_warmth - 1e-3 else "held")
    return (f"Night {night}: the room {trend} (warmth {f.warmth():+.2f}, "
            f"κ {f.concentration():.2f}) as {held}'s {top} pulled the field.")


def main():
    rng = np.random.default_rng(42)
    session = TapNightSession("The Tap", participants=_cast())
    n_nights = 14

    # Snapshot the initial priors so we can measure divergence at the end.
    initial = {n: p.dial_weights.copy() for n, p in session.participants.items()}

    prev_warmth = 0.0
    print(f"=== The Tap — {n_nights} evenings of the elephant ===\n")

    for night in range(1, n_nights + 1):
        session.start_session()
        idx = (night - 1) % max(len(w) for w in WORKS.values())
        # Each cast member reads one piece (rotating through their bank).
        for name in ORDER:
            text, reactions = WORKS[name][(night - 1 + ORDER.index(name))
                                          % len(WORKS[name])]
            session.speak(name, text, reactions=reactions)
        f = session.room_field()
        print(f"Night {night:>2}: warmth={f.warmth():+.2f} κ={f.concentration():.2f}"
              f" | top: {', '.join(session._top_dials(3))}")
        print("        " + _narrative(session, night, prev_warmth))
        prev_warmth = f.warmth()
        # Self-tune: each participant's dial_weights drift toward where they felt engaged.
        for name in ORDER:
            session.tune_participant(name)
        session.end_session()

    # ------------------------------------------------------------------ #
    # The DIVERGED taste table — who became which guitarist.              #
    # ------------------------------------------------------------------ #
    print("\n=== DIVERGED TASTES after %d nights ===\n" % n_nights)
    print(f"{'personality':<12} {'top dials (weights)':<48} {'accl':>5} {'char':>5}")
    print("-" * 74)
    for name in ORDER:
        p = session.participants[name]
        order = np.argsort(-p.dial_weights)
        top3 = ", ".join(f"{DIAL_NAMES[i]} {p.dial_weights[i]:.2f}"
                         for i in order[:3])
        print(f"{name:<12} {top3:<48} {p.acclimation_rate:>5.2f} {p.charisma:>5.2f}")

    # Pairwise distance between final dial_weights — the spread of the tastes.
    names = list(ORDER)
    print("\nPairwise dial_weights distance (final):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = float(np.linalg.norm(session.participants[names[i]].dial_weights
                                     - session.participants[names[j]].dial_weights))
            print(f"  {names[i]:<10} ↔ {names[j]:<10} {d:.3f}")

    init_spread = np.mean([
        float(np.linalg.norm(initial[a] - initial[b]))
        for a in names for b in names if a < b])
    final_spread = np.mean([
        float(np.linalg.norm(session.participants[a].dial_weights
                             - session.participants[b].dial_weights))
        for a in names for b in names if a < b])
    print(f"\nmean pairwise distance: initial {init_spread:.3f} → final "
          f"{final_spread:.3f} (tastes {'diverged' if final_spread > init_spread else 'converged'})")


if __name__ == "__main__":
    main()
