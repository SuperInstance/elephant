#!/usr/bin/env python3
"""demo_decompose.py — the decomposition doctrine in action.

The long time-span demo. A large model runs ONE narrow task — tending
bar — for a long time: 200 customer greetings, 200 answers, with four
latent voices hiding in its style (warm, clipped, jokey, formal). The
doctrine says: decompose that behavior into components that look like
other components, each with a simpler function; let an algorithmic
learning mechanism tune them over time (the guitarist principle:
settings are discovered by running); and keep a stochastic knob so the
body can vary its output when the application wants it.

Run:  python3 examples/demo_decompose.py
"""
import random
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elephant.decompose import DecompositionHarness

# --------------------------------------------------------------------- #
# The trace — one narrow task, a long time-span                         #
# --------------------------------------------------------------------- #
# Four latent styles, three variants each. The big model never explains
# itself; it just answers, night after night, drifting between voices.
STYLES = {
    "warm": [
        ("Hey there, how's your night going?",
         "Hey there! How's your night going? Can I get you something "
         "warm to drink, friend?"),
        ("Hi, how's it going?",
         "Hey, how's it going? It's going great now that you're here! "
         "What can I pour you that'll fix your day, friend?"),
        ("Hello, how are you?",
         "Hey there! How are you tonight? I'm good now that you're "
         "here — what's your pleasure, friend?"),
    ],
    "clipped": [
        ("Beer.", "Beer. Coming up."),
        ("Whiskey.", "Whiskey. Neat."),
        ("Shot.", "Shot. Down."),
    ],
    "jokey": [
        ("Hey! Guess what I brought.",
         "Well well well, if it isn't my favorite customer! Did you "
         "bring the dancing horse again?! The whole bar has been "
         "waiting for that horse all week!!"),
        ("Yo! Big night!",
         "The whole bar suddenly feels SMALL when you walk in! What's "
         "the damage tonight, my friend?! Hahaha!! Don't tell me you "
         "lost the dancing horse AGAIN!!"),
        ("Haha, I'm back!",
         "The legend RETURNS! We were JUST telling stories about the "
         "time you ordered the whole menu!! What'll it be tonight, "
         "champion?! Hahaha!!"),
    ],
    "formal": [
        ("A table for one, please.",
         "Certainly. Allow me to show you to your seat."),
        ("Could I see the wine list?",
         "Of course. I shall bring the list presently."),
        ("I have a reservation for eight.",
         "Excellent. Your table awaits, sir."),
    ],
}

N_PER_STYLE = 50
EPOCHS = 8
SEED = 7


def build_trace(n_per_style: int = N_PER_STYLE) -> list:
    """200 nights behind the bar: 50 of each latent style, shuffled."""
    rng = random.Random(42)
    trace = []
    for style, pairs in STYLES.items():
        for _ in range(n_per_style):
            trace.append(rng.choice(pairs))
    rng.shuffle(trace)
    return trace


def warm_reward(inp: str, out: str) -> float:
    """The application's taste: it wants WARM responses. Warm words in
    the answer, scaled — a clipped 'Beer. Coming up.' scores 0, a warm
    'Hey there! ... friend?' scores 1."""
    WARM_WORDS = {"good", "glad", "warm", "friend", "hey", "treating",
                  "cheers", "lovely", "welcome", "you're here",
                  "fix your day"}
    hits = sum(1 for w in WARM_WORDS if w in out.lower())
    return min(1.0, hits / 2.0)


def _line(c="=", n=72):
    return c * n


def _table(rows, headers):
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))
    def fmt(r):
        return "  ".join(str(c).ljust(w) for c, w in zip(r, widths))
    out = [fmt(headers), "-" * (sum(widths) + 2 * (len(widths) - 1))]
    out += [fmt(r) for r in rows]
    return "\n".join(out)


def describe_component(c, style_map) -> str:
    """What this organ got good at, from its members' styles."""
    counts = Counter()
    for _, out in c.table:
        for style, pairs in STYLES.items():
            if out in [p[1] for p in pairs]:
                counts[style] += 1
    top = counts.most_common(1)
    return top[0][0] if top else "?"


def main() -> None:
    trace = build_trace()
    print()
    print(_line())
    print("  THE DECOMPOSITION HARNESS — the doctrine in code")
    print("  a large model · one narrow task · a long time-span")
    print(_line())
    print()
    print("  > 'Given a long enough time-span running a narrow task, a large LLM")
    print("    can be decomposed into smaller pieces ... separated into components")
    print("    that look like other components and have simpler functions, and")
    print("    might do better with an algorithmic learning mechanism over time")
    print("    and some stochastic mechanisms for varying output.'")
    print("                                          — the captain, 2026-08-17")
    print()

    # The trace -------------------------------------------------------- #
    print("THE TRACE — the big model tending bar for a long time")
    print(f"  {len(trace)} nights: " + "  ·  ".join(
        f"{N_PER_STYLE} {s}" for s in STYLES))
    print("  One narrow task (greet the customer, pour the answer), four")
    print("  latent voices hiding in its style. The trace is the teacher.")
    print()

    # Move 1 + 2 — distill & decompose --------------------------------- #
    h = DecompositionHarness(seed=SEED, temperature=1.0)
    h.ingest(trace)
    h.distill(k=4)

    print("MOVE 1 — DISTILL (the long record becomes components)")
    print("  k-means on simple features: input length, output length,")
    print("  echo overlap, output entropy. 200 nights -> 4 organs.")
    print()
    print("MOVE 2 — DECOMPOSE (organs that look like organs)")
    print("  Every component is shaped like every other:")
    print("  id · prototype · learning_rate · temperature · hits ·"
          " correct · score")
    print()
    before_rows = []
    for c in sorted(h.components, key=lambda c: c.id):
        style = describe_component(c, STYLES)
        before_rows.append([c.id, style, "%.3f" % float(c.door[0]),
                            "%.1f" % float(c.style().mean()),
                            "%.3f" % c.score, c.hits, "—", c.output[:42]])
    print(_table(before_rows, ["id", "organ", "door", "style", "score",
                               "hits", "acc", "answers like"]))
    print()
    print("  Four near-identical organs at birth: uniform, unrun,")
    print("  interchangeable. One job is any job.")
    print()

    # Move 3 — learn --------------------------------------------------- #
    print("MOVE 3 — LEARN (the guitarist principle: settings are found")
    print("         by running, not designed)")
    print("  The application's taste: WARM responses. Replay the 200")
    print(f"  nights for {EPOCHS} epochs; every answer is scored; the")
    print("  winning organ's prototype moves toward the rewarded answer,")
    print("  away from the punished. Temperature 1.0 keeps exploration")
    print("  alive, so organs can still be sampled while they fade.")
    print()
    curve = h.learn(warm_reward, epochs=EPOCHS)
    arrow = "  ".join("%.2f" % r for r in curve)
    print(f"  epoch mean reward :  {arrow}")
    print()

    # AFTER — the body has diverged ------------------------------------ #
    print("AFTER — the body has diverged")
    after_rows = []
    for c in sorted(h.components, key=lambda c: -c.score):
        style = describe_component(c, STYLES)
        acc = "%.2f" % c.accuracy() if c.hits else "—"
        after_rows.append([c.id, style, "%.3f" % float(c.door[0]),
                           "%.1f" % float(c.style().mean()),
                           "%.1f" % c.score, c.hits, acc, c.output[:42]])
    print(_table(after_rows, ["id", "organ", "door", "style", "score",
                              "hits", "acc", "answers like"]))
    spec = h.specialization()
    print()
    print(f"  divergence (std of accuracy) : {spec['divergence']:.2f}")
    print("  The warm organ won the bar — highest score, perfect accuracy,")
    print("  its door now covers every greeting. The clipped, jokey and")
    print("  formal organs faded to the edge of the body: still shaped")
    print("  alike, still able to speak, no longer winning the room.")
    print()

    # The stochastic knob ---------------------------------------------- #
    print("THE STOCHASTIC KNOB — temperature is a dial, not a switch")
    print("  Same greeting, twenty pours per setting. At 0 the body pours")
    print("  one voice (deterministic); turn it up and the faded organs")
    print("  start to speak — varying output, when the application wants")
    print("  it.")
    print()
    greeting = "Hey there, how's your night going?"
    for label, temp, n in (("temperature 0.0", 0.0, 6),
                           ("temperature 1.0", 1.0, 12),
                           ("temperature 3.0", 3.0, 12)):
        pours = [h.respond(greeting, temperature=temp) for _ in range(n)]
        counts = Counter(pours)
        print(f"  {label}:")
        for out, c in counts.most_common():
            print(f"    x{c:<3} {out}")
        print()
    print("  At 0 the same organ always answers. At 1 the winner pours")
    print("  almost always. At 3 the body opens up: identical organs,")
    print("  different voices — the temperature is the divergence that")
    print("  keeps a body from collapsing into one loud dial.")
    print()

    # The body --------------------------------------------------------- #
    print("THE BODY — four organs, each with a simpler function,")
    print("          all shaped alike")
    body_rows = []
    for c in sorted(h.components, key=lambda c: c.id):
        body_rows.append([c.id, describe_component(c, STYLES),
                          "%.3f" % c.learning_rate, "%.1f" % c.temperature,
                          c.hits, "%.2f" % c.accuracy()])
    print(_table(body_rows, ["id", "organ", "learning_rate", "temperature",
                             "hits", "acc"]))
    print()
    print("  inspectable · upgradable · distillable · variable · learning")
    print(_line())
    print("  The black box is gone. The body remains — one narrow task,")
    print("  decomposed all the way down.")
    print(_line())
    print()


if __name__ == "__main__":
    main()
