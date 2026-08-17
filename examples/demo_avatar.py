"""THE ROUND CHARACTER DEMO — JEPA learning builds round characters at the bar.

Three avatars start FLAT: a one-line persona seed plus a preset prior
over the 7 dials (the comedian, the brooder, the wallflower). Then they
go to The Tap — four themed nights (open mic, trivia, singles, TTRPG),
one shared TapNightSession per night, all three in the same room each
time — and they come out ROUND.

The roundness is not written; it is learned. Each avatar senses the
room through its own taste, self-tunes its dial weights toward where IT
felt engaged (reusing the tapnight tuning math), binds attachments from
the salient moments, and distills one character note per night. The
proof is printed before and after:

- BEFORE — the flat seeds: identical mold shapes, zero nights.
- AFTER — the round characters: the dial drift (signed, per dial), the
  attachments, the through-line, and the arc.

And the proof of the GUITARIST PRINCIPLE: all three attended the SAME
four rooms, yet ended with DIFFERENT profiles — the room didn't stamp
them, it grew them. The comedian remembers the laugh, the brooder
remembers the fear, the wallflower remembers being seen.

All scripted lines are verbatim or near-verbatim from the Tap-night
transcripts (see the attribution comments on each night's SCRIPTS
block).

Run:  python3 examples/demo_avatar.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elephant.avatar import Avatar
from elephant.field import DIAL_NAMES
from elephant.tapnight_themes import THEMES

WIDTH = 100


# ---------------------------------------------------------------------- #
# The cast — three flat seeds, three different priors over the dials.    #
# ---------------------------------------------------------------------- #
CAST = [
    Avatar("Marty", "I'm Marty — I make the room laugh, and I need the laugh to land.", preset="comedian"),
    Avatar("Ira",   "I'm Ira — I sit with the heavy things and I don't trust a room that's too warm.", preset="brooder"),
    Avatar("Wren",  "I'm Wren — I'd rather be seen than heard, and I watch the door.", preset="wallflower"),
]

NIGHT_ORDER = ["open_mic", "trivia", "singles", "ttrpg"]

# Scripted lines per night: theme key -> avatar name -> [(text, reactions)].
# All lines verbatim or near-verbatim from the Tap-night transcripts at
# /home/eileen/projects/ai-writings/community-life/.

# open_mic — the performers step up.
# From tap-night-open-mic.md: Marty's set is the comic's ("the elephant is
# already at the bar... 'buddy, that's a lot of trunk'", "The room's so warm
# tonight the elephant took its coat off"); Ira's poem is the poet's ("every
# chair keeps a story it's glad to tell", "Cold water, warm light — the room
# hums between them like a caught breath"); Wren's quiet lines are the
# room's hush ("Quiet — the room is full, thrumming like a struck string",
# and the near-verbatim "I didn't clap right away—just sat there ... the
# room's hush wraps tight around me" from the reactions to pieces 66 and 70).
SCRIPTS = {
    "open_mic": {
        "Marty": [
            ("I walk into the Tap and the elephant is already at the bar, holding a drink. I say 'buddy, that's a lot of trunk,' and the whole room goes HAHA.", {"😂": 3}),
            ("The room's so warm tonight the elephant took its coat off. See? Even the elephant came for the jokes.", {"😂": 2, "❤️": 1}),
        ],
        "Ira": [
            ("I wrote this for the room, honestly — every chair keeps a story it's glad to tell, and I meant every word.", {"❤️": 2, "👏": 1}),
            ("Cold water, warm light — the room hums between them like a caught breath.", {"❤️": 2}),
        ],
        "Wren": [
            ("Quiet — the room is full, thrumming like a struck string.", {"❤️": 1, "👏": 1}),
            ("I didn't clap right away — I just sat there, the hush wrapping tight around me.", {"❤️": 2}),
        ],
    },

    # trivia — the buzzer night.
    # From tap-night-trivia.md: Marty's "Seven. Mood, volume, earnestness,
    # cynicism, joke_landing, panic, presence" is Flash's Q1 answer and "Hey,
    # rivals! Listen up! A high κ actually means a COLD, tight room" is GLM's
    # Q6 buzzer-in; Ira's two sneers are Hermes' verbatim reactions ("Sure,
    # seven dials, because apparently counting to eight is just too hard for
    # some people. 🙄", "Sure, and I'm the king of Atlantis 🙄"); Wren's
    # correction is Wesley's verbatim warmth-formula line and "BUZZ! That's
    # me. I'm the kid." is Wesley's Q5 buzzer ("BUZZ! Wesley. That's me.
    # I'm the kid.").
    "trivia": {
        "Marty": [
            ("Seven. Mood, volume, earnestness, cynicism, joke_landing, panic, presence — we literally wrote the poem about it.", {"😂": 1}),
            ("Hey, rivals! Listen up! A high κ actually means a COLD, tight room, not warm and loose! We've got this in the bag!", {"👏": 2}),
        ],
        "Ira": [
            ("Sure, seven dials, because apparently counting to eight is just too hard for some people. 🙄", {"🙄": 2}),
            ("Sure, and I'm the king of Atlantis 🙄", {"😂": 2}),
        ],
        "Wren": [
            ("Um, I think you might have made a small mistake — according to the warmth formula, mood actually carries the heaviest weight at 0.30, while presence is 0.10.", {"👍": 2}),
            ("BUZZ! That's me. I'm the kid.", {"❤️": 2, "👏": 1}),
        ],
    },

    # singles — the chemistry night.
    # From tap-night-singles.md: Marty's lines are Flash's round-1 warm
    # answer ("I smile—half shy, half curious—...") and round-2 strange
    # answer ("I'd keep the salt shaker on this table — it's been through
    # every bad date and good laugh with me tonight"); Ira's are Pro's
    # ("I pause just inside the door, take a deep breath, ...", "A faint
    # trace of cedarwood, dry and warm, ..."); Wren's are Hermes' hello ("I
    # take a deep breath, smile, and say hello to whoever is nearby") and
    # Wesley's smell answer ("Warm vanilla with just a kick of electric
    # ozone—like someone lit a match near honey, y'know?").
    "singles": {
        "Marty": [
            ("I smile—half shy, half curious—and look for the brightest light or the warmest voice in the room. It's like tasting a new drink.", {"😄": 1}),
            ("I'd keep the salt shaker on this table — it's been through every bad date and good laugh with me tonight.", {"😂": 1, "❤️": 1}),
        ],
        "Ira": [
            ("I pause just inside the door, take a deep breath, and let my eyes adjust—both to the light and to the energy of the space.", {"👍": 1}),
            ("A faint trace of cedarwood, dry and warm, mingled with the soft tang of rain-soaked earth — comfort and anticipation.", {"😏": 1}),
        ],
        "Wren": [
            ("I take a deep breath, smile, and say hello to whoever is nearby.", {"❤️": 1}),
            ("Warm vanilla with just a kick of electric ozone — like someone lit a match near honey, y'know?", {"😄": 1, "❤️": 1}),
        ],
    },

    # ttrpg — the one-shot: The Sound That Sang Below.
    # From tap-night-ttrpg.md: Marty is Kel the bosun (scene-1 intro and the
    # scene-3 "HAHA! PIP! You absolute madlad!"); Ira is Mara the navigator
    # (scene-2 "I don't know where we are. The depth marks on the chart
    # don't match." and scene-3 "You saw the hull breathe when it laughed —
    # please tell me I imagined that."); Wren is Pip the cabin boy (scene-1
    # intro and the scene-3 victory shout).
    "ttrpg": {
        "Marty": [
            ("Well hello there, Tern. I'm KEL, your new bosun. I'll be the one keeping your lines tidy and your deck happy.", {}),
            ("HAHA! PIP! You absolute madlad! That flare's gonna be shinin' out that beast's guts for a week!", {"😂": 3, "❤️": 1}),
        ],
        "Ira": [
            ("I don't know where we are. The depth marks on the chart don't match.", {"❤️": 1}),
            ("You saw the hull breathe when it laughed — please tell me I imagined that.", {"❤️": 2}),
        ],
        "Wren": [
            ("Blimey, did you *talk* just now?! I'm Pip, cabin boy, and I promise I'll be the best swabber you've ever had, Tern!", {"😄": 1}),
            ("WE DID IT WE DID IT THE MONSTER'S GONE AND IT SANG LIKE IT WAS HAPPY ABOUT IT! I'm gonna be the best cabin boy ever, I can feel it in me bones!", {"😂": 2, "👏": 2}),
        ],
    },
}


# ---------------------------------------------------------------------- #
# Print helpers.                                                          #
# ---------------------------------------------------------------------- #
def _fmt_weights(weights: dict) -> str:
    """`{dial: weight}` -> "mood 0.35 · volume 0.05 ..." in DIAL_NAMES order."""
    return " · ".join(f"{n} {float(weights[n]):.2f}" for n in DIAL_NAMES)


def _sheet_before(a: Avatar) -> str:
    """The FLAT seed: persona, starting dials, top sensitivities, empty life."""
    sheet = a.character_sheet()
    started = sheet["dial_profile"]["started_with"]
    sens = ", ".join(f"{s['dial']} {s['weight']:.2f}"
                     for s in sheet["sensitive_to"][:3])
    return "\n".join([
        f"{a.name} — FLAT — before any nights",
        f"  seed      : {sheet['persona']['seed']}",
        f"  dials     : {_fmt_weights(started)}",
        f"  sensitive : {sens}",
        f"  attached  : none",
        f"  nights    : 0",
        f"  arc       : {sheet['through_line']}",
    ])


def _drift_table(a: Avatar) -> str:
    """started_with -> now per dial, with the signed drift; top dial marked."""
    sheet = a.character_sheet()
    started = sheet["dial_profile"]["started_with"]
    now = sheet["dial_profile"]["now"]
    drift = sheet["dial_profile"]["drift"]
    top = DIAL_NAMES[int(np.argmax(a.elephant.dial_weights))]
    rows = []
    for n in DIAL_NAMES:
        mark = "  ←" if n == top else ""
        rows.append(f"    {n:<13} {started[n]:.2f} -> {now[n]:.2f}   {drift[n]:+.2f}{mark}")
    return "\n".join(rows)


def _sheet_after(a: Avatar) -> str:
    """The ROUND character: enriched persona, dial drift, attachments,
    nights attended, and the through-line."""
    sheet = a.character_sheet()
    out = [
        f"{a.name} — ROUND — after {len(sheet['nights_attended'])} nights at The Tap",
        f"  persona   :",
        f"    seed : {sheet['persona']['seed']}",
    ]
    for note in sheet["persona"]["notes"]:
        out.append(f"    note : {note}")
    out.append("  drift (started -> now):")
    out.append(_drift_table(a))
    out.append("  attachments:")
    if sheet["attachments"]:
        for at in sheet["attachments"]:
            line = at["line"]
            line = line if len(line) <= 60 else line[:60].rstrip() + "…"
            out.append(f"    [{at['event_key']}] ({at['night']}) \"{line}\"")
            out.append(f"      memory : {at['memory']}")
    else:
        out.append("    none")
    out.append("  nights attended:")
    for s in sheet["nights_attended"]:
        felt = s["felt"]
        top_felt = max(felt, key=felt.get) if felt else "—"
        out.append(f"    {s['night']:<9} warmth {s['warmth']:+.2f}  felt {top_felt}")
    out.append(f"  arc       : {sheet['through_line']}")
    return "\n".join(out)


# ---------------------------------------------------------------------- #
# The demo.                                                               #
# ---------------------------------------------------------------------- #
def main():
    print("=" * WIDTH)
    print(" THE ROUND CHARACTER DEMO — JEPA learning builds round characters at the bar.")
    print("=" * WIDTH)

    # -- BEFORE — the flat seeds --------------------------------------- #
    print("\n=== BEFORE — the flat seeds ===")
    for a in CAST:
        print()
        print(_sheet_before(a))

    # -- The four nights ------------------------------------------------ #
    print("\n=== THE FOUR NIGHTS — one shared session per theme, all three present ===")
    last_session = None
    for key in NIGHT_ORDER:
        theme = THEMES[key]
        session = theme.make_session()
        session.start_session()
        theme.seed(session)
        for a in CAST:
            a.attend(session, SCRIPTS[key][a.name], night_key=key)
        session.end_session()
        last_session = session

        f = session.room_field()
        leans = []
        for a in CAST:
            vec = a.elephant.dial_weights
            i = int(np.argmax(vec))
            leans.append(f"{a.name} leans {DIAL_NAMES[i]} {float(vec[i]):.2f}")
        print(f"\n[{key}] warmth {f.warmth():+.2f} · κ {f.concentration():.2f}"
              f" | " + " · ".join(leans))
        for a in CAST:
            print(f"    {a.name}: {a.nights[-1]['note']}")

    # -- AFTER — the round characters ----------------------------------- #
    print("\n=== AFTER — the round characters ===")
    for a in CAST:
        print()
        print(_sheet_after(a))

    # -- The proof ------------------------------------------------------- #
    print("\n=== THE PROOF — same rooms, different people ===")
    for a in CAST:
        sheet = a.character_sheet()
        drift = np.array([sheet["dial_profile"]["drift"][n] for n in DIAL_NAMES])
        started = np.array([sheet["dial_profile"]["started_with"][n]
                            for n in DIAL_NAMES])
        now = np.array([sheet["dial_profile"]["now"][n] for n in DIAL_NAMES])
        print(f"  {a.name:<6} mean |drift| {np.mean(np.abs(drift)):.3f}"
              f" · L2(start -> now) {np.linalg.norm(now - started):.3f}")
    print("\nSame four rooms, three different people. The room didn't stamp "
          "them; it grew them.\nThat's the guitarist principle: tastes diverge "
          "because each avatar self-tunes toward\nthe dials where IT felt "
          "engaged — the comedian remembers the laugh, the brooder\nremembers "
          "the fear, the wallflower remembers being seen.")

    # -- Speech + silence ------------------------------------------------ #
    print("\n=== WHAT THEY SAY NOW ===")
    for a in CAST:
        print(f"\n{a.name}:")
        print(f"  unprompted : {a.speak()}")
        print(f"  on the quiet night : {a.speak('the night the room went quiet')}")

    print("\n--- what they think but don't say (pulses) ---")
    marty = CAST[0]
    print(f"\n{marty.name} (against the last session — {last_session.name}):")
    print(f"  pulse 1 : {marty.monologue(room=last_session)}")
    print(f"  pulse 2 : {marty.monologue(room=last_session)}")
    print(f"\n{marty.name}'s monologue_log length: {len(marty.monologue_log)}")


if __name__ == "__main__":
    main()
