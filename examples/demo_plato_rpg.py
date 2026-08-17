#!/usr/bin/env python3
"""demo_plato_rpg — "The Fogbound Harbor", a one-shot session.

The elephant is the dungeon master. Three rooms, three round
characters, one goal: find the source of the fog.

Run:  python3 examples/demo_plato_rpg.py

The output is a session transcript — the cave wall telling the story:

- the SHADOWS: tinted room descriptions, perception reports in words
  (never raw vectors), pulse monologues, GM lines;
- the ROLLS: each player's perception check — the room's direction and
  rate of change over its recent history (two numbers show direction,
  three show rate of change);
- the RINGS: deadband crossings — a fight erupts, an anomaly spikes, a
  trend inverts — and each ring advances the plot;
- the SHEETS: final character sheets (the round characters, their
  personal elephants) and the rooms' vital signs.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elephant.plato_rpg import run_scenario

# ---------------------------------------------------------------------- #
# The scenario — as data.                                                #
# ---------------------------------------------------------------------- #
SCENARIO = {
    "name": "The Fogbound Harbor",
    "premise": (
        "The fog has come in off the sound for three nights running, and it "
        "has not left once. It hugs the harbor like it owns the place. The "
        "boats look drowned standing up. And on the old whaler beached past "
        "the wharf, in the haunted wheelhouse, a light has been burning since "
        "Tuesday — though there is no one aboard to light it."
    ),
    "goal": "find the source of the fog before it finds you",
    "goal_room": "The Haunted Wheelhouse",
    "max_turns": 8,
    "banter": True,

    "rooms": [
        {
            "name": "The Tap",
            "hour": 21.0,
            "description": (
                "the taproom of the Salt & Lantern: low beams, a hearth that "
                "never quite goes out, and a window full of fog that has been "
                "pressing at the glass for three nights."
            ),
            "deadband": {"panic": 0.6, "warmth_hi": 0.38},
            "seed_messages": [
                ("barmaid", "The hearth is lit! Good evening, all — the house is glad to see you.", 100),
                ("sailor", "A toast! Cheers — the fog outside can keep its cold; in here it's warm and kind.", 200),
                ("old woman", "Hush, you'll wake the night. But yes — the hearth's warm, and that's something.", 300),
                ("barmaid", "Pints up! They say the wheelhouse light's burning again — but not tonight, not in here.", 400),
                ("sailor", "To the fog — may it find something better to do!", 500),
                ("old woman", "Mark me — the fog wants something in that wheelhouse. But tonight we drink warm and stay.", 600),
            ],
        },
        {
            "name": "The Wharf",
            "hour": 22.5,
            "description": (
                "the wharf: stacked crates, a hung lantern swaying in a wind "
                "that isn't there, and one skiff still tied up with its oars "
                "standing in the locks, waiting for someone who never came."
            ),
            "deadband": {"panic": 0.4, "anomaly": 0.55},
            "seed_messages": [
                ("dockhand", "Every boat's in but the Gull. Skipper says he saw a light in the wheelhouse and turned back. Said the fog wasn't fog.", 100),
                ("dockhand", "I don't mind the dark. I mind the quiet. Even the gulls won't sit on the water tonight.", 200),
                ("night", "The lantern gutters, though the air is still.", 300),
                ("dockhand", "Crates of salt for the Gull, still strapped. Nobody's come to claim them.", 400),
            ],
        },
        {
            "name": "The Haunted Wheelhouse",
            "hour": 23.5,
            "description": (
                "the wheelhouse of the old whaler Fortune's Due, beached and "
                "shunned: brass fittings green with salt, a wheel that turns "
                "by itself in the fog, and a logbook open to a page that "
                "ends mid-sentence."
            ),
            "deadband": {"panic": 0.45, "anomaly": 0.5},
            "seed_messages": [
                ("logbook", "Night 40. The fog has followed us. It does not move like weather. It moves like something that has been told to wait.", 100),
                ("logbook", "Night 41. The wheel turns by itself now. I have lashed it. It turns anyway. I cannot find the lamp that lit the way home.", 200),
                ("chain", "Somewhere below, a chain rings against the hull — one pull, two, three, then silence.", 300),
                ("logbook", "Night 42. The lamp is lit though no one lit it. The light is not for us. It is a light for the ship that is still coming. We must not let it in.", 400),
                ("logbook", "Night 43. FIRE in the lamp. FIRE in the water. The fog is full of lights — code red, code red, abandon the wheel, RUN —", 500),
                ("crew", "All hands! Now! The light is in the water — EVERYONE to the boats, GO GO GO!", 600),
            ],
        },
    ],

    "edges": [
        ("The Tap", "The Wharf", "the harbor road"),
        ("The Wharf", "The Haunted Wheelhouse", "the rotting gangplank"),
    ],

    "players": [
        {"name": "Marnie", "archetype": "comedian", "start": "The Tap",
         "goal": "keep the others laughing so the dark can't get a word in"},
        {"name": "Ilsa", "archetype": "brooder", "start": "The Tap",
         "goal": "find the source of the fog before it finds them"},
        {"name": "Theo", "archetype": "wallflower", "start": "The Tap",
         "goal": "say one true thing"},
    ],

    "script": [
        (1, "Marnie", "joke", "The Tap"),
        (1, "Ilsa", "investigate", "The Tap"),
        (1, "Theo", "wait", "The Tap"),
        (2, "Marnie", "joke", "The Tap"),
        (2, "Ilsa", "investigate", "The Tap"),
        (2, "Theo", "comfort", "The Tap"),
        (3, "Marnie", "move", "The Wharf"),
        (3, "Ilsa", "move", "The Wharf"),
        (3, "Theo", "move", "The Wharf"),
        (4, "Ilsa", "investigate", "The Wharf"),
        (4, "Marnie", "joke", "The Wharf"),
        (4, "Theo", "investigate", "The Wharf"),
        (5, "Marnie", "move", "The Haunted Wheelhouse"),
        (5, "Theo", "move", "The Haunted Wheelhouse"),
        (5, "Ilsa", "investigate", "The Wharf"),
        (6, "Ilsa", "move", "The Haunted Wheelhouse"),
        (6, "Marnie", "fight", "The Haunted Wheelhouse"),
        (6, "Theo", "wait", "The Haunted Wheelhouse"),
        (7, "Ilsa", "resolve", "The Haunted Wheelhouse"),
        (7, "Marnie", "joke", "The Haunted Wheelhouse"),
        (7, "Theo", "resolve", "The Haunted Wheelhouse"),
    ],

    "events": {
        4: [{"room": "The Wharf",
             "text": "The fog rolls in and the lantern dies. From under the water, the Gull's bell rings — three long pulls, like a hand asking for HELP — an emergency, NOW, NOW, NOW!"}],
        6: [{"room": "The Haunted Wheelhouse",
             "text": "The wheel turns by itself — one full revolution, slow, deliberate, like it is listening."}],
    },

    "plot_lines": [
        "The taproom crests its band — the party finds its feet, and at the window the fog presses closer, annoyed to find a warm room.",
        "The night answers from across the water: the wheelhouse has been screaming since Tuesday — and it has been screaming for YOU.",
        "Across the water, the wheelhouse's own senses are ringing off — the fog does not behave, and the room knows it.",
        "The wharf's wrongness is confirmed: the bell under the water is asking to be found, and it knows your names.",
        "The bell stops. The water goes still. Whatever was ringing for help has stopped needing to ask.",
    ],
}


def sheet_lines(player):
    """The final character sheet — the round character, drawn."""
    s = player.character_sheet()
    e = s["elephant"]
    top_weights = sorted(e["dial_weights"].items(), key=lambda kv: -kv[1])[:3]
    top_txt = ", ".join(f"{k} {v:.2f}" for k, v in top_weights)
    acts = ", ".join(f"T{t} {v}" for t, v, _tg in s["acts"]) or "—"
    return [
        f"  {s['name']} — {s['title']}",
        f"    position: {s['position']}",
        f"    goal: {s['goal']}",
        f"    personal elephant: vibe warmth {e['vibe_warmth']:+.2f}, "
        f"charisma {e['charisma']:.2f}, reads {e['reads_first']} first",
        f"    cares about: {top_txt}",
        f"    pulses: {s['pulses']}, acts: {acts}",
        f"    last perception: {s['last_perception']}",
    ]


def main() -> None:
    print("=" * 74)
    print("  THE FOGBOUND HARBOR")
    print("  a Plato-based agentic RPG one-shot — the elephant as dungeon master")
    print("=" * 74)

    log = run_scenario(SCENARIO)

    print()
    print("\n".join(log.lines))
    print()

    # The epilogue summary.
    print("-" * 74)
    print(f"  SESSION END — {log.turns} turns, plot stage {log.plot_stage}, "
          f"{'goal reached' if log.goal_reached else 'goal not reached'}")
    print(f"  rings: {len(log.rings)}")
    for ring in log.rings:
        print(f"    {ring.marker()}")
    print()

    print("  FINAL CHARACTER SHEETS")
    for player in log.players:
        print()
        print("\n".join(sheet_lines(player)))

    print()
    print("  THE ROOMS' VITAL SIGNS (the terrain's projections)")
    for name, room in log.world.rooms.items():
        s = room.sheet()
        print(f"    {s['name']}: warmth {s['warmth']:+.3f}, κ {s['concentration']:.3f}, "
              f"trend {s['trend_dial']:+.3f}, anomaly {s['anomaly']:.3f}, "
              f"{s['messages']} messages")


if __name__ == "__main__":
    main()
