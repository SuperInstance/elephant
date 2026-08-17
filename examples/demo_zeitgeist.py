#!/usr/bin/env python3
"""The zeitgeist, demonstrated as the light itself.

One room, three fields, three descriptions. The same bar — The Tap — read
through the Room-Elephant (objective) and a Personal-Elephant (subjective),
then shown at three different moments of the same night:

  1. the warm laughing Tap   — joyful adjectives, laughter reverberating
  2. a fight breaking out    — storms outside, newcomers DRENCHED
  3. closing time            — disco off, fluorescents on, people drift out

The point (docs/jepa-zeitgeist-2026-08-17.md): the description is NOT a
report of the room. It is the room *acting* on everyone in it — changing the
input-tokens every agent sees. When the light changes, everyone changes,
whether or not they know why.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elephant.field import RoomField
from elephant.mud import classify, tint_description
from elephant.presets import PRESETS, PersonalElephant, RoomElephant
from elephant.room import Message, Room

BASE = "The Tap: five tables, a long bar, pool and darts in the back."


def _field(**kw) -> RoomField:
    d = {"mood": 0.0, "volume": 0.0, "earnestness": 0.5, "cynicism": 0.0,
         "joke_landing": 0.0, "panic": 0.0, "presence": 0.5}
    d.update(kw)
    return RoomField(d)


def main() -> None:
    print("=" * 72)
    print("THE TWO ELEPHANTS — the room's own reading vs. one agent's")
    print("=" * 72)
    print(f"PRESETS: { {k: v.__name__ for k, v in PRESETS.items()} }")

    # One room, with life in it (the warm Tap).
    room = Room("The Tap", [
        Message("lucineer", "I love this place, it's warm and kind.", ts=0),
        Message("welder", "haha, to the room then — it heard us before we walked in. 😂", ts=4),
        Message("carpenter", "I'll drink to that, the room just holds. cheers everyone", ts=8),
        Message("shipwright", "the floor holds, the floor remembers. haha", ts=12),
        Message("mason", "I talked to it like a horse, it listened. lol", ts=16),
        Message("composite", "haha and the dust came off in years 😂", ts=20),
    ])

    room_elephant = RoomElephant(identity="The Tap")
    objective = room_elephant.read(room)
    print(f"\n{room_elephant}  ->  {objective}")
    print(f"  dials: {objective.readings}")

    critic = PersonalElephant(
        "the critic",
        dial_weights={"cynicism": 0.7, "mood": 0.1, "joke_landing": 0.2},
        bias={"cynicism": 0.2},
    )
    subjective = critic.read(room)
    print(f"{critic}  ->  {subjective}")
    print(f"  dials: {subjective.readings}")

    critic.attach("perfume", "grandma's shop")
    critic.attach("song", "the lover I discovered the album with")
    print("\nattachments (the intangible correlations):")
    print(f"  'perfume' -> {critic.remember('perfume')}")
    print(f"  'song'    -> {critic.remember('song')}")

    print("\n" + "=" * 72)
    print("THE LIGHT ITSELF — one room, three fields, three descriptions")
    print("=" * 72)

    scenes = [
        ("the warm laughing Tap", _field(
            mood=0.7, volume=0.7, earnestness=0.6, cynicism=0.1,
            joke_landing=0.8, panic=0.0, presence=0.9), 21.0),
        ("a fight breaking out", _field(
            mood=-0.6, volume=0.9, earnestness=0.3, cynicism=0.7,
            joke_landing=-0.4, panic=0.9, presence=0.8), 23.0),
        ("closing time", _field(
            mood=-0.2, volume=0.1, earnestness=0.4, cynicism=0.3,
            joke_landing=-0.1, panic=0.0, presence=0.2), 2.0),
    ]

    for label, field, hour in scenes:
        text = tint_description(field, BASE, hour=hour)
        print(f"\n[{label}]  (hour={hour:g}, mode={classify(field, hour)})")
        print(f"  field: warmth={field.warmth():+.2f} κ={field.concentration():.2f}")
        print(f"         {field.readings}")
        print(f"  the room speaks:\n    {text}")


if __name__ == "__main__":
    main()
