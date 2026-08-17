"""MUD description tinting — the room's description is its body language.

The key downstream treatment of the Room-Elephant (`docs/jepa-zeitgeist-2026-08-17.md`):

    The MUD text is not static. The room's field **reverberates into how
    everyone sees things**. The description is NOT a report — it is the room
    acting on everyone in it, changing the input-tokens of every agent.

`tint_description(field, base_text, hour=None)` takes the room's objective
field and returns the text every agent sees, mutated by that field:

- **joyful** — high joke_landing / high mood + presence: joyful adjectives
  woven into the bar description; laughter reverberates into the words.
- **panic** — high panic: storms outside, newcomers described as DRENCHED,
  tension primed before the aftermath is even seen.
- **closing time** — late hour + low warmth + low volume: the light changes
  (disco off, fluorescents on), the music plays a little quieter, people start
  looking for the exit and closing their tabs.

It is the light-itself effect: when the disco lights go off and the
fluorescents come on, the people who forgot what time it is start looking for
the exit — even if they aren't thinking about it. The words here are that
light. They are drawn from small template banks (joyful / neutral / tense /
closing-time adjective + weather + light sets), deterministically seeded from
the field, numpy-only. Same field -> same words; a changed field -> a changed
room.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .field import RoomField

# --------------------------------------------------------------------------- #
# Template banks                                                              #
# --------------------------------------------------------------------------- #
WEATHER = {
    "joyful": [
        "A clear night, a warm breeze off the water.",
        "The kind of night where the windows steam with laughter.",
        "A soft night, the harbor still and silver.",
    ],
    "panic": [
        "A storm is lashing the windows outside.",
        "Rain hammers the roof; thunder rolls over the harbor.",
        "The sky has gone green-black; the storm is right on top of us.",
    ],
    "closing": [
        "The last of the night, the street gone quiet outside.",
        "The street outside is empty; the night is finishing itself.",
        "Nothing moving outside but the last light of the moon.",
    ],
    "neutral": [
        "An ordinary night outside.",
        "The night doing what nights do.",
        "A still night, nothing pressing at the glass.",
    ],
}

LIGHT = {
    "joyful": [
        "The lamps burn low and golden.",
        "Candlelight leans on every table.",
        "The light is warm and yellow, the way it gets when a room is happy.",
    ],
    "panic": [
        "The lights are still on, but no one trusts them.",
        "The neon buzzes, nervous and green.",
        "The light is hard and white, too bright for what's coming.",
    ],
    "closing": [
        "The disco lights are off, the fluorescents on.",
        "The dance lights have gone; the fluorescents hum, bright and ugly.",
        "The colored lights are dead; the fluorescents blink on.",
    ],
    "neutral": [
        "The lights are where they always are.",
        "The light has settled into its usual places.",
        "The lamps sit where the lamps always sit.",
    ],
}

# Joyful adjectives woven into the description when the room is laughing.
JOY_ADJ = [
    "bright", "glowing", "warm", "golden", "alive", "sparkling",
    "merry", "ringing", "humming", "soft-gold",
]

# Closing-time details: how people react when the light changes.
CLOSE_DETAIL = [
    "without quite deciding to",
    "a little slowly, a little sad",
    "like waking from a good dream",
]

# --------------------------------------------------------------------------- #
# Thresholds                                                                  #
# --------------------------------------------------------------------------- #
PANIC_HI = 0.5
JOY_JOKE_HI = 0.35
JOY_MOOD_HI = 0.1
JOY_PRESENCE_HI = 0.4
CLOSE_HOUR_LATE = 23.0
CLOSE_HOUR_EARLY = 3.0
CLOSE_WARMTH_LO = 0.0
CLOSE_VOLUME_LO = 0.4


def classify(field: RoomField, hour: Optional[float] = None) -> str:
    """The room's body-language mode: one of panic / joyful / closing / neutral.

    Precedence matters: a fight breaking out (panic) overrides everything;
    joy comes before the quiet of closing time (a warm laughing room at 11pm
    is still the warm bar).
    """
    r = field.readings
    if r.get("panic", 0.0) >= PANIC_HI:
        return "panic"
    if (r.get("joke_landing", 0.0) >= JOY_JOKE_HI
            and r.get("mood", 0.0) >= JOY_MOOD_HI
            and r.get("presence", 0.0) >= JOY_PRESENCE_HI):
        return "joyful"
    late = hour is not None and (hour >= CLOSE_HOUR_LATE or hour < CLOSE_HOUR_EARLY)
    if (late
            and field.warmth() < CLOSE_WARMTH_LO
            and r.get("volume", 0.0) < CLOSE_VOLUME_LO):
        return "closing"
    return "neutral"


def _seed(field: RoomField) -> int:
    """A deterministic integer from the field, so the same field always tints
    the same way (and a changed field changes the words). A rolling hash (not a
    plain sum) keeps the 7 dials' order significant, and micro-perturbations of
    a single dial change the seed."""
    h = 0
    for x in field.vector():
        h = (h * 1000003 + int(round(x * 1_000_000))) & 0x7FFFFFFF
    return h


def _pick(rng: np.random.Generator, bank) -> str:
    return str(bank[int(rng.integers(0, len(bank)))])


def tint_description(field: RoomField, base_text: str,
                     hour: Optional[float] = None,
                     seed: Optional[int] = None) -> str:
    """Mutate `base_text` by the room's field — the description the room speaks.

    `field` is the Room-Elephant's objective reading; `base_text` is the plain
    bar description; `hour` (0-24) is the time of day (used only for closing
    time). The returned string is deterministic for a given field (seeded from
    it, unless `seed` overrides). It is NOT a report of the field — it is the
    room *acting* on everyone in it: the words every agent will now read as
    its own input tokens.
    """
    mode = classify(field, hour)
    rng = np.random.default_rng(_seed(field) if seed is None else seed)
    weather = _pick(rng, WEATHER[mode])
    light = _pick(rng, LIGHT[mode])

    if mode == "joyful":
        adj = _pick(rng, JOY_ADJ)
        return (f"{weather} {base_text} The place feels {adj}. {light} "
                f"Laughter reverberates into the words; newcomers arrive "
                f"grinning, already half-smiling.")
    if mode == "panic":
        return (f"{weather} {base_text} {light} Newcomers arrive drenched, "
                f"dripping rain onto the floor, tension primed before anyone "
                f"sees the aftermath.")
    if mode == "closing":
        detail = _pick(rng, CLOSE_DETAIL)
        return (f"{weather} {base_text} {light} The music plays a little "
                f"quieter; people start looking for the exit and closing "
                f"their tabs, {detail}.")
    return f"{weather} {base_text} {light}"
