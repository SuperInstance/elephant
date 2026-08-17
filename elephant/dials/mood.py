"""Mood dial — the warmth or coldness of the room.

The first sense. Not what is said, but the temperature of what is
said. Warm rooms run positive-valence language with energy; cold rooms
run clipped, flat, negative. This is the dial that makes a room feel
like a sauna or a cold plunge before anyone has explained why.
"""
from __future__ import annotations

import re
from typing import Set

from ..dial import Dial
from ..room import Room

POSITIVE = {
    "good", "great", "love", "loved", "beautiful", "warm", "warmth", "kind",
    "glad", "happy", "cheers", "toast", "proud", "wonderful", "nice", "yes",
    "thank", "thanks", "home", "join", "joint", "held", "holds", "together",
    "relax", "relaxing", "peace", "soft", "gentle", "laugh", "laughing",
    "fun", "glow", "bright", "alive", "earnest", "sincere",
}
NEGATIVE = {
    "cold", "dead", "broke", "break", "fear", "afraid", "panic", "fire",
    "bad", "wrong", "hate", "lied", "lie", "fails", "failed", "sinking",
    "flood", "breach", "alarm", "crickets", "groan", "ugh", "no", "never",
    "dull", "flat", "empty", "stale", "tired", "trapped", "crash", "lost",
}
_CAPS = re.compile(r"\b[A-Z]{2,}\b")


class MoodDial(Dial):
    name = "mood"
    description = "warm/cold valence of the room, [-1 cold .. +1 warm]"

    def read(self, room: Room) -> float:
        if not room.messages:
            return 0.0
        pos = neg = 0
        for m in room.messages:
            words = set(m.words)
            pos += len(words & POSITIVE)
            neg += len(words & NEGATIVE)
        total = pos + neg
        if total == 0:
            return 0.0
        raw = (pos - neg) / max(total, 1) * 2.0  # scale to ~[-1, 1]
        return max(-1.0, min(1.0, raw))
