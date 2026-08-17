"""Earnestness dial — how much the room means it.

Sincere rooms say what they mean and mean what they say: first-person,
concrete, no hedging. Ironic rooms float above their own words. This
is the dial that tells you whether the compliment is a compliment.
"""
from __future__ import annotations

from ..dial import Dial
from ..room import Room

SINCERE = {
    "i", "me", "my", "we", "our", "really", "truly", "honestly", "actually",
    "mean", "meant", "felt", "feels", "remember", "remembered", "built",
    "held", "worked", "learned", "earnest", "sincere", "promise",
}
HEDGE = {
    "maybe", "perhaps", "sorta", "kinda", "kind of", "i guess", "whatever",
    "supposedly", "allegedly", "honestly?", "lol", "haha", "heh", "¯\\_(ツ)_/¯",
}


class EarnestnessDial(Dial):
    name = "earnestness"
    description = "how much the room means it, [0 ironic .. 1 sincere]"

    def read(self, room: Room) -> float:
        if not room.messages:
            return 0.5
        sincere = hedge = 0
        for m in room.messages:
            text = m.text.lower()
            words = set(m.words)
            sincere += len(words & SINCERE)
            hedge += sum(1 for h in HEDGE if h in text)
        total = sincere + hedge
        if total == 0:
            return 0.5
        return sincere / total
