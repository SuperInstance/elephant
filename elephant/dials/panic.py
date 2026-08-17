"""Panic dial — the stampede sense.

"Panic causing mob trampling triggered by a fire." This dial reads the
room's alarm state: alarm words, message-rate spikes, and the cascade
velocity of a triggering message. A room can go from warm to stampede
in three messages — the dial's job is to feel that transition before
anyone has said the word "fire" twice.
"""
from __future__ import annotations

import math

from ..dial import Dial
from ..room import Room

ALARM = {
    "fire", "flood", "breach", "leak", "alarm", "emergency", "evacuate",
    "sinking", "capsize", "mayday", "help", "panic", "stampede", "crash",
    "collision", "man overboard", "distress", "code red", "abandon", "run",
}
URGENCY = {"now", "immediately", "hurry", "fast", "everyone", "all hands",
           "!!!", "???", "now!", "right now", "go go go"}


class PanicDial(Dial):
    name = "panic"
    description = "stampede sense, [0 calm .. 1 trampling]"

    def read(self, room: Room) -> float:
        if not room.messages:
            return 0.0
        alarm_hits = 0
        urgency_hits = 0
        trigger_ripple = 0
        for m in room.messages:
            text = m.text.lower()
            alarm_hits += sum(1 for a in ALARM if a in text)
            urgency_hits += sum(1 for u in URGENCY if u in text)
            if any(a in text for a in ALARM):
                trigger_ripple = max(trigger_ripple, room.ripple(m))
        # Density spike: how hard the room is pounding right now.
        density = room.density(window=30.0)
        density_norm = 1.0 - math.exp(-density / 30.0)
        word_count = sum(len(m.words) for m in room.messages)
        alarm_norm = min(1.0, alarm_hits / max(word_count / 40.0, 1.0))
        urgency_norm = min(1.0, urgency_hits / 5.0)
        ripple_norm = min(1.0, trigger_ripple / 20.0)
        panic = 0.40 * alarm_norm + 0.25 * urgency_norm + 0.20 * ripple_norm + 0.15 * density_norm
        return max(0.0, min(1.0, panic))
