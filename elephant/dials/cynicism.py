"""Cynicism dial — how much the room is rolling its eyes.

The ironic twin of earnestness. A room can be warm and cynical (the
Tap after a long day), or cold and cynical (a thread that gave up).
This dial reads the sneer: scare quotes, "sure, sure", "oh great",
the emoji that means the opposite of what it shows.
"""
from __future__ import annotations

from ..dial import Dial
from ..room import Room

CYNICAL = {
    "sure", "right", "uh-huh", "uh huh", "yeah right", "oh great", "of course",
    "whatever", "suuuure", "great.", "nice.", "lovely.", "just great",
    "totally", "definitely not", "as if", "ha", "ha.", "lol ok", "ok sure",
    "sure thing", "obviously", "clearly", "sarcasm", "irony", "eyeroll",
}
_QUOTE = '"'
EYEROLL = {"🙄", "😒", "😏", "🤨"}


class CynicismDial(Dial):
    name = "cynicism"
    description = "how much the room is rolling its eyes, [0 earnest .. 1 sneering]"

    def read(self, room: Room) -> float:
        if not room.messages:
            return 0.5
        hits = 0
        quoted = 0
        total_words = 0
        for m in room.messages:
            text = m.text.lower()
            words = set(m.words)
            total_words += len(m.words)
            hits += len(words & CYNICAL)
            quoted += m.text.count(_QUOTE) // 2  # pairs of scare quotes
            hits += sum(1 for e in EYEROLL if e in m.text)
        if total_words == 0:
            return 0.5
        raw = (hits + quoted) / max(total_words, 1)
        return max(0.0, min(1.0, raw * 40.0))  # scale: 2.5% cynical tokens -> 1.0
