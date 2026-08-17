"""Joke-landing dial — the COLLECTIVE laugh or boo.

The captain: "whether a joke really landed because of the overall
sense of all the agents in the audience laughing or booing as a
collective." This is not about the joke itself — it's the room's
response to it. Find the joke, then read the audience: laughter
markers and warm reactions vs crickets, groans, and thumbs-down.
The dial is the audience, not the comedian.
"""
from __future__ import annotations

from ..dial import Dial
from ..room import Room

JOKE_MARKERS = {
    "lol", "haha", "heh", "funny", "joke", "that's what she said", "ba dum",
    "dad joke", "punchline", "kidding", "just kidding", "😂", "🤣", "😄",
}
LAUGH = {"lol", "lmao", "rofl", "haha", "hehe", "heh", "😂", "🤣", "😄", "💀", "gold", "dead"}
BOO = {"boo", "crickets", "groan", "👎", "🙄", "😐", "that was bad", "tough crowd",
       "womp", "yikes", "cringe", "😬", "no", "nope", "who let him cook"}


class JokeLandingDial(Dial):
    name = "joke_landing"
    description = "did the jokes land, [-1 booed/crickets .. +1 roared]"

    def read(self, room: Room) -> float:
        if not room.messages:
            return 0.0
        scores = []
        for i, m in enumerate(room.messages):
            text = m.text.lower()
            if not any(marker in text for marker in JOKE_MARKERS):
                continue
            # The audience = the room's response after the joke lands.
            window = room.messages[i + 1:i + 5]
            laugh = 0.0
            boo = 0.0
            for w in window:
                wt = w.text.lower()
                laugh += sum(1 for k in LAUGH if k in wt)
                boo += sum(1 for k in BOO if k in wt)
            # Reactions on the joke itself are also the crowd's hands.
            laugh += 0.5 * sum(
                c for e, c in m.reactions.items() if e in {"😂", "🤣", "😄", "👍", "❤️"}
            )
            boo += 0.5 * sum(
                c for e, c in m.reactions.items() if e in {"👎", "🙄", "😐", "💩"}
            )
            total = laugh + boo
            if total > 0:
                scores.append((laugh - boo) / total)
        if not scores:
            return 0.0
        return sum(scores) / len(scores)
