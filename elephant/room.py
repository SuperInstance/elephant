"""Rooms: message streams with gravity, reverberation, and ripples.

A room is any place agents leave traces — a message board, a chatroom,
a messenger conversation, an X thread, a bar. Rooms are not streams to
be ordered; they are fields with physics. A message has GRAVITY (how
hard it pulls attention), REVERBERATION (how it echoes after it's said),
and RIPPLE (how it propagates through replies, reactions, and the
collective).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field as dc_field
from typing import Dict, Iterable, List, Optional

WORD_RE = re.compile(r"\w+")


@dataclass
class Message:
    author: str
    text: str
    ts: float = 0.0
    channel: str = "default"
    reactions: Dict[str, int] = dc_field(default_factory=dict)
    replies: List["Message"] = dc_field(default_factory=list)

    @property
    def words(self) -> List[str]:
        return WORD_RE.findall(self.text.lower())

    @property
    def reaction_heat(self) -> int:
        """Total reaction weight — the crowd's hands."""
        return sum(self.reactions.values())


class Room:
    """A sequence of messages with the physics of a gathered room."""

    def __init__(self, name: str, messages: Optional[Iterable[Message]] = None):
        self.name = name
        self.messages: List[Message] = list(messages or [])
        self.messages.sort(key=lambda m: m.ts)

    # ------------------------------------------------------------------ #
    # Gravity — how hard a message pulls the room's attention            #
    # ------------------------------------------------------------------ #
    def gravity(self, msg: Message, half_life: float = 1800.0,
                engagement_weight: float = 1.0) -> float:
        """Recency-weighted pull: fresh + reacted-to messages dominate.

        Like a good line at the bar: it lands, and for a while the room
        leans toward it. Then it cools into the walls.
        """
        age = max(0.0, msg.ts - (self.messages[0].ts if self.messages else msg.ts))
        recency = 0.5 ** (age / half_life)
        engagement = 1.0 + engagement_weight * math.log1p(msg.reaction_heat + len(msg.replies))
        length = 1.0 + math.log1p(len(msg.words)) / 10.0
        return recency * engagement * length

    def gravity_series(self, half_life: float = 1800.0) -> List[float]:
        return [self.gravity(m, half_life) for m in self.messages]

    # ------------------------------------------------------------------ #
    # Reverberation — how the past echoes in the present                 #
    # ------------------------------------------------------------------ #
    def reverberation(self, window: int = 8) -> float:
        """Mean similarity between consecutive windows of message heat.

        High reverberation = the room keeps returning to the same theme,
        the same beat, the same argument — the sound bouncing off the
        wood walls. Low = new ground being broken.
        """
        heats = self.gravity_series()
        if len(heats) < 2 * window:
            return 0.0
        windows = [heats[i:i + window] for i in range(0, len(heats) - window, window)]
        if len(windows) < 2:
            return 0.0
        sims = []
        for a, b in zip(windows[:-1], windows[1:]):
            sims.append(_cosine(a, b))
        return sum(sims) / len(sims) if sims else 0.0

    # ------------------------------------------------------------------ #
    # Ripple — how a message propagates through the room                 #
    # ------------------------------------------------------------------ #
    def ripple(self, msg: Message, depth: int = 3) -> int:
        """Cascade size: the message's reach through replies + reactions.

        A joke that lands ripples through laughter. A fire ripples
        through panic. A cold take ripples through silence (small).
        """
        if depth <= 0:
            return 0
        size = msg.reaction_heat + len(msg.replies)
        for r in msg.replies:
            size += self.ripple(r, depth - 1)
        return size

    def ripple_series(self, depth: int = 3) -> List[int]:
        return [self.ripple(m, depth) for m in self.messages]

    # ------------------------------------------------------------------ #
    # Density — the room's pulse (messages per second)                   #
    # ------------------------------------------------------------------ #
    def density(self, window: float = 300.0) -> float:
        """How fast the room is talking. The pulse that panic speeds up
        and the sauna slows down."""
        if len(self.messages) < 2:
            return 0.0
        span = max(self.messages[-1].ts - self.messages[0].ts, 1e-9)
        return len(self.messages) / span * 60.0  # msgs per minute

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        return f"Room({self.name!r}, {len(self.messages)} messages)"


def _cosine(a: List[float], b: List[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)
