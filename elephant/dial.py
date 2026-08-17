"""The dial: one JEPA perceiving one dimension of the room.

Casey's reframing: JEPA is not the answer to everything — it is a
sensory perception attuned to the warmth or coldness of the room, one
dial among many. A room can have MANY JEPA models perceiving vibes on
more than one dimension at once: mood, volume, earnestness, cynicism,
whether a joke landed (the collective laugh or boo of the whole
audience), panic spreading like a stampede. Like pheromones — one
dimension of one of many JEPAs affecting agents constantly, without
words. It's a dial, not a description.

Each dial reads a Room and reports a scalar reading (and optionally a
windowed time-series). The ensemble of dials is the Field — the
elephant.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type

from .room import Room


class Dial(ABC):
    """A JEPA sense for one dimension of a room's vibe."""

    name: str = "dial"
    description: str = ""

    @abstractmethod
    def read(self, room: Room) -> float:
        """Current reading of this dimension for the room. Scalar, ~[-1,1]
        or [0,1] depending on the dial; must be self-normalizing."""

    def series(self, room: Room, window: int = 8) -> List[float]:
        """Optional: windowed readings so the dial can be trained over time."""
        return [self.read(room)]  # default: static reading

    def __repr__(self) -> str:
        return f"<Dial {self.name}>"


class DialBank:
    """Many JEPAs, many dimensions, one room. The perceiving ensemble."""

    def __init__(self, dials: Optional[Iterable[Dial]] = None):
        self.dials: List[Dial] = list(dials or [])

    def add(self, dial: Dial) -> "DialBank":
        self.dials.append(dial)
        return self

    def readings(self, room: Room) -> Dict[str, float]:
        return {d.name: d.read(room) for d in self.dials}

    def series(self, room: Room) -> Dict[str, List[float]]:
        return {d.name: d.series(room) for d in self.dials}

    def names(self) -> List[str]:
        return [d.name for d in self.dials]

    def __len__(self) -> int:
        return len(self.dials)
