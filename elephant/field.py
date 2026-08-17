"""The Field — the elephant.

A room is not its messages. It is the ensemble of what every dial
feels at once: mood, volume, earnestness, cynicism, whether the joke
landed, whether panic is spreading, whose pheromones still hang in
the air. That ensemble is the room's temperature — the elephant.

The v3 design (fleet-jepa-midi `research/elephant-sense-v3-design-2026-08-17.md`)
frames the field as a von Mises–Fisher direction μ̂ with concentration κ:
**cold room = high κ (tight, rigid, one way to be), warm room = low κ
(loose, many ways to be)**. This module is the pragmatic v0 of that:
a normalized dial-vector field with the same dynamics — contrast
between rooms, acclimation toward the room, charisma pulling the room.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .dial import DialBank
from .room import Room

DIAL_NAMES = [
    "mood", "volume", "earnestness", "cynicism",
    "joke_landing", "panic", "presence",
]


class RoomField:
    """The ensemble of dial readings — the room's temperature vector."""

    def __init__(self, readings: Dict[str, float]):
        self.readings = dict(readings)

    # ------------------------------------------------------------------ #
    # Vector form                                                        #
    # ------------------------------------------------------------------ #
    def vector(self, names: Optional[Sequence[str]] = None) -> np.ndarray:
        names = names or DIAL_NAMES
        return np.array([self.readings.get(n, 0.0) for n in names], dtype=float)

    def normalize(self, names: Optional[Sequence[str]] = None) -> np.ndarray:
        v = self.vector(names)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    # ------------------------------------------------------------------ #
    # Temperature                                                        #
    # ------------------------------------------------------------------ #
    def warmth(self) -> float:
        """The felt temperature: warm dials up, cold dials down.

        mood & joke_landing run [-1,+1]; the rest run [0,1] and are
        re-centered. Panic and cynicism are cold; presence and
        earnestness are warm; volume is heat (the sauna's roar).
        """
        r = self.readings
        return (
            0.30 * r.get("mood", 0.0)
            + 0.15 * r.get("joke_landing", 0.0)
            + 0.10 * (r.get("earnestness", 0.5) - 0.5) * 2
            + 0.10 * (r.get("presence", 0.0) - 0.5) * 2
            + 0.10 * (r.get("volume", 0.0) - 0.5) * 2
            - 0.15 * r.get("cynicism", 0.5)
            - 0.10 * r.get("panic", 0.0)
        )  # ~[-1, +1]

    def concentration(self) -> float:
        """κ: how tight the room is. Cold rooms are one way; warm rooms
        are many ways. Computed as the norm of the centered field."""
        return float(np.linalg.norm(self.vector() - 0.5)) * 2.0

    # ------------------------------------------------------------------ #
    # Contrast — the sauna / cold plunge                                 #
    # ------------------------------------------------------------------ #
    def distance(self, other: "RoomField") -> float:
        """The elephant gap: distance between two rooms' fields.

        This is what makes the elephant real. Inside one room it is
        invisible; walk into a different room and it is a very
        different elephant.
        """
        a = self.normalize()
        b = other.normalize()
        return float(np.linalg.norm(a - b))

    def sauna_plunge_gap(self, other: "RoomField") -> float:
        """Signed warmth contrast: + = this room is warmer than other,
        - = this room is colder. The plunge you feel on entry."""
        return self.warmth() - other.warmth()

    def __repr__(self) -> str:
        return f"RoomField(warmth={self.warmth():+.2f}, κ={self.concentration():.2f})"


def read_field(room: Room, bank: Optional[DialBank] = None) -> RoomField:
    """Read a room with the dial bank -> its field."""
    from .dials import DEFAULT_DIALS
    bank = bank or DialBank(DEFAULT_DIALS)
    return RoomField(bank.readings(room))


# ---------------------------------------------------------------------- #
# Dynamics: acclimation (agent -> room) and charisma (room -> agent)     #
# ---------------------------------------------------------------------- #
def acclimation_curve(
    agent: np.ndarray,
    room: np.ndarray,
    rate: float,
    t: float,
) -> np.ndarray:
    """Agent embedding relaxing toward the room field.

    The newcomer warms to the room — quickly or slowly depending on
    how experienced, talented, and trained they are at modulating
    their vibe toward the room. `rate` IS that skill (1/τ).
    Exponential relaxation: a(t) = room + (agent - room)·e^(-rate·t).
    """
    return room + (agent - room) * math.exp(-rate * t)


def acclimation_rate_from(
    agent_start: np.ndarray,
    agent_obs: np.ndarray,
    room: np.ndarray,
    t: float,
) -> float:
    """Invert the curve: given where the agent started, where they are
    now, and the room, recover their modulation skill rate."""
    delta = agent_start - room
    remain = agent_obs - room
    dn = float(np.linalg.norm(delta))
    if dn < 1e-9 or t <= 0:
        return 0.0
    ratio = max(0.0, min(1.0, float(np.dot(remain, delta)) / (dn * dn)))
    if ratio <= 0:
        return float("inf")
    return -math.log(ratio) / t


def charisma_pull(
    room: np.ndarray,
    agent: np.ndarray,
    charisma: float,
    interactions: int,
) -> np.ndarray:
    """A strong presence pulls the room's vibe toward them.

    Over time and interactions, the charismatic agent doesn't just
    acclimate — the room warms to THEM. Each interaction bends the
    field a little; `charisma` is the bend per interaction.
    """
    delta = agent - room
    return room + delta * (1.0 - math.exp(-charisma * interactions))


def field_vector_for(agent_or_room: "RoomField | np.ndarray",
                     names: Optional[Sequence[str]] = None) -> np.ndarray:
    if isinstance(agent_or_room, RoomField):
        return agent_or_room.vector(names)
    return np.asarray(agent_or_room, dtype=float)
