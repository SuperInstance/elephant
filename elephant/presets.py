"""Presets — the two elephants: Room-Elephant and Personal-Elephant.

The captain's architectural expansion (`docs/jepa-zeitgeist-2026-08-17.md`):

    A *personal* JEPA is subjective to the agent reading it — shaped by that
    agent's learned experience, their reflexes, and their intangible
    correlations (the perfume that is grandma's shop, the song that is the
    lover you discovered the album with). But the room itself also has an
    overall **zeitgeist** — a vibe that exists whether or not any particular
    agent is in it. That is a different animal: objective-as-first-class.

So the repo carries **two** presets, tuned for two distinct jobs:

- **RoomElephant** — the room's own reading. Objective, first-class, *not* any
  agent's view. It IS the room: neutral defaults, no agent bias, and a stable
  identity (the room's field does not drift with any one agent). This drives
  the MUD environment, the NPC vibes, the room's own description — the
  input-tokens every agent sees.

- **PersonalElephant** — one agent's subjective reading. Wraps the objective
  field with that agent's taste (`dial_weights` — which dials matter to them),
  their disposition (`bias`), and their `attachments` (the intangible
  correlations: event key -> memory). This drives that agent's reactions,
  decisions, and memories.

`PRESETS` is the registry both jobs hang off of:

    PRESETS = {"room": RoomElephant, "personal": PersonalElephant}

numpy-only. The objective field comes straight from the dial bank
(`field.read_field`); the subjective field is the objective field weighted by
taste, shifted by disposition, and clamped to the dials' bounds.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np

from .dial import DialBank
from .dials import DEFAULT_DIALS
from .field import DIAL_NAMES, RoomField, read_field
from .room import Room
from .tapnight import DIAL_BOUNDS, DIAL_CENTER


def _to_vector(x, default=0.0) -> np.ndarray:
    """Coerce a dial-space value into a 7-vector in DIAL_NAMES order.

    Accepts a dict {dial_name: value} (missing keys -> `default`, which may be
    a scalar or a per-dial dict), or any array-like of 7 floats.
    """
    if x is None:
        x = default
    if isinstance(x, dict):
        if isinstance(default, dict):
            return np.array([float(x.get(n, default[n])) for n in DIAL_NAMES],
                            dtype=float)
        return np.array([float(x.get(n, default)) for n in DIAL_NAMES],
                        dtype=float)
    if np.isscalar(x):
        return np.full(len(DIAL_NAMES), float(x), dtype=float)
    return np.asarray(list(x), dtype=float)


class RoomElephant:
    """The room's own reading — objective, first-class citizen.

    This is the zeitgeist, not any agent's view of it. It reads the room
    through the plain dial bank, with **neutral defaults**: presence rests at
    0.5, volume at 0 (quiet), the signed dials (mood, joke_landing) at 0, and
    earnestness/cynicism at their un-sneering, un-ironic center. There is no
    per-agent weight and no bias — two different agents reading the *same*
    room through the Room-Elephant get the *same* field. The field belongs to
    the room; it does not drift with any one agent (that drift is the
    Personal-Elephant's job, and the charisma dynamics in `field`/`tapnight`).
    """

    # First-class neutral defaults: the room at rest. This is the room's own
    # identity — the zero of the zeitgeist, against which every actual reading
    # is a deviation.
    NEUTRAL: Dict[str, float] = dict(DIAL_CENTER)

    def __init__(self, identity: str = "room",
                 bank: Optional[DialBank] = None):
        # `identity` is the room's stable name, not any agent's. It is what
        # makes the reading "the room's" rather than "someone's".
        self.identity = identity
        self.bank = bank or DialBank(DEFAULT_DIALS)

    def read(self, room: Room) -> RoomField:
        """The objective field — the room as it actually is.

        For an empty room there is nothing to read, so the field rests at its
        first-class neutral defaults (presence 0.5, volume 0, ...). For a room
        with life, this is exactly the dial bank's reading.
        """
        if room is None or len(room.messages) == 0:
            return RoomField(dict(self.NEUTRAL))
        return read_field(room, self.bank)

    def __repr__(self) -> str:
        return f"<RoomElephant identity={self.identity!r}>"


class PersonalElephant:
    """One agent's subjective reading of the room.

    Wraps the objective field (delegated to a Room-Elephant) with the agent's
    own three pieces of furniture:

    - `dial_weights` — the agent's **taste**: which dials matter to them. A
      prior over the 7 dimensions, normalized to sum to 1. A weight above 1/7
      amplifies that dial's *deviation from neutral* in the agent's reading;
      below 1/7 damps it. The dials an agent does not care about (weight ~0)
      read as **neutral**, not zero — an agent who ignores presence does not
      see the room as empty, they simply do not register it.
    - `bias` — the agent's **disposition**: a constant offset they bring to
      every room before anyone speaks. A warm writer leans mood; a sneering
      critic leans cynicism.
    - `attachments` — the **intangible correlations**: event key -> memory.
      The perfume that takes you to grandma's shop; the song that reminds you
      of the lover you discovered the album with. These are not dials; they
      are the subjective glue that makes one agent's room *feel* different
      from another's even at the same objective reading.
    """

    def __init__(self, name: str,
                 dial_weights: Optional[Union[Dict[str, float], Sequence[float]]] = None,
                 bias: Optional[Union[Dict[str, float], Sequence[float]]] = None,
                 room_elephant: Optional[RoomElephant] = None):
        self.name = name
        if dial_weights is None:
            self.dial_weights = np.full(len(DIAL_NAMES), 1.0 / len(DIAL_NAMES))
        else:
            self.dial_weights = _to_vector(dial_weights, default=0.0)
        self.dial_weights = np.maximum(self.dial_weights, 0.0)
        s = float(self.dial_weights.sum())
        if s > 1e-9:
            self.dial_weights = self.dial_weights / s
        self.bias = _to_vector(bias, default=0.0)
        self._center = np.array([DIAL_CENTER[n] for n in DIAL_NAMES], dtype=float)
        self.attachments: Dict[str, object] = {}
        # The objective anchor: every personal read is a deformation of the
        # room's own reading, so the two elephants are always comparable.
        self._room = room_elephant or RoomElephant()

    def objective(self, room: Room) -> RoomField:
        """The room as it actually is (the Room-Elephant's reading)."""
        return self._room.read(room)

    def read(self, room: Room) -> RoomField:
        """The agent's subjective field: objective dials weighted by taste,
        then shifted by disposition, clamped to the dials' bounds."""
        base = self._room.read(room).vector()
        # Weight the *deviation from neutral*, not the raw value: a dial the
        # agent doesn't care about (weight ~0) reads as neutral, not zero. A
        # uniform taste (1/7 each) returns the objective field un-deformed.
        delta = base - self._center
        subj = self._center + (self.dial_weights * len(DIAL_NAMES)) * delta + self.bias
        return RoomField(dict(zip(DIAL_NAMES, self._clamp(subj))))

    def attach(self, event_key: str, memory) -> "PersonalElephant":
        """Bind an intangible correlation: this event key -> this memory."""
        self.attachments[event_key] = memory
        return self

    def remember(self, event_key: str):
        """Recall the memory bound to an event key (None if none)."""
        return self.attachments.get(event_key)

    def _clamp(self, vec: np.ndarray) -> np.ndarray:
        out = vec.copy()
        for i, n in enumerate(DIAL_NAMES):
            lo, hi = DIAL_BOUNDS[n]
            out[i] = min(hi, max(lo, out[i]))
        return out

    def __repr__(self) -> str:
        return f"<PersonalElephant {self.name!r}>"


# The preset registry from the spec. Values are the *classes* (callable
# factories) so downstream can instantiate either elephant on demand.
PRESETS: Dict[str, type] = {
    "room": RoomElephant,
    "personal": PersonalElephant,
}
