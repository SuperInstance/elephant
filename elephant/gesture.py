"""Vibe as gesture — the geometry of a room's motion through dial-space.

A ``RoomField`` is a *point*: the room's temperature right now, a vector of dial
readings. `elephant` already knows the room moves — reverberation, ripples, a
zeitgeist that drifts — but Vibe has mostly been read after the fact, as the gap
between two snapshots (``RoomField.distance``).

This module reads the **motion itself**. A sequence of field readings is not a
list of points; it is the *path* the room traces through dial-space — a gesture.
And a gesture has geometry a single snapshot cannot hold:

* ``arc_length`` — how far the vibe has travelled (a placid hour vs. a wild one),
* ``bending_energy`` — how much the mood keeps *turning* (a steady warming vs. a
  room that lurches between moods),
* ``heading`` — the unit direction of the latest step: **d_mu**, where the vibe
  is going *now*. This is the first-class Vibe velocity the fleet's other nodes
  (musician-soul's ``soul_spline`` tangent, tensor-midi's ``Clip.tangent``) read
  as a persona's or a conversation's ``d_mu``; here it is the room's.

The framing is shared across the fleet: a tensor approximates a function; we
approximate the *abstraction* — the smooth motion between states. See
``docs/GESTURE.md``.

Pure ``numpy`` (elephant's core dependency). Never raises on empty/degenerate
input.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np

from .field import RoomField, field_vector_for

FieldLike = Union[RoomField, np.ndarray, Sequence[float]]


def _stack(fields: Sequence[FieldLike], names: Optional[Sequence[str]]) -> np.ndarray:
    """Stack a sequence of field readings into a (T, D) matrix.

    ``RoomField`` inputs are vectorized with ``names`` (defaulting to the
    canonical ``DIAL_NAMES`` order, so every row aligns); array-likes pass
    through. Returns an empty ``(0, 0)`` array for no input.
    """
    rows: List[np.ndarray] = [
        np.asarray(field_vector_for(f, names), dtype=float) for f in fields
    ]
    if not rows:
        return np.empty((0, 0))
    return np.vstack(rows)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros_like(v)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class VibeTrajectory:
    """The path a room's field traces through dial-space over time.

    Args:
        fields: an ordered sequence of ``RoomField`` (or plain vectors), oldest
            first — successive readings of the same room.
        names: dial ordering for vectorizing ``RoomField`` inputs; defaults to
            the canonical ``DIAL_NAMES``.
    """

    def __init__(self, fields: Sequence[FieldLike], names: Optional[Sequence[str]] = None):
        self.points = _stack(list(fields), names)  # shape (T, D)

    def __len__(self) -> int:
        return int(self.points.shape[0])

    @property
    def dim(self) -> int:
        return int(self.points.shape[1]) if self.points.ndim == 2 and self.points.shape[0] else 0

    def steps(self) -> np.ndarray:
        """The (T-1, D) array of consecutive step vectors (the discrete velocity)."""
        if len(self) < 2:
            return np.empty((0, self.dim))
        return np.diff(self.points, axis=0)

    def arc_length(self) -> float:
        """Total distance travelled through dial-space. 0.0 for < 2 readings."""
        s = self.steps()
        if s.shape[0] == 0:
            return 0.0
        return float(np.linalg.norm(s, axis=1).sum())

    def speed(self) -> float:
        """Magnitude of the latest step — how fast the vibe is moving now."""
        s = self.steps()
        return float(np.linalg.norm(s[-1])) if s.shape[0] else 0.0

    def heading(self) -> np.ndarray:
        """Unit direction of the latest step — the room's **d_mu** (where the
        vibe is heading now). Zero vector for < 2 readings or a still room."""
        s = self.steps()
        return _unit(s[-1]) if s.shape[0] else np.zeros(self.dim)

    def bending_energy(self) -> float:
        """Total turning of the gesture: summed ``1 - cos`` between consecutive
        step directions. 0.0 for a straight drift (the room warming steadily) at
        any speed; large for a room that keeps lurching between moods."""
        s = self.steps()
        if s.shape[0] < 2:
            return 0.0
        energy = 0.0
        for i in range(1, s.shape[0]):
            if np.linalg.norm(s[i - 1]) > 1e-12 and np.linalg.norm(s[i]) > 1e-12:
                energy += 1.0 - _cos(s[i - 1], s[i])
        return float(energy)

    def restlessness(self) -> float:
        """Bending energy per step — a scale-free "how much does this room keep
        changing its mind" in [0, ~2]. 0.0 when there are too few turns."""
        s = self.steps()
        turns = max(0, s.shape[0] - 1)
        return self.bending_energy() / turns if turns else 0.0


def heading_alignment(a: VibeTrajectory, b: VibeTrajectory) -> float:
    """Do two rooms trend the same way? Cosine of their d_mu headings, in
    [-1, 1] (1 = moving in the same direction, -1 = opposite). 0.0 if either is
    still or their dial spaces differ in dimension."""
    ha, hb = a.heading(), b.heading()
    if ha.shape != hb.shape:
        return 0.0
    return _cos(ha, hb)
