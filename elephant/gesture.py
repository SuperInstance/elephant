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

    def twist_energy(self) -> float:
        """Total **twist** of the gesture — its third-order structure: the
        turning that leaves the *osculating plane*.

        ``bending_energy`` is curvature — how much the vibe turns *within* a
        plane of mood-space. Twist is torsion — how much it turns *out* of that
        plane, into a fresh mood dimension. A room whose mood keeps swinging in
        one plane (mood up as panic falls, back and forth) has high bending but
        **zero twist**; a room whose swing keeps recruiting a *new* dial each
        turn has positive twist.

        Per interior vertex the contribution is ``sin θ``, where θ is the angle
        between the next step and the osculating plane of the previous two, so
        each vertex contributes in ``[0, 1]`` and a straight or planar drift
        contributes 0. Needs ≥4 readings to be non-zero.

        This is the fleet's *the property is in the twist* (see
        SuperInstance/twist-engine, musician-soul's ``AbstractionSpline.
        twist_energy``): new structure lives in the offset that leaves the
        current plane, not in more turning within it.
        """
        s = self.steps()
        if s.shape[0] < 3:
            return 0.0
        energy = 0.0
        for i in range(1, s.shape[0] - 1):
            s1, s2, s3 = s[i - 1], s[i], s[i + 1]
            n1 = float(np.linalg.norm(s1))
            if n1 < 1e-12:
                continue
            e1 = s1 / n1
            perp = s2 - np.dot(s2, e1) * e1  # s2 orthogonal to e1
            npn = float(np.linalg.norm(perp))
            if npn < 1e-12:
                continue  # s1 ∥ s2: no plane to leave
            e2 = perp / npn
            n3 = float(np.linalg.norm(s3))
            if n3 < 1e-12:
                continue
            d3 = s3 / n3
            out = d3 - np.dot(d3, e1) * e1 - np.dot(d3, e2) * e2
            energy += min(float(np.linalg.norm(out)), 1.0)
        return energy

    def planarity(self) -> float:
        """How flat the gesture stays, in ``[0, 1]``: ``1.0`` for a vibe whose
        whole motion lives in one plane (all bending, no twist), falling toward
        ``0.0`` as more of its turning leaves the plane. ``1.0`` for a
        trajectory too short to twist. The scale-free inverse of
        ``twist_energy``."""
        s = self.steps()
        vertices = max(0, s.shape[0] - 2)
        if vertices == 0:
            return 1.0
        return float(np.clip(1.0 - self.twist_energy() / vertices, 0.0, 1.0))


def heading_alignment(a: VibeTrajectory, b: VibeTrajectory) -> float:
    """Do two rooms trend the same way? Cosine of their d_mu headings, in
    [-1, 1] (1 = moving in the same direction, -1 = opposite). 0.0 if either is
    still or their dial spaces differ in dimension."""
    ha, hb = a.heading(), b.heading()
    if ha.shape != hb.shape:
        return 0.0
    return _cos(ha, hb)
