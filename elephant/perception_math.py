"""Perception-check math — the elephant's internal-monologue pulse.

The captain's macro-read, in one line:

    *Like macro-economic currency-exchange changes: the NUMBER doesn't
    matter, but TWO numbers show DIRECTION and MORE THAN TWO show
    RATE OF CHANGE.*

A dial's absolute reading is noise — exactly as an exchange rate's
absolute level is noise (nobody cares that EUR/USD sits at 1.08; they
care that it *moved*). What carries signal is the movement *between*
readings:

- **Two readings → direction** — the first difference
  ``d = x_t − x_{t−1}``. Sign and magnitude per dial.
- **Three+ readings → rate of change** — the second difference
  ``a = (x_t − x_{t−1}) − (x_{t−1} − x_{t−2})``. Whether the room is
  warming *faster* or *slower*; whether a move is exhausting
  (decelerating) or cascading (accelerating).

This is the same one-math as ``fleetmath.three_reading_kinematics``
(position → velocity → acceleration) transplanted from radar targets to
room dials: a dial reading is a "position", its first difference is a
"velocity"/direction, its second difference is an "acceleration"/rate.
Boats on radar and dials in a room are two domains of one derivative
ladder.

numpy-only and deliberately free of the package's heavier imports, so
the math reads and tests on its own.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

Number = Union[float, int]
# A series is a mapping of dial name -> chronological readings, or a
# bare 1-D sequence treated as a single unnamed dial keyed "value".
Series = Union[Dict[str, Sequence[Number]], Sequence[Number]]


def _as_dict_of_series(series: Series) -> Dict[str, np.ndarray]:
    """Normalize `series` to ``{name: float ndarray}``.

    Accepts a mapping ``{dial: [r0, r1, ...]}`` or a bare 1-D sequence
    (wrapped under the key ``"value"``). NaN entries pass through; they
    are handled (excluded) at the aggregate level.
    """
    if isinstance(series, dict):
        return {str(k): np.asarray(v, dtype=float).reshape(-1)
                for k, v in series.items()}
    return {"value": np.asarray(series, dtype=float).reshape(-1)}


def _gaps(dt: Optional[Sequence[Number]] or Number, n_steps: int) -> np.ndarray:
    """Per-step time gaps (length ``n_steps``) from the ``dt`` argument.

    ``dt`` is ``None`` (unit ticks), a scalar (uniform gap), or a
    sequence of per-step gaps of length ``n_steps`` (i.e. one gap per
    difference). Gaps are floored at ``1e-9`` so normalization never
    divides by zero.
    """
    if dt is None:
        gaps = np.ones(n_steps, dtype=float)
    elif np.isscalar(dt):
        gaps = np.full(n_steps, float(dt), dtype=float)
    else:
        gaps = np.asarray(dt, dtype=float).reshape(-1)
        if gaps.size != n_steps:
            raise ValueError(
                f"dt has {gaps.size} gap(s) but {n_steps} step(s) are required")
    return np.maximum(gaps, 1e-9)


def _floor_for(noise_floor: Union[Number, Dict[str, Number]],
               name: str) -> float:
    """Per-dial deadband floor: a scalar applied to all dials, or a
    ``{name: floor}`` mapping (missing names default to 0.0)."""
    if isinstance(noise_floor, dict):
        return float(noise_floor.get(name, 0.0))
    return float(noise_floor)


def _deadband(values: np.ndarray, floor: float) -> List[float]:
    """Zero out values whose magnitude is *below* ``floor``.

    The deadband is where the elephant has no opinion: a move smaller
    than the noise floor is not a move, so it reads as 0. NaN passes
    through untouched (a missing reading is not a zero reading).
    """
    out = np.asarray(values, dtype=float).copy()
    if floor > 0.0:
        out[np.abs(out) < floor] = 0.0
    return out.tolist()


def direction(series: Series,
              dt: Optional[Sequence[Number]] or Number = None,
              noise_floor: Union[Number, Dict[str, Number]] = 0.0
              ) -> Dict[str, List[float]]:
    """Per-dimension first difference (time-normalized, deadbanded).

    For each dial ``x_0 … x_{n−1}`` returns the per-second direction

        d_i = (x_{i+1} − x_i) / Δt_i,    i = 0 … n−2.

    The dial's absolute level never appears — only its movement. Two
    readings give one direction; one reading gives none.

    - ``dt`` — ``None`` (unit ticks), a scalar (uniform gap), or the
      per-step gaps (length n−1).
    - ``noise_floor`` — a scalar or ``{name: floor}``; a per-second move
      below the floor reads as 0.

    Returns ``{name: [d_0, …]}`` (n−1 values per dial; empty for n < 2).
    """
    data = _as_dict_of_series(series)
    out: Dict[str, List[float]] = {}
    for name, x in data.items():
        n = x.shape[0]
        if n < 2:
            out[name] = []
            continue
        gaps = _gaps(dt, n - 1)
        d = (x[1:] - x[:-1]) / gaps
        out[name] = _deadband(d, _floor_for(noise_floor, name))
    return out


def rate_of_change(series: Series,
                   dt: Optional[Sequence[Number]] or Number = None,
                   noise_floor: Union[Number, Dict[str, Number]] = 0.0
                   ) -> Dict[str, List[float]]:
    """Per-dimension second difference (time-normalized, deadbanded).

    For each dial, the per-second² acceleration from three consecutive
    readings:

        a_i = (d_{i+1} − d_i) / Δt̄_i,   Δt̄_i = (Δt_i + Δt_{i+1}) / 2,

    where ``d_i`` is the per-second direction. For uniform ``Δt = 1``
    this is exactly the second difference

        a_i = (x_{i+2} − x_{i+1}) − (x_{i+1} − x_i) = x_{i+2} − 2 x_{i+1} + x_i.

    Positive = the move is *accelerating* (cascading); negative =
    *decelerating* (exhausting). The room can be warming faster or
    slower even while its direction is unchanged — that is the thing
    three readings reveal that two cannot.

    This is the exact acceleration of the quadratic interpolant through
    any three consecutive readings (for arbitrary — even non-uniform —
    spacing), identical to ``fleetmath``'s central second divided
    difference up to the same factor of 2.

    Returns ``{name: [a_0, …]}`` (n−2 values per dial; empty for n < 3).
    """
    data = _as_dict_of_series(series)
    out: Dict[str, List[float]] = {}
    for name, x in data.items():
        n = x.shape[0]
        if n < 3:
            out[name] = []
            continue
        gaps = _gaps(dt, n - 1)
        v = (x[1:] - x[:-1]) / gaps          # per-second directions
        dt_mid = (gaps[:-1] + gaps[1:]) / 2.0
        a = (v[1:] - v[:-1]) / dt_mid        # per-second² rate of change
        out[name] = _deadband(a, _floor_for(noise_floor, name))
    return out


def _weights(weights: Optional[Dict[str, Number]],
             names: List[str]) -> Dict[str, float]:
    """Normalized dial weights. ``None`` → equal; a dict is normalized
    over the dials actually present so absent dials don't dilute."""
    n = len(names)
    if weights is None:
        return {name: 1.0 / max(n, 1) for name in names}
    raw = {str(k): float(v) for k, v in dict(weights).items()}
    total = sum(max(raw.get(name, 0.0), 0.0) for name in names)
    if total <= 0.0:
        return {name: 1.0 / max(n, 1) for name in names}
    return {name: max(raw.get(name, 0.0), 0.0) / total for name in names}


def _finite(x: float) -> bool:
    """True for finite numbers; False for NaN/None (a dead dial)."""
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def composite_read(series: Series,
                   weights: Optional[Dict[str, Number]] = None,
                   noise_floor: Union[Number, Dict[str, Number]] = 0.0
                   ) -> Dict[str, object]:
    """The whole-hand read: one number for the room's conversation.

    Folds per-dial direction and rate into the macro picture a skipper
    feels all at once:

    - ``macro_direction`` — weighted mean of each dial's *latest*
      direction (the room's overall drift; the warmth trend).
    - ``macro_rate`` — weighted mean of each dial's *latest* rate (the
      room's overall acceleration; is the drift speeding up or
      exhausting).
    - ``fastest_dial`` — the dial with the largest |direction| right
      now: *what is driving the room*.
    - ``accelerating_dials`` — the dials whose |rate| exceeds the noise
      floor, sorted by |rate| descending: *what is about to matter*.
    - ``directions`` / ``rates`` — the per-dial latest values (the raw
      table the aggregates summarize).

    NaN readings are excluded from every aggregate: a dead dial has no
    opinion and is simply dropped, not averaged in as 0.
    """
    data = _as_dict_of_series(series)
    names = list(data.keys())
    d = direction(series, noise_floor=noise_floor)
    r = rate_of_change(series, noise_floor=noise_floor)
    w = _weights(weights, names)

    def latest(mapping: Dict[str, List[float]], name: str) -> float:
        vals = mapping.get(name, [])
        return float(vals[-1]) if vals else 0.0

    d_now = {name: latest(d, name) for name in names}
    r_now = {name: latest(r, name) for name in names}

    def weighted_mean(values: Dict[str, float]) -> float:
        # A dead (NaN) dial is dropped and the survivors re-normalize,
        # so a room with one dead dial still reads as a full room.
        live = [name for name in names if _finite(values[name])]
        wsum = sum(w[name] for name in live)
        if wsum <= 0.0:
            return 0.0
        return sum(w[name] * values[name] for name in live) / wsum

    macro_direction = weighted_mean(d_now)
    macro_rate = weighted_mean(r_now)

    finite_dirs = {name: abs(d_now[name])
                   for name in names if _finite(d_now[name])}
    fastest_dial = (max(finite_dirs, key=finite_dirs.get)
                    if finite_dirs else None)

    accelerating = [
        name for name in names
        if _finite(r_now[name])
        and abs(r_now[name]) > _floor_for(noise_floor, name)
    ]
    accelerating.sort(key=lambda name: -abs(r_now[name]))

    return {
        "macro_direction": float(macro_direction),
        "macro_rate": float(macro_rate),
        "fastest_dial": fastest_dial,
        "accelerating_dials": accelerating,
        "directions": d_now,
        "rates": r_now,
    }
