"""The math of the fleet field — the inter-model temperature with sea legs.

This module is the *numeric* core of the elephant's sea legs. Where
`elephant/sensors.py` defines the dials (the JEPAs that read radar,
sounder, nav) and `elephant/field.py` defines the room field, this
module holds the rigorous mathematics underneath the fleet dials:

- **Kinematics** — recover direction, speed, and rate of change
  (acceleration) of every radar object from exactly *three* readings,
  with nearest-neighbour association and optional own-ship motion
  compensation (the lever-arm correction).

- **Fleet coherence** — the fleet as a von Mises–Fisher (vMF)
  distribution over boat headings/bearings, with concentration κ as the
  "tight (on fish) vs scattered (searching)" statistic, and dκ/dt as the
  scatter/bunch signal.

- **Inductive biomass** — the good-days → spotty-days induction. A
  Gaussian *anchor* over good-fishing-day feature vectors, and a
  Mahalanobis deviation score that answers "does this stretch of water
  feel like the good kind?"

Everything here is numpy-only and deliberately free of the package's
heavier imports, so the math can be read and tested on its own.

Units convention (fixed throughout): positions in **metres**, times in
**seconds**, speeds reported in **knots** (1 kt = 0.514444 m/s), angles
in **degrees**, accelerations in **m/s²**.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

MPS_TO_KTS = 1.9438444924406046      # metres/second -> knots
_LARGE_KAPPA = 1e4                   # cap for near-perfect alignment


# --------------------------------------------------------------------- #
# Kinematics — direction, speed, rate of change from three readings      #
# --------------------------------------------------------------------- #
def _positions(frame: Any) -> np.ndarray:
    """Extract an (N, 2) position array from a frame.

    Accepts a frame object with a `.data` attribute (as in
    `elephant/sensors.py` `SensorFrame`) or a bare array/list of (x, y).
    """
    # A numpy array *is* the positions (note: ndarray also has a `.data`
    # memoryview attribute, so test ndarray first).
    if isinstance(frame, np.ndarray):
        return frame.reshape(-1, 2)
    data = getattr(frame, "data", frame)
    if isinstance(data, np.ndarray):
        return data.reshape(-1, 2)
    return np.asarray(list(data), dtype=float).reshape(-1, 2)


def _ts(frame: Any) -> float:
    """Timestamp of a frame, defaulting to 0.0."""
    return float(getattr(frame, "ts", 0.0))


def _associate_idx(a: np.ndarray, b: np.ndarray,
                   gate: float = 2.0) -> List[Tuple[int, int]]:
    """Greedy nearest-neighbour association with a hard gate.

    Mirrors `elephant/sensors.py`'s `_associate` gating (nearest
    neighbour within `gate`), but returns *index* pairs so tracks can be
    linked across three frames unambiguously.
    """
    a = np.asarray(a, dtype=float).reshape(-1, 2)
    b = np.asarray(b, dtype=float).reshape(-1, 2)
    pairs: List[Tuple[int, int]] = []
    used: set = set()
    for i in range(a.shape[0]):
        best_j, best_d = -1, gate
        for j in range(b.shape[0]):
            if j in used:
                continue
            d = float(np.linalg.norm(a[i] - b[j]))
            if d < best_d:
                best_j, best_d = j, d
        if best_j >= 0:
            used.add(best_j)
            pairs.append((i, best_j))
    return pairs


def three_reading_kinematics(
    frames: Sequence[Any],
    own_ship: Optional[Sequence] = None,
    gate: float = 2.0,
) -> Dict[str, Any]:
    """Direction, speed, and acceleration of every radar object from
    exactly three readings.

    Given three radar sweeps at times ``t1 < t2 < t3`` (each a set of
    (x, y) target positions in the boat-relative frame), recover, for
    every object present at ``t3``:

    - **direction** ``dir_deg`` — angle of the most recent velocity
      (math convention, CCW from +x / "east"),
    - **speed** ``speed_kts`` — magnitude of that velocity, in knots,
    - **acceleration** ``accel`` — the central second difference
      ``2 (v23 - v12) / (t3 - t1)``, which is the *exact* acceleration
      of the quadratic interpolant through the three positions (and the
      best achievable with only three samples).

    ``own_ship``, if given (shape (3, 2) positions at the same three
    timestamps), removes the own-ship translation so the recovered
    velocities are water/geodetic-referenced rather than boat-relative
    (the lever-arm correction's dominant term).

    Returns a dict with ``objects`` (per-object kinematics), a
    ``fleet_mean_speed`` (knots), and a ``spread_rate`` (the rate at
    which the fleet's spatial spread changes — positive = scattering,
    negative = bunching). Objects without a three-frame track report
    ``accel = [0, 0]`` (unknown) rather than a fabricated value.
    """
    result: Dict[str, Any] = {"objects": [], "fleet_mean_speed": 0.0,
                              "spread_rate": 0.0}
    frames = list(frames)[-3:]
    if len(frames) < 3:
        return result

    t = [_ts(f) for f in frames]
    P = [_positions(f) for f in frames]

    if own_ship is not None:
        os_ = np.asarray(own_ship, dtype=float).reshape(-1, 2)
        os_ = os_[-3:]
        origin = os_[0]
        P = [p + (o - origin) for p, o in zip(P, os_)]

    t1, t2, t3 = t
    dt12 = max(t2 - t1, 1e-9)
    dt23 = max(t3 - t2, 1e-9)
    dt13 = max(t3 - t1, 1e-9)

    pairs12 = _associate_idx(P[0], P[1], gate)
    pairs23 = _associate_idx(P[1], P[2], gate)
    pred_of = {i2: i1 for (i1, i2) in pairs12}   # frame-2 index -> frame-1 index

    objects = []
    for (i2, i3) in pairs23:
        p2 = P[1][i2]
        p3 = P[2][i3]
        v23 = (p3 - p2) / dt23
        speed_mps = float(np.linalg.norm(v23))
        dir_deg = float(np.degrees(np.arctan2(v23[1], v23[0])))
        accel = np.zeros(2)
        if i2 in pred_of:
            p1 = P[0][pred_of[i2]]
            v12 = (p2 - p1) / dt12
            accel = 2.0 * (v23 - v12) / dt13
        objects.append({
            "pos": p2.tolist(),
            "vel": v23.tolist(),
            "dir_deg": dir_deg,
            "speed_mps": speed_mps,
            "speed_kts": speed_mps * MPS_TO_KTS,
            "accel": accel.tolist(),
            "accel_mag": float(np.linalg.norm(accel)),
        })

    speeds = [o["speed_kts"] for o in objects]
    spreads = [_spread(p) for p in P]
    spread_rate = (spreads[-1] - spreads[0]) / dt13 if dt13 > 0 else 0.0

    result["objects"] = objects
    result["fleet_mean_speed"] = float(np.mean(speeds)) if speeds else 0.0
    result["spread_rate"] = float(spread_rate)
    return result


def _spread(pts: np.ndarray) -> float:
    """Mean distance of points to their centroid (the model-free spread)."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if pts.shape[0] < 2:
        return 0.0
    c = pts.mean(axis=0)
    return float(np.mean(np.linalg.norm(pts - c, axis=1)))


def fleet_spread(positions: Sequence) -> float:
    """Spatial spread of the fleet: mean distance to centroid.

    The positional (not directional) counterpart to concentration κ.
    Clustered on fish -> small; searching -> large.
    """
    return _spread(_positions(positions))


# --------------------------------------------------------------------- #
# Fleet coherence — the vMF concentration κ                              #
# --------------------------------------------------------------------- #
def headings_to_vectors(headings_deg: Sequence) -> np.ndarray:
    """Heading angles (degrees) -> unit vectors on the circle S¹."""
    a = np.deg2rad(np.asarray(headings_deg, dtype=float))
    return np.stack([np.cos(a), np.sin(a)], axis=-1)


def vmf_kappa(unit_vectors: Sequence) -> float:
    """Maximum-likelihood concentration κ of a von Mises–Fisher (vMF)
    distribution from unit vectors.

    For unit vectors ``x_1 .. x_N`` on the sphere S^{d-1} (each row a
    d-dimensional unit vector), the MLE solves ``A_d(κ) = R`` where
    ``R = ||mean(x)||`` is the mean resultant length and ``A_d`` is the
    ratio of modified Bessel functions. Without SciPy we use the
    Banerjee–Dhillon–Ghosh–Sra (2005) approximation — the same formula
    the v3 design document uses:

        κ̂ = R (d − R²) / (1 − R²)

    κ ≈ 0 means uniform/isotropic (loose, scattered); κ ≫ 0 means
    tightly concentrated (cold, on fish). R → 1 sends κ → ∞ (capped).
    """
    X = np.asarray(unit_vectors, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[0] == 0:
        return 0.0
    d = X.shape[1]
    R = float(np.linalg.norm(X.mean(axis=0)))
    if R <= 1e-12:
        return 0.0
    if R >= 1.0 - 1e-12:
        return _LARGE_KAPPA
    kappa = R * (d - R * R) / (1.0 - R * R)
    return float(np.clip(kappa, 0.0, _LARGE_KAPPA))


def fleet_concentration(positions: Sequence,
                        headings: Optional[Sequence] = None) -> float:
    """vMF concentration κ of the fleet field.

    The fleet field is a distribution over boat *positions* and
    *headings*. κ is the directional coherence of that distribution:

    - If ``headings`` (degrees) is given, κ is the **same-tack**
      coherence — how aligned the boats' headings are (von Mises κ on
      S¹). On fish, boats drag/tack the same way -> high κ; searching
      -> low κ.

    - Otherwise κ is the **bearing** coherence — the vMF κ of the unit
      vectors pointing from the own ship (radar origin) toward each
      boat. A fleet bunched on fish subtends a narrow bearing sector ->
      high κ; a scattered fleet spans the horizon -> κ ≈ 0.

    Fewer than two boats carry no directional information -> 0.0.
    """
    if headings is not None:
        return vmf_kappa(headings_to_vectors(headings))
    pos = np.asarray(positions, dtype=float).reshape(-1, 2)
    if pos.shape[0] < 2:
        return 0.0
    ang = np.arctan2(pos[:, 1], pos[:, 0])
    return vmf_kappa(np.stack([np.cos(ang), np.sin(ang)], axis=-1))


def kappa_rate(frames: Sequence,
               times: Optional[Sequence] = None) -> float:
    """dκ/dt — the scatter/bunch signal.

    Computes κ at each sweep and returns the least-squares slope of κ
    against time. Positive = fleet bunching (κ rising — on fish);
    negative = fleet scattering (κ falling — searching). A least-squares
    fit (rather than a raw first difference) is robust to the noisy κ
    estimates of small fleets.

    ``frames`` is a chronological sequence of (N, 2) position arrays (or
    frame objects with a `.data`); ``times`` are aligned timestamps in
    seconds (default: unit spacing).
    """
    frames = list(frames)
    kappas = [fleet_concentration(_positions(f)) for f in frames]
    if len(kappas) < 2:
        return 0.0
    t = (np.asarray(times, dtype=float) if times is not None
         else np.arange(len(kappas), dtype=float))
    t = t - t[0]
    k = np.asarray(kappas, dtype=float)
    if np.ptp(t) < 1e-12:
        return 0.0
    return float(np.polyfit(t, k, 1)[0])


# --------------------------------------------------------------------- #
# Inductive biomass — the good-days anchor and its deviation             #
# --------------------------------------------------------------------- #
def _oas_shrinkage(X: np.ndarray) -> float:
    """Oracle Approximating Shrinkage (Chen et al. 2010) intensity.

    The optimal shrinkage toward ``trace(S)/d · I`` that keeps the
    covariance estimate well-conditioned even when N ≲ d — exactly the
    small-sample regime of "a week of good days."
    """
    n, d = X.shape
    if n <= 1:
        return 1.0
    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / (n - 1)
    trS = float(np.trace(S))
    trS2 = float(np.trace(S @ S))
    num = (1.0 - 2.0 / d) * trS2 + trS * trS
    den = (n + 1.0 - 2.0 / d) * (trS2 - trS * trS / d)
    if den <= 0.0:
        return 1.0
    return float(np.clip(num / den, 0.0, 1.0))


def biomass_anchor(good_day_vectors: Sequence,
                   shrinkage: Optional[float] = None) -> Dict[str, Any]:
    """Fit the *anchor* distribution over good-fishing-day feature vectors.

    ``good_day_vectors`` is an (N, d) array — one row per good day, one
    column per feature (sounder mean/variance, radar κ, fleet mean
    speed, spread rate, nav course stability, …). The anchor is a
    Gaussian ``N(μ, Σ)``; ``μ`` is the sample mean and ``Σ`` a
    shrinkage-regularized covariance (OAS by default, or an explicit
    ``shrinkage`` in [0, 1]) so the deviation stays well-defined for
    N ≲ d.

    Returns a dict with ``mean``, ``cov``, ``shrinkage``, ``n``, ``d``
    — everything ``biomass_deviation`` needs.
    """
    X = np.asarray(good_day_vectors, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n, d = X.shape
    mean = X.mean(axis=0)
    cov = np.cov(X, rowvar=False, ddof=1) if n > 1 else np.zeros((d, d))
    rho = shrinkage if shrinkage is not None else _oas_shrinkage(X)
    rho = float(np.clip(rho, 0.0, 1.0))
    scale = float(np.trace(cov)) / d if d > 0 else 0.0
    cov_reg = (1.0 - rho) * cov + rho * scale * np.eye(d)
    cov_reg = (cov_reg + cov_reg.T) / 2.0
    return {"mean": mean, "cov": cov_reg, "shrinkage": rho, "n": n, "d": d}


def biomass_deviation(vec: Sequence, anchor: Dict[str, Any]) -> float:
    """Mahalanobis distance of a feature vector from the good-days anchor.

    ``D = sqrt((x − μ)ᵀ Σ⁻¹ (x − μ))`` — the number of "good-day
    standard deviations" this stretch of water is away from the good
    kind. Small = feels like the good kind; large = a distribution
    shift (spotty). Uses a linear solve (not an explicit inverse) for
    numerical stability.
    """
    mu = np.asarray(anchor["mean"], dtype=float)
    cov = np.asarray(anchor["cov"], dtype=float)
    delta = np.asarray(vec, dtype=float) - mu
    solved = np.linalg.solve(cov, delta)
    m2 = float(delta @ solved)
    return float(np.sqrt(max(0.0, m2)))
