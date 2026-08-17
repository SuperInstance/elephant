"""Tests for the math of the fleet field — `elephant/fleetmath.py`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elephant.fleetmath import (
    biomass_anchor,
    biomass_deviation,
    fleet_concentration,
    headings_to_vectors,
    kappa_rate,
    three_reading_kinematics,
    vmf_kappa,
)


def _frame(ts, pts):
    class F:
        pass
    f = F()
    f.ts = ts
    f.data = np.asarray(pts, dtype=float)
    return f


# --------------------------------------------------------------------- #
# Kinematics                                                            #
# --------------------------------------------------------------------- #
def test_kinematics_recovers_constant_velocity():
    # One boat moving east at 10 m/s from t=0 to t=4.
    v = np.array([10.0, 0.0])
    frames = [
        _frame(0.0, [np.array([0.0, 0.0])]),
        _frame(2.0, [np.array([20.0, 0.0])]),
        _frame(4.0, [np.array([40.0, 0.0])]),
    ]
    out = three_reading_kinematics(frames, gate=100.0)
    obj = out["objects"][0]
    assert abs(obj["speed_mps"] - 10.0) < 1e-6
    assert abs(obj["speed_kts"] - 10.0 * 1.9438444924406046) < 1e-6
    assert abs(obj["dir_deg"] - 0.0) < 1e-6
    assert abs(obj["accel_mag"]) < 1e-6   # constant velocity -> no acceleration


def test_kinematics_recovers_acceleration():
    # Constant acceleration a=2 m/s^2 along +x, starting at rest.
    # x(t) = t^2  (v = 2t, a = 2).
    frames = [
        _frame(0.0, [np.array([0.0, 0.0])]),
        _frame(1.0, [np.array([1.0, 0.0])]),
        _frame(2.0, [np.array([4.0, 0.0])]),
    ]
    out = three_reading_kinematics(frames, gate=100.0)
    obj = out["objects"][0]
    assert abs(obj["accel"][0] - 2.0) < 1e-6
    assert abs(obj["accel_mag"] - 2.0) < 1e-6


def test_kinematics_recovers_direction_and_speed():
    # Velocity (3, 4) m/s -> speed 5 m/s, direction ~53.13 deg.
    v = np.array([3.0, 4.0])
    frames = [
        _frame(0.0, [np.array([0.0, 0.0])]),
        _frame(1.0, [v * 1.0]),
        _frame(2.0, [v * 2.0]),
    ]
    out = three_reading_kinematics(frames, gate=100.0)
    obj = out["objects"][0]
    assert abs(obj["speed_mps"] - 5.0) < 1e-6
    assert abs(obj["dir_deg"] - np.degrees(np.arctan2(4, 3))) < 1e-6


def test_kinematics_own_ship_compensation():
    # A boat stationary in the water; the own ship moves east at 10 m/s.
    # Boat-relative target drifts west at 10 m/s; compensation recovers 0.
    target_rel = [np.array([100.0, 0.0]), np.array([90.0, 0.0]),
                  np.array([80.0, 0.0])]
    own_ship = [np.array([0.0, 0.0]), np.array([10.0, 0.0]),
                np.array([20.0, 0.0])]
    frames = [_frame(0.0, [target_rel[0]]), _frame(1.0, [target_rel[1]]),
              _frame(2.0, [target_rel[2]])]
    out = three_reading_kinematics(frames, own_ship=own_ship, gate=100.0)
    obj = out["objects"][0]
    assert abs(obj["speed_mps"]) < 1e-6   # stationary in the water


def test_kinematics_association_tracks_multiple_objects():
    # Two objects crossing diagonally; association must link each to its own.
    frames = [
        _frame(0.0, [np.array([0.0, 0.0]), np.array([0.0, 10.0])]),
        _frame(1.0, [np.array([1.0, 1.0]), np.array([1.0, 11.0])]),
        _frame(2.0, [np.array([2.0, 2.0]), np.array([2.0, 12.0])]),
    ]
    out = three_reading_kinematics(frames, gate=5.0)
    speeds = sorted(o["speed_kts"] for o in out["objects"])
    # Both move at sqrt(2) m/s = 1.414... -> ~2.75 kts
    expected = np.hypot(1, 1) * 1.9438444924406046
    for s in speeds:
        assert abs(s - expected) < 1e-6


# --------------------------------------------------------------------- #
# vMF concentration κ                                                   #
# --------------------------------------------------------------------- #
def test_kappa_rises_when_boats_cluster():
    # Boats bunched into a narrow bearing sector -> high κ.
    ang = np.deg2rad([5.0, 10.0, 15.0, 20.0])
    clustered = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    # Boats spread across the horizon -> low κ.
    wide = headings_to_vectors([0.0, 90.0, 180.0, 270.0])
    k_cluster = fleet_concentration(clustered)
    k_wide = fleet_concentration(wide)
    assert k_cluster > k_wide
    assert k_cluster > 10.0        # tight cluster -> κ large
    assert k_wide < 2.0            # uniform -> κ near 0


def test_kappa_falls_when_scattering():
    # A sequence of fleets scattering over time -> κ decreases.
    seq = [
        headings_to_vectors([0.0, 5.0, 10.0]),
        headings_to_vectors([0.0, 30.0, 60.0]),
        headings_to_vectors([0.0, 120.0, 240.0]),
    ]
    kappas = [vmf_kappa(s) for s in seq]
    assert kappas[0] > kappas[1] > kappas[2]


def test_kappa_rate_sign():
    # Bunching (bearings converge onto the fish) -> positive rate;
    # scattering (bearings diverge) -> negative rate. Positions are placed
    # at unit radius at the given bearings, so bearing == angle.
    def at_bearings(angles_deg):
        a = np.deg2rad(angles_deg)
        return np.stack([np.cos(a), np.sin(a)], axis=1)

    bunch = [
        at_bearings([10.0, 40.0, 80.0]),
        at_bearings([5.0, 25.0, 55.0]),
        at_bearings([2.0, 10.0, 25.0]),
    ]
    scatter = [
        at_bearings([2.0, 10.0, 25.0]),
        at_bearings([5.0, 25.0, 55.0]),
        at_bearings([10.0, 40.0, 80.0]),
    ]
    assert kappa_rate(bunch) > 0
    assert kappa_rate(scatter) < 0


def test_vmf_kappa_degenerate():
    # Identical vectors -> huge κ; empty -> 0.
    assert vmf_kappa([[1.0, 0.0], [1.0, 0.0]]) > 1000.0
    assert vmf_kappa(np.empty((0, 2))) == 0.0


# --------------------------------------------------------------------- #
# Inductive biomass                                                     #
# --------------------------------------------------------------------- #
def test_biomass_deviation_small_for_anchor_like():
    rng = np.random.default_rng(0)
    good = rng.multivariate_normal([0.5, 0.5, 0.3], 0.05 * np.eye(3), size=40)
    anchor = biomass_anchor(good)
    # A vector close to the good-day mean -> small deviation.
    assert biomass_deviation([0.5, 0.5, 0.3], anchor) < 1.0


def test_biomass_deviation_large_for_shift():
    rng = np.random.default_rng(1)
    good = rng.multivariate_normal([0.5, 0.5, 0.3], 0.05 * np.eye(3), size=40)
    anchor = biomass_anchor(good)
    near = biomass_deviation([0.5, 0.5, 0.3], anchor)
    far = biomass_deviation([1.5, 1.5, 1.3], anchor)
    assert far > near
    assert far > 3.0            # many standard deviations out


def test_biomass_anchor_small_sample_invertible():
    # Fewer days than features must still yield an invertible (solved) cov.
    rng = np.random.default_rng(2)
    X = rng.normal(size=(3, 8))     # N=3 < d=8
    anchor = biomass_anchor(X)
    d = biomass_deviation(X[0], anchor)
    assert np.isfinite(d)
    assert np.linalg.matrix_rank(anchor["cov"]) == 8


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} fleetmath tests passed.")
