"""Trader's-board tests for the perception-check math.

The captain's macro-read: the NUMBER doesn't matter; TWO numbers show
DIRECTION, MORE THAN TWO show RATE OF CHANGE. These tests are the
trader's board — they check that the difference (not the level) is the
signal, and that the second difference sees the slowdown before the move
ends.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import numpy as np

from elephant.perception_math import (
    composite_read,
    direction,
    rate_of_change,
)
from elephant.fleetmath import three_reading_kinematics


def _frame(ts, pts):
    class F:
        pass
    f = F()
    f.ts = ts
    f.data = np.asarray(pts, dtype=float)
    return f


# --------------------------------------------------------------------- #
# Direction — two numbers                                               #
# --------------------------------------------------------------------- #
def test_direction_rising_is_positive():
    # The absolute level (0,1,2,3) is irrelevant; only the +1 steps.
    assert direction({"mood": [0.0, 1.0, 2.0, 3.0]})["mood"] == [1.0, 1.0, 1.0]


def test_direction_ignores_level():
    # Same movement, shifted by a constant -> same direction. The number
    # (the exchange rate's level) is noise; the difference is signal.
    a = direction({"x": [0.0, 1.0, 2.0]})["x"]
    b = direction({"x": [100.0, 101.0, 102.0]})["x"]
    assert a == b == [1.0, 1.0]


def test_direction_negative_and_zero():
    assert direction({"x": [2.0, 1.0, 0.0]})["x"] == [-1.0, -1.0]
    assert direction({"x": [3.0, 3.0, 3.0]})["x"] == [0.0, 0.0]


def test_short_series():
    # One reading -> no direction; two -> one direction, no rate.
    assert direction({"x": [1.0]})["x"] == []
    assert direction({"x": [1.0, 2.0]})["x"] == [1.0]
    assert rate_of_change({"x": [1.0, 2.0]})["x"] == []
    assert rate_of_change({"x": [1.0]})["x"] == []


# --------------------------------------------------------------------- #
# Rate of change — three+ numbers                                       #
# --------------------------------------------------------------------- #
def test_rate_accelerating_is_positive():
    # x = [0,1,3,6,10]: increments 1,2,3,4 -> constant +1 acceleration.
    x = [0.0, 1.0, 3.0, 6.0, 10.0]
    assert direction({"x": x})["x"] == [1.0, 2.0, 3.0, 4.0]
    assert rate_of_change({"x": x})["x"] == [1.0, 1.0, 1.0]


def test_rate_decelerates_before_direction_flattens():
    # The insight: a smoothly decelerating rise still reads "rising" in
    # the direction (first difference) while the rate (second difference)
    # has already gone negative. The macro reader feels the slowdown
    # BEFORE the move ends.
    x = [0.0, 1.0, 1.9, 2.7, 3.4, 4.0]   # increments 1,.9,.8,.7,.6
    d = direction({"x": x})["x"]
    a = rate_of_change({"x": x})["x"]
    assert all(di > 0 for di in d)        # still rising, end to end
    assert all(ai < 0 for ai in a)        # but decelerating the whole way
    assert d[-1] > 0 and a[-1] < 0        # rate turns negative first


def test_rate_rising_then_flat():
    # Hard plateau: direction goes to 0; the rate dips negative exactly
    # as the move exhausts, then returns to 0.
    x = [0.0, 1.0, 2.0, 3.0, 3.0, 3.0]
    assert direction({"x": x})["x"] == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert rate_of_change({"x": x})["x"] == [0.0, 0.0, -1.0, 0.0]


def test_rate_constant_series_is_zero():
    # The number (5.0) is irrelevant; a flat dial has no direction and
    # no rate of change.
    assert direction({"x": [5.0, 5.0, 5.0, 5.0]})["x"] == [0.0, 0.0, 0.0]
    assert rate_of_change({"x": [5.0, 5.0, 5.0, 5.0]})["x"] == [0.0, 0.0]


# --------------------------------------------------------------------- #
# Noise floor — small moves are not moves                               #
# --------------------------------------------------------------------- #
def test_noise_floor_zeroes_small_moves():
    x = [0.0, 0.001, 0.002, 0.001, 0.0]
    # Without a floor these are tiny non-zero moves...
    assert direction({"x": x})["x"] != [0.0, 0.0, 0.0, 0.0]
    # ...with a floor above the noise they all read as 0.
    assert direction({"x": x}, noise_floor=0.01)["x"] == [0.0, 0.0, 0.0, 0.0]
    assert rate_of_change({"x": x}, noise_floor=0.01)["x"] == [0.0, 0.0, 0.0]


def test_noise_floor_leaves_real_moves():
    # A real move (0.1/tick) survives a floor of 0.01.
    assert direction({"x": [0.0, 0.1, 0.2]}, noise_floor=0.01)["x"] == [0.1, 0.1]


def test_noise_floor_per_dial():
    # A per-dial floor lets one quiet dial read 0 while another moves.
    series = {"quiet": [0.0, 0.005, 0.010], "loud": [0.0, 0.5, 1.0]}
    d = direction(series, noise_floor={"quiet": 0.01, "loud": 0.01})
    assert d["quiet"] == [0.0, 0.0]
    assert d["loud"] == [0.5, 0.5]


# --------------------------------------------------------------------- #
# Non-uniform pulses — dt normalization                                 #
# --------------------------------------------------------------------- #
def test_non_uniform_dt_is_per_second():
    # Same physical rate (1 unit/second), sampled at different densities.
    # Per-tick differences would differ; per-second rates must not.
    fast = direction({"x": [0.0, 1.0, 2.0]}, dt=1.0)["x"]       # 1s ticks
    slow = direction({"x": [0.0, 2.0, 4.0]}, dt=2.0)["x"]       # 2s ticks
    assert fast == [1.0, 1.0]
    assert slow == [1.0, 1.0]
    assert slow == fast


def test_non_uniform_dt_rate_matches_quadratic():
    # x(t) = 0.5*3 t^2 + 2 t + 1, sampled at t = 0, 2, 5 (gaps 2, 3).
    # The exact acceleration is a = 3, recovered despite the uneven ticks.
    a = 3.0
    tt = [0.0, 2.0, 5.0]
    xx = [0.5 * a * t * t + 2.0 * t + 1.0 for t in tt]   # 1, 11, 48.5
    out = rate_of_change({"x": xx}, dt=[2.0, 3.0])["x"]
    assert abs(out[0] - a) < 1e-9


# --------------------------------------------------------------------- #
# The composite read — the whole hand                                  #
# --------------------------------------------------------------------- #
def test_composite_identifies_fastest_and_accelerating():
    # mood is moving fast; volume is drifting; panic is accelerating.
    series = {
        "mood": [0.0, 1.0, 2.0, 3.0],     # direction +1, rate 0
        "volume": [0.5, 0.6, 0.7, 0.8],   # direction +0.1, rate 0
        "panic": [0.0, 0.1, 0.4, 0.9],    # direction +.5, rate +.2,+... accelerating
    }
    out = composite_read(series)
    assert out["fastest_dial"] == "mood"       # |+1| > |+0.5| > |+0.1|
    # panic's rate of change is positive and non-zero -> accelerating.
    assert "panic" in out["accelerating_dials"]
    assert "mood" not in out["accelerating_dials"]  # constant rate 0
    assert out["macro_direction"] > 0


def test_composite_weighted_macro():
    series = {
        "mood": [0.0, 1.0, 2.0],     # direction +1
        "volume": [0.0, 0.5, 1.0],   # direction +0.5
    }
    # Equal weights: mean of (1, 0.5) = 0.75.
    eq = composite_read(series)
    assert abs(eq["macro_direction"] - 0.75) < 1e-12
    # All weight on mood -> macro_direction = 1.0.
    mood_only = composite_read(series, weights={"mood": 1.0, "volume": 0.0})
    assert abs(mood_only["macro_direction"] - 1.0) < 1e-12


def test_composite_handles_nan():
    # A dead dial (NaN) is dropped, not averaged in as 0.
    series = {
        "mood": [0.0, 1.0, 2.0],
        "dead": [0.0, float("nan"), float("nan")],
    }
    out = composite_read(series)
    assert abs(out["macro_direction"] - 1.0) < 1e-12
    assert out["fastest_dial"] == "mood"


# --------------------------------------------------------------------- #
# One math, two domains — the fleetmath cross-check                     #
# --------------------------------------------------------------------- #
def test_rate_of_change_matches_fleetmath_acceleration():
    # A 1-D quadratic trajectory read as radar positions (fleetmath) and
    # as a dial series (perception_math) must give the same acceleration.
    a = 2.0
    tt = [0.0, 1.0, 3.0]
    xx = [0.5 * a * t * t for t in tt]                     # 0, 1, 9
    frames = [
        _frame(tt[0], [np.array([xx[0], 0.0])]),
        _frame(tt[1], [np.array([xx[1], 0.0])]),
        _frame(tt[2], [np.array([xx[2], 0.0])]),
    ]
    accel_x = three_reading_kinematics(frames, gate=100.0)["objects"][0]["accel"][0]
    rate_x = rate_of_change({"x": xx}, dt=[1.0, 2.0])["x"][0]
    assert abs(accel_x - a) < 1e-9
    assert abs(rate_x - accel_x) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} perception-math tests passed.")
