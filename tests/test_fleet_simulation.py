"""Tests for the multi-boat fleet simulation — `examples/fleet_simulation.py`.

The fleet sim is the proof that the inter-model temperature scales from a
room (one boat) to a room of rooms (the fleet). These tests are light and
end-to-end: the sim must run, the fleet field must be well-formed, and the
30-day arc must be visible in the numbers — fleet κ rises in the good week
(days 1-7, boats bunch on the drag) and falls in the spotty week (days
8-14, boats scatter searching).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))

import numpy as np

from fleet_simulation import FleetSimulation, fleet_kappa, meta_room_warmth


def _sim(days=30, seed=7, dark_boat_event=True):
    return FleetSimulation(n_boats=4, seed=seed,
                           dark_boat_event=dark_boat_event).run(days=days)


def test_sim_runs_end_to_end():
    result = _sim()
    history = result["history"]
    assert len(history) == 30
    # Every day carries the meta-room field and the per-boat dial numbers.
    for day in history:
        assert day["day"] >= 1
        assert 0.0 <= day["kappa"] <= 1.0, day["kappa"]
        assert -1.0 <= day["warmth"] <= 1.0, day["warmth"]
        assert isinstance(day["centroid"], (list, tuple, np.ndarray))
        assert len(day["centroid"]) == 2
        assert len(day["fishing_day_per_boat"]) == 4
        assert len(day["nudge_mean"]) == 7
        for fd in day["fishing_day_per_boat"]:
            assert -1.0 <= fd <= 1.0, fd
        assert day["spread_km"] >= 0.0
    # The good-week anchor was built from catch-good days.
    assert "anchor" in result
    assert result["anchor"]["n"] >= 5


def test_fleet_kappa_rises_good_week_falls_spotty():
    # Use the FleetSimulation accessors for the phase means.
    sim = FleetSimulation(n_boats=4, seed=7, dark_boat_event=True)
    sim.run(days=30)
    good = sim.good_week_kappa()      # mean κ over days 5-7 (bunched)
    spotty = sim.spotty_week_kappa()  # mean κ over days 12-14 (scattered)
    assert good > spotty, (good, spotty)
    # The good week is genuinely tight and the spotty week genuinely loose.
    assert good > 0.5, good
    assert spotty < 0.35, spotty


def test_meta_room_field_well_formed():
    result = _sim()
    history = result["history"]
    # Warmth goes up in the good week and down in the spotty week.
    good_warmth = np.mean([d["warmth"] for d in history if 5 <= d["day"] <= 7])
    spotty_warmth = np.mean([d["warmth"] for d in history if 12 <= d["day"] <= 14])
    assert good_warmth > spotty_warmth, (good_warmth, spotty_warmth)
    # fishing_day is binned to {-1, 0, +1} on the wire (shared numbers only).
    for d in history:
        for fd in d["fishing_day_binned"]:
            assert fd in (-1, 0, 1), fd
    # The nudge prior is well-formed (a bounded 7-modality attention vector).
    for d in history:
        nudge = np.asarray(d["nudge_mean"], dtype=float)
        assert nudge.shape == (7,)
        assert float(np.all(nudge >= -1.0)) and float(np.all(nudge <= 1.0))


def test_deviation_recovers_after_spotty_week():
    result = _sim()
    history = result["history"]
    spotty_dev = np.mean([d["deviation"] for d in history if 9 <= d["day"] <= 14])
    recovery_dev = np.mean([d["deviation"] for d in history if 24 <= d["day"] <= 30])
    # The recovery stretch returns near the good-week anchor (small deviation).
    assert recovery_dev < spotty_dev, (recovery_dev, spotty_dev)


def test_fleet_kappa_helpers():
    # Clustered boats -> high κ; scattered boats -> low κ.
    clustered = np.array([[0.0, 0.0], [0.2, 0.1], [-0.1, 0.2], [0.1, -0.2]])
    scattered = np.array([[0.0, 0.0], [5.0, 4.0], [-5.0, 3.0], [4.0, -5.0]])
    assert fleet_kappa(clustered) > fleet_kappa(scattered)
    assert 0.0 < fleet_kappa(clustered) <= 1.0
    assert 0.0 <= fleet_kappa(scattered) < fleet_kappa(clustered)
    # warmth is monotone in the shared dial numbers.
    assert meta_room_warmth(0.6, 0.8, 0.7) > meta_room_warmth(-0.6, 0.1, -0.7)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} fleet simulation tests passed.")
