"""elephant — tests: the BoatHarness (sensors + conversation + anchors)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elephant.field import RoomField
from elephant.harness import BoatHarness
from elephant.nudge import MODALITIES


def tight_fleet(rng, center, n=5, radius=0.25):
    """5-6 (x, y) km points clustered within ~0.5 km of `center`."""
    n = int(rng.integers(n, n + 2))
    c = np.asarray(center, dtype=float)
    return [(float(x), float(y)) for x, y in c + rng.uniform(-radius, radius, (n, 2))]


def scattered_fleet(rng, center=(0.0, 0.0), n=5, radius=4.0):
    """5-6 (x, y) km points spread over ~a few km (searching)."""
    n = int(rng.integers(n, n + 2))
    c = np.asarray(center, dtype=float)
    return [(float(x), float(y)) for x, y in c + rng.uniform(-radius, radius, (n, 2))]


def feed_good_hours(h, rng, day, hours=6):
    for hour in range(hours):
        ts = float(day * hours + hour)
        h.ingest_radar(tight_fleet(rng, [1.0, 2.0]), ts=ts)
        h.ingest_sounder(0.80, ts=ts)
        h.ingest_nav(45.0, 2.0, ts=ts)


def test_harness_ingest_produces_field():
    h = BoatHarness()
    rng = np.random.default_rng(0)
    for i in range(4):
        ts = float(i)
        h.ingest_radar(tight_fleet(rng, [1.0, 2.0]), ts=ts)
        h.ingest_sounder(0.85, ts=ts)
        h.ingest_nav(45.0, 2.0, ts=ts)
    h.ingest_conversation("skipper", "We're on them. Good mark. Hold the drag.", ts=1.0)

    field = h.current_field()
    assert isinstance(field, RoomField)
    assert isinstance(field.warmth(), float)
    assert field.concentration() >= 0.0

    readings = h.readings()
    expected = {
        "radar_coherence", "sounder_biomass", "fishing_day",
        "mood", "volume", "earnestness", "cynicism",
        "joke_landing", "panic", "presence",
    }
    assert expected <= set(readings), set(readings)
    for key in expected:
        v = readings[key]
        assert isinstance(v, float), (key, type(v))
        assert -1.0 <= v <= 1.0, (key, v)


def test_nudge_prior_wellformed():
    h = BoatHarness()
    rng = np.random.default_rng(1)
    feed_good_hours(h, rng, day=0, hours=6)

    prior = h.current_nudge()
    assert isinstance(prior, np.ndarray)
    assert prior.ndim == 1
    assert len(prior) == len(MODALITIES) == 7
    assert float(np.all(prior >= -1.0)) and float(np.all(prior <= 1.0))
    assert prior[MODALITIES.index("radar")] > 0.0
    assert prior[MODALITIES.index("sounder")] > 0.0


def test_good_day_anchor_deviation():
    h = BoatHarness()
    rng = np.random.default_rng(2)

    # Several good days -> anchors stored.
    for day in range(3):
        feed_good_hours(h, rng, day=day, hours=6)
        anchor = h.day_memory(good_day_threshold=0.2)
        assert anchor is not None
        assert isinstance(anchor, np.ndarray)
        assert anchor.shape == (3,)
    assert len(h.good_days) == 3

    good_dev = h.inductive_signal()["biomass"]
    assert 0.0 <= good_dev < 0.2, good_dev

    # Spotty stretch: scattered fleet, thin biomass.
    for hour in range(6):
        ts = float(18 + hour)
        h.ingest_radar(scattered_fleet(rng, [5.0, 5.0]), ts=ts)
        h.ingest_sounder(0.15, ts=ts)
        h.ingest_nav(90.0, 8.0, ts=ts)

    spotty_dev = h.inductive_signal()["biomass"]
    assert spotty_dev > good_dev, (spotty_dev, good_dev)
    assert spotty_dev > 0.3, spotty_dev

    # The day fell below threshold: no new anchor.
    assert h.day_memory(good_day_threshold=0.2) is None
    assert len(h.good_days) == 3


if __name__ == "__main__":
    for fn in [test_harness_ingest_produces_field, test_nudge_prior_wellformed,
               test_good_day_anchor_deviation]:
        fn()
        print(f"PASS {fn.__name__}")
    print("\nAll harness tests passed.")
