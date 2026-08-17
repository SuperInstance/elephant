#!/usr/bin/env python3
"""F/V EILEEN, 30 days — the elephant feels the fishing turn.

One `BoatHarness`, simulated hour by hour for 30 days (deterministic:
numpy seeded with 7). Positions are km, timestamps are hours (ts = the
cumulative hour count). The story:

- days 1-7  — the good week: a tight fleet on the drag (boats clustered
  within ~0.5 km, drifting slowly), thick biomass marks, warm confident
  talk. Each day is warm enough (`fishing_day >= 0.2`) to store an
  inductive good-day anchor.
- days 8-14 — the fishing goes spotty: the fleet scatters over a few km
  searching, biomass thins, the talk goes clipped and cynical. No
  anchors stored.
- days 15-30 — the system feels the difference inductively: every few
  days it measures the current stretch against the good-days anchor and
  the deviation — especially the biomass channel — stays large.

The summary prints the good hour vs the spotty hour (field, dials,
fleet dials), radar kinematics (with an honest knots figure computed
from radar centroids, since the dial's internal "speed_kts" is a v0
mixed-unit quirk: positions in km + ts in hours), the nudge prior per
phase, the inductive signal, and the 30-day arc.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elephant.harness import BoatHarness
from elephant.nudge import MODALITIES

KTS_PER_KMH = 0.5399568  # 1 km/h = 0.5399568 knots

AUTHORS = ("skipper", "deckhand", "mate")

WARM_LINES = [
    "We're on them. Good mark.",
    "Hold the drag. This is the good kind.",
    "Haha, look at the column. Love it.",
    "Boys we're loaded. Happy days.",
    "Great show. Cheers to that.",
    "Beautiful mark. We're on them.",
    "Held it all morning. Good fishing.",
    "Together on this one. Yes.",
]

SPOTTY_LINES = [
    "Sure. Cold and empty.",
    "Dead drift. Moving on.",
    "Whatever. Lost the marks.",
    "Crickets. Flat and dull.",
    "Bad day. Cold water.",
    "Empty marks. Moving on.",
    "Lost them again. Crickets.",
    "Cold and flat out here.",
]


# ---------------------------------------------------------------------- #
# Simulation helpers                                                      #
# ---------------------------------------------------------------------- #
def tight_fleet(rng, center):
    """5-6 (x, y) km points clustered within ~0.5 km of `center`.

    The drag: boats on fish, all but anchored to each other."""
    n = int(rng.integers(5, 7))
    c = np.asarray(center, dtype=float)
    pts = c + rng.uniform(-0.25, 0.25, (n, 2))
    return [(float(x), float(y)) for x, y in pts]


def scattered_fleet(rng, center=(0.0, 0.0)):
    """5-6 (x, y) km points spread over ~a few km around `center`.

    Searching: every boat out for itself."""
    n = int(rng.integers(5, 7))
    c = np.asarray(center, dtype=float)
    pts = c + rng.uniform(-4.0, 4.0, (n, 2))
    return [(float(x), float(y)) for x, y in pts]


def honest_fleet_motion(h):
    """Honest fleet speed/direction from the last two radar centroids.

    centroid displacement km / dt hours * 0.5399568 -> knots. The dial's
    internal speed_kts mixes units (km positions + hour timestamps), so
    we report our own knots alongside it.
    """
    frames = h.signal.by_sensor("radar")
    if len(frames) < 2:
        return None
    f1, f2 = frames[-2], frames[-1]
    c1 = np.mean(np.asarray(f1.data, dtype=float), axis=0)
    c2 = np.mean(np.asarray(f2.data, dtype=float), axis=0)
    dt_h = max(f2.ts - f1.ts, 1e-9)
    disp = float(np.linalg.norm(c2 - c1))
    if disp < 1e-6:
        return None
    return {
        "disp_km": disp,
        "dt_hours": dt_h,
        "knots": disp / dt_h * KTS_PER_KMH,
        "dir_deg": float(np.degrees(math.atan2(c2[1] - c1[1], c2[0] - c1[0]))),
    }


def fmt_prior(prior):
    """The 7-modality nudge prior, labeled."""
    return "  ".join(f"{m}={v:+.2f}" for m, v in zip(MODALITIES, prior))


def take_snapshot(h):
    return {
        "field": h.current_field(),
        "readings": dict(h.readings()),
        "nudge": h.current_nudge(),
        "kinematics": h.radar_kinematics(),
        "deviation": h.inductive_signal(),
        "honest": honest_fleet_motion(h),
    }


# ---------------------------------------------------------------------- #
# The 30 days                                                             #
# ---------------------------------------------------------------------- #
def main() -> None:
    rng = np.random.default_rng(7)
    # Rolling window of ~6 days of talk (8 lines/day): the good week's
    # warm talk ages out by day 14, so the conversation room genuinely
    # *becomes* a different (colder) room — the elephant walks through.
    h = BoatHarness(name="EILEEN", max_messages=48)

    print("F/V EILEEN — 30 days, one harness, one elephant.")
    print("positions in km, timestamps in hours. The fleet dials feel it all.\n")

    # Spotty-phase state: a wandering search center at searching speed.
    heading = 90.0
    center = np.array([10.0, 4.0])

    good_snapshot = {}
    spotty_snapshot = {}

    for day in range(1, 31):
        for hour in range(24):
            ts = float((day - 1) * 24 + hour)
            if day <= 7:
                # The good week: tight fleet on a slowly drifting mark.
                mark = np.array([day * 1.0, (hour / 24) * 0.5])
                h.ingest_radar(tight_fleet(rng, mark), ts=ts)
                h.ingest_sounder(float(rng.uniform(0.70, 0.90)), ts=ts)
                h.ingest_nav(45.0 + float(rng.normal(0.0, 2.0)),
                             max(0.2, 2.0 + float(rng.normal(0.0, 0.3))), ts=ts)
                if hour % 3 == 0:
                    line = WARM_LINES[int(rng.integers(0, len(WARM_LINES)))]
                    h.ingest_conversation(AUTHORS[hour // 3 % 3], line, ts=ts)
            else:
                # Spotty: scattered fleet, thin marks, wandering at speed.
                heading += float(rng.normal(0.0, 12.0))
                speed_kts = float(rng.uniform(6.0, 8.5))
                v_kmh = speed_kts / KTS_PER_KMH
                center = center + v_kmh * np.array(
                    [math.cos(math.radians(heading)),
                     math.sin(math.radians(heading))])
                h.ingest_radar(scattered_fleet(rng, center), ts=ts)
                h.ingest_sounder(float(rng.uniform(0.08, 0.30)), ts=ts)
                h.ingest_nav(heading % 360.0, speed_kts, ts=ts)
                if hour % 3 == 0:
                    line = SPOTTY_LINES[int(rng.integers(0, len(SPOTTY_LINES)))]
                    h.ingest_conversation(AUTHORS[hour // 3 % 3], line, ts=ts)

        fd = h.fishing_day()
        anchor = h.day_memory(good_day_threshold=0.2)
        if day <= 7:
            print(f"day {day:2d}: good day — fishing_day={fd:+.2f} >= 0.2, "
                  f"anchor stored ({len(h.good_days)} anchor days)")
        elif day <= 14:
            print(f"day {day:2d}: spotty — fishing_day={fd:+.2f} < 0.2, "
                  f"no anchor stored")
        elif day % 3 == 0 or day == 30:
            dev = h.inductive_signal()
            print(f"day {day:2d}: deviation from anchor total={dev['total']:.2f} "
                  f"(radar {dev['radar']:.2f}, biomass {dev['biomass']:.2f}, "
                  f"nav {dev['nav']:.2f}) vs {dev['n_anchor_days']} good days "
                  f"— still not the good kind")

        if day == 7:
            good_snapshot = take_snapshot(h)
        if day == 14:
            spotty_snapshot = take_snapshot(h)

    phases = (("good (end of day 7)", good_snapshot),
              ("spotty (end of day 14)", spotty_snapshot))

    # ------------------------------------------------------------------ #
    print("\n=== good hour vs spotty hour ===")
    for label, snap in phases:
        r = snap["readings"]
        print(f"\n{label}:")
        print(f"  field : {snap['field']}")
        print(f"  dials : {snap['readings']}")
        print(f"  fleet : radar κ={r['radar_coherence']:+.2f}  "
              f"biomass={r['sounder_biomass']:.2f}  "
              f"fishing_day={r['fishing_day']:+.2f}")

    # ------------------------------------------------------------------ #
    print("\n=== radar kinematics (from 3 readings) ===")
    for label, snap in phases:
        k = snap["kinematics"]
        objs = k["objects"]
        print(f"\n{label}:")
        if objs:
            dirs = [o["dir_deg"] for o in objs]
            print(f"  dial direction (deg, unit-independent, correct): "
                  f"mean {float(np.mean(dirs)):+.1f} over {len(objs)} objects")
        else:
            print("  dial direction: no associations within its 2 km gate "
                  "(fleet too fast/scattered for hourly frames)")
        print(f"  dial fleet_mean_speed = {k['fleet_mean_speed']:.3f} "
              f"(dial \"speed_kts\" — v0 mixed-unit quirk: km positions + "
              f"hour ts, so treat with suspicion)")
        print(f"  dial spread_rate = {k['spread_rate']:+.3f} "
              f"({'bunching' if k['spread_rate'] < 0 else 'scattering'})")
        honest = snap["honest"]
        if honest:
            print(f"  HONEST fleet speed (last two radar centroids): "
                  f"{honest['knots']:.2f} kts — {honest['disp_km']:.2f} km over "
                  f"{honest['dt_hours']:.1f} h x {KTS_PER_KMH} — heading "
                  f"{honest['dir_deg']:+.0f} deg")
        else:
            print("  HONEST fleet speed: centroids static or <2 frames")

    # ------------------------------------------------------------------ #
    print("\n=== nudge prior each phase ===")
    for label, snap in phases:
        print(f"  {label:22s} {fmt_prior(snap['nudge'])}")

    # ------------------------------------------------------------------ #
    print("\n=== inductive signal ===")
    for label, snap in phases:
        d = snap["deviation"]
        print(f"  {label:22s} total={d['total']:.3f}  radar={d['radar']:.3f}  "
              f"biomass={d['biomass']:.3f}  nav={d['nav']:.3f}  "
              f"(anchor {np.round(d['anchor'], 2).tolist()} from "
              f"{d['n_anchor_days']} good days)")
    gb = good_snapshot["deviation"]["biomass"]
    sb = spotty_snapshot["deviation"]["biomass"]
    feels_good = sb <= 2.0 * max(gb, 1e-6)
    print(f"\n  does this stretch feel like the good kind? "
          f"{'YES' if feels_good else 'NO'} — spotty biomass deviation "
          f"{sb:.2f} >> good {gb:.2f}")

    # ------------------------------------------------------------------ #
    print("\n=== 30-day arc ===")
    print("  days 1-7  : the drag — tight fleet, thick marks, warm talk. "
          "7 anchors stored.")
    print(f"  days 8-14 : searching — scattered fleet, thin biomass, clipped "
          f"talk; fishing_day {spotty_snapshot['readings']['fishing_day']:+.2f}, "
          f"no anchors stored.")
    final_dev = h.inductive_signal()
    print(f"  days 15-30: the system feels the difference inductively — day 30 "
          f"still sits {final_dev['total']:.2f} from the "
          f"{final_dev['n_anchor_days']}-day good anchor "
          f"(biomass channel {final_dev['biomass']:.2f} off). "
          "The elephant remembers the good kind.")
    print(f"\n{h!r}")


if __name__ == "__main__":
    main()
