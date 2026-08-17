#!/usr/bin/env python3
"""Four boats, thirty days, one elephant — the fleet as a room of rooms.

This is the fleet-scale proof of the inter-model temperature: one
`BoatHarness` per boat (each boat is a room), and the fleet itself as
the meta-room whose temperature — fleet κ, meta-room warmth, the shared
nudge prior — rises and falls with the fishing. Everything is
deterministic: every random draw comes from one
`np.random.default_rng(seed)` (seed 7 by default), consumed in a fixed
order (catch -> per-boat motion -> per-frame radar noise).

The 30-day arc:

- days 1-7   GOOD — the boats bunch on a slowly-tightening drag around
  the ground point [12, 8] km. Fleet κ RISES (tight), the meta-room
  warms, the sounder is thick, catch is high. These seven days form
  the inductive anchor.
- days 8-14  SPOTTY — each boat drifts outward from its day-7 position,
  searching at speed. Fleet κ FALLS, the meta-room cools, biomass
  thins, catch dies.
- days 15-30 RECOVERY — the boats re-group on the drag (blending from
  their day-14 positions back to a tight home formation). The day's
  feature vector returns near the good-week anchor and the Mahalanobis
  deviation drops: "this stretch feels like the good kind."

The anchor is NOT self-confirming (the Seed-2.0-pro critique): the day
feature vector is `[kappa, mean_biomass, catch]` where `catch` is the
EXOGENOUS fish actually landed — drawn directly from the phase
schedule, never derived from any dial. `fishing_day` (a dial computed
from radar + sounder) never enters the anchor, so the induction has
something independent to agree with.

The DARK-BOAT CHARISMA RULE (design §4.2, FLAGGED, on by default): on
day 20 the highest-reputation boat goes dark — stops broadcasting. The
fleet field then uses the 3 active boats' positions plus ONE virtual
point at the dark boat's last position, weighted 3 x its reputation:
charisma still pulling the room. The remaining boats' recovery target
becomes that last position. With `dark_boat_event=False` nobody ever
goes dark and all four boats broadcast all 30 days (the rule never
fires, never crashes).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elephant.field import RoomField
from elephant.fleetmath import biomass_anchor, biomass_deviation
from elephant.harness import BoatHarness
from elephant.nudge import MODALITIES, describe

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

REPUTATIONS = (1.7, 1.0, 0.9, 0.6)          # boat 0 is the best skipper
BOAT_NAMES = ("EILEEN", "PETREL", "FULMAR", "SHEARWATER")
GROUND = np.array([12.0, 8.0])              # km — the drag center
DARK_DAY = 20                               # the day the charisma rule fires


# --------------------------------------------------------------------- #
# Weighted fleet geometry — the room-of-rooms' shape                      #
# --------------------------------------------------------------------- #
def weighted_median(values, weights=None) -> float:
    """The weighted median of a 1-D sequence.

    Sort by value, walk the cumulative weights, and stop at the first
    value where the running weight reaches half the total — the point
    that splits the weight mass in half. `weights=None` means uniform
    (the plain median, lower-middle convention on ties).
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    if v.size == 0:
        return 0.0
    w = (np.ones_like(v) if weights is None
         else np.asarray(weights, dtype=float).reshape(-1))
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    total = float(np.sum(w))
    if total <= 0.0:
        return float(np.median(v))
    cw = np.cumsum(w)
    idx = int(np.searchsorted(cw, total / 2.0))
    return float(v[min(idx, v.size - 1)])


def fleet_centroid(positions, weights=None) -> np.ndarray:
    """The fleet centroid: the weighted median, x and y separately.

    A median centroid (not a mean) is the fisherman's centroid — one
    crazy boat far out searching does not drag the middle of the room
    with it.
    """
    P = np.asarray(positions, dtype=float).reshape(-1, 2)
    w = None if weights is None else np.asarray(weights, dtype=float)
    return np.array([weighted_median(P[:, 0], w),
                     weighted_median(P[:, 1], w)])


def fleet_spread_km(positions, weights=None) -> float:
    """Weighted MEAN radial distance (km) from the fleet centroid."""
    P = np.asarray(positions, dtype=float).reshape(-1, 2)
    if P.shape[0] == 0:
        return 0.0
    w = (np.ones(P.shape[0]) if weights is None
         else np.asarray(weights, dtype=float))
    c = fleet_centroid(P, w)
    radial = np.linalg.norm(P - c, axis=1)
    total = float(np.sum(w))
    if total <= 0.0:
        return float(np.mean(radial))
    return float(np.sum(w * radial) / total)


def fleet_kappa(positions, weights=None) -> float:
    """Fleet tightness κ = 1 / (1 + spread_km), in [0, 1].

    Bunched on the drag -> spread ~ 0.1 km -> κ ~ 0.9; scattered over
    the horizon -> spread ~ 20 km -> κ ~ 0.05.
    """
    return 1.0 / (1.0 + fleet_spread_km(positions, weights))


def meta_room_warmth(mean_fishing_day: float, mean_biomass: float,
                     mean_radar_coherence: float) -> float:
    """The meta-room temperature — the fleet as one room.

    Half the day's luck, some of the biomass (re-centered to [-1, 1]),
    a little of the fleet's coherence: warm when the fleet is together
    on fish, a cold plunge when everyone is out searching alone.
    """
    warmth = (0.5 * mean_fishing_day
              + 0.3 * (2.0 * mean_biomass - 1.0)
              + 0.2 * mean_radar_coherence)
    return float(np.clip(warmth, -1.0, 1.0))


def _damping_bell(kappa: float) -> float:
    """Design §4.2: effective_kappa = kappa * (1 - clip(kappa, 0.2, 0.8)).

    The mid-tight fleet is the sweet spot; a perfectly bunched fleet
    (κ -> 1) or a fully scattered one (κ -> 0) carries less usable
    signal than the bell's middle.
    """
    return float(kappa * (1.0 - float(np.clip(kappa, 0.2, 0.8))))


def _bin_fishing_day(fd: float) -> int:
    """What goes on the wire: +1 good / -1 poor / 0 in between."""
    if fd > 0.25:
        return 1
    if fd < -0.25:
        return -1
    return 0


# --------------------------------------------------------------------- #
# One boat                                                                #
# --------------------------------------------------------------------- #
class Boat:
    """One boat. position (2,) km, heading deg, speed kts, reputation, harness.

    The harness is the boat's room: radar frames (the other active
    boats, relative), the sounder, nav, and one conversation line a
    day. The sim moves the boat; the harness feels it.
    """

    def __init__(self, name: str, position, heading: float, speed: float,
                 reputation: float):
        self.name = str(name)
        self.position = np.asarray(position, dtype=float).reshape(2)
        self.heading = float(heading)
        self.speed = float(speed)
        self.reputation = float(reputation)
        self.harness = BoatHarness(name=self.name, max_messages=120)
        # Per-day scratch state, filled in by the simulation loop.
        self.biomass = 0.0          # today's sounder truth
        self.line = ""              # today's conversation line
        self.author = AUTHORS[0]
        self.day14_position = None  # set once, at the end of day 14

    def __repr__(self) -> str:
        return (f"<Boat {self.name!r} rep={self.reputation:.1f} "
                f"pos=({self.position[0]:.1f},{self.position[1]:.1f})km>")


# --------------------------------------------------------------------- #
# The fleet simulation                                                    #
# --------------------------------------------------------------------- #
class FleetSimulation:
    """The room of rooms: `n_boats` boats over a 30-day fishing arc.

    Deterministic: one `np.random.default_rng(seed)` drives the catch
    schedule, the per-boat motion, the radar noise, and the day's
    conversation, in a fixed order.
    """

    def __init__(self, n_boats: int = 4, seed: int = 7,
                 dark_boat_event: bool = True):
        self.n_boats = int(n_boats)
        self.seed = int(seed)
        self.dark_boat_event = bool(dark_boat_event)
        self.ground = GROUND
        self.boats: list[Boat] = []
        self.history: list[dict] = []
        self.anchor: dict = {}
        self.dark_boat_name: str | None = None
        self.dark_pos: np.ndarray | None = None
        self._reset()

    # ---------------------------------------------------------------- #
    def _reset(self) -> None:
        """Fresh rng, boats, and arc state (so `run()` is idempotent)."""
        self.rng = np.random.default_rng(self.seed)
        self.history = []
        self.anchor = {}
        self.dark_boat_name = None
        self.dark_pos = None
        self.boats = []
        for i in range(self.n_boats):
            # Home geometry: one compass point per boat, just off the
            # ground point. Boat i sits at angle 2*pi*i/n, radius
            # 0.3 + 0.2*(i%2) km — close, but not on top of each other.
            home = self._home_dir(i)
            radius = 0.3 + 0.2 * (i % 2)
            start = self.ground + home * radius * 0.3
            name = BOAT_NAMES[i % len(BOAT_NAMES)]
            rep = REPUTATIONS[i % len(REPUTATIONS)]
            self.boats.append(Boat(name, start, 80.0, 1.5, rep))

    def _home_dir(self, i: int) -> np.ndarray:
        """Unit vector at angle 2*pi*i/n_boats (boat i's home bearing)."""
        a = 2.0 * np.pi * i / self.n_boats
        return np.array([np.cos(a), np.sin(a)])

    def _active_boats(self) -> list[Boat]:
        """Boats still broadcasting (all of them until the dark day)."""
        if self.dark_boat_name is None:
            return list(self.boats)
        return [b for b in self.boats if b.name != self.dark_boat_name]

    # ---------------------------------------------------------------- #
    # The anchor — exogenous catch, never a dial                        #
    # ---------------------------------------------------------------- #
    def _draw_catch(self, day: int) -> float:
        """Fleet-level scalar: the fish actually LANDED that day.

        Drawn straight from the phase schedule — high in the good
        week, dead in the spotty week, decent in recovery — and never
        derived from any dial. This is what keeps the inductive anchor
        honest (no tautology with `fishing_day`).
        """
        if day <= 7:
            return float(self.rng.uniform(0.70, 0.95))
        if day <= 14:
            return float(self.rng.uniform(0.05, 0.25))
        return float(self.rng.uniform(0.60, 0.80))

    # ---------------------------------------------------------------- #
    # One day                                                           #
    # ---------------------------------------------------------------- #
    def _simulate_day(self, day: int) -> tuple[dict, np.ndarray]:
        rng = self.rng

        # --- exogenous catch first (fixed draw order all day) ---------
        catch = self._draw_catch(day)

        # --- the dark-boat charisma rule (FLAGGED, design §4.2) -------
        # On day 20 the best-reputation boat stops broadcasting. Its
        # LAST position becomes a heavy virtual point in the fleet
        # field and the others' recovery target. If the event is off
        # this never fires.
        if (self.dark_boat_event and day == DARK_DAY
                and self.dark_boat_name is None):
            dark = max(self.boats, key=lambda b: b.reputation)
            self.dark_boat_name = dark.name
            self.dark_pos = dark.position.copy()

        dark_active = self.dark_boat_name is not None
        lines = WARM_LINES if (day <= 7 or day >= 15) else SPOTTY_LINES

        # --- per-boat motion + sensors (phase schedule below) ---------
        for i, boat in enumerate(self.boats):
            home = self._home_dir(i)
            radius = 0.3 + 0.2 * (i % 2)
            is_dark = dark_active and boat.name == self.dark_boat_name

            if day <= 7:
                # PHASE I — GOOD: the drag tightens from ~3 km down to
                # 0.3 km over the week; slow trawl, thick marks, warm
                # talk. tight(day 1)=3.0 ... tight(day 7)=0.3.
                tight = 0.3 + 2.7 * (7 - day) / 6.0
                pos = (self.ground + home * radius * tight
                       + rng.normal(0.0, 0.12, 2))
                heading = 80.0 + rng.normal(0.0, 5.0)
                speed = rng.uniform(1.0, 2.5)
                biomass = rng.uniform(0.70, 0.90)
            elif day <= 14:
                # PHASE II — SPOTTY: each boat drifts outward from its
                # current position along its home bearing (cumulative:
                # by day 14 they are 14-25 km out), searching at speed
                # over thin marks, clipped cynical talk.
                pos = (boat.position + home * (2.0 + 1.5 * (i % 2))
                       + rng.normal(0.0, 0.5, 2))
                heading = (float(np.degrees(np.arctan2(home[1], home[0])))
                           + rng.normal(0.0, 15.0))
                speed = rng.uniform(6.0, 8.5)
                biomass = rng.uniform(0.08, 0.30)
            else:
                # PHASE III — RECOVERY: blend from the day-14 position
                # back to a tight home formation (rc hits 1.0 on day
                # 18). With the dark boat active the target is ITS last
                # position — the fleet re-forms on the charismatic
                # boat's mark. The dark boat itself holds station
                # (position frozen; sensors keep flowing).
                rc = float(np.clip((day - 14) / 4.0, 0.0, 1.0))
                base = (boat.day14_position if boat.day14_position is not None
                        else boat.position)
                target = (self.dark_pos if dark_active
                          else self.ground + home * radius * 0.4)
                pos = (1.0 - rc) * base + rc * target + rng.normal(0.0, 0.15, 2)
                heading = 80.0 + rng.normal(0.0, 5.0)
                speed = rng.uniform(1.0, 2.5)
                biomass = rng.uniform(0.60, 0.75)

            line = lines[int(rng.integers(0, len(lines)))]
            author = AUTHORS[int(rng.integers(0, len(AUTHORS)))]

            if not is_dark:
                boat.position = pos
            boat.heading = float(heading)
            boat.speed = float(speed)
            boat.biomass = float(biomass)
            boat.line, boat.author = line, author
            if day == 14:
                boat.day14_position = boat.position.copy()

        # --- the fleet field (with the charisma virtual point) --------
        active = self._active_boats()
        if dark_active and self.dark_pos is not None:
            dark = next(b for b in self.boats
                        if b.name == self.dark_boat_name)
            field_positions = [b.position for b in active] + [self.dark_pos]
            field_weights = ([1.0] * len(active)
                             + [3.0 * dark.reputation])
        else:
            field_positions = [b.position for b in self.boats]
            field_weights = None
        centroid = fleet_centroid(field_positions, field_weights)
        spread_km = fleet_spread_km(field_positions, field_weights)
        kappa = fleet_kappa(field_positions, field_weights)

        # --- feed every harness: 3 sensor reads at 0h / 8h / 16h ------
        active_set = set(id(b) for b in active)
        for t_off in (0.0, 8.0, 16.0):
            ts = day * 24.0 + t_off
            for boat in self.boats:
                # Radar sees the other ACTIVE broadcasters, relative to
                # us, with 0.1 km of sensor noise.
                targets = [
                    (other.position - boat.position) + rng.normal(0.0, 0.1, 2)
                    for other in self.boats
                    if id(other) != id(boat) and id(other) in active_set
                ]
                boat.harness.ingest_radar(targets, ts=ts)
                boat.harness.ingest_sounder(
                    float(np.clip(boat.biomass, 0.0, 1.0)), ts=ts)
                boat.harness.ingest_nav(boat.heading, boat.speed, ts=ts)
        # One conversation line per day (with the evening watch).
        for boat in self.boats:
            boat.harness.ingest_conversation(boat.author, boat.line,
                                             ts=day * 24.0 + 16.0)

        # --- read the dials back ---------------------------------------
        fd_per_boat = [float(b.harness.fishing_day()) for b in self.boats]
        radar_per_boat = [float(b.harness.fleet_kappa()) for b in self.boats]
        biomass_per_boat = [float(b.harness.biomass()) for b in self.boats]
        nudge_mean = np.mean(
            np.stack([b.harness.current_nudge() for b in self.boats]),
            axis=0)
        mean_fd = float(np.mean(fd_per_boat))
        mean_radar = float(np.mean(radar_per_boat))
        mean_biomass = float(np.mean(biomass_per_boat))
        warmth = meta_room_warmth(mean_fd, mean_biomass, mean_radar)

        day_dict = {
            "day": day,
            "kappa": float(kappa),
            "effective_kappa": _damping_bell(float(kappa)),
            "warmth": warmth,
            "spread_km": float(spread_km),
            "centroid": [float(centroid[0]), float(centroid[1])],
            "catch": catch,
            "mean_fishing_day": mean_fd,
            "mean_biomass": mean_biomass,
            "mean_radar_coherence": mean_radar,
            "deviation": 0.0,          # filled once the anchor exists
            "fishing_day_per_boat": fd_per_boat,
            "fishing_day_binned": [_bin_fishing_day(fd) for fd in fd_per_boat],
            "nudge_mean": [float(v) for v in nudge_mean],
            "dark_boat": self.dark_boat_name,
            "positions": [[float(p[0]), float(p[1])] for p in
                          [b.position for b in self.boats]],
        }
        # The day feature for the inductive anchor: kappa + dial biomass
        # + EXOGENOUS catch (never fishing_day — no tautology).
        features = np.array([kappa, mean_biomass, catch], dtype=float)
        return day_dict, features

    # ---------------------------------------------------------------- #
    def run(self, days: int = 30) -> dict:
        """Run the arc. Returns {"history": [day-dicts], "anchor": dict}.

        The anchor is fit ONCE from the good week (days 1-7) over the
        [kappa, mean_biomass, catch] features, then every day's
        Mahalanobis deviation from it is written into its history dict.
        """
        self._reset()
        features: list[np.ndarray] = []
        for day in range(1, int(days) + 1):
            day_dict, feats = self._simulate_day(day)
            self.history.append(day_dict)
            features.append(feats)

        good = np.stack(features[:min(7, len(features))], axis=0)
        if good.shape[0] >= 2:
            self.anchor = biomass_anchor(good)
            for day_dict, feats in zip(self.history, features):
                day_dict["deviation"] = biomass_deviation(feats, self.anchor)
        else:
            # Too few days to fit a covariance: report a degenerate
            # anchor and zero deviation rather than a singular solve.
            self.anchor = {"mean": good.mean(axis=0),
                           "cov": np.eye(good.shape[1]),
                           "shrinkage": 1.0, "n": int(good.shape[0]),
                           "d": int(good.shape[1])}
            for day_dict in self.history:
                day_dict["deviation"] = 0.0
        return {"history": self.history, "anchor": self.anchor}

    # ---------------------------------------------------------------- #
    def _phase(self, lo: int, hi: int) -> list[dict]:
        return [d for d in self.history if lo <= d["day"] <= hi]

    def _ensure_run(self) -> None:
        if not self.history:
            self.run()

    def good_week_kappa(self) -> float:
        """Mean fleet κ over days 5-7 (1-indexed) — the tight drag."""
        self._ensure_run()
        return float(np.mean([d["kappa"] for d in self._phase(5, 7)]))

    def spotty_week_kappa(self) -> float:
        """Mean fleet κ over days 12-14 (1-indexed) — the scatter."""
        self._ensure_run()
        return float(np.mean([d["kappa"] for d in self._phase(12, 14)]))


# --------------------------------------------------------------------- #
# The report                                                              #
# --------------------------------------------------------------------- #
def _phase_report(sim: FleetSimulation, label: str, lo: int, hi: int) -> dict:
    """Per-phase means: κ raw + effective, warmth, per-boat fd, nudge, dev."""
    days = sim._phase(lo, hi)
    return {
        "label": label,
        "kappa": float(np.mean([d["kappa"] for d in days])),
        "effective_kappa": float(np.mean([d["effective_kappa"] for d in days])),
        "warmth": float(np.mean([d["warmth"] for d in days])),
        "catch": float(np.mean([d["catch"] for d in days])),
        "fd_per_boat": [float(np.mean([d["fishing_day_per_boat"][i]
                                       for d in days]))
                        for i in range(sim.n_boats)],
        "nudge": np.mean(np.stack([d["nudge_mean"] for d in days]), axis=0),
        "deviation": float(np.mean([d["deviation"] for d in days])),
        "dark": sorted({str(d["dark_boat"]) for d in days
                        if d["dark_boat"] is not None}),
    }


def main() -> None:
    sim = FleetSimulation(n_boats=4, seed=7, dark_boat_event=True)
    result = sim.run(days=30)
    history = result["history"]
    anchor = result["anchor"]

    print("FLEET SIMULATION — four boats, thirty days, one elephant "
          "(seed 7).")
    print("positions in km, timestamps in hours; fleet κ = 1/(1+spread).")
    print("the anchor feature is [kappa, mean_biomass, catch] with "
          "EXOGENOUS catch.\n")

    # --- the 30-day arc, one line per day -----------------------------
    print("day | κ (eff)     | warmth | catch | deviation | dark")
    for d in history:
        dark = d["dark_boat"] or "-"
        print(f"{d['day']:3d} | {d['kappa']:.2f} ({d['effective_kappa']:.2f})"
              f" | {d['warmth']:+.2f}  | {d['catch']:.2f}  | "
              f"{d['deviation']:9.2f} | {dark}")

    # --- per-phase summary --------------------------------------------
    phases = [
        _phase_report(sim, "PHASE I   good week  (days 1-7)", 1, 7),
        _phase_report(sim, "PHASE II  spotty     (days 8-14)", 8, 14),
        _phase_report(sim, "PHASE III recovery   (days 15-30)", 15, 30),
    ]
    spotty_dev = phases[1]["deviation"]
    recovery_dev = phases[2]["deviation"]
    for ph in phases:
        print(f"\n=== {ph['label']} ===")
        print(f"  fleet κ (raw/effective) : {ph['kappa']:.2f} / "
              f"{ph['effective_kappa']:.2f}")
        print(f"  meta-room warmth        : {ph['warmth']:+.2f}")
        print(f"  exogenous catch         : {ph['catch']:.2f}")
        print("  per-boat fishing_day    : "
              + "  ".join(f"{b.name}={fd:+.2f}" for b, fd in
                          zip(sim.boats, ph["fd_per_boat"])))
        print(f"  nudge prior (mean)      : {describe(ph['nudge'])}")
        feels_good = ph["deviation"] < 0.5 * spotty_dev
        print(f"  inductive deviation     : {ph['deviation']:.2f} good-day "
              f"sigmas — this stretch "
              f"{'FEELS like the good kind' if feels_good else 'does NOT feel like the good kind'}")

    # --- the dark-boat event ------------------------------------------
    if sim.dark_boat_name is not None:
        dark = next(b for b in sim.boats if b.name == sim.dark_boat_name)
        print(f"\n=== the dark-boat event (day {DARK_DAY}) ===")
        print(f"  {dark.name} (reputation {dark.reputation:.1f}, the best "
              f"skipper) stopped broadcasting at "
              f"({sim.dark_pos[0]:.1f}, {sim.dark_pos[1]:.1f}) km.")
        print(f"  fleet field since: 3 active positions + ONE virtual "
              f"point there, weight 3 x {dark.reputation:.1f} = "
              f"{3.0 * dark.reputation:.1f} — charisma still pulling the room.")
        print("  the remaining boats' recovery target became that last "
              "position: they re-grouped on HER mark.")
    else:
        print("\n=== no dark boat (event off) — all four broadcast all "
              "30 days ===")

    # --- the arc, in one breath ----------------------------------------
    good_k = sim.good_week_kappa()
    spotty_k = sim.spotty_week_kappa()
    print("\n=== the 30-day arc ===")
    print(f"  days 1-7   κ {good_k:.2f} (tight on the drag) -> days 12-14 "
          f"κ {spotty_k:.2f} (scattered searching) -> days 24-30 "
          f"κ {float(np.mean([d['kappa'] for d in sim._phase(24, 30)])):.2f} "
          "(re-grouped).")
    print(f"  deviation: spotty {spotty_dev:.1f} vs recovery "
          f"{recovery_dev:.1f} — the anchor recognizes the good kind "
          "when it comes back.")
    print(f"  anchor mean {np.round(anchor['mean'], 3).tolist()} from "
          f"{anchor['n']} good days (catch channel exogenous).")

    # --- each boat's final room ----------------------------------------
    print("\n=== the rooms at the end (day 30) ===")
    for b in sim.boats:
        field = RoomField(b.harness.readings())
        print(f"  {b.name:11s} {field!r}")
    print(f"\n{len(sim.boats)} rooms, one elephant. "
          f"{describe(np.mean(np.stack([d['nudge_mean'] for d in history[-1:]]), axis=0))}")


if __name__ == "__main__":
    main()
