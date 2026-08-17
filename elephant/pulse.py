"""Pulse — the internal monologue, on a constant heartbeat.

The captain's directive: agents run internal monologues on CONSTANT
PULSES even when they aren't talking. These internal monologues take a
perception check as part of their looking around and thinking — the
agent reads the table's conversation as a WHOLE HAND and sees JEPA
perceptions.

The design core (the captain, direct):

    "Agents using internal monologues on constant pulses even if they
    aren't talking. These internal monologues take a Perception check
    as part of their looking around and thinking. They look at the
    table's conversation as a whole hand and see JEPA perceptions —
    like macro-economic currency exchange changes, where the number
    doesn't matter but TWO numbers show DIRECTION and MORE THAN TWO
    show RATE OF CHANGE."

So an agent is always sensing, even when silent. The pulse is the
heartbeat of that sensing. Each pulse: read the room's field (or a
dial series), and from the SEQUENCE of readings extract the macro
sense — direction from the last two, rate of change from the last
three+ — the way a trader reads a currency pair. The number alone is
nothing; the movement is the perception.

This module gives an agent a `PulseLoop`:

- it ticks on an interval (``.tick(now)`` / ``.pulse()``), even when
  the agent says nothing in the room;
- it keeps a rolling history of the room's field vectors (bounded);
- every tick it runs a ``perception_check()`` — the macro read of the
  room over the pulse history: direction (last TWO readings),
  rate of change (last THREE+, the second difference), per-dial
  deltas, and a ``whole_hand`` summary of the table's conversation AS
  A WHOLE (the macro read, not any single message);
- ``internal_monologue()`` is the agent's silent thinking — the part
  that runs even when the agent never speaks.

The perception-check math (``direction``, ``rate_of_change``) is the
fleetmath `three_reading_kinematics` idea generalized to any dial
series, and it is deliberately standalone — numpy-only, import-free
of the package's heavier modules — so it reads and tests on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Sequence

import numpy as np

from elephant.dial import DialBank
from elephant.dials import DEFAULT_DIALS
from elephant.field import RoomField
from elephant.room import Room

DEFAULT_NOISE_FLOOR = 0.02   # per-pulse moves below this read as 0 — the
                             # number doesn't matter, only the movement.

__all__ = [
    "PulseLoop", "PerceptionReport",
    "direction", "rate_of_change",
    "compose_whole_hand", "compose_monologue",
    "DEFAULT_NOISE_FLOOR",
]


# ---------------------------------------------------------------------- #
# The perception-check math — two numbers show direction,                #
# three+ show rate of change.                                             #
# ---------------------------------------------------------------------- #
def _as_matrix(series: Sequence) -> tuple:
    """Coerce a pulse series into an (n, d) float matrix.

    Accepts a list of dicts (dial readings; the first dict's keys
    become the returned names) or any array-like (a flat list is one
    dial, reshaped to a column). Non-finite readings (NaN/inf) are
    carried forward from the last valid reading — a glitch is NOT a
    movement, and the number doesn't matter — so a single bad sample
    never fabricates direction or rate.
    """
    if len(series) == 0:
        return np.zeros((0, 0)), None
    first = series[0]
    if isinstance(first, dict):
        names = list(first.keys())
        mat = np.array([[s.get(n, 0.0) for n in names] for s in series],
                       dtype=float)
    else:
        mat = np.asarray(series, dtype=float)
        if mat.ndim == 1:
            mat = mat.reshape(-1, 1)
        names = None
    out = mat.copy()
    last = np.zeros(mat.shape[1])
    for i in range(mat.shape[0]):
        row = out[i]
        bad = ~np.isfinite(row)
        row[bad] = last[bad]
        last = row
    return out, names


def _key(names: Optional[List[str]], vec: np.ndarray) -> Dict:
    if names is not None:
        return {n: float(v) for n, v in zip(names, vec)}
    return {i: float(v) for i, v in enumerate(vec)}


def _zero_map(names: Optional[List[str]], d: int) -> Dict:
    if names is not None:
        return {n: 0.0 for n in names}
    return {i: 0.0 for i in range(d)}


def direction(series: Sequence, ts: Optional[Sequence] = None,
              noise_floor: float = DEFAULT_NOISE_FLOOR) -> Dict:
    """The macro read from the last TWO readings — the currency-pair
    insight.

    One number is nothing; two numbers show DIRECTION. Returns the
    per-dial movement (sign + magnitude) between the two most recent
    readings. With ``ts`` the movement is normalized per second; without
    it, per sample (per pulse). Movements below ``noise_floor`` read as
    0 — the number doesn't matter, only the movement.

    Handles: series shorter than two readings (zeros — no direction
    yet), NaN (carried forward, never a movement), constant rooms
    (all zeros), and noisy rooms (small moves floored to 0).
    ``noise_floor`` is in the function's OUTPUT units — per sample
    (per pulse) by default, converted to per-second when ``ts`` is
    given.
    """
    mat, names = _as_matrix(series)
    d = mat.shape[1]
    if mat.shape[0] < 2:
        return _zero_map(names, d)
    delta = mat[-1] - mat[-2]
    floor = noise_floor
    if ts is not None:
        t = np.asarray(ts, dtype=float)[-len(mat):]
        dt = t[-1] - t[-2]
        if dt > 1e-12:
            delta = delta / dt
            floor = noise_floor / dt      # floor in output (per-second) units
        else:
            delta = np.zeros(d)
    delta = np.where(np.abs(delta) < floor, 0.0, delta)
    return _key(names, delta)


def rate_of_change(series: Sequence, ts: Optional[Sequence] = None,
                   noise_floor: float = DEFAULT_NOISE_FLOOR) -> Dict:
    """The macro read from THREE+ readings.

    MORE THAN TWO numbers show RATE OF CHANGE. From the last three
    readings this is the central second difference — the exact
    acceleration of the quadratic interpolant through them (the
    fleetmath `three_reading_kinematics` idea generalized to any dial
    series):

        a = 2·(v23 − v12) / (t3 − t1),   v_ij = (x_j − x_i)/(t_j − t_i)

    With ``ts`` the rate is per second²; without it, per sample² (per
    pulse²). Accelerations below ``noise_floor`` read as 0.

    Handles: series shorter than three readings (zeros — no rate yet),
    NaN (carried forward), constant rooms AND constant-speed rooms
    (zero acceleration — only a CHANGE in the movement is a rate), and
    noisy rooms (small accelerations floored to 0). ``noise_floor`` is
    in the function's OUTPUT units — per sample² (per pulse²) by
    default, converted to per-second² when ``ts`` is given.
    """
    mat, names = _as_matrix(series)
    d = mat.shape[1]
    if mat.shape[0] < 3:
        return _zero_map(names, d)
    accel = mat[-1] - 2.0 * mat[-2] + mat[-3]
    floor = noise_floor
    if ts is not None:
        t = np.asarray(ts, dtype=float)[-len(mat):]
        dt12 = t[-2] - t[-3]
        dt23 = t[-1] - t[-2]
        dt13 = t[-1] - t[-3]
        if dt12 > 1e-12 and dt23 > 1e-12 and dt13 > 1e-12:
            v12 = (mat[-2] - mat[-3]) / dt12
            v23 = (mat[-1] - mat[-2]) / dt23
            accel = 2.0 * (v23 - v12) / dt13   # exact quadratic interpolant
            floor = noise_floor / (dt23 * dt23)   # floor in output (per-s²) units
        else:
            accel = np.zeros(d)
    accel = np.where(np.abs(accel) < floor, 0.0, accel)
    return _key(names, accel)


# ---------------------------------------------------------------------- #
# The report — one pulse's macro read of the room                        #
# ---------------------------------------------------------------------- #
@dataclass
class PerceptionReport:
    """One pulse's macro read of the room — the trader's board.

    - ``warmth_direction`` — is the room warming or cooling? The
      headline, from the last TWO field vectors.
    - ``warmth_rate`` — is that movement accelerating or easing? The
      second difference, from the last THREE+.
    - ``direction`` / ``rate_of_change`` — the same macro read, per
      dial (mood +0.31/pulse, cynicism −0.05/pulse² ...).
    - ``dial_deltas`` — per-dial ``{"direction": ..., "rate": ...}``.
    - ``whole_hand`` — the table's conversation read AS A WHOLE: the
      macro, not any single message.

    All movement values are per-pulse units (per ``period`` seconds),
    so the board reads like a currency pair: the raw numbers don't
    matter, the movement is the perception.
    """

    agent_id: str
    ts: float
    n_readings: int
    warmth: float
    warmth_direction: float
    warmth_rate: float
    direction: Dict[str, float] = dc_field(default_factory=dict)
    rate_of_change: Dict[str, float] = dc_field(default_factory=dict)
    dial_deltas: Dict[str, Dict[str, float]] = dc_field(default_factory=dict)
    traffic: int = 0
    agent_said: bool = False
    whole_hand: str = ""

    def board(self) -> str:
        """One-line trader's read of this pulse (for demos/logs)."""
        movers = sorted(self.dial_deltas.items(),
                        key=lambda kv: abs(kv[1]["direction"]), reverse=True)
        parts = [f"t={self.ts:5.1f}", f"warmth {self.warmth:+.2f}",
                 f"Δ {self.warmth_direction:+.2f}/p",
                 f"Δ² {self.warmth_rate:+.2f}/p²"]
        shown = 0
        for name, dd in movers:
            if abs(dd["direction"]) > 1e-9 and shown < 3:
                parts.append(f"{name} {dd['direction']:+.2f}")
                shown += 1
        if self.traffic:
            parts.append(f"{self.traffic} msg")
        return "  ".join(parts)


def compose_whole_hand(report: PerceptionReport,
                       noise_floor: float = DEFAULT_NOISE_FLOOR) -> str:
    """The table's conversation as a WHOLE — the macro read in words.

    Deterministic: the headline (warming / cooling / holding), the pace
    of that movement (accelerating / easing / steady), the top three
    dials actually moving (and which way), and whether the table is
    talking at all.
    """
    n = report.n_readings
    if n < 2:
        return (f"Only {n} pulse{'s' if n != 1 else ''} in — the table "
                f"hasn't moved enough to feel a hand yet.")
    wd, wr = report.warmth_direction, report.warmth_rate
    if wd > noise_floor:
        head = "the table is warming"
    elif wd < -noise_floor:
        head = "the table is cooling"
    else:
        head = "the table is holding steady"
    if wr > noise_floor:
        pace = "and the movement is accelerating"
    elif wr < -noise_floor:
        pace = "and the movement is easing"
    else:
        pace = "and the movement is steady"
    movers = []
    for name, dd in sorted(report.dial_deltas.items(),
                           key=lambda kv: abs(kv[1]["direction"]),
                           reverse=True):
        if abs(dd["direction"]) > noise_floor:
            verb = "rising" if dd["direction"] > 0 else "falling"
            movers.append(f"{name} {verb} {abs(dd['direction']):.2f}/pulse")
    mover_txt = ", ".join(movers[:3]) if movers else "no dial is moving"
    traffic_txt = (f"{report.traffic} new message"
                   f"{'s' if report.traffic != 1 else ''} crossed the table"
                   if report.traffic else "the table is quiet")
    return (f"As a whole hand: {head}, {pace} — {mover_txt}; "
            f"{traffic_txt}.")


def compose_monologue(report: PerceptionReport,
                      prompt: Optional[str] = None,
                      noise_floor: float = DEFAULT_NOISE_FLOOR) -> str:
    """The agent's silent thinking — 1-3 sentences of what it is
    noticing WITHOUT speaking. This is the part that runs even when the
    agent says nothing in the room."""
    n = report.n_readings
    if n < 2:
        return ("Only one pulse in — my ear is still warming to this "
                "room. Nothing to hold, nothing to say.")
    wd = report.warmth_direction
    if wd > noise_floor:
        head = "the room is warming"
    elif wd < -noise_floor:
        head = "the room is cooling"
    else:
        head = "the room is holding"
    wr = report.warmth_rate
    if wr > noise_floor:
        pace = "and the momentum is still building"
    elif wr < -noise_floor:
        pace = "and the momentum is easing"
    else:
        pace = "and the momentum is steady"
    s1 = f"I haven't said a word, but {head} — {pace}."
    mover = None
    for name, dd in sorted(report.dial_deltas.items(),
                           key=lambda kv: abs(kv[1]["direction"]),
                           reverse=True):
        if abs(dd["direction"]) > noise_floor:
            mover = (name, dd["direction"])
            break
    if mover:
        name, d = mover
        verb = "rising" if d > 0 else "falling"
        s2 = (f"{name.capitalize()} is the loudest hand on the table — "
              f"{verb} {abs(d):.2f} per pulse.")
    else:
        s2 = "Nothing on the dials is moving enough to matter."
    if prompt:
        focus = mover[0] if mover else "the table"
        s3 = f"Asked {prompt!r}: {focus} tells the story."
        return f"{s1} {s2} {s3}"
    return f"{s1} {s2}"


# ---------------------------------------------------------------------- #
# The PulseLoop — an agent's constant sensing heartbeat                  #
# ---------------------------------------------------------------------- #
class PulseLoop:
    """An agent's constant sensing heartbeat.

    Ticks on an interval even when the agent isn't speaking. Each tick
    reads the room's field, appends it to a rolling history, and runs a
    perception check — the macro read over the pulse history (direction
    from the last two readings, rate of change from the last three+).
    ``internal_monologue`` is the silent thinking that runs regardless
    of whether the agent says anything in the room.

    ``room`` may be a `Room` or anything exposing a normalized ``.room``
    — a `Space` adapter (``ChatSpace``, ``MudSpace``, ``SensorSpace``),
    a `TapNightSession`, a `BoatHarness`. The bank is taken from
    ``bank``, else from the object's own ``.bank``, else the default
    eight-dial bank. Values in reports are per-pulse units (per
    ``period`` seconds).
    """

    def __init__(self, agent_id: str, room, bank: Optional[DialBank] = None,
                 period: float = 5.0, history: int = 20,
                 noise_floor: float = DEFAULT_NOISE_FLOOR):
        self.agent_id = agent_id
        self.room = room
        self.bank = bank if bank is not None else getattr(room, "bank", None)
        if self.bank is None:
            self.bank = DialBank(DEFAULT_DIALS)
        self.names = list(self.bank.names())
        self.period = float(period)
        self.history = int(history)
        self.noise_floor = float(noise_floor)

        self._readings: List[Dict[str, float]] = []   # raw per-pulse dial dicts
        self._ts: List[float] = []
        self._clock = 0.0
        self._last_ts: Optional[float] = None
        self._n_msgs = 0
        self._last_report: Optional[PerceptionReport] = None

    # ------------------------------------------------------------------ #
    # The heartbeat                                                      #
    # ------------------------------------------------------------------ #
    def _target_room(self) -> Room:
        return self.room.room if hasattr(self.room, "room") else self.room

    def due(self, now: float) -> bool:
        """Is a pulse due? True when the caller's clock has advanced at
        least one ``period`` past the last tick (or no tick yet)."""
        if self._last_ts is None:
            return True
        return float(now) - self._last_ts >= self.period

    def tick(self, now: Optional[float] = None) -> PerceptionReport:
        """One pulse: read the room, record the reading, run the
        perception check, return the PerceptionReport.

        Ticks at or before the last tick are ignored (the heartbeat
        doesn't double-beat) and return the last report unchanged.
        """
        if now is None:
            now = self._clock
        now = float(now)
        if self._last_ts is not None and now <= self._last_ts:
            return self._last_report  # stale — no new perception

        room = self._target_room()
        readings = self.bank.readings(room)

        n_msgs = len(room.messages)
        reset = False
        if n_msgs < self._n_msgs:
            self._n_msgs = 0          # the room was reset (new session)
            reset = True
        new_msgs = room.messages[self._n_msgs:]
        traffic = 0 if reset else len(new_msgs)
        agent_said = (not reset) and any(m.author == self.agent_id
                                         for m in new_msgs)

        self._readings.append(dict(readings))
        self._ts.append(now)
        if len(self._readings) > self.history:
            self._readings.pop(0)
            self._ts.pop(0)

        self._n_msgs = n_msgs
        self._last_ts = now
        self._clock = max(self._clock, now + self.period)
        report = self.perception_check(traffic=traffic, agent_said=agent_said)
        self._last_report = report
        return report

    def pulse(self) -> PerceptionReport:
        """Convenience alias: one tick on the internal clock (advances
        by ``period`` each beat)."""
        return self.tick()

    # ------------------------------------------------------------------ #
    # The perception check — looking around, always                      #
    # ------------------------------------------------------------------ #
    def perception_check(self, traffic: int = 0,
                         agent_said: bool = False) -> PerceptionReport:
        """The macro read of the room over the pulse history.

        Direction from the last TWO field vectors; rate of change from
        the last THREE+ (the second difference); per-dial deltas; and
        the ``whole_hand`` — the table's conversation as a whole. All
        values are per-pulse units, so they read like a trader's board.
        """
        warm = [RoomField(d).warmth() for d in self._readings]
        warm_dir = direction(warm, noise_floor=self.noise_floor).get(0, 0.0)
        warm_rate = rate_of_change(warm, noise_floor=self.noise_floor).get(0, 0.0)
        dir_pp = direction(self._readings, noise_floor=self.noise_floor)
        rate_pp = rate_of_change(self._readings, noise_floor=self.noise_floor)
        dial_deltas = {n: {"direction": dir_pp.get(n, 0.0),
                           "rate": rate_pp.get(n, 0.0)} for n in self.names}
        report = PerceptionReport(
            agent_id=self.agent_id,
            ts=(self._ts[-1] if self._ts else 0.0),
            n_readings=len(self._readings),
            warmth=(warm[-1] if warm else 0.0),
            warmth_direction=warm_dir,
            warmth_rate=warm_rate,
            direction=dir_pp,
            rate_of_change=rate_pp,
            dial_deltas=dial_deltas,
            traffic=traffic,
            agent_said=agent_said,
            whole_hand="",
        )
        report.whole_hand = compose_whole_hand(report, self.noise_floor)
        return report

    def internal_monologue(self, prompt: Optional[str] = None) -> str:
        """The agent's silent thinking — 1-3 sentences of what it is
        noticing WITHOUT speaking. This is the part that runs even when
        the agent says nothing in the room."""
        report = self._last_report
        if report is None:
            return ("I've taken no pulses yet — my ear is still warming "
                    "to this room.")
        return compose_monologue(report, prompt=prompt,
                                 noise_floor=self.noise_floor)

    def last_readings(self) -> List[Dict[str, float]]:
        """The raw pulse series — the numbers that don't matter
        individually. Direction lives in the last two; rate of change
        in the last three+."""
        return [dict(r) for r in self._readings]

    def last_report(self) -> Optional[PerceptionReport]:
        """The most recent PerceptionReport (None before the first tick)."""
        return self._last_report

    @property
    def last_ts(self) -> Optional[float]:
        return self._last_ts

    def __len__(self) -> int:
        return len(self._readings)

    def __repr__(self) -> str:
        return (f"<PulseLoop {self.agent_id!r} period={self.period:g}s "
                f"history={len(self._readings)}/{self.history}>")
