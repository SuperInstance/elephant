"""Vision dial — the room's visual energy, felt from camera frames.

Cross-pollination from the PLATO nervous system's vision layer
(`SuperInstance/plato-vision-jepa`). Plato reads a room the way the
elephant does — from frames, not words. Its signal chain is

    Camera → Frame Histogram → VisionDeadband → JEPA → RoomVisionState

and its perception is a **16-dimensional room-state vector**: brightness,
motion, occupancy, anomaly, quadrant activity, temporal trends. This dial
is the elephant's version of that sense: it reads a `SignalRoom`'s camera
frames (a `SensorFrame` with `sensor="camera"`) whose `data` is either a
full 16-dim room-state vector or a plain `{brightness, motion, occupancy,
anomaly}` dict — and reports ONE reading: the room's visual
energy/aliveness.

The deadband travels with it. Plato's `VisionDeadband` only runs the JEPA
on frames whose histogram changed significantly (default threshold 0.05),
because a camera pushing the same room state every second is not telling
you anything new. This dial applies the same filter over the frame
sequence at read time: a frame that repeats the previous room state is
skipped, so a camera piling up identical frames can never dominate the
reading.

Range: `[0 dark+empty .. 1 bright+alive]`. No camera frames → `0.5` — the
room has no visual opinion, so the dial rests at neutral.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

from ..dial import Dial
from ..sensors import SensorFrame, SignalRoom

# --------------------------------------------------------------------- #
# plato-vision-jepa 16-dim layout (indices) — this dial reads 0-3;       #
# 4-7 are quadrant activity, 8-11 temporal trends, 12-15 reserved.       #
# --------------------------------------------------------------------- #
BRIGHTNESS_IDX, MOTION_IDX, OCCUPANCY_IDX, ANOMALY_IDX = 0, 1, 2, 3

# How much each field contributes to "aliveness" (weights sum to 1).
W_BRIGHTNESS, W_MOTION, W_OCCUPANCY = 0.40, 0.35, 0.25
# Anomaly is a bonus spike: it pushes the reading toward 1.0 by this
# fraction of the headroom left after the base energy.
ANOMALY_BONUS = 0.5

# Dict-form key spellings — plato's struct field names accepted too.
_KEYS = {
    "brightness": ("brightness",),
    "motion": ("motion", "motion_level"),
    "occupancy": ("occupancy", "occupants"),
    "anomaly": ("anomaly", "anomaly_score"),
}


def _norm(x: Any) -> float:
    """Coerce a field to a finite value in [0, 1].

    Plato semantics: all four fields are normalized 0-1 (occupancy is a
    normalized count). Out-of-range or non-finite values are clamped to
    the valid range so a raw person count or a NaN can't break the dial.
    """
    try:
        v = float(x)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if v != v or v in (float("inf"), float("-inf")):   # nan / inf
        return 0.0
    return max(0.0, min(1.0, v))


def _fields_from_data(data: Any) -> Optional[Tuple[float, float, float, float]]:
    """Extract (brightness, motion, occupancy, anomaly) from a frame's data.

    Accepts either form, documented:

    - **16-dim vector** — list/tuple of 16 floats, the plato
      `RoomVisionState.to_vector()` layout; indices 0-3 are brightness,
      motion_level, occupancy, anomaly_score (per the table above). Any
      sequence of ≥ 4 floats uses the first 4.
    - **dict** — keys `brightness`, `motion`, `occupancy`, `anomaly`
      (plato's spellings `motion_level` / `anomaly_score` accepted);
      missing keys read 0.0.

    Returns `None` when the data is unreadable (wrong shape/type).
    """
    if isinstance(data, dict):
        matched = False

        def pick(aliases: Tuple[str, ...]) -> float:
            nonlocal matched
            for k in aliases:
                if k in data:
                    matched = True
                    return _norm(data[k])
            return 0.0

        fields = (pick(_KEYS["brightness"]), pick(_KEYS["motion"]),
                  pick(_KEYS["occupancy"]), pick(_KEYS["anomaly"]))
        # A dict that names none of the room-state fields isn't a room
        # state at all — treat it as unreadable rather than "dark empty".
        return fields if matched else None
    if isinstance(data, (list, tuple)) and len(data) >= 4:
        return tuple(_norm(x) for x in data[:4])  # type: ignore[return-value]
    return None


def _state_diff(a: Tuple[float, float, float, float],
                b: Tuple[float, float, float, float]) -> float:
    """Mean absolute difference across the four fields, in [0, 1] — the
    elephant's stand-in for plato's histogram-intersection distance."""
    return sum(abs(x - y) for x, y in zip(a, b)) / 4.0


class VisionDial(Dial):
    name = "vision"
    description = "the room's visual energy/aliveness from camera frames, [0 dark+empty .. 1 bright+alive]"

    def __init__(self, deadband: float = 0.05):
        """`deadband` is plato's `VisionDeadband` threshold: a frame whose
        room-state differs from the previous frame by less than this is
        skipped — no significant visual change, no new information.
        Default 0.05 matches plato."""
        self.deadband = float(deadband)

    def read(self, room: "SignalRoom") -> float:
        # A plain text Room (the shared bank reads those too) has no
        # camera — no visual opinion, rest at neutral. SignalRooms expose
        # by_sensor; anything else is a room the vision sense can't see.
        by_sensor = getattr(room, "by_sensor", None)
        if by_sensor is None:
            return 0.5
        frames = by_sensor("camera")
        if not frames:
            return 0.5                       # no camera, no visual opinion
        states: List[Tuple[float, float, float, float]] = []
        prev: Optional[Tuple[float, float, float, float]] = None
        for f in frames:
            s = _fields_from_data(f.data)
            if s is None:
                continue                     # unreadable frame — skip
            if prev is not None and _state_diff(prev, s) <= self.deadband:
                continue                     # deadband: nothing significant
            prev = s
            states.append(s)
        if not states:
            return 0.5                       # frames present, none readable
        recent = states[-8:]                 # the recent visual field
        return sum(self._energy(s) for s in recent) / len(recent)

    @staticmethod
    def _energy(s: Tuple[float, float, float, float]) -> float:
        brightness, motion, occupancy, anomaly = s
        base = (W_BRIGHTNESS * brightness + W_MOTION * motion
                + W_OCCUPANCY * occupancy)
        # Anomaly is a bonus spike: it pushes the reading toward 1.0 by
        # the headroom left after the base energy.
        return max(0.0, min(1.0, base + ANOMALY_BONUS * anomaly * (1.0 - base)))
