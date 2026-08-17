"""Sensor dials — the elephant's sea legs.

The elephant is a highly modular tool: standalone in a harness, or
plugged into any system that has sensors. On F/V EILEEN every sensor
becomes a room, and every room gets dials:

- RADAR       — the distribution of boats. Three readings tell the
                direction, speed, and rate of change of every object.
                Clustered = on fish (same drag/tack). Scattered =
                searching. JEPA *feels* the distribution; the trope
                (boats together on fish) is a deduction we never make
                — the field just reads tight or loose.
- SOUNDER     — the biomass below. Specific targets sometimes; mostly
                a look, a texture, a density felt through experience.
- CAMERAS     — looking out, and on deck. Vision model's room.
- NAV/AP      — course, autopilot, where the boat is going.
- CONVERSATION — AI + crew. LOCAL ONLY for most boats: the dial runs
                on the boat, shares only its NUMBER, never the feed.

The dials' numbers are the nudge: they tell the vision model WHAT TO
COMPARE TOGETHER. A week of good fishing days becomes the warm room;
when fishing goes spotty the system feels the difference inductively
— it learns what driving over the right kind of biomass looks like.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dial import Dial


# --------------------------------------------------------------------- #
# Signal rooms — rooms whose "messages" are sensor frames               #
# --------------------------------------------------------------------- #
@dataclass
class SensorFrame:
    ts: float
    sensor: str                       # "radar" | "sounder" | "camera" | ...
    data: Any                         # radar: list[(x,y)] · sounder: float ...
    meta: Dict[str, Any] = dc_field(default_factory=dict)


class SignalRoom:
    """A room of sensor frames instead of messages.

    Same idea as a text room: a timestamped sequence whose FIELD is
    read by dials. The frames are the room's murmur; the dials are
    the elephant.
    """

    def __init__(self, name: str, frames: Optional[Sequence[SensorFrame]] = None):
        self.name = name
        self.frames: List[SensorFrame] = list(frames or [])
        self.frames.sort(key=lambda f: f.ts)

    def by_sensor(self, sensor: str) -> List[SensorFrame]:
        return [f for f in self.frames if f.sensor == sensor]

    def __len__(self) -> int:
        return len(self.frames)

    def __repr__(self) -> str:
        return f"SignalRoom({self.name!r}, {len(self.frames)} frames)"


# --------------------------------------------------------------------- #
# Radar dial — the fleet field                                          #
# --------------------------------------------------------------------- #
class RadarCoherenceDial(Dial):
    """Feels the radar's distribution of boats.

    Reading: [-1 scattered/searching .. +1 clustered/on fish].
    Plus `kinematics()`: per-object direction, speed, and rate of
    change, recovered from three readings — the JEPA way to know
    where everything is going without ever being told the trope.
    """

    name = "radar_coherence"
    description = "fleet field: tight (on fish) or scattered (searching), [-1 .. +1]"

    def read(self, room: "SignalRoom") -> float:
        frames = room.by_sensor("radar")
        if not frames:
            return 0.0
        recent = frames[-3:]
        spreads = [self._spread(f) for f in recent]
        base = 1.0 - min(1.0, float(np.mean(spreads)) / 4.0)   # km scale
        if len(spreads) >= 2:
            trend = spreads[-1] - spreads[0]                    # closing = tighter
            base += 0.30 * float(np.clip(-trend, -1, 1))
        return float(np.clip(base * 2.0 - 1.0, -1, 1))

    def kinematics(self, room: "SignalRoom") -> Dict[str, Any]:
        """Direction, speed, and rate of change of every object, from
        three radar readings."""
        frames = room.by_sensor("radar")
        if len(frames) < 3:
            return {"objects": [], "fleet_mean_speed": 0.0, "spread_rate": 0.0}
        f1, f2, f3 = frames[-3:]
        dt12 = max(f2.ts - f1.ts, 1e-6)
        dt23 = max(f3.ts - f2.ts, 1e-6)
        pairs12 = _associate(_targets(f1), _targets(f2))
        pairs23 = _associate(_targets(f2), _targets(f3))

        objects = []
        for (p2, p3) in pairs23:
            v = (p3 - p2) / dt23
            # find p2's predecessor for acceleration
            pred = next((a for (a, b) in pairs12 if np.linalg.norm(b - p2) < 0.5), None)
            acc = (v - (p2 - pred) / dt12) / dt23 if pred is not None else np.zeros(2)
            objects.append({
                "pos": p2.tolist(),
                "dir_deg": float(np.degrees(math.atan2(v[1], v[0]))),
                "speed_kts": float(np.linalg.norm(v)) * 1.94384,   # u/s -> kts
                "accel": acc.tolist(),
            })
        speeds = [o["speed_kts"] for o in objects]
        spreads = [self._spread(f) for f in frames[-3:]]
        spread_rate = (spreads[-1] - spreads[0]) / max(f3.ts - f1.ts, 1e-6)
        return {
            "objects": objects,
            "fleet_mean_speed": float(np.mean(speeds)) if speeds else 0.0,
            "spread_rate": float(spread_rate),   # + = scattering, - = bunching
        }

    @staticmethod
    def _spread(frame: SensorFrame) -> float:
        pts = _targets(frame)
        if len(pts) < 2:
            return 0.0
        c = np.mean(pts, axis=0)
        return float(np.mean(np.linalg.norm(pts - c, axis=1)))


# --------------------------------------------------------------------- #
# Sounder dial — the biomass look                                       #
# --------------------------------------------------------------------- #
class SounderBiomassDial(Dial):
    """Feels the depth sounder's biomass.

    Sometimes the sounder shows objects identifiable specifically;
    mostly it's a look — a density, a texture, felt through
    experience. This dial reads the biomass field and its trend.
    """

    name = "sounder_biomass"
    description = "biomass field under the keel, [0 empty .. 1 thick]"

    def read(self, room: "SignalRoom") -> float:
        frames = room.by_sensor("sounder")
        if not frames:
            return 0.0
        vals = [float(f.data) for f in frames]
        recent = np.mean(vals[-5:])
        base = float(np.clip(recent, 0, 1))
        if len(vals) >= 2:
            trend = vals[-1] - vals[0]
            base += 0.20 * float(np.clip(trend, -1, 1))
        return float(np.clip(base, 0, 1))


# --------------------------------------------------------------------- #
# Fishing-day dial — the composite luck field                           #
# --------------------------------------------------------------------- #
class FishingDayDial(Dial):
    """The room temperature of a fishing day.

    Composite over the boat's dials: radar coherence + sounder biomass
    + (optionally) catch meta. Good days are warm rooms. Poor days are
    cold plunges. The greater system learns the difference inductively
    and drives accordingly.
    """

    name = "fishing_day"
    description = "the day's luck field, [-1 poor .. +1 good]"

    def __init__(self, radar: Optional[RadarCoherenceDial] = None,
                 sounder: Optional[SounderBiomassDial] = None):
        self.radar = radar or RadarCoherenceDial()
        self.sounder = sounder or SounderBiomassDial()

    def read(self, room: "SignalRoom") -> float:
        r = self.radar.read(room)
        s = self.sounder.read(room)
        luck = 0.55 * r + 0.45 * (s * 2.0 - 1.0)
        return float(np.clip(luck, -1, 1))


# --------------------------------------------------------------------- #
# Association helper (nearest-neighbour gating)                         #
# --------------------------------------------------------------------- #
def _targets(frame: SensorFrame) -> np.ndarray:
    data = frame.data
    if isinstance(data, np.ndarray):
        return data.reshape(-1, 2)
    return np.asarray(list(data), dtype=float).reshape(-1, 2)


def _associate(a: np.ndarray, b: np.ndarray, gate: float = 2.0) -> List[Tuple[np.ndarray, np.ndarray]]:
    pairs = []
    used = set()
    for i, pa in enumerate(a):
        best, best_d = None, gate
        for j, pb in enumerate(b):
            if j in used:
                continue
            d = float(np.linalg.norm(pa - pb))
            if d < best_d:
                best, best_d = j, d
        if best is not None:
            used.add(best)
            pairs.append((pa, b[best]))
    return pairs
