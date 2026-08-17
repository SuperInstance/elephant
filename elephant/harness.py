"""BoatHarness — the nuts and bolts of the elephant on F/V EILEEN.

The elephant is a highly modular tool: standalone in a harness, or
plugged into any system that has sensors. This module is the standalone
harness — one boat, one place to plug every sense into:

- live sensor frames (radar targets, sounder biomass, nav, cameras)
  -> a rolling `SignalRoom` -> the fleet dials (`radar_coherence`,
  `sounder_biomass`, `fishing_day`);
- the crew's conversation -> a rolling text `Room` -> the vibe dials
  (mood, volume, earnestness, cynicism, joke_landing, panic, presence);
- the merged readings -> a `RoomField` (the room temperature) and a
  nudge prior over modalities (what the vision model should compare);
- the inductive good-day anchor: when `fishing_day` runs warm the day's
  3-dim feature vector [fleet κ, biomass, nav] is remembered, and the
  harness measures every later stretch against that anchor — the system
  feels the difference when the fishing turns, without ever being told
  the trope.

Everything is bounded (rolling rooms) so the harness can run for a
whole season without growing.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from elephant.dial import DialBank
from elephant.dials import DEFAULT_DIALS
from elephant.field import RoomField
from elephant.nudge import nudge_prior
from elephant.room import Message, Room
from elephant.sensors import (
    FishingDayDial,
    RadarCoherenceDial,
    SensorFrame,
    SignalRoom,
    SounderBiomassDial,
)


class BoatHarness:
    """One boat's elephant: sensor frames + conversation -> field + nudge.

    Holds a rolling `SignalRoom` (`.signal`) and a rolling text `Room`
    (`.conversation`), both trimmed to bounded length after every
    ingest (oldest dropped, lists stay sorted by ts). Reads merge the
    fleet dials over the signal room with the vibe dials over the
    conversation room.
    """

    def __init__(self, name: str = "EILEEN", max_signal_frames: int = 256,
                 max_messages: int = 400, step: float = 3600.0,
                 nav_speed_ref: float = 10.0):
        self.name = name
        self.max_signal_frames = int(max_signal_frames)
        self.max_messages = int(max_messages)
        self.step = float(step)             # auto-timestamp increment (s)
        self.nav_speed_ref = float(nav_speed_ref)  # kts that saturates nav at 1.0

        # Rolling rooms: sensor frames and conversation lines.
        self.signal = SignalRoom(f"{name}/sensors")
        self.conversation = Room(f"{name}/conversation")

        # Fleet dials over the signal room (fishing day shares the pair).
        self.radar_dial = RadarCoherenceDial()
        self.sounder_dial = SounderBiomassDial()
        self.fishing_dial = FishingDayDial(radar=self.radar_dial,
                                           sounder=self.sounder_dial)
        self.signal_bank = DialBank([self.radar_dial, self.sounder_dial,
                                     self.fishing_dial])
        # Vibe dials over the conversation room.
        self.text_bank = DialBank(DEFAULT_DIALS)

        # Inductive anchors: 3-dim day features from the good days.
        self.good_days: List[np.ndarray] = []

        # Internal clock for auto-timestamps.
        self._clock = 0.0

    # ------------------------------------------------------------------ #
    # Ingest                                                             #
    # ------------------------------------------------------------------ #
    def _next_ts(self, ts: Optional[float]) -> float:
        """Resolve a timestamp: explicit or auto-incrementing by `step`."""
        if ts is None:
            ts = self._clock
        ts = float(ts)
        self._clock = max(self._clock, ts + self.step)
        return ts

    def _trim(self) -> None:
        """Bound both rolling rooms; drop oldest, keep sorted by ts."""
        if len(self.signal.frames) > self.max_signal_frames:
            del self.signal.frames[: len(self.signal.frames) - self.max_signal_frames]
        if len(self.conversation.messages) > self.max_messages:
            del self.conversation.messages[: len(self.conversation.messages) - self.max_messages]

    def _add_frame(self, frame: SensorFrame) -> None:
        self.signal.frames.append(frame)
        self.signal.frames.sort(key=lambda f: f.ts)
        self._trim()

    def _add_message(self, msg: Message) -> None:
        self.conversation.messages.append(msg)
        self.conversation.messages.sort(key=lambda m: m.ts)
        self._trim()

    def ingest(self, frame: Any) -> "BoatHarness":
        """Ingest a `SensorFrame` (-> signal room) or `Message` (-> room)."""
        if isinstance(frame, SensorFrame):
            self._add_frame(frame)
        elif isinstance(frame, Message):
            self._add_message(frame)
        else:
            raise TypeError(
                f"ingest expects SensorFrame or Message, got {type(frame).__name__}"
            )
        return self

    def ingest_radar(self, targets, ts: Optional[float] = None,
                     meta: Optional[dict] = None) -> "BoatHarness":
        """Radar frame: `targets` is a list of (x, y) positions in km."""
        return self.ingest(SensorFrame(ts=self._next_ts(ts), sensor="radar",
                                       data=targets, meta=meta or {}))

    def ingest_sounder(self, biomass: float, ts: Optional[float] = None,
                       meta: Optional[dict] = None) -> "BoatHarness":
        """Sounder frame: `biomass` is a float in [0, 1]."""
        return self.ingest(SensorFrame(ts=self._next_ts(ts), sensor="sounder",
                                       data=float(biomass), meta=meta or {}))

    def ingest_nav(self, heading: float, speed: float,
                   ts: Optional[float] = None,
                   meta: Optional[dict] = None) -> "BoatHarness":
        """Nav frame: heading (deg) and speed (kts)."""
        data = {"heading": float(heading), "speed": float(speed)}
        return self.ingest(SensorFrame(ts=self._next_ts(ts), sensor="nav",
                                       data=data, meta=meta or {}))

    def ingest_camera(self, meta: Optional[dict] = None,
                      ts: Optional[float] = None) -> "BoatHarness":
        """Camera frame marker: the vision model's room; data lands in meta."""
        return self.ingest(SensorFrame(ts=self._next_ts(ts), sensor="camera",
                                       data=None, meta=meta or {}))

    def ingest_conversation(self, author: str, text: str,
                            ts: Optional[float] = None,
                            meta: Optional[dict] = None) -> "BoatHarness":
        """A conversation line -> the boat's text room.

        (`meta` is accepted for API symmetry; `Message` carries no meta.)
        """
        return self.ingest(Message(author=author, text=text,
                                   ts=self._next_ts(ts)))

    # ------------------------------------------------------------------ #
    # Read                                                               #
    # ------------------------------------------------------------------ #
    def readings(self) -> Dict[str, float]:
        """Merged dial readings: signal bank + text bank (text wins ties)."""
        merged = dict(self.signal_bank.readings(self.signal))
        merged.update(self.text_bank.readings(self.conversation))
        return merged

    def current_field(self) -> RoomField:
        """The merged readings as a room-temperature field."""
        return RoomField(self.readings())

    def current_nudge(self, modalities=None) -> np.ndarray:
        """Dial readings -> attention prior over modalities."""
        return nudge_prior(self.readings(), modalities)

    def fleet_kappa(self) -> float:
        """Fleet tightness: radar_coherence, [-1 scattered .. +1 clustered]."""
        return self.radar_dial.read(self.signal)

    def biomass(self) -> float:
        """Sounder biomass field, [0 empty .. 1 thick]."""
        return self.sounder_dial.read(self.signal)

    def fishing_day(self) -> float:
        """The day's luck field, [-1 poor .. +1 good]."""
        return self.fishing_dial.read(self.signal)

    def radar_kinematics(self) -> Dict[str, Any]:
        """Per-object direction/speed/accel from the last 3 radar frames."""
        return self.radar_dial.kinematics(self.signal)

    # ------------------------------------------------------------------ #
    # Day memory — the inductive good-day anchor                         #
    # ------------------------------------------------------------------ #
    def day_features(self) -> np.ndarray:
        """3-vector [fleet κ, biomass, nav] describing the day so far.

        nav = mean nav speed (kts) over nav frames, divided by
        `nav_speed_ref`, clipped to [0, 1] (0.0 with no nav frames).
        """
        nav_frames = self.signal.by_sensor("nav")
        if nav_frames:
            mean_speed = float(np.mean([float(f.data["speed"]) for f in nav_frames]))
            nav_feature = float(np.clip(mean_speed / self.nav_speed_ref, 0.0, 1.0))
        else:
            nav_feature = 0.0
        return np.array([self.fleet_kappa(), self.biomass(), nav_feature],
                        dtype=float)

    def day_memory(self, good_day_threshold: float = 0.2) -> Optional[np.ndarray]:
        """Store today's features as a good-day anchor, if the day was good.

        Returns the stored 3-dim feature vector when `fishing_day()` is
        at or above `good_day_threshold`, else None (nothing stored).
        """
        if self.fishing_day() >= good_day_threshold:
            feats = self.day_features()
            self.good_days.append(feats)
            return feats
        return None

    def inductive_signal(self, features=None) -> Dict[str, Any]:
        """Distance of the current (or given) features from the good-day anchor.

        Returns {"total", "radar", "biomass", "nav", "n_anchor_days",
        "anchor"}: total is the L2 norm of (features - anchor mean),
        radar/biomass/nav are the abs per-channel deviations. With no
        anchors yet, all deviations are 0.0 and the anchor is zeros.
        """
        feats = (self.day_features() if features is None
                 else np.asarray(features, dtype=float))
        if self.good_days:
            anchor = np.mean(np.stack(self.good_days, axis=0), axis=0)
            diff = feats - anchor
            total = float(np.linalg.norm(diff))
            devs = [abs(float(d)) for d in diff[:3]]
        else:
            anchor = np.zeros(3, dtype=float)
            total, devs = 0.0, [0.0, 0.0, 0.0]
        return {
            "total": total,
            "radar": devs[0],
            "biomass": devs[1],
            "nav": devs[2],
            "n_anchor_days": len(self.good_days),
            "anchor": anchor,
        }

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.signal) + len(self.conversation)

    def __repr__(self) -> str:
        return (f"<BoatHarness {self.name!r}: {len(self.signal)} frames, "
                f"{len(self.conversation)} msgs, {len(self.good_days)} good days>")
