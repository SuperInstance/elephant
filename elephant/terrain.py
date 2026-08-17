"""Terrain — the cave, the shadows, and the deadband.

The captain's founding reframing, implemented (docs/terrain-2026-08-17.md):

- The **Terrain** is the true state of a room: the full vectorized state
  over time — field vectors, dial readings, the Z_in/Z_out senses, the
  pulse history. The thing itself, never fully renderable. Nobody can
  look at the terrain and see it whole, and that is not the point.
- The **Shadow** is the witness mark: a deliberately LOSSY projection of
  the terrain that humans and agents can read — one line, labeled as a
  shadow, never claiming to be the terrain. Enough to recognize, never
  enough to be complete. The purpose is not fidelity; the purpose is
  **enough information to agree on the action**.
- The **Deadband** is the discipline: only when the terrain's movement
  crosses the band — a real warming, a real panic, a real anomaly — does
  it ring. Below significance, nothing rings; the room breathes, the
  shadows flicker, no one is disturbed. It rides the perception-math
  noise floor (pulse.py's 0.02) but as a CHAIN-OF-COMMAND gate.
- The **ChainOfCommand** is the captain's line made real: a deadband
  ringing up the chain of command. A ring reaches the lowest step first;
  if the terrain keeps crossing on subsequent checks the ring rises; a
  quiet room descends.

The elephant feels, the deadband decides, the chain acts.
"""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .dial import DialBank
from .dials import DEFAULT_DIALS
from .field import RoomField

__all__ = [
    "Terrain", "StateVector", "Shadow", "Ring", "Deadband",
    "ChainOfCommand", "read_state", "DEFAULT_STEPS",
]

DEFAULT_STEPS = ["host", "foreman", "captain"]

# The resting value per dial for shadow projection: where the dial
# reads when the room is not saying anything about that dimension.
# mood/joke_landing run [-1,1] and rest at 0; panic, volume, cynicism
# and model_vs_code rest near 0 (a calm room has no alarm, no shout,
# no sneer, no code); the rest sit at their 0.5 midpoints. A dial is
# "salient" by how far it has moved from its own resting value.
_NEUTRAL = {
    "mood": 0.0, "joke_landing": 0.0,
    "panic": 0.0, "volume": 0.0, "cynicism": 0.0,
    "model_vs_code": 0.0,
}
_LABEL_TAIL = ["warmth", "trend", "anomaly"]
# A dial must be this far off its resting value to earn a place in the
# shadow's line — a dial AT rest is not saying anything.
_SALIENCE_FLOOR = 0.10


def _neutral_for(name: str) -> float:
    return _NEUTRAL.get(name, 0.5)


# ===================================================================== #
# StateVector — one instant of the true state                           #
# ===================================================================== #
@dataclass
class StateVector:
    """The true state of the room at one instant — the thing itself.

    The field vectors, the dial readings, the Z_in/Z_out senses, all
    together. Never fully renderable; the shadows are what we can bear
    to see.
    """

    ts: float = 0.0
    space_id: str = ""
    dials: Dict[str, float] = dc_field(default_factory=dict)
    field: np.ndarray = dc_field(default_factory=lambda: np.zeros(0))
    warmth: float = 0.0
    kappa: float = 0.0
    trend: float = 0.0          # Z_out: where the room is going, [-1..+1]
    anomaly: float = 0.0        # Z_out: whether something is off, [0..1]
    zin: Optional[np.ndarray] = None    # the Z_in perception vector
    zout: Optional[np.ndarray] = None   # the Z_out prediction vector
    meta: Dict[str, Any] = dc_field(default_factory=dict)

    def salience(self) -> np.ndarray:
        """The vector the deadband compares: [field, warmth, trend,
        anomaly]. The READABLE senses of the room — the dials, the felt
        temperature, and the two Z_out senses. κ lives in the state
        (the true terrain holds it) but is not a sense that rings: the
        deadband crosses on a real warming, a real panic, a real
        anomaly, a real shift in the mix — never on the derived
        tightness of the field."""
        return np.concatenate([
            np.asarray(self.field, dtype=float).reshape(-1),
            [float(self.warmth), float(self.trend), float(self.anomaly)],
        ])

    def __repr__(self) -> str:
        return (f"StateVector(ts={self.ts:.1f}, space={self.space_id!r}, "
                f"warmth={self.warmth:+.2f}, κ={self.kappa:.2f}, "
                f"trend={self.trend:+.2f}, anomaly={self.anomaly:.2f})")


# ===================================================================== #
# Terrain — the true state over time                                    #
# ===================================================================== #
class Terrain:
    """A room's full vectorized state over time — the true ground.

    A time-indexed buffer of `StateVector`s: the field vectors, the dial
    readings, the Z_in/Z_out vectors, the pulse history — the thing
    itself, bounded (``capacity``) because even the true state is only
    kept as long as it matters.

    ``record`` accepts a ready `StateVector`, a dict of dial readings, a
    raw vector, or nothing (kwargs). ``record_room`` reads a Space
    adapter (or a bare Room) through the dial bank, and optionally a
    `DualDBRoom` for the Z_in/Z_out senses. ``hear`` records the trail
    of words — witness material for the shadows.
    """

    def __init__(self, space_id: str = "room",
                 names: Optional[Sequence[str]] = None,
                 capacity: int = 4096):
        self.space_id = space_id
        self.names: Optional[List[str]] = list(names) if names is not None else None
        self.capacity = max(1, int(capacity))
        self.records: List[StateVector] = []
        self.ts: List[float] = []
        self.transcript: List[Tuple[float, str, str]] = []

    # ------------------------------------------------------------------ #
    # Recording                                                          #
    # ------------------------------------------------------------------ #
    def record(self, state_vector: Union[StateVector, Dict[str, float],
                                         np.ndarray, Sequence[float], None] = None,
               ts: Optional[float] = None, space_id: Optional[str] = None,
               meta: Optional[Dict[str, Any]] = None, **kw) -> StateVector:
        """Store one instant of the true state; returns the stored vector.

        ``state_vector`` may be:
        - a `StateVector` (stored; ``ts``/``space_id``/``meta`` override),
        - a dict of dial readings {name: value} (partial is fine; warmth
          and κ are computed from what is given, defaults for the rest),
        - an array-like raw field vector (no dials),
        - None (build from kwargs: dials, warmth, kappa, trend, anomaly,
          zin, zout).
        Timestamps are monotonic: a record at or before the last one is
        bumped just past it, so the time index always advances.
        """
        if isinstance(state_vector, StateVector):
            from dataclasses import replace
            sv = replace(state_vector)          # never mutate the caller's state
            if self.names is None and sv.dials:
                self.names = list(sv.dials.keys())
        elif isinstance(state_vector, dict) or state_vector is None:
            dials: Dict[str, float] = dict(state_vector or {})
            dials.update(kw.get("dials") or {})
            if self.names is None and dials:
                self.names = list(dials.keys())
            names = self.names or []
            field = np.array([dials.get(n, 0.0) for n in names], dtype=float)
            rf = RoomField(dials)
            sv = StateVector(
                ts=0.0, space_id=self.space_id, dials=dials, field=field,
                warmth=float(kw.get("warmth", rf.warmth())),
                kappa=float(kw.get("kappa", rf.concentration())),
                trend=float(kw.get("trend", 0.0)),
                anomaly=float(kw.get("anomaly", 0.0)),
                zin=kw.get("zin"), zout=kw.get("zout"), meta={},
            )
        else:  # raw vector
            arr = np.asarray(state_vector, dtype=float).reshape(-1)
            if self.names is None:
                self.names = [f"dim{i}" for i in range(arr.shape[0])]
            sv = StateVector(
                ts=0.0, space_id=self.space_id, dials={}, field=arr,
                warmth=float(kw.get("warmth", 0.0)),
                kappa=float(kw.get("kappa", 0.0)),
                trend=float(kw.get("trend", 0.0)),
                anomaly=float(kw.get("anomaly", 0.0)),
                zin=kw.get("zin"), zout=kw.get("zout"), meta={},
            )
        if ts is not None:
            sv.ts = float(ts)
        if space_id is not None:
            sv.space_id = space_id
        if meta is not None:
            sv.meta = dict(meta)
        if self.records and sv.ts <= self.records[-1].ts:
            sv.ts = self.records[-1].ts + 1e-6
        self.records.append(sv)
        self.ts.append(sv.ts)
        if len(self.records) > self.capacity:
            excess = len(self.records) - self.capacity
            del self.records[:excess]
            del self.ts[:excess]
        return sv

    def record_room(self, space, bank: Optional[DialBank] = None,
                    dual=None, ts: Optional[float] = None,
                    meta: Optional[Dict[str, Any]] = None) -> StateVector:
        """Read a Space adapter (or a bare Room) into the terrain.

        Runs the dial bank over the space's room, computes warmth and κ
        from the field; if ``dual`` (a `DualDBRoom`) is given, perceives
        and predicts through it so the Z_in/Z_out senses join the state.
        """
        return self.record(read_state(space, bank=bank, dual=dual,
                                      ts=ts, meta=meta))

    def hear(self, author: str, text: str, ts: Optional[float] = None) -> None:
        """Record the trail of words — the raw material of shadows."""
        if ts is None:
            ts = self.records[-1].ts if self.records else 0.0
        self.transcript.append((float(ts), str(author), str(text)))

    # ------------------------------------------------------------------ #
    # Retrieval                                                          #
    # ------------------------------------------------------------------ #
    def state_at(self, ts: float) -> Optional[StateVector]:
        """The nearest record at-or-before ``ts`` (None if none)."""
        if not self.ts:
            return None
        i = bisect.bisect_right(self.ts, float(ts)) - 1
        if i < 0:
            return None
        return self.records[i]

    def recent(self, n: int = 1) -> List[StateVector]:
        """The last ``n`` records, newest last."""
        if n <= 0 or not self.records:
            return []
        return self.records[-n:]

    @property
    def last(self) -> Optional[StateVector]:
        return self.records[-1] if self.records else None

    # ------------------------------------------------------------------ #
    # The movement senses                                                #
    # ------------------------------------------------------------------ #
    def salience(self, sv: Optional[StateVector] = None) -> np.ndarray:
        """The deadband's view of a state (or the latest one)."""
        sv = sv if sv is not None else self.last
        if sv is None:
            return np.zeros(len(self.salience_labels()), dtype=float)
        return sv.salience()

    def salience_labels(self) -> List[str]:
        """Per-dimension names of the salience vector, for shadows/rings."""
        names = list(self.names) if self.names else []
        if not names:
            last = self.last
            d = last.field.shape[0] if last is not None else 0
            names = [f"dim{i}" for i in range(d)]
        return names + list(_LABEL_TAIL)

    def move(self) -> float:
        """Per-sample movement: max per-dim |delta| of the salience
        vector between the last two records (0.0 until there are two)."""
        if len(self.records) < 2:
            return 0.0
        a = self.records[-2].salience()
        b = self.records[-1].salience()
        d = np.abs(b - a)
        return float(d.max()) if d.size else 0.0

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        return (f"<Terrain {self.space_id!r} records={len(self.records)} "
                f"dims={len(self.salience_labels())}>")


def read_state(space, bank: Optional[DialBank] = None, dual=None,
               ts: Optional[float] = None,
               meta: Optional[Dict[str, Any]] = None) -> StateVector:
    """Build a `StateVector` from a Space adapter or a bare Room.

    Module-level convenience behind `Terrain.record_room`; usable
    standalone by anything that wants one witness of the room.
    """
    room = space.room if hasattr(space, "room") else space
    bank = bank if bank is not None else DialBank(DEFAULT_DIALS)
    names = list(bank.names())
    readings = bank.readings(room)
    field = RoomField(readings)

    zin: Optional[np.ndarray] = None
    zout: Optional[np.ndarray] = None
    trend = anomaly = 0.0
    if dual is not None:
        # Lazy import: dualdb is a heavy module; the terrain only needs
        # the two senses, and only when a bridge is actually wired.
        from .dualdb import PredictionType
        zin = dual.perceive()
        outs = dual.predict()
        if outs:
            zout = np.concatenate(
                [np.asarray(o.to_vector(), dtype=float) for o in outs]
            )
            for o in outs:
                if o.prediction_type == PredictionType.TREND:
                    trend = float(np.clip(o.value, -1.0, 1.0))
                elif o.prediction_type == PredictionType.ANOMALY_SCORE:
                    anomaly = float(np.clip(o.value, 0.0, 1.0))

    return StateVector(
        ts=float(ts) if ts is not None else 0.0,
        space_id=str(getattr(space, "name", getattr(room, "name", ""))),
        dials=readings,
        field=field.vector(names),
        warmth=field.warmth(),
        kappa=field.concentration(),
        trend=trend, anomaly=anomaly,
        zin=zin, zout=zout,
        meta=dict(meta or {}),
    )


# ===================================================================== #
# Shadow — the witness mark                                             #
# ===================================================================== #
class Shadow:
    """A lossy projection of the terrain — the shadow on the cave wall.

    ``project`` renders ONE line of witness from the terrain's latest
    state: labeled as a shadow, never claiming to be the terrain, and
    truthful — every number printed IS a number in the state it claims
    to witness. ``render_transcript`` renders the trail of words, each
    line a witness mark.
    """

    def __init__(self, terrain: Optional[Terrain] = None):
        self.terrain = terrain

    # ------------------------------------------------------------------ #
    # The one-liner                                                      #
    # ------------------------------------------------------------------ #
    def project(self, terrain: Optional[Terrain] = None,
                format: str = "text") -> str:
        """One line of witness from the terrain's latest state.

        Lossy by design: warmth, the three dials most off their resting
        value, and one phrase — enough to recognize the room, never
        enough to be the room. ``format="json"`` returns the same
        content as a JSON string (all dials, still a projection).
        """
        terrain = terrain or self.terrain
        space_id = terrain.space_id if terrain is not None else "?"
        if terrain is None or len(terrain) == 0:
            if format == "json":
                return json.dumps({"label": "shadow", "space_id": space_id,
                                   "state": "no state yet"})
            return f"shadow · {space_id} · no state yet"
        sv = terrain.last
        line = self.line(sv)
        if format == "json":
            top = self._top_dials(sv)
            return json.dumps({
                "label": "shadow",
                "space_id": sv.space_id or space_id,
                "ts": round(sv.ts, 3),
                "warmth": round(sv.warmth, 2),
                "kappa": round(sv.kappa, 2),
                "trend": round(sv.trend, 2),
                "anomaly": round(sv.anomaly, 2),
                "dials": {n: round(float(v), 2) for n, v in top},
                "phrase": self._phrase(sv),
            })
        return line

    def line(self, sv: StateVector) -> str:
        """The one-liner for a given state (helper; see ``project``)."""
        parts = [f"shadow · {sv.space_id} · t={sv.ts:.1f}",
                 f"warmth {sv.warmth:+.2f}"]
        for name, val in self._top_dials(sv):
            parts.append(f"{name} {float(val):+.2f}")
        return " · ".join(parts) + f" — {self._phrase(sv)}"

    def render_transcript(self, terrain: Optional[Terrain] = None,
                          since: Optional[float] = None,
                          limit: Optional[int] = None) -> str:
        """The trail of words as witness marks — one line per entry.

        ``since`` filters to entries at or after that time; ``limit``
        keeps only the newest entries. Every line is labeled a shadow.
        """
        terrain = terrain or self.terrain
        if terrain is None:
            return ""
        entries = [(t, a, x) for (t, a, x) in terrain.transcript
                   if since is None or t >= since]
        if limit is not None and limit >= 0:
            entries = entries[-limit:]
        return "\n".join(
            f"[shadow] t={t:6.1f}  {a}: {x}" for t, a, x in entries
        )

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _top_dials(self, sv: StateVector) -> List[Tuple[str, float]]:
        """The dials furthest from their resting value, up to three —
        dials AT rest say nothing, so they earn no place in the line."""
        ranked = sorted(
            ((n, float(v)) for n, v in sv.dials.items()
             if abs(float(v) - _neutral_for(n)) > _SALIENCE_FLOOR),
            key=lambda kv: abs(kv[1] - _neutral_for(kv[0])),
            reverse=True,
        )
        return ranked[:3]

    def _phrase(self, sv: StateVector) -> str:
        """One deterministic phrase — the shadow's reading of the room."""
        if sv.dials.get("panic", 0.0) >= 0.5:
            return "a fight is breaking out"
        if sv.dials.get("joke_landing", 0.0) >= 0.4:
            return "the room is laughing"
        if sv.anomaly >= 0.4:
            return "something is off — the room is not itself"
        if sv.warmth >= 0.25:
            return "the room is warm and loose"
        if sv.warmth <= -0.25:
            return "the room has gone cold"
        return "the room is holding steady"

    def __repr__(self) -> str:
        return f"<Shadow of {self.terrain!r}>"


# ===================================================================== #
# Ring — the escalation                                                 #
# ===================================================================== #
@dataclass
class Ring:
    """The escalation: who rings, what crossed, the shadow of the moment."""

    ts: float
    space_id: str
    magnitude: float          # the movement that crossed the band
    threshold: float
    what_crossed: str         # e.g. "panic +0.71", "mood -1.40"
    shadow: str               # the shadow of the moment
    who: str = "the elephant"

    def __str__(self) -> str:
        return (f"RING from {self.who} · {self.space_id} · t={self.ts:.1f} · "
                f"{self.what_crossed} ({self.magnitude:.2f} > {self.threshold:.2f}) — "
                f"{self.shadow}")


# ===================================================================== #
# Deadband — the discipline                                             #
# ===================================================================== #
class Deadband:
    """The gate: nothing rings below significance.

    A Schmitt trigger with a moving quiet anchor. Movement is the max
    per-dimension deviation of the terrain's salience vector from the
    anchor. While the room is quiet every check re-anchors (the room
    breathes); the first time movement crosses ``threshold`` the deadband
    rings and the anchor FREEZES — the terrain has left the quiet state,
    and it keeps crossing (and keeps ringing) until it comes back within
    ``threshold * hysteresis`` of where it was. That sustained crossing
    is what lets the ring climb the chain of command.

    The default threshold, 0.10, is five times the perception-math noise
    floor (pulse.py: 0.02 per pulse): the room must move that much to
    disturb anyone.

    This is a LEVEL gate, not a velocity gate: the spec's deadband
    crosses on "a dial crossing its significance threshold", "the room
    warming past the band" — the state being past the band, not merely
    the moment of change. So while a room stays past the band (even if
    it has stopped moving — a fight that holds its temperature), the
    deadband keeps the chain rung: the chain HOLDS the alarm rather than
    re-ringing it from scratch, and only a room that returns within the
    release level of its quiet state descends. A settled-but-changed
    room keeps the chain rung until it genuinely comes back — that is
    the discipline's price, and it is the safe direction to err.
    """

    def __init__(self, threshold: float = 0.10, hysteresis: float = 0.5,
                 who: str = "the elephant"):
        self.threshold = float(threshold)
        self.hysteresis = float(hysteresis)
        self.who = who
        self._anchor: Optional[np.ndarray] = None
        self._crossed = False
        self._cross_threshold: Optional[float] = None   # the threshold the
                                                        # current crossing
                                                        # actually tripped

    @property
    def crossed(self) -> bool:
        """Is the terrain currently across the band?"""
        return self._crossed

    def movement(self, terrain: Terrain) -> float:
        """The current movement: max per-dim |deviation| of the latest
        state from the quiet anchor (0.0 before the first check)."""
        if terrain is None or len(terrain) == 0 or self._anchor is None:
            return 0.0
        dev = np.abs(terrain.salience() - self._anchor)
        return float(dev.max()) if dev.size else 0.0

    def check(self, terrain: Terrain) -> Optional[Ring]:
        """One check of the terrain.

        Returns a `Ring` only when the terrain's movement crosses the
        deadband — a real warming, a real panic, a real anomaly. Below
        significance: None — the room breathes, no one is disturbed.
        """
        if terrain is None or len(terrain) == 0:
            return None
        last = terrain.last
        sal = terrain.salience(last)
        if self._anchor is None:
            # The terrain needs a baseline before it can cross anything.
            self._anchor = sal.copy()
            return None
        dev = np.abs(sal - self._anchor)
        movement = float(dev.max()) if dev.size else 0.0

        if not self._crossed:
            if movement > self.threshold:
                self._crossed = True
                self._cross_threshold = float(self.threshold)
                return self._ring(terrain, last, dev, movement)
            self._anchor = sal.copy()   # quiet: the room re-settles
            return None
        # Crossed: only coming back within the release level quiets it.
        # The release is keyed to the threshold that TRIPPED the crossing
        # (not a later-changed one), so the gate always lets go.
        release = (self._cross_threshold if self._cross_threshold is not None
                   else self.threshold) * self.hysteresis
        if movement < release:
            self._crossed = False
            self._cross_threshold = None
            self._anchor = sal.copy()
            return None
        return self._ring(terrain, last, dev, movement)

    def _ring(self, terrain: Terrain, sv: StateVector,
              dev: np.ndarray, movement: float) -> Ring:
        labels = terrain.salience_labels()
        idx = int(np.argmax(dev)) if dev.size else 0
        label = labels[idx] if idx < len(labels) else "dim"
        sal = sv.salience()
        anchor_val = (self._anchor[idx] if self._anchor is not None
                      and idx < len(self._anchor) else 0.0)
        signed = float(sal[idx]) - float(anchor_val)   # the real delta
        return Ring(
            ts=sv.ts,
            space_id=sv.space_id or terrain.space_id,
            magnitude=movement,
            threshold=self.threshold,
            what_crossed=f"{label} {signed:+.2f}",
            shadow=Shadow(terrain).project(),
            who=self.who,
        )

    def __repr__(self) -> str:
        state = "crossed" if self._crossed else "quiet"
        return f"<Deadband {self.threshold:g} ({state})>"


# ===================================================================== #
# ChainOfCommand — the captain's line, made real                        #
# ===================================================================== #
class ChainOfCommand:
    """A deadband ringing up the chain of command.

    ``escalate(ring)``: a ring reaches the LOWEST step first; if the
    terrain keeps crossing on subsequent checks, each consecutive ring
    rises one step. ``quiet()``: a quiet room descends — after
    ``quiet_after`` consecutive quiet checks the level drops one step,
    and sustained quiet keeps dropping one step per check. A ring after
    quiet starts at the lowest step again.

    ``ring()`` returns the current level — the step name, or None when
    no one is ringing.
    """

    def __init__(self, steps: Optional[Sequence[str]] = None,
                 quiet_after: int = 2):
        self.steps = list(steps) if steps is not None else list(DEFAULT_STEPS)
        self.quiet_after = max(1, int(quiet_after))
        self.history: List[Ring] = []
        self.last_ring: Optional[Ring] = None
        self._level: Optional[int] = None     # index into steps
        self._ring_streak = 0
        self._quiet_streak = 0

    def escalate(self, ring: Ring) -> Optional[str]:
        """A ring arrives. Returns the current level after escalation."""
        self.history.append(ring)
        self.last_ring = ring
        self._ring_streak += 1
        self._quiet_streak = 0
        if self._level is None:
            if self.steps:
                self._level = 0        # the lowest step hears it first
        elif self._level < len(self.steps) - 1:
            self._level += 1           # keeps crossing -> rises
        return self.ring()

    def quiet(self) -> Optional[str]:
        """A quiet check (no ring). Returns the level after descent."""
        self._quiet_streak += 1
        self._ring_streak = 0
        if (self._level is not None and self.steps
                and self._quiet_streak >= self.quiet_after):
            self._level -= 1
            if self._level < 0:
                self._level = None
        return self.ring()

    def report(self, ring: Optional[Ring]) -> Optional[str]:
        """Convenience: escalate if a ring crossed, else note the quiet."""
        if ring is not None:
            return self.escalate(ring)
        return self.quiet()

    def ring(self) -> Optional[str]:
        """The current level: a step name, or None when nothing is ringing."""
        if self._level is None or not self.steps:
            return None
        return self.steps[self._level]

    def __repr__(self) -> str:
        return f"<ChainOfCommand {' -> '.join(self.steps)} — ringing: {self.ring()!r}>"
