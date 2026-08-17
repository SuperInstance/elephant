"""plato_rpg — the elephant as dungeon master.

A GM-less agentic tabletop RPG engine built on the elephant's own
organs. The captain: "we can rapidly iterate as JEPA learning avatar
building round characters at Tap's bar but also as parts of
Plato-based Agentic RPGs." This module is the second half of that
sentence — a game where:

- **The WORLD is rooms**, each with an elephant reading (field, dials,
  trend, anomaly). A dungeon room, a tavern, a haunted wheelhouse —
  each has its own temperature.
- **The PLAYERS are avatars** — round characters carrying a
  PersonalElephant (vibe + dial_weights + charisma, the tapnight
  `Participant` anatomy), a pulse, and a voice. When a learned
  `Avatar` (elephant/avatar.py — the Tap's round characters) is
  attached, it speaks and monologues for the player.
- **The PERCEPTION CHECK is the roll**: a player entering a room reads
  the room's direction/rate of change over its recent history (the
  `pulse.perception_check` math — two numbers show direction, three
  show rate of change). The number doesn't matter; the movement is
  the perception.
- **The GM's narration is the SHADOW** (`mud.tint_description` + the
  perception reports in words) — the cave wall. The terrain (raw
  vectors) is never narrated.
- **The DEADBAND rings the plot**: when a room's field crosses its
  deadband (a fight erupts, an anomaly spikes, a trend inverts), the
  Ring escalates — GM line, plot stage, the dungeon answers.
- **Z_out predicts**: the room's trend dial (dualdb) tells the players
  what's coming around the corner — the ghost-trail ahead.

The three laws (docs/terrain-2026-08-17.md):

1. **Terrain** — the true state: the full field history, the vectors.
   Nobody sees it whole, and that is not the point.
2. **Shadow** — what anyone actually sees: the tinted description, the
   perception report in words, the monologue. Enough to agree on the
   action, never complete.
3. **Deadband** — the discipline: only significant movement rings, and
   a ring advances the plot. Not every flicker; only the moves that
   matter.

Design notes (from the Seed-2.0-pro critique, folded in):

- **Entry ripple**: when a player enters, the room re-reads WITH their
  charisma pull (scaled) — the room responds to the newcomer, and the
  perception check includes your own effect. The shadow moves because
  you moved the fire.
- **Ring discipline**: per-monitor hysteresis + a short cooldown, and
  at most three rings narrate per turn (cascading drama without spam).
  Each ring advances the plot.
- **Context-aware GM lines**: the prompt bank templates are slotted
  with the actual room, dial, player, and severity — plus the players'
  PersonalElephant filters the foreshadowing (a brooder hears a
  different fog than a comedian does).

numpy-only, deterministic given a seed, scenarios as data. The seam to
avatar.py is live: pass a learned `Avatar` to `RPGPlayer` and it
speaks and monologues for the player; without one, the archetype
presets (the flat characters) play.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field as dc_field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from elephant.dial import DialBank
from elephant.dials import DEFAULT_DIALS
from elephant.dualdb import DualDBRoom
from elephant.field import DIAL_NAMES, RoomField, read_field
from elephant.mud import tint_description
from elephant.pulse import (
    DEFAULT_NOISE_FLOOR,
    PerceptionReport,
    PulseLoop,
    compose_monologue,
    compose_whole_hand,
    direction,
    rate_of_change,
)
from elephant.room import Message, Room
from elephant.tapnight import DIAL_BOUNDS, Participant

__all__ = [
    "RPGWorld", "RPGRoom", "Deadband", "RingEvent",
    "RPGPlayer", "PersonalElephant", "ARCHETYPES",
    "RPGEngine", "RPGLog", "GM_BANK",
    "perception_check", "report_words", "run_scenario",
    "RIPPLE_STRENGTH",
]

# How strongly a newcomer's presence bends the room on entry: the
# charisma blend is (1 - exp(-charisma*n)) scaled by this. A strong
# presence warms the room the moment they walk in; a cold one chills
# it. The perception check then includes your own effect.
RIPPLE_STRENGTH = 1.5
# How many readings the room keeps in its shared record.
HISTORY_CAP = 12
# Game clock step per turn (seconds).
TURN_STEP = 300.0


# ===================================================================== #
# The deadband — the discipline that decides when the truth must ring  #
# ===================================================================== #
@dataclass
class RingEvent:
    """One crossing of a deadband — the witness mark that rang.

    - ``kind``: ``panic`` / ``warmth_hi`` / ``warmth_lo`` / ``anomaly`` /
      ``trend_invert`` / a named dial.
    - ``severity``: how far past the band (always >= 0).
    - ``player``: the last actor whose ripple crossed the line (if any).
    """
    turn: int
    room: str
    kind: str
    dial: str
    value: float
    prev: Optional[float]
    severity: float
    player: str = ""

    def marker(self) -> str:
        """One compact factual line — the ring in the game log."""
        who = f" (it is {self.player})" if self.player else ""
        return (f"⚡ [{self.kind}] {self.room} — {self.dial} crosses the "
                f"band +{self.severity:.2f}{who}.")


@dataclass
class Deadband:
    """Hysteresis thresholds per monitored signal.

    A monitor rings when its value crosses the threshold from below
    (``warmth_lo`` from above) and the cooldown has passed since the
    last ring; it re-arms only after dipping back under the band. The
    dungeon does not shout twice about the same fire.
    """
    dials: Dict[str, float] = dc_field(default_factory=dict)
    warmth_hi: Optional[float] = None
    warmth_lo: Optional[float] = None
    anomaly: Optional[float] = None
    trend_flip: Optional[float] = 0.15
    cooldown: int = 2


# ===================================================================== #
# The room — a room, its elephant, and its sense of itself             #
# ===================================================================== #
class RPGRoom:
    """A dungeon room: a message stream, an elephant (dial bank), a
    dual-DB bridge (Z_in perception / Z_out prediction), and a deadband.

    The room keeps a shared record — ``field_history`` — of effective
    readings (raw field + the charisma ripple of whoever entered). The
    record is the shadow the players share; the raw field is always
    recoverable via ``raw_field()`` (the terrain's projection).
    """

    def __init__(self, name: str, description: str = "",
                 seed_messages: Optional[Iterable[Message]] = None,
                 bank: Optional[DialBank] = None,
                 deadband: Optional[Deadband] = None,
                 hour: Optional[float] = None,
                 history_cap: int = HISTORY_CAP):
        self.name = name
        self.description = description
        self.bank = bank or DialBank(DEFAULT_DIALS)
        self.deadband = deadband or Deadband()
        self.hour = hour
        self.history_cap = int(history_cap)

        self.room = Room(name)
        # Z_in = the full RAW encoding of the dial readings (the
        # per-dial [value, confidence, ts] triples; confidence and ts
        # are pinned constant, so the moving dims ARE the dial values).
        self.bridge = DualDBRoom(self.room, zin_dim=None)
        self.field_history: List[Dict[str, float]] = []      # effective (ripples in)
        self.field_history_raw: List[Dict[str, float]] = []  # the terrain's projection
        self.field_ts: List[float] = []
        self._ring_state: Dict[str, dict] = {}
        self._trend_state: Optional[float] = None
        self._interactions: Dict[str, int] = {}
        self.last_actor: str = ""

        self._bootstrap(list(seed_messages or []))

    # ------------------------------------------------------------------ #
    # Bootstrap — the room's life before the players arrived             #
    # ------------------------------------------------------------------ #
    def _bootstrap(self, seeds: List[Message]) -> None:
        """Take a reading per distinct seed timestamp (capped): the room
        has a history before anyone walks in — the tavern that warmed
        through the evening, the wheelhouse that went mad by midnight.
        """
        if not seeds:
            self.re_read(t=0.0)
            return
        seeds.sort(key=lambda m: m.ts)
        # Group into beats by timestamp (1-decimal precision).
        groups: List[List[Message]] = []
        cur, cur_key = [], None
        for m in seeds:
            key = round(m.ts, 1)
            if cur_key is None or key == cur_key:
                cur.append(m)
                cur_key = key
            else:
                groups.append(cur)
                cur, cur_key = [m], key
        if cur:
            groups.append(cur)
        # Keep the most recent beats (dials read ALL messages, but the
        # reading record stays bounded).
        for group in groups[-6:]:
            for m in group:
                self.room.messages.append(m)
            self.room.messages.sort(key=lambda m: m.ts)
            self.re_read(t=float(group[-1].ts))

    # ------------------------------------------------------------------ #
    # Reading the room                                                   #
    # ------------------------------------------------------------------ #
    def raw_field(self) -> RoomField:
        """The terrain's projection: the dials' objective reading of the
        room's messages, before any charisma ripple."""
        return read_field(self.room, self.bank)

    def _pull(self, vec7: np.ndarray, actor) -> np.ndarray:
        """The entry ripple: the room bends toward the actor's vibe by
        their charisma — the field.py ``charisma_pull`` shape, scaled by
        RIPPLE_STRENGTH (an entrance is a fraction of a long presence)
        and clamped to the dial bounds."""
        n = self._interactions.get(actor.name, 0) + 1
        blend = (1.0 - math.exp(-actor.character.charisma * n))
        blend = min(blend * RIPPLE_STRENGTH, 1.0)
        vibe = np.asarray(actor.character.vibe, dtype=float)
        eff = vec7 + (vibe - vec7) * blend
        for i, name in enumerate(DIAL_NAMES):
            lo, hi = DIAL_BOUNDS.get(name, (0.0, 1.0))
            eff[i] = float(np.clip(eff[i], lo, hi))
        return eff

    def effective_field(self, actor=None) -> RoomField:
        """The field as felt: raw + the actor's entry ripple (if any).
        The ripple touches only the seven DIAL_NAMES; the remaining
        bank dials (model_vs_code, vision) ride along at their raw
        values — the record is the shared shadow, not a pure vector.
        """
        raw = self.raw_field().readings
        vec7 = np.array([raw.get(n, 0.0) for n in DIAL_NAMES], dtype=float)
        if actor is not None:
            vec7 = self._pull(vec7, actor)
        readings = dict(raw)
        readings.update(dict(zip(DIAL_NAMES, vec7.tolist())))
        return RoomField(readings)

    def re_read(self, actor=None, t: Optional[float] = None) -> Dict[str, float]:
        """Take a fresh reading and append it to the shared record.

        Every entry, act, and ambient event flows through here: the
        elephant re-reads, Z_in perceives, Z_out predicts. With an
        ``actor`` the reading includes their charisma ripple; the raw
        (pre-ripple) reading is recorded alongside, so the room's
        anomaly sense listens to the terrain, not to who walked in.
        """
        raw = self.raw_field().readings
        reading = self.effective_field(actor).readings
        if actor is not None:
            self._interactions[actor.name] = self._interactions.get(actor.name, 0) + 1
            self.last_actor = actor.name
        self.field_history.append(dict(reading))
        self.field_history_raw.append(dict(raw))
        self.field_ts.append(float(t if t is not None else len(self.field_history)))
        if len(self.field_history) > self.history_cap:
            self.field_history.pop(0)
            self.field_history_raw.pop(0)
            self.field_ts.pop(0)
        self.bridge.perceive()
        if len(self.bridge.zin) >= 2:
            self.bridge.predict()
        return dict(reading)

    def say(self, author: str, text: str, ts: Optional[float] = None,
            reactions: Optional[dict] = None) -> Message:
        """One message into the room's stream (the world acts). The
        default timestamp is after the room's latest message, so a say
        never sorts itself into the room's past."""
        if ts is None:
            ts = ((self.room.messages[-1].ts + 1.0) if self.room.messages
                  else 0.0)
        msg = Message(author=author, text=text, ts=float(ts),
                      reactions=dict(reactions or {}))
        self.room.messages.append(msg)
        self.room.messages.sort(key=lambda m: m.ts)
        return msg

    # ------------------------------------------------------------------ #
    # The room's sense of itself (Z_out)                                 #
    # ------------------------------------------------------------------ #
    def trend_dial(self) -> float:
        """Where the room is GOING: [-1 cooling .. +1 warming], the
        dual-db trend — the ghost-trail ahead."""
        return float(self.bridge.trend_dial())

    def anomaly(self) -> float:
        """How off the room is vs its own recent pattern: [0, 1].

        The elephant's anomaly sense, read from the RAW history (the
        terrain's projection — ripples of who walked in don't count;
        the room noticing ONE thing is wrong does): the largest
        per-dimension z-score of the latest raw reading vs the room's
        own recent record, mapped 1-exp(-z_max²/8). A panic spike, a
        mood collapse, a joke that changes everything — the single dial
        wildly off rings loud. (The dual-db bridge's multivariate
        Mahalanobis is the fleetmath original, but with short room
        histories it cannot see a one-dial spike — the typical radius
        sqrt(d) swallows it.)
        """
        if len(self.field_history_raw) < 3:
            return 0.0
        rows = [np.array([r[n] for n in DIAL_NAMES], dtype=float)
                for r in self.field_history_raw]
        X = np.stack(rows)
        mu = X[:-1].mean(axis=0)
        # The std floor is the dials' resolution: a still room that
        # moves by one dial notch is a real move, not max anomaly.
        sd = np.maximum(X[:-1].std(axis=0), 0.02)
        z = np.abs((X[-1] - mu) / sd)
        zmax = float(z.max())
        return float(1.0 - math.exp(-(zmax * zmax) / 8.0))

    # ------------------------------------------------------------------ #
    # The deadband check                                                 #
    # ------------------------------------------------------------------ #
    def deadband_check(self, turn: int) -> List[RingEvent]:
        """Does the room's field cross its deadband? Returns the rings
        (usually empty — the room breathes, no one is disturbed).

        Hysteresis per monitor: ring on the crossing from below
        (warmth_lo from above), then re-arm only after dipping back
        under the band; a cooldown stops the same ring repeating. A
        room that is ALREADY past its band on the first check rings
        once — the dungeon was screaming before you arrived.
        """
        rings: List[RingEvent] = []
        if not self.field_history_raw:
            return rings
        # The deadband listens to the TERRAIN — the raw field (the
        # room's truth), not the ripples of who walked in. The dungeon
        # does not re-arm because a calm person entered.
        reading = self.field_history_raw[-1]
        field = RoomField(reading)
        warmth = field.warmth()
        cd = max(0, int(self.deadband.cooldown))

        monitors: List[Tuple[str, float, str]] = []
        for dial_name, thr in self.deadband.dials.items():
            monitors.append((dial_name, float(thr), "gte"))
        if self.deadband.warmth_hi is not None:
            monitors.append(("warmth_hi", float(self.deadband.warmth_hi), "gte"))
        if self.deadband.warmth_lo is not None:
            monitors.append(("warmth_lo", float(self.deadband.warmth_lo), "lte"))
        if self.deadband.anomaly is not None:
            monitors.append(("anomaly", float(self.deadband.anomaly), "gte"))

        for key, thr, mode in monitors:
            if key == "warmth_hi" or key == "warmth_lo":
                v = warmth
                dial = "warmth"
            elif key == "anomaly":
                v = self.anomaly()
                dial = "anomaly"
            else:
                v = float(reading.get(key, 0.0))
                dial = key
            state = self._ring_state.setdefault(
                key, {"last": None, "above": False, "ring_turn": -99})
            crossed = v >= thr if mode == "gte" else v <= thr
            if crossed and not state["above"] and turn - state["ring_turn"] >= cd:
                severity = (v - thr) if mode == "gte" else (thr - v)
                rings.append(RingEvent(turn=turn, room=self.name, kind=key,
                                       dial=dial, value=v, prev=state["last"],
                                       severity=max(0.0, severity),
                                       player=self.last_actor))
                state["ring_turn"] = turn
            state["above"] = crossed
            state["last"] = v
            self._ring_state[key] = state

        # A trend inversion: the room's own prediction turned around.
        if self.deadband.trend_flip is not None and len(self.field_history) >= 2:
            t = self.trend_dial()
            prev = self._trend_state
            flip = float(self.deadband.trend_flip)
            if (prev is not None and abs(t) >= flip
                    and (prev > 0) != (t > 0) and abs(prev) >= flip):
                rings.append(RingEvent(turn=turn, room=self.name,
                                       kind="trend_invert", dial="trend_dial",
                                       value=t, prev=prev, severity=abs(t),
                                       player=self.last_actor))
            self._trend_state = t
        return rings

    # ------------------------------------------------------------------ #
    # The shadow                                                         #
    # ------------------------------------------------------------------ #
    def shadow(self, hour: Optional[float] = None) -> str:
        """The tinted description — the cave wall's rendering of the
        room. Deterministic for a given field; a changed field changes
        the words."""
        if hour is None:
            hour = self.hour
        return tint_description(self.effective_field(), self.description,
                                hour=hour)

    def sheet(self) -> dict:
        """The room's character sheet — the elephant's vital signs."""
        f = self.effective_field()
        return {
            "name": self.name,
            "warmth": round(f.warmth(), 3),
            "concentration": round(f.concentration(), 3),
            "readings": {k: round(v, 3) for k, v in f.readings.items()},
            "trend_dial": round(self.trend_dial(), 3),
            "anomaly": round(self.anomaly(), 3),
            "messages": len(self.room.messages),
            "history": len(self.field_history),
        }

    def __repr__(self) -> str:
        return (f"<RPGRoom {self.name!r} warmth={self.effective_field().warmth():+.2f} "
                f"history={len(self.field_history)}>")


# ===================================================================== #
# The world — rooms and the map between them                            #
# ===================================================================== #
class RPGWorld:
    """The dungeon: rooms + a simple graph of named edges + the clock."""

    def __init__(self, seed: int = 0):
        self.rooms: Dict[str, RPGRoom] = {}
        self.edges: List[Tuple[str, str, str]] = []
        self._adj: Dict[str, List[Tuple[str, str]]] = {}
        self.seed = int(seed)
        # The game clock starts after any room's seed history (the
        # world has been living before the players arrived). The big
        # base keeps direct player acts newer than seed messages even
        # without the engine; the engine re-bases it on the latest
        # seed timestamp anyway.
        self.clock = 1_000_000.0
        self.turn = 0
        self._action_ts: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Building                                                           #
    # ------------------------------------------------------------------ #
    def add_room(self, name: str, seed_messages: Optional[Iterable[Message]] = None,
                 description: str = "", bank: Optional[DialBank] = None,
                 deadband: Optional[Deadband] = None,
                 hour: Optional[float] = None) -> RPGRoom:
        """Build a room + its elephant. ``seed_messages`` is the room's
        recent history (its life before the players arrived)."""
        if name in self.rooms:
            raise ValueError(f"room {name!r} already exists")
        room = RPGRoom(name, description=description, seed_messages=seed_messages,
                       bank=bank, deadband=deadband, hour=hour)
        self.rooms[name] = room
        self._adj.setdefault(name, [])
        return room

    def add_edge(self, a: str, b: str, edge: Optional[str] = None) -> "RPGWorld":
        """A named passage between two rooms (undirected)."""
        for r in (a, b):
            if r not in self.rooms:
                raise KeyError(f"no room {r!r} — add it first")
        edge = edge or f"{a} → {b}"
        self.edges.append((a, b, edge))
        self._adj.setdefault(a, []).append((edge, b))
        self._adj.setdefault(b, []).append((edge, a))
        return self

    def neighbors(self, name: str) -> List[Tuple[str, str]]:
        """[(edge_name, room_name), ...] for the room."""
        return list(self._adj.get(name, []))

    def path(self, start: str, goal: str) -> List[str]:
        """Shortest room path [start, ..., goal] via BFS; [] if none."""
        if start == goal:
            return [start]
        prev: Dict[str, str] = {start: None}
        q: deque = deque([start])
        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for _edge, nb in self._adj.get(cur, []):
                if nb not in prev:
                    prev[nb] = cur
                    q.append(nb)
        if goal not in prev:
            return []
        path = [goal]
        while path[-1] != start:
            path.append(prev[path[-1]])
        return list(reversed(path))

    # ------------------------------------------------------------------ #
    # The shadows                                                        #
    # ------------------------------------------------------------------ #
    def describe(self, room: Union[str, RPGRoom]) -> str:
        """The tinted shadow of a room — the GM's description of it."""
        r = room if isinstance(room, RPGRoom) else self.rooms[room]
        return r.shadow()

    def foreshadow(self, room: Union[str, RPGRoom], player=None) -> str:
        """Z_out in words: what the room's own sense of itself says
        about what's coming. Filtered by the player's PersonalElephant
        — a brooder hears a different fog than a comedian."""
        r = room if isinstance(room, RPGRoom) else self.rooms[room]
        trend, anomaly = r.trend_dial(), r.anomaly()
        parts = []
        if anomaly >= 0.55:
            parts.append(f"{r.name} is ringing its own alarm — something there is not right")
        elif anomaly >= 0.35:
            parts.append(f"something at {r.name} is off — a wrongness its own senses keep reporting")
        if trend <= -0.25:
            parts.append(f"{r.name} is running cold — whatever waits there is getting worse")
        elif trend >= 0.25:
            parts.append(f"{r.name} is warming — whatever waits there is loosening")
        if not parts:
            parts.append(f"{r.name} keeps its own counsel — the dials there are quiet")
        line = "Ahead, the fog's own reading: " + "; ".join(parts) + "."
        if player is not None and hasattr(player, "character"):
            dial = player.character.top_dial()
            flavor = {
                "panic": "a tight, waiting hush",
                "cynicism": "an untrustworthy quiet",
                "joke_landing": "a room that could go either way",
                "mood": "a cold, heavy air",
                "presence": "a place that feels watched",
                "volume": "a silence with too much in it",
                "earnestness": "a room that means it",
            }.get(dial, "a wrongness with no name yet")
            line += f" {player.name}, who reads {dial} first, hears it as {flavor}."
        return line

    def next_ts(self) -> float:
        """The next action timestamp: one distinct moment per event or
        act within a turn (60s apart), so the room's density reads real
        pacing — a turn is a scene, not a single instant."""
        if self._action_ts is None:
            self._action_ts = self.clock
        else:
            self._action_ts += 60.0
        return float(self._action_ts)

    def __repr__(self) -> str:
        return f"<RPGWorld {len(self.rooms)} rooms, {len(self.edges)} edges>"


# ===================================================================== #
# The players — round characters as avatars                             #
# ===================================================================== #
class PersonalElephant:
    """A player's personal elephant: their vibe (native dial space), the
    dials they care about, their charisma, their acclimation skill.

    This is the round character — the same anatomy as a tapnight
    `Participant` (the Tap's regulars). When avatar.py lands, this is
    the seam it wraps.
    """

    def __init__(self, name: str, vibe: Optional[Dict[str, float]] = None,
                 dial_weights: Optional[Dict[str, float]] = None,
                 acclimation_rate: float = 0.25, charisma: float = 0.15,
                 title: str = ""):
        self.name = name
        self.title = title
        weights = dial_weights or {n: 1.0 / len(DIAL_NAMES) for n in DIAL_NAMES}
        self.participant = Participant(
            name, dial_weights=weights, acclimation_rate=acclimation_rate,
            charisma=charisma, vibe=vibe or {})

    @property
    def vibe(self) -> np.ndarray:
        return self.participant.vibe

    @property
    def dial_weights(self) -> np.ndarray:
        return self.participant.dial_weights

    @property
    def charisma(self) -> float:
        return self.participant.charisma

    @property
    def acclimation_rate(self) -> float:
        return self.participant.acclimation_rate

    def vibe_warmth(self) -> float:
        """The warmth of the character's native voice (RoomField.warmth
        on their vibe) — the temperature they carry into a room."""
        readings = dict(zip(DIAL_NAMES, self.vibe.tolist()))
        return RoomField(readings).warmth()

    def top_dial(self) -> str:
        """The dial this character always reads first."""
        w = list(zip(DIAL_NAMES, self.dial_weights.tolist()))
        return max(w, key=lambda t: t[1])[0]

    def top_dials(self, k: int = 3) -> List[str]:
        return [n for n, _ in sorted(zip(DIAL_NAMES, self.dial_weights.tolist()),
                                     key=lambda t: -t[1])[:k]]

    def personal_read(self, report: PerceptionReport,
                      noise_floor: float = DEFAULT_NOISE_FLOOR) -> str:
        """The same room, read through THIS character's weights — each
        prisoner sees a different shadow. The dial they care about most
        is the dial they notice moving first."""
        if report.n_readings < 2:
            return f"{self.name}'s ear is still warming to this room."
        scored = []
        for i, name in enumerate(DIAL_NAMES):
            dd = report.dial_deltas.get(name)
            if dd is not None:
                scored.append((abs(dd["direction"]) * float(self.dial_weights[i]),
                               name, dd["direction"]))
        scored.sort(key=lambda t: -t[0])
        if not scored or scored[0][0] <= max(noise_floor * 0.5, 1e-9):
            return (f"{self.name} feels nothing moving that matters "
                    f"to them — the dials they care about are still.")
        _w, name, d = scored[0]
        verb = "rising" if d > 0 else "falling"
        return (f"What {self.name} notices first: {name} {verb} — because "
                f"{name} is the dial {self.name} always reads first.")

    def sheet(self) -> dict:
        return {
            "vibe": {n: round(float(v), 3)
                     for n, v in zip(DIAL_NAMES, self.vibe.tolist())},
            "dial_weights": {n: round(float(w), 3)
                             for n, w in zip(DIAL_NAMES, self.dial_weights.tolist())},
            "charisma": round(self.charisma, 3),
            "acclimation_rate": round(self.acclimation_rate, 3),
            "vibe_warmth": round(self.vibe_warmth(), 3),
            "reads_first": self.top_dial(),
        }

    def __repr__(self) -> str:
        return (f"<PersonalElephant {self.name!r} warmth={self.vibe_warmth():+.2f} "
                f"charisma={self.charisma:.2f} reads_first={self.top_dial()}>")


# ---------------------------------------------------------------------- #
# Archetype presets — round-character templates with voices              #
# ---------------------------------------------------------------------- #
ARCHETYPES: Dict[str, dict] = {
    "comedian": {
        "title": "the comedian, who laughs at funerals and means it as a kindness",
        "vibe": {"mood": 0.60, "joke_landing": 0.65, "presence": 0.65,
                 "volume": 0.55, "earnestness": 0.50, "cynicism": 0.05,
                 "panic": 0.05},
        "dial_weights": {"joke_landing": 0.30, "mood": 0.25, "presence": 0.15,
                         "volume": 0.10, "earnestness": 0.10, "cynicism": 0.05,
                         "panic": 0.05},
        "charisma": 0.35, "acclimation_rate": 0.30,
        "lines": {
            "joke": [
                "So the fog walks into a bar, see — and the bar closes early. Joke's on the fog. 😂",
                "Fog's got opinions tonight, huh? I respect that in a weather system. 😂",
                "They say the fog's been hugging the harbor. I'd hug back, but I don't know where it's been. Heh.",
            ],
            "investigate": [
                "Marnie squints at {target} like it owes her money, then grins. \"Okay, so what's YOUR story?\"",
                "Marnie pokes {target} with one careful finger. \"Just checking it's not a metaphor. Correct?\"",
            ],
            "comfort": [
                "Marnie slides over and says, low: \"Hey. Whatever's out there, it's got to get through me first. And I'm VERY annoying.\"",
                "Marnie doesn't joke for a second, just stands close and says: \"You're not alone in this room. That's the whole trick.\"",
            ],
            "fight": [
                "Marnie grabs the wheel and yells into the fog: \"FIRE AWAY, then! Come on — show us your best!\" Then, quieter: \"And if that doesn't work — run.\"",
                "Marnie steps in front of the others, arms wide: \"You want a target? I'm RIGHT HERE — the loudest thing in this harbor! COME ON!\"",
            ],
            "wait": [
                "Marnie waits, tapping a rhythm on {target} that is, technically, a song.",
            ],
            "resolve": [
                "Marnie looks at {target} and says, suddenly quiet: \"Okay. I think I get it now. It wasn't angry. It was lost.\"",
            ],
            "banter": [
                "Marnie, to the room at large: \"See? This is why I never leave the house.\"",
                "Marnie: \"I'd make a joke, but the fog's already stolen my best material.\"",
            ],
        },
    },
    "brooder": {
        "title": "the brooder, who reads omens in the foam",
        "vibe": {"mood": -0.15, "joke_landing": -0.10, "presence": 0.15,
                 "volume": 0.20, "earnestness": 0.35, "cynicism": 0.70,
                 "panic": 0.30},
        "dial_weights": {"cynicism": 0.30, "panic": 0.20, "mood": 0.15,
                         "earnestness": 0.15, "presence": 0.10, "volume": 0.05,
                         "joke_landing": 0.05},
        "charisma": 0.18, "acclimation_rate": 0.20,
        "lines": {
            "joke": [
                "Ilsa's mouth does something that might be a smile. \"Heh. The fog's got a better sense of timing than this town.\"",
            ],
            "investigate": [
                "Ilsa kneels and studies {target} too long, then says: \"This place has been wrong since before we got here — dead quiet, cold, no one laughing, nothing right.\"",
                "Ilsa turns {target} over in her hands and goes very still. \"Someone's been waiting for us to find this.\"",
            ],
            "comfort": [
                "Ilsa says, without looking up: \"I don't do comfort. But I'll stand between you and whatever comes. That's the same thing, mostly.\"",
            ],
            "fight": [
                "Ilsa's voice drops to something flat and cold: \"You want through? Through me. I've been waiting for a fight that mattered — and I don't run.\"",
            ],
            "wait": [
                "Ilsa waits with her back to the wall, watching {target} like it might move.",
            ],
            "resolve": [
                "Ilsa reads the last line aloud, then blows out the lamp. \"The light was never for the living. Now it's out. Let's go home.\"",
            ],
            "banter": [
                "Ilsa: \"Don't look at me. I told you the fog wasn't weather.\"",
                "Ilsa, quietly: \"The jokes stop mattering when the thing in the water learns your name.\"",
            ],
        },
    },
    "wallflower": {
        "title": "the wallflower, who noticed the fog first and told no one",
        "vibe": {"mood": 0.0, "joke_landing": 0.0, "presence": 0.10,
                 "volume": 0.10, "earnestness": 0.60, "cynicism": 0.30,
                 "panic": 0.15},
        "dial_weights": {"earnestness": 0.30, "presence": 0.20, "mood": 0.15,
                         "panic": 0.15, "cynicism": 0.10, "volume": 0.05,
                         "joke_landing": 0.05},
        "charisma": 0.08, "acclimation_rate": 0.15,
        "lines": {
            "joke": [
                "Theo, almost to himself: \"If the fog wanted in, it would've knocked by now.\" He's not sure if it's a joke.",
            ],
            "investigate": [
                "Theo looks at {target} longer than anyone, then says one word: \"Warm.\"",
                "Theo touches {target} like it might bite, and reports, quietly: \"It's been moved. Recently.\"",
            ],
            "comfort": [
                "Theo brings the old woman a fresh pint and says, quiet: \"The hearth's warm. The company's good and kind. The fog can wait its turn — this is home, and the house drinks together.\"",
            ],
            "fight": [
                "Theo steps forward once, small and final: \"You'll have to go through me too. I'm harder to see — and I won't run.\"",
            ],
            "wait": [
                "Theo stands very still and listens to {target}, the way you'd listen to a heartbeat.",
            ],
            "resolve": [
                "Theo says one true thing: \"The fog was homesick. Now it can go home.\"",
            ],
            "banter": [
                "Theo, from somewhere near the wall: \"I saw it first. I just... didn't know what to do with seeing it.\"",
                "Theo: \"Someone has to be the one who noticed. I'm okay if it's me.\"",
            ],
        },
    },
    "traveler": {
        "title": "a traveler with no name yet",
        "vibe": {"mood": 0.2, "joke_landing": 0.1, "presence": 0.4,
                 "volume": 0.4, "earnestness": 0.5, "cynicism": 0.3,
                 "panic": 0.2},
        "dial_weights": {n: 1.0 / len(DIAL_NAMES) for n in DIAL_NAMES},
        "charisma": 0.2, "acclimation_rate": 0.25,
        "lines": {
            "joke": ["{name} tries a joke. It lands somewhere between brave and unwise."],
            "investigate": ["{name} studies the {target} carefully."],
            "comfort": ["{name} says something steady and kind."],
            "fight": ["{name} squares up — whatever comes, it comes through them."],
            "wait": ["{name} waits, watchful."],
            "resolve": ["{name} acts — the thing is done."],
            "banter": ["{name} has nothing to add, and says so."],
        },
    },
}


class RPGPlayer:
    """A player character: an avatar with a PersonalElephant and a pulse.

    - ``enter(room)`` — move + the perception roll (direction/rate of
      the room's recent history, including the player's own ripple).
    - ``act(verb, target)`` — the player acts; the room reacts.
    - ``perceive()`` — the pulse monologue of what it feels.
    """

    def __init__(self, name: str, archetype: str = "traveler",
                 world: Optional[RPGWorld] = None,
                 start: Optional[str] = None, goal: str = "",
                 elephant: Optional[PersonalElephant] = None,
                 avatar=None, period: float = 1.0, seed: int = 0):
        self.name = name
        self.archetype = archetype
        self.world = world
        self.start = start
        self.goal = goal
        self.goal_room: Optional[str] = None
        self.position: Optional[str] = start
        preset = ARCHETYPES.get(archetype, ARCHETYPES["traveler"])
        self.title = preset["title"]
        # The avatar seam: a learned round character (elephant/avatar.py)
        # speaks and monologues for the player; the game mechanics
        # (ripple, personal read) run on a PersonalElephant built from
        # the avatar's own vibe and dial weights. Without an avatar,
        # the archetype preset IS the flat character.
        self.avatar = avatar
        if avatar is not None:
            vibe = getattr(avatar, "_vibe", None)
            vibe_dict = None
            if vibe is not None:
                arr = np.asarray(vibe, dtype=float).reshape(-1)
                vibe_dict = {n: float(v) for n, v in zip(DIAL_NAMES, arr)}
            weights = dict(getattr(avatar, "dial_weights", None) or {}) or None
            elephant = PersonalElephant(
                name, vibe=vibe_dict, dial_weights=weights,
                acclimation_rate=0.2, charisma=0.2, title=self.title)
            self.title = (f"{avatar.persona.split('.')[0].strip()} — "
                          f"{preset['title']}")
        self.character = elephant or PersonalElephant(
            name, vibe=preset.get("vibe"), dial_weights=preset.get("dial_weights"),
            acclimation_rate=preset.get("acclimation_rate", 0.25),
            charisma=preset.get("charisma", 0.15), title=self.title)
        self.pulse = PulseLoop(name, room=Room(f"{name}"), period=period,
                               history=HISTORY_CAP)
        self.seed = int(seed)
        self._line_count: Dict[str, int] = {}
        self.acts: List[Tuple[int, str, str, str]] = []
        self.rings_witnessed: List[RingEvent] = []
        self.last_report: Optional[PerceptionReport] = None
        self.entry_note: str = ""
        self.pulses: int = 0

    # ------------------------------------------------------------------ #
    # The perception roll                                                #
    # ------------------------------------------------------------------ #
    def _check(self, room: RPGRoom, entry: bool = False) -> PerceptionReport:
        """The perception check over the room's recent history, and the
        player's pulse synced to it (so the monologue is grounded in
        what they just perceived). On entry the roll reads the room's
        movement up to the moment of arrival (``exclude_last``) — the
        room's own story — and the player's presence is reported
        separately as the bend."""
        report = perception_check(room, self.name,
                                  noise_floor=self.pulse.noise_floor,
                                  exclude_last=entry)
        # Prime the pulse with the room's record so the monologue reads
        # the room's whole recent history, then tick it once.
        self.pulse.room = room
        self.pulse._readings = [dict(r) for r in room.field_history]
        self.pulse._ts = list(room.field_ts)
        self.pulse._last_ts = room.field_ts[-1] if room.field_ts else None
        self.pulse._n_msgs = len(room.room.messages)
        self.pulse._last_report = self.pulse.perception_check(
            traffic=0, agent_said=False)
        self.last_report = report
        self.pulses += 1
        return report

    def _entry_bend(self, room: RPGRoom) -> str:
        """What the player's presence did to the room on arrival — the
        shadow moved because they moved the fire."""
        if len(room.field_history) < 2:
            return ""
        bend = (RoomField(room.field_history[-1]).warmth()
                - RoomField(room.field_history[-2]).warmth())
        floor = self.pulse.noise_floor
        if bend > floor:
            return (f"Your presence bends the room — it warms where "
                    f"you stand ({bend:+.2f}).")
        if bend < -floor:
            return (f"Your presence bends the room — it cools where "
                    f"you stand ({bend:+.2f}).")
        return "The room does not shift for you."

    def enter(self, room_name: str) -> PerceptionReport:
        """Walk into a room: the room re-reads with your ripple, then
        the perception roll reads the room's direction/rate over its
        recent history up to your arrival — its own story. Your effect
        on it is reported separately (``entry_note``)."""
        if self.world is None:
            raise RuntimeError("player is not attached to a world")
        if room_name not in self.world.rooms:
            raise KeyError(f"no room {room_name!r} in the world")
        self.position = room_name
        room = self.world.rooms[room_name]
        room.re_read(actor=self, t=self.world.next_ts())   # the entry ripple
        report = self._check(room, entry=True)
        self.entry_note = self._entry_bend(room)
        return report

    def read_room(self) -> PerceptionReport:
        """The perception roll on the current room without moving (the
        room as it stands after the last turn's events and acts)."""
        room = self.world.rooms[self.position]
        return self._check(room)

    def perceive(self) -> str:
        """The pulse monologue of what it feels — the silent thinking.
        (The personal read — what THIS character notices first — is
        printed separately by the engine alongside the roll.) With an
        avatar, the round character's own monologue speaks."""
        if self.avatar is not None:
            room = self.world.rooms[self.position] if self.world else None
            return self.avatar.monologue(room=room)
        if self.last_report is None:
            self.read_room()
        return self.pulse.internal_monologue()

    # ------------------------------------------------------------------ #
    # Acting                                                             #
    # ------------------------------------------------------------------ #
    def _voice(self, verb: str, target: Optional[str]) -> str:
        """The line for an act, round-robin over the archetype bank
        (deterministic) — or, with an avatar, the round character's own
        composed voice, given the act as context."""
        if self.avatar is not None:
            room = self.world.rooms[self.position].name if self.world else ""
            ctx = (f"the party {verb} toward {target or 'the room'}"
                   + (f" in {room}" if room else ""))
            return self.avatar.speak(prompt_context=ctx)
        preset = ARCHETYPES.get(self.archetype, ARCHETYPES["traveler"])
        bank = preset["lines"].get(verb) or preset["lines"].get("wait") \
            or ARCHETYPES["traveler"]["lines"]["wait"]
        i = self._line_count.get(verb, 0)
        self._line_count[verb] = i + 1
        tpl = bank[i % len(bank)]
        room = self.world.rooms[self.position].name if self.world else ""
        return tpl.format(name=self.name, room=room, target=target or "room")

    def act(self, verb: str, target: Optional[str] = None) -> str:
        """Act: the room ingests the line, the elephant re-reads the
        terrain (the words themselves move the dials — a fight spikes
        panic, a joke spikes joke_landing). The entrance ripple is for
        ``enter`` only; an act is the player reaching INTO the room.
        Returns the spoken line."""
        room = self.world.rooms[self.position]
        if verb in ("move", "enter", "go"):
            report = self.enter(target)
            line = f"{self.name} heads to {target}."
            self.acts.append((self.world.turn, verb, target or "", line))
            return line
        line = self._voice(verb, target)
        reactions = {}
        if verb == "joke":
            # The crowd's hands — the room's judgment of the joke,
            # read from the room's felt state (the shared record,
            # ripples included) as the joke lands.
            pre = RoomField(room.field_history[-1]).warmth() if room.field_history else 0.0
            reactions = {"😂": 1} if pre >= 0.05 else {"🙄": 1}
        room.say(self.name, line, ts=self.world.next_ts(), reactions=reactions)
        room.re_read(t=self.world.next_ts())   # the terrain re-reads — no ripple
        self.acts.append((self.world.turn, verb, target or "", line))
        return line

    def banter_line(self) -> str:
        """One archetype-voiced aside — the banter loop after a ring."""
        return self._voice("banter", None)

    # ------------------------------------------------------------------ #
    # Autonomy                                                           #
    # ------------------------------------------------------------------ #
    def decide(self) -> Tuple[str, Optional[str]]:
        """Goal-driven fallback: move toward the goal room, investigate
        when there, wait when done. Used when the script has no entry
        for this turn."""
        if self.position is None or self.goal_room is None:
            return ("wait", None)
        if self.position == self.goal_room:
            return ("investigate", None)
        path = self.world.path(self.position, self.goal_room)
        if len(path) >= 2:
            return ("move", path[1])
        return ("wait", None)

    # ------------------------------------------------------------------ #
    # The character sheet                                                #
    # ------------------------------------------------------------------ #
    def character_sheet(self) -> dict:
        sheet = {
            "name": self.name,
            "title": self.title,
            "archetype": self.archetype,
            "position": self.position,
            "goal": self.goal,
            "elephant": self.character.sheet(),
            "pulses": self.pulses,
            "acts": [(t, v, tg) for t, v, tg, _ in self.acts],
            "rings_witnessed": [r.marker() for r in self.rings_witnessed],
            "last_perception": (report_words(self.last_report) if self.last_report
                                else "no roll yet"),
        }
        if self.avatar is not None:
            sheet["avatar"] = {
                "through_line": self.avatar.through_line,
                "persona": self.avatar.persona,
                "nights_at_the_tap": len(self.avatar.nights),
            }
        return sheet

    def __repr__(self) -> str:
        return (f"<RPGPlayer {self.name!r} ({self.archetype}) at "
                f"{self.position!r}>")


# ===================================================================== #
# The perception check — shared                                          #
# ===================================================================== #
def perception_check(room: RPGRoom, agent_id: str,
                     noise_floor: float = DEFAULT_NOISE_FLOOR,
                     exclude_last: bool = False) -> PerceptionReport:
    """The RPG perception roll: the room's direction/rate of change
    over its recent history (the pulse's perception-check math over the
    room's shared record instead of one agent's pulse).

    With ``exclude_last=True`` the roll reads the room's movement up
    to the moment the player arrived — the room's own story before
    their ripple — used for the entry roll (the player's presence is
    then reported separately as the bend).
    """
    history = room.field_history
    ts = room.field_ts
    if exclude_last and len(history) >= 2:
        history = history[:-1]
        ts = ts[:-1]
    warm = [RoomField(d).warmth() for d in history]
    warm_dir = direction(warm, noise_floor=noise_floor).get(0, 0.0)
    warm_rate = rate_of_change(warm, noise_floor=noise_floor).get(0, 0.0)
    dir_pp = direction(history, noise_floor=noise_floor)
    rate_pp = rate_of_change(history, noise_floor=noise_floor)
    names = room.bank.names()
    dial_deltas = {n: {"direction": dir_pp.get(n, 0.0),
                       "rate": rate_pp.get(n, 0.0)} for n in names}
    report = PerceptionReport(
        agent_id=agent_id,
        ts=(ts[-1] if ts else 0.0),
        n_readings=len(history),
        warmth=(warm[-1] if warm else 0.0),
        warmth_direction=warm_dir,
        warmth_rate=warm_rate,
        direction=dir_pp,
        rate_of_change=rate_pp,
        dial_deltas=dial_deltas,
        traffic=0,
        agent_said=False,
        whole_hand="",
    )
    report.whole_hand = compose_whole_hand(report, noise_floor)
    return report


def report_words(report: PerceptionReport,
                 noise_floor: float = DEFAULT_NOISE_FLOOR) -> str:
    """The perception report as a shadow — words, never vectors. The
    number doesn't matter; two numbers show direction, three show rate
    of change."""
    n = report.n_readings
    if n < 2:
        return ("only one reading in — the room hasn't moved enough "
                "to feel a hand yet.")
    wd, wr = report.warmth_direction, report.warmth_rate
    if wd > noise_floor:
        head = "the room is warming"
    elif wd < -noise_floor:
        head = "the room is cooling"
    else:
        head = "the room is holding"
    if wr > noise_floor:
        pace = "the movement is building"
    elif wr < -noise_floor:
        pace = "the movement is easing"
    else:
        pace = "the movement is steady"
    movers = [(name, dd["direction"])
              for name, dd in report.dial_deltas.items()
              if abs(dd["direction"]) > noise_floor]
    movers.sort(key=lambda t: -abs(t[1]))
    if movers:
        name, d = movers[0]
        verb = "rising" if d > 0 else "falling"
        mover_txt = f"{name} is the loudest hand, {verb} {abs(d):.2f} per beat"
    else:
        mover_txt = "no dial is moving enough to matter"
    return f"{head}, and {pace} — {mover_txt}."


# ===================================================================== #
# The GM — the prompt bank, keyed to the ring                           #
# ===================================================================== #
GM_BANK: Dict[str, str] = {
    "panic": "The room screams — {dial} crosses the band in {room}{who}. Whatever is coming does not knock.",
    "warmth_hi": "{room} crests its band — the warmth peaks and the whole room leans in at once{who}.",
    "warmth_lo": "{room} goes cold — the warmth drains past the band and the room closes like a fist{who}.",
    "anomaly": "{room} is wrong. Its own senses are ringing off — {value:.2f} on the off-meter. The fog does not behave, and the room knows it.",
    "trend_invert": "The room's own prediction turns in {room} — it was going one way, and now it is going the other{who}.",
    "dial": "{room} shifts — {dial} crosses the band{who} ({severity:+.2f} past the mark).",
    "plot": "The dungeon answers. The plot tightens a notch.",
}


# ===================================================================== #
# The log and the engine                                                #
# ===================================================================== #
@dataclass
class RPGLog:
    """The session transcript — the shadows of the game, in order."""
    name: str = ""
    lines: List[str] = dc_field(default_factory=list)
    rings: List[RingEvent] = dc_field(default_factory=list)
    turns: int = 0
    plot_stage: int = 0
    goal_reached: bool = False
    world: Optional[RPGWorld] = None
    players: List[RPGPlayer] = dc_field(default_factory=list)

    def __str__(self) -> str:
        return "\n".join(self.lines)


class RPGEngine:
    """The game loop: perceive → act → the room re-reads → the deadband
    checks → rings advance the plot → Z_out foreshadows the next room.

    ``run(scenario, max_turns)`` — a scenario is data: rooms, players,
    a goal, a script of acts, ambient events, and plot lines. The
    narration is ALWAYS a shadow (the elephant's tint + the perception
    reports in words), never the terrain itself.
    """

    TURN_STEP = TURN_STEP
    MAX_RINGS_PER_TURN = 3   # at most three rings narrate per turn

    def __init__(self, world: Optional[RPGWorld] = None,
                 players: Optional[Sequence[RPGPlayer]] = None,
                 goal: str = "", goal_room: Optional[str] = None,
                 gm_bank: Optional[Dict[str, str]] = None,
                 seed: int = 0, noise_floor: float = 0.012,
                 banter: bool = True):
        self.world = world or RPGWorld(seed=seed)
        self.players = list(players or [])
        self.goal = goal
        self.goal_room = goal_room
        self.gm_bank = dict(gm_bank or GM_BANK)
        self.seed = int(seed)
        self.noise_floor = float(noise_floor)
        self.banter = bool(banter)

        self.plot_lines: List[str] = []
        self.script: List[Tuple[int, str, str, Optional[str]]] = []
        self.events: Dict[int, List[dict]] = {}
        self.premise: str = ""
        self.name: str = ""
        self.max_turns: int = 8

        self.plot_stage = 0
        self._resolved = False

    # ------------------------------------------------------------------ #
    # Scenario building (scenarios as data)                              #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_scenario(cls, data: dict, seed: int = 0) -> "RPGEngine":
        """Build an engine from a scenario dict:

        ``rooms``: name, description, hour, deadband, seed_messages
          [(author, text, ts) or (author, text, ts, reactions)]
        ``edges``: [(a, b, edge_name)]
        ``players``: name, archetype, start, goal
        ``goal``, ``goal_room``, ``premise``, ``name``, ``max_turns``
        ``script``: [(turn, player, verb, target)]
        ``events``: {turn: [{"room": ..., "text": ...}]}
        ``plot_lines``: [GM beats advancing the plot]
        ``banter``: bool
        """
        engine = cls(seed=seed)
        engine.name = data.get("name", "The Dungeon")
        engine.premise = data.get("premise", "")
        engine.goal = data.get("goal", "")
        engine.goal_room = data.get("goal_room")
        engine.plot_lines = list(data.get("plot_lines", []))
        engine.max_turns = int(data.get("max_turns", 8))
        engine.banter = bool(data.get("banter", True))
        world = engine.world

        for r in data.get("rooms", []):
            seeds = []
            for sm in r.get("seed_messages", []):
                author, text, ts = sm[0], sm[1], float(sm[2])
                rx = sm[3] if len(sm) > 3 else None
                seeds.append(Message(author=author, text=text, ts=ts,
                                     reactions=dict(rx or {})))
            db = r.get("deadband", {}) or {}
            deadband = Deadband(
                dials={k: float(v) for k, v in db.items()
                       if k not in ("warmth_hi", "warmth_lo", "anomaly",
                                    "trend_flip", "cooldown")},
                warmth_hi=db.get("warmth_hi"),
                warmth_lo=db.get("warmth_lo"),
                anomaly=db.get("anomaly"),
                trend_flip=db.get("trend_flip", 0.15),
                cooldown=int(db.get("cooldown", 2)),
            )
            world.add_room(
                r["name"], seed_messages=seeds,
                description=r.get("description", ""),
                deadband=deadband,
                hour=r.get("hour"))
        for a, b, *rest in data.get("edges", []):
            world.add_edge(a, b, rest[0] if rest else None)

        for p in data.get("players", []):
            player = RPGPlayer(
                p["name"], archetype=p.get("archetype", "traveler"),
                world=world, start=p.get("start"), goal=p.get("goal", ""),
                seed=seed)
            player.goal_room = engine.goal_room
            engine.players.append(player)

        for entry in data.get("script", []):
            engine.script.append((int(entry[0]), entry[1], entry[2],
                                  entry[3] if len(entry) > 3 else None))
        engine.events = {int(k): list(v) for k, v in data.get("events", {}).items()}

        # The clock starts after the rooms' own histories (the world has
        # been living before the players arrived).
        latest = 0.0
        for room in world.rooms.values():
            if room.field_ts:
                latest = max(latest, max(room.field_ts))
        world.clock = latest + engine.TURN_STEP
        return engine

    # ------------------------------------------------------------------ #
    # The loop                                                           #
    # ------------------------------------------------------------------ #
    def run(self, max_turns: Optional[int] = None) -> RPGLog:
        self.plot_stage = 0            # a run is a fresh session
        self._resolved = False
        max_turns = self.max_turns if max_turns is None else int(max_turns)
        log = RPGLog(name=self.name, world=self.world, players=self.players)
        w = self.world

        log.lines.append(f"--- {self.name} ---")
        if self.premise:
            log.lines.append(self.premise)
        log.lines.append(f"Goal: {self.goal}")
        log.lines.append("")

        for turn in range(1, max_turns + 1):
            w.turn = turn
            log.turns = turn
            w._action_ts = w.clock   # a turn is a scene: one timeline
            self._turn_rings = []

            # 1. The world breathes: ambient events (the terrain moves
            #    even where no player is looking).
            for ev in self.events.get(turn, []):
                room = w.rooms[ev["room"]]
                room.say("the night", ev["text"], ts=w.next_ts(),
                         reactions=ev.get("reactions"))
                room.re_read(t=w.next_ts())
                room.last_actor = ""   # the night is not a player
                log.lines.append(f"  The night: {ev['text']}")
                self._check_room(room, turn, log)

            # 2. The players perceive and act.
            for player in self.players:
                if player.position is None:
                    player.position = player.start
                if player.position is None:
                    raise ValueError(
                        f"player {player.name!r} has no start room — set "
                        f"'start' in the scenario's players")
                verb, target = self._action_for(player, turn)
                if verb in ("move", "enter", "go"):
                    report = player.enter(target)
                    log.lines.append(f"{player.name} enters {target}.")
                    log.lines.append(f"  {w.describe(target)}")
                    log.lines.append(f"  {player.name}'s read: {report_words(report, self.noise_floor)}")
                    log.lines.append(f"  {player.character.personal_read(report, self.noise_floor)}")
                    if player.entry_note:
                        log.lines.append(f"  {player.entry_note}")
                    log.lines.append(f"  {w.foreshadow(target, player)}")
                    log.lines.append(f"  {player.name} thinks: {player.perceive()}")
                    self._check_room(w.rooms[target], turn, log)
                else:
                    report = player.read_room()
                    log.lines.append(f"{player.name} looks around — {report_words(report, self.noise_floor)}")
                    log.lines.append(f"  {player.character.personal_read(report, self.noise_floor)}")
                    line = player.act(verb, target)
                    log.lines.append(f"{player.name}: “{line}”")
                    player.read_room()   # the room after the act — fresh pulse
                    log.lines.append(f"  {player.name} thinks: {player.perceive()}")
                    self._check_room(w.rooms[player.position], turn, log)

            # 3. The world breathes again: every room's deadband gets
            #    its say (rooms ring even when no player is looking).
            for room in w.rooms.values():
                self._check_room(room, turn, log)

            # 4. Banter — the party's voices bounce off the ring.
            if self._turn_rings and self.banter:
                for player in self.players:
                    log.lines.append(f"  {player.name}: “{player.banter_line()}”")

            # 5. Goal check.
            if self._goal_met():
                log.goal_reached = True
                log.lines.append("")
                log.lines.append(f"GM: {self.goal} — {self._epilogue()}")
                log.lines.append("The fog, out on the sound, begins to lift.")
                break

            w.clock += self.TURN_STEP
        else:
            log.lines.append("")
            log.lines.append("The night ends; the fog keeps what it keeps.")

        log.plot_stage = self.plot_stage
        return log

    def _check_room(self, room: RPGRoom, turn: int, log: RPGLog) -> None:
        """The deadband decides: every ring is narrated (up to the turn's
        cap), advances the plot, and draws a GM line keyed to the ring.
        """
        for ring in room.deadband_check(turn):
            if len(self._turn_rings) >= self.MAX_RINGS_PER_TURN:
                break
            self._turn_rings.append(ring)
            log.rings.append(ring)
            for p in self.players:
                if p.position == ring.room:
                    p.rings_witnessed.append(ring)
            log.lines.append(f"  {ring.marker()}")
            log.lines.append(f"  GM: {self._ring_line(ring)}")
            self.plot_stage += 1
            if self.plot_lines and self.plot_stage <= len(self.plot_lines):
                log.lines.append(f"  GM: {self.plot_lines[self.plot_stage - 1]}")
            elif self.plot_lines:
                tpl = self.gm_bank.get("plot", GM_BANK["plot"])
                log.lines.append(f"  GM: {tpl.format(room=ring.room, dial=ring.dial)}")

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _action_for(self, player: RPGPlayer,
                    turn: int) -> Tuple[str, Optional[str]]:
        for t, name, verb, target in self.script:
            if t == turn and name == player.name:
                return verb, target
        return player.decide()

    def _ring_line(self, ring: RingEvent) -> str:
        tpl = self.gm_bank.get(ring.kind) or self.gm_bank.get("dial", GM_BANK["dial"])
        who = f", and it is {ring.player} who pushes it over" if ring.player else ""
        try:
            return tpl.format(room=ring.room, dial=ring.dial, severity=ring.severity,
                              value=ring.value, who=who)
        except (KeyError, IndexError, ValueError):
            return GM_BANK["plot"].format(room=ring.room)

    def _goal_met(self) -> bool:
        if self.goal_room is None or self.plot_stage < 1:
            return False
        for player in self.players:
            for t, verb, target, _line in player.acts:
                if (verb == "resolve" and player.position == self.goal_room):
                    self._resolved = True
        return self._resolved

    def _epilogue(self) -> str:
        resolved_by = [p.name for p in self.players
                       if any(v == "resolve" for _t, v, _g in
                              [(a[0], a[1], a[2]) for a in p.acts])]
        names = " and ".join(resolved_by) if resolved_by else "the party"
        return (f"The source of the fog was a light left burning for a ship "
                f"that never came home — and {names} blew it out.")


def run_scenario(data: dict, max_turns: Optional[int] = None,
                 seed: int = 0) -> RPGLog:
    """One call: build the engine from scenario data, run it, return
    the session transcript."""
    engine = RPGEngine.from_scenario(data, seed=seed)
    return engine.run(max_turns=max_turns)
