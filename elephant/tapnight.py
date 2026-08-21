"""TapNightSession — the elephant at The Tap.

The Tap is where the crew gathers after work to read each other's creative
works and hear reactions — to their own piece and to others'. First-person,
not center of attention. This module is the session engine that builds the
elephant INTO the Tap:

- a `TapNightSession` ingests participant messages (the works read aloud) into
  a `Room`, runs the dial bank, and produces the room's field (warmth, κ, and
  all 7 dial readings);
- each `Participant` carries personal settings that EVOLVE over cycles — the
  taste that differs between guitarists. One guitar looks pretty, another
  sounds wonderful, another has a good neck: you don't know where the settings
  belong until different agents desire different settings and self-fine-tune
  to the moment they're in.

Reading the room is a relationship to the room, not a readout:

- *Within a session*, each participant's live `vibe` relaxes toward the room
  field at their `acclimation_rate` (they warm to the room), while their
  `charisma` pulls the room field toward their vibe over their interactions
  (the room warms to them).
- *Across cycles*, each participant's `dial_weights` self-fine-tune toward the
  dials where their *felt engagement* was highest. The engagement signal is
  anchored to the participant's own desire (`vibe`) measured against the cast's
  average desire (peer-relative, so it's robust to the v0 dials' tendency to
  saturate), amplified by the crowd's hands on the dial each reaction
  expresses, and selected by a ReLU-normalized target (so weight only ever
  moves toward the dials a participant is genuinely distinctive on — tastes
  diverge into multiple stable attractors instead of collapsing to the room's
  loudest dial).

numpy-only. Mirrors the `BoatHarness` pattern (rolling room + dial bank +
field), but for people reading each other's work.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass, field as dc_field
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from elephant.dial import DialBank
from elephant.dials import DEFAULT_DIALS
from elephant.field import DIAL_NAMES, RoomField, read_field
from elephant.room import Message, Room
from elephant.vmf import A7 as vmf_A7, edge as vmf_edge, vmf_fit, \
    record_with as vmf_record_with, windowed as vmf_windowed

# Dial value ranges: the room field is clamped to these (charisma saturates,
# never overshoots). mood & joke_landing are signed [-1,+1]; the rest run [0,1].
DIAL_BOUNDS: Dict[str, tuple] = {
    "mood": (-1.0, 1.0),
    "volume": (0.0, 1.0),
    "earnestness": (0.0, 1.0),
    "cynicism": (0.0, 1.0),
    "joke_landing": (-1.0, 1.0),
    "panic": (0.0, 1.0),
    "presence": (0.0, 1.0),
}
# Neutral (quiescent) value per dial. The dials rest at different values:
# signed dials (mood, joke_landing) and the "off" dials (volume = quiet,
# cynicism = not sneering, panic = calm) rest at 0; earnestness/presence rest
# at 0.5 (no signal). Used to default a participant's un-stated vibe dims and
# to rank which dials are prominent in a room.
DIAL_CENTER: Dict[str, float] = {
    "mood": 0.0,
    "volume": 0.0,
    "earnestness": 0.5,
    "cynicism": 0.0,
    "joke_landing": 0.0,
    "panic": 0.0,
    "presence": 0.5,
}


def _to_vector(x, default=0.5) -> np.ndarray:
    """Coerce a dial-space value into a 7-vector in DIAL_NAMES order.

    Accepts a dict {dial_name: value} (missing keys -> `default`, which may be
    a scalar or a per-dial dict), or any array-like of 7 floats.
    """
    if isinstance(x, dict):
        if isinstance(default, dict):
            return np.array([float(x.get(n, default[n])) for n in DIAL_NAMES],
                            dtype=float)
        return np.array([float(x.get(n, default)) for n in DIAL_NAMES],
                        dtype=float)
    return np.asarray(list(x), dtype=float)


# Reaction emoji -> the dial they express (the crowd's hands, attributed to
# the dimension the reaction signals).
REACTION_TO_DIAL: Dict[str, str] = {
    "😂": "joke_landing", "🤣": "joke_landing", "😄": "joke_landing",
    "💀": "joke_landing",
    "❤️": "mood",
    "👍": "earnestness",
    "👏": "presence",
    "🙄": "cynicism", "😏": "cynicism", "😒": "cynicism",
    "🤨": "cynicism", "👎": "cynicism",
}


@dataclass
class Participant:
    """One agent's personal settings — the taste that differs between
    guitarists. These are the persistent knobs that evolve over many Tap
    nights; `vibe` is the agent's native style in dial space (their "home"
    voice), while `dial_weights` is their prior over which dimensions matter.
    """

    name: str
    dial_weights: Union[Dict[str, float], Sequence[float]]  # -> 7-vector, sums to 1
    acclimation_rate: float = 0.25          # modulation skill (how fast they warm)
    charisma: float = 0.15                  # their pull on the room
    vibe: Union[Dict[str, float], Sequence[float]] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dial_weights = _to_vector(self.dial_weights, default=0.0)
        self.dial_weights = np.maximum(self.dial_weights, 0.0)
        s = float(self.dial_weights.sum())
        self.dial_weights = (self.dial_weights / s) if s > 1e-9 else np.full(
            len(DIAL_NAMES), 1.0 / len(DIAL_NAMES))
        # Un-stated vibe dims rest at the dial's neutral value, not 0.5, so a
        # participant who doesn't mention volume/panic isn't spuriously read as
        # caring about them.
        self.vibe = _to_vector(self.vibe, default=DIAL_CENTER)
        self.acclimation_rate = float(self.acclimation_rate)
        self.charisma = float(self.charisma)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dial_weights": self.dial_weights.tolist(),
            "acclimation_rate": self.acclimation_rate,
            "charisma": self.charisma,
            "vibe": self.vibe.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Participant":
        return cls(name=d["name"], dial_weights=d["dial_weights"],
                   acclimation_rate=float(d["acclimation_rate"]),
                   charisma=float(d["charisma"]), vibe=d["vibe"])

    def __repr__(self) -> str:
        return (f"<Participant {self.name!r} accl={self.acclimation_rate:.2f} "
                f"charisma={self.charisma:.2f}>")


class TapNightSession:
    """One after-work gathering. Ingests works, reads the room, lets the
    crew warm to it and pull it toward them, and self-tunes across cycles.
    """

    STEP = 60.0  # seconds of auto-timestamp per spoken line

    def __init__(self, name: str = "The Tap",
                 participants: Optional[Iterable[Participant]] = None,
                 bank: Optional[DialBank] = None,
                 log_path: Optional[str] = None,
                 identity: str = "elephant-v0",
                 W: int = 8,
                 reader_schema: int = 1,
                 staged_entries: Optional[Dict[str, "Participant"]] = None):
        self.name = name
        self.participants: Dict[str, Participant] = {}
        for p in (participants or []):
            self.participants[p.name] = p
        self.bank = bank or DialBank(DEFAULT_DIALS)

        # The room (messages) and the effective field (raw + charisma).
        self.room = Room(f"{name}")
        self.field: np.ndarray = np.zeros(len(DIAL_NAMES))

        # Per-participant runtime state (reset each session).
        self._vibe: Dict[str, np.ndarray] = {}
        self._vibe_start: Dict[str, np.ndarray] = {}
        self._interactions: Dict[str, int] = {}
        self._reaction_heat: Dict[str, int] = {}

        self._clock = 0.0
        self.cycle = 0
        self._log: List[str] = []

        # --- vMF edge log (spec §3, additive) ---
        self.W = int(W)
        self._log_path = log_path
        self._log_file = None
        self._identity = identity
        self._session_id: Optional[str] = None
        self._last_fit: Optional[dict] = None
        # v:2 per-reader logging (additive; 1 = legacy behavior, byte-stable)
        self._reader_schema = int(reader_schema)
        # staged cold entries: personas that engage at their FIRST speak
        # (no roster membership at open, no pre-entry acclimation)
        self._staged: Dict[str, Participant] = dict(staged_entries or {})
        self._reader_known: Dict[str, bool] = {}
        self._entry_mode: Dict[str, str] = {}
        self._reader_hist: Dict[str, list] = {}
        self._zc = np.array([DIAL_CENTER[n] for n in DIAL_NAMES])
        self._zscale = 2.0 / (np.array([DIAL_BOUNDS[n][1] for n in DIAL_NAMES])
                              - np.array([DIAL_BOUNDS[n][0] for n in DIAL_NAMES]))

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #
    def start_session(self) -> "TapNightSession":
        """Open the evening: fresh room, everyone starts at their native vibe."""
        self.room = Room(f"{self.name}")
        self._vibe = {n: p.vibe.copy() for n, p in self.participants.items()}
        self._vibe_start = {n: p.vibe.copy() for n, p in self.participants.items()}
        self._interactions = {}
        self._reaction_heat = {}
        self._clock = 0.0
        self.field = np.zeros(len(DIAL_NAMES))
        self._last_fit = None
        self._reader_known = {n: True for n in self.participants}
        self._entry_mode = {n: "roster" for n in self.participants}
        self._reader_hist = {n: [] for n in self.participants}
        self._session_id = uuid.uuid4().hex
        self._log.append(f"--- {self.name} opens: {len(self.participants)} "
                         f"souls at the table ---")
        self._emit(self._session_open())
        return self

    def speak(self, author: str, text: str, ts: Optional[float] = None,
              reactions: Optional[dict] = None) -> "TapNightSession":
        """Ingest one work read aloud and advance the field (charisma +
        acclimation). Unknown authors are registered lazily with neutral
        settings."""
        ts = float(self._clock if ts is None else ts)
        self._clock = max(self._clock, ts + self.STEP)
        msg = Message(author=author, text=text, ts=ts,
                      reactions=dict(reactions or {}))
        self.room.messages.append(msg)
        self.room.messages.sort(key=lambda m: m.ts)

        if author not in self.participants:
            self._register(author)
        # Capture the entry marker BEFORE the interaction counter increments.
        first_by_author = author not in self._interactions
        self._interactions[author] = self._interactions.get(author, 0) + 1
        self._reaction_heat[author] = (self._reaction_heat.get(author, 0)
                                       + msg.reaction_heat)

        # Raw dial field (persisted now — previously computed-and-discarded).
        raw = read_field(self.room, self.bank).vector()

        # Charisma: the room warms to strong presences. For a single agent this
        # reduces exactly to charisma_pull(raw, vibe, charisma, n); the /max(1,
        # total) keeps an aggregate of many pulls bounded to one step.
        delta = np.zeros(len(DIAL_NAMES))
        total = 0.0
        for pname, p in self.participants.items():
            n = self._interactions.get(pname, 0)
            if n <= 0:
                continue
            s = 1.0 - math.exp(-p.charisma * n)
            delta += s * (self._vibe[pname] - raw)
            total += s
        field = (raw + delta / max(1.0, total)) if total > 0 else raw
        self.field = self._clamp(field)

        # Acclimation: everyone warms to the room at their own rate.
        # (captured BEFORE this step: each reader's displaced field, using the
        #  pre-acclimation vibe that drove this speak's displacement — the
        #  exact quantity the v:2 schema logs as field_eff_to_reader)
        readers_pre = None
        if self._reader_schema >= 2:
            readers_pre = {}
            for pname, p in self.participants.items():
                n = self._interactions.get(pname, 0)
                s = 1.0 - math.exp(-p.charisma * n)
                eff = self._clamp(raw + s * (self._vibe[pname] - raw))
                self._reader_hist.setdefault(pname, []).append(eff)
                readers_pre[pname] = eff
        for pname, p in self.participants.items():
            alpha = 1.0 - math.exp(-p.acclimation_rate)
            self._vibe[pname] += (self.field - self._vibe[pname]) * alpha

        self._emit(self._speak_event(msg, raw, first_by_author,
                                      readers_pre=readers_pre))
        return self

    def end_session(self) -> str:
        """Close the evening and return a log line."""
        self.cycle += 1
        f = self.room_field()
        top = ", ".join(self._top_dials(3))
        line = (f"Night {self.cycle} closed: warmth={f.warmth():+.2f} "
                f"κ={f.concentration():.2f} | top: {top}")
        self._log.append(line)
        self._emit(self._session_close(top))
        self.close()
        return line

    # ------------------------------------------------------------------ #
    # Read the room                                                      #
    # ------------------------------------------------------------------ #
    def room_field(self) -> RoomField:
        """The effective field (raw dials displaced by charisma)."""
        return RoomField(dict(zip(DIAL_NAMES, self.field)))

    def raw_field(self) -> RoomField:
        """The un-displaced dial field — the room before charisma bent it."""
        return read_field(self.room, self.bank)

    def participant_state(self, name: str) -> dict:
        p = self.participants[name]
        return {
            "name": name,
            "dial_weights": p.dial_weights.tolist(),
            "acclimation_rate": p.acclimation_rate,
            "charisma": p.charisma,
            "vibe": self._vibe.get(name, p.vibe).tolist(),
            "vibe_start": self._vibe_start.get(name, p.vibe).tolist(),
            "interactions": self._interactions.get(name, 0),
            "reaction_heat": self._reaction_heat.get(name, 0),
        }

    # ------------------------------------------------------------------ #
    # Self-fine-tuning (across cycles)                                   #
    # ------------------------------------------------------------------ #
    def felt_engagement(self, name: str) -> np.ndarray:
        """Per-dial felt engagement this session (signed).

        Anchored to the participant's STABLE native style (`vibe` — the desire
        they bring) measured against the CAST's average desire, and amplified
        by the crowd's hands on the dial each reaction expresses:

        - `delta = vibe - cast_mean_vibe` is peer-relative: an agent feels
          engaged on the dials they care about *more than the rest of the
          table* — a warm writer leans mood, a sneering critic leans cynicism.
          This is what makes tastes diverge (different guitarists) instead of
          collapse to the room's loudest dial, and it is robust to the v0
          dials' tendency to saturate;
        - reaction heat is attributed per-dial (😂 -> joke_landing, ❤️ -> mood,
          🙄 -> cynicism, ...) so the crowd's hands reinforce the dial that
          actually landed, not a single shared dial.

        The room field still governs the relationship (acclimation warms the
        participant toward it, charisma pulls it toward them); the self-tuning
        signal is what the participant themselves is distinctive on.
        """
        p = self.participants[name]
        cast = np.mean(np.stack([q.vibe for q in self.participants.values()],
                                axis=0), axis=0)
        delta = p.vibe - cast
        rxn = np.zeros(len(DIAL_NAMES))
        for m in self.room.messages:
            if m.author != name:
                continue
            for emoji, count in m.reactions.items():
                d = REACTION_TO_DIAL.get(emoji)
                if d:
                    rxn[DIAL_NAMES.index(d)] += count
        return delta * (1.0 + rxn)

    def tune_participant(self, name: str, felt_engagement=None,
                         learning_rate: float = 0.15) -> "TapNightSession":
        """Self-fine-tune one participant's dial_weights toward the dials where
        their felt engagement was POSITIVE (their distinctive dials), zeroing
        the rest. The target is the normalized ReLU of engagement, so weight
        only ever moves toward dials the participant is genuinely distinctive
        on — this is what lets tastes diverge instead of collapsing to the
        room's loudest dial."""
        p = self.participants[name]
        if felt_engagement is None:
            eng = self.felt_engagement(name)
        elif np.isscalar(felt_engagement):
            eng = self.felt_engagement(name) * float(felt_engagement)
        else:
            eng = np.asarray(felt_engagement, dtype=float)

        pos = np.maximum(eng, 0.0)
        total = pos.sum()
        if total > 1e-9:
            # small epsilon keeps a trace of exploration on every dial
            target = (pos + 1e-3) / (total + 1e-3 * len(DIAL_NAMES))
            p.dial_weights = ((1 - learning_rate) * p.dial_weights
                              + learning_rate * target)
            p.dial_weights /= p.dial_weights.sum()
        return self

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #
    def settings(self) -> dict:
        return {n: p.to_dict() for n, p in self.participants.items()}

    def load_settings(self, data: dict) -> "TapNightSession":
        for name, d in data.items():
            if name in self.participants:
                p = self.participants[name]
                p.dial_weights = np.asarray(d["dial_weights"], dtype=float)
                p.acclimation_rate = float(d["acclimation_rate"])
                p.charisma = float(d["charisma"])
                p.vibe = np.asarray(d["vibe"], dtype=float)
        return self

    # ------------------------------------------------------------------ #
    # Edge log (spec §3 — append-only facts, JSONL)                      #
    # ------------------------------------------------------------------ #
    def _log_stream(self):
        if self._log_path is None:
            return None
        if self._log_file is None:
            d = os.path.dirname(os.path.abspath(self._log_path))
            if d:
                os.makedirs(d, exist_ok=True)
            self._log_file = open(self._log_path, "a", encoding="utf-8")
        return self._log_file

    def _emit(self, evt: dict) -> None:
        """Append one JSONL event. Logging must never break the session."""
        f = self._log_stream()
        if f is None:
            return
        try:
            f.write(json.dumps(evt, allow_nan=False) + "\n")
            f.flush()
        except (TypeError, ValueError):
            pass  # a broken log line is better than a broken session

    def close(self) -> None:
        """Flush and close the edge log (idempotent)."""
        if self._log_file is not None:
            try:
                self._log_file.flush()
                self._log_file.close()
            finally:
                self._log_file = None

    def _session_open(self) -> dict:
        evt = {
            "v": 1, "type": "session_open",
            "session_id": self._session_id,
            "space_id": self.name,
            "t_start": self._clock,
            "clock_mode": "auto60",
            "reader": {"kind": "RoomElephant", "identity": self._identity,
                       "bank": [d.__class__.__name__ for d in self.bank.dials]},
            "params": {"W": self.W, "standardization": "z=2(v-c)/(hi-lo)",
                       "estimator": "vmf-mle-newton-v1", "kappa_max": 500},
            "roster": {n: {**p.to_dict(), "vibe_start": self._vibe_start[n].tolist()}
                       for n, p in self.participants.items()},
        }
        if self._reader_schema >= 2:
            evt["reader_schema"] = {"version": 2,
                                    "field": "field_eff_to_reader",
                                    "lens": ["vibe_now", "weights_now"],
                                    "fit": "vmf-mle-newton-v1",
                                    "gate": "roster"}
            if self._staged:
                evt["staged_entries"] = {
                    n: {**p.to_dict(), "vibe_start": p.vibe.tolist()}
                    for n, p in self._staged.items()}
        return evt

    def _speak_event(self, msg: Message, raw: np.ndarray,
                     first_by_author: bool,
                     readers_pre: Optional[Dict[str, np.ndarray]] = None) -> dict:
        fit = vmf_fit(vmf_windowed(self.room, self.bank, W=self.W))
        edge = None
        if self._last_fit is not None and fit is not None:
            edge = vmf_edge(self._last_fit, fit)
            edge["real"] = None  # floor uncalibrated until measurement nights
        # Cell-ledger producer (quilt bridge, docs/quilt-bridge.md): book the
        # before→after field-edge the moment it happens — imbalance ≡ d_mu.
        # Emitted as its own append-only record (None when nothing to book).
        ledger = vmf_record_with(self._last_fit, fit,
                                 cell=f"room.field.{self.name}", ts=msg.ts)
        if ledger is not None:
            self._emit(dict(ledger, type="ledger"))
        if fit is not None:
            self._last_fit = fit
        trailing = self.room.messages[-self.W:]
        presence_mask = sorted({m.author for m in trailing})
        evt = {
            "v": 1, "type": "speak",
            "session_id": self._session_id,
            "space_id": self.name,
            "seq": self.room.messages.index(msg),
            "ts": msg.ts,
            "author": msg.author,
            "text_sha256": hashlib.sha256(msg.text.encode("utf-8")).hexdigest(),
            "len": len(msg.text),
            "reactions": dict(msg.reactions),
            "first_by_author": first_by_author,
            "presence_mask": presence_mask,
            "field_raw_after": raw.tolist(),
            "field_eff_after": self.field.tolist(),
            "interactions_after": dict(self._interactions),
            "fit": fit,
            "edge": edge,
        }
        if self._reader_schema >= 2 and readers_pre is not None:
            evt["v"] = 2
            readers = {}
            for pname, p in self.participants.items():
                eff = readers_pre[pname]
                readers[pname] = {
                    "reader_known": bool(self._reader_known.get(pname, False)),
                    "charisma": p.charisma,
                    "field_eff_to_reader": eff.tolist(),
                    "lens_now": {
                        "vibe_now": self._vibe[pname].tolist(),
                        "weights_now": p.dial_weights.tolist(),
                    },
                    "reader_fit": self._reader_fit(
                        self._reader_hist[pname][-self.W:], p.dial_weights),
                }
            evt["readers"] = readers
            evt["entry_mode"] = dict(self._entry_mode)
            reading_of = {}
            if msg.author in readers_pre:
                a = readers_pre[msg.author]
                na = np.linalg.norm(a)
                for member in presence_mask:
                    if member == msg.author:
                        reading_of[member] = {"cos": 1.0}
                        continue
                    b = readers_pre.get(member)
                    if b is None:
                        continue
                    nb = np.linalg.norm(b)
                    c = (float(np.dot(a, b) / (na * nb))
                         if na > 1e-12 and nb > 1e-12 else 0.0)
                    reading_of[member] = {"cos": c}
            evt["reading_of"] = reading_of
        return evt

    def _session_close(self, top_dials: List[str]) -> dict:
        fit = vmf_fit(vmf_windowed(self.room, self.bank, W=self.W))
        rf = self.raw_field()
        final = {
            "readings": rf.readings,  # all 9 raw bank readings
            "mu_hat": fit["mu_hat"] if fit else None,
            "kappa": fit["kappa"] if fit else None,
            "kappa_ci": fit["kappa_ci"] if fit else None,
            "warmth_v0": rf.warmth(),
            "warmth_vmf": fit["warmth_vmf"] if fit else None,
            "top_dials": top_dials,
        }
        evt = {
            "v": 1, "type": "session_close",
            "session_id": self._session_id,
            "space_id": self.name,
            "t_end": self._clock,
            "cycle": self.cycle,
            "final": final,
            "n_messages": len(self.room.messages),
            "notes": "",
        }
        if self._reader_schema >= 2:
            evt["reader_final"] = self._reader_final()
        return evt

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _register(self, name: str) -> None:
        """Lazily add an unknown speaker. A STAGED entry (pre-declared
        persona, cold: engaged at first speak, never pre-warmed) keeps its
        persona; an unstaged stranger gets neutral defaults."""
        if name in self._staged:
            src = self._staged[name]
            p = Participant(src.name,
                            dial_weights=src.dial_weights.copy(),
                            acclimation_rate=src.acclimation_rate,
                            charisma=src.charisma, vibe=src.vibe.copy())
            self._entry_mode[name] = "staged-cold"
        else:
            p = Participant(
                name, dial_weights=np.full(len(DIAL_NAMES), 1.0 / len(DIAL_NAMES)),
                acclimation_rate=0.2, charisma=0.1,
                vibe=np.full(len(DIAL_NAMES), 0.5))
            self._entry_mode[name] = "lazy-neutral"
        self.participants[name] = p
        self._vibe[name] = self.participants[name].vibe.copy()
        self._vibe_start[name] = self.participants[name].vibe.copy()
        self._reader_known[name] = False

    def _reader_fit(self, win: list, weights: np.ndarray) -> Optional[dict]:
        """Light vMF MLE over a reader's own attention-weighted reading
        window (per-reader schema doc: mu_hat/kappa/n; null under n < 3).
        Same Newton estimator as vmf.py's A7 solve; no bootstrap/jackknife
        and no NMIN=10 guard — the reader window is W speaks by design."""
        if len(win) < 3:
            return None
        g = weights / weights.max() if weights.max() > 1e-12 else np.ones(7)
        z = np.stack([self._zscale * g * (v - self._zc) for v in win])
        z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
        r = z.mean(0)
        rho = float(np.linalg.norm(r))
        if rho < 1e-12:
            return {"mu_hat": None, "kappa": None, "n": len(win)}
        mu = r / rho
        k = float(np.clip(rho * (7 - rho ** 2) / (1 - rho ** 2), 1e-6, 500.0))
        for _ in range(60):
            a = vmf_A7(k)
            gg = 1.0 - a * a - 6.0 * a / k
            if abs(gg) < 1e-12:
                break
            step = (a - rho) / gg
            k = float(np.clip(k - step, 1e-6, 500.0))
            if abs(step) < 1e-9:
                break
        return {"mu_hat": mu.tolist(), "kappa": k, "n": len(win)}

    def _clamp(self, vec: np.ndarray) -> np.ndarray:
        out = vec.copy()
        for i, n in enumerate(DIAL_NAMES):
            lo, hi = DIAL_BOUNDS[n]
            out[i] = min(hi, max(lo, out[i]))
        return out

    def _reader_final(self) -> dict:
        """Componentwise median of each reader's field_eff_to_reader over
        the night (schema doc: the greppable per-reader baseline fact)."""
        out = {}
        for name, hist in self._reader_hist.items():
            if not hist:
                continue
            out[name] = np.median(np.stack(hist), axis=0).tolist()
        return out

    def _top_dials(self, k: int) -> List[str]:
        dev = [(n, abs(self.field[i] - DIAL_CENTER[n]))
               for i, n in enumerate(DIAL_NAMES)]
        dev.sort(key=lambda t: -t[1])
        return [n for n, _ in dev[:k]]

    def __len__(self) -> int:
        return len(self.room.messages)

    def __repr__(self) -> str:
        return (f"<TapNightSession {self.name!r}: {len(self.room)} msgs, "
                f"{len(self.participants)} participants, night {self.cycle}>")
