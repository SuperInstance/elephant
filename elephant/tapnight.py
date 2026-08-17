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
  anchored to the agent's own distinctive voice vs the room-without-them (so
  charisma can't capture the loop), signed (so direction matters, not raw
  extremity), and selected with a softmax temperature (so tastes diverge into
  multiple stable attractors instead of collapsing to one loud dial).

numpy-only. Mirrors the `BoatHarness` pattern (rolling room + dial bank +
field), but for people reading each other's work.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from elephant.dial import DialBank
from elephant.dials import DEFAULT_DIALS
from elephant.field import DIAL_NAMES, RoomField, read_field
from elephant.room import Message, Room

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
                 bank: Optional[DialBank] = None):
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
        self._log.append(f"--- {self.name} opens: {len(self.participants)} "
                         f"souls at the table ---")
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
        self._interactions[author] = self._interactions.get(author, 0) + 1
        self._reaction_heat[author] = (self._reaction_heat.get(author, 0)
                                       + msg.reaction_heat)

        # Raw dial field.
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
        for pname, p in self.participants.items():
            alpha = 1.0 - math.exp(-p.acclimation_rate)
            self._vibe[pname] += (self.field - self._vibe[pname]) * alpha
        return self

    def end_session(self) -> str:
        """Close the evening and return a log line."""
        self.cycle += 1
        f = self.room_field()
        top = ", ".join(self._top_dials(3))
        line = (f"Night {self.cycle} closed: warmth={f.warmth():+.2f} "
                f"κ={f.concentration():.2f} | top: {top}")
        self._log.append(line)
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
        they bring) measured against the room-WITHOUT-them, and amplified by
        the crowd's hands on the dial each reaction expresses:

        - baseline = field of everyone ELSE's lines, so pulling the room toward
          yourself (charisma) is never rewarded in your own signal;
        - `delta = vibe - baseline` is signed: an agent engages the dials where
          their distinctive taste *differs* from the room — a warm writer leans
          mood, a sneering critic leans cynicism, each on their own dial;
        - reaction heat is attributed per-dial (😂 -> joke_landing, ❤️ -> mood,
          🙄 -> cynicism, ...) so the crowd's hands reinforce the dial that
          actually landed, not a single shared dial.
        """
        p = self.participants[name]
        others = [m for m in self.room.messages if m.author != name]
        if others:
            baseline = read_field(Room(f"{name}/others", others),
                                  self.bank).vector()
        else:
            baseline = np.full(len(DIAL_NAMES), 0.5)
        delta = p.vibe - baseline
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
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _register(self, name: str) -> None:
        """Lazily add an unknown speaker with neutral defaults."""
        self.participants[name] = Participant(
            name, dial_weights=np.full(len(DIAL_NAMES), 1.0 / len(DIAL_NAMES)),
            acclimation_rate=0.2, charisma=0.1,
            vibe=np.full(len(DIAL_NAMES), 0.5))
        self._vibe[name] = self.participants[name].vibe.copy()
        self._vibe_start[name] = self.participants[name].vibe.copy()

    def _clamp(self, vec: np.ndarray) -> np.ndarray:
        out = vec.copy()
        for i, n in enumerate(DIAL_NAMES):
            lo, hi = DIAL_BOUNDS[n]
            out[i] = min(hi, max(lo, out[i]))
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
