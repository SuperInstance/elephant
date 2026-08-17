"""Space — the elephant works in any room.

The elephant must not be coupled to the MUD system. It reads *any*
communication space between agents, humans, bots, or sensor arrays: a
message board, a chatroom, a messenger conversation, an X thread, a
Discord server, a Slack channel, an email thread, a team radio channel,
a sensor telemetry bus, a fish-finder feed, a camera watch.

Every space is normalized into the same thing the elephant already
reads — a `Room` (messages) or `SignalRoom` (frames) — through a thin
adapter. The core (`room.py`, `dial.py`, `field.py`, `dials/`) never
knows what the space *is*; it only knows Rooms, Messages, Frames,
DialBank, RoomField.

A `Space` wraps any native medium and exposes four seams:

    ingest(...)        — accept/read events from the native space
    .room              — the normalized Room (or SignalRoom) the elephant reads
    .tint_target()     — WHAT the elephant's description-mutation writes back
                         (MUD description text, a chat topic, a pinned message,
                         a status line, sensor alert phrasing...)
    .send_back(field)  — push the elephant's readout back in the space's own
                         idiom (optional per adapter, but every built-in does)

`read(bank)` runs a dial bank over `.room` and returns a `RoomField`.

The rule: **JEPA correlates; it never replaces.** The elephant nudges
what the space's own body-language compares — it is the light, and the
light works in every room that has one.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np

from .dial import DialBank
from .dials import DEFAULT_DIALS
from .field import RoomField, read_field
from .room import Message, Room
from .sensors import (
    FishingDayDial,
    RadarCoherenceDial,
    SensorFrame,
    SignalRoom,
    SounderBiomassDial,
)

__all__ = [
    "Space", "AdapterRegistry",
    "MudSpace", "ChatSpace", "SensorSpace",
    "read_space",
]

# Sensor dials: the array's own room-temperature (radar coherence,
# sounder biomass, the composite "fishing day" luck field).
SENSOR_DIALS = [RadarCoherenceDial(), SounderBiomassDial(), FishingDayDial()]


# ---------------------------------------------------------------------- #
# Protocol                                                               #
# ---------------------------------------------------------------------- #
class Space(ABC):
    """The adapter contract: wrap any communication medium as a room the
    elephant can read, and a tint target the elephant can write back to.
    """

    kind: str = "space"      # registry key, e.g. "mud", "chat", "sensor"
    step: float = 60.0       # auto-timestamp increment per event (seconds)

    def __init__(self, name: str):
        self.name = name
        self._clock = 0.0
        self._last_tint: Optional[str] = None

    # -- ingest ------------------------------------------------------- #
    @abstractmethod
    def ingest(self, *events) -> "Space":
        """Accept events from the native space (messages, frames, commits,
        whatever the medium produces); return self."""

    # -- normalized room ---------------------------------------------- #
    @property
    @abstractmethod
    def room(self) -> Union[Room, SignalRoom]:
        """The normalized room the elephant reads — same timestamped
        sequence, whatever the medium was."""

    # -- tint target -------------------------------------------------- #
    @abstractmethod
    def tint_target(self) -> str:
        """What the elephant's description-mutation writes back to: the
        MUD room description, a chat topic / pinned message / status line,
        sensor alert phrasing / display emphasis..."""

    # -- read --------------------------------------------------------- #
    def read(self, bank: Optional[DialBank] = None) -> RoomField:
        """Run a dial bank over this space's room -> its field."""
        bank = bank or DialBank(DEFAULT_DIALS)
        return read_field(self.room, bank)

    # -- tint + send_back --------------------------------------------- #
    @abstractmethod
    def tint(self, field: RoomField) -> str:
        """Transform a field into this space's own idiom — the tinted
        description/topic/alert that every participant sees."""

    def send_back(self, field: RoomField, tinted_text: Optional[str] = None) -> str:
        """Push the elephant's readout back into the space's native medium.

        `tinted_text` may be pre-computed; otherwise it is derived from
        `field` via `.tint`. Returns the text that was written back.
        """
        if tinted_text is None:
            tinted_text = self.tint(field)
        self._last_tint = tinted_text
        return tinted_text

    # -- internals ---------------------------------------------------- #
    def _next_ts(self, ts: Optional[float] = None) -> float:
        """Resolve a timestamp: explicit, or auto-incrementing by `step`."""
        if ts is None:
            ts = self._clock
        ts = float(ts)
        self._clock = max(self._clock, ts + self.step)
        return ts

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r} ({self.kind})>"


def read_space(space: Space, bank: Optional[DialBank] = None) -> RoomField:
    """Module-level convenience: the elephant reads a space."""
    return space.read(bank)


# ---------------------------------------------------------------------- #
# Registry                                                               #
# ---------------------------------------------------------------------- #
class AdapterRegistry:
    """Register and instantiate space adapters by kind string.

        AdapterRegistry.register("mud", MudSpace)
        AdapterRegistry.get("mud", "The Tap")   -> MudSpace("The Tap")
        AdapterRegistry.kinds()                 -> ["mud", "chat", ...]

    `register` works as a plain call or as a decorator.
    """

    _registry: Dict[str, Type[Space]] = {}

    @classmethod
    def register(cls, kind: str, adapter_cls: Optional[Type[Space]] = None):
        if adapter_cls is None:
            def deco(adapter_cls: Type[Space]) -> Type[Space]:
                cls._registry[kind] = adapter_cls
                return adapter_cls
            return deco
        cls._registry[kind] = adapter_cls
        return adapter_cls

    @classmethod
    def get(cls, kind: str, *args, **kwargs) -> Space:
        try:
            adapter_cls = cls._registry[kind]
        except KeyError:
            raise KeyError(
                f"no adapter registered for space kind {kind!r}; "
                f"registered kinds: {cls.kinds()}"
            ) from None
        return adapter_cls(*args, **kwargs)

    @classmethod
    def kinds(cls) -> List[str]:
        return sorted(cls._registry)

    @classmethod
    def has(cls, kind: str) -> bool:
        return kind in cls._registry


# ---------------------------------------------------------------------- #
# Event coercion — native events -> Message                              #
# ---------------------------------------------------------------------- #
def _coerce_message(event: Any, ts: float = 0.0) -> Message:
    """Turn a native event into a Message.

    Accepts a `Message` (used as-is), a bare string (a room/ambient
    event, authored by "[room]"), or a `(author, text[, ts])` pair.
    """
    if isinstance(event, Message):
        return event
    if isinstance(event, str):
        return Message(author="[room]", text=event, ts=ts)
    if isinstance(event, (tuple, list)) and len(event) >= 2:
        author, text = event[0], event[1]
        t = event[2] if len(event) > 2 else ts
        return Message(author=str(author), text=str(text), ts=float(t))
    raise TypeError(
        f"cannot coerce {type(event).__name__!r} to a Message "
        f"(want Message, str, or (author, text[, ts]))"
    )


# ---------------------------------------------------------------------- #
# Tint phrasing — the space's own body language                          #
# ---------------------------------------------------------------------- #
def _field_traits(field: RoomField) -> Tuple[float, float, float, float, float, float]:
    r = field.readings
    return (
        field.warmth(),
        field.concentration(),
        r.get("mood", 0.0),
        r.get("joke_landing", 0.0),
        r.get("panic", 0.0),
        r.get("volume", 0.0),
    )


def _mud_tint(field: RoomField) -> str:
    """Room description prose, keyed to the field.

    SEAM: if the parallel zeitgeist build adds `elephant/mud.py` with a
    `tint_description(field)` function, `MudSpace.tint` prefers it (see
    `MudSpace.tint`). Until then, this field-level phrasing stands in.
    """
    warm, _kappa, _mood, joke, panic, _volume = _field_traits(field)
    if panic >= 0.5:
        base = ("The room is in an uproar. Chairs scrape, someone shouts, "
                "and the door bangs open onto the storm.")
    elif warm >= 0.25:
        base = ("The room is warm and bright. Laughter hangs in the corners; "
                "the pool tables hum with easy talk and the darts clink in the back.")
    elif warm >= 0.0:
        base = ("A comfortable, lived-in room. Conversation drifts in low, "
                "friendly currents and the wood glows softly.")
    elif warm >= -0.25:
        base = ("A still room. The talk is clipped and quiet; a few glances "
                "slide toward the door.")
    else:
        base = ("A cold room. A storm rattles the windows, and newcomers step "
                "in drenched, shaking the water from their coats.")
    if joke >= 0.4:
        base += " Someone's story just landed — a ripple of laughter crosses the room."
    return base


def _chat_tint(name: str, field: RoomField, n_msgs: int) -> str:
    """A channel topic / status line keyed to the field."""
    warm, kappa, _mood, _joke, panic, volume = _field_traits(field)
    if panic >= 0.5 or volume >= 0.8:
        tag, phrase = "🔥", "heated — tempers up, the thread is moving fast"
    elif warm >= 0.25:
        tag, phrase = "✨", "good vibes — jokes landing, everyone's loose"
    elif warm >= 0.0:
        tag, phrase = "☕", "easy conversation, steady and comfortable"
    elif warm >= -0.25:
        tag, phrase = "🕯", "quiet — the thread has gone still"
    else:
        tag, phrase = "❄", "cold — the room has gone flat and sharp"
    return f"{tag} {name} — {phrase} (warmth {warm:+.2f}, κ {kappa:.2f}, {n_msgs} msgs)"


def _sensor_tint(name: str, field: RoomField) -> str:
    """Sensor alert phrasing / display emphasis keyed to the field."""
    r = field.readings
    panic = r.get("panic", 0.0)
    radar = r.get("radar_coherence", None)
    biomass = r.get("sounder_biomass", None)

    if panic >= 0.5:
        return f"⚠ {name}: ALERT — panic spike; verify camera_out / collision track"
    if radar is not None and biomass is not None:
        if radar > 0.3 and biomass > 0.6:
            return (f"🟢 {name}: fleet tight (κ {radar:+.2f}) + biomass "
                    f"{biomass:.2f} — on fish, hold the drag")
        if radar > 0.3:
            return f"🟡 {name}: radar coherent (κ {radar:+.2f}) — watch the cluster"
        if biomass > 0.6:
            return f"🟡 {name}: biomass {biomass:.2f} thick — sounder hot"
        if radar < -0.3:
            return f"🔵 {name}: fleet scattered (κ {radar:+.2f}) — still searching"
    warm = field.warmth()
    if warm < -0.1:
        return f"⚪ {name}: quiet deck — sparse signal, all nominal"
    return f"⚪ {name}: nominal — nothing the elephant needs to push"


# ---------------------------------------------------------------------- #
# MudSpace                                                               #
# ---------------------------------------------------------------------- #
class MudSpace(Space):
    """A MUD room. Room events + NPC chatter become messages; the room
    description is the tint target — the words every agent sees."""

    kind = "mud"

    def __init__(self, name: str, description: str = ""):
        super().__init__(name)
        self._room = Room(name)
        # `base_description` is the plain bar text the field mutates; `description`
        # is what agents actually read (tinted after send_back).
        self.base_description = description or "A low-ceilinged room, warm wood and a long counter."
        self.description = self.base_description

    def ingest(self, *events) -> "MudSpace":
        for e in events:
            self._room.messages.append(_coerce_message(e, self._next_ts()))
        self._room.messages.sort(key=lambda m: m.ts)
        return self

    def event(self, text: str, ts: Optional[float] = None) -> "MudSpace":
        """A room event (ambient line authored by the room itself)."""
        return self.ingest(Message(author=f"[{self.name}]", text=text,
                                   ts=self._next_ts(ts)))

    def chatter(self, author: str, text: str, ts: Optional[float] = None) -> "MudSpace":
        """NPC chatter (or any speaker) in the room."""
        return self.ingest(Message(author=author, text=text,
                                   ts=self._next_ts(ts)))

    @property
    def room(self) -> Room:
        return self._room

    def tint_target(self) -> str:
        return "the room description"

    def tint(self, field: RoomField) -> str:
        # SEAM: prefer the zeitgeist build's tint_description if it exists. It
        # mutates the plain `base_description` by the room's field (the room
        # acting on everyone in it). Signature: tint_description(field,
        # base_text, hour=None, seed=None).
        try:
            from .mud import tint_description  # type: ignore[import-not-found]
        except ImportError:
            tint_description = None
        if tint_description is not None:
            return tint_description(field, self.base_description)
        # Fallback: field-level phrasing (no mud.py).
        return _mud_tint(field)

    def send_back(self, field: RoomField, tinted_text: Optional[str] = None) -> str:
        text = super().send_back(field, tinted_text)
        self.description = text
        return text


# ---------------------------------------------------------------------- #
# ChatSpace                                                              #
# ---------------------------------------------------------------------- #
class ChatSpace(Space):
    """A chatroom / messenger / X thread. Authors, reactions, and reply
    trees become messages; gravity/reverb/ripple work as-is. The tint
    target is a topic / pinned / status line."""

    kind = "chat"

    def __init__(self, name: str, topic: str = ""):
        super().__init__(name)
        self._room = Room(name)
        self.topic = topic or f"{name} — no topic set"

    def ingest(self, *events) -> "ChatSpace":
        for e in events:
            self._room.messages.append(_coerce_message(e, self._next_ts()))
        self._room.messages.sort(key=lambda m: m.ts)
        return self

    def post(self, author: str, text: str, ts: Optional[float] = None,
             reactions: Optional[Dict[str, int]] = None,
             replies: Optional[List[Message]] = None) -> Message:
        """Post a message (with optional reactions + reply tree); returns it."""
        msg = Message(author=author, text=text, ts=self._next_ts(ts),
                      reactions=dict(reactions or {}),
                      replies=list(replies or []))
        self._room.messages.append(msg)
        self._room.messages.sort(key=lambda m: m.ts)
        return msg

    def react(self, index: int, emoji: str, n: int = 1) -> "ChatSpace":
        """Add `n` of `emoji` to the message at `index` (the crowd's hands)."""
        if 0 <= index < len(self._room.messages):
            m = self._room.messages[index]
            m.reactions[emoji] = m.reactions.get(emoji, 0) + n
        return self

    @property
    def room(self) -> Room:
        return self._room

    def tint_target(self) -> str:
        return "the channel topic / status line"

    def tint(self, field: RoomField) -> str:
        return _chat_tint(self.name, field, len(self._room))

    def send_back(self, field: RoomField, tinted_text: Optional[str] = None) -> str:
        text = super().send_back(field, tinted_text)
        self.topic = text
        return text


# ---------------------------------------------------------------------- #
# SensorSpace                                                            #
# ---------------------------------------------------------------------- #
def _point_count(data: Any) -> int:
    if isinstance(data, np.ndarray):
        return int(data.reshape(-1, 2).shape[0]) if data.size else 0
    return len(list(data))


def _render_frame(frame: SensorFrame) -> str:
    """Render a sensor frame as a line of text, so the shared dials
    (mood/volume/panic/presence...) can also feel the array."""
    s, d = frame.sensor, frame.data
    if s == "radar":
        n = _point_count(d)
        return f"radar: {n} target{'s' if n != 1 else ''} on the sweep"
    if s == "sounder":
        v = float(d)
        state = "thick" if v >= 0.6 else "thin" if v >= 0.3 else "sparse"
        return f"sounder: biomass {v:.2f} ({state})"
    if s == "nav":
        if isinstance(d, dict):
            return (f"nav: heading {d.get('heading', '?')}° "
                    f"at {d.get('speed', '?')} kts")
        return f"nav: {d}"
    if s == "camera":
        return "camera: watch frame"
    if s == "autopilot":
        return "autopilot: holding course"
    return f"{s}: {d}"


class SensorSpace(Space):
    """A sensor array read as a room. Reuses `SignalRoom`/`SensorFrame`
    (radar, sounder, cameras, nav, autopilot). The tint target is alert
    phrasing / display emphasis.

    Two rooms live here:
      - `.signal`   — the `SignalRoom` of raw frames, read by the fleet
                      dials (`radar_coherence`, `sounder_biomass`,
                      `fishing_day`);
      - `.room`     — a text rendering of those frames (plus any crew
                      chatter), so the SHARED seven dials (mood, volume,
                      panic, presence, ...) can feel the same array.
    """

    kind = "sensor"

    def __init__(self, name: str):
        super().__init__(name)
        self._signal = SignalRoom(name)
        self._room = Room(f"{name}/frames")
        self.alert = f"{name} — nominal"

        self.radar_dial = RadarCoherenceDial()
        self.sounder_dial = SounderBiomassDial()
        self.fishing_dial = FishingDayDial(radar=self.radar_dial,
                                           sounder=self.sounder_dial)
        self.sensor_bank = DialBank([self.radar_dial, self.sounder_dial,
                                     self.fishing_dial])

    # -- ingest ------------------------------------------------------- #
    def ingest(self, *events) -> "SensorSpace":
        for e in events:
            if isinstance(e, SensorFrame):
                self._add_frame(e)
            elif isinstance(e, Message):
                self._room.messages.append(e)          # crew chatter
                self._room.messages.sort(key=lambda m: m.ts)
            elif isinstance(e, (tuple, list)) and len(e) >= 2:
                sensor, data = e[0], e[1]
                ts = self._next_ts(e[2] if len(e) > 2 else None)
                self._add_frame(SensorFrame(ts=ts, sensor=str(sensor), data=data))
            else:
                raise TypeError(
                    f"ingest expects SensorFrame, Message, or (sensor, data[, ts]), "
                    f"got {type(e).__name__!r}"
                )
        return self

    def _add_frame(self, frame: SensorFrame) -> None:
        self._signal.frames.append(frame)
        self._signal.frames.sort(key=lambda f: f.ts)
        # The text view lets the shared dials feel the array too.
        self._room.messages.append(Message(author=frame.sensor,
                                           text=_render_frame(frame),
                                           ts=frame.ts))
        self._room.messages.sort(key=lambda m: m.ts)

    # -- convenience ingest ------------------------------------------- #
    def ingest_radar(self, targets, ts: Optional[float] = None,
                     meta: Optional[dict] = None) -> "SensorSpace":
        return self.ingest(SensorFrame(ts=self._next_ts(ts), sensor="radar",
                                       data=targets, meta=meta or {}))

    def ingest_sounder(self, biomass: float, ts: Optional[float] = None,
                       meta: Optional[dict] = None) -> "SensorSpace":
        return self.ingest(SensorFrame(ts=self._next_ts(ts), sensor="sounder",
                                       data=float(biomass), meta=meta or {}))

    def ingest_nav(self, heading: float, speed: float,
                   ts: Optional[float] = None,
                   meta: Optional[dict] = None) -> "SensorSpace":
        return self.ingest(SensorFrame(ts=self._next_ts(ts), sensor="nav",
                                       data={"heading": float(heading),
                                             "speed": float(speed)},
                                       meta=meta or {}))

    def ingest_alert(self, text: str, ts: Optional[float] = None) -> "SensorSpace":
        """A deck alarm / alert line — feeds the shared panic dial."""
        return self.ingest(Message(author="deck", text=text,
                                   ts=self._next_ts(ts)))

    # -- rooms -------------------------------------------------------- #
    @property
    def signal(self) -> SignalRoom:
        return self._signal

    @property
    def room(self) -> Room:
        return self._room

    # -- read --------------------------------------------------------- #
    def sensor_readings(self) -> Dict[str, float]:
        """The fleet dials over the raw signal room."""
        return self.sensor_bank.readings(self._signal)

    def full_read(self, bank: Optional[DialBank] = None) -> RoomField:
        """Merged field: fleet dials + shared dials (shared wins ties)."""
        merged = dict(self.sensor_readings())
        merged.update(self.read(bank).readings)
        return RoomField(merged)

    # -- tint --------------------------------------------------------- #
    def tint_target(self) -> str:
        return "sensor alert phrasing / display emphasis"

    def tint(self, field: RoomField) -> str:
        return _sensor_tint(self.name, field)

    def send_back(self, field: RoomField, tinted_text: Optional[str] = None) -> str:
        text = super().send_back(field, tinted_text)
        self.alert = text
        return text


# ---------------------------------------------------------------------- #
# Default registrations                                                  #
# ---------------------------------------------------------------------- #
# The three proof adapters.
AdapterRegistry.register("mud", MudSpace)
AdapterRegistry.register("chat", ChatSpace)
AdapterRegistry.register("sensor", SensorSpace)

# Aliases + forward placeholders. agent / human_bot / async / doc are
# chat-like today (authors, reactions, reply trees) until dedicated
# adapters land; `async` simply wants a longer gravity half-life at read
# time, which is a Room parameter, not an adapter change.
AdapterRegistry.register("messenger", ChatSpace)
AdapterRegistry.register("x_thread", ChatSpace)
AdapterRegistry.register("agent", ChatSpace)
AdapterRegistry.register("human_bot", ChatSpace)
AdapterRegistry.register("async", ChatSpace)
AdapterRegistry.register("doc", ChatSpace)
