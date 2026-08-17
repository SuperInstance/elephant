"""Four more rooms — the rest of the adapter set.

`space.py` ships the three proof adapters (MudSpace, ChatSpace,
SensorSpace) and leaves placeholders for the rest. This module lands the
remaining four, all registered into the same `AdapterRegistry` so the
elephant keeps ONE namespace for every room it can walk into:

    AgentSpace     — multi-agent channels (agent bars, the CNS bus):
                     agent messages + system events on the same clock.
    HumanBotSpace  — human + bot mixed channels: same as chat, but the
                     presence dial reads humans vs bots distinctly.
    AsyncSpace     — email / async threads: long time-deltas, a stretched
                     gravity half-life, a long-latency echo.
    DocSpace       — file / doc workspaces (repos, ai-writings): commits,
                     file events, review comments become messages.

The rule is unchanged: **JEPA correlates; it never replaces.** The
elephant nudges what each space's own body-language compares. These
adapters only translate the medium into the same Rooms, Frames, DialBank,
and RoomField the core already reads.
"""
from __future__ import annotations

import math
import subprocess
from typing import Dict, List, Optional

from .dial import Dial, DialBank
from .dials import DEFAULT_DIALS
from .field import RoomField, read_field
from .room import Message, Room
from .space import (
    AdapterRegistry,
    ChatSpace,
    Space,
    _coerce_message,
    _field_traits,
)

__all__ = [
    "AgentSpace", "HumanBotSpace", "AsyncSpace", "DocSpace",
    "PresenceSplitDial",
]


# ---------------------------------------------------------------------- #
# Shared helpers                                                         #
# ---------------------------------------------------------------------- #
def _cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length heat vectors."""
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)


def _presence(msgs: List[Message]) -> float:
    """The occupancy trace over a subset of messages — the same math the
    stock PresenceDial uses, factored out so it can be run per speaker kind."""
    if not msgs:
        return 0.0
    authors: Dict[str, dict] = {}
    t0 = msgs[0].ts
    t1 = msgs[-1].ts
    span = max(t1 - t0, 1e-9)
    for m in msgs:
        a = m.author
        e = authors.setdefault(a, {"first": m.ts, "last": m.ts, "n": 0})
        e["first"] = min(e["first"], m.ts)
        e["last"] = max(e["last"], m.ts)
        e["n"] += 1
    distinct = len(authors)
    recency = 1.0 - math.exp(-(t1 - t0) / max(span, 1e-9))
    longevity = 0.0
    for e in authors.values():
        life = (e["last"] - e["first"]) / span
        longevity += min(1.0, life * 2.0)
    longevity /= max(distinct, 1)
    activity = min(1.0, len(msgs) / 40.0)
    return max(0.0, min(1.0,
                        0.45 * distinct / 5.0 + 0.25 * recency
                        + 0.20 * longevity + 0.10 * activity))


# ---------------------------------------------------------------------- #
# AgentSpace — the CNS bus                                               #
# ---------------------------------------------------------------------- #
class AgentSpace(ChatSpace):
    """A multi-agent channel (an agent bar, the CNS bus). Agent messages and
    system events share one clock; authors are agent ids. The tint target is
    a status line / channel topic.

    Subclasses ChatSpace (authors, reactions, reply trees all work as-is)
    and adds `agent()` / `system()` conveniences plus agent-flavoured
    phrasing on the status line.
    """

    kind = "agent"

    def __init__(self, name: str, topic: str = ""):
        super().__init__(name, topic)
        # `status` is the bus's status line; `topic` is kept in sync so the
        # shared ChatSpace.send_back machinery still works unchanged.
        self.status = self.topic

    def agent(self, agent_id: str, text: str, ts: Optional[float] = None,
              reactions: Optional[Dict[str, int]] = None) -> Message:
        """An agent speaks on the bus."""
        return self.post(agent_id, text, ts=ts, reactions=reactions)

    def system(self, text: str, ts: Optional[float] = None) -> "AgentSpace":
        """A system event on the same clock (authored by the bus itself)."""
        self._room.messages.append(
            Message(author="[bus]", text=text, ts=self._next_ts(ts)))
        self._room.messages.sort(key=lambda m: m.ts)
        return self

    def tint_target(self) -> str:
        return "the status line / channel topic"

    def tint(self, field: RoomField) -> str:
        return _agent_tint(self.name, field, len(self._room))

    def send_back(self, field: RoomField, tinted_text: Optional[str] = None) -> str:
        text = super().send_back(field, tinted_text)
        self.status = text
        return text


def _agent_tint(name: str, field: RoomField, n_msgs: int) -> str:
    warm, kappa, _mood, _joke, panic, _volume = _field_traits(field)
    if panic >= 0.5:
        tag, phrase = "🚨", "bus in distress — alarms firing across agents"
    elif warm >= 0.25:
        tag, phrase = "🤝", "agents aligned — the bar hums in phase"
    elif warm >= 0.0:
        tag, phrase = "⚙️", "bus steady — agents trading, no contention"
    elif warm >= -0.25:
        tag, phrase = "🧊", "bus cooling — agents going quiet"
    else:
        tag, phrase = "💤", "bus dead — no traffic, agents offline"
    return f"{tag} {name} — {phrase} (warmth {warm:+.2f}, κ {kappa:.2f}, {n_msgs} msgs)"


# ---------------------------------------------------------------------- #
# HumanBotSpace — the presence dial reads humans vs bots                 #
# ---------------------------------------------------------------------- #
class PresenceSplitDial(Dial):
    """A presence dial that reads the human trace and the bot trace apart.

    Same occupancy math as the stock PresenceDial, but split by speaker kind
    — so a room of bots humming away and a room of humans lingering read as
    different presences, not one blurred pheromone trace.
    """

    name = "presence"
    description = "presence split by speaker kind (human vs bot)"

    def __init__(self, kinds: Dict[str, str]):
        self.kinds = kinds

    def read(self, room: Room) -> float:
        # The room's thrum is the louder of the two traces.
        return max(self.by_kind(room).values(), default=0.0)

    def by_kind(self, room: Room) -> Dict[str, float]:
        human = _presence([m for m in room.messages
                           if self.kinds.get(m.author) == "human"])
        bot = _presence([m for m in room.messages
                         if self.kinds.get(m.author) == "bot"])
        return {"human": human, "bot": bot}


class HumanBotSpace(ChatSpace):
    """A human + bot mixed channel. Same as chat — authors, reactions, reply
    trees — but the presence dial reads humans vs bots distinctly. Authors are
    tagged with a kind ("human" / "bot") and the elephant can feel the room's
    human presence separately from its bot presence. Tint target = a pinned
    message / greeting.
    """

    kind = "human_bot"

    def __init__(self, name: str, pinned: str = ""):
        super().__init__(name, pinned)
        self.pinned = self.topic
        self.author_kind: Dict[str, str] = {}
        self.presence_dial = PresenceSplitDial(self.author_kind)

    # -- tagging ------------------------------------------------------ #
    def mark(self, author: str, kind: str) -> "HumanBotSpace":
        if kind not in ("human", "bot"):
            raise ValueError(f"kind must be 'human' or 'bot', got {kind!r}")
        self.author_kind[author] = kind
        return self

    def human(self, author: str, text: str, ts: Optional[float] = None,
              reactions: Optional[Dict[str, int]] = None) -> Message:
        self.mark(author, "human")
        return self.post(author, text, ts=ts, reactions=reactions)

    def bot(self, author: str, text: str, ts: Optional[float] = None,
            reactions: Optional[Dict[str, int]] = None) -> Message:
        self.mark(author, "bot")
        return self.post(author, text, ts=ts, reactions=reactions)

    def kind_of(self, author: str) -> str:
        return self.author_kind.get(author, "unknown")

    def _kind(self, m: Message) -> str:
        return self.author_kind.get(m.author, "unknown")

    # -- presence split ---------------------------------------------- #
    def humans_vs_bots(self) -> float:
        """Ratio of human-authored to bot-authored messages.

        > 1 = humans lead; < 1 = bots lead; inf = no bots; 0 = no humans.
        """
        humans = sum(1 for m in self._room.messages if self._kind(m) == "human")
        bots = sum(1 for m in self._room.messages if self._kind(m) == "bot")
        if bots == 0:
            return float("inf") if humans else 0.0
        return humans / bots

    def presence_by_kind(self) -> Dict[str, float]:
        """The presence dial split by speaker kind: {"human": x, "bot": y}."""
        return self.presence_dial.by_kind(self._room)

    # -- read (kind-aware presence by default) ------------------------ #
    def read(self, bank: Optional[DialBank] = None) -> RoomField:
        if bank is None:
            dials = [d for d in DEFAULT_DIALS if d.name != "presence"]
            bank = DialBank(dials + [self.presence_dial])
        return read_field(self._room, bank)

    # -- tint --------------------------------------------------------- #
    def tint_target(self) -> str:
        return "a pinned message / greeting"

    def tint(self, field: RoomField) -> str:
        return _human_bot_tint(self.name, field, len(self._room),
                               self.humans_vs_bots())

    def send_back(self, field: RoomField, tinted_text: Optional[str] = None) -> str:
        text = super().send_back(field, tinted_text)
        self.pinned = text
        return text


def _human_bot_tint(name: str, field: RoomField, n_msgs: int, ratio: float) -> str:
    warm, kappa, _mood, _joke, panic, _volume = _field_traits(field)
    if ratio >= 1.0:
        mix = "humans lead"
    elif ratio > 0.0:
        mix = "bots lead"
    else:
        mix = "no one's here"
    if panic >= 0.5:
        tag, phrase = "🔥", f"heated — {mix}, the thread is moving fast"
    elif warm >= 0.25:
        tag, phrase = "✨", f"good vibes — {mix}, jokes landing"
    elif warm >= 0.0:
        tag, phrase = "☕", f"easy conversation — {mix}, steady"
    elif warm >= -0.25:
        tag, phrase = "🕯", f"quiet — {mix}, the thread has gone still"
    else:
        tag, phrase = "❄", f"cold — {mix}, the room has gone flat"
    return (f"{tag} {name} — {phrase} (warmth {warm:+.2f}, κ {kappa:.2f}, "
            f"{n_msgs} msgs, {ratio:.2f}:1 human:bot)")


# ---------------------------------------------------------------------- #
# AsyncSpace — long time-deltas, stretched gravity                       #
# ---------------------------------------------------------------------- #
class AsyncSpace(ChatSpace):
    """Email / async threads. Messages arrive with long time-deltas, so
    gravity cools slower — a stretched half-life (the caller passes a
    `half_life_scale`). Reverberation is exposed as a long-latency echo over
    the stretched gravity series. Tint target = thread subject/status.
    """

    kind = "async"

    def __init__(self, name: str, subject: str = "", half_life: float = 1800.0,
                 half_life_scale: float = 1.0):
        super().__init__(name, subject)
        self.subject = self.topic
        self.half_life = float(half_life)
        self.half_life_scale = float(half_life_scale)

    @property
    def effective_half_life(self) -> float:
        """The stretched half-life: `half_life * half_life_scale`."""
        return self.half_life * self.half_life_scale

    def email(self, author: str, subject: str, body: str,
              ts: Optional[float] = None) -> Message:
        """One email: subject + body become a single message (the body carries
        the mood the dials feel)."""
        return self.post(author, f"{subject} — {body}", ts=ts)

    # -- long-latency physics ----------------------------------------- #
    def gravity(self, msg: Message) -> float:
        """A message's pull under the stretched half-life."""
        return self._room.gravity(msg, half_life=self.effective_half_life)

    def gravity_series(self) -> List[float]:
        return [self.gravity(m) for m in self._room.messages]

    def reverberation(self, window: int = 8) -> float:
        """Long-latency echo: reverb over the stretched gravity series."""
        heats = self.gravity_series()
        if len(heats) < 2 * window:
            return 0.0
        windows = [heats[i:i + window]
                   for i in range(0, len(heats) - window, window)]
        if len(windows) < 2:
            return 0.0
        sims = [_cosine(a, b) for a, b in zip(windows[:-1], windows[1:])]
        return sum(sims) / len(sims) if sims else 0.0

    def tint_target(self) -> str:
        return "the thread subject / status"

    def tint(self, field: RoomField) -> str:
        return _async_tint(self.name, field, len(self._room),
                           self.effective_half_life)

    def send_back(self, field: RoomField, tinted_text: Optional[str] = None) -> str:
        text = super().send_back(field, tinted_text)
        self.subject = text
        return text


def _async_tint(name: str, field: RoomField, n_msgs: int, half_life: float) -> str:
    warm, kappa, _mood, _joke, panic, _volume = _field_traits(field)
    if panic >= 0.5:
        tag, phrase = "🚨", "urgent — replies firing, the thread is awake"
    elif warm >= 0.25:
        tag, phrase = "🌊", "long-form — thoughtful, unhurried, warm"
    elif warm >= 0.0:
        tag, phrase = "📮", "steady correspondence — the drift is easy"
    elif warm >= -0.25:
        tag, phrase = "🌫", "slow — replies come days apart"
    else:
        tag, phrase = "🪦", "dead thread — last reply long ago"
    return (f"{tag} {name} — {phrase} (warmth {warm:+.2f}, κ {kappa:.2f}, "
            f"{n_msgs} msgs, half-life {half_life:,.0f}s)")


# ---------------------------------------------------------------------- #
# DocSpace — commits, file events, review comments                       #
# ---------------------------------------------------------------------- #
class DocSpace(Space):
    """A file / doc workspace (a repo, ai-writings). Commits, file events, and
    review comments become messages — authors are committers, text is commit
    subjects / messages. Read-only adapter: it ingests a git log or a directory
    scan and never writes back to the tree. Tint target = a project status line.
    """

    kind = "doc"

    def __init__(self, name: str, repo_path: Optional[str] = None, status: str = ""):
        super().__init__(name)
        self._room = Room(name)
        self.repo_path = repo_path
        self.status = status or f"{name} — no status"

    # -- ingest ------------------------------------------------------- #
    def ingest(self, *events) -> "DocSpace":
        for e in events:
            self._room.messages.append(_coerce_message(e, self._next_ts()))
        self._room.messages.sort(key=lambda m: m.ts)
        return self

    def commit(self, author: str, subject: str,
               ts: Optional[float] = None) -> Message:
        return self.ingest(Message(author=author, text=subject,
                                   ts=self._next_ts(ts))).room.messages[-1]

    def file_event(self, path: str, action: str,
                   ts: Optional[float] = None) -> Message:
        return self.ingest(Message(author="[file]", text=f"{action}: {path}",
                                   ts=self._next_ts(ts))).room.messages[-1]

    def review_comment(self, author: str, text: str,
                       ts: Optional[float] = None) -> Message:
        return self.ingest(Message(author=author, text=text,
                                   ts=self._next_ts(ts))).room.messages[-1]

    def ingest_git_log(self, repo_path: Optional[str] = None,
                       max_count: int = 20, timeout: float = 30.0) -> "DocSpace":
        """Read a repo's git history into the room (read-only).

        `git log --format=%an%x09%s%x09%at -<max_count>` — authors become
        committers, text becomes commit subjects, and timestamps are
        normalized to start at 0 so the field reads the repo's *shape*,
        not its wall-clock age. Runs a blocking subprocess (bounded by
        `timeout` seconds).
        """
        repo = repo_path or self.repo_path
        if not repo:
            raise ValueError("repo_path required for git-log ingestion")
        proc = subprocess.run(
            ["git", "-C", repo, "log",
             "--format=%an%x09%s%x09%at", f"-{int(max_count)}"],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"git log failed in {repo!r}: {proc.stderr.strip()}")
        commits = []
        for line in proc.stdout.splitlines():
            line = line.rstrip("\n")
            if not line:
                continue
            author, rest = line.split("\t", 1)
            subject, _sep, ts_str = rest.rpartition("\t")
            if not _sep:
                continue
            commits.append((author, subject, float(ts_str)))
        if not commits:
            return self
        commits.sort(key=lambda c: c[2])
        base = commits[0][2]
        for author, subject, ts in commits:
            self.commit(author, subject, ts=ts - base)
        return self

    # -- room --------------------------------------------------------- #
    @property
    def room(self) -> Room:
        return self._room

    # -- tint --------------------------------------------------------- #
    def tint_target(self) -> str:
        return "a project status line"

    def tint(self, field: RoomField) -> str:
        return _doc_tint(self.name, field, len(self._room))

    def send_back(self, field: RoomField, tinted_text: Optional[str] = None) -> str:
        text = super().send_back(field, tinted_text)
        self.status = text
        return text


def _doc_tint(name: str, field: RoomField, n_msgs: int) -> str:
    warm, kappa, _mood, _joke, panic, _volume = _field_traits(field)
    if panic >= 0.5:
        tag, phrase = "🚧", "on fire — blockers and breakage in flight"
    elif warm >= 0.25:
        tag, phrase = "📈", "shipping clean — commits landing, reviews green"
    elif warm >= 0.0:
        tag, phrase = "📦", "steady — work moving, nothing screaming"
    elif warm >= -0.25:
        tag, phrase = "🧊", "quiet — few commits, the tree is still"
    else:
        tag, phrase = "💤", "dormant — no activity"
    return f"{tag} {name} — {phrase} (warmth {warm:+.2f}, κ {kappa:.2f}, {n_msgs} events)"


# ---------------------------------------------------------------------- #
# Registration — one namespace, seven rooms                              #
# ---------------------------------------------------------------------- #
# Overwrite the forward placeholders in space.py so "agent", "human_bot",
# "async", and "doc" resolve to their real adapters.
AdapterRegistry.register("agent", AgentSpace)
AdapterRegistry.register("human_bot", HumanBotSpace)
AdapterRegistry.register("async", AsyncSpace)
AdapterRegistry.register("doc", DocSpace)
