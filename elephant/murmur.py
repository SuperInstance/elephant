"""murmur — the shadow writer. The cave wall writes itself.

The captain's Terrain reframing (`docs/terrain-2026-08-17.md`): the true
state — the weighting and vectoring of every token, the whole field of
the room — is the **Terrain**. What a human (or an agent) actually sees
is the **Shadow**: witness marks of the terrain's activities. Murmur's
commits ARE shadows on the cave wall — witness marks of an agent's
internal monologue.

Murmur-agent (`/home/eileen/projects/murmur-agent`) is the all-night
thinking git-agent: every thought becomes a git commit, every insight a
file. This module is murmur's soul in elephant idiom — a self-populating
**shadow journal** where:

- every internal monologue becomes a **commit** (a witness mark), and
- the commit **carries its JEPA readings as terrain front-matter** — the
  shadow with enough terrain context to agree on the action.

`MurmurJournal` — a git-backed (or plain-dir) journal. `write_monologue`
writes a thought as a file with its readings attached as front-matter
and commits it (if the path is inside a git repo). `read_room` renders
the journal as a `Room` — the shadow-trail as a room the elephant reads
with the DialBank, or walks into through the `MurmurSpace` doc adapter
(DocSpace-style: commits as messages, tint target = a status line).
`retrieve` finds entries by feeling — "when did murmur last feel
like this?" — with deadband-gate ranges and target distances over the
terrain front-matter; when `elephant/jepa_rag.py` exists, `to_memory()`
indexes the witness marks into a `JepaMemory` and `retrieve_feeling()`
retrieves by vibe.

The seam (`murmurize_pulse`): PulseLoop's `internal_monologue()` runs on
every pulse even when the agent is silent; the seam takes that silent
thinking and commits it WITH that pulse's perception readings — the
raw readings, direction (last two), rate of change (last three+),
warmth and its kinematics. The elephant gives murmur its senses: the
thinking is informed by the terrain, and the wall writes itself.

The reverse seam (`read_room` / `MurmurSpace` / a plain `DocSpace` over
the journal's git log): the elephant reads murmur's history as a room —
the shadow-trail, with its terrain front-matter intact.

The shadow is not the thinking. The shadow is the witness. Enough
information to agree on the action.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .field import RoomField, read_field
from .room import Message, Room
from .space import Space, _field_traits

__all__ = [
    "MurmurJournal", "MurmurSpace",
    "murmurize_pulse", "pulse_profile",
    "front_matter", "parse_entry",
]

_GIT_TIMEOUT = 10.0
_ENTRY_RE = re.compile(r"^\d{4}-.*\.md$")


# ---------------------------------------------------------------------- #
# Front-matter — the terrain attached to each witness mark               #
# ---------------------------------------------------------------------- #
def front_matter(meta: Dict[str, Any]) -> str:
    """Serialize an entry's terrain metadata as a front-matter block.

    Every value is JSON-encoded so nested dicts (the readings profile)
    round-trip exactly. The block is the ``--- ... ---`` header of each
    witness file — the shadow with its terrain context.
    """
    lines = ["---"]
    for k in sorted(meta):
        lines.append(f"{k}: {json.dumps(_clean_json(meta[k]))}")
    lines.append("---")
    return "\n".join(lines)


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def _clean_json(value: Any) -> Any:
    """Round floats for the witness file — the noise floor is 0.02, so
    six decimals of terrain is more than enough fidelity."""
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: _clean_json(v) for k, v in value.items()}
    return value


def parse_entry(path: str) -> Optional[Dict[str, Any]]:
    """Parse one witness file into an entry dict.

    Returns None for malformed files (no front-matter block). The entry
    carries the front-matter metadata plus ``text`` (the monologue body)
    and ``path``.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
    except OSError:
        return None
    if not lines or lines[0].strip() != "---":
        return None
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return None
    meta: Dict[str, Any] = {}
    for ln in lines[1:end]:
        if ":" not in ln:
            continue
        k, v = ln.split(":", 1)
        meta[k.strip()] = _parse_value(v.strip())
    entry = dict(meta)
    entry["text"] = "\n".join(lines[end + 1:]).strip()
    entry["path"] = path
    return entry


# ---------------------------------------------------------------------- #
# MurmurJournal — the self-populating shadow journal                     #
# ---------------------------------------------------------------------- #
class MurmurJournal:
    """A git-backed (or plain-dir) journal of witness marks.

    Every entry = one internal monologue thought + its terrain
    front-matter (readings, ts, space_id, topic — and, from a pulse,
    direction / rate of change / warmth kinematics). If ``path`` is
    inside a git repo, each entry is committed — murmur's pattern:
    every thought a commit. If not, the journal is a plain directory of
    dated, indexed files (the /tmp overnight case).

    The journal IS a cave wall: it does not think; it witnesses. The
    elephant reads it as a room (`read_room`), walks into it through
    `MurmurSpace` (a DocSpace-style adapter), or retrieves by feeling
    (`retrieve`).
    """

    def __init__(self, path: str, space_id: str = "murmur"):
        self.path = os.path.abspath(path)
        self.space_id = space_id
        os.makedirs(self.path, exist_ok=True)
        self._git: Optional[bool] = None
        self._mem_cls = _load_jepa_memory()
        self._mem = None

    # ------------------------------------------------------------------ #
    # Writing — the wall writes itself                                   #
    # ------------------------------------------------------------------ #
    def write_monologue(self, text: str, readings: Dict[str, float],
                        ts: Optional[float] = None, topic: str = "pulse",
                        *,
                        space_id: Optional[str] = None,
                        direction: Optional[Dict[str, float]] = None,
                        rate_of_change: Optional[Dict[str, float]] = None,
                        warmth: Optional[float] = None,
                        warmth_direction: Optional[float] = None,
                        warmth_rate: Optional[float] = None,
                        confidence: Optional[float] = None,
                        commit: bool = True) -> Dict[str, Any]:
        """Commit one thought as a witness mark.

        ``text`` is the monologue; ``readings`` are the JEPA readings
        the thought was born under (the terrain context). Returns the
        entry dict (identical to what ``entries()`` and ``read_room()``
        will recover).
        """
        ts = float(ts) if ts is not None else time.time()
        idx = self._next_index()
        slug = _slug(topic)
        rel = f"{idx:04d}-{slug}.md"
        abs_path = os.path.join(self.path, rel)

        meta: Dict[str, Any] = {
            "index": idx,
            "ts": ts,
            "space_id": space_id or self.space_id,
            "topic": topic,
            "readings": dict(readings),
        }
        if direction is not None:
            meta["direction"] = dict(direction)
        if rate_of_change is not None:
            meta["rate_of_change"] = dict(rate_of_change)
        for k, v in (("warmth", warmth), ("warmth_direction", warmth_direction),
                     ("warmth_rate", warmth_rate)):
            if v is not None:
                meta[k] = float(v)
        if confidence is not None:
            meta["confidence"] = float(confidence)

        body = str(text).strip() or "(silence)"
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(front_matter(meta) + "\n\n" + body + "\n")

        if commit and self.git_enabled:
            self._commit(rel, f"murmur: {topic}")

        entry = dict(meta)
        entry["text"] = body
        entry["path"] = abs_path
        return entry

    def _commit(self, rel: str, message: str) -> bool:
        """Best-effort git commit of one witness mark. Never raises —
        a plain-dir journal is always a valid fallback. The repo is
        addressed explicitly via ``-C`` and inherited ``GIT_DIR``/
        ``GIT_WORK_TREE`` env is cleared so no ambient git state can
        hijack the commit."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("GIT_DIR", "GIT_WORK_TREE")}
        try:
            subprocess.run(["git", "-C", self.path, "add", "--", rel],
                           capture_output=True, text=True, env=env,
                           timeout=_GIT_TIMEOUT, check=True)
            subprocess.run(["git", "-C", self.path, "commit", "-q",
                            "-m", message],
                           capture_output=True, text=True, env=env,
                           timeout=_GIT_TIMEOUT, check=True)
            return True
        except (subprocess.SubprocessError, OSError):
            return False

    # ------------------------------------------------------------------ #
    # Reading — the elephant reads the shadow-trail                      #
    # ------------------------------------------------------------------ #
    def entries(self) -> List[Dict[str, Any]]:
        """All witness marks, in commit order (by index)."""
        out: List[Dict[str, Any]] = []
        for fn in sorted(os.listdir(self.path)):
            if not _ENTRY_RE.match(fn) and not fn.endswith(".md"):
                continue
            entry = parse_entry(os.path.join(self.path, fn))
            if entry is not None and isinstance(entry.get("index"), int):
                out.append(entry)
        out.sort(key=lambda e: (e["index"], e.get("ts", 0.0)))
        return out

    def read_room(self) -> Room:
        """The journal as a room: every witness mark becomes a
        Message(author="murmur", text=..., ts=...). The elephant reads
        the shadow-trail with the DialBank like any other room; each
        message's channel is its topic."""
        room = Room(f"murmur/{self.space_id}")
        for e in self.entries():
            room.messages.append(Message(
                author="murmur", text=e["text"], ts=e.get("ts", 0.0),
                channel=str(e.get("topic", "pulse")),
            ))
        room.messages.sort(key=lambda m: m.ts)
        return room

    def read_field(self) -> RoomField:
        """Convenience: the shadow-trail's field in one call."""
        return read_field(self.read_room())

    # ------------------------------------------------------------------ #
    # Retrieval — by feeling, "when did murmur last feel like this?"     #
    # ------------------------------------------------------------------ #
    def retrieve(self, query: Dict[str, Any],
                 topic: Optional[str] = None,
                 ts_range: Optional[Tuple[float, float]] = None,
                 k: int = 1,
                 weights: Optional[Dict[str, float]] = None,
                 threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Find the witness marks that felt the most like ``query``.

        ``query`` maps reading names to a target value (``{"panic": 0.7}``)
        or a range (``{"panic": (0.5, 1.0)}`` — "did panic ever cross the
        band?"). **Ranges are deadband gates**: an entry outside the band
        does not ring — it is excluded (unless ``threshold`` is given,
        which admits anything within that distance of the band). Scalar
        targets rank the surviving trail by nearness. The profile searched
        is the full terrain front-matter: ``readings`` plus ``warmth*``
        plus per-dial ``d_*`` (direction) and ``r_*`` (rate of change), so
        you can retrieve by feeling — "a night like this one" — or by
        movement ("the spike").

        The deterministic path — the deadband rings on LEVELS. When
        `elephant/jepa_rag.py` exists, `to_memory()` indexes the
        witness marks into a `JepaMemory` and `retrieve_feeling()`
        retrieves by vibe (cosine in reading space) — the two can
        disagree, and that is the difference between feeling and
        ringing.

        Returns up to ``k`` entries sorted by distance (each a copy with
        ``_distance`` attached). ``topic`` (substring) and ``ts_range``
        narrow the trail first; ``threshold`` drops entries farther than
        a bound.

        This is the deterministic path — the deadband rings on LEVELS.
        For vibe similarity through the JEPA-RAG memory (when
        `elephant/jepa_rag.py` exists), use `to_memory()` +
        `retrieve_feeling()`.
        """
        return self._retrieve_plain(query, topic=topic, ts_range=ts_range,
                                    k=k, weights=weights, threshold=threshold)

    def to_memory(self):
        """Index the journal's witness marks into a JepaMemory (from
        `elephant/jepa_rag.py`, when present) and return it.

        Every witness mark becomes a moment — the shadow (text) with its
        terrain context (readings, ts, space_id, and topic/index in
        meta) — exactly the shape the memory ingests. Returns None when
        the memory module is absent.
        """
        if self._mem_cls is None:
            return None
        if self._mem is None:
            self._mem = self._mem_cls()
            for e in self.entries():
                self._mem.ingest({
                    "text": e["text"],
                    "readings": e.get("readings") or {},
                    "ts": e.get("ts", 0.0),
                    "space_id": e.get("space_id", "unspecified"),
                    "meta": {"topic": e.get("topic"),
                             "index": e.get("index")},
                })
        return self._mem

    def retrieve_feeling(self, query: Dict[str, float], k: int = 5,
                         ) -> List[Dict[str, Any]]:
        """Retrieval BY VIBE — through the JEPA-RAG memory, when present.

        The memory ranks by cosine similarity in reading space ("the
        moment that felt most like this"), which can disagree with the
        deadband's LEVEL semantics: a low-panic quiet room whose only
        signal is panic can out-rank a loud warm room that also carries
        a little panic. That is the difference between feeling and
        ringing. Falls back to `retrieve()` when no memory exists.
        """
        mem = self.to_memory()
        if mem is None:
            return self.retrieve(query, k=k)
        try:
            hits = mem.query_readings(dict(query), top_k=max(1, int(k)))
        except Exception:
            return self.retrieve(query, k=k)
        return [{
            "text": h.text,
            "readings": dict(h.readings),
            "ts": h.ts,
            "space_id": h.space_id,
            "topic": (h.meta or {}).get("topic"),
            "index": (h.meta or {}).get("index"),
            "_distance": 1.0 - float(h.score),
        } for h in hits]

    def _retrieve_plain(self, query, *, topic, ts_range, k, weights,
                        threshold):
        trail = self.entries()
        if topic:
            t = topic.lower()
            trail = [e for e in trail if t in str(e.get("topic", "")).lower()]
        if ts_range is not None:
            lo, hi = float(ts_range[0]), float(ts_range[1])
            trail = [e for e in trail if lo <= e.get("ts", 0.0) <= hi]
        if not trail:
            return []

        profiles = [self.profile(e) for e in trail]
        scales = {}
        for key in query:
            vals = [p.get(key) for p in profiles if isinstance(p.get(key), (int, float))]
            if not vals:
                continue
            rng = max(vals) - min(vals)
            scales[key] = rng if rng > 1e-12 else 1.0

        weighted = []
        for e, p in zip(trail, profiles):
            dist, excluded = _query_distance(query, p, scales, weights,
                                             range_gate=threshold is None)
            if excluded:
                continue          # outside the band — it does not ring
            if threshold is not None and dist > threshold:
                continue
            weighted.append((dist, e))
        weighted.sort(key=lambda pair: (pair[0], pair[1].get("index", 0)))
        return [dict(e, _distance=d) for d, e in weighted[:max(0, int(k))]]

    def profile(self, entry: Dict[str, Any]) -> Dict[str, float]:
        """The entry's full terrain profile — the merged vector
        ``retrieve`` searches. Readings plus warmth kinematics plus
        per-dial direction (``d_*``) and rate (``r_*``)."""
        p: Dict[str, float] = {}
        for k, v in (entry.get("readings") or {}).items():
            p[str(k)] = _finite(float(v))
        for k in ("warmth", "warmth_direction", "warmth_rate"):
            v = entry.get(k)
            if v is not None:
                p[k] = _finite(float(v))
        for k, v in (entry.get("direction") or {}).items():
            p[f"d_{k}"] = _finite(float(v))
        for k, v in (entry.get("rate_of_change") or {}).items():
            p[f"r_{k}"] = _finite(float(v))
        return p

    # ------------------------------------------------------------------ #
    # Plumbing                                                           #
    # ------------------------------------------------------------------ #
    def _next_index(self) -> int:
        indices = [e["index"] for e in self.entries()]
        return (max(indices) + 1) if indices else 1

    @property
    def git_enabled(self) -> bool:
        """True when ``path`` sits inside a git repo (each witness mark
        is committed)."""
        if self._git is None:
            try:
                proc = subprocess.run(
                    ["git", "-C", self.path, "rev-parse",
                     "--is-inside-work-tree"],
                    capture_output=True, text=True, timeout=_GIT_TIMEOUT)
                self._git = proc.returncode == 0 and proc.stdout.strip() == "true"
            except (subprocess.SubprocessError, OSError):
                self._git = False
        return self._git

    def __len__(self) -> int:
        return len(self.entries())

    def __repr__(self) -> str:
        return (f"<MurmurJournal {self.space_id!r} path={self.path!r} "
                f"witnesses={len(self)} git={self.git_enabled}>")


def _slug(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(topic).lower()).strip("-")[:40]
    return slug or "monologue"


def _finite(v: float) -> float:
    """Non-finite readings read as 0 in a retrieval profile — a glitch
    is NOT a movement (the pulse math's rule), and a NaN must never
    poison the ordering of a query."""
    return v if math.isfinite(v) else 0.0


def _query_distance(query: Dict[str, Any], profile: Dict[str, float],
                    scales: Dict[str, float],
                    weights: Optional[Dict[str, float]],
                    range_gate: bool = True) -> Tuple[float, bool]:
    """Distance + gate between a query and one entry's profile.

    A scalar query value is a target (distance = |v − target|); a
    ``(lo, hi)`` pair is a band — distance 0 inside, distance to the
    nearest edge outside, and ``excluded=True`` when the entry is
    outside a band (the deadband: inside the band it doesn't ring).
    Per-key distances are normalized by the trail's observed scale so
    panic 0.9 counts like mood 0.9.
    """
    total = 0.0
    wsum = 0.0
    excluded = False
    for key, want in query.items():
        got = profile.get(key)
        if not isinstance(got, (int, float)):
            continue
        w = float((weights or {}).get(key, 1.0))
        if isinstance(want, (tuple, list)) and len(want) == 2:
            lo, hi = float(want[0]), float(want[1])
            if lo <= got <= hi:
                d = 0.0
            else:
                d = min(abs(got - lo), abs(got - hi))
                if range_gate:
                    excluded = True
        else:
            d = abs(got - float(want))
        scale = scales.get(key, 1.0)
        total += w * d / scale
        wsum += w
    if wsum <= 0:
        return float("inf"), False
    return total / wsum, excluded


# ---------------------------------------------------------------------- #
# The learned-memory hook — optional, jepa_rag.py if it ever lands       #
# ---------------------------------------------------------------------- #
def _load_jepa_memory():
    """The JEPA-RAG memory (`elephant/jepa_rag.py`), when present.

    Murmur's git history is a perfect memory to index: the seam is
    wired so that when a `JepaMemory` class exists, pure reading
    queries on ``retrieve`` go through it (moments = witness marks),
    and everything else falls back to the deterministic range filter.
    """
    try:
        from elephant.jepa_rag import JepaMemory  # type: ignore[import-not-found]
        return JepaMemory
    except Exception:
        return None


# ---------------------------------------------------------------------- #
# THE SEAM — the pulse feeds the wall                                    #
# ---------------------------------------------------------------------- #
def pulse_profile(report, readings: Optional[List[Dict[str, float]]] = None,
                  ) -> Tuple[Dict[str, float], Dict[str, float],
                             Dict[str, float], Optional[float],
                             Optional[float], Optional[float]]:
    """One pulse's perception bundle, split for the journal front-matter.

    Returns ``(readings, direction, rate_of_change, warmth,
    warmth_direction, warmth_rate)`` — the terrain context a pulse's
    monologue was born under: the raw dial readings (the numbers that
    don't matter individually) plus the macro read (direction from the
    last two, rate from the last three+, warmth kinematics).
    """
    readings = dict(readings[-1]) if readings else {}
    direction = dict(getattr(report, "direction", {}) or {})
    rate = dict(getattr(report, "rate_of_change", {}) or {})
    warmth = getattr(report, "warmth", None)
    wd = getattr(report, "warmth_direction", None)
    wr = getattr(report, "warmth_rate", None)
    return readings, direction, rate, warmth, wd, wr


def murmurize_pulse(loop, journal: MurmurJournal,
                    topic: Optional[str] = None, prompt: Optional[str] = None,
                    ts: Optional[float] = None, commit: bool = True,
                    ) -> Optional[Dict[str, Any]]:
    """THE SEAM: one pulse's silence becomes a witness mark.

    Takes a `PulseLoop`'s latest perception check, runs its internal
    monologue (the thinking that happens even when the agent says
    nothing), and commits it to ``journal`` WITH that pulse's perception
    readings as terrain front-matter — the raw readings, the direction
    (last two readings), the rate of change (last three+, the second
    difference), warmth and its kinematics.

    This is the self-populating sustaining system the captain named:
    murmur + elephant = the cave wall writing itself, informed by the
    terrain. Each witness carries a ``confidence`` — how loud the
    loudest hand was this pulse (the prisoner's estimate of how well it
    saw the shadow). Returns the written entry (None before the loop's
    first tick).
    """
    report = loop.last_report()
    # The seam is self-sufficient: fire the pulse if none has happened
    # yet, or if the caller's clock has advanced past a period. A pulse
    # is the heartbeat — the wall writes on the beat, never off it.
    if report is None:
        loop.tick(ts)
    elif ts is not None and loop.due(ts):
        loop.tick(ts)
    report = loop.last_report()
    if report is None:
        return None
    mono = loop.internal_monologue(prompt=prompt)
    readings, direction, rate, warmth, wd, wr = pulse_profile(
        report, loop.last_readings())
    # Confidence = how loud the loudest hand was this pulse: the biggest
    # per-dial movement (a move above the noise floor is a real read; a
    # flat room is a low-confidence read). The witness knows how sure it
    # is — the prisoner's estimate of how well it saw the shadow.
    loudest = max((abs(v) for v in direction.values()), default=0.0)
    confidence = min(1.0, loudest)
    return journal.write_monologue(
        mono,
        readings=readings,
        ts=report.ts if ts is None else float(ts),
        topic=topic or "pulse",
        space_id=journal.space_id,
        direction=direction,
        rate_of_change=rate,
        warmth=warmth,
        warmth_direction=wd,
        warmth_rate=wr,
        confidence=confidence,
        commit=commit,
    )


# ---------------------------------------------------------------------- #
# MurmurSpace — the shadow-trail as a room the elephant walks into       #
# ---------------------------------------------------------------------- #
def _murmur_tint(name: str, field: RoomField, n_entries: int) -> str:
    """A status line for the thinking journal, keyed to its field."""
    warm, kappa, _mood, _joke, panic, _volume = _field_traits(field)
    if panic >= 0.5:
        tag, phrase = "🌋", "the journal is on fire — a panic witness is ringing"
    elif warm >= 0.25:
        tag, phrase = "🫧", "the wall is bright — warm thoughts bubbling all night"
    elif warm >= 0.0:
        tag, phrase = "🕯", "the shadows are steady — a long, quiet think"
    elif warm >= -0.25:
        tag, phrase = "🌫", "the thinking has gone still and grey"
    else:
        tag, phrase = "❄", "a cold night — the wall is blank and sharp"
    return (f"{tag} {name} — {phrase} (warmth {warm:+.2f}, κ {kappa:.2f}, "
            f"{n_entries} witnesses)")


class MurmurSpace(Space):
    """The shadow-trail as a DocSpace-style room.

    Read-only adapter: the journal's witness marks become messages
    (author ``"murmur"``, text = the monologue, ts = the pulse time,
    channel = the topic), exactly like `DocSpace` renders a repo's
    commits. The tint target is a project status line — the elephant's
    readout of what the thinking felt like, written back in the doc
    idiom. Re-renders from the journal on every ``room`` access, so the
    wall is always current.
    """

    kind = "doc"

    def __init__(self, journal: MurmurJournal):
        super().__init__(f"murmur/{journal.space_id}")
        self.journal = journal
        self.status = f"{self.name} — no witnesses yet"

    def ingest(self, *events) -> "MurmurSpace":
        # Read-only: the room always re-renders from the journal.
        return self

    @property
    def room(self) -> Room:
        return self.journal.read_room()

    def tint_target(self) -> str:
        return "a project status line"

    def tint(self, field: RoomField) -> str:
        return _murmur_tint(self.name, field, len(self.journal))

    def send_back(self, field: RoomField, tinted_text: Optional[str] = None) -> str:
        text = super().send_back(field, tinted_text)
        self.status = text
        return text
