"""JEPA-RAG — retrieval where JEPA readings are first-class citizens.

The captain's directive (exact words):

    "Think about a RAG with Jepa readings as first-class citizens
    along side time and space stamps."

A normal RAG indexes text embeddings and retrieves similar text. This
system indexes MOMENTS instead. A moment is a shadow with its terrain
context:

    text      — the shadow: the witness words (a transcript chunk, a
                 bar line, a watch log, a speech)
    readings  — the JEPA reading vector: what the room FELT like
                 (mood, volume, panic, ... — the dial bank's senses)
    ts        — the time stamp: when it happened
    space_id  — the space stamp: which room it happened in
    meta      — anything else worth riding along

You can query it by TEXT ("what did the room feel like during the
fight?") — the normal way. By READING ("moments where mood > 0.6 and
panic < 0.2") — the JEPA reading as a FIRST-CLASS retrieval
dimension. By NEAR-FIELD ("find the moment that felt most like right
now") — nearest neighbor in JEPA space, the perfume that takes you to
grandma's shop. By TIME and by SPACE — the stamps as dimensions too.
Or by any weighted combination of all four: the captain's "alongside"
made concrete.

The answer to a query is a WITNESS with its terrain context — the
shadow plus the reading vector, ts, and space — enough to agree on
the action (docs/terrain-2026-08-17.md). The reading vector rides
along on every hit; it is not metadata, it is the citizen.

The math is deliberately small + honest: numpy matrices and
bag-of-words cosine. No learned embeddings, no vector database — a
few dozen moments, nine dials of meaning, all in plain arrays.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .dial import DialBank
from .dials import DEFAULT_DIALS
from .field import RoomField
from .room import Message, Room

_WORD_RE = re.compile(r"\w+")

# The JEPA dials — the retrieval dimensions. Order matters: it is the
# vector layout every moment is stored and queried in.
JEPA_DIAL_NAMES: List[str] = [d.name for d in DEFAULT_DIALS]

# Default combined-query weights: readings are the heaviest citizen,
# text next, time and space stamps alongside (the captain's "beside").
DEFAULT_WEIGHTS = {"text": 0.3, "readings": 0.5, "time": 0.1, "space": 0.1}

__all__ = [
    "JEPA_DIAL_NAMES",
    "DEFAULT_WEIGHTS",
    "MomentHit",
    "JepaMemory",
    "moment_from_text",
    "moment_from_room",
    "readings_from_text",
    "moments_from_markdown",
]


# ---------------------------------------------------------------------- #
# Small honest primitives                                                #
# ---------------------------------------------------------------------- #
def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def readings_from_text(text: str, bank: Optional[DialBank] = None) -> Dict[str, float]:
    """Run the JEPA dial bank over a piece of text -> a reading dict.

    The readings are COMPUTED, not hand-set: a witness earns its
    vector by being read. One message can only carry so much (no
    density, no ripple) — for whole conversations use
    `moment_from_room`.
    """
    bank = bank or DialBank(DEFAULT_DIALS)
    room = Room("witness")
    room.messages.append(Message(author="[witness]", text=str(text), ts=0.0))
    return bank.readings(room)


def moment_from_text(
    text: str,
    space_id: str = "unspecified",
    ts: float = 0.0,
    meta: Optional[Dict[str, Any]] = None,
    bank: Optional[DialBank] = None,
) -> Dict[str, Any]:
    """One moment from text: the shadow + its JEPA readings."""
    return {
        "text": str(text).strip(),
        "readings": readings_from_text(text, bank=bank),
        "ts": float(ts),
        "space_id": str(space_id),
        "meta": dict(meta or {}),
    }


def moment_from_room(
    room: Room,
    space_id: str = "unspecified",
    ts: Optional[float] = None,
    meta: Optional[Dict[str, Any]] = None,
    bank: Optional[DialBank] = None,
) -> Dict[str, Any]:
    """Read a whole Room through the dial bank -> one moment.

    For moments that are a whole conversation (a fight, a watch with
    several voices): the dials feel density, ripple, and presence,
    which a single message cannot carry. The shadow is the room's
    message trail joined as one witness.
    """
    bank = bank or DialBank(DEFAULT_DIALS)
    if ts is None:
        ts = room.messages[-1].ts if room.messages else 0.0
    return {
        "text": "\n".join(m.text for m in room.messages).strip(),
        "readings": bank.readings(room),
        "ts": float(ts),
        "space_id": str(space_id),
        "meta": dict(meta or {}),
    }


def _split_markdown_chunks(text: str, max_chars: int = 1600) -> List[str]:
    """Chunk markdown into witness-sized pieces.

    A heading starts a new chunk; a long section is broken on
    paragraph (blank-line) boundaries, so a chunk stays one readable
    moment. The heading leads its section's first chunk only.
    """
    lines = text.splitlines()
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    heading: Optional[str] = None
    heading_used = False
    pending_blank = 0
    for line in lines:
        if line.startswith("#"):
            if current:
                chunks.append("\n".join(current).strip())
            heading = line
            heading_used = False
            current = [line]
            current_len = len(line)
            pending_blank = 0
            continue
        if not line.strip():
            pending_blank += 1
            continue
        add = len(line) + 1
        if current_len + add > max_chars and current:
            current.extend([""] * pending_blank)
            chunks.append("\n".join(current).strip())
            current = [heading] if (heading and not heading_used) else []
            if heading:
                heading_used = True
            current_len = len(current[0]) if current else 0
            pending_blank = 0
        current.extend([""] * pending_blank)
        current.append(line)
        current_len += add + pending_blank
        pending_blank = 0
    if current:
        chunks.append("\n".join(current).strip())
    return [c for c in chunks if c and c.strip()]


def moments_from_markdown(
    path: str,
    space_id: str,
    base_ts: float = 0.0,
    step: float = 300.0,
    bank: Optional[DialBank] = None,
    max_chars: int = 1600,
) -> List[Dict[str, Any]]:
    """Chunk a markdown transcript into moments.

    Each chunk becomes a moment with its JEPA readings computed by
    the dial bank. ts = base_ts + chunk_index * step, so a file's
    moments keep their order as a stamp dimension.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    chunks = _split_markdown_chunks(text, max_chars=max_chars)
    moments = []
    for i, chunk in enumerate(chunks):
        moments.append(moment_from_text(
            chunk,
            space_id=space_id,
            ts=base_ts + i * step,
            meta={"source": os.path.basename(path), "chunk": i},
            bank=bank,
        ))
    return moments


# ---------------------------------------------------------------------- #
# The hit — a witness WITH its terrain context                           #
# ---------------------------------------------------------------------- #
@dataclass
class MomentHit:
    """One retrieved moment: the shadow with its terrain context.

    The JEPA reading vector rides along on every hit — it is a
    first-class citizen, not metadata. `score` is the retrieval score
    of the query that produced the hit; `vector` is the moment's
    reading vector in JEPA_DIAL_NAMES order.
    """

    text: str
    readings: Dict[str, float]
    ts: float
    space_id: str
    meta: Dict[str, Any]
    score: float
    vector: np.ndarray
    index: int = -1

    @property
    def field(self) -> RoomField:
        return RoomField(self.readings)

    @property
    def warmth(self) -> float:
        return self.field.warmth()

    def board(self) -> str:
        """One-line witness readout: score, warmth, κ, key dials."""
        r = self.readings
        return (
            f"score {self.score:+.3f}  warmth {self.warmth:+.2f}  "
            f"κ {self.field.concentration():.2f}  "
            f"mood {r.get('mood', 0.0):+.2f}  panic {r.get('panic', 0.0):.2f}  "
            f"@ {self.space_id}  ts {self.ts:.0f}"
        )

    def reading_line(self) -> str:
        """The reading vector in dial order — the citizen, printed."""
        return "  ".join(
            f"{name}={self.readings.get(name, 0.0):+.2f}"
            for name in JEPA_DIAL_NAMES
        )


# ---------------------------------------------------------------------- #
# The memory                                                             #
# ---------------------------------------------------------------------- #
class JepaMemory:
    """The elephant's memory: moments indexed by FEELING.

    Stores moments; `index()` builds the numpy retrieval structures (a
    reading-vector matrix, a field matrix, a timestamp array, a space
    array, and a bag-of-words term-frequency matrix). Queries are
    ranked lists of `MomentHit` — the shadow + its terrain context.

    The index rebuilds lazily whenever moments change, so ingest-then-
    query just works.
    """

    def __init__(self, dial_names: Optional[Sequence[str]] = None):
        self.dial_names: List[str] = (
            list(dial_names) if dial_names is not None else list(JEPA_DIAL_NAMES)
        )
        self._moments: List[Dict[str, Any]] = []
        self._dirty = True

        # index structures (built by index())
        self._vectors = np.zeros((0, len(self.dial_names)))
        self._unit = np.zeros((0, len(self.dial_names)))
        self._fields = np.zeros((0, 2))          # [warmth, concentration]
        self._ts = np.zeros(0)
        self._spaces = np.array([], dtype=object)
        self._vocab: List[str] = []
        self._vocab_index: Dict[str, int] = {}
        self._tf = np.zeros((0, 0))

    # ------------------------------------------------------------------ #
    # Ingest                                                             #
    # ------------------------------------------------------------------ #
    def ingest(self, moment: Dict[str, Any]) -> "JepaMemory":
        """Store one moment: {text, readings, ts, space_id, meta}.

        `readings` may be a partial dict (missing dials read 0.0 —
        the vector fills the gap); `ts` defaults to 0.0; `space_id`
        defaults to "unspecified". The shadow must not be empty.
        """
        text = str(moment.get("text", ""))
        if not text.strip():
            raise ValueError("a moment needs a shadow — text must be non-empty")
        self._moments.append({
            "text": text,
            "readings": dict(moment.get("readings") or {}),
            "ts": float(moment.get("ts", 0.0)),
            "space_id": str(moment.get("space_id", "unspecified")),
            "meta": dict(moment.get("meta") or {}),
        })
        self._dirty = True
        return self

    # ------------------------------------------------------------------ #
    # Index                                                              #
    # ------------------------------------------------------------------ #
    def index(self) -> "JepaMemory":
        """Build the retrieval structures from the stored moments.

        numpy only: a moments matrix (reading vectors), a field matrix
        (warmth, κ), a timestamp array, a space array, and a
        bag-of-words TF matrix for lexical retrieval.
        """
        n = len(self._moments)
        d = len(self.dial_names)
        self._vectors = np.zeros((n, d))
        self._unit = np.zeros((n, d))
        self._fields = np.zeros((n, 2))
        self._ts = np.zeros(n)
        self._spaces = np.empty(n, dtype=object)
        for i, m in enumerate(self._moments):
            field = RoomField(m["readings"])
            vec = field.vector(self.dial_names)
            self._vectors[i] = vec
            norm = float(np.linalg.norm(vec))
            if norm > 1e-12:
                self._unit[i] = vec / norm
            self._fields[i] = (field.warmth(), field.concentration())
            self._ts[i] = float(m["ts"])
            self._spaces[i] = m["space_id"]

        # bag-of-words TF matrix — the lexical retrieval dimension
        vocab: Dict[str, int] = {}
        rows: List[Dict[str, float]] = []
        for m in self._moments:
            counts: Dict[str, float] = {}
            for t in _tokenize(m["text"]):
                counts[t] = counts.get(t, 0.0) + 1.0
            for t in counts:
                if t not in vocab:
                    vocab[t] = len(vocab)
            rows.append(counts)
        self._vocab = [""] * len(vocab)
        for t, j in vocab.items():
            self._vocab[j] = t
        self._vocab_index = dict(vocab)
        self._tf = np.zeros((n, len(vocab)))
        for i, counts in enumerate(rows):
            for t, c in counts.items():
                self._tf[i, vocab[t]] = c

        self._dirty = False
        return self

    def _ensure_indexed(self) -> None:
        if self._dirty:
            self.index()

    # ------------------------------------------------------------------ #
    # Query: text — the normal way                                       #
    # ------------------------------------------------------------------ #
    def query_text(self, q: str, top_k: int = 5) -> List[MomentHit]:
        """Lexical retrieval: bag-of-words cosine against the shadows.

        The normal RAG way — what the words say. For what the room
        FELT, use `query_readings`; the feeling is the first-class
        citizen here.
        """
        self._ensure_indexed()
        if not self._moments or not (q or "").strip():
            return []
        scores = self._text_scores(str(q))
        # a moment with zero lexical overlap is not a hit — no shared
        # words means no evidence, so it does not rank
        scores = np.where(scores > 0.0, scores, -np.inf)
        return self._ranked(scores, top_k)

    # ------------------------------------------------------------------ #
    # Query: readings — the first-class-citizen query                    #
    # ------------------------------------------------------------------ #
    def query_readings(
        self, readings: Union[Dict[str, Any], RoomField, np.ndarray],
        top_k: int = 5,
    ) -> List[MomentHit]:
        """EXACT READING QUERY — rank by closeness in JEPA space.

        The first-class-citizen query: "moments where mood > 0.6 and
        panic < 0.2". Two profiles are accepted:

        - a plain dict of dial -> target (float): ranked by cosine
          similarity in reading space (the raw cosine — negative
          means anti-aligned, and that is honest). Unspecified dials
          read 0.0 — the vector's origin, exactly like
          RoomField.vector().
        - a dict of dial -> (lo, hi) RANGE constraints: ranked by the
          fraction of constraints the moment satisfies — the
          captain's "mood > 0.6, panic < 0.2" made literal.

        `readings` may also be a `RoomField` or a vector in dial
        order. top_k=None returns every moment, ranked.
        """
        self._ensure_indexed()
        if not self._moments:
            return []
        if isinstance(readings, dict) and any(
                isinstance(v, (tuple, list)) for v in readings.values()):
            return self._query_readings_constraints(readings, top_k)
        q = self._coerce_reading_vector(readings)
        qn = float(np.linalg.norm(q))
        if qn < 1e-12:
            return []
        return self._ranked(self._unit @ (q / qn), top_k)

    def _query_readings_constraints(
        self, profile: Dict[str, Any], top_k: Optional[int]
    ) -> List[MomentHit]:
        """Range-constraint profile: rank by the fraction of dials
        inside their (lo, hi) bounds — the threshold reading query,
        first-class and literal."""
        idx = {name: j for j, name in enumerate(self.dial_names)}
        gates = []
        for name, bounds in profile.items():
            if name in idx and isinstance(bounds, (tuple, list)) \
                    and len(bounds) == 2:
                lo, hi = bounds
                gates.append((idx[name], float(lo), float(hi)))
        if not gates:
            return []
        scores = np.zeros(len(self._moments))
        for i in range(len(self._moments)):
            v = self._vectors[i]
            scores[i] = (sum(1.0 for j, lo, hi in gates
                             if lo <= v[j] <= hi) / len(gates))
        return self._ranked(scores, top_k)

    # ------------------------------------------------------------------ #
    # Query: field — the perfume query                                   #
    # ------------------------------------------------------------------ #
    def query_field(
        self, field: Union[RoomField, Dict[str, float], np.ndarray],
        top_k: int = 5,
    ) -> List[MomentHit]:
        """NEAR-FIELD query — the moment that felt most like this one.

        Nearest neighbors in JEPA space to a field (a `RoomField`, a
        readings dict, or a vector): "find the moment that felt most
        like right now" — the perfume that takes you to grandma's
        shop. The field IS the ensemble of readings, so this is the
        same cosine as `query_readings`, named for what it is for.
        """
        return self.query_readings(field, top_k=top_k)

    # ------------------------------------------------------------------ #
    # Query: time and space — the stamps as dimensions                   #
    # ------------------------------------------------------------------ #
    def query_time(self, window: Union[Tuple[float, float], List[float], Dict[str, float], float],
                   top_k: Optional[int] = None) -> List[MomentHit]:
        """TIME query — the stamp as a retrieval dimension.

        `window` is (start, end), {"start": ..., "end": ...}, or a
        single instant (float — exact match). Moments outside the
        window are excluded (this is the hard filter); within it,
        ranked by proximity to the window's center. top_k=None
        returns every moment in the window.
        """
        self._ensure_indexed()
        start, end = self._parse_window(window)
        center = (start + end) / 2.0
        span = max(end - start, 1e-9)
        scores = np.full(len(self._moments), -np.inf)
        for i in range(len(self._moments)):
            t = self._ts[i]
            if start <= t <= end:
                scores[i] = 1.0 - min(abs(t - center) / (span / 2.0), 1.0)
        return self._ranked(scores, top_k)

    def query_space(self, space_id: str, top_k: Optional[int] = None) -> List[MomentHit]:
        """SPACE query — which room? The stamp as a dimension.

        Every moment from that space, ranked newest-first (recency
        score 1.0 -> 0.0). top_k=None returns them all — "what did
        the wheelhouse feel like last week?" starts here.
        """
        self._ensure_indexed()
        idx = [i for i in range(len(self._moments)) if self._spaces[i] == space_id]
        if not idx:
            return []
        ts = self._ts[idx]
        tmax, tmin = float(ts.max()), float(ts.min())
        span = max(tmax - tmin, 1e-9)
        scores = np.full(len(self._moments), -np.inf)
        for i in idx:
            scores[i] = (self._ts[i] - tmin) / span
        return self._ranked(scores, top_k)

    # ------------------------------------------------------------------ #
    # Query: combined — the captain's "alongside" made concrete          #
    # ------------------------------------------------------------------ #
    def query_combined(
        self,
        parts: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
        top_k: int = 5,
    ) -> List[MomentHit]:
        """The full RAG query — text, readings, time, space alongside.

        `parts` keys: "text" (str), "readings" (dict / RoomField /
        vector), "ts" (window), "space" (space_id). `weights` keys:
        "text", "readings", "time" (alias "ts"), "space"; defaults
        0.3 / 0.5 / 0.1 / 0.1 — the reading is the heaviest citizen.

        Weights are renormalized over the dimensions actually present,
        so a pure feeling query (only "readings") ranks on the full
        reading weight. Space and time are SOFT here: a wrong-space
        moment scores 0 on that dimension but can still rank on the
        others (use `query_space` / `query_time` for hard filters).
        Every dimension's score is clipped to [0, 1] so the weights
        mean what they say.
        """
        self._ensure_indexed()
        present = {k: v for k, v in parts.items() if v is not None}
        if not present or not self._moments:
            return []
        w = dict(DEFAULT_WEIGHTS if weights is None else weights)
        wmap: Dict[str, float] = {}
        for k in present:
            wk = "time" if k == "ts" else k
            wmap[k] = w.get(wk, 0.0)
        total = sum(wmap.values())
        if total <= 0:
            return []
        wnorm = {k: v / total for k, v in wmap.items()}

        scores = np.zeros(len(self._moments))
        for k, v in present.items():
            if k == "text":
                s = self._text_scores(str(v))
            elif k == "readings":
                s = self._reading_scores(self._coerce_reading_vector(v))
            elif k == "ts":
                s = self._time_scores(v)
            elif k == "space":
                s = (self._spaces == str(v)).astype(float)
            else:
                continue
            scores += wnorm[k] * s
        return self._ranked(scores, top_k)

    # ------------------------------------------------------------------ #
    # Accessors                                                          #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._moments)

    def __getitem__(self, i: int) -> MomentHit:
        self._ensure_indexed()
        return self._hit(int(i), float("nan"))

    @property
    def moments(self) -> List[Dict[str, Any]]:
        """The stored moments, as ingested (the shadows + readings)."""
        return list(self._moments)

    @property
    def reading_vectors(self) -> np.ndarray:
        """(N, d) moments matrix — the JEPA reading vectors."""
        self._ensure_indexed()
        return self._vectors.copy()

    @property
    def field_matrix(self) -> np.ndarray:
        """(N, 2) field matrix — [warmth, concentration] per moment."""
        self._ensure_indexed()
        return self._fields.copy()

    @property
    def timestamps(self) -> np.ndarray:
        """(N,) time stamps."""
        self._ensure_indexed()
        return self._ts.copy()

    @property
    def space_stamps(self) -> np.ndarray:
        """(N,) space stamps."""
        self._ensure_indexed()
        return self._spaces.copy()

    def spaces(self) -> List[str]:
        """The distinct spaces the elephant remembers."""
        self._ensure_indexed()
        return sorted(set(str(s) for s in self._spaces))

    def summary(self) -> str:
        self._ensure_indexed()
        return (
            f"JepaMemory({len(self)} moments, {len(self.dial_names)} dials, "
            f"{len(self._vocab)} tokens, spaces={self.spaces()})"
        )

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _coerce_reading_vector(self, x: Any) -> np.ndarray:
        if isinstance(x, RoomField):
            return x.vector(self.dial_names)
        if isinstance(x, dict):
            return RoomField(x).vector(self.dial_names)
        return np.asarray(x, dtype=float).reshape(-1)

    def _text_scores(self, q: str) -> np.ndarray:
        """Per-moment bag-of-words cosine with the query, clipped [0,1]."""
        qv = np.zeros(len(self._vocab))
        for t in _tokenize(q):
            j = self._vocab_index.get(t)
            if j is not None:
                qv[j] += 1.0
        s = np.zeros(len(self._moments))
        qn = float(np.linalg.norm(qv))
        if qn < 1e-12:
            return s
        qv = qv / qn
        row_norms = np.linalg.norm(self._tf, axis=1)
        for i in range(len(self._moments)):
            if row_norms[i] > 1e-12:
                s[i] = float(np.dot(self._tf[i] / row_norms[i], qv))
        return np.clip(s, 0.0, 1.0)

    def _reading_scores(self, q: np.ndarray) -> np.ndarray:
        """Per-moment cosine in JEPA space, clipped [0,1]."""
        s = np.zeros(len(self._moments))
        qn = float(np.linalg.norm(q))
        if qn < 1e-12:
            return s
        return np.clip(self._unit @ (q / qn), 0.0, 1.0)

    def _time_scores(self, window: Any) -> np.ndarray:
        """Per-moment proximity to a window center; 0 outside it."""
        start, end = self._parse_window(window)
        center = (start + end) / 2.0
        span = max(end - start, 1e-9)
        s = np.zeros(len(self._moments))
        for i in range(len(self._moments)):
            t = self._ts[i]
            if start <= t <= end:
                s[i] = 1.0 - min(abs(t - center) / (span / 2.0), 1.0)
        return s

    @staticmethod
    def _parse_window(window: Any) -> Tuple[float, float]:
        if isinstance(window, dict):
            return float(window["start"]), float(window["end"])
        if isinstance(window, (tuple, list)) and len(window) == 2:
            return float(window[0]), float(window[1])
        t = float(window)
        return t, t

    def _ranked(self, scores: np.ndarray, top_k: Optional[int]) -> List[MomentHit]:
        order = np.argsort(-scores, kind="stable")
        hits: List[MomentHit] = []
        for i in order:
            s = float(scores[int(i)])
            if s == -np.inf:
                break
            hits.append(self._hit(int(i), s))
            if top_k is not None and len(hits) >= top_k:
                break
        return hits

    def _hit(self, i: int, score: float) -> MomentHit:
        m = self._moments[i]
        vec = self._vectors[i]
        # The reading vector is the citizen — carry the full canonical
        # vector even if the moment was ingested with partial readings,
        # and keep any extra dials the moment carried beyond the bank.
        readings = dict(m["readings"])
        readings.update({name: float(v) for name, v in zip(self.dial_names, vec)})
        return MomentHit(
            text=m["text"],
            readings=readings,
            ts=float(m["ts"]),
            space_id=m["space_id"],
            meta=dict(m["meta"]),
            score=score,
            vector=vec.copy(),
            index=i,
        )

    def __repr__(self) -> str:
        return self.summary()
