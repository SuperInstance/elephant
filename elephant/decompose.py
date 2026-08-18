"""DecompositionHarness — the decomposition doctrine in code.

The doctrine (the captain, verbatim, 2026-08-17):

    "Given a long enough time-span running a narrow task, a large LLM
    can be decomposed into smaller pieces: distilled smaller LLMs that
    can themselves be distilled. But more than that — decomposed,
    separated into components that look like other components and have
    simpler functions, and might do better with an algorithmic learning
    mechanism over time and some stochastic mechanisms for varying
    output if desired for the application."

Three moves, in code:

1. DISTILL — ``ingest(trace)`` then ``distill(k)``. A trace is the long
   record of a large model doing ONE narrow task: a list of
   ``(input, output)`` pairs — a log of responses, a transcript of a
   Tap night, a run of a teacher. Distilling decomposes it into ``k``
   components: k-means (multi-restart, k-means++ seeded, lowest
   objective kept) on simple features of the pairs — input length,
   output length, input→output token overlap, output entropy. Each
   cluster becomes a Component with a centroid prototype and a simple
   function: nearest-centroid routing + a small per-component lookup
   with a mode fallback. Routing happens in DOOR space (each organ
   stands behind the centroid of its members' inputs: length +
   punctuation energy); learning happens in CLUSTER space (the
   prototype: what the organ is).

2. DECOMPOSE — every Component looks like every other Component. The
   same shape: ``{id, prototype, learning_rate, temperature, hits,
   correct, score}`` (plus the shared door/output/table every organ
   carries). One component's job is any component's job — they are
   interchangeable organs, like dials in a dial bank.

3. LEARN + VARY — ``learn(reward_fn)`` is the algorithmic learning
   mechanism over time (the guitarist principle: settings are not
   designed top-down, they are discovered by running). The harness
   replays the trace, each component answers, the reward_fn scores the
   answer, and the winning component's prototype moves toward the
   rewarded (teacher) output or away from the punished one — so
   components diverge into specializations instead of collapsing.
   ``respond(input, temperature)`` is the stochastic mechanism: at
   temperature 0 the nearest component answers (deterministic); above 0
   the response is softmax-sampled over the nearest prototypes, so the
   same component body produces varying output when the application
   wants it. ``specialization()`` reports the body: near-identical
   organs, each with a simpler function, each grown different by
   running.

numpy-only. Mirrors the fleet's patterns: the Tap night is the learning
loop, the dial bank is the body of components, the temperature is the
softmax divergence that keeps identical components from collapsing into
one loud dial.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field as dc_field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Reward above this counts as a "correct" hit (and moves the prototype
# toward the teacher's output); at or below it the prototype moves away.
REWARD_THRESHOLD = 0.5

# How many k-means restarts to run at distill time; the lowest-objective
# result becomes the body. Restarts are cheap and make the decomposition
# robust to bad seeds — near-identical organs still get their own.
KMEANS_RESTARTS = 12

__all__ = [
    "Component",
    "DecompositionHarness",
    "input_features",
    "output_features",
    "pair_features",
    "REWARD_THRESHOLD",
    "KMEANS_RESTARTS",
]


# ---------------------------------------------------------------------- #
# The simple features — what the harness can see of an (input, output)   #
# ---------------------------------------------------------------------- #
_WORD = re.compile(r"[a-zA-Z0-9']+")


def _words(text: str) -> List[str]:
    return _WORD.findall(text)


def _entropy(text: str) -> float:
    """Character-level Shannon entropy of the output — a cheap measure
    of how much the model varied its language (terse replies are
    low-entropy; elaborate ones are not)."""
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def input_features(text: str) -> np.ndarray:
    """Input-side features — the only things visible at respond time:
    word count and punctuation energy (per-word count of !, ?, and
    ALL-CAPS tokens). These are the DOOR space: a component stands
    behind the centroid of its members' inputs."""
    words = _words(text)
    n = max(1, len(words))
    energy = sum(1 for ch in text if ch in "!?") + _CAPS_COUNT(text)
    return np.array([float(len(words)), float(energy) / n], dtype=float)


_CAPS = re.compile(r"\b[A-Z]{2,}\b")


def _CAPS_COUNT(text: str) -> int:
    return len(_CAPS.findall(text))


def output_features(inp: str, out: str) -> np.ndarray:
    """Output-side features: output word count, input→output token
    overlap, and output entropy. These are what make one behavior look
    different from another (short terse vs long elaborate; a reply that
    echoes the customer vs one that goes off-script)."""
    in_words = set(_words(inp))
    out_words = _words(out)
    overlap = len(in_words & set(out_words)) / max(1, len(in_words))
    return np.array([float(len(out_words)), float(overlap),
                     _entropy(out)], dtype=float)


def pair_features(inp: str, out: str) -> np.ndarray:
    """The 4-vector a pair is clustered on: input length, output
    length, input→output overlap, output entropy — the spec's simple
    features. (The cluster space deliberately keeps only the input
    LENGTH from the door space: the extra door dims help routing, but
    they are noise for finding the organs — see docs.)"""
    return np.concatenate([[float(len(_words(inp)))],
                           output_features(inp, out)])


def _representative(outputs: Sequence[str]) -> str:
    """The cluster's typical answer: the mode, first-seen tiebreak."""
    counts: Dict[str, int] = {}
    for o in outputs:
        counts[o] = counts.get(o, 0) + 1
    # dicts preserve insertion order, so max() keeps the first-seen on ties
    return max(counts.items(), key=lambda kv: kv[1])[0]


# ---------------------------------------------------------------------- #
# The component — an organ that looks like every other organ             #
# ---------------------------------------------------------------------- #
@dataclass
class Component:
    """One organ of the body. Every component has the SAME shape — the
    same fields, the same simple function — so organs are
    interchangeable, like dials in a dial bank.

    - ``prototype`` — the centroid in pair-feature (cluster) space:
      dim 0 is the input length it answers; dims 1-3 are the STYLE
      (the behavior it produces). Learning moves this vector.
    - ``door`` — the centroid of the members' inputs in DOOR space
      (length + punctuation energy): the kinds of inputs this organ
      answers. Routing and the small lookup use the door; the
      prototype is what learning tunes and specialization reports.
    - ``output`` — the representative answer (mode of the cluster).
    - ``table`` — the cluster's own members, a small lookup: respond by
      nearest stored input, fall back to ``output``.
    - ``learning_rate`` / ``temperature`` — the algorithmic-learning
      step and the stochastic knob (0 = deterministic).
    - ``hits`` / ``correct`` / ``score`` — the running record of how
      the organ is doing, accumulated by ``learn``. ``score`` is the
      cumulative reward (it only goes up as the organ runs);
      ``accuracy`` is the ratio that says how well.
    """

    id: int
    prototype: np.ndarray                       # 4-vector centroid (cluster space)
    door: np.ndarray = dc_field(default_factory=lambda: np.zeros(2))
    output: str = ""                            # representative answer
    table: List[Tuple[np.ndarray, str]] = dc_field(default_factory=list)
    learning_rate: float = 0.1
    temperature: float = 0.0
    hits: int = 0
    correct: int = 0
    score: float = 0.0

    def style(self) -> np.ndarray:
        """The output-side half of the prototype (dims 1-3) — what this
        organ is good at producing."""
        return self.prototype[1:]

    def accuracy(self) -> float:
        return (self.correct / self.hits) if self.hits else 0.0

    def respond(self, query: np.ndarray) -> str:
        """The simple function: nearest stored input in the small
        lookup, falling back to the representative answer."""
        best: Optional[str] = None
        best_d = math.inf
        for feat, out in self.table:
            d = float(np.linalg.norm(feat - query))
            if d < best_d:
                best_d = d
                best = out
        return best if best is not None else self.output

    def __repr__(self) -> str:
        return (f"<Component {self.id} score={self.score:.3f} "
                f"hits={self.hits} acc={self.accuracy():.2f} "
                f"t={self.temperature:g}>")


# ---------------------------------------------------------------------- #
# The harness — distill, decompose, learn, vary                          #
# ---------------------------------------------------------------------- #
class DecompositionHarness:
    """A large model's narrow-task behavior, decomposed into a body.

    The long time-span is the trace; the harness is what the fleet does
    with it: distill the trace into components that look like other
    components, let each component self-tune over time (the guitarist
    principle), and keep a stochastic knob for varying output when the
    application wants it.
    """

    def __init__(self, seed: int = 7, learning_rate: float = 0.1,
                 temperature: float = 0.0, kmeans_iters: int = 25,
                 kmeans_restarts: int = KMEANS_RESTARTS):
        self.seed = int(seed)
        self.learning_rate = float(learning_rate)
        self.temperature = float(temperature)   # the stochastic knob
        self.kmeans_iters = int(kmeans_iters)
        self.kmeans_restarts = int(kmeans_restarts)
        self._rng = np.random.default_rng(self.seed)

        self.trace: List[Tuple[str, str]] = []
        self.components: List[Component] = []
        self._mean: Optional[np.ndarray] = None      # cluster-space scaler
        self._std: Optional[np.ndarray] = None
        self._in_mean: Optional[np.ndarray] = None   # door-space scaler
        self._in_std: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # DISTILL — the long record becomes components                       #
    # ------------------------------------------------------------------ #
    def ingest(self, trace: Sequence[Tuple[str, str]]) -> "DecompositionHarness":
        """Store the trace: the long record of the large model running
        one narrow task — (input, output) pairs. The trace is the
        teacher."""
        items = list(trace)
        if not items:
            raise ValueError("ingest() needs a non-empty trace — the "
                             "long time-span has to have happened")
        cleaned = []
        for i, item in enumerate(items):
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError(f"trace item {i} is not an (input, "
                                 f"output) pair: {item!r}")
            inp, out = str(item[0]), str(item[1])
            if inp == "" and out == "":
                raise ValueError(f"trace item {i} is empty — a teacher "
                                 f"that says nothing teaches nothing")
            cleaned.append((inp, out))
        self.trace = cleaned
        return self

    def distill(self, k: int = 4) -> List[Component]:
        """Decompose the trace into ``k`` components that look like
        other components.

        k-means (multi-restart, lowest objective kept) on the simple
        pair features (input length, output length, input→output
        overlap, output entropy); each cluster becomes a Component with
        a centroid prototype and a simple function (nearest-centroid
        responder with a small lookup). The scaler (mean/std) is fixed
        here so respond() and learn() live in the same space.
        """
        if not self.trace:
            raise ValueError("distill() before ingest() — there is no "
                             "trace to decompose")
        k = int(k)
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        # a body cannot have more organs than the trace has nights
        k = min(k, len(self.trace))

        # Cluster space: the simple pair features. Door space: the input
        # features (length + punctuation energy), scaled separately —
        # routing lives in the door, learning lives in the cluster.
        feats = np.stack([pair_features(i, o) for i, o in self.trace])
        mean = feats.mean(axis=0)
        std = feats.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)   # constant dims -> scaled to 0
        self._mean, self._std = mean, std
        scaled = np.nan_to_num((feats - mean) / std,
                               nan=0.0, posinf=0.0, neginf=0.0)

        ins = np.stack([input_features(i) for i, _ in self.trace])
        in_mean = ins.mean(axis=0)
        in_std = np.where(ins.std(axis=0) < 1e-6, 1.0, ins.std(axis=0))
        self._in_mean, self._in_std = in_mean, in_std
        doors = np.nan_to_num((ins - in_mean) / in_std,
                              nan=0.0, posinf=0.0, neginf=0.0)

        centroids, assign = self._kmeans(scaled, k)

        self.components = []
        for c in range(k):
            members = [i for i in range(len(self.trace)) if assign[i] == c]
            protos = scaled[members].mean(axis=0) if members else centroids[c]
            outs = [self.trace[i][1] for i in members]
            table = [(doors[i].copy(), self.trace[i][1]) for i in members]
            self.components.append(Component(
                id=c,
                prototype=np.nan_to_num(protos, nan=0.0),
                door=(doors[members].mean(axis=0) if members
                      else np.zeros(2)),
                output=_representative(outs) if outs else self.trace[0][1],
                table=table,
                learning_rate=self.learning_rate,
                temperature=self.temperature,
            ))
        return self.components

    # ------------------------------------------------------------------ #
    # RESPOND — the stochastic mechanism for varying output               #
    # ------------------------------------------------------------------ #
    def respond(self, inp: str, temperature: Optional[float] = None) -> str:
        """Route the input to the nearest component, and that component
        answers.

        ``temperature`` is the stochastic knob: 0 (default) is argmax —
        deterministic; > 0 softmax-samples among the nearest prototypes
        so the same body produces varying output when the application
        wants it.
        """
        comp = self._route(inp, temperature)
        return comp.respond(self._scale_input(inp))

    def route(self, inp: str, temperature: Optional[float] = None) -> Component:
        """Which component answers this input (the routing itself)."""
        return self._route(inp, temperature)

    def _route(self, inp: str, temperature: Optional[float]) -> Component:
        if not self.components:
            raise ValueError("distill() first — there are no components "
                             "to route through")
        temp = self.temperature if temperature is None else float(temperature)
        q = self._scale_input(inp)
        doors = np.stack([c.door for c in self.components])
        dists = np.linalg.norm(doors - q, axis=1)
        if temp <= 0:
            return self.components[int(np.argmin(dists))]
        # Softmax over the nearest doors — the divergence knob: near
        # doors share the answer, far doors stay (nearly) silent.
        k = min(len(self.components), 3)
        idx = np.argpartition(dists, k - 1)[:k]
        logits = -dists[idx] / max(temp, 1e-9)
        logits -= logits.max()                       # stabilize
        p = np.exp(logits)
        p = p / p.sum()
        pick = self._rng.choice(k, p=p)
        return self.components[int(idx[pick])]

    def _scale_input(self, inp: str) -> np.ndarray:
        """The query in DOOR space — what routing and the small lookup
        see."""
        if self._in_mean is None or self._in_std is None:
            raise ValueError("distill() first — the door space has "
                             "not been fixed yet")
        x = input_features(inp)
        return np.nan_to_num((x - self._in_mean) / self._in_std, nan=0.0)

    # ------------------------------------------------------------------ #
    # LEARN — the algorithmic learning mechanism over time               #
    # ------------------------------------------------------------------ #
    def learn(self, reward_fn: Callable[[str, str], float], epochs: int = 3,
              temperature: Optional[float] = None) -> List[float]:
        """The guitarist principle: settings are discovered by running.

        Replay the trace ``epochs`` times; each (input, output) is
        answered by a component (routed, with ``temperature`` exploration
        when given); the reward_fn scores the answer; and the winning
        component's prototype moves toward the teacher's output when
        rewarded, away when punished — so components specialize into
        different organs instead of collapsing into one loud dial.

        Returns the mean reward per epoch (the learning curve).
        """
        if not self.components:
            raise ValueError("distill() first — there is nothing to learn")
        if not self.trace:
            raise ValueError("ingest() first — the teacher is not here")
        temp = self.temperature if temperature is None else float(temperature)
        curve: List[float] = []
        for _ in range(max(1, int(epochs))):
            rewards = []
            for inp, teacher_out in self.trace:
                comp = self._route(inp, temp)
                q = self._scale_input(inp)
                response = comp.respond(q)
                try:
                    r = float(reward_fn(inp, response))
                except (TypeError, ValueError):
                    r = 0.0
                if not math.isfinite(r):
                    r = 0.0
                r = min(1.0, max(0.0, r))
                comp.hits += 1
                comp.score += r
                if r > REWARD_THRESHOLD:
                    comp.correct += 1
                # The per-component learning signal — the guitarist
                # principle: settings are discovered by running.
                #   prototype (what it is): toward the teacher's answer
                #     on reward, away on punishment;
                #   door (what it answers): toward this input on reward
                #     (it wins here — widen), away on punishment (it
                #     loses here — vacate the region).
                # The step is clipped and the vectors boxed, so a
                # punished organ FADES instead of exploding — organs
                # drift to the edge of the body and go quiet, they
                # never blow up.
                target = pair_features(inp, teacher_out)
                target = np.nan_to_num((target - self._mean) / self._std,
                                       nan=0.0)
                direction = 1.0 if r > REWARD_THRESHOLD else -1.0
                delta = np.clip(target - comp.prototype, -1.0, 1.0)
                comp.prototype += direction * comp.learning_rate * delta
                np.clip(comp.prototype, -4.0, 4.0, out=comp.prototype)
                d_target = np.clip(q - comp.door, -1.0, 1.0)
                comp.door += direction * comp.learning_rate * d_target
                np.clip(comp.door, -4.0, 4.0, out=comp.door)
                rewards.append(r)
            curve.append(float(np.mean(rewards)))
        return curve

    # ------------------------------------------------------------------ #
    # SPECIALIZATION — the body after running                            #
    # ------------------------------------------------------------------ #
    def specialization(self) -> Dict:
        """Report how the components DIVERGED: each organ's score, what
        it got good at (its style and representative answer), and the
        temperature-driven diversity — the body of near-identical
        organs, each with a simpler function.

        ``divergence`` is the spread of accuracies (how far the organs
        have grown apart in performance — 0 when no one has run yet);
        ``prototype_spread`` is the mean pairwise distance between the
        organs' style vectors (how far apart the behaviors have
        drifted). Two numbers because one can lie alone: identical
        organs that are all equally bad show divergence 0 but a real
        body shows its shape in both."""
        rows = []
        for c in self.components:
            rows.append({
                "id": c.id,
                "score": round(float(c.score), 4),
                "hits": c.hits,
                "correct": c.correct,
                "accuracy": round(float(c.accuracy()), 4),
                "temperature": float(c.temperature),
                "door": [round(float(v), 3) for v in c.door],
                "style": [round(float(v), 3) for v in c.style()],
                "output": c.output,
            })
        accs = [r["accuracy"] for r in rows if r["hits"]]
        divergence = float(np.std(accs)) if accs else 0.0
        styles = np.stack([c.style() for c in self.components])
        spread = 0.0
        if len(styles) > 1:
            pairs = 0.0
            total = 0.0
            for i in range(len(styles)):
                for j in range(i + 1, len(styles)):
                    total += float(np.linalg.norm(styles[i] - styles[j]))
                    pairs += 1.0
            spread = total / max(1.0, pairs)
        return {
            "n": len(rows),
            "divergence": round(divergence, 4),
            "prototype_spread": round(spread, 4),
            "components": rows,
        }

    # ------------------------------------------------------------------ #
    # k-means (numpy-only, k-means++ seeded, multi-restart, empty-safe)  #
    # ------------------------------------------------------------------ #
    def _kmeans(self, X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Best of ``kmeans_restarts`` runs: each restart is k-means++
        seeded from a deterministic derivation of the harness seed; the
        lowest within-cluster-sum-of-squares result becomes the body.
        Restarts are what keep near-identical organs from collapsing
        into one loud cluster."""
        best_cent, best_assign, best_obj = None, None, math.inf
        for r in range(self.kmeans_restarts):
            rng = np.random.default_rng(self.seed * 2654435761
                                        + r * 7919 + 1)
            cent = self._kmeans_pp(X, k, rng)
            for _ in range(self.kmeans_iters):
                assign = self._assign(X, cent)
                new = np.stack([
                    X[assign == c].mean(axis=0) if np.any(assign == c)
                    else self._farthest(X, cent)
                    for c in range(k)
                ])
                if np.allclose(new, cent):
                    cent = new
                    break
                cent = new
            obj = self._objective(X, self._assign(X, cent))
            if obj < best_obj:
                best_obj = obj
                best_cent = cent
                best_assign = self._assign(X, cent)
        return best_cent, best_assign

    @staticmethod
    def _kmeans_pp(X: np.ndarray, k: int, rng) -> np.ndarray:
        """k-means++ seeding: first centroid uniform, then each next
        one with probability ∝ distance² to the nearest already chosen
        — so every latent behavior has a chance to be seeded."""
        n = len(X)
        idx = [int(rng.integers(n))]
        while len(idx) < k:
            d = np.linalg.norm(X - X[idx][:, None, :], axis=2)
            nearest = d.min(axis=0)
            p = nearest ** 2
            total = p.sum()
            if total <= 1e-12:
                idx.append(int(rng.integers(n)))   # degenerate: any seed
            else:
                idx.append(int(rng.choice(n, p=p / total)))
        return X[idx].copy()

    @staticmethod
    def _assign(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        return np.argmin(d, axis=1)

    @staticmethod
    def _objective(X: np.ndarray, assign: np.ndarray) -> float:
        tot = 0.0
        for c in range(int(assign.max()) + 1):
            m = X[assign == c]
            if len(m):
                tot += float(np.sum((m - m.mean(axis=0)) ** 2))
        return tot

    @staticmethod
    def _farthest(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        nearest = d.min(axis=1)
        return X[int(np.argmax(nearest))].copy()

    def __len__(self) -> int:
        return len(self.components)

    def __repr__(self) -> str:
        return (f"<DecompositionHarness trace={len(self.trace)} "
                f"components={len(self.components)} temp={self.temperature:g}>")
