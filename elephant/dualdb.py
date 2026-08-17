"""Dual-DB bridge — the room perceives itself, and predicts itself.

The elephant reads a room: dials feel the vibe, the field is the
temperature. That is the Z_in side — what the room PERCEIVES. This
module adds the Z_out side — what the room PREDICTS about itself.
Every room gets two databases:

    Z_in  — perception history: sensor frames, dial readings, field
            vectors, each encoded as a vector (plato-perception's
            ladder: raw, normalized, hash-projected, random-projected,
            learned-projected).
    Z_out — prediction history: where the field is going, what the
            next reading will be, how anomalous the room is right
            now, and the trend (plato-prediction's outputs, encoded
            the way that repo encodes them).

The bridge is `DualDBRoom`: it wraps a `SignalRoom` (frames) or a
`Room` (messages), keeps the Z_in history, and from it produces
Z_out predictions. Two of those predictions are NEW dials the field
can read:

    trend_dial — the room's own predicted direction,
                 [-1 cooling .. +1 warming]. Not what the room IS,
                 but where it's GOING. A dial about the room, read
                 by the room — the room predicting itself.
    anomaly    — how unusual the current state is vs the room's own
                 recent pattern, [0, 1]. The room noticing something
                 is off (a Mahalanobis deviation — the same math as
                 fleetmath's good-days anchor).

The seam to pulse.py: `elephant/pulse.py` is the internal-monologue
engine — a `PulseLoop` that ticks on a constant heartbeat and reads a
room's field through a `DialBank`. The Dual-DB bridge feeds it two
ways: the `on_pulse` callback (perception in, prediction out — call
`pulse_loop.tick()` from it), and the `TrendDial` / `AnomalyDial`
wrappers, which cast the two new senses as real `Dial`s so they join
any `DialBank` / `PulseLoop` / `read_field` unchanged. The trend dial
is the forward-looking counterpart to pulse's backward-looking
`direction()` (the last two readings); together they are the past and
future of the same room.

Math is deliberately simple + honest: linear extrapolation, rolling
mean/std, a regularized Mahalanobis. No learned models — that is the
LearnedProjection stub's job later, seeded from elephant/learned.py.

Cross-pollinated from plato-perception (Z_in) and plato-prediction
(Z_out) — docs/dualdb-bridge.md is the full writeup.
"""
from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .dial import Dial
from .fleetmath import biomass_anchor, biomass_deviation
from .room import Room
from .sensors import (
    FishingDayDial,
    RadarCoherenceDial,
    SensorFrame,
    SignalRoom,
    SounderBiomassDial,
)

__all__ = [
    "EncodingMethod",
    "ZInEncoder",
    "PredictionType",
    "PredEncodingMethod",
    "PredictionOutput",
    "PredictionEncoder",
    "ZOutPredictor",
    "DualDBRoom",
    "TrendDial",
    "AnomalyDial",
]


# ===================================================================== #
# Z_in — perception encoding (cross-pollinated from plato-perception)   #
#                                                                       #
# plato-perception turns a SensorReading into a raw triple              #
# [value, confidence, timestamp_norm] and then runs that through an     #
# encoding ladder: Raw -> Normalized (z-score) -> HashProjection ->     #
# RandomProjection -> LearnedProjection. The elephant's frames and      #
# dial readings are the same animal: a reading is a reading.            #
# ===================================================================== #
class EncodingMethod(str, enum.Enum):
    RAW = "raw"
    NORMALIZED = "normalized"
    HASH_PROJECTION = "hash_projection"
    RANDOM_PROJECTION = "random_projection"
    LEARNED_PROJECTION = "learned_projection"


def _fnv1a(s: str) -> int:
    """FNV-1a 64-bit hash — the deterministic mixer under the
    hash projection, ported straight from plato-perception."""
    h = 14695981039346656037
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def _hash_project(input_vec: np.ndarray, output_dim: int, seed_str: str) -> np.ndarray:
    """Deterministic projection: each input coordinate stirs every
    output coordinate by a hash-derived weight; result unit-normalized.

    Same structure as plato-perception's `deterministic_project`.
    """
    out = np.zeros(output_dim, dtype=float)
    for i, val in enumerate(input_vec):
        h = _fnv1a(f"{seed_str}-{i}-{output_dim}")
        for j in range(output_dim):
            mix = (((h + (j * 2654435761)) % 1_000_000) / 1_000_000.0) - 0.5
            out[j] += val * mix
    n = float(np.linalg.norm(out))
    if n > 0.0:
        out /= n
    return out


def _random_project(input_vec: np.ndarray, output_dim: int, seed: int) -> np.ndarray:
    """Seeded random projection: x @ W with W ~ N(0, 1/sqrt(d_in)),
    unit-normalized. Deterministic for a given seed."""
    d_in = input_vec.shape[0]
    if d_in == 0 or output_dim == 0:
        return np.zeros(output_dim, dtype=float)
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0 / math.sqrt(d_in), size=(d_in, output_dim))
    out = input_vec @ w
    n = float(np.linalg.norm(out))
    if n > 0.0:
        out /= n
    return out


def _resize(vec: np.ndarray, output_dim: int) -> np.ndarray:
    """Pad with zeros or truncate to `output_dim` (plato's `resize`)."""
    out = np.zeros(output_dim, dtype=float)
    k = min(vec.shape[0], output_dim)
    out[:k] = vec[:k]
    return out


class ZInEncoder:
    """Perception encoder — SensorFrame / dial readings -> vector.

    Port of plato-perception's `PerceptionEncoder`. Every reading
    becomes a raw triple [value, confidence, timestamp_norm]; the
    triples are then run through the chosen EncodingMethod.
    """

    def __init__(
        self,
        output_dim: Optional[int] = None,
        seed: int = 42,
        method: EncodingMethod = EncodingMethod.RAW,
    ):
        self.output_dim = output_dim
        self.seed = seed
        self.method = EncodingMethod(method)

    # ------------------------------------------------------------------ #
    # Reading triples                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def scalarize(data: Any) -> float:
        """Reduce any frame payload to one scalar.

        - scalar -> itself (NaN -> 0.0)
        - 1-D array -> mean
        - 2-D point cloud (N, 2) -> spread (mean distance to centroid),
          the same feel as RadarCoherenceDial._spread: tight = on fish.
        """
        try:
            if data is None:
                return 0.0
            arr = np.asarray(data, dtype=float)
        except (TypeError, ValueError):
            return 0.0
        if arr.ndim == 0:
            v = float(arr)
            return 0.0 if math.isnan(v) else v
        if arr.ndim == 1:
            v = float(np.mean(arr)) if arr.size else 0.0
            return 0.0 if math.isnan(v) else v
        if arr.ndim == 2:
            if arr.shape[0] < 2:
                return 0.0
            c = arr.mean(axis=0)
            v = float(np.mean(np.linalg.norm(arr - c, axis=1)))
            return 0.0 if math.isnan(v) else v
        return 0.0

    def triples(self, source: Any) -> np.ndarray:
        """Source -> (N, 3) array of [value, confidence, ts_norm].

        Accepts a `SensorFrame`, a `SignalRoom`, a sequence of frames,
        a `Room` (read through its dials), or a dict of dial readings
        {name: reading}. Mirrors plato's `to_raw_vector()`:
        timestamp_norm = (ts % 1_000_000) / 1_000_000.
        """
        # A single frame.
        if isinstance(source, SensorFrame):
            return self.triples([source])
        # A room of frames.
        if isinstance(source, SignalRoom):
            return self.triples(source.frames)
        # A dict of dial readings {name: reading}.
        if isinstance(source, dict):
            if not source:
                return np.zeros((0, 3), dtype=float)
            rows = [
                [float(v), 1.0, 0.0]
                for v in source.values()
            ]
            return np.asarray(rows, dtype=float)
        # A text Room -> its dial readings (the elephant path).
        if isinstance(source, Room) or (hasattr(source, "messages") and not hasattr(source, "frames")):
            from .dial import DialBank
            from .dials import DEFAULT_DIALS
            from .field import read_field
            return self.triples(read_field(source, DialBank(DEFAULT_DIALS)).readings)
        # A sequence of frames.
        rows = []
        for f in source:
            conf = 1.0
            meta = getattr(f, "meta", None)
            if isinstance(meta, dict) and meta.get("confidence") is not None:
                conf = float(meta["confidence"])
            ts = float(getattr(f, "ts", 0.0))
            rows.append([self.scalarize(getattr(f, "data", f)), conf, (ts % 1_000_000.0) / 1_000_000.0])
        return np.asarray(rows, dtype=float) if rows else np.zeros((0, 3), dtype=float)

    # ------------------------------------------------------------------ #
    # Encoding                                                           #
    # ------------------------------------------------------------------ #
    def encode_triples(self, triples: np.ndarray,
                       method: Optional[EncodingMethod] = None) -> np.ndarray:
        """Core: (N, 3) reading triples -> one vector under `method`."""
        triples = np.asarray(triples, dtype=float)
        method = EncodingMethod(method or self.method)
        out_dim = self.output_dim if self.output_dim is not None else triples.size

        if triples.shape[0] == 0 or triples.size == 0:
            return np.zeros(out_dim, dtype=float)

        if method == EncodingMethod.NORMALIZED:
            # plato's encode_normalized: normalize_readings then resize.
            triples = self._zscore_values(triples)
            method = EncodingMethod.RAW

        raw = triples.reshape(-1)

        if method == EncodingMethod.RAW:
            return _resize(raw, out_dim)
        if method == EncodingMethod.HASH_PROJECTION:
            return _hash_project(raw, out_dim, f"hash-{self.seed}")
        if method == EncodingMethod.RANDOM_PROJECTION:
            return _random_project(raw, out_dim, self.seed)
        if method == EncodingMethod.LEARNED_PROJECTION:
            # STUB: same deterministic structure as the hash projection
            # until a real learned backbone lands (elephant/learned.py
            # trains dials, not projections; that seam is the TODO).
            return _hash_project(raw, out_dim, f"learned-{self.seed}")
        raise ValueError(f"unknown encoding method: {method!r}")

    def encode(self, source: Any,
               method: Optional[EncodingMethod] = None) -> np.ndarray:
        """Encode a frame / room / readings dict into a perception vector."""
        return self.encode_triples(self.triples(source), method)

    @staticmethod
    def _zscore_values(triples: np.ndarray) -> np.ndarray:
        """Z-score the value column across readings (plato's
        `normalize_readings`): std == 0 leaves values unchanged."""
        out = triples.copy()
        vals = out[:, 0]
        std = float(vals.std())
        if std > 1e-12:
            out[:, 0] = (vals - float(vals.mean())) / std
        return out

    def __repr__(self) -> str:
        return f"ZInEncoder(method={self.method.value}, output_dim={self.output_dim}, seed={self.seed})"


# ===================================================================== #
# Z_out — prediction encoding (cross-pollinated from plato-prediction)  #
#                                                                       #
# plato-prediction carries a PredictionOutput (type, value, confidence) #
# and encodes it with Raw / Confidence / Hierarchical / MultiHead       #
# methods. The elephant's predictor produces the same animals.          #
# ===================================================================== #
class PredictionType(str, enum.Enum):
    VALUE_PREDICTION = "value_prediction"
    CLASSIFICATION = "classification"
    ANOMALY_SCORE = "anomaly_score"
    ACTION = "action"
    TREND = "trend"
    MULTI_TARGET = "multi_target"


_PREDICTION_TYPE_CODES: Dict[PredictionType, float] = {
    PredictionType.VALUE_PREDICTION: 1.0,
    PredictionType.CLASSIFICATION: 2.0,
    PredictionType.ANOMALY_SCORE: 3.0,
    PredictionType.ACTION: 4.0,
    PredictionType.TREND: 5.0,
    PredictionType.MULTI_TARGET: 6.0,
}


@dataclass
class PredictionOutput:
    """One prediction — plato-prediction's `PredictionOutput` ported.

    `targets` carries the payload for MULTI_TARGET predictions (the
    variant payload plato stores inside the enum).
    """
    prediction_type: PredictionType
    value: float
    confidence: float
    model_name: str = "dualdb.zout.linear"
    latency_ms: float = 0.0
    targets: Tuple[float, ...] = ()

    def to_vector(self) -> np.ndarray:
        """[value, confidence, type_code] (+ targets for MULTI_TARGET)."""
        code = _PREDICTION_TYPE_CODES.get(self.prediction_type, 0.0)
        v = [self.value, self.confidence, code]
        if self.prediction_type == PredictionType.MULTI_TARGET:
            v.extend(self.targets)
        return np.asarray(v, dtype=float)


class PredEncodingMethod(str, enum.Enum):
    RAW = "raw"
    CONFIDENCE = "confidence"
    HIERARCHICAL = "hierarchical"
    MULTI_HEAD = "multi_head"


class PredictionEncoder:
    """Turns a PredictionOutput into a vector — the plato-prediction
    `PredictionEncoder` port: Raw, Confidence (weight everything by
    confidence), Hierarchical (append mean/max/min), MultiHead (repeat
    the raw vector `heads` times); then pad/truncate to output_dim."""

    def __init__(self,
                 output_dim: Optional[int] = None,
                 method: PredEncodingMethod = PredEncodingMethod.RAW,
                 heads: int = 2):
        self.output_dim = output_dim
        self.method = PredEncodingMethod(method)
        self.heads = max(1, int(heads))

    def encode(self, output: PredictionOutput) -> np.ndarray:
        raw = output.to_vector()
        if self.method == PredEncodingMethod.RAW:
            v = raw
        elif self.method == PredEncodingMethod.CONFIDENCE:
            v = raw * output.confidence
        elif self.method == PredEncodingMethod.HIERARCHICAL:
            mean = float(np.mean(raw)) if raw.size else 0.0
            mx = float(np.max(raw)) if raw.size else 0.0
            mn = float(np.min(raw)) if raw.size else 0.0
            v = np.concatenate([raw, np.asarray([mean, mx, mn])])
        elif self.method == PredEncodingMethod.MULTI_HEAD:
            v = np.tile(raw, self.heads)
        else:
            raise ValueError(f"unknown prediction encoding method: {self.method!r}")
        if self.output_dim is not None:
            v = _resize(v, self.output_dim)
        return v

    def __repr__(self) -> str:
        return f"PredictionEncoder(method={self.method.value}, output_dim={self.output_dim})"


class ZOutPredictor:
    """Predicts the room's future from its Z_in history.

    Simple + honest: per-dimension least-squares linear extrapolation,
    rolling mean/std statistics, and a regularized Mahalanobis
    deviation (fleetmath's biomass_anchor / biomass_deviation — the
    same math as the good-days anchor) for the anomaly sense.

    `predict(series)` returns three PredictionOutputs, in order:
    TREND ([-1 cooling .. +1 warming]), VALUE_PREDICTION (the mean
    next-field-value one step ahead), ANOMALY_SCORE ([0, 1]).
    """

    def __init__(self, history_window: int = 10, min_history: int = 3, seed: int = 42):
        self.history_window = max(2, int(history_window))
        self.min_history = max(2, int(min_history))
        self.seed = seed

    # ------------------------------------------------------------------ #
    # Linear statistics                                                  #
    # ------------------------------------------------------------------ #
    def trends(self, series: Sequence[np.ndarray]) -> np.ndarray:
        """Per-dimension normalized slope in [-1, +1] (up / down / flat)."""
        X = self._matrix(series)
        n, d = X.shape
        if n < 2:
            return np.zeros(d, dtype=float)
        t = np.arange(n, dtype=float)
        slopes = np.asarray([np.polyfit(t, X[:, j], 1)[0] for j in range(d)])
        scale = np.maximum(X.std(axis=0), 1e-9)
        return np.clip(slopes / scale, -1.0, 1.0)

    def next_value(self, series: Sequence[np.ndarray]) -> np.ndarray:
        """Per-dimension one-step-ahead linear extrapolation."""
        X = self._matrix(series)
        n, d = X.shape
        if n == 0:
            return np.zeros(d, dtype=float)
        if n < 2:
            return X[-1].copy()
        t = np.arange(n, dtype=float)
        preds = []
        for j in range(d):
            slope, intercept = np.polyfit(t, X[:, j], 1)
            preds.append(slope * n + intercept)   # one step past index n-1
        return np.asarray(preds, dtype=float)

    def fit_quality(self, series: Sequence[np.ndarray]) -> Optional[float]:
        """Mean R² of the per-dimension linear fits; None if too short.

        Constant dimensions (a missing sensor encoding as 0.0, a
        confidence slot pinned at 1.0) carry no fit signal — they are
        skipped rather than counted as a perfect fit, so confidence
        reflects the dimensions that are actually moving."""
        X = self._matrix(series)
        n, d = X.shape
        if n < 3:
            return None
        t = np.arange(n, dtype=float)
        r2s = []
        for j in range(d):
            y = X[:, j]
            denom = float(np.sum((y - y.mean()) ** 2))
            if denom < 1e-12:
                continue                    # constant dim: no fit signal
            slope, intercept = np.polyfit(t, y, 1)
            resid = float(np.sum((y - (slope * t + intercept)) ** 2))
            r2s.append(1.0 - resid / denom)
        if not r2s:
            return 1.0                      # every dim flat: confident it stays flat
        return float(np.mean(r2s))

    # ------------------------------------------------------------------ #
    # Anomaly — the room noticing something is off                       #
    # ------------------------------------------------------------------ #
    def anomaly(self, series: Sequence[np.ndarray]) -> float:
        """Mahalanobis deviation of the newest state from the room's
        recent pattern, mapped to [0, 1].

        The anchor distribution is fit over the history (shrinkage-
        regularized, exactly like fleetmath's good-days anchor); the
        newest vector's deviation D = sqrt((x-μ)ᵀ Σ⁻¹ (x-μ)) is then
        scored against the typical radius sqrt(d): a room at its own
        typical distance reads 0, at ~3x the typical radius reads 1.
        If the covariance is degenerate (a perfectly still room) it
        falls back to a diagonal Mahalanobis with a tiny floor, so
        any real jump reads as anomalous.
        """
        X = self._matrix(series)
        n, d = X.shape
        if n < 3 or d == 0:
            return 0.0
        hist, latest = X[:-1], X[-1]
        mu = hist.mean(axis=0)
        delta = latest - mu

        # Full-covariance path (fleetmath) when well-conditioned.
        D: Optional[float] = None
        try:
            anchor = biomass_anchor(hist, shrinkage=0.5)
            cov = anchor["cov"]
            if (np.trace(cov) > 1e-12 and np.isfinite(np.linalg.cond(cov))
                    and np.linalg.cond(cov) < 1e12):
                D = biomass_deviation(latest, anchor)
        except (np.linalg.LinAlgError, ValueError):
            D = None
        if D is None or not math.isfinite(D):
            floor = 1e-9 * max(1.0, float(np.mean(np.abs(mu))) if d else 1.0)
            stds = np.maximum(hist.std(axis=0), floor)
            D = float(np.sqrt(np.sum((delta / stds) ** 2)))

        # Nakagami mean of a d-dimensional Gaussian's Mahalanobis D is
        # sqrt(d - 0.5), not sqrt(d). Score the *excess* above that and
        # map smoothly: normal reads ~0, ~2x typical ~0.45, ~3x ~0.99.
        # No dead zones, no binary trigger — a graded "weirdness" meter.
        typical = math.sqrt(max(d - 0.5, 0.0))
        excess = max(D - typical, 0.0)
        return float(1.0 - math.exp(-(excess * excess) / (2.0 * d)))

    # ------------------------------------------------------------------ #
    # predict                                                            #
    # ------------------------------------------------------------------ #
    def predict(self, series: Sequence[np.ndarray]) -> List[PredictionOutput]:
        """Z_in history (newest LAST) -> three Z_out predictions."""
        start = time.perf_counter()
        windowed = list(series)[-self.history_window:]
        X = self._matrix(windowed)
        n, d = X.shape

        if n == 0:
            outs = [
                PredictionOutput(PredictionType.TREND, 0.0, 0.05),
                PredictionOutput(PredictionType.VALUE_PREDICTION, 0.0, 0.05),
                PredictionOutput(PredictionType.ANOMALY_SCORE, 0.0, 0.05),
            ]
        elif n == 1:
            last = X[-1]
            outs = [
                PredictionOutput(PredictionType.TREND, 0.0, 0.05),
                PredictionOutput(PredictionType.VALUE_PREDICTION, float(np.mean(last)), 0.05),
                PredictionOutput(PredictionType.ANOMALY_SCORE, 0.0, 0.05),
            ]
        else:
            trend = float(np.clip(np.mean(self.trends(windowed)), -1.0, 1.0))
            value = float(np.mean(self.next_value(windowed)))
            anomaly = self.anomaly(windowed)
            r2 = self.fit_quality(windowed)
            confidence = float(np.clip(r2, 0.05, 0.99)) if r2 is not None else 0.5
            outs = [
                PredictionOutput(PredictionType.TREND, trend, confidence),
                PredictionOutput(PredictionType.VALUE_PREDICTION, value, confidence),
                PredictionOutput(PredictionType.ANOMALY_SCORE, anomaly, confidence),
            ]

        latency_ms = (time.perf_counter() - start) * 1000.0
        for o in outs:
            o.latency_ms = latency_ms
        return outs

    @staticmethod
    def _matrix(series: Sequence[np.ndarray]) -> np.ndarray:
        if not series:
            return np.zeros((0, 0), dtype=float)
        rows = [np.asarray(s, dtype=float).reshape(-1) for s in series]
        d = max((r.shape[0] for r in rows), default=0)
        X = np.zeros((len(rows), d), dtype=float)
        for i, r in enumerate(rows):
            k = min(r.shape[0], d)
            X[i, :k] = r[:k]
        return X

    def __repr__(self) -> str:
        return f"ZOutPredictor(window={self.history_window}, min_history={self.min_history})"


# ===================================================================== #
# DualDBRoom — the bridge: Z_in in, Z_out out                           #
# ===================================================================== #
_FRAME_SENSORS = ("sounder", "radar", "camera", "nav")


class DualDBRoom:
    """Wraps a SignalRoom (frames) or Room (messages); keeps Z_in
    (perception history) and Z_out (prediction history).

    perceive() snapshots the room's current state into a Z_in vector;
    predict() turns the history into Z_out predictions. The two new
    dials — trend_dial() (where the room is going) and anomaly()
    (whether something is off) — join the field via dial().
    """

    def __init__(
        self,
        room: Union[SignalRoom, Room],
        zin_method: EncodingMethod = EncodingMethod.RAW,
        zin_dim: Optional[int] = None,
        predictor: Optional[ZOutPredictor] = None,
        on_pulse: Optional[Callable[[np.ndarray, List[PredictionOutput]], None]] = None,
        encoder_seed: int = 42,
    ):
        self.room = room
        self.zin_method = EncodingMethod(zin_method)
        self.zin_dim = zin_dim
        self.predictor = predictor or ZOutPredictor()
        self.on_pulse = on_pulse
        self.encoder_seed = encoder_seed
        self.zin: List[np.ndarray] = []
        self.zout: List[List[PredictionOutput]] = []
        self._encoder = ZInEncoder(output_dim=zin_dim, seed=encoder_seed,
                                   method=zin_method)
        self._cap = max(self.predictor.history_window, 8)
        # Fleet dials for the SignalRoom path (the room's own senses).
        self._radar = RadarCoherenceDial()
        self._sounder = SounderBiomassDial()
        self._fishing = FishingDayDial(radar=self._radar, sounder=self._sounder)

    # ------------------------------------------------------------------ #
    # Z_in — perception                                                  #
    # ------------------------------------------------------------------ #
    def _snapshot_triples(self) -> np.ndarray:
        """Latest frame per sensor (fixed layout, missing -> zero triple)
        + the three fleet dial readings appended raw.

        NOTE: the timestamp component is pinned to 0.0 here. The raw
        triple is plato's [value, confidence, timestamp_norm], but a
        recency ramp is metadata about the *vector*, not a property of
        the room's *state* — feeding it to the trend dial leaks the
        clock and plants a phantom +warming bias on an empty room.
        ZInEncoder.encode still emits the faithful triple; the bridge
        drops the clock deliberately.
        """
        rows = []
        for sensor in _FRAME_SENSORS:
            frames = self.room.by_sensor(sensor) if isinstance(self.room, SignalRoom) else []
            if frames:
                f = frames[-1]
                conf = 1.0
                meta = getattr(f, "meta", None)
                if isinstance(meta, dict) and meta.get("confidence") is not None:
                    conf = float(meta["confidence"])
                rows.append([
                    self._encoder.scalarize(getattr(f, "data", None)),
                    conf,
                    0.0,
                ])
            else:
                rows.append([0.0, 0.0, 0.0])
        return np.asarray(rows, dtype=float)

    def perceive(self) -> np.ndarray:
        """Encode the room's CURRENT state into a Z_in vector.

        SignalRoom path: [sounder, radar, camera, nav] triples encoded
        under zin_method, then [radar_coherence, sounder_biomass,
        fishing_day] appended raw (they are already self-normalizing
        senses; RAW keeps the across-time level that trends need).
        Room path: the shared dial readings encoded under zin_method.
        """
        if isinstance(self.room, SignalRoom):
            encoded = self._encoder.encode_triples(self._snapshot_triples(),
                                                   self.zin_method)
            dials = np.asarray([
                self._radar.read(self.room),
                self._sounder.read(self.room),
                self._fishing.read(self.room),
            ], dtype=float)
            vector = np.concatenate([encoded, dials])
        else:
            from .dial import DialBank
            from .dials import DEFAULT_DIALS
            from .field import read_field
            readings = read_field(self.room, DialBank(DEFAULT_DIALS)).readings
            vector = self._encoder.encode(readings, self.zin_method)
        self.zin.append(vector)
        if len(self.zin) > self._cap:
            self.zin = self.zin[-self._cap:]
        return vector

    # ------------------------------------------------------------------ #
    # Z_out — prediction                                                 #
    # ------------------------------------------------------------------ #
    def predict(self) -> List[PredictionOutput]:
        """Z_in history -> Z_out predictions (TREND, VALUE_PREDICTION,
        ANOMALY_SCORE). No-op until at least two perceptions exist."""
        if len(self.zin) < 2:
            return []
        outputs = self.predictor.predict(self.zin)
        self.zout.append(outputs)
        if self.on_pulse is not None:
            self.on_pulse(self.zin[-1], outputs)
        return outputs

    # ------------------------------------------------------------------ #
    # The two new dials                                                  #
    # ------------------------------------------------------------------ #
    def trend_dial(self) -> float:
        """The room's own predicted direction: [-1 cooling .. +1 warming].

        A NEW kind of dial — it doesn't read what the room IS, it
        reads where the room is GOING. Auto-perceives/predicts if the
        bridge is quiet.
        """
        if not self.zout:
            if not self.zin:
                self.perceive()
            self.predict()
        for outputs in reversed(self.zout):
            for o in outputs:
                if o.prediction_type == PredictionType.TREND:
                    return float(np.clip(o.value, -1.0, 1.0))
        return 0.0

    def anomaly(self) -> float:
        """How unusual the room is vs its own recent pattern: [0, 1]."""
        if not self.zout:
            if not self.zin:
                self.perceive()
            self.predict()
        for outputs in reversed(self.zout):
            for o in outputs:
                if o.prediction_type == PredictionType.ANOMALY_SCORE:
                    return float(np.clip(o.value, 0.0, 1.0))
        return 0.0

    def dial(self) -> Dict[str, float]:
        """The Dual-DB as dial readings the field can read."""
        return {"trend_dial": self.trend_dial(), "anomaly": self.anomaly()}

    # ------------------------------------------------------------------ #
    # History access                                                     #
    # ------------------------------------------------------------------ #
    def perception_history(self) -> np.ndarray:
        """(N, d) array of Z_in vectors, oldest first."""
        if not self.zin:
            return np.zeros((0, 0), dtype=float)
        return np.stack(self.zin)

    def prediction_history(self,
                           ptype: Optional[PredictionType] = None) -> List[PredictionOutput]:
        """All Z_out outputs, flattened; optionally filtered by type."""
        flat = [o for outs in self.zout for o in outs]
        if ptype is not None:
            flat = [o for o in flat if o.prediction_type == ptype]
        return flat

    def __repr__(self) -> str:
        name = getattr(self.room, "name", type(self.room).__name__)
        return f"DualDBRoom({name!r}, zin={len(self.zin)}, zout={len(self.zout)})"


# ===================================================================== #
# The two new senses as dials — the pulse.py seam                        #
#                                                                        #
# Cast the Dual-DB's two predictions as real `Dial`s so they join any    #
# `DialBank` / `PulseLoop` / `read_field` unchanged. The trend dial is   #
# the forward-looking counterpart to pulse's backward-looking            #
# `direction()`: pulse reads where the room HAS been (last two readings) #
# — the trend dial reads where it is GOING. Together: past + future.     #
# ===================================================================== #
class TrendDial(Dial):
    """The Dual-DB's trend sense as a dial — the room predicting itself.

    Reads a `DualDBRoom`'s `trend_dial()`: the room's own predicted
    direction, [-1 cooling .. +1 warming]. A dial about the room, read
    by the room — and because it is a `Dial`, it slots straight into a
    `DialBank` (and so a `PulseLoop` / `read_field`) with no changes.
    """

    name = "trend_dial"
    description = "the room's own predicted direction, [-1 cooling .. +1 warming]"

    def __init__(self, bridge: "DualDBRoom"):
        self.bridge = bridge

    def read(self, room=None) -> float:
        return float(self.bridge.trend_dial())


class AnomalyDial(Dial):
    """The Dual-DB's anomaly sense as a dial — the room noticing
    something is off, [0, 1]."""

    name = "anomaly"
    description = "how unusual the room is vs its own recent pattern, [0, 1]"

    def __init__(self, bridge: "DualDBRoom"):
        self.bridge = bridge

    def read(self, room=None) -> float:
        return float(self.bridge.anomaly())

