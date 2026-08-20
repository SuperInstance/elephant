"""E2 instrument — premise measurement on the schedule-diversified corpus.

Implements the E2 registration (research/prototype/e2-antecedent-test/
REGISTRATION-2026-08-19.md) quantities:

  reading_R(t) = CENTER + g_R ⊙ (field_eff_to_reader[R](t) − CENTER)
    — consumed from logged v:2 facts on new nights, reconstructed by the
      premise-measurement replay on v:1 nights (identical equations; the
      equality is asserted numerically on the v:2 nights).

  drift(reader)  = mean over signal strata transitions of
                   ‖mean(k+1) − mean(k)‖ / corpus_sd
  E-seg spread   = RMS over (night, stratum) cells of [RMS over dials of the
                   across-reader sd (ddof=1) of per-cell MEDIAN baselines]
  E-cont spread  = RMS over dials of across-reader sd of global-mean
                   baselines (the pre-measurement §4 estimator)
  ratio          = spread / mean drift        (corpus-sd units)
  ICC            = per-dial σ²_between / (σ²_between + σ²_within) on
                   per-(reader,night) median baselines, schedule means
                   removed; aggregate = unweighted mean over dials.

numpy-only, CPU, read-only against the nights. Bootstrap seed: 20260819.
"""
from __future__ import annotations

import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elephant.field import DIAL_NAMES
from elephant.tapnight import DIAL_BOUNDS, DIAL_CENTER

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NIGHTS_DIR = os.path.join(ROOT, "data", "nights")
PERSONAS = os.path.join(ROOT, "data", "e2", "e2-personas.json")

D = 7
CENTER = np.array([DIAL_CENTER[n] for n in DIAL_NAMES])
LO = np.array([DIAL_BOUNDS[n][0] for n in DIAL_NAMES])
HI = np.array([DIAL_BOUNDS[n][1] for n in DIAL_NAMES])
BOOT_SEED = 20260819
KILL_LO, KILL_HI = 0.3, 0.6

# Registered strata: (label, lo, hi, kind); transitions = consecutive pairs.
NIGHT_SPECS = {
    "A":      ("night-A.jsonl", [("warm", 0, 19, "signal"), ("cynical", 20, 39, "signal")]),
    "B":      ("night-B.jsonl", [("warm", 0, 19, "signal"), ("cynical", 20, 39, "signal")]),
    "C":      ("night-C.jsonl", [("warm", 0, 19, "signal"), ("cynical", 20, 39, "signal")]),
    "D":      ("night-D.jsonl", [("pre", 0, 23, "signal"), ("post", 24, 45, "signal")]),
    "D-cold": ("night-D-cold.jsonl", [("pre", 0, 23, "signal"), ("post", 24, 45, "signal")]),
    "S1":     ("night-S1.jsonl", [("warm", 0, 19, "signal"), ("cynical", 20, 39, "signal")]),
    "S2":     ("night-S2.jsonl", [("warm", 0, 7, "signal"), ("cynical", 8, 27, "signal")]),
    "S3":     ("night-S3.jsonl", [("warm", 0, 19, "signal"), ("cynical", 20, 27, "signal")]),
    "S4a":    ("night-S4a.jsonl", [("warm-pre", 0, 11, "signal"), ("warm-entry", 12, 19, "signal"), ("cynical", 20, 45, "signal")]),
    "S4b":    ("night-S4b.jsonl", [("warm", 0, 19, "signal"), ("cynical-pre", 20, 27, "signal"), ("cynical-entry", 28, 44, "signal")]),
    "S5":     ("night-S5.jsonl", [("warm-a", 0, 9, "null"), ("warm-b", 10, 19, "null")]),
}

PRIMARY_NIGHTS = ["A", "D", "D-cold", "S1", "S2", "S3", "S4a", "S4b", "S5"]
SIGNAL_NIGHTS = ["A", "D", "D-cold", "S1", "S2", "S3", "S4a", "S4b"]
ORIG6 = ["writer", "poet", "essayist", "engineer", "critic", "captain"]

# Registered field attendance (reader -> nights present as a persona reader).
FIELD_NIGHTS = {
    **{n: list(PRIMARY_NIGHTS) for n in ORIG6},
    "drifter": ["D", "S4a", "S4b"],
    "barkeep": ["S1", "S3"],
    "fiddler": ["S1", "S4a"],
    "cartographer": ["S1", "S4b"],
    "singer": ["S2", "S4a"],
    "lamplighter": ["S2", "S5"],
    "tinker": ["S2", "S4b"],
    "blacksmith": ["S3", "S4b"],
    "weaver": ["S3", "S5"],
}

# Cold-entry nights per reader (readings begin at the reader's first speak).
COLD_ENTRY = {"drifter": ["S4a", "S4b"]}


def archetype_labels():
    doc = json.load(open(PERSONAS, encoding="utf-8"))
    labels = {n: n for n in ORIG6}
    labels["drifter"] = "drifter"
    for n, p in doc["new_personas"].items():
        labels[n] = p["archetype"]
    return labels


class Night:
    def __init__(self, name):
        fn, strata = NIGHT_SPECS[name]
        self.name = name
        self.path = os.path.join(NIGHTS_DIR, fn)
        rows = [json.loads(l) for l in open(self.path, encoding="utf-8") if l.strip()]
        self.open = next(r for r in rows if r["type"] == "session_open")
        self.speaks = [r for r in rows if r["type"] == "speak"]
        self.strata = strata
        self.v2 = "readers" in self.speaks[0]
        self.params = {}
        for n, p in self.open["roster"].items():
            self.params[n] = dict(p)
        for n, p in self.open.get("staged_entries", {}).items():
            self.params.setdefault(n, dict(p))
        for n in self.params:
            self.params[n]["dial_weights"] = np.asarray(self.params[n]["dial_weights"], float)
            self.params[n]["vibe_start"] = np.asarray(self.params[n]["vibe_start"], float)
        # canonical presence timeline (addendum A3): per-speak mean attendee
        # interaction count — a logged, reader-independent fact of the night
        self.canon_n = [float(np.mean(list(r["interactions_after"].values())))
                        if r["interactions_after"] else 0.0
                        for r in self.speaks]

    def first_speak_seq(self, author):
        for r in self.speaks:
            if r["author"] == author:
                return r["seq"]
        return None


def replay_readings(params, speaks, source, start_seq=None, canon_n=None):
    """The premise-measurement replay. `start_seq` models a cold entry:
    no readings and NO vibe evolution before the entry speak (the engine
    does not acclimate unregistered participants). `canon_n` (addendum A3)
    substitutes the per-speak mean attendee interaction count for the
    reader's own count — canonical presence, participation-deconfounded."""
    w = np.asarray(params["dial_weights"], float)
    g = w / w.max() if w.max() > 1e-12 else np.ones(D)
    charisma = float(params["charisma"])
    alpha = 1.0 - math.exp(-float(params["acclimation_rate"]))
    vibe = np.asarray(params["vibe_start"], float).copy()
    out = []
    for i, row in enumerate(speaks):
        if start_seq is not None and row["seq"] < start_seq:
            continue
        raw = np.asarray(row["field_raw_after"], float)
        room_eff = np.asarray(row["field_eff_after"], float)
        n = canon_n[i] if canon_n is not None else row["interactions_after"].get(source, 0)
        s = 1.0 - math.exp(-charisma * n)
        eff = np.minimum(HI, np.maximum(LO, raw + s * (vibe - raw)))
        out.append((row["seq"], CENTER + g * (eff - CENTER)))
        vibe = vibe + (room_eff - vibe) * alpha
    return out


def logged_readings(night, reader):
    """(seq, reading) pairs from logged v:2 facts; presence from first
    logged appearance (cold entrants read from entry onward)."""
    w = np.asarray(night.params[reader]["dial_weights"], float)
    g = w / w.max() if w.max() > 1e-12 else np.ones(D)
    out = []
    for r in night.speaks:
        blk = r.get("readers", {}).get(reader)
        if blk is None:
            continue
        eff = np.asarray(blk["field_eff_to_reader"], float)
        out.append((r["seq"], CENTER + g * (eff - CENTER)))
    return out


def readings_for(night, reader, params=None, source=None, cold=False,
                 presence="actual"):
    """Readings of `reader` on `night`.
    presence="actual", params=None (field mode): logged facts on v:2 nights,
    replay from the night's own roster on v:1 nights. params given
    (synthetic/ladder mode): replay with those params. presence="canonical"
    (addendum A3): replay with the night's canonical presence timeline
    (params default to the night's roster entry for real readers)."""
    start = night.first_speak_seq(reader) if cold else None
    if presence == "canonical":
        p = params if params is not None else night.params[reader]
        return replay_readings(p, night.speaks, source or reader, start,
                               canon_n=night.canon_n)
    if params is not None:
        return replay_readings(params, night.speaks, source or reader, start)
    if night.v2:
        return logged_readings(night, reader)
    return replay_readings(night.params[reader], night.speaks, reader, start)


def assert_replay_matches_log(night, reader, cold=False):
    """Field-channel consistency: the v:1 replay equations must reproduce
    the logged v:2 field_eff_to_reader exactly (same engine semantics)."""
    logged = logged_readings(night, reader)
    if not logged:
        return
    replayed = replay_readings(night.params[reader], night.speaks, reader,
                               night.first_speak_seq(reader) if cold else None)
    a = np.array([v for _, v in replayed][-len(logged):])
    b = np.array([v for _, v in logged])
    assert np.abs(a - b).max() < 1e-9, \
        f"replay!=log on {night.name}/{reader}: {np.abs(a-b).max()}"


def corpus_sd(nights):
    raw = np.array([r["field_raw_after"] for n in nights for r in n.speaks])
    sd = raw.std(axis=0, ddof=1)
    return float(np.sqrt(np.mean(sd ** 2))), sd


def cell_vecs(seq_vecs, lo, hi):
    return np.array([v for sq, v in seq_vecs if lo <= sq <= hi])


class Measurement:
    """Per-reader cached quantities + both estimators + ICC + bootstrap."""

    def __init__(self, readers_nights, sd, include_nights=None,
                 include_null_drift=False, presence="actual"):
        """readers_nights: {reader: {"params": dict|None,
                                     "nights": {night: source_name},
                                     "cold": [nights]}}.
        sd: corpus_sd scalar. include_nights: nights allowed (default
        PRIMARY_NIGHTS). include_null_drift: S5 pseudo-transition in drift."""
        self.sd = sd
        allow = list(include_nights) if include_nights is not None else PRIMARY_NIGHTS
        self.nights = {n: Night(n) for n in allow}
        self.include_null = include_null_drift
        self.presence = presence
        self.readers = sorted(readers_nights)
        self.arch = archetype_labels()
        self._build(readers_nights)

    def _build(self, readers_nights):
        self.readings = {}
        for r in self.readers:
            spec = readers_nights[r]
            cold = set(spec.get("cold", []))
            self.readings[r] = {}
            for night in spec["nights"]:
                if night not in self.nights:
                    continue
                seq_vecs = readings_for(self.nights[night], r,
                                        params=spec.get("params"),
                                        source=spec["nights"][night],
                                        cold=night in cold,
                                        presence=self.presence)
                if seq_vecs:
                    self.readings[r][night] = seq_vecs

        self.cell_base, self.night_base = {}, {}
        for night in self.nights:
            for label, lo, hi, kind in self.nights[night].strata:
                cell = {}
                for r in self.readers:
                    if night not in self.readings[r]:
                        continue
                    v = cell_vecs(self.readings[r][night], lo, hi)
                    if len(v):
                        cell[r] = np.median(v, axis=0)
                if cell:
                    self.cell_base[(night, label)] = cell
            nb = {}
            for r in self.readers:
                if night in self.readings[r]:
                    v = np.array([x for _, x in self.readings[r][night]])
                    nb[r] = np.median(v, axis=0)
            if nb:
                self.night_base[night] = nb

        self.drift, self.trans = {}, {}
        for r in self.readers:
            sig, null, tr = [], [], []
            for night in sorted(self.readings[r]):
                strata = self.nights[night].strata
                for (l0, lo0, hi0, k0), (l1, lo1, hi1, k1) in zip(strata, strata[1:]):
                    a = cell_vecs(self.readings[r][night], lo0, hi0)
                    b = cell_vecs(self.readings[r][night], lo1, hi1)
                    if not len(a) or not len(b):
                        continue
                    val = float(np.linalg.norm(b.mean(0) - a.mean(0))) / self.sd
                    tr.append((night, f"{l0}->{l1}", val))
                    if k0 == "null" or k1 == "null":
                        null.append(val)
                    else:
                        sig.append(val)
            self.drift[r] = (float(np.mean(sig)) if sig else np.nan,
                             float(np.mean(null)) if null else np.nan)
            self.trans[r] = tr

        self.cell_order = [k for k in self.cell_base
                           if self._kind(k) != "null" or self.include_null]

    def _kind(self, cell_key):
        night, label = cell_key
        for l, lo, hi, kind in self.nights[night].strata:
            if l == label:
                return kind
        return "signal"

    # ---------------- point estimators (multiset-aware) ---------------- #
    def _multiset(self, readers):
        return list(self.readers) if readers is None else list(readers)

    def drift_mean(self, readers=None):
        rs = self._multiset(readers)
        vals = [self.drift[r][0] for r in rs if not np.isnan(self.drift[r][0])]
        return float(np.mean(vals)) if vals else float("nan")

    def null_drift_mean(self):
        vals = [self.drift[r][1] for r in self.readers
                if not np.isnan(self.drift[r][1])]
        return float(np.mean(vals)) if vals else float("nan")

    def spread_seg(self, readers=None, class_residual=False):
        rs = self._multiset(readers)
        sqs = []
        for key in self.cell_order:
            present = [r for r in rs if r in self.cell_base[key]]
            if len(present) < 2:
                continue
            vecs = {r: self.cell_base[key][r] for r in present}
            if class_residual:
                groups = collections.defaultdict(list)
                for r in present:
                    groups[self.arch[r]].append(r)
                for r in present:
                    gm = np.mean([vecs[x] for x in groups[self.arch[r]]], axis=0)
                    vecs[r] = vecs[r] - gm
            B = np.stack([vecs[r] for r in present])
            sqs.append(float(np.mean(B.std(axis=0, ddof=1) ** 2)))
        return float(np.sqrt(np.mean(sqs))) / self.sd if sqs else float("nan")

    def spread_cont(self, readers=None):
        rs = self._multiset(readers)
        base = {}
        for r in rs:
            vecs = [v for night in self.readings[r]
                    for _, v in self.readings[r][night]]
            if vecs:
                base[r] = np.mean(vecs, axis=0)
        if len(base) < 2:
            return float("nan")
        B = np.stack([base[r] for r in rs if r in base])
        return float(np.sqrt(np.mean(B.std(axis=0, ddof=1) ** 2))) / self.sd

    def ratio_seg(self, readers=None):
        d = self.drift_mean(readers)
        s = self.spread_seg(readers)
        return s / d if d and not np.isnan(d) and not np.isnan(s) else float("nan")

    def ratio_cont(self, readers=None):
        d = self.drift_mean(readers)
        s = self.spread_cont(readers)
        return s / d if d and not np.isnan(d) and not np.isnan(s) else float("nan")

    # ---------------- ICC ---------------- #
    def icc(self, readers=None):
        """(aggregate, per-dial). Night (schedule) means removed first."""
        rs = self._multiset(readers)
        per_dial = {}
        for d in range(D):
            rows = []
            for night, nb in self.night_base.items():
                vals = [nb[r][d] for r in rs if r in nb]
                if len(vals) < 2:
                    continue
                m = float(np.mean(vals))
                for r in rs:
                    if r in nb:
                        rows.append((r, nb[r][d] - m))
            if not rows:
                per_dial[DIAL_NAMES[d]] = float("nan")
                continue
            by_r = collections.defaultdict(list)
            for r, v in rows:
                by_r[r].append(v)
            within = [float(np.var(vs, ddof=1)) for vs in by_r.values()
                      if len(vs) > 1]
            s2w = float(np.mean(within)) if within else 0.0
            means = [float(np.mean(vs)) for vs in by_r.values()]
            s2b = float(np.var(means, ddof=1)) if len(means) > 1 else 0.0
            per_dial[DIAL_NAMES[d]] = (s2b / (s2b + s2w)
                                       if (s2b + s2w) > 0 else float("nan"))
        finite = [v for v in per_dial.values() if not np.isnan(v)]
        agg = float(np.mean(finite)) if finite else float("nan")
        return agg, per_dial

    # ---------------- bootstrap over readers ---------------- #
    def bootstrap(self, B=2000, seed=BOOT_SEED):
        rng = np.random.default_rng(seed)
        n = len(self.readers)
        seg, cont, iccs = [], [], []
        for _ in range(B):
            rs = [self.readers[i] for i in rng.integers(0, n, n)]
            for val, sink in ((self.ratio_seg(rs), seg),
                              (self.ratio_cont(rs), cont)):
                if not np.isnan(val):
                    sink.append(val)
            agg, _ = self.icc(rs)
            if not np.isnan(agg):
                iccs.append(agg)

        def ci(xs):
            return (float(np.percentile(xs, 2.5)),
                    float(np.percentile(xs, 97.5))) if xs else (float("nan"),) * 2

        return {"seg_ci": ci(seg), "cont_ci": ci(cont), "icc_ci": ci(iccs),
                "draws": len(seg)}


def verdict(lo, hi):
    if lo > KILL_HI:
        return "CLEAR: CI entirely above 0.6 — premise holds (field)"
    if hi < KILL_LO:
        return "KILL: CI entirely below 0.3 — doctrine dies by its own registration"
    return "INDETERMINATE: CI touches the 0.3–0.6 band"


def power_analysis(point, lo, hi, n_readers):
    h = (hi - lo) / 2.0
    d = min(abs(point - KILL_LO), abs(point - KILL_HI))
    if d <= 0 or h <= 0:
        return None
    return {"ci_half_width": h, "distance_to_nearest_edge": d,
            "n_required": int(np.ceil(n_readers * (h / d) ** 2))}
