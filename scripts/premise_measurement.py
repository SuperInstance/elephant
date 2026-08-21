"""Premise measurement — the Antecedent Test for the second-order doctrine.

The doctrine under test: the Nurse JEPA's second-order reading (the doctor
reads the nurse's CHANGE from her baseline) is only real if readers have
IDIOSYNCRATIC, STABLE baselines. The synthetic reader-delta test assumed this
premise; this script measures it in the field corpus (data/nights/*.jsonl).

Post-Switch-Test annotation (2026-08-21, zeroclaw-dissertation d59bf17 — NO
CLEAN WIN): "second-order" is retained as the structural term for baseline-
relativity only; the reader-delta object is a mean-shift, baseline-relative
delta (reads the step, not the change-of-reading). This script measures the
PREMISE (baseline-spread-to-drift ratio), not the object's temporal claim, and
its in-band result (0.5599 real-only / 0.4898 grounded) stands unaffected.

The devil's kill band: if the field's baseline-spread-to-drift ratio lands
below ~0.3-0.6 (corpus-sd units), the doctrine dies by its own registration.

What this script does (read-only against the logs; modifies nothing):

1. PER-READER DISPLACED FIELDS. The logs carry only ROOM-level field_eff
   (charisma-displaced once for the whole room, tapnight.py:239-257). For each
   reader R in each night's roster we compute R's personal reading of the room
   field by replaying the session dynamics from the log alone:

     s_R(t)      = 1 - exp(-charisma_R * n_R(t))          # n_R from interactions_after
     eff_R(t)    = clamp(raw(t) + s_R(t) * (vibe_R(t) - raw(t)))
                   # single-agent case of the engine's charisma displacement
                   # (tapnight.py:239-241: "for a single agent this reduces
                   #  exactly to charisma_pull(raw, vibe, charisma, n)")
     g_R         = w_R / max(w_R)                          # per-dial attention gain
                   # from R's dial_weights: the reader reads the room through
                   # their own dials; unattended dims attenuate toward neutral
     reading_R(t)= center + g_R * (eff_R(t) - center)
     vibe_R(t+1) = vibe_R(t) + (1 - exp(-rate_R)) * (field_eff_logged(t) - vibe_R(t))
                   # acclimation replay, exactly the engine's update
                   # (tapnight.py:254-257), using the LOGGED room field_eff

2. HARVEST the real readers (roster members across nights A, B, C, D, D-cold).

3. MEASURE per reader: (a) baseline = mean of their own readings (direction +
   magnitude), (b) baseline spread across readers in corpus-sd units,
   (c) drift = within-reader displacement across strata (SEG1 warm vs SEG2
   cynical in A/B/C; pre/post newcomer entry in D and D-cold), in corpus-sd.

4. THE KILL NUMBER: ratio = baseline_spread_z / mean_drift_z, with the
   three-way verdict (below 0.3 / in 0.3-0.6 / above 0.6).

5. HONEST N: the real roster has 7 readers < 10, and the nights share one
   schedule, so the >=10-readers / >=5-strata-transitions discipline is NOT
   met by the real corpus alone. We therefore ALSO run a clearly-labeled
   synthetic-grounded bootstrap: 13 extra readers sampled from the real
   roster's observed parameter distribution (charisma, acclimation_rate,
   dial_weights, vibe_start), each inheriting the interaction timeline of a
   real grounding archetype. Real-only and real+synthetic are reported
   SEPARATELY.

Run:  python3 scripts/premise_measurement.py
CPU-only, numpy only. Append-only corpus: this script writes nothing.
"""
from __future__ import annotations

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

# The premise corpus. Excluded: coarse-anchor (different room + roster — the
# sanity anchor, not the field under test) and night-A-repro (byte-replay of A).
NIGHT_FILES = {
    "A": "night-A.jsonl",
    "B": "night-B.jsonl",
    "C": "night-C.jsonl",
    "D": "night-D.jsonl",
    "D-cold": "night-D-cold.jsonl",
}

# Strata: A/B/C run the 40-message schedule (SEG1 warm seq 0-19, SEG2 cynical
# seq 20-39). D and D-cold run the 46-message schedule with the newcomer
# landing at seq 24; there the stratum is pre/post entry (a SEG split at 20
# would be confounded with the entry at 24).
def _strata_for(night, speaks):
    if night in ("A", "B", "C"):
        return {"SEG1": [r for r in speaks if r["seq"] <= 19],
                "SEG2": [r for r in speaks if r["seq"] >= 20]}
    entry = next(r["seq"] for r in speaks if r["author"] == "drifter")
    return {"pre-entry": [r for r in speaks if r["seq"] < entry],
            "post-entry": [r for r in speaks if r["seq"] >= entry]}


D = len(DIAL_NAMES)
CENTER = np.array([DIAL_CENTER[n] for n in DIAL_NAMES])
LO = np.array([DIAL_BOUNDS[n][0] for n in DIAL_NAMES])
HI = np.array([DIAL_BOUNDS[n][1] for n in DIAL_NAMES])

KILL_LO, KILL_HI = 0.3, 0.6
N_SYNTHETIC = 13  # 7 real + 13 synthetic-grounded = 20 (>=10 discipline)
SYNTH_SEED = 0


# --------------------------------------------------------------------------- #
# Log loading + per-reader displacement replay                                #
# --------------------------------------------------------------------------- #
def load_night(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    roster = next(r for r in rows if r["type"] == "session_open")["roster"]
    speaks = [r for r in rows if r["type"] == "speak"]
    return roster, speaks


def replay_readings(roster_entry, speaks, interaction_source=None):
    """Replay one reader's personal readings over one night's speak rows.

    roster_entry: {dial_weights, charisma, acclimation_rate, vibe_start}.
    interaction_source: name whose interactions_after timeline supplies n_R(t)
        (None => the reader's own name from the log). Returns list of
        (seq, reading 7-vector).
    """
    w = np.asarray(roster_entry["dial_weights"], dtype=float)
    g = w / w.max() if w.max() > 1e-12 else np.ones(D)
    charisma = float(roster_entry["charisma"])
    alpha = 1.0 - math.exp(-float(roster_entry["acclimation_rate"]))
    vibe = np.asarray(roster_entry["vibe_start"], dtype=float).copy()
    src = interaction_source or roster_entry["name"]

    out = []
    for row in speaks:
        raw = np.asarray(row["field_raw_after"], dtype=float)
        room_eff = np.asarray(row["field_eff_after"], dtype=float)
        n = row["interactions_after"].get(src, 0)
        s = 1.0 - math.exp(-charisma * n)
        eff = np.minimum(HI, np.maximum(LO, raw + s * (vibe - raw)))
        reading = CENTER + g * (eff - CENTER)
        out.append((row["seq"], reading))
        vibe = vibe + (room_eff - vibe) * alpha  # acclimation (engine order)
    return out


# --------------------------------------------------------------------------- #
# Measurement                                                                 #
# --------------------------------------------------------------------------- #
def corpus_sd(all_speaks):
    """RMS of the per-dial std of the raw room field over the whole corpus."""
    raw = np.array([r["field_raw_after"] for sp in all_speaks for r in sp])
    sd = raw.std(axis=0, ddof=1)
    return float(np.sqrt(np.mean(sd ** 2))), sd


def fit_readers(readers):
    """readers: {name: {"params": roster_entry, "nights": {night: source_name}}}.
    Returns {name: {"readings": {night: [(seq, vec)]}, "by_stratum":
    {(night, stratum): [vecs]}}}.
    """
    fitted = {}
    for name, spec in readers.items():
        per_night, by_stratum = {}, {}
        for night, src in spec["nights"].items():
            speaks = ALL_SPEAKS[night]
            readings = replay_readings(spec["params"], speaks,
                                       interaction_source=src)
            per_night[night] = readings
            for stratum, rows in STRATA[night].items():
                seqs = {r["seq"] for r in rows}
                vecs = [v for sq, v in readings if sq in seqs]
                if vecs:
                    by_stratum[(night, stratum)] = vecs
        fitted[name] = {"readings": per_night, "by_stratum": by_stratum}
    return fitted


def measure(fitted, sd_scalar):
    """Baselines, baseline spread, per-reader drift, and the kill ratio."""
    baselines = {}
    for name, f in fitted.items():
        vecs = [v for night in f["readings"].values() for _, v in night]
        b = np.mean(vecs, axis=0)
        n = float(np.linalg.norm(b))
        baselines[name] = {"vec": b, "magnitude": n,
                           "direction": b / n if n > 1e-12 else b,
                           "n_readings": len(vecs)}

    # Baseline spread: RMS across dials of the across-reader std of baselines.
    B = np.stack([baselines[r]["vec"] for r in fitted])
    spread = float(np.sqrt(np.mean(B.std(axis=0, ddof=1) ** 2)))
    spread_z = spread / sd_scalar

    # Drift: per reader, per strata transition, ||mean(stratum2)-mean(stratum1)||
    # in corpus-sd. Also the stratum-vs-own-baseline variant as a robustness check.
    drift, drift_vs_base, transitions = {}, {}, {}
    for name, f in fitted.items():
        ds, dvb = [], []
        for night in f["readings"]:
            strata = [s for (nt, s) in f["by_stratum"] if nt == night]
            if len(strata) < 2:
                continue
            m = {s: np.mean(f["by_stratum"][(night, s)], axis=0) for s in strata}
            d = float(np.linalg.norm(m[strata[1]] - m[strata[0]])) / sd_scalar
            ds.append(d)
            transitions.setdefault(f"{night}: {strata[0]}->{strata[1]}",
                                   []).append(d)
            for s in strata:
                dvb.append(float(np.linalg.norm(m[s] - baselines[name]["vec"]))
                           / sd_scalar)
        drift[name] = float(np.mean(ds)) if ds else float("nan")
        drift_vs_base[name] = float(np.mean(dvb)) if dvb else float("nan")

    mean_drift = float(np.nanmean(list(drift.values())))
    mean_dvb = float(np.nanmean(list(drift_vs_base.values())))
    ratio = spread_z / mean_drift if mean_drift > 1e-12 else float("inf")
    ratio_vb = spread_z / mean_dvb if mean_dvb > 1e-12 else float("inf")
    return {"baselines": baselines, "spread": spread, "spread_z": spread_z,
            "drift": drift, "drift_vs_base": drift_vs_base,
            "mean_drift_z": mean_drift, "mean_drift_vs_base_z": mean_dvb,
            "ratio": ratio, "ratio_vs_base": ratio_vb,
            "transitions": {k: float(np.mean(v)) for k, v in transitions.items()}}


def verdict(ratio):
    if ratio < KILL_LO:
        return "below band: doctrine dies by registration"
    if ratio <= KILL_HI:
        return "in band: indeterminate"
    return "above band: premise holds"


# --------------------------------------------------------------------------- #
# Synthetic-grounded bootstrap                                                #
# --------------------------------------------------------------------------- #
def synthesize(real_readers, n=N_SYNTHETIC, seed=SYNTH_SEED):
    """Sample n readers from the real roster's observed parameter
    distribution. Each synthetic reader is grounded on a real archetype
    (whose interaction timeline and night coverage it inherits), with:
      charisma, acclimation_rate ~ archetype value + N(0, observed sd),
                                  truncated to the observed range;
      dial_weights ~ archetype weights * lognormal(0, 0.15), renormalized;
      vibe_start   ~ archetype vibe + N(0, pooled per-dial sd), clipped to
                                  dial bounds.
    Clearly labeled synthetic-grounded: names carry the `synth-` prefix.
    """
    rng = np.random.default_rng(seed)
    names = list(real_readers)
    chars = np.array([real_readers[k]["params"]["charisma"] for k in names])
    rates = np.array([real_readers[k]["params"]["acclimation_rate"]
                      for k in names])
    vibes = np.stack([real_readers[k]["params"]["vibe_start"] for k in names])
    sd_char = max(float(chars.std(ddof=1)), 1e-3)
    sd_rate = max(float(rates.std(ddof=1)), 1e-3)
    sd_vibe = max(float(vibes.std(ddof=1)), 1e-3)

    synth = {}
    for i in range(n):
        base = names[rng.integers(len(names))]
        bp = real_readers[base]["params"]
        ch = float(np.clip(bp["charisma"] + rng.normal(0, sd_char),
                           chars.min(), chars.max()))
        ra = float(np.clip(bp["acclimation_rate"] + rng.normal(0, sd_rate),
                           rates.min(), rates.max()))
        w = np.asarray(bp["dial_weights"], dtype=float)
        w = w * np.exp(rng.normal(0, 0.15, size=D))
        w = (w / w.sum()).tolist()
        vb = np.asarray(bp["vibe_start"], dtype=float)
        vb = np.minimum(HI, np.maximum(LO, vb + rng.normal(0, sd_vibe, size=D)))
        synth[f"synth-{i:02d}({base})"] = {
            "params": {"name": f"synth-{i:02d}", "dial_weights": w,
                       "charisma": ch, "acclimation_rate": ra,
                       "vibe_start": vb.tolist()},
            "nights": dict(real_readers[base]["nights"]),
            "grounded_on": base,
        }
    return synth


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
ALL_SPEAKS, STRATA = {}, {}


def main():
    rosters = {}
    for night, fn in NIGHT_FILES.items():
        roster, speaks = load_night(os.path.join(NIGHTS_DIR, fn))
        rosters[night] = roster
        ALL_SPEAKS[night] = speaks
        STRATA[night] = _strata_for(night, speaks)

    sd_scalar, sd_per_dial = corpus_sd(list(ALL_SPEAKS.values()))

    # Harvest real readers: union of rosters. Params come from the first
    # roster that defines them (identical across A/B/C/D by construction).
    # Nights: every night whose roster contains them. The drifter is only a
    # roster member in night D (D-cold runs his lines with a neutral lazy
    # registration, i.e. no persona — so he is harvested from D only).
    real = {}
    for night, roster in rosters.items():
        for name, entry in roster.items():
            if name not in real:
                real[name] = {"params": dict(entry), "nights": {}}
            real[name]["nights"][night] = name  # own interaction timeline

    print("=" * 74)
    print("PREMISE MEASUREMENT — Antecedent Test (baseline spread vs drift)")
    print("=" * 74)
    print("\n[1] Per-reader displacement model (mirrors tapnight charisma at reader grain):")
    print("    s_R(t)       = 1 - exp(-charisma_R * n_R(t))     # n_R from log interactions_after")
    print("    eff_R(t)     = clamp(raw(t) + s_R(t) * (vibe_R(t) - raw(t)))")
    print("    g_R          = dial_weights_R / max(dial_weights_R)   # attention gain")
    print("    reading_R(t) = DIAL_CENTER + g_R * (eff_R(t) - DIAL_CENTER)")
    print("    vibe_R(t+1)  = vibe_R(t) + (1-exp(-rate_R)) * (field_eff_logged(t) - vibe_R(t))")

    print("\n[2] Corpus scale (raw field, all 5 nights, 206 speak events):")
    for n, s in zip(DIAL_NAMES, sd_per_dial):
        print(f"    sd[{n:<13}] = {s:.4f}")
    print(f"    corpus_sd (RMS over dials) = {sd_scalar:.4f}")

    n_real = len(real)
    print(f"\n[3] Real readers harvested: {n_real} -> {sorted(real)}")
    for name in sorted(real):
        print(f"    {name:<9} nights={sorted(real[name]['nights'])}")

    fitted_real = fit_readers(real)
    m_real = measure(fitted_real, sd_scalar)

    print("\n[4a] Per-reader baselines (REAL, fit from own readings only):")
    for name in sorted(real):
        b = m_real["baselines"][name]
        d = " ".join(f"{x:+.3f}" for x in b["direction"])
        print(f"    {name:<9} |baseline|={b['magnitude']:.4f} "
              f"direction=[{d}] n_readings={b['n_readings']}")

    print(f"\n[4b] Baseline spread across readers (REAL): "
          f"{m_real['spread']:.4f} = {m_real['spread_z']:.4f} corpus-sd")

    print("\n[4c] Drift per reader (REAL, corpus-sd; mean over strata transitions):")
    for name in sorted(real):
        print(f"    {name:<9} drift={m_real['drift'][name]:.4f} "
              f"(vs-own-baseline variant: {m_real['drift_vs_base'][name]:.4f})")
    print("    per-transition means (corpus-sd):")
    for k, v in m_real["transitions"].items():
        print(f"      {k:<28} {v:.4f}")
    print(f"    mean drift (reader-mean) = {m_real['mean_drift_z']:.4f} corpus-sd")

    print("\n[5] THE KILL NUMBER (REAL ONLY):")
    print(f"    baseline-spread / drift = {m_real['spread_z']:.4f} / "
          f"{m_real['mean_drift_z']:.4f} = {m_real['ratio']:.4f}")
    print(f"    robustness (drift-vs-own-baseline): ratio = {m_real['ratio_vs_base']:.4f}")
    print(f"    kill band = [{KILL_LO}, {KILL_HI}] corpus-sd")
    print(f"    VERDICT (real-only): {verdict(m_real['ratio'])}")

    # --- synthetic-grounded bootstrap ------------------------------------- #
    synth = synthesize(real)
    merged = dict(real)
    merged.update(synth)
    fitted_all = fit_readers(merged)
    m_all = measure(fitted_all, sd_scalar)

    print(f"\n[6] N CAVEAT + synthetic-grounded bootstrap:")
    print(f"    real readers = {n_real} (< 10 required); distinct schedules = 2 "
          f"(SEG shift; newcomer entry), A/B/C byte-share one schedule")
    print(f"    -> >=10-readers / >=5-strata-transitions discipline NOT met by real corpus")
    print(f"    bootstrap: {len(synth)} synthetic-grounded readers "
          f"(seed={SYNTH_SEED}), sampled from the real roster's observed")
    print(f"    charisma/acclimation/dial_weights/vibe_start distribution, each")
    print(f"    inheriting a real archetype's interaction timeline; labeled `synth-*`")
    print(f"    real+synthetic: N={len(merged)} readers")
    print(f"    baseline spread = {m_all['spread_z']:.4f} corpus-sd; "
          f"mean drift = {m_all['mean_drift_z']:.4f} corpus-sd")
    print(f"    KILL NUMBER (real+synthetic) = {m_all['ratio']:.4f} "
          f"(vs-own-baseline variant: {m_all['ratio_vs_base']:.4f})")
    print(f"    VERDICT (real+synthetic): {verdict(m_all['ratio'])}")

    print("\n[7] ONE-SENTENCE VERDICTS:")
    print(f"    real-only:       ratio={m_real['ratio']:.4f} -> {verdict(m_real['ratio'])}")
    print(f"    real+synthetic:  ratio={m_all['ratio']:.4f} -> {verdict(m_all['ratio'])}")


if __name__ == "__main__":
    main()
