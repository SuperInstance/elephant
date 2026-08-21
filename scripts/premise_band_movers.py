"""E2/E3 PREMISE BAND-MOVERS — the registered temporal/exposure decomposition.

Implements E2E3-premise-band-movers-design-2026-08-21.md (registered as a
dated addendum to research/topic.md, dissertation repo, before this run):

  Moving-window premise score rho_R(t) = o_R(t)/d_R(t) on W=12-speak windows
  (stride 1): o_R = idiosyncratic offset of R's windowed mean from the
  present-roster windowed mean; d_R = split-half displacement (6/6); both in
  corpus_sd units so the 0.3/0.6 kill band applies unchanged. Band states
  with hysteresis (>=0.05 beyond the edge, >=3 consecutive windows). Four
  registered legs, per wave, never pooled:

    A  timing     mean over counted down-crossings of 1[within 3 speaks of a
                  registered strata transition] vs 10,000 circular-shift
                  nulls per reader-night (seed 20260821)
    D  direction  fraction of night-level signal transitions with a counted
                  down-crossing in +-3 speaks vs the null-night pseudo-
                  transition rate (T9/S5 at their midpoints); exact binomial
    P  persistence cosine of pre- vs post-transition offset vectors over the
                  roster (reliable subspace) vs persistence-at-rest;
                  P_trans >= 0.5 x P_rest => idiosyncrasy survives the step
    S  exposure   per-night median score ~ night warmth x + reader FE vs a
                  roster-composition competitor; reader-clustered bootstrap
                  B=2000 + nested permutation 10,000

  PRIMARY = wave-2 T-nights (21 readers, 66 signal + 7 null reader-nights);
  REPLICATION = wave-1 (15 readers, 65 signal + 8 null reader-nights;
  orig-6 span all nine primary nights, S5 null). A/B/C/D/D-cold (v:1,
  replay-mediated) anchor the estimator-continuity ladder only. S6/S7 and
  night-H* excluded. Channel: canonical presence (primary, registered);
  actual presence = labeled sensitivity.

  Kill/void rules (design sec.5) applied honestly; the bare crossing rate is
  never read as evidence (sec.0 tautology guard — only A, P, S carry
  content; D carries direction structure).

numpy-only analysis; scipy.stats only for the exact binomial CI. Read-only
against data/nights/; the Stage-2 wave gate is INVOKED, not edited.
Seeds: 20260821 family. Run:  python3 scripts/premise_band_movers.py
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats as sstats

from elephant.field import DIAL_NAMES
from elephant.vmf import CENTER, SCALE, WARM
from scripts.e2_field import field_readers
from scripts.e2_instrument import (FIELD_NIGHTS, FIELD_NIGHTS_W2,
                                   PRIMARY_NIGHTS, W2_NIGHT_LIST,
                                   Measurement, Night, archetype_labels,
                                   assert_replay_matches_log, corpus_sd)
from scripts.slope_regression import room_warmth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_SLOPE = os.path.join(ROOT, "data", "slope", "premise-band-movers-results.json")
OUT_DIR = os.path.join(ROOT, "data", "premise-band-movers")

SEED = 20260821
N_SHIFT = 10_000      # circular-shift null draws (leg A)
B_BOOT = 2_000        # reader-clustered bootstrap (leg S) / event bootstrap (leg P)
N_PERM = 10_000       # nested permutation (leg S competitor)
W_PRIMARY = 12
HYST_MARGIN = 0.05
HYST_HOLD = 3
EDGE_LO, EDGE_HI = 0.3, 0.6
TOL = 3               # speaks: crossing-to-transition tolerance
RHO_CAP = 100.0
CENTER_OFF = (W_PRIMARY - 1) / 2.0   # window-center position offset (deviation
                                       # note 1: speak-position referent for
                                       # crossing timing/coverage statistics)

RELIABLE = ["mood", "volume", "earnestness", "presence"]   # ICC reliable subspace
RIDX = [DIAL_NAMES.index(d) for d in RELIABLE]
WARM_REL = WARM[RIDX] / np.linalg.norm(WARM[RIDX])

# Filed warmth ladder (roster-invariant, a priori; stage2_wave_gate.LADDER +
# SLOPE-REGRESSION-2026-08-20.md sec.1). x_N for leg S / trajectory.
X_W2 = {"T1": 0.6551, "T2": 0.3187, "T3": 0.6551, "T4a": 0.4465,
        "T4b": 0.6319, "T5": 0.6293, "T5c": 0.6293, "T8": 0.7409,
        "T9": 0.7589}
X_W1 = {"S1": 0.6551, "S2": 0.3187, "S3": 0.7409, "S4a": 0.4465,
        "S4b": 0.6319, "D": 0.6293, "D-cold": 0.6293, "A": 0.6551,
        "S5": 0.7589}

SIGNAL_W2 = ["T1", "T2", "T3", "T4a", "T4b", "T5", "T5c", "T8"]
SIGNAL_W1 = ["A", "D", "D-cold", "S1", "S2", "S3", "S4a", "S4b"]
NULL_W2 = ["T9"]
NULL_W1 = ["S5"]

S = ["kill", "in", "clear"]  # state indices 0/1/2


# --------------------------------------------------------------------------- #
# Hysteresis band-state machine (design sec.3)                                #
# --------------------------------------------------------------------------- #
def plain_state(x: float) -> int:
    if x > EDGE_HI:
        return 2
    if x < EDGE_LO:
        return 0
    return 1


def entry_ok(cur: int, tgt: int, x: float) -> bool:
    """x must sit >= HYST_MARGIN beyond EVERY edge between cur and tgt."""
    if cur == tgt:
        return False
    if tgt > cur:                                  # upward move
        if cur == 0 and tgt == 1:
            return x >= EDGE_LO + HYST_MARGIN
        return x >= EDGE_HI + HYST_MARGIN          # 1->2 and 0->2 (both edges)
    if cur == 2 and tgt == 1:                      # downward move
        return x <= EDGE_HI - HYST_MARGIN
    return x <= EDGE_LO - HYST_MARGIN              # 1->0 and 2->0 (both edges)


def counted_crossings(vals: np.ndarray) -> list[dict]:
    """Hysteresis-counted crossings over a FINITE (no-NaN) rho segment.
    Crossing position = index of the first window of the >=HYST_HOLD
    confirmation run. A direct clear->kill (or kill->clear) confirmed move
    records one event per edge crossed (disclosed convention)."""
    n = len(vals)
    if n == 0:
        return []
    cur = plain_state(float(vals[0]))
    cand = None
    events = []
    for i in range(n):
        x = float(vals[i])
        tgt = plain_state(x)
        if cand is not None and (tgt != cand[0]
                                 or not entry_ok(cur, cand[0], x)):
            cand = None
        if cand is None and tgt != cur and entry_ok(cur, tgt, x):
            cand = (tgt, i)
        if cand is not None and i - cand[1] + 1 >= HYST_HOLD:
            j0, j1 = cur, cand[0]
            step = 1 if j1 > j0 else -1
            for k in range(j0, j1, step):
                edge = EDGE_LO if min(k, k + step) == 0 else EDGE_HI
                events.append({"pos": int(cand[1]), "edge": float(edge),
                               "dir": "up" if step > 0 else "down"})
            cur = j1
            cand = None
    return events


def finite_series(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(values, night-positions) of the reader's finite rho segment."""
    mask = ~np.isnan(rho)
    return rho[mask], np.flatnonzero(mask)


# --------------------------------------------------------------------------- #
# Moving-window estimator (design sec.3)                                      #
# --------------------------------------------------------------------------- #
def night_windows(m: Measurement, night: str, sd: float, W: int,
                  class_residual: bool = False) -> dict:
    """Per-night windowed o/d/rho (corpus-sd units) + population R(t).

    Presence = roster membership with full-window coverage: a reader is in
    window t iff they have a reading at every speak in [t, t+W-1] (windows
    before a cold entrant's first speak simply lack that reader).

    Norm convention (pinned by the continuity ladder, design sec.0/sec.3):
    numerator o_R = RMS over dials of the deviation from the reference
    mean, /corpus_sd (the FILED E2 spread convention — the dial-RMS scale
    of the honesty guard's "numerator ≈ 0.46-0.56"); denominator d_R =
    Euclidean split-half displacement norm /corpus_sd (the FILED drift
    convention: 0.29 noise floor, 0.75-0.93 transition spikes). Population
    RMS_R o_R reproduces the filed spread and mean_R d_R the filed drift,
    so R(t) is commensurate with the filed ratio channel and 0.3/0.6 apply
    unchanged (a Euclidean 7-norm numerator would inflate the score
    ~sqrt(7) and break the ladder; deviation disclosed in the run doc).
    """
    sp = m.nights[night].speaks
    T = len(sp)
    npos = T - W + 1
    readers = [r for r in m.readers if night in m.readings[r]]
    o = {r: np.full(npos, np.nan) for r in readers}
    d = {r: np.full(npos, np.nan) for r in readers}
    rho = {r: np.full(npos, np.nan) for r in readers}
    present = {r: np.zeros(npos, bool) for r in readers}
    Rt = np.full(npos, np.nan)
    if npos <= 0:
        return {"T": T, "positions": [], "readers": readers, "rho": rho,
                "o": o, "d": d, "Rt": Rt.tolist(), "present": present}

    half = W // 2
    seqd = {r: dict(m.readings[r][night]) for r in readers}
    arch = m.arch

    for t in range(npos):
        cur, vecs = [], {}
        for r in readers:
            vs = [seqd[r].get(sp[i]["seq"]) for i in range(t, t + W)]
            if all(v is not None for v in vs):
                cur.append(r)
                vecs[r] = np.stack(vs)
        if len(cur) < 2:
            continue
        M = {r: vecs[r].mean(axis=0) for r in cur}
        if class_residual:
            groups = {}
            for r in cur:
                groups.setdefault(arch[r], []).append(r)
            gmean = {a: np.mean([M[r] for r in rs], axis=0)
                     for a, rs in groups.items()}
            ref = {r: gmean[arch[r]] for r in cur}
        else:
            bbar = np.mean([M[r] for r in cur], axis=0)
            ref = {r: bbar for r in cur}
        os_, ds_ = [], []
        for r in cur:
            ov = float(np.sqrt(np.mean((M[r] - ref[r]) ** 2))) / sd
            dv = float(np.linalg.norm(vecs[r][half:].mean(axis=0)
                                      - vecs[r][:half].mean(axis=0))) / sd
            o[r][t], d[r][t] = ov, dv
            rho[r][t] = min(ov / dv if dv > 1e-12 else RHO_CAP, RHO_CAP)
            os_.append(ov)
            ds_.append(dv)
            present[r][t] = True
        Rt[t] = float(np.sqrt(np.mean(np.square(os_))) / np.mean(ds_))

    return {"T": T, "positions": list(range(npos)), "readers": readers,
            "rho": rho, "o": o, "d": d, "Rt": Rt.tolist(),
            "present": present}


def strata_transitions(m: Measurement, night: str) -> list[dict]:
    """Registered consecutive-strata transitions; boundary = first speak seq
    of the new stratum (seqs contiguous 0..T-1, asserted at load)."""
    strata = m.nights[night].strata
    out = []
    for (l0, lo0, hi0, k0), (l1, lo1, hi1, k1) in zip(strata, strata[1:]):
        out.append({"from": l0, "to": l1, "boundary": lo1,
                    "kind": "signal" if (k0 == "signal" and k1 == "signal")
                            else "null"})
    return out


# --------------------------------------------------------------------------- #
# Leg A — timing vs circular-shift null                                       #
# --------------------------------------------------------------------------- #
def _shift_table(vals: np.ndarray, pos: np.ndarray, anchors: list[float],
                  direction: str) -> tuple[np.ndarray, np.ndarray]:
    """For every circular shift s of the finite segment: (n_events,
    sum_of_timing_indicators). Event speak-position = window start +
    (W-1)/2 (the window's temporal center — the position referent that
    makes the registered +-3-speaks tolerance meaningful: the causal event
    of a transition dip is the half-split, which sits at the center; a
    window-START referent is arithmetically blind to the design's own
    predicted effect, since W/2 = 6 > TOL = 3. Deviation note 1 in the run
    doc; the start-referent value is carried as a labeled sensitivity)."""
    n = len(vals)
    nev = np.zeros(n)
    sind = np.zeros(n)
    for s in range(n):
        for e in counted_crossings(np.roll(vals, s)):
            if e["dir"] != direction:
                continue
            p = float(pos[e["pos"]]) + CENTER_OFF
            ind = 1 if (anchors and min(abs(p - a) for a in anchors) <= TOL) \
                else 0
            nev[s] += 1
            sind[s] += ind
    return nev, sind


def leg_A(win: dict, m: Measurement, signal_nights: list[str],
          seed: int, n_draws: int = N_SHIFT, direction: str = "down",
          mid_anchor: bool = False, pos_ref: str = "center") -> dict:
    rng = np.random.default_rng(seed)
    rn = []
    for night in signal_nights:
        w = win[night]
        bounds = [t["boundary"] for t in strata_transitions(m, night)
                  if t["kind"] == "signal"]
        anchors = [b + W_PRIMARY // 2 for b in bounds] if mid_anchor \
            else bounds
        for r in w["readers"]:
            vals, pos = finite_series(w["rho"][r])
            if len(vals) == 0:
                continue
            rn.append((night, r, vals, pos, anchors))

    def speak_pos(t0, pos_arr):
        return float(pos_arr[t0]) + (CENTER_OFF if pos_ref == "center" else 0.0)

    obs_events = []
    for night, r, vals, pos, anchors in rn:
        for e in counted_crossings(vals):
            if e["dir"] != direction:
                continue
            p = speak_pos(e["pos"], pos)
            obs_events.append({"night": night, "reader": r,
                               "pos": p, "edge": e["edge"],
                               "within": 1 if (anchors and
                                               min(abs(p - a) for a in anchors)
                                               <= TOL) else 0})
    n_obs = len(obs_events)
    if n_obs == 0:
        return {"n_events": 0, "A": None, "p": None, "null95": None,
                "null_mean": None, "n_valid_draws": 0, "events": [],
                "pos_ref": pos_ref}
    A_obs = float(np.mean([e["within"] for e in obs_events]))

    tabs = [_shift_table(vals, pos, anchors, direction)
            for _, _, vals, pos, anchors in rn]
    lens = np.array([len(t[0]) for t in tabs])
    n_rn = len(tabs)
    maxT = int(lens.max())
    nev = np.zeros((n_rn, maxT))
    sind = np.zeros((n_rn, maxT))
    for i, (a, b) in enumerate(tabs):
        nev[i, :len(a)] = a
        sind[i, :len(b)] = b
    draws_idx = np.stack([rng.integers(0, lens[i], size=n_draws)
                          for i in range(n_rn)], axis=1)  # (draws, n_rn)
    rows = np.arange(n_rn)
    tot_n = nev[rows[None, :], draws_idx].sum(axis=1)
    tot_i = sind[rows[None, :], draws_idx].sum(axis=1)
    valid = tot_n > 0
    A_null = np.divide(tot_i, tot_n, out=np.full(n_draws, np.nan),
                       where=valid)[valid]
    return {"n_events": n_obs, "A": A_obs,
            "p": float(np.mean(A_null >= A_obs)),
            "null95": float(np.percentile(A_null, 95)),
            "null_mean": float(np.mean(A_null)),
            "n_valid_draws": int(valid.sum()), "events": obs_events,
            "pos_ref": pos_ref}


# --------------------------------------------------------------------------- #
# Leg D — transition coverage                                                 #
# --------------------------------------------------------------------------- #
def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(sstats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(sstats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def _down_events_near(win: dict, readers: list[str], b: float) -> bool:
    for r in readers:
        vals, pos = finite_series(win["rho"][r])
        for e in counted_crossings(vals):
            if e["dir"] == "down" \
                    and abs(float(pos[e["pos"]]) + CENTER_OFF - b) <= TOL:
                return True
    return False


def leg_D(win: dict, m: Measurement, signal_nights: list[str],
          null_nights: list[str]) -> dict:
    trans = []
    for night in signal_nights:
        for tr in strata_transitions(m, night):
            if tr["kind"] == "signal":
                trans.append((night, tr["boundary"]))
    covered = [int(_down_events_near(win[night], win[night]["readers"], b))
               for night, b in trans]
    k, n = int(sum(covered)), len(covered)
    D_signal = k / n if n else None
    ci = clopper_pearson(k, n) if n else (None, None)

    null_pseudo = []
    for night in null_nights:
        w = win[night]
        mid = w["T"] // 2
        null_pseudo.append({"night": night, "midpoint": mid,
                            "covered": int(_down_events_near(w, w["readers"],
                                                             mid))})
    D_null = float(np.mean([c["covered"] for c in null_pseudo])) \
        if null_pseudo else None

    def rn_rate(nights):
        tot, hit = 0, 0
        for night in nights:
            w = win[night]
            for r in w["readers"]:
                vals, _ = finite_series(w["rho"][r])
                ev = [e for e in counted_crossings(vals)
                      if e["dir"] == "down"]
                tot += 1
                hit += int(len(ev) > 0)
        return (hit / tot if tot else None), hit, tot

    sig_rate, sig_hit, sig_tot = rn_rate(signal_nights)
    null_rate, null_hit, null_tot = rn_rate(null_nights)
    return {"D_signal": D_signal, "k": k, "n_transitions": n,
            "ci_exact_binomial": list(ci), "covered": covered,
            "transitions": [{"night": a, "boundary": b} for a, b in trans],
            "D_null": D_null, "null_pseudo": null_pseudo,
            "signal_rn_crossing_rate": sig_rate,
            "null_rn_crossing_rate": null_rate,
            "signal_rn_hits": sig_hit, "signal_rn_total": sig_tot,
            "null_rn_hits": null_hit, "null_rn_total": null_tot}


# --------------------------------------------------------------------------- #
# Leg P — offset persistence through transitions                              #
# --------------------------------------------------------------------------- #
def leg_P(win: dict, m: Measurement, sd: float, W: int,
          signal_nights: list[str], class_residual: bool = False) -> dict:
    """Cosine between pre ([b-W, b-1]) and post ([b, b+W-1]) transition
    offset matrices over the roster, restricted to the reliable subspace;
    persistence-at-rest = the same cosine over the first adjacent window
    pair inside a single >=2W signal stratum. Pooled per wave by Fisher z.
    Events whose windows do not fit are dropped and listed."""
    sp = {n: m.nights[n].speaks for n in m.nights}

    def offset_vecs(night, t0, cres):
        w = win[night]
        M = {}
        for r in w["readers"]:
            if not w["present"][r][t0]:
                continue
            vs = [dict(m.readings[r][night])[sp[night][i]["seq"]]
                  for i in range(t0, t0 + W)]
            M[r] = np.mean(np.stack(vs), axis=0)
        if not M:
            return {}
        if cres:
            groups = {}
            for r in M:
                groups.setdefault(m.arch[r], []).append(r)
            keep = {r for r in M if len(groups[m.arch[r]]) >= 2}
            if len(keep) < 2:
                return {}
            gm = {a: np.mean([M[r] for r in rs if r in keep], axis=0)
                  for a, rs in groups.items() if any(r in keep for r in rs)}
            return {r: (M[r] - gm[m.arch[r]]) / sd for r in keep}
        bbar = np.mean(list(M.values()), axis=0)
        return {r: (M[r] - bbar) / sd for r in M}

    def cos_pair(Oa, Ob):
        common = sorted(set(Oa) & set(Ob))
        if len(common) < 2:
            return None, 0
        A = np.stack([Oa[r][RIDX] for r in common]).ravel()
        B = np.stack([Ob[r][RIDX] for r in common]).ravel()
        na, nb = np.linalg.norm(A), np.linalg.norm(B)
        if na < 1e-12 or nb < 1e-12:
            return 0.0, len(common)
        return float(A @ B / (na * nb)), len(common)

    trans_events, dropped = [], []
    for night in signal_nights:
        T = len(sp[night])
        for tr in strata_transitions(m, night):
            if tr["kind"] != "signal":
                continue
            b = tr["boundary"]
            if b - W < 0 or b + W > T:
                dropped.append({"night": night, "boundary": b,
                                "reason": "pre/post window does not fit"})
                continue
            cos, nr = cos_pair(offset_vecs(night, b - W, class_residual),
                               offset_vecs(night, b, class_residual))
            if cos is None:
                dropped.append({"night": night, "boundary": b,
                                "reason": f"<2 common readers ({nr})"})
                continue
            trans_events.append({"night": night, "boundary": b,
                                 "n_readers": nr, "cos": cos})

    rest_events = []
    for night in signal_nights:
        for label, lo, hi, kind in m.nights[night].strata:
            if kind != "signal" or (hi - lo + 1) < 2 * W:
                continue
            cos, nr = cos_pair(offset_vecs(night, lo, class_residual),
                               offset_vecs(night, lo + W, class_residual))
            if cos is None:
                continue
            rest_events.append({"night": night, "stratum": label,
                                "n_readers": nr, "cos": cos})

    def fisher_pool(evs):
        if not evs:
            return None
        zs = [math.atanh(min(max(e["cos"], -0.999), 0.999)) for e in evs]
        return float(math.tanh(float(np.mean(zs))))

    P_trans, P_rest = fisher_pool(trans_events), fisher_pool(rest_events)

    rng = np.random.default_rng(SEED + 1)

    def boot(evs):
        if not evs:
            return (None, None), []
        zs = np.array([math.atanh(min(max(e["cos"], -0.999), 0.999))
                       for e in evs])
        draws = [math.tanh(float(zs[rng.integers(0, len(zs), len(zs))].mean()))
                 for _ in range(B_BOOT)]
        return (float(np.percentile(draws, 2.5)),
                float(np.percentile(draws, 97.5))), draws

    (trans_lo, trans_hi), _ = boot(trans_events)
    (rest_lo, rest_hi), rest_draws = boot(rest_events)
    half = [0.5 * v for v in rest_draws] if rest_draws else []
    half_lo = float(np.percentile(half, 2.5)) if half else None
    half_hi = float(np.percentile(half, 97.5)) if half else None
    holds = (P_trans is not None and P_rest is not None
             and P_trans >= 0.5 * P_rest)
    mech_kill = bool(P_trans is not None and P_rest is not None
                     and P_trans < 0.5 * P_rest
                     and trans_hi is not None and half_lo is not None
                     and trans_hi < half_lo)
    return {"P_trans": P_trans, "P_rest": P_rest,
            "trans_ci": [trans_lo, trans_hi],
            "rest_ci": [rest_lo, rest_hi],
            "half_rest_ci": [half_lo, half_hi],
            "holds_at_half": bool(holds),
            "mechanism_kill": mech_kill,
            "trans_events": trans_events, "rest_events": rest_events,
            "dropped": dropped}


# --------------------------------------------------------------------------- #
# Leg S — exposure (per-night score ~ warmth x + reader FE)                   #
# --------------------------------------------------------------------------- #
def persona_warmth(m: Measurement) -> dict:
    """A-priori baseline warmth of each reader's persona: direction cosine
    of the z-standardized reliable-subspace vibe_start against the
    restricted renormalized vmf.WARM (no measured readings involved)."""
    out = {}
    for r in m.readers:
        for night in sorted(m.nights):
            if r in m.nights[night].params:
                z = SCALE * (np.asarray(m.nights[night].params[r]
                                        ["vibe_start"], float) - CENTER)
                zr = z[RIDX]
                n = float(np.linalg.norm(zr))
                out[r] = float(WARM_REL @ (zr / n)) if n > 1e-12 else 0.0
                break
    return out


def leg_S(win: dict, m: Measurement, signal_nights: list[str],
          x_map: dict, class_residual: bool = False, seed: int = SEED) -> dict:
    cells = []
    for night in signal_nights:
        w = win[night]
        for r in w["readers"]:
            vals = w["rho"][r]
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            cells.append({"reader": r, "night": night,
                          "score": float(np.median(vals)), "x": x_map[night]})
    roster = {night: sorted({c["reader"] for c in cells
                             if c["night"] == night})
              for night in signal_nights}
    pwarm = persona_warmth(m)
    archwarm = {night: float(np.mean([pwarm[r] for r in roster[night]]))
                for night in signal_nights}
    for c in cells:
        c["size"] = float(len(roster[c["night"]]))
        c["archwarm"] = archwarm[c["night"]]

    readers = sorted({c["reader"] for c in cells})
    by_reader = {r: [dict(c) for c in cells if c["reader"] == r]
                 for r in readers}

    def fit(cells_):
        rows = []
        for r in sorted({c["reader"] for c in cells_}):
            cs = [c for c in cells_ if c["reader"] == r]
            for key in ("score", "x", "size", "archwarm"):
                mu = float(np.mean([c[key] for c in cs]))
                for c in cs:
                    c["_" + key] = c[key] - mu
            rows.extend(cs)
        y = np.array([c["_score"] for c in rows])
        return (y,
                np.array([c["_x"] for c in rows]),
                np.array([c["_size"] for c in rows]),
                np.array([c["_archwarm"] for c in rows]))

    def ols_slope(y, x):
        vx = float(np.var(x, ddof=1))
        if vx <= 1e-15 or len(x) < 3:
            return None
        return float(np.cov(x, y, ddof=1)[0, 1] / vx)

    def rss(y, cols):
        A = np.column_stack(cols)
        if A.shape[0] <= A.shape[1]:
            return float(np.sum(y ** 2))
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        return float(np.sum((y - A @ beta) ** 2))

    y, X, sz, aw = fit(cells)
    slope = ols_slope(y, X)

    rng = np.random.default_rng(seed)
    slopes, eff = [], 0
    for _ in range(B_BOOT):
        take = [readers[i]
                for i in rng.integers(0, len(readers), len(readers))]
        bc = [dict(c) for r in take for c in by_reader[r]]
        if len({c["x"] for c in bc}) < 2:
            continue
        s = ols_slope(*fit(bc)[:2])
        if s is not None:
            slopes.append(s)
            eff += 1
    ci = (float(np.percentile(slopes, 2.5)),
          float(np.percentile(slopes, 97.5))) if slopes else (None, None)

    delta_obs = rss(y, [sz, aw]) - rss(y, [sz, aw, X])
    nights = sorted(signal_nights)
    xs = np.array([x_map[n] for n in nights])
    rng2 = np.random.default_rng(seed + 1)
    deltas = []
    for _ in range(N_PERM):
        xp = xs[rng2.permutation(len(xs))]
        cmap = {n: float(xp[i]) for i, n in enumerate(nights)}
        bc = [dict(c) for c in cells]
        for c in bc:
            c["x"] = cmap[c["night"]]
        yp, Xp, szp, awp = fit(bc)
        deltas.append(rss(yp, [szp, awp]) - rss(yp, [szp, awp, Xp]))
    p_nested = float(np.mean(np.array(deltas) >= delta_obs))

    contains0 = ci[0] is not None and ci[0] <= 0 <= ci[1]
    beats = bool(delta_obs > 0 and p_nested < 0.05)
    return {"cells": [{"reader": c["reader"], "night": c["night"],
                       "score": c["score"], "x": c["x"]} for c in cells],
            "n_cells": len(cells), "n_readers": len(readers),
            "slope_x": slope, "slope_ci": list(ci),
            "contains_0": bool(contains0),
            "delta_rss_x_beyond_competitor": delta_obs,
            "nested_perm_p": p_nested,
            "beats_competitor": beats,
            "x_invariant": bool(contains0 and not beats),
            "competitor_covariates": {
                "roster_size": {n: len(roster[n]) for n in nights},
                "archwarm": archwarm},
            "effective_draws": eff}


# --------------------------------------------------------------------------- #
# Continuity ladder (design sec.3, mandatory before any leg)                  #
# --------------------------------------------------------------------------- #
def ladder_rung1(m: Measurement, sd: float, signal_nights: list[str],
                 channel: str) -> dict:
    """W = full night, halves = registered strata split, pooled over signal
    reader-nights; exact anchor = the filed E-cont global arithmetic."""
    os_, ds_ = [], []
    for night in signal_nights:
        n = m.nights[night]
        per = {}
        for r in m.readers:
            if night in m.readings[r]:
                per[r] = np.mean([v for _, v in m.readings[r][night]], axis=0)
        if len(per) < 2:
            continue
        bbar = np.mean(list(per.values()), axis=0)
        strata = n.strata
        for r, base in per.items():
            os_.append(float(np.sqrt(np.mean((base - bbar) ** 2))) / sd)
            sv = m.readings[r][night]
            ds = []
            for (l0, lo0, hi0, k0), (l1, lo1, hi1, k1) in zip(strata,
                                                              strata[1:]):
                a = np.array([v for sq, v in sv if lo0 <= sq <= hi0])
                b = np.array([v for sq, v in sv if lo1 <= sq <= hi1])
                if len(a) and len(b):
                    ds.append(float(np.linalg.norm(b.mean(0) - a.mean(0))) / sd)
            if ds:
                ds_.append(float(np.mean(ds)))
    from scripts.e5_identity_propagation import cont_baselines, cont_spread
    gspread = cont_spread(cont_baselines(m), sd)
    gdrift = m.drift_mean()
    return {"channel": channel,
            "R_pooled_fullnight": float(np.sqrt(np.mean(np.square(os_)))
                                        / np.mean(ds_)),
            "pooled_spread": float(np.sqrt(np.mean(np.square(os_)))),
            "pooled_drift": float(np.mean(ds_)),
            "exact_anchor_global_spread": gspread,
            "exact_anchor_global_drift": gdrift,
            "exact_anchor_ratio": gspread / gdrift}


def ladder_rung2() -> dict:
    """v:1 anchor corpus (A,B,C,D,D-cold; 7 harvested real readers; +13
    synthetic-grounded, seed 0): the same full-night pooled o/d machinery
    must reproduce 0.5599 (real-only) / 0.4898 (grounded) within +-0.10."""
    import scripts.premise_measurement as pm
    rosters = {}
    for night, fn in pm.NIGHT_FILES.items():
        roster, speaks = pm.load_night(os.path.join(pm.NIGHTS_DIR, fn))
        rosters[night] = roster
        pm.ALL_SPEAKS[night] = speaks
        pm.STRATA[night] = pm._strata_for(night, speaks)
    sd_scalar, _ = pm.corpus_sd(list(pm.ALL_SPEAKS.values()))
    real = {}
    for night, roster in rosters.items():
        for name, entry in roster.items():
            if name not in real:
                real[name] = {"params": dict(entry), "nights": {}}
            real[name]["nights"][night] = name
    synth = pm.synthesize(real)
    merged = dict(real)
    merged.update(synth)

    def pooled_R(readers_dict):
        fitted = pm.fit_readers(readers_dict)
        by_night = {}
        for name, f in fitted.items():
            for night, readings in f["readings"].items():
                if readings:
                    by_night.setdefault(night, {})[name] = \
                        np.mean([v for _, v in readings], axis=0)
        os_ = []
        for night, bases in by_night.items():
            if len(bases) < 2:
                continue
            bbar = np.mean(list(bases.values()), axis=0)
            for b in bases.values():
                os_.append(float(np.sqrt(np.mean((b - bbar) ** 2)))
                           / sd_scalar)
        ds_ = []
        for name, f in fitted.items():
            for night in f["readings"]:
                strata = list(pm.STRATA[night])
                ds = []
                for s0, s1 in zip(strata, strata[1:]):
                    if (night, s0) in f["by_stratum"] \
                            and (night, s1) in f["by_stratum"]:
                        a = np.mean(f["by_stratum"][(night, s0)], axis=0)
                        b = np.mean(f["by_stratum"][(night, s1)], axis=0)
                        ds.append(float(np.linalg.norm(b - a)) / sd_scalar)
                if ds:
                    ds_.append(float(np.mean(ds)))
        return (float(np.sqrt(np.mean(np.square(os_))) / np.mean(ds_)),
                float(np.mean(os_)), float(np.mean(ds_)), len(os_))

    R_real = pooled_R(real)
    R_grounded = pooled_R(merged)
    m_real = pm.measure(pm.fit_readers(real), sd_scalar)
    m_all = pm.measure(pm.fit_readers(merged), sd_scalar)
    assert abs(m_real["ratio"] - 0.5599) < 5e-3, "premise real ratio drifted"
    assert abs(m_all["ratio"] - 0.4898) < 5e-3, "premise grounded drifted"
    return {"R_pooled_real": R_real[0], "R_pooled_grounded": R_grounded[0],
            "pooled_real_o_mean": R_real[1], "pooled_real_d_mean": R_real[2],
            "n_real_cells": R_real[3],
            "exact_premise_real_ratio": m_real["ratio"],
            "exact_premise_grounded_ratio": m_all["ratio"],
            "target_real": 0.5599, "target_grounded": 0.4898}


# --------------------------------------------------------------------------- #
# Wave assembly                                                               #
# --------------------------------------------------------------------------- #
def drift_by_phase(win: dict, m: Measurement, signal_nights: list[str]) -> dict:
    near, far = [], []
    for night in signal_nights:
        w = win[night]
        bounds = [t["boundary"] for t in strata_transitions(m, night)
                  if t["kind"] == "signal"]
        for r in w["readers"]:
            d = w["d"][r]
            for t in range(len(d)):
                if np.isnan(d[t]):
                    continue
                if any(abs(t - b) < W_PRIMARY for b in bounds):
                    near.append(float(d[t]))
                else:
                    far.append(float(d[t]))
    return {"d_transition_phase_mean": float(np.mean(near)) if near else None,
            "d_stable_phase_mean": float(np.mean(far)) if far else None,
            "n_transition_phase": len(near), "n_stable_phase": len(far)}


def analyze_wave(label, nights, signal_nights, null_nights, x_map,
                 nights_map, W=W_PRIMARY, presence="canonical",
                 class_residual=False, seed=SEED):
    sd, _ = corpus_sd([Night(n) for n in nights])
    m = Measurement(nights_map, sd, include_nights=nights, presence=presence)
    win = {n: night_windows(m, n, sd, W, class_residual) for n in nights}
    res = {"label": label, "W": W, "presence": presence,
           "class_residual": class_residual, "corpus_sd": sd,
           "n_readers": len(m.readers)}
    A = leg_A(win, m, signal_nights, seed)
    A_up = leg_A(win, m, signal_nights, seed + 2, direction="up",
                 mid_anchor=True)
    A_start = leg_A(win, m, signal_nights, seed + 4, pos_ref="start")
    res["A"] = {k: v for k, v in A.items() if k != "events"}
    res["A_up_mirror"] = {k: v for k, v in A_up.items() if k != "events"}
    res["A_start_ref_sensitivity"] = {k: v for k, v in A_start.items()
                                       if k != "events"}
    res["D"] = leg_D(win, m, signal_nights, null_nights)
    res["P"] = leg_P(win, m, sd, W, signal_nights, class_residual)
    res["S"] = leg_S(win, m, signal_nights, x_map, class_residual, seed)
    traj = {}
    for n in signal_nights:
        w = win[n]
        traj[n] = {"Rt": [None if not np.isfinite(v) else round(float(v), 4)
                          for v in w["Rt"]],
                   "boundaries": [t["boundary"] for t in
                                  strata_transitions(m, n)
                                  if t["kind"] == "signal"], "x": x_map[n]}
    res["trajectory"] = traj
    res["null_night_scores"] = {
        n: {r: (None if np.all(np.isnan(win[n]["rho"][r])) else
                round(float(np.nanmedian(win[n]["rho"][r])), 4))
            for r in win[n]["readers"]} for n in null_nights}
    res["mean_d_by_phase"] = drift_by_phase(win, m, signal_nights)
    return res, m, win


# --------------------------------------------------------------------------- #
# Guards                                                                      #
# --------------------------------------------------------------------------- #
def run_guards() -> dict:
    print("=" * 78)
    print("[0] GUARDS (Stage-2 wave gate invoked verbatim + wave-1 re-asserts)")
    print("=" * 78)
    gate_path = os.path.join(ROOT, "data", "slope", "stage2-wave-gate.json")
    before = open(gate_path, "rb").read() if os.path.exists(gate_path) \
        else None
    proc = subprocess.run([sys.executable,
                           os.path.join(ROOT, "scripts",
                                        "stage2_wave_gate.py")],
                          capture_output=True, text=True, cwd=ROOT,
                          timeout=900)
    assert proc.returncode == 0, \
        f"wave gate crashed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    after = open(gate_path, "rb").read()
    gate = json.loads(after)
    assert gate["all_pass"], "Stage-2 wave gate FAILED -> VOID sec.5.1"
    print(f"  stage2_wave_gate: ALL CHECKS PASS (byte-identical re-run: "
          f"{before == after})")

    nights1 = {n: Night(n) for n in PRIMARY_NIGHTS}
    sd1, _ = corpus_sd(list(nights1.values()))
    m1 = Measurement(field_readers(), sd1, presence="canonical")
    from scripts.e5_identity_propagation import cont_baselines, cont_spread
    spread1 = cont_spread(cont_baselines(m1), sd1)
    drift1 = m1.drift_mean()
    print(f"  wave-1 guards: corpus_sd={sd1:.4f} (0.2367)  E-cont spread="
          f"{spread1:.4f} (0.4556)  drift={drift1:.4f} (0.7483)")
    assert abs(sd1 - 0.2367) < 1e-3 and abs(spread1 - 0.4556) < 1e-3 \
        and abs(drift1 - 0.7483) < 1e-3, "wave-1 guards drifted -> VOID"
    for n in PRIMARY_NIGHTS:
        assert abs(room_warmth(nights1[n]) - X_W1[n]) < 5e-5, \
            f"wave-1 warmth ladder drift {n}"
    print("  wave-1 warmth ladder reproduced to 4 decimals (9/9)")

    for tag, rdr, cold in [("T1", "poet", False), ("T4a", "drifter", True),
                           ("T2", "engineer", False), ("T9", "essayist", False),
                           ("S4a", "drifter", True), ("S5", "weaver", False)]:
        assert_replay_matches_log(Night(tag), rdr, cold=cold)
    print("  assert_replay_matches_log: 6/6 sampled reader-nights exact")
    return {"gate_all_pass": True, "gate_byte_identical": before == after,
            "wave1": {"corpus_sd": sd1, "spread": spread1, "drift": drift1},
            "gate_guards_w2": gate.get("guards_w2")}


# --------------------------------------------------------------------------- #
# Verdict (registered kill/void rules, design sec.5)                          #
# --------------------------------------------------------------------------- #
def verdict_for(primary: dict) -> dict:
    v = {"void": False, "void_reasons": [], "branches": {}}
    A, Dd, P, Sx = primary["A"], primary["D"], primary["P"], primary["S"]

    if Dd["null_rn_crossing_rate"] is not None \
            and Dd["signal_rn_crossing_rate"] is not None \
            and Dd["null_rn_crossing_rate"] >= \
            0.5 * Dd["signal_rn_crossing_rate"]:
        v["void"] = True
        v["void_reasons"].append(
            "sec.5.2 null-night crossing rate "
            f"{Dd['null_rn_crossing_rate']:.3f} >= 50% of signal rate "
            f"{Dd['signal_rn_crossing_rate']:.3f}")
    if A["n_events"] < 20:
        v["void"] = True
        v["void_reasons"].append(
            f"sec.5.3 only {A['n_events']} counted down-crossings (< 20)")
    if Sx.get("effective_draws", B_BOOT) < 1500:
        v["void"] = True
        v["void_reasons"].append("sec.5.5 bootstrap effective draws "
                                 f"{Sx.get('effective_draws')} < 1500")

    a_fires = A["A"] is not None and (A["A"] > A["null95"]
                                      or A["p"] <= 0.05)
    lo, hi = Dd["ci_exact_binomial"]
    # D_signal - D_null exact-binomial difference CI: [lo - D_null,
    # hi - D_null]; contains 0 iff lo <= D_null <= hi
    d_diff_contains0 = bool(Dd["D_null"] is not None and lo <= Dd["D_null"]
                            <= hi)
    d_le_half = bool(Dd["D_signal"] is not None and Dd["D_signal"] <= 0.50)
    d_fails = d_le_half or d_diff_contains0
    v["branches"] = {
        "A_fires": bool(a_fires),
        "A_value": A["A"], "A_p": A["p"], "A_null95": A["null95"],
        "D_signal": Dd["D_signal"], "D_null": Dd["D_null"],
        "D_signal_le_50pct": d_le_half,
        "D_minus_D_null_ci_contains_0": d_diff_contains0,
        "D_fails": d_fails,
        "P_holds_at_half": bool(P["holds_at_half"]),
        "P_mechanism_kill": bool(P["mechanism_kill"]),
        "P_trans": P["P_trans"], "P_rest": P["P_rest"],
        "S_x_coef_ci_excludes_0": bool(not Sx["contains_0"]),
        "S_beats_competitor": bool(Sx["beats_competitor"]),
        "S_slope": Sx["slope_x"], "S_slope_ci": Sx["slope_ci"],
    }
    if v["void"]:
        v["verdict"] = "VOID BY RULE — " + "; ".join(v["void_reasons"])
        return v
    hbm_killed = (not a_fires) and d_fails
    alignment_falsified = (not Sx["contains_0"]) and Sx["beats_competitor"]
    pro_premise = a_fires and P["holds_at_half"] and Sx["x_invariant"]
    if hbm_killed:
        v["verdict"] = ("KILL (H-BM): A <= circular-shift null 95th pct AND "
                        "D fails — the ratio's movement is noise; static "
                        "in-band verdict stands as phase-averaged noise; "
                        "retirement confirmed, sharpened")
    elif P["mechanism_kill"]:
        v["verdict"] = ("PREMISE KILLED TEMPORALLY: P_trans < 0.5 x P_rest "
                        "with gap CI excluding overlap — steps erase "
                        "idiosyncrasy; readers are not instruments through "
                        "steps (nurse-as-index stable-phase-only)")
    elif alignment_falsified:
        v["verdict"] = ("ALIGNMENT ARM FALSIFIED (collapse): S x-coef CI "
                        "excludes 0 and beats the roster competitor — the "
                        "idiosyncrasy was room geometry wearing a reader's "
                        "name; premise dies with an explanation")
    elif pro_premise:
        v["verdict"] = ("SURVIVED (capped): A fires, P holds, S x-invariant — "
                        "readers are instruments except at steps; the "
                        "in-band ratio was an average over phases "
                        "(phase-conditional claim; Branch A/B stay closed)")
    else:
        v["verdict"] = ("INDETERMINATE: no registered branch fires cleanly "
                        "(two-branch discipline; see branch flags)")
    return v


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def prune(res: dict, keep_desc: bool = False) -> dict:
    out = {k: v for k, v in res.items()}
    if not keep_desc:
        out.pop("trajectory", None)
        out.pop("null_night_scores", None)
    return out


def _f4(x):
    return "nan" if x is None else f"{x:.4f}"


def _fs(x):
    return "nan" if x is None else f"{x:.3f}"


def print_wave(res: dict):
    A, Dd, P, Sx = res["A"], res["D"], res["P"], res["S"]
    f4, fs = _f4, _fs
    print(f"  A: {A['n_events']} counted down-crossings; A = {f4(A['A'])} "
          f"(pos_ref=center, deviation note 1); circular-shift null p = "
          f"{f4(A['p'])}, null95 = {f4(A['null95'])} "
          f"(null mean {f4(A['null_mean'])}, valid draws "
          f"{A['n_valid_draws']}); start-referent sensitivity A = "
          f"{f4(res['A_start_ref_sensitivity']['A'])} "
          f"(p={f4(res['A_start_ref_sensitivity']['p'])})")
    up = res["A_up_mirror"]
    print(f"     up-crossing mirror (secondary): {up['n_events']} events, "
          f"A_up = {f4(up['A'])} (p = {f4(up['p'])}, null95 "
          f"{f4(up['null95'])})")
    print(f"  D: signal {Dd['k']}/{Dd['n_transitions']} = "
          f"{f4(Dd['D_signal'])} (exact CI "
          f"[{f4(Dd['ci_exact_binomial'][0])}, "
          f"{f4(Dd['ci_exact_binomial'][1])}]); null-night pseudo "
          f"{f4(Dd['D_null'])}; rn crossing rates: signal "
          f"{f4(Dd['signal_rn_crossing_rate'])} "
          f"({Dd['signal_rn_hits']}/{Dd['signal_rn_total']}) vs null "
          f"{f4(Dd['null_rn_crossing_rate'])} "
          f"({Dd['null_rn_hits']}/{Dd['null_rn_total']})")
    print(f"  P: P_trans = {f4(P['P_trans'])} CI "
          f"[{f4(P['trans_ci'][0])}, {f4(P['trans_ci'][1])}] over "
          f"{len(P['trans_events'])} transitions "
          f"({len(P['dropped'])} dropped: "
          f"{[d['night'] + '@' + str(d['boundary']) for d in P['dropped']]});"
          f" P_rest = {f4(P['P_rest'])} over {len(P['rest_events'])} refs "
          f"{[e['night'] + ':' + e['stratum'] for e in P['rest_events']]}; "
          f"half-rest = {f4(None if P['P_rest'] is None else 0.5 * P['P_rest'])}; "
          f"holds: {P['holds_at_half']}")
    print(f"  S: {Sx['n_cells']} cells / {Sx['n_readers']} readers; "
          f"slope_x = {f4(Sx['slope_x'])} CI "
          f"[{f4(Sx['slope_ci'][0])}, {f4(Sx['slope_ci'][1])}] "
          f"(contains 0: {Sx['contains_0']}); deltaRSS(x|competitor) = "
          f"{Sx['delta_rss_x_beyond_competitor']:.5f}, nested perm p = "
          f"{Sx['nested_perm_p']:.4f} (beats competitor: "
          f"{Sx['beats_competitor']}; x-invariant: {Sx['x_invariant']})")
    ph = res["mean_d_by_phase"]
    print(f"  denominator check: d transition-phase "
          f"{f4(ph['d_transition_phase_mean'])} vs stable-phase "
          f"{f4(ph['d_stable_phase_mean'])} "
          f"({ph['n_transition_phase']}/{ph['n_stable_phase']} windows)")


def main():
    f4, fs = _f4, _fs
    guards = run_guards()
    gate = json.load(open(os.path.join(ROOT, "data", "slope",
                                       "stage2-wave-gate.json"),
                          encoding="utf-8"))
    w2_channel = gate["guards_w2"]["cont_spread"] / gate["guards_w2"]["drift"]

    print("\n" + "=" * 78)
    print("[1] CONTINUITY LADDER (estimator gate; VOID if any rung off > 0.10)")
    print("=" * 78)
    sd2, _ = corpus_sd([Night(n) for n in W2_NIGHT_LIST])
    m2g = Measurement(field_readers(FIELD_NIGHTS_W2), sd2,
                      include_nights=W2_NIGHT_LIST, presence="canonical")
    r1_w2 = ladder_rung1(m2g, sd2, SIGNAL_W2, "wave-2")
    sd1 = guards["wave1"]["corpus_sd"]
    m1g = Measurement(field_readers(), sd1, presence="canonical")
    r1_w1 = ladder_rung1(m1g, sd1, SIGNAL_W1, "wave-1")
    r2 = ladder_rung2()
    w1_channel, w1_seg_channel = 0.6088, 0.6853   # filed E2-at-power primary / E-seg
    ladder = {
        "wave2": {**r1_w2, "filed_channel": w2_channel,
                  "ok": abs(r1_w2["R_pooled_fullnight"] - w2_channel) <= 0.10,
                  "exact_ok": abs(r1_w2["exact_anchor_ratio"] - w2_channel)
                  <= 0.10},
        "wave1": {**r1_w1, "filed_channel": w1_channel,
                  "filed_seg_channel": w1_seg_channel,
                  "ok": abs(r1_w1["R_pooled_fullnight"] - w1_channel) <= 0.10,
                  "exact_ok": abs(r1_w1["exact_anchor_ratio"] - w1_channel)
                  <= 0.10},
        "anchor_v1": {**r2,
                      "ok_real": abs(r2["R_pooled_real"] - 0.5599) <= 0.10,
                      "ok_grounded": abs(r2["R_pooled_grounded"] - 0.4898)
                      <= 0.10},
    }
    print(f"  wave-2: R_full(pooled) = {r1_w2['R_pooled_fullnight']:.4f} vs "
          f"filed channel {w2_channel:.4f} (gate spread "
          f"{gate['guards_w2']['cont_spread']:.4f} / drift "
          f"{gate['guards_w2']['drift']:.4f}) -> "
          f"{'OK' if ladder['wave2']['ok'] else 'OFF > 0.10'}; exact global "
          f"anchor {r1_w2['exact_anchor_ratio']:.4f} -> "
          f"{'OK' if ladder['wave2']['exact_ok'] else 'OFF'}")
    print(f"  wave-1: R_full(pooled) = {r1_w1['R_pooled_fullnight']:.4f} vs "
          f"filed channel {w1_channel:.4f} (E-seg {w1_seg_channel:.4f}) -> "
          f"{'OK' if ladder['wave1']['ok'] else 'OFF > 0.10'}; exact global "
          f"anchor {r1_w1['exact_anchor_ratio']:.4f} -> "
          f"{'OK' if ladder['wave1']['exact_ok'] else 'OFF'}")
    print(f"  v:1 anchor: R(real) = {r2['R_pooled_real']:.4f} vs 0.5599 -> "
          f"{'OK' if ladder['anchor_v1']['ok_real'] else 'OFF'}; "
          f"R(grounded) = {r2['R_pooled_grounded']:.4f} vs 0.4898 -> "
          f"{'OK' if ladder['anchor_v1']['ok_grounded'] else 'OFF'} "
          f"(premise exact reproduction: "
          f"{r2['exact_premise_real_ratio']:.4f} / "
          f"{r2['exact_premise_grounded_ratio']:.4f})")
    ladder_all_ok = (ladder["wave2"]["ok"] and ladder["wave1"]["ok"]
                     and ladder["anchor_v1"]["ok_real"]
                     and ladder["anchor_v1"]["ok_grounded"])
    if not ladder_all_ok:
        print("  !! LADDER OFF > 0.10 on >=1 rung -> VOID by rule sec.5.4")

    print("\n" + "=" * 78)
    print("[2] PRIMARY — wave-2 T-nights (never pooled with wave-1)")
    print("=" * 78)
    primary, m2, win2 = analyze_wave("wave-2-primary", W2_NIGHT_LIST,
                                     SIGNAL_W2, NULL_W2, X_W2,
                                     field_readers(FIELD_NIGHTS_W2))
    print_wave(primary)

    print("\n" + "=" * 78)
    print("[3] REPLICATION — wave-1 (labeled; never pooled)")
    print("=" * 78)
    repl, m1, win1 = analyze_wave("wave-1-replication", PRIMARY_NIGHTS,
                                  SIGNAL_W1, NULL_W1, X_W1, field_readers())
    print_wave(repl)

    print("\n" + "=" * 78)
    print("[4] SENSITIVITIES (labeled; the registered primaries stand)")
    print("=" * 78)
    sens = {}
    for Wv in (8, 16):
        s2, _, _ = analyze_wave(f"wave-2 W={Wv}", W2_NIGHT_LIST, SIGNAL_W2,
                                NULL_W2, X_W2,
                                field_readers(FIELD_NIGHTS_W2), W=Wv,
                                seed=SEED + 10 + Wv)
        s1, _, _ = analyze_wave(f"wave-1 W={Wv}", PRIMARY_NIGHTS, SIGNAL_W1,
                                NULL_W1, X_W1, field_readers(), W=Wv,
                                seed=SEED + 20 + Wv)
        sens[f"W{Wv}"] = {"wave2": prune(s2), "wave1": prune(s1)}
        print(f"  W={Wv}: wave-2 A={f4(s2['A']['A'])} "
              f"(p={f4(s2['A']['p'])}, n={s2['A']['n_events']}), "
              f"D={fs(s2['D']['D_signal'])}, P_trans="
              f"{f4(s2['P']['P_trans'])} vs rest {f4(s2['P']['P_rest'])}; "
              f"wave-1 A={f4(s1['A']['A'])} "
              f"(p={f4(s1['A']['p'])}, n={s1['A']['n_events']}), "
              f"D={fs(s1['D']['D_signal'])}, P_trans="
              f"{f4(s1['P']['P_trans'])} vs rest {f4(s1['P']['P_rest'])}")
    s2a, _, _ = analyze_wave("wave-2 actual", W2_NIGHT_LIST, SIGNAL_W2,
                             NULL_W2, X_W2, field_readers(FIELD_NIGHTS_W2),
                             presence="actual", seed=SEED + 40)
    s1a, _, _ = analyze_wave("wave-1 actual", PRIMARY_NIGHTS, SIGNAL_W1,
                             NULL_W1, X_W1, field_readers(),
                             presence="actual", seed=SEED + 50)
    sens["actual_presence"] = {"wave2": prune(s2a), "wave1": prune(s1a)}
    print(f"  actual-presence: wave-2 A={f4(s2a['A']['A'])} "
          f"(p={f4(s2a['A']['p'])}, n={s2a['A']['n_events']}), "
          f"D={fs(s2a['D']['D_signal'])}; wave-1 A={f4(s1a['A']['A'])} "
          f"(p={f4(s1a['A']['p'])}, n={s1a['A']['n_events']}), "
          f"D={fs(s1a['D']['D_signal'])}")
    c2, _, _ = analyze_wave("wave-2 cres", W2_NIGHT_LIST, SIGNAL_W2, NULL_W2,
                            X_W2, field_readers(FIELD_NIGHTS_W2),
                            class_residual=True, seed=SEED + 60)
    c1, _, _ = analyze_wave("wave-1 cres", PRIMARY_NIGHTS, SIGNAL_W1,
                            NULL_W1, X_W1, field_readers(),
                            class_residual=True, seed=SEED + 70)
    sens["class_residual"] = {"wave2_P": prune(c2)["P"],
                              "wave2_S": prune(c2)["S"],
                              "wave1_P": prune(c1)["P"],
                              "wave1_S": prune(c1)["S"]}
    print(f"  class-residual P: wave-2 {f4(c2['P']['P_trans'])} "
          f"(rest {f4(c2['P']['P_rest'])}); wave-1 {f4(c1['P']['P_trans'])} "
          f"(rest {f4(c1['P']['P_rest'])})")
    print(f"  class-residual S: wave-2 slope {f4(c2['S']['slope_x'])} CI "
          f"[{f4(c2['S']['slope_ci'][0])}, {f4(c2['S']['slope_ci'][1])}]; "
          f"wave-1 {f4(c1['S']['slope_x'])} CI "
          f"[{f4(c1['S']['slope_ci'][0])}, {f4(c1['S']['slope_ci'][1])}]")

    print("\n" + "=" * 78)
    print("[5] VERDICT (registered kill/void rules, design sec.5)")
    print("=" * 78)
    verd = verdict_for(primary)
    if not ladder_all_ok:
        verd = {"void": True,
                "void_reasons": ["sec.5.4 continuity ladder off > 0.10"],
                "verdict": "VOID BY RULE — continuity ladder off > 0.10",
                "branches": verd.get("branches", {})}
    print(f"  {verd['verdict']}")
    for k in ("A_fires", "D_signal_le_50pct", "D_minus_D_null_ci_contains_0",
              "D_fails", "P_holds_at_half", "P_mechanism_kill",
              "S_x_coef_ci_excludes_0", "S_beats_competitor"):
        if k in verd.get("branches", {}):
            print(f"    {k}: {verd['branches'][k]}")
    # identity-propagation booking (E5 sensitivity)
    pop_holds = primary["P"]["holds_at_half"]
    cres = sens["class_residual"]["wave2_P"]
    id_prop = bool(pop_holds and cres["P_trans"] is not None
                   and cres["P_rest"] is not None
                   and cres["P_trans"] < 0.5 * cres["P_rest"])
    print(f"    identity-propagation booking (population P holds, residual "
          f"P fails): {id_prop}")

    def beta_upd(p, fired):
        a, b = p * 20, (1 - p) * 20
        a, b = a + (1 if fired else 0), b + (0 if fired else 1)
        return round(a / (a + b), 4)

    bf = verd.get("branches", {})
    priors_updated = {
        "A_fires": {"prior": 0.55,
                    "posterior": beta_upd(0.55, bool(bf.get("A_fires"))),
                    "observed": {"A": primary["A"]["A"],
                                 "p": primary["A"]["p"],
                                 "null95": primary["A"]["null95"]}},
        "P_holds": {"prior": 0.50,
                    "posterior": beta_upd(0.50, bool(bf.get("P_holds_at_half"))),
                    "observed": {"P_trans": primary["P"]["P_trans"],
                                 "P_rest": primary["P"]["P_rest"]}},
        "x_invariance": {"prior": 0.60,
                         "posterior": beta_upd(
                             0.60, not bf.get("S_x_coef_ci_excludes_0", False)),
                         "observed": {"slope": primary["S"]["slope_x"],
                                      "ci": primary["S"]["slope_ci"]}},
        "note": "Beta bookkeeping at the registered prior strength "
                "(concentration 20); one run moves priors modestly — the "
                "leg values carry the information."}
    print(f"\n  priors: A 0.55 -> {priors_updated['A_fires']['posterior']}; "
          f"P 0.50 -> {priors_updated['P_holds']['posterior']}; "
          f"x-invariance 0.60 -> "
          f"{priors_updated['x_invariance']['posterior']}")

    results = {
        "date": "2026-08-21",
        "test": "E2/E3 premise band-movers (registered design "
                "E2E3-premise-band-movers-design-2026-08-21.md; topic.md "
                "addendum 2026-08-21)",
        "channel": "canonical presence (primary); actual = sensitivity",
        "guards": guards,
        "ladder": ladder, "ladder_all_ok": ladder_all_ok,
        "primary_wave2": prune(primary, keep_desc=True),
        "replication_wave1": prune(repl, keep_desc=True),
        "sensitivities": sens,
        "identity_propagation_booking": id_prop,
        "verdict": verd,
        "priors_updated": priors_updated,
        "seeds": {"main": SEED, "shift_null": SEED, "boot_P": SEED + 1,
                  "boot_S": SEED, "nested_perm": SEED + 1,
                  "variants": "SEED+10..90 (all 20260821-family)"},
    }
    os.makedirs(os.path.dirname(OUT_SLOPE), exist_ok=True)
    with open(OUT_SLOPE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    with open(os.path.join(OUT_DIR, "wave2.json"), "w", encoding="utf-8") as f:
        json.dump(prune(primary, keep_desc=True), f, indent=1)
    with open(os.path.join(OUT_DIR, "wave1.json"), "w", encoding="utf-8") as f:
        json.dump(prune(repl, keep_desc=True), f, indent=1)
    print(f"\n[premise-band-movers] results -> {OUT_SLOPE}")
    print(f"[premise-band-movers] per-wave -> {OUT_DIR}/"
          "{results,wave2,wave1}.json")


if __name__ == "__main__":
    main()
