"""Cross-strata transfer test — session grain -> memory/identity grain.

Open question #4 (research/topic.md v3): the August clause that ran was
cross-condition (the admitted sixth laundering); THIS is the true
cross-strata version, now runnable on E2's v:2 per-reader schema
(night-S1..S7, reader-grain logging).

THE TWO GRAINS, DEFINED HONESTLY FROM THE CORPUS
================================================

Session grain (fast, within-night): per reader R, the registered E2 drift
estimator (scripts/e2_instrument.py, reused unmodified) on logged v:2
readings — d_R = mean over signal strata transitions of
||mean(next cell) - mean(prev cell)|| / corpus_sd, where cells are the
REGISTERED strata segments (NIGHT_SPECS) and readings are the logged
field_eff_to_reader facts passed through R's lens. The S5 no-flip split
supplies the per-reader null drift (the primary reading of the instrument:
"this reader has NOT drifted from herself").

Memory/identity grain (slow, across-night): the corpus has NO cross-night
state carryover — each night's session resets every lens to vibe_start
(verified: roster vibe == vibe_start in every session_open; lens_now at
seq 0 is one acclimation step past vibe_start). The only across-night,
per-reader objects the corpus carries are:

  identity      b_R  = mean over R's nights of M_R(k)   (stable baseline;
                     the E2 ICC object's center)
  memory-of-room M_R(k) = R's per-night summary reading = componentwise
                     median of R's readings on night k. This is EXACTLY the
                     logged session_close reader_final fact (verified
                     numerically), i.e. the schema doc's "greppable
                     per-reader baseline fact". It is the closest honest
                     proxy for "memory of that room" the corpus supports,
                     and it is what the task statement names.

Memory-grain behavior quantities (per reader):
  plasticity   p_R = mean_k ||M_R(k) - b_R|| / corpus_sd
                (how far R's night-memories sit from R's identity — the
                between-night, slow-grain movement)
  room-tracking t_R = mean_k cos(M_R(k) - b_R, mu_k - mu_bar)
                (alignment of R's memory displacement with the room's own
                displacement mu_k - mu_bar, mu_k = night k's final room
                mu_hat; separates room-driven memory from idiosyncratic)

WHAT "TRANSFER" MEANS HERE
==========================
Primary transfer coefficient rho = corr_across_readers(d_R, p_R).
Transfer (rho > 0): readers whose readings move more WITHIN sessions also
carry night-memories further FROM their stable identity — fast-grain
movement is not absorbed; it propagates into slow-grain memory structure.
Null (rho = 0): the strata decouple — within-session movement is mean-
reverted (or reader-independent) and does NOT predict memory/identity
organization; a reader can be a big session mover and a stable-identity
keeper, and vice versa. Negative rho: compensatory organization (big
within-session movers form MORE identity-locked memories) — booked as a
finding either way. What counts as a null: bootstrap CI covering 0.

CAVEATS (pre-stated)
====================
1. Shared observable: both grains derive from the same field_eff_to_reader
   series — but p_R is BETWEEN-night structure of per-night MEDIANS while
   d_R is WITHIN-night structure of segment MEANS; coupling requires the
   within-night movement to survive median-averaging and night aggregation.
   That survival is the substantive claim, not a mechanical identity.
2. Mechanism-amplitude confound: lens gain, charisma, and vibe extremity
   scale displacement at BOTH grains. Reported honestly via Frisch-Waugh
   partial correlations (residualizing both variables on the knobs).
   NOTE the tension: these knobs are themselves identity parameters, so
   the partial is a mechanism probe, not a cleaner estimate of transfer.
3. Room-schedule spread is common to co-attendees: part of p_R is the
   rooms differing, not the reader differing. t_R (secondary) and the
   memory-grain variance decomposition address this.
4. N = 15 readers (6 originals at 8 nights, 6 field draws at 3, drifter/
   cartographer/tinker at 2) — heterogeneous precision; n_nights >= 3
   subset reported as robustness. Bootstrap over readers, 10k draws.

numpy-only, CPU, read-only against the corpus. Run:
  python3 scripts/cross_strata_transfer.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from e2_instrument import (COLD_ENTRY, DIAL_NAMES, FIELD_NIGHTS_EXT,  # noqa: E402
                           Measurement, Night, ORIG6, corpus_sd)

S_NIGHTS = ["S1", "S2", "S3", "S4a", "S4b", "S5", "S6", "S7"]
FAMILY = {
    "S1": "flip@20 canonical", "S2": "flip@8 early", "S3": "flip@20 late",
    "S4a": "cold-entry pre-flip", "S4b": "cold-entry post-flip",
    "S5": "no-flip (null)", "S6": "double reversal", "S7": "oscillation",
}
BOOT_B, BOOT_SEED = 10_000, 20260820
PERM_B, PERM_SEED = 10_000, 20260821


# --------------------------------------------------------------------------- #
# Stats helpers (numpy-only)                                                   #
# --------------------------------------------------------------------------- #
def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    def rank(v):
        v = np.asarray(v, float)
        order = np.argsort(v)
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=float)
        return r
    return pearson(rank(x), rank(y))


def boot_ci(fn, n, B=BOOT_B, seed=BOOT_SEED):
    """Reader-level bootstrap: resample reader indices, recompute fn(idx)."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        v = fn(idx)
        if np.isfinite(v):
            draws.append(v)
    if not draws:
        return (float("nan"),) * 2, 0
    return (float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5))), len(draws)


def perm_p(x, y, B=PERM_B, seed=PERM_SEED):
    """Two-sided permutation p for corr(x, y); NaN pairs dropped."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    obs = pearson(x, y)
    if not np.isfinite(obs):
        return float("nan"), obs
    rng = np.random.default_rng(seed)
    yy = y.copy()
    hits = 0
    for _ in range(B):
        rng.shuffle(yy)
        r = pearson(x, yy)
        if np.isfinite(r) and abs(r) >= abs(obs) - 1e-12:
            hits += 1
    return (hits + 1) / (B + 1), obs


def ols_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() < 1e-12:
        return float("nan")
    xm, ym = x.mean(), y.mean()
    return float(np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2))


def resid(v, Z):
    """Frisch-Waugh: residual of v on controls Z (with intercept)."""
    v, Z = np.asarray(v, float), np.asarray(Z, float)
    ok = np.isfinite(v) & np.isfinite(Z).all(axis=1)
    v, Z = v[ok], Z[ok]
    X = np.column_stack([np.ones(len(Z)), Z])
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    out = np.full(len(v), np.nan)
    out[ok] = v - X @ beta
    return out


# --------------------------------------------------------------------------- #
# Build the measurement (reused registered instrument, logged v:2 facts)       #
# --------------------------------------------------------------------------- #
def build():
    spec = {}
    for r, nights in FIELD_NIGHTS_EXT.items():
        sn = [n for n in nights if n in S_NIGHTS]
        if not sn:
            continue
        spec[r] = {"params": None, "nights": {n: r for n in sn},
                   "cold": COLD_ENTRY.get(r, [])}
    sd, _ = corpus_sd([Night(n) for n in S_NIGHTS])
    m = Measurement(spec, sd, include_nights=S_NIGHTS,
                    include_null_drift=False, presence="actual")
    return m, sd


def room_mu():
    """Final room mu_hat per S-night, from session_close.final."""
    mu = {}
    for n in S_NIGHTS:
        rows = [json.loads(l) for l in
                open(os.path.join(ROOT, "data", "nights",
                                  f"night-{n}.jsonl"), encoding="utf-8")]
        close = next(r for r in rows if r["type"] == "session_close")
        mu[n] = np.asarray(close["final"]["mu_hat"], float)
    return mu


def main():
    m, sd = build()
    readers = sorted(m.readings)
    mu = room_mu()
    mu_bar = np.mean([mu[n] for n in S_NIGHTS], axis=0)

    # --- per-reader grain quantities -------------------------------------- #
    rows = {}
    for r in readers:
        d_sig = m.drift[r][0]
        d_null = m.drift[r][1]
        Mk = {n: m.night_base[n][r] for n in m.readings[r]}
        b = np.mean(list(Mk.values()), axis=0)
        p = float(np.mean([np.linalg.norm(Mk[n] - b) for n in Mk])) / sd
        ts = []
        for n, v in Mk.items():
            u, w = v - b, mu[n] - mu_bar
            if np.linalg.norm(u) > 1e-9 and np.linalg.norm(w) > 1e-9:
                ts.append(float(u @ w /
                                (np.linalg.norm(u) * np.linalg.norm(w))))
        t = float(np.mean(ts)) if ts else float("nan")
        first_night = sorted(Mk, key=lambda n: S_NIGHTS.index(n))[0]
        par = m.nights[first_night].params[r]
        w = np.asarray(par["dial_weights"], float)
        rows[r] = {
            "n_nights": len(Mk), "nights": sorted(Mk), "d": d_sig,
            "d0": d_null, "p": p, "t": t, "Mk": Mk, "b": b,
            "gain": float(np.mean(w / w.max())),
            "charisma": float(par["charisma"]),
            "extremity": float(np.linalg.norm(
                np.asarray(par["vibe_start"], float) - 0.5)),
        }

    names = [r for r in readers if np.isfinite(rows[r]["d"])
             and np.isfinite(rows[r]["p"])]
    d = np.array([rows[r]["d"] for r in names])
    p = np.array([rows[r]["p"] for r in names])
    t = np.array([rows[r]["t"] for r in names])
    gain = np.array([rows[r]["gain"] for r in names])
    cha = np.array([rows[r]["charisma"] for r in names])
    ext = np.array([rows[r]["extremity"] for r in names])
    nn = np.array([rows[r]["n_nights"] for r in names])

    print("=" * 78)
    print("CROSS-STRATA TRANSFER TEST — session grain -> memory/identity grain")
    print("corpus: night-S1..S7 (v:2 per-reader schema), 15 registered readers")
    print("instrument: scripts/e2_instrument.py (reused, unmodified), actual presence")
    print("=" * 78)

    print("\n[0] GRAIN DEFINITIONS (honest):")
    print("    session grain   d_R : E2 registered drift on logged v:2 readings")
    print("                     (mean over signal strata transitions of segment-mean")
    print("                     displacement, corpus-sd units; S5 split = null d0_R)")
    print("    identity        b_R : mean over nights of per-night reading medians")
    print("    memory-of-room  M_R(k): per-night median of R's readings == logged")
    print("                     session_close reader_final (verified: engine median)")
    print("    NO cross-night state exists (lens resets to vibe_start each night);")
    print("    reader_final is the closest honest memory proxy the corpus supports.")
    print("    plasticity      p_R : mean_k ||M_R(k)-b_R|| / corpus_sd")
    print("    room-tracking   t_R : mean_k cos(M_R(k)-b_R, mu_k - mu_bar)")
    print(f"    corpus_sd (S-nights raw field, RMS over dials) = {sd:.4f}")

    print("\n[1] PER-READER TABLE (session grain vs memory grain):")
    print(f"    {'reader':<13}{'n_nights':>8}{'drift d':>9}{'null d0':>9}"
          f"{'plast p':>9}{'track t':>9}{'gain':>7}{'char':>6}{'extr':>7}")
    for r in names:
        q = rows[r]
        print(f"    {r:<13}{q['n_nights']:>8}{q['d']:>9.4f}"
              f"{(q['d0'] if np.isfinite(q['d0']) else float('nan')):>9.4f}"
              f"{q['p']:>9.4f}{q['t']:>9.4f}{q['gain']:>7.3f}"
              f"{q['charisma']:>6.2f}{q['extremity']:>7.3f}")

    print("\n[2] DRIFT PER SCHEDULE FAMILY (mean over attending readers,")
    print("    corpus-sd; registered strata transitions):")
    fam_trans = {}
    for r in names:
        for night, lab, val in m.trans[r]:
            fam_trans.setdefault((night, lab), []).append(val)
    for (night, lab) in sorted(fam_trans):
        vals = fam_trans[(night, lab)]
        print(f"    {night:<4}{FAMILY[night]:<22}{lab:<18}"
              f"n={len(vals):<3}{float(np.mean(vals)):>7.4f}")
    sig = [rows[r]["d"] for r in names]
    nul = [rows[r]["d0"] for r in names if np.isfinite(rows[r]["d0"])]
    print(f"    signal drift (reader-mean) = {np.mean(sig):.4f}"
          f"  |  null drift (S5, n={len(nul)}) = "
          f"{np.mean(nul) if nul else float('nan'):.4f}"
          f"  |  signal/null = "
          f"{np.mean(sig) / np.mean(nul) if nul else float('nan'):.2f}x")

    # --- memory-grain variance decomposition (ORIG6 balanced 6x8 panel) ---- #
    print("\n[3] MEMORY-GRAIN VARIANCE DECOMPOSITION (ORIG6, balanced 6x8 panel,")
    print("    per dial: share of variance in M_R(k) carried by READER (identity)")
    print("    vs NIGHT (room schedule) vs interaction):")
    shares = {"reader": [], "night": [], "resid": []}
    for dd in range(len(DIAL_NAMES)):
        X = np.array([[rows[r]["Mk"][n][dd] for n in S_NIGHTS]
                      for r in ORIG6])
        X = X - X.mean()
        ss_tot = float(np.sum(X ** 2))
        ss_r = float(np.sum(X.mean(axis=1) ** 2)) * len(S_NIGHTS)
        ss_n = float(np.sum(X.mean(axis=0) ** 2)) * len(ORIG6)
        ss_e = max(ss_tot - ss_r - ss_n, 0.0)
        if ss_tot <= 1e-12:
            continue
        shares["reader"].append(ss_r / ss_tot)
        shares["night"].append(ss_n / ss_tot)
        shares["resid"].append(ss_e / ss_tot)
        print(f"    {DIAL_NAMES[dd]:<14} reader {ss_r / ss_tot:5.2f}  "
              f"night {ss_n / ss_tot:5.2f}  resid {ss_e / ss_tot:5.2f}")
    print(f"    mean shares: reader(identity) {np.mean(shares['reader']):.2f}  "
          f"night(room) {np.mean(shares['night']):.2f}  "
          f"resid {np.mean(shares['resid']):.2f}")

    # --- primary + secondary transfer -------------------------------------- #
    print("\n[4] PRIMARY TRANSFER COEFFICIENT  rho = corr(d_R, p_R):")
    r_dp, obs = pearson(d, p), pearson(d, p)
    (lo, hi), nb = boot_ci(lambda idx: pearson(d[idx], p[idx]), len(names))
    p_dp, _ = perm_p(d, p)
    beta = ols_slope(d, p)
    (blo, bhi), _ = boot_ci(lambda idx: ols_slope(d[idx], p[idx]), len(names))
    print(f"    Pearson r = {r_dp:+.4f}   bootstrap CI [{lo:+.4f}, {hi:+.4f}]"
          f"  ({nb} draws)   perm p = {p_dp:.4f}")
    print(f"    slope beta (p per unit d) = {beta:+.4f}  CI [{blo:+.4f}, {bhi:+.4f}]")
    print("    Spearman (robustness) = "
          f"{spearman(d, p):+.4f}")

    print("\n[5] SECONDARY  corr(d_R, t_R)  (does drift predict ROOM-tracking):")
    r_dt = pearson(d, t)
    (tlo, thi), _ = boot_ci(lambda idx: pearson(d[idx], t[idx]), len(names))
    p_dt, _ = perm_p(d, t)
    print(f"    Pearson r = {r_dt:+.4f}   CI [{tlo:+.4f}, {thi:+.4f}]"
          f"   perm p = {p_dt:.4f}")

    print("\n[6] MECHANISM-AMPLITUDE CONTROLS (Frisch-Waugh partials; see")
    print("    caveat 2 — these knobs are themselves identity parameters):")
    for lab, Z in (("gain", np.column_stack([gain])),
                   ("gain+charisma+extremity",
                    np.column_stack([gain, cha, ext]))):
        rd, rp = resid(d, Z), resid(p, Z)
        pr = pearson(rd, rp)
        pp, _ = perm_p(rd, rp)
        print(f"    partial corr(d,p | {lab:<24}) = {pr:+.4f}   perm p = {pp:.4f}")
    rt, rp_ = pearson(d, t), pearson(d, p)
    print(f"    corr(d, gain) = {pearson(d, gain):+.4f}   "
          f"corr(p, gain) = {pearson(p, gain):+.4f}   "
          f"corr(d, extremity) = {pearson(d, ext):+.4f}   "
          f"corr(p, extremity) = {pearson(p, ext):+.4f}")

    print("\n[7] ROBUSTNESS SUBSETS:")
    sub = nn >= 3
    (slo, shi), _ = boot_ci(
        lambda idx, s=sub: pearson(d[idx][s[idx]], p[idx][s[idx]]), len(names))
    print(f"    n_nights>=3 (n={sub.sum()}): r = {pearson(d[sub], p[sub]):+.4f}"
          f"   CI [{slo:+.4f}, {shi:+.4f}]")
    bk = np.array([r != "barkeep" for r in names])
    (blo2, bhi2), _ = boot_ci(
        lambda idx, s=bk: pearson(d[idx][s[idx]], p[idx][s[idx]]), len(names))
    print(f"    barkeep excluded (n={bk.sum()}): r = {pearson(d[bk], p[bk]):+.4f}"
          f"   CI [{blo2:+.4f}, {bhi2:+.4f}]")
    o6 = np.array([r in ORIG6 for r in names])
    print(f"    ORIG6 only (n={o6.sum()}, descriptive): "
          f"r = {pearson(d[o6], p[o6]):+.4f}")

    # --- verdict ------------------------------------------------------------ #
    print("\n" + "=" * 78)
    print("[8] VERDICT")
    print("=" * 78)
    ok = np.isfinite(r_dp) and np.isfinite(lo) and np.isfinite(hi)
    if ok and lo > 0:
        v = ("TRANSFER: session-grain drift positively predicts memory-grain "
             "plasticity (CI excludes 0 above). Within-session movement "
             "propagates into between-night memory structure — not absorbed.")
    elif ok and hi < 0:
        v = ("NEGATIVE TRANSFER: compensatory organization — big within-session "
             "movers form MORE identity-locked memories (CI excludes 0 below).")
    else:
        v = ("NULL / NO TRANSFER: CI covers 0. Session grain and memory/"
             "identity grain decouple — within-session drift does NOT predict "
             "memory plasticity; fast-grain movement is absorbed before it "
             "reaches memory/identity structure.")
    print(f"    rho = {r_dp:+.4f}   CI [{lo:+.4f}, {hi:+.4f}]   perm p = {p_dp:.4f}")
    print(f"    {v}")
    print("\n    CAVEATS: shared observable (see docstring caveat 1) — the two")
    print("    grains derive from one series; coupling requires survival across")
    print("    median-averaging and night aggregation, but a residual")
    print("    mechanism-amplitude story is not fully excluded by N=15.")
    print("    Exploratory-but-registered: book as designed, not as confirmed.")
    print("=" * 78)


if __name__ == "__main__":
    main()
