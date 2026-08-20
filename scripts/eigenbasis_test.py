"""Eigenbasis diagonalization test (registered) — topic.md harvest fold #2.

Registered claim (research/topic.md, fold 2): "diagonalize the dial-covariance
matrix over the E2 per-reader data; if the top eigen-dimensions align with
the ICC-reliable subspace, the reliability finding is *eigenbasis
conservation* — a deeper account of why those four dials survive."

MATRIX CHOICE (stated, per registration discipline):

  PRIMARY: the BETWEEN-READER covariance of per-reader baseline vectors,
  C_between = cov across readers (ddof=1) of B, where

      baseline_R = mean over R's nights of (per-(reader,night) MEDIAN
                   reading vector, minus that night's mean over readers)

  built through the registered E2 instrument VERBATIM (scripts/e2_instrument
  .Measurement, canonical presence, PRIMARY_NIGHTS, the 15 field readers of
  FIELD_NIGHTS) — this is exactly the data path of the published ICC, whose
  per-dial sigma^2_between IS the diagonal of C_between (asserted below).

  WHY between-reader and not pooled-across-readings: the ICC reliability
  finding is a statement about BETWEEN-reader variance — which dials carry
  reader identity. Its numerator structure is C_between. "Eigenbasis
  conservation" therefore means: the dominant eigen-directions of
  reader-identity variance coincide with the reliable dials. A covariance
  across all readings pools in night-schedule and within-reader variance —
  variance that is by definition NOT reader identity — and is reported only
  as a labeled contrast. The non-trivial content of the test is the
  OFF-diagonal structure: the diagonal ordering is only partially inherited
  from the ICC arithmetic (ICC ranks sigma^2_b against sigma^2_w; it does
  not say the reliable dials dominate the total variance mass, nor that
  they form a coherent low-dimensional block).

OPERATIONALIZATION of "the top eigen-dimensions align" (fixed in this file
before the first run; no data was inspected first):

  u = unit indicator vector of {mood, volume, earnestness, presence}
  cos_k  = ||P_k u||, P_k = projector onto the top-k eigenspace of the matrix
  w_d    = sum_{j<=4} v_jd^2  (top-4 eigen-subspace mass on dial d)
  EXACT subset test: among all C(7,4)=35 coordinate 4-subspaces, rank the
  reliable subset by cos_4; p = (#subsets at least as aligned)/35.

  CONSERVATION fires iff  cos_4 >= 0.95  AND  reliable-subset rank == 1
                          (exact p = 1/35 ~ 0.029).
  PARTIAL if              cos_4 >= 0.85  AND rank <= 3.
  Otherwise NOT conservation.

numpy-only, deterministic (no RNG), read-only against the corpus; writes only
the report EIGENBASIS-TEST-2026-08-20.md at the repo root.

Run:  python3 scripts/eigenbasis_test.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elephant.field import DIAL_NAMES  # canonical 7-dial order
from scripts.e2_field import field_readers
from scripts.e2_instrument import (D, PRIMARY_NIGHTS, Measurement, Night,
                                   corpus_sd)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "data", "e2", "e2-field-results.json")
REPORT = os.path.join(ROOT, "EIGENBASIS-TEST-2026-08-20.md")

RELIABLE = ("mood", "volume", "earnestness", "presence")
REL_IDX = tuple(DIAL_NAMES.index(d) for d in RELIABLE)  # (0, 1, 2, 6)
UNRELIABLE = ("cynicism", "joke_landing", "panic")
UNREL_IDX = tuple(DIAL_NAMES.index(d) for d in UNRELIABLE)

COS_FULL, COS_PARTIAL, RANK_PARTIAL = 0.95, 0.85, 3
K_SUB = 4  # matched comparison: top-4 eigenspace vs the 4 reliable dials


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def ranks(x):
    """Ranks (1 = largest) without ties (values here are continuous)."""
    order = np.argsort(-np.asarray(x), kind="stable")
    r = np.empty(len(x), dtype=int)
    r[order] = np.arange(1, len(x) + 1)
    return r


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def eig_desc(M):
    """Symmetric eigendecomposition, descending eigenvalues."""
    lam, V = np.linalg.eigh(M)  # ascending
    order = np.argsort(lam)[::-1]
    return lam[order], V[:, order]


def subspace_cos(V, k, idx):
    """||P_k u|| for u = unit indicator of the dial set `idx`."""
    u = np.zeros(D)
    u[list(idx)] = 1.0
    u /= np.linalg.norm(u)
    return float(np.linalg.norm(V[:, :k].T @ u))


def subset_ranks(V, k=K_SUB):
    """cos_k for every coordinate 4-subset; returns ({subset: cos}, sorted desc)."""
    out = {}
    for S in itertools.combinations(range(D), k):
        out[S] = subspace_cos(V, k, S)
    return out, sorted(out.items(), key=lambda kv: -kv[1])


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    published = json.load(open(RESULTS, encoding="utf-8"))

    # The registered instrument, exactly as the E2 field arm used it.
    nights = {n: Night(n) for n in PRIMARY_NIGHTS}
    sd, _ = corpus_sd(list(nights.values()))
    m = Measurement(field_readers(), sd, presence="canonical")

    print("=" * 76)
    print("EIGENBASIS DIAGONALIZATION TEST (registered) — topic.md fold #2")
    print("=" * 76)
    print(f"\n[0] corpus: {len(PRIMARY_NIGHTS)} primary nights, corpus_sd={sd:.4f}, "
          f"{len(m.readers)} field readers, canonical presence (as published)")

    # ---- guard: instrument state reproduces the published ICC ------------ #
    icc_agg, icc_dial = m.icc()
    drift = [abs(icc_dial[d] - published["icc"]["per_dial"][d])
             for d in DIAL_NAMES]
    assert max(drift) < 1e-9, f"ICC drift vs published: {max(drift)}"
    print(f"[1] ICC reproduced exactly (max |diff| = {max(drift):.2e}); "
          f"aggregate = {icc_agg:.4f}")

    # ---- baselines: per-reader mean of night-mean-removed night medians --- #
    # (identical row construction to Measurement.icc(), so diag(C_between)
    #  equals the ICC's per-dial sigma^2_between by construction — asserted)
    adj = {r: {} for r in m.readers}  # reader -> dial-vector per night
    for night, nb in m.night_base.items():
        present = [r for r in m.readers if r in nb]
        if len(present) < 2:
            continue
        M = np.stack([nb[r] for r in present])          # readers x dials
        M = M - M.mean(axis=0, keepdims=True)           # night (schedule) means
        for i, r in enumerate(present):
            adj[r][night] = M[i]
    B = np.stack([np.mean(list(adj[r].values()), axis=0) for r in m.readers])
    print(f"[2] per-reader schedule-adjusted baselines: B = {B.shape} "
          f"(readers x dials); nights per reader: "
          f"{min(len(v) for v in adj.values())}-{max(len(v) for v in adj.values())}")

    C = np.cov(B, rowvar=False, ddof=1)
    lam, V = eig_desc(C)
    assert abs(lam.sum() - np.trace(C)) < 1e-9, "eigendecomposition trace mismatch"

    # diag(C) == ICC sigma^2_between (recompute per dial to assert)
    s2b = np.array([np.var(B[:, d], ddof=1) for d in range(D)])
    assert np.abs(np.diag(C) - s2b).max() < 1e-12
    icc_vec = np.array([published["icc"]["per_dial"][d] for d in DIAL_NAMES])
    s2w = s2b * (1.0 - icc_vec) / icc_vec  # back out within-variance for report
    print("[3] C_between = cov across readers of B (7x7, ddof=1); "
          "diag == ICC sigma^2_between (asserted, max |diff| "
          f"{np.abs(np.diag(C) - s2b).max():.1e})")

    # ---- eigendecomposition ---------------------------------------------- #
    share = lam / lam.sum()
    print("\n[4] EIGENSYSTEM of C_between (descending):")
    print("    j |  eigenvalue  | var-share | cumulative | reliable-mass "
          "| dominant loadings (|v| >= .30)")
    rel_mass = []
    for j in range(D):
        rm = float(np.sum(V[REL_IDX, j] ** 2))
        rel_mass.append(rm)
        loads = ", ".join(f"{DIAL_NAMES[d]}:{V[d, j]:+.2f}"
                          for d in range(D) if abs(V[d, j]) >= 0.30)
        print(f"    {j} | {lam[j]:11.6f} | {share[j]:.4f}    | "
              f"{share[:j + 1].sum():.4f}     | {rm:.3f}        | {loads}")

    # ---- alignment with the ICC-reliable subspace ------------------------- #
    cos_k = [subspace_cos(V, k, REL_IDX) for k in range(1, D + 1)]
    print("\n[5] ALIGNMENT: cos(top-k eigenspace, reliable indicator u)")
    for k, c in enumerate(cos_k, 1):
        print(f"    k={k}: cos = {c:.4f}")
    cos4 = cos_k[K_SUB - 1]

    w = (V[:, :K_SUB] ** 2).sum(axis=1)  # top-4 subspace mass per dial
    h2 = np.array([np.sum(lam[:K_SUB] * V[d, :K_SUB] ** 2) / C[d, d]
                   for d in range(D)])   # share of dial variance captured

    print("\n[6] per-dial eigen-mass w_d (top-4 subspace) and communality "
          "h^2_d (variance captured):")
    print("    dial           | ICC    | sigma^2_b | w_d (rank) | h^2_d")
    for d in range(D):
        tag = "R" if d in REL_IDX else "."
        print(f"    {DIAL_NAMES[d]:<14} | {icc_vec[d]:.4f} | {s2b[d]:.4f}   "
              f"| {w[d]:.4f} ({ranks(w)[d]}) | {h2[d]:.4f}  {tag}")

    # exact subset test
    subsets, ranked = subset_ranks(V, K_SUB)
    rel_set = tuple(sorted(REL_IDX))
    order = [S for S, _ in ranked]
    rel_rank = order.index(rel_set) + 1
    n_ge = sum(1 for c in subsets.values() if c >= subsets[rel_set] - 1e-12)
    p_exact = n_ge / len(subsets)
    print(f"\n[7] EXACT SUBSET TEST (all {len(subsets)} coordinate 4-subspaces, "
          f"top-{K_SUB} eigenspace):")
    for S, c in ranked[:4]:
        names = "+".join(DIAL_NAMES[d] for d in S)
        mark = "  <-- ICC-reliable" if S == rel_set else ""
        print(f"    {c:.4f}  {names}{mark}")
    print(f"    reliable-subset rank = {rel_rank}/{len(subsets)}; "
          f"exact p = {p_exact:.4f} ({n_ge}/{len(subsets)} at least as aligned)")

    print(f"\n[8] correlations: ICC vs sigma^2_b: pearson "
          f"{pearson(icc_vec, s2b):+.4f}, spearman {spearman(icc_vec, s2b):+.4f}; "
          f"ICC vs w_d: pearson {pearson(icc_vec, w):+.4f}, "
          f"spearman {spearman(icc_vec, w):+.4f}")

    # ---- robustness: standardized (correlation) matrix -------------------- #
    R = np.corrcoef(B, rowvar=False)
    lam_r, V_r = eig_desc(R)
    cos4_r = subspace_cos(V_r, K_SUB, REL_IDX)
    subs_r, ranked_r = subset_ranks(V_r, K_SUB)
    rel_rank_r = [S for S, _ in ranked_r].index(rel_set) + 1
    print(f"\n[9] ROBUSTNESS (correlation-matrix version, scale removed): "
          f"cos_4 = {cos4_r:.4f}, reliable-subset rank = "
          f"{rel_rank_r}/{len(subs_r)}")

    # ---- contrast: pooled covariance across readings ---------------------- #
    pooled = np.array([v for r in m.readings for nt in m.readings[r]
                       for _, v in m.readings[r][nt]], dtype=float)
    C_p = np.cov(pooled, rowvar=False, ddof=1)
    lam_p, V_p = eig_desc(C_p)
    cos4_p = subspace_cos(V_p, K_SUB, REL_IDX)
    subs_p, ranked_p = subset_ranks(V_p, K_SUB)
    rel_rank_p = [S for S, _ in ranked_p].index(rel_set) + 1
    print(f"[10] CONTRAST (pooled covariance across ALL canonical readings "
          f"(n={len(pooled)}), between+within+schedule): cos_4 = {cos4_p:.4f}, "
          f"reliable-subset rank = {rel_rank_p}/{len(subs_p)}")

    # ---- verdict ---------------------------------------------------------- #
    if cos4 >= COS_FULL and rel_rank == 1:
        verdict_txt = ("EIGENBASIS CONSERVATION CONFIRMED: the top eigen-"
                       "dimensions of the between-reader dial-covariance ARE "
                       "the ICC-reliable subspace")
    elif cos4 >= COS_PARTIAL and rel_rank <= RANK_PARTIAL:
        verdict_txt = ("PARTIAL eigenbasis conservation: top eigenspace leans "
                       "reliable but does not coincide with it")
    else:
        verdict_txt = ("NOT eigenbasis conservation: the reliability finding "
                       "does not survive as eigen-subspace structure")
    print("\n" + "=" * 76)
    print(f"VERDICT: {verdict_txt}")
    print(f"  (cos_4 = {cos4:.4f} vs threshold {COS_FULL}; "
          f"reliable-subset rank {rel_rank}/35, exact p = {p_exact:.4f})")
    print("=" * 76)

    # ---- report ------------------------------------------------------------ #
    md = []
    md.append("# Eigenbasis Diagonalization Test (registered) — 2026-08-20\n")
    md.append("**Registered in** `research/topic.md` fold 2: *diagonalize the "
              "dial-covariance matrix over the E2 per-reader data; if the top "
              "eigen-dimensions align with the ICC-reliable subspace, the "
              "reliability finding is eigenbasis conservation.*\n")
    md.append("## Matrix (choice stated per the task)\n")
    md.append("**PRIMARY: between-reader covariance of per-reader baselines.** "
              "`baseline_R = mean over R's nights of (per-(reader,night) median "
              "reading, night-mean removed)`, built through the registered E2 "
              "instrument verbatim (`scripts/e2_instrument.Measurement`, "
              "canonical presence, 9 primary nights, 15 field readers) — the "
              "ICC's per-dial sigma^2_between is exactly the diagonal of this "
              "matrix (asserted numerically). **Why between-reader:** the "
              "reliability finding is a claim about which dials carry *reader "
              "identity*; its variance object is the between-reader "
              "covariance. Pooled across-readings covariance mixes in "
              "schedule and within-reader variance (non-identity) and is "
              "reported only as a contrast.\n")
    md.append(f"Corpus: 9 primary nights (A, D, D-cold, S1-S5), "
              f"corpus_sd = {sd:.4f}, {len(m.readers)} readers; published ICC "
              f"reproduced exactly (aggregate {icc_agg:.4f}).\n")
    md.append("## Eigensystem of C_between\n")
    md.append("| j | eigenvalue | var-share | cum | reliable-mass | dominant "
              "loadings (|v|>=.30) |")
    md.append("|---|-----------|-----------|-----|---------------|"
              "--------------------------|")
    for j in range(D):
        loads = ", ".join(f"{DIAL_NAMES[d]} {V[d, j]:+.2f}"
                          for d in range(D) if abs(V[d, j]) >= 0.30)
        md.append(f"| {j} | {lam[j]:.6f} | {share[j]:.4f} | "
                  f"{share[:j + 1].sum():.4f} | {rel_mass[j]:.3f} | {loads} |")
    md.append("")
    md.append("## Alignment with the ICC-reliable subspace "
              "(mood/volume/earnestness/presence)\n")
    md.append("cos(top-k eigenspace, reliable indicator): "
              + ", ".join(f"k={k}: {c:.4f}" for k, c in enumerate(cos_k, 1))
              + "\n")
    md.append("| dial | ICC | sigma^2_b | w_d top-4 mass (rank) | h^2_d "
              "variance captured | reliable |")
    md.append("|------|-----|-----------|-----------------------|"
              "-----------------------------|----------|")
    for d in range(D):
        tag = "yes" if d in REL_IDX else "no"
        md.append(f"| {DIAL_NAMES[d]} | {icc_vec[d]:.4f} | {s2b[d]:.6f} | "
                  f"{w[d]:.4f} ({ranks(w)[d]}) | {h2[d]:.4f} | {tag} |")
    md.append("")
    md.append(f"**Exact subset test** (all 35 coordinate 4-subspaces vs the "
              f"top-4 eigenspace): reliable subset ranks **{rel_rank}/35** "
              f"(exact p = {p_exact:.4f}, {n_ge} subsets at least as aligned); "
              f"most-aligned subset: "
              + "+".join(DIAL_NAMES[d] for d in ranked[0][0])
              + f" (cos {ranked[0][1]:.4f}) vs reliable "
              f"{cos4:.4f}.\n")
    md.append(f"Correlations across dials: ICC vs sigma^2_b pearson "
              f"{pearson(icc_vec, s2b):+.4f} / spearman "
              f"{spearman(icc_vec, s2b):+.4f}; ICC vs w_d pearson "
              f"{pearson(icc_vec, w):+.4f} / spearman "
              f"{spearman(icc_vec, w):+.4f}.\n")
    md.append(f"**Robustness** (correlation-matrix version): cos_4 = "
              f"{cos4_r:.4f}, rank {rel_rank_r}/35. ")
    md.append(f"**Contrast** (pooled covariance across all {len(pooled)} "
              f"canonical readings): cos_4 = {cos4_p:.4f}, rank "
              f"{rel_rank_p}/35 — the between-reader isolation is what "
              "carries (or fails to carry) the alignment.\n")
    md.append(f"**Within-reader sigma^2_w (backed out of ICC and sigma^2_b):** "
              + ", ".join(f"{DIAL_NAMES[d]} {s2w[d]:.4f}" for d in range(D))
              + "\n")
    md.append(f"## Verdict\n")
    md.append(f"**{verdict_txt}** — cos_4 = {cos4:.4f} "
              f"(threshold {COS_FULL}), reliable-subset rank {rel_rank}/35 "
              f"(exact p = {p_exact:.4f}).\n")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\n[eigenbasis] report -> {REPORT}")


if __name__ == "__main__":
    main()
