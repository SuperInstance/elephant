"""REG-1 — the W-vs-v* rotation test (registered).

Registered as the foundation's next experiment (foundation-synthesis-
2026-08-21.md, axiom 7 + "Registered Experiment 1") and sequenced as S0 of
the wave-3 plan (wave3-generation-plan-2026-08-21.md §2/§4). This run
executes it on the EXISTING filed corpora — wave-1 (field S-nights) and
wave-2 (T-nights) — never pooled.

THE TEST (geometric foundation §3.4): the data-derived room temperature
axis v* is the leading generalized eigenvector of

    C_room · v = λ · C_pers · v

with (foundation §3.4 verbatim: "both already computed above" from §3.1)
    C_pers = between-reader covariance of reader offsets
             (baseline mean per reader minus the roster mean) — the
             personality fixed-effect covariance, and
    C_room  = within-reader between-night covariance of baselines
             (baseline minus reader mean, pooled over reader-nights) —
             the room response (night/schedule effects seen through
             readers, i.e. the variance of the roster-mean trajectory).

v* maximizes room-response variance per unit personality variance:
maximally room-responsive, minimally personality-loaded. The a-priori
warm direction W (elephant/vmf.py WARM, z-space) is then located against
v* and against PC1_personality (leading eigenvector of C_pers).

DATA PATH — the filed instrument VERBATIM (read-only):
  scripts/e2_instrument.Measurement, canonical presence (registered
  primary channel), per-(reader,night) MEDIAN baselines (night_base);
  wave-1 = FIELD_NIGHTS attendance over PRIMARY_NIGHTS (15 readers,
  9 nights — the published ICC's data path, as eigenbasis_test.py);
  wave-2 = FIELD_NIGHTS_W2 over W2_NIGHT_LIST (21 readers, 9 T-nights,
  as slope_regression_w2.py). Baselines standardized to z-space
  (z = SCALE·(v − CENTER), the vmf.py chart) before all covariance
  work — matching the geometric team's computations.

SUBSPACE CHOICE (documented per registration):
  PRIMARY = full 7-dial z-space (the filed warmth functional reads all
  seven dials; the eigenproblem itself whitens C_pers, so unreliable
  dials are downweighted by their personality covariance, not dropped).
  SENSITIVITY = ICC-reliable subspace {mood, volume, earnestness,
  presence} (RIDX, premise_band_movers' registered reliable set) — the
  space in which the geometric team validated cos(W, v_temp) ∈
  [0.24, 0.40]; this is the branch-1 comparability check.

VERDICT BRANCHES (pre-stated, operationalized before the first run):
  A  warmth is room temperature      cos(W, v*) ≥ 0.80 AND bootstrap
                                     95% CI lower bound > 0.60
  B  warmth is reader personality    cos(W, PC1_pers) ≥ 0.80 AND
                                     cos(W, v*) < 0.80 (W parallel to
                                     the personality axis, off the
                                     temperature axis)
  C  between: instrument needs a     otherwise — the rotation is
     rotated axis                    reported (angle + v* itself)
  Branch-1 range check (foundation): cos(W, v*) in the reliable
  subspace ∈ [0.24, 0.40] → confirmed, else a dated deviation is filed.

Axes are sign-free: cosines reported as |cos| (sign of an eigenvector is
arbitrary; the warm pole fixes W's sign, v* carries none).

numpy + scipy.linalg.eigh only; read-only against data/nights/; the only
writes are data/slope/reg1-rotation-results.json (+ the run doc, filed
separately). Bootstrap: reader-clustered, B=2000, seed 20260821.

Run:  python3 scripts/reg1_rotation.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elephant.field import DIAL_NAMES
from elephant.vmf import CENTER, SCALE, WARM
from scripts.e2_field import field_readers
from scripts.e2_instrument import (COLD_ENTRY_W2, FIELD_NIGHTS_W2,
                                   PRIMARY_NIGHTS, W2_NIGHT_LIST,
                                   Measurement, Night, corpus_sd)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "slope", "reg1-rotation-results.json")

SEED = 20260821
B_BOOT = 2000
EPS_FLOOR = 1e-2         # C_pers eigenvalue floor (fraction of its leading
                        # eigenvalue) for the whitened generalized solve;
                        # sensitivity surface reported alongside
RELIABLE = ["mood", "volume", "earnestness", "presence"]   # ICC reliable
RIDX = [DIAL_NAMES.index(d) for d in RELIABLE]

W = np.asarray(WARM, float)                       # z-space warm direction
W /= np.linalg.norm(W)
W_REL = WARM[RIDX] / np.linalg.norm(WARM[RIDX])   # restricted, renormalized
W_MASS_RELIABLE = float(np.linalg.norm(WARM[RIDX]))   # mass inside RIDX

# pre-stated branch thresholds (see module docstring)
ALIGNED_COS = 0.80
ALIGNED_CI_LO = 0.60
PERS_PAR_COS = 0.80
BRANCH1_LO, BRANCH1_HI = 0.24, 0.40


def w2_field_readers():
    rn = {}
    for r, nights in FIELD_NIGHTS_W2.items():
        rn[r] = {"params": None, "nights": {n: r for n in nights},
                 "cold": COLD_ENTRY_W2.get(r, [])}
    return rn


# --------------------------------------------------------------------------- #
# Covariance construction (foundation §3.1/§3.4)                              #
# --------------------------------------------------------------------------- #
def z_cells(m: Measurement, cell_base: bool = False) -> dict:
    """{(reader, night[, label]): z-space baseline vector}.

    Primary: per-(reader,night) median (m.night_base). cell_base=True:
    per-(night, stratum) cell medians (m.cell_base) — the finer room-
    trajectory grain (within-night room response included)."""
    src = (m.cell_base if cell_base else
           {(n, ""): nb for n, nb in m.night_base.items()})
    out = {}
    for key, nb in src.items():
        night, label = key          # both src shapes are (night, label)
        for r, v in nb.items():
            out[(r, night, label)] = SCALE * (np.asarray(v, float) - CENTER)
    return out


def decompose(cells: dict, ridx, remove_night_mean: bool = False):
    """Build C_room, C_pers (+ offsets/devs) from z-cells.

    remove_night_mean=True: the ICC-style construction — subtract each
    night's roster mean first (bounds the unbalanced-attendance leakage
    of schedule into reader offsets; labeled sensitivity, NOT primary:
    night means carry the schedule, i.e. the room signal itself)."""
    ridx = list(ridx)
    Z = {}
    groups = sorted({k[1:] for k in cells})
    for g in groups:
        zs = [cells[k][ridx] for k in cells if k[1:] == g]
        mu = np.mean(zs, axis=0)
        for k in cells:
            if k[1:] == g:
                Z[k] = (cells[k][ridx] - mu) if remove_night_mean \
                    else cells[k][ridx]
    readers = sorted({k[0] for k in Z})
    grand = np.mean([Z[k] for k in Z], axis=0)
    offs = {}
    for r in readers:
        rs = [Z[k] for k in Z if k[0] == r]
        offs[r] = np.mean(rs, axis=0) - grand
    devs = [Z[k] - (offs[k[0]] + grand) for k in Z]   # z − m_R
    O = np.stack([offs[r] for r in readers])
    Dm = np.stack(devs)
    C_pers = np.cov(O, rowvar=False, ddof=1) if len(readers) > 1 else None
    C_room = np.cov(Dm, rowvar=False, ddof=1)
    return {"C_room": C_room, "C_pers": C_pers, "offsets": offs,
            "n_readers": len(readers), "n_cells": len(Z), "Z": Z,
            "readers": readers}


def solve_gen(C_room, C_pers, eps=1e-2):
    """Leading generalized eigenvectors of C_room v = λ C_pers v.

    Construction: whiten C_pers by its (floored) eigendecomposition —
    C_pers = V diag(e) V', e_floored = max(e, eps·e_max) — then
    diagonalize the whitened C_room. The floor is the honesty guard for
    near-null personality directions: with 15–21 readers, a C_pers
    eigenvalue below ~1% of the leading one is statistically
    indistinguishable from zero, and an unfloored solve returns a
    spurious huge-λ direction along the null space (observed: panic in
    wave-1 is a dead dial — zero between-reader AND between-night
    variance — and the raw ridge solve returned a 1600-norm vector
    there). eps = 1e-2 primary; the eps-sensitivity surface is reported
    alongside. Eigenvectors are returned Euclideally normalized (the
    solve is B-orthogonal; cosines are scale-invariant either way)."""
    d = C_room.shape[0]
    e, V = np.linalg.eigh(C_pers)   # (eigenvalues, eigenvectors)
    emax = float(e[-1]) if e[-1] > 0 else 1.0
    ef = np.maximum(e, eps * emax)
    A = V @ np.diag(ef ** -0.5)              # floored whitening
    M = A.T @ C_room @ A
    M = (M + M.T) / 2.0
    theta, U = np.linalg.eigh(M)
    order = np.argsort(theta)[::-1]
    theta, U = theta[order], U[:, order]
    evecs = A @ U
    evecs = evecs / np.linalg.norm(evecs, axis=0, keepdims=True)
    return {"evals": theta, "evecs": evecs,
            "cond_pers": float(np.linalg.cond(C_pers)),
            "n_floored": int((e < eps * emax).sum()), "eps": eps}


def pc1(C):
    evals, evecs = np.linalg.eigh(C)
    return evecs[:, np.argmax(evals)], float(np.max(evals))


def abs_cos(a, b):
    return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def icc_along(Z: dict, v) -> float:
    """ICC of the projection v·z (night means removed, reader
    between/within split) — the e2_instrument.icc recipe for one
    derived dial."""
    by_group = {}
    for k, z in Z.items():
        by_group.setdefault(k[1:], []).append(float(v @ z))
    rows = {}
    for g, xs in by_group.items():
        mu = float(np.mean(xs))
        for k, z in [(k, z) for k, z in Z.items() if k[1:] == g]:
            rows.setdefault(k[0], []).append(float(v @ z) - mu)
    within = [float(np.var(vs, ddof=1)) for vs in rows.values() if len(vs) > 1]
    s2w = float(np.mean(within)) if within else 0.0
    means = [float(np.mean(vs)) for vs in rows.values()]
    s2b = float(np.var(means, ddof=1)) if len(means) > 1 else 0.0
    return s2b / (s2b + s2w) if (s2b + s2w) > 0 else float("nan")


def wave_space(m: Measurement, cells: dict, ridx, w_vec,
               remove_night_mean: bool = False, dials: list = None):
    """One (wave × subspace × construction) cell of the analysis."""
    dec = decompose(cells, ridx, remove_night_mean=remove_night_mean)
    sol = solve_gen(dec["C_room"], dec["C_pers"])
    v_star = sol["evecs"][:, 0]
    v2 = sol["evecs"][:, 1] if sol["evecs"].shape[1] > 1 else None
    p1, p1_ev = pc1(dec["C_pers"])
    r1, r1_ev = pc1(dec["C_room"])
    # geometric team's original v* proxy: PC1 of C_room alone
    out = {
        "dials": dials or [DIAL_NAMES[i] for i in ridx],
        "n_readers": dec["n_readers"], "n_reader_nights": dec["n_cells"],
        "gen_evals": [float(x) for x in sol["evals"]],
        "lambda_star": float(sol["evals"][0]),
        "v_star": {d: float(x) for d, x in zip(out_dials(ridx), v_star)},
        "v2": ({d: float(x) for d, x in zip(out_dials(ridx), v2)}
               if v2 is not None else None),
        "cos_W_vstar": abs_cos(w_vec, v_star),
        "cos_W_v2": abs_cos(w_vec, v2) if v2 is not None else None,
        "cos_W_pc1_pers": abs_cos(w_vec, p1),
        "cos_W_pc1_room": abs_cos(w_vec, r1),
        "pc1_pers_eval_share": None,   # filled below
        "icc_along_vstar": icc_along(dec["Z"], v_star),
        "icc_along_W": icc_along(dec["Z"], w_vec),
        "cond_pers": sol["cond_pers"],
        "n_pers_dirs_floored": sol["n_floored"],
        "eps_floor": sol["eps"],
    }
    ev = np.linalg.eigvalsh(dec["C_pers"])[::-1]
    out["pers_evals"] = [float(x) for x in ev]
    out["pc1_pers_eval_share"] = float(ev[0] / ev.sum()) if ev.sum() > 0 else None
    # eps-sensitivity of the headline cosine (floor level of the whitening)
    out["cos_W_vstar_eps_sensitivity"] = {
        f"eps={eps:g}": abs_cos(w_vec, solve_gen(dec["C_room"], dec["C_pers"],
                                                 eps=eps)["evecs"][:, 0])
        for eps in (1e-4, 1e-3, 1e-2, 5e-2)}
    return out, dec


def out_dials(ridx):
    return [DIAL_NAMES[i] for i in ridx]


def bootstrap(cells: dict, B=B_BOOT):
    """Reader-clustered bootstrap: resample readers, rebuild both
    covariances, re-solve, re-collect the cosines. Full + reliable."""
    rng = np.random.default_rng(SEED)
    readers = sorted({k[0] for k in cells})
    full, rel = [], []
    for _ in range(B):
        draw = [readers[i] for i in rng.integers(0, len(readers), len(readers))]
        bcells = {}
        for i, r in enumerate(draw):
            for k, v in cells.items():
                if k[0] == r:
                    # positional tag keeps repeated draws of one reader
                    # as distinct multiset members
                    bcells[(f"{r}#{i}",) + k[1:]] = v
        try:
            dec = decompose(bcells, list(range(7)))
            sol = solve_gen(dec["C_room"], dec["C_pers"])
            full.append((abs_cos(W, sol["evecs"][:, 0]),
                         abs_cos(W, pc1(dec["C_pers"])[0])))
            dec_r = decompose(bcells, RIDX)
            sol_r = solve_gen(dec_r["C_room"], dec_r["C_pers"])
            rel.append((abs_cos(W_REL, sol_r["evecs"][:, 0]),
                        abs_cos(W_REL, pc1(dec_r["C_pers"])[0])))
        except Exception:
            continue

    def ci(xs):
        return [float(np.percentile([x[0] for x in xs], 2.5)),
                float(np.percentile([x[0] for x in xs], 97.5))] if xs else None

    def ci2(xs):
        return [float(np.percentile([x[1] for x in xs], 2.5)),
                float(np.percentile([x[1] for x in xs], 97.5))] if xs else None

    return {"cos_W_vstar_ci": ci(full), "cos_W_pc1_ci": ci2(full),
            "cos_W_vstar_ci_reliable": ci(rel),
            "cos_W_pc1_ci_reliable": ci2(rel), "draws": len(full)}


def branch(cos_v, cos_pc1, ci):
    lo = ci[0] if ci else float("nan")
    if cos_v >= ALIGNED_COS and lo > ALIGNED_CI_LO:
        return ("A: W aligns with v* — warmth IS room temperature")
    if cos_pc1 >= PERS_PAR_COS and cos_v < ALIGNED_COS:
        return ("B: W does not align with v* and is parallel to the "
                "personality axis — warmth is reader personality")
    return ("C: W between the axes — the instrument needs a rotated "
            "axis (rotation reported)")


def run_wave(wave: int, presence: str = "canonical"):
    if wave == 1:
        nights_list = list(PRIMARY_NIGHTS)
        rn = field_readers()
    else:
        nights_list = list(W2_NIGHT_LIST)
        rn = w2_field_readers()
    sd, _ = corpus_sd([Night(n) for n in nights_list])
    m = Measurement(rn, sd, include_nights=nights_list, presence=presence)

    icc_agg, icc_dial = m.icc()
    cells = z_cells(m)

    full, dec = wave_space(m, cells, list(range(7)), W)
    rel, _ = wave_space(m, cells, RIDX, W_REL, dials=RELIABLE)

    boot = bootstrap(cells)

    # labeled sensitivities (primary stands)
    nmr_full, _ = wave_space(m, cells, list(range(7)), W,
                             remove_night_mean=True)
    nmr_rel, _ = wave_space(m, cells, RIDX, W_REL, remove_night_mean=True,
                            dials=RELIABLE)
    cell_cells = z_cells(m, cell_base=True)
    cell_full, _ = wave_space(m, cell_cells, list(range(7)), W)

    # roster-trajectory variant: C_room = cov over nights of roster means
    night_means = {}
    for n in sorted({k[1] for k in cells}):
        night_means[n] = np.mean([cells[k] for k in cells if k[1] == n],
                                 axis=0)
    NM = np.stack(list(night_means.values()))
    C_traj = np.cov(NM, rowvar=False, ddof=1)
    sol_t = solve_gen(C_traj, dec["C_pers"])
    traj = {"cos_W_vstar_traj": abs_cos(W, sol_t["evecs"][:, 0]),
            "gen_evals": [float(x) for x in sol_t["evals"]],
            "v_star": {d: float(x) for d, x in
                       zip(out_dials(range(7)), sol_t["evecs"][:, 0])}}

    # actual-presence sensitivity
    m_act = Measurement(rn, sd, include_nights=nights_list, presence="actual")
    act_full, _ = wave_space(m_act, z_cells(m_act), list(range(7)), W)

    v_star = np.array([full["v_star"][d] for d in DIAL_NAMES])
    verdict_full = branch(full["cos_W_vstar"], full["cos_W_pc1_pers"],
                          boot["cos_W_vstar_ci"])
    in_range = BRANCH1_LO <= rel["cos_W_vstar"] <= BRANCH1_HI

    return {
        "n_readers": len(m.readers), "n_nights": len(nights_list),
        "n_reader_nights": len(cells), "corpus_sd": sd,
        "icc_aggregate": icc_agg, "icc_per_dial": icc_dial,
        "primary_full7": full,
        "reliable_subspace": rel,
        "bootstrap": boot,
        "sensitivities": {
            "night_mean_removed_full7": nmr_full,
            "night_mean_removed_reliable": nmr_rel,
            "cell_level_C_room_full7": cell_full,
            "roster_trajectory_C_room": traj,
            "actual_presence_full7": act_full,
        },
        "branch1_range_check": {
            "range": [BRANCH1_LO, BRANCH1_HI],
            "cos_W_vstar_reliable": rel["cos_W_vstar"],
            "in_range": bool(in_range),
            "note": ("confirmed" if in_range else
                     "dated deviation filed (see run doc)")},
        "rotation": {
            "angle_deg_W_to_vstar":
                float(np.degrees(np.arccos(min(1.0, full["cos_W_vstar"])))),
            "angle_deg_W_to_vstar_reliable":
                float(np.degrees(np.arccos(min(1.0, rel["cos_W_vstar"])))),
        },
        "verdict_full7": verdict_full,
    }


def main():
    print("=" * 78)
    print("REG-1 — W-vs-v* rotation test (generalized eigenproblem, both waves)")
    print("=" * 78)
    print(f"\nW (z-space, |W|={np.linalg.norm(W):.4f}): "
          + " ".join(f"{d}={x:+.3f}" for d, x in zip(DIAL_NAMES, W)))
    print(f"W mass inside ICC-reliable subspace {{mood,volume,earnestness,"
          f"presence}}: ||W[RIDX]|| = {W_MASS_RELIABLE:.4f}")

    results = {
        "date": "2026-08-21",
        "test": "REG-1 W-vs-v* rotation test — generalized eigenproblem "
                "C_room v = lambda C_pers v on the filed waves 1+2",
        "registration": [
            "foundation-synthesis-2026-08-21.md (axiom 7 + Registered "
            "Experiment 1)",
            "wave3-generation-plan-2026-08-21.md (§2 REG-1 integration, "
            "S0 sequencing)",
            "math-foundation-geometric-2026-08-21.md (§3.4 the eigenproblem)",
        ],
        "W_z": {d: float(x) for d, x in zip(DIAL_NAMES, W)},
        "W_mass_reliable_subspace": W_MASS_RELIABLE,
        "method": {
            "C_room": "within-reader between-night covariance of median "
                      "baselines (baseline minus reader mean), z-space",
            "C_pers": "between-reader covariance of reader offsets "
                      "(reader mean minus roster mean), z-space",
            "baselines": "per-(reader,night) median, canonical presence, "
                         "e2_instrument.Measurement verbatim",
            "primary_space": "full 7-dial z-space; reliable subspace "
                             "{mood,volume,earnestness,pres} = sensitivity "
                             "+ branch-1 comparability",
            "cosines": "|cos| (axes are sign-free)",
            "solver": "floored-whitening generalized eigensolve "
                     "(C_pers eigenvalue floor eps=1e-2 of its leading "
                     "eigenvalue; eps-sensitivity reported)",
            "bootstrap": {"B": B_BOOT, "seed": SEED, "unit": "reader"},
            "branch_thresholds": {"aligned_cos": ALIGNED_COS,
                                  "aligned_ci_lo": ALIGNED_CI_LO,
                                  "pers_parallel_cos": PERS_PAR_COS,
                                  "branch1_range": [BRANCH1_LO, BRANCH1_HI]},
        },
        "waves": {},
    }

    for wave in (1, 2):
        r = run_wave(wave)
        results["waves"][f"wave{wave}"] = r
        f7, rr = r["primary_full7"], r["reliable_subspace"]
        print(f"\n----- wave-{wave}: {r['n_readers']} readers, "
              f"{r['n_nights']} nights, {r['n_reader_nights']} reader-nights "
              f"(ICC agg {r['icc_aggregate']:.4f}) -----")
        print(f"[full 7-dial]  lambda* = {f7['lambda_star']:.3f}   "
              f"spectrum: " + ", ".join(f"{x:.2f}" for x in f7['gen_evals']))
        print(f"  v* = " + "  ".join(f"{d[:4]}:{x:+.2f}"
                                      for d, x in f7['v_star'].items()))
        print(f"  cos(W, v*)          = {f7['cos_W_vstar']:.4f}   "
              f"CI {boot_fmt(r['bootstrap']['cos_W_vstar_ci'])}")
        print(f"  cos(W, v2)          = {f7['cos_W_v2']:.4f}")
        print(f"  cos(W, PC1_pers)    = {f7['cos_W_pc1_pers']:.4f}   "
              f"CI {boot_fmt(r['bootstrap']['cos_W_pc1_ci'])}")
        print(f"  cos(W, PC1_room)    = {f7['cos_W_pc1_room']:.4f}   "
              f"(geometric team's v* proxy; their filed: 0.40 w1 / 0.24 w2)")
        print(f"  eps-sensitivity of cos(W, v*): " +
              ", ".join(f"{k}={v:.3f}" for k, v in
                         f7['cos_W_vstar_eps_sensitivity'].items()) +
              f"  [floored dirs: {f7['n_pers_dirs_floored']}]")
        print(f"  ICC along v*        = {f7['icc_along_vstar']:.4f}   "
              f"(along W: {f7['icc_along_W']:.4f})")
        print(f"[reliable {len(RELIABLE)}-dial]  lambda* = "
              f"{rr['lambda_star']:.3f}")
        print(f"  v* = " + "  ".join(f"{d[:4]}:{x:+.2f}"
                                      for d, x in rr['v_star'].items()))
        print(f"  cos(W, v*)          = {rr['cos_W_vstar']:.4f}   "
              f"CI {boot_fmt(r['bootstrap']['cos_W_vstar_ci_reliable'])}")
        print(f"  cos(W, PC1_pers)    = {rr['cos_W_pc1_pers']:.4f}   "
              f"CI {boot_fmt(r['bootstrap']['cos_W_pc1_ci_reliable'])}")
        print(f"  cos(W, PC1_room)    = {rr['cos_W_pc1_room']:.4f}")
        print(f"  eps-sensitivity of cos(W, v*): " +
              ", ".join(f"{k}={v:.3f}" for k, v in
                         rr['cos_W_vstar_eps_sensitivity'].items()) +
              f"  [floored dirs: {rr['n_pers_dirs_floored']}]")
        print(f"  ICC along v*        = {rr['icc_along_vstar']:.4f}")
        b1 = r["branch1_range_check"]
        print(f"[branch-1 range] cos_reliable = "
              f"{b1['cos_W_vstar_reliable']:.4f} in "
              f"[{BRANCH1_LO}, {BRANCH1_HI}] -> {b1['note']}")
        rot = r["rotation"]
        print(f"[rotation] W->v* angle: {rot['angle_deg_W_to_vstar']:.1f} deg "
              f"(full) / {rot['angle_deg_W_to_vstar_reliable']:.1f} deg "
              f"(reliable)")
        s = r["sensitivities"]
        print(f"[sensitivities] night-mean-removed: "
              f"cos {s['night_mean_removed_full7']['cos_W_vstar']:.4f} | "
              f"cell-level C_room: "
              f"{s['cell_level_C_room_full7']['cos_W_vstar']:.4f} | "
              f"roster-trajectory C_room: "
              f"{s['roster_trajectory_C_room']['cos_W_vstar_traj']:.4f} | "
              f"actual presence: "
              f"{s['actual_presence_full7']['cos_W_vstar']:.4f}")
        print(f"[verdict, full-7 primary] {r['verdict_full7']}")

    # overall verdict across waves
    v1 = results["waves"]["wave1"]
    v2 = results["waves"]["wave2"]
    same = v1["verdict_full7"].split(" — ")[0] == v2["verdict_full7"].split(" — ")[0]
    results["verdict"] = {
        "wave1": v1["verdict_full7"], "wave2": v2["verdict_full7"],
        "waves_agree": bool(same),
        "overall": (v1["verdict_full7"] if same else
                    "SPLIT across waves — see per-wave verdicts"),
    }
    print("\n===== OVERALL: " + results["verdict"]["overall"])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"\n[reg1] results -> {OUT}")


def boot_fmt(ci):
    if not ci:
        return "[n/a]"
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


if __name__ == "__main__":
    main()
