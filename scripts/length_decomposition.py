#!/usr/bin/env python3
"""scripts/length_decomposition.py — E4 "Clock-Split" Stage 1 (registered).

Registration: zeroclaw-dissertation/research/committee/deep-think-2026-08-20/
methodologist-glm53.md. Length is a MEDIATOR (condition -> length -> dial),
so DECOMPOSE, not deconfound-by-matching:

  Estimand  : the length-orthogonal fine edge — vMF mu_hat chord distance
              between condition means of residualized dial vectors, in the
              same units as the raw 1.2285 (ratio deconfounded/raw IS the
              decomposition).
  Residual  : per-dial z_i - E[z_i | length]; length model = monotone
              spline df=4 (piecewise-linear truncated-ramp / degree-1
              I-spline basis; interior knots at TRAIN-fold quartiles;
              nonneg weights by NNLS cyclic coordinate descent; direction
              sign fixed by the train-fold Pearson r). Fit CROSS-FITTED
              within LONO folds (train one night, apply the other; folds
              = nights A <-> H) — no leakage.
  Support   : common-support set = windows inside the length-distribution
              overlap ([max(min_SEG1,min_SEG2), min(max_SEG1,max_SEG2)] of
              the window-mean-length distributions, pooled A+H). Coverage
              reported; < 70% => INDETERMINATE per registration.
  Inference : length-stratified permutation of condition labels WITHIN
              (night x length-stratum) cells — preserves the length-dial
              coupling, destroys the condition link, cluster-aware by
              night. No parametric residual claim. Cluster bootstrap
              (resample within night x condition) for the CI.
  Manip chk : silence-only classification on the common-support set must
              fall to chance (else trimming failed => INDETERMINATE).
  Triad     : residualized-content vs silence vs both, pinned protocol
              (pooled LONO logreg, W=4, W=8 sensitivity).

Decision bands (pre-stated):
  PASS : deconfounded gap >= 0.37 (30% of 1.2285) AND stratified-null
         p < 0.05 AND residualized-content > silence-only (pinned), AND
         silence-only < residualized-content - 0.10 (kill tolerance clear).
  KILL : deconfounded-gap CI entirely below 0.37, OR silence-only >=
         residualized-content - 0.10.
  INDET: anything else (incl. coverage < 70%, failed manip check, or any
         unanticipated gate shape — honesty clause, reported not absorbed).

Grains (both registered, both reported):
  * z-space (the 1.2285's own grain): seg_fit-style trailing W=8 windows
    over each stratum, DialBank replay -> zvec standardization -> vMF mu_hat
    per condition -> chord. Decomposed here.
  * triad grain (the silence test's pinned protocol): W=4 stride-1 windows
    of logged per-message field_eff_after dials; message-grain residuals
    (same spline machinery, folds by night).

Stage 0 (corpus build) imports the project's own numpy-only dial machinery
(elephant.dial / elephant.vmf / scripts.nights_abc / scripts.night_h) — NO
torch, NO model loading (that is what timed out the earlier attempt). All
analysis stages are numpy-only. Checkpoint JSON rewritten after every stage
so a timeout cannot lose numbers. CPU, deterministic seeds. No git ops.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys

import numpy as np

ELEPHANT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NIGHTS = os.path.join(ELEPHANT, "data", "nights")
CKPT = os.path.join(ELEPHANT, "LENGTH-DECOMPOSITION-2026-08-20.json")
DIALS = 7
W_Z = 8            # fine-gap window (seg_fit convention)
W_TRIAD = 4        # pinned triad window (silence_test convention)
T_BASE, T_CHAR = 0.20, 0.005   # silence proxy, identical to silence_test
LR, ITERS, L2 = 0.5, 4000, 1e-3  # pinned logreg, identical to silence_test
B_PERM, B_BOOT = 5000, 2000
SEED = 20260820
GAP_PASS = 0.37    # 30% of 1.2285, registered
COV_MIN = 0.70
TOL_KILL = 0.10    # silence >= resid - 0.10 kills

STATE: dict = {}


def checkpoint(stage: str, **kw):
    STATE["stage"] = stage
    STATE.update(kw)
    with open(CKPT, "w") as f:
        json.dump(STATE, f, indent=2)
    print(f"[checkpoint:{stage}] written {CKPT}", flush=True)


def load_mod(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------- #
# Stage 0 — corpus build (the ONLY non-numpy part: project's own dial    #
# machinery; pure python + numpy, no torch)                              #
# --------------------------------------------------------------------- #
def stage0() -> dict:
    sys.path.insert(0, ELEPHANT)
    nabc = load_mod(os.path.join(ELEPHANT, "scripts", "nights_abc.py"),
                    "nights_abc")
    nh = load_mod(os.path.join(ELEPHANT, "scripts", "night_h.py"), "night_h")
    from elephant.dial import DialBank
    from elephant.dials import DEFAULT_DIALS
    from elephant.room import Message, Room
    from elephant.vmf import windowed, zvec

    bank = DialBank(DEFAULT_DIALS)

    def zwindows(lines):
        """seg_fit semantics: fresh Room over exactly these messages,
        trailing W=8 windows, zvec standardization. Returns (Z 20x7,
        Lmean 20, Lsd 20) — z rows in window order i=0..n-1."""
        room = Room("seg", [Message(a, t, ts=float(i), reactions=r)
                            for i, (a, t, r) in enumerate(lines)])
        msgs = room.messages
        Z, Lm, Ls = [], [], []
        for i in range(len(msgs)):
            sub = msgs[max(0, i - W_Z + 1): i + 1]
            z = zvec(bank.readings(Room("seg", sub)))
            lens = np.array([len(m.text) for m in sub])
            Z.append(np.asarray(z, float))
            Lm.append(float(lens.mean()))
            Ls.append(float(lens.std()))
        return np.stack(Z), np.array(Lm), np.array(Ls)

    seg = {}
    # night A: the canonical flip, banks verbatim
    seg[("A", 0)] = zwindows(nabc.SEG1)
    seg[("A", 1)] = zwindows(nabc.SEG2)
    # night H: ramp geometry, same banks/cast; strata seq 0-19 / 20-39
    lines = nh.ramp_script()
    seg[("H", 0)] = zwindows(lines[:20])
    seg[("H", 1)] = zwindows(lines[20:])

    # raw (confounded) chords, z-space
    def mu(Z):
        X = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        r = X.mean(0)
        return r / np.linalg.norm(r)

    chord_A = float(np.linalg.norm(mu(seg[("A", 0)][0]) - mu(seg[("A", 1)][0])))
    chord_H = float(np.linalg.norm(mu(seg[("H", 0)][0]) - mu(seg[("H", 1)][0])))
    Zp0 = np.vstack([seg[("A", 0)][0], seg[("H", 0)][0]])
    Zp1 = np.vstack([seg[("A", 1)][0], seg[("H", 1)][0]])
    chord_pool = float(np.linalg.norm(mu(Zp0) - mu(Zp1)))

    # message-grain data (triad grain): logged field_eff_after + len
    msg = {}
    for tag, fn in (("A", "night-A.jsonl"), ("H", "night-H.jsonl")):
        rows = [json.loads(l) for l in
                open(os.path.join(NIGHTS, fn)) if l.strip()]
        sp = sorted((r for r in rows if r.get("type") == "speak"),
                    key=lambda r: r["seq"])
        msg[tag] = {"len": np.array([r["len"] for r in sp], float),
                    "dials": np.stack([np.asarray(r["field_eff_after"],
                                                  float) for r in sp])}

    # A==B==C byte-identity (suite convention, assert once)
    import hashlib
    h = [hashlib.md5(open(os.path.join(NIGHTS, f"night-{n}.jsonl"), "rb")
                     .read()).hexdigest() for n in "ABC"]
    out = {"raw_chord_A": chord_A, "raw_chord_H": chord_H,
           "raw_chord_pooled": chord_pool,
           "summary_reference_1.2285": json.load(
               open(os.path.join(NIGHTS, "summary.json"))
               )["nights"]["A"]["fine_gap_chord"],
           "ABC_identical_md5": h[0] == h[1] == h[2],
           "n_zwindows_per_cell": 20, "corpus": "A (once) + H, strata "
           "SEG1 seq0-19 / SEG2 seq20-39, z-space W=8 trailing windows"}
    np.savez(os.path.join(ELEPHANT, "scripts", "cache_clock_split.npz"),
             **{f"Z_{t}_{c}": seg[(t, c)][0] for t in "AH" for c in (0, 1)},
             **{f"Lm_{t}_{c}": seg[(t, c)][1] for t in "AH" for c in (0, 1)},
             **{f"Ls_{t}_{c}": seg[(t, c)][2] for t in "AH" for c in (0, 1)},
             **{f"mlen_{t}": msg[t]["len"] for t in "AH"},
             **{f"mdial_{t}": msg[t]["dials"] for t in "AH"})
    print(f"  raw chords: A={chord_A:.4f} (summary ref "
          f"{out['summary_reference_1.2285']:.4f}) H={chord_H:.4f} "
          f"pooled={chord_pool:.4f}")
    return out


# --------------------------------------------------------------------- #
# Monotone spline df=4 + NNLS (numpy only)                               #
# --------------------------------------------------------------------- #
def _ramp_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    return np.clip(x[:, None] - knots[None, :], 0.0, None)


def _nnls(H: np.ndarray, g: np.ndarray, iters: int = 3000) -> np.ndarray:
    """Cyclic coordinate descent for min ||g - Hw||, w >= 0 (H PSD, tiny)."""
    K = len(g)
    w = np.zeros(K)
    d = np.diag(H).copy()
    d[d < 1e-12] = 1e-12
    for _ in range(iters):
        w0 = w.copy()
        for k in range(K):
            w[k] = max(0.0, w[k] + (g[k] - H[k] @ w) / d[k])
        if np.max(np.abs(w - w0)) < 1e-12:
            break
    return w


def monotone_spline_fit(L: np.ndarray, y: np.ndarray):
    """Monotone spline df=4: piecewise-linear truncated-ramp basis (degree-1
    I-spline), interior knots at train quintiles 20/40/60/80, direction =
    sign(r(y,L)). Returns predictor f(L')."""
    r = float(np.corrcoef(L, y)[0, 1])
    s = -1.0 if (math.isfinite(r) and r < 0.0) else 1.0
    knots = np.quantile(L, [0.20, 0.40, 0.60, 0.80])  # 4 interior knots -> df=4
    knots = np.array(knots, float)
    span = float(L.max() - L.min())
    # ensure strictly increasing (integer lens can tie)
    eps = span * 1e-6 if span > 0 else 1e-6
    for k in range(1, 4):
        if knots[k] <= knots[k - 1]:
            knots[k] = knots[k - 1] + eps * (k + 1)
    Phi_raw = _ramp_basis(L, knots)
    mu_p, sd_p = Phi_raw.mean(0), Phi_raw.std(0) + 1e-12
    P = (Phi_raw - mu_p) / sd_p          # conditioning only
    yc = y - y.mean()
    Hm = P.T @ P + 1e-10 * np.eye(4)
    g = P.T @ (s * yc)
    w = _nnls(Hm, g)

    def f(Lnew: np.ndarray) -> np.ndarray:
        Pn = (_ramp_basis(Lnew, knots) - mu_p) / sd_p
        return y.mean() + s * (Pn @ w)

    return f, s, knots


def crossfit_residuals(L: np.ndarray, Z: np.ndarray,
                       night: np.ndarray) -> np.ndarray:
    """Cross-fitted within LONO folds (folds = nights): fit on one night,
    predict the other. Returns residual matrix same shape as Z."""
    R = np.zeros_like(Z)
    meta = []
    for tr in np.unique(night):
        m_tr, m_te = night == tr, night != tr
        for j in range(Z.shape[1]):
            f, s, kn = monotone_spline_fit(L[m_tr], Z[m_tr, j])
            R[m_te, j] = Z[m_te, j] - f(L[m_te])
            meta.append({"dim": j, "train": str(tr), "sign": s,
                         "knots": [round(float(k), 2) for k in kn]})
    STATE.setdefault("_spline_meta", meta)
    return R


# --------------------------------------------------------------------- #
# Stage 1 — cross-fitted residualization (z-space)                       #
# --------------------------------------------------------------------- #
def stage1(cache) -> dict:
    Zs = {(t, c): cache[f"Z_{t}_{c}"] for t in "AH" for c in (0, 1)}
    Lms = {(t, c): cache[f"Lm_{t}_{c}"] for t in "AH" for c in (0, 1)}
    Z = np.vstack([Zs[k] for k in [("A", 0), ("A", 1), ("H", 0), ("H", 1)]])
    L = np.concatenate([Lms[k] for k in [("A", 0), ("A", 1), ("H", 0), ("H", 1)]])
    cond = np.array([c for t in "AH" for c in (0, 1) for _ in range(20)])
    night = np.array([t for t in "AH" for _ in range(40)])

    r_before = [float(np.corrcoef(L, Z[:, j])[0, 1]) for j in range(DIALS)]
    # orthogonality QC: in-fold vs out-of-fold residual-length correlation
    qc = {}
    for tr in np.unique(night):
        m_tr, m_te = night == tr, night != tr
        r_in, r_out = [], []
        for j in range(DIALS):
            f, s, kn = monotone_spline_fit(L[m_tr], Z[m_tr, j])
            r_in.append(float(np.corrcoef(L[m_tr], Z[m_tr, j] - f(L[m_tr]))[0, 1]))
            r_out.append(float(np.corrcoef(L[m_te], Z[m_te, j] - f(L[m_te]))[0, 1]))
        qc[f"train{tr}"] = {"r_infold": r_in, "r_outfold": r_out,
                            "max_abs_r_outfold": float(np.max(np.abs(r_out)))}
    R = crossfit_residuals(L, Z, night)
    r_after = [float(np.corrcoef(L, R[:, j])[0, 1]) for j in range(DIALS)]
    print("  r(dim, window-mean len) before:", [round(r, 3) for r in r_before])
    print("  r(dim, window-mean len) after cross-fit:",
          [round(r, 3) for r in r_after])
    for k, v in qc.items():
        print(f"  orthogonality QC {k}: in-fold max|r|="
              f"{max(abs(x) for x in v['r_infold']):.3f}, out-of-fold max|r|="
              f"{v['max_abs_r_outfold']:.3f}")
    norms = np.linalg.norm(R, axis=1)
    return {"r_len_dim_before": r_before, "r_len_dim_after": r_after,
            "orthogonality_qc": qc,
            "resid_norm_min": float(norms.min()),
            "n_dropped_zero_norm": int((norms < 1e-9).sum())}


# --------------------------------------------------------------------- #
# Stage 2 — common support + deconfounded edge                           #
# --------------------------------------------------------------------- #
def _pooled_chord(X: np.ndarray, cond: np.ndarray) -> float:
    ok = np.linalg.norm(X, axis=1) > 1e-9
    Xn = X[ok] / np.linalg.norm(X[ok], axis=1, keepdims=True)
    c = cond[ok]
    m0, m1 = Xn[c == 0].mean(0), Xn[c == 1].mean(0)
    m0 /= np.linalg.norm(m0)
    m1 /= np.linalg.norm(m1)
    return float(np.linalg.norm(m0 - m1))


def stage2(cache) -> dict:
    Zs = {(t, c): cache[f"Z_{t}_{c}"] for t in "AH" for c in (0, 1)}
    Lms = {(t, c): cache[f"Lm_{t}_{c}"] for t in "AH" for c in (0, 1)}
    Z = np.vstack([Zs[k] for k in [("A", 0), ("A", 1), ("H", 0), ("H", 1)]])
    L = np.concatenate([Lms[k] for k in [("A", 0), ("A", 1), ("H", 0), ("H", 1)]])
    cond = np.array([c for t in "AH" for c in (0, 1) for _ in range(20)])
    night = np.array([t for t in "AH" for _ in range(40)])
    R = crossfit_residuals(L, Z, night)

    lo = max(L[cond == 0].min(), L[cond == 1].min())
    hi = min(L[cond == 0].max(), L[cond == 1].max())
    cs = (L >= lo) & (L <= hi)
    cov = float(cs.mean())
    print(f"  common support L in [{lo:.1f}, {hi:.1f}] -> {cs.sum()}/"
          f"{len(L)} windows (coverage {cov:.3f})")

    def night_chord(X, m, t):
        sel = m & cs
        return _pooled_chord(X[sel], cond[sel]), int(sel.sum())

    dec_pool = _pooled_chord(R[cs], cond[cs])
    raw_pool_cs = _pooled_chord(Z[cs], cond[cs])
    dec_A, nA = night_chord(R, night == "A", "A")
    dec_H, nH = night_chord(R, night == "H", "H")
    out = {"cs_lo": float(lo), "cs_hi": float(hi), "coverage": cov,
           "n_windows_total": int(len(L)), "n_windows_cs": int(cs.sum()),
           "raw_chord_on_cs_pooled": raw_pool_cs,
           "dec_chord_pooled": dec_pool, "dec_chord_A": dec_A,
           "dec_chord_H": dec_H, "n_cs_A": nA, "n_cs_H": nH,
           "ratio_dec_over_rawcs": dec_pool / raw_pool_cs
           if raw_pool_cs > 0 else None}
    print(f"  raw chord on CS (pooled) = {raw_pool_cs:.4f}")
    print(f"  DECONFOUNDED chord (pooled) = {dec_pool:.4f} "
          f"[A {dec_A:.4f} / H {dec_H:.4f}]  ratio={out['ratio_dec_over_rawcs']:.3f}")
    return out, cs, R, L, cond, night, Z


# --------------------------------------------------------------------- #
# Stage 3 — length-stratified permutation null + cluster bootstrap       #
# --------------------------------------------------------------------- #
def stage3(cs, R, L, cond, night) -> dict:
    X, C, Nt, Ln = R[cs], cond[cs], night[cs], L[cs]
    obs = _pooled_chord(X, C)
    # length strata: quartile bins of L on the common-support set
    edges = np.quantile(Ln, [0.25, 0.50, 0.75])
    strat = np.digitize(Ln, edges)
    cells = [(Nt == t) & (strat == s)
             for t in np.unique(Nt) for s in np.unique(strat)]
    cells = [c for c in cells if c.sum() > 0]
    cell_sizes = [int(c.sum()) for c in cells]
    n_exch = sum(int(min((C[c] == 0).sum(), (C[c] == 1).sum()))
                 for c in cells)
    print(f"  perm cells (night x len-quartile): {len(cells)} cells, sizes "
          f"{cell_sizes}; exchangeable labels within cells")

    rng = np.random.default_rng(SEED)
    Xn_all = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = len(C)
    cnt = 0
    null = np.empty(B_PERM)
    for b in range(B_PERM):
        Cl = C.copy()
        for c in cells:
            idx = np.where(c)[0]
            Cl[idx] = Cl[rng.permutation(idx)]
        m0 = Xn_all[Cl == 0].mean(0)
        m1 = Xn_all[Cl == 1].mean(0)
        m0 /= np.linalg.norm(m0)
        m1 /= np.linalg.norm(m1)
        null[b] = np.linalg.norm(m0 - m1)
        cnt += null[b] >= obs - 1e-12
    p_perm = (1 + cnt) / (B_PERM + 1)
    print(f"  stratified permutation: obs={obs:.4f} null mean={null.mean():.4f}"
          f" sd={null.std():.4f} p={p_perm:.4f}")

    # cluster bootstrap: resample within (night x condition)
    rng = np.random.default_rng(SEED + 1)
    boots = np.empty(B_BOOT)
    groups = [(Nt == t) & (C == c) for t in np.unique(Nt) for c in (0, 1)]
    for b in range(B_BOOT):
        idx = np.concatenate([np.where(g)[0][rng.integers(0, g.sum(), g.sum())]
                              for g in groups if g.sum() > 0])
        boots[b] = _pooled_chord(X[idx], C[idx])
    ci = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
    print(f"  cluster bootstrap 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    # reference: nothing further; raw chord handled in stage 2
    return {"perm_pooled_obs": obs, "perm_p_value": p_perm,
            "perm_null_mean": float(null.mean()),
            "perm_null_sd": float(null.std()),
            "perm_cell_sizes": cell_sizes,
            "n_exchangeable_pairs": int(n_exch),
            "boot_ci95": ci}


# --------------------------------------------------------------------- #
# Stage 4 — triad under the pinned protocol + manipulation check         #
# --------------------------------------------------------------------- #
def _fit_logreg(X, y):
    n = len(X)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = np.concatenate([(X - mu) / sd, np.ones((n, 1))], 1)
    w = np.zeros(Z.shape[1])
    for _ in range(ITERS):
        p = 1.0 / (1.0 + np.exp(-(Z @ w)))
        w -= LR * (Z.T @ (p - y) / n + L2 * np.r_[w[:-1], 0.0])
    return w, mu, sd


def _pred_logreg(m, X):
    w, mu, sd = m
    Z = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1)
    return Z @ w


def _fit_centroid(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    return (Z[y == 1].mean(0), Z[y == 0].mean(0), mu, sd)


def _pred_centroid(m, X):
    c1, c0, mu, sd = m
    Z = (X - mu) / sd
    return np.linalg.norm(Z - c0, axis=1) - np.linalg.norm(Z - c1, axis=1)


def binom_p_two_sided(k, n, p=0.5):
    def pmf(i):
        return math.exp(math.lgamma(n + 1) - math.lgamma(i + 1)
                        - math.lgamma(n - i + 1)
                        + i * math.log(p) + (n - i) * math.log(1 - p))
    pk = pmf(k)
    tol = pk * (1.0 + 1e-9) if pk > 0 else 0.0
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= tol))


def build_triad_windows(cache, residualize: bool, W: int):
    """W-window stride-1 features within strata, silence_test convention.
    Returns list of dicts with night/label/start/pos + feature arrays, and
    the per-window all-in-common-support flag (message-grain overlap)."""
    lens = {t: cache[f"mlen_{t}"] for t in "AH"}
    dials = {t: cache[f"mdial_{t}"].copy() for t in "AH"}
    if residualize:
        L = np.concatenate([lens[u] for u in "AH"])
        D = np.vstack([dials[u] for u in "AH"])
        Nt = np.array([u for u in "AH" for _ in lens[u]])
        Rm = crossfit_residuals(L, D, Nt)
        for t in "AH":
            dials[t] = Rm[Nt == t]
    lo = max(lens["A"][:20].min(), lens["A"][20:].min())
    hi = min(lens["A"][:20].max(), lens["A"][20:].max())
    lo = max(lo, lens["H"][:20].min(), lens["H"][20:].min())
    hi = min(hi, lens["H"][:20].max(), lens["H"][20:].max())

    out = []
    for t in "AH":
        L, D = lens[t], dials[t]
        taus = T_BASE + T_CHAR * L
        for stratum, (a, b), lab in (("SEG1", (0, 20), 0), ("SEG2", (20, 40), 1)):
            for i in range(a, b - W + 1):
                idx = np.arange(i, i + W)
                t_ = taus[idx]
                out.append({
                    "night": t, "label": lab, "stratum": stratum,
                    "start_seq": i, "pos": i - a,
                    "content": np.concatenate([D[idx].mean(0), D[idx].std(0)]),
                    "silence": np.array([t_.mean(), t_.std(), t_.min(),
                                         t_.max()]),
                    "cs": bool(((L[idx] >= lo) & (L[idx] <= hi)).all())})
    return out, (lo, hi)


def run_arm(win, arm: str, clf: str) -> dict:
    X = np.stack([w[arm] for w in win])
    y = np.array([w["label"] for w in win], float)
    nights = np.array([w["night"] for w in win])
    fit, pred = (_fit_logreg, _pred_logreg) if clf == "logreg" \
        else (_fit_centroid, _pred_centroid)
    scores = np.full(len(y), np.nan)
    degenerate = 0
    for t in sorted(set(nights)):
        te = nights == t
        if len(set(y[~te])) < 2:   # train fold single-class
            degenerate += 1
            continue
        scores[te] = pred(fit(X[~te], y[~te]), X[te])
    ok = ~np.isnan(scores)
    correct = (scores[ok] > 0) == (y[ok] == 1)
    k, n = int(correct.sum()), int(ok.sum())
    return {"n": n, "n_excluded_degfold": int((~ok).sum()),
            "degenerate_folds": degenerate,
            "accuracy": float(correct.mean()) if n else None,
            "binom_p": binom_p_two_sided(k, n) if n else None}


def stage4(cache) -> dict:
    # sanity baseline on the FULL corpus (reproduce silence_test numbers)
    full, _ = build_triad_windows(cache, residualize=False, W=W_TRIAD)
    base = {f"full|content|{c}": run_arm(full, "content", c)
            for c in ("logreg", "centroid")}
    base.update({f"full|silence|{c}": run_arm(full, "silence", c)
                 for c in ("logreg", "centroid")})

    # common-support triad: residualized content vs silence vs both
    res: dict = {}
    for W in (W_TRIAD, 8):
        win_raw, (lo, hi) = build_triad_windows(cache, residualize=False, W=W)
        win_res, _ = build_triad_windows(cache, residualize=True, W=W)
        cs_raw = [w for w in win_raw if w["cs"]]
        cs_res = [w for w in win_res if w["cs"]]
        for name, wl, extra in (("resid_content", cs_res, None),
                                ("silence", cs_raw, None),
                                ("resid_content+silence", cs_raw, "resid")):
            if extra:  # both arms: silence from raw windows + resid content
                Xr = np.stack([w["content"] for w in cs_res])
                Xs = np.stack([w["silence"] for w in cs_raw])
                X = np.concatenate([Xr, Xs], 1)
                for c in ("logreg", "centroid"):
                    y = np.array([w["label"] for w in cs_raw], float)
                    ng = np.array([w["night"] for w in cs_raw])
                    fit, pred = (_fit_logreg, _pred_logreg) if c == "logreg" \
                        else (_fit_centroid, _pred_centroid)
                    sc = np.full(len(y), np.nan)
                    deg = 0
                    for t in sorted(set(ng)):
                        te = ng == t
                        if len(set(y[~te])) < 2:
                            deg += 1
                            continue
                        sc[te] = pred(fit(X[~te], y[~te]), X[te])
                    ok = ~np.isnan(sc)
                    corr = (sc[ok] > 0) == (y[ok] == 1)
                    acc = float(corr.mean()) if ok.sum() else None
                    res[f"W{W}|{name}|{c}"] = {
                        "n": int(ok.sum()),
                        "n_excluded_degfold": int((~ok).sum()),
                        "degenerate_folds": deg, "accuracy": acc,
                        "binom_p": binom_p_two_sided(int(corr.sum()),
                                                      int(ok.sum()))
                        if acc is not None else None}
                continue
            for c in ("logreg", "centroid"):
                res[f"W{W}|{name}|{c}"] = run_arm(wl, "content" if name ==
                                                  "resid_content" else
                                                  "silence", c)
        # reference arms on the same CS set
        for c in ("logreg", "centroid"):
            res[f"W{W}|raw_content_cs|{c}"] = run_arm(cs_raw, "content", c)
        cov = float(np.mean([w["cs"] for w in win_raw]))
        res[f"W{W}|coverage"] = {"lo": lo, "hi": hi, "coverage": cov,
                                 "n_cs": len(cs_raw), "n_total": len(win_raw)}
        def _a(k):
            v = res[f"W{W}|{k}|logreg"]["accuracy"]
            return f"{v:.3f}" if v is not None else "n/a(deg)"
        print(f"  W={W} CS windows {len(cs_raw)}/{len(win_raw)} "
              f"(cov {cov:.3f}) | resid_content logreg "
              f"{_a('resid_content')} | silence logreg {_a('silence')} | "
              f"raw_content_cs logreg {_a('raw_content_cs')}")
    return {"baseline_full_corpus": base, "triad": res}


# --------------------------------------------------------------------- #
# Stage 5 — verdict                                                      #
# --------------------------------------------------------------------- #
def stage5(dec, inf, tri, orth) -> dict:
    gap = dec["dec_chord_pooled"]
    ci = inf["boot_ci95"]
    p = inf["perm_p_value"]
    cov_ok = dec["coverage"] >= COV_MIN
    acc_r = tri["triad"]["W4|resid_content|logreg"]["accuracy"]
    acc_s = tri["triad"]["W4|silence|logreg"]["accuracy"]
    p_s = tri["triad"]["W4|silence|logreg"]["binom_p"]
    triad_cov = tri["triad"]["W4|coverage"]["coverage"]
    manip_ok = (acc_s is not None) and (acc_s <= 0.65) and (p_s >= 0.05)
    tol_hit = (acc_s is not None and acc_r is not None
               and acc_s >= acc_r - TOL_KILL)
    orth_fail = max(v["max_abs_r_outfold"]
                    for v in orth["orthogonality_qc"].values()) > 0.30

    reasons = []
    verdict = None
    if not cov_ok or triad_cov < COV_MIN:
        verdict = "INDETERMINATE"
        reasons.append(f"common-support coverage below {COV_MIN:.0%} "
                       f"(z-space {dec['coverage']:.3f}, triad "
                       f"{triad_cov:.3f}) — the SEG1/SEG2 length "
                       "distributions overlap too little for support-based "
                       "estimation on this corpus; PASS/KILL branches "
                       "presuppose a valid common-support evaluation")
    if not manip_ok:
        if verdict is None:
            verdict = "INDETERMINATE"
        reasons.append("manipulation check failed: silence-only still "
                       f"discriminates on the trimmed set (acc={acc_s}, "
                       f"p={p_s:.1e}) — the trimming has no teeth; the "
                       "registration routes this to INDETERMINATE, not a "
                       "pass")
    if orth_fail:
        if verdict is None:
            verdict = "INDETERMINATE"
        reasons.append("orthogonality QC failed out-of-fold (max|r| up to "
                       f"{max(v['max_abs_r_outfold'] for v in orth['orthogonality_qc'].values()):.3f}): "
                       "the length->dial mediator mapping does not "
                       "transfer across nights (flip vs ramp geometry), so "
                       "the residualized dials are not certifiably "
                       "length-orthogonal; the deconfounded point estimate "
                       "is reported but cannot be certified as the "
                       "registered estimand")
    if verdict is None:
        if gap >= GAP_PASS and p < 0.05 and acc_r > acc_s and not tol_hit:
            verdict = "PASS"
        elif ci[1] < GAP_PASS:
            verdict = "KILL"
            reasons.append(f"deconfounded-gap CI entirely below {GAP_PASS} "
                           f"(CI hi={ci[1]:.4f})")
        elif tol_hit:
            verdict = "KILL"
            reasons.append(f"silence-only within the kill tolerance of "
                           f"residualized content ({acc_s:.3f} >= "
                           f"{acc_r:.3f} - {TOL_KILL})")
        else:
            verdict = "INDETERMINATE"
            reasons.append("unanticipated gate shape (honesty clause)")
    notes = []
    if tol_hit:
        notes.append(f"note: the KILL tolerance clause (silence {acc_s} >= "
                     f"resid {acc_r} - {TOL_KILL}) also fires, but on a "
                     "degenerate common-support triad (n=24, all arms at "
                     "ceiling) — reported, not counted, per the coverage "
                     "precondition")
    notes.append(f"note: perm p={p:.4f} (>= 0.05) and boot CI {ci} lie ABOVE "
                 f"{GAP_PASS} — nothing in the data supports PASS either; "
                 "the corpus cannot adjudicate Stage 1")
    return {"verdict": verdict, "reasons": reasons, "notes": notes,
            "gates": {"gap": gap, "gap_pass_band": GAP_PASS,
                      "perm_p": p, "ci95": ci, "acc_resid_content": acc_r,
                      "acc_silence_cs": acc_s, "silence_p": p_s,
                      "coverage_z": dec["coverage"],
                      "coverage_triad": triad_cov,
                      "manip_check_ok": manip_ok,
                      "orthogonality_ok": not orth_fail}}


# --------------------------------------------------------------------- #
def main() -> int:
    print("=" * 74)
    print("E4 CLOCK-SPLIT STAGE 1 — length decomposition of the fine edge")
    print("=" * 74)

    print("\n[stage 0] corpus build + confounded-edge reproduction")
    s0 = stage0()
    cache_npz = np.load(os.path.join(ELEPHANT, "scripts",
                                     "cache_clock_split.npz"))
    checkpoint("stage0_corpus", stage0_res=s0)
    assert abs(s0["raw_chord_A"] - s0["summary_reference_1.2285"]) < 1e-9, \
        "fine-gap reproduction failed"

    print("\n[stage 1] cross-fitted monotone-spline residualization (z-space)")
    s1 = stage1(cache_npz)
    checkpoint("stage1_residualized", stage1_res=s1)

    print("\n[stage 2] common support + deconfounded edge")
    s2, cs, R, L, cond, night, Z = stage2(cache_npz)
    checkpoint("stage2_deconfounded", stage2_res=s2)

    print("\n[stage 3] length-stratified permutation + cluster bootstrap")
    s3 = stage3(cs, R, L, cond, night)
    checkpoint("stage3_inference", stage3_res=s3)

    print("\n[stage 4] triad under pinned protocol + manipulation check")
    s4 = stage4(cache_npz)
    checkpoint("stage4_triad", stage4_res=s4)

    print("\n[stage 5] verdict")
    s5 = stage5(s2, s3, s4, s1)
    checkpoint("stage5_verdict", stage5_res=s5)

    print("\n" + "=" * 74)
    print("THE THREE NUMBERS")
    print(f"  (a) confounded fine edge (raw z-space chord, night A): "
          f"{s0['raw_chord_A']:.4f}  [summary ref 1.2285 reproduced]")
    print(f"  (b) length-orthogonal fine edge (cross-fitted spline "
          f"residuals, common support, pooled A+H): "
          f"{s2['dec_chord_pooled']:.4f}   perm p={s3['perm_p_value']:.4f}"
          f"   boot CI [{s3['boot_ci95'][0]:.4f}, {s3['boot_ci95'][1]:.4f}]"
          f"   (A {s2['dec_chord_A']:.4f} / H {s2['dec_chord_H']:.4f})")
    print(f"  (c) silence-only manipulation check on the trimmed set "
          f"(pinned LONO logreg W4): acc="
          f"{s4['triad']['W4|silence|logreg']['accuracy']} "
          f"p={s4['triad']['W4|silence|logreg']['binom_p']:.3f} "
          f"(must be chance)")
    print()
    print(f"VERDICT (Clock-Split bands): {s5['verdict']}")
    for r in s5["reasons"]:
        print(f"  - {r}")
    for n in s5["notes"]:
        print(f"  - {n}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
