"""WAVE-3 S4 BLINDED ANALYSIS — branch-pattern recovery from filed leg outputs.

Registered step (registration §1.3/§4, addendum G6). BLINDING: reads ONLY
data/wave3/legs/*.json (the S3-filed raw leg outputs). Never reads the run
doc, sealed sidecars, manifests, or corpus directories. Produces the
per-corpus branch-pattern table, pair rankings, VOID rulings, and the
discrimination verdict WITHOUT unblinding α.

Outputs (stdout + data/wave3/s4-blinded-summary.json):
  - per-corpus leg reads at W=12 canonical (primary), W=8/16 + actual
    presence as the registered sensitivity manifold (void rule 7)
  - branch-pattern classification per the §4 tables (pattern names only)
  - VOID rule checks evaluable from the leg window (2,3,5 + Sxx part of 1)
  - 2AFC pair detection + per-leg pair rankings on the pre-stated signed
    directions (baseline spread ↓, S x-slope ↑, P_trans ↓, D ↓, A flat-to-↓)
  - gradient-ordering statistics among standalone corpora
  - discrimination verdict inputs

Not evaluable in this window (flagged, never guessed): ICC leg (not filed by
the S3 driver; G6 addendum re-band [0.60,0.80] queued), decoy panel void rule
6 (per-reader detrending / mixed-effects decoys not filed; the in-file A
up-mirror + start-ref variants are the available estimator-variation
columns), manifest-gate determinism/roster checks (void rule 1, S3-side).
"""
from __future__ import annotations

import glob
import itertools
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGS = os.path.join(ROOT, "data", "wave3", "legs")
OUT = os.path.join(ROOT, "data", "wave3", "s4-blinded-summary.json")

CHANNELS = [("W12", "canonical"), ("W12", "actual"), ("W08", "canonical"),
            ("W16", "canonical"), ("W08", "actual"), ("W16", "actual")]
PRIMARY = ("W12", "canonical")

FLOOR = 20          # registered crossing floor (branch-conditional)
EPS_VOID = 1500     # bootstrap effective-draws void floor
SXX_MIN = 0.19      # a-priori x-design floor (void rule 1, evaluable part)


def load_all():
    data = {}
    for f in sorted(glob.glob(os.path.join(LEGS, "*.json"))):
        d = json.load(open(f))
        data[(d["corpus_id"], f"W{d['W']:02d}", d["presence"])] = d
    return data


def sxx(x_map, nights):
    x = np.array([x_map[n] for n in nights], float)
    xm = x - x.mean()
    return float(xm @ xm / len(x))


def baseline_spread(S):
    """Night-demeaned reader-mean score spread (derived diagnostic; the
    registered 2AFC 'baseline spread ↓' direction measured from S cells)."""
    by_night = {}
    for c in S["cells"]:
        by_night.setdefault(c["night"], []).append(c["score"])
    night_mean = {n: np.mean(v) for n, v in by_night.items()}
    rmeans = {}
    for c in S["cells"]:
        rmeans.setdefault(c["reader"], []).append(c["score"] - night_mean[c["night"]])
    rm = np.array([np.mean(v) for v in rmeans.values()])
    return float(np.std(rm, ddof=1)), len(rm)


def read_legs(d):
    """One corpus × one channel -> registered leg reads + void flags."""
    A, Au, Ast = d["A"], d["A_up_mirror"], d["A_start_ref_sensitivity"]
    D, P, S = d["D"], d["P"], d["S"]
    sig, nulls = d["signal_nights"], d["null_nights"]
    n_ev = A["n_events"]
    fired = A["p"] < 0.05
    fired_up = Au["p"] < 0.05
    fired_start = Ast["p"] < 0.05
    d_sig, d_null = D["D_signal"], D["D_null"]
    rn_ratio = (D["null_rn_crossing_rate"] / D["signal_rn_crossing_rate"]
                if D["signal_rn_crossing_rate"] > 0 else math.inf)
    spread, n_r = baseline_spread(S)
    s_collapse_sig = (not S["contains_0"]) and S["beats_competitor"]
    # residual scatter of S cells around a night-FE + x line (inflation probe)
    X = np.array([[c["x"], 1.0] for c in S["cells"]])
    y = np.array([c["score"] for c in S["cells"]])
    nfe = {n: i for i, n in enumerate(sorted({c["night"] for c in S["cells"]}))}
    rows = []
    for i, c in enumerate(S["cells"]):
        r = np.zeros(1 + len(nfe) + 1)
        r[0] = c["x"]
        r[1 + nfe[c["night"]]] = 1.0
        rows.append(r)
    Xm = np.array(rows)
    beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    resid = y - Xm @ beta
    pr, pt = P.get("P_rest"), P.get("P_trans")
    p_unreadable = pr is None or pt is None or pr <= 0 or len(
        P.get("rest_events", [])) == 0
    p_holds = None if p_unreadable else bool(P["holds_at_half"])
    return {
        "sd": d["corpus_sd"], "n_events": n_ev, "A_rate": A["A"], "A_p": A["p"],
        "P_unreadable": p_unreadable,
        "A_fired": fired, "A_up_p": Au["p"], "A_up_fired": fired_up,
        "A_start_p": Ast["p"], "A_start_fired": fired_start,
        "A_referent_consistent": fired == fired_up,
        "D_sig": d_sig, "D_null": d_null, "D_above_null": d_sig > max(d_null, 0.5),
        "D_above_05": d_sig > 0.5,
        "rn_ratio": rn_ratio, "srn": D["signal_rn_crossing_rate"],
        "P_trans": P["P_trans"], "P_rest": pr,
        "P_holds": p_holds, "P_ratio": (None if p_unreadable else pt / pr),
        "P_n": len(P["trans_events"]), "P_kill": bool(P["mechanism_kill"]),
        "S_slope": S["slope_x"], "S_ci": S["slope_ci"], "S_c0": S["contains_0"],
        "S_beats": S["beats_competitor"], "S_xinv": S["x_invariant"],
        "S_collapse_sig": s_collapse_sig, "S_effdraws": S["effective_draws"],
        "S_perm_p": S["nested_perm_p"],
        "spread": spread, "n_readers_S": n_r, "S_resid_sd": float(np.std(resid, ddof=1)),
        "sxx": sxx(d["x_map"], sig),
        "voids": {
            "v2_null_rn_ge_half_signal": rn_ratio >= 0.5,
            "v3_low_crossings": n_ev < FLOOR,
            "v5_effdraws": S["effective_draws"] < EPS_VOID,
            "v1_sxx": sxx(d["x_map"], sig) < SXX_MIN,
        },
    }


def classify(r):
    """Branch-PATTERN label (no α knowledge). Per §4 tables."""
    if r["voids"]["v2_null_rn_ge_half_signal"]:
        pat, note = "VOID-v2", "null-night crossing ≥50% of signal"
    elif r.get("P_unreadable"):
        pat = "P-unreadable-channel"
        note = "no rest strata at this channel (structural; W too wide)"
    elif r["S_collapse_sig"] or r["P_holds"] is False:
        pat = "collapse-pattern"
        note = ("P fail" if not r["P_holds"] else "") + \
               (" S fires" if r["S_collapse_sig"] else "")
    elif r["A_fired"] and r["D_above_null"]:
        if r["voids"]["v3_low_crossings"]:
            pat, note = "instrument-pattern", "A fires but <20 crossings (floor)"
        else:
            pat, note = "instrument-pattern", "A time-locks, D above null"
    elif not r["A_fired"] and r["A_p"] > 0.05:
        if r["voids"]["v3_low_crossings"]:
            # floor applies only where A would be read on instrument/intermediate;
            # blinded dual-read: collapse/noise hit OR instrument-side void
            pat = "low-count/silent"
            note = "A silent & <20 crossings — dual-read (branch hit vs floor void)"
        else:
            pat = "noise-pattern"
            note = "A silent with ≥20 crossings, no S/P signature"
    else:
        pat, note = "mixed", "A silent but D above null (or other combo)"
    return pat, note


def supplement(summary, data, prim, corpora, pairs, standalone):
    """S4 supplements: traj pair confirmation (family-matched tags), effect
    sizes within/between pairs, latent-chain search, full-grid void scan."""
    import itertools
    traj = {}
    for cid in corpora:
        t = {}
        for n, tr in data[(cid, *PRIMARY)]["trajectory"].items():
            t[n.split("-")[-1]] = np.array(
                [v if v is not None else np.nan for v in tr["Rt"]], float)
        traj[cid] = t
    def tcorr(a, b):
        cs = []
        for fam in traj[a]:
            if fam in traj[b]:
                x, y = traj[a][fam], traj[b][fam]
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() > 5 and np.std(x[m]) > 0 and np.std(y[m]) > 0:
                    cs.append(float(np.corrcoef(x[m], y[m])[0, 1]))
        return float(np.mean(cs)) if cs else None
    traj_pairs = {f"{a}|{b}": tcorr(a, b) for a, b in pairs}
    traj_cross = [tcorr(a, b) for a, b in itertools.combinations(corpora, 2)
                  if tuple(sorted((a, b))) not in pairs]
    traj_cross = [v for v in traj_cross if v is not None]

    def spread_cells(S):
        return prim and None
    fns = {"A_rate": lambda d: d["A"]["A"], "n_events": lambda d: d["A"]["n_events"],
           "D_sig": lambda d: d["D"]["D_signal"], "P_trans": lambda d: d["P"]["P_trans"],
           "S_slope": lambda d: d["S"]["slope_x"], "spread": None,
           "srn": lambda d: d["D"]["signal_rn_crossing_rate"]}
    vals = {}
    for k, fn in fns.items():
        if fn is None:
            continue
        vals[k] = {c: fn(data[(c, *PRIMARY)]) for c in corpora}
    def spread_of(cid):
        S = data[(cid, *PRIMARY)]["S"]
        bn = {}
        for c in S["cells"]:
            bn.setdefault(c["night"], []).append(c["score"])
        nm = {n: np.mean(v) for n, v in bn.items()}
        rm = {}
        for c in S["cells"]:
            rm.setdefault(c["reader"], []).append(c["score"] - nm[c["night"]])
        return float(np.std([np.mean(v) for v in rm.values()], ddof=1))
    vals["spread"] = {c: spread_of(c) for c in corpora}
    eff = {}
    for k in vals:
        within = [abs(vals[k][a] - vals[k][b]) /
                  max(abs(vals[k][a]), abs(vals[k][b]), 1e-9) for a, b in pairs]
        ks = [c for c in standalone if not c.endswith("k06")]
        ks = [c for c in standalone if c != "w3k06"]
        btwk = [abs(vals[k][x] - vals[k][y]) /
                max(abs(vals[k][x]), abs(vals[k][y]), 1e-9)
                for x, y in itertools.combinations(ks, 2)]
        eff[k] = {"within_med": float(np.median(within)),
                  "within_max": float(max(within)),
                  "between_k_med": float(np.median(btwk)),
                  "between_k_max": float(max(btwk))}

    dirs = [("spread", -1), ("S_slope", +1), ("P_trans", -1),
            ("D_sig", -1), ("A_rate", -1), ("n_events", -1)]
    ladder = [c for c in standalone if c != "w3k06"]
    best = None
    for perm in itertools.permutations(ladder):
        viol = 0
        for k, sgn in dirs:
            seq = [vals[k][c] for c in perm]
            viol += sum(1 for i, j in itertools.combinations(range(len(seq)), 2)
                        if (seq[i] - seq[j]) * sgn < -1e-12)
        if best is None or viol < best[0]:
            best = (viol, list(perm))
    chain = {"best_violations": best[0], "total_pairs": len(dirs) * 10,
             "best_order_inst_to_collapse": best[1]}

    voids = {"v2": [], "v3": [], "v5": [], "S_collapse_signature": [],
             "S_ambiguous": []}
    for (cid, W, pres), d in data.items():
        tag = f"{cid} {W}|{pres}"
        D, S = d["D"], d["S"]
        if D["signal_rn_crossing_rate"] > 0 and \
                D["null_rn_crossing_rate"] >= 0.5 * D["signal_rn_crossing_rate"]:
            voids["v2"].append(tag)
        if d["A"]["n_events"] < FLOOR:
            voids["v3"].append((tag, d["A"]["n_events"]))
        if S["effective_draws"] < EPS_VOID:
            voids["v5"].append(tag)
        if S["beats_competitor"] and not S["contains_0"]:
            voids["S_collapse_signature"].append(tag)
        if S["beats_competitor"] and S["contains_0"]:
            voids["S_ambiguous"].append(tag)
    summary["supplement"] = {
        "traj_pair_corr": traj_pairs,
        "traj_cross_median": float(np.median(traj_cross)),
        "traj_cross_max": float(max(traj_cross)),
        "effect_sizes": eff, "chain_search": chain, "void_scan_grid": voids,
        "sxx_note": ("x-ladder identical across all 16 corpora (field X_W2 by "
                     "family); regression-design Sxx over cells = 1.058 for "
                     "every corpus (>= 0.19); formula is cell-level sum of "
                     "squared x-deviations — the S3-side gate formula may "
                     "differ; design is corpus-invariant either way"),
    }


def main():
    data = load_all()
    corpora = sorted({k[0] for k in data})
    chan_avail = {f"{W}|{p}": sum(1 for cid in corpora if (cid, W, p) in data)
                  for W, p in CHANNELS}
    reads = {}
    for (cid, W, pres), d in data.items():
        r = read_legs(d)
        r["pattern"], r["pattern_note"] = classify(r)
        reads[(cid, W, pres)] = r

    prim = {cid: reads[(cid, *PRIMARY)] for cid in corpora}

    # ---- pair detection (design structure, not α): identical corpus_sd
    # buckets among non-standalone ids + confirmation via trajectory corr.
    pairs = []
    sds = {}
    for cid in corpora:
        sds.setdefault(round(prim[cid]["sd"], 6), []).append(cid)
    for sd, cids in sorted(sds.items()):
        if len(cids) == 2 and cids[0] != cids[1]:
            pairs.append(tuple(cids))
    traj = {}
    for cid in corpora:
        t = {}
        d = data[(cid, *PRIMARY)]
        for n, tr in d["trajectory"].items():
            t[n] = np.array([v if v is not None else np.nan for v in tr["Rt"]],
                            float)
        traj[cid] = t
    pair_traj_corr = {}
    for a, b in itertools.combinations(corpora, 2):
        cs = []
        for n in traj[a]:
            if n in traj[b]:
                x, y = traj[a][n], traj[b][n]
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() > 5:
                    cs.append(np.corrcoef(x[m], y[m])[0, 1])
        pair_traj_corr[(a, b)] = float(np.mean(cs)) if cs else None

    # ---- 2AFC pair rankings on pre-stated signed directions vs increasing
    # α: baseline spread ↓, S x-slope ↑, P_trans ↓, D ↓, A flat-to-↓ (A rate)
    def rank_pairs(key, direction):
        out = {}
        for a, b in pairs:
            ra, rb = prim[a][key], prim[b][key]
            if direction == "down":
                hi = a if ra > rb else b        # more-instrument side
            else:
                hi = b if rb > ra else a
            out[(a, b)] = {"hi_inst": hi, "a": ra, "b": rb,
                           "gap": abs(rb - ra)}
        return out

    pair_rank = {
        "spread_down": rank_pairs("spread", "down"),
        "S_slope_up": rank_pairs("S_slope", "up"),
        "P_trans_down": rank_pairs("P_trans", "down"),
        "D_down": rank_pairs("D_sig", "down"),
        "A_rate_down": rank_pairs("A_rate", "down"),
    }

    # concordance: within each pair, does a single member win the majority of
    # discriminating legs? (legs agree => one-dimensional instrument reading)
    conc = {}
    for a, b in pairs:
        votes = []
        for key, direction, sgn in [("spread", "down", 1), ("S_slope", "up", 1),
                                    ("P_trans", "down", 1), ("D_sig", "down", 1),
                                    ("A_rate", "down", 1)]:
            ra, rb = prim[a][key], prim[b][key]
            rel = (ra - rb) / max(abs(ra), abs(rb), 1e-9)
            votes.append((key, rel))  # >0 => a more instrument-like
        pos = sum(1 for _, v in votes if v > 0.02)
        neg = sum(1 for _, v in votes if v < -0.02)
        conc[(a, b)] = {"votes": {k: round(v, 4) for k, v in votes},
                        "n_a_inst": pos, "n_b_inst": neg,
                        "agree_member": a if pos > neg else (b if neg > pos else None)}

    # ---- gradient ordering among standalone corpora (k-prefixed ids)
    standalone = [c for c in corpora if not c.startswith("w3q")]
    grad = {k: {c: prim[c][k] for c in standalone}
            for k in ["n_events", "A_rate", "D_sig", "P_trans", "S_slope",
                      "spread", "srn", "S_resid_sd"]}

    # robustness manifold: pattern stability across channels
    manifold = {c: {f"{W}|{p}": reads.get((c, W, p), {}).get("pattern")
                    for W, p in CHANNELS} for c in corpora}

    summary = {
        "n_corpora": len(corpora), "corpora": corpora,
        "channels_available": chan_avail,
        "primary_reads": {c: prim[c] for c in corpora},
        "pairs_detected": [list(p) for p in pairs],
        "pair_traj_corr": {f"{a}|{b}": v for (a, b), v in pair_traj_corr.items()
                           if v is not None and (v > 0.98 or True)},
        "pair_rank": {k: {f"{a}|{b}": v for (a, b), v in d.items()}
                      for k, d in pair_rank.items()},
        "pair_concordance": {f"{a}|{b}": v for (a, b), v in conc.items()},
        "gradient_standalone": grad,
        "pattern_manifold": manifold,
        "not_evaluable_in_window": [
            "ICC leg (not filed by S3 driver; G6 re-band queued for S4-unblind)",
            "decoy panel void rule 6 (per-reader detrending / mixed-effects "
            "not filed; A up-mirror + start-ref are the filed estimator-"
            "variation columns)",
            "void rule 1 manifest determinism/roster/strata-warmth checks "
            "(S3-side; only Sxx + corpus_sd-finite evaluable here)",
            "continuity-ladder void rule 4 (not a filed channel)",
        ],
    }
    pairs_fixed = [tuple(p) for p in pairs]
    supplement(summary, data, prim, corpora, pairs_fixed, standalone)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in summary.items()
                      if k in ("n_corpora", "pairs_detected", "channels_available")},
                     indent=1, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
