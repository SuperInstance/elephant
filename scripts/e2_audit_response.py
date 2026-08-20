"""E2 audit response — devil's-audit robustness runs (addendum 3, C1/C2).

Runs AFTER addendum 3 is committed. Two pieces, both labeled:

1. LADDER ROBUSTNESS on the 11-night corpus (9 primary + S6 double-reversal
   + S7 oscillation), attendance template = FIELD_NIGHTS_EXT. NOT a new
   ladder attempt (no estimator rebuild; the passed estimator re-checked).
   Criterion per addendum C1: STRICT MONOTONICITY of recovered ratio in
   planted ratio, per estimator; ±0.1 recovery reported as secondary; bias
   curve table emitted. Planting: lambda-family [0,12] then directional
   family [0,1] (addendum 2), truth anchored on S5, canonical presence.

2. FIELD EXTENDED VARIANTS (field): (a) extended-corpus ratio over all 11
   nights; (b) non-monotonic-drift variant (drift from S6/S7 transitions
   only, estimators otherwise unchanged) with bootstrap CIs; outcome classes
   per addendum C3 (fully-inside-band => undecidable with this instrument
   class) and C2 (clear eligibility requires clearing on BOTH).

Also asserts the monotonicity of the frozen attempt-3 results (on fixtures)
for the record. All fixture numbers carry "on fixtures"; reader numbers
carry "field". Bootstrap seed 20260819.

Run:  python3 scripts/e2_audit_response.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.e2_instrument import (COLD_ENTRY, EXTENDED_NIGHTS, FIELD_NIGHTS,
                                   FIELD_NIGHTS_EXT, NONMONO_NIGHTS, Night,
                                   PRIMARY_NIGHTS, Measurement, corpus_sd,
                                   replay_readings, cell_vecs, NIGHT_SPECS,
                                   verdict)
from scripts.e2_ladder import (RUNGS, TOL, BASE_SEED, corpus_geometry,
                               draw_population, existing_personas,
                               scale_all, scale_directional, transition_vals)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "e2", "e2-audit-response-results.json")
SD_FIXED = None  # set in main from the registered 9-night corpus


# --------------------------------------------------------------------- #
# 1. ladder robustness on the extended corpus                             #
# --------------------------------------------------------------------- #
def reader_replay(params, night, slot):
    cold = night.name in COLD_ENTRY.get(slot, [])
    start = night.first_speak_seq(slot) if cold else None
    return replay_readings(params, night.speaks, slot, start,
                           canon_n=night.canon_n)


def truth_of_ext(params_by_slot, nights, sd, s5, template_map):
    anchors, drifts = [], []
    for slot, attended in template_map.items():
        p = params_by_slot[slot]
        sv = reader_replay(p, s5, slot)
        anchors.append(np.mean([v for _, v in sv], axis=0))
        vals = []
        for night in attended:
            if night in nights:
                vals.extend(transition_vals(
                    reader_replay(p, nights[night], slot), night, sd))
        assert vals, f"slot {slot} has no signal transitions"
        drifts.append(float(np.mean(vals)))
    B = np.stack(anchors)
    spread = float(np.sqrt(np.mean(B.std(axis=0, ddof=1) ** 2))) / sd
    drift = float(np.mean(drifts))
    return {"spread": spread, "drift": drift,
            "ratio": spread / drift if drift > 1e-9 else float("inf")}


def bisect_ext(scaler, drawn, cast_mean, rung, truth_fn, lam_hi,
               tol=0.005):
    grid = np.linspace(0.0, lam_hi, 25)
    vals = [truth_fn(scaler(drawn, cast_mean, x))["ratio"] for x in grid]
    if rung > vals[-1]:
        return None, None, None
    i = int(np.argmax(np.array(vals) >= rung))
    lo, hi = grid[max(i - 1, 0)], grid[i]
    for _ in range(40):
        mid = (lo + hi) / 2
        t = truth_fn(scaler(drawn, cast_mean, mid))
        if abs(t["ratio"] - rung) <= tol:
            return mid, scaler(drawn, cast_mean, mid), t
        if t["ratio"] < rung:
            lo = mid
        else:
            hi = mid
    lam = (lo + hi) / 2
    p = scaler(drawn, cast_mean, lam)
    return lam, p, truth_fn(p)


def ladder_robustness():
    nights = {n: Night(n) for n in EXTENDED_NIGHTS}
    sd, _ = corpus_sd([Night(n) for n in PRIMARY_NIGHTS])  # registered scale
    s5 = nights["S5"]
    _, cast_mean = existing_personas()
    template_map = {r: FIELD_NIGHTS_EXT[r] for r in sorted(FIELD_NIGHTS_EXT)}
    delta, Delta = corpus_geometry({k: nights[k] for k in PRIMARY_NIGHTS})
    warm_mean = np.mean([r["field_raw_after"] for r in nights["S5"].speaks[:20]],
                        axis=0)

    from scripts.e2_instrument import LO, HI

    def scaler_dir(d, cm, lam):
        return scale_directional(lam, cm, delta, Delta, warm_mean,
                                 sorted(template_map))[0]

    rows = []
    for ri, rung in enumerate(RUNGS):
        for rep in range(3):
            seed = BASE_SEED + 1000 * ri + rep
            drawn = draw_population(seed)
            if rung == 0.0:
                p = scale_all(drawn, cast_mean, 0.0)
                fam, lam = "lambda", 0.0
                t = truth_of_ext(p, nights, sd, s5, template_map)
            else:
                tfn = lambda pp: truth_of_ext(pp, nights, sd, s5, template_map)
                lam, p, t = bisect_ext(scale_all, drawn, cast_mean, rung,
                                       tfn, 12.0)
                fam = "lambda"
                if p is None:
                    lam, p, t = bisect_ext(scaler_dir, drawn, cast_mean, rung,
                                           tfn, 1.0)
                    fam = "directional"
                if p is None:
                    rows.append({"rung": rung, "rep": rep, "seed": seed,
                                 "error": "unreachable"})
                    continue
            rn = {slot: {"params": p[slot],
                         "nights": {n: slot for n in template_map[slot]},
                         "cold": COLD_ENTRY.get(slot, [])}
                  for slot in template_map}
            m = Measurement(rn, sd, include_nights=EXTENDED_NIGHTS,
                            presence="canonical")
            est_seg, est_cont = m.ratio_seg(), m.ratio_cont()
            rows.append({"rung": rung, "rep": rep, "seed": seed,
                         "family": fam, "lambda": lam,
                         "true_ratio": t["ratio"],
                         "est_seg": est_seg, "est_cont": est_cont,
                         "seg_bias": est_seg - rung,
                         "cont_bias": est_cont - rung})
            print(f"[robustness] rung {rung:.2f} rep {rep} [{fam} "
                  f"lam={lam:.3f}] truth={t['ratio']:.4f} | "
                  f"E-seg={est_seg:.4f} E-cont={est_cont:.4f}")

    def mono(vals):
        return all(b > a for a, b in zip(vals, vals[1:]))

    rep0 = [r for r in rows if r.get("rep") == 0 and "est_seg" in r]
    seg_vals = [r["est_seg"] for r in sorted(rep0, key=lambda x: x["rung"])]
    cont_vals = [r["est_cont"] for r in sorted(rep0, key=lambda x: x["rung"])]
    summary = {
        "seg_monotone": mono(seg_vals), "cont_monotone": mono(cont_vals),
        "seg_vals": seg_vals, "cont_vals": cont_vals,
        "seg_within_tol": [abs(r["seg_bias"]) <= TOL for r in
                           sorted(rep0, key=lambda x: x["rung"])],
        "cont_within_tol": [abs(r["cont_bias"]) <= TOL for r in
                            sorted(rep0, key=lambda x: x["rung"])],
        "rows": rows,
    }
    print(f"[robustness] E-seg monotone: {summary['seg_monotone']} "
          f"{[round(v,3) for v in seg_vals]}")
    print(f"[robustness] E-cont monotone: {summary['cont_monotone']} "
          f"{[round(v,3) for v in cont_vals]}")
    return summary


# --------------------------------------------------------------------- #
# 2. field extended variants                                              #
# --------------------------------------------------------------------- #
def field_variants():
    nights9 = [Night(n) for n in PRIMARY_NIGHTS]
    sd, _ = corpus_sd(nights9)   # registered scale, frozen (addendum C2)

    def readers_of(att_map):
        return {r: {"params": None, "nights": {n: r for n in ns},
                    "cold": COLD_ENTRY.get(r, [])}
                for r, ns in att_map.items()}

    m_ext = Measurement(readers_of(FIELD_NIGHTS_EXT), sd,
                        include_nights=EXTENDED_NIGHTS, presence="canonical")

    def drift_nonmono(rs):
        vals = [v for r in rs for (nt, lab, v) in m_ext.trans[r]
                if nt in NONMONO_NIGHTS]
        return float(np.mean(vals)) if vals else float("nan")

    def ratio_ext_seg(rs): return m_ext.ratio_seg(rs)
    def ratio_ext_cont(rs): return m_ext.ratio_cont(rs)
    def ratio_nonmono_seg(rs):
        d = drift_nonmono(rs)
        s = m_ext.spread_seg(rs)
        return s / d if d and not np.isnan(d) else float("nan")
    def ratio_nonmono_cont(rs):
        d = drift_nonmono(rs)
        s = m_ext.spread_cont(rs)
        return s / d if d and not np.isnan(d) else float("nan")

    rng = np.random.default_rng(20260819)
    n = len(m_ext.readers)
    draws = {k: [] for k in ("ext_seg", "ext_cont", "nonmono_seg",
                             "nonmono_cont")}
    fns = {"ext_seg": ratio_ext_seg, "ext_cont": ratio_ext_cont,
           "nonmono_seg": ratio_nonmono_seg, "nonmono_cont": ratio_nonmono_cont}
    for _ in range(2000):
        rs = [m_ext.readers[i] for i in rng.integers(0, n, n)]
        for k, fn in fns.items():
            v = fn(rs)
            if not np.isnan(v):
                draws[k].append(v)

    def ci(xs):
        return [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]

    point = {k: fn(m_ext.readers) for k, fn in fns.items()}
    cis = {k: ci(v) for k, v in draws.items()}
    icc_ext, icc_dial = m_ext.icc()

    out = {"point": point, "ci": cis, "icc_ext": icc_ext,
           "n_readers": n,
           "drift_nonmono": drift_nonmono(m_ext.readers),
           "drift_ext": m_ext.drift_mean(),
           "spread_ext_cont": m_ext.spread_cont(),
           "spread_ext_seg": m_ext.spread_seg()}
    for k in ("ext_cont", "ext_seg", "nonmono_cont", "nonmono_seg"):
        lo, hi = cis[k]
        if lo > 0.6:
            cls = "above band"
        elif hi < 0.3:
            cls = "below band"
        elif lo >= 0.3 and hi <= 0.6:
            cls = ("fully inside band -> undecidable with this instrument "
                   "class (addendum C3)")
        else:
            cls = "touches band -> indeterminate"
        out[f"verdict_{k}"] = cls
        print(f"[field] {k}: {point[k]:.4f} CI [{lo:.4f}, {hi:.4f}] -> {cls}")
    print(f"[field] ICC (extended corpus) = {icc_ext:.4f}")
    return out


def main():
    # 0. monotonicity of the frozen attempt-3 ladder (on fixtures), for record
    lad = json.load(open(os.path.join(ROOT, "data", "e2",
                                      "e2-ladder-results.json"),
                         encoding="utf-8"))
    rep0 = sorted([r for r in lad["repetitions"] if r.get("rep") == 0],
                  key=lambda x: x["rung"])
    seg3 = [r["est_seg"] for r in rep0]
    cont3 = [r["est_cont"] for r in rep0]
    mono3 = {"seg": all(b > a for a, b in zip(seg3, seg3[1:])),
             "cont": all(b > a for a, b in zip(cont3, cont3[1:]))}
    print(f"[attempt-3 record, on fixtures] E-seg monotone: {mono3['seg']} "
          f"{[round(v,3) for v in seg3]} | E-cont monotone: {mono3['cont']} "
          f"{[round(v,3) for v in cont3]}")

    robust = ladder_robustness()
    field = field_variants()

    results = {"attempt3_monotonicity_on_fixtures": mono3,
               "ladder_robustness_11nights": robust,
               "field_extended_variants": field}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"[e2-audit-response] results -> {OUT}")


if __name__ == "__main__":
    main()
