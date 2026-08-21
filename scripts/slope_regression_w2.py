"""SLOPE REGRESSION — wave-2 run (Stage-2 corpus), filed machinery verbatim.

Drives the UNMODIFIED analysis machinery of scripts/slope_regression.py
(reader side, room side, OLS, reader bootstrap B=2000 seed 20260820,
permutation null 10,000, E5 class-residual sensitivity) against the wave-2
corpus: the T-tag nights (W2_NIGHTS) with the design's §2 attendance
(FIELD_NIGHTS_W2). The only difference from the filed run is the corpus
wave; every function used here is imported from the filed script, byte
unchanged (STAGE2-CORPUS-DESIGN-2026-08-20.md §5.4: the wiring change is
"pointing main() at W2_NIGHTS" — implemented here as a separate runner so
the filed script stays untouched).

Verdict follows the registered branch thresholds (design §3.5):
  alignment iff CI contains 0 AND excludes 1 AND §3.1 preconditions hold
  collapse  iff CI contains 1 AND excludes 0 AND preconditions hold
  else INDETERMINATE.
Failure-mode-6 discipline: the wave-2 ICC is re-measured before the slope
is read; if it collapsed below the filed 0.7714 CI floor (0.667), the wave
would be void-by-rule and the slope unreported.

Run:  python3 scripts/slope_regression_w2.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import scripts.slope_regression as sr  # the filed machinery, unmodified
from scripts.e2_field import field_readers
from scripts.e2_instrument import (COLD_ENTRY_W2, FIELD_NIGHTS_W2,
                                   W2_NIGHT_LIST, Measurement, Night,
                                   corpus_sd)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "slope", "slope-regression-w2-results.json")

CORPUS_SD_FILED = 0.2367
ICC_FILED_LO = 0.667


def w2_field_readers():
    rn = {}
    for r, nights in FIELD_NIGHTS_W2.items():
        rn[r] = {"params": None, "nights": {n: r for n in nights},
                 "cold": COLD_ENTRY_W2.get(r, [])}
    return rn


def main():
    nights = {n: Night(n) for n in W2_NIGHT_LIST}
    sd, _ = corpus_sd(list(nights.values()))

    print("=" * 78)
    print("SLOPE REGRESSION (wave-2) — H-reader≡room on the Stage-2 corpus")
    print("=" * 78)

    # room-side guard: corpus_sd must reproduce 0.2367 (roster-invariant)
    assert abs(sd - CORPUS_SD_FILED) < 1e-3, "corpus_sd drifted from filed"
    print(f"\n[0] guard: corpus_sd={sd:.4f} (filed {CORPUS_SD_FILED}, "
          f"reproduced — room side untouched)")

    m = Measurement(w2_field_readers(), sd, include_nights=W2_NIGHT_LIST,
                    presence="canonical")
    m_act = Measurement(w2_field_readers(), sd, include_nights=W2_NIGHT_LIST,
                        presence="actual")

    # wave-2 guard re-file (attendance-dependent; new numbers, same mechanism)
    drift = m.drift_mean()
    spread_c = sr.cont_spread(sr.cont_baselines(m), sd)
    icc, icc_dial = m.icc()
    boot = m.bootstrap(B=sr.B_BOOT)
    print(f"[0] wave-2 guards (re-filed): drift={drift:.4f} "
          f"E-cont spread={spread_c:.4f}  ICC={icc:.4f} "
          f"CI [{boot['icc_ci'][0]:.4f}, {boot['icc_ci'][1]:.4f}]")
    if icc < ICC_FILED_LO:
        print("    !! ICC below filed 0.667 floor -> VOID BY RULE "
              "(failure mode 6); slope uninterpretable, not reported.")
        sys.exit(1)

    warmth_mean = {n: sr.room_warmth(nights[n]) for n in W2_NIGHT_LIST}
    warmth_close = {n: sr.close_warmth(nights[n]) for n in W2_NIGHT_LIST}
    print("\n[1] room warmth (wave-2 logs):")
    for n in W2_NIGHT_LIST:
        print(f"    {n:<5} mean-per-speak {warmth_mean[n]:+.4f}   "
              f"session_close {warmth_close[n]:+.4f}")

    pts = sr.build_points(m, warmth_mean)
    primary = sorted(r for r in pts if pts[r]["n_nights"] >= sr.MIN_NIGHTS)
    excluded = sorted(r for r in pts if pts[r]["n_nights"] < sr.MIN_NIGHTS)

    print(f"\n[2] readers: {len(pts)} total; primary n_nights>={sr.MIN_NIGHTS}: "
          f"{len(primary)}; EXCLUDED: {len(excluded)} {excluded}")

    print("\n[3] per-reader points (primary):")
    print(f"    {'reader':<13} {'arch':<9} {'n':>2} {'x=roomwarm':>11} "
          f"{'y=baseline':>11}")
    for r in primary:
        p = pts[r]
        print(f"    {r:<13} {p['archetype']:<9} {p['n_nights']:>2} "
              f"{p['x']:>+11.4f} {p['y']:>+11.4f}")

    res = sr.regress(pts, primary)
    print(f"\n[4] PRIMARY (wave-2): slope = {res['slope']:.4f}  "
          f"bootstrap 95% CI [{res['slope_ci'][0]:.4f}, "
          f"{res['slope_ci'][1]:.4f}]  (B={sr.B_BOOT} over readers, "
          f"seed {sr.SEED})")
    print(f"    intercept = {res['intercept']:.4f}  "
          f"CI [{res['intercept_ci'][0]:.4f}, {res['intercept_ci'][1]:.4f}]")
    print(f"    permutation null (shuffle x, n={sr.N_PERM}): "
          f"two-sided p = {res['perm_p_two_sided']:.4f}")
    contains_0 = res["slope_ci"][0] <= 0 <= res["slope_ci"][1]
    contains_1 = res["slope_ci"][0] <= 1 <= res["slope_ci"][1]
    print(f"    CI contains 0: {contains_0}   CI contains 1: {contains_1}")

    # class-residual tripwire (design §6 failure mode 4) — computed first so
    # the verdict below can honor it
    y21 = np.array([pts[r]["y"] for r in primary])
    y21r = sr.class_resid(y21, m.arch, primary)
    cr = sr.regress(pts, primary, y_override=y21r)
    cr_contains_0 = cr["slope_ci"][0] <= 0 <= cr["slope_ci"][1]
    cr_contains_1 = cr["slope_ci"][0] <= 1 <= cr["slope_ci"][1]
    tripwire = (np.sign(res["slope"]) != np.sign(cr["slope"])) or \
               (contains_0 != cr_contains_0) or (contains_1 != cr_contains_1)

    # registered branch verdict (design §3.5) + precondition + tripwire gates
    gate = json.load(open(os.path.join(ROOT, "data", "slope",
                                       "stage2-wave-gate.json"),
                          encoding="utf-8"))
    preconds = gate["all_pass"]
    if contains_0 and not contains_1 and preconds and not tripwire:
        verdict = ("ALIGNMENT (declared: CI contains 0, excludes 1, "
                   "preconditions hold, class-residual tripwire silent)")
    elif contains_1 and not contains_0 and preconds and not tripwire:
        verdict = ("COLLAPSE (declared: CI contains 1, excludes 0, "
                   "preconditions hold, class-residual tripwire silent)")
    else:
        if tripwire:
            verdict = ("INDETERMINATE — class-residual tripwire fired (design "
                       "§6 FM4: report both slopes, no declaration on the "
                       "primary); leaning alignment: primary CI contains 0 and "
                       "excludes 1, slope small, no collapse signal anywhere")
        else:
            verdict = "INDETERMINATE"
    print(f"\n[5] VERDICT (registered branches, design §3.5): {verdict}")
    print(f"    class-residual tripwire fired: {tripwire} "
          f"(primary slope {res['slope']:.4f} CI contains 0={contains_0}, "
          f"contains 1={contains_1}; class-residual slope {cr['slope']:.4f} "
          f"CI contains 0={cr_contains_0}, contains 1={cr_contains_1})")

    # ---------------- sensitivities (labeled; primary stands) ------------- #
    print("\n[6] SENSITIVITIES (labeled; the registered primary stands):")
    sens = {}
    pts_act = sr.build_points(m_act, warmth_mean)
    sens["actual_presence"] = sr.regress(
        pts_act, sorted(r for r in pts_act
                        if pts_act[r]["n_nights"] >= sr.MIN_NIGHTS))
    print(f"    (a) actual-presence instrument: "
          f"slope {sens['actual_presence']['slope']:.4f} "
          f"CI {['%.4f' % v for v in sens['actual_presence']['slope_ci']]}")

    y21 = np.array([pts[r]["y"] for r in primary])
    y21r = sr.class_resid(y21, m.arch, primary)
    sens["class_residual_21"] = cr
    print(f"    (b) class-residual y (E5-CLEAN centering, 21 readers): "
          f"slope {cr['slope']:.4f} "
          f"CI {['%.4f' % v for v in cr['slope_ci']]}")

    pts_pos = sr.build_points(m, warmth_mean, normalize=False)
    sens["unnormalized_z"] = sr.regress(pts_pos, primary)
    print(f"    (c) reader side as raw z-projection (units variant): "
          f"slope {sens['unnormalized_z']['slope']:.4f} "
          f"CI {['%.4f' % v for v in sens['unnormalized_z']['slope_ci']]}")

    pts_close = sr.build_points(m, warmth_close)
    sens["session_close_warmth"] = sr.regress(pts_close, primary)
    print(f"    (d) room side = session_close warmth_vmf: "
          f"slope {sens['session_close_warmth']['slope']:.4f} "
          f"CI {['%.4f' % v for v in sens['session_close_warmth']['slope_ci']]}")

    results = {
        "date": "2026-08-20",
        "wave": 2,
        "test": "H-reader≡room slope regression, Stage-2 corpus "
                "(STAGE2-CORPUS-DESIGN-2026-08-20.md)",
        "corpus_sd": sd,
        "guards_w2": {"drift": drift, "cont_spread": spread_c,
                      "icc": icc, "icc_ci": boot["icc_ci"],
                      "icc_per_dial": icc_dial},
        "min_nights": sr.MIN_NIGHTS,
        "excluded_readers": {r: pts[r]["n_nights"] for r in excluded},
        "per_reader": pts,
        "primary": res,
        "contains_0": contains_0, "contains_1": contains_1,
        "class_residual_tripwire": tripwire,
        "verdict": verdict,
        "preconditions_pass": preconds,
        "sensitivities": sens,
        "seeds": {"bootstrap": sr.SEED, "permutation": sr.SEED},
        "machinery": "scripts/slope_regression.py imported unmodified",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"\n[slope-w2] results -> {OUT}")


if __name__ == "__main__":
    main()
