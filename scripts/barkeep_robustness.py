"""Barkeep-excluded robustness check for the E2 antecedent test (registered).

The E2 field run (REPORT-2026-08-19.md, data/e2/e2-field-results.json) flagged
the seeded reader `barkeep` (drift 2.3852 corpus-sd vs reader-median ~0.5,
only 2 nights: S1, S3) as possibly dragging the drift denominator of the
primary ratio (0.6088). This is the registered cheap-prerequisite robustness
check: rerun the premise ratio + baseline ICC with barkeep EXCLUDED from the
reader set, everything else frozen — same 9 primary nights, same corpus_sd
(reader-independent: computed from the room logs, not from readers), same
canonical-presence instrument, same bootstrap seed (20260819) and B=2000.

Scope, honestly: this can only move the SHADE of the INDETERMINATE verdict,
not adjudicate. Excluding an outlier selected by inspecting the field results
is a sensitivity analysis, not a new registered measurement; the R7 retirement
of the premise stands either way, and the E2 kill condition is evaluated on
the registered 15-reader run.

Reuses the committed instrument (scripts/e2_instrument.py) verbatim via
scripts/e2_field.field_readers; asserts exact reproduction of the committed
field numbers (ratio 0.6088, ICC 0.7714) before reporting the exclusion run.

Run:  python3 scripts/barkeep_robustness.py
numpy-only, CPU, read-only against the logs; writes nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.e2_field import field_readers
from scripts.e2_instrument import (FIELD_NIGHTS, Measurement, Night,
                                   PRIMARY_NIGHTS, corpus_sd, power_analysis,
                                   verdict)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELD_RESULTS = os.path.join(ROOT, "data", "e2", "e2-field-results.json")
NIGHTS_MANIFEST = os.path.join(ROOT, "data", "e2", "e2-nights-manifest.json")
EXCLUDED = "barkeep"
TOL = 1e-9


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def check_manifest():
    man = json.load(open(NIGHTS_MANIFEST, encoding="utf-8"))
    ok, checked = True, 0
    for tag, spec in man["nights"].items():
        if tag not in PRIMARY_NIGHTS:
            continue
        path = os.path.join(ROOT, "data", "nights", spec["file"])
        good = sha256(path) == spec["sha256"]
        ok &= good
        checked += 1
        print(f"    {tag:<4} sha256 {'OK' if good else 'MISMATCH'}")
    return checked, ok


def summarize(m, B=2000):
    seg = m.ratio_seg()
    cont = m.ratio_cont()
    drift = m.drift_mean()
    spread_c = m.spread_cont()
    spread_s = m.spread_seg()
    null_drift = m.null_drift_mean()
    boot = m.bootstrap(B=B)
    icc, icc_dial = m.icc()
    lo, hi = boot["cont_ci"]
    v = verdict(lo, hi)
    kill = ("FIRES (CI upper <= 0.6)" if hi <= 0.6
            else "does not fire (CI upper > 0.6)")
    pa = (power_analysis(cont, lo, hi, len(m.readers))
          if "INDETERMINATE" in v else None)
    drifts = [m.drift[r][0] for r in m.readers if not np.isnan(m.drift[r][0])]
    return {"n": len(m.readers), "ratio": cont, "seg_ratio": seg,
            "spread": spread_c, "spread_seg": spread_s, "drift": drift,
            "drift_median": float(np.median(drifts)),
            "null_drift": null_drift, "ci": (lo, hi),
            "seg_ci": boot["seg_ci"], "icc": icc, "icc_ci": boot["icc_ci"],
            "icc_dial": icc_dial, "verdict": v, "kill": kill, "pa": pa}


def main():
    ref = json.load(open(FIELD_RESULTS, encoding="utf-8"))

    print("=" * 76)
    print(f"BARKEEP-EXCLUDED ROBUSTNESS — E2 antecedent test (registered check)")
    print("=" * 76)

    print("\n[0] corpus integrity (v:2 nights vs SHA-pinned manifest):")
    n_checked, ok = check_manifest()
    if not ok:
        raise SystemExit("night logs do not match the registered manifest")

    sd, _ = corpus_sd([Night(n) for n in PRIMARY_NIGHTS])
    print(f"\n[1] corpus: 9 primary nights, corpus_sd = {sd:.4f} "
          f"(reader-independent — unchanged by any reader exclusion)")

    m15 = Measurement(field_readers(), sd, presence="canonical")
    r15 = summarize(m15)
    d_ratio = abs(r15["ratio"] - ref["primary"]["ratio"])
    d_icc = abs(r15["icc"] - ref["icc"]["aggregate"])
    d_ci = max(abs(r15["ci"][0] - ref["primary"]["ci"][0]),
               abs(r15["ci"][1] - ref["primary"]["ci"][1]))
    print(f"\n[2] reproduction (all 15 readers, canonical presence, "
          f"seed 20260819):")
    print(f"    ratio = {r15['ratio']:.10f} vs committed "
          f"{ref['primary']['ratio']:.10f}  |diff| = {d_ratio:.2e}")
    print(f"    CI    = [{r15['ci'][0]:.4f}, {r15['ci'][1]:.4f}] vs committed "
          f"[{ref['primary']['ci'][0]:.4f}, {ref['primary']['ci'][1]:.4f}]  "
          f"|diff| = {d_ci:.2e}")
    print(f"    ICC   = {r15['icc']:.10f} vs committed "
          f"{ref['icc']['aggregate']:.10f}  |diff| = {d_icc:.2e}")
    assert max(d_ratio, d_icc, d_ci) < TOL, "does not reproduce the field run"
    print("    -> reproduces the committed field numbers exactly; "
          "v:1 nights guarded by this equality")

    print(f"\n[3] EXCLUDED: `{EXCLUDED}` (seeded field-distribution draw, "
          f"archetype=critic; nights S1+S3 only; "
          f"drift {m15.drift[EXCLUDED][0]:.4f} vs reader-median "
          f"{r15['drift_median']:.4f} corpus-sd)")

    reduced = {r: ns for r, ns in FIELD_NIGHTS.items() if r != EXCLUDED}
    m14 = Measurement(field_readers(reduced), sd, presence="canonical")
    r14 = summarize(m14)

    print(f"\n[4] PRIMARY RERUN (14 readers, E-cont-canonical):")
    print(f"    ratio = {r14['spread']:.4f}/{r14['drift']:.4f} "
          f"= {r14['ratio']:.4f} corpus-sd   (was {r15['ratio']:.4f}; "
          f"delta {r14['ratio'] - r15['ratio']:+.4f})")
    print(f"    bootstrap 95% CI over readers: [{r14['ci'][0]:.4f}, "
          f"{r14['ci'][1]:.4f}]   (was [{r15['ci'][0]:.4f}, "
          f"{r15['ci'][1]:.4f}])")
    print(f"    E-seg-canonical: {r14['seg_ratio']:.4f} "
          f"CI [{r14['seg_ci'][0]:.4f}, {r14['seg_ci'][1]:.4f}]   "
          f"(was {r15['seg_ratio']:.4f} CI [{r15['seg_ci'][0]:.4f}, "
          f"{r15['seg_ci'][1]:.4f}])")
    print(f"    spread = {r14['spread']:.4f} (was {r15['spread']:.4f}) | "
          f"mean drift = {r14['drift']:.4f} (was {r15['drift']:.4f}) | "
          f"drift median = {r14['drift_median']:.4f}")
    print(f"    S5 null-drift control = {r14['null_drift']:.4f} "
          f"(was {r15['null_drift']:.4f})")

    print(f"\n[5] SECONDARY RERUN (14 readers): baseline ICC = "
          f"{r14['icc']:.4f}   (was {r15['icc']:.4f}; "
          f"delta {r14['icc'] - r15['icc']:+.4f})")
    print(f"    ICC bootstrap 95% CI: [{r14['icc_ci'][0]:.4f}, "
          f"{r14['icc_ci'][1]:.4f}]   (was [{r15['icc_ci'][0]:.4f}, "
          f"{r15['icc_ci'][1]:.4f}])")
    for d in r14["icc_dial"]:
        print(f"    ICC[{d:<13}] = {r14['icc_dial'][d]:+.4f} "
              f"(was {r15['icc_dial'][d]:+.4f})")

    print(f"\n[6] VERDICT vs kill band [0.3, 0.6] (14 readers):")
    print(f"    {r14['verdict']}   (was: {r15['verdict']})")
    print(f"    E2 kill condition: {r14['kill']}   (was: {r15['kill']})")
    edge = 0.6
    for tag, r in (("15-reader", r15), ("14-reader", r14)):
        print(f"    {tag} point sits {r['ratio'] - edge:+.4f} from the 0.6 "
              f"edge; CI lower {r['ci'][0] - edge:+.4f} from it")
    if r14["pa"]:
        print(f"    power analysis (14): half-width {r14['pa']['ci_half_width']:.4f}, "
              f"distance to nearest edge {r14['pa']['distance_to_nearest_edge']:.4f} "
              f"-> N required ~= {r14['pa']['n_required']} readers "
              f"(15-reader run: ~{r15['pa']['n_required']})")

    print("\n[7] SHADE:")
    if "INDETERMINATE" in r14["verdict"]:
        above = r14["ratio"] > 0.6
        print(f"    still INDETERMINATE — CI still touches the band — but the "
              f"point estimate now sits "
              f"{'clearly above' if above else 'at/inside'} the 0.6 edge "
              f"({r14['ratio']:.4f}, was 0.0088 above it)")
        print("    the barkeep drag was real but not verdict-changing: "
              "removing him de-inflates mean drift "
              f"({r15['drift']:.4f} -> {r14['drift']:.4f}) and lifts the ratio; "
              "the reader-sample CI remains too wide to adjudicate")
    else:
        print(f"    verdict CHANGES to: {r14['verdict']}")
        print("    (still not adjudication: post-hoc exclusion selected by "
              "inspection; the registered 15-reader verdict stands)")

    print("\n" + "=" * 76)
    print(f"FINAL: ratio {r15['ratio']:.4f} -> {r14['ratio']:.4f} | "
          f"ICC {r15['icc']:.4f} -> {r14['icc']:.4f} | "
          f"verdict: INDETERMINATE (shade: point now "
          f"{r14['ratio'] - edge:+.4f} vs 0.6 edge, CI "
          f"[{r14['ci'][0]:.4f}, {r14['ci'][1]:.4f}] still touches)")
    print("=" * 76)


if __name__ == "__main__":
    main()
