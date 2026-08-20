"""E2 field arm — the Antecedent Test at Power, on the real expanded corpus.

Runs only after the calibration ladder passes (addendum 2, attempt 3):
primary estimator E-cont-canonical (flagged: continuity estimator; E-seg
failed rung 0.9 by 0.0096 on fixtures and is reported alongside).

Field corpus (registration R2/R3/R4): nights A, D, D-cold (frozen v:1) +
S1, S2, S3, S4a, S4b, S5 (v:2, cold-entry protocol); 15 real readers
(7 existing personas frozen verbatim + 8 seeded field-distribution draws,
archetype-labeled); B/C excluded from the primary (byte-replays of A).

Outputs: ratio + bootstrap CI + verdict vs the kill band; ICC + CI;
null-drift control; class-conditional tables; labeled sensitivities
(actual presence, B/C in, S5-null in, old-corpus continuity).
All numbers here are FIELD numbers (real readers, real nights).

Run:  python3 scripts/e2_field.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.e2_instrument import (COLD_ENTRY, FIELD_NIGHTS, NIGHT_SPECS,
                                   Night, PRIMARY_NIGHTS, Measurement,
                                   assert_replay_matches_log, corpus_sd,
                                   power_analysis, verdict)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSONAS = os.path.join(ROOT, "data", "e2", "e2-personas.json")
OUT = os.path.join(ROOT, "data", "e2", "e2-field-results.json")

ORIG7 = ["writer", "poet", "essayist", "engineer", "critic", "captain",
         "drifter"]
NEW8 = ["barkeep", "singer", "fiddler", "lamplighter", "cartographer",
        "blacksmith", "tinker", "weaver"]
OLD_ONLY = {  # pre-measurement continuity attendance (old corpus)
    **{n: ["A", "B", "C", "D", "D-cold"] for n in ORIG7 if n != "drifter"},
    "drifter": ["D"],
}


def field_readers(nights_map=None):
    rn = {}
    for r, nights in (nights_map or FIELD_NIGHTS).items():
        rn[r] = {"params": None, "nights": {n: r for n in nights},
                 "cold": COLD_ENTRY.get(r, [])}
    return rn


def main():
    ladder = json.load(open(os.path.join(ROOT, "data", "e2",
                                         "e2-ladder-results.json"),
                            encoding="utf-8"))
    assert ladder.get("primary"), "ladder must pass before the field arm"

    nights = {n: Night(n) for n in PRIMARY_NIGHTS}
    sd, per_dial = corpus_sd(list(nights.values()))

    print("=" * 76)
    print("E2 FIELD ARM — Antecedent Test at Power (schedule-diversified)")
    print("=" * 76)
    print(f"\n[0] corpus: 9 primary nights, corpus_sd = {sd:.4f}")

    # instrument consistency: v:1 replay equations reproduce logged v:2 facts
    worst = 0.0
    checked = 0
    for tag in ("S1", "S2", "S3", "S4a", "S4b", "S5"):
        n = nights[tag]
        for r in n.params:
            try:
                assert_replay_matches_log(n, r, cold=tag in COLD_ENTRY.get(r, []))
                checked += 1
            except AssertionError:
                raise
    print(f"[1] replay==log consistency asserted on {checked} reader-nights "
          f"(max |diff| < 1e-9)")

    m = Measurement(field_readers(), sd, presence="canonical")
    m_act = Measurement(field_readers(), sd, presence="actual")

    n_readers = len(m.readers)
    print(f"[2] field readers: {n_readers} (7 existing frozen + 8 seeded draws)")
    for r in m.readers:
        nights_here = sorted(m.readings[r])
        print(f"    {r:<13} archetype={m.arch[r]:<9} nights={nights_here}")

    # ---- primary + secondary ----
    seg = m.ratio_seg()
    cont = m.ratio_cont()
    drift = m.drift_mean()
    spread_c = m.spread_cont()
    spread_s = m.spread_seg()
    null_drift = m.null_drift_mean()
    boot = m.bootstrap(B=2000)
    icc, icc_dial = m.icc()
    lo, hi = boot["cont_ci"]

    print(f"\n[3] PRIMARY (field, E-cont-canonical, ladder-flagged): "
          f"ratio = {spread_c:.4f}/{drift:.4f} = {cont:.4f}")
    print(f"    bootstrap 95% CI over readers: [{lo:.4f}, {hi:.4f}]")
    print(f"    E-seg-canonical (reported, failed top rung on fixtures): "
          f"{seg:.4f}  CI {tuple(round(x,4) for x in boot['seg_ci'])}")
    print(f"    between-reader spread (E-cont) = {spread_c:.4f} corpus-sd | "
          f"mean within-reader drift = {drift:.4f} corpus-sd")
    print(f"    S5 no-flip null drift (control, excluded from primary): "
          f"{null_drift:.4f} corpus-sd")
    v = verdict(lo, hi)
    print(f"\n[4] VERDICT vs kill band [0.3, 0.6]: {v}")
    kill_condition = "FIRES (CI upper <= 0.6: the powered estimate never clears)" \
        if hi <= 0.6 else "does not fire (CI upper > 0.6)"
    print(f"    E2 kill condition (CI upper <= 0.6): {kill_condition}")

    pa = None
    if "INDETERMINATE" in v:
        pa = power_analysis(cont, lo, hi, n_readers)
        print(f"    power analysis: half-width {pa['ci_half_width']:.4f}, "
              f"distance to nearest edge {pa['distance_to_nearest_edge']:.4f} "
              f"-> N required ≈ {pa['n_required']} readers")

    print(f"\n[5] SECONDARY (field): baseline ICC = {icc:.4f} "
          f"(bootstrap 95% CI {tuple(round(x,4) for x in boot['icc_ci'])})")
    for d, val in icc_dial.items():
        print(f"    ICC[{d:<13}] = {val:+.4f}")

    # ---- class-conditional discipline (rule 2) ----
    print("\n[6] class-conditional (field):")
    groups = {}
    for r in m.readers:
        groups.setdefault(m.arch[r], []).append(r)
    for a in sorted(groups):
        ds = [m.drift[r][0] for r in groups[a] if not np.isnan(m.drift[r][0])]
        print(f"    archetype {a:<9} n={len(groups[a])} "
              f"mean drift = {np.mean(ds):.4f} corpus-sd")
    resid = m.spread_seg(class_residual=True)
    print(f"    class-residual spread (E-seg variant) = {resid:.4f} corpus-sd "
          f"-> class-residual ratio = {resid/drift:.4f} "
          f"(population ratio {cont:.4f}; the difference is class structure)")

    # ---- sensitivities ----
    print("\n[7] sensitivities (field):")
    s = {}
    s["actual_presence"] = {
        "cont": m_act.ratio_cont(), "seg": m_act.ratio_seg(),
        "note": ("attempt-1 instrument; participation-conflated "
                 "(bias +0.18 at planted 0, on fixtures)")}
    m_bc = Measurement(field_readers({**FIELD_NIGHTS,
                                      **{n: FIELD_NIGHTS[n] + ["B", "C"]
                                         for n in FIELD_NIGHTS if len(FIELD_NIGHTS[n]) > 3}}),
                       sd, include_nights=PRIMARY_NIGHTS + ["B", "C"],
                       presence="canonical")
    s["with_BC"] = {"cont": m_bc.ratio_cont(), "seg": m_bc.ratio_seg()}
    m_null = Measurement(field_readers(), sd, include_null_drift=True,
                         presence="canonical")
    s["with_S5_null_drift"] = {"cont": m_null.ratio_cont(),
                               "null_included": m_null.drift_mean()}
    old_nights = ["A", "B", "C", "D", "D-cold"]
    old_sd, _ = corpus_sd([Night(n) for n in old_nights])
    m_old = Measurement(field_readers(OLD_ONLY), old_sd,
                        include_nights=old_nights, presence="actual")
    s["old_corpus_continuity"] = {"cont": m_old.ratio_cont(),
                                  "corpus_sd": old_sd,
                                  "note": "pre-measurement estimator+corpus; "
                                          "continuity check vs 0.5599"}
    for k, v2 in s.items():
        line = f"    {k:<22}"
        if "cont" in v2:
            line += f" E-cont={v2['cont']:.4f}"
        if "seg" in v2 and not np.isnan(v2.get("seg", float('nan'))):
            line += f" E-seg={v2['seg']:.4f}"
        if "note" in v2:
            line += f"  [{v2['note']}]"
        print(line)

    # per-reader table
    print("\n[8] per-reader (field, canonical):")
    for r in m.readers:
        print(f"    {r:<13} drift={m.drift[r][0]:.4f} "
              f"null={m.drift[r][1]:.4f} nights={len(m.readings[r])}")

    results = {
        "date": "2026-08-19", "corpus_sd": sd,
        "per_dial_sd": per_dial.tolist(),
        "n_readers": n_readers,
        "primary": {"estimator": "E-cont-canonical (flagged)",
                    "ratio": cont, "spread": spread_c, "drift": drift,
                    "ci": [lo, hi], "seg_ratio": seg,
                    "seg_ci": list(boot["seg_ci"])},
        "icc": {"aggregate": icc, "per_dial": icc_dial,
                "ci": list(boot["icc_ci"])},
        "null_drift_control": null_drift,
        "verdict": v, "kill_condition": kill_condition,
        "power_analysis": pa,
        "class_residual": {"spread": resid, "ratio": resid / drift},
        "archetype_drift": {a: float(np.mean([m.drift[r][0]
                                              for r in groups[a]]))
                            for a in groups},
        "sensitivities": s,
        "per_reader": {r: {"drift": m.drift[r][0], "null": m.drift[r][1],
                           "n_nights": len(m.readings[r])}
                       for r in m.readers},
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"\n[e2-field] results -> {OUT}")


if __name__ == "__main__":
    main()
