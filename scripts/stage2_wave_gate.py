"""Stage-2 wave gate — generation-time validation of the wave-2 corpus.

Implements STAGE2-CORPUS-DESIGN-2026-08-20.md §5.5 (the new wave gate,
ladder-style) and the §3.1(ii)/(iii) x-side validity preconditions. Runs
read-only against the generated T-logs. numpy-only, CPU, deterministic.

Checks (fail-fast, void-by-rule on the §3 preconditions):
  1. logged session_open rosters == designed ATTENDANCE (design §2 matrix)
  2. corpus_sd reproduces the filed 0.2367 (roster-invariant room side;
     engine-drift guard)          [failure mode 3]
  3. per-night warmth reproduces the filed ladder to 4 decimals
     (T1 .6551, T2 .3187, T3 .6551, T4a .4465, T4b .6319, T5 .6293,
      T5c .6293, T8 .7409, T9 .7589)         [failure mode 3]
  4. §2 band means land within ±0.02 of targets (0.48 / 0.64 / 0.71)
  5. all 21 readers have >= 3 logged nights (attrition guard; mid band's
     4-night buffer is the schedule-level backstop)   [failure mode 2]
  6. §3.1 preconditions, asserted (any failure => VOID — INDETERMINATE
     BY RULE, no branch reading): (i) no two readers share a visited-room
     set (21/21 unique); (ii) >= 3 distinct x and Sxx >= 0.19; (iii) no
     archetype majority in any band; x-range >= 0.15 (design target 0.254)
  7. wave-2 guard re-file: drift and E-cont spread are attendance-dependent
     and are re-measured here (same guard mechanism, new numbers); ICC is
     re-measured (failure mode 6) against the filed 0.7714 CI [0.667, 0.810]

Run:  python3 scripts/stage2_wave_gate.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.e2_field import field_readers
from scripts.e2_instrument import (COLD_ENTRY_W2, FIELD_NIGHTS_W2,
                                   W2_NIGHT_LIST, Measurement, Night,
                                   archetype_labels, corpus_sd)
from scripts.e5_identity_propagation import cont_baselines, cont_spread
from scripts.slope_regression import room_warmth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "slope", "stage2-wave-gate.json")

# Filed per-family warmth ladder (SLOPE-REGRESSION-2026-08-20.md §1,
# reproduced from the frozen logs; T-tags map to families per design §5.1).
LADDER = {"T1": 0.6551, "T2": 0.3187, "T3": 0.6551, "T4a": 0.4465,
          "T4b": 0.6319, "T5": 0.6293, "T5c": 0.6293, "T8": 0.7409,
          "T9": 0.7589}
BAND_TARGETS = {"cold": 0.48, "mid": 0.64, "warm": 0.71}
BANDS = {
    "cold": ["writer", "engineer", "drifter", "lamplighter", "tinker",
             "new-1", "new-2"],
    "mid": ["poet", "critic", "singer", "cartographer", "blacksmith",
            "new-3", "new-4"],
    "warm": ["essayist", "captain", "barkeep", "fiddler", "weaver",
             "new-5", "new-6"],
}
CORPUS_SD_FILED = 0.2367
ICC_FILED_LO, ICC_FILED_HI = 0.667, 0.810


def check(cond, label):
    results["checks"].append({"check": label, "pass": bool(cond)})
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    if not cond:
        results["all_pass"] = False
    return bool(cond)


results = {"date": "2026-08-20", "wave": 2, "all_pass": True,
           "checks": []}


def main():
    from scripts.e2_nights import ATTENDANCE  # designed matrix (session-open)
    nights = {n: Night(n) for n in W2_NIGHT_LIST}

    # 1. logged rosters == designed ATTENDANCE (failure mode 1)
    print("[1] logged rosters vs designed ATTENDANCE:")
    for tag in W2_NIGHT_LIST:
        logged = sorted(nights[tag].open["roster"])
        check(logged == sorted(ATTENDANCE[tag]),
              f"{tag}: logged roster == designed ({len(logged)})")

    # 2. corpus_sd reproduces 0.2367 (failure mode 3; roster-invariant)
    sd, per_dial = corpus_sd([nights[n] for n in W2_NIGHT_LIST])
    results["corpus_sd"] = sd
    print(f"\n[2] corpus_sd = {sd:.4f} (filed {CORPUS_SD_FILED}):")
    check(abs(sd - CORPUS_SD_FILED) < 1e-3,
          "corpus_sd reproduces 0.2367 (room side untouched)")

    # 3. per-night warmth reproduces the filed ladder (failure mode 3)
    print("\n[3] per-night warmth vs filed ladder (4 decimals):")
    warmth = {n: room_warmth(nights[n]) for n in W2_NIGHT_LIST}
    for tag in W2_NIGHT_LIST:
        ok = abs(warmth[tag] - LADDER[tag]) < 5e-5
        check(ok, f"{tag}: warmth {warmth[tag]:+.4f} vs filed {LADDER[tag]:+.4f}")
    results["per_night_warmth"] = warmth

    # 4. band means within ±0.02 of targets
    print("\n[4] band means (x from measured warmth + design attendance):")
    x = {}
    for r, ns in FIELD_NIGHTS_W2.items():
        x[r] = float(np.mean([warmth[n] for n in ns]))
    for band, rs in BANDS.items():
        mean = float(np.mean([x[r] for r in rs]))
        ok = abs(mean - BAND_TARGETS[band]) < 0.02
        check(ok, f"{band}: mean x = {mean:.4f} vs target {BAND_TARGETS[band]:.2f}")
        results.setdefault("band_means", {})[band] = mean

    # 5. all 21 readers >= 3 logged nights (failure mode 2)
    print("\n[5] attendance completeness (21 readers, >= 3 nights):")
    m = Measurement(field_readers(FIELD_NIGHTS_W2), sd,
                    include_nights=W2_NIGHT_LIST, presence="canonical")
    n_nights = {r: len(m.readings[r]) for r in m.readers}
    results["n_nights_per_reader"] = n_nights
    for r in sorted(m.readers):
        check(n_nights[r] >= 3, f"{r}: {n_nights[r]} nights (>= 3)")
    check(len(m.readers) == 21, "21 readers measured")

    # 6. §3.1 x-side preconditions (void-by-rule if any fail)
    print("\n[6] registered x-side preconditions (§3.1; void-by-rule):")
    sets = {r: tuple(sorted(FIELD_NIGHTS_W2[r])) for r in sorted(FIELD_NIGHTS_W2)}
    check(len(set(sets.values())) == len(sets),
          f"21/21 unique visited-room sets ({len(set(sets.values()))} unique)")
    xs = np.array([x[r] for r in sorted(FIELD_NIGHTS_W2)])
    xbar = float(xs.mean())
    sxx = float(np.sum((xs - xbar) ** 2))
    results["sxx"] = sxx
    results["x_range"] = float(xs.max() - xs.min())
    results["n_distinct_x"] = len(set(np.round(xs, 10).tolist()))
    check(results["n_distinct_x"] >= 3,
          f">= 3 distinct x ({results['n_distinct_x']} distinct)")
    check(sxx >= 0.19, f"Sxx = {sxx:.4f} >= 0.19")
    check(results["x_range"] >= 0.15,
          f"x-range = {results['x_range']:.4f} >= 0.15 (design 0.254)")
    arch = archetype_labels()
    for band, rs in BANDS.items():
        counts = {}
        for r in rs:
            counts[arch[r]] = counts.get(arch[r], 0) + 1
        mx = max(counts.values())
        check(mx <= len(rs) / 2,
              f"no archetype majority in {band} (max {mx}/7: {counts})")

    # 7. wave-2 guard re-file: drift, E-cont spread, ICC (failure modes 3/6)
    print("\n[7] wave-2 guard re-file (attendance-dependent, new numbers):")
    drift = m.drift_mean()
    spread_c = cont_spread(cont_baselines(m), sd)
    icc, icc_dial = m.icc()
    boot = m.bootstrap(B=2000)
    results["guards_w2"] = {"drift": drift, "cont_spread": spread_c,
                            "icc": icc, "icc_ci": boot["icc_ci"],
                            "icc_per_dial": icc_dial}
    print(f"  drift = {drift:.4f} corpus-sd (wave-1 filed 0.7483, "
          f"attendance-dependent)")
    print(f"  E-cont spread = {spread_c:.4f} corpus-sd (wave-1 filed 0.4556)")
    print(f"  ICC = {icc:.4f}  bootstrap 95% CI "
          f"[{boot['icc_ci'][0]:.4f}, {boot['icc_ci'][1]:.4f}] "
          f"(filed 0.7714 CI [0.667, 0.810])")
    icc_ok = icc >= ICC_FILED_LO
    results["icc_ok"] = bool(icc_ok)
    if not icc_ok:
        results["all_pass"] = False
        print("  !! ICC below filed CI floor -> void-by-rule (failure mode 6): "
              "slope uninterpretable")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"\n[stage2-wave-gate] {'ALL CHECKS PASS' if results['all_pass'] else 'FAILURES PRESENT'} -> {OUT}")


if __name__ == "__main__":
    main()
