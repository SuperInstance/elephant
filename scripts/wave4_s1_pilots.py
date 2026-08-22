"""WAVE-4 S1 PILOT DRIVER — unsealed α∈{0,1} pilots on the fiber v4.

The pre-stated design gate (docs/wave4-registration-draft-2026-08-22.md §1.2
design gate, §4 row 3): generate UNSEALED pilot corpora with --fiber v4 at
α=0 and α=1 (5 nights: T1/T2/T3/T4a signal + T9 null; the canonical 6-reader
ATTENDANCE roster), as a MATCHED PAIR (shared --pair-seed ⇒ identical room
paths, rosters, authors, κ(t); targets differ only through α), run the
registered legs on them through the G5 adapter, read ICC, and file a
machine-readable adjudication JSON.

Nothing here touches data/wave3/** (sealed) — pilots live in
data/wave4-pilots/, legs output to data/wave4-pilots/legs/.

Usage:
    python3 scripts/wave4_s1_pilots.py            # generate + legs + adjudicate
    python3 scripts/wave4_s1_pilots.py --skip-gen # legs + adjudicate only
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOTS = os.path.join(ROOT, "data", "wave4-pilots")
LEGS_OUT = os.path.join(PILOTS, "legs")

# pilot skeleton: 4 signal nights (T4a carries the staged entrant) + T9 null
PILOT_NIGHTS = ["T1", "T2", "T3", "T4a", "T9"]
SEED = 20260822          # the draft §1.6 fresh master seed (no wave-3 overlap)
PAIR_SEED = 20260822     # G13 matched pair: rooms shared bit-for-bit


def generate_pilots():
    import scripts.riverbed_generator as rg
    rg.NIGHT_ORDER = list(PILOT_NIGHTS)   # 5-night pilot skeleton
    corpora = {}
    for label, branch in (("a0", "instrument"), ("a1", "collapse")):
        outdir = os.path.join(PILOTS, label)
        corpora[label] = outdir
        rg.generate_wave(outdir, branch_name=branch, seed=SEED,
                         tag_prefix=f"p4-{label}", pair_seed=PAIR_SEED,
                         fiber="v4")
    return corpora


def run_pilot_legs(corpora):
    from scripts.wave3_s3_legs import run_legs
    filed = {}
    for label, c in corpora.items():
        filed[label] = run_legs(c, LEGS_OUT, [12, 8, 16],
                                ["canonical", "actual"])
    return filed


def read_result(label, W=12, presence="canonical"):
    fn = os.path.join(LEGS_OUT, f"{label}.W{W:02d}.{presence}.json")
    return json.load(open(fn, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gen", action="store_true")
    args = ap.parse_args()
    corpora = {l: os.path.join(PILOTS, l) for l in ("a0", "a1")}
    if not args.skip_gen:
        generate_pilots()
    run_pilot_legs(corpora)

    # --- ICC (registered Measurement, G5 adapter) ----------------------- #
    from scripts.riverbed_adapter import load_wave
    icc = {}
    for label, c in corpora.items():
        w = load_wave(c)
        agg, per_dial = w["measurement"].icc()
        icc[label] = {"aggregate": agg,
                      "per_dial": {k: round(v, 4) for k, v in per_dial.items()}}

    # --- adjudication (canonical primary channel W=12) ------------------ #
    r0, r1 = read_result("a0"), read_result("a1")
    p0, p1 = r0["P"], r1["P"]
    a0, a1 = r0["A"], r1["A"]

    p_fires = (p1["P_trans"] is not None and p1["P_rest"] is not None
               and p1["P_trans"] < 0.5 * p1["P_rest"])
    a_gap = abs(a1["A"] - a0["A"])
    A_ENVELOPE = {"median": 0.011, "max": 0.085}   # wave-3 within-pair
    a_blind = a_gap <= A_ENVELOPE["max"]
    alpha0_parity = {
        "P_holds_099_class": (p0["P_trans"] is not None
                              and p0["P_trans"] >= 0.5 * p0["P_rest"]
                              and min(p0["P_trans"], p0["P_rest"]) >= 0.95),
        "A_fires": (a0["p"] is not None and a0["p"] <= 0.05),
    }

    out = {
        "run": "wave4-S1-hardening", "date": "2026-08-22",
        "fiber": "v4", "seed": SEED, "pair_seed": PAIR_SEED,
        "nights": PILOT_NIGHTS, "sealed": False,
        "P_a0": {k: p0[k] for k in ("P_trans", "P_rest", "trans_ci",
                                    "rest_ci", "holds_at_half",
                                    "mechanism_kill")},
        "P_a1": {k: p1[k] for k in ("P_trans", "P_rest", "trans_ci",
                                    "rest_ci", "holds_at_half",
                                    "mechanism_kill")},
        "A_a0": {k: a0[k] for k in ("n_events", "A", "p", "null95")},
        "A_a1": {k: a1[k] for k in ("n_events", "A", "p", "null95")},
        "gate": {
            "P_fires_at_alpha1": p_fires,
            "A_gap_alpha1_vs_alpha0": round(a_gap, 4),
            "A_blind_within_envelope": a_blind,
            "A_envelope_reference": A_ENVELOPE,
            "alpha0_instrument_parity": alpha0_parity,
            "PASS": bool(p_fires and a_blind
                         and all(alpha0_parity.values())),
        },
        "ICC": icc,
        "G6_reference_band": [0.60, 0.80],
    }
    opath = os.path.join(PILOTS, "wave4-S1-adjudication.json")
    with open(opath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["gate"], indent=1))
    print(f"ICC a0={icc['a0']['aggregate']:.3f} a1={icc['a1']['aggregate']:.3f}")
    print(f"-> {opath}")


if __name__ == "__main__":
    main()
