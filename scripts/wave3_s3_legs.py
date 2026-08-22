"""WAVE-3 S3 LEG DRIVER — A/D/P/S on riverbed corpora through the G5 adapter.

Registered legs run UNMODIFIED: this driver mirrors
premise_band_movers.analyze_wave()'s exact call sequence (leg_A / leg_A up /
leg_A start-ref / leg_D / leg_P / leg_S / trajectory / null-night scores /
mean_d_by_phase, same seeds 20260821/+2/+4) with ONE substitution the G5
adapter exists for: the Measurement is scripts.riverbed_adapter's
RiverbedMeasurement (the registered e2_instrument.Measurement subclass that
redirects ONLY night construction to generated files). No registered script
is modified; premise_band_movers + e2_instrument consume the generated
nights unmodified through the adapter.

Per corpus it files RAW leg outputs (no interpretation — S4 blinded
analysis is a separate registered step) to:

    data/wave3/legs/<corpus-id>.W<W>.<presence>.json

Channels filed per corpus (the field run's structure, premise_band_movers
main(): canonical primary + labeled sensitivities):
  - W=12 canonical  (the registered field-primary channel)
  - W=12 actual     (adapter-default channel; labeled sensitivity)
  - W=8 / W=16 canonical (the registered W-manifold sensitivity surface,
                    registration §1.4 void rule 7)

Signal/null nights and the a-priori x ladder are the field's W2 design
mapped by FAMILY (the generated families ARE the frozen T-families):
SIGNAL = T1,T2,T3,T4a,T4b,T5,T5c,T8; NULL = T9; x = X_W2[family].

Usage:
    python3 scripts/wave3_s3_legs.py --corpus data/wave3/<id> [--corpus ...]
        [--out data/wave3/legs] [--W 12 8 16] [--presence canonical actual]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.premise_band_movers import (
    SEED, W_PRIMARY, X_W2, NULL_W2, SIGNAL_W2, drift_by_phase, leg_A, leg_D,
    leg_P, leg_S, night_windows, strata_transitions)
from scripts.riverbed_adapter import build_measurement, load_wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "data", "wave3", "legs")


def run_legs(corpus_dir, outdir, W_values, presences):
    w = load_wave(corpus_dir)
    man = w["manifest"]
    fam_of = {t: meta["family"] for t, meta in man["nights"].items()}
    nights = list(w["nights"])
    signal = [t for t in nights if fam_of[t] in SIGNAL_W2]
    nulls = [t for t in nights if fam_of[t] in NULL_W2]
    x_map = {t: X_W2[fam_of[t]] for t in nights}
    label = man.get("corpus_id", os.path.basename(os.path.normpath(corpus_dir)))
    filed = []
    for presence in presences:
        m, sd = build_measurement(w["nights"], w["attendance"], w["cold"],
                                  presence=presence)
        for W in W_values:
            t0 = time.time()
            win = {n: night_windows(m, n, sd, W) for n in nights}
            res = {"wave": "riverbed", "corpus_id": label,
                   "manifest": os.path.relpath(
                       os.path.join(w["dir"], "riverbed-manifest.json"), ROOT),
                   "label": f"{label} W={W} presence={presence}",
                   "W": W, "presence": presence, "corpus_sd": sd,
                   "n_readers": len(m.readers),
                   "signal_nights": signal, "null_nights": nulls,
                   "x_map": x_map, "family_of": fam_of}
            A = leg_A(win, m, signal, SEED)
            A_up = leg_A(win, m, signal, SEED + 2, direction="up",
                         mid_anchor=True)
            A_start = leg_A(win, m, signal, SEED + 4, pos_ref="start")
            res["A"] = {k: v for k, v in A.items() if k != "events"}
            res["A_up_mirror"] = {k: v for k, v in A_up.items() if k != "events"}
            res["A_start_ref_sensitivity"] = {k: v for k, v in A_start.items()
                                              if k != "events"}
            res["D"] = leg_D(win, m, signal, nulls)
            res["P"] = leg_P(win, m, sd, W, signal)
            res["S"] = leg_S(win, m, signal, x_map, False, SEED)
            traj = {}
            for n in signal:
                wn = win[n]
                traj[n] = {"Rt": [None if not np.isfinite(v)
                                  else round(float(v), 4) for v in wn["Rt"]],
                           "boundaries": [t["boundary"] for t in
                                          strata_transitions(m, n)
                                          if t["kind"] == "signal"],
                           "x": x_map[n]}
            res["trajectory"] = traj
            res["null_night_scores"] = {
                n: {r: (None if np.all(np.isnan(win[n]["rho"][r])) else
                        round(float(np.nanmedian(win[n]["rho"][r])), 4))
                    for r in win[n]["readers"]} for n in nulls}
            res["mean_d_by_phase"] = drift_by_phase(win, m, signal)
            res["runtime_s"] = round(time.time() - t0, 1)
            os.makedirs(outdir, exist_ok=True)
            out = os.path.join(outdir, f"{label}.W{W:02d}.{presence}.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=1)
            filed.append(out)
            print(f"[legs] {label} W={W} {presence}: corpus_sd={sd:.4f} "
                  f"({res['runtime_s']}s) -> {os.path.relpath(out, ROOT)}")
    return filed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", action="append", required=True,
                    help="riverbed corpus dir (repeatable)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--W", type=int, nargs="+", default=[12, 8, 16])
    ap.add_argument("--presence", nargs="+",
                    default=["canonical", "actual"])
    args = ap.parse_args()
    for c in args.corpus:
        run_legs(c, args.out, args.W, args.presence)


if __name__ == "__main__":
    main()
