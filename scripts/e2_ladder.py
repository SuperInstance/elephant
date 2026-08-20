"""E2 calibration ladder — ATTEMPT 3 (canonical presence; addenda 1+2).

Attempt 1 (as registered) failed; its raw results are preserved at
data/e2/e2-ladder-attempt1.json and the diagnosis + rebuild are registered
in the dissertation repo (ADDENDUM-LADDER-REBUILD-2026-08-19.md) BEFORE
this run. Changes from attempt 1, per the addendum:

  * canonical presence everywhere (truth AND estimator): each reader's
    replay substitutes the per-speak mean attendee interaction count for
    their own — participation-deconfounded by construction (λ=0 ⇒ ratio
    exactly 0 at both truth and estimate);
  * planting extension (addendum 2, B3): any rung unreachable by the
    λ-family within [0, 12] is planted via the DIRECTIONAL-GAIN family
    (λ_dir in [0, 1]: gain camps on the top-2 dials by stable-deviation /
    flip-displacement ratio, vibes pushed to the warm side of the camp
    dial). λ_vibe (addendum 1 A4) is retired: structurally capped by
    acclimation.

Thresholds unchanged: recover every rung within ±0.1 (rep 0 decides; reps
1-2 are stability flags). PASS: E-seg primary; else E-cont (flagged);
else second consecutive ladder failure — the measurement is killed and no
field number is filed.

All numbers are FIXTURES ("on fixtures"). Seeds as registered.
Run:  python3 scripts/e2_ladder.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.e2_instrument import (COLD_ENTRY, D, FIELD_NIGHTS, HI, LO,
                                   NIGHT_SPECS, Night, PRIMARY_NIGHTS,
                                   cell_vecs, corpus_sd, replay_readings)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSONAS = os.path.join(ROOT, "data", "e2", "e2-personas.json")
OUT = os.path.join(ROOT, "data", "e2", "e2-ladder-results.json")

RUNGS = [0.0, 0.15, 0.3, 0.6, 0.9]
TOL = 0.1
BASE_SEED = 20260819
PRESENCE = "canonical"
ATTEMPT = 3
TEMPLATE = sorted(FIELD_NIGHTS)


def existing_personas():
    doc = json.load(open(PERSONAS, encoding="utf-8"))
    return doc["existing_personas_frozen"], doc["cast_mean"]


def draw_population(seed):
    """15 synthetic readers, one per field-template slot, drawn from the
    field distribution (mirrors scripts/e2_personas.py's sampler)."""
    frozen, _ = existing_personas()
    names = sorted(frozen)
    rng = np.random.default_rng(seed)
    chars = np.array([frozen[n]["charisma"] for n in names])
    rates = np.array([frozen[n]["acclimation_rate"] for n in names])
    vibes = np.stack([np.array(frozen[n]["vibe_start"]) for n in names])
    sd_char = float(chars.std(ddof=1))
    sd_rate = float(rates.std(ddof=1))
    sd_vibe = vibes.std(axis=0, ddof=1)
    pop = {}
    for slot in TEMPLATE:
        arch = frozen[names[int(rng.integers(len(names)))]]
        ch = float(np.clip(arch["charisma"] + rng.normal(0, sd_char),
                           chars.min(), chars.max()))
        ra = float(np.clip(arch["acclimation_rate"] + rng.normal(0, sd_rate),
                           rates.min(), rates.max()))
        w = np.asarray(arch["dial_weights"]) * np.exp(rng.normal(0, 0.15, D))
        w = w / w.sum()
        vb = np.clip(np.asarray(arch["vibe_start"]) + rng.normal(0, sd_vibe),
                     LO, HI)
        pop[slot] = {"dial_weights": w, "vibe_start": vb,
                     "charisma": ch, "acclimation_rate": ra}
    return pop


def scale_all(drawn, cast_mean, lam):
    """params(λ) = cast_mean + λ·(draw − cast_mean) for every parameter."""
    out = {}
    cm_w = np.asarray(cast_mean["dial_weights"], float)
    cm_v = np.asarray(cast_mean["vibe_start"], float)
    for slot, p in drawn.items():
        w = cm_w + lam * (p["dial_weights"] - cm_w)
        w = w / w.sum()
        vb = np.clip(cm_v + lam * (p["vibe_start"] - cm_v), LO, HI)
        ch = max(float(cast_mean["charisma"]
                       + lam * (p["charisma"] - cast_mean["charisma"])), 0.01)
        ra = max(float(cast_mean["acclimation_rate"]
                       + lam * (p["acclimation_rate"] - cast_mean["acclimation_rate"])),
                 0.01)
        out[slot] = {"dial_weights": w, "vibe_start": vb,
                     "charisma": ch, "acclimation_rate": ra}
    return out


def corpus_geometry(nights):
    """Per-dial stable deviation delta (warm stratum mean vs dial center)
    and flip displacement Delta (cynical minus warm), from the A/S1 logs
    (deterministic; committed into the results)."""
    from scripts.e2_instrument import CENTER as C
    warm, cyn = [], []
    for tag in ("A", "S1"):
        n = nights[tag]
        warm += [r["field_raw_after"] for r in n.speaks if r["seq"] <= 19]
        cyn += [r["field_raw_after"] for r in n.speaks if r["seq"] >= 20]
    warm, cyn = np.array(warm, float), np.array(cyn, float)
    delta = np.abs(warm.mean(0) - C)
    Delta = np.abs(cyn.mean(0) - warm.mean(0))
    return delta, Delta


def scale_directional(lam, cast_mean, delta, Delta, warm_mean, template):
    """Addendum 2 B3: gain camps on the top-2 dials by delta/max(Delta,0.05)
    restricted to delta>0.08; vibes pushed to the warm side of the camp
    dial; charisma/acclimation at cast mean. Out-of-field-distribution
    fixture construct — never enters the field number."""
    ratio = delta / np.maximum(Delta, 0.05)
    ratio = ratio * (delta > 0.08)
    d1, d2 = [int(x) for x in np.argsort(-ratio)[:2]]
    cm_w = np.asarray(cast_mean["dial_weights"], float)
    cm_v = np.asarray(cast_mean["vibe_start"], float)
    out = {}
    for i, slot in enumerate(template):
        pick = d1 if i % 2 == 0 else d2
        w = cm_w * (1.0 - lam)
        w[pick] += lam * 0.8
        target = float(np.clip(warm_mean[pick], LO[pick], HI[pick]))
        offset = 0.6 if delta[pick] > 0.3 else 0.2
        vb = cm_v.copy()
        vb[pick] = np.clip(cm_v[pick]
                           + lam * (target + offset * np.sign(target or 1.0)
                                    - cm_v[pick]), LO[pick], HI[pick])
        out[slot] = {"dial_weights": w / w.sum(), "vibe_start": vb,
                     "charisma": float(cast_mean["charisma"]),
                     "acclimation_rate": float(cast_mean["acclimation_rate"])}
    return out, (d1, d2)


def reader_replay(params, night, slot):
    cold = night.name in COLD_ENTRY.get(slot, [])
    start = night.first_speak_seq(slot) if cold else None
    return replay_readings(params, night.speaks, slot, start,
                           canon_n=night.canon_n)


def transition_vals(seq_vecs, night_name, sd):
    strata = NIGHT_SPECS[night_name][1]
    vals = []
    for (l0, lo0, hi0, k0), (l1, lo1, hi1, k1) in zip(strata, strata[1:]):
        a, b = cell_vecs(seq_vecs, lo0, hi0), cell_vecs(seq_vecs, lo1, hi1)
        if len(a) and len(b) and k0 != "null" and k1 != "null":
            vals.append(float(np.linalg.norm(b.mean(0) - a.mean(0))) / sd)
    return vals


def truth_of(params_by_slot, nights, sd, s5):
    """TRUE spread/drift/ratio: canonical-S5 stable-room baselines
    (reader-intrinsic by construction) ÷ signal-transition drift."""
    anchors, drifts = [], []
    for slot in TEMPLATE:
        p = params_by_slot[slot]
        sv = reader_replay(p, s5, slot)
        anchors.append(np.mean([v for _, v in sv], axis=0))
        vals = []
        for night in FIELD_NIGHTS[slot]:
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


def bisect_family(scaler, drawn, cast_mean, nights, sd, s5, rung,
                  lam_hi, tol=0.005):
    grid = np.linspace(0.0, lam_hi, 25)
    vals = [truth_of(scaler(drawn, cast_mean, x), nights, sd, s5)["ratio"]
            for x in grid]
    if rung > vals[-1]:
        return None, None, None
    i = int(np.argmax(np.array(vals) >= rung))
    lo, hi = grid[max(i - 1, 0)], grid[i]
    for _ in range(40):
        mid = (lo + hi) / 2
        t = truth_of(scaler(drawn, cast_mean, mid), nights, sd, s5)
        if abs(t["ratio"] - rung) <= tol:
            return mid, scaler(drawn, cast_mean, mid), t
        if t["ratio"] < rung:
            lo = mid
        else:
            hi = mid
    lam = (lo + hi) / 2
    p = scaler(drawn, cast_mean, lam)
    return lam, p, truth_of(p, nights, sd, s5)


def calibrate(drawn, cast_mean, nights, sd, s5, rung, geom):
    """λ-family [0,12] first; directional-gain family [0,1] (addendum 2)."""
    if rung == 0.0:
        p = scale_all(drawn, cast_mean, 0.0)
        return {"family": "lambda", "lambda": 0.0, "params": p,
                "truth": truth_of(p, nights, sd, s5)}
    lam, p, t = bisect_family(scale_all, drawn, cast_mean, nights, sd, s5,
                              rung, 12.0)
    if p is not None:
        return {"family": "lambda", "lambda": lam, "params": p, "truth": t}
    delta, Delta, warm_mean = geom

    def scaler(d, cm, lam):
        return scale_directional(lam, cm, delta, Delta, warm_mean, TEMPLATE)[0]

    lam, p, t = bisect_family(scaler, drawn, cast_mean, nights, sd, s5,
                              rung, 1.0)
    if p is not None:
        _, picks = scale_directional(lam, cast_mean, delta, Delta, warm_mean,
                                     TEMPLATE)
        return {"family": "directional", "lambda": lam, "params": p,
                "truth": t, "picks": picks}
    return {"family": "unreachable", "lambda": None, "params": None, "truth": None}


def main():
    from scripts.e2_instrument import Measurement

    nights = {n: Night(n) for n in PRIMARY_NIGHTS}
    sd, _ = corpus_sd(list(nights.values()))
    s5 = nights["S5"]
    _, cast_mean = existing_personas()
    print(f"[e2-ladder attempt 2] presence={PRESENCE} corpus_sd={sd:.4f}")

    delta, Delta = corpus_geometry(nights)
    warm_mean = np.mean([r["field_raw_after"] for n in nights.values()
                         for r in n.speaks[:20]], axis=0)  # S5 warm reference
    geom = (delta, Delta, warm_mean)
    results = {"attempt": ATTEMPT, "presence": PRESENCE, "rungs": RUNGS,
               "tolerance": TOL, "corpus_sd": sd,
               "corpus_geometry": {"delta": delta.tolist(),
                                   "Delta": Delta.tolist()},
               "repetitions": []}

    for ri, rung in enumerate(RUNGS):
        for rep in range(3):
            seed = BASE_SEED + 1000 * ri + rep
            drawn = draw_population(seed)
            cal = calibrate(drawn, cast_mean, nights, sd, s5, rung, geom)
            if cal["params"] is None:
                results["repetitions"].append(
                    {"rung": rung, "rep": rep, "seed": seed,
                     "error": "rung unreachable by both planting families"})
                print(f"rung {rung:.2f} rep {rep}: UNREACHABLE (both families)")
                continue
            readers_nights = {
                slot: {"params": cal["params"][slot],
                       "nights": {n: slot for n in FIELD_NIGHTS[slot]},
                       "cold": COLD_ENTRY.get(slot, [])}
                for slot in TEMPLATE}
            m = Measurement(readers_nights, sd, presence=PRESENCE)
            est_seg, est_cont = m.ratio_seg(), m.ratio_cont()
            boot = m.bootstrap(B=2000)
            rec = {
                "rung": rung, "rep": rep, "seed": seed,
                "family": cal["family"], "lambda": cal["lambda"],
                "picks": cal.get("picks"),
                "true_ratio": cal["truth"]["ratio"],
                "true_spread": cal["truth"]["spread"],
                "true_drift": cal["truth"]["drift"],
                "est_seg": est_seg, "est_seg_ci": list(boot["seg_ci"]),
                "est_cont": est_cont, "est_cont_ci": list(boot["cont_ci"]),
                "seg_err": est_seg - rung, "cont_err": est_cont - rung,
                "seg_pass": abs(est_seg - rung) <= TOL,
                "cont_pass": abs(est_cont - rung) <= TOL,
            }
            results["repetitions"].append(rec)
            print(f"rung {rung:.2f} rep {rep} [{cal['family']} "
                  f"λ={cal['lambda']:.3f}] truth={cal['truth']['ratio']:.4f} | "
                  f"E-seg={est_seg:.4f} (err {est_seg - rung:+.4f}) "
                  f"E-cont={est_cont:.4f} (err {est_cont - rung:+.4f})")

    rep0 = [r for r in results["repetitions"] if r.get("rep") == 0]
    seg_pass = len(rep0) == len(RUNGS) and all(r.get("seg_pass") for r in rep0)
    cont_pass = len(rep0) == len(RUNGS) and all(r.get("cont_pass") for r in rep0)
    if seg_pass:
        primary, verdict = "E-seg", (f"LADDER (attempt {ATTEMPT}) PASSES — primary "
                                     "estimator: E-seg, canonical presence")
    elif cont_pass:
        primary, verdict = "E-cont", (f"LADDER (attempt {ATTEMPT}) PASSES — E-seg failed "
                                      "a rung; primary: E-cont (flagged)")
    else:
        primary, verdict = None, ("LADDER FAILS A SECOND TIME — the measurement "
                                  "is killed: the field cannot currently "
                                  "measure its own antecedent; no field number")
    results["seg_pass"] = seg_pass
    results["cont_pass"] = cont_pass
    results["primary"] = primary
    results["verdict"] = verdict
    print("\n" + verdict)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"[e2-ladder] results -> {OUT}")


if __name__ == "__main__":
    main()
