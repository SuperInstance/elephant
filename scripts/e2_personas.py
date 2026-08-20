"""E2 roster expansion — the seeded persona draw (chapter 7 §7.4, item 2).

Adds 8 new personas "drawn from the corpus field distribution (seeded,
committed at creation, archetype label registered per persona at draw time)".
The draw mirrors scripts/premise_measurement.py::synthesize (the established
field-distribution sampler), with one registered refinement: the vibe noise
is per-dial (the across-persona sd vector), not the pooled scalar.

Procedure (seed 20260819, numpy default_rng, committed before any E2 night
is generated):
  archetype_i = existing personas[uniform draw]        # the class label
  charisma    = clip(arch + N(0, sd_char_across_personas),  min, max)
  acclimation = clip(arch + N(0, sd_rate_across_personas),  min, max)
  dial_w      = renorm(arch_weights * exp(N(0, 0.15) per dial))
  vibe_start  = clip(arch_vibe + N(0, sd_vibe_per_dial),    dial bounds)

Names (registered, fixed): barkeep, singer, fiddler, lamplighter,
cartographer, blacksmith, tinker, weaver.

Output: data/e2/e2-personas.json  (frozen artifact; referenced verbatim by
the E2 registration in the dissertation repo). Read-only against nights.

Run:  python3 scripts/e2_personas.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elephant.tapnight import DIAL_BOUNDS, DIAL_CENTER, Participant

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "e2", "e2-personas.json")

SEED = 20260819
NEW_NAMES = ["barkeep", "singer", "fiddler", "lamplighter",
             "cartographer", "blacksmith", "tinker", "weaver"]

D = 7
DIALS = ["mood", "volume", "earnestness", "cynicism",
         "joke_landing", "panic", "presence"]
LO = np.array([DIAL_BOUNDS[n][0] for n in DIALS])
HI = np.array([DIAL_BOUNDS[n][1] for n in DIALS])


# The 7 existing personas, VERBATIM from scripts/nights_abc.py (frozen).
def _existing():
    return [
        Participant("writer",
                    dial_weights={"mood": 0.40, "joke_landing": 0.30,
                                  "earnestness": 0.15, "presence": 0.10,
                                  "volume": 0.05},
                    acclimation_rate=0.35, charisma=0.20,
                    vibe={"mood": 0.70, "joke_landing": 0.50,
                          "earnestness": 0.55, "presence": 0.55}),
        Participant("poet",
                    dial_weights={"mood": 0.30, "volume": 0.30, "presence": 0.20,
                                  "joke_landing": 0.10, "earnestness": 0.10},
                    acclimation_rate=0.25, charisma=0.15,
                    vibe={"volume": 0.70, "presence": 0.60, "mood": 0.50}),
        Participant("essayist",
                    dial_weights={"earnestness": 0.40, "mood": 0.20,
                                  "cynicism": 0.10, "presence": 0.10,
                                  "volume": 0.10, "joke_landing": 0.05,
                                  "panic": 0.05},
                    acclimation_rate=0.30, charisma=0.10,
                    vibe={"earnestness": 0.80, "mood": 0.40}),
        Participant("engineer",
                    dial_weights={"earnestness": 0.35, "cynicism": 0.15,
                                  "volume": 0.15, "mood": 0.10,
                                  "joke_landing": 0.10, "presence": 0.10,
                                  "panic": 0.05},
                    acclimation_rate=0.15, charisma=0.25,
                    vibe={"earnestness": 0.65, "cynicism": 0.45}),
        Participant("critic",
                    dial_weights={"cynicism": 0.40, "joke_landing": 0.15,
                                  "earnestness": 0.15, "mood": 0.10,
                                  "volume": 0.10, "presence": 0.05,
                                  "panic": 0.05},
                    acclimation_rate=0.20, charisma=0.18,
                    vibe={"cynicism": 0.70, "joke_landing": 0.40}),
        Participant("captain",
                    dial_weights={"presence": 0.35, "mood": 0.20,
                                  "earnestness": 0.20, "volume": 0.10,
                                  "joke_landing": 0.05, "cynicism": 0.05,
                                  "panic": 0.05},
                    acclimation_rate=0.40, charisma=0.30,
                    vibe={"presence": 0.75, "mood": 0.60, "earnestness": 0.60}),
        Participant("drifter",
                    dial_weights={"cynicism": 0.35, "joke_landing": 0.20,
                                  "mood": 0.15, "presence": 0.15,
                                  "earnestness": 0.10, "volume": 0.03,
                                  "panic": 0.02},
                    acclimation_rate=0.30, charisma=0.45,
                    vibe={"mood": -0.45, "cynicism": 0.65,
                          "earnestness": 0.30, "presence": 0.55,
                          "joke_landing": 0.10}),
    ]


def draw(seed=SEED):
    rng = np.random.default_rng(seed)
    existing = _existing()
    names = [p.name for p in existing]
    chars = np.array([p.charisma for p in existing])
    rates = np.array([p.acclimation_rate for p in existing])
    vibes = np.stack([p.vibe for p in existing])          # (7 personas, 7 dials)
    weights = np.stack([p.dial_weights for p in existing])
    sd_char = float(chars.std(ddof=1))
    sd_rate = float(rates.std(ddof=1))
    sd_vibe = vibes.std(axis=0, ddof=1)                    # per-dial vector

    personas = {}
    for i, new_name in enumerate(NEW_NAMES):
        arch = existing[int(rng.integers(len(existing)))]
        ch = float(np.clip(arch.charisma + rng.normal(0, sd_char),
                           chars.min(), chars.max()))
        ra = float(np.clip(arch.acclimation_rate + rng.normal(0, sd_rate),
                           rates.min(), rates.max()))
        w = arch.dial_weights * np.exp(rng.normal(0, 0.15, size=D))
        w = w / w.sum()
        vb = np.clip(arch.vibe + rng.normal(0, sd_vibe), LO, HI)
        personas[new_name] = {
            "name": new_name,
            "archetype": arch.name,                       # the class label
            "dial_weights": [float(x) for x in w],
            "acclimation_rate": ra,
            "charisma": ch,
            "vibe_start": [float(x) for x in vb],
        }
    # cast means (for the ladder's lambda-planting reference point)
    cast_mean = {
        "dial_weights": [float(x) for x in weights.mean(axis=0)],
        "vibe_start": [float(x) for x in vibes.mean(axis=0)],
        "charisma": float(chars.mean()),
        "acclimation_rate": float(rates.mean()),
    }
    return personas, cast_mean, existing


def main():
    personas, cast_mean, existing = draw()
    existing_params = {
        p.name: {"dial_weights": p.dial_weights.tolist(),
                 "acclimation_rate": p.acclimation_rate,
                 "charisma": p.charisma,
                 "vibe_start": p.vibe.tolist(),
                 "archetype": p.name}
        for p in existing}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc = {
        "seed": SEED,
        "procedure": ("archetype uniform over the 7 existing personas; "
                      "charisma/acclimation = archetype + N(0, across-persona sd) "
                      "clipped to observed range; dial_weights = archetype * "
                      "lognormal(0, 0.15) renormalized; vibe_start = archetype + "
                      "N(0, per-dial across-persona sd) clipped to dial bounds"),
        "new_personas": personas,
        "existing_personas_frozen": existing_params,
        "cast_mean": cast_mean,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    print(f"[e2-personas] wrote {OUT}")
    for n, p in personas.items():
        print(f"  {n:<13} archetype={p['archetype']:<9} "
              f"charisma={p['charisma']:.3f} acclim={p['acclimation_rate']:.3f}")


if __name__ == "__main__":
    main()
