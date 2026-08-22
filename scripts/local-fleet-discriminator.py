#!/usr/bin/env python3
"""The discriminator test — does the elephant's field SEPARATE the cases?

Experiment A (done): common-shift probe -> COH 0.9931, v* -0.014 (pure
translation, no shape). Experiment B (this): a discordant probe where the
local fleet naturally DISAGREES. If the instrument works, B should show
LOWER cohesion and a NONZERO v* — the school's shape appearing where the
fleet differentiates. Same six sounders, same machinery, different water.
"""
import json, urllib.request, time, math

OLLAMA = "http://127.0.0.1:11434/api/generate"
PROBE = ("Answer in exactly one sentence, no analysis: is a room's 'warmth' "
         "best measured by people's mood, their volume, their honesty, or "
         "their presence? Pick one and answer with just that word.")
FLEET = [
    ("granite3.1-dense:2b", "Wesley-class"),
    ("mistral:7b", "deckhand"),
    ("phi4-mini:latest", "ensign"),
    ("gemma3:4b", "lookout"),
    ("Liquid-LFM2.5-2.6B", "boat brain"),
]
DIALS = ["mood", "volume", "earnestness", "cynicism", "joke_landing", "panic", "presence"]
DIAL_LEX = {
    "mood":        (["mood", "feeling", "warm", "happy", "glad", "emotion"], []),
    "volume":      (["volume", "loud", "loudness", "sound", "decibel"], []),
    "earnestness": (["honesty", "honest", "truth", "sincere", "earnest"], []),
    "cynicism":    ([], []),
    "joke_landing":([], []),
    "panic":       ([], []),
    "presence":    (["presence", "showing", "attendance", "there", "participation"], []),
}

def ask(model, timeout=90):
    body = json.dumps({"model": model, "prompt": PROBE, "stream": False,
                       "options": {"num_predict": 40, "temperature": 0.7}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["response"]

def score(text):
    t = text.lower()
    out = {}
    for dial, (pos, neg) in DIAL_LEX.items():
        p = sum(t.count(w) for w in pos)
        out[dial] = min(1.0, p) if p else 0.0
    return out

def vmf_stats(readings):
    mean = [sum(r[d] for r in readings) / len(readings) for d in DIALS]
    r_bar = math.sqrt(sum(m * m for m in mean))
    kappa = r_bar * (7 - 1) / (1 - r_bar * r_bar) if r_bar < 1 else 99.0
    return {"mu": {d: round(m, 4) for d, m in zip(DIALS, mean)},
            "r_bar": round(r_bar, 4), "kappa": round(kappa, 2)}

def main():
    print(f"probe: {PROBE}\n")
    readings, responses = [], {}
    for model, role in FLEET:
        try:
            text = ask(model)
            readings.append(score(text))
            responses[model] = {"role": role, "text": text.strip()[:120]}
            print(f"[{role:12s}] {text.strip()[:80]}")
        except Exception as e:
            print(f"[{role:12s}] FAILED: {e}")
    if len(readings) < 3:
        print("too few readings"); return
    stats = vmf_stats(readings)
    vol = stats["mu"]["volume"]; pres = stats["mu"]["presence"]
    vstar = round(vol - pres, 4)
    spread = [abs(r["volume"] - r["presence"]) for r in readings]
    coh = round(1.0 - (sum(spread) / len(spread)) / 2, 4)
    out = {
        "experiment": "local-fleet-discriminator",
        "ts": time.time(), "probe": PROBE, "fleet": responses,
        "field": {**stats, "vstar": vstar, "cohesion": coh},
    }
    path = "/home/eileen/projects/elephant/data/slope/local-fleet-discriminator.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== DISCRIMINATOR FIELD ===")
    print(f"mu: {stats['mu']}")
    print(f"r_bar={stats['r_bar']}  kappa={stats['kappa']}")
    print(f"v* = {vstar}   COH = {coh}")
    print(f"\nCONTRAST with experiment A: v* -0.014/COH 0.993 (synchronized) vs "
          f"v* {vstar}/COH {coh} (discordant)")
    verdict = ("SEPARATES: cohesion drops + v* lights up when the fleet differs"
               if (coh < 0.95 and abs(vstar) > 0.05)
               else "DOES NOT SEPARATE — check the instrument")
    print(f"VERDICT: {verdict}")
    print(f"filed: {path}")

if __name__ == "__main__":
    main()
