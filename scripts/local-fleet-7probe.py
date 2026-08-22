#!/usr/bin/env python3
"""The 7-probe room-field — the elephant's actual RoomField pipeline, fed by
real local hardware. Six sounders on the RTX 4050 answer seven dial-probes
(mood, volume, earnestness, cynicism, joke_landing, panic, presence); each
response is scored on its dial; the result is a true 6-model × 7-dial matrix
that the elephant's field machinery (warmth, kappa, v*, COH) reads.

This is the hundred-boats doctrine made literal: the fleet IS the room, the
sounders ARE the boats, and the field is what no single boat contains.
"""
import json, urllib.request, time, math, re

OLLAMA = "http://127.0.0.1:11434/api/generate"
FLEET = [
    ("granite3.1-dense:2b", "Wesley-class"),
    ("mistral:7b", "deckhand"),
    ("phi4-mini:latest", "ensign"),
    ("gemma3:4b", "lookout"),
    ("Liquid-LFM2.5-2.6B", "boat brain"),
]
DIALS = ["mood", "volume", "earnestness", "cynicism", "joke_landing", "panic", "presence"]
# Each probe asks the model to rate the room on a 0-10 scale for one dial.
PROBES = {
    "mood":        "On a scale of 0 to 10, rate the room's overall mood right now (0=dark, 10=bright). Answer with just a number.",
    "volume":      "On a scale of 0 to 10, how much volume or energy is in this room right now (0=quiet, 10=deafening)? Answer with just a number.",
    "earnestness": "On a scale of 0 to 10, how honest or sincere is the conversation in this room (0=performative, 10=raw)? Answer with just a number.",
    "cynicism":    "On a scale of 0 to 10, how cynical or skeptical is the room's tone (0=trusting, 10=scornful)? Answer with just a number.",
    "joke_landing":"On a scale of 0 to 10, how well are jokes landing in this room right now (0=bombed, 10=roaring)? Answer with just a number.",
    "panic":       "On a scale of 0 to 10, how much panic or tension is in this room (0=calm, 10=stampede)? Answer with just a number.",
    "presence":    "On a scale of 0 to 10, how present and engaged are the people in this room (0=checked out, 10=all here)? Answer with just a number.",
}

def ask(model, prompt, timeout=60):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"num_predict": 8, "temperature": 0.2}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["response"]

def extract_number(text):
    m = re.search(r"(\d{1,2})(?:\.\d)?", text or "")
    if not m:
        return None
    v = int(m.group(1))
    return max(0, min(10, v))

def vmf_stats(matrix):
    """matrix: list of dicts (per-model dial readings 0..1)."""
    n = len(matrix)
    mean = [sum(m[d] for m in matrix) / n for d in DIALS]
    r_bar = math.sqrt(sum(x * x for x in mean))
    kappa = r_bar * (len(DIALS) - 1) / (1 - r_bar * r_bar) if r_bar < 1 else 99.0
    return {"mu": {d: round(x, 4) for d, x in zip(DIALS, mean)},
            "r_bar": round(r_bar, 4), "kappa": round(kappa, 2)}

def main():
    print("7-probe room-field — six sounders, seven dials, one room\n")
    matrix, raw = [], {}
    for model, role in FLEET:
        row, raw_row = {}, {}
        for dial in DIALS:
            try:
                text = ask(model, PROBES[dial])
                v = extract_number(text)
                raw_row[dial] = {"answer": text.strip()[:60], "raw": v}
                row[dial] = (v / 10.0) if v is not None else 0.5
            except Exception as e:
                raw_row[dial] = {"error": str(e)}
                row[dial] = 0.5
        matrix.append(row)
        raw[model] = {"role": role, "dials": raw_row}
        print(f"[{role:12s}] " + " ".join(f"{d[0]}={row[d]:.1f}" for d in DIALS))
    stats = vmf_stats(matrix)
    vol, pres = stats["mu"]["volume"], stats["mu"]["presence"]
    vstar = round(vol - pres, 4)
    spreads = [abs(m["volume"] - m["presence"]) for m in matrix]
    coh = round(1.0 - (sum(spreads) / len(spreads)) / 2, 4)
    out = {"experiment": "local-fleet-7probe-roomfield", "ts": time.time(),
           "fleet": raw, "matrix": matrix,
           "field": {**stats, "vstar": vstar, "cohesion": coh,
                     "warmth": round(stats["mu"]["mood"], 4)}}
    path = "/home/eileen/projects/elephant/data/slope/local-fleet-7probe.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== THE ROOM-FIELD (7-probe, real readings) ===")
    print(f"mu: {stats['mu']}")
    print(f"r_bar={stats['r_bar']}  kappa={stats['kappa']}")
    print(f"v* = {vstar}   COH = {coh}   warmth = {stats['mu']['mood']}")
    print(f"filed: {path}")

if __name__ == "__main__":
    main()
