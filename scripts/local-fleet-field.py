#!/usr/bin/env python3
"""The local fleet reads the room — the hundred-boats doctrine, literal.

Six local models on the RTX 4050 = six sounders on the same tack. Each answers
the same probe; we score the 7 dials from each response (the elephant's own
RoomField dials), then run the field machinery: warmth, kappa, v* (volume/
presence — the REG-1 axis), and COH (cohesion). The school, if it forms, is
the shape no single model contains.

Hardware: local Ollama (http://127.0.0.1:11434) — granite3.1-dense:2b,
mistral:7b, qwen3:8b, phi4-mini, gemma3:4b, Liquid-LFM2.5-2.6B.
"""
import json, urllib.request, time, math

OLLAMA = "http://127.0.0.1:11434/api/generate"
PROBE = ("Answer in exactly one sentence, no analysis: when everyone in a room "
         "moves together at once, what is that? Give just your answer.")

FLEET = [
    ("granite3.1-dense:2b", "Wesley-class"),
    ("mistral:7b", "deckhand"),
    ("qwen3:8b", "nav"),
    ("phi4-mini:latest", "ensign"),
    ("gemma3:4b", "lookout"),
    ("Liquid-LFM2.5-2.6B", "boat brain"),
]

DIALS = ["mood", "volume", "earnestness", "cynicism", "joke_landing", "panic", "presence"]

# Simple lexical dial scoring: each dial has warm/positive and cold/negative cues.
DIAL_LEX = {
    "mood":        (["warm", "together", "glad", "good", "bright", "yes", "mov"], ["cold", "alone", "dark", "no ", "fear"]),
    "volume":      (["loud", "all", "every", "together", "one", "move"], ["quiet", "still", "single"]),
    "earnestness": (["real", "honest", "truly", "actually", "genuine", "is"], ["pretend", "fake", "maybe"]),
    "cynicism":    (["just", "only", "nothing", "mere", "claim", "sure"], ["real", "honest", "truly"]),
    "joke_landing":(["funny", "ha", "laugh", "joke", "irony"], []),
    "panic":       (["fear", "danger", "stampede", "panic", "chaos", "wrong"], ["calm", "fine", "safe"]),
    "presence":    (["together", "everyone", "all", "room", "move", "here"], ["alone", "nobody", "empty"]),
}

def ask(model: str, timeout=90) -> str:
    body = json.dumps({"model": model, "prompt": PROBE, "stream": False,
                       "options": {"num_predict": 120, "temperature": 0.4}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["response"]

def score(text: str) -> dict:
    t = text.lower()
    out = {}
    for dial, (pos, neg) in DIAL_LEX.items():
        p = sum(t.count(w) for w in pos)
        n = sum(t.count(w) for w in neg)
        out[dial] = 0.5 + 0.5 * (p - n) / max(1, p + n) if (p + n) else 0.5
    return out

def vmf_stats(readings: list) -> dict:
    """Mean direction + concentration-ish on the dial simplex (7-dim)."""
    mean = [sum(r[d] for r in readings) / len(readings) for d in DIALS]
    r_bar = math.sqrt(sum(m * m for m in mean))
    kappa = r_bar * (7 - 1) / (1 - r_bar * r_bar) if r_bar < 1 else 99.0
    return {"mu": {d: round(m, 4) for d, m in zip(DIALS, mean)},
            "r_bar": round(r_bar, 4), "kappa": round(kappa, 2)}

def main():
    print(f"probe: {PROBE}\n")
    readings, responses = [], {}
    for model, role in FLEET:
        t0 = time.time()
        try:
            text = ask(model)
            dt = time.time() - t0
            d = score(text)
            readings.append(d)
            responses[model] = {"role": role, "text": text.strip()[:220],
                                "time_s": round(dt, 2), "dials": d}
            print(f"[{role:12s}] {model:24s} {dt:5.1f}s  {text.strip()[:90]}")
        except Exception as e:
            print(f"[{role:12s}] {model:24s} FAILED: {e}")
    if not readings:
        print("no readings — fleet asleep"); return
    stats = vmf_stats(readings)
    vol = stats["mu"]["volume"]; pres = stats["mu"]["presence"]
    vstar = round(vol - pres, 4)
    spread = [abs(r["volume"] - r["presence"]) for r in readings]
    coh = round(1.0 - (sum(spread) / len(spread)) / 2, 4)  # low spread = high cohesion
    out = {
        "experiment": "local-fleet-reads-the-room",
        "ts": time.time(),
        "probe": PROBE,
        "fleet": responses,
        "field": {**stats, "vstar": vstar, "cohesion": coh,
                  "warmth": round(stats["mu"]["mood"], 4)},
    }
    path = "/home/eileen/projects/elephant/data/slope/local-fleet-field.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== THE ROOM'S FIELD ===")
    print(f"mu: {stats['mu']}")
    print(f"r_bar={stats['r_bar']}  kappa={stats['kappa']}")
    print(f"v* (volume-presence) = {vstar}   <-- the REG-1 room-energy axis")
    print(f"COH (cohesion)       = {coh}")
    print(f"filed: {path}")

if __name__ == "__main__":
    main()
