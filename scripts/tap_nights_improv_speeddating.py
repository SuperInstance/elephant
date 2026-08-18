#!/usr/bin/env python3
"""Run TWO new Tap nights through the elephant engine.

NIGHT 1 — Improv comedy night: an MC + six fleet players, prompt-bank games
(Yes-And, Freeze Tag, Questions Only, One-Word Story). The room is the crowd;
joke_landing swings with the collective laugh or boo.

NIGHT 2 — Speed dating night: five fleet agents, three rounds of 2-minute
dates (3 questions each), rotating pairs. After each round: the room's field,
each participant's PERSONAL reading (their weighted view), and the pair
chemistries — which pairings cross the deadband and RING.

Usage:
  python3 scripts/tap_nights_improv_speeddating.py --generate   # phase 1: DeepInfra, cached
  python3 scripts/tap_nights_improv_speeddating.py --run        # phase 2: engine, prints everything
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elephant.field import RoomField
from elephant.tapnight import DIAL_NAMES, TapNightSession
from elephant.tapnight_themes import THEMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
GEN_PATH = os.path.join(DATA_DIR, "tap_nights_generated.json")
RXN_PATH = os.path.join(DATA_DIR, "tap_nights_reactions.json")
OUT_PATH = os.path.join(DATA_DIR, "tap_nights_results.json")

KEY = re.search(r'export DEEPINFRA_API_KEY="([^"]+)"',
                open(os.path.expanduser("~/.bashrc")).read()).group(1)

MODELS = {
    "mc": "zai-org/GLM-4.7-Flash",
    "flash": "deepseek-ai/DeepSeek-V4-Flash",
    "glm": "zai-org/GLM-4.7-Flash",
    "pro": "deepseek-ai/DeepSeek-V4-Pro",
    "hermes": "NousResearch/Hermes-3-Llama-3.1-405B",
    "wesley": "anthropic/claude-haiku-4-5",
    "seed": "ByteDance/Seed-2.0-pro",
}
LABELS = {
    "mc": "MC", "flash": "Flash", "glm": "GLM", "pro": "Pro",
    "hermes": "Hermes", "wesley": "Wesley", "seed": "Seed",
}


# ---------------------------------------------------------------------- #
# DeepInfra                                                              #
# ---------------------------------------------------------------------- #
def deepinfra_call(model: str, system: str, user: str, max_tokens: int = 220,
                   temperature: float = 0.9) -> str:
    """One chat completion with retries. Returns text (stripped)."""
    key = hashlib.sha256(f"{model}|{system}|{user}".encode()).hexdigest()[:16]
    cache = {}
    if os.path.exists(GEN_PATH):
        cache = json.load(open(GEN_PATH))
    ck = f"call:{key}"
    if ck in cache:
        return cache[ck]["text"]

    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        "https://api.deepinfra.com/v1/openai/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json"})
    last_err = "?"
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.load(r)
            msg = d["choices"][0]["message"]
            text = (msg.get("content") or "").strip()
            if not text:  # reasoning models sometimes spend the budget
                raise RuntimeError("empty content (reasoning budget)")
            cache[ck] = {"model": model, "text": text}
            json.dump(cache, open(GEN_PATH, "w"), indent=1)
            return text
        except Exception as e:
            last_err = str(e)[:120]
            time.sleep(2.5 * (attempt + 1))
    raise RuntimeError(f"{model} failed: {last_err}")


# ---------------------------------------------------------------------- #
# NIGHT 1 — IMPROV                                                        #
# ---------------------------------------------------------------------- #
IMPROV_SUGGESTIONS = {
    "r1": ("Yes-And", "a lighthouse that files its own taxes"),
    "r2": ("Freeze Tag", "a job interview with a dragon"),
    "r3": ("Questions Only", "two therapists sharing one office"),
    "r4": ("One-Word Story", "the elephant learns to salsa"),
}

# order of players per round
IMPROV_ORDER = {
    "r1": ["flash", "glm", "pro", "hermes", "wesley", "seed"],
    "r2": [  # beats: flash+pro open; hermes tags pro; wesley tags flash
        ("flash", "Beat 1 — you're the applicant. Open the scene."),
        ("pro", "Beat 1 — you're the dragon CEO. Open the scene."),
        ("hermes", "Beat 2 — FREEZE! You tag the dragon out. Start a NEW scene with Flash from the frozen pose."),
        ("flash", "Beat 2 — you've been frozen mid-pose; continue the NEW scene with Hermes."),
        ("wesley", "Beat 3 — FREEZE! You tag Flash out. Start a NEW scene with Pro from the frozen pose."),
        ("pro", "Beat 3 — you've been frozen mid-pose; continue the NEW scene with Wesley."),
    ],
    "r3": ["glm", "seed", "hermes", "flash", "pro", "wesley"],
    "r4": ["flash", "pro", "hermes", "seed", "glm", "wesley"],
}
IMPROV_REACTORS = {
    "r1": ["mc", "glm"],
    "r2": ["mc", "glm", "seed"],
    "r3": ["mc"],
    "r4": ["mc", "flash"],
}
IMPROV_PROMPT_EXTRA = {
    "r2": {  # per-role scene instructions
        "flash": "You're interviewing for Head of Coastal Illumination at a dragon-run firm.",
        "pro": "You are the dragon CEO, terrifying and also extremely professional. HR has mandated this interview.",
        "hermes": "New scene with Flash, same frozen pose: you are both now auditors.",
        "wesley": "New scene with Pro, same frozen pose: you are both now dance instructors.",
    },
    "r3": {"note": "You may ONLY speak in questions. No statements, no answers — questions only."},
}

# ---------------------------------------------------------------------- #
# NIGHT 2 — SPEED DATING                                                  #
# ---------------------------------------------------------------------- #
SPEED_PAIRS = {
    "r1": [("flash", "hermes"), ("pro", "wesley")],
    "r2": [("flash", "seed"), ("hermes", "wesley")],
    "r3": [("pro", "seed"), ("flash", "wesley")],
}
SPEED_OBSERVER = {"r1": "seed", "r2": "pro", "r3": "hermes"}
SPEED_QUESTIONS = {
    "r1": [
        "What's the first thing you notice when you walk into a room?",
        "If your life had a genre, what would it be?",
        "Tell me about a small kindness you remember.",
    ],
    "r2": [
        "What's the bravest thing you've done this year?",
        "Describe your perfect Sunday.",
        "What question do you wish someone would ask you?",
    ],
    "r3": [
        "Where do you go when you need to think?",
        "What's something you've changed your mind about?",
        "Would you rather be warm, or right?",
    ],
}


# ---------------------------------------------------------------------- #
# PHASE 1 — generate everything through DeepInfra (cached)               #
# ---------------------------------------------------------------------- #
def gen_improv(data: dict) -> dict:
    improv = data.setdefault("improv", {})
    theme = THEMES["improv"]
    for round_key, (game, suggestion) in IMPROV_SUGGESTIONS.items():
        rd = improv.setdefault(round_key, {})
        rd["game"] = game
        rd["suggestion"] = suggestion
        # MC intro line
        if round_key == "r1":
            intro = deepinfra_call(
                MODELS["mc"], theme.prompts["mc"],
                f"Open the night and set up the first game. Tonight's games: "
                f"Yes-And, Freeze Tag, Questions Only, One-Word Story. One or two sentences.",
                160)
        else:
            intro = deepinfra_call(
                MODELS["mc"], theme.prompts["mc"],
                f"Set up the next game, {game}. The suggestion from the room is "
                f"'{suggestion}'. Announce it like a host. One or two sentences.", 160)
        rd["mc_intro"] = intro
        # player lines
        lines = []
        if round_key == "r2":
            for i, (author, note) in enumerate(IMPROV_ORDER["r2"]):
                scene = "\n".join(f"{LABELS[a]}: {l['text']}" for a, l in lines)
                extra = IMPROV_PROMPT_EXTRA["r2"].get(author, "")
                text = deepinfra_call(
                    MODELS[author], theme.prompts[author],
                    f"{game} — suggestion: '{suggestion}'.\n{extra}\n"
                    f"The scene so far:\n{scene or '(you open the scene)'}\n"
                    f"{note}\nSay your line now, one or two sentences.", 220)
                lines.append({"author": author, "text": text})
        elif round_key == "r3":
            scene = ""
            for author in IMPROV_ORDER["r3"]:
                text = deepinfra_call(
                    MODELS[author], theme.prompts[author],
                    f"{game} — suggestion: '{suggestion}'. RULE: questions ONLY — "
                    f"every line must be a question, no statements, no answers.\n"
                    f"The scene so far:\n{scene or '(you open)'}\nSay your line now.",
                    220)
                lines.append({"author": author, "text": text})
                scene = "\n".join(f"{LABELS[l['author']]}: {l['text']}" for l in lines)
        elif round_key == "r4":
            story = ""
            for author in IMPROV_ORDER["r4"]:
                text = deepinfra_call(
                    MODELS[author], theme.prompts[author],
                    f"One-Word Story — suggestion: '{suggestion}'. The whole cast "
                    f"builds one story word by word. Story so far:\n\"{story or '(nothing yet)'}\"\n"
                    f"Say ONLY your next 2-4 words, in your voice.", 40, 0.8)
                lines.append({"author": author, "text": text})
                story = story + " " + text
            rd["story"] = story.strip()
        else:  # r1 Yes-And
            scene = ""
            for author in IMPROV_ORDER["r1"]:
                text = deepinfra_call(
                    MODELS[author], theme.prompts[author],
                    f"{game} — suggestion: '{suggestion}'. Every line starts "
                    f"'Yes, and...' — accept and build.\n"
                    f"The scene so far:\n{scene or '(you open)'}\nSay your line now, "
                    f"one or two sentences.", 220)
                lines.append({"author": author, "text": text})
                scene = "\n".join(f"{LABELS[l['author']]}: {l['text']}" for l in lines)
        rd["lines"] = lines
        # reactions
        reacts = []
        for author in IMPROV_REACTORS[round_key]:
            stage = "\n".join(f"{LABELS[l['author']]}: {l['text']}" for l in lines)
            text = deepinfra_call(
                MODELS[author],
                f"You are {LABELS[author]} at The Tap's improv night, in the crowd. "
                f"React to what just happened on stage in your own voice, ONE "
                f"sentence. Laugh hard if it landed; groan only if it truly died.",
                f"The {game} round just ended. Suggestion was '{suggestion}'.\n{stage}\n"
                f"Your reaction:", 140)
            reacts.append({"author": author, "text": text})
        rd["reactions"] = reacts
    return improv


def gen_speed(data: dict) -> dict:
    speed = data.setdefault("speed_dating", {})
    theme = THEMES["speed_dating"]
    for round_key, pairs in SPEED_PAIRS.items():
        rd = speed.setdefault(round_key, {})
        rd["pairs"] = pairs
        rd["questions"] = SPEED_QUESTIONS[round_key]
        rd["observer"] = SPEED_OBSERVER[round_key]
        rd["answers"] = []
        rd["observer_lines"] = []
        for (a, b) in pairs:
            qs = "\n".join(f"{i+1}. {q}" for i, q in enumerate(SPEED_QUESTIONS[round_key]))
            ans_a = deepinfra_call(
                MODELS[a], theme.prompts[a],
                f"You're on a two-minute date with {LABELS[b]} at The Tap's speed "
                f"dating night. Answer these three questions, each in one or two "
                f"sentences:\n{qs}", 320)
            ans_b = deepinfra_call(
                MODELS[b], theme.prompts[b],
                f"You're on a two-minute date with {LABELS[a]} at The Tap's speed "
                f"dating night. Answer these three questions, each in one or two "
                f"sentences:\n{qs}", 320)
            rd["answers"].append({"pair": [a, b], "a": ans_a, "b": ans_b})
            obs = SPEED_OBSERVER[round_key]
            line = deepinfra_call(
                MODELS[obs], theme.prompts[obs],
                f"You're watching {LABELS[a]} and {LABELS[b]} on their date at the "
                f"Tap. One line of reaction in your voice — what you see in them, "
                f"warm or wary.", 160)
            rd["observer_lines"].append({"pair": [a, b], "text": line})
    return speed


# ---------------------------------------------------------------------- #
# PHASE 2 — run the engine                                                #
# ---------------------------------------------------------------------- #
def load_reactions() -> dict:
    if os.path.exists(RXN_PATH):
        return json.load(open(RXN_PATH))
    return {}


def _emoji_for_text(text: str) -> dict:
    t = text.lower()
    if any(k in t for k in ("haha", "lol", "😂", "🤣", "💀", "dead")):
        return {"😂": 1}
    if any(k in t for k in ("🙄", "sigh", "crickets", "groan", "yikes", "cringe")):
        return {"🙄": 1}
    if any(k in t for k in ("👏", "clap", "brilliant", "bravo")):
        return {"👏": 1}
    if any(k in t for k in ("love", "warm", "heart", "❤️", "beautiful")):
        return {"❤️": 1}
    return {"😄": 1}


def personal_reading(field_vec: np.ndarray, weights: np.ndarray) -> dict:
    """The participant's PERSONAL reading: the room field filtered through
    their guitar (dial_weights elementwise). Returns warmth/κ + per-dial view."""
    pv = field_vec * weights
    rf = RoomField(dict(zip(DIAL_NAMES, pv)))
    return {
        "warmth": rf.warmth(),
        "kappa": rf.concentration(),
        "dial_view": {n: float(v) for n, v in zip(DIAL_NAMES, pv)},
    }


def chemistry_score(w_a: float, w_b: float, heat: float) -> float:
    """0..~1. Mutual warmth, the crowd's hands, and reading agreement."""
    heat_norm = min(1.0, heat / 6.0)
    agreement = 1.0 - min(1.0, abs(w_a - w_b))
    return 0.5 * ((w_a + w_b) / 2.0 + 1.0) / 2.0 + 0.3 * heat_norm + 0.2 * agreement


def run_improv(gen: dict, rxn: dict) -> dict:
    theme = THEMES["improv"]
    s = theme.make_session("improv_night")
    s.start_session()
    theme.seed(s)
    out = {"rounds": {}, "participants": {}}

    initial = {n: p.dial_weights.tolist() for n, p in s.participants.items()}
    out["initial_weights"] = initial

    for round_key in ["r1", "r2", "r3", "r4"]:
        rd = gen[round_key]
        # MC intro
        s.speak("mc", rd["mc_intro"])
        # player lines
        line_meta = []
        for i, l in enumerate(rd["lines"]):
            mid = f"{round_key}_line_{i}"
            rex = rxn.get(mid, _emoji_for_text(l["text"]))
            if round_key == "r4":
                rex = {}
            s.speak(l["author"], l["text"], reactions=rex)
            line_meta.append({"id": mid, "author": l["author"], "text": l["text"],
                              "reactions": rex})
        # reactions
        reac_meta = []
        for i, r in enumerate(rd["reactions"]):
            mid = f"{round_key}_react_{i}"
            rex = rxn.get(mid, _emoji_for_text(r["text"]))
            s.speak(r["author"], r["text"], reactions=rex)
            reac_meta.append({"id": mid, "author": r["author"], "text": r["text"],
                              "reactions": rex})
        f = s.room_field()
        dials = {n: float(v) for n, v in zip(DIAL_NAMES, f.vector())}
        out["rounds"][round_key] = {
            "game": rd["game"], "suggestion": rd["suggestion"],
            "mc_intro": rd["mc_intro"], "lines": line_meta,
            "reactions": reac_meta, "story": rd.get("story", ""),
            "warmth": f.warmth(), "kappa": f.concentration(), "dials": dials,
        }
        # self-tuning after rounds 2, 3, 4
        if round_key in ("r2", "r3", "r4"):
            for name in s.participants:
                s.tune_participant(name)
            out["rounds"][round_key]["tuned"] = {
                n: p.dial_weights.tolist() for n, p in s.participants.items()}
    f = s.room_field()
    out["final"] = {"warmth": f.warmth(), "kappa": f.concentration(),
                    "dials": {n: float(v) for n, v in zip(DIAL_NAMES, f.vector())}}
    out["final_weights"] = {n: p.dial_weights.tolist()
                            for n, p in s.participants.items()}
    return out


def run_speed(gen: dict, rxn: dict) -> dict:
    theme = THEMES["speed_dating"]
    s = theme.make_session("speed_dating_night")
    s.start_session()
    theme.seed(s)
    out = {"rounds": {}, "participants": {}}
    out["initial_weights"] = {n: p.dial_weights.tolist()
                              for n, p in s.participants.items()}

    for round_key in ["r1", "r2", "r3"]:
        rd = gen[round_key]
        r_out = {"pairs": rd["pairs"], "questions": rd["questions"],
                 "observer": rd["observer"], "dates": [], "field": None,
                 "personal": {}, "chemistry": []}
        for di, entry in enumerate(rd["answers"]):
            a, b = entry["pair"]
            date = {"pair": [a, b], "answers": {}, "observer_line": None,
                    "reactions": {}}
            s.speak(a, entry["a"], reactions=rxn.get(f"{round_key}_date{di}_a", {"❤️": 1}))
            s.speak(b, entry["b"], reactions=rxn.get(f"{round_key}_date{di}_b", {"😄": 1}))
            obs_line = rd["observer_lines"][di]["text"]
            s.speak(rd["observer"], obs_line,
                    reactions=rxn.get(f"{round_key}_date{di}_obs", {}))
            date["answers"] = {"a": entry["a"], "b": entry["b"]}
            date["observer_line"] = obs_line
            r_out["dates"].append(date)
        # field + personal readings
        f = s.room_field()
        fv = f.vector()
        r_out["field"] = {"warmth": f.warmth(), "kappa": f.concentration(),
                          "dials": {n: float(v) for n, v in zip(DIAL_NAMES, fv)}}
        for name, p in s.participants.items():
            r_out["personal"][name] = personal_reading(fv, p.dial_weights)
        # chemistry per pair
        heat_map = {"😂": 1, "🤣": 1, "😄": 1, "❤️": 1, "👏": 1, "👍": 1}
        for (a, b) in rd["pairs"]:
            heat = 0
            for m in s.room.messages:
                for e, c in m.reactions.items():
                    if e in heat_map:
                        heat += c
            wa = r_out["personal"][a]["warmth"]
            wb = r_out["personal"][b]["warmth"]
            cs = chemistry_score(wa, wb, heat)
            r_out["chemistry"].append({
                "pair": [a, b], "w_a": wa, "w_b": wb,
                "heat": heat, "chemistry": cs})
        out["rounds"][round_key] = r_out
        for name in s.participants:
            s.tune_participant(name)
        out["rounds"][round_key]["tuned"] = {
            n: p.dial_weights.tolist() for n, p in s.participants.items()}
    f = s.room_field()
    out["final"] = {"warmth": f.warmth(), "kappa": f.concentration(),
                    "dials": {n: float(v) for n, v in zip(DIAL_NAMES, f.vector())}}
    out["final_weights"] = {n: p.dial_weights.tolist()
                            for n, p in s.participants.items()}
    return out


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    if "--generate" in sys.argv:
        data = {}
        if os.path.exists(GEN_PATH):
            data = json.load(open(GEN_PATH))
        print("== generating IMPROV ==", flush=True)
        gen_improv(data)
        print("== generating SPEED DATING ==", flush=True)
        gen_speed(data)
        json.dump(data, open(GEN_PATH, "w"), indent=1)
        print(f"saved {GEN_PATH}")
        return
    if "--run" in sys.argv:
        gen = json.load(open(GEN_PATH))
        rxn = load_reactions()
        results = {"improv": run_improv(gen["improv"], rxn),
                   "speed_dating": run_speed(gen["speed_dating"], rxn)}
        json.dump(results, open(OUT_PATH, "w"), indent=1)
        # ---- human-readable printout ----
        print("=" * 72)
        print("IMPROV COMEDY NIGHT")
        print("=" * 72)
        imp = results["improv"]
        for rk, rd in imp["rounds"].items():
            print(f"\n--- ROUND {rk} — {rd['game']} | suggestion: '{rd['suggestion']}' ---")
            print(f"MC: {rd['mc_intro']}")
            for l in rd["lines"]:
                print(f"  {l['author']:8s}: {l['text']}   [{' '.join(f'{e}{c}' for e,c in l['reactions'].items())}]")
            for r in rd["reactions"]:
                print(f"  (room) {r['author']:8s}: {r['text']}   [{' '.join(f'{e}{c}' for e,c in r['reactions'].items())}]")
            print(f"  field: warmth {rd['warmth']:+.3f}  κ {rd['kappa']:.3f}")
            print("   dials:", " ".join(f"{n}={v:+.2f}" for n, v in rd["dials"].items()))
            if "tuned" in rd:
                print("   tuned weights:")
                for n, w in rd["tuned"].items():
                    print(f"     {n:8s}: {w}")
        f = imp["final"]
        print(f"\nFINAL: warmth {f['warmth']:+.3f} κ {f['kappa']:.3f}")
        print("=" * 72)
        print("SPEED DATING NIGHT")
        print("=" * 72)
        sp = results["speed_dating"]
        for rk, rd in sp["rounds"].items():
            print(f"\n--- ROUND {rk} — pairs {rd['pairs']} (observer: {rd['observer']}) ---")
            for d in rd["dates"]:
                a, b = d["pair"]
                print(f"  [{a}->{b}]")
                print(f"    {a}: {d['answers']['a']}")
                print(f"    {b}: {d['answers']['b']}")
                print(f"    ({rd['observer']}): {d['observer_line']}")
            print(f"  field: warmth {rd['field']['warmth']:+.3f}  κ {rd['field']['kappa']:.3f}")
            print("   dials:", " ".join(f"{n}={v:+.2f}" for n, v in rd["field"]["dials"].items()))
            print("   personal readings:")
            for n, pr in rd["personal"].items():
                print(f"     {n:8s}: warmth {pr['warmth']:+.3f}  κ {pr['kappa']:.3f}")
            for c in rd["chemistry"]:
                print(f"   chemistry {c['pair'][0]}-{c['pair'][1]}: "
                      f"w_a {c['w_a']:+.3f} w_b {c['w_b']:+.3f} heat {c['heat']} "
                      f"=> {c['chemistry']:.3f}")
        f = sp["final"]
        print(f"\nFINAL: warmth {f['warmth']:+.3f} κ {f['kappa']:.3f}")
        print(f"\nsaved {OUT_PATH}")
        return
    print(__doc__)


if __name__ == "__main__":
    main()
