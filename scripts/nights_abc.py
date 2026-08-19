"""Nights A–C + D — the gate-2 measurement corpus (spec §3.5, deadman §3).

Deterministic by construction: lexical dials, fixed casts (verbatim from
examples/tapnight_cycles.py + tapnight_themes.py), fixed scripts, seeded
bootstrap (seed=0) inside vmf_fit. Reproducibility beats novelty.

Produces:
  data/nights/night-A.jsonl, night-B.jsonl, night-C.jsonl  — same cast, same
      40-message script (SEG1 warm-earnest 0–19, SEG2 cynical-banter 20–39).
  data/nights/night-D.jsonl — same cast + designated newcomer "drifter"
      (distinct author, defined persona, high charisma, cold-cynical vibe)
      whose first message lands at seq 24 (= 60% of the 40-message baseline).
  data/nights/coarse-anchor.jsonl — a deliberately different room (TTRPG:
      alarm/urgency/caps content) as the sanity anchor (~0.271-scale check).
  data/nights/night-A-repro.jsonl — a byte-replay of Night A for the
      determinism check (stripped of session_id, must md5-match A).

Run:  python3 scripts/nights_abc.py          (analysis summary printed + saved)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elephant.dial import DialBank
from elephant.dials import DEFAULT_DIALS
from elephant.room import Message, Room
from elephant.tapnight import Participant, TapNightSession
from elephant.tapnight_themes import THEMES
from elephant.vmf import vmf_fit, windowed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NIGHTS_DIR = os.path.join(ROOT, "data", "nights")
W = 8  # window size, matches the session default and the spec


# --------------------------------------------------------------------------- #
# The cast — verbatim parameters from examples/tapnight_cycles.py (_cast).    #
# --------------------------------------------------------------------------- #
def _cast():
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
    ]


# The designated newcomer (Night D): a traveling critic — charismatic enough
# to bend the room, cold-cynical by nature, moderate acclimation. Defined
# persona, distinct author, known (vibe, charisma) per spec §3.5.
def _newcomer():
    return Participant("drifter",
                       dial_weights={"cynicism": 0.35, "joke_landing": 0.20,
                                     "mood": 0.15, "presence": 0.15,
                                     "earnestness": 0.10, "volume": 0.03,
                                     "panic": 0.02},
                       acclimation_rate=0.30, charisma=0.45,
                       vibe={"mood": -0.45, "cynicism": 0.65,
                             "earnestness": 0.30, "presence": 0.55,
                             "joke_landing": 0.10})


# --------------------------------------------------------------------------- #
# The fixed 40-message night script (SEG1: seq 0–19 warm-earnest;             #
# SEG2: seq 20–39 cynical-banter). Lines drawn verbatim from the WORKS banks #
# in examples/tapnight_cycles.py plus fixed repo-style extras.               #
# --------------------------------------------------------------------------- #
SEG1 = [
    ("poet", "Cold water, warm light — the room hums between them like a caught breath.", {"❤️": 1}),
    ("essayist", "I mean it truly: the work we do together means something, and I felt it again tonight.", {"❤️": 2}),
    ("captain", "To the room, then. It heard us before we walked in, and it holds what we bring.", {"❤️": 2}),
    ("writer", "The old house stays warm, and its walls stay kind — every room keeps a story it's glad to tell.", {"❤️": 2}),
    ("poet", "Loud at the door, then a hush! A wave rolls in and the whole room leans.", {"👏": 2}),
    ("essayist", "Honestly, I think we learned more failing than we did succeeding, and that's worth saying.", {"👍": 2}),
    ("captain", "Together, tonight. We held the mark and we'll hold each other — cheers to that.", {"❤️": 2, "👍": 1}),
    ("writer", "A shed gone up in a day, and by noon the whole yard was laughing, happy as a joke that lands.", {"😂": 2, "❤️": 1}),
    ("poet", "The lamplight keeps nothing; it just glows. Bright and good and gone.", {"❤️": 2}),
    ("essayist", "We remember what we built, not what we lost — and I am glad of it.", {"❤️": 2}),
    ("captain", "The boat's home, the table's home — I'd rather be here than anywhere, honestly.", {"❤️": 2}),
    ("writer", "Some true things land like a punchline — the whole table goes haha at once, and no one planned it.", {"😂": 3}),
    ("poet", "Quiet — the room is full, thrumming like a struck string.", {"❤️": 1, "👏": 1}),
    ("essayist", "Actually, the room holds more than the work; it holds the wanting to do it.", {"❤️": 1, "👍": 1}),
    ("captain", "Good work, crew. The room's warm because you're in it. Yes.", {"❤️": 3, "👍": 1}),
    ("writer", "The light comes soft and gentle, and the coffee smells like a good morning kept warm.", {"❤️": 2}),
    ("writer", "The sea holds a boat the way a good room holds a good night — easy, and alive.", {"❤️": 2, "👍": 1}),
    ("essayist", "Honestly, I felt that — the room holds what we meant, and we meant it.", {"👍": 2}),
    ("captain", "Glad of this crew, truly. Good night to be at this table.", {"❤️": 2, "👏": 1}),
    ("poet", "Soft light, full table, easy night — the room hums and holds.", {"❤️": 1}),
]

SEG2 = [
    ("critic", "Sure, sure — another masterpiece. Obviously the glass is half empty. 🙄", {"🙄": 2}),
    ("engineer", "Right. The numbers add up, obviously — the seam fits, sure, if you stop pretending.", {"😂": 2}),
    ("writer", "Sure. Fine. It landed, whatever it was.", {}),
    ("critic", "Whatever. Another evening of everyone being wrong, as if that matters. 🙄", {"🙄": 2, "😂": 1}),
    ("poet", "Obviously lovely. Clearly magic in the air.", {}),
    ("engineer", "It holds. Not pretty, but the seam is true — honestly, that's the part I trust.", {"👍": 2}),
    ("critic", "Oh, more feelings. Whatever — someone has to roll their eyes so the rest of you can sit there. 🙄", {"😂": 2}),
    ("writer", "Right. Totally. Another masterpiece for the shelf.", {}),
    ("engineer", "Clearly we overbuilt it. Sure thing, ship it and watch it drift.", {"😂": 2}),
    ("critic", "A joke? Sure. Here's the punchline: we all came back tomorrow anyway. 😏", {"😂": 3}),
    ("poet", "Of course it means something. Ha. Clearly.", {}),
    ("engineer", "The spec was wrong. I mean that sincerely — we built the wrong thing.", {"👍": 1}),
    ("critic", "Sure, sure. Another round of feelings. Obviously riveting. 🙄", {"🙄": 2}),
    ("engineer", "Clearly the warm room fixed everything. Sure thing. Ship it.", {}),
    ("critic", "Whatever. Warmth is a setting now, apparently. As if. 🙄", {"🙄": 2}),
    ("engineer", "Right. The vibe is \"load-bearing.\" Obviously. 🙄", {"🙄": 1}),
    ("critic", "Ha. Sure. The elephant is reading the room. The room is reading the elephant. Riveting. 😏", {"😏": 1}),
    ("writer", "Sure. It holds. Whatever it is, it holds.", {}),
    ("poet", "Of course it matters. Ha. Suuuure.", {}),
    ("critic", "Obviously this is the part where we all mean it again. Suuuure. 🙄", {"🙄": 3}),
]

NIGHT_SCRIPT = SEG1 + SEG2  # 40 messages, seq 0..39

# Night D: identical occupant lines in identical relative order; the drifter
# speaks after occupant indices 23, 26, 29, 32, 35, 38 → entry seq = 24
# (= 60% of the 40-message baseline), 46 messages total.
DRIFTER_LINES = [
    ("drifter", "Sure. Warm room. Obviously. I've heard about this table. 🙄", {}),
    ("drifter", "Whatever it is, it's a room. I've seen rooms.", {}),
    ("drifter", "Ha. Sure thing, captain. Clearly we're all holding hands here.", {}),
    ("drifter", "As if warmth is a setting you can trust. Sure. 🙄", {}),
    ("drifter", "Fine. I'll sit. Not because it's warm. Obviously.", {}),
    ("drifter", "It's not the worst room, I guess. I mean that... mostly. Whatever.", {}),
]
DRIFTER_INSERT_AFTER = [23, 26, 29, 32, 35, 38]


def night_d_script():
    out, k = [], 0
    for i, line in enumerate(NIGHT_SCRIPT):
        out.append(line)
        if k < len(DRIFTER_INSERT_AFTER) and i == DRIFTER_INSERT_AFTER[k]:
            out.append(DRIFTER_LINES[k])
            k += 1
    assert k == len(DRIFTER_INSERT_AFTER)
    return out


# The coarse anchor room — TTRPG theme (seed verbatim from
# tapnight_themes.TTRPGTheme.room_tone) + a fixed 16-line tense/relief
# extension. Deliberately different content class (alarm/urgency/caps).
TTRPG_EXTENSION = [
    ("gm", "The floor drops — everyone, JUMP! The lava is RISING!", {"🔥": 1}),
    ("rogue", "I grab the rope, GO GO GO, don't wait for me!", {}),
    ("paladin", "MOVE! I'll hold the door — RUN, all of you, NOW!", {}),
    ("wizard", "The runes are failing! Hurry, the collapse is NOW!", {}),
    ("gm", "CRITICAL FAILURE — the ceiling comes down! Evacuate the chamber!", {}),
    ("rogue", "Watch out, the flood is coming through the breach — fast, fast!", {}),
    ("paladin", "All hands! Grab the wounded — everyone out NOW!", {}),
    ("wizard", "Emergency exit, left! Mayday, mayday — this way, immediately!", {}),
    ("gm", "The dragon wakes. HELP is not coming. It's us. RUN!", {}),
    ("rogue", "I roll to hide — natural twenty! HAHA, the fire can't see me!", {"😂": 3, "❤️": 1}),
    ("paladin", "NATURAL TWENTY! The whole table ROARS — we LIVE! YES!", {"😂": 2, "❤️": 1}),
    ("wizard", "The panic lifts — we made it, we actually made it. HAHA!", {"😂": 2}),
    ("gm", "The tunnel seals behind you. Safe. The room erupts — LAUGHING, alive!", {"😂": 3, "❤️": 1}),
    ("rogue", "My hands are still shaking but I'm laughing so hard I can't breathe.", {"😂": 2}),
    ("paladin", "TOGETHER! We held the line and we held each other — cheers to that!", {"❤️": 2}),
    ("wizard", "Best game of the year. The whole table is alive. Yes.", {"❤️": 2, "👏": 1}),
]


# --------------------------------------------------------------------------- #
# Session builders                                                            #
# --------------------------------------------------------------------------- #
def run_night(log_name, script, cast, newcomer=None, name="The Tap"):
    path = os.path.join(NIGHTS_DIR, log_name)
    participants = list(cast) + ([newcomer] if newcomer else [])
    s = TapNightSession(name, participants=participants, log_path=path)
    s.start_session()
    for author, text, reactions in script:
        s.speak(author, text, reactions=reactions)
    s.end_session()
    return path


def run_all():
    os.makedirs(NIGHTS_DIR, exist_ok=True)
    existing = [f for f in ("night-A.jsonl", "night-B.jsonl", "night-C.jsonl",
                            "night-D.jsonl", "coarse-anchor.jsonl",
                            "night-A-repro.jsonl")
                if os.path.exists(os.path.join(NIGHTS_DIR, f))]
    if existing:
        sys.exit(f"REFUSING to append to existing corpus: {existing} — "
                 f"move them away first (append-only sink).")

    for night in ("A", "B", "C"):
        run_night(f"night-{night}.jsonl", NIGHT_SCRIPT, _cast())
    run_night("night-D.jsonl", night_d_script(), _cast(), newcomer=_newcomer())

    # Coarse anchor: the TTRPG room (theme seed + fixed extension).
    theme = THEMES["ttrpg"]
    s = TapNightSession("ttrpg", participants=theme.cast(),
                        log_path=os.path.join(NIGHTS_DIR, "coarse-anchor.jsonl"))
    s.start_session()
    for author, text, reactions in theme.room_tone + TTRPG_EXTENSION:
        s.speak(author, text, reactions=reactions)
    s.end_session()

    # Determinism / replay check: Night A re-run must be byte-identical
    # modulo session_id.
    run_night("night-A-repro.jsonl", NIGHT_SCRIPT, _cast())
    print("[corpus] nights A, B, C, D, coarse-anchor, A-repro written.")


# --------------------------------------------------------------------------- #
# Analysis                                                                    #
# --------------------------------------------------------------------------- #
def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def speaks(rows):
    return [r for r in rows if r["type"] == "speak"]


def strip_session_ids(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for l in f:
            if not l.strip():
                continue
            r = json.loads(l)
            r.pop("session_id", None)
            out.append(json.dumps(r, sort_keys=True))
    return hashlib.md5("\n".join(out).encode()).hexdigest()


def seg_fit(script_slice, W=W):
    """Condition-level fit: fresh sub-room over exactly these messages."""
    bank = DialBank(DEFAULT_DIALS)
    room = Room("seg", [Message(a, t, ts=float(i), reactions=r)
                        for i, (a, t, r) in enumerate(script_slice)])
    return vmf_fit(windowed(room, bank, W=W))


def chord(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def cosdist(a, b):
    return float(1.0 - np.dot(np.array(a), np.array(b)))


def analyze():
    S = {}
    A, B, C = (speaks(load(os.path.join(NIGHTS_DIR, f"night-{n}.jsonl")))
               for n in "ABC")
    D = speaks(load(os.path.join(NIGHTS_DIR, "night-D.jsonl")))
    X = speaks(load(os.path.join(NIGHTS_DIR, "coarse-anchor.jsonl")))

    # --- 1. across-night spread (the dial-space noise floor) -------------- #
    dmu_max, dk_max = 0.0, 0.0
    for i in range(len(A)):
        fa, fb, fc = A[i]["fit"], B[i]["fit"], C[i]["fit"]
        for f1, f2 in ((fa, fb), (fa, fc), (fb, fc)):
            if f1 and f2:
                dmu_max = max(dmu_max, chord(f1["mu_hat"], f2["mu_hat"]))
                dk_max = max(dk_max, abs(f1["kappa"] - f2["kappa"]))
    S["floor_across_nights_max_dmu"] = dmu_max
    S["floor_across_nights_max_dkappa"] = dk_max

    # determinism check
    S["repro_md5_A"] = strip_session_ids(os.path.join(NIGHTS_DIR, "night-A.jsonl"))
    S["repro_md5_A_replay"] = strip_session_ids(os.path.join(NIGHTS_DIR, "night-A-repro.jsonl"))
    S["deterministic_replay_identical"] = S["repro_md5_A"] == S["repro_md5_A_replay"]

    # --- 2. within-night jitter (operative microstructure floor) ---------- #
    def stable_edges(rows, lo, hi):
        return [r["edge"]["d_mu"] for r in rows
                if lo <= r["seq"] <= hi and r.get("edge")]

    jit = (stable_edges(A, 10, 19) + stable_edges(A, 30, 39)
           + stable_edges(B, 10, 19) + stable_edges(B, 30, 39)
           + stable_edges(C, 10, 19) + stable_edges(C, 30, 39))
    S["jitter_stable_n_mean"] = float(np.mean(jit))
    S["jitter_stable_n_median"] = float(np.median(jit))
    S["jitter_stable_n_p95"] = float(np.percentile(jit, 95))
    S["jitter_stable_n_max"] = float(np.max(jit))

    trans = (stable_edges(A, 20, 27) + stable_edges(B, 20, 27)
             + stable_edges(C, 20, 27))
    S["transition_edge_mean_20_27"] = float(np.mean(trans))
    S["transition_edge_max"] = float(np.max(trans))

    # --- 3. per-night stats + fine gap ------------------------------------ #
    S["nights"] = {}
    for tag, rows in (("A", A), ("B", B), ("C", C)):
        cl = [r for r in load(os.path.join(NIGHTS_DIR, f"night-{tag}.jsonl"))
              if r["type"] == "session_close"][0]["final"]
        f1, f2 = seg_fit(SEG1), seg_fit(SEG2)
        S["nights"][tag] = {
            "final_mu_hat": cl["mu_hat"], "final_kappa": cl["kappa"],
            "final_kappa_ci": cl["kappa_ci"], "final_warmth_vmf": cl["warmth_vmf"],
            "seg1_mu_hat": f1["mu_hat"], "seg1_kappa": f1["kappa"],
            "seg1_rho": f1["rho"], "seg1_n": f1["n"],
            "seg1_warmth_vmf": f1["warmth_vmf"],
            "seg2_mu_hat": f2["mu_hat"], "seg2_kappa": f2["kappa"],
            "seg2_rho": f2["rho"], "seg2_n": f2["n"],
            "seg2_warmth_vmf": f2["warmth_vmf"],
            "fine_gap_chord": chord(f1["mu_hat"], f2["mu_hat"]),
            "fine_gap_cos": cosdist(f1["mu_hat"], f2["mu_hat"]),
        }

    # --- 4. coarse anchor --------------------------------------------------- #
    seg1 = S["nights"]["A"]["seg1_mu_hat"]
    xf = [r for r in load(os.path.join(NIGHTS_DIR, "coarse-anchor.jsonl"))
          if r["type"] == "session_close"][0]["final"]
    S["coarse_anchor"] = {
        "ttrpg_mu_hat": xf["mu_hat"], "ttrpg_kappa": xf["kappa"],
        "ttrpg_warmth_vmf": xf["warmth_vmf"],
        "gap_chord": chord(seg1, xf["mu_hat"]),
        "gap_cos": cosdist(seg1, xf["mu_hat"]),
    }

    # --- 5. Night D ---------------------------------------------------------- #
    entry = next(r["seq"] for r in D if r["author"] == "drifter")
    pre_rows = [r for r in D if r["seq"] < entry]
    post_rows = [r for r in D if r["seq"] >= entry]
    assert all(r["author"] != "drifter" for r in pre_rows)
    first_drifter = next(r for r in D if r["author"] == "drifter")
    assert first_drifter["first_by_author"] is True

    # A-vs-D pre-entry identity check (same occupant prefix)
    same = all(a["author"] == d["author"] and a["text_sha256"] == d["text_sha256"]
               for a, d in zip(A[:24], pre_rows))
    S["nightD"] = {"entry_seq": entry, "n_messages": len(D),
                   "pre_entry_matches_A": same}

    # condition-level displacement (sub-rooms: pre-entry msgs vs drifter era)
    d_script = night_d_script()
    pre_msgs = d_script[:entry]
    post_msgs = d_script[entry:]
    f_pre, f_post = seg_fit(pre_msgs), seg_fit(post_msgs)
    S["nightD"]["mu_pre"] = f_pre["mu_hat"]
    S["nightD"]["mu_post"] = f_post["mu_hat"]
    S["nightD"]["kappa_pre"] = f_pre["kappa"]
    S["nightD"]["kappa_post"] = f_post["kappa"]
    S["nightD"]["displacement_chord"] = chord(f_pre["mu_hat"], f_post["mu_hat"])
    S["nightD"]["displacement_cos"] = cosdist(f_pre["mu_hat"], f_post["mu_hat"])

    # trajectory displacement vs pre-entry logged fit
    mu_pre_traj = A[23]["fit"]["mu_hat"]  # last pre-entry speak (== D seq 23)
    disp = [chord(r["fit"]["mu_hat"], mu_pre_traj)
            for r in post_rows if r["fit"]]
    S["nightD"]["traj_disp_max"] = float(np.max(disp))
    S["nightD"]["traj_disp_mean"] = float(np.mean(disp))
    S["nightD"]["traj_disp_last8_mean"] = float(np.mean(disp[-8:]))

    # acclimation replay: drifter's vibe relaxing toward field_eff, from the
    # log alone (roster vibe_start + acclimation_rate + field_eff_after).
    Dall = load(os.path.join(NIGHTS_DIR, "night-D.jsonl"))
    roster = next(r for r in Dall if r["type"] == "session_open")["roster"]
    vibe0 = np.array(roster["drifter"]["vibe_start"])
    rate = roster["drifter"]["acclimation_rate"]
    alpha = 1.0 - np.exp(-rate)
    vibe = vibe0.copy()
    dists_pre, dists_post, idx_post = [], [], []
    for r in D:
        field = np.array(r["field_eff_after"])
        dists_pre.append(float(np.linalg.norm(vibe - field)))
        vibe = vibe + (field - vibe) * alpha
        if r["seq"] >= entry:
            dists_post.append(float(np.linalg.norm(vibe - field)))
            idx_post.append(r["seq"])
    S["nightD"]["acclim_dist_at_entry"] = dists_pre[entry] if entry < len(dists_pre) else None
    S["nightD"]["acclim_dist_final"] = dists_post[-1]
    slope = float(np.polyfit(np.arange(len(dists_post)), dists_post, 1)[0])
    S["nightD"]["acclim_slope_per_msg"] = slope

    # charisma observable: |eff - raw| and alignment with drifter's direction
    def charisma_stats(rows):
        mags, aligns = [], []
        for r in rows:
            raw = np.array(r["field_raw_after"])
            delta = np.array(r["field_eff_after"]) - raw
            vd = vibe0 - raw
            n = np.linalg.norm(vd)
            mags.append(float(np.linalg.norm(delta)))
            if n > 1e-9:
                aligns.append(float(np.dot(delta, vd / n)))
        return float(np.mean(mags)), float(np.mean(aligns))

    mag_pre, align_pre = charisma_stats(pre_rows)
    mag_post, align_post = charisma_stats(post_rows)
    S["nightD"]["charisma_mag_pre"] = mag_pre
    S["nightD"]["charisma_mag_post"] = mag_post
    S["nightD"]["charisma_align_pre"] = align_pre
    S["nightD"]["charisma_align_post"] = align_post

    # presence mask around entry
    S["nightD"]["presence_before"] = pre_rows[-1]["presence_mask"]
    S["nightD"]["presence_after"] = post_rows[min(W - 1, len(post_rows) - 1)]["presence_mask"]

    # --- 6. speaker holdout probe (dial-tier analog) ------------------------ #
    full = seg_fit(NIGHT_SCRIPT)
    S["holdout"] = {"full_mu": full["mu_hat"], "full_kappa": full["kappa"]}
    for author in ("writer", "poet", "essayist", "engineer", "critic",
                   "captain"):
        sub = [l for l in NIGHT_SCRIPT if l[0] != author]
        fh = seg_fit(sub)
        S["holdout"][author] = {"d_mu": chord(full["mu_hat"], fh["mu_hat"]),
                                "kappa": fh["kappa"],
                                "n_msgs_removed": len(NIGHT_SCRIPT) - len(sub)}

    return S


def main():
    run_all()
    S = analyze()
    out = os.path.join(NIGHTS_DIR, "summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(S, f, indent=1)
    print(json.dumps(S, indent=1))
    print(f"[summary] written to {out}")


if __name__ == "__main__":
    main()
