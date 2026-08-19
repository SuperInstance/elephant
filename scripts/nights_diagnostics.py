"""Nights diagnostics — D′ (cold-entry newcomer) + pre-registered checks.

Companion to scripts/nights_abc.py (which must have been run first). This
script:

1. Generates data/nights/night-D-cold.jsonl — the SAME 46-message script as
   Night D, but the drifter joins the ROSTER at entry (seq 24) instead of at
   session open. Runner-side roster timing only; no engine changes. Why: the
   shipped engine acclimates every rostered participant from message 0, so
   the Night-D drifter is already pre-warmed at entry (measured distance at
   entry ≈ 0.001) and the post-entry acclimation curve is flat by construction.
   The cold-entry variant is the spec §3.5 newcomer in the strict sense
   ("entering at seq ≈ 60%"), and yields a real acclimation curve.
   Deviation to note: session_open's roster does not contain the drifter in
   this variant (the persona + vibe_start live in this runner).

2. Adds the pre-registered diagnostics to data/nights/summary.json:
   - κ W-sensitivity sweep (W ∈ {4, 8, 16}) on seg1 / seg2 / full night / TTRPG.
   - corr(warmth_vmf, log κ) tripwire across room-level fits (|r| > 0.8 ⇒
     investigate; the warmth/κ confound surviving).
   - post-hoc deadband flags (d_mu vs 2·max SE) per stretch — how often the
     shipped deadband fires on stable / transition / post-entry edges.

Run:  python3 scripts/nights_diagnostics.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elephant.tapnight import TapNightSession
from elephant.room import Message, Room
from elephant.dial import DialBank
from elephant.dials import DEFAULT_DIALS
from elephant.vmf import vmf_fit, windowed

from nights_abc import (NIGHTS_DIR, NIGHT_SCRIPT, SEG1, SEG2, TTRPG_EXTENSION,
                        _cast, _newcomer, night_d_script, seg_fit, chord,
                        cosdist, load, speaks, W)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(NIGHTS_DIR, "summary.json")


def run_d_cold():
    path = os.path.join(NIGHTS_DIR, "night-D-cold.jsonl")
    if os.path.exists(path):
        return path
    script = night_d_script()
    s = TapNightSession("The Tap", participants=_cast(), log_path=path)
    s.start_session()
    added = False
    for author, text, reactions in script:
        if author == "drifter" and not added:
            # roster-at-entry: the cold newcomer joins the table now
            p = _newcomer()
            s.participants["drifter"] = p
            s._vibe["drifter"] = p.vibe.copy()
            s._vibe_start["drifter"] = p.vibe.copy()
            added = True
        s.speak(author, text, reactions=reactions)
    s.end_session()
    assert added
    return path


def acclim_replay(rows, vibe0, rate, entry):
    """Replay one participant's vibe from field_eff_after (log-only)."""
    alpha = 1.0 - np.exp(-rate)
    vibe = np.asarray(vibe0, float).copy()
    dist, seqs = [], []
    for r in rows:
        field = np.array(r["field_eff_after"])
        if r["seq"] >= entry:
            dist.append(float(np.linalg.norm(vibe - field)))
            seqs.append(r["seq"])
        vibe = vibe + (field - vibe) * alpha
    return np.array(dist), seqs


def main():
    run_d_cold()
    S = json.load(open(SUMMARY, encoding="utf-8"))

    A = speaks(load(os.path.join(NIGHTS_DIR, "night-A.jsonl")))
    D = speaks(load(os.path.join(NIGHTS_DIR, "night-D.jsonl")))
    Dc = speaks(load(os.path.join(NIGHTS_DIR, "night-D-cold.jsonl")))
    entry = S["nightD"]["entry_seq"]

    # identical message streams D vs D'
    same_stream = all(d["author"] == c["author"] and d["text_sha256"] == c["text_sha256"]
                      for d, c in zip(D, Dc)) and len(D) == len(Dc)
    pre_same_as_A = all(a["author"] == c["author"] and a["text_sha256"] == c["text_sha256"]
                        for a, c in zip(A[:24], Dc[:24]))
    # raw-field trajectory must be identical D vs D' (roster timing does not
    # touch the raw bank); effective field differs post-entry (charisma).
    raw_same = all(d["field_raw_after"] == c["field_raw_after"] for d, c in zip(D, Dc))
    eff_diverge = next(i for i, (d, c) in enumerate(zip(D, Dc))
                       if d["field_eff_after"] != c["field_eff_after"])

    newcomer = _newcomer()
    dist_cold, seqs_cold = acclim_replay(Dc, newcomer.vibe.tolist(),
                                         newcomer.acclimation_rate, entry)
    slope_cold = float(np.polyfit(np.arange(len(dist_cold)), dist_cold, 1)[0])
    S["nightD_cold"] = {
        "entry_seq": entry,
        "same_stream_as_D": same_stream,
        "pre_entry_matches_A": pre_same_as_A,
        "raw_field_identical_to_D": raw_same,
        "field_eff_first_divergence_seq": eff_diverge,
        "dist_at_entry": float(dist_cold[0]),
        "dist_final": float(dist_cold[-1]),
        "acclim_slope_per_msg": slope_cold,
        "acclim_curve": [round(float(x), 4) for x in dist_cold],
        "half_life_msgs": float(np.log(2) / (-slope_cold / max(dist_cold[0], 1e-9)))
        if slope_cold < 0 else None,
    }

    # charisma observable, cold variant (alignment with NATIVE cold vibe)
    def charisma_stats(rows):
        mags, aligns = [], []
        v0 = np.asarray(newcomer.vibe.tolist(), float)
        for r in rows:
            raw = np.array(r["field_raw_after"])
            delta = np.array(r["field_eff_after"]) - raw
            vd = v0 - raw
            n = np.linalg.norm(vd)
            mags.append(float(np.linalg.norm(delta)))
            if n > 1e-9:
                aligns.append(float(np.dot(delta, vd / n)))
        return float(np.mean(mags)), float(np.mean(aligns))

    pre_rows = [r for r in Dc if r["seq"] < entry]
    post_rows = [r for r in Dc if r["seq"] >= entry]
    mag_pre, align_pre = charisma_stats(pre_rows)
    mag_post, align_post = charisma_stats(post_rows)
    S["nightD_cold"]["charisma_mag_pre"] = mag_pre
    S["nightD_cold"]["charisma_mag_post"] = mag_post
    S["nightD_cold"]["charisma_align_pre"] = align_pre
    S["nightD_cold"]["charisma_align_post"] = align_post

    # ---- W-sensitivity sweep -------------------------------------------- #
    ttrpg_theme_msgs = None
    from elephant.tapnight_themes import THEMES
    th = THEMES["ttrpg"]
    ttrpg_msgs = th.room_tone + TTRPG_EXTENSION
    sweep = {}
    for label, msgs in (("seg1", SEG1), ("seg2", SEG2),
                        ("full_night", NIGHT_SCRIPT), ("ttrpg", ttrpg_msgs)):
        bank = DialBank(DEFAULT_DIALS)
        room = Room("sweep", [Message(a, t, ts=float(i), reactions=r)
                              for i, (a, t, r) in enumerate(msgs)])
        sweep[label] = {str(Wx): (lambda f: {"kappa": f["kappa"], "n": f["n"]})(vmf_fit(windowed(room, bank, W=Wx)))
                        for Wx in (4, 8, 16)}
    S["w_sweep"] = sweep

    # ---- warmth / log-kappa confound tripwire ---------------------------- #
    pts = []
    for key, fit_key in (("seg1", "seg1"), ("seg2", "seg2")):
        f = S["nights"]["A"]
        pts.append((f[f"{fit_key}_warmth_vmf"], f[f"{fit_key}_kappa"]))
    pts.append((S["nights"]["A"]["final_warmth_vmf"], S["nights"]["A"]["final_kappa"]))
    pts.append((S["coarse_anchor"]["ttrpg_warmth_vmf"], S["coarse_anchor"]["ttrpg_kappa"]))
    pts.append((None, S["nightD"]["kappa_pre"]))  # warmth filled below
    pts[-1] = (float(np.dot(__import__("elephant.vmf", fromlist=["WARM"]).WARM,
                            S["nightD"]["mu_pre"])), S["nightD"]["kappa_pre"])
    pts.append((float(np.dot(__import__("elephant.vmf", fromlist=["WARM"]).WARM,
                             S["nightD"]["mu_post"])), S["nightD"]["kappa_post"]))
    w = np.array([p[0] for p in pts])
    lk = np.log([max(p[1], 1e-9) for p in pts])
    r_conf = float(np.corrcoef(w, lk)[0, 1])
    S["warmth_logkappa_corr"] = {"r": r_conf, "n_fits": len(pts),
                                 "tripwire_0.8": bool(abs(r_conf) > 0.8)}

    # ---- post-hoc deadband flags ----------------------------------------- #
    def deadband_flags(rows, lo, hi):
        fits = {r["seq"]: r["fit"] for r in rows if r["fit"]}
        flagged = tot = 0
        for r in rows:
            sq = r["seq"]
            if not (lo <= sq <= hi) or sq - 1 not in fits or r["fit"] is None:
                continue
            f0, f1 = fits[sq - 1], r["fit"]
            d = float(np.linalg.norm(np.array(f1["mu_hat"]) - np.array(f0["mu_hat"])))
            db = 2 * max(f0["mu_se"], f1["mu_se"])
            tot += 1
            flagged += bool(d > db)
        return flagged, tot

    S["deadband_posthoc"] = {}
    for label, rows, lo, hi in (
            ("A_stable_seg1", A, 10, 19), ("A_stable_seg2", A, 30, 39),
            ("A_transition", A, 20, 27), ("D_post_entry", Dc, entry, len(Dc) - 1)):
        f, t = deadband_flags(rows, lo, hi)
        S["deadband_posthoc"][label] = {"flagged": f, "total": t}

    with open(SUMMARY, "w", encoding="utf-8") as f:
        json.dump(S, f, indent=1)
    print(json.dumps({k: S[k] for k in ("nightD_cold", "w_sweep",
                                        "warmth_logkappa_corr",
                                        "deadband_posthoc")}, indent=1))
    print("[diagnostics] merged into", SUMMARY)


if __name__ == "__main__":
    main()
