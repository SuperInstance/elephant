"""Night-H — the ramp-night drift-geometry redesign (registered runner-up).

Registration: research/prototype/e2-e3-side-by-side.md §7 ("Runner-up
(registered): the ramp-night drift-geometry redesign (night-H)"):

  "subtler transitions shrink the denominator (drift), giving the premise
   its best shot. Pro-premise by construction; best case = boundary-condition
   booking."

What this script does:

1. GENERATE data/nights/night-H.jsonl — one 40-message night over the SAME
   verbatim text banks as nights A/B/C (SEG1 warm-earnest, SEG2
   cynical-banter, imported from scripts/nights_abc.py) and the SAME cast
   (the 6 original occupants, frozen verbatim via nights_abc._cast()).
   The ONLY difference from the canonical S1 composition (SEG1+SEG2 with a
   hard flip at seq 20) is the TRANSITION GEOMETRY: instead of a flip, the
   script ramps the cynical fraction through five 8-message blocks —

       seq  0- 7: 8 warm + 0 cynical
       seq  8-15: 6 warm + 2 cynical
       seq 16-23: 4 warm + 4 cynical
       seq 24-31: 2 warm + 6 cynical
       seq 32-39: 0 warm + 8 cynical

   (minority lines spread as evenly as possible within each block). The
   strata are registered with the SAME convention as A/B/C — SEG1 = seq
   0-19, SEG2 = seq 20-39 — so the estimator is byte-identical and the
   comparison is apples-to-apples: same estimator, same readers, same
   strata, only the content transition is subtler (the era means are pulled
   together: SEG1 is 80% warm / 20% cynical, SEG2 is 20% warm / 80%
   cynical, vs 100/0 and 0/100 under the flip).

   Roster = the 6 original occupants ONLY (no new personas): the
   premise-measurement reader set (7 real readers: the 6 originals +
   drifter-from-D) stays EXACTLY the same. Additive, append-only: refuses
   to overwrite an existing night-H.jsonl. v:2 reader schema, consistent
   with the E2-era S-nights.

2. MEASURE with the registered premise estimator: this script imports the
   UNMODIFIED functions from scripts/premise_measurement.py (load_night,
   replay_readings, corpus_sd, fit_readers, measure, verdict, synthesize)
   and drives them over the extended corpus {A, B, C, D, D-cold, H} with
   the identical flow main() uses. It FIRST reproduces the registered
   baseline over the 5-night corpus (must print ratio 0.5599 real-only /
   0.4898 real+synthetic), THEN adds night-H and re-measures. No estimator
   code is re-implemented or invented.

3. REPORT: prints the new ratio + band verdict for real-only and
   real+synthetic, and writes NIGHT-H-REDESIGN-2026-08-20.md with the
   numbers and the honest verdict (pro-premise-by-construction caveat
   included).

Run:  python3 scripts/night_h.py            (generate + measure + report)
      python3 scripts/night_h.py --verify   (determinism re-run check only)
CPU-only, numpy only. No existing files are modified; nothing committed.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elephant.tapnight import TapNightSession
from scripts.nights_abc import SEG1, SEG2, _cast
import scripts.premise_measurement as pm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NIGHTS_DIR = os.path.join(ROOT, "data", "nights")
H_PATH = os.path.join(NIGHTS_DIR, "night-H.jsonl")
REPORT = os.path.join(ROOT, "NIGHT-H-REDESIGN-2026-08-20.md")

ORIG6 = ["writer", "poet", "essayist", "engineer", "critic", "captain"]

# Registered ramp composition: 5 blocks x 8 lines, cynical fraction
# 0 -> 1 in quarter steps. (warm_slice, cynical_slice) per block.
RAMP_BLOCKS = [
    (SEG1[0:8], SEG2[0:0]),
    (SEG1[8:14], SEG2[0:2]),
    (SEG1[14:18], SEG2[2:6]),
    (SEG1[18:20], SEG2[6:12]),
    (SEG1[20:20], SEG2[12:20]),
]


def _spread(minority, majority):
    """Interleave minority lines as evenly as possible (deterministic)."""
    if not minority:
        return list(majority)
    if not majority:
        return list(minority)
    n = len(minority) + len(majority)
    slots = [None] * n
    step = n / len(minority)
    for i, line in enumerate(minority):
        idx = min(n - 1, int(round((i + 0.5) * step)))
        while slots[idx] is not None:
            idx = (idx + 1) % n
        slots[idx] = line
    j = 0
    for i in range(n):
        if slots[i] is None:
            slots[i] = majority[j]
            j += 1
    assert j == len(majority)
    return slots


def ramp_script():
    out = []
    for warm, cyn in RAMP_BLOCKS:
        minority, majority = (cyn, warm) if len(cyn) <= len(warm) else (warm, cyn)
        out += _spread(minority, majority)
    assert len(out) == 40, len(out)
    # verbatim-bank integrity: 20 warm + 20 cynical lines, authors in ORIG6
    assert len([1 for l in out if l in SEG1]) == 20
    assert len([1 for l in out if l in SEG2]) == 20
    assert set(l[0] for l in out) <= set(ORIG6)
    return out


def stripped_md5(path):
    out = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        r.pop("session_id", None)
        out.append(json.dumps(r, sort_keys=True))
    return hashlib.md5("\n".join(out).encode()).hexdigest()


def run_night(outdir=NIGHTS_DIR):
    path = os.path.join(outdir, "night-H.jsonl")
    cast = _cast()  # writer, poet, essayist, engineer, critic, captain
    participants = [p for p in cast if p.name in ORIG6]
    assert len(participants) == 6
    s = TapNightSession("The Tap", participants=participants,
                        log_path=path, reader_schema=2, staged_entries=None)
    s.start_session()
    for author, text, reactions in ramp_script():
        s.speak(author, text, reactions=reactions)
    s.end_session()
    return path


def generate():
    if os.path.exists(H_PATH):
        sys.exit(f"REFUSING to overwrite {H_PATH} (append-only corpus)")
    path = run_night()
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    speaks = [r for r in rows if r["type"] == "speak"]
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    md5 = stripped_md5(path)
    print(f"[night-h] generated {os.path.basename(path)}: "
          f"{len(speaks)} msgs, roster={len(next(r for r in rows if r['type']=='session_open')['roster'])}")
    print(f"[night-h] sha256={sha}")
    print(f"[night-h] stripped_md5={md5}")

    # determinism: regenerate into a temp dir, compare stripped md5
    with tempfile.TemporaryDirectory() as tmp:
        run_night(outdir=tmp)
        assert stripped_md5(os.path.join(tmp, "night-H.jsonl")) == md5, "night-H"
    print("[night-h] determinism: night-H byte-identical on re-run (stripped of session_id)")
    return sha, md5


def verify():
    man = json.load(open(REPORT.replace(".md", ".json"), encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        run_night(outdir=tmp)
        md5 = stripped_md5(os.path.join(tmp, "night-H.jsonl"))
        ok = md5 == man["night_H"]["stripped_md5"]
        print(f"  night-H: {'OK' if ok else 'MISMATCH'}")
        assert ok
    print("[verify] night-H reproduces")


# --------------------------------------------------------------------------- #
# Measurement — the premise_measurement estimator, driven over the corpus.    #
# Same functions, same flow as premise_measurement.main(); only the night     #
# set differs (5 nights for the baseline reproduction, +H for the redesign).  #
# --------------------------------------------------------------------------- #
def strata_for(night, speaks):
    """Registered strata: H uses the SAME SEG1/SEG2 convention as A/B/C."""
    if night == "H":
        return {"SEG1": [r for r in speaks if r["seq"] <= 19],
                "SEG2": [r for r in speaks if r["seq"] >= 20]}
    return pm._strata_for(night, speaks)


def measure_corpus(night_files, label):
    """Drive the unmodified premise_measurement functions over a night set.
    Returns the full result dict (mirrors pm.main()'s flow exactly)."""
    rosters = {}
    for night, fn in night_files.items():
        roster, speaks = pm.load_night(os.path.join(NIGHTS_DIR, fn))
        rosters[night] = roster
        pm.ALL_SPEAKS[night] = speaks
        pm.STRATA[night] = strata_for(night, speaks)

    sd_scalar, sd_per_dial = pm.corpus_sd(list(pm.ALL_SPEAKS.values()))

    # Harvest real readers exactly as pm.main() does (union of rosters;
    # params from the first roster that defines them; nights = every night
    # whose roster contains them; drifter from D only).
    real = {}
    for night, roster in rosters.items():
        for name, entry in roster.items():
            if name not in real:
                real[name] = {"params": dict(entry), "nights": {}}
            real[name]["nights"][night] = name

    fitted_real = pm.fit_readers(real)
    m_real = pm.measure(fitted_real, sd_scalar)

    synth = pm.synthesize(real)
    merged = dict(real)
    merged.update(synth)
    fitted_all = pm.fit_readers(merged)
    m_all = pm.measure(fitted_all, sd_scalar)

    return {"label": label, "night_files": dict(night_files),
            "n_real": len(real), "n_total": len(merged),
            "sd_scalar": sd_scalar, "sd_per_dial": sd_per_dial,
            "real": m_real, "all": m_all,
            "readers": sorted(real),
            "nights_per_reader": {n: sorted(real[n]["nights"]) for n in real}}


def fmt_verdict(r):
    return pm.verdict(r)


def main():
    sha, md5 = generate()

    base_files = dict(pm.NIGHT_FILES)                 # A, B, C, D, D-cold
    h_files = dict(base_files)
    h_files["H"] = "night-H.jsonl"

    print("\n" + "=" * 74)
    print("NIGHT-H — RAMP-NIGHT DRIFT-GEOMETRY REDESIGN (premise measurement)")
    print("=" * 74)

    # 1) Continuity: the registered baseline must reproduce exactly.
    base = measure_corpus(base_files, "baseline (A,B,C,D,D-cold)")
    assert abs(base["real"]["ratio"] - 0.5599) < 1e-3, base["real"]["ratio"]
    assert abs(base["all"]["ratio"] - 0.4898) < 1e-3, base["all"]["ratio"]
    print("\n[1] CONTINUITY CHECK — registered baseline reproduced:")
    print(f"    real-only      ratio={base['real']['ratio']:.4f} "
          f"-> {fmt_verdict(base['real']['ratio'])}")
    print(f"    real+synthetic ratio={base['all']['ratio']:.4f} "
          f"-> {fmt_verdict(base['all']['ratio'])}")

    # 2) The redesign: same estimator, same readers, + night-H.
    h = measure_corpus(h_files, "with night-H (ramp transition)")
    r_real, r_all = h["real"]["ratio"], h["all"]["ratio"]
    v_real, v_all = fmt_verdict(r_real), fmt_verdict(r_all)

    print("\n[2] NIGHT-H COMPOSITION (registered, 40 msgs, verbatim banks):")
    print("    seq  0- 7: 8 warm + 0 cynical | seq  8-15: 6 warm + 2 cynical")
    print("    seq 16-23: 4 warm + 4 cynical | seq 24-31: 2 warm + 6 cynical")
    print("    seq 32-39: 0 warm + 8 cynical")
    print("    strata: SEG1 = seq 0-19, SEG2 = seq 20-39 (same convention as A/B/C)")
    print("    roster: 6 originals only -> real reader set unchanged: "
          f"{h['readers']}")

    print("\n[3] CORPUS SCALE:")
    print(f"    baseline: corpus_sd = {base['sd_scalar']:.4f} "
          f"({sum(len(pm.ALL_SPEAKS[n]) for n in base_files)} speak events)")
    print(f"    with H:   corpus_sd = {h['sd_scalar']:.4f} "
          f"({sum(len(pm.ALL_SPEAKS[n]) for n in h_files)} speak events)")

    print("\n[4] PER-NIGHT TRANSITION DRIFT (mean over readers, corpus-sd):")
    for k, v in base["real"]["transitions"].items():
        print(f"    {k:<30} {v:.4f}   [baseline]")
    for k, v in h["real"]["transitions"].items():
        mark = "   <-- night-H (ramp)" if k.startswith("H:") else "   [with H]"
        print(f"    {k:<30} {v:.4f}{mark}")

    print("\n[5] THE KILL NUMBER:")
    print(f"    REAL ONLY:      spread {h['real']['spread_z']:.4f} / "
          f"drift {h['real']['mean_drift_z']:.4f} = {r_real:.4f}   "
          f"(baseline {base['real']['ratio']:.4f})")
    print(f"    REAL+SYNTHETIC: spread {h['all']['spread_z']:.4f} / "
          f"drift {h['all']['mean_drift_z']:.4f} = {r_all:.4f}   "
          f"(baseline {base['all']['ratio']:.4f})")
    print(f"    kill band = [{pm.KILL_LO}, {pm.KILL_HI}] corpus-sd")
    print(f"    VERDICT (real-only):      {v_real}")
    print(f"    VERDICT (real+synthetic): {v_all}")
    print(f"    robustness (vs-own-baseline): real {h['real']['ratio_vs_base']:.4f} "
          f"| real+synth {h['all']['ratio_vs_base']:.4f}")

    print("\n[6] ONE-SENTENCE VERDICTS:")
    print(f"    real-only:       ratio={r_real:.4f} -> {v_real}  "
          f"(baseline 0.5599)")
    print(f"    real+synthetic:  ratio={r_all:.4f} -> {v_all}  "
          f"(baseline 0.4898)")

    # Per-reader drift detail for the report
    per_reader = {}
    for name in sorted(h["real"]["drift"]):
        per_reader[name] = {"drift": h["real"]["drift"][name],
                            "vs_base": h["real"]["drift_vs_base"][name]}

    # ---------------------------------------------------------------- #
    # Report                                                                 #
    # ---------------------------------------------------------------- #
    moved = "CLEARED" if (r_real > pm.KILL_HI or r_all > pm.KILL_HI) else \
            ("STAYED" if r_real >= pm.KILL_LO and r_all >= pm.KILL_LO else "MOVED")
    crossed = "yes" if (r_real > pm.KILL_HI) != (base["real"]["ratio"] > pm.KILL_HI) \
              or (r_all > pm.KILL_HI) != (base["all"]["ratio"] > pm.KILL_HI) else "no"

    results = {
        "generated": "2026-08-20", "script": "scripts/night_h.py",
        "night_H": {"file": "data/nights/night-H.jsonl", "sha256": sha,
                    "stripped_md5": md5, "n_msgs": 40,
                    "roster": ORIG6,
                    "composition": "5x8 ramp blocks, cynical fraction 0->1",
                    "strata": "SEG1 seq 0-19 / SEG2 seq 20-39 (same as A/B/C)"},
        "baseline": {"ratio_real": base["real"]["ratio"],
                     "ratio_all": base["all"]["ratio"],
                     "verdict_real": fmt_verdict(base["real"]["ratio"]),
                     "verdict_all": fmt_verdict(base["all"]["ratio"])},
        "with_H": {"ratio_real": r_real, "ratio_all": r_all,
                   "spread_z_real": h["real"]["spread_z"],
                   "drift_z_real": h["real"]["mean_drift_z"],
                   "spread_z_all": h["all"]["spread_z"],
                   "drift_z_all": h["all"]["mean_drift_z"],
                   "ratio_vs_base_real": h["real"]["ratio_vs_base"],
                   "ratio_vs_base_all": h["all"]["ratio_vs_base"],
                   "verdict_real": v_real, "verdict_all": v_all,
                   "transitions": dict(h["real"]["transitions"]),
                   "per_reader": per_reader,
                   "n_real": h["n_real"], "n_total": h["n_total"],
                   "band": [pm.KILL_LO, pm.KILL_HI],
                   "band_crossed": crossed, "movement": moved},
    }
    with open(REPORT.replace(".md", ".json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"\n[night-h] results -> {REPORT.replace('.md', '.json')}")

    print("\n" + "=" * 74)
    print(f"FINAL: real-only ratio = {r_real:.4f}  ({v_real})")
    print(f"FINAL: real+synthetic ratio = {r_all:.4f}  ({v_all})")
    print(f"FINAL: band verdict moved? {crossed.upper()} — "
          f"baseline 0.5599/0.4898 -> {r_real:.4f}/{r_all:.4f}")
    print("=" * 74)


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify()
    else:
        main()
