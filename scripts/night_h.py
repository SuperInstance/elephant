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

GEOMETRY SWEEP EXTENSION (2026-08-20, drift-geometry follow-up): night-H's
ramp landed at 0.5980 real-only — 0.002 short of the 0.6 clear edge. This
extension adds three further drift geometries over the SAME verbatim banks,
cast, roster, strata convention, and estimator, to see whether ANY geometry
clears 0.6 or exits the band:

  H2 — multi-stage ramp: 8 blocks x 5 lines, cynical fraction
       0 -> 1 in finer steps (0, .2, .4, .4, .6, .6, .8, 1.0);
       SEG1 ends 75% warm / 25% cynical (vs H's 80/20).
  H3 — oscillation with tightening envelope: 8 blocks x 5 lines, cynical
       fraction oscillates around 0.5 with amplitude 0.5 -> 0.1
       (0, 1.0, 0.2, 0.8, 0.4, 0.6, 0.4, 0.6); SEG1 and SEG2 both land
       at 50/50 — the era means nearly coincide (the extreme pro-premise
       geometry: the transition is real in the text but invisible to a
       SEG1/SEG2 split).
  H4 — long-dwell-plateau: 5 blocks x 8 lines, cynical fraction
       0 -> 0.5 (24-msg plateau) -> 1.0 (0, .5, .5, .5, 1.0).

Each variant is measured as baseline (A,B,C,D,D-cold) + that one geometry
night — same estimator, same readers, same flow as the night-H run. The
sweep writes NIGHT-H-GEOMETRY-SWEEP-2026-08-20.json. Every gain is
pro-premise by construction (denominator shrink); the report says so.

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
SWEEP_REPORT = os.path.join(ROOT, "NIGHT-H-GEOMETRY-SWEEP-2026-08-20.md")

ORIG6 = ["writer", "poet", "essayist", "engineer", "critic", "captain"]


def _slices(counts, sizes):
    """(warm_slice, cynical_slice) per block from cynical counts + block sizes."""
    assert sum(counts) == 20 and sum(sizes) == 40
    assert all(c <= s for c, s in zip(counts, sizes))
    blocks, wi, ci = [], 0, 0
    for c, s in zip(counts, sizes):
        w = s - c
        blocks.append((SEG1[wi:wi + w], SEG2[ci:ci + c]))
        wi += w
        ci += c
    assert wi == 20 and ci == 20
    return blocks


# Registered ramp composition: 5 blocks x 8 lines, cynical fraction
# 0 -> 1 in quarter steps. (warm_slice, cynical_slice) per block.
RAMP_BLOCKS = [
    (SEG1[0:8], SEG2[0:0]),
    (SEG1[8:14], SEG2[0:2]),
    (SEG1[14:18], SEG2[2:6]),
    (SEG1[18:20], SEG2[6:12]),
    (SEG1[20:20], SEG2[12:20]),
]

# Geometry sweep variants (drift-geometry follow-up, 2026-08-20). Same banks,
# cast, roster, strata convention; only the transition geometry differs.
# name -> (cynical counts per block, block sizes, one-line description)
GEOMETRIES = {
    "H": (None, None, "ramp: 5x8 blocks, cynical 0 -> 1 in quarter steps"),
    "H2": ([0, 1, 2, 2, 3, 3, 4, 5], [5] * 8,
           "multi-stage ramp: 8x5 blocks, cynical 0,.2,.4,.4,.6,.6,.8,1"),
    "H3": ([0, 5, 1, 4, 2, 3, 2, 3], [5] * 8,
           "oscillation, tightening envelope: cynical 0,1,.2,.8,.4,.6,.4,.6"),
    "H4": ([0, 4, 4, 4, 8], [8] * 5,
           "long-dwell-plateau: cynical 0 -> .5 (24-msg plateau) -> 1"),
}


def geometry_blocks(name):
    if name == "H":
        return RAMP_BLOCKS
    counts, sizes, _ = GEOMETRIES[name]
    return _slices(counts, sizes)


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


def build_script(blocks):
    out = []
    for warm, cyn in blocks:
        minority, majority = (cyn, warm) if len(cyn) <= len(warm) else (warm, cyn)
        out += _spread(minority, majority)
    assert len(out) == 40, len(out)
    # verbatim-bank integrity: 20 warm + 20 cynical lines, authors in ORIG6
    assert len([1 for l in out if l in SEG1]) == 20
    assert len([1 for l in out if l in SEG2]) == 20
    assert set(l[0] for l in out) <= set(ORIG6)
    return out


def ramp_script():
    return build_script(RAMP_BLOCKS)


def stripped_md5(path):
    out = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        r.pop("session_id", None)
        out.append(json.dumps(r, sort_keys=True))
    return hashlib.md5("\n".join(out).encode()).hexdigest()


def run_night(name="H", outdir=NIGHTS_DIR):
    path = os.path.join(outdir, f"night-{name}.jsonl")
    cast = _cast()  # writer, poet, essayist, engineer, critic, captain
    participants = [p for p in cast if p.name in ORIG6]
    assert len(participants) == 6
    s = TapNightSession("The Tap", participants=participants,
                        log_path=path, reader_schema=2, staged_entries=None)
    s.start_session()
    for author, text, reactions in build_script(geometry_blocks(name)):
        s.speak(author, text, reactions=reactions)
    s.end_session()
    return path


def night_path(name):
    return os.path.join(NIGHTS_DIR, f"night-{name}.jsonl")


def generate():
    """Generate every geometry night (append-only: existing files are kept,
    not overwritten). Returns {name: (sha256, stripped_md5)}."""
    manifest = {}
    for name in GEOMETRIES:
        path = night_path(name)
        if os.path.exists(path):
            print(f"[night-{name.lower()}] {os.path.basename(path)} exists "
                  f"(append-only) — keeping, not regenerating")
        else:
            run_night(name)
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        speaks = [r for r in rows if r["type"] == "speak"]
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
        md5 = stripped_md5(path)
        print(f"[night-{name.lower()}] {os.path.basename(path)}: "
              f"{len(speaks)} msgs, roster="
              f"{len(next(r for r in rows if r['type']=='session_open')['roster'])}")
        print(f"[night-{name.lower()}] sha256={sha}")
        print(f"[night-{name.lower()}] stripped_md5={md5}")

        # determinism: regenerate into a temp dir, compare stripped md5
        with tempfile.TemporaryDirectory() as tmp:
            run_night(name, outdir=tmp)
            assert stripped_md5(os.path.join(tmp, f"night-{name}.jsonl")) == md5, name
        print(f"[night-{name.lower()}] determinism: byte-identical on re-run "
              f"(stripped of session_id)")
        manifest[name] = (sha, md5)
    return manifest


def verify():
    man = json.load(open(SWEEP_REPORT.replace(".md", ".json"), encoding="utf-8"))
    for name in GEOMETRIES:
        with tempfile.TemporaryDirectory() as tmp:
            run_night(name, outdir=tmp)
            md5 = stripped_md5(os.path.join(tmp, f"night-{name}.jsonl"))
            ok = md5 == man["nights"][name]["stripped_md5"]
            print(f"  night-{name}: {'OK' if ok else 'MISMATCH'}")
            assert ok
    print("[verify] all geometry nights reproduce")


# --------------------------------------------------------------------------- #
# Measurement — the premise_measurement estimator, driven over the corpus.    #
# Same functions, same flow as premise_measurement.main(); only the night     #
# set differs (5 nights for the baseline reproduction, +H for the redesign).  #
# --------------------------------------------------------------------------- #
def strata_for(night, speaks):
    """Registered strata: all H-family nights use the SAME SEG1/SEG2
    convention as A/B/C."""
    if night.startswith("H"):
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
    manifest = generate()

    base_files = dict(pm.NIGHT_FILES)                 # A, B, C, D, D-cold

    print("\n" + "=" * 78)
    print("NIGHT-H GEOMETRY SWEEP — DRIFT-GEOMETRY FOLLOW-UP (premise measurement)")
    print("=" * 78)

    # 1) Continuity: the registered baseline must reproduce exactly.
    base = measure_corpus(base_files, "baseline (A,B,C,D,D-cold)")
    assert abs(base["real"]["ratio"] - 0.5599) < 1e-3, base["real"]["ratio"]
    assert abs(base["all"]["ratio"] - 0.4898) < 1e-3, base["all"]["ratio"]
    print("\n[1] CONTINUITY CHECK — registered baseline reproduced:")
    print(f"    real-only      ratio={base['real']['ratio']:.4f} "
          f"-> {fmt_verdict(base['real']['ratio'])}")
    print(f"    real+synthetic ratio={base['all']['ratio']:.4f} "
          f"-> {fmt_verdict(base['all']['ratio'])}")

    # 2) Sweep: same estimator, same readers, baseline + one geometry night.
    print("\n[2] GEOMETRY VARIANTS (verbatim banks, 6 originals, SEG1/SEG2 strata):")
    for name, (_, _, desc) in GEOMETRIES.items():
        print(f"    {name:<3} {desc}")

    sweep = {}
    for name in GEOMETRIES:
        files = dict(base_files)
        files[name] = f"night-{name}.jsonl"
        m = measure_corpus(files, f"baseline + night-{name}")
        sweep[name] = m
        trans_key = f"{name}: SEG1->SEG2"
        own = m["real"]["transitions"][trans_key]
        print(f"\n[3.{name}] baseline + night-{name}:")
        print(f"      own transition drift = {own:.4f} corpus-sd "
              f"(A/B/C flip: {m['real']['transitions']['A: SEG1->SEG2']:.4f})")
        print(f"      real-only:      spread {m['real']['spread_z']:.4f} / "
              f"drift {m['real']['mean_drift_z']:.4f} = "
              f"{m['real']['ratio']:.4f}  -> {fmt_verdict(m['real']['ratio'])}")
        print(f"      real+synthetic: spread {m['all']['spread_z']:.4f} / "
              f"drift {m['all']['mean_drift_z']:.4f} = "
              f"{m['all']['ratio']:.4f}  -> {fmt_verdict(m['all']['ratio'])}")

    # 4) Cumulative: baseline + ALL geometry nights at once.
    all_files = dict(base_files)
    for name in GEOMETRIES:
        all_files[name] = f"night-{name}.jsonl"
    cum = measure_corpus(all_files, "baseline + all geometry nights")
    print(f"\n[4] CUMULATIVE (baseline + H + H2 + H3 + H4):")
    print(f"      real-only:      spread {cum['real']['spread_z']:.4f} / "
          f"drift {cum['real']['mean_drift_z']:.4f} = "
          f"{cum['real']['ratio']:.4f}  -> {fmt_verdict(cum['real']['ratio'])}")
    print(f"      real+synthetic: spread {cum['all']['spread_z']:.4f} / "
          f"drift {cum['all']['mean_drift_z']:.4f} = "
          f"{cum['all']['ratio']:.4f}  -> {fmt_verdict(cum['all']['ratio'])}")

    # 5) The table.
    print("\n[5] THE SWEEP TABLE (kill band = "
          f"[{pm.KILL_LO}, {pm.KILL_HI}] corpus-sd):")
    hdr = (f"    {'geometry':<38} {'own drift':>9} {'ratio(R)':>9} "
           f"{'ratio(R+S)':>10} {'verdict (real-only)'}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    rows = [("baseline (A,B,C,D,D-cold; flip nights)", None,
             base["real"]["ratio"], base["all"]["ratio"])]
    for name in GEOMETRIES:
        m = sweep[name]
        rows.append((f"+{name}: {GEOMETRIES[name][2]}",
                     m["real"]["transitions"][f"{name}: SEG1->SEG2"],
                     m["real"]["ratio"], m["all"]["ratio"]))
    rows.append(("+H+H2+H3+H4 (cumulative)", None,
                 cum["real"]["ratio"], cum["all"]["ratio"]))
    for label, own, rr, ra in rows:
        own_s = f"{own:.4f}" if own is not None else "—"
        print(f"    {label:<38} {own_s:>9} {rr:>9.4f} {ra:>10.4f} "
              f"{fmt_verdict(rr)}")

    any_clear = any(r[2] > pm.KILL_HI or r[3] > pm.KILL_HI for r in rows)
    any_below = any(r[2] < pm.KILL_LO or r[3] < pm.KILL_LO for r in rows)
    print(f"\n    any geometry CLEARS 0.6?   {'YES' if any_clear else 'NO'}")
    print(f"    any geometry EXITS band?   {'YES' if any_clear or any_below else 'NO'}")

    # ---------------------------------------------------------------- #
    # Machine-readable results                                             #
    # ---------------------------------------------------------------- #
    results = {
        "generated": "2026-08-20", "script": "scripts/night_h.py",
        "follow_up_of": "NIGHT-H-REDESIGN-2026-08-20.md",
        "nights": {name: {"file": f"data/nights/night-{name}.jsonl",
                          "sha256": manifest[name][0],
                          "stripped_md5": manifest[name][1], "n_msgs": 40,
                          "roster": ORIG6,
                          "geometry": GEOMETRIES[name][2],
                          "strata": "SEG1 seq 0-19 / SEG2 seq 20-39 (same as A/B/C)"}
                   for name in GEOMETRIES},
        "baseline": {"ratio_real": base["real"]["ratio"],
                     "ratio_all": base["all"]["ratio"]},
        "sweep": {name: {
            "own_transition_drift": sweep[name]["real"]["transitions"]
                [f"{name}: SEG1->SEG2"],
            "ratio_real": sweep[name]["real"]["ratio"],
            "ratio_all": sweep[name]["all"]["ratio"],
            "spread_z_real": sweep[name]["real"]["spread_z"],
            "drift_z_real": sweep[name]["real"]["mean_drift_z"],
            "spread_z_all": sweep[name]["all"]["spread_z"],
            "drift_z_all": sweep[name]["all"]["mean_drift_z"],
            "ratio_vs_base_real": sweep[name]["real"]["ratio_vs_base"],
            "ratio_vs_base_all": sweep[name]["all"]["ratio_vs_base"],
            "verdict_real": fmt_verdict(sweep[name]["real"]["ratio"]),
            "verdict_all": fmt_verdict(sweep[name]["all"]["ratio"])}
            for name in GEOMETRIES},
        "cumulative": {"ratio_real": cum["real"]["ratio"],
                       "ratio_all": cum["all"]["ratio"],
                       "verdict_real": fmt_verdict(cum["real"]["ratio"]),
                       "verdict_all": fmt_verdict(cum["all"]["ratio"])},
        "band": [pm.KILL_LO, pm.KILL_HI],
        "any_clear": any_clear, "any_exit": any_clear or any_below,
    }
    out_json = SWEEP_REPORT.replace(".md", ".json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"\n[night-h] results -> {out_json}")

    print("\n" + "=" * 78)
    print("FINAL TABLE:")
    for label, own, rr, ra in rows:
        print(f"  {label:<38} real {rr:.4f} | real+synth {ra:.4f}")
    print(f"VERDICT: any clear? {'YES' if any_clear else 'NO'} — "
          f"any band exit? {'YES' if any_clear or any_below else 'NO'}")
    print("(all movement is pro-premise by construction: denominator shrink)")
    print("=" * 78)


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify()
    else:
        main()
