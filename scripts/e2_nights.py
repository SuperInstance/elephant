"""E2 nights wave-2 (Stage-2 corpus) — the x-resolved attendance redesign.

Stage-2 (STAGE2-CORPUS-DESIGN-2026-08-20.md §5.1): a FRESH corpus wave on the
SAME frozen script families/strata as the E2 S-series — new logs (new
rosters) under T-tags, one log per ladder family:

  T1  = S1 family      (canonical warm→cynical flip@20, SEG1+SEG2, 40 msgs)
  T2  = S2 family      (early flip@8, SEG1[:8]+SEG2, 28 msgs)
  T3  = A family       (canonical warm→cynical; A/S1 warmth-identical pair,
                        SEG1+SEG2, 40 msgs — the design's A↔S1 swap slot)
  T4a = S4a family     (newcomer cold entry PRE-flip @12, 46 msgs, staged)
  T4b = S4b family     (newcomer cold entry POST-flip @28, 45 msgs, staged)
  T5  = D family       (newcomer entry @24, night_d_script, 46 msgs, staged)
  T5c = D-cold family  (same 46-msg script as D; cold entry @24, staged)
  T8  = S3 family      (late flip@20, SEG1+SEG2[:8], 28 msgs)
  T9  = S5 family      (no-flip control, SEG1 only, 20 msgs)

The T4a/T4b/T5/T5c drifter lines are spoken by the STAGED drifter (cold entry
at his first speak) — the DRIFTER_LINES text is part of each family's warmth
(roster-invariant by measurement), and the drifter's measurement attendance
is exactly the design's matrix (T4a, T4b, T2). His staged appearances on
T5/T5c are warmth-content only, not attendance (FIELD_NIGHTS_W2 excludes
them).

Attendance: the §2 matrix of STAGE2-CORPUS-DESIGN-2026-08-20.md (21 readers,
unique 3–4 night subsets, three x-bands ≈0.48/0.64/0.71, night loads 7–10).

All nights emit v:2 per-reader logs (additive; v:1 and S-series nights are
never regenerated — append-only corpus). Manifest: data/e2/
e2-nights-manifest-w2.json (the filed wave-1 manifest is untouched).

Run:  python3 scripts/e2_nights.py            (generate; refuses overwrite)
      python3 scripts/e2_nights.py --verify   (determinism re-run check)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elephant.tapnight import Participant, TapNightSession

from scripts.nights_abc import (DRIFTER_LINES, SEG1, SEG2, _cast, _newcomer,
                                night_d_script)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NIGHTS_DIR = os.path.join(ROOT, "data", "nights")
PERSONAS = os.path.join(ROOT, "data", "e2", "e2-personas.json")
MANIFEST = os.path.join(ROOT, "data", "e2", "e2-nights-manifest-w2.json")

ORIG6 = ["writer", "poet", "essayist", "engineer", "critic", "captain"]

# Wave-2 attendance (STAGE2-CORPUS-DESIGN-2026-08-20.md §2, session-open
# rosters; staged drifter on T4a/T4b/T5/T5c is engine mechanics, not roster).
ATTENDANCE = {
    "T1": ["lamplighter", "poet", "singer", "cartographer",
           "blacksmith", "captain", "new-4", "new-5", "new-6"],
    "T2": ["writer", "engineer", "drifter", "lamplighter", "tinker",
           "new-1", "new-2"],
    "T3": ["tinker", "critic", "singer", "cartographer", "blacksmith",
           "barkeep", "new-6"],
    "T4a": ["writer", "engineer", "lamplighter", "tinker", "new-1", "new-2"],
    "T4b": ["new-2", "poet", "critic", "cartographer", "blacksmith",
            "new-3", "essayist", "new-5"],
    "T5": ["writer", "new-1", "new-2", "poet", "critic", "singer",
           "blacksmith", "new-3", "new-4", "fiddler"],
    "T5c": ["engineer", "new-1", "poet", "critic", "singer", "cartographer",
            "weaver", "new-3", "new-4"],
    "T8": ["essayist", "captain", "barkeep", "fiddler", "weaver", "new-5",
           "new-6"],
    "T9": ["essayist", "captain", "barkeep", "fiddler", "weaver", "new-5",
           "new-6"],
}

# Nights whose family contains DRIFTER_LINES: the drifter is staged (cold
# entry at his first speak), reproducing the family's text verbatim.
STAGED_TAGS = ("T4a", "T4b", "T5", "T5c")

# Registered schedules (occupant indices for drifter insertion).
S4A_INSERT_AFTER = [11, 14, 17, 20, 23, 26]        # entry seq 12 (warm era)
S4B_INSERT_AFTER = [27, 30, 33, 36, 38]            # entry seq 28 (cynical era)
D_INSERT_AFTER = [23, 26, 29, 32, 35, 38]          # entry seq 24 (D family)


def _insert(script, lines, after_indices):
    out, k = [], 0
    for i, line in enumerate(script):
        out.append(line)
        if k < len(after_indices) and i == after_indices[k]:
            out.append(lines[k])
            k += 1
    assert k == len(after_indices)
    return out


def scripts():
    night_script = SEG1 + SEG2
    d_script = night_d_script()  # D family verbatim (46 msgs, drifter @24)
    return {
        "T1": list(night_script),                       # S1 family
        "T2": SEG1[:8] + SEG2,                          # S2 family
        "T3": list(night_script),                       # A family (≡S1 warmth)
        "T4a": _insert(night_script, DRIFTER_LINES, S4A_INSERT_AFTER),
        "T4b": _insert(night_script, DRIFTER_LINES[:5], S4B_INSERT_AFTER),
        "T5": list(d_script),                           # D family
        "T5c": list(d_script),                          # D-cold family
        "T8": SEG1 + SEG2[:8],                          # S3 family
        "T9": list(SEG1),                               # S5 family
    }


def personas():
    doc = json.load(open(PERSONAS, encoding="utf-8"))
    return {n: p for n, p in doc["new_personas"].items()}


def new_participant(p):
    return Participant(p["name"], dial_weights=p["dial_weights"],
                       acclimation_rate=p["acclimation_rate"],
                       charisma=p["charisma"], vibe=p["vibe_start"])


def run_night(tag, script, names, staged, outdir=NIGHTS_DIR):
    path = os.path.join(outdir, f"night-{tag}.jsonl")
    cast = _cast()  # writer, poet, essayist, engineer, critic, captain (order)
    by_name = {p.name: p for p in cast}
    participants = [by_name[n] for n in ORIG6 if n in names] + \
                   [new_participant(personas()[n]) for n in names
                    if n not in by_name and n != "drifter"]
    if "drifter" in names:   # rostered-from-open (T2); staged elsewhere
        participants.append(_newcomer())
    staged_entries = {"drifter": _newcomer()} if staged else None
    s = TapNightSession("The Tap", participants=participants,
                        log_path=path, reader_schema=2,
                        staged_entries=staged_entries)
    s.start_session()
    for author, text, reactions in script:
        s.speak(author, text, reactions=reactions)
    s.end_session()
    return path


def stripped_md5(path):
    out = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        r.pop("session_id", None)
        out.append(json.dumps(r, sort_keys=True))
    return hashlib.md5("\n".join(out).encode()).hexdigest()


def generate(verify=False):
    os.makedirs(NIGHTS_DIR, exist_ok=True)
    sp = scripts()
    missing = [t for t in sp
               if not os.path.exists(os.path.join(NIGHTS_DIR, f"night-{t}.jsonl"))]
    if not missing and not verify:
        sys.exit("REFUSING to overwrite existing nights "
                 f"(all {len(sp)} T-tags present; append-only corpus)")

    if verify:
        verify_determinism(sp)
        return

    # generate only the missing tags (append-only)
    for tag in missing:
        staged = tag in STAGED_TAGS
        path = run_night(tag, sp[tag], ATTENDANCE[tag], staged)
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        speaks = [r for r in rows if r["type"] == "speak"]
        entry = None
        if staged:
            entry = next(r["seq"] for r in speaks if r["author"] == "drifter")
        print(f"[e2-nights] night-{tag}: {len(speaks)} msgs, "
              f"roster={len(next(r for r in rows if r['type']=='session_open')['roster'])}"
              + (f", cold entry @ {entry}" if entry is not None else ""))

    # rebuild the FULL manifest from disk (metadata is file-derived)
    manifest = {"generated": "2026-08-20", "reader_schema": 2,
                "wave": 2, "nights": {}}
    for tag in sp:
        path = os.path.join(NIGHTS_DIR, f"night-{tag}.jsonl")
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        speaks = [r for r in rows if r["type"] == "speak"]
        staged = tag in STAGED_TAGS
        entry = None
        if staged:
            entry = next(r["seq"] for r in speaks if r["author"] == "drifter")
        manifest["nights"][tag] = {
            "file": os.path.basename(path),
            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "stripped_md5": stripped_md5(path),
            "n_msgs": len(speaks),
            "roster": sorted(next(r for r in rows if r["type"] == "session_open")["roster"]),
            "cold_entry_seq": entry,
            "schedule_family": {"T1": "S1 family: warm→cynical canonical (flip@20)",
                                "T2": "S2 family: early flip@8",
                                "T3": "A family: canonical warm→cynical (≡S1 warmth)",
                                "T4a": "S4a family: newcomer cold entry pre-flip @12",
                                "T4b": "S4b family: newcomer cold entry post-flip @28",
                                "T5": "D family: newcomer entry @24",
                                "T5c": "D-cold family: newcomer cold entry @24",
                                "T8": "S3 family: late flip@20",
                                "T9": "S5 family: no-flip control"}[tag],
        }

    # determinism check: regenerate into a temp dir, compare stripped md5s
    with tempfile.TemporaryDirectory() as tmp:
        for tag in manifest["nights"]:
            run_night(tag, sp[tag], ATTENDANCE[tag], tag in STAGED_TAGS,
                      outdir=tmp)
            md5 = stripped_md5(os.path.join(tmp, f"night-{tag}.jsonl"))
            assert md5 == manifest["nights"][tag]["stripped_md5"], tag
            manifest["nights"][tag]["deterministic_replay_identical"] = True
    print(f"[e2-nights] determinism: all {len(manifest['nights'])} nights "
          f"byte-identical on re-run (stripped of session_id)")

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"[e2-nights] manifest -> {MANIFEST}")


def verify_determinism(sp):
    man = json.load(open(MANIFEST, encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        for tag, meta in man["nights"].items():
            run_night(tag, sp[tag], ATTENDANCE[tag], tag in STAGED_TAGS,
                      outdir=tmp)
            md5 = stripped_md5(os.path.join(tmp, f"night-{tag}.jsonl"))
            ok = md5 == meta["stripped_md5"]
            print(f"  night-{tag}: {'OK' if ok else 'MISMATCH'}")
            assert ok, tag
    print("[verify] all nights reproduce")


if __name__ == "__main__":
    generate(verify="--verify" in sys.argv)
