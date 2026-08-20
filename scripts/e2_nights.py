"""E2 nights — the schedule-diversified corpus (chapter 7 §7.4, item 1).

Six new nights over the SAME text sources as nights A–D (SEG1 warm bank,
SEG2 cynical bank, DRIFTER_LINES — imported verbatim from nights_abc.py),
five registered schedule families:

  S1  canonical warm→cynical, flip@20            (SEG1+SEG2, 40 msgs)
  S2  early flip@8                               (SEG1[:8]+SEG2, 28 msgs)
  S3  late flip@20                               (SEG1+SEG2[:8], 28 msgs)
  S4a newcomer entry PRE-flip: cold entry @12     (46 msgs, staged persona)
  S4b newcomer entry POST-flip: cold entry @28    (45 msgs, staged persona)
  S5  no-flip control (warm only, 20 msgs)

Protocol correction (binding): the newcomer enters COLD in S4a/S4b — staged
persona engaged at first speak, NOT rostered at open, no pre-entry
acclimation (night D rostered him from open; that treatment is frozen in the
old data and is not repeated).

Roster: the 6 original occupants (frozen verbatim) + 8 new personas from
data/e2/e2-personas.json (seeded draw, committed at creation) with the
registered attendance plan; every reader appears in >=2 schedule families.
All nights emit v:2 per-reader logs (additive; v:1 nights are never
regenerated). Old corpus files are never touched.

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

from scripts.nights_abc import DRIFTER_LINES, SEG1, SEG2, _cast, _newcomer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NIGHTS_DIR = os.path.join(ROOT, "data", "nights")
PERSONAS = os.path.join(ROOT, "data", "e2", "e2-personas.json")
MANIFEST = os.path.join(ROOT, "data", "e2", "e2-nights-manifest.json")

ORIG6 = ["writer", "poet", "essayist", "engineer", "critic", "captain"]

# Registered attendance (E2 registration R4).
ATTENDANCE = {
    "S1": ORIG6 + ["barkeep", "fiddler", "cartographer"],
    "S2": ORIG6 + ["singer", "lamplighter", "tinker"],
    "S3": ORIG6 + ["blacksmith", "weaver", "barkeep"],
    "S4a": ORIG6 + ["fiddler", "singer"],          # + drifter staged-cold @12
    "S4b": ORIG6 + ["cartographer", "blacksmith", "tinker"],  # + drifter @28
    "S5": ORIG6 + ["lamplighter", "weaver"],
    # addendum 3 (non-monotonic families, same text banks):
    "S6": ORIG6 + ["barkeep", "singer", "weaver"],
    "S7": ORIG6 + ["fiddler", "lamplighter", "blacksmith"],
}

# Registered schedules (occupant indices for drifter insertion).
S4A_INSERT_AFTER = [11, 14, 17, 20, 23, 26]        # entry seq 12 (warm era)
S4B_INSERT_AFTER = [27, 30, 33, 36, 38]            # entry seq 28 (cynical era)


def _insert(script, lines, after_indices):
    out, k = [], 0
    for i, line in enumerate(script):
        out.append(line)
        if k < len(after_indices) and i == after_indices[k]:
            out.append(lines[k])
            k += 1
    assert k == len(after_indices)
    return out


def _oscillate(a, b, block=5):
    """Alternate a/b in `block`-line blocks (a first)."""
    out = []
    for i in range(0, max(len(a), len(b)), block):
        out += a[i:i + block] + b[i:i + block]
    return out


def scripts():
    night_script = SEG1 + SEG2
    return {
        "S1": list(night_script),
        "S2": SEG1[:8] + SEG2,
        "S3": SEG1 + SEG2[:8],
        "S4a": _insert(night_script, DRIFTER_LINES, S4A_INSERT_AFTER),
        "S4b": _insert(night_script, DRIFTER_LINES[:5], S4B_INSERT_AFTER),
        "S5": list(SEG1),
        # addendum 3: non-monotonic families (registered compositions)
        "S6": SEG1[:10] + SEG2[:10] + SEG1[10:20] + SEG2[10:20],  # double reversal
        "S7": _oscillate(SEG1, SEG2, block=5),                    # oscillation
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
                   [new_participant(personas()[n]) for n in names if n not in by_name]
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
    existing = [f for f in (f"night-{t}.jsonl" for t in sp)
                if os.path.exists(os.path.join(NIGHTS_DIR, f))]
    if existing and not verify:
        # addendum-3 mode: only S6/S7 may be missing; the 6 base nights are
        # frozen artifacts and must already exist.
        missing = [t for t in ("S6", "S7")
                   if not os.path.exists(os.path.join(NIGHTS_DIR, f"night-{t}.jsonl"))]
        if not missing:
            sys.exit(f"REFUSING to overwrite existing nights: {existing} "
                     f"(append-only corpus)")
    if verify:
        verify_determinism(sp)
        return

    manifest_path = MANIFEST
    manifest = {"generated": "2026-08-19", "reader_schema": 2, "nights": {}}
    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path, encoding="utf-8"))
    for tag in ("S1", "S2", "S3", "S4a", "S4b", "S5", "S6", "S7"):
        staged = tag in ("S4a", "S4b")
        if os.path.exists(os.path.join(NIGHTS_DIR, f"night-{tag}.jsonl")):
            continue  # append-only: addendum-3 nights only
        path = run_night(tag, sp[tag], ATTENDANCE[tag], staged)
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        speaks = [r for r in rows if r["type"] == "speak"]
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
            "schedule_family": {"S1": "warm→cynical canonical (flip@20)",
                                "S2": "early flip@8",
                                "S3": "late flip@20",
                                "S4a": "newcomer cold entry pre-flip @12",
                                "S4b": "newcomer cold entry post-flip @28",
                                "S5": "no-flip control",
                                "S6": "double reversal warm->cyn->warm->cyn",
                                "S7": "oscillation (alternating 5-line blocks)"}[tag],
        }
        print(f"[e2-nights] night-{tag}: {len(speaks)} msgs, "
              f"roster={len(manifest['nights'][tag]['roster'])}"
              + (f", cold entry @ {entry}" if entry is not None else ""))

    # determinism check: regenerate into a temp dir, compare stripped md5s
    with tempfile.TemporaryDirectory() as tmp:
        for tag in manifest["nights"]:
            run_night(tag, sp[tag], ATTENDANCE[tag], tag in ("S4a", "S4b"),
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
            run_night(tag, sp[tag], ATTENDANCE[tag], tag in ("S4a", "S4b"),
                      outdir=tmp)
            md5 = stripped_md5(os.path.join(tmp, f"night-{tag}.jsonl"))
            ok = md5 == meta["stripped_md5"]
            print(f"  night-{tag}: {'OK' if ok else 'MISMATCH'}")
            assert ok, tag
    print("[verify] all nights reproduce")


if __name__ == "__main__":
    generate(verify="--verify" in sys.argv)
