"""Riverbed adapter (G5) — generated nights into the registered pipeline.

The verification of 2026-08-21 proved the generator's night JSONL is
schema-compatible with the filed wave-2 T-nights (byte-exact key sets:
session_open / speak v:2 / readers block / fit / session_close) and that
`e2_instrument.logged_readings` + `premise_band_movers.night_windows`
consume generated data unchanged. This module makes that bridge a REUSABLE
function instead of a one-off: it builds pipeline-ready Night objects,
the wave attendance map, and a Measurement over any riverbed corpus —
with the registered estimators (e2_instrument.Measurement and everything
downstream) running UNMODIFIED.

Design notes:
  - `NightFromFile` re-uses scripts/e2_instrument.Night's load semantics
    against an explicit (tag, path, strata): the registered Night keys
    NIGHT_SPECS/W2_NIGHTS by name with a hard-coded data/nights dir, so
    generated corpora (which must never touch data/nights) need their
    own loader. Same equations, different path.
  - `RiverbedMeasurement` subclasses the registered Measurement and only
    replaces night construction (self.nights = caller-provided objects);
    every estimator — _build, drift, spread, ICC, bootstrap — is the
    registered code, untouched.
  - Attendance mirrors FIELD_NIGHTS_W2/COLD_ENTRY_W2 semantics from the
    manifest: open-roster members attend; the staged entrant attends only
    on entrant_is_attendance nights (his T5/T5c line-readings are warmth
    content, not attendance — exactly the filed wave-2 rule), entering
    cold like COLD_ENTRY_W2.
  - Strata: the 9 canonical families reuse the REGISTERED W2_NIGHTS
    strata verbatim (the generated families are those families); custom
    families get derived strata (flip/entry cuts; no-flip ⇒ null kind).

Blind-corpus safe: everything here reads only redacted-manifest fields
(design facts, branch-free). Nothing in this module reads branch params.

Run (module):  python3 -c "from scripts.riverbed_adapter import load_wave; ..."
  w = load_wave("data/nights/generated/<corpus>/riverbed-manifest.json")
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.e2_instrument import (D, W2_NIGHTS, Measurement, Night,
                                   archetype_labels, corpus_sd)
from scripts.riverbed_generator import NIGHT_FAMILIES

__all__ = ["NightFromFile", "RiverbedMeasurement", "family_strata",
           "wave_attendance", "wave_cold", "build_measurement", "load_wave"]


# ----------------------------------------------------------------------- #
# Strata                                                                  #
# ----------------------------------------------------------------------- #
def _derived_strata(n, flip, entries):
    """Strata for a custom family: cuts at the flip and the entry events;
    a no-flip family is a null (control) night, mirroring the T9 rule."""
    cuts = sorted({c for c in ([0, flip] if flip is not None else [0])
                   + list(entries or []) if 0 <= c < n})
    segs = []
    for i, lo in enumerate(cuts):
        hi = (cuts[i + 1] - 1) if i + 1 < len(cuts) else n - 1
        segs.append((f"s{i}", lo, hi, "signal" if flip is not None else "null"))
    return segs


def family_strata(meta):
    """Registered strata for a manifest night meta dict. Canonical families
    (family name + schedule + n matching the generator's frozen families)
    reuse W2_NIGHTS strata VERBATIM; anything else derives them."""
    fam = meta.get("family")
    sched = meta.get("schedule", {})
    flip, entries = sched.get("flip_seq"), sched.get("entry_seqs", [])
    n = int(meta["n_msgs"])
    if fam in W2_NIGHTS and fam in NIGHT_FAMILIES:
        _, nf, ff, ef = NIGHT_FAMILIES[fam]
        strata = W2_NIGHTS[fam][1]
        if (nf == n and ff == flip and list(ef) == list(entries or [])
                and strata[-1][2] == n - 1):
            return list(strata)
    return _derived_strata(n, flip, entries)


# ----------------------------------------------------------------------- #
# Night loader (G5)                                                       #
# ----------------------------------------------------------------------- #
class NightFromFile(Night):
    """A Night over an explicit (tag, path, strata) — the registered
    Night.__init__ semantics (open/speaks/params incl. staged_entries/
    canon_n) pointed at a generated file."""

    def __init__(self, tag, path, strata):
        self.name = tag
        self.path = path
        rows = [json.loads(l) for l in open(path, encoding="utf-8")
                if l.strip()]
        self.open = next(r for r in rows if r["type"] == "session_open")
        self.speaks = [r for r in rows if r["type"] == "speak"]
        self.strata = strata
        self.v2 = "readers" in self.speaks[0]
        self.params = {}
        for n, p in self.open["roster"].items():
            self.params[n] = dict(p)
        for n, p in self.open.get("staged_entries", {}).items():
            self.params.setdefault(n, dict(p))
        for n in self.params:
            self.params[n]["dial_weights"] = np.asarray(
                self.params[n]["dial_weights"], float)
            self.params[n]["vibe_start"] = np.asarray(
                self.params[n]["vibe_start"], float)
        self.canon_n = [float(np.mean(list(r["interactions_after"].values())))
                        if r["interactions_after"] else 0.0
                        for r in self.speaks]

    def first_speak_seq(self, author):
        for r in self.speaks:
            if r["author"] == author:
                return r["seq"]
        return None


# ----------------------------------------------------------------------- #
# Measurement over generated nights (G5)                                  #
# ----------------------------------------------------------------------- #
class RiverbedMeasurement(Measurement):
    """The registered Measurement with ONLY the night construction
    redirected: self.nights come from the caller (NightFromFile objects),
    never from the hard-coded NIGHT_SPECS/W2_NIGHTS tables. Every
    downstream quantity (_build, drift, spread_seg/cont, ICC, bootstrap)
    is e2_instrument code, unmodified."""

    def __init__(self, readers_nights, sd, night_objects,
                 include_null_drift=False, presence="actual"):
        self.sd = sd
        self.nights = dict(night_objects)   # {tag: NightFromFile}
        self.include_null = include_null_drift
        self.presence = presence
        self.readers = sorted(readers_nights)
        self.arch = archetype_labels()
        self._build(readers_nights)


def wave_attendance(manifest):
    """{reader: [tags]} from a riverbed manifest, mirroring the filed
    FIELD_NIGHTS_W2 semantics: open-roster attendance + the staged entrant
    only on entrant_is_attendance nights."""
    att = {}
    for tag, meta in manifest["nights"].items():
        members = set(meta["roster"])
        if meta.get("staged_entrant") and meta.get("entrant_is_attendance"):
            members.add(meta["staged_entrant"])
        for r in members:
            att.setdefault(r, []).append(tag)
    return att


def wave_cold(manifest):
    """{reader: [tags]} of cold entries — the staged entrant enters cold on
    his attendance nights (mirrors the filed COLD_ENTRY_W2)."""
    cold = {}
    for tag, meta in manifest["nights"].items():
        e = meta.get("staged_entrant")
        if e and meta.get("entrant_is_attendance"):
            cold.setdefault(e, []).append(tag)
    return cold


def build_measurement(night_objects, attendance, cold=None, presence="actual"):
    """(RiverbedMeasurement, corpus_sd) — the corpus's OWN corpus_sd is the
    normalization (gate-target holdout rule: never a filed field number)."""
    sd, _ = corpus_sd(list(night_objects.values()))
    spec = {}
    for r, tags in attendance.items():
        spec[r] = {"nights": {t: r for t in tags if t in night_objects},
                   "cold": list((cold or {}).get(r, []))}
    return RiverbedMeasurement(spec, sd, night_objects,
                               presence=presence), sd


def load_wave(path):
    """Full bridge for a generated wave: manifest (file or corpus dir) ->
    {manifest, dir, nights, attendance, cold, measurement, sd}. The
    Measurement runs the registered estimators over the generated nights;
    premise_band_movers.night_windows etc. consume it unchanged."""
    if os.path.isdir(path):
        mpath = os.path.join(path, "riverbed-manifest.json")
    else:
        mpath = path
    base = os.path.dirname(os.path.abspath(mpath))
    man = json.load(open(mpath, encoding="utf-8"))
    nights = {}
    for tag, meta in man["nights"].items():
        nights[tag] = NightFromFile(tag, os.path.join(base, meta["file"]),
                                    family_strata(meta))
    attendance = wave_attendance(man)
    cold = wave_cold(man)
    m, sd = build_measurement(nights, attendance, cold)
    return {"manifest": man, "dir": base, "nights": nights,
            "attendance": attendance, "cold": cold, "measurement": m,
            "sd": sd}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="riverbed -> pipeline adapter")
    ap.add_argument("path", help="generated corpus dir or manifest json")
    args = ap.parse_args()
    w = load_wave(args.path)
    m, sd = w["measurement"], w["sd"]
    print(f"[adapter] {len(w['nights'])} nights, {len(m.readers)} readers, "
          f"corpus_sd={sd:.4f}")
    print(f"[adapter] drift={m.drift_mean():.4f} corpus-sd  "
          f"spread_seg={m.spread_seg():.4f}  ICC={m.icc()[0]:.4f}")
