"""Tests for the wave-3 S1 riverbed hardening (plan §3 gaps G1/G3/G4/G5/G13
+ self-test extensions G7/G9), scripts/riverbed_generator.py rework of
2026-08-21.

Registered behavior untouched: premise_band_movers, slope_regression_w2,
stage2_wave_gate, e2_instrument/e2_nights are consumed read-only. All
generation goes to pytest tmp dirs; data/ corpus files are never written.
"""

import json
import os
import shutil

import numpy as np
import pytest

from scripts.e2_instrument import W2_NIGHTS
from scripts.e2_nights import ATTENDANCE
from scripts.premise_band_movers import night_windows
from scripts.riverbed_adapter import (NightFromFile, build_measurement,
                                      family_strata, load_wave, wave_attendance,
                                      wave_cold)
from scripts.riverbed_generator import (BRANCHES, ENTRANT_NAME,
                                        ENTRANT_SPEAKS, NIGHT_FAMILIES,
                                        SEALED_FIELDS, generate_night,
                                        generate_wave, load_personas,
                                        persona_deviations, unblind)
from scripts.riverbed_wave_gate import run_gate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_T4A = os.path.join(ROOT, "data", "nights", "night-T4a.jsonl")


def _rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _speaks(path):
    return [r for r in _rows(path) if r["type"] == "speak"]


# --------------------------------------------------------------------------- #
# Shared generated corpora (module-scoped: generation is ~seconds each)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pair_waves(tmp_path_factory):
    """Two full waves, alpha 0 vs 1, sharing --pair-seed (2AFC pair)."""
    d0 = tmp_path_factory.mktemp("rb-pair-a0")
    d1 = tmp_path_factory.mktemp("rb-pair-a1")
    m0 = generate_wave(str(d0), alpha=0.0, seed=20260821, pair_seed=4242,
                       tag_prefix="pairA")
    m1 = generate_wave(str(d1), alpha=1.0, seed=20260821, pair_seed=4242,
                       tag_prefix="pairB")
    return (str(d0), m0), (str(d1), m1)


@pytest.fixture(scope="module")
def wave(tmp_path_factory):
    """A full instrument wave on the registered design (21 readers x 9)."""
    d = tmp_path_factory.mktemp("rb-wave")
    man = generate_wave(str(d), branch_name="instrument", seed=20260821)
    return str(d), man


@pytest.fixture(scope="module")
def blind_wave(tmp_path_factory):
    d = tmp_path_factory.mktemp("rb-blind")
    man = generate_wave(str(d), branch_name="collapse", blind=True,
                        corpus_id="PYTEST", seed=20260821)
    return str(d), man


# --------------------------------------------------------------------------- #
# G1 — mid-night entrants (field roster mechanics)
# --------------------------------------------------------------------------- #
class TestG1Entrants:
    def test_staged_entrant_semantics(self, wave):
        d, man = wave
        meta = next(m for m in man["nights"].values() if m["family"] == "T4a")
        rows = _rows(os.path.join(d, meta["file"]))
        opens = next(r for r in rows if r["type"] == "session_open")
        speaks = [r for r in rows if r["type"] == "speak"]
        e, eseq = meta["staged_entrant"], meta["entry_seq"]
        assert e == ENTRANT_NAME and eseq == 12
        assert e not in opens["roster"], "entrant never in the open roster"
        assert list(opens["staged_entries"]) == [e]
        ent = opens["staged_entries"][e]
        assert set(ent) == set(next(iter(opens["roster"].values())))
        in_readers = [e in r["readers"] for r in speaks]
        assert not any(in_readers[:eseq])
        assert all(in_readers[eseq:])
        assert next(r["seq"] for r in speaks if r["author"] == e) == eseq
        assert [q for q in ENTRANT_SPEAKS["T4a"] if q < len(speaks)] == \
            [r["seq"] for r in speaks if r["author"] == e]
        assert e not in speaks[eseq - 1]["entry_mode"]
        assert speaks[eseq]["entry_mode"][e] == "staged-cold"

    def test_non_staged_families_have_no_entrant(self, wave):
        d, man = wave
        for tag, meta in man["nights"].items():
            if meta["family"] in ("T4a", "T4b", "T5", "T5c"):
                assert meta["staged_entrant"] == ENTRANT_NAME
            else:
                assert meta["staged_entrant"] is None
                opens = next(r for r in _rows(os.path.join(d, meta["file"]))
                             if r["type"] == "session_open")
                assert "staged_entries" not in opens

    def test_nan_before_entry_convention_fires(self, wave):
        """The analysis NaN-before-entry path (night_windows full-window
        coverage) must fire exactly as on the field corpus."""
        d, man = wave
        w = load_wave(os.path.join(d, "riverbed-manifest.json"))
        tag = next(t for t, m in man["nights"].items() if m["family"] == "T4a")
        m = w["measurement"]
        win = night_windows(m, tag, w["sd"], 12)
        rho = win["rho"][ENTRANT_NAME]
        # windows whose span [t, t+11] reaches before the entry seq lack the
        # entrant (NaN); from the first full-post-entry window he is present
        entry = 12
        for t in range(len(rho)):
            if t + 11 < entry:
                assert np.isnan(rho[t]) or not win["present"][ENTRANT_NAME][t], t
        finite_from = next(t for t in range(len(rho))
                           if np.isfinite(rho[t]))
        assert finite_from >= entry, "entrant present in a pre-entry window"

    def test_readings_start_at_entry(self, wave):
        d, man = wave
        w = load_wave(os.path.join(d, "riverbed-manifest.json"))
        tag = next(t for t, m in man["nights"].items() if m["family"] == "T4a")
        reads = w["measurement"].readings[ENTRANT_NAME][tag]
        assert len(reads) == NIGHT_FAMILIES["T4a"][1] - 12


# --------------------------------------------------------------------------- #
# G9 — staged-night schema parity vs the filed corpus
# --------------------------------------------------------------------------- #
class TestG9StagedParity:
    def test_generated_staged_matches_field_t4a_keys(self, wave):
        d, man = wave
        meta = next(m for m in man["nights"].values() if m["family"] == "T4a")
        gen = _rows(os.path.join(d, meta["file"]))
        real = _rows(REAL_T4A)
        g_open = next(r for r in gen if r["type"] == "session_open")
        r_open = next(r for r in real if r["type"] == "session_open")
        assert set(g_open) == set(r_open), "staged open key sets differ"
        assert set(g_open["staged_entries"]) == set(r_open["staged_entries"])
        g_se = g_open["staged_entries"][ENTRANT_NAME]
        r_se = r_open["staged_entries"][ENTRANT_NAME]
        assert set(g_se) == set(r_se), "staged_entries shape differs"
        g_sp = [r for r in gen if r["type"] == "speak"]
        r_sp = [r for r in real if r["type"] == "speak"]
        assert all(set(a) == set(b) for a, b in zip(g_sp, r_sp))
        for a, b in zip(g_sp, r_sp):
            if ENTRANT_NAME in a["readers"]:
                assert set(a["readers"][ENTRANT_NAME]) == \
                    set(b["readers"][ENTRANT_NAME])
                break
        # staged == non-staged speak key sets (plan: verified on the field)
        meta1 = next(m for m in man["nights"].values() if m["family"] == "T1")
        g1 = _speaks(os.path.join(d, meta1["file"]))
        assert set(g1[15]) == set(g_sp[15])


# --------------------------------------------------------------------------- #
# G5 — adapter (generator output -> registered pipeline input)
# --------------------------------------------------------------------------- #
class TestG5Adapter:
    def test_load_wave_full_measurement(self, wave):
        d, man = wave
        w = load_wave(os.path.join(d, "riverbed-manifest.json"))
        m = w["measurement"]
        assert len(m.readers) == 21
        assert all(len(v) >= 3 for v in m.readings.values())
        icc, per_dial = m.icc()
        assert np.isfinite(icc)
        assert np.isfinite(m.drift_mean())
        assert np.isfinite(m.spread_seg())
        boot = m.bootstrap(B=200)
        assert boot["draws"] > 0, "registered bootstrap runs on generated data"

    def test_strata_verbatim_from_w2(self, wave):
        d, man = wave
        for tag, meta in man["nights"].items():
            strata = family_strata(meta)
            assert strata == W2_NIGHTS[meta["family"]][1], tag

    def test_attendance_mirrors_field_semantics(self, wave):
        d, man = wave
        att = wave_attendance(man)
        cold = wave_cold(man)
        # open-roster attendance for everyone...
        for fam, names in ATTENDANCE.items():
            tag = next(t for t, m2 in man["nights"].items()
                       if m2["family"] == fam)
            for n in names:
                assert tag in att[n], (n, tag)
        # ...plus the staged drifter on T4a/T4b ONLY (T5/T5c warmth-content)
        drifter_nights = sorted(att[ENTRANT_NAME])
        fams = sorted(man["nights"][t]["family"] for t in drifter_nights)
        assert fams == ["T2", "T4a", "T4b"]
        assert sorted(cold[ENTRANT_NAME]) and all(
            man["nights"][t]["family"] in ("T4a", "T4b")
            for t in cold[ENTRANT_NAME])

    def test_corpus_sd_is_own_number(self, wave):
        d, man = wave
        w = load_wave(os.path.join(d, "riverbed-manifest.json"))
        from scripts.e2_instrument import corpus_sd
        sd2, _ = corpus_sd(list(w["nights"].values()))
        assert abs(w["sd"] - sd2) < 1e-12
        assert w["sd"] > 0


# --------------------------------------------------------------------------- #
# G7 — realized ICC honesty (measured, not analytic)
# --------------------------------------------------------------------------- #
class TestG7RealizedICC:
    def test_instrument_icc_in_filed_band(self, wave):
        d, man = wave
        w = load_wave(os.path.join(d, "riverbed-manifest.json"))
        icc, _ = w["measurement"].icc()
        assert 0.85 <= icc <= 0.96, f"realized ICC {icc:.4f}"


# --------------------------------------------------------------------------- #
# G4 — parameterized riverbed wave gate
# --------------------------------------------------------------------------- #
class TestG4Gate:
    def test_gate_green_on_instrument(self, wave):
        d, man = wave
        results = run_gate(os.path.join(d, "riverbed-manifest.json"))
        assert results["all_pass"], [
            c for c in results["checks"] if not c["pass"]]
        # gate-target holdout: the corpus's OWN sd, never the field's 0.2367
        assert results["corpus_sd"] == pytest.approx(
            load_wave(os.path.join(d, "riverbed-manifest.json"))["sd"])
        # a-priori design x (deterministic): the design's exact Sxx
        assert results["sxx"] == pytest.approx(0.19707, abs=1e-4)
        assert results["sxx"] >= 0.19

    def test_gate_detects_tampering(self, wave, tmp_path):
        d, man = wave
        copy = tmp_path / "tampered"
        shutil.copytree(d, copy)
        meta = man["nights"][next(iter(man["nights"]))]
        fn = copy / meta["file"]
        rows = _rows(str(fn))
        rows[-1]["notes"] = "tampered"
        with open(fn, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        results = run_gate(str(copy / "riverbed-manifest.json"))
        assert not results["all_pass"]
        assert any("sha256" in c["check"] and not c["pass"]
                   for c in results["checks"])

    def test_gate_no_field_targets_leak(self):
        """The gate must never impose the field's corpus_sd/ladder on a
        generated corpus (gate-target holdout rule)."""
        import inspect
        src = inspect.getsource(run_gate)  # the gate logic itself
        assert "abs(sd -" not in src      # no filed-value comparison form
        assert "CORPUS_SD" not in src     # no filed corpus_sd constant
        assert "LADDER" not in src        # no filed-ladder table at all


# --------------------------------------------------------------------------- #
# G3 — blinding (redacted manifest + sealed sidecar)
# --------------------------------------------------------------------------- #
class TestG3Blinding:
    def test_manifest_redacted_and_opaque(self, blind_wave):
        d, man = blind_wave
        for k in SEALED_FIELDS:
            assert k not in man, f"redacted manifest leaks {k}"
        assert man["blinded"] and man["corpus_id"] == "PYTEST"
        for tag in man["nights"]:
            assert "PYTEST" in tag and "collapse" not in tag
            assert "alpha" not in tag
        # design facts stay (branch-free): schedule, rosters, entry info
        meta = next(iter(man["nights"].values()))
        for k in ("schedule", "roster", "sha256", "n_msgs", "family"):
            assert k in meta

    def test_unblind_round_trip(self, blind_wave):
        d, man = blind_wave
        sealed = unblind(os.path.join(d, man["sealed"]["file"]))
        assert sealed["branch"] == "collapse"
        assert sealed["alpha"] == 1.0
        assert sealed["seed"] == 20260821

    def test_tampered_seal_rejected(self, blind_wave, tmp_path):
        d, man = blind_wave
        spath = os.path.join(d, man["sealed"]["file"])
        sealed = json.load(open(spath))
        sealed["alpha"] = 0.0   # forge a different branch
        forged = tmp_path / "forged-sealed.json"
        forged.write_text(json.dumps(sealed))
        with pytest.raises(AssertionError):
            unblind(str(forged), manifest_path=os.path.join(
                d, "riverbed-manifest.json"))

    def test_tampered_night_rejected(self, blind_wave, tmp_path):
        d, man = blind_wave
        copy = tmp_path / "copy"
        shutil.copytree(d, copy)
        tag = next(iter(man["nights"]))
        fn = copy / man["nights"][tag]["file"]
        rows = _rows(str(fn))
        rows[10]["len"] = 999
        with open(fn, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with pytest.raises(AssertionError):
            unblind(str(copy / man["sealed"]["file"]))


# --------------------------------------------------------------------------- #
# G13 — pair-matching mode (branch-invariant room paths)
# --------------------------------------------------------------------------- #
class TestG13PairMode:
    def test_room_paths_identical_across_branches(self, pair_waves):
        (d0, m0), (d1, m1) = pair_waves
        for t0 in m0["nights"]:
            fam = m0["nights"][t0]["family"]
            t1 = next(t for t, m in m1["nights"].items()
                      if m["family"] == fam)
            s0 = _speaks(os.path.join(d0, m0["nights"][t0]["file"]))
            s1 = _speaks(os.path.join(d1, m1["nights"][t1]["file"]))
            assert len(s0) == len(s1)
            assert all(a["field_raw_after"] == b["field_raw_after"]
                       for a, b in zip(s0, s1)), f"room path differs ({fam})"
            assert all(a["author"] == b["author"] for a, b in zip(s0, s1)), \
                f"author schedule differs ({fam})"

    def test_fibers_diverge(self, pair_waves):
        (d0, m0), (d1, m1) = pair_waves
        t0 = next(iter(m0["nights"]))
        s0 = _speaks(os.path.join(d0, m0["nights"][t0]["file"]))
        t1 = next(t for t, m in m1["nights"].items()
                  if m["family"] == m0["nights"][t0]["family"])
        s1 = _speaks(os.path.join(d1, m1["nights"][t1]["file"]))
        shared = sorted(set(s0[15]["readers"]) & set(s1[15]["readers"]))
        r = shared[0]
        assert any(a["readers"][r] != b["readers"][r]
                   for a, b in zip(s0, s1)), "fibers did not diverge with alpha"

    def test_pair_waves_deterministic(self, pair_waves):
        for d, m in pair_waves:
            assert all(v.get("deterministic_replay_identical")
                       for v in m["nights"].values())

    def test_tag_keyed_default_differs(self, tmp_path):
        """Without --pair-seed the tag-keyed rng gives different room paths
        (the pre-G13 behavior, which corrupted 2AFC pairing)."""
        personas = load_personas()
        roster = ["writer", "poet", "engineer", "critic"]
        anchors = persona_deviations(roster, personas)
        fam, famname = NIGHT_FAMILIES["T1"], "T1"
        p0, _ = generate_night("np-a-T1", fam, roster, personas, anchors, {},
                               BRANCHES["instrument"], 7, str(tmp_path),
                               fam=famname)
        p1, _ = generate_night("np-b-T1", fam, roster, personas, anchors, {},
                               BRANCHES["instrument"], 7, str(tmp_path),
                               fam=famname)
        s0, s1 = _speaks(p0), _speaks(p1)
        assert any(a["field_raw_after"] != b["field_raw_after"]
                   for a, b in zip(s0, s1))

    def test_pair_mode_single_night_alignment(self, tmp_path):
        personas = load_personas()
        roster = ["writer", "poet", "engineer", "critic"]
        anchors = persona_deviations(roster + [ENTRANT_NAME], personas)
        fam, famname = NIGHT_FAMILIES["T4a"], "T4a"
        outs = {}
        for label, branch in (("i", BRANCHES["instrument"]),
                              ("c", BRANCHES["collapse"])):
            outs[label], _ = generate_night(
                f"pr-{label}-T4a", fam, roster, personas, anchors, {},
                branch, 7, str(tmp_path), pair_seed=99, fam=famname)
        si = _speaks(outs["i"])
        sc = _speaks(outs["c"])
        assert all(a["field_raw_after"] == b["field_raw_after"]
                   for a, b in zip(si, sc))
        # the staged entrant behaves identically in pair mode (entry @12)
        assert ENTRANT_NAME not in si[11]["readers"]
        assert ENTRANT_NAME in si[12]["readers"]


# --------------------------------------------------------------------------- #
# NightFromFile / build_measurement unit checks (G5 plumbing)
# --------------------------------------------------------------------------- #
class TestAdapterUnits:
    def test_derived_strata_for_custom_family(self):
        meta = {"family": "custom-x", "n_msgs": 30, "schedule":
                {"base_warmth": 0.6, "flip_seq": 10, "entry_seqs": []}}
        s = family_strata(meta)
        assert [x[0] for x in s] == ["s0", "s1"]
        assert (s[0][1], s[0][2]) == (0, 9) and (s[1][1], s[1][2]) == (10, 29)
        assert all(k == "signal" for _, _, _, k in s)
        meta2 = {"family": "custom-y", "n_msgs": 20, "schedule":
                 {"base_warmth": 0.6, "flip_seq": None, "entry_seqs": [5]}}
        s2 = family_strata(meta2)
        assert all(k == "null" for _, _, _, k in s2)   # no flip => control

    def test_build_measurement_explicit(self, wave):
        d, man = wave
        w = load_wave(os.path.join(d, "riverbed-manifest.json"))
        t1 = next(t for t in w["nights"] if man["nights"][t]["family"] == "T1")
        t3 = next(t for t in w["nights"] if man["nights"][t]["family"] == "T3")
        nights = {t1: w["nights"][t1], t3: w["nights"][t3]}
        att = {"singer": [t1, t3], "cartographer": [t1, t3]}  # real subsets
        m, sd = build_measurement(nights, att)
        assert m.readers == ["cartographer", "singer"]
        assert len(m.readings["singer"]) == 2
        assert np.isfinite(m.drift_mean())
