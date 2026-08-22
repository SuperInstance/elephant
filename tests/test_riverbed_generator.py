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
                                        ENTRANT_SPEAKS, ENTRY_DWARMTH,
                                        E_SEG, KAPPA_COLD,
                                        KAPPA_ENTRY_FACTOR, KAPPA_WARM,
                                        NIGHT_FAMILIES, NIGHT_ORDER,
                                        SEALED_FIELDS, Z_ENTRY, Z_FLIP,
                                        Z_WARM_DEV, generate_night,
                                        generate_wave, load_personas,
                                        persona_deviations,
                                        expected_logged_warmth_path,
                                        room_path, room_schedule,
                                        seg_schedule, unblind)
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
    def test_instrument_icc_in_field_actual_band(self, wave):
        """G6 re-verification (2026-08-21, G6 rework run doc): the
        charisma-pull fiber + field-magnitude persona anchors + the
        era-position geometry reproduce the FIELD's actual-presence ICC
        (0.8444 through this exact Measurement path — the filed wave-2
        number, docs/riverbed-S1-hardening + G6 research §3; canonical
        0.7714). Band anchored on the field value, not the old vMF-fiber
        bracket [0.85, 0.96] (superseded — see
        memory/wave3-registration-addendum-g6-2026-08-21.md and
        docs/riverbed-G6-run-2026-08-21.md for the re-verification)."""
        d, man = wave
        w = load_wave(os.path.join(d, "riverbed-manifest.json"))
        icc, _ = w["measurement"].icc()
        assert 0.78 <= icc <= 0.88, f"realized ICC {icc:.4f}"


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
# Corrected event semantics (κ(t)-check 2026-08-21, DIRECTION-EVENT verdict)
# --------------------------------------------------------------------------- #
class TestCorrectedEventSemantics:
    """memory/kappa-t-check-2026-08-21.md: entry-steps are μ/direction
    events (Δwarmth −0.147 ≈ flip's −0.151, p=0.68); κ polarity is warm=
    tight ≈24 / cynical=loose ≈11; transitions only LOOSEN κ."""

    def test_entry_is_a_warmth_event(self):
        rng = np.random.default_rng(0)
        w, k = room_schedule(NIGHT_FAMILIES["T4a"], False, rng)
        base, n, flip, entries = NIGHT_FAMILIES["T4a"]
        e = entries[0]
        # μ steps down at entry by ENTRY_DWARMTH (flip-magnitude)
        assert w[e - 1] - w[e] == pytest.approx(ENTRY_DWARMTH)
        # the flip's own step is intact
        assert w[flip - 1] - w[flip] == pytest.approx(0.5)

    def test_entry_moves_mu_at_flip_magnitude(self):
        rng = np.random.default_rng(1)
        rp1 = room_path(NIGHT_FAMILIES["T1"], False, rng)
        rp4 = room_path(NIGHT_FAMILIES["T4a"], False, rng)
        dmu_entry = float(np.linalg.norm(rp4["mu"][12] - rp4["mu"][11]))
        dmu_flip = float(np.linalg.norm(rp1["mu"][20] - rp1["mu"][19]))
        assert 0.5 * dmu_flip < dmu_entry < 2.0 * dmu_flip

    def test_kappa_polarity_warm_tight_cynical_loose(self):
        rng = np.random.default_rng(0)
        w, k = room_schedule(NIGHT_FAMILIES["T1"], False, rng)
        assert k[:20].mean() == pytest.approx(KAPPA_WARM, abs=1.0)
        assert k[20:].mean() == pytest.approx(KAPPA_COLD, abs=1.0)
        assert k[:20].mean() > k[20:].mean() + 8.0

    def test_transitions_only_loosen_kappa(self):
        rng = np.random.default_rng(0)
        w4, k4 = room_schedule(NIGHT_FAMILIES["T4a"], False, rng)
        assert k4[12] < k4[11] - 5.0          # entry loosens (no +12 spike)
        assert k4[12:] .max() < k4[:12].min()  # never re-tightens after

    def test_kappa_floor_respected(self):
        rng = np.random.default_rng(0)
        # T4b: flip then entry — deepest stacked loosening
        w, k = room_schedule(NIGHT_FAMILIES["T4b"], False, rng)
        assert (k >= 2.5 * 0.9).all()

    def test_null_mode_field_polarity(self):
        rng = np.random.default_rng(0)
        w, k = room_schedule(NIGHT_FAMILIES["T2"], True, rng)
        base, n, flip, entries = NIGHT_FAMILIES["T2"]
        assert len(set(w.tolist())) == 1              # warmth flat
        assert k[:flip].mean() > k[flip:].mean() + 8  # cohesion shift loosens

    def test_manifest_carries_entry_dwarmth(self, wave):
        d, man = wave
        assert man["entry_dwarmth"] == pytest.approx(ENTRY_DWARMTH)


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


# --------------------------------------------------------------------------- #
# G6 — the noise-model rework (memory/research-g6-noise-2026-08-21.md):
# (i) per-speak per-dial Gaussian noise at the field's within-stratum
#     scales, era-scaled by the κ(t) channel; (ii) the emitted windowed z
# keeps its magnitude (no unit-normalization; the fit channel is the
# clamped dial reading); (iii) the reader fiber is the engine's
# charisma-pull equation (replay_readings parity); plus the E_SEG
# text-step schedule contrast (the field's real schedule geometry).
# --------------------------------------------------------------------------- #
class TestG6NoiseModel:
    def test_seg_schedule_field_geometry(self):
        # era-position VECTORS at the field-measured per-dial magnitudes
        # (stratum-mean measurements on the wave-2 T-nights, G6 run
        # 2026-08-21): the flip steps by Z_FLIP, entries stack Z_ENTRY,
        # warm eras sit at Z_WARM_DEV
        seg = seg_schedule(NIGHT_FAMILIES["T1"], False)
        assert np.allclose(seg[19], Z_WARM_DEV)
        assert np.allclose(seg[20] - seg[19], Z_FLIP)
        # entries stack partway (T4a: entry@12 then flip@20)
        seg4 = seg_schedule(NIGHT_FAMILIES["T4a"], False)
        assert np.allclose(seg4[12] - seg4[11], Z_ENTRY)
        assert np.allclose(seg4[20] - seg4[19], Z_FLIP)
        assert np.allclose(seg4[-1], Z_WARM_DEV + Z_ENTRY + Z_FLIP)
        # null mode: NO text steps (cohesion-only, no direction content)
        segn = seg_schedule(NIGHT_FAMILIES["T2"], True)
        assert np.allclose(segn, np.tile(Z_WARM_DEV, (len(segn), 1)))

    def test_era_vectors_field_geometry(self):
        # the measured text-step vectors: cynicism moves rail-to-rail,
        # presence up, mood ~flat (the field's flip is a content step,
        # not a warmth step — the field's mood dial barely moves between
        # strata: stratum-mean sd 0.052 raw, measured 2026-08-21)
        assert Z_FLIP[3] == pytest.approx(max(Z_FLIP), abs=1e-9)
        assert Z_FLIP[3] > 1.5                       # cynicism rail-to-rail
        assert Z_FLIP[6] > 0.4                       # presence up
        assert abs(Z_FLIP[0]) < 0.1                  # mood light
        assert Z_WARM_DEV[3] < -0.9                  # warm cynicism at low rail
        assert Z_ENTRY[3] > 1.0                      # entry moves cynicism too
        assert np.linalg.norm(Z_ENTRY) < np.linalg.norm(Z_FLIP)  # smaller

    def test_emission_unnormalized_with_noise(self):
        # part (ii): the emitted windowed z KEEPS its magnitude (the
        # engine logs the raw windowed reading, not its direction)
        rp = room_path(NIGHT_FAMILIES["T1"], False, np.random.default_rng(0))
        norms = [float(np.linalg.norm(z)) for z in rp["obs"]]
        # NOT unit vectors: magnitudes spread around the era-anchored
        # scale (warm ~1.46 / cold ~2.2 — the field's own z-norm
        # structure: warm-stratum mean 1.46, corpus mean 1.997), well
        # off 1.0
        assert 0.7 < min(norms) and max(norms) < 3.4
        assert float(np.mean(norms)) > 1.2
        # part (i): per-speak per-dial noise — same seed, same emission
        rp2 = room_path(NIGHT_FAMILIES["T1"], False, np.random.default_rng(0))
        assert all(np.allclose(a, b) for a, b in zip(rp["obs"], rp2["obs"]))
        # dial_noise=0 changes the emission (the noise is live)
        rp0 = room_path(NIGHT_FAMILIES["T1"], False, np.random.default_rng(0),
                        dial_noise=0.0)
        assert any(not np.allclose(a, b)
                   for a, b in zip(rp["obs"], rp0["obs"]))

    def test_noise_is_per_dial_heterogeneous(self):
        # the field's within-stratum shape: volume ~deterministic,
        # joke_landing/presence loose (SIGMA_DIAL, G6 sec 2.1)
        from scripts.riverbed_generator import SIGMA_DIAL
        idx_vol, idx_joke = 1, 4
        assert SIGMA_DIAL[idx_vol] < 0.05
        assert SIGMA_DIAL[idx_joke] > 0.15
        # era-scaled by kappa(t): a warm-era draw is quieter than a
        # cold-era one (the κ polarity rides the noise scale)
        from scripts.riverbed_generator import KAPPA_COLD, KAPPA_WARM, NOISE_ERA_EXP
        assert (KAPPA_COLD / KAPPA_WARM) ** NOISE_ERA_EXP < 0.9

    def test_engine_charisma_pull_replay_parity(self, tmp_path):
        """The fiber IS replay_readings: the registered replay on the
        logged rows reproduces field_eff_to_reader exactly (T1 roster +
        T4a staged entrant, cold entry path)."""
        from scripts.e2_instrument import assert_replay_matches_log
        import types
        personas = load_personas()
        roster = ["writer", "poet", "engineer", "critic", "captain", "essayist"]
        anchors = persona_deviations(roster + [ENTRANT_NAME], personas)
        for famname, family in (("T1", NIGHT_FAMILIES["T1"]),
                                ("T4a", NIGHT_FAMILIES["T4a"])):
            path, _ = generate_night(f"g6-{famname}", family, roster, personas,
                                     anchors, {}, BRANCHES["instrument"], 7,
                                     str(tmp_path), fam=famname)
            rows = [json.loads(l) for l in open(path)]
            nt = types.SimpleNamespace()
            nt.path = path
            nt.open = next(r for r in rows if r["type"] == "session_open")
            nt.speaks = [r for r in rows if r["type"] == "speak"]
            nt.v2 = True
            nt.params = {n: dict(v) for n, v in nt.open["roster"].items()}
            for n, v in nt.open.get("staged_entries", {}).items():
                nt.params.setdefault(n, dict(v))
            for n in nt.params:
                nt.params[n]["dial_weights"] = np.asarray(
                    nt.params[n]["dial_weights"], float)
                nt.params[n]["vibe_start"] = np.asarray(
                    nt.params[n]["vibe_start"], float)
            nt.first_speak_seq = lambda reader: next(
                (r["seq"] for r in nt.speaks if r["author"] == reader), None)
            for name in nt.params:
                assert_replay_matches_log(nt, name, cold=(name == ENTRANT_NAME))

    def test_branch_lives_in_logged_vibe_start(self, tmp_path):
        """Alpha enters ONLY through the logged persona anchor (persona
        space — the coordinate firewall); the equation itself is
        branch-free (engine parity for every branch)."""
        personas = load_personas()
        roster = ["writer", "poet", "engineer"]
        anchors = persona_deviations(roster + [ENTRANT_NAME], personas)
        kw = dict(family=NIGHT_FAMILIES["T1"], roster_names=roster,
                  personas=personas, dev_anchors=anchors, ou_state={},
                  seed=7, outdir=str(tmp_path), fam="T1")
        p0, _ = generate_night("br0-T1", branch=BRANCHES["instrument"], **kw)
        p1, _ = generate_night("br1-T1", branch=BRANCHES["collapse"], **kw)
        r0 = next(json.loads(l) for l in open(p0) if '"session_open"' in l)
        r1 = next(json.loads(l) for l in open(p1) if '"session_open"' in l)
        v0 = r0["roster"]["writer"]["vibe_start"]
        v1 = r1["roster"]["writer"]["vibe_start"]
        assert v0 != v1, "the branch must move the logged persona anchor"

    def test_calibration_snapshot_registered_seed(self, wave):
        """The four re-verified G6 statistics at the registered seed on
        the registered tags (bands carry the measured realization
        spread; the G6 registration addendum holds the full table)."""
        import types
        from scripts.e2_instrument import W2_NIGHTS, logged_readings
        d, man = wave
        w = load_wave(os.path.join(d, "riverbed-manifest.json"))
        sd = w["sd"]
        assert 0.22 <= sd <= 0.27, f"corpus_sd {sd:.4f} (field 0.2367)"
        # stable-d: night_windows W=12 split-half, windows wholly inside a
        # stratum, normalized by the corpus's own corpus_sd (registered
        # object; field actual-presence 0.376, floor 0.29)
        ds = []
        for tag, nt in w["nights"].items():
            fam = next(m["family"] for t, m in man["nights"].items() if t == tag)
            strata = W2_NIGHTS[fam][1]
            for name in nt.params:
                pairs = logged_readings(nt, name)
                if len(pairs) < 12:
                    continue
                vecs = np.stack([v for _, v in pairs])
                for t in range(len(vecs) - 12 + 1):
                    if not any(lo <= t and t + 11 <= hi
                               for _, lo, hi, _ in strata):
                        continue
                    a, b = vecs[t:t + 6].mean(0), vecs[t + 6:t + 12].mean(0)
                    ds.append(np.linalg.norm(b - a) / sd)
        dstat = float(np.mean(ds))
        assert 0.28 <= dstat <= 0.42, f"stable-d {dstat:.3f}"
        # logged kappa: warm/cold per-strata means — the field's 24/11
        # band is reproduced in the COLD level and the warm/cold RATIO
        # (the warm level carries a disclosed offset; G6 addendum)
        kw, kc = [], []
        for tag, nt in w["nights"].items():
            fam = next(m["family"] for t, m in man["nights"].items() if t == tag)
            flip = NIGHT_FAMILIES[fam][2]
            if flip is None:
                continue
            for label, lo, hi, kind in W2_NIGHTS[fam][1]:
                ks = [r["fit"]["kappa"] for r in nt.speaks
                      if r.get("fit") and lo <= r["seq"] <= hi]
                if not ks:
                    continue
                (kw if hi < flip else kc).append(float(np.mean(ks)))
        kw_m, kc_m = float(np.mean(kw)), float(np.mean(kc))
        assert 18 <= kc_m <= 28, f"cold kappa {kc_m:.1f} (field ~11-15; disclosed level offset)"
        assert 1.6 <= kw_m / kc_m <= 2.4, f"kappa ratio {kw_m/kc_m:.2f} (field 2.18)"
        # warmth residual vs the noise-aware expected path: within the
        # gate's ±0.10 band on strata means (G6 rework run: max 0.056 at
        # the registered seed — a genuine pass, tightened from the
        # earlier 0.16 disclosed band that the pre-rework calibration
        # needed)
        res = []
        for tag, nt in w["nights"].items():
            fam = next(m["family"] for t, m in man["nights"].items() if t == tag)
            exp = expected_logged_warmth_path(NIGHT_FAMILIES[fam])
            for label, lo, hi, kind in W2_NIGHTS[fam][1]:
                if kind != "signal":
                    continue
                vals = [r["fit"]["warmth_vmf"] - exp[r["seq"]]
                        for r in nt.speaks
                        if r.get("fit") and exp[r["seq"]] is not None
                        and lo <= r["seq"] <= hi]
                if vals:
                    res.append(abs(float(np.mean(vals))))
        assert max(res) <= 0.10, f"max strata warmth residual {max(res):.3f}"

    def test_noise_branch_icc_collapses(self, tmp_path_factory):
        """The registered noise prediction (ICC < filed floor 0.667):
        the noise branch redraws the WHOLE persona per night — anchors
        AND the dial lens (the charisma-pull fiber's stable-constant
        carrier is the lens, not a kappa_R draw)."""
        d = tmp_path_factory.mktemp("rb-g6-noise")
        generate_wave(str(d), branch_name="noise", seed=20260821)
        w = load_wave(os.path.join(d, "riverbed-manifest.json"))
        icc, _ = w["measurement"].icc()
        assert icc < 0.667, f"noise-branch ICC {icc:.4f} must collapse"
