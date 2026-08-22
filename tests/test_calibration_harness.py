"""Tests for scripts/calibration_harness.py — the riverbed calibration
harness (summit gear #1, 2026-08-21).

Deterministic and fast: all corpora are built from the calibrated
parameter snapshot (the values calibrate() converged to on seed 20260822;
regenerating them takes ~60s so the tests pin the snapshot — the full
search is exercised by the harness itself and asserted fresh-seed in
assert_targets).
"""

import numpy as np
import pytest

from scripts import calibration_harness as ch
from scripts.e2_instrument import NIGHT_SPECS, W2_NIGHTS
from scripts.premise_band_movers import (counted_crossings, finite_series,
                                         leg_D, night_windows,
                                         strata_transitions)

# Calibrated snapshot (seed 20260822 family; see
# data/calibration/riverbed-calibration-results.json).
ROOM = {"m0": 1.2574, "kappa0": 300.0, "ar": 0.98}
FIBER = {"sigma_a": 0.2851, "sigma_j": 0.08, "phi": 0.55,
         "sigma_eta": 0.02, "nu": 0.06}
DX0 = 0.14


@pytest.fixture(scope="module")
def corpus():
    room = ch.build_room(DX0, ROOM, ch.SEED)
    return ch.build_corpus(room, FIBER, {}, ch.SEED + 1)


class TestVMFSampler:
    def test_concentration_pulls_to_mean_direction(self):
        rng = np.random.default_rng(7)
        mu = np.zeros(7)
        mu[0] = 1.0
        xs = ch.vmf_sample(rng, mu, 40.0, 3000)
        cos = xs @ mu
        assert float(np.mean(cos)) > 0.85

    def test_warmth_cos_matches_target_at_high_kappa(self):
        rng = np.random.default_rng(8)
        from elephant.vmf import WARM
        x = 0.6
        mu = x * WARM + np.sqrt(1 - x ** 2) * ch._orthogonal(rng)
        mu /= np.linalg.norm(mu)
        xs = ch.vmf_sample(rng, mu, 60.0, 4000)
        m = xs.mean(axis=0)
        m /= np.linalg.norm(m)
        assert abs(float(WARM @ m) - x) < 0.03

    def test_deterministic_given_seed(self):
        a = ch.vmf_sample(np.random.default_rng(3), np.eye(7)[0], 5.0, 10)
        b = ch.vmf_sample(np.random.default_rng(3), np.eye(7)[0], 5.0, 10)
        assert np.array_equal(a, b)


class TestCoordinateFirewall:
    def test_sandbox_names_disjoint_from_filed_specs(self):
        assert not (set(ch.K_NIGHTS)
                    & (set(NIGHT_SPECS) | set(W2_NIGHTS)))

    def test_data_nights_untouched(self, corpus, tmp_path):
        before = ch.snapshot_nights()
        # exercise the estimator against the sandbox corpus
        from scripts.e2_instrument import corpus_sd
        corpus_sd(list(corpus.nights.values()))
        assert ch.snapshot_nights() == before

    def test_seeds_disjoint_from_filed_families(self):
        for filed in (20260819, 20260820, 20260821):
            assert ch.SEED > filed
            assert ch.PERSONA_SEED > filed

    def test_firewall_guard_passes(self):
        fw = ch.firewall(ch.snapshot_nights())
        assert fw["data_nights_sha_stable"] and fw["names_disjoint"]


class TestWave2XSide:
    """The filed attendance matrix + warmth ladder reproduce the stage-2
    x-side design stats by construction (design constants, not data)."""

    def test_sxx_and_bands(self):
        xs = ch.x_side_stats()
        assert abs(xs["sxx"] - 0.1971) < 0.02
        for got, want in zip(xs["band_means"], (0.4793, 0.6384, 0.7094)):
            assert abs(got - want) < 0.02
        assert xs["n_distinct_x"] == 15
        assert abs(xs["x_range"] - 0.2535) < 0.01

    def test_attendance_shape_mirrors_wave2(self):
        assert len(ch.ATTENDANCE) == 21
        assert sum(len(v) for v in ch.ATTENDANCE.values()) \
            == sum(len(v) for v in __import__(
                "scripts.e2_instrument", fromlist=["FIELD_NIGHTS_W2"]
            ).FIELD_NIGHTS_W2.values())
        null_rn = sum("K9" in v for v in ch.ATTENDANCE.values())
        assert null_rn == 7      # the wave-2 null reader-nights


class TestSimulatorDeterminismAndPairs:
    def test_same_seed_identical_corpus(self):
        room = ch.build_room(0.10, ROOM, 123)
        a = ch.build_corpus(room, FIBER, {}, 124)
        b = ch.build_corpus(room, FIBER, {}, 124)
        for r in a.readings:
            for n in a.readings[r]:
                va = np.array([v for _, v in a.readings[r][n]])
                vb = np.array([v for _, v in b.readings[r][n]])
                assert np.array_equal(va, vb)

    def test_adversarial_pair_shares_room_and_personas(self):
        room = ch.build_room(DX0, ROOM, 55)
        off = ch.build_corpus(room, FIBER, {"slope_lambda": 0.0}, 56)
        on = ch.build_corpus(room, FIBER, {"slope_lambda": 1.0}, 56)
        # identical persona vibes (a-priori params) ...
        for n in off.nights:
            for r in off.nights[n].params:
                assert np.allclose(off.nights[n].params[r]["vibe_start"],
                                   on.nights[n].params[r]["vibe_start"])
        # ... identical room field ... (same room object)
        # ... but different readings (the one registered quantity moved)
        va = np.array([v for _, v in off.readings["poet"]["K1"]])
        vb = np.array([v for _, v in on.readings["poet"]["K1"]])
        assert np.abs(va - vb).max() > 0.02

    def test_differential_pair_differs_only_from_boundaries(self):
        room = ch.build_room(DX0, ROOM, 77)
        rigid = ch.build_corpus(room, FIBER, {"diff_zeta": 0.0}, 78)
        scram = ch.build_corpus(room, FIBER, {"diff_zeta": 0.8}, 78)
        # before the first flip boundary (K1@20): identical readings
        pre_r = np.array([v for s, v in rigid.readings["poet"]["K1"]
                          if s < 20])
        pre_s = np.array([v for s, v in scram.readings["poet"]["K1"]
                          if s < 20])
        assert np.allclose(pre_r, pre_s)
        post_r = np.array([v for s, v in rigid.readings["poet"]["K1"]
                           if s >= 20])
        post_s = np.array([v for s, v in scram.readings["poet"]["K1"]
                           if s >= 20])
        assert np.abs(post_r - post_s).max() > 0.01


class TestQRigidity:
    """THE q-rule test: a rigid common shift must NOT register as
    persistence; a differential step must."""

    def test_rigid_step_is_uninformative(self):
        room = ch.build_room(DX0, ROOM, ch.SEED)
        m = ch.build_corpus(room, FIBER, {}, ch.SEED + 1)
        from scripts.e2_instrument import corpus_sd
        q = ch.q_rule(m, corpus_sd(list(m.nights.values()))[0])
        assert q["verdict"] == "uninformative"
        assert q["q_trans"] < 2.0 * q["q_rest"] + 0.02

    def test_differential_step_is_violated(self):
        room = ch.build_room(DX0, ROOM, ch.SEED)
        m = ch.build_corpus(room, FIBER, {"diff_zeta": 1.2}, ch.SEED + 1)
        from scripts.e2_instrument import corpus_sd
        q = ch.q_rule(m, corpus_sd(list(m.nights.values()))[0])
        assert q["verdict"] == "persistence_violated"


class TestEstimatorIntegration:
    """The filed E2/E3 estimator runs unmodified on sandbox output."""

    def test_night_windows_produces_finite_rho(self, corpus):
        from scripts.e2_instrument import corpus_sd
        sd = corpus_sd(list(corpus.nights.values()))[0]
        w = night_windows(corpus, "K1", sd, 12)
        vals = np.concatenate([finite_series(w["rho"][r])[0]
                               for r in w["readers"]])
        assert len(vals) > 100
        assert np.isfinite(vals).all()

    def test_leg_D_runs_and_mirrors_field_coverage_shape(self, corpus):
        from scripts.e2_instrument import corpus_sd
        sd = corpus_sd(list(corpus.nights.values()))[0]
        win = {n: night_windows(corpus, n, sd, 12) for n in ch.K_NIGHTS}
        d = leg_D(win, corpus, ch.K_SIGNAL, ch.K_NULL)
        assert d["n_transitions"] == 10          # the wave-2 count
        assert 0.0 <= d["D_signal"] <= 1.0
        assert d["null_rn_crossing_rate"] is not None

    def test_dose_response_of_crossings(self):
        m0, sd0 = ch._probe(ROOM, FIBER, 0.0, ch.SEED + 3)
        mhi, sdhi = ch._probe(ROOM, FIBER, 0.28, ch.SEED + 3)
        assert ch.n_down_events(m0, sd0) <= 3    # null: no step-less crossings
        assert ch.n_down_events(mhi, sdhi) >= 12  # hard flips make band-movers

    def test_calibrated_field_stats(self, corpus):
        """The snapshot parameters land on the wave-2 targets."""
        from scripts.e2_instrument import corpus_sd
        sd = corpus_sd(list(corpus.nights.values()))[0]
        spread, drift = ch.sandbox_spread_drift(corpus, sd)
        icc = ch.sandbox_icc(corpus)
        q = ch.q_rule(corpus, sd)
        assert abs(sd - 0.2367) < 0.02
        assert abs(spread - 0.4883) < 0.08
        assert 0.667 - 0.05 <= icc <= 0.810 + 0.05
        assert 0.07 <= q["q_trans"] <= 0.20
        assert ch.n_down_events(corpus, sd) >= 8


# --------------------------------------------------------------------------- #
# q_rule — red-team common-shift guard (wave-4 S1 unit-test obligation)
# --------------------------------------------------------------------------- #
class TestQRule:
    def _fake_m(self, differential):
        """One 60-speak night, two signal strata cut at 24, three readers
        with stable per-reader offsets; the transition applies a COMMON
        rigid step plus (if differential) per-reader idiosyncratic steps —
        the exact adversarial pair the q-rule exists to separate."""
        import types
        readers = ["r0", "r1", "r2"]
        base = np.zeros(7)
        offs = {r: 0.30 * np.eye(7)[i + 1] for i, r in enumerate(readers)}
        dsteps = {r: 0.50 * np.eye(7)[i + 4] for i, r in enumerate(readers)}
        common = 0.40 * np.eye(7)[0]
        n = 60
        cut = 24   # both strata >= 2W=24 so rest events exist
        speaks = [{"seq": t} for t in range(n)]
        night = types.SimpleNamespace(
            speaks=speaks,
            strata=[("s0", 0, cut - 1, "signal"),
                    ("s1", cut, n - 1, "signal")])
        m = types.SimpleNamespace(
            readers=readers, nights={"N": night}, readings={})
        for r in readers:
            seqd = {}
            for t in range(n):
                v = base + offs[r]
                if t >= cut:
                    v = v + common + (dsteps[r] if differential else 0.0)
                seqd[t] = v
            m.readings[r] = {"N": seqd}
        return m

    def test_common_rigid_step_is_uninformative(self):
        res = ch.q_rule(self._fake_m(differential=False), sd=1.0,
                        signal_nights=["N"])
        assert res["verdict"] == "uninformative"
        assert res["n_trans"] >= 1 and res["n_rest"] >= 1

    def test_differential_step_flags_persistence_violated(self):
        res = ch.q_rule(self._fake_m(differential=True), sd=1.0,
                        signal_nights=["N"])
        assert res["q_trans"] > 2.0 * res["q_rest"] + 0.02
        assert res["verdict"] == "persistence_violated"

    def test_q_rule_on_calibration_corpus_is_wellformed(self, corpus):
        from scripts.e2_instrument import corpus_sd
        m, sd = corpus, corpus_sd(list(corpus.nights.values()))[0]
        res = ch.q_rule(m, sd)
        assert set(res) >= {"q_trans", "q_rest", "n_trans", "n_rest",
                            "verdict"}
        assert res["verdict"] in ("uninformative", "persistence_violated")
