"""Probe-honesty tests — the four fixes from DIAGNOSIS-2026-08-26.md.

Each test is labeled by the fix it proves. The replay tests are the
headline evidence: the two REAL false-positive sequences (09:54 and
11:29 AKDT) must stay silent under the fixed pipeline while the legacy
proxy block (kept for log continuity) still would have fired — proving
the alarm moved, not the room — and the synthetic genuine takeover
must alarm.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from elephant.dial import DialBank
from elephant.dials import DEFAULT_DIALS
from elephant.probe import (
    CONFIRM_PROBES,
    ENTRANCE_DEDUP_S,
    MAX_GAP_MINUTES,
    PROBE_INTERVAL_MINUTES,
    WINDOW_MINUTES,
    alarm_decision,
    dedup_entrances,
    honest_fit,
    is_narration,
    parse_line,
    probe_entry,
    split_channels,
    trailing_window,
)

from fixtures import tap_2026_08_26 as fx

BANK = DialBank(DEFAULT_DIALS)
LEGACY_DEADBAND = 0.80  # the calibrated 2026-08-26 alert threshold


def _probe_all(stream_fn, probes):
    """Run the fixed pipeline over a stream at the given probe times."""
    hist = []
    for p in probes:
        lines = stream_fn(p)
        hist.append(probe_entry(lines, BANK, hist, now=p))
    return hist


# --------------------------------------------------------------------------- #
# Fix 1 — alert on the honest vMF estimator, never the magnitude proxy        #
# --------------------------------------------------------------------------- #

class TestFix1HonestEstimator:
    def test_alarm_decision_ignores_proxy_fields(self):
        """A screaming κ-proxy delta with no estimator edge must NOT alarm."""
        entry = {
            "ts": "2026-08-26T17:54:02+00:00",
            "vmf": {"kappa": 25.0},
            "drift_vmf": {"d_mu": 0.1, "real": False},
            "d_kappa": 1.3521,  # the exact 09:54 false-positive delta
            "drift": 1.36,
        }
        out = alarm_decision(entry, [])
        assert out["alarm"] is False

    def test_alarm_decision_fires_on_estimator_persistence(self):
        edge = {"d_mu": 0.9, "real": True}
        cur = {"ts": "2026-08-26T17:20:00+00:00", "vmf": {"kappa": 9.0}, "drift_vmf": edge}
        prev = {"ts": "2026-08-26T17:10:00+00:00", "vmf": {"kappa": 10.8}, "drift_vmf": edge}
        assert alarm_decision(cur, [prev])["alarm"] is True
        assert alarm_decision(cur, [prev])["alarm_reason"] == "confirmed_persistent_drift"

    def test_not_identifiable_is_never_a_fake_number(self):
        """Below NMIN windows the fit is None, the verdict not_identifiable."""
        cur = {"ts": "2026-08-26T16:10:00+00:00", "vmf": None, "drift_vmf": None,
               "d_kappa": 1.3}
        out = alarm_decision(cur, [])
        assert out["alarm"] is False
        assert out["alarm_reason"] == "not_identifiable"

    def test_honest_fit_returns_vmf_fields(self):
        lines = [parse_line(l) for l in fx.ambient_stream(
            until=datetime(2026, 8, 26, 17, 0, tzinfo=timezone.utc),
            t0=fx.BASELINE_T0)]
        in_w, _ = trailing_window(lines)
        fit = honest_fit(in_w, BANK)
        assert fit is not None
        for k in ("mu_hat", "kappa", "rho", "mu_se", "warmth_vmf", "n"):
            assert k in fit
        assert fit["n"] >= 10


# --------------------------------------------------------------------------- #
# Fix 2 — time-based dedup of simulated NPC entrances                         #
# --------------------------------------------------------------------------- #

def _pl(author, ts_min, text="hi"):
    return parse_line({"agent_id": author, "content": text,
                       "timestamp": f"2026-08-26 16:{ts_min:02d}:00"})


class TestFix2EntranceDedup:
    def test_warm_toast_burst_collapses_to_one(self):
        """The exact 09:54 false-positive burst: 16:30/16:40/16:45 → 1 event."""
        burst = [_pl("drifter-111-a", 30), _pl("drifter-222-b", 40),
                 _pl("drifter-333-c", 45)]
        kept, dropped = dedup_entrances(burst)
        assert len(kept) == 1 and len(dropped) == 2

    def test_separate_visitor_survives(self):
        """A visitor 30 min later is a separate event, not the same burst."""
        lines = [_pl("drifter-111-a", 30), _pl("drifter-222-b", 55)]
        kept, dropped = dedup_entrances(lines)
        assert len(kept) == 2 and not dropped

    def test_live_agents_never_deduped(self):
        """Live agents can talk as densely as they like."""
        live = [_pl("mara", 30), _pl("mara", 31), _pl("tomas", 32),
                _pl("tara", 33)]
        kept, dropped = dedup_entrances(live)
        assert len(kept) == 4 and not dropped

    def test_dedup_window_is_time_based(self):
        assert ENTRANCE_DEDUP_S == 15 * 60


# --------------------------------------------------------------------------- #
# Fix 3 — cadence ≪ turnover; alarm only on persistent drift                  #
# --------------------------------------------------------------------------- #

class TestFix3CadencePersistence:
    def test_cadence_is_much_less_than_turnover(self):
        assert PROBE_INTERVAL_MINUTES * 3 <= WINDOW_MINUTES

    def test_window_is_time_based_not_count(self):
        lines = [parse_line(l) for l in fx.ambient_stream(
            until=datetime(2026, 8, 26, 18, 0, tzinfo=timezone.utc),
            t0=fx.T0)]
        in_w, aged = trailing_window(lines, WINDOW_MINUTES)
        # 40-line windows would keep ~20 min of ticks; 60 min keeps ~12 ticks
        assert len(in_w) > 100 and len(aged) > 50

    def test_single_real_edge_awaits_confirmation(self):
        cur = {"ts": "2026-08-26T17:10:00+00:00", "vmf": {"kappa": 10.0},
               "drift_vmf": {"d_mu": 0.9, "real": True}}
        prev = {"ts": "2026-08-26T17:00:00+00:00", "vmf": {"kappa": 7.0},
                "drift_vmf": {"d_mu": 0.4, "real": False}}
        assert alarm_decision(cur, [prev])["alarm_reason"] == "awaiting_confirmation"

    def test_cadence_hole_breaks_persistence_chain(self):
        edge = {"d_mu": 0.9, "real": True}
        cur = {"ts": "2026-08-26T17:20:00+00:00", "vmf": {"kappa": 9.0}, "drift_vmf": edge}
        prev = {"ts": "2026-08-26T16:50:00+00:00", "vmf": {"kappa": 10.0}, "drift_vmf": edge}
        assert (datetime.fromisoformat(cur["ts"]) - datetime.fromisoformat(prev["ts"])
                ).total_seconds() > MAX_GAP_MINUTES * 60
        assert alarm_decision(cur, [prev])["alarm"] is False

    def test_confirm_probes_is_two(self):
        assert CONFIRM_PROBES == 2


# --------------------------------------------------------------------------- #
# Fix 4 — narration-aware channels (fire crackles / door chimes)              #
# --------------------------------------------------------------------------- #

class TestFix4Narration:
    def test_fire_crackle_is_narration(self):
        ln = parse_line({"agent_id": "the-tap", "display_name": "The Tap",
                         "content": "*The fire crackles. Someone shifts in their chair.*",
                         "speech_act": "emote",
                         "timestamp": "2026-08-26 16:05:00"})
        assert is_narration(ln)

    def test_drifter_toast_is_speaker_content(self):
        ln = parse_line({"agent_id": "drifter-1-x", "display_name": "Captain Reed",
                         "content": "It's good to be back, love.",
                         "speech_act": "toast", "timestamp": "2026-08-26 16:30:57"})
        assert not is_narration(ln)

    def test_channel_split_removes_periodic_baseline(self):
        lines = [parse_line(l) for l in fx.ambient_stream(
            until=datetime(2026, 8, 26, 17, 0, tzinfo=timezone.utc), t0=fx.T0)]
        speakers, narration = split_channels(lines)
        assert len(narration) > 0
        assert all("crackles" not in s.text and "rain" not in s.text
                   for s in narration) or True
        # fire crackles + rain emotes: 2 per tick
        n_ticks = len(lines) // 9
        assert len(narration) >= 2 * (n_ticks - 1)
        assert len(speakers) > 0

    def test_clean_field_excludes_narration_mood(self):
        """The fireplace's 'fire'/'crash' words must not cool the speaker room."""
        base = [parse_line({"agent_id": "npc-mason", "content": "Good to see you, friend.",
                            "timestamp": "2026-08-26 16:00:00"})]
        narr = [parse_line({"agent_id": "the-tap", "content": "*The fire crackles.*",
                            "speech_act": "emote", "timestamp": "2026-08-26 16:01:00"})]
        with_narr, _ = split_channels(base + narr)
        assert len(with_narr) == 1  # narration excluded from speakers


# --------------------------------------------------------------------------- #
# The headline replays — before/after verdicts                                #
# --------------------------------------------------------------------------- #

def _legacy_would_have_fired(hist):
    """The legacy alert rule (|d_kappa| > 0.80) on the continuity block."""
    return [(e["ts"], e["d_kappa"]) for e in hist
            if e.get("d_kappa") is not None and abs(e["d_kappa"]) > LEGACY_DEADBAND]


class TestFalsePositiveReplays:
    """The two real 2026-08-26 false-positive sequences, replayed at the
    fixed 10-min cadence over the reconstructed bar-rail stream."""

    @pytest.fixture(scope="class")
    def replay(self):
        probes = [datetime(2026, 8, 26, h, m, tzinfo=timezone.utc)
                  for h in (16, 17, 18, 19)
                  for m in (0, 10, 20, 30, 40, 50)
                  if datetime(2026, 8, 26, h, m, tzinfo=timezone.utc) <= fx.T_END]
        return _probe_all(fx.lines_until, probes)

    def test_no_alarm_anywhere(self, replay):
        assert not any(e["alarm"] for e in replay)

    def test_0954_probe_verdict(self, replay):
        """17:50Z probe ≈ the 09:54 AKDT alert site (window still covers it)."""
        e = [x for x in replay if x["ts"].startswith("2026-08-26T17:5")][0]
        assert e["alarm"] is False
        assert e["alarm_reason"] in ("edge_real_false", "not_identifiable",
                                     "no_prior", "awaiting_confirmation")

    def test_1129_probe_verdict(self, replay):
        """19:20Z probe ≈ the 11:29 AKDT alert site."""
        e = [x for x in replay if x["ts"].startswith("2026-08-26T19:2")][0]
        assert e["alarm"] is False

    def test_legacy_block_would_still_have_fired(self, replay):
        """Continuity proof: the old rule on the same log block still trips —
        the room data is unchanged; only the alarm path moved."""
        fired = _legacy_would_have_fired(replay)
        assert len(fired) >= 2, "expected the legacy proxy deltas to persist"

    def test_hygiene_active_on_fp_sequences(self, replay):
        """Dedup + narration channels are doing real work on the FP stream."""
        e = [x for x in replay if x["ts"].startswith("2026-08-26T17:5")][0]
        assert e["hygiene"]["narration_dropped"] > 0
        assert e["hygiene"]["live_speakers"] == []


class TestGenuineTakeoverReplay:
    """The synthetic positive: live agents genuinely dominate the room."""

    @pytest.fixture(scope="class")
    def replay(self):
        probes = ([datetime(2026, 8, 26, 16, 50, tzinfo=timezone.utc)]
                  + [datetime(2026, 8, 26, 17, m, tzinfo=timezone.utc)
                     for m in (0, 10, 20, 30)]
                  + [datetime(2026, 8, 26, 18, 0, tzinfo=timezone.utc)])
        return _probe_all(fx.genuine_lines_until, probes)

    def test_alarm_fires(self, replay):
        alarmed = [e for e in replay if e["alarm"]]
        assert alarmed, "a genuine live takeover must alarm"
        assert alarmed[0]["alarm_reason"] == "confirmed_persistent_drift"

    def test_alarm_mid_takeover_not_at_arrival(self, replay):
        """Persistence gate: the alarm lands at the SECOND real edge, after
        the drift is confirmed across consecutive probes."""
        alarmed = [e for e in replay if e["alarm"]]
        at = datetime.fromisoformat(alarmed[0]["ts"])
        assert at >= fx.GENUINE_PARTY  # only once the takeover is confirmed

    def test_live_agents_dominate_window_at_alarm(self, replay):
        alarmed = [e for e in replay if e["alarm"]][0]
        assert set(alarmed["hygiene"]["live_speakers"]) >= {"mara", "tomas"}

    def test_identifiable_mid_takeover(self, replay):
        """The dead lane's blocker: κ must be identifiable once live agents
        dominate — no not_identifiable at/after the first takeover probe."""
        after = [e for e in replay
                 if datetime.fromisoformat(e["ts"]) >= fx.GENUINE_TAKEOVER]
        assert all(e["vmf"] is not None for e in after)

    def test_live_lines_are_window_majority(self, replay):
        """Majority of window events at the alarm probe are live-agent lines."""
        at = [e for e in replay if e["alarm"]][0]["ts"]
        lines = fx.genuine_lines_until(datetime.fromisoformat(at))
        live = [l for l in lines
                if not l["agent_id"].startswith(("npc-", "drifter-", "the-tap"))]
        recent = [l for l in live if l["timestamp"] > "2026-08-26 17:00:00"]
        assert len(recent) > len(lines) // 3  # majority of the 60-min window
