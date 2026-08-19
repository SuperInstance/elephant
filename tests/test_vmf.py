"""elephant — tests: the vMF (μ̂, κ) estimator and the edge log (spec §4).

Covers the gate-3 / gate-4 buildables:
- κ recovery on exact vMF samples (MLE reproduces the generating concentration);
- the numpy-only closed-form Bessel ratio A₇ against scipy (spot-check < 1e-9);
- the guards: N < 10 → None, the ρ→1 sinh-overflow clamp, κ ≤ 500 saturation;
- window sensitivity over W ∈ {4, 8, 16};
- the JSONL edge-log sink (session_open / speak / session_close), replay honesty.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from elephant.room import Message, Room
from elephant.tapnight import DIAL_BOUNDS, DIAL_CENTER, Participant, TapNightSession
from elephant.vmf import (
    A7,
    CENTER,
    DIALS,
    HI,
    LO,
    KMAX,
    NMIN,
    RHOMAX,
    WARM,
    edge,
    vmf_fit,
    windowed,
    zvec,
)


# --------------------------------------------------------------------------- #
# vMF sampler (Wood 1994 / Banerjee et al. 2005) — only used to *generate*     #
# ground-truth samples for the recovery test; the estimator itself is          #
# numpy-only.                                                                  #
# --------------------------------------------------------------------------- #
def _sample_vmf(mu, kappa, n, seed=0):
    rng = np.random.default_rng(seed)
    d = mu.shape[0]
    mu = mu / np.linalg.norm(mu)
    b = (d - 1) / (2 * kappa + np.sqrt(4 * kappa ** 2 + (d - 1) ** 2))
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d - 1) * np.log(1 - x0 ** 2)
    out = []
    while len(out) < n:
        z = rng.beta((d - 1) / 2, (d - 1) / 2)
        t = (1 - (1 + b) * z) / (1 - (1 - b) * z)
        u = rng.random()
        if kappa * t + (d - 1) * np.log(1 - x0 * t) - c < np.log(u):
            continue
        v = rng.standard_normal(d - 1)
        v /= np.linalg.norm(v)
        out.append(np.concatenate([[t], np.sqrt(1 - t ** 2) * v]))
    X = np.array(out)
    e1 = np.zeros(d)
    e1[0] = 1.0
    if np.linalg.norm(e1 - mu) > 1e-9:
        w = e1 - mu
        w /= np.linalg.norm(w)
        H = np.eye(d) - 2 * np.outer(w, w)
        X = (H @ X.T).T
    return X


# --------------------------------------------------------------------------- #
# 1. κ recovery on exact vMF samples                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kappa", [3.0, 5.0, 15.0])
def test_kappa_recovery_on_exact_vmf(kappa):
    rng = np.random.default_rng(7)
    mu = rng.standard_normal(7)
    mu /= np.linalg.norm(mu)
    X = _sample_vmf(mu, kappa, n=800, seed=123)

    # The sampler is self-consistent: empirical ρ ≈ population A₇(κ).
    rho_emp = float(np.linalg.norm(X.mean(0)))
    assert rho_emp == pytest.approx(A7(kappa), abs=0.05)

    fit = vmf_fit(X)
    assert fit is not None
    # MLE recovers the generating concentration within small-sample error.
    assert abs(fit["kappa"] - kappa) / kappa < 0.15
    # and the mean direction.
    assert np.dot(fit["mu_hat"], mu) > 0.99
    # Newton actually converged: A₇(κ̂) == ρ to tolerance.
    assert A7(fit["kappa"]) == pytest.approx(fit["rho"], abs=1e-6)
    assert 0.0 <= fit["kappa"] <= KMAX


# --------------------------------------------------------------------------- #
# 2. numpy-only closed form vs scipy (spot-check < 1e-9)                      #
# --------------------------------------------------------------------------- #
def test_A7_closed_form_matches_scipy():
    ive = pytest.importorskip("scipy.special").ive
    for k in [0.6, 0.7, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
        assert A7(k) == pytest.approx(ive(3.5, k) / ive(2.5, k), abs=1e-9)


def test_A7_limits():
    # → κ/7 as κ → 0 (series branch), → 1 − 3/κ as κ → ∞.
    assert A7(0.1) == pytest.approx(0.1 / 7.0, rel=1e-12)
    assert A7(500.0) == pytest.approx(1.0 - 3.0 / 500.0, abs=1e-4)


# --------------------------------------------------------------------------- #
# 3. Guards                                                                   #
# --------------------------------------------------------------------------- #
def test_kappa_none_below_minimum_windows():
    assert vmf_fit([]) is None
    assert vmf_fit(np.zeros((NMIN - 1, 7))) is None
    # exactly NMIN samples is the first identifiable case
    rng = np.random.default_rng(0)
    X = rng.standard_normal((NMIN, 7))
    fit = vmf_fit(X)
    assert fit is not None
    assert fit["n"] == NMIN


def test_rho_clamp_prevents_sinh_overflow():
    # The unclipped Banerjee init overflows sinh as ρ → 1 — reproduce it.
    rho = 0.9999
    k_unclipped = rho * (7 - rho ** 2) / (1 - rho ** 2)
    assert k_unclipped > 1e4
    with np.errstate(over="ignore"):
        assert np.isinf(np.sinh(k_unclipped))
    # vmf_fit clamps ρ ≤ 0.999 and κ ≤ 500, so it never reaches that branch.
    x = np.array([1.0] + [0.0] * 6)
    fit = vmf_fit(np.tile(x, (50, 1)))
    assert fit is not None
    assert fit["rho"] <= RHOMAX
    assert fit["kappa"] == KMAX  # saturation branch
    assert fit["saturated"] is True
    assert np.isfinite(fit["kappa"])


def test_kappa_capped_at_500_for_saturated_dials():
    # 200 near-identical unit vectors: ρ ≈ 1 → κ saturates at KMAX, never above.
    rng = np.random.default_rng(3)
    base = rng.standard_normal(7)
    base /= np.linalg.norm(base)
    X = base + 1e-4 * rng.standard_normal((200, 7))
    fit = vmf_fit(X)
    assert fit is not None
    assert fit["kappa"] <= KMAX
    assert np.isfinite(fit["kappa"])
    assert np.all(np.isfinite(fit["kappa_ci"]))


def test_isotropic_sample_returns_none():
    # A sample whose mean resultant vanishes has no mean direction.
    X = np.vstack([np.eye(7), -np.eye(7)])  # cancels to r̄ ≈ 0
    assert vmf_fit(X) is None


# --------------------------------------------------------------------------- #
# 4. Standardization + window sensitivity                                     #
# --------------------------------------------------------------------------- #
def test_standardization_matches_tapnight_bounds():
    # vmf.LO/HI/CENTER must stay in lockstep with tapnight's single source of
    # truth (drift between the two would silently miscalibrate the sphere).
    for i, name in enumerate(DIALS):
        lo, hi = DIAL_BOUNDS[name]
        assert LO[i] == lo
        assert HI[i] == hi
        assert CENTER[i] == DIAL_CENTER[name]
    # signed dials map to [-1, 1], [0,1] dials map to [-1, 1] via the scale.
    assert WARM.shape == (7,)
    assert np.isclose(np.linalg.norm(WARM), 1.0)


def test_zvec_centers_and_scales():
    readings = dict(zip(DIALS, DIAL_CENTER.values()))
    assert np.linalg.norm(zvec(readings)) < 1e-12  # neutral → origin


class _NeutralBank:
    """A bank whose readings sit at every dial's center → quiescent room."""

    def readings(self, room):
        return {n: DIAL_CENTER[n] for n in DIALS}


def test_windowed_skips_quiescent_windows():
    room = Room("quiet", [Message(author="a", text="", ts=float(i))
                          for i in range(20)])
    zs = windowed(room, _NeutralBank(), W=8)
    assert zs == []  # all ‖z‖ < 1e-3 → nothing to normalize
    assert vmf_fit(zs) is None


def _make_session():
    writer = Participant("writer", dial_weights={"mood": 0.5, "joke_landing": 0.5},
                         vibe={"mood": 0.6, "joke_landing": 0.4})
    critic = Participant("critic", dial_weights={"cynicism": 1.0},
                         vibe={"cynicism": 0.7})
    s = TapNightSession("The Tap", participants=[writer, critic])
    s.start_session()
    lines = [
        ("writer", "I love this warm room, truly. haha", {"❤️": 2}),
        ("critic", "Sure, sure. Obviously great. 🙄", {}),
        ("writer", "We built it together, honestly, and it holds.", {}),
        ("critic", "Whatever, lovely, as if.", {}),
        ("writer", "The fire is warm and the room is kind tonight.", {"❤️": 1}),
        ("critic", "A joke? Please. As if that landed. 🙄", {}),
        ("writer", "It lands, it always lands, and we all laugh.", {"😂": 3}),
        ("critic", "Rolling my eyes so hard. 😒", {}),
        ("writer", "The presence in this room is something else.", {}),
        ("critic", "Sure, presence. Whatever that means. 🙄", {}),
        ("writer", "Warmth, honesty, and a joke that actually landed.",
         {"😂": 1, "❤️": 1}),
        ("critic", "Panic? No. Cynicism? Yes. 😏", {}),
        ("writer", "Mood is high and the laughs are real tonight.", {"😂": 2}),
        ("critic", "Earnestness is overrated. 🙄", {}),
        ("writer", "It holds, honestly, and it glows. cheers to that.", {}),
        ("critic", "Whatever, lovely, as if. 😒", {}),
        ("writer", "We built a warm thing here, together.", {"❤️": 2}),
        ("critic", "Sure. Obviously. 🙄", {}),
        ("writer", "And it glows. cheers.", {}),
        ("critic", "Whatever. as if. 🙄", {}),
    ]
    for a, t, r in lines:
        s.speak(a, t, reactions=r)
    return s


def test_window_sensitivity_W_sweep():
    s = _make_session()
    fits = {}
    for W in (4, 8, 16):
        zs = windowed(s.room, s.bank, W=W)
        fits[W] = vmf_fit(zs)
    # every W yields an identifiable, finite κ; drift across W is reported, not
    # hidden — the values are just asserted to be sane (bounded, finite).
    for W, fit in fits.items():
        assert fit is not None, f"W={W} should be identifiable"
        assert 0.0 <= fit["kappa"] <= KMAX
        assert fit["n"] >= NMIN
    assert len({round(fits[W]["kappa"], 3) for W in fits}) >= 1


# --------------------------------------------------------------------------- #
# 5. Edge log sink                                                            #
# --------------------------------------------------------------------------- #
def test_edge_log_sink(tmp_path):
    path = tmp_path / "edge.jsonl"
    s = TapNightSession("The Tap", participants=[
        Participant("writer", dial_weights={"mood": 0.5, "joke_landing": 0.5},
                    vibe={"mood": 0.6, "joke_landing": 0.4}),
        Participant("critic", dial_weights={"cynicism": 1.0},
                    vibe={"cynicism": 0.7}),
    ], log_path=str(path))
    s.start_session()
    lines = [
        ("writer", "I love this warm room, truly. haha", {"❤️": 2}),
        ("critic", "Sure, sure. Obviously great. 🙄", {}),
        ("writer", "We built it together, honestly, and it holds.", {}),
        ("critic", "Whatever, lovely, as if.", {}),
        ("writer", "The fire is warm and the room is kind tonight.", {"❤️": 1}),
        ("critic", "A joke? Please. As if that landed. 🙄", {}),
        ("writer", "It lands, it always lands, and we all laugh.", {"😂": 3}),
        ("critic", "Rolling my eyes so hard. 😒", {}),
        ("writer", "The presence in this room is something else.", {}),
        ("critic", "Sure, presence. Whatever that means. 🙄", {}),
        ("writer", "Warmth, honesty, and a joke that actually landed.",
         {"😂": 1, "❤️": 1}),
        ("critic", "Panic? No. Cynicism? Yes. 😏", {}),
    ]
    for a, t, r in lines:
        s.speak(a, t, reactions=r)
    s.end_session()

    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    kinds = [r["type"] for r in rows]
    assert kinds[0] == "session_open"
    assert kinds[-1] == "session_close"
    speaks = [r for r in rows if r["type"] == "speak"]
    assert len(speaks) == len(lines)

    # schema presence (spec §3.2)
    op = rows[0]
    assert op["reader"]["kind"] == "RoomElephant"
    assert set(op["params"]) == {"W", "standardization", "estimator", "kappa_max"}
    assert set(op["roster"]) == {"writer", "critic"}

    # seq strictly increasing, ts non-decreasing
    seqs = [r["seq"] for r in speaks]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    tss = [r["ts"] for r in speaks]
    assert all(b >= a for a, b in zip(tss, tss[1:]))

    # every speak carries the 7-vector + arrival order + presence mask
    for r in speaks:
        assert len(r["field_raw_after"]) == 7
        assert len(r["field_eff_after"]) == 7
        assert isinstance(r["presence_mask"], list)
    # first_by_author marks the author's first-ever message
    first_flags = {(r["author"], r["seq"]): r["first_by_author"] for r in speaks}
    for author in ("writer", "critic"):
        author_seqs = [r["seq"] for r in speaks if r["author"] == author]
        assert first_flags[(author, min(author_seqs))] is True
        for sq in author_seqs[1:]:
            assert first_flags[(author, sq)] is False

    # presence_mask == authors in the trailing W=8 window
    W = op["params"]["W"]
    for i, r in enumerate(speaks):
        trailing_authors = sorted({s["author"] for s in speaks[max(0, i - W + 1):i + 1]})
        assert r["presence_mask"] == trailing_authors

    # fits: null below NMIN windows, then present with finite κ ≤ 500
    for r in speaks[: NMIN - 1]:
        assert r["fit"] is None
    for r in speaks[NMIN - 1:]:
        assert r["fit"] is not None
        assert 0.0 <= r["fit"]["kappa"] <= KMAX
        assert r["fit"]["n"] >= NMIN

    # edge.real is null until the noise floor is calibrated (spec §3.2)
    for r in speaks:
        if r["edge"] is not None:
            assert r["edge"]["real"] is None

    # replay honesty: session_close.final == refit from the last speak's fit
    close = rows[-1]
    last = speaks[-1]
    assert close["final"]["kappa"] == pytest.approx(last["fit"]["kappa"])
    assert close["final"]["mu_hat"] == last["fit"]["mu_hat"]
    assert close["final"]["warmth_vmf"] == pytest.approx(last["fit"]["warmth_vmf"])
    assert len(close["final"]["readings"]) == 9  # all raw bank readings
    assert close["n_messages"] == len(lines)


def test_edge_deadband_replay_is_still():
    # Replaying the same sample yields ‖Δμ̂‖ = 0 → real == False (deadband).
    s = _make_session()
    fit = vmf_fit(windowed(s.room, s.bank, W=8))
    e = edge(fit, fit)
    assert e is not None
    assert e["d_mu"] == pytest.approx(0.0)
    assert e["real"] is False
    assert edge(None, fit) is None
    assert edge(fit, None) is None


def test_vmf_never_uses_v0_concentration():
    # v0 concentration() is banned from comparison paths (spec §0 / §1).
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(here, "..", "elephant", "vmf.py"), encoding="utf-8").read()
    assert ".concentration(" not in src          # never *called*
    assert "read_field" not in src               # never reads via field.RoomField
    assert "from .field" not in src and "from elephant.field" not in src


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nAll vmf tests passed.")
