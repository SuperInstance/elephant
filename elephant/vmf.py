"""von Mises–Fisher MLE — the honest (μ̂, κ) snapshot of a DialBank room.

This module implements the §1 math of the vMF engineering spec: window the
room over trailing messages, standardize the dials onto a comparable centered
cube, unit-normalize, and solve the vMF maximum-likelihood equations for the
mean direction μ̂ and concentration κ.

The v0 field already ships a `RoomField.concentration` — but that is a
center-mismatched extremity proxy (2·‖v − 0.5·𝟙‖), monotone in field magnitude
and therefore collinear with |warmth|. It stays for back-compat logging and is
**banned from comparison paths**; do not import it here.

Key design points (see the spec §1):

- **Standardization** — z_k = s_k·(v_k − c_k), s_k = 2/(hi_k − lo_k), with
  (lo, hi, c) mirroring `tapnight.DIAL_BOUNDS` / `DIAL_CENTER`. Without this,
  signed dials and [0,1] dials are incommensurable and the centered-at-0.5
  offset is double-counted.
- **Estimator** — κ solves A₇(κ) = ρ, A_d(κ) = I_{d/2}(κ)/I_{d/2−1}(κ), by
  Newton's method with a closed-form half-integer Bessel ratio (numpy-only).
  The Banerjee et al. formula `κ₀ = ρ(d−ρ²)/(1−ρ²)` is used as the *init* and
  the *bootstrap CI shortcut only*, never as the final estimate.
- **Warmth** — a fixed linear projection ŵ·μ̂ on the linearized warm direction.
  It reads μ̂ only; κ reads ρ only; ρ is rotation-invariant, so warmth cannot
  move κ *by construction*. (This is the disambiguation gate-condition 3 asks
  for.)
- **Guards** — κ = None under N < 10 windows (not identifiable, never a fake
  number); ρ clamped ≤ 0.999 with a clipped init (the unclipped init overflows
  sinh as ρ → 1 — see `vmf_fit`); κ ≤ 500 (dial saturation); bootstrap CI for
  κ; jackknife SE(μ̂) doubling as the drift deadband.

Honesty note for the future (spec §1.3): this estimator on 384-d encoder
embeddings at N ≈ 15 clips has E‖r̄‖ ≈ √(N/d) ≈ 0.20 *under uniformity* — raw
mean ρ is garbage there and needs small-sample shrinkage. In dial space
(d = 7, N ≥ 10) the bias is mild, but the warning must travel with the code.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .room import Room

# The 7-dim field space (DIAL_NAMES slice — the 9-key bank is NOT the field).
DIALS = ["mood", "volume", "earnestness", "cynicism",
         "joke_landing", "panic", "presence"]

# Ranges / centers mirror tapnight.DIAL_BOUNDS / DIAL_CENTER (kept local so
# vmf.py does not import tapnight — which imports vmf for the edge log — and
# so there is no cycle). tests/test_vmf.py asserts these stay in lockstep.
LO = np.array([-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0], float)
HI = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], float)
CENTER = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5], float)
SCALE = 2.0 / (HI - LO)  # z_k ∈ [−1, 1]

# The linearized warm direction in z-space (spec §1.4's fixed linear form),
# normalized so warmth_vMF = Ŵ · μ̂ is a signed cosine.
WARM = np.array([0.30, 0.10, 0.10, -0.15, 0.15, -0.10, 0.10], float)
WARM = WARM / np.linalg.norm(WARM)

D = 7            # dimension of the field space (S⁶)
KMAX = 500.0     # κ saturation cap — the v0 dials DO saturate
NMIN = 10        # below this many windows, κ is not identifiable
RHOMAX = 0.999   # ρ clamp — the unclipped init overflows sinh as ρ → 1


def zvec(readings: Dict[str, float]) -> np.ndarray:
    """Standardize a readings dict onto the comparable centered cube z ∈ [−1,1]."""
    return SCALE * (np.array([readings.get(n, 0.0) for n in DIALS]) - CENTER)


def windowed(room: Room, bank, W: int = 8, step: int = 1,
             cap: int = 64) -> List[np.ndarray]:
    """Trailing-window z-samples of the room *as it was* at each arrival.

    Each R_i = Room(messages[max(0, i−W+1) .. i]) is read by the bank and
    standardized; quiescent windows (‖z‖ < 1e-3) are skipped — nothing to
    normalize. Overlapping windows (step 1) are the trajectory; use
    non-overlapping (step = W) when bootstrap CIs must not be artificially
    narrow from window autocorrelation.
    """
    out: List[np.ndarray] = []
    msgs = room.messages[-cap:]
    for i in range(0, len(msgs), step):
        sub = Room(room.name, msgs[max(0, i - W + 1):i + 1])
        z = zvec(bank.readings(sub))
        if float(np.linalg.norm(z)) > 1e-3:
            out.append(z)
    return out


def A7(k: float) -> float:
    """A₇(κ) = I_{7/2}(κ) / I_{5/2}(κ) — closed-form half-integer Bessel ratio.

    numpy-only (the √(2/πκ) factors cancel). The closed form catastrophic-
    cancels for small κ (numerator and denominator are O(κ³)/O(κ⁴) against
    O(1) terms), so a series branch A₇ ≈ κ/7 handles κ < 0.5. Verified against
    scipy's ive(3.5)/ive(2.5) to < 1e-9 for κ ∈ [0.6, 500]; → κ/7 as κ→0 and
    → 1 − 3/κ as κ→∞.
    """
    if k < 0.5:
        return k / 7.0  # series branch (leading Taylor term)
    s, c = np.sinh(k), np.cosh(k)
    return (((1 + 15 / k ** 2) * c - (6 / k + 15 / k ** 3) * s)
            / ((1 + 3 / k ** 2) * s - (3 / k) * c))


def vmf_fit(zs, B: int = 200, seed: int = 0) -> Optional[dict]:
    """Joint (μ̂, κ) vMF MLE over a sample of z-vectors (rows = windows).

    Returns None when N < NMIN (κ not identifiable) or when the sample mean
    resultant vanishes (isotropic — no direction). Otherwise returns a dict
    with mu_hat[7], kappa, rho, n, kappa_ci, warmth_vmf, mu_se, axis_spread,
    and `saturated` (ρ hit RHOMAX or κ hit KMAX).
    """
    X = np.asarray(zs, float)
    N = len(X)
    if N < NMIN:
        return None  # κ is not identifiable — never a fake number

    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    r = X.mean(0)
    rho = min(float(np.linalg.norm(r)), RHOMAX)
    if rho < 1e-12:
        return None  # isotropic sample — no mean direction
    mu = r / rho

    # Banerjee et al. init, clipped — the unclipped value overflows sinh as
    # ρ → 1 (this is the sinh-overflow bug the spec's guard exists for).
    k = float(np.clip(rho * (D - rho ** 2) / (1 - rho ** 2), 1e-6, KMAX))

    # Newton solve on A₇(κ) = ρ. A′(κ) = 1 − A² − (d−1)·A/κ.
    for _ in range(60):
        a = A7(k)
        g = 1.0 - a * a - (D - 1.0) * a / k
        if abs(g) < 1e-12:
            break
        step = (a - rho) / g
        k = float(np.clip(k - step, 1e-6, KMAX))
        if abs(step) < 1e-9:
            break

    # Bootstrap CI on κ (Banerjee shortcut per sample — cheap, honest spread).
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(B):
        rb = X[rng.integers(0, N, N)].mean(0)
        rh = min(float(np.linalg.norm(rb)), RHOMAX)
        if rh < 1e-12:
            ks.append(0.0)
        else:
            ks.append(float(np.clip(rh * (D - rh ** 2) / (1 - rh ** 2),
                                    1e-6, KMAX)))

    # Jackknife SE(μ̂) — doubles as the drift deadband (gate 4).
    jk = np.stack([np.delete(X, i, 0).mean(0) for i in range(N)])
    jk /= np.linalg.norm(jk, axis=1, keepdims=True)
    mu_se = float(np.sqrt((N - 1) / N * ((jk - jk.mean(0)) ** 2).sum()))

    return {
        "mu_hat": mu.tolist(),
        "kappa": k,
        "rho": rho,
        "n": N,
        "kappa_ci": [float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))],
        "warmth_vmf": float(WARM @ mu),
        "mu_se": mu_se,
        "axis_spread": X.std(0).tolist(),
        "saturated": bool(rho >= RHOMAX or k >= KMAX),
    }


def edge(fb: Optional[dict], fa: Optional[dict], db_factor: float = 2.0) -> Optional[dict]:
    """Field drift between two fits, with the jackknife-SE deadband.

    `real` is True only when ‖Δμ̂‖ > db_factor · max(SE) — the drift deadband
    (gate 4). The edge-log schema records `real: null` until the dial-space
    noise floor is calibrated (measurement nights A–C), so this flag is a
    *derived* quantity, computed post-hoc, not baked into the append-only log.
    """
    if not fb or not fa:
        return None
    d_mu = float(np.linalg.norm(np.array(fa["mu_hat"]) - np.array(fb["mu_hat"])))
    return {
        "d_mu": d_mu,
        "d_warmth": fa["warmth_vmf"] - fb["warmth_vmf"],
        "d_log_kappa": float(np.log(fa["kappa"] / fb["kappa"])),
        "real": d_mu > db_factor * max(fb["mu_se"], fa["mu_se"]),
    }


def record_with(expected: Optional[dict], output: Optional[dict],
                cell: str = "room.field", ts: float = 0.0,
                db_factor: float = 2.0) -> Optional[dict]:
    """One cell-ledger entry — the elephant's producer half of the quilt seam.

    Wire contract (docs/quilt-bridge.md, quilt-rust/docs/cell-ledger.md): the
    ledger stores the before→after directed edge as
    ``{v, cell, ts, before, after, delta, imbalance, expected, provenance}``.
    ``expected`` is the forecast the outcome was scored against; under the
    default persistence prior (predict(b) = b) it IS ``before``, so
    ``imbalance == ‖expected − output‖ == d_mu`` by construction — identity 4
    of the bridge (on the unit sphere imbalance ≡ d_mu, the field-edge).

    Honesty gates coincide with the ledger's: with no prior, the entry books
    ``imbalance: null`` (the genesis entry — never fake a number); with no
    reading at all (``output`` None — e.g. N < NMIN), nothing is booked.
    """
    if output is None:
        return None  # no reading — nothing to book
    if expected is None:
        # Genesis entry: no prior, no surprise claimed (ledger §3).
        return {
            "v": 1, "cell": cell, "ts": float(ts),
            "before": None, "after": output["mu_hat"],
            "delta": None, "imbalance": None, "expected": None,
            "provenance": {"origin": "reading",
                            "producer": "elephant.vmf.record_with"},
        }
    e = edge(expected, output, db_factor=db_factor)
    if e is None:
        return None
    return {
        "v": 1, "cell": cell, "ts": float(ts),
        "before": expected["mu_hat"],
        "after": output["mu_hat"],
        "delta": [float(a) - float(b)
                  for a, b in zip(output["mu_hat"], expected["mu_hat"])],
        "imbalance": e["d_mu"],          # identity 4: imbalance ≡ d_mu
        "expected": expected["mu_hat"],  # persistence prior, sealed pre-outcome
        "d_warmth": e["d_warmth"],       # the signed leg the ledger discards
        "d_log_kappa": e["d_log_kappa"],
        "real": e["real"],
        "provenance": {"origin": "reading",
                        "producer": "elephant.vmf.record_with"},
    }
