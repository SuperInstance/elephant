# FIELD-MATH-GPU — 2026-08-20

**The room-field thermometer as a CUDA batch farm.**
`scripts/field_math_gpu.py` · RTX 4050 Laptop (6,140 MB VRAM) · torch 2.13.0+cu130 · float64 parity mode

---

## Verdict

> **PASS — GPU batched field math matches the CPU serial reference (`elephant/vmf.py`) to 3.3e-12 max abs diff (tol 1e-6), invalid windows flagged identically, and runs the full-corpus fit (vMF MLE + 200-sample bootstrap CI + jackknife SE + edges + ledger) 111.6× faster: 43.23 s CPU-serial → 0.39 s GPU-batched over 15,058 windows. Core solver alone: 78.6× (2.72 s → 0.03 s). Peak VRAM 735 MB of 6,140 MB.**

| Path | Time | Throughput |
|---|---|---|
| CPU serial, full `vmf_fit` (Newton + bootstrap B=200 + jackknife) | 43.23 s | 348 fits/s |
| CPU serial, core solver only | 2.72 s | 5,537 fits/s |
| GPU batch, core solver only | 0.03 s | 435,288 fits/s |
| **GPU batch, full (upload + bootstrap + jackknife + download)** | **0.39 s** | **38,861 fits/s** |

**Speedup: full fit 111.6× · core 78.6×.**

## Corpus

21 independent z-sample streams: `roomd-field-log.jsonl` (2 rooms × 7,245 snapshots — the log is live and appending, so event counts grow between runs), 18 night logs (`data/nights/*.jsonl`, `field_raw_after` speak rows; `night-A-repro` excluded as a byte-replay of A), and `production-log.jsonl` (bar-rail probe). One log row = one bank-level reading (the `premise_measurement.py` reading of the logs), standardized by `elephant.vmf.zvec`, quiescent rows (‖z‖ < 1e-3) skipped exactly as `vmf.windowed` does. Trailing windows (cap 64, step 1, N ≥ NMIN=10) → **15,058 fit events**, N ∈ [10, 64], plus 15,037 consecutive-fit edges and 77 ledger before→after pairs.

## What was batched

1. **`vmf_fit_batch(X, N)`** — the entire `vmf_fit` for every window at once:
   row unit-normalization, masked mean, ρ (clamped at RHOMAX=0.999), μ̂ = r/ρ,
   Banerjee init `κ₀ = ρ(d−ρ²)/(1−ρ²)` clipped to [1e-6, 500] (vectorized),
   the 60-iteration Newton solve on A₇(κ) = ρ **as a fixed loop with
   per-element freeze masks** that replicate `vmf.py`'s exact break semantics
   (`|g| < 1e-12` → stop without update; `|step| < 1e-9` → apply update, then
   stop), the bootstrap CI, the jackknife SE(μ̂), axis spread, warmth
   (`WARM @ μ̂`, signed cosine), and the saturation flag.
2. **`edge_batch`** — field edges between consecutive fits of every stream:
   d_mu, d_warmth, d_log_kappa, and the jackknife-SE deadband (`real`),
   batched.
3. **`ledger_batch`** — the production-ledger imbalance on GPU: v0 warmth
   formula, κ proxy 2·‖v−0.5‖, drift = L2(d_warmth, d_kappa) and full-field
   L2 before→after, mirroring `examples/production_probe.py`.

### The bootstrap-parity trick

`vmf_fit` creates `np.random.default_rng(0)` **per call** and draws B=200
resample index rows — so every window of length n uses the *same* index
stream. The GPU path generates those exact streams on the host with numpy
(`default_rng(0)`, B×n draws, identical call sequence) and uploads them once
per distinct n. The GPU CI is therefore numerically the CPU CI (max diff
3.3e-12), not a lookalike from a different RNG.

## Correctness (full corpus, 15,058 windows vs CPU serial `vmf_fit`)

| Quantity | max abs diff |
|---|---|
| μ̂ (7-dim) | 9.99e-16 |
| κ | 7.25e-13 |
| ρ | 4.44e-16 |
| warmth_vmf (WARM·μ̂) | 5.55e-16 |
| κ CI (2.5/97.5 pct) | 3.30e-12 |
| SE(μ̂) jackknife | 8.47e-15 |
| axis_spread | 8.88e-16 |

Valid/invalid agreement: **15,058 / 15,058** (0 CPU-only, 0 GPU-only — the
isotropic/None gate fires identically). Saturated-flag agreement
15,058/15,058 (14,472 saturated — the live roomd log repeats identical
snapshots, so the near-sphere path is the *common* case, not a corner).
A₇ closed form vs `vmf.A7` over κ ∈ [1e-6, 600]: max diff 8.85e-15.

**Edges:** 15,037 consecutive-fit edges across 21 streams — max diff vs
`elephant.vmf.edge`: d_mu 1.4e-17, d_warmth 0.0, d_log_kappa 0.0, deadband
`real` disagreements: 0. Against **tapnight's own logged edges** (499
night-log edges written by the production consumer): d_mu 1.4e-17,
d_warmth 0.0, d_log_kappa 0.0.

**Ledger:** 77 before→after pairs; max |Δ d_warmth| 1.10e-4,
|Δ d_kappa| 1.02e-4, |Δ drift| 1.61e-4 — the ledger stores values rounded
to 4 dp *before* differencing (`production_probe.py`), so this is exact
parity up to the log's own rounding. Full-field L2 before→after: mean
0.359, max 1.382.

### Synthetic edge-case battery (CPU vs GPU, worst diff 1.2e-11)

| case | cpu | gpu | d_mu | d_κ | d_SE | d_CI | saturated |
|---|---|---|---|---|---|---|---|
| loose κ≈0.5 (series branch) | ✓ | ✓ | 1.1e-16 | 7.1e-15 | 5.6e-17 | 2.7e-15 | no |
| loose κ≈2 | ✓ | ✓ | 5.6e-17 | 0 | 0 | 1.8e-14 | no |
| κ≈20 cluster | ✓ | ✓ | 2.2e-16 | 0 | 3.5e-18 | 0 | **yes (κ=500)** |
| identical unit vectors | ✓ | ✓ | 2.2e-16 | 0 | 1.0e-16 | 0 | **yes (κ=500)** |
| near-RHOMAX tiny noise | ✓ | ✓ | 1.1e-16 | 0 | 1.7e-17 | 0 | **yes (κ=500)** |
| antipodal (isotropic) | None | None | — | — | — | — | — |
| exactly N=NMIN=10 | ✓ | ✓ | 0 | 0 | 2.1e-17 | 3.0e-12 | no |
| ρ just under clamp | ✓ | ✓ | 1.1e-16 | 0 | 1.0e-17 | 1.2e-11 | no |

## Edge cases & how they're handled

- **sinh overflow / κ divergence near the sphere (ρ → 1).** The unclipped
  Banerjee init `ρ(7−ρ²)/(1−ρ²)` → ~3,000 as ρ → 0.999 and would overflow
  `sinh` in A₇. Both CPU and GPU clip the init to [1e-6, KMAX=500] and clamp
  ρ at RHOMAX=0.999; saturated windows pin κ = 500 (Newton steps push into
  the clip and stay), flagged `saturated`. This is the *dominant* regime in
  the live roomd corpus (14,472/15,058 windows) and matches bit-for-bit.
- **A₇ catastrophic cancellation for κ < 0.5.** The closed form's numerator
  and denominator are O(κ³)/O(κ⁴) against O(1); below 0.5 both implementations
  use the series branch A₇ ≈ κ/7. On GPU the closed form is evaluated on a
  `k`-safe copy (never below 0.5) and selected with `torch.where`, so no
  inf/nan is ever produced even at κ = 1e-6.
- **Newton derivative vanishing (|g| < 1e-12).** Step suppressed per element
  (`g`-safe divisor), mirroring the CPU break; converged elements freeze via
  an `active` mask instead of `break` — including the apply-then-break
  semantics of the `|step| < 1e-9` exit.
- **Isotropic windows (ρ < 1e-12).** The batch analogue of `vmf_fit`
  returning `None`: masked to NaN, never a fake number. Flagged identically
  to CPU (verified by the antipodal battery case and 0/0 only-CPU/only-GPU
  counts over the corpus).
- **float64 parity caveats (documented, measured harmless).** libdevice vs
  libm `sinh/cosh` differ in the last ulp, and the jackknife's leave-one-out
  mean `(Σ − xᵢ)/(n−1)` reassociates vs `np.delete(...).mean(0)` — both land
  at ≤ 7e-13 on κ, ≤ 1e-14 on SE, five orders below tolerance. Everything
  runs in float64; consumer-GPU FP64 throughput is not a bottleneck at this
  workload (0.39 s end-to-end).
- **Memory.** Bootstrap gather is chunked to ≤ 2²⁵ elements (~256 MB);
  measured peak allocation 735 MB of 6,140 MB — comfortable headroom.

## Reproduce

```
python3 scripts/field_math_gpu.py            # full run (uses CPU cache when the
                                             # corpus digest is unchanged)
python3 scripts/field_math_gpu.py --refresh  # force CPU serial re-reference
python3 scripts/field_math_gpu.py --device cpu  # everything runs on torch-CPU too
```

Read-only against the corpus; writes only its own cache
(`scripts/field-math-gpu-cache.npz`). Note the roomd field log is append-only
and live — the event count grows between runs and the cache re-keys on the
corpus digest.

**One line:** the thermometer's math is unchanged — same estimator, same
resample streams, same guards — it just stopped waiting for Python: 43 s of
serial fitting per corpus sweep is now 0.39 s, and every number it prints
is the CPU's number to twelve decimal places.
