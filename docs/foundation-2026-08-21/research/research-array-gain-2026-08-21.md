# Research: Array Gain — What the Hundred Boats Actually Buy

**Filed: 2026-08-21** · Read-only research (this doc is the sole write). Commissioned by the Captain's sonar doctrine: *"if all the sounders of the ~100 boats on this tack were connected, the echograms could have enough data in real time to animate the school like a 3D pixelated video."* Inputs: `foundation-synthesis-2026-08-21.md` (skew product), `math-foundation-geometric-2026-08-21.md` (the eigenproblem, fiber geometry), `research-impact-2026-08-21.md` (applications, blocker inventory), `wave3-registration-2026-08-21.md` (frozen H-GEN design), `elephant/REG1-RUN-2026-08-21.md` (v* = volume–presence, branch B), `quilt-synergy-map-2026-08-21.md` (the fleet = the boats). Every number below is arithmetic on filed quantities — no corpus was read, no run executed.

---

## 0. The doctrine, restated in the fleet's own math

In sonar, **array gain** is the SNR improvement from coherently combining R sensors: +10·log₁₀(R) dB for perfectly coherent signal and independent noise, eroded by inter-sensor noise correlation and by sensors that can't see the target at all. The fleet's translation is exact and was already latent in the foundation:

- **The school = the room base orbit** μ(t) — the cynicism-steered trajectory on the low-dim base, one track every reader rides (skew product, geometric doc §3.1).
- **Each boat's hull fog = the personality fiber** o_r — a fixed per-reader offset, 95.1% of spread in a 3-dim subspace, a near-constant of the room flow (pooled cos 0.993, P_trans 0.994).
- **The water column = κ(t)** — warm-tight (κ≈24) / cold-loose (κ≈11) heterogeneity; the shared choppiness every sounder sees through.
- **The echogram = the registered statistic** — (μ̂, κ) snapshots, edges with deadband, the A/D/P/S legs.

The question the doctrine forces: when R readers each observe μ(t) + o_r + ε_r(t), what is the school's effective SNR? Answer: **it depends entirely on which channel you beamform** — and the fleet has two, with opposite gain laws.

---

## 1. The array equation

### 1.1 The observation model (from the filed geometry)

Per-reader windowed reading (z-space, direction u in dial space):

  **x_r(t) = μ(t) + o_r + ε_r(t),  ε_r(t) = c(t) + η_r(t)**

- μ(t): shared base orbit (the signal);
- o_r: fixed fiber point (personality; cancels in within-reader differences, dominates levels);
- c(t): **common-mode** room-heterogeneity fluctuation — the noise every reader shares, i.e. κ(t)'s wobble; E[η_r·η_s] = 0 for r ≠ s, c ⊥ η.

### 1.2 Two channels, two gain laws

**(i) Step/timing channel** (edges, A-leg timing, D coverage — everything computed from within-reader differences, where o_r cancels exactly):

  Δx_r = Δμ + Δc + Δη_r  →  roster-mean step variance = 2(σ²_c + σ²_η/R)

  **SNR_step(R) = ‖Δμ‖ / √(2(σ²_c + σ²_η/R))**, i.e.

  **G(R) = √( qR / (1 + (qR − 1)·ρ_κ) )** relative to one eligible reader,

  with **ρ_κ = σ²_c/(σ²_c + σ²_η)** the common-mode correlation (the array's saturation constant) and **q = P(o_r eligible)** the fiber-eligibility factor. Three deviations from textbook √R, all measured:
  1. **Saturation**: as R → ∞, G → √(q/ρ_κ) — the array cannot average below the shared κ noise. Gain in dB: 10·log₁₀(qR) − 10·log₁₀(1 + (qR−1)ρ_κ).
  2. **Eligibility attenuation q**: the filed band design thresholds on ρ = o_r/d_r, so only readers with small offsets can ever cross (measured: crossings at ō = 0.356 vs population 0.479, sd 0.171 → q ≈ Φ(−0.35) ≈ **0.37**). Counting statistics scale as √(qR) ≈ √(0.37R): **the fleet currently fields ~37% of its nominal aperture.** Dividend: the geometric team's reader-invariant units (φ = d/d_noise, §2.4 of their doc) set q → 1 — a ~2.7× event-rate recovery that shows up as pure array gain.
  3. **Heteroskedastic κ**: per-reader angular noise ≈ κ^(−1/2); cold/loose strata (κ≈11) carry (24/11) ≈ 2.2× the noise variance of warm/tight strata (κ≈24). Same √R, 2.2× worse baseline: **cold-stratum detection costs 2.2× the reader-nights.** The corrected generator contract (wave-3 §3) makes this a pre-stateable curve.

**(ii) Level/warmth channel** (instantaneous position, S-leg regressions — o_r does *not* cancel):

  μ̂(t) = x̄(t) − ô̄, per-axis precision σ_η(u)/√R — the √R is real — **but the estimand is fiber-contaminated**: along W, 99.5% of single-reader variance is between-reader, cos(W, PC1_pers) = 0.978, and the cross-roster centroid noise σ_o(W)/√R (≈ 0.055 at R = 21, from sd 0.254) stays ~50% of a typical room step's warm component (0.112). Arrays shrink the fog around W; **no R rotates it away.** Along v* (the REG-1 generalized eigenvector, volume+ / presence−), ICC is 0.38–0.50 — roughly half the variance is room response — so v* is where the level channel actually earns its √R. The confound annotation cos(W, v*) ≤ 0.44 is array-invariant.

### 1.3 The beamformer was already computed: v*

The optimal fleet beamformer (max room-response per unit personality noise) is the leading generalized eigenvector of (C_room, C_pers) — **REG-1 didn't just diagnose the confound; it computed the array's beam direction.** The dual weighting is the geometric doc's irony made explicit:

- **Step array weights ∝ Σ_η⁻¹** (whiten idiosyncratic noise) → favors the ICC-*unreliable* dials — cynicism (73% of step variance, ICC 0.64), joke. The base orbit lives off the reliable subspace.
- **Level array weights ∝ (Σ_o + Σ_η)⁻¹** → favors the ICC-*reliable* subspace — and, rotated to v*, the fiber-orthogonal directions.

"Reliable" selects for the wrong invariance per channel; the array needs both weightings and must label which one each statistic uses.

### 1.4 The fiber aperture dividend (what different fibers buy)

Readers at *spread* fiber points — unlike co-located sensors — can discriminate base motion from fiber motion: a displacement shared by all readers = base step (measurable **cohesion**, Wesley's channel); a displacement pattern matching the offset subspace = fiber artifact ("the rooms were furniture"). Discriminator: step energy inside the fiber span ≈ 3/7 = 0.43 isotropic (measured 0.40–0.44 ✓). Minimum array sizes this forces:

| threshold | R | why |
|---|---|---|
| geometric minimum | 4 | any fiber-orthogonal contrast needs R > dim(U_fib) = 3 |
| covariance stability | 8–10 | top-3 C_pers eigenvalues estimable (REG-1: directions < 1% of leading personality variance not estimable even at 15–21) |
| event floor (9 nights) | ≈ 12/night sustained | ≥ 20 crossings at 0.194 crossings/reader-night |
| near-full transition coverage | ≈ 30 | D → 0.97 (§2) |
| "animation" scale | ≈ 100 | D → 1.00, S CI half-width 0.71, events 8.7× floor (§2–3) |

**The array equation, one line:** SNR_fleet(R) = SNR₁·√(qR/(1+(qR−1)ρ_κ)) on the step/timing channel (level channel: √R precision on a fiber-contaminated estimand; beam along v*).

---

## 2. Detection thresholds: power-vs-R for the four legs

Back-of-envelope from filed quantities (wave-2: 21 readers, 72/189 r-nights attendance, 14 counted down-crossings at W=12 → **0.194 crossings/reader-night**; coverage 0.60 at effective R≈8/night → per-reader per-transition coverage p₁ ≈ 0.108; wave-2 S slope 1.41, CI [−0.31, 2.80] → SE 0.79 at R=21). These are pre-registered *predictions* for the subsampling experiment (§5), not runs.

| R (/night, 9 nights) | events ≈ 0.194·9R | Leg A (floor ≥ 20; Z≫2.8 once read) | Leg D coverage 1−(1−0.108)^R | Leg P / ICC | Leg S: CI half-width ≈ 1.55·√(21/R) |
|---|---|---|---|---|---|
| 5 | ≈ 9 | VOID (floor) | 0.44 | saturating | ±3.2 |
| 8 (today, effective) | ≈ 14 | VOID — **the field corpus's exact spot** | 0.60 (filed ✓) | saturated | ±2.5 |
| 10 | ≈ 17 | VOID (17 — the filed headline void) | 0.68 | saturated | ±2.2 |
| 15 | ≈ 26 | PASS floor | 0.82 | saturated | ±1.8 |
| 21 (full attendance) | ≈ 37 | PASS | 0.91 | saturated | ±1.55 (reproduces filed CI ✓) |
| 30 | ≈ 52 | comfortable | 0.97 | saturated | ±1.3 |
| 100 | ≈ 175 | 8.7× floor | ≈ 1.00 | saturated | ±0.71 |

Readings:
- **Leg A is floor-dominated, not slope-dominated.** Once ≥ 20 crossings exist, the timing statistic is overwhelmingly powered (Z ≈ 6+ at the filed A = 0.647 vs shift-null ≈ 0.10); below the floor it is a void by registration. A's power-vs-R curve is a step function at R ≈ 12/night sustained.
- **The headline field VOID was an attendance problem, not an effect problem**: 21 enrolled readers at 38% attendance = effective R ≈ 8 → 14–17 events < 20 floor. **The Captain's doctrine — connect the boats — is precisely the registered fix**: either sustained attendance ≥ 12/night or more readers. The apparatus was never under-powered per event; it was under-crewed.
- **Leg D is the leg the 100-boat array transforms**: per-transition coverage is an OR across readers — an exponential-approach (logistic) curve in R, *not* √R. 0.60 → 0.99+ between R = 8 and R = 100. D saturates only near the doctrine's own number.
- **Leg P and ICC are flat-fast** (roster-level cosine/ICC estimates: SE ∝ 1/√R, threshold 0.5·P_rest generous; instrument-vs-collapse discrimination trivial by R ≈ 5).
- **Leg S is the slowest and the least honest beneficiary**: SE ∝ 1/√(RN); excluding 0 at the wave-2 slope (1.41) needs R ≈ 26 sustained; halving the CI needs 4× the readers. But the geometric doc §3.5 stands: S's x-invariance cannot distinguish instrument from thermostat **at any R** — precision without identifiability. More boats sharpen a blurred question.
- **2AFC α-resolution**: pair-ranking power rises as Φ(d′·√(R/21)); with the wave-3 prior (endpoint recovery P≈0.7 at R=21), resolving adjacent α pairs (Δα = 0.25) at P≈0.9 needs roughly 4× reader-nights; intermediate-α ordering (prior P≈0.5) needs ~8×+. The array is the α-axis microscope.

---

## 3. The 100-boat claim, honestly

**Verdict: one-third theorem, one-third engineering bill, one-third metaphor.**

**Real consequences of the math (the theorem third):**
1. Event scaling: 175 counted crossings over 9 nights at R = 100 — 8.7× the void floor; the A/D legs stop being coverage-limited entirely.
2. Coverage: D → ≈1.00 — every registered transition gets a boat over it inside ±3 speaks.
3. SNR: +10·log₁₀(qR) dB ≈ +15.7 dB over a single eligible reader (q = 0.37), +6.8 dB over today's effective R = 8 — minus saturation losses 10·log₁₀(1+(qR−1)ρ_κ), which is exactly why ρ_κ must be measured (§5) before promising dB.
4. The fiber aperture: 100 boats at spread fiber points make base-vs-fiber discrimination routine (cohesion becomes a calibrated output channel, not a confound).

**The metaphor third — what "3D pixelated video" overclaims:**
- All readers observe **the same orbit point at time t**. The array is a *single-target tracker with redundant sensors*, not a tomographic array: 100 boats do not sample 100 parts of the school. The honest render is a **comet, not a video**: nucleus = roster-mean base trajectory (1-D track on a low-dim orbit), tail = the reader cloud on the 3-dim fiber (κ = tail thickness). Pixels: only directions with both room variance and personality aperture light up — **dead dials (panic in wave-1) are unrecoverable pixels**; the W-warmth pixel renders permanently behind its annotation (cos(W, v*) ≤ 0.44 — arrays average fog, they don't rotate it away); the crisp pixels are v* (volume–presence) and the cynicism step axis.
- What *would* make it genuinely school-like: **multiple rooms observed simultaneously** (the Tap's three rooms, the officers-quarters 12 tiles, quilt-fleet) — cross-room base orbits = multiple fish. The tomographic array is the *room*-array, not the reader-array. The Captain's instinct is right but the unit of the array's third dimension is the room, not the reader.

**The engineering bill (what rendering would actually take):**
1. **Spatial array**: sustained reader coverage ≥ 12/night per room (the floor), spread across the fiber (attendance design — wave-3's a-priori persona-balanced attendance, impact doc's "clears §5.3 in an afternoon").
2. **Time-synchronization = referent-lock, not clock-lock**: the kernel-centroid theorem is the fleet's NTP — every boat must window with the covariant referent c_W(t) = t + (W−1)/2. A boat on the window-start referent is displaced −5.5 speaks and contributes **zero** timing information (A_start = 0.0, p = 1.0, both waves). The matched-filter lead ℓ(W) ≈ W/5 means referent mismatches smear the animation by whole pixels. The crab-traps wire format's `ts` + sha256 chain is already the sync substrate.
3. **The coordinate firewall across boats**: each boat computes dials/edges in its own chart; nothing on the boat side may compute offsets from roster means (wave-3 honesty guard 6); the array fuses *after* chain-sealing per-boat edges, and never feeds the roster mean back — otherwise the array launders its own readings (registration axiom 2; the decoy panel is the audit).
4. **Per-boat calibration**: N ≥ 9 nights per boat for offset SE (0.194 events/reader-night is itself a per-boat calibration constant); the wave-3 α-sweep + 2AFC pairs are the array's calibration certificate — without them the "animation" could be the matched filter watching itself (the geometric doc's one risk, §5.2: the kernel peaks at transitions *by construction*).
5. **Streaming registration**: window-registration instead of night-registration (impact doc artifact #5).

---

## 4. The fleet as the array — what exists today

| array element | surface | status |
|---|---|---|
| single-room readout | elephant `roomd :4073 GET /field`, `GET /rooms/{name}/field` | ✅ live |
| the synapse (array packet + sync log) | elephant-sim-worker → crab-traps `POST /edge` (D1, sha256 chain, before/after/delta/imbalance/ts/provenance) | ✅ **live in production** — the only segment already interoperating |
| fusion identity | `imbalance ≡ d_mu`, proven to 1e-12 (quilt-rust field-edge-bridge) | ✅ proven, unwired elephant-side |
| 1-pixel display | fleet-radio "quilt weather" broadcast nightly | ✅ live |
| array memory | collective-unconscious — readings as first-class metadata | ✅ live |
| multiplexing fabric | quilt `field`/`reading` sensor-cell kind (synergy handoff #1) | 🟡 designed |
| multi-target aperture (the real 3rd dimension) | the Tap's 3 rooms; officers-quarters 12 tiles; quilt-fleet federation | 🟡 rooms exist, unfused |
| **the beamformer** | v* (volume–presence) wired into field outputs | 🔴 computed (REG-1), not wired |

**The missing piece is the one the demo-wiring priority (impact doc artifact #3) names**: the end-to-end live path — real Tap room traffic → roomd field computation → crab-traps D1 ledger → visible dashboard → policy loop — *designed at every segment, wired at none*. Plus three array-specific gaps: (a) the referent-lock protocol (covariant c_W(t) across all boats), (b) the multi-room fusion node (fleet-rooms is the natural keel), (c) v* as a first-class field output so the array beams along the clean axis from day one. Zero new math is required for the wiring — the identity is proven; this is the rare array that has its fusion equation before its power cable.

---

## 5. Registered experiment: REG-4 ARRAY-GAIN (proposal, to be frozen before reading)

**Design.** Subsample readers on existing sealed corpora (field waves 1+2 now; wave-3's 6 corpora when generated, post K-leg rework): R ∈ {5, 10, 15, 21} (and 8 = today's effective attendance), ecological thinning of actual nightly attendance lists (primary; full-attendance imputation as flagged secondary), stratified across strata; ≥ 200 subsample draws per R; CIs by reader-clustered bootstrap B = 2000, seed 20260821; never pool waves; q-rule referent-invariance inherited as a robustness column; all wave-3 void rules inherited per subsample. Read-only against corpora; the subsampling scheme is the registered object (seed frozen before first draw).

**Pre-stated predictions (the √R-vs-flat contrast, per leg):**
- **A**: point estimate Â flat in R (unbiased); event count n(R) = 0.194·9R (wave-2 class); Z grows ∝ √(qR) above the floor; floor void below R ≈ 12/night. *Shape: step function at the floor, then √(qR).*
- **D**: coverage sigmoid 1−(1−0.108)^R — exponential approach, **not √R** — 0.44/0.68/0.82/0.91 at R = 5/10/15/21. *Shape: logistic.* (D's non-√R shape is itself a discriminator: if D grows like √R instead, per-reader coverage is not independent — an ρ_κ-class finding.)
- **P / ICC**: flat-fast; SE ∝ 1/√R; saturated by R ≈ 10.
- **S**: SE ∝ 1/√R (CI half-width 3.2/2.2/1.8/1.55 at 5/10/15/21); slope point estimate stable; **identifiability unchanged** (pre-stated: S's x-invariance verdict does not flip with R — if it flips, that is a finding about attendance composition, not exposure).
- **ICC**: within [0.85, 0.96] bracket for all R ≥ 15; wider CIs below.
- **κ-channel**: estimate ρ_κ = mean pairwise within-night residual correlation across readers, separately for warm/tight (κ≈24) and cold/loose (κ≈11) strata. **Pre-stated: ρ_κ(loose) > ρ_κ(tight)** (heteroskedastic array); ρ_κ ≈ 0 in tight strata (no saturation up to R = 21).
- **Eligibility recovery**: rerun counting statistics with reader-invariant band units (φ = d/d_noise): q: 0.37 → ≈ 1; event counts jump ≈ 2.7× at fixed R. Pre-stated as a design dividend visible as array gain.

**Branch verdicts (pre-stated):**
| outcome | verdict |
|---|---|
| Z-type statistics grow ∝ √(qR); shapes match per-leg predictions | **SCALE** — the doctrine's math confirmed; publish the power-vs-R certificate |
| statistics plateau below √(qR) growth | **SATURATE** — common-mode noise dominates; file ρ_κ as a fleet constant; array gain capped at √(q/ρ_κ) |
| statistics grow *faster* than √(qR) | **SUPERLINEAR (pathological)** — shared estimator artifact: the matched filter watching itself at fleet scale (the one-risk guard); triggers the phase-structured null before any array claim |
| no growth, statistics at null for all R | **NOISE-DOMINATED** — the school isn't there; the honest negative with a mechanism |

**Why it matters**: REG-4 converts the Captain's doctrine from metaphor into a calibration certificate — the R-ladder (4 / 8–10 / 12 / 30 / 100) becomes the fleet's staffing and wiring specification, and ρ_κ becomes the number that tells us whether a hundred connected boats buy +16 dB or +7.

---

## Provenance

Read (read-only): the six input documents above. No corpus, repo, or registered file touched; no runs; all quantities are arithmetic on filed numbers (flagged as back-of-envelope where they feed §2's table as predictions). **STRICT read-only honored: this document is the sole write.**
