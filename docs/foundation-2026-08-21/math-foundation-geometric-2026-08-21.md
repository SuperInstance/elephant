# Geometric / Topological / Dynamical Foundation for the Elephant Thesis
*Written by the Mathematician (geometric/topological/dynamical seat), 2026-08-21. Read-only analysis — no repo files were modified; all numbers below were computed in-memory against the unmodified code and data (seeds 0/20260821-family where relevant; numpy only).*

Read: `elephant/field.py`, `elephant/vmf.py`, `elephant/sensors.py`, `elephant/dials/`, `scripts/premise_band_movers.py`, `scripts/slope_regression_w2.py` (+ `slope_regression.py` reader side), `scripts/e2_instrument.py`, and `data/slope/premise-band-movers-results.json` (E2/E3 run). Companion: the leader's round-1 vision, the algebraic position (`math-foundation-algebraic-2026-08-21.md`), the E2/E3 design doc.

---

## 1. The Geometry

### 1.1 The projection tower (the atlas, honestly)

The stack is not one space; it is a tower of charts, each arrow non-injective:

  Messages (Room) →[DialBank, 10 functionals] R^7_raw →[zvec: z = SCALE·(v−CENTER)] C=[−1,1]^7 →[π: z ↦ z/‖z‖] S⁶ ⊂ R⁷

- `field.RoomField.vector()` is the R⁷_raw coordinate; `normalize()` is π; `vmf.zvec()` is the affine chart (`vmf.py:zvec`, with `LO/HI/CENTER/SCALE` mirroring the dial bounds).
- The **metric is a convention, twice over**. The premise machinery uses the dial-RMS norm (‖x‖² = (1/7)Σx_k²) for the numerator o_R and the full Euclidean norm for the denominator d_R (`night_windows()` docstring pins this to the continuity ladder: "a Euclidean 7-norm numerator would inflate the score ~√7"). Geometrically: numerator and denominator live under **different metric tensors** — M₁ = I/7 for offsets, M₂ = I for displacements. The ratio ρ = o/d is therefore not a scalar under the dial-chart group; it is gauge-fixed by the filed corpus_sd convention (see §1.6 on the H3 reductio).
- `sensors.py` is the same tower in a different chart: radar frames are point clouds in a fleet configuration space; `RadarCoherenceDial._spread` = mean distance to centroid — literally a **concentration statistic** in R² (the ρ/κ of the fleet cloud); `kinematics()` with nearest-neighbor gating is a zeroth-order correspondence ≈ a discrete connection on configuration space; `FishingDayDial.read` = 0.55·radar + 0.45·(2·sounder−1), an **a-priori warm functional** — the same object class as `WARM`, with the same confound exposure (§3).

### 1.2 The state space is not S⁶ — it is the vMF phase cylinder

The leader's "vMF random field over S⁶" conflates the sample space of one reading with the state space of the field. The vMF fit returns (μ̂, κ) ∈ S⁶ × R₊ (`vmf_fit`). The natural state space is the **open cylinder** S⁶ × R₊ — the total space of the positive radial ray bundle over the sphere. κ is not a coordinate on S⁶; it is the fiber coordinate — inverse temperature (vMF is the max-entropy measure on S⁶ at fixed mean resultant; κ plays β in the Gibbs analogy; KMAX=500 is the dial-saturation phase boundary).

The **field trajectory** is the curve
  t ↦ (μ̂(t), κ(t)) ∈ S⁶ × R₊,
discretized at three grains: per-trailing-window (W=8, `vmf.windowed`), per-window-position (W=12, `night_windows`), per-stratum/per-night (the filed E2 quantities). The trajectory is the primary dynamical object; every "snapshot" in the codebase is a coarse-graining of it. Note what the cylinder view immediately exposes: `RoomField.concentration()` = 2‖v − 0.5·𝟙‖ is a **radial** statistic of the raw vector (banned from comparison paths, correctly) — it attempts to read the fiber coordinate κ from the base coordinate ‖v‖. The ban is geometrically necessary: radius and concentration are coordinates of different factors and are coupled only through the estimator.

### 1.3 Room warmth is a height function — and the height function is misplaced

`WARM` (z-space, normalized) = (0.717, 0.239, 0.239, −0.359, 0.359, −0.239, 0.239): mood dominates, cynicism/joke secondary. Warmth is the **height function of the Morse pair** (S⁶, W):

  h_W : S⁶ → [−1,1],  h_W(x) = W·x,  ∇_S h_W(x) = W − (W·x)x ∈ T_x S⁶.

Critical points: exactly ±W (warm pole / cold pole); level sets are geodesic S⁵'s. Room warmth dynamics is the meridian coordinate of the trajectory:
  d/dt warmth_vmf = ⟨W, μ̇(t)⟩ = ⟨∇_S h_W(μ̂), μ̇(t)⟩,
i.e., warmth change is the pairing of the tangent field μ̇ with the ambient-constant vector field W. **"Warmth" is well-defined geometrically — the question is whether W points where the dynamics live** (§3.3: it does not).

Dual-warming seam resolved: `field.RoomField.warmth()` and `vmf.WARM` carry **identical weights** (mood .30, joke .15, earnest/presence/volume .10, cyn −.15, panic −.10 — verified line by line). The two numbers differ only because they evaluate the same functional at **different rungs of the tower**: h_W on the raw vector v (radial part included) vs h_W on the normalized mean direction μ̂ (radial part projected out). The seam is a chart-consistency requirement: one functional, one rung. Forced code shape: warmth is only defined on the direction factor; the radial contribution W·v − W·μ̂‖v‖ should be logged separately or dropped.

### 1.4 Edges: tangent vectors, not chords; and the algebraic composition law is false as stated

`vmf.edge()` records d_mu = ‖μ̂_B − μ̂_A‖ (ambient chord). Geometrically natural objects:

- **Geodesic distance**: d_geo = 2·arcsin(d_chord/2) = d_chord + d_chord³/24 + O(d⁵). At the filed magnitudes (d_chord ≲ 1) the chord underestimates by ≤ 4% — tolerable, but state it.
- **Log-map edge**: E(A→B) = log_{μ̂_A}(μ̂_B) ∈ T_{μ̂_A}S⁶ = {μ̂_B − (μ̂_B·μ̂_A)μ̂_A}/(1 − (μ̂_B·μ̂_A)²)^{1/2} · d_geo. This is a genuine tangent vector; edges along a trajectory compose by **parallel transport** (Levi-Civita connection of the round metric; discrete implementation = Schild's ladder), and loops acquire **holonomy** — ambient chords have trivial holonomy. Loop events are rare in this corpus, so chord vs log is presently untestable; but only the connection view supports the leader's "edges as temporal gradient" (a gradient lives in the tangent bundle).

Correction to the algebraic position: it claims d_mu(A→C) = d_mu(A→B) + d_mu(B→C) "by transitivity of difference in R⁷". **False for the scalars the code computes** — chord lengths satisfy the strict triangle inequality; only the ambient displacement vectors μ̂_B − μ̂_A compose additively. The thin-category structure of the algebraic doc is carried by the ambient vector differences (a legitimate "chordal calculus" in the linear R⁷), not by `edge()`'s scalar triple.

The `real` flag is the best geometry in the codebase: `d_mu > db_factor · max(SE)` (`edge()`, db_factor=2) quotients the state space by the **noise cone**. Formally: define [μ̂] ~ [μ̂'] iff ‖μ̂−μ̂'‖ ≤ ε with ε = 2·max(SE) (jackknife SE from `vmf_fit`). The ledger's objects should be ε-equivalence classes; the recorded edges are the 1-skeleton of the **Vietoris–Rips complex** of the trajectory at scale ε. That is a metric coarse-graining — and it upgrades the algebraic "thin category" to what it actually is: the **action groupoid of ε-distinguishable transitions**, with `real: null/true` the computable boundary map. The noise quotient is the principled version of the leader's Problem 1 (edges tied to the noise floor): it already half-exists in code; formalize it and the hard-coded hysteresis margins inherit a derivation (§2.4).

### 1.5 The window-referent ambiguity is not an ambiguity — it is the kernel-centroid theorem

Every windowed statistic is the response of a boxcar kernel K_W of width W placed at lattice position t. For the split-half drift d_R(t), a step discontinuity of size s at boundary speak b produces the triangular response pulse

  R_W(t; b) = s · g((b − c_W(t))/(W/2)),  c_W(t) = t + (W−1)/2,

with g a tent of slope W/2 peaking at 1 when the **kernel center** c_W(t) = b. Two theorems fall out:

1. **Centroid referent is forced.** The response is symmetric about the kernel center; the unique time-coordinate under which the statistic is (i) translation-equivariant (shift the speak grid by δ ⇒ the event shifts by δ) and (ii) symmetric in event time is c_W(t). The window-**start** referent evaluates the same event at −(W−1)/2 = −5.5 speaks; with TOL=3 it is arithmetically blind to any event cloud of width ≤ 6. The measured result confirms exactly this: **A_center = 0.647 (p=0.0013) wave-2 / 0.632 (p=0.0001) wave-1; A_start = 0.0 (p=1.0) both waves.** The "start-referent sensitivity" is not a sensitivity — it is the same statistic read in a non-covariant chart.
2. **The event cloud has a measurable systematic lead.** Direct recomputation (kernel-center convention, wave-2, all counted down-crossings): median offset c−b = **−1.5 / −2.5 / −3.5 speaks at W = 8/12/16** — crossings lead the boundary by ≈ W/5, widening with W. This is the matched-filter ramp: the confirmation convention (`counted_crossings` records the *first* window of the ≥3-run) plus entry-margin means the recorded position sits on the rising flank, before peak response. The lead is a derivable quantity: ℓ(W) ≈ (W/2)·(θ_entry/s_step) + (HOLD−1)/2·stride with θ_entry the entry threshold in d-units (o_R/0.55). A foundation that derives ℓ(W) replaces both ad-hoc constants CENTER_OFF and TOL with one function.

**Scale-covariance bug (finding).** `premise_band_movers.py` fixes `CENTER_OFF = (W_PRIMARY−1)/2 = 5.5` at module level and uses it in `leg_A`/`_shift_table` for **all** W. The W-sensitivities therefore run with a non-covariant referent: at W=16 the correct offset is 7.5, so every event is displaced −2 speaks; the event cloud (quartiles −5.5..−2.0 even at correct centering) falls outside TOL=3 and the filed A collapses to 0.074 (p=0.99). Recomputed with covariant offsets, the near-boundary fractions are **0.667 / 0.647 / 0.370 at W = 8/12/16** — timing structure persists at all scales, decaying, instead of vanishing. Symmetrically, at W=8 the +2 error tightens the cloud and inflates A to 0.917. The catastrophic W-sensitivity flip in the filed results is thus **half referent-artifact, half physics** (spurious far crossings do grow: 4 → 6 → 17 far events, because a stratum of width 8 < W admits no fully-interior window at W=16).

**Scale-space verdict.** A principled windowing scheme exists and the code should be derived from it: the axioms (translation equivariance; causality — no new extrema under blurring; semigroup structure across scales) select a causal smoothing family (exponential/triangular kernels), **not** the trailing boxcar (`vmf.windowed`) or the split-half difference kernel (`night_windows`), whose difference response has sidelobes and admits spurious extrema as W grows — exactly the 17 far crossings at W=16. The forced conventions: referent p_W(t) = t + (W−1)/2 (covariant), tolerance TOL_W = c·W (scale-covariant; the registered TOL=3 at all W is a scale-space mismatch), thresholds derived from the response shape R_W, and W chosen from the scale axis by the data (the A statistic becomes a curve over scale, and its stability across W — 0.67/0.65/0.37 — is itself the reportable object, replacing single-W point estimates).

### 1.6 The H3 reductio, restated as gauge

The thesis's rotation argument (drift-geometry is geometry-malleable) says: the drift statistic depends on the metric/connection gauge. The geometric resolution is not to abandon drift but to **fix the gauge explicitly**: the dial-RMS/Euclidean pair (M₁, M₂) pinned by the continuity ladder to the filed corpus_sd convention. A gauge-fixed quantity is honest; an unacknowledged gauge is an artifact. The ledger should record (M₁, M₂, corpus_sd) as part of every ρ — the gauge is a parameter of the measurement, not a property of the room.

---

## 2. Hysteresis, Topologically

### 2.1 The automaton is a delayed relay; its Markov state space is S × {0,…,H}

`plain_state`/`entry_ok`/`counted_crossings` implement a **relay (non-ideal, Prager-type)** on ρ with two edges (0.3, 0.6), margin 0.05, dwell HOLD=3. The band state alone is non-Markovian (the algebraic doc says this correctly), but the **extended state** (band, counter) ∈ {kill, in, clear} × {0,1,2,3} — 12 states — is Markov: the counter is a clock variable; the system is a discrete retarded functional-differential system. Consequence for inference: crossing rates are computable from the 12-state chain only if the ρ-increments' autocorrelation is specified — and it is: stride-1 windows share W−1 speaks, so the correlation length of ρ(t) is ≈ W, not 1.

### 2.2 Relation to Preisach

A single relay is the atomic Preisach hysteresis operator. A continuum of relays R_{α,β} weighted by a measure μ(α,β) gives the full Preisach model, whose memory is the "staircase" partition of the (α,β)-plane by the input's past extrema; output is a rate-independent functional of the extremal history. The code's two-edge chain is a **two-relay Preisach cell with shared thresholds** — a degenerate Preisach measure (two point masses), plus a dwell-time extension (which breaks rate-independence and moves it toward play-type operators with delay). The topological content is real: the input–state graph of a relay is a **branched two-sheeted 1-manifold** over the input line; the loop area is the dissipated/hysteretic information per cycle. So yes — the hysteresis is a genuine topological feature **of the estimator's response**, independent of threshold values.

### 2.3 But the band structure in physical units is a personality artifact (the sharp finding)

Unpack ρ_R = o_R/d_R. The design's own honesty guard: numerator ≈ constant (ō = 0.479, sd 0.171 over 1918 wave-2 windows; the three-measurement agreement 0.46–0.56). Then the band edges are **level sets of the denominator, scaled per-reader by personality**:

  down through 0.6  ⇔  d_R ≥ o_R/0.55  (median needed: 0.85 corpus-sd),
  down through 0.3 (kill) ⇔  d_R ≥ o_R/0.30  (median needed: 1.56 corpus-sd).

Measured d-distribution (wave-2, all windows): mean 0.349, p90 0.676, p99 1.928. Transition-phase mean 0.363 vs stable 0.317 (`mean_d_by_phase`) — a 1.27× contrast, **not** the 0.29→0.75–0.93 bimodality the design predicted (those were stratum-mean-grain numbers; the windowed kernel attenuates — see §1.5). Consequences, all verified in the data:

- **The kill state is effectively unreachable by signal.** Transition-phase windows average 0.36; the kill threshold needs 1.56. Kills live in the noise tail (2.1% of all windows satisfy d > o/0.3). Crossing inventory at W=12 (wave-2): down-through-0.6: 14, down-through-0.3: 3. **The three-state automaton operates as a single relay at the 0.6 edge; the 0.3/kill edge is decorative.** The "premise band" 0.3/0.6 is, at window grain, a one-threshold system with a rarely-visited noise attic.
- **Band-mover status is filtered on personality.** Crossings occur at ō = 0.356 (vs population 0.479) — windows where the reader's offset happens to be small. A reader with o_R = 0.75 needs d > 1.36 to even enter the band (never happens); a reader with o_R = 0.23 enters at d > 0.42 (often). **The set of band-movers is selected by o_R, a stable personality quantity** (ICC 0.77–0.91). Every statistic computed over band-movers (all four legs) inherits this selection.
- Genuine topology vs threshold artifact, answered: the **relay topology is genuine** (§2.2) but the **level sets {ρ = 0.3}, {ρ = 0.6} are not features of a room field** — they are reader-personality-scaled images of the level sets {d = o_R/0.3}, {d = o_R/0.55} of the drift field. The invariant object is the drift field's own level structure relative to its noise floor.

### 2.4 Forced derivations (replacing HYST_MARGIN=0.05, HYST_HOLD=3, 0.3/0.6)

- Margin: in d-units the margin should be z_{α}·SE(d) — SE estimable by the same jackknife machinery already in `vmf_fit` (SE(μ̂)); 0.05 in ρ-units translates to a reader-dependent d-margin (0.05·ρ²/o_R), i.e., the code's uniform margin is not uniform where it matters.
- Hold: with correlation length ≈ W, three consecutive windows share W−3 speaks — **~1.25 effectively independent samples**. A dwell rule that actually enforces independence needs H ≈ τ_int/stride ≈ W/2 (≈6 at W=12). HOLD=3 mostly re-imposes the margin; it does not debounce at the estimator's true autocorrelation scale.
- Bands: express thresholds in **reader-invariant units**: the phase variable φ(t) = d_R(t)/d_noise (noise floor ≈ 0.29 filed, 0.317 measured stable-phase mean), with universal thresholds on φ. ρ then becomes a derived, convention-labeled quantity (gauge-fixed per §1.6), and band-mover selection on o_R disappears from the primary statistic.

---

## 3. The Personality-Confound Dissent, Geometrically

### 3.1 The empirical splitting: dial space ≈ personality fiber ⊕ room base

Computed read-only on wave-2 T-nights (canonical presence, z-space, reliable subspace {mood, volume, earnestness, presence}):

- **Reader offsets** (m_R − b̄) concentrate in a 3-dim subspace: top-3 eigenvalues of the offset covariance = 95.1% of variance (0.085, 0.055, 0.010 vs 0.006, 0.0012, …).
- **Room steps** (roster-mean pre/post transition displacements, 10 signal transitions) are dominated by a **cynicism axis**: stepPC1 = (−0.19, −0.04, −0.08, **−0.97**, −0.04, −0.01, −0.10), 73% of step variance, **cos(stepPC1, WARM) = 0.147**. Step energy inside the offset top-3 span: 0.40–0.44 ≈ isotropic (3/7 = 0.43) — steps are essentially **uncoupled to the personality subspace**.
- **Offsets are near-constants of the room flow.** Through transitions, per-reader offset directions change by 1−cos = 0.004–0.024 (≈ 5–12° at most; magnitudes by 3–21%). Pooled cos = 0.993 — the filed P_trans = 0.994.

So the geometry is a **skew product**: a low-dimensional base orbit (the room trajectory, cynicism-steered) × a 3-dimensional fiber of reader offsets on which the flow acts trivially to within ~5°. This *is* the leader's Problem 3 (JEPA of a JEPA) — but it is not an add-on to the riverbed; it is the load-bearing wall (§5.1).

### 3.2 W is the personality axis

- **83% of WARM's weight mass lies inside the ICC-reliable subspace** (‖WARM[RIDX]‖ = 0.828); within it, W_REL = (0.866, 0.289, 0.289, 0.289) — 87% mood.
- **Between-reader (personality) covariance, reliable subspace**: wave-1 PC1 = (−0.94, −0.13, −0.19, −0.26), 55% of variance, **cos(PC1, W_REL) = 0.978**. The dominant axis of stable between-reader variation *is* the warm direction. Wave-2: cos(PC1) = 0.710, cos(PC2) = 0.667 — W_REL lies in the top-2 personality span (92% of variance), roughly along the bisector.
- **Variance along W decomposes 99.5% between-reader / 0.5% within-reader** (wave-2; 99.3/0.7 wave-1). Caveat I insist on: a random direction in the reliable subspace already has median between-reader share 0.990 [p5 0.967, p95 0.997] — the whole baseline field is personality-dominated (that is ICC 0.77–0.91). **The specific, non-generic fact is the axis alignment** (0.978), not the variance share. The dissent should be stated precisely: W is not merely personality-contaminated like everything else; it is *parallel to the leading personality axis*.

### 3.3 The room's dynamics are off-W

- Within-reader night-effect covariance (baseline minus reader-mean, the room response): PC1 (58–66% share) has cos(W_REL) = 0.24/0.40 — the main room response is **orthogonal to warmth**; **PC2 (≈20% share) has cos(W_REL) = 0.958 (wave-2) / 0.801 (wave-1)** — warmth is the *second* room axis.
- Room steps: mean |step·WARM| = 0.112 vs between-reader offsets·WARM sd = 0.254 — **one reader's personality coordinate along W is ~2.3× the typical room step's warm component**, and the roster-mean jitter along W (≈0.25/√8 ≈ 0.09) is ~20% of the entire experimental ladder range (0.32–0.76).

The irony worth filing: the ICC-reliable subspace is reader-*stable* (personality fixtures); the room's dominant dynamical dial (**cynicism**, 73% of step variance) is precisely the dial *excluded* from the reliable subspace. "Reliable" selects for the wrong invariance: a room thermometer wants directions where readers agree and rooms differ; ICC selects directions where readers differ and rooms barely matter. The confound is structural, not incidental.

### 3.4 The geometric test (do this; it is one generalized eigenproblem)

The clean temperature axis is the direction maximizing room-response variance per unit personality variance:

  **C_room · v = λ · C_pers · v**  (generalized eigenproblem; C_room = within-reader between-night covariance of baselines, C_pers = between-reader covariance; both already computed above).

The leading eigenvector v* is the **data-derived warm direction** — maximally room-responsive, minimally personality-loaded. Report: (i) λ(v*), (ii) cos(v*, WARM), (iii) the ICC-along-v* (should be ≪ 0.99). My computations already locate W: v* is the night-effect PC1 direction (cos ≈ 0.24–0.40 with W), and W itself sits near the *second* generalized eigenvector. The registered prediction this replaces: "warmth is the room's leading axis" is **empirically false at both grains** (steps and night-effects); the foundation should derive W from (C_room, C_pers) or defend the a-priori choice against it.

Secondary protocol (cheap, registered): per-archetype offset means define the archetype axes; test whether W ∈ span(archetype axes) — E5's 93–96% between-archetype baseline variance says it will be.

### 3.5 What this does to leg S (the algebra of the uninformative result)

Score ρ_R,N is built from o_R (personality at 99%+) over d_R (noise + step response). If readers are pure personality constants, score ~ x regresses to slope 0 with reader FE absorbing the personality — **exactly the observed "x-invariant" outcome** (wave-2 slope 1.41, CI [−0.31, 2.80] ∋ 0; beats_competitor false). If readers are perfect room instruments, score ~ x also regresses to 0 (an instrument does not move when it measures correctly — that is the alignment arm's own definition). **The x-invariance outcome cannot distinguish the two hypotheses**; it is uninformative between "instrument" and "thermostat setting," because the outcome's dominant variance is the personality fiber either way. Meanwhile wave-1 (CI [0.33, 1.98], excludes 0, doesn't beat the competitor) and wave-2 class-residual (slope 0.70, CI [0.12, 1.47]) lean the *other* way. The S leg as designed measures the wrong contrast; the discriminating contrast is the generalized-eigenvector one (§3.4) or the room-response share along W (0.5% — measured, tiny).

---

## 4. The Derivable Code Shape (what the geometry forces)

1. **One kernel family, covariant referent, scale-covariant tolerance** (§1.5). Replace the trailing boxcar (`vmf.windowed`) and split-half (`night_windows`) with a causal kernel family; referent p_W(t) = t + (W−1)/2 as a *function of W* (fixes the CENTER_OFF module-constant bug class — the W-sensitivity numbers in the filed results are partly referent artifacts); TOL_W = c·W; report the A statistic as a curve over W (0.67/0.65/0.37 at covariant centering), not a point.
2. **Log-map edges with parallel transport** (§1.4): `edge()` gains log_{μ̂_A}(μ̂_B) alongside the chord; the ledger transports tangent edges by Schild's ladder; loop holonomy becomes a (future) invariant; chord retained with its O(d³/24) error bound stated.
3. **The noise-quotient ledger** (§1.4): objects = ε-equivalence classes at ε = db_factor·SE (the `real` flag generalized); the ledger is the Rips 1-skeleton; `real: null/true` is the boundary map of the groupoid, not a flag.
4. **One cross-correlogram object unifying legs A and D**: C(τ) = Σ dN_cross(t)·dN_trans(t+τ) between the crossing and transition point processes; A = C-mass in |τ| ≤ TOL_W normalized; D = per-transition coverage. The circular-shift null is then transparently the torus-shift null of one process against the other. (This also resolves the algebraic doc's "A and D are correlated" — they are two projections of the same coupling measure C.)
5. **P-leg replacement** (saturation): cos-of-offsets is pinned at 0.99 by personality dominance (‖O‖ ≈ 1.2 per reader vs ‖ΔO‖/‖O‖ ≈ 0.03–0.2); the 0.5·P_rest threshold ≈ 0.497 was unreachable by construction. Forced: report the **relative-change statistics with dynamic range** — per-reader 1−cos of direction (0.004–0.024 measured) and ‖ΔO‖/‖O‖ (3–21%), with the registered threshold on those.
6. **Reader-invariant band units** (§2.3–2.4): phase variable φ = d/d_noise with universal thresholds; margins from SE(d); hold from τ_int ≈ W/2; ρ demoted to a gauge-labeled derived quantity; band-mover selection on o_R removed from primary statistics.
7. **Warmth from the generalized eigenproblem** (§3.4): data-derived W = leading eigenvector of (C_room, C_pers), logged next to the a-priori WARM with their cosine; warmth claims carry the cos as a confound annotation. Same treatment for `FishingDayDial`'s 0.55/0.45 functional in sensor space.
8. **Tower consistency** (§1.3): one warmth functional, evaluated only on the direction factor; radial warmth logged separately or dropped.

None of these eight is a rewrite; each is a derivation of an existing constant or convention from the geometry the code already half-embodies. That is the sense in which the code "naturally flows over" this foundation.

---

## 5. Dissent and the One Risk

### 5.1 Dissent from the riverbed

The leader's vision — "a time-indexed vMF random field over S⁶; rooms are single-time snapshots; edges are gradients of sufficient statistics" — puts the sphere first. **The sphere is the wrong primary object.** The empirical geometry is the skew product (base orbit ⊕ 3-dim personality fiber, §3.1); the fiber dominates every scalar statistic the thesis instruments (99.5% of variance along W; 95% of offset variance in 3 dims; P pinned at 0.99; band thresholds personality-scaled §2.3). Fitting one vMF to the pooled window directions — which is what `vmf_fit` does — **conflates the two levels into one (μ̂, κ)**: μ̂ then reads mostly fiber (roster composition), and κ reads the mixture's concentration. The leader's own Problem 3 (two-level JEPA) is not the fifth problem; it is the foundation. The corrected riverbed: **a base-valued trajectory (room field on a low-dim orbit, cynicism-steered) with a fiber of reader charts (g_R, vibe_R), all statistics decomposed base/fiber before any thresholding.** Everything else in the vision survives — edges as transported tangents, hysteresis as relay topology, ledger as noise-quotient skeleton — but on the product, not on the bare sphere. Second dissent, sharper: "premise bands are hysteretic thresholds on the idiosyncrasy/drift ratio" — the ratio is personality-scaled by construction (§2.3); bands on the ratio are bands on personality-filtered drift, and the 0.3 edge is empirically decorative. Bands belong on φ = d/d_noise.

### 5.2 The one risk that could sink the thesis

**The thesis has exactly one load-bearing empirical leg, and it is kernel-geometry-conditioned.** Walk the chain: the personality fiber's projection noise dominates every level-and-direction instrument (P saturated, S uninformative between hypotheses §3.5, x-ladder's roster jitter ≈ 20% of its own range); the room's decisive signal survives only in the **timing statistic** A — and A's magnitude is a function of the (referent, W, TOL) calibration triple (§1.5): 0.65 covariant-centered vs 0.0 start-referent vs 0.074 non-covariant W16 in the filed results. If the timing structure turns out to be matched-filter self-calibration rather than room dynamics — the response pulse is peaked at the boundary *by construction of the kernel*, so any step-caused crossing lands near the boundary regardless of what the step is — then A measures "were the crossings step-caused" (which is D's content), and the *distinct* information in A collapses. The corpus cannot currently separate "crossings time-lock to transitions" from "the kernel peaks at transitions"; separating them requires the scale-curve (A over W at covariant referents — measurable, §4.1) and a phase-structured null (shift within stable strata, not across the whole night). If A falls with a phase-structured null and the scale-curve is flat at the null mean, the premise dies with nothing temporal left to measure, and the personality confound (§3) becomes the thesis's epitaph rather than its annotation: the instrument read the readers, the rooms were furniture.

### Mitigations that already exist and should be kept

The ledger's R4 annotation discipline, the registered competitor models in S, the class-residual sensitivities, and the null-night void rules are exactly the right posture toward all of this — the geometric view sharpens them (cos(W, v*) as a standing confound annotation; the scale-curve as a registered sensitivity), it does not replace them.

---

## Summary of Mappings

| Geometric object | Formula | Code |
|---|---|---|
| Chart tower | Messages→R⁷→[−1,1]⁷→S⁶ | `field.RoomField.vector/normalize`, `vmf.zvec` |
| vMF phase cylinder | (μ̂,κ) ∈ S⁶×R₊, κ=β⁻¹, KMAX=500 | `vmf.vmf_fit` |
| Warmth = height function | h_W(x)=W·x, ∇h_W=W−(W·x)x | `vmf.WARM`, `warmth_vmf`; `field.RoomField.warmth` (same weights, different rung) |
| Splitting base⊕fiber | offsets 95% in 3-dim span; steps uncoupled (0.40≈3/7) | `night_windows` o/d; this doc's computations |
| Personality axis ∥ W | cos(PC1_pers, W_REL)=0.978 (w1) | §3.2, read-only computation |
| Room step axis ⊥ W | stepPC1=cynicism, cos=0.147 | §3.3 |
| Clean temperature axis | C_room v = λ C_pers v | §3.4 (to be added, one eigensolve) |
| Kernel-centroid referent | c_W(t)=t+(W−1)/2; lead ℓ≈W/5 (−1.5/−2.5/−3.5 at W=8/12/16) | `CENTER_OFF` (currently non-covariant — bug), `TOL` |
| Relay/delayed automaton | 12-state (band,counter) chain; Preisach degenerate cell | `plain_state/entry_ok/counted_crossings` |
| Bands in physical units | d_kill=o_R/0.3≈1.56, d_in=o_R/0.55≈0.85; kill edge decorative (3/17) | §2.3; `EDGE_LO/HI` |
| Noise-quotient ledger | [μ̂] at ε=2·max(SE); Rips 1-skeleton | `vmf.edge(real=…)` |
| A+D unification | C(τ) cross-correlogram, torus-shift null | `leg_A`, `leg_D` |
| P saturation | cos=1−‖ΔO‖²/2‖O‖²; ‖ΔO‖/‖O‖≈0.03–0.2 | `leg_P`, this doc's recomputation |

*Written by the Mathematician (geometric/topological/dynamical seat), 2026-08-21. Read-only: in-memory computations only; no repo files written or modified.*
