# The Math of the Fleet Field

**Author:** fleet mathematics lead (subagent)
**Status:** v0 — implemented in `elephant/fleetmath.py`, tested in
`tests/test_fleetmath.py`
**Reviewed by:** Qwen3.6-35B-A3B (kinematics + vMF), Seed-2.0-pro
(second opinion), DeepSeek V4-Pro (inductive biomass) — see "Review
notes" at the end.

This is the numeric spine of the elephant's sea legs: the mathematics
underneath the radar/sounder/nav dials in `elephant/sensors.py` and the
nudge in `elephant/nudge.py`. Four pieces:

1. **Kinematics** — direction, speed, and rate of change of every radar
   object from *exactly three readings*.
2. **Fleet coherence** — the fleet as a von Mises–Fisher field; κ as the
   "tight (on fish) vs scattered (searching)" statistic.
3. **Inductive biomass** — the good-days → spotty-days induction as a
   distribution-shift / novelty objective.
4. **Nudge math** — how a dial number becomes an attention multiplier.

**Units.** Positions in metres, times in seconds, speeds in knots
(1 kt = 0.514444 m/s), angles in degrees, accelerations in m/s².

---

## 1. Kinematics — three readings, everything moving

A radar sweep returns a *set* of 2-D target positions (range/bearing
converted to boat-relative Cartesian). We get three sweeps at times
`t1 < t2 < t3`. From them we must recover, **per object**, its
direction, speed, and rate of change. There is no velocity sensor — the
tropes ("boats together on fish") are deductions we never make; the
field is the reading. The only raw material is positions.

### 1.1 Association — nearest-neighbour gating

The sweeps are *sets*, not tracks: we do not know which point at `t2`
belongs to which point at `t1`. We associate greedily, exactly as
`elephant/sensors.py`'s `_associate`: each point in frame *k* claims its
nearest neighbour in frame *k+1* **within a gate radius `g`**, each
target claimed at most once:

```
for each a_i in frame k:
    j* = argmin_j ‖a_i − b_j‖   subject to  ‖a_i − b_j‖ < g,  b_j unclaimed
    pair (a_i, b_{j*})
```

Two failure modes, both bounded by the gate:

- **Missing association** — a target moves more than `g` between sweeps
  (fast boat × long sweep interval) or leaves the beam. Result: a track
  fragment; kinematics reported only for the fragments.
- **Mismatched association** — two targets swap nearest neighbours
  (crossing tracks, or targets closer to each other than their motion).
  Greedy NN is a *local* heuristic, not a global minimum; it can lock
  the wrong pairs. For 3–10 boats at fishing speeds the ambiguity is
  small, but it is the dominant *systematic* error in κ and in
  acceleration, so it is worth keeping in mind (§1.6).

Three subtler consequences of gating worth naming explicitly:

- **Selection (gating) bias.** Greedy gating prefers whichever
  measurement happens to land nearest the prediction, so it
  *systematically under-reads* large displacements — measured
  acceleration is biased toward zero. This is a hard bias, not noise.
- **Swap amplification.** The `1, −2, 1` acceleration stencil is exactly
  maximally sensitive to association errors: a single wrong pairing
  between adjacent frames injects a spurious acceleration of order
  `2 v / Δ` — enormous unphysical outliers.
- **Filter before associate.** Production trackers run a motion filter
  (constant-velocity/Kalman predictor) *before* the association gate, not
  after; the predictor keeps the gate small and resolves crossings. The
  v0 greedy gate is adequate at fishing densities; the filter is the
  documented next step.

### 1.2 Velocity from two positions

Between two sweeps `(t_a, p_a)` and `(t_b, p_b)` the velocity is the
forward difference:

```
v = (p_b − p_a) / (t_b − t_a)
```

Speed is `‖v‖`; direction is `atan2(v_y, v_x)` (math convention, CCW
from +x / "east"; compass bearing = `(90° − dir) mod 360°`).

### 1.3 Acceleration as the second difference

With three positions `p1, p2, p3` we form two velocities

```
v12 = (p2 − p1)/(t2 − t1),      v23 = (p3 − p2)/(t3 − t2)
```

and the acceleration is the central second divided difference:

```
a = 2 (v23 − v12) / (t3 − t1)                       (1)
```

This is not an ad-hoc choice. The second *divided difference*
`f[t1,t2,t3] = (v23 − v12)/(t3 − t1)` is, up to the factor 2, the exact
second derivative of the unique quadratic through p1, p2, p3 — i.e.
equation (1) is the **exact acceleration of the quadratic interpolant**
for *arbitrary* (non-uniform) spacing, not just uniform. (Concretely: the
interpolant is `p(t) = p1 + v12(t−t1) + f[t1,t2,t3](t−t1)(t−t2)`, whose
second derivative is the constant `2 f[t1,t2,t3]`.) For uniform spacing
`t2 − t1 = t3 − t2 = Δ` it collapses to the familiar centred second
difference

```
a = (p3 − 2 p2 + p1) / Δ²                              (2)
```

*Note.* The v0 code in `sensors.py` divided by `dt23` instead of
`(t3 − t1)/2`, which is correct only when `dt12 = dt23` and even then
doubles the magnitude. `fleetmath.py` uses (1) and reports acceleration
at `t2`.

### 1.4 Error growth with dt and noise

Let each coordinate carry independent Gaussian noise of std `σ`
(radar range/bearing error propagated to Cartesian). Propagating:

| quantity | error std | scales as |
|----------|-----------|-----------|
| position | `σ` | `σ` |
| velocity | `σ_v = √2 σ / Δ` | `σ / Δ` |
| acceleration | `σ_a = √6 σ / Δ²` | `σ / Δ²` |

The `√2` comes from differencing two independent positions; the `√6` =
`√(1² + 2² + 1²)` from the `1, −2, 1` stencil of (2). The message is
unavoidable: **acceleration is a second difference and its noise blows
up as `1/Δ²`**. Halving the sweep interval quadruples the acceleration
noise. This is why "three readings" gives a usable velocity but a noisy
acceleration, and why the sign of dκ/dt (a second-derivative-like
quantity) needs a least-squares fit rather than a raw difference.

### 1.5 When three readings are insufficient, and what 4+ buys

Three readings fail when:

1. **`Δ` too small** — `σ_a ∝ σ/Δ²` floods the acceleration.
2. **`Δ` too large** — the constant-acceleration (quadratic) assumption
   breaks (a boat manoeuvres), and association breaks (motion exceeds the
   gate, §1.1).
3. **Non-uniform sampling** — `dt12 ≠ dt23` biases (1) toward whichever
   interval is shorter; the central-difference weights are no longer
   symmetric.
4. **Entering/leaving targets** — a target present in only two sweeps
   yields a velocity but no acceleration; the code reports
   `accel = 0` (unknown) rather than fabricating one.

What a fourth (and fifth, …) reading buys:

- **Noise averaging** — fitting `p(t) = p0 + v t + ½ a t²` by least
  squares over *n* points drives the acceleration error down as
  `σ_a ∝ σ / √(Σ (tᵢ − t̄)⁴)`, far better than `√6 σ/Δ²`, and lets a
  higher-order (jerk) term be tested.
- **Better association** — a track (rather than a pairwise match) plus a
  constant-velocity or Kalman predictor resolves crossings that greedy NN
  gets wrong.
- **Manoeuvre detection** — residuals of the quadratic fit flag the
  moment a boat turns; three points can't distinguish a turn from noise.

### 1.6 Coordinate frame and the lever-arm correction

Radar measures **boat-relative** positions. The own ship moves and
rotates, so a stationary object's boat-relative position *changes*; that
change is the own ship's motion, not the object's. To recover water/geodetic
motion we must remove the own-ship contribution.

Let `p_rel(t)` be the target's boat-relative position, `p_true(t)` its
inertial position, `p_ship(t)` the ship's inertial position, `θ(t)` the
ship heading, and `r_ant` the (fixed) antenna lever arm in ship
coordinates. Then

```
p_true = p_ship + R(θ) (p_rel + r_ant),       R(θ) = [[cosθ, −sinθ],[sinθ, cosθ]]
```

Differentiating and rearranging gives the true velocity as

```
v_true = R(θ) ( v_rel + ω × (p_rel + r_ant) ) + v_ship
```

where `ω` is the ship's turn rate and `v_rel = d p_rel/dt`. Three terms
correct the raw relative velocity `R v_rel`:

1. **Own-ship translation** `+ v_ship` — the dominant term; implemented in
   `fleetmath.three_reading_kinematics(own_ship=…)` by shifting each
   sweep's targets by the own-ship displacement `(p_ship(t) − p_ship(t1))`
   before differencing.
2. **Own-ship rotation** `+ R ω × (p_rel + r_ant)` — matters only when the
   ship turns between sweeps; a second-order correction documented here,
   not (yet) implemented.
3. **Antenna lever arm** `r_ant` — enters only through the rotation term
   (a constant offset otherwise cancels in the differences).

Omitting these leaves a **false velocity** equal to `−v_ship` on every
stationary target — the classic radar artefact, and the reason the
`own_ship` argument exists.

---

## 2. Fleet coherence — the vMF concentration κ

The fleet field is a **distribution over boat headings and positions**.
"Clustered = on fish (same drag/tack); scattered = searching" is a
statement about that distribution's *shape*, and the right summary
statistic of shape is the von Mises–Fisher concentration κ.

### 2.1 Why κ, not just spread

The naive statistic is the spread (mean distance to centroid). Two
problems:

1. **Spread is scale-bound and unit-bound.** It answers "how far apart,
   in metres," not "how *coherent*." A fleet of ten boats drifting a
   mile apart but all pointing the same tack is highly *coherent* despite
   a large spread; a fleet bunched in 50 m but tacking every direction is
   incoherent. The field's felt property is *alignment*, and alignment
   lives on the sphere.
2. **Directions are circular data.** Headings wrap (359° → 1°). Their
   arithmetic mean is ill-defined (the mean of 359° and 1° is not 180°);
   spread of raw angles is meaningless across the wrap. The vMF is the
   natural distribution on the circle/sphere, and its concentration κ is
   the natural, wrap-safe coherence.

κ is dimensionless and bounded below by 0 (uniform/isotropic — loose,
warm, searching) and unbounded above (perfectly aligned — cold, tight,
on fish). This is exactly the room-temperature reading of the design
document: **cold = high κ, warm = low κ**, here applied to a fleet of
boats instead of a room of speakers.

### 2.2 The vMF and its MLE

The von Mises–Fisher distribution on the sphere S^{d−1} (unit vectors in
R^d) has density

```
p(x; μ, κ) ∝ exp( κ μᵀ x ),      ‖μ‖ = 1
```

with mean direction `μ` and concentration `κ`. Given unit vectors
`x_1 … x_N`, the MLE of the mean direction is the normalized resultant

```
r̄ = (1/N) Σ x_i,      μ̂ = r̄ / ‖r̄‖
```

and `R = ‖r̄‖` is the **mean resultant length** — the single number that
captures coherence (`R = 0` uniform, `R = 1` all-identical). The MLE for
κ solves

```
A_d(κ) = R,      A_d(κ) = I_{d/2}(κ) / I_{d/2 − 1}(κ)
```

where `I_ν` are modified Bessel functions of the first kind. Without
SciPy we use the Banerjee–Dhillon–Ghosh–Sra (2005) approximation — the
same formula the v3 design document uses:

```
κ̂ = R (d − R²) / (1 − R²)                            (3)
```

For the circle (d = 2, von Mises over headings/bearings) this is
`κ̂ = R(2 − R²)/(1 − R²)`. `R → 1` sends `κ̂ → ∞` (capped at 1e4 in the
code); `R → 0` sends `κ̂ → 0`. Exact inversion of `A_2` is a one-line
`scipy.special` step if precision matters; the approximation is within a
few percent for the κ range that matters here.

### 2.3 Two κ's: same-tack and bearing

A fleet field has two directional quantities, and the code exposes both:

- **Heading κ (same-tack)** — the vMF κ of the boats' *headings*
  (unit vectors `(cos θ, sin θ)`). On fish, boats drag/tack the same way
  → high κ; searching → low κ. This is the cleanest "coherence" reading.
- **Bearing κ** — the vMF κ of the unit vectors *from the own ship toward
  each boat* (angles `atan2(y, x)`). A fleet bunched on fish subtends a
  narrow bearing sector → high κ; a fleet spread around the horizon →
  κ ≈ 0. This is what `fleet_concentration(positions)` returns when no
  headings are given, and it is what "clustered vs scattered" means in
  the raw radar frame.

Both are legitimate; heading κ needs heading data (from §1's kinematics),
bearing κ needs only positions. The **positional** counterpart to κ is
the model-free `fleet_spread` (mean distance to centroid) — the scale of
the field, complementary to its shape.

### 2.4 How κ evolves — the scatter/bunch signal

When boats bunch onto fish, headings align *and* bearings tighten, so κ
**rises**. When the fleet gives up and searches, κ **falls**. The
derivative `dκ/dt` is therefore the fleet's *decision* read directly
off the field, with no trope-deduction:

- `dκ/dt > 0` — bunching (the fleet is converging; someone found fish).
- `dκ/dt < 0` — scattering (the fleet is dispersing; the fish are gone).

`kappa_rate(frames)` computes κ at each sweep and returns the
least-squares slope of κ vs time. A least-squares fit (rather than a raw
first difference) is deliberate: κ estimates from 3–10 boats are noisy,
and — per §1.4 — a derivative of a noisy quantity needs averaging.

### 2.5 Statistical significance of a coherence change (small fleets)

With N boats, the mean resultant length has (for κ = 0) a known null
distribution. By the central limit theorem the resultant `Σ x_i` is a
2-D Gaussian with covariance `(N/2) I` (each unit vector has coordinate
variance ½), so `R` is asymptotically Rayleigh with scale `1/√(2N)`:

```
P(R > r | κ = 0) ≈ exp( −N r² )                        (4)
E[R | κ = 0]    ≈ √( π / (4N) ) ≈ 0.886 / √N
```

The exact finite-N mean sits *below* the naive `1/√N` overestimate: for
N = 3 it is 0.525, for N = 10 it is 0.282 (both ~0.886/√N). A coherence
*change* between two sweeps of N boats is significant only if the
difference in R exceeds this `~1/√N` sampling noise. Two practical
consequences:

1. **Don't trust dκ/dt from a handful of boats on one sweep pair** —
   average over several sweeps (as `kappa_rate` does).
2. **Report κ changes with N attached.** A κ jump of X on a 3-boat fleet
   is far weaker evidence than the same X on a 10-boat fleet. The null
   (4) gives the exact p-value: `p = exp(−N r²)` against "no coherence."

---

## 3. Inductive biomass — the good-days anchor

The induction: *a week of good fishing, then it gets spotty. The good
days teach the system what driving over the right kind of biomass looks
like — inductively.* The mathematics of that induction is a **distribution
anchor** plus a **deviation score** that says "this stretch of water feels
like the good kind."

### 3.1 The anchor distribution

Each good day is reduced to a feature vector `x ∈ R^d` over the boat's
dials — e.g. sounder mean biomass, sounder variance (texture), radar
coherence κ, fleet mean speed, spread rate, nav course stability. The
good days form a cloud of points; we model that cloud as a Gaussian

```
anchor = N(μ, Σ),     μ = (1/N) Σ x_i,     Σ = (1/(N−1)) Σ (x_i − μ)(x_i − μ)ᵀ
```

(`biomass_anchor`). μ is "what a good day feels like"; Σ is "how much
good days vary."

### 3.2 The deviation — Mahalanobis, regularized

The question "does this stretch of water feel like the good kind?" is
answered by the Mahalanobis distance

```
D(x) = sqrt( (x − μ)ᵀ Σ⁻¹ (x − μ) )                   (5)
```

This is the number of *good-day standard deviations* `x` sits from the
anchor, in the anchor's own coordinate system — so it is scale- and
correlation-aware (a biomass dip along a normally-noisy axis scores low;
the same dip along a rock-steady axis scores high). `D` small = "the good
kind"; `D` large = a distribution shift (spotty water).

Two honest complications:

1. **Small N.** With only a week of good days, `N ≲ d` and the sample Σ
   is singular — (5) is undefined. Fix: **shrinkage** toward a scaled
   identity,
   ```
   Σ_shrunk = (1 − ρ) Σ + ρ · (trace Σ / d) · I
   ```
   with the Oracle Approximating Shrinkage intensity (Chen et al. 2010)
   computed in `_oas_shrinkage`. OAS keeps Σ well-conditioned and
   invertible even when `N < d`, and is the right default precisely
   because the anchor is built from *few* days.
2. **Outlier sensitivity.** The Gaussian MLE is not robust; a single
   anomalous "good" day inflates Σ. For v0 this is acceptable; a robust
   MCD/trimmed anchor is the documented next step.

Under the anchor, `D² ~ χ²_d`, so `D` can be converted to a percentile
("how typical is this water") via the χ² CDF — the same percentile-rank
idea used for acclimation in the v3 design.

### 3.3 Sample complexity — how many good days before the anchor is trustworthy

The anchor is two quantities with different sample requirements:

- **Mean μ** — trustworthy after `N = O(d)` days (the sample mean's
  error is `√(trace Σ / N)`).
- **Covariance Σ** — needs `N > d` days *without* shrinkage to be
  invertible, and more to be stable (the sample covariance's relative
  error is `~ √(2/(N−1))` per eigenvalue). With OAS shrinkage it is
  well-defined for any `N ≥ 2`, but the *directions* of the low-variance
  axes — the ones that matter for "spotty" detection — need `N` on the
  order of several times `d` to be reliable.

Rule of thumb for a `d ≈ 5–8` feature vector: **~10–20 good days** give
a usable anchor; the deviation is meaningful as a novelty signal before
the covariance is fully stable, because the shrinkage keeps it honest.

### 3.4 Updating as the season changes

The biomass shifts with the season, so a fixed anchor goes stale (every
day eventually reads as "spotty"). The fix is an **online** update — the
Normal–Inverse-Wishart (conjugate) posterior, or its cheap cousin an
exponential moving average:

```
μ_{t+1} = (1 − α) μ_t + α x_t
Σ_{t+1} = (1 − α) Σ_t + α (x_t − μ_t)(x_t − μ_t)ᵀ
```

α sets the memory: small α = slow drift (stable anchor), larger α = the
elephant relearns the season. The deviation `D` is then computed against
the *current* anchor, so the system tracks what "good" means *now* while
still flagging the *abrupt* drops (spotty days) as deviations. This is
the inductive loop: good days pull the anchor toward themselves, and the
anchor's own recent history is the yardstick.

---

## 4. Nudge math — dial numbers as attention multipliers

The dials do not replace the vision model; they *correlate*. Each dial
emits a number, and those numbers nudge the vision model about **what to
compare together**. This is a modality-weighting prior, and it is worth
writing down exactly.

### 4.1 From dial number to attention multiplier

`elephant/nudge.py` maps each dial reading to a modality with a sign
(`radar_coherence → radar +1`, `sounder_biomass → sounder +1`, `panic →
camera_out +1`, …). A modality's prior is the signed sum of its dials:

```
prior[m] = Σ_{dial d → (m, s)}  s · clip(r_d, −1, 1)          (6)
```

normalized so the strongest opinion is at most 1. The prior is then
blended into the vision model's cross-attention as a **multiplicative**
reweighting:

```
attention[m] ← attention[m] · (1 + strength · prior[m])        (7)
```

with `strength ≈ 0.15` by default ("the elephant nudges, it doesn't
drive"). Zero prior = compare as usual; a positive prior up-weights a
modality; a negative prior down-weights it.

### 4.2 Why multiplicative, and what it does to comparison cost

Suppose the vision model has an attention budget `B` (comparisons per
step) to spread over `M` modalities and, within each, over `F` frames.
Without a nudge the attention is uniform: `B/M` comparisons per modality,
every frame pair compared with equal weight — a cost that scales with
`M · F²` if the model compares all-within-all, or wastes `B` on
uninformative modalities.

The nudge turns the flat prior into a reweighted distribution
`q_m ∝ (1 + strength · prior[m])`. The information gained by the nudge —
the "surprise" of the field — is the KL divergence from uniform:

```
D_KL(q ‖ uniform) = log M − H(q),      H(q) = −Σ q_m log q_m
```

When one dial spikes (`prior[radar] → 1`), `q` concentrates on radar,
`H(q)` drops, and the KL rises: the field is *telling* the model where to
look. Operationally the comparison cost is concentrated rather than
spread:

- **The saving is comparing the right frames.** A high sounder biomass +
  rising radar coherence says "compare *this* water column to *last
  week's good hour*" — a tiny, targeted comparison. A flat sounder says
  "don't burn attention there" — the model skips it. The nudge's value is
  that the model spends its fixed budget `B` on the frames the elephant
  flags, instead of exhausting it on a uniform search.
- **Expected-comparisons argument.** If the "right" frame is one of `F`
  and the model samples frames with weight `∝ (1 + strength · prior)`, the
  expected number of samples to hit it falls from the uniform `F` to
  `F / (1 + strength · prior)`. Every unit of prior `strength · prior` is
  a direct divisor on the search cost over the flagged modality.

In short: the dial number is a **prior over where to spend attention**;
equation (7) is the mechanism; the saving is the attention budget the
model no longer wastes on flat modalities. The nudge doesn't make the
vision model *see better* — it makes it *look in the right place*.

---

## Review notes

The math above was checked by three models (DeepInfra API + self):

- **Qwen3.6-35B-A3B** reviewed §1 (kinematics) and §2 (vMF).
- **Seed-2.0-pro** gave a second opinion on the same.
- **DeepSeek V4-Pro** (self) reviewed §3 (inductive biomass).

What the reviewers confirmed, and what they got wrong — a caution worth
recording because *reviewer corrections are data to verify, not oracles*:

- **Confirmed** (both): the `√2` / `√6` noise scalings (§1.4), the
  Banerjee κ approximation (§2.2), and the `~1/√N` sampling-noise scale
  for R (§2.5).
- **Seed proposed two "corrections" that I verified are wrong** and
  therefore rejected. (a) It claimed `a = 2(v23−v12)/(t3−t1)` is "only
  valid for uniform timesteps" and offered an alternative; a direct
  numeric check on a quadratic (`5t²−2t+3` at t = 0, 2, 5) returns 10 for
  the divided-difference form and 2.8 for Seed's — the divided difference
  is exact for non-uniform spacing, as §1.3 states. (b) It claimed the
  null tail is `exp(−N r²/2)`, a factor-of-2 error; Monte Carlo and the
  CLT (covariance `(N/2)I`) both give `exp(−N r²)`, as (4) states.
- **Adopted from Seed** (valid): the gating/selection-bias and
  swap-amplification failure modes of greedy nearest-neighbour association
  (§1.1), and the "filter before associate" prescription.
- **Self-corrected** (from my own Monte Carlo, not a reviewer): the
  noise-floor *mean* of R is `√(π/4N) ≈ 0.886/√N`, not the `1/√N` I
  first wrote — `1/√N` is a slight overestimate of E[R] (exact small-N:
  0.525 for N=3, 0.282 for N=10).

Substantive design points, independent of review: the second-difference
acceleration denominator (vs the v0 `dt23` doubling/units error, §1.3),
OAS shrinkage for the small-sample covariance (§3.2), and least-squares
dκ/dt rather than a raw first difference (§2.4, per the `σ/Δ²` analysis
of §1.4).
