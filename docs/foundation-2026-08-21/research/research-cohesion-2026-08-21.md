# Research: Cohesion as a First-Class Registered Quantity (COH)

**Author:** Research subagent (cohesion seat), 2026-08-21 14:32 AKDT
**Status:** STRICT read-only against `/home/eileen/projects/elephant`; this doc only.
**Sources:** `memory/math-foundation-wesley-localgpu-2026-08-21.md` (the reframe),
`memory/math-foundation-redteam-2026-08-21.md` (q-rule, o_R/d_R decomposition),
`memory/math-foundation-probabilistic-2026-08-21.md` (P crisis),
`memory/math-foundation-geometric-2026-08-21.md` (skew product, stepPC1),
`memory/foundation-synthesis-2026-08-21.md` (axiom 3 adopted cohesion provisionally),
`scripts/premise_band_movers.py` (offset_vecs/leg_P/night_windows; the code path COH reuses).

**Wesley's claim (granite3.1, verbatim core):** "if everyone moves as a collective, it's
not the room warming — it's the energy and cohesion emanating from our synchronized
actions." The team read the common shift as the confound that breaks P; Wesley read it
as the object. This doc supplies the operational spec the synthesis's axiom 3
("common shift is defined as measurable cohesion") adopted but never specified.

---

## 1. Cohesion, defined

**Definition.** At a transition with boundary speak `b`, split each common present
reader's step into common + residual parts (the red-team's decomposition,
`premise_band_movers.py:421-443` provides the machinery):

```
δ_R  = M_R(post) − M_R(pre)          (per-reader step, W=12 pre/post window means)
ĉ    = mean_R δ_R                    (the common shift — the roster-mean step)
r_R  = δ_R − ĉ                       (residual; note Δo_R = r_R exactly, offsets are mean-centered)
COH  = ‖ĉ‖ / corpus_sd               (cohesion: collective synchronization magnitude)
q    = RMS(r_R) / RMS(o_pre)         (purity/noise floor, the red-team q-rule)
```

**What it measures:** did everyone lean the same way, and how far? COH is the
magnitude of the room's synchronized translation — the impulse response of the *base
orbit* at a registered boundary. It is already half-computed by the code:
`night_windows` forms `b̄(t)` (the present-roster mean) at every window and uses it
only as the reference to subtract when building offsets. **Cohesion is the trajectory
of the very reference the code throws away.** The confound was the signal.

**Three objects, orthogonal by construction** (this is the definitional core):

| object | formal seat | reads | units / statistics |
|---|---|---|---|
| **Warmth** | height function `h_W(x) = W·x` on S⁶ | *direction* of the mean along the a-priori axis | `warmth_vmf`, X-ladder 0.32–0.76 |
| **Cohesion (COH)** | base-orbit translation `‖Δb̄‖` | *collective motion magnitude*, direction-agnostic | corpus_sd; q as purity |
| **Personality fiber** | offsets `o_R`, 95.1% in 3 dims | *idiosyncratic shape* — who leans differently | o_R, ICC 0.77–0.91, κ |

The measured geometry enforces the split: stepPC1 (73% of step variance) is the
cynicism axis with **cos(stepPC1, WARM) = 0.147** — the room's synchronized motion is
nearly orthogonal to warmth; and step energy inside the offset top-3 span is 0.40–0.44
≈ isotropic (3/7) — steps are uncoupled to the personality fiber. Warmth says *where
the room points*; the fiber says *how the readers differ*; cohesion says *how far the
whole room moved as one*. None of the three reduces to another.

**Natural units:** corpus_sd (commensurate with `d_R` and `o_R`, the registered
convention); the q-residual as the noise floor; the stepPC1 projection as the
direction channel (log `ĉ`'s components — the school's velocity *vector*, not just
its speed). Two magnitude channels must be registered separately: **COH_4**
(reliable subspace, RIDX — comparable to P's world) and **COH_7** (full dial space,
where the cynicism-steered motion actually lives; the reliable subspace excludes the
step's dominant axis by construction).

## 2. The reframe of the P crisis

The filed reading: P_trans = 0.9940 vs P_rest = 0.9935 against a 0.497 kill bar —
saturated, unfalsifiable, "the P leg is empty." The red-team proved the mechanism
(step ≈ rigid common translation on the 4-dial subspace; q_trans ≈ q_rest ≈ 0.13) and
the probabilistic doc's centering guard is a no-op.

**Wesley's inversion:** if the common shift is a quantity, P = 0.994 is not a failed
test of fiber invariance — it is a *successful measurement of base rigidity*. The
cosine saturated because the school translated without deforming: shape frozen
(P ≈ 1, offsets 95% in 3 stable dims), translation real (‖ĉ‖ = 0.14–0.36 corpus_sd
across the 8 wave-2 signal transitions). The corpus realized the "null" — and the
null has a magnitude. The question stops being "does idiosyncrasy survive the step?"
(unanswerable here, per the red-team) and becomes **"what does synchronization
predict?"** — which is answerable, because COH varies 2.5× across transitions.

**Testable hypotheses (each pinned to an existing measurable):**

- **H1 — Flip magnitude/coverage (near-mechanical, the floor test):** ‖ĉ‖ at a
  transition predicts per-reader drift response (`d_R` straddle peaks) and D-style
  crossing coverage. Direction: positive. Content is limited (d_R ≈ ‖ĉ + r_R‖), so
  the registered form is the *residual*: **purity (low q), not magnitude, predicts
  cross-reader coordination** — a coherent step moves many readers through the band
  together; an incoherent one of equal size does not.
- **H2 — Entry receptivity:** at roster-entry transitions the entrant has not
  synchronized; q should be elevated (measured: entry-family q = 0.135–0.204 vs hard
  flips 0.079–0.133 — T4a@12, the cold-entrant flip, is the corpus maximum at 0.204).
  Prediction: q_entry > q_flip with separation; and the entrant's own r_R is the
  largest in the roster.
- **H3 — School formation (κ):** COH predicts post-step concentration — a room that
  moves together tightens. Measurable: |Δlog κ| from `vmf_fit` on pre/post windows.
  The κ(t) check already hints flips are bigger κ-events than entries; H3 registers
  the direction: |Δlog κ|_flip > |Δlog κ|_entry > |Δlog κ|_null.
- **H4 — Warmth independence (annotation-grade):** |cos(ĉ, W)| stays small across
  steps (stepPC1 says 0.147). Cohesion is *not* warmth motion — which is exactly why
  it is a new axis: the warmth ladder spans 0.32–0.76 while the room's actual
  synchronized motion runs off-W. Every COH claim carries this cosine as its
  confound annotation, mirroring the cos(W, v*) discipline.

Post-hoc honesty: H2/H3 directions are read off the same 8 wave-2 transitions the
red-team tabled. They are *hypotheses for* the pre-registered test (§5, wave-1 +
nulls), not evidence — ZeroClaw's registration rule applies to us too.

## 3. The registered quantity: COH-v1

**Spec.** Per transition, per wave (never pooled), canonical presence, W=12:

- Roster = readers present in **both** the pre window `[b−W, b−1]` and post window
  `[b, b+W−1]` (the `leg_P::offset_vecs` presence construction, reused verbatim —
  **common-readers guard**: entries/exits must not move `b̄` by composition).
- Register the triple **(COH_4, COH_7, q)** + direction log `ĉ` and cos(ĉ, W).
- **Noise floor:** the same triple over adjacent in-stratum window pairs (the
  `rest_events` construction). The red-team measured q_rest (0.116–0.139) but never
  ĉ_rest — the common motion *at rest* is the one missing number, and it is COH's
  floor. Naive model: per-reader window noise σ_ε ≈ 0.3 corpus_sd over W=12, n≈8
  common readers gives floor ≈ σ_ε/√(W·n) ≈ 0.03 — an order below the observed
  0.14–0.36; but within-stratum schedule drift inflates it. Measuring it IS the
  experiment (§5).
- **Branches:** (a) COH at hard flips vs entry-family transitions vs null
  pseudo-transitions vs rest; (b) purity q by family (H2); (c) the κ cross-check
  (H3). Branch (a) is the survival branch; (b)/(c) are the differentiation branches.

**What COH adds beyond the existing legs.** A/D/P/S are all *fiber* statistics —
they measure reader offsets' timing (A), coverage (D), invariance (P), and warmth
coupling (S) *conditioned on* base events. COH is the first registered **base**
statistic: it measures the base orbit's impulse response itself. Concretely: (i) it
retires the saturated cosine — P's binary holds/kill becomes a two-channel readout
(rigidity ‖ĉ‖ + purity q) with dynamic range (0.140–0.359 measured, 2.5×); (ii) it
gives the D leg a covariate (was the transition covered *because it was big*, or
*because it was coherent?*); (iii) it opens the cynicism channel the reliable
subspace excludes — mood residual ratio 0.095 (rigid) vs cynicism 1.650 (idiosyncratic
on top of a large common cynicism step) is currently invisible to every registered
leg.

**Known instrument floor (register it):** readings are gain-masked projections
(`m_R = CENTER + g_R ⊙ (eff − CENTER)`), so even a perfectly rigid room step produces
*different* per-reader steps (g_R ⊙ Δeff). Part of q ≈ 0.13 may be mask dispersion,
not reader disagreement. Cheap check: recompute q under unit gains; if q drops, the
fiber's charts blur the base and the purity channel needs gain-normalizing.

## 4. The echo to the sonar doctrine

Hundred boats, one school. REG-1's volume/presence axis reads the school's **shape**
— its extent, how the fish are spread (the personality fiber; presence is literally a
reliable dial; κ is the shape's concentration). Cohesion is the school **moving
together** — the translation of the whole.

**Is cohesion the school's velocity?** Yes, in event-time: `b̄(t)` is the school's
center of mass; ‖Δb̄‖ per transition is its impulse speed. At rest COH decays to the
common-mode jitter (the drift current); at steps it spikes (the flick). So COH(t) as a
time series is the school's speed profile — near-floor between events, sharp peaks at
boundaries. The measured corpus state is precisely *a rigid school translating*:
shape frozen (P = 0.994), translation real (Ĉ up to 0.36). Wesley's "robust,
interconnected network" is the correct sonar picture: what the array logged as a
confound was the fleet changing course.

**Can the array measure it?** It already does, inverted. Every reader is a hydrophone;
the roster is the array. Classic array processing *rejects* the common mode to find
the local signal (a fish); the elephant's four legs do exactly that (offsets subtract
`b̄`). COH is the polarity flip: **the common mode is the signal** (the school), the
rejected part is the fish. The q-rule is the array's purity certificate — it says how
much of the apparent common motion is really local (q ≈ 0.13: on the mood dial, >90%
of the step is common). One caveat from §3: the hydrophones have different gains
(g_R), so the beamformer (mean over readers) sees an attenuated, differentially
weighted image of the room step — the array can measure the school's velocity, but
only through the readers' charts.

## 5. The pre-registered experiment (COH-1)

**Design (all read-only; analysis in /tmp per red-team precedent; seeds 20260821-family).**
For wave-2 primary and wave-1 replication separately, compute (COH_4, COH_7, q, Δlog κ,
cos(ĉ, W)) at:

1. **Signal transitions** (8 per wave), split into hard-flip vs entry-family
   (entry-family = T5/T5c/T4a wave-2; the roster-change strata);
2. **Null pseudo-transitions** — T9 (wave-2) / S5 (wave-1) midpoints *plus* matched
   stratum-interior pseudo-boundaries on signal nights (same spacing, no registered
   event);
3. **Rest pairs** — the `leg_P` rest_events positions (the noise floor).

Reader-clustered bootstrap B=2000 for CIs; verdicts require wave-2 primary with
wave-1 labeled replication.

**Pre-stated predictions (direction):**
- **P1:** COH_signal > 2× COH_rest with CI separation (steps are real collective
  motion, not accumulated jitter). Uncertainty declared: at stratum grain the
  per-reader contrast is 2.6–3.2× (0.75–0.93 vs 0.29), but at W=12 window grain the
  dilute contrast is only 1.14–1.5× (0.363/0.317; straddle 0.48/far 0.32). The
  boundary-anchored leg_P construction sits between; the honest prior is ~2×, i.e.
  the verdict is genuinely at risk — which is what makes it a test.
- **P2:** q_entry > q_flip (entries are mushier; the new fish doesn't move with the
  school).
- **P3:** |Δlog κ| orders flip > entry > null.
- **P4 (annotation):** |cos(ĉ, W)| < 0.5 everywhere.

**Pre-stated verdict rule:**
- COH_signal/COH_rest > 2 with CI separation **in both waves** → **COH registers as
  a first-class quantity (COH-v1)**; P's saturation is re-booked as base-rigidity
  measurement; branches (b)/(c) attach as differentiators.
- Null ≈ signal (ratio < 2 or CIs overlap) → **cohesion stays a confound**: the
  common shift is indistinguishable from within-stratum common drift, P remains
  uninformative under the q-rule, and the reframe dies as a nice metaphor.
- Wave-2 passes / wave-1 fails → COH-v1 registered with a replication-failure label;
  downgraded to descriptive, no branch claims.

**My pre-registered expectation:** signal ĉ (0.14–0.36) clears a rest floor that the
σ_ε/√(Wn) model puts near 0.03–0.05 — but the floor is the unmeasured number, and
within-stratum schedule drift (d_stable = 0.317 per-reader!) could push it toward
0.1–0.2 and squeeze the ratio to the bar. Call it 70/30 survives. The purity and
differentiation branches (P2/P3) are the richer payoff: magnitude alone does not
separate flip families cleanly (T3 = 0.140 min, T4a = 0.359 max, families
interleaved), but purity does (entry-family occupies the top of the q range).

---

## Verdict (this doc's answer)

**Cohesion survives as a quantity — provisionally, pending COH-1.** The evidence
already on file favors quantity over confound: the common step varies 2.5× across
transitions (not a constant artifact), has a stable structured direction (stepPC1 =
cynicism, 73%), is orthogonal to warmth (cos 0.147) and uncoupled to the fiber
(isotropic 0.40 ≈ 3/7) — noise has no preferred axis and no event-lock. The one
missing measurement (COH at rest/null) is exactly what the pre-registered experiment
supplies, with a real, pre-stated chance (~30%) of collapsing back to confound. The
deeper point stands regardless of outcome: the code computes `b̄(t)` at every window
and discards its motion — Wesley noticed what the subtraction throws away.

*No repo files were read-modified; no new repo files; analysis proposed, not run.*
