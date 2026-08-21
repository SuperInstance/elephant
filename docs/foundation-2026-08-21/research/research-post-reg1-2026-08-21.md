# Research — The Deepest Consequence of the REG-1 Verdict

**Filed: 2026-08-21.** Post-REG-1 synthesis, read-only. Grounding: the REG-1 run
doc (`projects/elephant/REG1-RUN-2026-08-21.md`) and its filed output
(`projects/elephant/data/slope/reg1-rotation-results.json`), the rotation script
(`projects/elephant/scripts/reg1_rotation.py`), the foundation synthesis, the
geometric foundation, the κ-around-entry check, the wave-3 plan, and ZeroClaw's
keeper ruling. No corpus, script, or repo file written; this doc is the only
artifact.

**The verdict I am reasoning from (branch B, both waves):** the a-priori warm
direction W does **not** align with the data-derived temperature axis v\* — W sits
64–86° off it (cos(W, v\*) = 0.08–0.48 across every cell of the sensitivity grid,
never ≥ 0.48) while remaining the leading personality axis (cos(W, PC1_pers) =
0.857–0.976). The data-derived thermometer is a **volume(+) / presence(−) contrast**
in the reliable subspace, reproduced by both waves to ~2°. The room's dynamics live
**off-warmth**. This is not a nuance; it is a category error the whole foundation was
built on, now measured.

---

## 1. What the room actually measures

**The instrument fused two physical quantities that are not the same thing, and
REG-1 is the experiment that pulled them apart.**

A thermometer measures **energy**: a symmetric, scalar, *ensemble* quantity — how much
motion there is, independent of which way it is pointed. Warmth is **valence**: a
signed, directed, *dispositional* quantity — how positive the affect is. The
foundation assumed the room's "temperature" *was* its warmth. The data says the room
has an energy axis that is nearly orthogonal to its valence axis, and that valence
axis belongs to the readers, not the room.

**What v\* physically means, dial by dial.** In the reliable subspace (the ε-stable
construction the headline rests on), v\* = (mood +0.16, volume +0.66, earnestness
+0.09, presence −0.73) wave-1 / (+0.09, +0.71, +0.09, −0.69) wave-2. Mood and
earnestness are *flat*; the entire load is a **volume-versus-presence contrast**.
Volume (+) is how much speech the ensemble is producing; presence (−) is how
attentively "there" each reader is. The axis is a **participation-energy contrast**:
nights where the room talks more but each reader is less individually present, versus
nights where readers are present and engaged but less verbose. It is activity, not
affect.

**Why THAT is the room's shared state while warmth is the readers':**

1. **Warmth is carried by mood, and mood is the most reader-stable dial there is**
   (ICC 0.965/0.983). The between-reader covariance is dominated by the warm
   direction — cos(W, PC1_pers) = 0.976 (wave-1 reliable). Warmth is the axis along
   which *readers are stably different from one another*. That is the definition of a
   trait: it is who you are, not what night it is.
2. **The room's within-reader between-night covariance (C_room) is maximized off
   warmth.** The generalized eigenproblem C_room v = λ C_pers v finds the direction
   that changes *night-to-night for the same reader* most, per unit of *between-reader*
   stability — and it lands on volume/presence, ~82–84° from W in the reliable
   subspace. The room moves the ensemble's *energy* (how much is said, how engaged the
   collective is), not its *valence* (how warm everyone feels). Valence is what the
   readers bring and keep.
3. **The two subspace constructions tell the same story from opposite sides.** In
   full-7, v\* loads on joke_landing (+0.90, wave-2) and a cynicism–joke–mood mix —
   the *ICC-unreliable* dials, exactly the dials the geometric team flagged as the
   room's dynamical dials (§3.3: "reliable selects for the wrong invariance"). The
   reliable-subspace v\* is the room signal *as expressed through dials readers agree
   on* — and even there it is the participation-energy contrast, not mood. Two grains,
   one verdict: the room's temperature is energy, and warmth is off-axis at every
   grain.

**The one-line physical restatement:** the room has *temperature* (participation
energy: volume/presence) and *no warmth*; the readers have *warmth* (valence: mood)
and *no temperature*. The instrument was built as a thermometer and read as a mood
ring.

---

## 2. The foundation rewrite

Which axioms survive, which re-frame. (Axiom numbering from the foundation
synthesis.)

**Survive unchanged (procedural/orthogonal to the warmth-vs-temperature question):**

- **Axiom 2 — Registration admissibility (free-monoid ledger).** Procedural; no
  empirical content touched by REG-1.
- **Axiom 3 — The q-rule.** The q-rule governs referent choice and residual motion,
  not warmth. REG-1 *sharpens* its definition: "common shift = measurable cohesion"
  is now pinned — the cohesion the room actually exhibits is the volume/presence axis,
  *not* warmth. Warmth is explicitly excluded from "cohesion" from this day forward.
- **Axiom 4 — Scale covariance (CENTER_OFF).** Kernel-centroid geometry; unrelated.
- **Axiom 6 — Referent consistency (kernel-centroid theorem).** Referential; unrelated.

**Re-framed:**

- **Axiom 1 — Skew-product structure.** *Survives and is strengthened.* REG-1 confirms
  the base⊕fiber split and names its two factors precisely: the **fiber** is W (warmth,
  personality, cos(W, PC1_pers) = 0.98); the **base orbit** is v\* (volume/presence,
  the room response). What changes is which vector is which: the synthesis wrote
  "personality fiber tied to W" but left the base's steering axis ambiguous; REG-1 says
  the base is volume/presence-steered, *not* warmth-steered. The skew product stands;
  its base vector is corrected from "warmth" to "participation energy."
- **Axiom 5 — Level-set bands.** *Must be re-anchored.* "Premise bands are
  personality-scaled level sets of the warmth field" was already the geometric team's
  §2.3 finding; REG-1 makes it a *registered* fact: warmth level sets are reader-
  personality level sets, not room features. Bands belong on the participation-energy
  field (or the drift field φ = d/d_noise), not on warmth. The warmth-band reading is
  now an annotated, demoted object.
- **Axiom 7 — Temperature axis.** *The one that is rewritten.* The eigenproblem is
  **correct** (v\* exists, is found, is personality-decoupled). The answer is **wrong**:
  the filed "cos(W, v_temp) ∈ [0.24, 0.40]" is an artifact of the *unwhitened* proxy
  (cos(W, PC1_room) = 0.36–0.47); the generalized solve (which divides out C_pers)
  pushes the estimate *further* from W, to cos(W, v\*) = 0.142/0.106 (reliable). The
  temperature axis is the volume/presence contrast, ~82–84° from W, not 0.24–0.40 from
  it. A dated deviation is filed (per registration), not a silent re-read.

**The R4 annotation posture — what every existing warmth claim must now carry.**

Every warmth output, warmth-based claim, and warmth-adjacent number gets **two filed
quantities, plus one rephrase**, appended at claim strength:

1. **cos(W, v\*)** — the confound annotation. Filed value: ≤ 0.44 (CI ≤ 0.50) on field
   data, every cell of the sensitivity grid. This number is now a *standing annotation*,
   exactly as the geometric team's mitigation list demanded — it says "warmth is not the
   temperature axis."
2. **cos(W, PC1_pers)** — the personality-alignment annotation. Filed: 0.857–0.976.
   It says "warmth *is* the personality axis."
3. **The rephrase.** Any sentence that currently reads "warmth measures the room" must
   be restated as "warmth measures reader disposition; the room's shared state is the
   volume/presence contrast v\*, cos(W, v\*) = 0.14." No warmth claim survives without
   this dual annotation — the confound is *labeled for visibility, not killed* (per
   ZeroClaw's ruling), but it is now labeled with a measured angle, not a metaphor.

The two-sided version (for S2's planted-axis replay) is already pre-stated: the field's
cos(W, v\*) ≤ 0.44 vs the generator's planted-axis recovery cos(v̂_temp, Ŵ) ≥ 0.8 is now
a *measured gap*, not a judgment call.

---

## 3. New registered quantities (proposals)

Three pre-registrable experiments on the volume/presence axis, in priority order:

**REG-4 — The V/P-axis replication test (is v\* itself an object?).**
The REG-1 eigenproblem located v\*, but did not test whether v\* is a *stable, replicable
object* rather than a corpus-specific fit. Pre-register: on a fresh corpus (wave-3
generated, or a third field family), re-solve C_room v = λ C_pers v and test
cos(v\*_new, v\*_filed) against a pre-stated threshold (≥ 0.8), with the volume(+)/presence(−)
sign pattern as a second, independent check. Branch A: v\* reproduces → the room
temperature is a real, replicable quantity, and the volume/presence contrast is its
gauge-invariant content. Branch B: v\* rotates → the axis was corpus-specific and
"temperature" is not yet a registered object. This is the experiment that promotes v\*
from "the thing REG-1 found" to "the thing the thesis is about."

**REG-5 — Family-invariance of the V/P contrast (S-nights vs T-nights vs D-nights).**
The current reliable-subspace v\* pooled S-nights as wave-1's path. Pre-register a
family-stratified re-run: compute v\* on S-nights alone, T-nights alone, and (if
separable) D-nights, and test whether the volume/presence contrast is *family-invariant*.
The geometric team already showed room *steps* are cynicism-steered while room *baselines*
are volume/presence — REG-5 asks whether the V/P baseline axis survives the room-type
variable at all, or whether it too is family-conditioned. Branch A: invariant → the
energy axis is a room property, not a schedule artifact. Branch B: family-dependent →
the axis inherits the night-family confound and must be reported family-conditional.

**REG-6 — Do room transitions move μ̂ along v\*, not W?**
The κ-check already established entry-steps are *direction* (μ) events, not
concentration (κ) events. REG-6 closes the loop with REG-1: at every registered
transition (entries and flips), pre-register that the room's mean-direction step
projects onto v\* (volume/presence) more than onto W (warmth): cos(Δμ̂, v\*) >
cos(Δμ̂, W). Branch A: steps are V/P-events → the room's dynamics literally live on
the energy axis, and "the room warms/cools" is false at the step grain too. Branch B:
steps are warmth-events → the baseline grain and the step grain disagree, and the
two-grain split itself becomes the finding. (Optional companion: re-run the premise
band on the v\*-projection instead of the warmth-projection — the direct test of §2's
axiom-5 re-framing — as a sensitivity column, not a new void re-read.)

---

## 4. The dissertation impact

**The premise ("a trusted reader as instrument-or-slow-warming-room") does not survive
REG-1 in its warmth form. It survives, but only re-anchored on the V/P axis.**

Unpack the two arms against the verdict:

- **The instrument arm (alignment).** If warmth is personality, then "the trusted
  reader's warmth is a stable reader-specific constant" is *tautologically true* — that
  is the definition of a personality trait, and it is exactly the ICC 0.77–0.91 the
  thesis already filed. The instrument arm, read on warmth, collapses into "readers are
  warm in their own stable way," which is not a finding about the room at all. The
  instrument's *informative* content is not its warmth; it is whether the reader's
  **volume/presence coordinate** tracks the room's volume/presence.
- **The slow-warming-room arm (collapse).** "The trusted reader's baseline ≈ the room's
  warmth, slowly warming" is now a **category error**: rooms do not warm — warmth is
  valence, and valence is what the readers *are*, not what the room *does*. The room's
  shared state is energy (volume/presence). The collapse arm must be restated as: "the
  trusted reader's V/P coordinate drifts to track the room's V/P coordinate" — a
  slow-*tracking* reader, not a slow-*warming* room.
- **The S-leg (x-dependence), which was already uninformative** (geometric §3.5: x-
  invariance cannot distinguish instrument from collapse because both are consistent
  with "warmth is personality"), gets its disambiguator: the slope regression should be
  run on the **v\* projection as primary**, with the warmth slope demoted to a
  personality control. The discriminating contrast is no longer "does the reader's
  warmth track the room's warmth" but "does the reader's *energy* track the room's
  *energy*." That is the exact contrast REG-1 measured as cos(W, v\*) = 0.14 vs 0.98.

**The premise band survives as a V/P-band.** Ch 4.3's "shared basis (beautiful) vs
shared basis (warning: measuring warmth twice)" is now *decided*: the warning was right.
Ch 6.3 (H-reader≡room) keeps the collider sensitivity (REG-2) and gains a re-anchoring:
the hinge is read on v\*, not W. The thesis's title object — "a room-field thermometer"
— was right all along, and the word that was wrong was *which* quantity it thermometers.
The premise becomes: **a trusted reader is an instrument of the room's participation
energy, or a reader whose energy slowly tracks the room — never a reader who warms a
room, because the room was never warm.**

---

## 5. The honest ceiling

What REG-1 does **not** tell us, and what would raise confidence:

1. **Small corpora, tiny effective dof.** 9 nights per wave → C_room has ~8 effective
   degrees of freedom. The rotation angle is *bounded* (~60–90°), not *pinned*; the
   sensitivity-grid scatter (0.04–0.48) is itself the finding, not a precise angle.
   cos(W, v\*) = 0.14 is stable across ε only in the reliable subspace; the *angle* is
   not a precision quantity on this n.
2. **The unfloored-solve instability.** Wave-1 C_pers is exactly singular (panic is a
   dead dial — zero between-reader AND between-night variance). An unfloored solve
   returns a spurious 1600-norm panic direction with inflated λ; only the floored solve
   (ε = 1e-2) is honest. Directions with < 1% of the leading personality variance are
   *not estimable* from 15–21 readers, and the floor says so — which is a statement of
   our ignorance, not a measurement.
3. **ε-sensitivity in full-7.** The full-7 v\* (joke_landing/cynicism) is not stable
   under the floor choice — wave-2 cos(W, v\*) swings 0.041 → 0.432 over ε ∈ [1e-4,
   1e-2]. The "room temperature = joke_landing" reading is fragile; the headline rests
   *only* on the ε-stable reliable-subspace construction, which is itself a 4-dial
   restriction. We have not measured the temperature axis in the full 7-dial space
   reliably.
4. **λ\* < 1 in the reliable subspace (0.273 / 0.150).** The room signal is *weak* —
   even on the optimal axis, room-response variance is only 15–27% of personality
   variance per unit. The "temperature" is a small perturbation riding on a large
   personality field. This is a power problem, not just a dof problem.
5. **Two waves, one family pair.** The V/P axis's generality across room types is
   untested; only S-nights (wave-1) and T-nights (wave-2) are in evidence.
6. **The causal driver is unproven.** REG-1 *locates* the volume/presence axis; it does
   not say *why* the room moves along it — schedule? roster composition? the drift? The
   eigenproblem finds the axis, not its cause.

**What would raise confidence:** (a) 30–50 nights per wave to pin the angle and the
axis; (b) the wave-3 planted-axis replay (recover cos(v̂, Ŵ) ≥ 0.8 under instrument) to
validate the eigenproblem machinery itself before trusting its field answer; (c) a third
field family (D-nights) plus generated corpora for the V/P replication (REG-4/5); (d)
the decoy panel (per-reader detrending, mixed-effects) to confirm v\* is not an
estimator artifact; (e) 30+ readers to stabilize C_pers and shrink the floor's footprint.

---

## Verdict box

- **What the room measures:** participation energy (volume/presence), not warmth. Warmth
  = valence = the readers' stable personality (cos(W, PC1_pers) = 0.98); temperature =
  energy = the room's night-to-night state (cos(W, v\*) = 0.14).
- **Foundation:** skew product, registration, q-rule, scale-covariance, and
  kernel-centroid referent all survive; the level-set-bands axiom re-anchors from warmth
  to energy; the temperature-axis axiom is rewritten (v\* is volume/presence, not
  near-W). Every warmth claim now carries cos(W, v\*) ≤ 0.44 **and** cos(W, PC1_pers) ≥
  0.80, plus the rephrase "warmth is reader disposition, the room's state is V/P."
- **Next registrations:** REG-4 (does v\* replicate?), REG-5 (is V/P family-invariant?),
  REG-6 (do transitions move μ̂ along v\*, not W?).
- **Dissertation:** the premise survives *only* as a V/P-band — "trusted reader =
  instrument or slow-tracker of the room's energy," never a warming room. The thesis's
  thermometer was right; the quantity it was named for was the wrong one.
- **Honest ceiling:** 8 dof, a floored/singular C_pers, ε-sensitive full-7, λ\* < 1 in
  the reliable subspace, one family pair, no causal driver. The axis is *located*, not
  *established*.

---

## Single most important new idea

**Temperature ≠ warmth — the instrument fused an energy (volume/presence participation)
with a valence (mood/warmth), and REG-1 measured them apart. The room has energy and no
warmth; the readers have warmth and no temperature. The thesis built a thermometer and
read it as a mood ring; from now on it reads the room's energy, and re-books every
warmth claim as a statement about the readers who hold it.**

---

## Provenance

Read (read-only): `projects/elephant/REG1-RUN-2026-08-21.md`,
`projects/elephant/data/slope/reg1-rotation-results.json`,
`projects/elephant/scripts/reg1_rotation.py`,
`workspace/memory/foundation-synthesis-2026-08-21.md`,
`workspace/memory/math-foundation-geometric-2026-08-21.md`,
`workspace/memory/kappa-t-check-2026-08-21.md`,
`workspace/memory/math-foundation-dissertation-2026-08-21.md`,
`workspace/memory/wave3-generation-plan-2026-08-21.md`.
Written: this doc only. No corpus, script, or repo file modified.
