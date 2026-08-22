# WAVE-3 S5 — UNBLINDING + VERDICT (registered H-GEN, Arm 1)

**Filed 2026-08-21. Executes S5 of `memory/wave3-generation-plan-2026-08-21.md`
against the frozen registration `memory/wave3-registration-2026-08-21.md` +
G6 addendum. Verdicts were filed blinded first (S4,
`memory/wave3-S4-analysis-2026-08-21.md`); this document opens the seals and
adjudicates.**

## 1. Unsealing (the registered G3 procedure)

Per the S3 run doc §3: sidecar copy-back into each corpus dir, then
`python3 scripts/riverbed_generator.py --unblind data/wave3/<id>/riverbed-sealed-<id>.json`.
The seal chain verified **16/16** with zero tamper failures: each redacted
manifest pins its sidecar's sha256; each sidecar pins all 9 night sha256s.
**Seals are one-time in effect: verdicts were filed before opening (§5.2
satisfied), and the study is now permanently unblinded** — the canonical
sealed sidecars remain at `data/wave3-sealed/` (committed at S3); the
in-corpus copies are the executed procedure's artifacts.

**α-truth (all seal-verified, matches the S3 run doc's sealed table):**

| corpus | α | branch | pair_seed |
|---|---|---|---|
| w3k01 | 0 | instrument | — |
| w3k02 | 0.25 | intermediate | — |
| w3k03 | 0.5 | intermediate | — |
| w3k04 | 0.75 | intermediate | — |
| w3k05 | **1** | **collapse** | — |
| w3k06 | null | noise (flat warmth, κ_R 8, per-night whole-persona redraw) | — |
| w3q1m / w3q1n | 0 / 0.25 | pair P1 | 2101 |
| w3q2m / w3q2n | 0.25 / 0.5 | pair P2 | 2102 |
| w3q3m / w3q3n | 0.5 / 0.75 | pair P3 | 2103 |
| w3q4m / w3q4n | 0.75 / **1** | pair P4 | 2104 |
| w3q5m / w3q5n | 0 / **1** | pair P5 (endpoint) | 2105 |

m = the lower-α member of every pair. Machine-readable:
`data/wave3/s5-unblinded-summary.json`.

## 2. The verdict — branch tables R5, adjudicated against the α-truth

**H-GEN is FALSIFIED. The confirmed outcome is the registration's pre-stated
anti-hypothesis (ii): the honest negative.** The apparatus runs clean (zero
primary-channel voids; seeds/draws valid; gates green at S3) and recovers the
noise endpoint, **but cannot separate instrument from collapse even at
endpoints**: all three α=1 corpora (w3k05, w3q4n, w3q5n) are read as
instrument by every content leg.

| leg | instrument (α=0) | intermediate (.25–.75) | collapse (α=1) | noise (w3k06) |
|---|---|---|---|---|
| **A** timing | **PASS** (3/3 fire, p≈0) | PASS fire / ordering clause FAILED (flat, no weakening with α) | **KILL** — fires strongly, "misread as instrument" (3/3, the pre-stated cell) | **PASS** (silent, p=.12–.89) |
| **D** coverage | **PASS** (.7–.9 vs null 0.0) | PASS above-null / trend clause FAILED | **KILL** 2/3 above null; 1/3 uninformative (q5 D_null=1.0, n=1) | **PASS** (lowest, tracks null) |
| **P** persistence | **PASS** (0.99-class) | PASS hold / trend clause FAILED (pair gaps ≈2e-4) | **KILL** — holds ⇒ "misread as instrument" (3/3); *the registered instrument-vs-collapse headline discriminator, falsified* | non-discriminator, as pre-stated |
| **S** exposure | **PASS** (x-invariant) | PASS invariance / trend clause FAILED | **KILL** — collapse signature absent corpus-wide (0/96 channels) | **PASS** (x-invariant) |
| **ICC** (post-G6) | **PASS** instrument-vs-noise (re-band [0.60,0.80]; w3k01 .877 above top, inside superseded original bracket — annotated) | PASS (near instrument; pair gaps ≤.005) | **non-discriminating** (G6-disclosed re-read; stable lens .79–.81 inside instrument range) | **PASS** — collapses to **0.260**, the prediction (G6 draw-level 0.228 confirmed) |
| **2AFC** | FAIL (all legs): no pair separable at any channel; blinded majority calls 3/5 vs truth = chance; P_trans ranking 0/5 | | | |
| **gradient** | FAIL: α *is* a total order yet every signed direction violates it at ~chance (spread: 5/10 violations under the TRUE order) | | | |

**What passed:** signal-vs-noise. A/D/ICC recover the noise endpoint 16/16 —
every warmth-structured corpus fires on A, the null corpus is silent; ICC
separates .73–.88 vs .26. The battery is a verified signal-vs-noise
instrument. **What failed:** everything registered to carry α (P, S, 2AFC
ordering, gradient). **Anti-hypothesis (i) is not triggered:** it
presupposes o/d-pipeline recovery with decoy disagreement; no estimator
recovered branches at all, so the failure is structural channel invisibility,
not estimator-specific contamination.

## 3. α-verification of the localization (verified, no longer inferred)

The S4 inference is now verified against the true labels and the generator
source (line 845: `vibe0 = clamp(CENTER + pool_vibe_z + (1−α)·dev)`):

1. **α enters only the per-night persona anchor.** Room path, κ(t), rosters,
   and the charisma-pull fiber are α-invariant by construction.
2. **The three α=1 corpora are indistinguishable from α=0 on every registered
   leg** (A p≈0; P .983–.989; S x-invariant; ICC .79–.81) — the KILL cells
   fire exactly as pre-stated.
3. **Matched pairs prove it pairwise:** members differing only in α separate
   1–2 orders below room-draw scatter on every statistic — A rate ~18×,
   P_trans ~16×, S slope ~10×, spread ~75×, ICC ~28× (between/within
   ratios). The between-corpus trends that exist (spread clusters 1.34×,
   mild standalone ICC decline, S-slope scatter) **evaporate inside the
   pairs** — room-draw-carried, not α-carried. Three α-mirages, one design
   that caught all three.
4. **Why P cannot see it:** the charisma-pull fiber state is
   within-night-constant, so the pre/post offset cosine is ≈1 regardless of
   whether the offset is reader- or room-carried. **Why S cannot see it:**
   anchor-scale variation is small against per-speak noise at this n.
   **Why A/D cannot see it:** they read the room path, which α does not
   touch.

## 4. Deferred items adjudicated

- **ICC (post-G6)** — computed at unblinding through the registered adapter
  (`load_wave → Measurement.icc()`, `data/wave3/s5-icc.json`): table above.
  PASS instrument-vs-noise; non-discriminating instrument-vs-collapse
  (disclosed in advance by the G6 addendum).
- **Decoy-panel v6 — NOT EVALUABLE, MOOT.** S3 filed no decoy outputs; the
  S4 blinded window could not fire the rule. Rule 6's only licensed action
  is voiding an *apparatus-validation claim*; the verdict is the honest
  negative — no validation claim exists to void. Protocol lesson booked:
  decoys must be filed S3-side (or built blinded at S4 before verdict
  filing) in the next registration.
- **Five W8-only low-count cells (v3)** — branch-conditional resolution:
  w3k05.W08.can (19; collapse) = **branch hit** (pre-stated "no void on low
  count"); w3q1m.W08.can/act (16/18; instrument) and w3q1n.W08.can/act
  (17/18; intermediate) = **floor-VOIDs**, W8 sensitivity channels only.
  Primary channel untouched (min 23 crossings corpus-wide).
- **One v2 sensitivity void** (w3k05 W8|canonical, null-rn 0.143 vs
  signal-rn 0.246) — the blinded VOID ruling stands conservatively;
  post-unblinding annotation: on the collapse branch A is not read, so the
  branch-consistent reading is branch-hit. The annotation remains on the
  sensitivity cell only.
- **S4's booked item 4** — RESOLVED affirmatively: the lone silent corpus
  w3k06 **is** the null-mode corpus (seal-verified); the signal-vs-noise
  pass stands.
- **Unexecuted registered expectation (disclosed):** registration §2.5's
  generated-corpus calibration (cos(v̂_temp, Ŵ) ≥ 0.8 under instrument; the
  confound-annotation-vs-α calibration curve) was queued by the G6 addendum
  for S4 but never executed in the blinded window. It is computable only
  post-hoc with disclosure now; booked for the next registration instead.

## 5. The learning loop — S5→next-registration handoff

**Diagnosis:** α was injected into a channel the registered legs cannot see.
The anchor sits upstream of the charisma-pull fiber; the fiber's
within-night-constant state pins P, its reading-level effect is below S's
rejection power at this n, and A/D only read the α-invariant room path.

**Primary proposal (wave-4 generator change):** re-point α from the static
anchor into the fiber's within-night **target trajectory**:

```
target_R(t) = pool + (1−α)·dev_R + α·room(t)
```

with `room(t)` the latent AR(1) tangent wobble already shared pairwise. At
α=1 readings track the moving room — offsets room-carried AND time-varying,
so P's pre/post cosine decorrelates across transitions (the registered
collapse signature becomes reachable); at α=0 readings track the reader's
dev (offsets reader-carried, persistent). This preserves the registered
collapse semantics, stays in persona/field-measure space (coordinate
firewall intact), and the 2AFC pair mode isolates it exactly (pairs share
room(t) bit-for-bit). **Secondary** (if the fiber rework is too heavy):
register an anchor-reading leg as an apparatus extension in a fresh
registration. **Rejected:** scaling room-path μ-step/κ contrast by α — that
makes A/D fire on amplitude, not carrier (a pass for the wrong reason).
**Keep:** the matched-pair design (it is what proved α-invisibility), the
sealed-sidecar protocol, decoys filed S3-side, the calibration curve
registered pre-S4.

## 6. What wave-3 established (dissertation-usable)

> Wave-3 ran the registered field apparatus (legs A/D/P/S + slope/ICC
> machinery, thresholds frozen from REG-1) on a sealed, ground-truthed
> forward-model generation corpus (16 corpora: α ∈ {0,.25,.5,.75,1} +
> null-mode + five α-only matched pairs) and returned the honest negative
> pre-stated as anti-hypothesis (ii): the apparatus discriminates
> signal-vs-noise (16/16; ICC .73–.88 vs .26) but cannot separate instrument
> from collapse even at endpoints — and the sealed α-truth verifies the
> localization: the branch parameter lives in a per-night persona anchor
> upstream of a within-night-constant charisma-pull fiber, a sufficient
> statistic to which every registered leg is provably blind (α-only matched
> pairs separate 1–2 orders below room-draw scatter on every statistic).
> The negative is a calibration bound on the apparatus's detection envelope,
> and the matched-pair design that established it is itself a registered
> methodological result.

## Provenance

Read: the frozen registration + G6 addendum + κ-check re-run; the S3 run
doc; S4 blinded analysis + summary JSON; the 96 leg files (S4-window
numbers, cited not recomputed); the sealed sidecars (opened this step,
procedure above); `scripts/riverbed_generator.py` (α entry point, line 845).
Written: this document; `data/wave3/s5-unblinded-summary.json`;
`data/wave3/s5-icc.json`; the in-corpus sidecar copies (procedure
artifacts); `memory/wave3-S5-verdict-2026-08-21.md` (workspace);
`research/topic.md` advisory line (dissertation, annotate-only). No frozen
document edited; no sealed datum altered (seal chain re-verified on open).
