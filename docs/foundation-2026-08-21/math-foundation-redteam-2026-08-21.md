# EMPIRICAL RED-TEAM — Implementation Reality vs the Foundations

**Author:** Mathematician (empirical red-team arm; deepseek-v4-flash runner)
**Filed:** 2026-08-21
**Method:** read-only against `/home/eileen/projects/elephant` (no repo writes,
no commits, no new repo files); analysis scripts in `/tmp` only. Ran the
existing pytest suite. Reproduced `leg_P` and the full `night_windows`
pipeline directly against the corpus and re-derived every quantity in this
doc from live code paths (fidelity check: my `leg_P` call reproduces the
filed `P_trans = 0.9940217` / `P_rest = 0.9935313` to all 6 decimals).
Companion positions: `discussion-leader-round1-2026-08-21.md` (riverbed),
`math-foundation-probabilistic-2026-08-21.md`, `math-foundation-algebraic-2026-08-21.md`.

---

## 0. Verdict summary

1. **THE P CRISIS IS CONFIRMED — with a sharper mechanism than the
   probabilistic doc, and one correction to its proposed fix.** `P_trans =
   0.9940` vs the 0.497 kill bar is real and reproduces exactly. The common-
   shift critique is *structurally* right but the doc's §4 guard ("subtract
   the roster's common step before computing offsets") is a **no-op**: the
   offsets are already deviations from the present-roster mean, so a constant
   de-shift cancels identically (I verified `cos_guard == cos` to 4+ decimals
   on every transition). The real mechanism, measured: **the step is nearly a
   rigid common translation on exactly the 4-dial reliable subspace the P
   statistic uses** (pooled residual motion `q = RMS(r_R)/RMS(o_R) = 0.132`
   across steps vs `0.116–0.139` at rest — indistinguishable), and the
   non-common motion lives in the dials P excludes (cynicism residual ratio
   1.65). P is unfalsifiable **in this corpus** because the corpus only
   realizes the common-shift null, and that null predicts cos ≈ 1 by
   construction.
2. **Test suite: 277 tests, all pass (12s). The E2/E3 pipeline has ZERO
   tests.** `premise_band_movers.py`, `e2_instrument.py`, `e2_field.py`,
   `premise_measurement.py` are absent from `tests/`; the only run-time
   checks are asserts inside `main()` (replay equality, wave gate, ladder
   ±0.10), which pytest never executes.
3. **Window-referent critique confirmed with real numbers:** A(center) =
   0.647 (p=0.0013) vs A(start) = **0.0 (p=1.0)** — the same 17 events, one
   constant referent shift, complete verdict flip. The timing leg is also
   W-fragile: A = 0.917 (W=8) → 0.647 (W=12) → 0.074 (W=16).
4. **Ratio-of-means vs mean-of-ratios critique: direction confirmed, magnitude
   trivial, and the real driver is W, not the ordering.** R(t) = RMS_R(o)/mean_R(d)
   median 1.89 vs mean_R(o/d) median 2.18 (~15% Jensen gap, as predicted) —
   but both are clear-side; the in-band static 0.57–0.61 is unreachable by
   the W=12 estimator (only 1.7–5.6% of windows enter the band). The static
   number is a *different estimator* (full-stratum displacement), and the
   ladder certifies the W→∞ limit reproduces it **by construction**.
5. **Two descriptive claims in the team's own documents are wrong against the
   code's own numbers** (details §6): the RUN doc's "transition spike d ≈
   0.6–0.9 vs stable 0.15–0.3" (actual W=12 straddle mean 0.48, median 0.30,
   only 25% of straddle windows reach ≥0.6), and the probabilistic doc's
   `δ₁ ≈ 0.6–0.9` dilution arithmetic (the straddle windows' spike is a tail
   phenomenon, not the typical value).

**Overall:** the filed E2/E3 verdict is VOID (§5.3: 17 < 20 crossings) and
stays void under the red-team reading — but for a sharper reason than "too
few events": **A is underpowered AND referent-flipping, P is saturated and
common-shift-blind, D is split, S is collider-adjacent.** The foundations
should build *around* these facts, not over them.

---

## 1. Task 1 — Verify the P crisis empirically

### 1.1 Reproduction (exact)

Called `leg_P(win, m, sd, W=12, SIGNAL_W2)` directly on the wave-2 canonical
Measurement (same construction as `analyze_wave`):

| quantity | my run | filed (results.json) |
|---|---|---|
| P_trans | 0.994022 | 0.994022 |
| P_rest | 0.993531 | 0.993531 |
| kill bar 0.5×P_rest | 0.496766 | 0.497 (rounded) |
| holds_at_half | True | True |
| mechanism_kill | False | False |

The 0.9940 is real, reproducible, and (with the bootstrap CI `[0.991, 0.996]`
vs half-rest CI `[0.495, 0.498]`) the mechanism-kill branch `trans_hi <
half_lo` can never fire — the required gap is ~0.5 on a CI of width 0.005.

### 1.2 Is the common-shift critique correct as implemented?

**The statistic:** `leg_P::offset_vecs` (premise_band_movers.py:421-443)
computes `o_R = (M_R − b̄)/sd` where `b̄` = mean over present readers — offsets
are **already deviations from the roster mean**. `cos_pair` (:445-463)
stacks the 4-dial reliable subspace (mood, volume, earnestness, presence;
RIDX at :56-57) over common readers and takes the cosine.

**Structural fact (doc right):** if the roster moved rigidly in reading
space (`m_R → m_R + Δ` for all R), then `o_R` is exactly invariant → cos =
1.0 automatically. A cosine of 0.994 is exactly what a near-rigid shift
produces, and is observationally equivalent to "idiosyncrasy survives."

**The doc's §4 fix is a no-op (doc wrong).** "Subtract the roster's common
step (mean of Δm over readers) before computing offsets": since `b̄` is
already the roster mean, de-shifting all post-window means by any constant
`ĉ` gives `o'_R(post) = (M_R − ĉ) − (b̄' − ĉ) = o_R(post)`. I applied it
literally on all 8 transitions: `cos_guard` reproduces `cos` exactly
(0.9964→0.9964, 0.9971→0.9971, 0.9800→0.9800, …). **The one-line guard the
probabilistic doc proposes changes nothing.** The fix must change the
statistic, not the centering.

### 1.3 The measured mechanism (why 0.994)

Decompose each step's per-reader displacement `δ_R = M_R(post) − M_R(pre)`
into common `ĉ = mean_R δ_R` and residual `r_R = δ_R − ĉ` (note: `Δo_R =
r_R` exactly, since offsets are mean-centered). In the **reliable 4-dial
subspace** (what P actually measures), in corpus-sd units:

| transition | cos4 | RMS ĉ | RMS r_R | q = RMS r_R / RMS o_pre |
|---|---|---|---|---|
| T1@20 | 0.9964 | 0.182 | 0.113 | 0.092 |
| T3@20 | 0.9971 | 0.140 | 0.102 | 0.079 |
| T4a@12 | 0.9800 | 0.311 | 0.230 | 0.204 |
| T4a@20 | 0.9918 | 0.359 | 0.178 | 0.158 |
| T4b@20 | 0.9953 | 0.183 | 0.134 | 0.111 |
| T4b@28 | 0.9952 | 0.222 | 0.154 | 0.133 |
| T5@24 | 0.9931 | 0.225 | 0.172 | 0.138 |
| T5c@24 | 0.9940 | 0.199 | 0.160 | 0.135 |
| **pooled** | | | | **0.132** |

**At rest** (adjacent W-window pair inside a stratum, same construction):
T4a-cynical q=0.116, T5-pre q=0.133, T5c-pre q=0.139 → **q_rest ≈ q_trans ≈
0.13**. The step's idiosyncratic (non-common) component is *the same size as
ordinary rest motion*. The cosine is saturated at rest AND across steps for
the same reason: offset motion is ~13% of offset scale everywhere.

**Per-dial residual ratios** (RMS r_R / RMS o_R across steps): mood **0.095**
(the dominant dial, offset scale 1.08), volume 0.154, earnestness 0.168,
presence 0.530, joke_landing 0.214, panic 0.547, **cynicism 1.650**. The
step is nearly rigid exactly on mood — the dial that dominates the stacked
cosine — and the non-common motion concentrates in cynicism, which the
reliable subspace **excludes**. Also: mean per-reader `cos(r_R, o_R(pre)) =
−0.50` (residual motion anti-aligns with pre-offset — partial mean-reversion
toward the roster mean), which further protects the cosine.

**One-line version:** P = 0.994 not because "idiosyncrasy survives the
step" but because *the step doesn't touch idiosyncrasy* — in the subspace P
measures, it is a common translation at the rest-noise level.

### 1.4 What the guard actually requires in code

Not a centering change (`offset_vecs` is already centered). Options that
would actually bind, in increasing order of invasiveness:

1. **Report q, not just cos** (2 lines): `q = RMS(r_R)/RMS(o_pre)` per
   transition, with the honest rule `q_trans ≫ q_rest` (step moves offsets
   more than rest) → kill; `q_trans ≈ q_rest` → **uninformative** (this
   corpus). The current rule reads cos≈1 as "holds"; q says "nothing to
   measure."
2. **Permutation null on the residual**: shuffle reader labels of `o_post`
   within each transition → null cos distribution; observed 0.994 sits
   beyond any re-draw null, which is *still* not evidence — it only rules
   out re-draw, not rigidity. Needs the q-condition above to separate.
3. **Engineer non-common steps**: the corpus's steps are mood-rigid by
   construction of the strata flips; P can only be exercised by a step that
   moves readers differently (archetype-specific or per-reader rotations).
   This is a corpus-design fix, not a code fix.

---

## 2. Task 2 — Test count and E2/E3 coverage

- `python -m pytest -q` in the repo root: **277 passed in 12.03s**, 0
  failures. Matches pyproject's declared "277 tests across 25 files".
- **Zero tests exist for the E2/E3 pipeline.** Grep of `tests/` for
  `premise|band_mover|night_windows|leg_P|leg_A|e2_instrument` hits only a
  dungeon-quest "premise" string in `test_plato_rpg.py` (false positive).
  No `test_premise_band_movers.py`, no `test_e2_*.py`, no `test_premise_measurement.py`.
- `pyproject.toml` sets `testpaths = ["tests"]` and explicitly excludes
  `scripts/*_test.py` (e.g. `tap_test_models.py`) from collection.
- The pipeline's only automated checks are asserts inside `main()`:
  `assert_replay_matches_log` (6 sampled reader-nights), the stage-2 wave
  gate (byte-identical re-run), and the continuity ladder ±0.10 gate — all
  execute only when the script runs, never under pytest. **A regression in
  `night_windows` or `leg_P` would be caught by nothing.**

---

## 3. Task 3 — Window referent and ratio critique, with real numbers

### 3.1 Window-referent (CENTER_OFF = 5.5, TOL = 3; :77, :271, :297, :357)

Same events, one referent shift:

| referent | A (wave-2) | p | reading |
|---|---|---|---|
| window-center (registered) | 0.647 | 0.0013 | "fires" |
| window-start (sensitivity) | **0.0** | **1.0** | "never fires" |
| up-crossing mirror (center) | 0.233 | 0.2218 | no signal |

The `_shift_table` docstring (line ~265) is honest about why: the causal
event of a transition dip is the half-split at window center, and W/2 = 6 >
TOL = 3 makes the start referent arithmetically blind. **That honesty does
not rescue the statistic: it means the entire timing verdict is carried by
an untested positional convention** (CENTER_OFF = (W−1)/2 is a hard-coded
choice, not derived). Under the riverbed's own "generative over descriptive"
rule, the referent must be derived from detection latency, not chosen so the
test can fire.

W-fragility of the same statistic (filed sensitivities): A = 0.917 (W=8) →
0.647 (W=12) → **0.074 (W=16)**. The timing leg is a function of the window
grid; none of the three is "right" under the current foundation.

### 3.2 Ratio-of-means vs mean-of-ratios (W=12, wave-2, all signal nights)

- `R(t) = RMS_R o_R / mean_R d_R` (the code's population score, :232):
  median **1.89**, mean 1.93, **5.6%** of windows in [0.3, 0.6].
- `mean_R ρ_R(t)` (mean of per-reader ratios): median **2.18**, mean 2.24,
  **1.7%** in band.
- Gap 1.89 vs 2.18: Jensen's inequality, direction exactly as the
  probabilistic doc predicts (E[o/d] ≥ E[o]/E[d]), magnitude ~15% — real but
  **verdict-irrelevant**: both are clear-side, and neither approaches the
  band.
- The in-band static number (0.5737 pooled / 0.6139 exact anchor) is
  unreachable by the W=12 estimator. The driver is **not** the ordering of
  averages — it is the **estimator scale**: static drift = full-stratum-to-
  full-stratum displacement (0.79–0.83), while W=12 `d_R` = 6+6-speak
  split-half displacement (median 0.25, phase means 0.32–0.36). The ladder's
  W=∞ rung is the only W where split-half *is* the stratum split, so the
  ladder certifies the W→∞ estimator reproduces the number the filed
  estimator produces — a tautology gate (probabilistic doc Dissent 1,
  confirmed with numbers).

### 3.3 The straddle-window d profile (checking the RUN doc's descriptive §3)

The RUN doc claims "transition drift spike (split-half d ≈ 0.6–0.9) vs
stable ≈ 0.15–0.3". Actual W=12 numbers from the code's own arrays:

- straddle windows (half-split crosses a boundary): mean d = **0.48**,
  median **0.30**; only **24.7%** reach ≥ 0.6, only 11.6% sit in [0.6, 0.9].
- far/stable windows: mean 0.32 (45% in [0.15, 0.3] — the floor claim is
  right).
- straddle/far ratio 1.50×, not the 2–6× the descriptive §3 implies.
- ρ straddle median 1.37 vs far 1.58 — the "transition dip ≈ 1.4–1.8" is
  approximately right; 16.7% of straddle ρ enters the band (vs 1.7%
  overall), which is where the 17 counted crossings come from.

The spike is a **tail phenomenon** (hard transitions T2/T3/T8 show 0.57–1.02
peaks; entry-family transitions T5/T5c/T4a barely move), not the typical
straddle window. The probabilistic doc's `δ₁ ≈ 0.6–0.9` dilution arithmetic
inherits this overstatement.

---

## 4. Task 4 — Where the theory breaks: concrete file:line seams

1. **P is common-shift-blind and its kill branch can never fire.**
   `scripts/premise_band_movers.py:412-524` (`leg_P`): offsets mean-centered
   at :421-443, cos on the 4-dial subspace at :445-463, kill rule
   `P_trans < 0.5*P_rest` with CI separation at :513-518. With
   P_trans≈P_rest≈0.994 and CI widths ~0.005, `trans_hi < half_lo` requires
   a 0.5-scale gap the data cannot produce. The registered "holds" branch
   (:514, :925 `pro_premise`) reads saturation as survival. **This is the
   single most consequential seam: it can certify a premise claim from a
   statistic that is constant under the corpus's only realized null.**
2. **The field-edge gate is κ-blind.** `elephant/vmf.py:184-189`: `real` is
   `d_mu > 2·max(SE)` — pure chordal mean-direction change; a pure rigidity
   step (κ-only) is certified "no real drift." The KL combine is 2 lines
   from `A7` (probabilistic doc §1.3, agreed).
3. **Mixed norms + ratio-of-means inside ρ.**
   `scripts/premise_band_movers.py:224-232`: numerator `ov` is dial-RMS
   (:224), denominator `dv` Euclidean 7-norm (:225-226), population score
   `Rt = RMS(o)/mean(d)` (:232). The 0.3/0.6 band is calibrated on this
   mixed convention (docstring at :177-187 admits the Euclidean-numerator
   alternative would inflate the score ~√7 and break the ladder) — the band
   is a dimensional artifact of the norm choice.
4. **The continuity ladder is a same-estimator tautology.**
   `scripts/premise_band_movers.py:687-688` (`R_pooled_fullnight = RMS(o)/
   mean(d)` at W=∞) gated at :1039-1044 against the filed channel within
   ±0.10. Filed: 0.5737 vs 0.6139 → OK, by design. The rung reproduces the
   number because the full-night split-half is the strata split.
5. **Hysteresis and window size are hard-coded, not derived.**
   `scripts/premise_band_movers.py:71-77`: `W_PRIMARY=12`, `HYST_MARGIN=0.05`,
   `HYST_HOLD=3`, `EDGE_LO/HI=0.3/0.6`, `CENTER_OFF=5.5` — five constants
   that jointly determine the verdict (A flips across W; referent flips A;
   the band is unreachable at W=12), none derived from detection theory.
6. **S is structurally selection-prone.** `leg_S` at :549-647 demeans within
   reader (FE) but `x` (night warmth) is a night-level constant assigned to
   all readers on that night; roster × warmth selection in `FIELD_NIGHTS_W2`
   (e2_instrument.py:132-153) is therefore absorbed into `x` and cannot be
   removed by the FE. The knife-edge is visible in the filed split: primary
   CI contains 0, class-residual CI excludes 0 (wave-2 [0.119, 1.473]) —
   disclosed, unread.

---

## 5. Task 5 — The common-shift guard as a mini-experiment (done, read-only)

I ran the minimal experiment already in §1.3 (scripts in `/tmp`, nothing
written to the repo): for every signal transition, decompose the step into
common + residual motion, measure `q = RMS(r_R)/RMS(o_pre)` in the P
statistic's own subspace, compare against the same quantity at rest.

**Results:** q_trans(pooled) = 0.132 vs q_rest = 0.116–0.139.

**What a minimal fix looks like (code shape, ~5 lines):** in `leg_P`,
replace the `holds_at_half`/`mechanism_kill` pair with a q-rule:

```
q_trans = RMS_R ‖(M_R^post − M_R^pre) − mean_R(ΔM)‖_rel / RMS_R ‖o_R^pre‖_rel
q_rest  = same over adjacent in-stratum window pairs
kill iff q_trans ≫ q_rest (bootstrap-separated)
holds  iff q_trans ≪ 1 with CI (offsets genuinely survive a REAL step)
else   uninformative
```

**What it changes about the E2/E3 verdicts:** the current P branch
"holds_at_half: True" (which feeds `pro_premise` at :925 and the
identity-propagation booking at :1159-1166, which does **not** fire here)
would become **UNINFORMATIVE**: the step's idiosyncratic component is at the
rest level, so P_trans ≈ 1 is a statement about the step's rigidity, not
about persistence. The composite verdict is already VOID (17 < 20); under
the q-rule it stays void but for the right reason — A underpowered, P
unfalsifiable. The RUN doc's §6 route to "SURVIVED (capped)" (fix the
crossing count, keep P's 0.994 as evidence) is **closed**: P cannot carry
that weight.

---

## 6. Corrections to the team's positions (so the foundation doesn't inherit them)

1. **Probabilistic doc §4 guard is a no-op** (see §1.2). The single most
   important change to the thesis's fix list: the guard must be a *q-rule /
   re-draw null on the residual*, not a centering change.
2. **Probabilistic doc §2.2's δ numbers are overstated.** Transition spikes
   are a tail phenomenon: straddle-window d mean 0.48 (median 0.30), 25%
   over 0.6; the "dilution of δ₁≈0.75" arithmetic inherits the RUN doc's
   descriptive overstatement. The ratio-of-means critique survives in
   direction (1.89 vs 2.18) but its *magnitude* is small; the dominant
   artifact is estimator scale (stratum-mean vs split-half displacement).
3. **P_rest = 0.9935 is not "persistence at rest" either** — it is the same
   saturation: offset motion is ~13% of offset scale at rest. Both P numbers
   measure the estimator's noise floor, not a property of steps.
4. **Algebraic doc's "thin category" reading of `edge` is safe** (chord
   composition is what the code does) but inherits seam 2: the category is
   over the *direction* factor only; κ moves are invisible morphisms.
5. **The RUN doc's own numbers contradict its narrative**: it reports
   d_transition_phase 0.363 vs d_stable 0.317 in results.json while the
   prose claims 0.6–0.9 vs 0.15–0.3. Future foundations should cite the
   code's phase means.

---

## 7. Bottom line for the Captain

The code is honest — the void is declared, the referent sensitivity is
carried, the norm deviation is disclosed, the replay asserts hold. But the
registered statistics are not yet *measurements* of the premise: A is a
function of a positional convention and the window grid; P is saturated
under the corpus's only realized null (rigid-on-mood steps); the in-band
static ratio is produced by a W→∞ estimator that the W=12 instrument never
observes locally. The deeper foundation should make all three *derivable*
(edge as KL divergence, ρ as phase-conditional ratio with one norm, P as a
q-rule, referent from detection latency) — and, critically, the corpus needs
steps that are **not** common translations before "idiosyncrasy survives the
step" is a testable claim at all.

---

## 8. Formula → file index (red-team additions)

| quantity | location |
|---|---|
| `P_trans/P_rest` cos on 4-dial stacked offsets | `scripts/premise_band_movers.py:412-524` (offset_vecs :421-443, cos_pair :445-463, kill rule :513-518) |
| common-shift residual `r_R = δ_R − mean(δ)` (no-op guard proof) | derived from :421-443; verified numerically |
| `q = RMS(r_R)/RMS(o_pre)`: trans 0.132, rest 0.116–0.139 (measured) | this doc §1.3 |
| per-dial residual: mood 0.095, cynicism 1.650 (measured) | this doc §1.3 |
| mixed-norm ρ / R(t) ratio-of-means | `premise_band_movers.py:224-232` |
| ladder tautology gate | `premise_band_movers.py:687-688, 1039-1044` |
| referent constants | `premise_band_movers.py:71-77, 271, 297, 357` |
| κ-blind edge gate | `elephant/vmf.py:184-189` |
| hard-coded hysteresis | `premise_band_movers.py:72-74` |
| S night-level x / selection | `premise_band_movers.py:549-647`; `e2_instrument.py:132-153` |

*No repo files modified. Analysis scripts: `/tmp/pbm_redteam.py`, `/tmp/pbm_redteam2.py`, `/tmp/pbm_redteam3.py` (kept for reproducibility; all read-only against the corpus).*
