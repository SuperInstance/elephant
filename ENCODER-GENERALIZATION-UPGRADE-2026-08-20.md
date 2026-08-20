# Encoder Generalization Upgrade — the retrieval half (2026-08-20)

**Date:** 2026-08-20 (America/Anchorage), 14:20–15:25 AKDT
**Machine:** WSL2 host `eileen`, CPU-only (`CUDA_VISIBLE_DEVICES=""` before any
torch import; torch 2.13.0 CPU, numpy 2.4.6; 24 cores, OMP 4 threads/process).
**Status:** COMPLETE — all five runs + postprocess done.
**Deliverables:** `scripts/encoder_generalization_upgrade.py`,
`scripts/encoder_generalization_postprocess.py`, this report, artifacts in
`checkpoints/contrast/upgrade/<mode>/`. **No git commit** (per dispatch).

## Why this run exists

The honest held-out re-test (2026-08-19, `CONTRAST-HELDOUT-ROOM-2026-08-19.md`)
showed the text contrast head memorizes: trained on 17 rooms (tap-night-1/2 +
15 non-tap), held-out fine gaps on UNSEEN tap-night-3/4 were
**0.0713 / 0.1073 / 0.0295** vs the registered 0.05 binary noise floor —
seed 2 below floor, so room identity is NOT demonstrated for unseen nights at
the 3-consecutive-seeds standard (mean 0.0694). The dissertation's open
question #2 (`research/topic.md`) registers the upgrades: **more training
nights, a second/different split, a generalizing (contrastive) objective**
(details in `RESEARCH-NOTE-MEMORIZATION-GEOMETRY-2026-08-19.md`: positives as
independent views of the same identity — token-dropout or split-half views —
so point-attractors become smooth basins).

## Method (this run)

One self-contained script, `scripts/encoder_generalization_upgrade.py`, five
modes, ALL against the same frozen v2 trunk (`learned_dials.pt`), the same
clip definition (W=8 non-overlapping), the same loss family (multi-positive
InfoNCE τ=0.15 + spread hinge vs the frozen baseline), the same schedule
(200×60 batches, LR 1e-4 cosine), the same 3 seeds (0/1/2), the same
held-out metric (fine gap = same-room minus cross-room cosine, floor 0.05,
PASS = 3/3 — the registered rule). A postprocess script rebuilds the merged
`results.json` per mode from the saved checkpoints (parallel seed processes
each wrote their own seed file; identical numbers, verified).

| mode | train (rooms / clips) | holdout (never trained on) | objective | registered attempt |
|------|----------------------|----------------------------|-----------|--------------------|
| `baseline` | 17 / 1097 | tap-night-3 + tap-night-4 (18) | plain | reproduction of the committed FAIL |
| `more-nights` | 19 / 1111 | tap-night-3 + tap-night-4 (18) | plain + 2 extra same-cast training nights | (a) more training nights |
| `second-split` | 17 / 1105 | tap-night-1 + tap-night-2 (10) | plain | (b) swapped/different split |
| `views` | 17 / 1097 | tap-night-3 + tap-night-4 (18) | split-half views | (c1) generalizing objective |
| `dropout` | 17 / 1097 | tap-night-3 + tap-night-4 (18) | token-dropout views (p=0.3) | (c2) generalizing objective |

Frozen-trunk anchors on the held-out clips (the honest reference): nights 3+4
gap **0.0019**, disc 0.611; nights 1+2 gap **0.0378**, disc 0.900.
Frozen-baseline train-corpus gap 0.0151 (committed 0.0148/0.0151 — consistent).

## (0) Baseline reproduction — ✅ CONFIRMED (the committed FAIL reproduces)

| seed | TRAIN gap | HELD-OUT gap | > 0.05? | disc | sp-heldout | venue-disc |
|------|-----------|--------------|---------|------|------------|------------|
| 0 | 0.4810 | **0.0710** | yes | 0.556 | 0.611 | 0.556 |
| 1 | 0.4407 | **0.1023** | yes | 0.500 | 0.611 | 0.889 |
| 2 | 0.4366 | **0.0309** | **no** | 0.611 | 0.611 | 0.667 |
| mean | 0.4528 | **0.0681** | 2/3 | 0.556 | 0.611 | 0.704 |

Committed reference: 0.0713/0.1073/0.0295, mean 0.0694, seed 2 FAIL.
**Reproduced to within 0.001–0.005 per seed; seed 2 again below the floor.**
Held-out margin over the frozen trunk: +0.066.

## (a) More training nights (`more-nights`) — ✅ 3/3 PASS (strongest result)

**Structural note on the first attempt (3 seeds run, discarded):** holding
out ONE night only makes the fine-gap metric UNDEFINED (a single held-out
room has no cross-room pairs → gap = nan; kNN disc was trivially 1.0).
**Correction:** held-out set kept EXACTLY the baseline's (nights 3+4, 18
clips) and two genuinely NEW same-cast TRAINING nights were added from the
unused 2026-08-16 tap-trades material: `tap-questions` (questions/, 2 clips)
and `tap-sequels` (sequels/ + sequels-night2/, 12 clips). The ONLY change vs
baseline: +2 training nights of the same cast (19 rooms / 1111 clips).

| seed | TRAIN gap | HELD-OUT gap | > 0.05? | disc | sp-heldout | venue-disc |
|------|-----------|--------------|---------|------|------------|------------|
| 0 | 0.4688 | **0.1201** | yes | 0.722 | 0.667 | 0.611 |
| 1 | 0.4502 | **0.0742** | yes | 0.722 | 0.667 | 0.667 |
| 2 | 0.4156 | **0.1355** | yes | 0.667 | 0.667 | 0.667 |
| mean | 0.4449 | **0.1099** | **3/3** | 0.704 | 0.667 | 0.648 |

**All three seeds clear the floor, and the previously-failing seed 2 jumps
from 0.0309 → 0.1355.** Mean 0.1099 = 1.6× the baseline mean; held-out
margin over the frozen trunk +0.108 (the trunk sits at 0.0019 on these
nights, so nearly all of this separation is learned, not pre-existing).
The same schedule/objective/seeds as baseline — the ONLY difference is two
additional nights of the same cast in training. This is the registered
upgrade working as the geometry note predicted the memorization would NOT
be fixable by capacity — it was fixable by more identity data.

## (b) Second/different split (`second-split`) — ✅ 3/3 PASS (with a disclosed caveat)

Swapped split: baseline's training nights (tap-night-1 + tap-night-2, 10
clips) become the held-out nights; nights 3+4 become training nights.

| seed | TRAIN gap | HELD-OUT gap | > 0.05? | disc | sp-heldout | venue-disc |
|------|-----------|--------------|---------|------|------------|------------|
| 0 | 0.5743 | **0.0522** | yes (hairline) | 0.500 | 0.500 | 0.900 |
| 1 | 0.5467 | **0.1105** | yes | 0.500 | 0.500 | 0.900 |
| 2 | 0.5605 | **0.0603** | yes | 0.400 | 0.400 | 0.900 |
| mean | 0.5605 | **0.0743** | **3/3** | 0.467 | 0.467 | 0.900 |

**Passes the registered 3/3 bar — but the honest caveat is large:** the
frozen trunk ALREADY separates nights 1+2 (gap 0.0378, disc 0.900) — unlike
nights 3+4 (0.0019). The head's marginal contribution over the trunk is only
+0.0365 (mean), below the noise floor; seed 0 clears by a hairline (0.0522)
and kNN discrimination sits AT chance (0.40–0.50). Reading: nights 1+2 are
inherently more separable in the v2 trunk's space (they share more surface
vocabulary with the open-mic room — see the geometry note), so this PASS
partly measures a pre-existing property, not learned night-identity
generalization. It counts by the registered letter; by the honest margin
over trunk it is the weakest pass of the two.

## (c1) Generalizing objective — split-half views (`views`) — ❌ 2/3 FAIL

Positives = content-disjoint independent views of the same night (fresh
random split of each night's clips into two halves every batch; a clip's
positives = the other half of its own night ONLY). Eval on full unseen text.

| seed | TRAIN gap | HELD-OUT gap | > 0.05? | disc | sp-heldout | venue-disc |
|------|-----------|--------------|---------|------|------------|------------|
| 0 | 0.4241 | **0.0331** | **no** | 0.667 | 0.722 | 0.722 |
| 1 | 0.4012 | **0.0698** | yes | 0.611 | 0.611 | 0.722 |
| 2 | 0.3734 | **0.1028** | yes | 0.556 | 0.556 | 0.889 |
| mean | 0.3996 | **0.0686** | 2/3 | 0.611 | 0.630 | 0.778 |

**Does NOT clear the floor 3/3** (seed 0 at 0.0331). Mean 0.0686 ≈ baseline
mean 0.0681 — the objective reshuffled the failure pattern without creating
generalizing basins. TRAIN gap dropped (0.40 vs 0.45 baseline): split-half
positives weaken in-sample separation by construction (half the within-room
positives excluded) and buy nothing held-out. Margin over trunk +0.067 ≈
baseline's +0.066 — no net improvement.

## (c2) Generalizing objective — token-dropout views (`dropout`) — ❌ 0/3 FAIL (worst result)

Every clip re-masked each batch (each non-pad token dropped p=0.3);
positives = same-room masked clips. Eval on FULL unseen text.

| seed | TRAIN gap | HELD-OUT gap | > 0.05? | disc | sp-heldout | venue-disc |
|------|-----------|--------------|---------|------|------------|------------|
| 0 | 0.3589 | **−0.0095** | **no** | 0.556 | 0.556 | 0.889 |
| 1 | 0.3462 | **0.0060** | **no** | 0.444 | 0.556 | 0.889 |
| 2 | 0.3166 | **0.0175** | **no** | 0.500 | 0.556 | 0.833 |
| mean | 0.3406 | **0.0047** | 0/3 | 0.500 | 0.556 | 0.870 |

**Active harm:** token-dropout invariance training collapsed held-out
night-separability to the frozen-trunk level (mean 0.0047 vs trunk 0.0019;
seed 0 even negative). TRAIN gap also collapsed (0.34 vs 0.45 baseline).
The encoder became invariant to the very token content that distinguishes
nights — masked-variant positives erased the night signature instead of
smoothing it into a basin. The geometry note's remedy #2 fails empirically;
venue-disc (0.87) confirms only cast-level signal survives.

## Verdict — does ANY registered upgrade clear the generalization floor?

**YES — two of the three registered upgrade families clear it 3/3:**

| attempt | held-out fine gaps (3 seeds) | mean | 3/3 > 0.05? | margin over frozen trunk |
|---------|------------------------------|------|-------------|--------------------------|
| baseline (repro) | 0.0710 / 0.1023 / 0.0309 | 0.0681 | ❌ 2/3 | +0.066 |
| **(a) more training nights** | 0.1201 / 0.0742 / 0.1355 | **0.1099** | ✅ **3/3** | **+0.108** |
| **(b) second/different split** | 0.0522 / 0.1105 / 0.0603 | **0.0743** | ✅ **3/3** | +0.0365 (trunk pre-carries 0.0378) |
| (c1) split-half views | 0.0331 / 0.0698 / 0.1028 | 0.0686 | ❌ 2/3 | +0.067 |
| (c2) token-dropout views | −0.0095 / 0.0060 / 0.0175 | 0.0047 | ❌ 0/3 | +0.003 |

**The one-line honest answer: the held-out claim now PASSES for the data
upgrades — (a) more training nights is a strong, clean 3/3 pass (the
previously-failing seed 2 goes 0.0309 → 0.1355, nearly all of it learned
over a trunk that sits at 0.0019 on those nights), and (b) a different split
also passes 3/3 though weakly (the frozen trunk already carries 0.0378 of
the 0.0743 mean, and kNN discrimination is at chance). The generalizing
OBJECTIVES fail: split-half views is a 2/3 reshuffle at the baseline mean,
and token-dropout is actively destructive (0/3, collapsed to trunk level).**

Honest reading for the dissertation (open question #2):
- **More identity data, not a cleverer objective, is what fixes encoder
  generalization here.** This matches the geometry note's own diagnosis in
  an unexpected way: it claimed capacity wasn't the bottleneck and the
  training signal contained no night-invariant feature — the fix turned out
  to be *more nights of the same cast* (the signal exists, but 2 training
  nights were too few to separate it from room-level memorization).
- The two objective-based upgrades — the registered "basin" remedies — both
  fail; token-dropout actively destroys the signal. The prototype
  (`encoder_generalization_prototype.py`) showed views beating plain on a
  toy geometry; the real corpus does not reproduce that (the toy's cast
  themes were fresh-drawn per night; the real nights' vocabulary overlap is
  dominated by the open-mic venue, so view-invariance pressure erases the
  night dimension that actually needs separating).
- Weak-pass caveat on (b) must be carried into any thesis text: a 3/3 pass
  on a split where the frozen trunk already scores 0.0378 is a partial
  measurement of a pre-existing trunk property.
- The strong claim to book: **room identity IS recoverable for unseen
  nights of the same cast — with more training nights of that cast**
  (3/3, mean 0.1099, ~2.4× the failure-seed-free baseline floor). The
  retrieval tier graduates from "retrieval fact, not measurement
  instrument" to "retrieval fact with demonstrated (data-limited)
  generalization"; the instrument caveat for the dissertation's claim
  inventory stands unchanged.

## Secondary reading — venue/cast discrimination (reported, not promoted)

Top-1 training-neighbor venue (tap family = nights 1–4 + open-mic) for each
held-out clip: baseline 0.70, more-nights 0.65, second-split 0.90, views
0.78, dropout 0.87 (means). Consistent with the geometry note: cast/venue
signal transfers across every objective; night-identity is the fragile
dimension — exactly the dimension the data upgrades, and only they, repair.

## Artifacts (all additive, nothing committed touched)

- `scripts/encoder_generalization_upgrade.py` — the 5-mode runner
  (modes: baseline / more-nights / second-split / views / dropout;
  `--seeds`; CPU-only; per-seed crash-safe JSON + merged results).
- `scripts/encoder_generalization_postprocess.py` — rebuilds merged
  `results.json` from saved checkpoints (numbers verified identical to the
  training logs).
- `checkpoints/contrast/upgrade/<mode>/model_seed{0,1,2}.pt` +
  `results.json` (+ `results_seed{k}.json` for the new-code modes).
- This report. Committed checkpoints (`text_contrast_heldout_seed*.pt`,
  `text_contrast_results.json`, `text_frozen_baseline.json`, etc.) were
  never written to.

**Runtime notes:** all runs CPU-only; 3 seeds × 5 modes = 15 training runs.
Under 6–9-way CPU contention, per-seed wall time was 456–685 s (~8–11 min)
with severe oversubscription inflation (~2.7× CPU-min at 9 procs); the final
wave ran at full 4-thread utilization. Total wall ≈ 75 min of training plus
postprocessing. The first more-nights attempt (single-room holdout) is
documented as a structural invalid and superseded.
