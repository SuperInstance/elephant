# Contrast Head — ROOM-HELDOUT Eval (the honest test, gate 3 follow-up)

**Date:** 2026-08-19 (America/Anchorage), 16:35–16:49 AKDT
**Machine:** WSL2 host `eileen`, CPU-only (`CUDA_VISIBLE_DEVICES=""`, enforced in-script before torch import). 3 seeds, ~4.5 min/seed.

## Why this run exists (the committee's finding, verified)

The registered eval's "speaker heldout" is speaker-level only
(`elephant/contrast.py::room_discrimination(holdout_speaker=True)` removes
clips sharing the query's SPEAKER key — never a ROOM). And the trained
checkpoints `text_contrast_seed{0,1,2}.pt` were trained on ALL 19 rooms
(`build_text_corpus()`), so the registered TAP-subset fine gaps
(0.4359/0.5187/0.4806) were **in-sample for the tap nights** — the
soft-narration trap. This run answers the real question: **is room identity
recoverable for NIGHTS THE MODEL NEVER SAW, same cast?**

## Method (retrain-with-exclusion, not checkpoint-reload)

`scripts/contrast_train_text.py --holdout tap-night-3,tap-night-4 --seeds 0,1,2`

The new `--holdout` flag (this commit; without it, behavior identical to the
registered training run):

- **excludes** the listed rooms from the training corpus, clips, `room_names`,
  AND the spread-hinge targets — printed before/after as verification;
- trains on the remaining rooms (identical math: τ=0.15 multi-positive
  InfoNCE + spread hinge vs FROZEN baseline spread, 200×60 batches, LR 1e-4
  cosine, CPU);
- after training each seed, **re-embeds the excluded clips with the trained
  model** and computes fine gap + room-discrimination + speaker-heldout on
  those clips ONLY;
- writes `text_contrast_heldout_seed{k}.pt` / `text_contrast_heldout_results.json`
  — the registered checkpoints, `text_contrast_results.json`, and
  `text_frozen_baseline.json` are NOT touched (frozen-baseline committed
  guard fired: committed gap 0.0148 kept, recomputed train-only check
  0.0151, file not rewritten).

### Exclusion verification (exact output)

```
[text-heldout] pre-holdout corpus: rooms=19 clips=1115
[text-heldout] train rooms=17 clips=1097  (holdout EXCLUDED: tap-night-3,tap-night-4)
[text-heldout] heldout (NEVER trained on): rooms=['tap-night-3', 'tap-night-4'] clips=18
```

Training corpus = tap-night-1, tap-night-2 + all 15 non-tap rooms
(channel-42-dawn, compass-01..07, dogs-fell-in-love, fleet-radio-004/005,
speeches, tap-open-mic, tavern-night, wesley-stream). Held out: tap-night-3
(13 clips) + tap-night-4 (5 clips) = **18 clips never trained on**.

Held-out fine gap definition (probe-exact `separability` on the 18 held-out
clips): same-room mean (within night-3 + within night-4) minus cross-room
mean (night-3 vs night-4). n_same=88 pairs, n_cross=65 pairs. Room
discrimination on the held-out set is 2-way (chance = 0.50).

**Reference anchor — the frozen v2 trunk on the same 18 held-out clips:**
fine gap **0.0019**, disc 0.611, speaker-heldout 0.611 (the trunk does NOT
separate unseen nights either; 0.611 ≈ 11/18, near chance).

## Results

| seed | TRAIN-corpus fine gap (17 rooms, in-sample) | HELD-OUT fine gap (nights 3-4, unseen) | HELD-OUT disc (chance 0.50) | HELD-OUT speaker-heldout |
|------|----------------------------------------------|----------------------------------------|------------------------------|--------------------------|
| 0 | 0.4814 | **0.0713** ✅ (> 0.05) | 0.611 | 0.611 |
| 1 | 0.4413 | **0.1073** ✅ (> 0.05) | 0.500 | 0.611 |
| 2 | 0.4349 | **0.0295** ❌ (< 0.05) | 0.611 | 0.611 |
| mean | 0.4525 | 0.0694 | 0.574 | 0.611 |

Per-seed held-out detail (exact): seed0 same=0.6851 cross=0.6137;
seed1 same=0.6365 cross=0.5292; seed2 same=0.5494 cross=0.5199. Full numbers in
`checkpoints/contrast/text_contrast_heldout_results.json`
(`heldout_eval` block per seed).

## Verdict (the one-line honest answer)

**HELD-OUT CLAIM: FAIL — fine gaps [seed0=0.0713, seed1=0.1073, seed2=0.0295];
NOT all 3 seeds exceed the 0.05 noise floor (seed 2 does not), so room
identity is NOT demonstrated for unseen nights of the same cast at the
registered 3-consecutive-seeds standard.**

The 0.10 training deadman was deliberately NOT applied to held-out data
(it was calibrated on training rooms); the held-out bar is the binary
0.05 noise floor — and it still fails 1/3 seeds.

## Reading (honest interpretation)

- The registered 0.44–0.52 "tap fine gaps" collapse to 0.03–0.11 when the
  tap nights are actually held out: **~4–15× drop**. Most of the registered
  margin was room-level memorization of nights the model trained on, exactly
  as the devil's advocate suspected.
- 2/3 seeds do clear the noise floor and all seeds sit far above the frozen
  trunk's held-out 0.0019 — the head moves held-out night separability by
  15–56× over trunk. There is a *weak, seed-unstable* night-signature
  generalization, not a reliable one.
- Held-out 2-way discrimination 0.50–0.611 vs chance 0.50 and
  speaker-heldout 0.611 (n=18): barely above chance, consistent with weak
  generalization, not with a robust room-identity channel.
- n is small (18 clips, 88/65 pairs). A seed-stable claim would need either
  more held-out nights, a different split (e.g. hold out 2 non-tap rooms
  too), or an architecture/objective that generalizes rather than memorizes.

## Artifacts (new, additive)

- `scripts/contrast_train_text.py` — `--holdout` flag (no behavior change without it)
- `checkpoints/contrast/text_contrast_heldout_seed{0,1,2}.pt`
- `checkpoints/contrast/text_contrast_heldout_results.json`

Untouched: `text_contrast_seed{0,1,2}.pt`, `text_contrast_results.json`,
`text_frozen_baseline.json`, `registered_eval_text.json`, and all registered
eval semantics in `scripts/contrast_eval.py`.

**Chain:** 3a56756 (requirements) → 24c6eb8 (addendum) → 4ea7892
(hardening) → 2052cb4 (registered training + eval, PASS in-sample) →
**this event (honest held-out re-test: FAIL 1/3 seeds on the binary bar).**
