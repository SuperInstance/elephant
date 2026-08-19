# Contrast Head — Text-Tier Training & Registered Eval (gate 3, phase 1+2)

**Date:** 2026-08-19 (America/Anchorage), 14:18–14:35 AKDT
**Machine:** WSL2 host `eileen`, CPU-only (24 cores, 15 GiB RAM). No CUDA touched
(`CUDA_VISIBLE_DEVICES=""` in all three hardened scripts, per 4ea7892).

**Pre-registration chain (untouched by this event):** 3a56756 (requirements) →
24c6eb8 (addendum: PRIMARY = REGISTER axis, room-axis secondary) → 4ea7892
(hardening). This training ran AFTER all three, on the frozen baselines
committed at 77b8aa4 (never regenerated — the recomputed-check guard fired
each run with delta 0.00e+00).

## Phase 1 — training (per-seed processes, crash-safe)

`scripts/contrast_train_text.py --seeds {0,1,2}` (the per-seed CLI block added
here for the text tier, mirroring the audio script's 4ea7892 block — identical
math). Corpus: **19 rooms, 1115 non-overlapping clips** (W=8, stride=8).
τ=0.15 multi-positive InfoNCE + anti-collapse spread hinge vs FROZEN baseline
spread; 200 epochs × 60 batches, LR 1e-4 cosine, CPU ~11 cores/seed, ~4.5 min/seed.

| seed | final batch loss | full-corpus fine gap | disc | heldout | mean_spread | checkpoint |
|------|------------------|----------------------|------|---------|-------------|------------|
| 0 | 6.594 | 0.5350 | 0.962 | 0.868 | 0.081 | `text_contrast_seed0.pt` ✅ |
| 1 | 2.977 | 0.4983 | 0.958 | 0.860 | 0.084 | `text_contrast_seed1.pt` ✅ |
| 2 | 4.763 | 0.4587 | 0.939 | 0.860 | 0.085 | `text_contrast_seed2.pt` ✅ |

Spread preservation ratios 0.87–7.3 per room (spread GREW — no collapse).

## Phase 2 — registered eval (`scripts/contrast_eval.py` eval_text + verdict)

Checkpoint-reload, re-embed, re-probe from scratch (probe-exact). Frozen
text baseline: fine 0.0148 / disc 0.753 / heldout 0.621 (tap: 0.0191 / 0.571 / 0.536).

**TAP subset (the registered fine leg) — apples-to-apples vs the probe:**

| seed | fine gap (probe 0.0146 → deadman ≥0.10) | disc | speaker-heldout (≥0.50) |
|------|------------------------------------------|------|--------------------------|
| 0 | **0.4359** ✅ | 0.964 | **0.964** ✅ |
| 1 | **0.5187** ✅ | 1.000 | **1.000** ✅ |
| 2 | **0.4806** ✅ | 1.000 | **1.000** ✅ |
| mean | **0.4784** (min 0.4359) | 0.988 | **0.988** ✅ |

All three consecutive seeded runs clear both deadmen (fine ≥ 0.10 by ≥4.4×;
heldout ≥ 0.50, chance 0.25). Full `registered_eval_text.json` in
`checkpoints/contrast/`; embeddings + clip meta saved for the fusion stage.

## Coarse — reported SEPARATELY, per the addendum

- **PRIMARY (REGISTER axis):** audio-tier metric. The audio contrast head is
  NOT yet trained (no `audio_contrast_seed*.pt`); the register-axis coarse
  comparison against the frozen audio baseline (0.0955) is deferred to the
  audio-tier training event. The text tier carries no coarse leg by design.
- **SECONDARY (room axis, dial-space, committed cd00bb8):** gap_chord 0.9409,
  gap_cos 0.4426 vs within-room floor 0.0 (jitter ≈ 0.028). Reported for
  cross-tier triangulation only; not promoted.

## Verdict

Text-tier gate-3: **PASS** (3/3 seeds, both deadmens, spread preserved).
