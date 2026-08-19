# Contrast Head — Threshold Re-Registration Event

**2026-08-19 11:37 AKDT — this event precedes ALL completed training below.**

The thesis author's rule (pre-committed, non-negotiable): *a threshold
re-registration that happens after training has begun is INVALID.
Re-registration MUST precede training.* This file is that event. No
contrast-trained weights existed at this timestamp: the first text run
(`text_contrast_seed0.pt`, started ~10:50 AKDT, before this registration
existed) was killed by a gateway restart mid-write — it survives on disk as
a **0-byte artifact**, contains nothing, and is **discarded as
exploratory**. Every checkpoint reported after this file carries training
that post-dates the registration.

## Baseline provenance (measured artifacts, not doc claims)

The committed probe — `fleet-jepa-midi/checkpoints/elephant_probe.json`
(the shipped measurement of the frozen audio-JEPA v2 encoder on the
tap-trades corpus) — contains exactly:

| metric (probe key) | value |
|---|---|
| `separability.gap` (fine gap) | **0.014613** |
| `room_discrimination_plain` | 0.338984 |
| `room_discrimination_speaker_heldout` | 0.355932 |
| `chance` | 0.25 |
| corpus | 4 rooms (episode-1..4), 59 clips |

`grep -c 0.271 elephant_probe.json` → **0**. The coarse sauna/plunge gap
(speech vs music, **0.271**) exists ONLY as a claimed result in design
docs — `elephant-sense-v3-design.md` §8 (lines 348/352/366) and
`jepa-cross-pollination-map.md` (lines 18/58/92). It is **not** in the
measured artifact. Structural cause: the music corpus is FLAT —
`/home/eileen/projects/ai-writings/music` holds 231 top-level mp3s, zero
`episode-*` directories — so the shipped probe's room discovery (which
requires `episode-*` subdirectories) cannot even find the music corpus,
let alone measure a speech-vs-music gap from it. **0.271 is a doc-claim
without a measured artifact.**

## Fine leg — thresholds RE-AFFIRMED (it reproduces)

The fine leg reproduces in the committed probe (0.0146 / 0.339 / 0.356,
above), and the harness re-measures the frozen tap-only baseline before
every training run and prints the fidelity check against those numbers.
Registered thresholds, re-affirmed unchanged:

* **fine gap: 0.015 → ≥ 0.10** — on the tap-trades control (4 rooms,
  same cast — the designed killer control), **all seeds** (min over seeds
  gates).
* **speaker-heldout discrimination ≥ 0.50** (chance 0.25) — mean over
  seeds.
* three consecutive runs (seeds 0/1/2); within-room spread preserved
  (hinge, spread ≥ 0.9× frozen baseline); noise-margin read: fine gap >
  0.05 + 2σ_cross.

## Coarse leg — baseline RE-REGISTERED (0.271 retired)

The 0.271 number is retired as a baseline: it cannot be re-measured and
was never in the artifact. The coarse gap is **re-defined on the structure
that exists**, effective this timestamp:

* **Definition (audio tier):** `cross_group_gap` — mean within-A
  same-room cosine minus mean A-vs-B cosine, where **A = the four tap
  speech rooms** (`tap-1..4`, the probe's own corpus) and **B = the music
  cold-plunge rooms** (`music-a/b/c`: the flat music corpus's mp3s ≥ 50 KB,
  sorted by filename, split deterministically into 3 rooms). Computed on
  the audio tier, where the speech-vs-modality axis exists.
* **Frozen baseline (measured pre-registration, committed with this
  event):** coarse gap = **0.095507** (`checkpoints/contrast/
  audio_frozen_baseline.json`, 230 clips / 15 rooms).
* **Registered requirement:** the coarse gap must NOT degrade under fine
  contrast training — coarse ≥ frozen baseline (0.0955) per seed. The
  coarse axis may not be traded away to buy the fine gap.

## Registered run structure (both tiers)

Seeds (0, 1, 2); objective = multi-positive InfoNCE τ=0.15 (fixed),
anchor=clip, batch = all clips of 2–3 rooms; anti-collapse spread hinge
(slack 0.9, λ=5.0, targets = frozen baseline spread, frozen per run).
Text tier = the `learned.py` TextEncoder trunk (`learned_dials.pt`),
loaded and fine-tuned, vocab frozen. Audio tier = the fleet-jepa-midi
ConvEncoder v2 (`audio_jepa_v2.pt`), loaded and fine-tuned, BN stats
frozen. Fusion (v3 §5): per-modality L2-norm + p95 distance-distribution
matching, late-fusion projector, modality dropout 0.3.

**Verdict gates (binding):** fine ≥ 0.10 on all seeds AND heldout ≥ 0.50
AND spread preserved AND coarse ≥ 0.0955. Anything else = FAIL, reported
plainly.
