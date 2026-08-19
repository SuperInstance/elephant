# Contrast Head — Re-Registration Addendum (dated, pre-training)

**Date:** 2026-08-19 (America/Anchorage)
**Status:** PRE-TRAINING. No post-event training has run: three prior dispatches
were killed by host kernel crashes before any training step completed. This
addendum is written and committed before the first training artifact exists —
a dated addendum, not a retroactive narration.

Amends the re-registration event (77b8aa4), which re-defined the coarse
baseline but raced `CONTRAST-RE-REGISTRATION-REQUIREMENTS.md` (3a56756) and
did not name PRIMARY vs SECONDARY. Per that requirements file, this addendum
names both explicitly:

## PRIMARY coarse definition: REGISTER AXIS

**Flat-music-vs-speech — the corpus as it structurally exists.**

Audio tier, `cross_group_gap` with A = the speech/tap rooms, B = the music
cold-plunge rooms (`music-a` / `music-b` / `music-c`), on the frozen
audio-JEPA v2 encoder's clip embeddings (probe-exact: same definitions, same
kNN, same speaker-holdout as `checkpoints/elephant_probe.json` in
fleet-jepa-midi).

- Frozen baseline (committed at 77b8aa4, `checkpoints/contrast/audio_frozen_baseline.json`):
  **coarse gap 0.0955** (within-A mean 0.5587, A×B mean 0.4631, n_cross 11016).
- This — not the probe-era 0.271 scale, and not the room axis — is the number
  the trained head's coarse leg is compared against.

## SECONDARY (reported, NOT a fallback): ROOM AXIS

**Cross-room contrast from `data/nights/` (gate-2 measurement corpus).**

Room-to-room displacement between the Tap nights' dial-field μ̂ and the TTRPG
anchor room's μ̂ (deterministic corpus, cd00bb8):

- **gap_chord 0.9409, gap_cos 0.4426** (`data/nights/summary.json`,
  `coarse_anchor`), alongside the within-room across-nights floor
  (`floor_across_nights_max_dmu` = 0.0, jitter-stable n ≈ 0.028).

Reported alongside the primary for cross-tier triangulation. It is a
dial-space field measurement at a different grain than the head's
embedding-space coarse leg; it is a reported number and does **not** substitute
for the primary if the primary disappoints.

## Rationale

The coarse leg is a head metric on the same embedding corpus the fine leg and
the deadman live on (audio tier: 230 clips, 15 rooms, which structurally
contain the music pole). The register split is observable at that tier without
crossing tiers or re-instrumenting. The room axis lives in the dial-space
nights corpus — a different measurement instrument — so it is reported as a
secondary number and never promoted.

## Fine leg re-affirmed (unchanged)

Tap-subset fine gap **0.0146 → ≥ 0.10 deadman**, speaker-heldout
discrimination **≥ 0.50** (chance 0.25), three consecutive seeded runs
(0/1/2), τ = 0.15 fixed, anti-collapse hinge vs the frozen baseline spread.
Fine and coarse are reported **separately**, never composited.
