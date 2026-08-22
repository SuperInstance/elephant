# Riverbed G6 run — the three-part decoupling, calibrated (S3 gate)

**2026-08-21. This run note supersedes the numeric rows of the G6
registration addendum's re-verification table
(`memory/wave3-registration-addendum-g6-2026-08-21.md`, written against
commit `9df2581`); the addendum itself is annotated, never edited.**
Spec: `memory/research-g6-noise-2026-08-21.md` (§2 the three-part fix,
§3 the calibration targets). Registered seed 20260821, registered
9-family instrument design, all numbers through the registered
instruments (adapter → e2_instrument Measurement / vmf fits / the
generator's own noise-aware expected path / `riverbed_wave_gate`).

The three G6 parts landed in `9df2581` (with the charisma-pull fiber,
the clamped fit channel, the noise-aware expected path, and this
session's persona-anchor/author-schedule work merged in). **This run's
commit is the calibration completion**: the E_SEG single-direction
contrast is replaced by the FIELD-MEASURED era-position vectors, the
latent warmth layer is scaled to engine parity, and the noise constants
are re-tuned — moving four of the five calibration statistics onto their
field targets simultaneously (the earlier state hit them only one at a
time; see the "before" column).

## Per-part changes (this commit, `scripts/riverbed_generator.py`)

| part | what | where |
|---|---|---|
| era geometry | `Z_WARM_DEV` / `Z_FLIP` / `Z_ENTRY`: the field's per-dial stratum-mean vectors (measured on data/nights, 2026-08-21) replace `baseline·BASELINE_Z + seg·E_SEG` — the flip moves cynicism rail-to-rail (+1.70 z) with presence (+.53), mood ~flat (the field's own stratum-mean sd on mood is 0.052 raw); the harness's normalized E_SEG (kept for reference) missed the per-dial mix | constants 252–277, `seg_schedule` 516, `room_path` 583 |
| warmth scale | `WARMTH_SCALE = 0.45`: the direct sampler stacks a direction-only warmth channel on the era anchor; the field has no second layer and its warm-era ‖z‖ ≈ 1.46 is reproduced at 0.45 (full scale inflates ‖z‖ to ~1.9 → estimator-facing noise σ/‖z‖ shrinks → logged κ inflates, and the doubled mood swings crowd out the era channels) | 286, `room_path` 597, expected path 688 |
| emission levels | `EMISSION_BASELINE = 1.0` (the field's own per-dial mean positions — 0.35 starves the charisma×era-swing between-reader channel, sinking realized ICC to 0.53–0.67); `DIAL_NOISE = 2.2` (the free dials' within-era scatter at the T1-warm anatomy level, ~2× the pooled within-stratum shape — the pooled SIGMA_DIAL averages rail-pinned eras where the clamp cuts the noise); `NOISE_ERA_EXP = 0.25` (era scaling of the noise by κ(t) — the level needs σ, the logged-κ RATIO needs the era scaling) | 218–233, 279 |
| tests | `TestG6NoiseModel` rewritten to the era-vector geometry (flip vector, field content assertions), the calibration snapshot re-banded to the achieved-and-honest values (warmth residual band TIGHTENED 0.16 → 0.10 — a genuine pass now), the ICC test re-anchored on the field's own value | `tests/test_riverbed_generator.py` 499–700, 227 |

Earlier-session work already in `9df2581` (context): the engine
charisma-pull fiber with bit-exact replay parity (`eff = clamp(raw +
s·(vibe−raw))`, generator 933–934), the unnormalized emission + clamped
fit channel (`obs_fit`, 621), the noise-aware expected path
(`expected_logged_warmth_path`, 646), field-magnitude persona anchors
(`ANCHOR_SCALE = 1.0`, 123), and the balanced author bag (871).

## Calibration table — registered seed 20260821 (instrument wave, 9 nights)

| statistic | field target | `9df2581` (before) | **this run** | verdict |
|---|---|---|---|---|
| corpus_sd | 0.2367 (gate: own numbers, never handed over) | 0.2568 (+0.02) | **0.2426** | ✅ in [0.22, 0.27]; within 0.006 of field |
| stable-d (W=12 split-half, own-sd) | floor 0.29, band 0.26–0.40 (canonical 0.261 / actual 0.376) | 0.414 (over) | **0.365** (median 0.333) | ✅ in band; sits between the field's canonical and actual values |
| logged κ warm / cold | ~21–26 / ~11–15 (ratio 2.18) | 37.4 / 18.7 (ratio 2.00) | **45.9 / 24.5** (ratio 1.87) | ⚠️ levels carry a disclosed ×~1.9 offset (was ×8 pre-G6: 200/90); ratio and the Δlogκ responses below re-verified |
| flip Δlogκ | −0.746 (pooled flip windows) | −0.53 | **−0.435** | ⚠️ right sign, under-responding — re-run the κ-check protocol at S2 |
| entry Δlogκ | −0.320, band [−0.418, −0.205] | — | **−0.302** | ✅ IN the registered band |
| warmth-SNR (strata-mean vs noise-aware expected path, max abs) | ±0.10 (gate band) | 0.141 (breach, band fudged to 0.16) | **0.056** | ✅ GENUINE pass — every stratum ≤ 0.056, no disclosed breach needed; gate check 5 ALL PASS (drops within ±0.10, final levels within ±(0.10+σ_fit)) |
| realized ICC (instrument, actual presence) | 0.8444 field (canonical 0.7714) — measured through this exact path | 0.627 (test band re-set to [0.60, 0.80], which does not even contain the field value) | **0.815** | ✅ within 0.03 of field; test band [0.78, 0.88] anchored ON the field value |
| ICC (noise branch) | collapses < 0.667 (the registered prediction) | 0.228 | **collapses** (test green) | ✅ prediction holds |
| wave gate | 9/9 checks | — | **ALL PASS** | ✅ |

Per-dial ICC (this run vs field): mood .98/.98, volume .93/.95,
earnestness .90/.92, cynicism .86/.73, joke .85/.76, presence .82/.90,
panic .37/.68 — five of seven within ±.04 of the field, cynicism/joke
EXCEED it; **panic is the one residual dial** (the cast's personas share
a single panic vibe value and a near-zero lens — its field ICC is a
ratio of near-zero variances; disclosed, not manufactured).

corpus_sd decomposition (raw RMS): within-stratum 0.179, stratum
contrast 0.167 (√(w²+c²) = 0.246) vs field 0.134/0.196 — the contrast
now carries the measured era vectors (cynicism pooled z-sd 0.94 vs field
0.969, presence 0.38 vs 0.403); the within component rides at the
T1-warm anatomy level (see DIAL_NOISE above).

## Why the era vectors (the structural finding of this run)

The field's stratum-mean geometry is NOT a single direction: the flip
moves cynicism rail-to-rail (+1.70 z) while the mood dial — the WARM
direction's heaviest loading — barely moves (stratum-mean sd 0.052 raw).
The old construction expressed the flip in WARM space (mood swings) and
then E_SEG space (one normalized direction); both miss the per-dial mix.
The measured vectors deliver the contrast at field scale AND feed the
charisma×era-swing between-reader channel (the engine fiber's reading =
raw + s·(vibe − raw): dials that actually swing give heterogeneous-s
readers between-reader variance — volume ICC .65 → .93, presence .36 →
.82 when the anchor sits at the field's own dial positions).

## Honest residuals (disclosed for the S2 re-verification)

1. **logged κ levels 45.9/24.5 vs field ~24/11** (×~1.9, down from the
   pre-G6 ×8). The σ/κ trade is a curve, not a knob: the κ level wants
   σ ≳ 2.6, at which corpus_sd overshoots the band top (0.257+) and
   stable-d crosses 0.40. The landed point favors corpus_sd + warmth
   residual + ICC + the entry response over the warm-κ level, per the
   G6 research §2.4 ordering. The flip Δlogκ (−0.435 vs −0.746)
   under-responds for the same reason — re-run the κ-check protocol
   (per-strata logged fits, the registered object) before S3 generates.
2. **Panic-dial ICC** .37 vs field .68 — structurally starved persona
   channel (see above), one-seventh of the aggregate.
3. The registration's old ICC bracket [0.85, 0.96] is superseded by the
   field-anchored band [0.78, 0.88] (the bracket was the vMF-fiber
   calibration; the engine-faithful fiber's target is the field's own
   0.8444, which this path reproduces exactly on the filed corpus).

## Verification

- Full suite: **340 passed, 0 failed** (includes the 10 TestG6NoiseModel
  mechanism tests: era-vector geometry, unnormalized emission with live
  noise, per-dial heterogeneous shape with era scaling, replay parity on
  every reader incl. the staged entrant, branch-in-vibe-start, the
  registered-seed calibration snapshot, noise-branch ICC collapse).
- `riverbed_generator.py --self-test`: ALL 15 checks pass.
- `riverbed_wave_gate.py` on the registered instrument wave: ALL PASS.

## Concurrent-session note

This task ran in parallel with a sibling session that landed the G6
mechanisms as `9df2581` (merging this session's anchor/author-schedule
work in, as its addendum notes). This commit is the calibration
completion on top: era-position geometry, warmth scale, level
re-tuning, honest re-banding, and this run note. The G6 addendum's §2
numeric rows are superseded by the table above; its sequencing section
(S3 precondition) still holds — with the κ-check protocol re-run as the
one open item before S3 generates.
