# Slope Regression (H-reader≡room) — The Registered Decisive Test

**Dated: 2026-08-20.** The experiment the side-by-side named "the single most decisive next experiment" (e2-e3-side-by-side.md §7; registered advisor 2026-08-19, topic.md v3 convergence note): regress each reader's baseline (mean reliable-subspace reading, per reader, n_nights ≥ 3) on the measured warmth of the rooms they actually visited. Slope ≈ 0 ⇒ alignment (the baseline is a reader-specific instrument constant carrying information the room doesn't have). Slope ≈ 1 ⇒ collapse (the baseline is a slow warmth estimate; "trusted reader" = "reader who agrees with the room"; nurse-as-index dissolves into room-warmth retrieval wearing a reader's name). This document reports the run. Script: `scripts/slope_regression.py` (numpy-only, CPU, read-only against `data/`, writes `scripts/slope_regression_results.json`).

## One-line verdict

**The registered between-reader slope is INDETERMINATE at the achievable N — point estimate 1.085, 95% CI [−1.670, 3.813], landing ON the collapse anchor but spanning both anchors — while the higher-power within-reader reading of the same quantity (slope 0.133, CI [0.023, 0.253], 66 reader-nights) leans ALIGNMENT: baselines are reader-constants with a small (~13%) warmth-tracking component, so H-reader≡room is not established, and the collapse reading is excluded wherever there is power to exclude it.**

## Method (and the warmth source, documented)

**Corpus.** The E2 v:2 per-reader corpus, S-series only as registered for this test: `night-S1, S2, S3, S4a, S4b, S5, S6, S7.jsonl` (8 log files — S4a/S4b are the two cold-entry-family halves; the five schedule families of S1–S5 plus the two addendum-3 non-monotonic families S6/S7). Attendance is derived from the logs themselves (union of per-event `readers` blocks) and matches the instrument's `FIELD_NIGHTS_EXT` template exactly. 15 readers seen; **12 kept at n_nights ≥ 3** (the 6 originals — writer, poet, essayist, engineer, critic, captain — span all 8 nights; barkeep, fiddler, singer, lamplighter, blacksmith, weaver have 3 each). Dropped at n_nights = 2: cartographer, tinker, drifter.

**Room warmth — the apparatus's own thermometer, not a new one.** `warmth_vmf = WARM @ mu_hat` (elephant/vmf.py:59,167): the v0 warmth form linearized in z-space and normalized to a signed cosine, read on the vMF mean direction of the room's trailing raw windows. It is logged per speak event in `fit.warmth_vmf` and at `session_close`. Per-night room warmth = the mean of the identifiable per-event fits (the estimator returns None below NMIN = 10 windows — never a fake number; 11–37 of 20–46 events per night are identifiable, the nulls are the warm-up tail). Because the vMF fits run on raw room windows, this warmth is charisma-free: the room side of the regression is independent of the readers' displacements. Night warmths (signed cosines): S1 +0.655, S2 +0.319, S3 +0.741, S4a +0.447, S4b +0.632, S5 +0.759, S6 +0.397, S7 +0.220.

**Reader baseline — the registered schema rule, asymmetric by design.** Room snapshot keeps all 7 dials; reader baseline uses the reliable subspace only — mood, volume, earnestness, presence (E2 ICC .97/.98/.95/.91) — with panic excluded by rule (the reader-in-disguise stays out of the index), cynicism and joke_landing also out (not reliable). Per (reader, night): the E2 reading definition (`e2_instrument.logged_readings`), `reading = CENTER + g_R·(field_eff_to_reader − CENTER)` with g_R the reader's attention gain from the roster; each reading z-standardized exactly as `vmf.zvec` does; scalar = projection on the SAME warmth direction restricted to the reliable subspace and renormalized to unit norm (same signed-cosine units as warmth_vmf on both sides of the regression). Per-night value = mean over the night's events (quiescent ‖z‖ < 1e-3 skipped, mirroring `vmf.windowed`); reader baseline = mean over nights, equal weight per visited room. The regression: OLS of baseline on visited-room mean warmth, across the 12 readers; 95% CI = reader-level pairs bootstrap, B = 10,000, seed 20260820, degenerate resamples (zero x-variance) skipped and counted.

**Verdict bands (the operationalization of "slope ≈ 0 / ≈ 1", declared before the run):** alignment if the CI lies entirely in [−0.25, +0.25]; collapse if entirely in [0.75, 1.25]; otherwise INDETERMINATE with the lean reported.

## Numbers

**Per-reader table (primary treatment):**

| reader | n_nights | baseline | visited-room warmth |
|---|---|---|---|
| writer | 8 | +0.889 | +0.521 |
| poet | 8 | +0.952 | +0.521 |
| essayist | 8 | +0.660 | +0.521 |
| engineer | 8 | +0.510 | +0.521 |
| critic | 8 | +0.430 | +0.521 |
| captain | 8 | +0.847 | +0.521 |
| barkeep | 3 | +0.444 | +0.598 |
| weaver | 3 | +0.668 | +0.632 |
| blacksmith | 3 | +0.872 | +0.531 |
| fiddler | 3 | +0.306 | +0.441 |
| lamplighter | 3 | +0.744 | +0.433 |
| singer | 3 | +0.294 | +0.387 |

**Primary (registered, between-reader):** baseline = +0.079 + **1.085** × room_warmth; Pearson r = +0.313; **slope 95% CI [−1.670, 3.813]** (10,000 draws, 0 degenerate). Verdict: **INDETERMINATE (lean collapse; CI contains both 0 and 1)**. Design-power honesty: 6 of the 12 readers — the originals — share one visited-room warmth by construction, so the between-reader regression has 7 distinct x-values and its slope is carried by the 6 field draws (singer at the cool end, weaver at the warm end). This is the E2 power lesson repeating at the reader grain: no feasible N of this design adjudicates a between-reader slope when the roster's backbone attends everything.

**Robustness variants (all INDETERMINATE, all lean collapse, all CIs spanning both anchors):** reader mean-direction projection 1.342 [−2.320, 4.698]; `reader_final` median fact 1.680 [−1.493, 4.847]; pooled-events weighting 1.062 [−1.718, 3.815]; room warmth = close fit 1.112 [−1.956, 3.658]; room warmth = close `warmth_v0` 4.049 [−1.679, 15.964] (context only — the v0 form is not a renormalized cosine, so its slope is in different units).

**Supplement — the same registered quantity at the power the corpus actually has:** the within-reader (fixed-effects) panel, per-night baseline on per-night warmth, reader means removed, 12 readers × 66 reader-nights, cluster bootstrap over readers. Slope **0.133, 95% CI [+0.023, +0.253]**; under the close-fit room warmth, 0.099 [−0.002, +0.217] — entirely inside the alignment band. Seed-stable (seeds 20260820/1/7). Both treatments exclude the collapse anchor (1.0) by 3–4 CI half-widths; the primary warmth treatment excludes 0 by a hair (a real, small warmth-tracking component of ~10–13%).

## What this settles

- **H-reader≡room: not established — and the collapse reading fails wherever there is power to test it.** The registered between-reader slope cannot adjudicate (CI spans both anchors), but the point estimate's landing on 1.0 is exactly what the within-reader panel says it cannot be: night by night, a reader's baseline moves ~0.13 per unit of room warmth, not ~1. The "trusted reader = reader who agrees with the room" reading would require the baseline to track the rooms visited; it barely does.
- **The convergence note is disambiguated in the "shared basis, beautiful-and-warning" direction, at partial strength.** The reliable subspace does overlap the warmth form's heavy weights, and ~13% of a room-warmth unit shows up in the reader's baseline — the two readings share a real component. But the ICC's 0.7714 stands as what it measured: baselines are mostly reader-specific instrument constants, not slow warmth estimates wearing names.
- **The premise's numerator half gains its sharpest pass available at this grain.** Idiosyncratic baseline structure is not room geometry in disguise; the E2/E3 divergence explanation "embedded readers ARE rooms" is bounded at ~13%.
- **The committee hinge gets its number.** The slope CI was registered as the hinge of the nurse-as-index claim (principle vs metaphor): the hinge swings alignment-ward at every treatment with power, and the collapse sentence is unavailable at every treatment.

## Honesty

The registered verdict belongs to the primary between-reader regression, and it is INDETERMINATE — the supplement cannot be laundered into a clear any more than E3's miss could be laundered into a kill. The within-reader panel is a post-hoc higher-power reading of the same quantity (labeled as such; the registered wording fixes per-reader baselines, not per-night panels), its cluster bootstrap has 12 clusters, and its CI upper bound (0.253) sits a hair outside the pre-declared alignment band edge (0.25) under the primary warmth treatment — "lean alignment, at the band edge," not "alignment." The verdict bands (±0.25) are this run's operationalization of the registered "≈", declared before the run. No new measurement was taken: everything is a read-only replay of logged facts, deterministic, input md5s recorded in the results JSON.

## Reproduce

```
cd /home/eileen/projects/elephant
python3 scripts/slope_regression.py   # writes scripts/slope_regression_results.json
```

Provenance: registration — topic.md v3 convergence note + e2-e3-side-by-side.md §7 (both pre-dated); instrument reuse — `elephant/vmf.py` (WARM, zvec), `elephant/tapnight.py` (bounds/centers), `scripts/e2_instrument.py` (reading definition); corpus — the committed E2 v:2 S-nights (commit 54a0af1 extended the S-series; md5s in the results JSON). Not committed, per instruction.

*The count of launderings stands at six; none added by this document.*
