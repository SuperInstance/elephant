# Cross-Strata Transfer — Session Grain → Memory/Identity Grain

**Filed: 2026-08-20.** Status: the exploratory-but-registered transfer test, open question #4 (`research/topic.md` v3). The clause that ran in August was **cross-condition** (the admitted sixth laundering); this document reports the **true cross-strata version**, now runnable on E2's v:2 per-reader schema (night-S1..S7, reader-grain logging). Script: `scripts/cross_strata_transfer.py` (numpy-only, read-only, reuses `scripts/e2_instrument.py` unmodified). No existing scripts modified; nothing committed.

---

## 0. The one-paragraph verdict

**TRANSFER, POSITIVE, ROBUST-BUT-EXPLORATORY.** The transfer coefficient is **ρ = corr(d_R, p_R) = +0.784, bootstrap CI [+0.29, +0.99] over readers, permutation p = 0.0032 (N = 15 registered readers, 8 S-nights, 5+ schedule families).** Readers whose readings drift more *within* sessions (session grain, the registered E2 drift estimator on logged v:2 facts) are the same readers whose per-night memories-of-room sit further from their stable identity (memory/identity grain): within-session movement is **not absorbed** — it propagates into between-night memory structure, at roughly **β = +0.49 corpus-sd of memory plasticity per 1 corpus-sd of session drift**. The effect survives barkeep-exclusion (+0.70), the n_nights ≥ 3 subset (+0.71), rank correlation (+0.79), and — importantly — Frisch–Waugh partials on the mechanism-amplitude knobs (lens gain, charisma, vibe extremity), which leave it at +0.80. The secondary coefficient is **null**: drift does *not* predict room-tracking of memory (r = −0.21, CI [−0.61, +0.40]) — the propagated displacement is **idiosyncratic, not room-aligned**. Booked as designed, not as confirmed: N = 15, one corpus, shared observable (caveats in §6).

## 1. The two grains, defined honestly

**Session grain (fast, within-night).** Per reader R, the registered E2 drift estimator (`e2_instrument.Measurement`), consumed from the logged v:2 `field_eff_to_reader` facts passed through R's lens: d_R = mean over signal strata transitions of ‖mean(next cell) − mean(prev cell)‖ / corpus_sd, over the registered strata (flip@20/@8, cold-entry splits, reversals, oscillation blocks). The S5 no-flip split supplies the per-reader null d⁰_R — the instrument's primary negative reading ("this reader has NOT drifted from herself"). Corpus scale: corpus_sd = 0.2405 (S-nights raw field, RMS over dials). Field-channel integrity re-asserted for this run: replay == log for all 72 (reader, night) channels, max dev < 1e-9.

**Memory/identity grain (slow, across-night).** The corpus has **no cross-night state carryover** — every night's session resets each lens to `vibe_start` (verified: roster `vibe == vibe_start` in every `session_open`; `lens_now` at seq 0 is one acclimation step past `vibe_start`). The engine therefore has no literal across-night memory, and we say so: the identity/memory grain is defined from the only across-night per-reader objects the corpus carries —

- **identity** b_R = mean over R's nights of M_R(k) — the stable baseline (the E2 ICC object's center);
- **memory-of-room** M_R(k) = componentwise median of R's readings on night k — which is *exactly* the logged `session_close.reader_final` fact (verified numerically; the schema doc's "greppable per-reader baseline fact"). This is the closest honest proxy for "their reading of each night's room," and it is what the task names. No cleaner memory object exists in the corpus; anything more would be laundered in.

Memory-grain behavior per reader: **plasticity** p_R = mean_k ‖M_R(k) − b_R‖ / corpus_sd (how far night-memories sit from identity) and **room-tracking** t_R = mean_k cos(M_R(k) − b_R, μ_k − μ̄) (alignment of memory displacement with the room's own displacement μ_k − μ̄, μ_k = night k's final room μ̂).

## 2. What "transfer" means, and what counts as a null

**Transfer (ρ > 0):** readers who move more within sessions also carry night-memories further from their stable identity — fast-grain movement survives median-averaging and night aggregation and shows up as slow-grain memory structure. **Null (ρ = 0):** the strata decouple — within-session movement is mean-reverted/absorbed before reaching memory/identity structure; a reader can be a big session mover and a stable-identity keeper (and vice versa). Decision rule fixed before reading output: bootstrap CI covering 0 ⇒ null; CI lower bound > 0 ⇒ transfer; CI upper bound < 0 ⇒ negative (compensatory) transfer — booked as a finding either way. A structural note: the two grains share one observable series, so *perfect* absorption (night-medians identical across nights for everyone) would force ρ undefined, and pure room-lockstep (all readers' memories moving only with the room) would collapse between-reader variance — both degenerate cases read as null; the test has teeth precisely between them.

## 3. Session-grain facts (per reader and per family)

Signal drift (reader-mean over the S-corpus signal transitions) = **1.134** corpus-sd vs null (S5, n = 8 readers) = **0.463** — signal/null = **2.45×** (E2's field numbers on the primary nights were 0.748 vs 0.291, 2.6×; consistent separability, different night set — S6/S7 non-monotonic families included here). Per-reader nulls span 0.25–0.94; lamplighter is the one reader whose null exceeds his signal (0.64 vs 0.94) — a genuinely no-flip-movable reader.

| Reader | nights | drift d | null d⁰ | plast p | track t | gain | char | extr |
|---|---|---|---|---|---|---|---|---|
| barkeep | 3 | 2.670 | — | 1.680 | 0.567 | 0.399 | 0.10 | 1.525 |
| cartographer | 2 | 2.720 | — | 1.585 | 0.258 | 0.330 | 0.13 | 0.836 |
| singer | 3 | 2.255 | — | 0.330 | −0.118 | 0.345 | 0.44 | 0.778 |
| fiddler | 3 | 1.501 | — | 1.491 | 0.546 | 0.407 | 0.40 | 1.038 |
| critic | 8 | 1.385 | 0.484 | 0.876 | 0.784 | 0.357 | 0.18 | 0.894 |
| tinker | 2 | 1.137 | — | 0.736 | 0.384 | 0.554 | 0.17 | 0.998 |
| drifter | 2 | 0.917 | — | 0.294 | 0.164 | 0.408 | 0.45 | 1.276 |
| weaver | 3 | 0.826 | 0.484 | 0.648 | 0.901 | 0.406 | 0.10 | 1.071 |
| engineer | 8 | 0.742 | 0.451 | 0.493 | 0.747 | 0.408 | 0.25 | 1.012 |
| lamplighter | 3 | 0.637 | 0.936 | 0.495 | 0.549 | 0.398 | 0.23 | 0.977 |
| writer | 8 | 0.489 | 0.369 | 0.292 | 0.398 | 0.357 | 0.20 | 0.892 |
| blacksmith | 3 | 0.490 | — | 0.242 | 0.264 | 0.420 | 0.16 | 0.956 |
| captain | 8 | 0.411 | 0.316 | 0.246 | 0.250 | 0.408 | 0.30 | 1.040 |
| essayist | 8 | 0.428 | 0.248 | 0.360 | 0.737 | 0.357 | 0.10 | 1.049 |
| poet | 8 | 0.403 | 0.415 | 0.242 | 0.271 | 0.476 | 0.15 | 0.894 |

Family-level transitions (mean over attending readers, corpus-sd): flips dominate (S1 1.85, S2 1.63, S6 w1→c1 1.93, S7 b0→b1 1.75); later oscillation transitions die (S7 b4→b5 onward ≈ 0.09–0.12); the S4b late cold entry into an already-cynical room moves little (0.47); the S5 no-flip null sits at 0.46. The corpus's schedule families deliver genuinely different session-grain forcing — which is what makes the across-families transfer question well-posed.

**Memory-grain variance decomposition** (ORIG6, balanced 6×8 panel, per dial: share of M_R(k) variance carried by reader-identity vs night/room): mood 0.97/0.02, volume 0.97/0.01, earnestness 0.91/0.05, presence 0.72/0.08, cynicism 0.52/0.21, joke_landing 0.26/0.57, panic 0.17/0.53. Mean shares: **identity 0.65, room 0.21, residual 0.15.** The memory grain is identity-organized on exactly the ICC-reliable dials and room-organized on the unreliable ones — the ICC finding's decomposition echo, now at the memory grain.

## 4. The transfer coefficients

**Primary: ρ = corr(d_R, p_R) = +0.7839**, reader bootstrap CI **[+0.2927, +0.9911]** (10k draws, seed 20260820), permutation p = **0.0032**; slope β = +0.494 corpus-sd per corpus-sd, CI [+0.141, +0.776]; Spearman +0.789.

**Robustness:** n_nights ≥ 3 (n=12): +0.712, CI [+0.108, +0.995]. Barkeep excluded (n=14): +0.698, CI [+0.121, +0.985]. ORIG6 only (n=6, descriptive): +0.986. The two extreme high-high readers (barkeep, cartographer) are *not* carrying the effect alone — the within-ORIG6 ordering alone reproduces it.

**Mechanism-amplitude controls (Frisch–Waugh partials):** corr(d,p | gain) = +0.787 (p = 0.0009); corr(d,p | gain+charisma+extremity) = **+0.802** (p = 0.0006). The transfer is not a lens-amplitude artifact: partialling the knobs out *strengthens* it slightly. (Honest tension, pre-stated: gain/charisma/extremity are themselves identity parameters, so the partial is a mechanism probe, not a cleaner estimand.)

**Secondary: corr(d_R, t_R) = −0.209**, CI [−0.614, +0.396], p = 0.452 — **null**. On average memories do track the room (mean t across readers ≈ +0.45), but the *between-reader* differences in session drift do not predict room-tracking. Read jointly with the primary: what propagates across strata is **idiosyncratic displacement, not the room's own signal**. High-drift readers' memories are further from their identity, in directions of their own.

## 5. Verdict (printed by the script)

```
rho = +0.7839   CI [+0.2927, +0.9911]   perm p = 0.0032
TRANSFER: session-grain drift positively predicts memory-grain plasticity
(CI excludes 0 above). Within-session movement propagates into
between-night memory structure — not absorbed.
```

With the secondary rider: the propagation is **not room-tracking** — session-grain movement predicts how *far* memories sit from identity, not alignment with the room. The sentence the dissertation may carry: *within-session drift and memory plasticity are one quantity measured at two timescales (ρ ≈ 0.78), while the room's own signal is not what crosses the strata.*

## 6. Caveats and honest limits

1. **Shared observable.** Both grains derive from the one `field_eff_to_reader` series. The coupling is not mechanical (medians of medians vs within-night segment means; perfect absorption would null it), but a residual same-source story cannot be fully excluded at N = 15.
2. **N = 15, one corpus, heterogeneous attendance** (8/3/2 nights). The two 2-night readers sit at the extremes of both axes. Subsets bracket, do not remove, this.
3. **Exploratory-but-registered.** The corpus pre-existed (E2); grains, estimands, decision rule, and null were fixed in the script's docstring before any output was read. No counterfactual re-runs were performed; the one design iteration was mechanical (param lookup for late-roster readers), before results were inspected.
4. **Memory is a proxy.** The engine resets lenses nightly; "memory-of-room" is the per-night reading summary (`reader_final`), the corpus's only greppable per-reader across-night fact. A true carry-memory claim would need a corpus with cross-night lens persistence — a registered candidate for the night-H family redesign.
5. **The launderings count stands at six.** This test is the de-launer's discharge of the sixth: the clause that ran in August claimed cross-*strata* and delivered cross-*condition*; this one delivers cross-strata or nothing. It delivered.

## 7. Provenance and reproduce

- Corpus: `data/nights/night-S1..S7.jsonl` (v:2 reader schema; manifest `data/e2/e2-nights-manifest.json`, reader_schema 2, deterministic replay verified).
- Instrument reuse: `scripts/e2_instrument.py` (registered strata, drift estimator, attendance `FIELD_NIGHTS_EXT`, cold-entry protocol) — imported unmodified; `scripts/premise_measurement.py` machinery superseded by the logged v:2 facts it was built to approximate.
- This run: `python3 scripts/cross_strata_transfer.py` (numpy-only, CPU, read-only; bootstrap/permutation seeds 20260820/20260821). Field-channel assertion: replay == log on 72/72 (reader, night) channels.
- Context: `research/topic.md` v3 §open questions #4; `research/prototype/e2-e3-side-by-side.md` (the reader-baseline/drift framing this test consumes).
