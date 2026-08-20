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

---

# Addendum A — Registered rerun (2026-08-20, subagent pass)

Script: `scripts/cross_strata_transfer_registered.py` (superset of the filed script; base numbers reproduced bit-for-bit — same instrument, same seeds 20260820/20260821). This addendum records one provenance **correction** and four robustness probes that materially qualify the filed verdict's "ROBUST" characterization. Addenda are registered probes of the primary estimand, not re-specifications.

## A1. Provenance correction: `reader_final` is PRE-LENS (the filed claim was false)

The filed report (§1) claimed M_R(k) — the median of R's lens-applied readings — is "EXACTLY the logged `session_close.reader_final` fact (verified numerically)". **That claim is false, and we say so.** Engine source: `elephant/tapnight.py` `_reader_final()` (line 618) takes the componentwise median of `_reader_hist`, and `_reader_hist` stores the **pre-lens** displaced field `eff = clamp(raw + s·(vibe − raw))` (line 282) — i.e. exactly `field_eff_to_reader`, before any dial-lens application. The lens is affine with constant per-reader gain g, so the two medians are related by

```
M_R(k) = CENTER + g_R ⊙ (reader_final_R(k) − CENTER)
```

and coincide **only on dials where g_i = 1** (the max-weight dial). Numerically verified: `med_raw[volume]/g = reader_final[volume]` to 1e-9 on every g ≠ 0 dial (writer/S1). The primary transfer numbers do **not** depend on the false claim (the script computes M_R(k) from its own `night_base`), and the memory-grain definition per the task statement — "componentwise median of R's readings" — is the lens-applied median, which the script correctly uses. Only the provenance identification was wrong. It is corrected here, and the proxy choice is now tested explicitly (A4).

Field-channel integrity re-verified this run: **replay == log on 72/72 (reader, night) channels** (max dev < 1e-9).

## A2. Room-volatility common-cause control (schedule-family composition) — SURVIVES

d_R and p_R are measured on the nights R attended; if readers attended different schedule families with different room forcing, both grains could be elevated by the room, not the reader. Control: per-night room volatility v_k = mean over strata transitions of ‖mean(`field_eff_after` in next cell) − mean(… prev cell)‖ / corpus_sd — a **reader-independent logged fact** — and per reader v_R = mean over attended nights. Per-family v_k: S1 0.105, S2 0.309, S3 0.107, S4a 0.526, S4b 0.331, S5 0.230, S6 0.092, S7 0.100.

```
corr(d, v_R)   = −0.0965      corr(p, v_R) = −0.3504
partial corr(d, p | v_R) = +0.8046   CI [+0.3693, +0.9937]   perm p = 0.0016
```

The transfer **survives** controlling for the volatility of the nights each reader attended (it slightly *strengthens* — attendance composition is not the common cause).

## A3. Mechanical-coupling probe (S5 null drift) — SIGNAL-SPECIFIC

Both grains share one observable series; a pure same-source artifact should appear in ANY movement statistic. On the n = 8 readers who attended S5, the NULL (no-flip) drift d⁰_R — within-night movement in the absence of any room signal — against the same p_R:

```
corr(d_signal, p) = +0.9658   CI [+0.8834, +0.9972]   (same readers)
corr(d0_null,  p) = +0.3352   CI [−0.0460, +0.9079]   perm p = 0.3953
```

Null drift does **not** predict memory plasticity (CI covers 0); signal drift on the same readers predicts it near-perfectly. The shared-series artifact alone does not reproduce the coefficient — the propagation is specific to **signal-strata** movement.

## A4. Memory-proxy robustness, full 2×2 (lensed vs RAW) — THE TRANSFER IS LENSED-SPACE-SPECIFIC

The lens is affine with constant gain, so swapping the memory object to the logged pre-lens `reader_final` is exactly a gain reweighting. All four cells (n = 15; bootstrap seeds 20260822, permutation same):

```
corr(d_lens, p_lens) = +0.7839   CI [+0.2837, +0.9903]   perm p = 0.0024   ← primary cell
corr(d_lens, p_raw ) = +0.1379   CI [−0.3446, +0.5878]   perm p = 0.6285   ← dies
corr(d_raw , p_lens) = +0.5846   CI [+0.1698, +0.8667]   perm p = 0.0195
corr(d_raw , p_raw ) = +0.5123   CI [+0.0354, +0.8486]   perm p = 0.0590   ← borderline
corr(p_lens, p_raw)  = +0.4366   (the two memory proxies are only moderately related)
```

The strong coefficient lives **only** in the lensed × lensed cell. Lensed drift does not predict raw (pre-lens) memory plasticity (+0.14, null). Raw drift predicts both lensed and raw plasticity only moderately (+0.58 / +0.51, the latter p = 0.059). **The transfer is a property of the reader's lensed reading space; the raw room channel does not carry it.**

## A5. Dial-concentration mechanism control — THE CONTROL THE FILED REPORT MISSED; CI COVERS 0

The filed report's mechanism-amplitude partials (gain = mean g, charisma, vibe extremity; §4: +0.79–0.80) controlled the **mean** gain. The right control is the **concentration** of the lens on high-variance dials: var_R = Σ_i g_i² σ_i², the variance of the raw room channel as seen through R's lens (σ² = per-dial corpus variance).

```
corr(var_R, d) = +0.7887      corr(var_R, p) = +0.6087
partial corr(d, p | var_R) = +0.6228   CI [−0.1512, +0.9887]   perm p = 0.0148
```

Lensed drift is almost entirely dial-concentration-driven (r = +0.79 with a reader constant that contains no session behavior at all). Holding the lens's dial concentration fixed, the between-reader transfer **bootstrap CI covers 0** — the filed "robust to mechanism-amplitude controls" claim is **superseded**. Readers whose lenses sit on variable dials read bigger at BOTH grains; that shared mechanical root is not fully removable at N = 15, and the permutation p (0.015) and the bootstrap CI disagree — book the residual transfer as suggestive, not robust.

## A6. Final verdict (this addendum)

**Primary registered estimand, unchanged: ρ = corr(d_R, p_R) = +0.7839, reader bootstrap CI [+0.2927, +0.9911] (10k draws, seed 20260820), permutation p = 0.0032 — TRANSFER under the decision rule (CI excludes 0).** Slope β = +0.4942 corpus-sd per corpus-sd, CI [+0.1411, +0.7756].

**The filed "ROBUST-BUT-EXPLORATORY" characterization is downgraded to: TRANSFER ON THE REGISTERED ESTIMAND, LENSED-SPACE-SPECIFIC, MECHANISM-UNRESOLVED.** What survives: room-volatility common-cause control (+0.80, CI clear of 0); mechanical-coupling probe (signal-specific, null drift does not predict); ORIG6-within ordering (+0.99); n_nights ≥ 3 and barkeep-excluded subsets (+0.71 / +0.70, CIs clear of 0). What does not survive: the pre-lens memory proxy (+0.14, null) and the dial-concentration control (CI covers 0). Both failures point the same way — **the quantity that transfers is the reader's own lensed read, and its amplitude is substantially set by which dials the lens attends to.** That is consistent with the reader-as-room framing (the reader's read, not the raw room, is the object), but it is not evidence of a *reader-level* fast→slow coupling independent of the lens's dial geometry. The sixth laundering's discharge stands at the registered estimand; the mechanism story the dissertation may carry is the weaker, better-supported one: *within-session movement and memory plasticity are one quantity in the reader's lensed reading space (ρ ≈ 0.78), and that quantity's amplitude is dial-concentration-driven; the raw room channel shows no such coupling.*
