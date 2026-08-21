# STAGE-2 CORPUS DESIGN — Making the Slope Test Decisive

**Filed: 2026-08-20.** Companion to `SLOPE-REGRESSION-2026-08-20.md` (the INDETERMINATE run) and `research/topic.md` v3 (the registration). This document **designs** the next corpus wave; it changes nothing. Read-only on all repos: no commits, no pushes, no script edits. Design only — generation changes are named (§5) but not made.

## Verdict (6 lines)

1. **Schedule:** keep all 9 night families (warmth ladder 0.319–0.759 already spans 0.44); 21 readers × 9 nights, 7–10 readers per night, every reader attends a **unique 3–4 night subset** in one of **three x-bands** (cold ≈0.48, mid ≈0.64, warm ≈0.71) — no reader attends all rooms, no two readers share a visited-room set.
2. **x-variance:** design delivers x-range **0.254** (target ≥ 0.15; current gap 0.0381), **15 distinct x values**, **Sxx = 0.197** (current filed 15-reader set: Sxx = 0.111).
3. **Power:** n = **21** (7/7/7 bands) → SE = 0.264 (σ=0.117, observed) / 0.338 (σ=0.15, conservative); power to exclude 1 at β=0: **0.947 / 0.801**. Current corpus at n=15 was never decisive (power ≈ 0.62).
4. **Registration amendment:** append a dated addendum to topic.md — n_nights ≥ 3 **unchanged**, exposure-weighting **rejected**, scalarization + aggregation + branch thresholds **now registered** (the run's choices), and a new **void-by-rule x-side validity precondition** (≥3 distinct x, Sxx ≥ 0.19, unique sets) so degeneracy can never again be misread as a slope.
5. **Generation route:** reuse `scripts/e2_nights.py` (new T-tags over the same frozen scripts/strata) + new attendance map; `scripts/e2_instrument.py` gains the wave-2 night list + `FIELD_NIGHTS_W2`; `slope_regression.py` runs unmodified against the new night set. Verified: per-speak `warmth_vmf` is **roster-invariant** (bit-identical across 1- vs 9-reader rosters on S2/S3/S4a), so the x-side is computable a priori.
6. **Failure-mode guards:** logged-roster assertion + determinism re-run (attendance drift); 4-night mid-band buffer (attrition); corpus_sd 0.2367 + per-night warmth reproduction checks (engine drift); stratified archetype assignment + class-residual tripwire (imbalance); ≥3 distinct x by rule (two-point recurrence); ICC re-measured before the slope is read (baseline stability).

---

## §1 Diagnosis — why the registered test came back INDETERMINATE

The registered slope (7 readers, n_nights ≥ 3) collapsed to a drifter-vs-rest contrast because **six of seven included readers attended all nine rooms** — their x (mean warmth of visited rooms) is the grand mean +0.6073 to four decimals. The x-variance the test needs lived only in the n=2 readers the rule excludes (filed x: 0.3826 → 0.7499, range 0.367). Two structural facts, both **verified empirically today**, frame the fix:

1. **The room thermometer is roster-invariant.** Per-speak `warmth_vmf` is fit from `vmf_fit(vmf_windowed(room, bank, W=10))` — the **text-content** field, not the charisma-pulled `field_eff_after`. Replaying the S2/S3/S4a scripts with 1–2 readers instead of 9–10 produced per-speak warmth **bit-identical to the filed logs** (max |Δwarmth| = 0.00e+00 on all 19/19 and 37/37 speaks tested). Room warmth is a property of the script composition; the x-side of any wave built on the frozen scripts is computable a priori. Night warmth ladder (filed, reproduced): S2 +0.3187 · S4a +0.4465 · D/D-cold +0.6293 · S4b +0.6319 · A/S1 +0.6551 · S3 +0.7409 · S5 +0.7589.
2. **The reader side is roster-dependent.** The same replay shows `field_eff_after` shifting up to 0.44 per dial when the roster changes (charisma pulls + acclimation). Wave-2 baselines and drifts are therefore **new measurements**, not re-uses of wave-1 y-values. This is fine — E-cont convention is per-wave — but it means the guard numbers (drift 0.7483, spread 0.4556) are attendance-dependent and must be **re-filed at wave generation**, while `corpus_sd` (fit from roster-invariant `field_raw_after`) must reproduce 0.2367 exactly (that reproduction is itself the "room side untouched" guard).

The fix is therefore **purely an attendance redesign** on the same 9 frozen script families. No new room content, no new ladder, no change to the analysis code's semantics.

## §2 Attendance schedule (requirement 1)

**Design principle:** the slope's x-resolution comes from *where in the warmth ranking a reader's subset sits*, not from how many nights they attend. Three nights at the extremes beats nine nights at the mean. Minimum 3 nights per reader is sufficient **iff** subsets are drawn from opposite ends of the warmth-ranked pool; 4 nights is used for the mid band (attrition buffer; mid x stays mid under D/D-cold/S4b/A/S1 swaps, all 0.629–0.655).

**Should the 9-room structure change? No.** (a) Uniqueness combinatorics: 21 readers need 21 unique ≥3-subsets; C(9,3) = 84 vs C(7,3) = 35 for a 7-room design — fewer rooms starves the uniqueness requirement, not the warmth contrast. (b) The 9 families' warmth ladder is known, frozen, and roster-invariant; abandoning it forfeits the only fully-certain half of the apparatus. (c) Rotating subsets per night is exactly what the design does (7–10 readers/night, everyone on 3–4 nights).

**Night×reader matrix (21 readers = 15 existing frozen personas + 6 new seeded draws; T-tags denote wave-2 logs of the same families):**

| reader | nights (T-family) | x (a priori) | band |
|---|---|---|---|
| writer | T2, T4a, T5 | +0.4648 | cold |
| engineer | T2, T4a, T5c | +0.4648 | cold |
| drifter | T4a, T4b, T2 (staged entry kept) | +0.4657 | cold |
| lamplighter | T2, T4a, T1 | +0.4734 | cold |
| tinker | T2, T4a, T3 | +0.4734 | cold |
| new-1 | T2, T4a, T5, T5c | +0.5060 | cold |
| new-2 | T2, T4a, T5, T4b | +0.5060 | cold |
| poet | T5, T5c, T4b, T1 | +0.6364 | mid |
| critic | T5, T5c, T4b, T3 | +0.6364 | mid |
| singer | T5, T5c, T1, T3 | +0.6399 | mid |
| cartographer | T5c, T4b, T1, T3 | +0.6429 | mid |
| blacksmith | T5, T4b, T1, T3 | +0.6429 | mid |
| new-3 | T5, T5c, T4b | +0.6302 | mid |
| new-4 | T5, T5c, T1 | +0.6399 | mid |
| essayist | T9, T8, T4b | +0.7106 | warm |
| captain | T9, T8, T1 | +0.7183 | warm |
| barkeep | T9, T8, T3 | +0.7183 | warm |
| fiddler | T9, T8, T5 | +0.7097 | warm |
| weaver | T9, T8, T5c | +0.7097 | warm |
| new-5 | T9, T8, T4b, T1 | +0.6967 | warm |
| new-6 | T9, T8, T1, T3 | +0.7025 | warm |

**Verified properties:** 21/21 unique visited sets; 15 distinct x values; x-range 0.2535 (cold mean 0.4793, mid 0.6384, warm 0.7094); Sxx = 0.1971; night loads 7–10 (T5/T5c carry the most, ≤10; D↔D-cold swaps are the load-balancing lever — warmth-identical); every reader has ≥2 signal transitions (drift measurable for all — S5-only readers are impossible by construction); band membership survives any D↔D-cold or A↔S1 swap (warmth-identical pairs).

**Archetype stratification** (E5: 93–96% of baseline y-variance is between-archetype, so balance is mandatory): original-6 split 2/2/2 across bands (writer+engineer / poet+critic / essayist+captain); drifter → cold; the 8 new personas split 2/3/3 (lamplighter+tinker / singer+cartographer+blacksmith / barkeep+fiddler+weaver); 6 new draws 2/2/2. No band holds an archetype majority.

## §3 Registration amendment (requirement 2)

Diff-ready as an **appended, dated addendum** to `research/topic.md` (doctrine: annotate, never delete — the 2026-08-20 run stands under the original registration). Proposed text:

> **### ADDENDUM 2026-08-20 (Stage-2 corpus wave — resolves the run's committee-open items; nothing above is altered)**
>
> 1. **Corpus-design requirement (replaces the de-facto attendance degeneracy).** Before analysis, the included set (n_nights ≥ 3) must satisfy: (i) **no two readers share the same visited-room set**; (ii) **≥ 3 distinct x values** and **Sxx ≥ 0.19**; (iii) readers stratified into three x-bands (targets ≈ 0.48/0.64/0.71) with **no archetype majority in any band**. If (i)–(iii) fail, the run is **VOID — INDETERMINATE BY RULE**, and no branch reading is taken.
> 2. **Committee-open §1 (scalarization) — registered:** PRIMARY = direction cosine of the z-standardized reliable-subspace baseline against the subspace-restricted, renormalized `vmf.WARM` (the run's choice; unit commensurability with `warmth_vmf` is what gives "slope ≈ 1" its meaning). Raw z-projection remains a labeled sensitivity.
> 3. **Committee-open §2 (room-warmth aggregation) — registered:** PRIMARY = unweighted per-night mean of logged per-speak `warmth_vmf`. **Exposure-weighting is rejected**: night lengths are schedule-determined, and the registered object is the warmth of the rooms *visited*, not per-speak exposure. `session_close` final-window fit remains a labeled sensitivity.
> 4. **Committee-open §3 (x-side degeneracy) — resolved by design:** inclusion rule **unchanged** (n_nights ≥ 3); the registration-relaxed all-readers variant remains a labeled sensitivity only. The corpus, not the rule, guarantees x-variance.
> 5. **Committee-open §6 (branch thresholds) — registered:** ≈0 = bootstrap CI contains 0; ≈1 = CI contains 1. **Alignment** declared iff CI contains 0 AND excludes 1 AND §1 preconditions hold; **collapse** iff CI contains 1 AND excludes 0 AND preconditions hold; else INDETERMINATE.
> 6. **Power target (filed):** n = 21 readers (7 per band), design Sxx = 0.197, x-range = 0.254; inference unchanged (bootstrap over readers, B = 2000; permutation null, 10,000).

## §4 Power (requirement 3 — the math, shown)

**Model.** OLS with intercept; per-reader points (x = mean warmth of visited rooms, y = baseline warmth-cos). Under the null of interest (true slope β = 0), b̂ ~ N(0, SE²), SE = σ_ε/√Sxx.

**σ_ε.** The y-spread under β=0 is the reader-side scatter: sd of the filed 15 baselines = **0.117** (range 0.584–0.998 matches the observed 0.58–0.99). Conservative bound used throughout: **σ = 0.15**.

**Decision rule.** Alignment is declared when the bootstrap 95% CI contains 0 and excludes 1. At β = 0, P(CI excludes 1) = P(b̂ + t·SE < 1) = F_t(1/SE − t), df = n−2. For 80% power: 1/SE − t ≥ F_t⁻¹(0.8).

**Design SE at n = 21, Sxx = 0.197:** SE = 0.117/√0.197 = **0.264**; SE = 0.15/√0.197 = **0.338**. Required SE for 80% (df = 19: t = 2.093, F⁻¹(0.8) = 0.861): SE ≤ 1/(2.093+0.861) = **0.339**. The design sits exactly at the threshold under the conservative σ, well inside under the observed σ:

| n (design) | Sxx | SE (σ=0.117 / 0.15) | power σ=0.117 | power σ=0.15 |
|---|---|---|---|---|
| 12 (3-cluster) | 0.127 | 0.328 / 0.421 | 0.78 | 0.56 |
| 15 (3-cluster) | 0.159 | 0.293 / 0.376 | 0.88 | 0.69 |
| 18 (3-cluster) | 0.191 | 0.268 / 0.343 | 0.94 | 0.78 |
| **21 (3-cluster, this design)** | **0.197** | **0.264 / 0.338** | **0.95** | **0.80** |
| 24 (3-cluster) | 0.254 | 0.232 / 0.298 | 0.98 | 0.89 |
| 14 (2-cluster fallback) | 0.212 | 0.254 / 0.326 | 0.95 | 0.80 |

**Why the filed corpus could never have decided it:** even the registration-relaxed 15-reader set had Sxx = 0.111 → SE = 0.351 (σ=0.117) → power ≈ 0.62, *and* it was dominated by six coincident x = 0.6073 points. The 2-cluster n=14 fallback reaches the same power with fewer readers but is exactly the two-point contrast shape that misled the drifter run; the 3-band design's 15 distinct x values give the permutation null genuine resolution. **Verdict: n = 21 (7/7/7).** No feasible N below 18 clears the 0.8 bar at σ=0.15.

## §5 Generation route (requirement 4 — named, not made)

The wave is a **fresh corpus on the frozen script families**: same SEG banks (`scripts/nights_abc.py`), same schedule-family compositions, same strata — new logs (new rosters) under new names. Files and minimal changes:

1. **`scripts/e2_nights.py`** — the generator that drives nights (cast + `TapNightSession` + per-speak v:2 logs + manifest + determinism check).
   - Add T-tags to `scripts()`: T1 = S1 family, T2 = S2, T3 = S3, T4a = S4a (staged drifter entry kept — its DRIFTER_LINES are part of the family's warmth), T4b = S4b, T5 = D, T5c = D-cold, T8 = S3, T9 = S5 (any tag names; the point is one log per family).
   - Replace `ATTENDANCE` with the §2 matrix (night → roster).
   - `generate()` already refuses overwrite (append-only) — the T-tags are new files, so the filed S1–S7 artifacts are untouched.
   - `run_night`/`stripped_md5`/determinism verification: unchanged.
2. **`scripts/e2_personas.py` / `data/e2/e2-personas.json`** — add 6 new seeded, archetype-labeled persona draws (the `new-N` readers), same schema.
3. **`scripts/e2_instrument.py`** — add `W2_NIGHTS` (T-tags → filenames + strata, same as the mapped families) and `FIELD_NIGHTS_W2` (the §2 matrix); `PRIMARY_NIGHTS` and `FIELD_NIGHTS` are **untouched** (filed wave-1 constants stay).
4. **`scripts/slope_regression.py`** — runs **unmodified** against the wave: the only wiring change is pointing `main()` at `W2_NIGHTS` (e.g., a `--wave` arg or a wave-2 entry constant); its reader-side, room-side, inference, and sensitivity machinery are already generic over nights/readers. `MIN_NIGHTS = 3` unchanged.
5. **New wave gate (generation-time, ladder-style):** a small check (design-only here) asserting, before any analysis: logged rosters == designed `ATTENDANCE`; `corpus_sd` reproduces 0.2367; per-night warmth reproduces the filed ladder to 4 decimals; the §2 band means land within ±0.02 of target; all 21 readers have ≥3 logged nights. Re-file the wave's guard values (drift, E-cont spread) — attendance-dependent, so new numbers at generation, same guard mechanism.

No changes made; this section is the change list for whoever builds the wave.

## §6 Failure modes and guards (requirement 5)

| # | Failure mode | Why it could bite | Guard |
|---|---|---|---|
| 1 | **Attendance drift** — designed matrix ≠ logged rosters | Roster baking bug or generation skip silently reproduces a degenerate x | Logged-roster assertion == designed `ATTENDANCE` before any analysis (fail-fast, regenerate the drifted night — deterministic); `e2_nights.py --verify` byte-determinism re-run |
| 2 | **Reader attrition** — a scheduled reader logs <3 nights | n_nights ≥ 3 rule excludes them; if it's a band anchor, x-variance shrinks | Mid band scheduled at 4 nights (buffer); pre-run validity check: all 21 ≥ 3 nights or the wave is incomplete (regenerate missing cells, not "run with fewer"); §3.1(ii) Sxx ≥ 0.19 makes any residual drop **void-by-rule**, never a silent slope |
| 3 | **Room-warmth saturation** — bands don't land | **Retired by measurement:** warmth_vmf is text-content-determined and roster-invariant (verified bit-identical on S2/S3/S4a). Residual risk is only engine/script drift | `corpus_sd` = 0.2367 reproduction (roster-invariant by construction — fit from `field_raw_after`) + per-night warmth reproduction vs filed ladder at generation; scripts frozen, never regenerated |
| 4 | **Archetype imbalance across bands** — the E5 trap | 93–96% of y-variance is between-archetype; a skewed assignment manufactures a spurious slope under alignment | Stratified assignment (§2: 2/2/2, 2/3/3, 2/2/2); class-residual sensitivity as the tripwire — if primary vs class-residual slopes diverge in sign or CI disposition, report both and make **no alignment declaration** on the primary |
| 5 | **Two-point degeneracy recurrence** | Two x-levels collapse the permutation null into a label-swap test (the drifter lesson) | ≥3 distinct x values by rule (§3.1(ii)); design delivers 15; max single-reader leverage bounded (any reader ≤ ~1/15 of Sxx vs 100% of the contrast in the degenerate run) |
| 6 | **Baseline instability under new rosters** | Reader-side dynamics are roster-dependent (field_eff shifts ≤0.44/dial); if baselines stop being stable reader constants, the slope is uninterpretable | Re-measure ICC on wave-2 data before reading the slope; if the ICC collapses below its filed 0.7714 CI [0.667, 0.810], the wave is not measuring stable baselines → void-by-rule (or at minimum reported as its own finding, slope unreported) |
| 7 | **Bootstrap coverage at small n** | Reader-bootstrap CIs degrade as n shrinks | n = 21 registered; B = 2000 over readers unchanged; report effective draws; the n≥18/σ=0.15 power bar from §4 is the floor |

## Provenance

- Grounding: `SLOPE-REGRESSION-2026-08-20.md` (esp. §3 committee-open), `research/topic.md` v3, `e2-e3-side-by-side.md` §7, `scripts/slope_regression.py`, `scripts/e2_field.py`, `scripts/e2_instrument.py`, `scripts/e2_nights.py`, `elephant/tapnight.py`, `elephant/vmf.py`.
- Empirical checks run today (read-only, temp dirs only): roster-invariance of per-speak `warmth_vmf` (S2/S3/S4a, 1–2 vs 9–10 reader rosters, max |Δ| = 0.00e+00); roster-dependence of `field_eff_after` (max 0.44/dial); design Sxx/power/band/uniqueness computations (numpy, seeds irrelevant — no randomness).
- No git commit; no file outside this document was written.
