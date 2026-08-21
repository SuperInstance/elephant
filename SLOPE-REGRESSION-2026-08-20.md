# SLOPE REGRESSION — H-reader≡room: the Registered Decisive Test, Run

**Dated: 2026-08-20.** Implements the advisor-registered slope regression (research/topic.md v3; e2-e3-side-by-side.md §7 — "the single most decisive next experiment") on the committed E2 field corpus, read-only. Script: `scripts/slope_regression.py` (numpy-only, CPU, seeds 20260820; writes `data/slope/slope-regression-results.json`). No git commit made. Guards reproduce the filed E2 numbers exactly (corpus_sd 0.2367, E-cont spread 0.4556, drift 0.7483).

## Method (5 lines)

1. **Reader side (y):** E2 canonical-presence readings; per-reader baseline = mean reading vector (E-cont convention, E5 machinery verbatim), restricted to the ICC reliable subspace (mood, volume, earnestness, presence — panic excluded by the two-line schema rule), scalarized as the direction cosine of the z-standardized subspace baseline against `vmf.WARM` restricted to the same subspace and renormalized.
2. **Room side (x):** per-night warmth = mean of the logged per-speak `warmth_vmf` (the vMF μ̂-projection thermometer, reader-independent); a reader's x = mean warmth over the nights they actually attended.
3. **Regression:** OLS with intercept over per-reader points; registered inclusion n_nights ≥ 3 → **7 readers** (8 excluded, listed below).
4. **Inference:** bootstrap over readers (B=2000, seed 20260820) for 95% CIs; permutation null (10,000 shuffles of x, same seed) for "any relationship at all".
5. **E5 discipline:** the primary reader side is the unresidualized registered baseline; the E5-corrected clean class-residualization runs only as a labeled sensitivity (the buggy in-place mutation of `e2_instrument.spread_seg` is never touched).

## Numbers

**PRIMARY (registered): slope = −3.086, bootstrap 95% CI [−6.373, −0.107]; intercept 2.701 [1.005, 4.572]; permutation p = 0.468. CI contains neither 0 nor 1.**

Per-reader points (primary set):

| reader | archetype | n_nights | x = rooms' warmth | y = baseline warmth-cos |
|---|---|---|---|---|
| writer | writer | 9 | +0.6073 | +0.9383 |
| poet | poet | 9 | +0.6073 | +0.9938 |
| essayist | essayist | 9 | +0.6073 | +0.7197 |
| engineer | engineer | 9 | +0.6073 | +0.5844 |
| critic | critic | 9 | +0.6073 | +0.8103 |
| captain | captain | 9 | +0.6073 | +0.9140 |
| drifter | drifter | 3 | +0.5692 | +0.9442 |

Excluded (registered rule n_nights ≥ 3; all n_nights = 2, listed): barkeep, blacksmith, cartographer, fiddler, lamplighter, singer, tinker, weaver.

**The structural fact the table shows:** six of the seven primary readers attended all nine rooms and therefore share x = +0.6073 to four decimals. The primary slope is a two-point contrast — drifter (the only reader with differentiated room exposure) versus everyone else — and the permutation null cannot reject no-relationship (p = 0.468). The registered design's x-variance lives entirely in the n = 2 readers the registered rule excludes.

## Interpretation (strictly two-branch)

- **Collapse (slope ≈ 1): does not fire.** The CI excludes 1 in every variant that has any x-variance at all (primary, actual-presence, all-15, class-residual, session-close).
- **Alignment (slope ≈ 0): cannot be declared at the registered bar.** The primary CI [−6.373, −0.107] excludes 0 — but on a degenerate two-point x this is a drifter artifact, not a measurement of the reader–room identity; the permutation null (p = 0.468) says the slope is indistinguishable from no relationship.
- **Honest verdict: INDETERMINATE at the registered rule — the test as registered is uninformative on this corpus, because the x-side degenerates under the n ≥ 3 inclusion rule.** The x-informative, registration-relaxed readings (below) both land on the alignment side: point estimates near 0, CIs containing 0 and excluding 1. *Leaning alignment, not declared alignment* — the same epistemic posture the side-by-side took on the premise.

## Sensitivity notes (labeled; the registered primary stands)

| variant | slope | 95% CI | note |
|---|---|---|---|
| (a) all 15 readers (n ≥ 2, registration-relaxed) | −0.223 | [−0.876, +0.356] | the only variant with real x-variance; contains 0, excludes 1 → leans alignment |
| (b) actual-presence instrument | −3.325 | [−6.589, −0.375] | same degenerate x; participation-conflated (+0.18 floor) |
| (c) class-residual y, E5-clean, 15 readers | +0.034 | [−0.005, +0.119] | archetype-conditioned reader side; contains 0, excludes 1; **vacuous on the 7-reader primary set** (all singleton archetypes → residuals identically 0) |
| (d) reader side as raw z-projection (position, not direction cosine) | +8.445 | [+3.635, +13.267] | units variant only — demonstrates the "slope ≈ 1" semantics exist only in matched (cosine) units |
| (e) room side = session_close warmth_vmf (final-window fit) | −1.407 | [−2.906, −0.049] | same degenerate x |

(d) is the quiet warning of this run: under unmatched units the point estimate moves by >11 units and changes sign across variants; the test's decisiveness is exactly as good as its unit convention, which the registration did not specify (committee-open, below).

## Committee-open (decisions the registration did not cover)

1. **Scalarization of "mean reliable-subspace reading"** — the registration names the subspace but not the scalar map. Chosen: direction cosine against the subspace-restricted, renormalized warm direction, for unit commensurability with `warmth_vmf` = Ŵ·μ̂ (without it, "slope ≈ 1" has no meaning). Alternative (raw z-projection) reported as sensitivity (d), not swapped.
2. **Room-warmth aggregation** — per-night mean of logged per-speak fits chosen; session-close final-window fit reported as sensitivity (e).
3. **The x-side degeneracy** — the registration did not anticipate that every n ≥ 3 reader visited the same 9 rooms, collapsing the primary regression to a drifter-vs-rest contrast. Whether "rooms actually visited" should be exposure-weighted, or the inclusion rule relaxed, is a committee decision; the relaxed version (a) is filed as sensitivity only.
4. **Class-residual reader side** — E5's erratum note (archetype-conditioned baselines as the right object) is *unrunnable* on the primary set (7 readers, 7 singleton archetypes); run on the 15-reader set (c) with the E5-clean centering. The filed buggy path was not used anywhere.
5. **S5 (null night)** included in baselines and visited-room means, following the instrument's E-cont convention.
6. **Branch thresholds** — "≈0" / "≈1" operationalized as CI-contains-0 / CI-contains-1; no numeric band was registered.

## Reproduce

```
cd /home/eileen/projects/elephant
python3 scripts/slope_regression.py    # <1 min, numpy-only, CPU
```

Outputs: console + `data/slope/slope-regression-results.json`. The reproduction guard (filed 0.2367 / 0.4556 / 0.7483) asserts before any regression output is produced. Read-only on the corpus; no commit, no push.
