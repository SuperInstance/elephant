# LENGTH DECOMPOSITION (E4 CLOCK-SPLIT, STAGE 1) — 2026-08-20

**The registered decomposition of the fine condition edge, run.** Registration: `zeroclaw-dissertation/research/committee/deep-think-2026-08-20/methodologist-glm53.md` (written before the run; the bands below are quoted from it, not re-derived). Predecessor: `SILENCE-TEST-2026-08-20.md` — silence-only (length proxy) 0.897 beats content 0.765 at the condition grain; cynicism/volume correlate −0.30 with message length; identity grain survives. The question this run answers: **does the fine condition edge survive de-confounding — and can this corpus even deliver that answer?**

Runner: `scripts/length_decomposition.py` (numpy-only analysis; Stage 0 imports the project's own numpy-only dial machinery to rebuild the exact z-space of the 1.2285 — no torch, no model loading, which is what killed the earlier attempt). Deterministic (re-run bit-identical; seeds 20260820/20260821). Checkpointed after every stage (`LENGTH-DECOMPOSITION-2026-08-20.json`) so a timeout cannot lose numbers. No git operations.

**Corpus:** night A (≡B≡C by md5, used once) + night-H, strata SEG1 seq 0–19 / SEG2 seq 20–39, exactly as registered. Two grains, both registered: **z-space** (the 1.2285's own: `seg_fit` trailing W=8 DialBank windows, `zvec` standardization, vMF μ̂ per condition) and the **triad grain** (the silence test's pinned protocol: W=4 stride-1 windows of logged `field_eff_after` dials, pooled-LONO logistic).

---

## 1. Reproduction: the confounded edge, bit-exact

The registered fine-gap chain, rebuilt from scratch over nights A and H:

| Quantity | Value |
|---|---|
| raw chord, night A (SEG1 vs SEG2 μ̂) | **1.2285** — bit-exact vs `summary.json` (`1.2285097805010095`) |
| raw chord, night H (ramp geometry) | 0.4208 |
| raw chord, pooled A+H | 0.8478 |
| baseline triad, full corpus (pinned LONO logreg) | content 0.603 (p=0.114, ns) / **silence 0.897 (p=7.4e-12)** |
| baseline triad, centroid | content 0.765 (p=1.4e-05) / silence 0.882 (p=5.7e-11) |

The silence test's numbers reproduce exactly. The confound is confirmed on the same corpus before anything is decomposed.

## 2. The decomposition, as registered

**Length model:** monotone spline df=4 (degree-1 I-spline / truncated-ramp basis, interior knots at train-fold quintiles 20/40/60/80, nonnegative weights by NNLS), cross-fitted within LONO folds (train one night, apply the other). Residualized dials = z − E[z | window-mean length].

**Orthogonality QC (the estimand's own validity check):**

| Fold | in-fold max \|r(resid, len)\| | out-of-fold max \|r\| | worst out-of-fold dims |
|---|---|---|---|
| train A → apply H | 0.145 | **0.892** | mood −0.892, earnestness −0.475, cynicism +0.321 |
| train H → apply A | 0.049 | **0.423** | mood +0.423, earnestness +0.336, cynicism −0.297 |

In-fold the spline is clean; **out-of-fold it fails badly: the length→dial mediator mapping does not transfer across nights.** The mood–length relation *flips sign* between the flip night (r=+0.39) and the ramp night (r=−0.15) — with different transition geometry, the same window length means a different room state. The pooled post-residualization correlations (all |r| ≤ 0.053) are a **cancellation across nights, not orthogonality**: within each night the residuals still carry length. An honest length-orthogonal dial set cannot be constructed by cross-night residualization from two nights of different geometry.

**Common support (registered: windows inside the length-distribution overlap; ≥70% or INDETERMINATE):**

| Grain | overlap band (chars) | kept/total | coverage | composition |
|---|---|---|---|---|
| z-space (window-mean len) | [67.8, 78.0] | 23/80 | **28.8%** | 17 SEG1 vs 6 SEG2 — the overlap is SEG1's lower tail |
| triad W=4 (all 4 msgs) | [57, 96] | 24/68 | **35.3%** | one LONO fold single-class (11 windows excluded from eval) |
| triad W=8 | [57, 96] | 10/52 | **19.2%** | fully degenerate (both folds single-class, n=0) |

The flip design makes SEG2 short and SEG1 long (message-grain Welch d=0.98; window-mean separation ~2.8 sd) — the conditions' length distributions barely overlap, and the overlap that exists is class-imbalanced.

**The number the procedure produces (reported; see verdict for its status):** deconfounded pooled chord on the common support = **1.829** (A 1.527 / H 0.879), *above* the raw 1.229 — an over-correction/normalization artifact of non-transportable residualization (residual vectors center near the origin, unit-normalization amplifies whatever misfit remains). Length-stratified permutation (night × length-quartile cells, cluster-aware): null mean 1.589 ± 0.246, **p = 0.096** — even taken at face value it misses the PASS gate. Cluster bootstrap (night × condition): 95% CI [1.498, 1.919].

## 3. Manipulation check: the trimming has no teeth

Silence-only on the common-support set (pinned LONO logreg, W=4): **acc = 1.000** (n=13 after the degenerate fold; p = 2.4e-04; centroid 0.923). Range-overlap trimming does **not** collapse the silence arm — within the overlap band the two conditions' length *distributions* still differ in mass, and silence reads the difference. Per the registration's own sentence: *if it does not fall to chance, the trimming failed — INDETERMINATE, not a pass.*

The full triad on the trimmed set is degenerate ceiling: residualized-content 1.000, silence 1.000, both 1.000, raw content 1.000 — every arm saturates; the comparison carries no arm-separating information.

## 4. The three numbers

| | Quantity | Value |
|---|---|---|
| (a) | confounded fine edge (raw z-space chord, night A) | **1.2285** (reproduced bit-exact) |
| (b) | length-orthogonal fine edge (cross-fitted spline residuals, common support, pooled) | **not certifiable** — the procedure returns 1.829 (perm p=0.096, CI [1.50, 1.92]), but the orthogonality QC fails out-of-fold (mood −0.89), so this is not the registered estimand |
| (c) | silence-only manipulation check on the trimmed set | **1.000** (p=2.4e-04) — fails; trimming cannot neutralize the clock arm |

## 5. Verdict (pre-registered bands, applied literally)

**INDETERMINATE.** All registered INDETERMINATE triggers fire at once:

1. **Common-support coverage < 70%** (0.288 z-space / 0.353 triad / 0.192 W=8). The PASS and KILL branches both presuppose a valid common-support evaluation; the corpus cannot supply one.
2. **Manipulation check failed** — silence-only still discriminates on the trimmed set.
3. **Honesty clause — unanticipated gate shape:** the registered cross-fitting (LONO folds = nights) produces non-transportable length models; the residualized dials are not length-orthogonal within nights, so the deconfounded point estimate exceeds the raw edge — a shape the branches did not anticipate, reported, not absorbed. Additionally: the KILL tolerance clause (silence ≥ residualized − 0.10) fires, but on ceiling-degenerate triad numbers under a failed coverage precondition — reported, not counted. And nothing supports PASS even at face value: perm p = 0.096 ≥ 0.05.

**What Stage 1 decisively establishes** (a negative result with content): the existing 2-night corpus **cannot adjudicate the fine edge's decomposition** — not because the analysis was underpowered for a point estimate, but for three structural reasons: (i) the conditions' length distributions are nearly disjoint by design, collapsing common support to SEG1's lower tail; (ii) the length→dial mediator mapping is geometry-dependent (mood flips sign across flip/ramp nights), so cross-fitted residualization cannot produce length-orthogonal dials; (iii) support-based trimming cannot neutralize the silence arm because the conditional length distributions differ within any overlap. The 1.229 therefore remains **confounded and undecomposed** — neither certified length-orthogonal nor re-registered as length-carried. Per the registration's own logic, the only remaining route for the condition grain is **Stage 2: length-matched generation** (≥6 independent condition nights, matched on length marginals *and* window-scale statistics, with the content-validity gate). The L2 identity-grain object result (content 0.893 vs silence 0.36) is untouched by any of this.

## 6. Limitations (stated, not buried)

1. **Two independent nights, two geometries.** Cross-fitting across LONO folds is what the registration pinned; with flip-vs-ramp as the only fold pair, mediator transportability was untestable in advance and failed in practice. Stage 2's independent nights would also give the folds homogeneity the current corpus lacks.
2. **Range-overlap was the pre-declared common-support reading.** A mass-overlap (distributional) trimming would be more aggressive — but it was not the declared instantiation, it would leave even fewer windows (the mass overlap is thinner than the range overlap), and switching definitions after seeing the manip check fail would be outcome-driven. Not run; not substituted.
3. **The deconfounded chord's scale** (1.83 > raw 1.23) reflects residual-normalization geometry, not an amplified edge; it is reported because the registration's CI-based KILL branch needs the number on the record, with its uncertified status attached.
4. **Permutation cells are small** (8 cells of 2–4 windows) — the stratified null is honest but coarse; the bootstrap clusters (2 nights × 2 conditions) cannot capture night-level uncertainty. Both are corpus-size facts, which is the finding.
5. The triad's degenerate folds are excluded and flagged (`degenerate_folds`, `n_excluded_degfold` in the JSON), never silently imputed.

## 7. What this buys the dissertation

- The Clock-Split's Stage 1 is **closed with a registered verdict**: INDETERMINATE, mechanism identified, no branch fished. The committee's decisive question — "does the fine edge survive de-confounding?" — has a registered answer: **on the existing corpus, the question is not answerable, and here is exactly why.**
- The silence test's annotation requirement is strengthened: the 1.229 must now be reported with *both* its silence-confound (0.897 vs 0.765) *and* the decomposition's structural failure modes (disjoint support; non-transportable mediator; un-trimmable clock arm).
- Stage 2 is now the **only** registered path for the condition grain, and its design constraints are sharper than in the registration: matching must cover window-scale length statistics (the silence arm's fuel), and nights must share transition geometry or provide enough geometries that the length model is fit within geometry — otherwise the mediator stays non-transportable.
- The identity-grain OBJECT claim needs no defense from this run and receives none; the split verdict of the silence test stands, now with the condition-grain half explicitly marked *undecomposed* rather than merely *confounded*.

**The one-sentence registered claim, answered:** *the length-orthogonal component of the 1.229 is not measurable on the existing corpus; Stage 1 returns INDETERMINATE on three independent registered triggers, and the condition-grain edge remains clock-confounded pending length-matched generation.*

*(Raw numbers: `LENGTH-DECOMPOSITION-2026-08-20.json` — checkpointed stage by stage. Runner: `scripts/length_decomposition.py`, deterministic, CPU, numpy-only analysis. No git operations.)*
