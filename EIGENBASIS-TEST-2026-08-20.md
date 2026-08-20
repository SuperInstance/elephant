# Eigenbasis Diagonalization Test (registered) — 2026-08-20

**Registered in** `research/topic.md` fold 2: *diagonalize the dial-covariance matrix over the E2 per-reader data; if the top eigen-dimensions align with the ICC-reliable subspace, the reliability finding is eigenbasis conservation.*

## Matrix (choice stated per the task)

**PRIMARY: between-reader covariance of per-reader baselines.** `baseline_R = mean over R's nights of (per-(reader,night) median reading, night-mean removed)`, built through the registered E2 instrument verbatim (`scripts/e2_instrument.Measurement`, canonical presence, 9 primary nights, 15 field readers) — the ICC's per-dial sigma^2_between is exactly the diagonal of this matrix (asserted numerically). **Why between-reader:** the reliability finding is a claim about which dials carry *reader identity*; its variance object is the between-reader covariance. Pooled across-readings covariance mixes in schedule and within-reader variance (non-identity) and is reported only as a contrast.

Corpus: 9 primary nights (A, D, D-cold, S1-S5), corpus_sd = 0.2367, 15 readers; published ICC reproduced exactly (aggregate 0.7714).

## Eigensystem of C_between

| j | eigenvalue | var-share | cum | reliable-mass | dominant loadings (|v|>=.30) |
|---|-----------|-----------|-----|---------------|--------------------------|
| 0 | 0.057530 | 0.8039 | 0.8039 | 0.879 | mood -0.92, cynicism +0.35 |
| 1 | 0.009254 | 0.1293 | 0.9332 | 0.932 | earnestness +0.96 |
| 2 | 0.002946 | 0.0412 | 0.9744 | 0.300 | mood -0.36, cynicism -0.82 |
| 3 | 0.001130 | 0.0158 | 0.9902 | 0.576 | volume +0.44, cynicism +0.41, joke_landing -0.51, presence +0.62 |
| 4 | 0.000405 | 0.0057 | 0.9958 | 0.315 | joke_landing +0.83, presence +0.47 |
| 5 | 0.000299 | 0.0042 | 1.0000 | 0.998 | volume -0.83, presence +0.55 |
| 6 | 0.000000 | 0.0000 | 1.0000 | 0.000 | panic +1.00 |

## Alignment with the ICC-reliable subspace (mood/volume/earnestness/presence)

cos(top-k eigenspace, reliable indicator): k=1: 0.5633, k=2: 0.7181, k=3: 0.7214, k=4: 0.9040, k=5: 0.9833, k=6: 1.0000, k=7: 1.0000

| dial | ICC | sigma^2_b | w_d top-4 mass (rank) | h^2_d variance captured | reliable |
|------|-----|-----------|-----------------------|-----------------------------|----------|
| mood | 0.9652 | 0.049549 | 0.9916 (2) | 0.9999 | yes |
| volume | 0.9766 | 0.000837 | 0.2390 (6) | 0.7188 | yes |
| earnestness | 0.9533 | 0.008710 | 0.9787 (3) | 0.9991 | yes |
| cynicism | 0.6406 | 0.009503 | 0.9973 (1) | 0.9999 | no |
| joke_landing | 0.6485 | 0.000907 | 0.3155 (5) | 0.6944 | no |
| panic | 0.3020 | 0.000000 | 0.0000 (7) | 0.5032 | no |
| presence | 0.9138 | 0.002058 | 0.4779 (4) | 0.9130 | yes |

**Exact subset test** (all 35 coordinate 4-subspaces vs the top-4 eigenspace): reliable subset ranks **3/35** (exact p = 0.0857, 3 subsets at least as aligned); most-aligned subset: mood+earnestness+cynicism+presence (cos 0.9299) vs reliable 0.9040.

Correlations across dials: ICC vs sigma^2_b pearson +0.3808 / spearman +0.2500; ICC vs w_d pearson +0.5033 / spearman +0.1071.

**Robustness** (correlation-matrix version): cos_4 = 0.7881, rank 4/35. 
**Contrast** (pooled covariance across all 2686 canonical readings): cos_4 = 0.7793, rank 16/35 — the between-reader isolation is what carries (or fails to carry) the alignment.

**Within-reader sigma^2_w (backed out of ICC and sigma^2_b):** mood 0.0018, volume 0.0000, earnestness 0.0004, cynicism 0.0053, joke_landing 0.0005, panic 0.0000, presence 0.0002

## Verdict

**PARTIAL eigenbasis conservation: top eigenspace leans reliable but does not coincide with it** — cos_4 = 0.9040 (threshold 0.95), reliable-subset rank 3/35 (exact p = 0.0857).
