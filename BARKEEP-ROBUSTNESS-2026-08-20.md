# Barkeep-Excluded Robustness — E2 Antecedent Test

**Dated: 2026-08-20.** The registered cheap-prerequisite robustness check on the E2 field numbers (REPORT-2026-08-19). Trigger: the per-reader table flagged `barkeep` (seeded field-distribution draw, critic archetype, nights S1+S3 only, n_nights=2) with drift **2.3852** corpus-sd vs reader-median 0.5084 — a possible drag on the drift denominator of the primary ratio.

## Method

Everything frozen from the field run; only the reader set changes (15 → 14). Reuses the committed instrument verbatim (`scripts/e2_instrument.py` via `scripts/e2_field.field_readers`): same 9 primary nights, same corpus_sd = 0.2367 (reader-independent — computed from room logs, unchanged by exclusion), canonical presence, bootstrap seed 20260819, B=2000. v:2 night logs verified against the SHA-pinned manifest; the 15-reader rerun reproduces the committed field numbers **exactly** (ratio, CI, ICC; |diff| = 0), which guards the v:1 nights by equality. Script: `scripts/barkeep_robustness.py` (numpy-only, read-only, writes nothing).

## Numbers (field, E-cont-canonical)

| quantity | 15 readers (committed) | 14 readers (barkeep out) |
|---|---|---|
| primary ratio | **0.6088** = 0.4556 / 0.7483 | **0.7339** = 0.4634 / 0.6314 |
| ratio 95% CI | [0.3710, 0.9210] | [0.4888, 1.0142] |
| E-seg (reported) | 0.6853 [0.394, 0.963] | 0.8080 [0.4725, 1.0235] |
| spread / mean drift | 0.4556 / 0.7483 | 0.4634 / 0.6314 |
| baseline ICC | **0.7714** [0.667, 0.810] | **0.7569** [0.6597, 0.7984] |
| verdict | INDETERMINATE (CI touches band) | INDETERMINATE (CI touches band) |
| kill condition | does not fire | does not fire (CI upper 1.014 > 0.6) |

Per-dial ICC moves are small except joke_landing (0.6485 → 0.5204 — barkeep carried real between-reader structure on that dial) and cynicism (0.6406 → 0.6608); the aggregate stays entirely above the 0.265 ICC-equivalent threshold.

## Does the shade of INDETERMINATE change?

**Yes — from "edge-hugging" to "clearly above, CI still touching."** The 15-reader point sat +0.0088 from the 0.6 edge; the 14-reader point sits **+0.1339** above it, but the CI lower bound (0.4888) is still below 0.6, so the registered verdict and the kill condition are unchanged. Mechanism: the drag was entirely denominator — mean drift de-inflates 0.7483 → 0.6314 while spread is essentially unmoved (0.4556 → 0.4634). Barkeep is a high-drift reader, not a spread outlier.

One shade-level consequence for §5 of the report: the registered power formula now yields N ≈ **54** readers (half-width 0.2627, edge-distance 0.1339) instead of ≈ 14,533 — the "no feasible N adjudicates" conclusion was substantially an artifact of the barkeep-dragged point hugging the edge. With barkeep out, decisive measurement at this design point looks feasible in principle.

## Honesty

This can only reweight the shade, not adjudicate: the exclusion was chosen by inspecting the field results (post hoc), barkeep's 2-night drift estimate is the noisiest kind in the roster (mean of two transitions), and the registered verdict belongs to the 15-reader run, which stands — INDETERMINATE, R7 retirement intact, kill condition not firing. The right reading: the primary conclusion (baselines real and stable, spread-vs-drift unresolvable at this design point) survives the flagged outlier; the outlier was responsible for the point estimate's unnerving proximity to the band edge, not for the verdict.

## Reproduce

```
cd /home/eileen/projects/elephant
python3 scripts/barkeep_robustness.py
```
