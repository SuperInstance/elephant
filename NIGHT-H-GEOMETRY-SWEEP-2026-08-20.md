# Night-H Geometry Sweep — Drift-Geometry Follow-Up (Premise Measurement)

**Filed: 2026-08-20.** Status: follow-up to `NIGHT-H-REDESIGN-2026-08-20.md`. Night-H's ramp landed at **0.5980 real-only — 0.002 short of the 0.6 clear edge**. This sweep asks the registered follow-up question: does ANY drift geometry clear 0.6 or exit the band? Script: `scripts/night_h.py` (extended — adds three geometry variants over the SAME verbatim SEG1/SEG2 banks, cast, roster, strata convention, and the **unmodified** `scripts/premise_measurement.py` estimator). No existing scripts modified; no estimator invented; nothing committed.

---

## 0. The one-paragraph verdict

**Yes — geometries exist that clear 0.6, and the sweep exits the band; and no — this does not rescue the premise.** All three new variants clear the 0.6 edge real-only (H2 multi-stage ramp **0.6084**, H3 oscillation-with-tightening-envelope **0.6076**, H4 long-dwell-plateau **0.6053**), and the cumulative corpus (baseline + all four geometry nights) clears in **both** arms (real-only **0.6988**, real+synthetic **0.6320**). But every fraction of the movement is the registered mechanism — denominator shrink by construction: each variant's own transition drift (0.37–0.42 corpus-sd vs 0.82 for the A/B/C flips) is designed in, not discovered. The single-variant clears are razor-thin (+0.005 to +0.008 over the edge), and the N-discipline arm (real+synthetic) stays **in band for every single variant** — only piling all four pro-premise nights into the corpus at once pushes it over. This is exactly the registration's prediction made literal: *the premise lands at the edge of the constructible cone, and with enough pro-premise geometry the cone's edge can be pushed past 0.6 — by construction.* The band verdict is therefore not "the premise holds"; it is "the ratio is geometry-malleable across the band edge," which is the denominator-divergence finding of the side-by-side (§7) confirmed from the geometry side. The premise remains **retired, leaning false — not proven false**; the slope regression (H-reader≡room) remains the decisive test.

## 1. What was run

Same machinery as the night-H run, extended. Continuity check first: the registered 5-night baseline reproduces exactly (**0.5599** real-only / **0.4898** real+synthetic). Then, for each geometry, the corpus is **baseline (A,B,C,D,D-cold) + that one geometry night** — same estimator, same 7 real readers (6 originals + drifter-from-D), same synthetic-grounded bootstrap (seed 0, 20 readers), same strata (SEG1 = seq 0–19, SEG2 = seq 20–39), same kill band **[0.3, 0.6]** corpus-sd. A cumulative row (baseline + all four geometry nights) is also measured.

The variants (all 40 messages, verbatim 20 warm + 20 cynical bank lines, 6-original roster, minority lines spread evenly within blocks, deterministic, append-only, verified byte-identical on re-run):

| night | geometry | per-block cynical fraction | SEG1 mix (warm/cyn) |
|-------|----------|---------------------------|---------------------|
| H | ramp (registered) | 5×8 blocks: 0, .25, .5, .75, 1 | 80/20 |
| H2 | multi-stage ramp | 8×5 blocks: 0, .2, .4, .4, .6, .6, .8, 1 | 75/25 |
| H3 | oscillation, tightening envelope | 8×5 blocks: 0, 1, .2, .8, .4, .6, .4, .6 (amplitude .5→.1 around 0.5) | 50/50 |
| H4 | long-dwell-plateau | 5×8 blocks: 0, .5, .5, .5, 1 (24-msg plateau at 50/50) | ~70/30 |

## 2. The sweep table

| geometry | own transition drift (corpus-sd) | ratio (real-only) | ratio (real+synth) | verdict (real-only) |
|---|---|---|---|---|
| baseline (A,B,C,D,D-cold; flip nights) | 0.8196 (flip) | **0.5599** | **0.4898** | in band: indeterminate |
| +H (ramp) | 0.4951 | **0.5980** | **0.5283** | in band: indeterminate |
| +H2 (multi-stage ramp) | 0.4116 | **0.6084** | **0.5397** | **above band: premise holds** |
| +H3 (oscillation, tightening envelope) | 0.3692 | **0.6076** | **0.5360** | **above band: premise holds** |
| +H4 (long-dwell-plateau) | 0.4208 | **0.6053** | **0.5357** | **above band: premise holds** |
| +H+H2+H3+H4 (cumulative) | — | **0.6988** | **0.6320** | **above band: premise holds (both arms)** |

**Does any geometry clear 0.6? YES** — all three new variants, real-only; the cumulative corpus clears in both arms. **Does any exit the band? YES** — upward, in the pro-premise direction only. No geometry moves kill-ward.

Decomposition of a representative clear (H2): spread 0.4643 / drift 0.7632 = 0.6084. Against the baseline (0.4627 / 0.8264), the numerator is essentially flat (+0.0016); the entire movement is the denominator (−0.0632), of which the variant's own transition contributes −0.41 relative to a flip night and the rest is averaging. The mechanism is pure denominator shrink, exactly as registered.

## 3. Why the clears do not rescue the premise (the honest caveats)

1. **Pro-premise by construction — the direction was designed in.** Every variant's only mechanism is shrinking the drift denominator. A clear produced this way is not independent confirmation of the premise; it is the registration's warning ("it will land at the edge of the constructible cone") verified. The sweep measures the *malleability of the ratio under geometry*, not the premise.
2. **The single-variant clears are razor-thin.** +0.0053 to +0.0084 over the 0.6 edge — inside any reasonable numerical slack, and each is one night's worth of denominator dilution against five flip nights that still dominate the reader-mean drift (0.76 vs the variant's own 0.37–0.42).
3. **The N-discipline arm does not clear for any single variant.** Real+synthetic stays mid-band (0.5357–0.5397) for H2/H3/H4. The ≥10-readers discipline is met only by the bootstrap arm, and that arm says indeterminate for every individually-added geometry. Only the cumulative pile-up — four pro-premise nights added at once, a corpus-construction choice, not a field observation — pushes it to 0.6320.
4. **H3 is the reductio.** The oscillation-with-tightening-envelope makes SEG1 and SEG2 both 50/50: the regime change is real in the text but structurally invisible to a SEG-split estimator. Its "clear" demonstrates that with full control of transition geometry the denominator can be driven toward zero and the ratio toward infinity — the ratio is a property of the geometry choice as much as of the readers. That is the side-by-side's denominator-divergence finding (E2 drift 0.748 vs E3 drift 3.46 under two frames) reproduced from the design side: **the denominator is not instrument-independent, and now it is shown to not be geometry-independent either.**
5. **The monotone dose-response is the finding.** Own-transition drift 0.82 → 0.50 → 0.42 → 0.41 → 0.37 maps monotonically to ratio 0.5599 → 0.5980 → 0.6053 → 0.6084 → 0.6076 (real-only). The ratio is a smooth function of constructed transition sharpness near the band edge — meaning a boundary-hugging truth (0.56–0.61) cannot be adjudicated by adding geometry nights at all; each new night just re-poses the estimand.

## 4. Bottom line

- **Task question answered:** yes, drift geometries exist that clear 0.6 (three of three new variants, real-only) and the band is exited upward (cumulatively in both arms). No geometry exits kill-ward.
- **What it means:** the clear is constructible but not earned — it is the premise's best-case boundary-condition booking (Branch B's shape) arrived at by construction, with the N-discipline arm dissenting on every single-variant row. The honest sentence is unchanged: **the premise stays retired, leaning false — not proven false.** What the sweep adds is not a verdict change but a measured fact: the kill number is geometry-malleable across the 0.6 edge, so no further drift-geometry redesign can adjudicate it. The decisive test remains the registered slope regression (H-reader≡room).

## 5. Provenance and reproduce

- **Corpus:** `data/nights/night-H.jsonl` (pre-existing, untouched — registered md5 `52e299c0…` reproduced), plus new append-only `night-H2.jsonl`, `night-H3.jsonl`, `night-H4.jsonl`. All prior nights untouched byte-for-byte.
- **Generator:** `scripts/night_h.py` — `python3 scripts/night_h.py` (generate + sweep + report); `python3 scripts/night_h.py --verify` (determinism re-run check, all four nights OK).
- **Estimator:** the **unmodified** functions from `scripts/premise_measurement.py`, driven over each 6-night corpus with the identical flow `main()` uses; baseline continuity reproduced (0.5599 / 0.4898) before any variant was measured.
- **Machine-readable numbers:** `NIGHT-H-GEOMETRY-SWEEP-2026-08-20.json`.
- Nothing committed; no existing file modified.
