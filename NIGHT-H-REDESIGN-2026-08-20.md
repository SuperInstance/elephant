# Night-H — Ramp-Night Drift-Geometry Redesign (Premise Measurement)

**Filed: 2026-08-20.** Status: the registered runner-up (`prototype/e2-e3-side-by-side.md` §7) — the ramp-night drift-geometry redesign, run on the premise estimator. Script: `scripts/night_h.py` (generates `data/nights/night-H.jsonl`, then drives the **unmodified** `scripts/premise_measurement.py` functions over the extended corpus). No existing scripts modified; no estimator invented; nothing committed.

---

## 0. The one-paragraph verdict

**The band verdict does not move, and the premise still does not clear.** Adding night-H (a 40-message night over the verbatim SEG1/SEG2 banks whose transition is a **ramp** — cynical fraction 0 → 1 across five 8-message blocks — instead of the canonical flip at seq 20) moves the kill number from **0.5599 → 0.5980** (real-only) and **0.4898 → 0.5283** (real+synthetic). The ramp does exactly what it was designed to do on its own transition — night-H's SEG1→SEG2 drift is **0.4951 corpus-sd vs 0.8195** for the A/B/C flip nights, a 40% denominator shrink — but the ratio lands **0.002 short of the 0.6 clear edge** (real-only), and the synthetic-grounded reading stays mid-band. Both verdicts remain **in band: indeterminate**. The registered geometry warning in §7 holds verbatim: *it will land at the edge of the constructible cone like its predecessor.* The premise's best shot has now been taken; it does not clear, and the honest sentence is unchanged: **the premise stays retired, leaning false — not proven false.** The redesign is pro-premise by construction, its best case was always a boundary-condition booking, and this run delivers the boundary — not the booking.

## 1. What was run (registration, verbatim)

From `e2-e3-side-by-side.md` §7:

> **Runner-up (registered): the ramp-night drift-geometry redesign (night-H).** It is the only registered path that could change the premise's own band verdict from INDETERMINATE — subtler transitions shrink the denominator and give the premise its best shot. But it is pro-premise by construction, its best case is a boundary-condition booking (Branch B's outcome, which the registration already anticipates), and the geometry warns it will land at the edge of the constructible cone like its predecessor. Run it only if the committee wants the premise's band verdict adjudicated to the last inch; it cannot produce a decisive kill, and its clear is capped.

**Composition (registered in `scripts/night_h.py`, deterministic):** 40 messages, verbatim SEG1 (warm-earnest) + SEG2 (cynical-banter) banks from `scripts/nights_abc.py`, 20 warm + 20 cynical lines in five 8-message ramp blocks:

| seq | warm lines | cynical lines | cynical fraction |
|-----|-----------|---------------|------------------|
| 0–7 | 8 | 0 | 0.00 |
| 8–15 | 6 | 2 | 0.25 |
| 16–23 | 4 | 4 | 0.50 |
| 24–31 | 2 | 6 | 0.75 |
| 32–39 | 0 | 8 | 1.00 |

Minority lines are spread as evenly as possible within each block (deterministic). **Strata registered with the same convention as A/B/C:** SEG1 = seq 0–19, SEG2 = seq 20–39 — the estimator, the strata rules, and the readers are unchanged; only the *content transition* is subtler (SEG1 is 80% warm / 20% cynical; SEG2 is 20% warm / 80% cynical, vs 100/0 and 0/100 under the flip).

**Roster:** the 6 original occupants only (no new personas) — the premise-measurement real reader set stays exactly the 7 registered readers (6 originals + drifter-from-D). v:2 reader schema (consistent with the E2-era S-nights), additive and append-only (refuses overwrite), deterministic re-run verified byte-identical (stripped of `session_id`):

- `sha256 = c55025aca61eccfc822f63036aca3d147abdcb94bb0b22d4bfb0391bccead343`
- `stripped_md5 = 52e299c05d84b792a4295619fbc20532`

## 2. Continuity check (the estimator must reproduce the registration first)

Before night-H was added, the driver re-ran the registered 5-night corpus through the same functions and reproduced the kill numbers exactly:

| | real-only | real+synthetic |
|---|---|---|
| registered baseline | **0.5599** | **0.4898** |
| this run (reproduced) | **0.5599** | **0.4898** |

## 3. The kill number, before and after night-H

Same estimator (`premise_measurement.py` functions, unmodified), same readers (7 real; 20 with the synthetic-grounded bootstrap, seed 0), same kill band **[0.3, 0.6] corpus-sd**.

| quantity | baseline (5 nights) | with night-H (6 nights) |
|---|---|---|
| corpus_sd | 0.2292 | 0.2291 |
| spread_z (real-only) | 0.4627 | 0.4675 |
| mean drift_z (real-only) | 0.8264 | 0.7817 |
| **ratio (real-only)** | **0.5599** | **0.5980** |
| **verdict (real-only)** | in band: indeterminate | **in band: indeterminate** |
| spread_z (real+synthetic) | 0.4179 | 0.4236 |
| mean drift_z (real+synthetic) | 0.8533 | 0.8017 |
| **ratio (real+synthetic)** | **0.4898** | **0.5283** |
| **verdict (real+synthetic)** | in band: indeterminate | **in band: indeterminate** |
| robustness, ratio_vs_base (real / +synth) | 1.1185 / 0.9779 | 1.1573 / 1.0206 |

**Per-transition drift (mean over readers, corpus-sd):** the ramp's denominator shrink is visible on its own transition:

| transition | baseline corpus | with night-H |
|---|---|---|
| A: SEG1→SEG2 | 0.8195 | 0.8196 |
| B: SEG1→SEG2 | 0.8195 | 0.8196 |
| C: SEG1→SEG2 | 0.8195 | 0.8196 |
| D: pre→post entry | 0.7379 | 0.7380 |
| D-cold: pre→post entry | 0.8799 | 0.8801 |
| **H: SEG1→SEG2 (ramp)** | — | **0.4951** |

**Per-reader drift with night-H (corpus-sd; vs-own-baseline variant):**

| reader | drift | vs-own-baseline |
|---|---|---|
| captain | 0.3428 | 0.1758 |
| poet | 0.3547 | 0.1830 |
| writer | 0.4792 | 0.2470 |
| essayist | 0.6950 | 0.3521 |
| engineer | 0.9411 | 0.4898 |
| drifter | 0.9322 | 0.4661 |
| critic | 1.7268 | 0.9138 |

## 4. Can it clear 0.6? Can it exit the band? — the honest answers

- **Can it clear 0.6 (real-only)?** No — **0.5980**, two-tenths of a percent short of the edge. The ramp shrinks night-H's own transition to 0.4951 (vs 0.8195 under the flip), but the per-reader drift is averaged over all six nights, and the A/B/C/D/D-cold flips still dominate the reader-mean denominator (0.7817). The spread half moved *up* slightly (0.4627 → 0.4675) as baselines absorbed the new night, which eats part of the denominator gain.
- **Can it exit the band?** No. Both readings remain inside [0.3, 0.6] — real-only at 0.5980 (0.002 from the clear edge, 0.298 from the kill edge), real+synthetic at 0.5283. The band verdict is **unchanged: INDETERMINATE** in both arms.
- **Direction of movement:** pro-premise, as registered. The ratio rose in both arms (+0.038 real-only, +0.039 real+synthetic), entirely from the denominator shrink the design was built to produce.

## 5. The honest verdict (pro-premise caveat included)

1. **Pro-premise by construction.** Night-H was registered as *the premise's best shot*: its only mechanism is shrinking the drift denominator. Movement toward clear is therefore expected by construction and cannot be read as independent confirmation of the premise — the direction of travel is designed in, not discovered.
2. **The clear is capped and did not land.** The registration's best case was a boundary-condition booking (Branch B: E2 clears ⇒ grade-dependence). It did not fire: the ratio stops 0.002 below the edge, and the synthetic-grounded reading (the N-discipline arm) sits mid-band. The verdict category did not change.
3. **What the run does establish:** the drift denominator is *geometrically compressible* — halving the sharpness of one transition (0.8195 → 0.4951) moves the reader-mean drift from 0.8264 to 0.7817, and the ratio responds monotonically. The premise's failure is therefore not "drift is irreducible"; it is that even the premise's own preferred geometry cannot lift the ratio clear of the band in this corpus.
4. **N caveat unchanged.** 7 real readers < 10; the ≥10-readers / ≥5-strata-transitions discipline is still met only by the clearly-labeled synthetic-grounded bootstrap — which is the arm that stays farthest from clear.
5. **Bottom line:** the ramp-night redesign adjudicates the band verdict to the last inch, as the registration commissioned — and the last inch says **indeterminate, leaning pro-premise but short of clear**. The premise remains **retired, leaning false — not proven false**; the slope regression (H-reader≡room) remains the decisive test, exactly as §7 ordered.

## 6. Provenance and reproduce

- **Corpus:** `data/nights/night-H.jsonl` (new, append-only; all prior nights untouched byte-for-byte).
- **Generator:** `scripts/night_h.py` — `python3 scripts/night_h.py` (generate + measure + report); `python3 scripts/night_h.py --verify` (determinism re-run check).
- **Estimator:** the **unmodified** functions from `scripts/premise_measurement.py` (`load_night`, `replay_readings`, `corpus_sd`, `fit_readers`, `measure`, `verdict`, `synthesize`), driven over {A, B, C, D, D-cold, H} with the identical flow `main()` uses; continuity reproduced before the H night was added.
- **Machine-readable numbers:** `NIGHT-H-REDESIGN-2026-08-20.json`.
- Nothing committed; no existing file modified.
