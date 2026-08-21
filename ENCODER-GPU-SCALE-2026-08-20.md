# Encoder GPU Scale — CUDA port + same-cast corpus scale (2026-08-20)

**Date:** 2026-08-20 (America/Anchorage), 16:10–17:20 AKDT
**Machine:** WSL2 host `eileen`, **NVIDIA GeForce RTX 4050 Laptop GPU (6.4 GB
VRAM)**, torch 2.13.0+cu130, CUDA verified, numpy 2.4.6.
**Status:** COMPLETE — 12 CUDA training runs (4 corpora × 3 seeds) + 3 CPU
runs for the speedup measurement.
**Deliverables:** `scripts/encoder_gpu_scale.py` (the CUDA port + corpus-scale
runner), this report, artifacts under
`checkpoints/contrast/gpu_scale/<corpus>/` (per-seed crash-safe JSON +
model checkpoints + merged `results_{cuda,cpu}.json`). **No git commit** (per
dispatch).

## Why this run exists

`ENCODER-GENERALIZATION-UPGRADE-2026-08-20.md` established: **more identity
data clears the held-out floor** — with 4 same-cast training nights the
held-out fine gap on UNSEEN tap-night-3/4 goes 3/3 seeds above the 0.05
noise floor (mean 0.1099), vs the 2-night baseline FAIL (mean 0.0681, 2/3).
This run GPU-scales that probe: (1) port the contrastive training loop to
CUDA, (2) assemble MORE same-cast training nights (target 4–6) from the
unused tap-trades material, (3) re-run the held-out generalization at scale,
3 seeds, same 0.05 floor, same schedule, (4) measure GPU vs CPU speedup and
ask whether the gain is still "more data helps" rather than
capacity-limited.

The registered scripts pin `CUDA_VISIBLE_DEVICES=""` at import (the old
GPU-pinning bug era); this port deliberately does not — it is the GPU lane.
All probe metrics (fine gap, kNN room discrimination, speaker-heldout,
venue-discrimination) are the registered numpy definitions, byte-identical
to the harness. The held-out set is **untouched**: tap-night-3 + tap-night-4,
the same 18 clips, in every corpus.

## Corpora (dose axis: same-cast training nights)

All extras are previously-unused 2026-08-16 tap-trades filings, same cast as
the held-out nights (LUCINEER/WELDER/CARPENTER/SHIPWRIGHT/MASON/COMPOSITE/
WESLEY). NOT used: joint-map meta-analysis docs, and the improv/speed-dating
nights (different cast — fleet-model cast, not the trades cast).

| corpus | same-cast train nights | what the nights are | train rooms / clips |
|--------|------------------------|---------------------|---------------------|
| `baseline` | 2 | evening-1, evening-2 (+ open-mic always) | 17 / 1097 |
| `more` | 4 | + questions/, sequels/ merged | 19 / 1111 |
| `scale` | 5 | + tap-pieces (the six source trade monologues: carpenter, mason, shipwright, welder, composite, wesley-the-room) | 20 / 1116 |
| `scale-split` | 6 | + sequels split into its two real night filings (sequels/ = n1, sequels-night2/ = n2) | 21 / 1115 |

Held-out: tap-night-3 + tap-night-4, 18 clips, in all four. The `more`
corpus is byte-identical to the registered `more-nights` (19 rooms / 1111
clips) — the reproducibility anchor.

## (0) CUDA port verification — GPU reproduces the registered CPU numbers

Same code path, only the device changed. Per-seed held-out gaps, GPU vs the
registered CPU runs (upgrade run):

| corpus | seed0 | seed1 | seed2 | mean | vs registered |
|--------|-------|-------|-------|------|---------------|
| baseline GPU | 0.0710 | 0.1039 | 0.0287 | **0.0679** | registered 0.0710/0.1023/0.0309 = **0.0681** (Δ ≤ 0.0036/seed) |
| more GPU | 0.1186 | 0.0688 | 0.1399 | **0.1091** | registered 0.1201/0.0742/0.1355 = **0.1099** (Δ ≤ 0.0054/seed) |

Both verdicts reproduce exactly: baseline **2/3 FAIL** (seed 2 below floor on
GPU too, 0.0287), more **3/3 PASS**. The GPU port is faithful; the small
per-seed deltas are the pre-existing run-to-run noise band (±0.005–0.007,
the same band seen between the two registered CPU runs).

## (1) Held-out generalization at scale (all CUDA, 200×60 schedule, 3 seeds)

| corpus | nights | held-out fine gaps s0/s1/s2 | **mean** | 3/3 > 0.05? | train gap mean | ho disc | sp-heldout | venue |
|--------|--------|------------------------------|----------|--------------|----------------|---------|------------|-------|
| baseline | 2 | 0.0710 / 0.1039 / 0.0287 | **0.0679** | ❌ 2/3 | 0.452 | 0.556 | 0.611 | 0.685 |
| more | 4 | 0.1186 / 0.0688 / 0.1399 | **0.1091** | ✅ 3/3 | 0.444 | 0.704 | 0.667 | 0.574 |
| scale | 5 | 0.1439 / 0.1049 / 0.1429 | **0.1306** | ✅ 3/3 | 0.437 | 0.611 | 0.630 | 0.611 |
| scale-split | 6 | 0.1725 / 0.1645 / 0.1528 | **0.1633** | ✅ 3/3 | 0.392 | 0.556 | 0.630 | 0.648 |

Frozen-trunk anchors (identical in every corpus): train gap 0.0161, held-out
gap **0.0019** on nights 3+4 — nearly all of the learned separation is
learned, not pre-existing.

**Dose-response is monotone:** 2→4→5→6 same-cast training nights →
0.0679 → 0.1091 → 0.1306 → 0.1633. Every added night raises the mean, by
~+0.02–0.03 per night, with no sign of flattening at 6. The previously
failing seed 2 heals monotonically: 0.0287 → 0.1399 → 0.1429 → 0.1528.
At 6 nights the mean is **2.4× the 2-night mean and 3.3× the 0.05 floor**;
the weakest seed at scale (0.1049) still clears the floor by 2×.

## (2) GPU vs CPU speedup (identical code path, scale corpus, 3 seeds)

Per-seed wall time (training loop only, 200×60 = 12,000 steps):

| seed | GPU (cuda) | CPU | speedup |
|------|-----------|-----|---------|
| 0 | 64.8 s (185 steps/s) | 204.7 s (59 steps/s) | 3.16× |
| 1 | 67.9 s (177 steps/s) | 219.5 s (55 steps/s) | 3.23× |
| 2 | 67.5 s (178 steps/s) | 267.4 s (45 steps/s) | 3.96× |
| **mean** | **66.7 s (180 steps/s)** | **230.5 s (53 steps/s)** | **3.46×** |

- **GPU ≈ 3.46× faster per seed** (wall), ~3.4× on steps/s.
- **Peak VRAM 0.17 GB of 6.4 GB** — the entire corpus ids tensor (1116×256
  longs ≈ 2.3 MB), the small model, and the largest contrast batch (~900
  clips when wesley-stream is sampled → 900×900 sim matrix ≈ 3.2 MB) all fit
  trivially. No VRAM-driven batch shrinking was needed; the registered batch
  definition (all clips of 2–3 rooms) runs unchanged.
- Whole probe: 12 CUDA training runs ≈ **13 min wall**; the same probe on
  CPU would be ≈ 46 min. The earlier 5-mode upgrade run took ~75 min CPU;
  this 4-corpus scale probe cost a fraction of that.
- Honest framing: 3.46× is modest because the model is tiny (d=64) and the
  batches are small GEMMs — a laptop 4050 cannot saturate on this workload,
  and torch's default 12 CPU threads keep CPU competitive. The GPU's real
  win at this scale is wall-time per seed plus freeing the CPU for parallel
  seed dispatch; the speedup would be far larger on bigger trunks/batches.

## (3) Is the gain still "more data helps" — or capacity-limited?

**Verdict: still data-limited, not capacity-limited.** Evidence:

1. **Monotone dose-response with a FIXED model** (same 64-dim trunk, same
   schedule, same objective, same seeds across all corpora). If capacity
   were the limit, held-out gap would flatten or the model would overfit;
   instead every added same-cast night raises the mean held-out gap
   (0.068 → 0.109 → 0.131 → 0.163) with no saturation at 6 nights.
2. **No overfit signature.** Train-corpus gap does NOT collapse toward 1.0
   as nights are added; it drifts mildly DOWN (0.452 → 0.392) — more
   within-class structure with more identity data, not memorization. The
   spread hinge holds room geometry; mean train discrimination stays
   ~0.93–0.96 throughout.
3. **The failure mode itself is data-healed.** The exact failing seed at 2
   nights (seed 2, 0.0287 < floor) passes at 4, 5, and 6 nights with growing
   margin (0.1399 / 0.1429 / 0.1528) — the signature of a signal that exists
   but needs more same-cast nights to separate from room-level memorization,
   exactly the upgrade run's diagnosis.
4. **Held-out separation is learned, not carried by the trunk.** The frozen
   v2 trunk sits at 0.0019 on nights 3+4 in every corpus; at 6 nights the
   head adds +0.161 of margin over trunk.

## Honest caveats (carry into thesis text)

- **scale-split's 6th "night" is the softest increment**: the sequels
  filing split into its two real night documents (sequels/ + sequels-night2/
  — the joint-map's own reading that these are "the same argument run a
  second night"). Defensible, but it is a filing split, not two wholly
  independent nights. The cleanest new data point is **scale (5 nights)**.
- **tap-pieces (5th night)** are the six source trade monologues — same
  cast, same venue family, previously unused; real new identity text.
- **Noise band**: same-seed GPU-vs-CPU deltas up to ~0.005–0.007 (consistent
  with the registered CPU run-to-run band). The 0.0679→0.1633 trend is
  ~15–20× the band and robust; single-corpus means carry ±0.005.
- Room-discrimination on the held-out clips stays moderate (0.56–0.70) at
  every dose — fine gap is the registered bar and it is the number that
  scales; kNN disc is the secondary reading.

## Verdict

1. **(a) Held-out generalization at scale vs 0.1099 @ 4 nights:**
   5 nights → **0.1306**, 6 nights → **0.1633** (both 3/3 PASS). The
   more-data result is not a ceiling artifact — it keeps climbing.
2. **(b) GPU vs CPU speedup: 3.46× per seed** (66.7 s vs 230.5 s mean,
   scale corpus, 3 seeds, identical code path); peak VRAM 0.17 GB of 6.4 GB;
   the whole 12-run probe ≈ 13 min on GPU vs ≈ 46 min CPU-equivalent.
3. **(c) The gain is still "more data helps", NOT capacity-limited** —
   monotone dose-response with a fixed model, no overfit collapse, and the
   2-night failing seed heals monotonically with added same-cast nights.

**One-line answer: porting to CUDA reproduced the registered numbers to
±0.005, scaled the corpus to 5–6 same-cast training nights, and the held-out
fine gap keeps rising (0.068 → 0.109 → 0.131 → 0.163 mean across 2/4/5/6
nights, 3/3 PASS at ≥4) — the encoder's room-identity generalization is
data-limited, and more identity data remains the fix at scale.**

## Artifacts (all additive; registered checkpoints untouched)

- `scripts/encoder_gpu_scale.py` — CUDA-port + scale runner
  (`--corpus baseline|more|scale|scale-split`, `--seeds`, `--device
  auto|cuda|cpu`, `--epochs`; per-seed crash-safe JSON + model checkpoints;
  merged results; peak-VRAM tracking; CPU lane for the speedup measurement).
- `checkpoints/contrast/gpu_scale/<corpus>/results_{cuda,cpu}.json` +
  `results_seed{k}_{device}.json` + `model_seed{k}_{device}.pt`.
- This report.

**Runtime notes:** GPU runs 59–73 s/seed (~180–200 steps/s, 0.17 GB peak).
CPU runs 205–267 s/seed (~45–59 steps/s, torch default 12 threads, no
OMP_NUM_THREADS override). The GLM-5.3 dispatch that preceded this lane had
completed all CUDA runs and one CPU seed before timing out; this lane
re-ran the CPU lane to completion (3/3 seeds, same script), verified the
corpus composition per dose, verified merged-vs-per-seed consistency, and
confirmed the GPU numbers reproduce the registered CPU runs.
