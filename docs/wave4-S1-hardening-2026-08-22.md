# WAVE-4 S1 HARDENING SWEEP — DESIGN GATE FAILED (P does not fire at α=1)

**RUN — S1, UNSEALED. Filed 2026-08-22.** Executed against
docs/wave4-registration-draft-2026-08-22.md §1.2 (design gate), §1.4
(carrier-purity guard, ICC calibration), §4 row 3 (S1 obligations).
Fiber v4 implementation at commit 5b8af52 (`--fiber v4`). Nothing sealed;
nothing in data/wave3/** was read or written.

---

## 1. VERDICT (the kill test, pre-stated)

**GATE: FAIL — the fiber v4 rework does NOT deliver the collapse signature
to P. Per the pre-stated rule (§1.2): the rework failed and must not be
registered. No redesign attempted here; goes to Casey.**

| gate clause | pre-stated bar | result | verdict |
|---|---|---|---|
| P fires at α=1 | P_trans < 0.5×P_rest | P_trans 0.9754, P_rest 0.9948 — holds (ratio 0.981); mechanism_kill False | **FAIL** |
| A stays α-blind | gap α1-vs-α0 ≤ wave-3 within-pair envelope (median ~0.011, max ~0.085) | gap = 0.0333 (A 0.500 → 0.467) | PASS |
| α=0 instrument parity | P 0.99-class, A fires (p≈0) | P_trans 0.9756 / P_rest 0.9947 (holds), A fires (A=0.500, p=0.0398) | PASS |

Headline diagnostic (for Casey, not a redesign): **α is demonstrably live in
the fiber — ICC declines 0.885 (α=0) → 0.744 (α=1), exactly the §1.4
predicted direction — yet P moves not at all** (P_trans 0.9756 vs 0.9754;
the two corpora are statistically indistinguishable on P). The wobble carrier
moves the targets (reader constants ride the room-night w_ar) but does not
decorrelate the pre/post offset cosines the P leg measures. The draft's own
flagged risk (§1.4: "decorrelation magnitude depends on wobble persistence
vs window" — AR_PHI=0.9 wobble adjacent-window persistence vs the trailing
W=12 windowed estimator) is the natural suspect, as is leg_P's roster-mean
centering (a common-mode moving target subtracts out of offsets relative to
the per-window roster mean). Diagnosis stops here per the kill rule.

## 2. Setup (provenance)

- Pilots: **UNSEALED**, 5 nights each (T1/T2/T3/T4a signal + T9 null — the
  canonical families, registered W2 strata verbatim through the G5 adapter),
  canonical 6-reader ATTENDANCE roster (+ staged entrant on T4a).
- Generated as a **G13 matched pair**: shared `--pair-seed 20260822`
  (identical room paths, rosters, authors, κ(t); pair members' targets
  differ ONLY through α), `--fiber v4`, seed 20260822 (the §1.6 fresh
  master seed — no wave-3 stream overlap). Driver:
  `scripts/wave4_s1_pilots.py`; corpora `data/wave4-pilots/{a0,a1}/`
  (determinism re-run: all stripped-md5 identical); legs filed unmodified
  via `scripts/wave3_s3_legs.py` → `data/wave4-pilots/legs/`.
- Legs: registered A/D/P/S (premise_band_movers, seeds 20260821/+2/+4)
  through RiverbedMeasurement; canonical W=12 primary channel, W∈{8,16}
  sensitivity, presence canonical+actual. Adjudication JSON:
  `data/wave4-pilots/wave4-S1-adjudication.json`.

## 3. Leg numbers (W=12 canonical primary)

| corpus | P_trans | P_rest | P holds @½ | mechanism_kill | A (n_events) | A p | corpus_sd |
|---|---|---|---|---|---|---|---|
| α=0 (a0) | 0.9756 | 0.9947 | yes | no | 0.500 (16) | 0.0398 | 0.2476 |
| α=1 (a1) | 0.9754 | 0.9948 | yes | no | 0.467 (15) | 0.0715 | 0.2476 |

W=8 sensitivity identical in shape (a0: 0.9811/0.9979; a1: 0.9824/0.9980 —
no kill). W=16: transition windows do not fit on the pilot night lengths
(P_trans None — events dropped, consistent with the registered drop rule).

Carrier-purity guard detail: the A gap (0.0333) sits inside the wave-3
within-pair envelope (≤ max 0.085) but well above its median (0.011); at
α=1 A's p drifts to 0.0715 (above the 0.05 fire bar on this 5-night pilot —
event-count-poor, not a leak finding). No corpus pair is separable on A at
envelope resolution; the guard passes, noted not absorbed.

## 4. ICC calibration (§1.4 — for the S2 freeze, reported despite the kill)

| corpus | ICC aggregate | per-dial range | vs G6 band [0.60, 0.80] |
|---|---|---|---|
| α=0 | **0.885** | .77–.94 | above band (matches wave-3 instrument .73–.88 upper edge) |
| α=1 | **0.744** | .55–.87 | inside band |

The pre-registered **decline prediction lands** (0.885 → 0.744, −0.141):
reader constants ride the nightly-fresh room-carried w_ar exactly as §1.4
predicted — the α=1 fiber IS doing what it was built to do at the anchor
level. What fails is the delivery to P, not the mechanism. If a future
rework re-runs this sweep, the ICC band question reopens; nothing here is
frozen.

## 5. S1 obligations (§4 row 3)

1. **Replay-parity v2** — already present as unit tests at 5b8af52
   (tests/test_riverbed_generator.py TestFiberV4):
   `test_alpha0_replay_v2_from_logged_target` reconstructs every
   `field_eff_to_reader` bit-for-bit from logged rows + `lens_now.target_now`
   (per-t target logging); `test_alpha0_structural_identity_with_v3` and
   `test_alpha1_common_moving_target_from_room_stream` pin the §1.1/§1.2
   structure. Re-verified green this run. **PASS.**
2. **q-rule unit tests** — added (tests/test_calibration_harness.py
   TestQRule, 3 tests): common rigid step ⇒ `uninformative` (the q-rule
   removes the common component — the wave-3 guard's no-op fix); per-reader
   differential step ⇒ `persistence_violated` (q_trans > 2·q_rest + 0.02);
   calibration-corpus run ⇒ well-formed. **PASS.**
3. **kl_sym unit tests** — `kl_sym` implemented in elephant/vmf.py (REG-3
   §2.3: symmetric vMF divergence from the A₇ machinery;
   kl_sym(a,b) = KL(a‖b)+KL(b‖a), log C₇ via scipy ive, log-space) + 4 unit
   tests (tests/test_vmf.py): zero on identical fits, symmetric + positive,
   **fires on pure concentration change with d_mu = 0** (the rigidity blind
   spot), monotone in angular separation. **PASS.**
4. Full suites green: tests/{test_vmf, test_calibration_harness,
   test_riverbed_generator}.py — **88 passed**.

## 6. Consequences

- The §1 rework **must not be registered** (pre-stated kill). §1.5/§2
  clauses that depend on the α-gradient (2AFC ordering, P monotone ↓,
  REG-2's S-leg on the new fiber) are **moot** on this design; REG-3's
  kl_sym edge and the q-rule survive as estimator-side gear independent of
  the fiber.
- No corpus was sealed; no wave-3 artifact touched. Pilots and this doc are
  the complete record. **Escalated to Casey per the kill rule.**

## Provenance

Read: docs/wave4-registration-draft-2026-08-22.md (§1–§4, gate + §1.4 +
obligations); scripts/riverbed_generator.py (fiber v4 block, self-tests,
generate_wave/G13 streams); scripts/wave3_s3_legs.py + scripts/
riverbed_adapter.py (interfaces verified before wiring: archetype_labels,
leg_S/leg_D signatures, manifest roster handling); scripts/
premise_band_movers.py (leg_A/leg_P return contracts); scripts/
calibration_harness.py (q_rule); elephant/vmf.py (A₇). The aborted prior
attempt left NOTHING (git status clean; no data/wave4-pilots/, no prior
wave4-S1 doc — verified before starting). Wave-3 envelope numbers (median
~0.011, max ~0.085) taken from the task brief / draft §1.4 — data/wave3/**
never read.

Written: this document; scripts/wave4_s1_pilots.py; elephant/vmf.py
(kl_sym added; nothing else touched); tests/test_vmf.py + tests/
test_calibration_harness.py (appended tests); data/wave4-pilots/**
(generated corpora, legs, adjudication JSON).
