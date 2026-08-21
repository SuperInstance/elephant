# Riverbed S1 hardening run-note — 2026-08-21

Executed S1 of the wave-3 generation-corpus plan
(`memory/wave3-generation-plan-2026-08-21.md`): generator gaps G1, G3, G4,
G5, G13 + self-test extensions G7, G9. G2 (engine-native arm) left as a
documented extension point per plan (separate registration addendum).

Everything here is PRE-registration (plan §4 S2 has not committed the
addendum); calibration decisions below are therefore legitimate inputs to
S2's frozen thresholds, and deviations are flagged for it.

## Per-gap changes

**G1 — entrants present from t=0 (fixed).** Families with `entry_seqs`
(T4a/T4b/T5/T5c) now STAGE the drifter exactly as the engine does on the
filed corpus (positions measured on `night-{T4a,T4b,T5,T5c}.jsonl`):
declared in `session_open.staged_entries` (never the roster), omitted from
every `readers` block before the entry seq, present from it with
`entry_mode` "staged-cold", first speak AT the entry seq, engine speak
positions mirrored (`ENTRANT_SPEAKS`). Manifest records
`staged_entrant/entry_seq/entrant_is_attendance` (T4a/T4b attendance =
True, T5/T5c warmth-content-only — the FIELD_NIGHTS_W2 rule). The
NaN-before-entry convention now fires: `logged_readings` returns
`n − entry_seq` rows; `night_windows` lacks the entrant in pre-entry
windows (tested).

**G3 — blinding (added).** `--blind[=CORPUS-ID]`: redacted manifest
(SEALED_FIELDS: branch/alpha/ou_phi/kappa_R/redraw/null_mode/seed/
pair_seed all withheld; tags = opaque `rb-<corpus-id>-<family>`; design
facts stay — they are branch-free) + sealed sidecar bound by sha256 both
ways (manifest pins the sidecar's sha256; the sidecar pins every night
file's sha256). `--unblind <sealed.json>` verifies the seal and prints the
withheld parameters; any tamper (sidecar, manifest, night) raises.
Determinism re-run unaffected. Under `--blind`, `--tag-prefix` is ignored
(the opaque id enforces the seal).

**G4 — parameterized gate (added).** `scripts/riverbed_wave_gate.py`
implements plan §1.4.1 on a corpus's OWN numbers: roster == designed
ATTENDANCE; sha256 integrity; in-log G1 entry discipline; determinism
flags; warmth-vs-schedule with cumulative-fit lag accounted; corpus_sd
computed from the corpus itself (finite/>0, recorded as its
normalization — the field's 0.2367 and filed ladder are never imposed;
regression-tested); a-priori x-design Sxx ≥ 0.19 (design x from the
manifest schedule + attendance = 0.19707 exactly, the §2 matrix's own
value; realized-x Sxx recorded as a guard observation only);
≥3 nights × 21 readers; null-night present. Exit 1 on any failure.
`stage2_wave_gate.py` untouched (the registered field gate).

**G5 — Measurement adapter (added).** `scripts/riverbed_adapter.py`:
`NightFromFile` (registered Night load semantics over an explicit
path+strata), canonical-family strata verbatim from `W2_NIGHTS` /
derived for custom families, `wave_attendance`/`wave_cold` (FIELD_NIGHTS_W2
+ COLD_ENTRY_W2 semantics from the manifest), and
`RiverbedMeasurement(Measurement)` — the registered estimators run
UNMODIFIED with only night construction redirected. `load_wave(dir)` is
the one-call bridge; `build_measurement` the reusable builder. Blind-corpus
safe (reads no branch fields).

**G13 — pair-matching mode (added).** `--pair-seed`: room path and fiber
draw from SEPARATE rngs keyed `(pair_seed, family)` — branch-invariant
(the tag carries the branch, the key does not) — so paired corpora share
room paths, rosters, authors, κ(t); α enters only through the fiber mean.
Fiber draw counts are κ-determined → streams align across α at fixed κ_R
(plan note, holds). Tested at wave level: α=0 vs α=1 waves are
byte-identical in `field_raw_after` and `author` on all 9 nights, readers
diverge; without `--pair-seed` the tag-keyed default differs (unchanged).

**G2 — extension point only.** TODO block at `BRANCHES`; design (persona-
resampling map on TapNightSession inputs) deferred to its own addendum.

**G7 (self-test ext) — realized ICC, with calibration.** Measured through
the registered Measurement via the G5 adapter on a full 21×9 instrument
wave. Calibration changes (all documented at the constants):
- tangent walk THREADED across the wave (one persistent latent path —
  The Tap is one space); fresh-per-night e⊥ directions sink realized ICC
  to ~0.74;
- `ORTH_WALK` 0.02 → 0.005 (the corpus path now diffuses ~0.1 rad);
- `persona_deviations` now PROPORTIONAL (engine-like heterogeneous
  magnitudes) instead of per-reader unit-norm — unit-norm starved weak
  dials of between-reader spread (panic-dial ICC ~0.1);
- `DEV_SCALE` 0.55 → 0.70, `ICC_TARGET` 0.9076 → 0.96 (the analytic OU
  level; geometry contributes the rest of the within-variance).
Realized: **0.8724** at the self-test seed (7); **0.8813** at the
registered seed 20260821 (`rb-instrument` tags) — in the filed band
[0.85, 0.96]. Reference: real wave-2 actual-presence ICC 0.8444
(canonical-presence 0.7714); wave-1 filed 0.9076.

**G9 (self-test ext) — staged-night parity.** Generated staged night vs
`night-T4a.jsonl`: session_open key sets IDENTICAL (incl. staged_entries),
staged-entry shape identical (6 keys), speak key sets identical, entrant
reader-block keys identical; staged == non-staged speak key sets.

## Honest findings flagged for S2/S3 (G6 territory, NOT fixed in S1)

1. **Warmth-gate realization noise (dated deviation).** The plan's
   ±0.10 strata-mean warmth check, read literally, misfires on correlated
   cumulative-fit noise: the logged fit carries a decaying "lead-in
   memory" of the night's first ~10 latent draws (measured: T1/T3, same
   schedule, level residuals −0.04/−0.11; T5c α=0.75 ramps 0.25→0.59 on a
   flat schedule). The gate therefore checks the same design content in a
   noise-aware form — per-night FINAL-fit level + per-stratum DROP
   residuals (correlated noise cancels), both ±0.10 — documented in the
   gate source. Even so, at the current room-κ the (seed, tag) lottery
   fails one stratum on ~2 of 6 registered corpus configs (α=0.5 T5 drop
   0.155; α=0.75 T2 final 0.111, T5c drop 0.19). instrument / null /
   collapse / noise / α=0.25 configs gate green.
2. **Room-noise model tension (G6 sweep required pre-S3).** With the
   current vMF noise model the three field targets cannot be met at once:
   corpus_sd ≈ 0.094–0.13 (field 0.2367), stable-phase d ≈ 0.86 corpus-sd
   (field ≈ 0.29), and warm-stratum warmth-SNR. Lowering room κ moves sd/d
   toward the field but fattens warmth noise further. Candidate structural
   fix (engine-faithful): per-message dial-space noise decoupled from the
   direction concentration. This changes registered-statistic behavior and
   belongs to G6 + S2, not S1.
3. **ICC robustness surface.** Across a (seed, tag-prefix) lottery the
   full-design instrument ICC ranges ≈ [0.81, 0.90] (5 seeds × 2–4
   prefixes measured during calibration); the registered-seed value is
   0.88. All well above the void-guard floor 0.667. If S2 wants the
   §1.3 [0.85, 0.96] prediction to hold per-corpus regardless of tag,
   either freeze tags at registration or widen after the G6 sweep.

## Verification

- Generator self-test: 13/13 checks (schema parity vs filed T2/T4a,
  pipeline consumption, determinism, direction-only warmth, κ/μ event
  split, endpoint separation, null mode, G1, G9, G7, G13, G3).
- Full suite: **323 passed** (299 baseline + 24 new in
  `tests/test_riverbed_generator.py`), zero regressions; registered
  scripts (`premise_band_movers.py`, `slope_regression_w2.py`,
  `e2_instrument.py`, `e2_nights.py`, `stage2_wave_gate.py`) untouched.
- `data/` corpus files untouched (git-clean; generation confined to tmp).
