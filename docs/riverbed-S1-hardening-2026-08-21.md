# Riverbed S1 hardening run-note — 2026-08-21

Executed S1 of the wave-3 generation-corpus plan
(`memory/wave3-generation-plan-2026-08-21.md`): generator gaps G1, G3, G4,
G5, G13 + self-test extensions G7, G9. G2 (engine-native arm) left as a
documented extension point per plan (separate registration addendum).

**Amended same day** by the κ(t)-around-entry-steps check
(`memory/kappa-t-check-2026-08-21.md`, DIRECTION-EVENT verdict): the
plan's pre-registered "entry = κ-event, μ continuous" presupposition is
FALSIFIED by the field — entry-steps are μ/direction events, κ polarity
is warm=tight/cynical=loose, and transitions only loosen κ. The entry
mechanism and κ schedule were corrected accordingly (section below).
G3/G4/G5/G13 unaffected (they are plumbing, not event semantics).

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

## K-leg rework (κ-check fold-in, 2026-08-21)

The κ(t)-around-entry check measured, on the filed corpora, that the
generator's founding "κ-trajectory-first" design had the entry mechanism
backwards and the κ polarity sign-flipped. Corrected in `room_schedule` +
module header (constants at the top of `riverbed_generator.py`):

- **Entry = μ event.** `ENTRY_DWARMTH = 0.485` (= 0.97 × FLIP_SIZE; field
  pooled Δwarmth entry/flip = 0.147/0.151, p=0.68): the schedule steps
  warmth down at each entry exactly like a (marginally smaller) flip. μ
  is no longer continuous at entry. Self-test 6 asserts
  |Δμ(entry)| ≈ |Δμ(flip)| on the realized paths.
- **κ polarity corrected.** `KAPPA_WARM = 24`, `KAPPA_COLD = 11` (field
  per-strata logged κ: warm ≈ 21–24 tight, cynical ≈ 11–15 loose; the old
  10/18 ran opposite).
- **Transitions only loosen κ.** The flip response IS the level change
  (ln(11/24) ≈ −0.78 ≈ the measured flip Δlogκ −0.746); entries multiply
  latent κ by `KAPPA_ENTRY_FACTOR = 0.28` (sized so the LOGGED window
  response ≈ the measured entry Δlogκ −0.32 through the cumulative-fit
  renewal fraction); combined by pointwise min (never re-tighten), floor
  2.5. The old +12 entry tightening pulse is gone.
- **Null-mode polarity corrected:** the cohesion-only common κ shift now
  loosens (24 → 11 at the would-be flip) — field transition polarity.
- **Manifest** now records `entry_dwarmth` (design fact, branch-free,
  blind-safe) and the gate's expected-path reconstruction includes entry
  steps.
- **G7 recalibrated under corrected κ** (loose entry eras add within-night
  variance): `DEV_SCALE = 0.85`, analytic `ICC_TARGET = 0.99`; realized
  ICC **0.8861** at the registered seed 20260821 (self-test now uses the
  registered seed; scratch tags, per the §5.1 boundary). Seed-7 lottery
  draw: 0.807 (documented spread; see finding 3).

Plan consequence for S2: §1.2's "entry seqs as κ-events" and the
K-leg-as-primary-path framing must be revised in the registration
addendum — κ is now a designed channel MATCHING the engine's measured
semantics (text-determined, warm-tight/cynical-loose, entry-loosening),
and μ carries both flip and entry events.

## Honest findings flagged for S2/S3 (G6 territory, NOT fixed in S1)

1. **Warmth-gate realization noise (dated deviation).** The plan's
   ±0.10 strata-mean warmth check, read literally, misfires on correlated
   cumulative-fit noise: the logged fit carries a decaying "lead-in
   memory" of the night's first ~10 latent draws (measured: T1/T3, same
   schedule, level residuals −0.04/−0.11). The gate therefore checks the
   same design content in a noise-aware form — per-night FINAL-fit level +
   per-stratum DROP residuals (correlated noise cancels), both ±0.10 —
   documented in the gate source. Post-correction sweep (7 configs,
   registered seed): 6/7 gate green; the noise corpus misses ONE drop
   residual by 0.0007 (T4b cynical-pre→entry 0.1007). Residual scatter
   across corpora is draw-dominated (−0.04..+0.13, no consistent sign) —
   not an absorbable bias; at current noise settings the registered
   generation carries a real (~1 stratum in ~50) void risk on this check.
   S2 options: G6 noise-model rework (see 2), or accept + pre-plan the
   void handling.
2. **Room-noise model tension (G6 sweep required pre-S3).** With the
   current vMF noise model the three field targets cannot be met at once:
   corpus_sd ≈ 0.094–0.13 (field 0.2367), stable-phase d ≈ 0.86 corpus-sd
   (field ≈ 0.29), and warm-stratum warmth-SNR. Lowering room κ moves sd/d
   toward the field but fattens warmth noise further. Candidate structural
   fix (engine-faithful): per-message dial-space noise decoupled from the
   direction concentration. This changes registered-statistic behavior and
   belongs to G6 + S2, not S1.
3. **ICC robustness surface.** Across a (seed, tag-prefix) lottery the
   full-design instrument ICC ranges ≈ [0.80, 0.90] under the corrected κ
   semantics (registered seed 0.886; a seed-7 draw 0.807). All far above
   the void-guard floor 0.667. If S2 wants the §1.3 [0.85, 0.96]
   prediction to hold per-corpus regardless of tag, either freeze tags at
   registration or widen after the G6 sweep.

## Verification

- Generator self-test: 13/13 checks (schema parity vs filed T2/T4a,
  pipeline consumption, determinism, direction-only warmth, CORRECTED
  event semantics (entry μ-event + κ polarity/loosening), endpoint
  separation, null mode, G1, G9, G7, G13, G3).
- Full suite: **330 passed** (299 baseline + 31 new in
  `tests/test_riverbed_generator.py` — 24 S1 tests + 7 corrected-event-
  semantics tests), zero regressions; registered scripts
  (`premise_band_movers.py`, `slope_regression_w2.py`,
  `e2_instrument.py`, `e2_nights.py`, `stage2_wave_gate.py`) untouched.
- Gate sweep (7 branch configs, registered seed): instrument / collapse /
  null / α=0.25 / α=0.5 / α=0.75 ALL GREEN; noise fails one drop residual
  by 0.0007 (finding 1).
- `data/` corpus files untouched (git-clean; generation confined to tmp).
