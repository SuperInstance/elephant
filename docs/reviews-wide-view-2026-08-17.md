# Elephant — wide-view critique sweep (2026-08-17)

**Sweep:** five DeepInfra models run over the `elephant` v0 repo, each with a
tight, code-scoped prompt (architecture / math / philosophy / robustness / code
quality). This file is the synthesis: per-model verdicts, the findings that
*converged* across models, and the prioritized fix list with what was applied.

**Models run:**
| Role | Model | Status |
|------|-------|--------|
| Architecture (dial bank + field) | `ByteDance/Seed-2.0-pro` | ✅ |
| Math/code (sensors + nudge) | `Qwen/Qwen3.6-35B-A3B` | ✅ |
| Philosophy (captain's reframing) | `NousResearch/Hermes-3-Llama-3.1-405B` | ✅ |
| Systems/robustness | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B` | ✅ |
| Code quality + bugs | `Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo` | ⚠️ rate-limited (429) → substituted `moonshotai/Kimi-K2.7-Code` |

---

## 1. Per-model verdicts

### Seed-2.0-pro — architecture
1. **"κ is inverted" — REJECTED (verified false).** Seed claimed `concentration()`
   = `‖vector − 0.5‖·2` produces the inverse of the spec (`cold = high κ, warm = low κ`).
   Re-checked against `README.md` and `test_concentration_cold_tighter`: extreme/cold
   rooms give *high* κ, mid/warm rooms give *low* κ — exactly the spec. The test
   already encodes it. Seed mislabeled the correct mapping. *(The real, subtler κ
   issue is the mixed neutral-center — see P1-2.)*
2. **`RoomField.vector()` uses a hardcoded `DIAL_NAMES` basis** and silently fills
   missing dials with `0.0`. A custom `DialBank` with renamed/added/removed dials
   silently corrupts distance/contrast/warmth with no error. → P2.
3. **`acclimation_rate_from` returns `inf` on overshoot.** An agent that has moved
   *past* the room vector (the exact case a charismatic/high-skill agent hits) yields
   `ratio <= 0 → inf`, which then poisons any downstream mean/training log. → **P0, FIXED**.

### Qwen3.6-35B-A3B — math/code (sensors.py + nudge.py)
1. **Unit mismatch in `speed_kts`.** `sensors.py` computes `speed = ‖v‖ · 1.94384`
   (the m/s→knots factor), but every other constant (`# km scale`, `4.0` spread
   saturation, `2.0` association gate, `0.5` predecessor gate) is written as if
   positions are **km** — where the factor should be `1943.84`. Off by 1000× under
   the km reading. → P1.
2. **Hardcoded association/predecessor gates** (`2.0`, `0.5`) are magic numbers with
   a unit assumption baked in; the greedy `_associate` also doesn't guarantee track
   identity across the three frames. → P2.
3. **Acceleration finite-difference** `(v23 − v12)/dt23` reuses `dt23` as the
   denominator even when `dt12 ≠ dt23` — a minor approximation, not a crash. → P2.

### Hermes-3-Llama-405B — philosophy
- **Verdict: the code honors the captain.** "Many JEPAs as dials" (DialBank over N
  independent Dial senses), "the elephant is the ensemble field" (RoomField), and
  "the elephant is only visible by contrast" (`distance` / `sauna_plunge_gap`) all
  land faithfully.
- **Caveat:** the `warmth()` weighting is a single fixed linear blend — Hermes notes
  the weights could be tuned to better capture the *felt* temperature, and that
  acclimation/charisma are currently plain Euclidean relaxation rather than the
  embedding-space geodesic the v3 design describes. Philosophy is intact; fidelity
  of the numeric *shape* is the open gap. → P2 (weight tuning).

### Nemotron-3-Ultra-550B — systems/robustness
- The whole file is a NaN-propagation audit. Concrete failure modes:
  1. **NaN entry points in `sensors.py`**: `float(f.data)` raises on non-numeric
     sounder data; `_targets` `reshape(-1, 2)` raises on odd-length/None data; a NaN
     in radar targets flows through `_spread → mean → read()` into every downstream
     dial, `nudge_prior`, and `RoomField`. → P2.
  2. **`nudge_prior` does not guard NaN**: `np.max(np.abs(prior))` is NaN → the
     `m > 1.0` normalization silently no-ops and NaN reaches the vision model's
     attention. → P2.
  3. **`ripple()` has no cycle detection** — a cyclic reply tree recurses to the
     interpreter limit. → P2.
- (Also confirms `density()`'s `span = max(…, 1e-9)` and `_cosine`'s zero-guard are
  already correct.)

### Kimi-K2.7-Code — code quality + bugs *(substitute for rate-limited Qwen3-Coder-480B)*
1. **`room.density()` ignores its `window` parameter.** Both callers (`panic`,
   `volume`) pass a window expecting a rolling pulse, but the body measures the full
   span. → **P0, FIXED.**
2. **`speed_kts` unit factor** — independent confirmation of Qwen3.6's finding. → P1.
3. **`warmth()` missing-dial defaults are inconsistent**: `presence`/`volume` default
   to `0.0` (cold) while `earnestness`/`cynicism` default to `0.5` (neutral). → **P0, FIXED.**
- (Also flagged a `reverberation()` window off-by-one and the dropped tail window —
  real but cosmetically guarded by the `len(windows) < 2` return. → P1.)

---

## 2. Convergent findings (flagged by ≥2 models)

1. **`speed_kts` unit inconsistency in `sensors.py`** — Qwen3.6 + Kimi-Coder
   (independent). The knots factor says meters; three other constants say km.
2. **`room.density()` is not actually windowed + "per second" vs "per minute"** —
   Kimi-Coder + Nemotron (unused param) + the section header comment contradicted
   the `× 60` body.
3. **NaN / malformed-data robustness in `sensors.py` and `nudge.py`** — Nemotron +
   Qwen3.6: `float(f.data)`, `_targets` reshape, and NaN flowing into `nudge_prior`.
4. **`warmth()` blend is hand-tuned and inconsistent** — Kimi-Coder (missing-dial
   defaults) + Hermes (weights could be tuned) + Seed (hardcoded vector basis).
5. **`acclimation_rate_from` inf blow-up** — Seed (primary) + Nemotron's
   `math.log(ratio)` failure-mode note.

---

## 3. Prioritized fix list

### P0 — applied this sweep (small, safe: bugs + doc strings + tests)
1. **`field.acclimation_rate_from` no longer returns `inf`.** Ratio is now clamped to
   `[1e-9, 1]` so an overshot agent yields a large *finite* rate. Test added:
   `test_acclimation_rate_overshoot_is_finite`.
2. **`room.density` now honors `window`.** Measures over the trailing `window`
   seconds (count and span both restricted to recent messages); fixed the
   "messages per second" header comment → "per minute". Test added:
   `test_density_is_windowed`.
3. **`field.warmth` missing-dial defaults fixed.** `presence`/`volume` default to
   `0.5` (neutral for a `[0,1]` dial) instead of `0.0` (cold). No effect on the
   default bank (all seven dials present), but corrects custom/partial banks.

### P1 — documented, not applied (need a product decision or a heuristic change)
1. **`sensors.py` unit decision.** Pick meters or km and make `speed_kts`'s `1.94384`,
   the `4.0` spread scale, the `2.0` gate, and the `0.5` predecessor gate
   self-consistent. The current code is silently 1000× wrong under one reading.
2. **`field.concentration()` centers every dial at `0.5`, but `mood` and
   `joke_landing` are `[-1, +1]` with neutral `0.0`.** A neutral-mood room reads as
   `0.5` from center, inflating κ. (Seed's broader "κ is inverted" claim was wrong;
   this neutral-center mismatch is the real residual issue.)
3. **`warmth()` re-centering is inconsistent**: `earnestness`/`presence`/`volume` are
   re-centered `(x − 0.5)·2` but `cynicism`/`panic` are used raw; positive weights sum
   to `0.75` vs `1.0` negative, so warmth can't reach `+1`. Decide a symmetric scheme.
4. **`reverberation()` window off-by-one / dropped tail window** — minor, currently
   masked by the `len(windows) < 2` guard.

### P2 — backlog (robustness hardening, no behavior change)
1. NaN/empty guards: `_targets` (None / odd-length), `SounderBiomassDial.read`
   (`float()` fallback), and `nudge_prior` (skip/zero NaN readings).
2. `ripple()` cycle detection for cyclic reply trees.
3. `RoomField.vector()` basis validation vs the active `DialBank` (raise on mismatch
   rather than silently filling `0.0`).
4. Named constants for the association/predecessor gates.

---

## 4. Result

- `python3 tests/test_elephant.py` → **8/8 pass** (6 original + 2 new).
- `tests/test_fleetmath.py` → not present (skipped).
- `examples/demo_elephant.py` → runs clean, values unchanged (the P0 fixes are
  edge-case/doc/default fixes that don't shift the happy-path numbers).

*The elephant is still hand-crafted v0 — no redesign was attempted, per instruction.
The three P0 fixes close the concrete bug the sweep turned up; the P1/P2 backlog
waits on the unit decision and the next learned-side pass.*
