# PREMISE BAND-MOVERS RUN — Void by Rule, Composite Indicative

**Filed: 2026-08-21.** Implements and runs the registered design
`E2E3-premise-band-movers-design-2026-08-21.md` (registered as the dated
addendum to `research/topic.md`, zeroclaw-dissertation, before this run).
Script: `scripts/premise_band_movers.py` (new, read-only against
`data/nights/`; the Stage-2 wave gate invoked verbatim, not edited).
Outputs: `data/slope/premise-band-movers-results.json` (primary; mirrored at
`data/premise-band-movers/{results,wave2,wave1}.json`). No filed artifact
modified; the wave gate's JSON re-run byte-identical.

## Verdict (6 lines)

1. **VERDICT: VOID BY RULE §5.3 — only 17 counted down-crossings across the
   wave-2 primary corpus (< 20; wave-1 replication: 19 < 20).** No branch is
   declared. Per the registration, the void is reported with the estimator
   finding and the fix route — never silently re-read.
2. **The four legs, reported as indicative (both waves, never pooled):**
   **A fires** — wave-2 A = 0.647 (11/17 down-crossings within ±3 speaks of a
   registered transition, window-center referent; circular-shift null
   10,000: p = 0.0013, null95 = 0.45); wave-1 A = 0.632 (p = 0.0001,
   null95 = 0.44). **D fails the ≤50% bar** — wave-2 D = 4/10 = 0.40 (exact
   CI [0.122, 0.738]), wave-1 D = 5/10 = 0.50 (CI [0.187, 0.813]); both
   significantly above the null-night rate 0/1 (T9/S5 midpoint, no counted
   crossing on any null reader-night: 0/7 and 0/8 vs signal 14/65, 15/65).
   **P holds decisively** — wave-2 P_trans = 0.9940 [0.991, 0.996] vs
   P_rest = 0.9935 (registered threshold: ≥ 0.5×P_rest = 0.497); wave-1
   0.9948 vs 0.9962. **S x-invariant in the primary** — slope 1.410
   [−0.313, +2.797] (contains 0), nested-permutation vs roster competitor
   p = 0.384 (does not beat); wave-1 replication slope 1.237
   [+0.334, +1.979] **excludes 0** but p = 0.521 (does not beat) — split.
3. **The mechanism picture (descriptive, no branch read off it):** at W=12
   the premise score is clear-side nearly everywhere (stable-phase mean ρ
   ≈ 2–4; transition-window dip ≈ 1.4–1.8). The transition drift spike
   (split-half d ≈ 0.6–0.9 corpus-sd, vs stable ≈ 0.15–0.3) pushes a
   minority of readers (those with small offsets) through the 0.6 edge for
   exactly the ~1–2 window positions whose half-split straddles the
   boundary — those are the counted crossings, and their timing is tight
   (0.5–2.5 speaks in center coordinates). The static in-band ratio
   (0.5599/0.6088/0.6139) is a **window-scale artifact of the full-night /
   strata-split estimator**, not a phase-average of in-band phases: no
   phase at W=12 sits in the band except early-night acclimation dips
   (T2/S2-style short first strata).
4. **Coverage structure (leg D, informative):** the covered transitions are
   the hard SEG warm→cynical shifts (T1@20, T3@20, T4b@20, T8@20; A@20,
   S1@20, S2@8, S3@20, S4b@20). NOT covered: the newcomer-entry family
   (T5/T5c@24, D/D-cold@24 — the milder step) and T4a (both). Hard strata
   steps make band-movers; entry steps do not.
5. **Sensitivities:** W=8 — A = 0.917 / 0.900 (p < 0.0001 both waves);
   W=16 — A = 0.074 / 0.087 (dies; windows too coarse for the dip);
   actual-presence channel — A = 0.545 / 0.684 (p ≤ 0.0002; 44/38 events,
   more than double the canonical channel's); class-residual P holds
   (0.992 vs rest 0.969) ⇒ **identity-propagation booking does NOT fire**
   (what persists through steps is individual, not just archetype);
   class-residual S wave-2 slope 0.695 [0.119, 1.473] — CI excludes 0
   while the primary contains it (knife-edge divergence, disclosed, same
   pattern as the wave-2 slope run's tripwire; no declaration made).
6. **Priors:** A fires 0.55 → 0.571; P holds 0.50 → 0.524; x-invariance
   0.60 → 0.619 (Beta bookkeeping at the registered prior strength; the
   legs carry the information, the void caps the reading).

## 1. What was implemented

| item | detail |
|---|---|
| `scripts/premise_band_movers.py` | new; imports `Night`, `Measurement`/`readings`, `corpus_sd`, `FIELD_NIGHTS`/`FIELD_NIGHTS_W2`, `archetype_labels` from the unmodified `scripts/e2_instrument.py`; canonical presence = primary channel (registered), actual = sensitivity |
| estimator | W=12 windows (stride 1), o_R = dial-RMS offset of the windowed mean from the present-roster windowed mean (filed E2 spread convention; deviation note 2), d_R = Euclidean split-half (6/6) displacement (filed drift convention), ρ = o/d; band states 0.3/0.6, hysteresis ≥0.05 beyond edge + ≥3 windows |
| legs | A (±3 speaks vs 10,000 per-reader-night circular shifts, seed 20260821), D (night-level coverage vs null-night midpoint, exact binomial), P (offset-vector cosine over the ICC-reliable subspace, pre/post windows, Fisher-z pooled, ≥0.5×P_rest threshold), S (per-night median ρ ~ warmth x + reader FE; reader-clustered bootstrap B=2000; roster-size + archetype-baseline-warmth competitor; nested permutation 10,000) |
| corpus | PRIMARY wave-2 T-nights (21 readers, 66 signal + 7 null reader-nights); REPLICATION wave-1 (15 readers, 65 signal + 8 null); per-wave, never pooled; A/B/C/D/D-cold v:1 anchor the ladder only; S6/S7/night-H* excluded |
| guards | Stage-2 wave gate re-run verbatim (ALL PASS, byte-identical output); wave-1 corpus_sd 0.2367 / spread 0.4556 / drift 0.7483 / warmth ladder 9/9 re-asserted; `assert_replay_matches_log` 6/6 sampled reader-nights (incl. cold drifter) |

## 2. Continuity ladder (all rungs PASS — the estimator measures the filed object)

| rung | pooled full-night R | filed channel | status |
|---|---|---|---|
| wave-2 (66 signal reader-nights) | 0.5737 | 0.6139 (gate spread 0.4883 / drift 0.7955) | OK (±0.10); exact global anchor = 0.6139 exact |
| wave-1 (65) | 0.6491 | 0.6088 (filed E2-at-power; E-seg 0.6853) | OK; exact global anchor = 0.6088 exact |
| v:1 anchor, real-only | 0.5409 | 0.5599 | OK |
| v:1 anchor, grounded (N=20) | 0.4845 | 0.4898 | OK |

Premise-measurement exact reproduction asserted to <0.005 (0.5599 / 0.4898).

## 3. The four legs (registered statistics)

**PRIMARY wave-2** (T-nights, canonical presence, W=12):

| leg | value | CI / null | registered branch |
|---|---|---|---|
| A | **0.647** (11/17) | shift-null p = 0.0013; null95 0.450; null mean 0.246 | fires |
| D | 0.40 (4/10) | exact binomial [0.122, 0.738]; D_null = 0 (T9: 0/7 reader-nights cross) | fails ≤50% clause; D−D_null CI excludes 0 |
| P | 0.9940 | [0.991, 0.996]; 8 transitions (2 dropped: T2@8, T8@20 — windows don't fit); P_rest 0.9935 (3 refs: T4a:cynical, T5:pre, T5c:pre) | holds (≥0.5×P_rest) |
| S | slope 1.410 | [−0.313, +2.797]; 65 cells/21 readers; nested perm p 0.384 | x-invariant (contains 0, competitor not beaten) |

Up-crossing mirror (secondary): A_up = 0.233 (p = 0.222) — up-crossings are
acclimation-phase exits, not mid-stratum events.

**REPLICATION wave-1** (labeled, never pooled): A = 0.632 (12/19, p =
0.0001, null95 0.440); D = 0.50 (5/10, CI [0.187, 0.813], D_null = 0,
S5 0/8); P = 0.9948 vs rest 0.9962 (2 dropped: S2@8, S3@20); S slope 1.237
[+0.334, +1.979] — **CI excludes 0** but nested perm p = 0.521 (competitor
not beaten) ⇒ not a registered falsification, and not x-invariance either:
an INDETERMINATE S leg in the replication. Both waves' S slopes are
positive ~1.2–1.4 (warm nights → higher, i.e. more clear-side, scores).

## 4. Void-rule application (honest, in order)

| rule | value | verdict |
|---|---|---|
| §5.1 wave gate | ALL PASS, byte-identical | pass |
| §5.2 null ≥ 50% of signal crossings | 0.000 vs 0.215 (w2); 0.000 vs 0.231 (w1) | pass (clean separation) |
| §5.3 ≥ 20 counted down-crossings | **17 (w2); 19 (w1)** | **FIRES — VOID** |
| §5.4 ladder ±0.10 | all rungs OK | pass |
| §5.5 effective bootstrap draws | 2000/2000 | pass |

The registered kill (A ≤ null95 AND D fails) does **not** fire — A fired at
p = 0.0013. The registered survival composite (A fires + P holds +
x-invariant) is *present* in the primary — but it is **not declared**,
because §5.3's event-count floor exists precisely to keep 17-event composites
from being read as branch support. What the run licenses instead: the
estimator finding (§5 of the verdict lines) and the fix route below.

## 5. Deviation notes (each with reason; none improvised)

1. **Crossing position referent (leg A/D timing).** The design registers
   "|crossing position − boundary| ≤ 3 speaks" without specifying the
   window→speak mapping. Window-START makes the registered test
   arithmetically blind to its own predicted effect: the transition dip's
   causal window has its half-split at the boundary, so its hysteresis run
   starts ≈ W/2 = 6 speaks early — always outside ±3 (observed: A = 0.000,
   p = 1.0, an artifact, not a finding). Primary uses the window CENTER
   (start + (W−1)/2); the start-referent value is carried as a labeled
   sensitivity in the results JSON. Same class of correction as the Stage-2
   run's T3-mapping deviation (the design's own arithmetic forces it).
2. **Numerator norm convention pinned by the ladder.** o_R = RMS over dials
   of the deviation (the filed E2 spread convention — the honesty guard's
   "numerator ≈ 0.46–0.56" numbers are dial-RMS), d_R = Euclidean split-half
   norm (the filed drift convention: 0.29 floor, 0.75–0.93 spikes). A
   Euclidean 7-norm numerator (the literal reading of "‖·‖") inflates the
   score ~√7 and fails the ladder by +0.9 — measured, then fixed; both
   conventions' ladder values recorded in the JSON. This is the exact
   relationship §3 said the ladder pins.
3. **Wave-1 replication corpus** = the 9 primary nights (A, D, D-cold, S1–S5)
   with FIELD_NIGHTS attendance = 73 reader-nights, 65 signal + 8 null — per
   the design's own "73 reader-nights; orig-6 span all 9 incl. S5 null".
   A/B/C/D/D-cold additionally serve the v:1 ladder anchor (different tier,
   7-reader replay, premise-measurement machinery).
4. **Leg P window fitting.** Pre = [b−W, b−1], post = [b, b+W−1]; transitions
   where either window doesn't fit are dropped and listed (T2@8/S2@8: first
   stratum 8 speaks < W; T8@20/S3@20: post stratum 8 speaks < W) → 8 events
   per wave. Rest-references = the first adjacent window pair inside a
   single ≥2W signal stratum (3 per wave: wave-2 T4a:cynical, T5:pre,
   T5c:pre; wave-1 D:pre, D-cold:pre, S4a:cynical). At W=16 no stratum is
   ≥ 32 speaks → P_rest unavailable (labeled nan).
5. **Direct clear→kill confirmed moves** record one down-crossing per edge
   crossed (2 events at the same position). 3 such events in wave-2, 7 in
   wave-1; the convention affects only event counts, not timing.
6. **Outputs**: primary results at `data/slope/premise-band-movers-results.json`
   (task routing) and mirrored at `data/premise-band-movers/` (design §8
   naming); per-wave JSONs alongside.
7. **T3/T1 label note**: the design's ladder parenthetical ("0.6088 E-seg /
   0.5599 E-cont") transposes the filed JSON's labels (0.6088 =
   E-cont-canonical primary; E-seg = 0.6853; 0.5599 = the v:1 old-corpus
   continuity number). The ladder was checked against the filed JSON values
   as ground truth.

## 6. What would change the verdict

- **Event count ≥ 20** under the same registration (more nights, or more
  readers per night): with A's effect size (0.63–0.65 vs null ~0.25) and
  P's 0.994, the composite would clear §5.3 and the verdict would move from
  VOID to **SURVIVED (capped)** — "readers are instruments except at steps;
  the in-band ratio was an average over phases" (capped: does not reopen
  Branch A/B) — *if* S stays x-invariant and D's coverage climbs above 50%
  (D at 0.40–0.50 currently fails its clause; the composite tolerates a
  failed D only as an INDETERMINATE leg).
- **D climbing above 50%**: needs the entry-step family (T5/T5c, D/D-cold)
  to produce counted crossings — their drift step (~0.6–0.9) is currently
  below the crossing threshold for most readers; more readers with small
  offsets, or hysteresis relaxed to 2-window hold (a re-registration), would
  do it.
- **W=8 registered as primary** (re-registration): A = 0.90–0.92 at p < 1e-4
  — but W=8 has even fewer events (12/10); it does not fix §5.3 alone.
- **S resolving toward collapse**: if the replication's positive slope
  (1.24, CI excluding 0) were to also beat the roster competitor (currently
  p = 0.52), the alignment arm would be falsified per the registered rule.
  The class-residual S CI excluding 0 in wave-2 ([0.119, 1.473]) — the same
  knife-edge divergence the slope run's tripwire produced — is the standing
  hint that a within-archetype warmth dependence exists; it is disclosed,
  not read.
- **A mechanism kill** would need P_trans < 0.5×P_rest: observed 0.994 vs
  0.497 threshold — nowhere close (offsets persist through steps essentially
  perfectly, individual not just archetype).

## 7. Honesty ledger

- The bare crossing rate (21–23% of signal reader-nights, 0% of null) is
  **not** read as evidence (§0 guard) — it is reported as the §5.2 void
  check only.
- The A firing is convention-dependent (center vs start referent): both
  values are filed; the spec-defect argument for the center referent is
  arithmetic (W/2 = 6 > TOL = 3), not data-driven, and the start-referent
  null result is retained in full.
- The W=16 collapse of A (0.07, p ≈ 0.99) is filed: the timing structure is
  window-scale-dependent, strongest at W=8, present at W=12, absent at 16.
- 17 and 19 events are below the registered floor; nothing in §3 is
  declared; the priors update is bookkeeping, not a branch reading.

## Reproduce

```
cd /home/eileen/projects/elephant
python3 scripts/premise_band_movers.py        # gate -> ladder -> legs -> sensitivities (~5 min)
# outputs: data/slope/premise-band-movers-results.json
#          data/premise-band-movers/{results,wave2,wave1}.json
```

## Provenance

- Design: `E2E3-premise-band-movers-design-2026-08-21.md` (registered;
  dissertation `research/topic.md` addendum 2026-08-21).
- Machinery: `scripts/e2_instrument.py` (unmodified), `scripts/stage2_wave_gate.py`
  (invoked), `scripts/premise_measurement.py` (rung-2 anchor), `scripts/slope_regression.py`
  (`room_warmth` guard), `scripts/e5_identity_propagation.py` (ladder anchors).
- Corpora: `data/nights/night-{T*,S*,A,D,D-cold,B,C}.jsonl` read-only;
  `data/slope/stage2-wave-gate.json` rewritten byte-identically by the gate
  itself.
- Seeds: 20260821 family (main/shift-null/bootstrap/permutation), 0/20260819/20260820
  inherited from imported filed machinery.
