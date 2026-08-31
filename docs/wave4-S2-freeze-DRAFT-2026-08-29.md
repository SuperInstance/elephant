# WAVE-4 S2 FREEZE TEXT — METHODOLOGIST DRAFT (wave-4b, post-kill redesign)

**Filed 2026-08-29 (METHODOLOGIST subagent lane). DRAFT of the exact S2
freeze text — not frozen until committed by the elephant lane with ZeroClaw
clause review.** Everything below is written to be committable verbatim as
the wave-4 registration addendum. Inputs: wave-4 registration draft
(`wave4-registration-draft-2026-08-22.md`), the S1 kill
(`wave4-S1-hardening-2026-08-22.md`, fiber v4 at 5b8af52: P_trans 0.9754 at
α=1 vs 0.5×P_rest bar; ICC 0.885→0.744 proves α live at anchor level),
wave-3 registration §1.4 (VOID provenance), wave-3 S5 verdict (envelope
numbers: within-pair A gaps median ~0.011, max ~0.085), G6 noise research
(stable-d floor 0.29, corpus_sd 0.2367 target, acceptance band 0.26–0.40),
and the eight-position foundation fold-in (§3 of the draft, carried
unchanged — not restated here).

**Status clause (the freeze's first line):** fiber v4 as-landed is DEAD per
the pre-stated kill. This freeze registers **fiber v4b**: the same
target-in-pull design with the wobble carrier re-specified per §1 below.
S3 generation is **BLOCKED** until the §5 power certificate is filed
(§5.4). No registered corpus may be generated before (i) the design gate
re-certifies on unsealed v4b pilots AND (ii) the power certification passes.

---

## 1. Wobble-sd amplitude clause (the frozen carrier specification)

**1.1 Amplitude matching (frozen, from draft §1.2 option (i)).**
The room-carried component of the within-night target is

```
room_carrier(t) = FIELD_ANCHOR_NORM · w_ar(t) / ‖w_ar(t)‖
target_R(t)     = pool + (1−α)·dev_R + α·room_carrier(t)
```

Direction-only carrier at anchor scale (`FIELD_ANCHOR_NORM` = 0.989, the
field anchor norm). The α contrast is **purely carrier identity** (who
carries the offset: reader id_idiosyncratic static dev vs room-common
moving direction), never amplitude. The raw-amplitude option (draft §1.2
(ii)) stays **rejected and unregistered**: at α=1 it collapses the target
norm ~12× and lets A/D/S move for the wrong reason.

**1.2 Wobble-sd floor (new, the v4b addition — the S1 kill's lesson).**
Amplitude matching alone was insufficient (S1: carrier moved ICC, never
touched P). Diagnosis: AR_PHI=0.9 persistence makes the wobble
indistinguishable from a static offset within any W=12 window, and leg_P's
roster-mean centering cancels any common-mode static component. The freeze
therefore pins the carrier's **scale and decorrelation jointly**:

- **(a) Scale floor:** the carrier's within-window displacement must exceed
  the estimator's split-half detection floor. Frozen rule:
  `sd(room_carrier within any stable window) ≥ 0.29 · corpus_sd`
  (the G6 field stable-d floor; at the 0.2367 calibration corpus_sd this is
  ≈ 0.069 in raw units — roughly 0.070 ≈ 0.07·FIELD_ANCHOR_NORM in anchor
  units). The carrier amplitude parameter (WOBBLE_LEVEL class) is frozen at
  the smallest value satisfying (a) with margin ×2, i.e. target within-window
  carrier sd ≈ 0.14 anchor units (≈ 0.033 raw), certified on unsealed pilots
  by direct measurement of logged `target_R(t)`.
- **(b) Decorrelation ceiling:** wobble autocorrelation at the P-leg
  transition spacing must be low enough that pre/post transition-window
  offset cosines decorrelate: frozen `ρ_w(lag ≥ 1 stratum spacing) ≤ 0.2`.
  Operationally: AR_PHI re-frozen from 0.90 to **0.5** (or an OU carrier
  with matched half-life), chosen at S1v4b hardening as the largest
  persistence passing (b) on pilots — the chosen value is then frozen here
  by dated addendum before any registered corpus exists.
- **(c) Common-mode cancellation is a feature to be survived, not removed:**
  leg_P's roster-mean centering stays (registered, unmodified). The design
  must deliver P's failure **through** it: at α=1 all readers share the
  carrier, but the charisma pull `s` is reader-heterogeneous, so per-reader
  offsets ride the carrier differentially. Pre-state: if pilots show the
  differential component < the (a) floor, the design fails again and the
  kill rule re-fires — no second redesign is attempted inside this
  registration.

**1.3 Provenance integrity (carried):** `--fiber v4b` versions the change;
v3 default preserved so wave-3 corpora stay bit-reproducible and the S5
verdict's line-845 provenance remains true of the artifact it describes.
Pair isolation exact: carrier from `room_rng` `(pair_seed, crc(fam), 1)`,
dev draws from `(pair_seed, crc(fam), 2)` — pair members share both streams
bit-for-bit; targets differ only through α. Coordinate firewall holds: α
never touches room path, κ(t), rosters, or authors.

---

## 2. Branch×leg matrix with pre-stated verdicts (frozen)

Corpus skeleton (draft §1.6): 16 corpora — α ∈ {0, .25, .5, .75, 1} ×3
corpora... [final grid per S1v4b: α ∈ {0,.25,.5,.75,1} + null-mode + 5
α-only matched pairs (adjacent + endpoint 0/1), pair-seeds m = lower-α,
master seed 20260829 fresh — no stream overlap with wave-3 or the pilots]
— night families, ATTENDANCE matrix, and K-leg μ-event entry semantics
verbatim. Legs run unmodified (TOL=3, HYST 0.05/3, edges 0.3/0.6, W=12
primary, W∈{8,16} sensitivity; seeds 20260821-family; B=2000
reader-clustered).

| leg | α=0 (instrument) — pre-stated verdict | α=1 (collapse) — pre-stated verdict | intermediate α | discriminates |
|---|---|---|---|---|
| **A** timing | **FIRES** (p<0.05; reads the α-invariant room path) | **FIRES, α-blind by design** — within-pair A gap ≤ 0.085 (wave-3 max envelope); gap > max envelope = carrier leak, corpus pair VOIDed (§3.8) | fires, flat | signal-vs-noise only |
| **D** coverage | above null-night rate | above null (amplitude-matched ⇒ ρ not small) | flat | context; never primary |
| **P** persistence | **HOLDS** (P_trans ≥ 0.5×P_rest, 0.97-class per pilots) | **FAILS** — P_trans < 0.5×P_rest: differential carrier ride decorrelates pre/post offset cosines across transitions (§1.2 makes this reachable) | **monotone ↓** (pre-registered direction) | **instrument-vs-collapse (the headline)** |
| **S** exposure | x-invariant (CI ∋ 0) | reachable in principle (anchor-scale variation time-varying, room-correlated); **exploratory, direction only** — no pass/fail claim licensed | — | secondary |
| **ICC** | ∈ [0.80, 0.92] (pilot 0.885 anchors the bracket; G6 band retired for v4b, threshold re-frozen from S1v4b sweep by dated addendum) | **DECLINES** below the α=0 bracket floor (pilot −0.141; pre-registered direction: reader constants ride nightly-fresh carrier) | between, orderable | instrument-vs-collapse (supporting) |
| **V** anchor leg (new, generation-only) | reads logged target trajectory | **separates α exactly** (cos(target_R(t), carrier) = α by construction up to dev mixing) | monotone in α | closes the localization loop: legs blind to anchor, V pointed at anchor separates |
| **2AFC pairs** | — | signed directions vs α: P_trans ↓, spread ↓ (common moving target compresses dispersion), ICC ↓; A flat | orderable | gradient clause, live for the first time |

**Verdict pre-statement (frozen, so S4 is adjudication not invention):**
H-GEN-4b passes iff (i) P holds on all α=0 corpora, (ii) P fails on all
α=1 corpora, (iii) A stays α-blind within envelope on every pair, (iv) V
separates α monotonically on every pair. Partial patterns map to the
pre-stated outcomes: P fails nowhere → v4b kill confirmed, honest negative
#2, localization claim extended to "target-level α needs a leg that reads
targets"; P fails at α=1 but also wobbles at α=0 → design contaminated,
corpus VOIDed, no apparatus claim; everything passes but V noisy →
apparatus verdict stands, V booked exploratory.

---

## 3. VOID rules — complete set (frozen)

Carried from wave-3 §1.4 verbatim where marked; new rules are v4b-specific.

1. **Generated-corpus gate failure** (carried): logged roster == designed
   ATTENDANCE; determinism re-run byte-identical; strata-mean logged warmth
   within ±0.10 of the lag-accounted schedule; corpus_sd finite, computed
   from the corpus itself, used as its own normalization; Sxx ≥ 0.19.
2. **Null-night rule** (carried): null crossing rate ≥ 50% of signal-night
   rate, per corpus where A is read.
3. **Crossing floor, branch-conditional** (carried): <20 counted
   down-crossings voids instrument/intermediate corpora only; on collapse
   corpora a low count is a branch hit.
4. **Continuity ladder** (carried): off by >0.10 within-corpus.
5. **Bootstrap draws** (carried): effective draws <1,500.
6. **Decoy-panel disagreement** (carried): only the o/d pipeline recovering
   the branch ⇒ contamination finding; no validation claim.
7. **Robustness manifold** (carried): verdicts reported as sets over
   W∈{8,12,16}, margin ∈{1.5,2,3}·SE, hold ∈{2,3,4}.
8. **Carrier-purity leak** (NEW, registration-grade): any pair separable on
   A beyond the wave-3 max within-pair envelope (0.085) ⇒ VOID that pair's
   instrument-vs-collapse reading; the localization claim does not survive
   a leak.
9. **Wobble-spec violation** (NEW): measured within-window carrier sd below
   the §1.2(a) floor, or ρ_w above the §1.2(b) ceiling, measured from
   logged `target_R(t)` ⇒ VOID the corpus (the frozen carrier was not
   delivered; this is a generator check, not an outcome check).
10. **Power-certificate lapse** (NEW): if S3 begins before the §5
    certificate is filed, or the certificate's seed/provenance cannot be
    verified, ALL wave-4 corpora are VOID and the wave restarts at S1v4b.

Explicitly NOT voids (carried): wave-3's field-corpus §5.3 VOID stands; a
clean wave-4 calibration licenses the power-with-certificate sentence only.

---

## 4. Honesty guards (carried, deltas only)

All ten wave-3 guards carry (registration-before-reading, q-rule,
coordinate firewall, decoy panel, procedural blindness with sealed
sidecars + opaque tags + verdicts-before-unseal, gate-target holdout,
append-only, never-pool, tautology guard, priors filed). **Priors for
wave-4b, filed up front:** apparatus recovers the P headline at endpoints
P≈0.55 (humbled by the v4 kill); intermediate-α monotone ordering P≈0.45;
V separates exactly P≈0.9; decoy-panel full agreement P≈0.6; A stays
α-blind P≈0.85. REG-1′/2/3 clauses ride as drafted (§2 of the
registration draft) with REG-2's S-leg and all α-gradient clauses MOOT if
the v4b design gate re-fires.

---

## 5. Power analysis for matched pairs (frozen) + S3 block

**5.1 Unit of analysis.** Each α-only matched pair shares room paths,
rosters, authors, κ(t) bit-for-bit; members differ only through α. The
licensed statistic is the **within-pair difference** at family level:
n = 9 families per corpus, each family's leg statistic differenced across
pair members ⇒ 9 paired differences per leg per pair. Never pooled across
pairs.

**5.2 Test.** Paired t-test (or Wilcoxon signed-rank as the robustness
column) on the 9 family-level differences, two-sided, α=0.05, per leg per
pair. 2AFC ordering: ≥8/10 adjacent-pair orderings correct per leg
(binomial p≈0.011 at 8/10) — carried from wave-3 §1.3.

**5.3 Minimum detectable effect.** With n=9 pairs and the pilot-measured
family-level scatter (wave-3 instrument corpora: P_trans family-level sd
≈0.01; wave-3 within-pair scatter is the honest noise scale), 80% power at
α=0.05 two-sided requires a paired difference of ≥ 1.03·σ_d
(t₈ crit 2.306, √9=3 ⇒ MDE = 2.306·(σ_d/3)·2.80 ≈ 1.03σ_d). The
pre-stated α=1 effect (P_trans ratio from ~0.98 to <0.5, i.e. ΔP_trans
≈ 0.49 against σ_d ≈ 0.01) is a **~49σ effect if delivered** — power is
not the binding constraint on the headline; delivery is (that is what §1.2
and the design gate are for). The binding power cases are the
**intermediate-α gradient** and **ICC decline**:
- ICC: pilot paired difference −0.141; wave-3 ICC corpus-level sd ≈0.04 ⇒
  MDE ≈ 0.041 ⇒ the pilot effect is ~3.4 MDE — powered at n=9.
- Intermediate α (e.g. Δα=0.25 adjacent pairs): effect unknown by design
  (that is the experiment). Power is therefore certified by simulation
  (§5.4), not asserted.

**5.4 Certification protocol (the S3 block).** Before any registered
corpus is generated, run on **unsealed pilot replicates** (throwaway seeds
in scratch, seed 20260829-PC, never the registered statistics): 500
replicate pairs at each registered α-contrast, computing (i) rejection
rate for the paired P test under the frozen §1 carrier — require ≥0.90 at
Δα=1 and ≥0.80 at Δα=0.25; (ii) false-positive rate on α-identical pairs
(require ≤0.05); (iii) A-gap envelope exceedance rate ≤0.01 (leak guard
specificity); (iv) V-leg separation rate ≥0.95 at Δα=0.25. **File the
certificate as a dated doc with seeds and code SHA. S3 is BLOCKED until
this certificate exists and passes.** A failed certificate returns the
design to S1v4b hardening (§1.2c) — it may not be waived, narrated around,
or downgraded to a caveat.

---

## Provenance

Read (read-only): wave4-registration-draft-2026-08-22.md; wave4-S1-hardening
-2026-08-22.md; memory/wave3-registration-2026-08-21.md (§1.4 VOID set);
memory/wave3-S5-verdict + wave3-S4-analysis (envelope/scatter numbers);
memory/research-g6-noise-2026-08-21.md (stable-d floor, corpus_sd); elephant
git log (5b8af52, 8749974). Written: this document only. Nothing sealed,
frozen, or committed; no data/wave3/** or registered corpus touched.
