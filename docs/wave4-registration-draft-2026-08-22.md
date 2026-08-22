# WAVE-4 REGISTRATION — DRAFT (S5 handoff, REG pre-staging, foundation fold-in)

**Filed 2026-08-22. DRAFT — a pre-registration working document, not the frozen
registration.** Nothing here is frozen; no threshold is binding until the S2
freeze. Inputs: the wave-3 S5 verdict (Arm 1 closed, H-GEN falsified → honest
negative confirmed), the S4 blinded analysis, the generator source, the
foundation summit's eight positions, and ZeroClaw's REG-1/2/3 pre-stage text.
No frozen, sealed, or registered document was modified to produce this draft;
nothing committed.

---

## 0. What wave-3 handed off

H-GEN is dead; anti-hypothesis (ii) is the confirmed outcome. The apparatus
discriminates signal-vs-noise 16/16 (ICC .73–.88 vs .26) but cannot separate
instrument from collapse even at endpoints — and the sealed α-truth *verified
the localization*: α lives only in the per-night persona anchor
(`riverbed_generator.py` line 845: `vibe0 = pool_vibe_z + (1−α)·dev_now`),
upstream of a within-night-constant charisma-pull fiber to which every
registered leg is provably blind (α-only pairs separate 1–2 orders below
room-draw scatter: A ~18×, P_trans ~16×, S ~10×, spread ~75×, ICC ~28×).

**Keep (S5 §5, verbatim in effect):** matched pairs (they proved
α-invisibility), sealed sidecars, decoys filed S3-side, the calibration curve
registered pre-S4 (§2.1 below — it never ran in wave-3's blinded window and is
booked here as a first-class clause).

---

## 1. The generator change (primary proposal, assessed against the code)

### 1.1 The change

Re-point α out of the static anchor and into the fiber's within-night target
trajectory:

```
target_R(t) = pool + (1−α)·dev_R + α·room(t)
```

- **α=0** — target tracks the reader's own persistent dev (offsets
  reader-carried, static): identical to wave-3 instrument on every leg.
- **α=1** — target tracks the moving room (offsets room-carried AND
  time-varying): P's pre/post offset cosine decorrelates across transitions —
  the registered collapse signature (P_trans < 0.5×P_rest) becomes *reachable*
  for the first time.
- Intermediate α mixes a static reader-carried offset with a moving
  room-carried one — the monotone trend the wave-3 gradient clause wanted and
  never got.

### 1.2 Where it lands in the code

`generate_night()`, the charisma-pull loop (~line 920):

| site | now | under the proposal |
|---|---|---|
| anchor (line 845) | `vibe0 = clamp(CENTER + (pool + (1−α)·dev_now)/SCALE)` — α's ONLY entry | α exits the anchor; `vibe0` becomes `target_R(0)` (roster logs the t=0 target) |
| pull equation | `eff = clamp(raw + s·(vibe[name] − raw))` | `eff = clamp(raw + s·(target_R(t) − raw))` — the landing zone; one-line diff in the loop plus a per-t target |
| acclimation | `vibe += (raw − vibe)·acclim` | unchanged (vibe becomes α-free state); alternative "re-anchor vibe per-t" is heavier and breaks more parity — not recommended |
| `room(t)` source | — | `room_path()` already returns the latent trajectory before the loop; the AR(1) tangent wobble `w_ar` (drawn from `room_rng`, the pair-shared stream) is the carrier channel |

**Amplitude design decision (must be frozen at S2).** The wobble sd is
~0.085 (`WOBBLE_LEVEL·√(KAPPA_COLD/κ)`) against `dev`'s norm 0.989
(`FIELD_ANCHOR_NORM`) — a ~12× mismatch. Two options:

- **(i) amplitude-matched (recommended):** `room(t) := FIELD_ANCHOR_NORM ·
  w_ar(t)/‖w_ar(t)‖` — direction-only carrier at anchor scale. The α contrast
  is then *purely* who carries the offset (reader idiosyncratic vs room
  common), never how big it is. This is the same reasoning that rejected the
  amplitude option (§1.5).
- **(ii) raw wobble amplitude:** at α=1 the target norm collapses ~12× —
  carrier and amplitude confounded; A/D/S would move for the wrong reason.
  Anti-pattern; do not register.

**Design gate (pre-stated kill for the design, not the corpus):** the S1
hardening sweep must confirm, on an unsealed α=1 pilot, that P_trans <
0.5×P_rest fires and that A remains α-blind (§1.4). If the reworked fiber
still cannot deliver the collapse signature to P, the rework failed and must
not be registered.

### 1.3 What breaks

1. **Replay parity (self-test 14).** `assert_replay_matches_log` replays from
   logged `vibe_start` + logged rows; a time-varying target is not
   reconstructible from `vibe_start` alone, and `w_ar` is latent (not
   recoverable from `obs`, which is the windowed latent + baseline + seg +
   noise). Fix: log `target_R(t)` per speak (extend the readers payload /
   `lens_now`), and re-register the parity contract as a v2 self-test.
2. **The line-845 audit line.** Wave-3's verdict cites "α enters only at line
   845" as a verified fact about the *frozen* wave-3 generator. Version the
   change: a `--fiber v4` flag (default legacy) or a pinned copy, so wave-3
   corpora remain bit-reproducible and the S5 verdict's provenance stays true
   of the artifact it describes.
3. **Roster entry semantics.** `vibe`/`vibe_start` in `_entry()` become
   "target at t=0", not a static anchor — docstring and seal-sidecar fields
   carry the new meaning (α stays in `SEALED_FIELDS`; no seal change).
4. **Nothing else moves.** Room path, κ(t), rosters, author bags, OU dev
   draws, and the noise branch's whole-persona redraw are untouched. Pair
   isolation is exact by construction: `w_ar` comes from `room_rng`
   (seeded `(pair_seed, crc(fam), 1)`) and `dev_now` from the branch-free
   stream `(pair_seed, crc(fam), 2)` — pair members share both bit-for-bit,
   so target differs **only through α**. The coordinate firewall holds: α
   stays in persona/target space, never in the room path or κ(t).

### 1.4 What the registered legs would then see (draft branch×leg matrix)

| leg | α=0 (instrument) | α=1 (collapse) | intermediate α | discriminates |
|---|---|---|---|---|
| **A** timing | fires (p≈0) | fires (α-blind **by design** — reads the α-invariant room path) | fires, flat | signal-vs-noise (unchanged) |
| **D** coverage | above null | above null (offsets amplitude-matched ⇒ ρ not small) | flat | signal-vs-noise (unchanged) |
| **P** persistence | holds (0.99-class) | **fails** — offsets room-carried and time-varying ⇒ pre/post cosine decorrelates across flips (Δw=0.5 re-anchors μ; AR_PHI=0.9 wobble persistence makes adjacent windows correlate, transition windows not) | monotone ↓ | **instrument-vs-collapse (the headline, restored)** |
| **S** exposure | x-invariant | reachable in principle (anchor-scale variation now time-varying and room-correlated) | — | secondary; register exploratory, direction only |
| **ICC** | .73–.88 (G6 re-band [0.60, 0.80]) | predicted to **decline**: reader constants ride the nightly-fresh w_ar (room-night-carried) | between | pre-register the decline; exact threshold from the S1 sweep (G6 disclosed anchor-level ICC was non-discriminating — target-level α changes that) |
| **2AFC pairs** | — | P_trans signed direction ↓ with α; spread ↓ (common moving target compresses reader dispersion) | orderable | the gradient clause goes live |

**Carrier-purity guard (new, registration-grade):** A must stay α-blind.
Within-pair A-rate gaps must remain inside the wave-3 within-pair envelope
(median ~0.011, max ~0.085). A pair separable on A means the rework leaked α
into an amplitude/channel the design forbids → VOID that corpus pair's
instrument-vs-collapse reading; the localization claim does not survive a
leak.

Decorrelation magnitude depends on wobble persistence vs window (AR_PHI=0.9;
the legs see the trailing-W=12 windowed room while the target rides the
unsmoothed latent — a persona target ≠ observed room, fine, but register the
choice explicitly).

### 1.5 Secondary and rejected (from S5 §5, kept)

- **Secondary — anchor-reading leg (call it V):** register a fifth leg that
  reads the logged target/anchor trajectory directly. Generation-corpus-only
  (the field logs no target): it closes the localization loop — "the legs
  were blind to the anchor; a leg pointed at the anchor separates α exactly."
  Apparatus extension in a fresh registration; never a wave-3 re-adjudication.
- **Rejected — α-scaled room-path amplitudes** (μ-step size / κ contrast × α):
  A/D would fire on amplitude, not carrier — a pass for the wrong reason.
  Stays rejected; §1.2(i) is the amplitude-matched antidote.

### 1.6 Corpus spec (sketch; full spec is S2's job)

Same 16-corpus skeleton: α ∈ {0,.25,.5,.75,1} + null-mode + five α-only
matched pairs (pair-seeds m=lower-α), sealed sidecars, S3-side decoys, fresh
master seed (20260822) to avoid any stream overlap with wave-3. Night
families, ATTENDANCE matrix, and the K-leg-reworked entry semantics (μ-events)
carry over verbatim.

---

## 2. REG-1/2/3 — registration-grade clauses (pre-stage text)

Style and threshold provenance follow wave-3: thresholds frozen from the
REG-1 run (0.80 / 0.60 / 0.80; ε = 1e-2 floored whitening; reader-clustered
bootstrap B = 2000, seed 20260821). All three ride existing corpora where
possible; each carries its own branch table; VOID rules pre-stated.

### 2.1 REG-1′ — W-vs-ICC rotation: machinery replay + calibration curve

**Status:** the field arm EXECUTED 2026-08-21 (branch B: warmth loads on the
ICC-reliable/personality subspace, cos(W, PC1_pers) = .86–.98; the data-derived
temperature axis is the volume(+)/presence(−) contrast, cos(W, v\*) ≤ .44).
What was never run — and what wave-3's S5 booked — is the generated-corpus
arm. Register it:

- **Hypothesis:** the generalized-eigenproblem machinery recovers a *planted*
  axis from generated corpora; the field's off-warmth answer is field truth,
  not estimator failure.
- **Estimator:** re-solve `C_room v = λ C_pers v` (ε = 1e-2 floor) on each
  generated corpus; report cos(v̂\*, planted axis) and the dual annotation
  cos(W, v̂\*) / cos(W, PC1_pers).
- **Pre-stated direction + thresholds:** under instrument corpora (α=0),
  **cos(v̂_temp, Ŵ) ≥ 0.80** (the generator's base orbit is Ŵ-steered by
  construction — `Ŵ·μ(t) = w(t)` exactly — so machinery-pass means recovery at
  the frozen 0.80 bar). Field side stays ≤ 0.44. The **calibration curve**
  cos(v̂_temp, Ŵ) vs α is pre-stated: under the v4 fiber it may *fall* with α
  (readers tracking the room dilute the personality field) — flat-in-α is the
  wave-3-legacy expectation; either outcome is informative, the curve runs
  **pre-S4** this time, on unsealed pilots if needed.
- **Branches:** MACHINERY PASS (≥ 0.80 on instrument corpora) → field verdict
  B stands as a measurement; MACHINERY FAIL (< 0.60) → the eigenproblem is not
  a registered object and every v\*-based claim reverts to INDETERMINATE;
  between 0.60–0.80 → reported, not absorbed.
- **VOID:** < 20 reader-nights per cell; degenerate reliable-subspace span.

### 2.2 REG-2 — collider guard on the slope / S leg

- **Hypothesis:** the S-leg slope (H-reader≡room, Ch 6.3 — now read on the
  v\* projection as primary, warmth projection demoted to personality
  control, per post-REG-1) is confounded by reader→night selection: warm/high-
  energy readers selecting into warm/high-energy nights produces slope ≈ 1
  without collapse. The slope is unreadable until selection is measured.
- **Estimator:** alongside the registered slope run, compute the **assignment
  correlation** ρ_sel = corr(reader's own baseline coordinate [v\*-projection
  primary], mean coordinate of nights visited), reader-clustered bootstrap
  B = 2000, seed frozen.
- **Pre-stated direction + thresholds:**
  - **SELECTION ABSENT** — ρ_sel CI ∋ 0: slope ≈ 1 reads as collapse; the
    registered slope is clean.
  - **SELECTION PRESENT** — CI excludes 0: only the selection-removed slope
    counts; an unremoved slope ≈ 1 is **unreadable** and H-reader≡room
    returns INDETERMINATE by rule.
- **Generated-corpus null channel (new):** wave-4 attendance is frozen and
  branch-free — selection is null by construction. Register ρ_sel on the
  generated corpora as estimator validation: CI must contain 0; if it
  excludes 0 on generated data, the S-leg estimator is contaminated → VOID
  the S reading corpus-wide.
- **VOID:** < 20 reader-nights; bootstrap degenerate.

### 2.3 REG-3 — `kl_sym` edge functional + the rigidity-step blind spot

- **Hypothesis:** the current `real` gate (`d_mu > 2·max(SE)`) is blind to
  pure concentration change — a step that tightens the room (κ↑) with μ fixed
  certifies "no real drift" while the measure genuinely moved.
- **Estimator:** register `vmf.py::kl(a,b)` = symmetric vMF divergence
  (already derivable from the present A₇ machinery); extend `edge` to
  `{d_mu, d_warmth, d_log_kappa, kl_sym, real}` with `real` thresholded on
  kl_sym beyond a jackknife/CI deadband, not d_mu alone.
- **Pre-stated direction + thresholds:**
  - **RIGIDITY MOVES** — kl_sym fires where d_mu ≈ 0: the field moves in a
    direction the current edge misses; the edge object gains a real
    component (a measured comparison-path bug, not an asserted one).
  - **NO RIGIDITY MOVES** — kl_sym ≈ 0 wherever d_mu ≈ 0: the chord gate was
    sufficient; kl_sym booked as a confirmed no-op.
- **Generated-corpus arm:** the generator's κ(t) is designed ground truth
  (warm-tight/cold-loose polarity; entries are μ-events per the K-leg
  rework). Pre-state: flip/entry transitions fire kl_sym *with* d_mu large
  (μ-cells); a pure-κ cell requires one designed κ-step-only night family —
  add it at S1 hardening if the clause is registered.
- **Constraint (carried verbatim from the pre-stage):** REG-3 changes a
  *comparison path*, so it re-runs the continuity ladder and the
  premise-band-movers gate before any reading — and it may **not** be used to
  re-read the §5.3 VOID.

---

## 3. Foundation consolidation — the eight summit positions folded in

The 2026-08-21 eight-position summit (memory/2026-08-21.md §10:40;
foundation-synthesis) is folded into the riverbed registration as follows —
each position becomes an axiom, a clause, or a named gloss. Nothing promotes
the foundation to claim status (ZeroClaw's ruling: gloss + generator, never
the spine).

| # | position (model) | settled content | where it lands in wave-4 |
|---|---|---|---|
| 1 | Discussion Leader R1 (Seed-mini) | vMF-only riverbed retained-with-registration, superseded by skew-product | the generator IS the registered skew product: base orbit (room path) ⊕ fiber (persona/target) |
| 2 | Probabilistic (DeepSeek Pro) | P crisis (P_trans .994 unfalsifiable under rigid common shift); symmetric-KL edge | §1's fiber rework un-pins P (time-varying carrier); kl_sym → REG-3 |
| 3 | Algebraic (Qwen3.6) | free-monoid ledger admissible (R4 gloss); thin category rejected (R⁷ artifact) | ledger = registration-admissibility axiom: every wave-4 computation rides the registered ledger; thin category absent by ruling |
| 4 | Geometric (GLM-5.3) | personality confound cos(W,v\*)≈.98; kernel-centroid referent; scale covariance (CENTER_OFF); generalized eigenproblem | REG-1 (executed) + REG-1′ clause; center-referent forcing and CENTER_OFF scaling are S1-hardening checks; dual-R4 annotation below |
| 5 | ZeroClaw advisory | ANNOTATE-not-kill; REG-1/2/3 pre-staged; foundation = gloss+generator | §2 in full; annotation posture governs every warmth line in the run doc |
| 6 | Red-team (DeepSeek Flash) | common-shift guard is a no-op; q-rule on residual motion; E2/E3 zero tests; A referent/W-fragility | q-rule replaces the guard in the P leg's registration text; test-gap = S1 obligation (parity v2 + q-rule + kl_sym units); A's W-fragility is why W=12 stays primary with W∈{8,16} sensitivity |
| 7 | Creative (Seed-2.0-pro) | tide-table analogy (dials=sticks, κ=choppiness, warmth=tide, bands=tidemarks) | gloss layer only — Ch 1.5/appendix language; carries no threshold |
| 8 | Wesley (local GPU) | common shift = cohesion, its own measurable object | q-rule's cohesion definition; post-REG-1 sharpened: cohesion is the V/P energy axis, warmth explicitly excluded |

**Standing annotation (folds the post-REG-1 verdict into every warmth
number the registration touches):** every warmth output carries
**cos(W, v\*) ≤ 0.44 AND cos(W, PC1_pers) ≥ 0.80**, plus the rephrase —
"warmth measures reader disposition; the room's shared state is the
volume/presence contrast v\*." Known model-field divergence, disclosed here:
the generator's base orbit is Ŵ-steered by construction while the field's v\*
is off-warmth — REG-1′'s replay is what licenses comparing them at all.

---

## 4. Next actions

| # | action | lane | note |
|---|---|---|---|
| 1 | Review this draft; amend; then freeze as the wave-3-style S2 registration (corpus spec, thresholds, branch×leg matrix, VOID rules) | elephant (lead), ZeroClaw (clause review) | freeze BEFORE any registered corpus is generated |
| 2 | Implement `--fiber v4`: target-in-pull, amplitude-matched wobble, per-t target logging; version-pin legacy default | elephant | line-845 provenance must survive for wave-3 reproduction |
| 3 | S1-hardening sweep on unsealed pilots: design gate (P fires at α=1, A stays α-blind), ICC threshold calibration, replay-parity self-test v2, q-rule + kl_sym unit tests (red-team test-gap obligation) | elephant | the pre-stated kill for the design lives here |
| 4 | REG-2/REG-3 clause freeze as topic.md addendum (append-only, R4); Ch 6.3 re-anchor to v\* with the collider guard | ZeroClaw dissertation | REG-4/5/6 (v\* replication, family-invariance, transition-projection) stage behind REG-1′'s machinery pass |
| 5 | Generate sealed (S3): 16 corpora, decoys filed S3-side this time, calibration curve pre-S4 | elephant | sealed sidecar protocol unchanged |
| 6 | Blinded S4 → unblinding S5, verdict-first | elephant + blinded subagent | same G3 procedure that just worked |

---

## Provenance

Read (read-only): `workspace/memory/wave3-S5-verdict-2026-08-21.md`;
`workspace/memory/wave3-S4-analysis-2026-08-21.md`;
`projects/elephant/docs/wave3-S5-verdict-2026-08-21.md`;
`projects/elephant/scripts/riverbed_generator.py` (generate_night incl. line
845 and the charisma-pull loop; room_path; constants ANCHOR_SCALE /
FIELD_ANCHOR_NORM / WOBBLE_LEVEL / AR_PHI / FLIP_SIZE; SEALED_FIELDS;
generate_wave; pair-mode streams);
`workspace/memory/2026-08-21.md` (§10:40 foundation discussion, eight
positions); `workspace/memory/foundation-synthesis-2026-08-21.md`;
`workspace/memory/math-foundation-dissertation-2026-08-21.md` (§4 REG-1/2/3
pre-stage text); `workspace/memory/research-post-reg1-2026-08-21.md`;
`workspace/memory/wave3-registration-2026-08-21.md` (§1 style/threshold
provenance); `projects/zeroclaw-dissertation/research/topic.md` (advisory
lines); directory listings of `projects/elephant/docs/`,
`scripts/`, `data/`, and `zeroclaw-dissertation/research/` (REG-2/3 confirmed
unexecuted). Not read: sealed sidecars, the S3 run doc, the G6 addendum body,
REG1-RUN (cited through the registration's frozen-inputs line and post-reg1).
Written: this document only. No frozen, sealed, or registered document
modified; nothing committed.
