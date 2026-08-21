# Kimi Ideation — Elephant/JEPA Next Move

**Filed: 2026-08-21.** Ideation-only; no repo changes, no runs. Grounded in
`elephant-next-move-research-2026-08-21.md` (Paths A/B) and
`discussion-leader-round1-2026-08-21.md` (the Riverbed Vision). File citations
refer to the elephant repo at `/home/eileen/projects/elephant`.

---

## 0. The Reframe: Riverbed Dissolves the A/B Fork

The research note poses Path A (more reader-nights / re-registered hysteresis)
vs Path B (length-matched generation corpus) as alternatives with a sequencing
question. The Riverbed framing says they are the same object viewed from two
directions:

- The **field corpus** is the *inverse problem*: given readings, infer which
  field process generated them.
- The **generation corpus** is the *forward model*: given a field process
  (branch parameters), produce readings.

Once the generation corpus is specified as a **sample path of the time-indexed
vMF random field** — room trajectory (μ̂_room(t), κ(t)) on S⁶ plus a second-level
reader-deviation field — the registered statistics stop being a battery of
one-off tests and become an **inversion of a known forward model**. The verdict
question becomes model selection: which branch of the forward model inverts to
the field corpus? That is a stronger dissertation claim than either path alone,
and it tells us what to build first: the forward model, because it is also the
calibration instrument. **Path B is not the endgame after Path A; it is the
instrument that makes Path A's wave-3 interpretable.**

Recommendation in one line: **build the riverbed generator (two simulators,
α-sweep branches, adversarial-pair design), then run wave-3 through an
attendance matrix designed on a-priori persona warmth, with hysteresis derived
from the noise floor rather than re-picked.**

---

## 1. Generation-Corpus Design Angles

### 1.1 The corpus IS a field sample path, not a text factory

Specify generation at the level of the Riverbed's sufficient statistics:

- **Room level:** μ̂_room(t), κ(t) trajectories per night. Registered strata
  transitions (the SEG warm→cynical flips) are jumps in μ̂_room; entry steps are
  primarily **κ events** (the room gains a mode / re-concentrates) — see §4.
- **Reader level (the JEPA-of-a-JEPA, Agenda Problem 3):** each reader is a
  second-level vMF field with their own (μ̂_R, κ_R) around their deviation from
  the room. The branch parameter α then has an exact field meaning:

  - **Instrument (α=0):** μ̂_R ⊥ room; κ_R high; persistent across nights.
  - **Collapse (α=1):** reader's sampling distribution *is* the room's
    (μ̂_R = μ̂_room(t), no persistent second level).
  - **Noise:** κ_R ≈ 0, μ̂_R redrawn per night.
  - **Intermediate:** mixture — reader deviates from room by
    (1−α)·(μ̂_R − μ̂_room) plus noise.

  This is cleaner than the research note's "α × room warmth + (1−α) × fixed
  vector" (a linear-warmth scalar parametrization): it lives in the same
  geometry as the estimator, so "recovery" means recovering a *direction and a
  concentration*, not a scalar weight on the very axis the S leg regresses on.

### 1.2 Two simulators, one field measure (the contamination firewall)

Generate every corpus condition twice:

1. **Engine-native (TapNightSession):** branch parameters expressed purely in
   *persona space* — `dial_weights`, `acclimation_rate`, `charisma`,
   `vibe_start` (the constructor inputs at `scripts/e2_nights.py:121-125`) —
   plus script text. The readings emerge through the engine's actual dynamics
   (acclimation relaxation, `elephant/field.py:108-121`; charisma pull,
   `elephant/field.py:147-160`). Nothing in this path ever computes an offset
   from a roster mean.
2. **Direct vMF sampler:** sample readings straight from the field measure
   (room path + reader deviations + deadband noise), bypassing text and engine
   entirely. Cheap, so it carries the power sweeps (how many nights/readers
   does the apparatus need per branch?) and the α-sweep.

The two simulators answer different contamination questions: the vMF sampler
tests whether the *statistics* discriminate (do A/D/P/S separate the branches
when the truth is vMF-shaped?); the engine path tests whether they discriminate
*through the full text→dial→reading transformation* — the "hard version" the
research note's §5.1 asks for. **If the apparatus recovers the branch under one
simulator but not the other, that gap is itself the finding** (it localizes
where the inference leaks: statistical vs engine-level).

### 1.3 Adversarial pairs, not absolute recovery

Absolute recovery ("does the slope CI contain 0 when α=0?") is low-power and
verdict-fragile. The sharper design is **paired discrimination**: generate
corpus pairs that differ *only* in α (matched speak counts, warmth ladders,
rosters, κ(t) profiles), and ask the registered pipeline to *rank* the pair.
This is a 2AFC psychophysics design — it converts the calibration question
from "is the CI in the right place" (needs many corpus replicates) into "does
the statistic order the conditions correctly" (needs far fewer), and it is
robust to any bias that affects both conditions equally. Pre-register: for
each leg, the signed direction of the statistic's response to increasing α.

### 1.4 Match the field corpus's *hard* statistics, not the easy ones

"Length-matched" must mean more than speak-count histograms (20–46). The field
corpus's difficulty lives in:

- **ICC = 0.9076** — reader baselines are stable but *not* perfectly stable
  across nights. A synthetic corpus with frozen reader vectors (between-night
  variance = 0) is strictly easier than the field and the calibration will
  overstate the apparatus's power. Model reader baselines as OU-like processes
  in the tangent space of S⁶ around μ̂_R, with between-night variance tuned to
  reproduce ICC ≈ 0.91. **This is the single most important honesty parameter.**
- **The drift floor:** the filed 0.29 noise floor and 0.75–0.93 transition
  spikes in d_R (commented at `scripts/premise_band_movers.py:178-181`). The
  generator's deadband noise must reproduce the stable-phase d distribution,
  or every hysteresis constant is calibrated against the wrong floor.
- **The ladder:** Sxx ≥ 0.19 over 0.319–0.759, three x-bands — already in the
  spec, keep it.
- **P ≈ 0.994 within-night persistence.** If the generator's reader process
  doesn't reproduce within-night persistence *and* between-night ICC
  simultaneously, the generation model is structurally wrong, not just
  miscalibrated — that joint check is a free validation of Agenda Problem 3's
  two-level model before any branch test runs.

### 1.5 Minimal viable corpus

The research note's open question 4 (matched to what?) has a clean answer
under the pair design: **one wave** (9 nights, 21 readers, the same family
templates and transition positions as `scripts/e2_nights.py:100-113`) per
branch condition, times the α-grid {0, 0.25, 0.5, 0.75, 1} via the vMF
sampler, plus engine-native generation at the three endpoints {0, 1} and the
field-like α*. That is 5 cheap corpora + 3 engine corpora, not 36-night
replication. The engine corpora are ~minutes each (the wave-2 precedent).

---

## 2. Contamination Control: Reader Baselines Independent of the o/d Decomposition

The research note's hard question 1 is exactly right: if the generator sets
"baseline = room mean + fixed vector," the estimator's o_R numerator *is* the
generation parameter (`night_windows` computes o_R as RMS deviation from the
roster windowed mean, `scripts/premise_band_movers.py:220-228`). Rules to keep
it honest:

### 2.1 The coordinate firewall

The estimator's coordinate system is: roster-mean subtraction, split-half
(6/6) displacement, corpus_sd normalization, the ICC-reliable subspace
`["mood", "volume", "earnestness", "presence"]` (`premise_band_movers.py:81`),
and the WARM projection (`elephant/vmf.py:57-60`). **The generator must never
express a branch parameter in any of these coordinates.** Persona space
(dial_weights, vibe_start, acclimation_rate, charisma) and text space only.
Concretely, define the branches as statements about *persona resampling*:

- **Instrument:** each reader keeps one persona across all nights they attend
  (identity-preserving attendance — what the field corpus assumes).
- **Collapse:** each night, every attending reader's persona is *redrawn* from
  a room-warmth-conditioned persona distribution. The reader name persists;
  the instrument doesn't. The estimator must discover that "the same reader"
  is a different instrument in warm vs cold rooms.
- **Noise:** persona redrawn per night unconditionally.

The generator then has no access to o/d structure — collapse vs instrument is
a fact about the *persona assignment map*, which is exactly what the estimator
is claiming to infer. The o/d machinery appears only on the analysis side.

### 2.2 The decoy-estimator audit

Contamination that survives the firewall shows up as *estimator-specificity*:
only the o/d decomposition recovers the branch. Pre-register a panel of three
analysis pipelines on every generated corpus:

1. The registered o/d pipeline (`premise_band_movers.py`).
2. A per-reader detrending estimator (reader fixed effects on raw z-dials,
   no roster-mean subtraction).
3. A mixed-effects model on raw dials (reader random intercept, room-warmth
   fixed effect).

**Pass condition: branch-consistent verdicts across all three.** If only (1)
separates the branches, the result is about the estimator, not the data — the
contamination signature made measurable. This directly answers "did we measure
the apparatus's discrimination power or the generation model's prior": the
decoys share the data but not the decomposition.

### 2.3 Procedural blindness

The research note's adversarial split (one agent generates, another analyzes)
is cheap and worth formalizing: the analysis script takes a corpus path and a
manifest *with the α field redacted*; verdicts are filed before unblinding.
The existing manifest discipline (`data/e2/e2-nights-manifest-w2.json`,
sha256 + stripped-md5 + determinism check, `scripts/e2_nights.py:147-155,
184-226`) is exactly the right substrate — add a `branch` field that is
written but sealed until after the registered run.

### 2.4 Hold out the forensic channel

Keep one analysis-side object *unknown* to the generator by construction:
`corpus_sd` and the WARM direction are computed from corpora the generator
never sees (the filed wave-1/wave-2 data). The generator must produce a corpus
whose *own* corpus_sd and ladder land inside the wave gate without ever being
told the targets — if it can only pass the gate by being handed 0.2367, the
corpus is fit-to-gate, not field-like.

---

## 3. Hysteresis Re-Registration — Derive, Don't Re-Pick

The current constants are hard-coded: `HYST_MARGIN = 0.05`, `HYST_HOLD = 3`,
edges 0.3/0.6, W=12 (`scripts/premise_band_movers.py:71-75`). Re-registering
3→2 or 12→8 to clear §5.3 is legitimate under the file-before-measure
doctrine, but it *reads* like threshold-shopping because the constants have no
derivation. The Riverbed's Problem 5 supplies the missing derivation — so
**re-register the derivation, not the values**:

### 3.1 Margin from the field's noise floor

The margin exists to ignore score jitter. But the apparatus already ships a
jitter measure: the jackknife SE(μ̂) in `vmf_fit` (`elephant/vmf.py:156-159`),
which the edge detector already uses as a deadband (`elephant/vmf.py:174-190`,
`real = d_mu > db_factor · max(SE)`). Register: **margin = c·SE(ρ_R(t))**,
where SE(ρ) is obtained by jackknifing speaks within each window, and c is
filed (2.0, matching the existing `db_factor` convention). The margin then
varies with the local noise floor — tight in stable strata, wide near
transitions — which is precisely the phase structure the static-ratio finding
described. This likely *increases* counted crossings in stable regions
(smaller margin) while honestly suppressing noise crossings near steps.

### 3.2 Hold from the window geometry

A transition's effect on d_R persists for exactly ~W windows as the step
crosses the split-half window — the dip duration is a deterministic function
of W and transition sharpness, not a free parameter. Register
**hold = f(W)** derived from the sharpest registered transition (the SEG
flip): at W=12, a hard flip produces a sustained ρ excursion of W−1 windows;
requiring hold=3 confirms *sustained departure* at a quarter of the dip
length. File the derivation; the value may stay 3, but now it has a reason.

### 3.3 The event-pooling question (flag for the Captain)

§5.3's floor (≥20 counted down-crossings) is a power requirement for leg A's
circular-shift null, and it is a property of the *event set*, not of any leg.
Wave-1 and wave-2 produced 19 and 17 events — 36 pooled. The design's
"never pooled" rule governs the *legs*; whether it governs the *floor* is a
filing question. A narrow addendum — "§5.3 evaluates on the pooled event set
of all gate-passing waves; legs A/D/P/S remain per-wave, never pooled" —
clears the void without touching hysteresis at all. **This is the smallest
possible re-registration and should be put to the Captain as option 0** before
any margin/hold/W change. Risk: a hostile reader may see the floor as part of
the test's registration; mitigated by the fact that pooled events serve the
floor's exact statistical purpose (null power).

### 3.4 Register a robustness manifold, not a point

Whatever primary is filed, pre-register the verdict's invariance across a
grid: margin ∈ {1.5, 2, 3}·SE, hold ∈ {2, 3, 4}, W ∈ {8, 12, 16}. Report the
verdict *set*. This inverts the threshold-shopping accusation into a
registered sensitivity surface — the same move the run already makes with the
W=8/W=16 sensitivities and the start-referent sensitivity
(`premise_band_movers.py:799-803`), promoted from afterthought to
registration.

---

## 4. Entry Steps Without Rigging

### 4.1 The mechanistic hypothesis: entry steps are κ events, not μ̂ events

Why do hard SEG flips produce crossings and mild entry steps don't? Under the
Riverbed, a scripted warm→cynical flip moves μ̂_room for *every* speaker at
once — a large Δμ̂, hence a large split-half d_R for the whole roster, hence
ρ collapses through the 0.6 edge. A single newcomer entering a 7–10-person
room shifts the roster mean by ~1/n and injects drift mostly into *their own*
trajectory: the μ̂ step is small for incumbents. **Entry steps are
concentration and roster-composition events; flips are mean-direction
events.** The current D leg only looks for μ̂-flavored signatures (ρ
crossings driven by d_R spikes). If this hypothesis is right, pushing D over
50% by making entries "harder" fights the estimator's geometry.

**Cheap decisive check, available today, read-only:** the field corpus on disk
already contains the answer. Compute κ(t) per window (`vmf_fit` on windowed
readings) around the registered entry steps (T4a@12, T4b@28, T5/T5c@24) vs
matched stable windows. If κ(t) moves at entries while μ̂ barely does, the
hypothesis is confirmed and the correct fix is a **K leg** — κ-gradient at
registered transitions vs null nights — which uses information the current
registered statistics simply never touch. κ is already estimated with a
bootstrap CI in `elephant/vmf.py:144-154`; the leg writes itself.

### 4.2 The entrant's own trajectory

`acclimation_curve` (`elephant/field.py:108-121`) is the engine's own model of
what an entrant does: relax from off-room to room at their acclimation rate.
The entrant's ρ(t) around entry is the natural place to look for an entry
signature — they arrive with large o_R (far from room) and large d_R
(acclimating), and the *ratio's trajectory* during relaxation is a
discriminating shape (both numerator and denominator decay, at rates set by
different mechanisms). Currently the entrant's pre-entry windows are NaN by
construction (`premise_band_movers.py:170-172`), so their first windows are
an unexamined signal. Register a **D-entry companion**: fraction of entry
events where the entrant's own ρ(t) shows the registered acclimation
signature within the window budget. This measures entry steps where the
signal actually lives, instead of demanding the whole roster flinch.

### 4.3 Attendance-matrix design: a-priori persona warmth, not measured offsets

Assigning small-offset readers to entry-step nights is legitimate **iff** the
assignment variable is computed before measurement and from persona
parameters only. That variable exists: `persona_warmth`
(`premise_band_movers.py:532-546`) — the direction cosine of the persona's
z-standardized `vibe_start` against WARM, no measured readings involved.
Register: wave-3 attendance is a balanced incomplete block design on a-priori
persona warmth, ensuring every night type (including entry nights) spans the
offset range, so *some* reader sits near the 0.6 edge at every transition by
design. This is experimental design (how you'd stratify a trial), not rigging:
it never looks at outcomes, and it is exactly as filed as the current
ATTENDANCE matrix (`scripts/e2_nights.py:59-77`).

### 4.4 If stronger text is wanted: new frozen families, not tweaked ones

If the Captain wants harder entry steps in text: write a *new* registered
family (e.g., "hostile entry" — a newcomer whose staged lines are maximally
field-displacing entering a warm room), freeze the text, file it, generate
under new tags. The corpus is append-only and the T-tag precedent
(`e2_nights.py` header) is exactly this: new logs, frozen scripts, never
regenerate old ones. Strengthening an intervention is not rigging when the
intervention text is frozen before measurement and all void rules still
apply — but do it *after* the κ check (4.1), because if entries are κ events,
harder μ̂-text still won't produce ρ crossings and the effort is wasted.

---

## 5. How the Riverbed Changes What We Build

The six seams in the discussion-leader note are not a refactor list; they are
the experiment list. Each agenda problem retires a specific empirical
vulnerability:

1. **Problem 5 (hysteresis from detection uncertainty)** → retires the
   re-registration smell (§3 here). Build: SE(ρ) jackknife in the windowed
   estimator; filed derivation of margin/hold.
2. **Problem 4 (window referent)** → retires deviation note 1
   (`premise_band_movers.py:77-79`, 256-263). The center-referent is currently
   a post-hoc fix defended by arithmetic (W/2 = 6 > TOL = 3); the causal
   answer is that a transition becomes *detectable* when it occupies the
   split-half boundary, which sits at window center — that argument should be
   a filed lemma, not a deviation note.
3. **Problem 2 (static = phase-average of temporal)** → converts the
   0.5599/0.6088 "window-scale artifact" from an embarrassment into a theorem:
   the static ratio is the time-average of R(t) over strata, weighted by
   stratum length. Then the E2 in-band verdict and the band-movers phase
   structure are the same fact at two temporal resolutions, and the
   "continuity ladder" (`premise_band_movers.py:656-761`) becomes a corollary
   check instead of an independent anchor.
4. **Problem 1 (edges as gradients of sufficient statistics)** → **plural**
   statistics: μ̂ *and* κ. This is the K leg of §4.1. Note the codebase
   already half-knows this: `edge()` returns `d_mu`, `d_warmth`, AND
   `d_log_kappa` (`elephant/vmf.py:184-189`) — but only d_mu gates the
   `real` flag. The registered statistics inherited that single-channel
   blindness.
5. **The dual-warming seam** — worth stating precisely, because it's subtle:
   `RoomField.warmth()` (`elephant/field.py:52-68`) and `warmth_vmf`
   (`elephant/vmf.py:167`) use the *same weight vector* (vmf.py's WARM is the
   field.py weights, renormalized, in DIALS order) applied to *different
   objects* — raw re-centered readings vs the unit direction μ̂. So
   `warmth()` is magnitude-contaminated (collinear with field extremity —
   the `concentration()` proxy's docstring problem, `elephant/field.py:70-73`
   and `vmf.py:8-11`), while `warmth_vmf` reads direction only. The filed
   warmth ladders (X_W1/X_W2, `premise_band_movers.py:87-92`) and
   `room_warmth` should be re-expressed as Ŵ·μ̂ everywhere — one warmth, on
   the sphere — before the generation corpus bakes the old ladder in as its
   matching target.
6. **Problem 3 (two-level reader field)** → the generation corpus's reader
   model (§1.1) and the collapse-branch semantics (§2.1). This is the piece
   that makes "reader baseline" a well-posed parameter at all.

**Net build order the Riverbed implies:**

1. **κ(t) on the existing field corpus** (read-only, hours). Decides whether
   D's gap is a reframe (K leg) or a power problem (wave-3 design).
2. **The riverbed generator**: field-measure spec + direct vMF sampler +
   engine-native persona simulator, α-sweep, adversarial pairs, decoy-estimator
   audit (§1–2). This is Path B, and it calibrates everything downstream.
3. **Filed derivations** for margin/hold/referent + the §5.3 event-pooling
   question for the Captain (§3).
4. **Wave-3** with a-priori persona-warmth-balanced attendance (§4.3),
   analyzed with the derived hysteresis — now Path A runs on an instrument
   whose discrimination power is *measured*, not assumed.

The thesis-level payoff: if the generator shows the apparatus recovers
branches under contamination control, the field corpus's VOID becomes "the
instrument is sound; the field under-delivered events" — a power statement
with a calibration certificate attached. If it shows the apparatus *cannot*
separate instrument from collapse even at α endpoints, that is the honest
negative the research note's §4 contemplates — and the Riverbed then tells
you *why* (which sufficient statistic the estimator is blind to), turning
even the kill into a method contribution with a mechanism.

---

*Ideation only. No repos modified, nothing run.*
