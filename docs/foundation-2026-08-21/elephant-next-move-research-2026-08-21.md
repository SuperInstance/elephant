# Elephant Next Move — Deep Research

**Filed: 2026-08-21.** Strictly read-only research synthesis for the ideation session. No commits, no runs.

---

## 1. State of the Evidence

The premise-band-movers run (`PREMISE-BAND-MOVERS-RUN-2026-08-21.md`, results at `data/slope/premise-band-movers-results.json`) is **VOID BY RULE §5.3**: 17 counted down-crossings in the wave-2 primary (19 in wave-1 replication), both below the 20-event floor. No branch is declared. But the four registered legs are strongly informative:

- **A (timing) fires hard.** Wave-2: 0.647 (11/17), circular-shift null p = 0.0013. Wave-1: 0.632, p = 0.0001. Crossings time-lock to registered strata transitions with high precision. Both waves agree. W=8 sensitivity: 0.917/0.900 (p < 0.0001). W=16: dies (0.074). The timing structure is real but window-scale-dependent.
- **D (direction/coverage) fails the ≤50% bar but is clearly above null.** Wave-2: 4/10 = 0.40, CI [0.122, 0.738]. Wave-1: 5/10 = 0.50. Null-night rate: 0/7 and 0/8. D−D_null > 0. The missed transitions are the milder newcomer-entry steps (T5/T5c, D/D-cold), not the hard SEG warm→cynical shifts.
- **P (persistence) is a landslide for the premise.** P_trans = 0.994 vs P_rest = 0.9935; threshold is 0.5×P_rest = 0.497. Offsets survive strata steps essentially perfectly. Crucially, class-residual P also holds (0.992 vs rest 0.969) → identity-propagation booking does NOT fire — the persistent structure is individual, not just archetype.
- **S (x-invariance) is mixed.** Wave-2 primary: slope 1.41 [−0.31, +2.80], CI contains 0, competitor not beaten (p = 0.384) → x-invariant. Wave-1 replication: slope 1.24 [+0.33, +1.98], CI excludes 0 but competitor not beaten (p = 0.521) → indeterminate, not a registered falsification. The positive slopes (~1.2–1.4) hint warmer nights → higher scores, but the roster competitor absorbs it.

**The estimator finding (booked, descriptive):** at W=12 the premise score is clear-side (ρ ≈ 2–4) nearly everywhere; the static in-band ratio (0.5599/0.6088) is a window-scale artifact of the full-night estimator, not a phase-average of in-band phases.

**Prior state:** the wave-2 slope regression (Stage-2 run, 2026-08-20) was INDETERMINATE leaning alignment (slope −0.36, CI contains 0, excludes 1; class-residual tripwire fired on a knife-edge CI exclusion). ICC = 0.9076 — baselines are stable reader constants. The side-by-side cross-instrument verdict is "retired, leaning false."

**Bottom line:** A and P strongly support H-BM; D is borderline; S is the standing uncertainty. The void is a coverage problem, not a content problem. The data says the premise's temporal structure is real — there just aren't enough crossing events to satisfy the registered floor.

---

## 2. The Two Paths to a Verdict

### Path A: More Reader-Nights / Re-Registered Hysteresis

**What it would take:** Generate additional wave-3 reader-nights (or re-register with relaxed hysteresis) to push the counted down-crossing count from 17→≥20.

**Options:**
1. **Wave-3 corpus:** New T-tagged logs of the same 9 frozen families with an attendance matrix designed to maximize crossing events — specifically, more readers with small offsets (the ones that cross the 0.6 edge at transitions). The design §7 power analysis estimated ~85 reader-night transition events yielding ≥40 crossings; the actual yield was 17/85 ≈ 20%, suggesting hysteresis + the offset distribution are the bottleneck.
2. **Re-register with W=8 as primary:** A fires at 0.917/0.900 (p < 0.0001). But W=8 has even fewer events (12/10) — does not fix §5.3 alone.
3. **Re-register with 2-window hysteresis hold (currently 3):** The run doc notes this could increase event count but would be a re-registration (new thresholds filed before measurement).
4. **Add entry-step nights designed to be strong enough to push D above 50%.** Currently the mild entry steps (T5/T5c, D/D-cold) produce no crossings; a design with harder entry steps (or readers specifically chosen for small offsets on those nights) would help D.

**What must be satisfied:** All registered void rules from the design + the addendum: wave gate (roster match, corpus_sd 0.2367, ladder reproduction), ≥20 counted down-crossings, null-night rate < 50% of signal, continuity ladder ±0.10, bootstrap effective draws ≥1500.

**What already exists to reuse:** The entire generation apparatus (`e2_nights.py`, `e2_personas.py`, `e2_instrument.py`, the frozen script families, the 9 warmth-ladder positions, the Stage-2 wave gate). A wave-3 is a new attendance matrix + T-tagged runs — ~5 minutes of generation. The analysis script `premise_band_movers.py` is generic over wave. The filed wave-1 and wave-2 corpora stay untouched.

**Cost estimate:** Near-zero. Generation is CPU-local, deterministic, ~5 minutes. Analysis is ~5 minutes. The only cost is the Captain's time to approve the re-registration and the attendance matrix design.

**Single biggest risk:** Even with more reader-nights, the event count may not clear 20 without changing the hysteresis rules — and changing the rules opens the re-registration door, which the advisor's doctrine handles cleanly (file thresholds before measurement), but which may look like threshold-shopping to a reader unfamiliar with the registration discipline. A second risk: D may never clear 50% if the fundamental mechanism is that only hard SEG transitions produce crossings, and there are only so many of those per night.

### Path B: The Length-Matched Generation Corpus (the Endgame)

**What it would take:** Build a synthetic corpus where every branch is pre-stated: generated rooms at known warmth values, generated readers with known baseline properties, and the premise score computed on data whose ground truth is fully specified. This is abstract sentence 6's decisive experiment.

**Why it's decisive:** The field corpus's indeterminacy stems from not knowing whether reader baselines are instruments or room-warmth proxies. A generation corpus where you *set* both the reader's true baseline and the room's true warmth eliminates this ambiguity — the slope becomes a calibration check, not a discovery.

**What must be satisfied (from the registered rules + abstract sentence 6):** See §3 below — the full requirements spec.

**What already exists to reuse:** The elephant engine (TapNightSession, SEG banks, vmf fitting, the 7-dial schema), the e2_instrument measurement pipeline (Night, readings, corpus_sd, the premise-score machinery), the Stage-2 corpus design's attendance-matrix methodology, the wave gate framework. The generation infrastructure is battle-tested.

**Cost estimate:** Higher. Requires: (a) designing the generation model for readers (how to parametrize "true baseline" in a way that maps onto the 7-dial measurement), (b) generating enough nights to match the field corpus's length statistics (20–46 speaks/night, 7–10 readers/night, the same warmth ladder), (c) pre-stating every branch (instrument vs collapse vs noise) as generation parameters, (d) running the same analysis pipeline on the generated data and checking the registered verdict against the known ground truth. The generation itself is cheap; the design work (contamination control, ensuring the generation doesn't bake in the answer) is the real cost.

**Single biggest risk:** **Contamination.** If the generation model's reader-baseline mechanism shares statistical structure with the analysis's estimator, the calibration "passes" for the wrong reason — you've measured the generation model's prior, not the apparatus's capacity to discriminate. This is the hard problem: making the generation corpus honest enough that a correct verdict is evidence of the apparatus's discrimination power, not of the generation model's faithfulness.

---

## 3. The Length-Matched Corpus — Requirements Spec

Every registered constraint the generation corpus must meet, extracted from the design documents and the advisor note:

1. **Length matching:** Generated nights must match the field corpus's length distribution (20–46 speaks/night). Each generated night's speak count falls within the field range; the corpus-level distribution is not statistically distinguishable from the field (KS test or equivalent, registered).

2. **Branch pre-statement:** Before generation, every branch of the premise is pre-stated as a generation parameter:
   - **Instrument branch:** reader baseline = a fixed vector in dial-space, invariant across rooms.
   - **Collapse branch:** reader baseline = room warmth × 1 + noise (the H-reader≡room identity).
   - **Noise branch:** reader baseline = random draw per night, no cross-night structure.
   - **Intermediate:** reader baseline = α × room warmth + (1−α) × fixed vector + noise, for specified α ∈ (0, 1).
   Each generated corpus condition corresponds to one branch specification. The analysis is run blind to which condition generated which data.

3. **Reader counts:** Same structure as the field — 21 readers (or the wave-1 15), attending 3–4 nights each, 7–10 readers per night. The attendance matrix is a design parameter, not a random variable.

4. **x-band stratification:** The field's three warmth bands (cold ≈0.48, mid ≈0.64, warm ≈0.71) must be represented. Generated room warmth values must span the same ladder range (0.319–0.759, Sxx ≥ 0.19).

5. **Null nights:** At least one night per corpus with no strata transitions (like T9/S5). The null-night void rule must be satisfiable: crossing rate on null nights < 50% of signal-night rate.

6. **VOID rules (all must be satisfiable on the generated corpus):**
   - §5.1: Wave gate passes (roster match, corpus_sd, ladder).
   - §5.2: Null-night crossing rate < 50% of signal.
   - §5.3: ≥20 counted down-crossings.
   - §5.4: Continuity ladder within ±0.10.
   - §5.5: Bootstrap effective draws ≥1500.

7. **The registered statistics must recover the pre-stated branch:**
   - If the generation sets instrument (α = 0): the slope regression should declare alignment (CI contains 0, excludes 1); S should be x-invariant; P should hold; A should fire (if transitions are present and strong enough to produce crossings).
   - If the generation sets collapse (α = 1): the slope should declare collapse (CI contains 1, excludes 0); S should falsify the alignment arm (x-coefficient CI excludes 0 and beats roster competitor).
   - If the generation sets noise: the premise-band-movers should void or the legs should fail to fire.

8. **Strata transitions:** Generated nights must contain registered strata transitions (the SEG warm→cynical shifts) at known positions. The field corpus's transition positions (speak 8, 20, 24, depending on the night family) are the templates.

9. **The 7-dial schema:** Generated reader readings must live in the same 7-dimensional dial space (mood, volume, earnestness, cynicism, joke_landing, panic, presence). The ICC-reliable subspace (mood, volume, earnestness, presence) is the analysis target; panic is excluded by the two-line schema rule.

10. **Determinism and reproducibility:** Generation must be deterministic (seeded, byte-identical on re-run), matching the field corpus's convention.

11. **Contamination control (the new constraint, not yet registered):** The generation model must not share the estimator's statistical structure in a way that guarantees recovery. Specifically: the generation model's mechanism for producing reader baselines must be independent of the analysis's ρ = o/d computation. This likely requires an adversarial design step — have one agent design the generation, another run the analysis, and check recovery.

---

## 4. Gaps & Open Questions

**Genuinely unknown:**
- Whether the S leg's positive slope (~1.2–1.4) in wave-1 and the class-residual knife-edge are real signals or small-sample artifacts. The class-residual S CI excluding 0 in wave-2 ([+0.12, +1.47]) while the primary contains it is exactly the same pattern as the slope run's tripwire — but with 65 reader-night cells and 21 readers, the power to resolve it is marginal.
- The mechanism by which only hard SEG transitions produce crossings. The run doc identifies the coverage pattern (warm→cynical shifts yes; newcomer-entry steps no) but doesn't explain it mechanistically — it's likely a drift-magnitude threshold (hard steps produce d ≈ 0.6–0.9, mild steps produce lower), but this hasn't been formally tested.
- Whether the window-center vs window-start referent issue generalizes. The design's registered "≤3 speaks" was arithmetically blind at window-start (W/2 = 6 > 3), producing A = 0.000 — an artifact, not a finding. The center-referent fix is defensible, but it's a post-hoc correction to the registered spec.

**Needs the Captain:**
- Whether to pursue Path A (re-registration + wave-3) or Path B (length-matched generation corpus) or both. The advisor's note ranks Path B as "the endgame" and Path A as "the vanguard," implying Path B is the strategic target and Path A is the tactical next step — but the Captain decides sequencing.
- Whether to re-register the hysteresis hold (3→2 windows) or the primary window size (12→8) to clear §5.3. Both are legitimate re-registrations (file before measure), but they change the floor and thus the verdict space.
- The JEPA naming question (consensus OPEN #2) — this affects how the generation corpus's reader model is framed in the dissertation.

**Could kill the thesis at this stage:**
- If a properly designed length-matched generation corpus shows the apparatus cannot recover the pre-stated branch (contamination control + honest generation → the registered statistics don't discriminate). This would mean the entire measurement framework — the slope, the band-movers, the ICC — is structurally unable to distinguish instrument from collapse, and the thesis's central contribution (the discipline that made honest measurement possible) would be the method chapter only, not a claim about readers.
- If the S leg resolves to collapse in a future wave (the replication's positive slope excludes 0; if it also beats the roster competitor, the alignment arm is falsified per the registered rule). This kills the premise *with an explanation* — the idiosyncrasy was room geometry wearing a reader's name.
- Neither of these kills the thesis's method contribution (claim 6 in topic.md: "reproducibility makes cheap adversarial audits possible; cheap audits make solo reasoning survivable"). They kill the *empirical* claims about readers and rooms.

---

## 5. Research Frontier — Hard Questions for the Ideation Session

1. **How do you generate a reader baseline that is *independent* of the estimator's o/d decomposition?** The estimator measures idiosyncrasy as deviation from the room mean, and drift as split-half displacement. If the generation model sets "reader baseline = fixed vector" by simply offsetting from the room mean by that vector, the estimator trivially recovers it — the o_R numerator is the generation parameter itself. The hard version: generate at the *text* level (synthetic scripts producing synthetic field_eff readings via the engine), and check whether the estimator recovers the generation-model parameters *through* the engine's transformation. This is the contamination problem made concrete.

2. **What does a length-matched corpus buy you that the field corpus + re-registration doesn't?** The field corpus's void is an event-count problem (17 vs 20). A re-registered wave-3 with more reader-nights could clear that floor in an afternoon. The generation corpus's advantage is branch pre-statement — but is that advantage worth the design cost if the field path is viable? Under what conditions does the generation corpus become *necessary* rather than *nice-to-have*?

3. **How strong must an entry step be to produce a counted down-crossing, and can you design for it without rigging?** The coverage gap (D = 40–50%) is driven by mild entry steps (T5/T5c, D/D-cold at speak 24) not producing crossings. To push D above 50%, either the entry steps need to be harder (but the frozen scripts determine the step magnitude — you'd need new script families) or the readers attending those nights need smaller offsets (designable via the attendance matrix). Is designing readers-onto-nights to maximize crossings legitimate (it's just the attendance matrix, which is already a design parameter) or does it cross into rigging?

4. **What's the minimal generation corpus that would satisfy abstract sentence 6?** The advisor says "length-matched" — but matched to what? The full field corpus (18 nights × 2 waves = 36 night-logs)? One wave (9 nights)? One night per branch condition? The answer determines the generation burden and the contamination surface area.

5. **If the generation corpus shows the apparatus works (recovers branches correctly), does that validate the field corpus's VOID verdict or override it?** This is the external-validity question: a clean calibration on synthetic data doesn't retroactively make 17 events into 20. The generation corpus calibrates the *instrument*; the field corpus's void says the instrument wasn't given enough signal in the field. Can the thesis use the generation corpus to argue "the instrument is sound; the field just didn't cooperate" — or does the field void stand regardless of calibration?

---

*No files modified, no commits, no runs. Read-only research only.*