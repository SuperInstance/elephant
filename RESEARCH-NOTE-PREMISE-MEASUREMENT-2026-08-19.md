# RESEARCH NOTE — Premise Measurement (Antecedent Test), 2026-08-19

**Status:** field measurement, real corpus, read-only. Script: `scripts/premise_measurement.py` (numpy-only, CPU, writes nothing). Data: `data/nights/{night-A,night-B,night-C,night-D,night-D-cold}.jsonl` (206 speak events total). `coarse-anchor` (different room/roster) and `night-A-repro` (byte-replay of A) excluded.

## The premise under test

The Nurse JEPA doctrine's second-order reading — the doctor reads the nurse's *change from her baseline* — is only real if readers have **idiosyncratic, stable** baselines. The synthetic reader-delta test assumed this; both the devil's pass-4 review and the rival's review demanded it be measured in the field. The devil's kill band: **if the field's baseline-spread-to-drift ratio lands below ~0.3–0.6 (corpus-sd units), the doctrine dies by its own registration.**

## 1. Per-reader displacement model

The logs carry only ROOM-level `field_eff` (charisma-displaced once for the whole room, `elephant/tapnight.py:239-257`). Each reader R's personal reading is computed by replaying the session dynamics from the log alone:

```
s_R(t)       = 1 - exp(-charisma_R * n_R(t))        # n_R from logged interactions_after
eff_R(t)     = clamp(raw(t) + s_R(t) * (vibe_R(t) - raw(t)))
               # single-agent case of the engine's displacement — tapnight.py:239-241:
               # "for a single agent this reduces exactly to charisma_pull(raw, vibe, charisma, n)"
g_R          = dial_weights_R / max(dial_weights_R)  # per-dial attention gain:
               # the reader reads the room through their own dials;
               # unattended dims attenuate toward neutral
reading_R(t) = DIAL_CENTER + g_R * (eff_R(t) - DIAL_CENTER)
vibe_R(t+1)  = vibe_R(t) + (1 - exp(-rate_R)) * (field_eff_logged(t) - vibe_R(t))
               # acclimation replay, the engine's exact update (tapnight.py:254-257),
               # driven by the LOGGED room field_eff
```

Presence = roster membership (a rostered reader reads the room all night; before their first interaction s_R = 0, so only their attention shapes the reading). The drifter is harvested from night D only: in D-cold his lines run but he is not in the roster (lazy neutral registration — no persona, hence "cold").

## 2. Corpus scale

Per-dial std of `field_raw` over all 206 speak events: mood 0.1592, volume 0.0067, earnestness 0.0856, cynicism 0.4897, joke_landing 0.2353, panic 0.0374, presence 0.1958. **corpus_sd (RMS over dials) = 0.2292.**

## 3. Real readers harvested

**7 real readers** — captain, critic, drifter, engineer, essayist, poet, writer. The six occupants appear in all 5 nights (212 readings each); the drifter in D only (46 readings).

## 4. Measurements (REAL)

Per-reader baselines (fit from each reader's own readings only):

| reader | |baseline| | direction (mood, vol, earn, cyn, joke, panic, pres) |
|---|---|---|
| captain | 0.9696 | [+.421 +.031 +.699 +.018 +.016 +.001 +.577] |
| critic | 0.8998 | [+.240 +.019 +.715 +.286 +.101 +.002 +.581] |
| drifter | 0.9378 | [+.384 +.007 +.647 +.222 +.135 +.001 +.606] |
| engineer | 1.0753 | [+.227 +.028 +.820 +.096 +.063 +.002 +.512] |
| essayist | 1.1090 | [+.379 +.017 +.780 +.070 +.025 +.002 +.492] |
| poet | 1.1521 | [+.674 +.090 +.531 +.000 +.048 +.000 +.503] |
| writer | 1.1088 | [+.675 +.011 +.562 +.000 +.092 +.000 +.469] |

The baselines are genuinely idiosyncratic in direction: poet/writer lean mood, engineer/essayist lean earnestness, critic/drifter carry the only substantial cynicism components, captain leans presence.

**(b) Baseline spread across readers: 0.1060 = 0.4627 corpus-sd.**

**(c) Drift per reader (corpus-sd, mean over strata transitions):** captain 0.3515, poet 0.3622, writer 0.4709, essayist 0.7296, drifter 0.9320, engineer 1.0038, critic 1.9348. Mean drift (reader-mean) = **0.8264 corpus-sd**. Per-transition means: A SEG1→SEG2 0.8195, B 0.8195, C 0.8195 (byte-identical schedules → identical values, zero independent variance), D pre/post-entry 0.7379, D-cold pre/post-entry 0.8799.

## 5. THE KILL NUMBER

| measure | ratio | verdict |
|---|---|---|
| **REAL ONLY (N=7)** | 0.4627 / 0.8264 = **0.5599** | **in band: indeterminate** |
| robustness variant (drift measured vs own baseline) | 1.1185 | above band |
| **REAL + SYNTHETIC (N=20)** | 0.4179 / 0.8533 = **0.4898** | **in band: indeterminate** |

One-sentence verdicts:

- **real-only: ratio = 0.5599 → in band: indeterminate.**
- **real+synthetic: ratio = 0.4898 → in band: indeterminate.**

The primary estimator (stratum-to-stratum displacement) lands inside the 0.3–0.6 kill band on both reader sets — the premise is **not killed, but not cleared** by this corpus. The vs-own-baseline variant (each stratum's mean against the reader's global baseline) doubles the ratio (1.1185 real-only) because with two roughly balanced strata the stratum means sit ~half a stratum-gap from the baseline; both estimators are reported so the committee can pick its registration.

## 6. Honest N caveat

The real roster has **7 readers (< 10 required)**, and the five nights instantiate only **2 distinct schedules** (the SEG warm→cynical shift; the newcomer entry) — A/B/C byte-share one schedule and D/D-cold share another, so the ≥5-independent-strata-transitions discipline is **not met** by the real corpus. The real-only kill number therefore rests on 7 readers × effectively 2 conditions.

Minimal bootstrap (proposed by the review, run here, clearly labeled): **13 synthetic-grounded readers** (`synth-00`…`synth-12`, seed 0) sampled from the real roster's *observed* parameter distribution — charisma/acclimation_rate = archetype value + N(0, observed sd) truncated to the observed range; dial_weights = archetype weights × lognormal(0, 0.15) renormalized; vibe_start = archetype vibe + N(0, pooled per-dial sd) clipped to dial bounds — each inheriting a real archetype's interaction timeline and night coverage. These are **not** new field data; they only thicken the reader-population sample against the same rooms. Reported separately above, never pooled into the real-only number.

## 7. What would settle it

The band verdict is a data-limitation verdict, not a doctrine verdict. To move the ratio out of the band: (i) more distinct real readers (≥10) with genuinely different dial_weights/vibes — spread is driven mostly by the cynicism and mood dims, where only critic/drifter and poet/writer separate; (ii) more independent schedules — A/B/C contribute one condition three times; (iii) readers observed across ≥5 real strata transitions each, so drift is estimated within-reader rather than from two strata.

## Reproduction

```
python3 scripts/premise_measurement.py
```

No trained files, checkpoints, or logs were modified; the script is read-only against `data/nights/` and imports only `DIAL_NAMES`/`DIAL_BOUNDS`/`DIAL_CENTER` constants from the elephant package.
