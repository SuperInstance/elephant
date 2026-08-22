# κ(t)-check RE-RUN on the G6-reworked generator — 2026-08-21

**Read-only.** Generated corpus in `/tmp/kappa-rerun` (never `data/`), analysis
in `/tmp/kappa_rerun_check.py` + `/tmp/kappa_final.py`. No commits, no repo
writes. Grounding: the original κ-check
(`memory/kappa-t-check-2026-08-21.md`), the G6 research
(`memory/research-g6-noise-2026-08-21.md`), and the G6 run note's flagged
pre-S3 item (`docs/riverbed-G6-run-2026-08-21.md` — "flip Δlogκ −0.435
under-responds, re-run the κ-check protocol at S2").

**Verdict (4 lines):**

1. **The flagged open item does NOT reproduce.** Committed G6 flip Δlogκ = **−0.775** (T1 −0.855, T3 −0.842, T8 −0.626; T2@8 has no pre-fit as in the field), essentially **at** the field's −0.746 — slightly *over*-responding, not under. The flip is a κ-loosening event at field magnitude. ✅ **OPEN ITEM RESOLVED.**
2. **Entry Δlogκ is the new marginal: −0.417 pooled [−0.499, −0.330]**, sitting **on the band's lower edge −0.418** (band [−0.418, −0.205]). 2/4 events in-band (T4b −0.283, T5 −0.388), 2/4 below/too-loose (T4a −0.526, T5c −0.472). The run note's claimed −0.302 does not reproduce; the committed generator loosens ~0.1 logκ *more* than the field's −0.320.
3. **Logged κ levels 59.2 / 27.8 (warm/cold)** vs field 24/11 — **×~2.5 offset** (the run note's ×1.9 understates it), ratio 2.13 ≈ field 2.18 (ratio preserved).
4. **Qualitative κ-check verdict is REPRODUCED**: warm tight / cold loose; entry and flip both *loosen* κ; flip loosens **more** than entry (−0.775 vs −0.417, same ordering as the field's −0.746 vs −0.320). The DIRECTION-EVENT structure holds in the committed generator.

**RECOMMENDATION: S3-GO as-is.** The flip under-response that blocked the gate
is gone; the two remaining deviations (entry Δlogκ marginally at band edge,
κ levels ×~2.5) are branch-invariant absolute residuals that do not touch the
branch-relative predictions (α-sweep, 2AFC pairs) the registration protects.

---

## 1. What I did

Generated the full 9-family instrument wave at the registered seed 20260821
(`python3 scripts/riverbed_generator.py --branch instrument --seed 20260821
--outdir /tmp/kappa-rerun --tag-prefix kappacheck`; 9 nights, determinism
re-run identical). Then re-ran the **exact** protocol from the original field
check (verbatim logic copied from `/tmp/kappa_check.py`):

- Δlogκ = log κ(e+7) / κ(e−1), where κ(t) = the per-speak **cumulative**
  `fit.kappa` (vmf_fit over `obs_fit[:t+1]`, the engine's trailing-W=8
  smoothed cumulative fit) — identical to the field check's instrument.
- Event sets restricted to the families that exist in the generated corpus
  (no wave-1 S/D/A replay families): ENTRY = T4a@12, T4b@28, T5@24, T5c@24;
  FLIP = T1@20, T2@8, T3@20, T8@20 (pure-flip families only, matching the
  field check, which never used entry families for its flip pool).

## 2. Numbers (seed 20260821, generated)

### Δlogκ per event (window [e−1, e+7])

| class | event | generated Δlogκ | field Δlogκ | band/target |
|---|---|---|---|---|
| ENTRY | T4a@12 | **−0.526** | −0.461 | [−0.418, −0.205] ⚠️ below |
| ENTRY | T4b@28 | −0.283 | −0.068 | ✅ in band |
| ENTRY | T5@24 | −0.388 | −0.374 | ✅ in band |
| ENTRY | T5c@24 | −0.472 | −0.374 | ⚠️ below |
| FLIP | T1@20 | **−0.855** | −0.746 | −0.746 |
| FLIP | T3@20 | −0.842 | −0.746 | −0.746 |
| FLIP | T8@20 | −0.626 | −0.746 | −0.746 |
| FLIP | T2@8 | — (no pre-fit) | — (no pre-fit) | — |

### Pooled

| metric | generated | field | verdict |
|---|---|---|---|
| entry Δlogκ | **−0.417 [−0.499, −0.330]** (n=4) | −0.320 [−0.418, −0.205] | at band edge, slightly too loose |
| flip Δlogκ | **−0.775** (n=3 valid) | −0.746 | on target (slightly over) |
| logged κ warm/cold | **59.2 / 27.8** (ratio 2.13) | ~24 / ~11 (ratio 2.18) | ×~2.5 level offset, ratio preserved |

(κ levels: per-stratum means over flip families' warm strata `hi<flip` and
cold strata `lo≥flip`, the `g6_calib.measure` definition. A looser inclusive
definition — all warm-era speaks incl. no-flip T9 — gives 52.8/23.9; the
conclusion is the same.)

## 3. The discrepancy with the run note / commit message

Commit `1bdeaab`'s message and the run note claim **entry −0.302 IN band** and
**flip −0.435 under-responding**. Neither reproduces on the committed code:
the committed generator's entry Δlogκ is −0.417 (at/below the band edge) and
its flip Δlogκ is −0.775 (on target). The claimed −0.435/−0.302 appear to be
stale numbers carried forward from the calibration sweep (e.g. the
`/tmp/g6era.py` monkeypatch used `baseline·grand` and no `NOISE_ERA_EXP`
era-scaling; the committed code uses fixed `BASELINE_Z` + era-scaled noise),
not a measurement of the landed constants. The flip number was the flagged
blocker; re-measuring it on the landed code clears the block.

## 4. Does the residual matter for wave-3?

- **Flip:** resolved — no residual. The flip response (−0.775) is at field
  magnitude; no flip-tune is needed.
- **Entry Δlogκ −0.417:** a ~0.1 logκ overshoot vs the field's −0.320,
  concentrated in **T4b@28** (generated −0.283 vs field −0.068). Cause is
  structural: `KAPPA_ENTRY_FACTOR = 0.28` is applied **unconditionally** via
  pointwise `min`, so an entry that lands *after a flip* (κ already loose)
  loosens κ again — the field's entry is a secondary μ-event side-effect that
  saturates when κ is already loose. It is **branch-invariant** (branch params
  live only in persona anchors), so it does not change branch discrimination;
  it only shifts the within-night κ trajectory uniformly and the absolute κ
  level (already a disclosed residual).
- **κ levels ×~2.5:** disclosed absolute-level residual; the ratio (2.13 ≈
  2.18) — what the log-ratio Δlogκ targets actually read — is preserved.

None of the three invert the registration's branch-relative predictions
(H-GEN within-corpus α-sweep, 2AFC pairs, noise-branch ICC collapse).

## 5. Recommendation

**S3-GO as-is.** The flip under-response (the one open pre-S3 item) does not
reproduce on the committed generator — flips respond at field magnitude. Ship
S3 with the disclosed residuals **updated** to the honest re-measurement: (i)
entry Δlogκ −0.417 (marginally at the −0.418 band edge, not the claimed
−0.302); (ii) logged κ levels ×~2.5 (not ×1.9). Optional non-blocking
one-parameter follow-up, only if the entry band-edge overshoot matters: give
`KAPPA_ENTRY_FACTOR` a saturation rule — apply the entry loosening as a step
**toward a floor** (or scale it by the current κ) instead of an unconditional
`min`, so a post-flip entry stops re-loosening an already-loose κ (kills the
T4b@28 −0.283→~−0.07 deviation). Branch-invariant; not an S3 precondition.

*Analysis: `/tmp/kappa_rerun_check.py`, `/tmp/kappa_final.py`, corpus
`/tmp/kappa-rerun/`. No repo file written, no commit, no `data/` change.*
