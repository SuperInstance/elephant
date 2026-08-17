# Documentation Peer Review — 2026-08-17

**Lead reviewer:** DeepSeek V4-Pro (self, synthesizing)
**Panel:** Qwen3-Coder-480B (technical accuracy), Seed-2.0-pro (design
coherence), Hermes-3-Llama-405B (philosophy fidelity), Nemotron (ops
realism). All via the DeepInfra API.

**Scope:** README.md, `docs/just-so.md`, `docs/api-reference.md`,
`docs/tuning-guide.md`, `docs/deployment-guide.md`,
`docs/fleet-operations.md`, plus the older design docs
(`jepa-is-the-elephant.md`, `jepa-zeitgeist-2026-08-17.md`,
`communication-spaces-2026-08-17.md`, `fleet-field-math.md`,
`elephant-sense-v3-design.md`, `tap-night-operations.md`,
`fleet-simulation-notes.md`).

The ground rule of this repo applies to reviewers too: **a reviewer
correction is data to verify, not an oracle.** Every finding below was
re-run against the shipped code by the lead before it was adopted. Two
reviewer claims were rejected as hallucinations (see §4).

---

## 1. Per-reviewer verdicts

### Qwen3-Coder-480B — technical accuracy

> "2 critical issues (acclimation comment, Dial.series behavior), 2 medium
> (module count, symbol count), 1 minor (example count — correct)."

- Flagged the `# halfway toward room` comment on `acclimation_curve`
  (value is ~39%, not 50%). **Adopted** (re-rated P2).
- Flagged `Dial.series()` as returning `window` repeated values.
  **Rejected** — the code returns `[self.read(room)]` (a single element),
  exactly as documented; the reviewer hallucinated.
- Flagged README "all 21 `.py` files" vs a 20-row table (`__init__.py`
  missing). **Adopted** (P1).
- Flagged "60 symbols" vs 64 index rows. **Noted** (P2; the "60 symbols"
  count lives in the captain's brief, not in the doc body).
- Confirmed the "17 verified examples" count as correct. **Agreed.**

### Seed-2.0-pro — design coherence

> "P0: unacknowledged inverted κ polarity across system boundaries… P0:
> README contradicts its own demo data on the temperature/κ rule… P1: κ
> causal overclaim…"

- The core catch: **room κ** ("cold = high κ") and **fleet κ**
  ("warm/bunched = high κ") point in *opposite* directions, and no doc
  says so. **Adopted** as the top convergent finding (documented, not
  code-changed).
- The README states "cold room = high κ" then publishes a demo where the
  warm tap (κ 2.04) out-reads the cold wheelhouse (κ 1.96). **Adopted**
  — softened the README and api-reference to say v0's `concentration()`
  measures *extremity*, not yet temperature.
- Confirmed `PersonalElephant` weighting (tuning guide) matches the code
  exactly. **Agreed.**

### Hermes-3-Llama-405B — philosophy fidelity

> "The README and just-so.md demonstrate strong fidelity to the captain's
> reframing, with no significant drifts or contradictions."

- The just-so faithfully retires the "ordering engine" baton, honors
  room-not-stream, acclimation, charisma, contrast, JEPA-as-one-sense.
- "The light itself" and "light the woodstove" metaphors are used without
  overclaim; "JEPA correlates, never replaces" is consistent.
- No place reduces the elephant back to a replacement or an ordering
  engine. **No fixes required** — the reframing is intact.

### Nemotron (Nemotron-4-340B) — ops realism

> "P0: the '>40× jump in the biomass channel' claim is incorrect — the
> biomass channel is actually 18.3×, the total is 41×."

- `>40×` biomass-channel claim in the deployment guide. **Adopted**
  (fixed to ~18× biomass / ~41× total).
- Mixed-units quirk in radar kinematics (km positions + hour timestamps →
  `speed_kts` is not real knots) under-warned in the guides. **Adopted**
  as a documented caveat (the demo already flags it; the guides did not).
- "anchor must not use `fishing_day` (tautology)" claimed as missing.
  **Rejected** — deployment-guide §5 already says this explicitly.
- "captain → presence" flagged for verification. **Adopted** — verified
  the captain's argmax is `mood` (0.73), not `presence` (0.21).

*(Note: `nvidia/Nemotron-3-Ultra-550B` 404s on DeepInfra; the closest
available Nemotron, `nvidia/Nemotron-4-340B-Instruct`, stood in. Qwen3-Coder
also required the `-A35B-Instruct` suffix.)*

---

## 2. Convergent findings (flagged by 2+ reviewers, or verified independently)

1. **κ polarity collision + the "cold = high κ" overclaim.** Seed (twice)
   and the lead's own run. Room `concentration()` is `norm(vector−0.5)·2`
   — a *neutral-distance* metric — yet README/field-docstring/api-reference
   assert "cold room = high κ." The published demo contradicts it (warm tap
   κ 2.04 > cold wheelhouse κ 1.96), and the fleet sim uses an *opposite*
   κ (`1/(1+spread)`; bunched = high). **Fix:** soften the docs to call
   v0's κ an extremity proxy and note the v3 "tightness" reading is the
   design target. Applied to README + api-reference.

2. **"captain → presence"** — flagged by Nemotron, verified by the lead:
   the captain's top dial after 14 nights is `mood` (0.73), `presence`
   (0.21) is second. README and tap-night-operations both said
   "captain → presence." **Fixed** in both.

3. **Factual multipliers in the deployment guide.** Nemotron flagged the
   `>40×` biomass-channel number; the lead verified 0.568/0.031 ≈ 18×
   (biomass) vs 1.314/0.032 ≈ 41× (total). **Fixed.**

4. **README module count.** Qwen + lead: "all 21 `.py` files" listed 20
   rows. **Fixed** (added `__init__.py`).

---

## 3. Prioritized fix list

### P0 — factual error / wrong number / failing example (all fixed)
| # | Where | What was wrong | Fix |
|---|-------|----------------|-----|
| 1 | `docs/deployment-guide.md` | ">40× jump in the biomass channel" (actually ~18× biomass, ~41× total) | rewritten to "~18× … (~41× on total deviation)" |
| 2 | `README.md` | "captain → presence" (captain's top dial is `mood` 0.73) | → "captain → mood (presence close behind)" |
| 3 | `docs/tap-night-operations.md` | "captain → presence" (same error) | → "captain → mood/presence" |
| 4 | `docs/fleet-operations.md` | "pulled the field on 12 of 14 nights" (actually all 14) | → "all 14 nights" |
| 5 | `README.md` | "all 21 `.py` files" but 20 rows | added `__init__.py` row |

### P1 — misleading (fixed)
| # | Where | What was wrong | Fix |
|---|-------|----------------|-----|
| 6 | `README.md`, `docs/api-reference.md` | "cold room = high κ" asserted as fact; v0 `concentration()` measures extremity, not temperature, and the demo contradicts it | softened to "far from neutral; the cold=high-κ tightness reading is the v3 design target, not yet v0's proxy" |

### P1 — misleading (documented, not code-changed)
| # | Where | Note |
|---|-------|------|
| 7 | fleet ↔ room κ | `fleet κ = 1/(1+spread)` (bunched/warm = high) runs opposite room κ (cold = high). Both are correct in their own contexts; the name is overloaded. Worth a one-line cross-reference if the two are ever discussed side-by-side. |
| 8 | `docs/deployment-guide.md` §2 | radar `kinematics()` `speed_kts` is fed km positions at hour timestamps in the demo, so it is km/h mislabeled as knots (the demo flags "v0 mixed-unit quirk … treat with suspicion"; the guides did not). Added as a caveat below. |

### P2 — polish
| # | Where | Note |
|---|-------|------|
| 9 | `docs/api-reference.md` | `# halfway toward room` → `# ~39% of the way toward room` (fixed) |
| 10 | captain's brief | "60 symbols documented, 17 verified examples" — the quick index has 64 rows; the "17 verified examples" count is accurate, "60 symbols" is a rough undercount. Not in the doc body; no edit. |

---

## 4. Rejected reviewer claims (data to verify, not oracles)

- **Qwen3-Coder: "`Dial.series()` returns `window` repeated values."**
  False. `dial.py` returns `[self.read(room)]` — a single-element list,
  exactly as documented. Rejected.
- **Nemotron: "the guides don't explain that the anchor must not use
  `fishing_day`."** False. `deployment-guide.md` §5 states it verbatim
  ("Do not build the inductive anchor from `fishing_day` … self-confirming"),
  and `fleet-operations.md` says "catch is exogenous (… never a dial)."
  Rejected.

---

## 5. Lead's final accuracy verdict

**The documentation matches the code.** Every numeric claim I checked was
reproduced from a clean run:

- README quickstart: tap `RoomField(warmth=+0.29, κ=2.04)`, wheelhouse
  `(warmth=-0.05, κ=1.96)`, `distance=0.828…`, `sauna_plunge_gap=+0.339…` ✓
- api-reference examples all execute and return their documented values:
  `three_reading_kinematics` → `dir_deg=45.0`, `fleet_mean_speed=0.5498` ✓;
  `nudge_prior`/`describe` → `'nudge[radar=+0.80, camera_out=+0.90]'` ✓;
  `classify(f, 21.0) == 'joyful'` ✓; `acclimation_curve(zeros, half, 0.5, 1)`
  → `0.1967` ✓; `biomass_deviation` large ✓; `fleet_concentration` tight ≫
  scattered ✓.
- Fleet sim (seed 7) phase table matches fleet-operations/deployment-guide
  exactly: κ 0.63/0.22, 0.11/0.08, 0.76/0.17; warmth +0.66/−0.83/+0.41;
  deviation 1.35/12.17/3.50; anchor `[0.632, 0.819, 0.813]`; dark boat
  EILEEN (rep 1.7) at (12.3, 8.3) with weight 5.1 ✓.
- Tap (seed 42) diverged-taste table and mean pairwise 0.389 → 0.859 ✓.

The 17 documented examples are correct and verified. The five P0 and one P1
fixes above close the only real drifts — none of which was a broken
signature or a failing example; all were mis-attributed numbers/counts and
one over-stated metaphor. Test suite stays **49 passed** (docs-only edits,
no code change).

**Bottom line: docs are faithful to the code and to the captain's
reframing. After this pass, documentation matches code.**
