# JEPA Implementation Path for the Elephant
*Research subagent, 2026-08-21. Read-only synthesis.*

---

## 1. What JEPA Actually Is, For This Room-Field Instrument

LeCun's JEPA **predicts representations of the future from representations of the past, entirely in latent space**. The mapping onto the elephant is exact and non-metaphorical:

| JEPA Term | Elephant Object |
|---|---|
| **Encoder** | The existing field pipeline: 7 dial readings → `vmf_fit()` → latent state `(μ̂, κ, v*, coherence)`. **This is already built.** The encoder is frozen — it is the registered instrument, not trained via JEPA. |
| **Latent Space** | The tangent bundle of S⁶ at the current field position. This is not the mu-hat point itself — it is the **normal bundle of the tide line**: the direction perpendicular to the waterline that everyone in the room feels but cannot name. Explicitly: the latent vector is the edge displacement `d_mu / dt` (field velocity) plus residual κ-change. |
| **Context** | The state of the room before an event: `field_before`, with the event's interior masked out (exactly I-JEPA's context masking). |
| **Predictor** | The thing we do not yet have: a small feedforward network that maps `(field_before, event_mask)` → `predicted_field_after`. The predictor runs *entirely in latent tangent space*. |
| **Target** | The measured `field_after` state — *produced by the same frozen encoder*. No pixels. No tokens. No reconstruction. |
| **Loss** | The **field edge**: the chord distance / vMF symmetric divergence between predicted and measured `field_after`. Exactly the existing `edge()` function. |
| **Anti-Collapse Guard** | The encoder is *frozen*. This eliminates JEPA's canonical collapse problem entirely. There is no trivial constant solution because the target encoder is not trainable. Collapse immunity is bought by anchoring the latent space to the existing registered instrument. |

The v* / generalized eigenproblem machinery slots in as the **latent basis**: the predictor does not operate on arbitrary tangent vectors — it operates on the registered basis `(v*, v₂, v₃, ...)` where `v*` is the maximally-room-responsive, minimally-personality-loaded axis (REG-1 verdict). The q-rule is the loss mask: we only backpropagate loss for edges where `q = RMS(r_R) / RMS(o_pre)` passes threshold — we do not train on rigid common-shift steps that are personality artifacts.

---

## 2. Minimal Viable JEPA-v1 Specification

This is a concrete, buildable architecture scoped to existing corpora:

### Architecture Sketch
- **Input:** 10-minute sliding window of dial readings, with the central 2-minute interval masked out (context = before + after edges, target = interior)
- **Frozen Encoder:** Unmodified `vmf_fit()` + REG-1 rotation → 7-dimensional latent vector: `(v* projection, κ, coherence, 4 residual dial projections)`
- **Predictor:** 2-layer MLP (64 → 32 → 7) that maps the context latent state → predicted target latent state
- **Loss (3 terms):**
  1. **Primary:** L1 distance between predicted and measured `v*` projection (volume/presence contrast axis)
  2. **Secondary:** L1 distance between predicted and measured `κ` (choppiness / presence)
  3. **Mask:** Apply the q-rule mask — zero loss for edges where `q < 0.2` (common-shift artifacts)
- **Latent dimensionality:** 7, exactly the registered dial space. No learned compression.

### Scope
- **Buildable now:** This can be trained on the existing wave-1 and wave-2 corpora (9 nights × 15 readers). No additional data required. The only new code is the predictor network and the window/masking logic.
- **What needs wave-3 corpus:** Only the fine-grained generalization to event types (jokes, silences, stories). JEPA-v1 can be built and validated *before* wave-3 lands, using only the existing filed data.
- **Difference from current thermometer:** The thermometer is a *filter* that measures the current state. JEPA-v1 is a *predictor* that outputs what the next state *should* be, and measures the residual (the surprise) when reality differs.

---

## 3. Novelty Guard (Honest Citations + Contribution)

Per the novelty audit, **all underlying JEPA machinery is known**. The elephant's specific contributions are:

1. **Frozen domain-specific encoder as anti-collapse guard.** All existing JEPA implementations use an EMA target encoder that is trained alongside the predictor. This implementation uses a *frozen, externally validated instrument* as the encoder — eliminating the collapse problem entirely, and anchoring the latent space to a pre-existing measurement standard. This is a novel variant of the JEPA architecture.
2. **Room-field thermometer as JEPA substrate.** No prior JEPA implementation has been built for group conversational state, or used a directional-statistics (vMF) instrument as the latent anchor.
3. **The personality fiber as variance the predictor *must not* model.** The q-rule explicitly tells the predictor to ignore, and not predict, the common-shift personality variance. JEPA is normally trained to predict everything predictable — this one is trained to predict only the *room variance*, and actively ignore the personality variance. This is an inversion of the standard JEPA objective.

**Required citations:**
- LeCun (2022) for the JEPA architecture
- Assran et al. (2023) I-JEPA for masking design
- Mardia & Jupp (2000) for vMF directional statistics
- Bardes et al. (2022) VICReg for the anti-collapse analogy
- Fisher (1936) for the generalized eigenproblem / CSP axis selection

---

## 4. Demo Value

A working JEPA-v1 changes the fleet's story completely:

> The elephant stops being an aspirational metaphor. It becomes a working predictive model that, given the room's state right now, outputs what the room's state *will be* in two minutes. When the prediction is wrong, that residual is the surprise — the thing that just happened that no one expected. The "school's next move" is no longer poetry. It is a vector in latent space.

For the fleet:
- The deadband becomes the prediction error floor
- Premise bands become the contours where prediction error reliably drops to zero
- The ledger becomes a record of prediction residuals, not just measurements
- Every conversation can be scored by how much it surprised the predictor

This is the line between "a thermometer that measures the room" and "a model that understands the room".

---

## 5. Sequencing + Risk

### Slotting Post Wave-3
- **S4 (immediately after REG-1):** Implement JEPA-v1 on existing wave-1/2 corpora
- **S5 (after wave-3 lands):** Add event-type conditioning and train on the larger wave-3 corpus
- **S6:** Add latent variable z for multi-modal future prediction

### Biggest Risks
1. **Overclaiming:** Do not claim this is "a JEPA that understands conversation". It is a JEPA that predicts the *output of the elephant instrument*. That is enough.
2. **Small corpus overfitting:** The wave-1/2 corpus is only 9 nights. The predictor must be extremely small (2-layer MLP, 64 hidden units) and heavily regularized.
3. **Persistence baseline beating:** The null hypothesis is that predicting "the field will be exactly the same" (persistence) will outperform the JEPA. If this happens, the entire project is falsified.

### Falsification Design
The test is simple and pre-registered:
> Train JEPA-v1 on 7 nights, test on held-out 2 nights. Compute RMSE of predicted `v*` and `κ`. If JEPA-v1 does not beat the persistence baseline by >15% on both metrics, the JEPA hypothesis is falsified.

---

## Summary (5 Lines + One-Line Architecture)

**Doc path:** `memory/research-jepa-implementation-2026-08-21.md`

1. JEPA maps exactly onto the elephant: frozen vmf encoder = encoder, field_before = context, field edge = loss, normal bundle = latent space.
2. Minimal JEPA-v1 is a 2-layer MLP predictor trained on existing corpora, no new data required.
3. Novelty is frozen instrument anti-collapse, room-field substrate, and explicitly not predicting personality variance.
4. A working JEPA turns the elephant from a thermometer into a predictive model of the room's next state.
5. Sequenced for S4 after REG-1, falsifiable via persistence baseline test.

**One-line architecture:** *Frozen vMF field encoder → 2-layer MLP predictor → L1 loss on registered v*/κ axes, q-rule masked, no reconstruction.*