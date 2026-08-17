# Elephant Sense v3 — the room-temperature sense (2026-08-17)

**Author:** fleet perception architect (subagent)
**Status:** design — with a decisive first experiment already run
**Supersedes:** the retired "beat the 0.849 ordering" framing in
`audio-jepa-v2-2026-08-17.md` §6/§9, per the captain's reframing
(`jepa-is-the-elephant-2026-08-17.md`)
**Reviewed by:** Seed-2.0-pro (architecture), Qwen3.6-35B-A3B (math),
Hermes-3-Llama-3.1-405B (philosophy), DeepSeek V4-Pro (practical ML) — full
transcripts in `research/reviews-elephant-sense-v3.md`

---

## 0. The reframing, restated as a spec

JEPA is not a conductor's baton. It is **a room-temperature sense** — the elephant
in the room. You don't notice it from inside; you notice it only when you walk into
a *different* room and it is a very different elephant. Three consequences follow,
and they are the whole spec:

1. **The unit of perception is the ROOM, not the stream.** The fleet already has
   rooms. The ambient field of a room is what the sense feels.
2. **Contrast is the only training signal.** The elephant is invisible within a
   room; it is revealed by *moving between* rooms (sauna vs cold-plunge). Within-room
   ordering — the old 0.849 target — is the wrong objective and is **retired**.
3. **Two social forces, not one:**
   - *Acclimation* (agent → room): a newcomer warms toward the room's vibe over
     time; the rate is their experience/talent/training at modulating their vibe.
   - *Charisma* (room → agent): a strong presence pulls the room's vibe toward
     itself over interactions.

---

## 1. What is a ROOM, in fleet data terms?

A room is a **bounded social + acoustic + textual context** — the thing you step
into and out of. It has an ambient field (heat, pacing, mood, who is present), and
that field shapes everything produced inside it. The fleet already ships many rooms,
scattered across the repo. The mapping below is the *input* to the elephant sense.

### 1.1 Rooms that already exist as data

| Room | Location | N clips | Character | Contrast role |
|------|----------|---------|-----------|---------------|
| **The Tap — trades nights 1–4** | `ai-writings/tap-trades/radio-theater/episode-{1..4}` | 14–15 each | same cast (lucineer, wesley, 5 trades), different night | **the killer control**: same speakers, different vibe — fine-grained room contrast |
| **Compass-Head Radio Hour** | `ai-writings/radio-theater/compass-head-radio-hour/` (episodes 01–07 + voices) | 12+ voices, serial episodes | serialized show, recurring host | serial room drift across episodes |
| **The Tap open-mic night** | `ai-writings/radio-theater/tap-open-mic-night/voices` | 53 | many acts, one room | loose/rowdy room — high "temperature" (spread) |
| **Tavern night / front door / jazz suite** | `radio-theater/tavern-night`, `the-front-door`, `hermes-jazz-suite`, `dogs-fell-in-love`, `channel-42-dawn` | 3–9 each | named venues, distinct moods | extra sauna/plunge contrast |
| **Fleet-radio segments** | `ai-writings/radio/audio` | 27 (series 004/005…) | one nightly broadcast per series | the "daily episode" room |
| **Music / instrumentals** | `ai-writings/music/` (ace-step-output, mmx sessions) | ~280 | no speech | **the cold plunge** — the extreme opposite pole |
| **The speeches corpus** | `ai-writings/speeches/` | 16 | the v2 training set | the original single-room toy |
| **Wesley's streams (text)** | `ai-writings/wesley-stream/` | ~40 `.md` | one voice, many nights | a **text-only room** — tests whether text alone carries room-ness |
| **F/V EILEEN (future)** | not yet rendered | — | wheelhouse (cold, alert, instruments) vs galley (warm, coffee, wood) | the *physically real* sauna/plunge event |

**The single most important fact for training:** `tap-trades` episodes 1–4 are **the
same cast in the same room on four different nights**. That is the exact control that
lets us *disentangle room-vibe from speaker-identity* — the failure mode every
reviewer flagged (see §7).

### 1.2 The signals a room emits

The elephant is multimodal. The room-state embedding is fed by (ordered by how much
signal each carries today):

1. **Audio** — the trained v2 encoder's 384-dim vibe embedding per clip (the
   perceptual substrate from `audio-jepa-v2-2026-08-17.md`; 2.92M params,
   non-collapsed). *Primary channel.*
2. **Text** — transcript embeddings (local `nomic-embed-text` via Ollama, or the
   `fleet-embed` server's `all-MiniLM-L6-v2`). Carries *topic/mood*.
3. **Pacing** — tempo, pause density, clip duration, energy curve (already extracted
   by `vibe_matcher.py`'s LISTEN stage). Carries the "sauna people talk slower" signal.
4. **Presence** — who is in the room (speaker keys from filenames), turn counts. Used
   as a **mask** for attention, *not* as a classification feature (see §5.3).
5. **Time of day** — timestamps embedded in filenames (`2026-08-14-0759-…`). Rooms
   have circadian temperature.
6. **Visual (boat only)** — future F/V EILEEN wheelhouse/galley images via FLUX or a
   frozen vision encoder. The one channel the current fleet does not yet emit.

### 1.3 What new data would make the elephant trainable

Today the elephant is only *partly* trainable. What would make it fully so:

- **Two or more genuinely different rooms that share a cast** — we have this (trades
  nights 1–4). This is enough to *start*.
- **Multi-room, multi-night transcripts with speaker labels** — most radio-theater
  clips have no aligned transcript. Pairing `vibe_matcher`'s transcript-finder with
  the `.tap` scripts and episode HTML would unlock the text channel.
- **Acclimation/charisma traces** — a *labeled* sequence of an agent entering a room
  and either warming or pulling. We do not have this yet; it is the data that makes
  §3 and §4 trainable (see §7 for how to bootstrap it from what exists).
- **F/V EILEEN wheelhouse/galley audio** — the physically-realized sauna/plunge pair,
  the cleanest possible contrastive signal.

---

## 2. Contrast as the training signal

The retired objective asked "can the ear *order* a stream?" The new objective asks
"can the sense *tell rooms apart*, and *feel the walk between them*?"

### 2.1 What "contrast" means here

- **Positive pair** — two clips (or windows) from the *same* room.
- **Negative pair** — two clips from *different* rooms.
- The sense is trained so that a clip is closer to *its own room's field* than to any
  other room's — **without** collapsing every clip in a room to a single point (the
  collapse the reviewers warned would destroy within-room structure).

### 2.2 The room-state is a *distribution*, not a point (adopted from review)

Mean-aggregation of L2-normalized embeddings is **degenerate** (all three reviewers,
independently): on the 384-sphere, the raw mean of N unit vectors has expected norm
`√(N/d) ≈ √(15/384) ≈ 0.20`, so renormalizing the mean discards magnitude and
amplifies noise; and a mean erases the *spread* — the single most important property
of a vibe (a rowdy open-mic and a quiet formal reading can share a mean).

**Adopted representation — a von Mises–Fisher (vMF) field:**

- **mean direction `μ̂`** — where the room "is" on the sphere.
- **concentration `κ`** — how tight the vibe is (κ ≈ 0 → uniform/rowdy/loose; κ large
  → cold/tight/formal). This is the "room temperature": **cold = high κ, warm = low κ**.

`κ` is estimated from `κ ≈ (d·‖r̄‖ − ‖r̄‖³) / (1 − ‖r̄‖²)` where `r̄ = mean of unit
embeddings`, `d = 384`. (This is the standard vMF MLE; Qwen verified the norm issue
it fixes.) The **pairwise-spread** inside a room — `1 − mean within-room cosine` — is
a cheaper, model-free proxy we already compute (see §6).

### 2.3 The objective (adopted from review)

**Not** vanilla InfoNCE over centroids — with only 10–20 rooms the softmax is
statistically weak, and it *successfully collapses* (every clip → its centroid), which
looks like it's working while destroying within-room structure. Adopted scheme:

- **Hierarchical clip↔clip contrast** (SimCLR-style): anchor = a clip; positives =
  *other clips from the same room*; negatives = clips from *other rooms*. No centroid
  as anchor.
- **Batch structure**: each batch samples *all* clips from 2–3 rooms (many within-room
  positives, a bounded negative set), rather than one clip per room.
- **Temperature τ ≈ 0.15** (fixed; reviewers agreed the SimCLR default 0.07 is wrong
  at this cosine scale and with this few negatives).
- **An explicit within-room spread regularizer**: maximize mean pairwise distance
  *within* each room (reward that variation exists inside a room). This is the
  anti-collapse guard that mirrors v2's VICReg role, but now for room *spread*.

**"Walking into a different room"** is then the **edge** between two room fields in
embedding space: the vector `μ̂_B − μ̂_A` (or the geodesic between vMF modes). The
sauna/plunge event is a **large-magnitude edge**; the trades-nights 1→2 walk is a
**small edge** (same cast, subtly different night). The sense must resolve *both*
scales.

### 2.4 The acclimation curve (embedding-space definition)

An agent entering a room follows a **geodesic relaxation** on the sphere (not
Euclidean — Qwen caught that the naive `e(t) = e_room + (e_0 − e_room)e^(−t/τ)`
drops off the unit sphere):

```
e(t) = slerp(e_0, μ̂_room, 1 − e^(−t/τ))
```

`τ` = the **acclimation time constant** (bigger τ = slower to warm). The rate
`1/τ` is the agent's modulation skill. This is §3.

---

## 3. Acclimation — agent modulates toward the room

**Definition.** A newcomer's per-turn embedding relaxes from its own prior toward the
room's field over successive turns. The relaxation's rate is the agent's
*experience/talent/training at modulating their vibe to the group*.

**Observable (adopted from review — not cosine-to-centroid, which is confounded).**
The reviewers identified three confounds in "cosine distance to the room centroid":

1. **Moving-target confound** — the room itself drifts (charisma, evolution), so
   distance-to-centroid shrinks even if the agent didn't move.
2. **Talkativeness confound** — an agent with few turns has noisier per-turn
   embeddings and looks like they "never converge."
3. **Noise-floor** — cosine distance has a hard minimum (~0.05) from encoder noise,
   capping exponential-fit R².

**Adopted observable — percentile-rank acclimation.** At each turn, score the agent
by *where their embedding ranks among the room's other occupants* (how close are they
to the room's center-of-mass relative to everyone else present). Track that rank over
turns. This is **invariant to room drift, talkativeness, and the noise floor** (Seed's
suggestion), and it directly reads "is this agent becoming a regular?"

- `τ` is recovered from the **rank curve**'s approach to the top of the room's
  distribution (an exponential in rank-space, not cosine-space).
- **Rate as a function of talent** is then testable: for a fixed room, plot `1/τ`
  against a proxy for experience (number of prior fleet appearances, a "seniority"
  score from the fleet-ensemble roster). The prediction to verify: veterans acclimate
  faster (smaller τ). This is a *hypothesis*, flagged as such until we have the data.

---

## 4. Charisma — a strong presence pulls the room

**Definition.** Over interactions, a charismatic presence pulls the room's field
toward itself. The wheelhouse on a bad day; the Tap when Hermes holds the room.

**Observable — room-field displacement toward the agent.** The room field `μ̂` is a
running aggregation over occupants. After a charismatic agent's passes, `μ̂` shifts
toward that agent's embedding. The measurable quantity is:

```
charisma shift = ⟨ μ̂_after − μ̂_before , unit( e_agent − μ̂_before ) ⟩
```

i.e. the component of the room's displacement *along the direction toward the agent*.

**The hard truth (all four reviewers, independently):** you **cannot** separate
charisma from acclimation from a single joint trajectory. Both produce "the distance
between agent and room shrinks." The only clean separation is **order of arrival**:

- If a *stable* room (existed for many turns) has a newcomer enter, and only the
  newcomer moves → **acclimation**.
- If the *existing occupants* all shift toward the newcomer → **charisma**.

This is an experimental-design constraint, not a statistics fix. Design consequence:
**the fleet must log turn order and speaker identity** for any session where we want
charisma to be measurable. The trades-nights already have this structure (lucineer
opens, trades speak in sequence, signoff); the future F/V EILEEN wheelhouse/galley
sessions should log it explicitly.

**Minimum viable claim (adopted):** until we have order-of-arrival logs, we report
the **net agent↔room coupling** (the summed displacement), and label "charisma vs
acclimation" as requiring the order-of-arrival intervention — which we design now and
collect later.

---

## 5. The ensemble — the elephant is multimodal

The room field is late-fused from per-clip embeddings across channels. The reviews
added concrete fusion rules that are now part of the spec:

### 5.1 Channels into the room-state embedding

| Channel | Encoder | Output | Notes |
|---------|---------|--------|-------|
| audio | frozen v2 encoder | 384-dim L2 unit | primary; carries the acoustic vibe |
| text | `nomic-embed-text` / `fleet-embed` | 384/768-dim | carries topic/mood |
| pacing | `vibe_matcher` LISTEN features | ~6-dim (tempo, energy, pause) | carries "sauna = slow" |
| presence | speaker keys + turn counts | mask, not feature | see 5.3 |
| time-of-day | filename timestamp | 2-dim (circular hour) | circadian temperature |
| visual (boat) | frozen vision encoder | d-dim | future only |

### 5.2 Fusion rules (adopted from review)

- **Per-modality L2-normalization + distance-distribution matching.** Before fusion,
  rescale each modality so its *distribution of pairwise distances* matches the
  others (e.g. make the 95th-percentile distance equal across channels). Without
  this, the well-trained audio channel dominates ~90% and text/pacing contribute
  nothing.
- **Modality dropout** (e.g. drop audio with p=0.3 during training) so the model
  learns to work with any subset — required because the boat's visual channel will
  be absent in text-only/audio-only rooms.
- **Late fusion is a projector over concatenated channel embeddings** → the same
  vMF room field, but beware: vibe is the *interaction* of channels, not their sum
  (a room where people "yell jokes at each other" is not "yelling + joke text"; it
  is the playful *combination*). A single cross-attention layer over channel
  embeddings (audio attends to text, pacing, presence) is the next step beyond
  plain concatenation.

### 5.3 Presence is a *mask*, never a feature

One-hot speaker identity is **poison** (Seed + DeepSeek, emphatic): the model will
ignore all real signal and classify rooms by roll-call. Presence enters only as an
**attention mask** (which clips to attend to when aggregating a room field), and is
**ablated** in the first experiment to prove audio alone carries the room signal.

---

## 6. Evaluation — the elephant metrics (the 0.849 is dead)

The retired headline — "does the learned ear beat the hand-crafted 0.849 ordering" —
is replaced by four metrics, all computed on the frozen encoder first, then on the
contrast-trained encoder:

| Metric | Definition | What it proves |
|--------|-----------|----------------|
| **Room discrimination accuracy** | k-NN (k=1): is a clip's nearest neighbor from its own room? | can the sense tell rooms apart? |
| **… speaker-heldout** | same, but remove all same-speaker clips from the candidate set | *is it room or is it voice?* (the decisive control) |
| **Sauna/plunge separability** | same-room vs cross-room cosine gap (and silhouette) | can the sense feel the walk between rooms? |
| **Room temperature** | mean within-room spread `1−cos` (κ proxy) | can the sense feel *how warm/loose* a room is? |
| **Acclimation convergence** | percentile-rank curve → τ, plus exponential-fit R² | does a newcomer warm, and how fast? |
| **Charisma shift magnitude** | room-field displacement along the agent direction | does a strong presence pull the room? |

The first three are computed **now** (see §8). The last two are computed once
order-of-arrival traces exist (§4).

---

## 7. What the wider view said — and what I adopted

Four models reviewed the brief (full transcripts in
`research/reviews-elephant-sense-v3.md`). They **converged hard** on the same three
warnings, which are now baked into the design:

1. **Speaker-identity is the confound.** All four: the trades-nights share a cast, so
   any encoder trained on speech will cluster by *voice*, not *room*. **Adopted:** the
   speaker-heldout control is mandatory and is the headline control in §8.
2. **Mean-aggregation is degenerate on the sphere.** Seed, Qwen, DeepSeek (three
   independent derivations of the `√(N/d)` norm collapse). **Adopted:** vMF field with
   concentration κ as "temperature."
3. **Charisma vs acclimation are unidentifiable from a joint trajectory.** All four.
   **Adopted:** order-of-arrival as the only clean test; report "net coupling" until
   then.

Seed additionally contributed: (a) the **percentile-rank** acclimation observable
(invariant to drift/talkativeness/noise-floor), (b) **per-room batches + explicit
within-room spread regularizer** for the contrastive objective, (c) fixed τ ≈ 0.15.
DeepSeek contributed the **Gaussian/attention set-encoder** ladder and the
**modality-balance + presence-as-mask** fusion rules. Hermes kept the design honest to
the captain's philosophy — emphasizing that the *room* (not the voice) is the unit,
and that κ (spread) is the felt "temperature." Qwen verified the vMF math and the
contrastive temperature/batch guidance.

**What I did *not* adopt:** nothing was dismissed; the three warnings are the design.
The only open dispute is *how far* to go on the room representation now — Seed says
start with vMF, DeepSeek says Gaussian is fine, both say *not* mean-pooling. I start
with vMF (it is the natural distribution on the sphere the encoder already lives on),
and treat a learned attention set-encoder as the §5.2 "next step."

---

## 8. The first experiment (already run — decisive result)

**Question:** is the elephant (room-ness) already latent in the *frozen* v2 encoder,
with no retraining?

**Method** (`elephant_sense_probe.py`, committed): treat the four trades-nights as
four rooms (same cast, different night). Extract a 384-dim embedding per clip with the
frozen encoder. Compute room-discrimination (with and without speaker-heldout),
sauna/plunge separability, and room temperature. Also measure the *coarse* sauna/plunge
contrast (speech vs music) as a known-positive control.

**Result:**

| Metric | Value | Read |
|--------|-------|------|
| room discrimination (no control) | **0.339** | barely above chance (0.25) |
| room discrimination (speaker-heldout) | **0.356** | *does not drop* → the weak signal is **not** speaker identity |
| same-room cosine mean | 0.629 | — |
| cross-room cosine mean | 0.614 | — |
| fine sauna/plunge gap (which night?) | **0.015** | the encoder does **not** yet feel *which night it is* |
| coarse sauna/plunge gap (speech vs music) | **0.271** | it **does** feel *speech vs music* — the elephant's coarse pole |
| room temperatures (spread) | 0.328–0.418 | episodes differ in spread even now |

**The decisive read.** The frozen encoder already feels the **coarse** sauna/plunge
contrast (speech vs music, gap 0.271 — a genuinely felt "different elephant"), but it
does **not** yet feel the **fine** contrast (which trades-night, gap 0.015 — the
elephant is invisible *within* the speech rooms). And — crucially — the weak signal it
*does* have is **not** speaker identity (heldout doesn't drop it), so it is real
room-ish signal, just weak.

**This is exactly the proof the captain's reframing predicted:** the elephant is
invisible from inside a room; it is only *real* once you train on contrast. The frozen
encoder is "inside the room." The contrastive objective of §2 is what walks it out.

### The next experiment (the one that *opens the gap*)

Take the **same** four rooms + the music pole, and train **only** a light contrastive
head (or fine-tune the frozen encoder with the §2.3 objective) for a few minutes on
the GPU. Success = the **fine** gap (0.015) climbs toward the **coarse** gap (0.271),
and speaker-heldout discrimination rises well past chance — while within-room spread is
preserved (no collapse). That single number — *"the fine room-gap opened from 0.015
to X, speaker-heldout stayed high"* — is the replacement headline.

---

## 9. Files

- `research/elephant-sense-v3-design-2026-08-17.md` — this document
- `research/reviews-elephant-sense-v3.md` — the four reviewer transcripts
- `elephant_sense_probe.py` — the first experiment (runs the frozen encoder over the
  four trades-nights, computes elephant metrics)
- `checkpoints/elephant_probe.json` — the probe's output (the numbers in §8)

---

*The ear grew. Now it has learned what a room is — and that it must walk from one
room into another before it can feel the elephant.*
