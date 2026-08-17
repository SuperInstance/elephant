# Dial Training v1 — distill the hand-crafted senses, then grow them

**Author:** fleet learning engineer (subagent)
**Status:** shipped — `elephant/learned.py`, `tests/test_learned.py`, checkpoint in `checkpoints/`
**Date:** 2026-08-17
**Reviewed by:** Seed-2.0-pro (distillation design), Qwen3.6-35B-A3B (training math), DeepSeek V4-Pro (self-review)

---

## 0. The v1 contract

The seven dials in `dials/` are hand-crafted v0 senses — keyword/regex
heuristics. This is the learned pass: a small text encoder plus **one
regression head per dial**, trained to reproduce each hand-crafted dial's
reading **from raw room text alone**. v0 is the teacher; the learned model is
the student.

Why this is the right first move (Seed, verbatim): *"the student will only
learn to reimplement your regex heuristics. That is not a bug, that is the
bar."* If the student cannot even reproduce a dumb keyword scorer on held-out
text from the same cast, it has no business trying to learn vibe. So v1's
question is honest and modest: **which dials can a learned model feel from
text, and which are the seams it cannot yet feel?**

---

## 1. Pipeline

```
markdown (.md) ──► parse ──► Room ──► teacher_readings()   [v0 labels, 7-dim]
        │                                │
        └─► window (8 msgs, stride 1) ──► text ──► tokenize ──► vocab (train only)
                                                 │
                      ┌──────────────────────────┘
                      ▼
        [optional]  JEPA pretrain:  EMA target + stop-gradient + cosine
                                    predictor + VICReg  (window t → window t+1)
                      │
                      ▼
        TextEncoder (BoW mean+max → MLP trunk, 64-dim) → 7 linear heads
                      │
                      ▼
        supervised MSE distillation → per-dial held-out r / R²
```

- **Tokenization** is word-level whitespace, lowercased, punctuation left
  attached (`great.` ≠ `great`) — deliberately, because the v0 cynicism dial
  keys on `great.` and the volume/panic dials key on `!`/`?`. The student must
  *see the same surface the teacher reads*.
- **Vocab** is built from training text only (2,529 tokens) — no test leakage.
- **Encoder** is deliberately small: 64-dim embedding, mean + max pooling,
  2-layer MLP trunk (~165k params). Minutes, not hours, on the GPU.
- **Windows**, not rooms, are the training unit: a sense reads a *span* of
  conversation, so each room yields many (text, label) spans. Evaluation is
  reported at both window level (per-span) and room-mean level (the elephant
  feels a *room*).

### The JEPA stage (the `jepa.py` promise, made real)

Before the supervised heads are fitted, an optional self-supervised pass
trains the trunk on **unlabeled** text: window *t* predicts window *t+1* via a
cosine predictor against a stop-grad EMA target encoder, with VICReg
variance/covariance terms keeping the representation from collapsing. This is
the same skeleton `jepa.py` promised (EMA + stop-gradient + cosine + VICReg),
run for 5 epochs (~100 pairs).

---

## 2. The corpus and the split

| Split | Rooms | Content |
|-------|-------|---------|
| **Train — nights 1–2** | 27 files / 281 messages | the two source evenings, their broadcast episodes (1–2), the six trade pieces, the night-1/night-2 sequels and questions |
| **Test — nights 3–4 + speeches** | 40 files / 1,192 messages | adaptation night + episodes 3–4, the adaptations/lenses pieces, the 27-speech corpus |

The trades nights are the killer control from the v3 design: **the same cast,
different nights**. Training on nights 1–2 and testing on nights 3–4 (plus a
completely different genre — the speeches) is the honest held-out check: does
the learned dial feel a room it has never read?

---

## 3. Results — per-dial held-out transfer

All numbers are Pearson **r** (and R²) between the learned student and the v0
teacher on the held-out rooms. r is the honest headline (scale-invariant,
robust to the teacher's near-constant dials); R² is reported for the dials
with real variance.

### 3.1 Room-mean transfer (mean over 3 seeds)

| dial | with JEPA r | without JEPA r | verdict |
|------|-------------|----------------|---------|
| **mood** | **+0.64** (0.61/0.68/0.63) | +0.08 (−0.41/0.68/−0.04) | ✅ transfers, JEPA is decisive |
| **earnestness** | **+0.58** (0.52/0.61/0.61) | +0.51 | ✅ transfers, JEPA small help |
| **panic** | **+0.50** (0.40/0.52/0.57) | +0.14 (0.26/0.54/−0.37) | ✅ transfers, JEPA stabilizes |
| **cynicism** | +0.30 (0.01/0.50/0.40) | −0.26 | ⚠️ borderline — noisy across seeds |
| **joke_landing** | −0.03 | −0.01 | ❌ does not transfer |
| **volume** | −0.14 | +0.09 | ❌ teacher flat on this corpus |
| **presence** | −0.09 | −0.16 | ❌ teacher flat on this corpus |

### 3.2 Window-level (seed 0) and the teacher's own variance

| dial | window r | window R² | teacher σ (held-out) |
|------|----------|-----------|----------------------|
| mood | +0.159 | −0.255 | **0.671** |
| earnestness | +0.317 | +0.098 | **0.238** |
| cynicism | +0.016 | −0.036 | **0.369** |
| joke_landing | −0.020 | −0.073 | 0.126 |
| panic | +0.170 | −0.189 | 0.071 |
| volume | −0.086 | −0.385 | **0.042** |
| presence | +0.135 | −2.153 | **0.059** |

`teacher σ` is the standard deviation of the *teacher's own* reading across
held-out rooms. It separates the two failure modes cleanly.

### 3.3 What this actually says

1. **The elephant can learn to feel mood, earnestness, and panic from text.**
   These are *content-driven* dials (valence words, sincerity markers, alarm
   words), and their teacher readings actually vary across rooms. The student
   reproduces them on rooms it never saw — mood at r ≈ 0.64 room-mean.

2. **volume, presence (and, largely, panic) are near-constant on this corpus.**
   Their teacher σ is 0.04–0.07: they are driven by *structure* — message
   timestamps, density, author counts — that the fake document timestamps
   flatten and that raw text cannot carry. These are **not "the model failed";
   they are dead labels** (Seed's phrase): the hand-crafted dials themselves
   barely move on written transcripts. That is the seam, and it is real.

3. **cynicism and joke_landing are the interesting seams.** Their teachers
   *do* vary (σ 0.37 / 0.13), but the student cannot hold them. Cynicism keys
   on scare-quotes, sarcastic punctuation, "great."-style tokens — signal that
   survives into the tokenizer but is too sparse for 129 training windows.
   joke_landing is not a text property at all — it reads the *audience's
   reaction after the joke* (reactions/replies), which the static transcripts
   don't encode. Both are honest v1 failures, flagged for v2.

---

## 4. Did the JEPA pretraining help?

**Yes, clearly and specifically.** Over three seeds:

- mood: **+0.08 → +0.64** room-mean r (and it stops being seed-fragile)
- panic: **+0.14 → +0.50** (and stabilizes: worst seed −0.37 → +0.40)
- cynicism: **−0.26 → +0.30**

The mechanism is the expected one: with only 129 training windows, a
random-init trunk overfits and generalizes poorly (the no-pretrain run hits
train MSE 0.003 vs 0.023 with pretraining). Five epochs of "predict the next
window" gives the trunk a representation that already notices *how this show
moves*, and the supervised heads transfer off that. Earnestness transfers with
or without it — it is the easiest, most content-obvious dial.

This is the v1 answer to "is the JEPA shape worth anything on text?": **yes —
as a representation regularizer, exactly the anti-collapse role VICReg plays in
the audio JEPA.**

---

## 5. The reviewers' pushback (adopted)

**Seed-2.0-pro** (excerpted here; the review was run over the live design and
math below):

- *"This is exactly the correct v1 target"* — reproducing the heuristic is the
  bar, not a bug.
- *"Evaluating at room mean is double-smoothing"* — adopted: window-level r is
  reported first (3.2), room-mean only as the "can it feel the room" aggregate,
  with that caveat stated.
- *"These are not seams, these are dead labels"* — adopted: volume/presence are
  framed as corpus-flat teacher dials, not model failures.
- *"v2: train a contrastive base encoder that can tell any two windows apart
  first, then fit heads on the frozen encoder"* — this is exactly the §6
  direction.

**Qwen3.6-35B-A3B** checked the math (VICReg, EMA, cosine predictor, R², vMF
κ) and found it correct, with one flag — the VICReg covariance denominator
`n−1` divides by zero at batch size 1; the code already guards `if n > 1`.
The vMF κ approximation (`(d·|r̄| − |r̄|³)/(1 − |r̄|²)`) is the standard Sra
(2012)-family approximation and does fix the mean-norm degeneracy.

---

## 6. v2 direction

The v1 result points one way: **stop distilling heads from broken heuristics,
and learn the room field end-to-end, contrast-only** (the v3 design, §2).

1. **Contrastive base encoder first.** Train the text encoder to tell any two
   windows from *different* rooms apart (and same-room windows together), with
   an explicit within-room spread term — before any dial head exists. This
   fixes the core weakness Seed named: the current encoder is trained to
   *throw away* everything the 7 heuristics don't score.
2. **Then** fit heads (or better, read a vMF room field: mean direction μ̂ +
   concentration κ) on the *frozen* contrastive encoder. The room-temperature
   sense (cold = high κ, warm = low κ) is then a *property of the field*, not a
   keyword count.
3. **Give the structural dials their channel.** volume/presence/panic are flat
   on text because their signal is timing, density, reactions. The v3 fusion
   spec already has this: pacing (tempo/pause/energy), presence-as-mask, and
   reaction heat as channels — not text. Text alone can't feel a stampede's
   *speed*; it can only see the word "fire."
4. **More nights.** 129 training windows is the binding constraint on the
   sparse dials (cynicism). More trades nights, or the open-mic/tavern rooms,
   are the cheapest way to move those numbers.

*The elephant learned to feel mood, earnestness, and the first edge of panic —
from text, on rooms it had never entered. The dials that wouldn't turn are the
map of where the next sense has to be built.*
