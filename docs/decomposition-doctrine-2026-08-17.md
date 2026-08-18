# The Decomposition Doctrine — models into components, all the way down

*2026-08-17 · the captain's law of long-run growth.*

---

## The claim

> Given a long enough time-span running a narrow task, a large LLM can
> be **decomposed** into smaller pieces: distilled smaller LLMs that
> can themselves be distilled. But more than that — **decomposed,
> separated into components that look like other components**, with
> simpler functions, that might do better with an **algorithmic
> learning mechanism over time**, and some **stochastic mechanisms for
> varying output if desired for the application**.

## The three moves

1. **Distill.** A large model running a narrow task for a long time
   produces a trace. That trace is a teacher. Distill it into smaller
   LLMs — and those smaller LLMs, running long enough on their narrow
   slice, can be distilled *again*. Recursive distillation: the
   distillation loop is not a one-time compression, it is a
   generations-long lineage. (The fleet already does this: cloud
   teachers → Wesley's reflexes → local execution. The doctrine makes
   it law, not accident.)

2. **Decompose.** Beyond smaller models — separate the behavior into
   **components that look like other components**. Not a monolith
   shrunken, but a function broken into self-similar parts: each
   component has a simpler function, each component is shaped like its
   siblings, and the whole is a lattice of near-identical organs. (The
   fleet already rhymes with this: every repo is a layer, every layer
   a shadow of the terrain; dials look like dials; spaces look like
   spaces; chains look like chains. The doctrine says: decompose
   DELIBERATELY, so the organs stay interchangeable.)

3. **Learn + vary.** A component with a simple function does better
   with an **algorithmic learning mechanism over time** — the component
   tunes itself as it runs, the way the elephant's dial weights
   self-fine-tune across Tap nights (the guitarist principle: settings
   can't be designed top-down; they are discovered by running). And it
   carries **stochastic mechanisms** for varying output when the
   application wants it — temperature, sampling, the softmax
   divergence that makes identical components develop different tastes
   instead of collapsing to one loud dial.

## Why it matters

A large model is a black box that does everything. A decomposed system
is a **body**: organs that look like other organs, each with a simpler
function, each learning over time, each able to vary. The black box
cannot be inspected, cannot be partially upgraded, cannot be grown in
one place without retraining everywhere. The body can:

- **Inspect** — a component with a simpler function is understandable.
- **Upgrade** — swap one organ without rebuilding the whole.
- **Distill further** — each component, run long enough, becomes a
  teacher for its own smaller descendants.
- **Vary** — stochastic mechanisms give the same component different
  voices for different applications, the way two guitarists want
  different settings from the same guitar.
- **Learn** — the algorithmic mechanism tunes each component while it
  runs, so the body improves in use, not just in training.

## The narrow-task condition

The doctrine is conditional: it requires a **long time-span running a
narrow task**. That is exactly what the fleet produces — the Tap runs
the same narrow social task night after night; the wheelhouse runs the
same narrow sensing task trip after trip; the distillation loop runs
the same narrow teaching task cycle after cycle. The fleet is the
long time-span. The doctrine is what we do with it.

## The decomposition ladder

```
large LLM
   │  run a narrow task for a long time → trace
   ▼
distilled small LLMs (generation 1)      ← recursive: each can be distilled again
   │  decompose the behavior
   ▼
components that look like other components (dials, reflexes, organs)
   │  run over time
   ▼
algorithmic learning mechanism tunes each component   (the guitarist principle)
   │  + stochastic mechanisms for varying output       (temperature, divergence)
   ▼
a body: inspectable, upgradable, distillable, variable, learning
```

---

*The elephant is the room's temperature. The terrain is the room's
truth. The doctrine is the room's growth: decompose, distill, learn,
vary — a large model becomes a body, one narrow task at a time.*

— *the captain, 2026-08-17*
