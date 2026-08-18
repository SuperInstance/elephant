# The Decomposition Harness — the doctrine's first implementation

*2026-08-17 · `elephant/decompose.py` · the captain's law, made code.*

The doctrine (decomposition-doctrine-2026-08-17.md) says: a large model
running a narrow task for a long time can be **decomposed** into
components that look like other components, each with a simpler function,
each tuned over time by an **algorithmic learning mechanism**, each able
to **vary** through a stochastic mechanism when the application wants it.
The harness is that sentence in numpy: `ingest` → `distill` → `learn` →
`respond` → `specialization`. The three moves, in code.

---

## The API — the three moves

```python
from elephant.decompose import DecompositionHarness

h = DecompositionHarness(seed=7, temperature=1.0)
h.ingest(trace)            # DISTILL source: (input, output) pairs from the big model
h.distill(k=4)             # DECOMPOSE: k-means on simple features -> 4 components
h.learn(warm_reward, epochs=8)   # LEARN: the guitarist principle, over time
h.respond("Hey there, how's your night going?")   # VARY: temperature 0 = argmax,
                                                  #       > 0 = softmax sampling
h.specialization()         # the body after running: score, door, style, divergence
```

- **`ingest(trace)`** — the trace is the teacher: the long record of the
  large model doing ONE narrow task (a log of responses, a transcript of
  a Tap night, a run of a teacher). Stored as `(input, output)` pairs.
- **`distill(k=4)`** — decompose the trace into `k` components that look
  like other components. k-means (k-means++ seeded, multi-restart,
  lowest objective kept) on **simple features**: input length, output
  length, input→output token overlap, output entropy. Each cluster
  becomes a `Component {id, prototype, learning_rate, temperature,
  hits, correct, score}` with a centroid prototype and a simple function
  (nearest-centroid routing + a small per-component lookup with a mode
  fallback). Two spaces, on purpose:

  - **cluster space** (the prototype) — the simple pair features the
    organs are found by; learning moves the prototype here;
  - **door space** (the door) — each organ also stands behind the
    centroid of its members' *inputs* (length + punctuation energy);
    routing and the small lookup live here. The door is what makes
    ``respond`` route by what the input *is*, not by what the answer
    was.
- **`respond(input, temperature=0.0)`** — route to the nearest
  component (in door space); that component answers. `temperature 0`
  is argmax — deterministic. `temperature > 0` softmax-samples among
  the nearest prototypes — the stochastic mechanism for varying output
  when the application wants it.
- **`learn(reward_fn, epochs, temperature)`** — the algorithmic learning
  mechanism over time. Replay the trace; each component answers; the
  reward_fn scores the answer; and the winning component learns on
  BOTH vectors: its prototype moves **toward** the rewarded (teacher)
  output and **away** from the punished one, and its door moves toward
  the inputs it wins on and **away** from the inputs it loses on (an
  organ vacates the regions where it fails). Both updates are bounded —
  a punished organ fades to the edge of the body, it never explodes.
  Settings are discovered by running, the way dial weights self-fine-
  tune across Tap nights.
- **`specialization()`** — report the body after running: each organ's
  score, what it got good at (door, style, representative answer), and
  two divergence numbers — `divergence` (std of accuracy across
  organs) and `prototype_spread` (mean pairwise distance between the
  organs' style vectors). Two numbers because one can lie alone: a
  body that is all-terrible and a body that is one-winner-plus-faded
  both need a shape, and the shape is in both.

---

## How it maps to the fleet

The harness is not an island; it is the fleet's existing rhythms made
explicit. Every piece of it already existed somewhere in the fleet —
the harness is what they look like when the doctrine makes them law.

| Doctrine piece | The fleet already had | The harness does |
|---|---|---|
| **The long time-span** | `pulse.py`'s `PulseLoop` — constant pulses, the agent sensing even in silence | the trace: a long record of one narrow task |
| **Distill (trace → smaller)** | the compaction/acclimation teachers → Wesley's reflexes; cloud teachers → local execution | `ingest` + `distill`: the trace becomes k components |
| **Components that look like other components** | `dials/` — a dial is a component; a bank of dials is a body of near-identical organs, same shape, one job each | `Component` — same fields, same simple function, interchangeable organs |
| **Algorithmic learning over time** | `tapnight.py` — dial weights self-fine-tune across nights (the guitarist principle: settings are discovered by running) | `learn(reward_fn)` — replay, score, move toward reward / away from punishment |
| **Stochastic mechanisms for varying output** | the Tap's softmax temperature, so tastes diverge instead of collapsing to one loud dial | `respond(temperature)` — softmax sampling over nearest prototypes |

### The distillation loop is the trace source

The fleet's teachers (compaction_teacher, acclimation_teacher) already
do move 1 in prose: read the long record, extract what survives, write
it where it lasts. The harness makes the same move mechanical: any
teacher's run — any long log of a large model on a narrow task — is a
trace, and `ingest` + `distill` turns it into organs. And the doctrine's
recursive clause holds: each component, run long enough on its narrow
slice, produces a trace of its own, which can be distilled again — a
generations-long lineage, not a one-time compression.

### The dial bank is the body

`elephant/dials/` is the fleet's existing body of organs: mood, volume,
earnestness, cynicism, joke_landing, panic, presence — each a JEPA
perceiving ONE dimension, each shaped like its siblings (the `Dial`
interface: `read(room) -> float`), each swappable without rebuilding the
bank. A harness body is the same lattice for behavior: k components,
identical shape, simpler functions, interchangeable. The `Component`
shape — `{id, prototype, learning_rate, temperature, hits, correct,
score}` — is the `Dial` interface of the decomposition world.

### The Tap night is the learning mechanism

`tapnight.py` is the algorithmic-learning reference: participants carry
`dial_weights` that self-fine-tune across cycles toward the dials where
their *felt engagement* was highest — anchored peer-relative and
ReLU-normalized so tastes **diverge into multiple stable attractors
instead of collapsing to the room's loudest dial**. The harness's
`learn` is the same principle on components: each component keeps its
own running record (`hits`, `correct`, `score`) and its own prototype,
updated by what actually happened when it answered. Settings are not
designed top-down; they are discovered by running — for the guitarist,
for the dial, for the organ.

### The temperature is the stochastic knob

The Tap's softmax temperature keeps identical participants from
collapsing into one taste. The harness's `respond(temperature)` is the
same divergence in miniature: at 0, argmax — the nearest organ always
answers, deterministic; above 0, softmax sampling over the nearest
prototypes — identical organs develop different voices, and a faded
organ can still occasionally speak when the application wants variety.
The demo shows the knob doing its job: after learning prefers warm
responses, temperature 1.0 pours warm 18 times out of 20 — and the
jokey organ still gets its two.

---

## The demo — doctrine in action

`examples/demo_decompose.py` is the long time-span demo: 200 synthetic
nights of one narrow task (bar greetings) with four latent styles
(warm, clipped, jokey, formal). It shows the full arc:

1. **BEFORE** — distill finds four pure organs (50/50/50/50), all
   shaped alike, all unrun: score 0, hits 0. A body of interchangeable
   organs.
2. **LEARN** — eight epochs with a reward that prefers warm responses;
   the epoch mean-reward curve climbs 0.38 → 0.86 as the body warms.
3. **AFTER** — the warm organ won the bar: 1364 hits, accuracy 1.00,
   its door now covers every greeting. The others faded to the edge of
   the body (doors pinned at the box edge, scores ~0) — still shaped
   alike, still able to speak, no longer winning the room. Divergence
   0.43.
4. **VARY** — the stochastic knob as a dial, not a switch: at
   temperature 0.0 the body pours one voice (deterministic); at 1.0
   the winner pours almost always; at 3.0 the body opens up — the same
   greeting comes back warm, jokey, and formal across pours. The body
   varies when the application wants it.
5. **THE BODY** — four organs, each with a simpler function, all shaped
   alike: inspectable, upgradable, distillable, variable, learning.

## The v1 direction — real distilled models per component

The v0 component responds from a prototype: a centroid plus a small
lookup table (nearest stored input, mode fallback). That is the
doctrine's shape without its substance — an organ that has a *record*
of the teacher but not a *mind* of its own. The v1 direction is to
replace the prototype with a **real distilled model per component**:

- each component holds a small distilled LM (or reflex) trained on its
  cluster's slice of the trace — the nearest-centroid routing stays, but
  the answer is generated, not looked up;
- `learn` then tunes the distilled model's own weights (or the dials
  around it) instead of a feature centroid — same guitarist principle,
  richer instrument;
- and each component, run long enough, becomes a teacher for its own
  descendants — the recursive distillation the doctrine promises.

Two further v1 moves the design review flagged, both about the body's
health: **retirement** (an organ below a hit-rate floor after an epoch
is killed and re-seeded from the trace — v0 lets it fade and stay able
 to speak, which is the doctrine's variety; v1 lets the body also grow)
 and **richer doors** (route on more than length + punctuation — the
 v0 door is deliberately simple so the organs stay findable).

The harness's job was to make the doctrine's skeleton visible and
testable: distill, decompose, learn, vary. v1 is what grows on that
skeleton — a body of real small minds, one narrow task at a time.

---

*The black box is gone. The body remains. Decompose, distill, learn,
vary — a large model becomes a body, one narrow task at a time.*

— *the decomposition engineer, 2026-08-17*
