# Elephant — Tuning Guide

How to tune the elephant without breaking it. The elephant is the room's
**field** — the ensemble of seven dial readings plus the dynamics that shape
them (acclimation, charisma, self-tuning, nudge, tint). Every knob below is a
real surface in the code, with the file and symbol you touch.

The through-line: **you cannot design the settings top-down.** The settings are
*discovered* by running cycles — different agents desiring different settings
and self-fine-tuning to the moment they're in. Your job as the engineer is to
set the *conditions* (lexicons, ranges, priors, rates), not to hand-pick the
outcomes.

---

## 1. The dials' lexicons — adding words

The seven v0 dials are **keyword-matchers**. Their entire behavior is the
module-level word/phrase sets they scan. This is the cheapest, most direct
tuning surface, and the one that most changes what the elephant feels.

| Dial | File | Lexicons (add words here) | Effect |
|---|---|---|---|
| mood | `elephant/dials/mood.py` | `POSITIVE`, `NEGATIVE` | warm/cold valence |
| earnestness | `elephant/dials/earnestness.py` | `SINCERE`, `HEDGE` | sincere vs ironic |
| cynicism | `elephant/dials/cynicism.py` | `CYNICAL`, `EYEROLL` | sneer detection |
| joke_landing | `elephant/dials/joke_landing.py` | `JOKE_MARKERS`, `LAUGH`, `BOO` | did the joke land |
| panic | `elephant/dials/panic.py` | `ALARM`, `URGENCY` | stampede sense |
| volume | `elephant/dials/volume.py` | (no lexicon — regex + density) | loudness |
| presence | `elephant/dials/presence.py` | (no lexicon — occupancy) | pheromone trace |

Mechanics to respect when you edit a lexicon:

- **mood** matches `set(words) ∩ POSITIVE` (and `NEGATIVE`) per message — so
  only single-word tokens count, and duplicates within one message count once.
  Output: `(pos−neg)/max(total,1) · 2`, clamped `[-1, 1]`.
- **earnestness** matches `set(words) ∩ SINCERE` vs substring scan of `HEDGE`
  phrases; output `sincere/(sincere+hedge)`, `0.5` when empty.
- **cynicism** counts `CYNICAL` hits **plus** scare-quote pairs (`"..."`) plus
  `EYEROLL` emoji, divided by total words and scaled `×40` (≈2.5% cynical
  tokens saturates to `1.0`). To make the dial *more sensitive*, lower the `40`;
  *less*, raise it.
- **joke_landing** requires a message to contain a `JOKE_MARKERS` token, then
  reads the **next 4 messages** as the audience (`LAUGH` vs `BOO`), plus
  laugh/boo reactions on the joke. To catch more jokes, add markers; to catch
  the audience's mood better, extend `LAUGH`/`BOO`.
- **panic** weights `0.40·alarm + 0.25·urgency + 0.20·ripple + 0.15·density`.
  `URGENCY` is divided by `5.0` — adding many urgency words without tuning that
  divisor will mute their contribution.

**Saturation caveat.** These dials are naive and *saturate*: a room of earnest
writers reads `earnestness ≈ 1.0`; `"great."` reads *warm* on the mood dial
(no sarcasm detection). v1 will train dials; until then, don't fight saturation
at the lexicon level — the self-tuning signal (§7) is **peer-relative**
precisely to survive it.

---

## 2. Dial weights — which dimensions matter

Two places use a 7-vector `dial_weights` (each normalized to sum to 1):

- **`PersonalElephant`** (`elephant/presets.py`) — an agent's *taste*. In
  `read`, the objective field's **deviation from neutral** is multiplied by
  `dial_weights · 7`:

  ```python
  subj = center + (dial_weights * 7) * (objective - center) + bias
  ```

  A weight **above 1/7 amplifies** that dial's deviation; **below 1/7 damps**
  it; a weight **≈ 0 reads that dial as neutral** (not zero). A uniform taste
  (`1/7` each) returns the objective field **un-deformed**.

- **`Participant`** (`elephant/tapnight.py`) — the same prior, but it
  **self-tunes** across evenings via `tune_participant` (§7). Set *different*
  priors per regular to seed divergence (the different guitarists), then let the
  cycles do the rest:

  ```python
  from elephant.tapnight import Participant
  writer = Participant("writer", dial_weights={"mood": 0.4, "joke_landing": 0.3},
                       vibe={"mood": 0.7, "joke_landing": 0.5})
  critic = Participant("critic", dial_weights={"cynicism": 0.4},
                       vibe={"cynicism": 0.7})
  ```

**Pitfall — zero-weight ≠ zero.** A dial an agent ignores must read as
*neutral* (`DIAL_CENTER`: mood/volume/cynicism/panic rest at `0.0`,
earnestness/presence at `0.5`), never as raw `0`. If you weight the raw value
instead of the deviation, an agent who ignores presence will "see" an empty
room (`presence=0`) rather than simply not registering presence. Both
`PersonalElephant.read` and `Participant.__post_init__` already do this
correctly — keep it that way if you touch them.

---

## 3. `acclimation_rate` — the modulation skill

`acclimation_rate` is **1/τ**, the rate at which an agent's live embedding
relaxes toward the room field. It appears in:

- `field.acclimation_curve(agent, room, rate, t)` → `room + (agent−room)·e^(−rate·t)`
- `field.acclimation_rate_from(agent_start, agent_obs, room, t)` → inverts it
- `Participant.acclimation_rate` (default `0.25`)
- `TapNightSession.speak`, which applies one discrete step:
  `vibe += (field − vibe)·(1 − e^(−acclimation_rate))`

Tuning:
- **High** (e.g. `0.35+`) = a newcomer who "warms to the room" fast — few
  interactions to blend in.
- **Low** (e.g. `0.05`) = someone who holds their own temperature against the
  room's — the sauna stays a sauna even when they walk in.
- The default `0.25` is a middling "gets comfortable over a few exchanges."

You can **measure** an agent's skill from a trajectory with
`acclimation_rate_from(start, observed, room, t)` — the projection of the
remaining gap onto the initial gap gives `e^(−rate·t)`, clamped so an agent
that has *overshot* the room yields a large finite rate, not `inf` (which would
poison downstream averages).

---

## 4. Charisma — the room warms to them

`charisma` is the bend-per-interaction by which a strong presence pulls the
room's field toward their vibe:

- `field.charisma_pull(room, agent, charisma, interactions)` →
  `room + (agent−room)·(1 − e^(−charisma·interactions))`
- `Participant.charisma` (default `0.15`)
- In `TapNightSession.speak`, the aggregate of all participants' pulls is divided
  by `max(1, total)` so a *sum* of many pulls is bounded to one step, and the
  result is clamped to `DIAL_BOUNDS`.

Tuning: higher `charisma` = the room visibly bends toward that agent by the end
of an evening (visible as `room_field() − raw_field()`). **Keep it modest** —
see the pitfall in §8.2.

---

## 5. `PersonalElephant` — bias and attachments

- **`bias`** (`dict` or 7-vector, default `0`) — a *constant* disposition offset
  the agent brings to every room before anyone speaks. A warm writer leans
  `{"mood": +0.1}`; a sneering critic leans `{"cynicism": +0.2}`. It is added
  *after* the taste weighting, then clamped to `DIAL_BOUNDS`.
- **`attachments`** — the intangible correlations: `attach(event_key, memory)`
  binds a memory to a key; `remember(event_key)` recalls it (or `None`). These
  are **not dials** — they are the subjective glue that makes one agent's room
  *feel* different from another's at the same objective reading. Use them for
  the perfume-that-is-grandma's-shop correlations; don't route them through the
  field.

```python
from elephant.presets import PersonalElephant
pe = PersonalElephant("casey", dial_weights={"mood": 0.3, "cynicism": 0.25},
                      bias={"mood": 0.1})
pe.attach("lover_album", "the song we found it to")
pe.remember("lover_album")          # "the song we found it to"
```

**Rule:** the objective field always comes from the `RoomElephant` (the
`PersonalElephant` delegates to it via `.objective()`). Never let the subjective
read drift the objective one — the two must stay comparable, because their
*comparison* (where they agree, where they diverge) is the observable of
relationship.

---

## 6. Nudge strength — the elephant nudges, it doesn't drive

The nudge turns dial readings into an attention prior the vision/correlation
model consumes. Two knobs:

- **`NUDGE_MAP`** (`elephant/nudge.py`) — which dial feeds which modality, and
  its sign. E.g. `"panic": ("camera_out", +1.0)` = alarm → look *out*, now.
  Change the **sign** to invert a dial's steering, or the **weight** to
  down-rank a modality (e.g. `"presence": ("camera_deck", +0.3)` is already a
  soft opinion).
- **`apply_nudge(attention, prior, strength=0.15)`** — the blend strength:
  `attention · (1 + strength·prior)`. Default `0.15` is deliberately small.

The design invariant (from the fleet build history): **the nudge loop must be
damped.** Raw κ into the nudge is an unbounded positive-feedback loop — a
feedback gain > 1 is a death spiral, not a vibe. Keep `strength` well below `1`,
and prefer a low-pass filter on nudge values if you close a loop at fleet scale.
The elephant correlates; it never replaces the vision model.

---

## 7. Tint template banks — `elephant/mud.py`

`tint_description` mutates a room description by the objective field. Its words
come from four template banks, each a `dict[classify-mode → list of strings]`
(`WEATHER`, `LIGHT`) or a flat `list` (`JOY_ADJ`, `CLOSE_DETAIL`):

- **`WEATHER` / `LIGHT`** — keyed by `panic` / `joyful` / `closing` / `neutral`.
- **`JOY_ADJ`** — joyful adjectives woven in when the room is laughing.
- **`CLOSE_DETAIL`** — how people react when the light changes.

Extend a bank by **appending strings** — the picker is
`bank[rng.integers(0, len(bank))]`, so order and length are both free. The seed
is a rolling hash of the field's 7 dials, so the **same field always tints the
same way**, and a one-dial change changes the words. To force reproducibility in
tests, pass `seed=`.

The thresholds are module constants at the top of `mud.py`:

```python
PANIC_HI = 0.5
JOY_JOKE_HI = 0.35; JOY_MOOD_HI = 0.1; JOY_PRESENCE_HI = 0.4
CLOSE_HOUR_LATE = 23.0; CLOSE_HOUR_EARLY = 3.0
CLOSE_WARMTH_LO = 0.0; CLOSE_VOLUME_LO = 0.4
```

Precedence is fixed and deliberate — **panic overrides everything; joy comes
before closing time.** A warm, laughing room at 11pm is still the warm bar, not
"closing." Tune the thresholds, not the precedence.

Remember the metaphor the whole module is built on: **the description is not a
report; it is the room acting on everyone in it.** The words are the light.
Same field → same words; changed field → changed room.

---

## 8. TapNightSession self-tuning — peer-relative signal, learning rate, cycles

The self-tuning loop is where the settings get *discovered*. Per evening:

```python
for evening in range(1, N + 1):
    session.start_session()
    for author, text, reactions in load_evening(evening):
        session.speak(author, text, reactions=reactions)
    field = session.room_field()
    for name in session.participants:
        session.tune_participant(name)      # learning_rate=0.15 default
    print(session.end_session())
    save(session.settings())
```

Three things to understand:

### 8.1 The signal is peer-relative, not absolute

`felt_engagement(name)` returns, per dial:

```python
delta = p.vibe - cast_mean_vibe                     # peer-relative
rxn   = per-dial reaction heat from REACTION_TO_DIAL  # crowd's hands
return delta * (1.0 + rxn)
```

The agent feels engaged on the dials they care about **more than the rest of
the table** — a warm writer leans mood, a sneering critic leans cynicism. This
is what makes tastes **diverge** (different guitarists) instead of collapsing
to the room's loudest dial, and it is robust to the v0 dials' tendency to
saturate.

### 8.2 The target is ReLU-normalized

`tune_participant` builds `target = (pos + ε)/(total + ε·7)` where `pos =
relu(engagement)` (the small `ε = 1e-3` keeps a trace of exploration on every
dial), then blends:

```python
dial_weights = (1 - learning_rate)·dial_weights + learning_rate·target
```

Weight only ever moves **toward** dials the participant is genuinely distinctive
on — so tastes settle into multiple stable attractors rather than one loud dial.

### 8.3 Learning rate and cycles

`learning_rate=0.15` is an exponential-moving-average step. **"Many cycles"
means dozens of evenings, not two or three.** Reaction noise in any single
evening averages out; the *structure* (who is systematically distinctive on
which dial) is what survives. If you raise `learning_rate`, tastes converge
faster but swing harder on noisy evenings. Seed with `np.random.default_rng(42)`
if you want reproducible runs.

**What to look for** (from the ops runbook): (1) mean pairwise `‖wᵢ − wⱼ‖`
should *rise* and each participant's argmax dial settle onto a different one;
(2) κ should settle and warmth stop swinging as the regulars establish the
vibe; (3) charisma shows as `room_field() − raw_field()`; (4) the elephant is
only visible when you *change rooms* — contrast `distance()` /
`sauna_plunge_gap()` between Tap and wheelhouse; (5) **don't hand-tune** — if a
taste looks "wrong" after many cycles, look at the inputs (the works, the
reactions), not the knobs.

---

## 9. Pitfalls (from the actual build history)

### 9.1 Mean-aggregation collapse on the sphere

The single most-flagged design failure. Mean-aggregating L2-normalized
embeddings is **degenerate**: on the d-sphere, the raw mean of N unit vectors
has expected norm `√(N/d)`, so renormalizing the mean discards magnitude and
amplifies noise — and a mean **erases the spread**, which is the single most
important property of a vibe (a rowdy open-mic and a quiet formal reading can
share a mean). The adopted fix is the **vMF field**: a direction μ̂ *plus* a
concentration κ, never a renormalized sum. In this codebase that survives as
`RoomField.concentration()` (`‖vector − 0.5‖·2`) and `fleetmath.vmf_kappa`.
**Never** replace the field with a renormalized mean of per-message vectors.

### 9.2 Charisma capturing the loop (the strong agent homogenizes everyone)

An over-strong charisma is a positive-feedback loop: a charismatic agent bends
the field toward themselves; the field then becomes the target everyone
acclimates *to*; the agent's own vibe is reinforced; and every distinct taste
collapses into the charismatic agent's. The guards in the code: pulls are
summed and divided by `max(1, total)` (so a crowd of pulls is one step, not
N), the field is clamped to `DIAL_BOUNDS` (charisma saturates, never
overshoots), and `charisma` defaults to a small `0.15`. Keep charisma low and
the aggregate-bounded structure intact — a homogenized room is a dead elephant.

### 9.3 Zero-weight dials collapsing to 0 instead of neutral

Weighting the **raw** dial value by taste makes a dial an agent ignores read as
`0.0` — which for a `[0,1]` dial like presence *means* "empty room." The agent
then spuriously "sees" an empty room instead of simply not registering
presence. The fix (already in `PersonalElephant.read` and
`Participant.__post_init__`): weight the **deviation from neutral**
(`DIAL_CENTER`), so weight ≈ 0 reads as neutral. Preserve this whenever you
touch the weighting.

### 9.4 Speaker-identity confounds

The killer confound in room-vibe learning: when the same cast appears across
rooms (the trades-nights), an encoder clusters by **voiceprint**, not room
context, and "room vibe" is really "speaker identity." One-hot speaker identity
is **poison** as a feature. The decisive control is **speaker-heldout** —
remove all same-speaker clips from the candidate set and confirm room
discrimination does *not* drop. In this repo the analogue is: presence uses
author keys only as a **mask/occupancy trace**, never as a feature that could
proxy identity, and the self-tuning signal is cast-relative (vibe minus the
cast mean) rather than author-identity-indexed.

### 9.5 "Settings can't be designed top-down" — the guitarist principle

The whole reason self-tuning exists. A skilled guitarist is the only one who
can recognize a well-built guitar — one looks pretty, another sounds wonderful,
another has a good neck. You don't know where the settings belong until
**different agents desire different settings and self-fine-tune to the moment
they're in** — *reading the room is a relationship to the room*. So: set the
conditions (lexicons, ranges, priors, rates), run many cycles, and let the
tastes diverge. Do not hand-pick the final dial weights and call it done.

### 9.6 Feedback stability (fleet-scale)

At fleet scale, raw κ into the nudge is an unbounded positive-feedback loop
(the 30-day diary would blow apart by day 10). The adopted guard is a
**damping bell** on κ (`κ·(1 − clamp(κ, 0.2, 0.8))`, see
`docs/fleet-simulation-notes.md`) and a low-pass filter on nudge values. If you
wire the elephant into a live loop, add damping before you add gain.

---

## Cheat sheet — "which knob do I turn?"

| I want… | Touch |
|---|---|
| The elephant to feel a new word as warm | add to `mood.POSITIVE` |
| Sarcasm to read as cynicism, not warmth | add to `cynicism.CYNICAL` (and note v0 can't detect sarcasm) |
| More jokes detected | add to `joke_landing.JOKE_MARKERS` |
| Panic to fire sooner | add to `panic.ALARM` / `URGENCY` or raise `0.40` weight |
| An agent to ignore a dial | set that dial's `dial_weights` ≈ 0 (reads neutral) |
| An agent to warm to the room faster | raise `acclimation_rate` |
| A strong presence to bend the room | raise `charisma` (modestly) |
| The room description to change tone | extend `mud.WEATHER`/`LIGHT`/`JOY_ADJ`/`CLOSE_DETAIL` |
| The vision model to heed the elephant more | raise `apply_nudge(..., strength=...)` |
| Tastes to diverge faster | raise `tune_participant(..., learning_rate=...)` and run more cycles |
