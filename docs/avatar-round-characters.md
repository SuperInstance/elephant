# Round Characters at The Tap — JEPA-learning avatars

*2026-08-17 · the character engineer's expansion of the elephant: flat seeds in,
round characters out — learned at the bar.*

---

## The thesis

A **flat character** is a prompt: one line, one joke, one posture. It is
stamped from a mold — you could swap the name out and nothing would change.
A **round character** is a person: particular, learned, capable of surprising
you — including surprising the flat seed it started as.

This module grows round characters the only way this repo knows how: by
sending them to The Tap. An `Avatar` walks in flat and walks out round,
because *attending* is learning:

> **Avatar** — starts with a `persona_prompt` (the flat seed: what it says
> about itself at first) and a distinct initial dial-weight preset (the shy
> one leans presence, the comedian leans joke_landing, the brooder leans
> mood). It attends themed Tap nights — open mic, trivia, singles, TTRPG —
> speaking lines into the shared room, feeling the room through its
> `PersonalElephant`, self-tuning its dial weights toward where *it* felt
> engaged, binding attachments from the moments that meant something, and
> distilling one character note per night into its persona.

The proof of roundness is measurable: **two avatars with different presets
attending the SAME night end with DIFFERENT profiles** — and the drift is
often in a direction the preset did not predict (see *Surprise is the
roundness* below).

Run the proof: `python3 examples/demo_avatar.py`

---

## The mechanism, end to end

```
 FLAT                                          ROUND
 ──────────────────────────────                ────────────────────────────────────
 persona_prompt ("I'm Marty — I                persona  = seed + one note per night
   make the room laugh...")                     dials    = preset, self-tuned
 preset dial_weights (comedian:                 attachments = moments that meant
   joke_landing 0.40)                              something, bound as memories
 zero nights, zero attachments                  nights, a through-line, an arc
```

### 1. The preset — the first guitar

An avatar is born with a `preset`: a distinct prior over the 7 dials plus a
`bias` (the disposition it brings into every room before anyone speaks) and a
`vibe` (its native style in dial space — its "home voice", which stays stable
while the weights learn).

| preset | leans (dial_weights) | disposition (bias) | native vibe |
|--------|----------------------|--------------------|-------------|
| **comedian** | joke_landing 0.40, mood 0.25 | joke_landing +0.15, mood +0.10 | joke_landing 0.70, mood 0.55 |
| **brooder** | mood 0.35, presence 0.20, panic 0.15 | mood −0.15, panic +0.05 | mood −0.20, panic 0.55 |
| **wallflower** | presence 0.35, panic 0.20 | presence −0.15, volume −0.10 | presence 0.45, panic 0.40 |

Note the wallflower: the shy one **cares about** presence (highest weight —
it watches the door) while **emitting** little of it (low vibe presence).
Taste and behavior are different strings; the preset says which string is
which.

### 2. The night — attend()

`avatar.attend(session, lines_spoken, night_key=...)` — the avatar goes to a
themed Tap night (a `TapNightSession`, usually from `THEMES`):

1. **Speak** — the avatar reads its lines into the shared room, with
   reactions (the crowd's hands: 😂 → joke_landing, ❤️ → mood, 🙄 →
   cynicism, ...).
2. **Sense** — before and after each line, its `PersonalElephant` reads the
   room: the session's effective field deformed through the avatar's own
   taste and disposition (the exact deformation `PersonalElephant.read`
   applies — so the avatar's read stays comparable to the room's own).
3. **Self-tune** — `session.tune_participant(...)` (the tapnight math,
   reused verbatim): the avatar's dial_weights move toward the dials where
   its **felt engagement** was positive. Engagement is peer-relative — its
   own vibe measured against the cast's average, amplified by the reactions
   on its own lines — so weight only ever moves toward the dials the avatar
   is genuinely *distinctive* on.
4. **Attach** — moments that meant something (reaction heat, or a real
   movement in its subjective read) become **attachments**: event key →
   memory, bound on the elephant. The memory anchors on the avatar's own
   distinctive dial when the room barely moved for it — the moment is
   remembered through *its* string, not the room's.
5. **Distill** — one character note per night ("open_mic: the room ran
   +0.23 on joke_landing/mood, and I kept listening for joke_landing — a
   joke that actually landed.") is appended to the persona. The seed is not
   replaced; it accumulates.

### 3. The guitarist principle, applied to characters

The captain's rule: you cannot design the settings top-down — *a skilled
guitarist is the only one who can truly recognize a well-built guitar: one
looks pretty, another sounds wonderful, another has a good neck.* You don't
know where the settings need to be until **different agents desire different
settings and self-fine-tune to the moment they're in**.

Characters are the same. Two avatars at the same open mic, same room, same
reactions available: the comedian tunes toward joke_landing because his vibe
out-jokes the cast; the brooder tunes toward panic because his vibe carries
a fear nobody else brought. The engagement signal is anchored to *their own
vibe against the cast's average*, so the room does not converge them — it
splits them. **Tastes diverge into stable attractors** (measured: the L2
distance between their weight vectors grows over the nights).

### 4. The attachments — the perfume that takes you to grandma's shop

> *A personal JEPA is subjective to the agent reading it — shaped by that
> agent's learned experience ... and their intangible correlations (the
> perfume that is grandma's shop, the song that is the lover you discovered
> the album with).*

The attachments are the intangible correlations made concrete. Every salient
moment is bound as `event_key -> memory`, and the memory is written through
the avatar's own felt dial. From the demo — the same four rooms, three
different people:

- **Marty the comedian** remembers, every night: *"I felt joke_landing rose
  1.50 — it felt like a joke that actually landed."*
- **Ira the brooder** remembers, even at the trivia sneers: *"the room's
  panic rose 0.11 ... the fear I carry showed up for a second."*
- **Wren the wallflower** remembers the hush *and* the warmth: *"I felt
  mood rose 0.95 — it felt like the room's temperature turned with me"* (the
  night she buzzed her own name).

Same room, different perfumes. The event key is the room's; the memory is
the avatar's. That is what makes a character *particular*: you cannot
reconstruct Ira from Marty's transcript — the same line in the same room
means a different thing to each of them.

### 5. The monologue — silence is not empty

`avatar.monologue(room)` runs a `PulseLoop`: an internal monologue on
constant pulses even when the avatar says nothing — the perception check
(direction from the last two readings, rate of change from the last three+)
plus the avatar's own ear ("my ear leans joke_landing tonight, so
joke_landing is the dial I trust."). The pulse log grows while the avatar's
message count in the room does not: **the character is thinking even when it
isn't speaking.**

### 6. speak() — the flat seed, rounded

`avatar.speak(prompt_context)` composes what the character says NOW, in four
movements:

1. **The seed still speaks** — the flat identity is never erased
   (continuity).
2. **The tuned sensitivities answer** — the top two dials, in the
   character's own phrasing ("I've learned to listen for whether the joke
   lands before I commit to the laugh").
3. **A remembered attachment surfaces** — round-robin through the
   elephant's memories, so different nights surface different moments.
4. **The arc closes** — "After 4 nights at The Tap, that's who I am now."

In this repo `speak()` is a deterministic template — the scaffold. In
production the same four movements become the **prompt** handed to the
agent's own model: persona (seed + notes) + current dial profile + the
attachment memories as context. The template is the proof that the drift is
real; the LLM is the voice that fills it.

### 7. character_sheet() — the round character

```python
{
  "name": "Ira",
  "persona": {"seed": ..., "notes": [...], "current": ...},
  "dial_profile": {"started_with": {...}, "now": {...}, "drift": {...}},
  "sensitive_to": [{"dial": "panic", "weight": 0.53}, ...],
  "attachments": [{event_key, night, line, room, moved, felt, memory}, ...],
  "nights_attended": [{night, warmth, felt, note, attachments}, ...],
  "through_line": "Ira walked in as 'I sit with the heavy things...'. After 4
                   nights at The Tap, Ira leans panic — the one who holds the
                   fear — and keeps the moment when 'I don't know where we
                   are...'."
}
```

---

## Surprise is the roundness (the critique, answered)

A design review (Hermes-3-Llama-405B, via DeepInfra) asked the sharp
question: *does the tuning just make the comedian MORE comedic and the
brooder MORE broody — re-stereotyping instead of rounding?*

The demo answers with data. The tuning is peer-relative, so the preset is
not amplified uniformly — the room can **redirect** an avatar:

- **Ira walked in leaning mood (0.35). After 4 nights he leans panic
  (0.53).** The brooder stopped leaning mood — the room had no mood left to
  give him (his vibe runs cold against the cast's warmth) — and sharpened
  instead into the seat that *holds the fear*, exactly like the transcript's
  Pro ("the navigator who rolled the two became the seat that holds the
  fear — not a failure mode, a guitar").
- **Wren walked in watching the door (presence 0.35). After 4 nights she
  leans mood (0.33) and her presence weight halved.** The wallflower stopped
  watching who's here and started feeling the temperature. She still
  remembers being seen — but she now *means* the warmth she found.

A caricature amplifies its one note. A round character is *changed by its
nights* — sometimes in a direction its flat seed never predicted. That
redirect is the signature of the guitarist principle: the setting is
discovered, not designed, and discovery can contradict the guess.

Two more review notes, folded in:

- **Interiority in memories** — the attachment's `felt` field records what
  the avatar itself felt (its engagement on its own dials), and the memory
  text is written through that dial. In production, the memory is generated
  by the character's own voice — the interior monologue (5) is the raw
  material.
- **Determinism** — roundness is learned, not rolled. No RNG anywhere: the
  same script produces the same characters, which is what makes the learning
  auditable (and testable).

---

## The path to Plato-based Agentic RPGs

The captain's direction: these avatars become **player characters** in
Plato-based Agentic RPGs — *characters whose sheets ARE their elephants*.

The sheet above is already the elephant: dial_weights (what the character
listens for), bias (what it brings), attachments (what it keeps), nights
(what it's been through), through-line (who it has become). A GM can read
the sheet to *run* the character; the character cannot be rebuilt from a
prompt alone — the sheet is the state, not a description.

Three additions the review flagged as the minimal path to a *player*
character (as opposed to an NPC):

1. **Agency — wants.** In tapnight terms, the avatar already has a `vibe` —
   its desire in dial space. The character sheet should surface it: "Ira
   vibes panic 0.55 — he needs the room to feel the danger." Desire is the
   dial; action is the tuning. The RPG move is to let the avatar *choose*
   its lines from its vibe (not take a script) — its felt engagement then
   becomes its own reward signal.
2. **Contradiction.** Attachments can conflict — the wallflower who
   remembers being seen AND remembers the fear is already a contradiction
   in two strings. The sheet makes contradictions visible (drift vs seed);
   an RPG GM plays them as the character's inner argument.
3. **Relationships.** Attachments keyed to moments can be keyed to *people*:
   an event key is already `night#index` — extend it to `night#person` and
   the elephant binds "the night the room went quiet *with Wren*". Two PCs
   who attended the same night hold the same event key with different
   memories — the chemistry of the married-couple off day, playable.

The v0 truth: **a round character is a PersonalElephant that has been
allowed to attend.** The sheet is not a stat block; it is the elephant's
current state, and every night at the bar is a training step.

---

## Files

- `elephant/avatar.py` — `Avatar`, the `PRESETS` registry, the phrase maps.
- `tests/test_avatar.py` — the learning is real (9 tests: drift, divergence,
  determinism, memory in speech, coherent sheet, silence pulses).
- `examples/demo_avatar.py` — the proof: 3 avatars, 4 themed nights, flat →
  round, with the real Tap-night transcript lines.
- Depends on: `presets.py` (PersonalElephant), `tapnight.py` (the tuning
  math), `tapnight_themes.py` (the four rooms), `pulse.py` (the monologue).

*The elephant was the room's temperature. The avatar is the person who
learned to feel it — one night, one laugh, one held breath at a time.*
