# Murmur × Elephant — the Cave Wall Writes Itself

*2026-08-17 · the captain's Terrain reframing, cross-pollinated with
murmur-agent: the self-populating shadow journal.*

---

## The reframing, restated

The Terrain (`docs/terrain-2026-08-17.md`) says it plainly: the true
state of any agentic conversation — the weighting and vectoring of
every token, every JEPA reading, the whole field of it — is beyond any
human's rendering. That is the **Terrain**: the real ground, the thing
itself. What we actually *see* — the trail of words, the monologues,
the transcripts, the logs — are **Shadows on the cave wall**: witness
marks of the terrain's activities. Lossy projections. Enough to
recognize, never enough to be complete.

And the Terrain names murmur-agent explicitly, as one of the cave's
walls:

> the agent's internal monologues in a self-populating sustaining
> system (murmur-agent, or any other) — *are Shadows on the cave wall*.

This module makes that sentence code. **Murmur is the fleet's shadow
writer**: a self-populating system where every internal monologue
becomes a commit — a witness mark. The elephant gives murmur its
senses: the thinking is *informed by* the elephant's readings of the
rooms it sits in, and every witness mark *carries* those JEPA readings
as terrain front-matter — the shadow with enough terrain context to
agree on the action. And the elephant can read murmur's history as a
room — the shadow-trail — with the same DialBank that reads every other
room.

Two pieces, one loop:

```
                 ┌────────────────────────────────────────────┐
                 │  THE TERRAIN (the elephant's senses)        │
                 │  a room, its field, its dials, its pulses  │
                 └───────────────┬────────────────────────────┘
                                 │  pulse readings (direction/rate)
                                 ▼
   ┌─────────────────────────────────────────────┐
   │  THE WALL (murmur — the shadow writer)       │
   │  every internal monologue → a witness mark   │
   │  committed WITH its readings as front-matter │
   └─────────────────────────────────────────────┘
                                 │
                                 ▼
                 ┌────────────────────────────────────────────┐
                 │  THE ELEPHANT READS THE WALL               │
                 │  the journal as a Room (field + dials)     │
                 │  retrieval by feeling ("when did murmur    │
                 │  last feel like this?")                    │
                 └────────────────────────────────────────────┘
```

## What murmur is (from its own charter)

Murmur-agent is the fleet's Scout-class all-night thinker: drop it
into a project, give it a topic, let it think while you sleep. **Every
thought becomes a Git commit, every insight a file.** Five thinking
strategies (explore, connect, contradict, synthesize, question), a
knowledge tensor, budget-agnostic, tolerant of silence. Its commits
are, in the captain's reframing, exactly what a witness mark is: the
visible residue of an invisible process.

`elephant/murmur.py` does not re-implement the TypeScript/C engine. It
implements the **pattern** in elephant idiom — a self-populating
shadow journal — and the **seam** that makes it self-populating: the
internal-monologue pulse.

## `MurmurJournal` — the wall

```python
from elephant.murmur import MurmurJournal, MurmurSpace, murmurize_pulse
from elephant.pulse import PulseLoop
from elephant.space import ChatSpace

journal = MurmurJournal("/tmp/my-wall", space_id="the-tap")
```

A git-backed (or plain-dir) journal. Every entry is one internal
monologue thought plus its terrain front-matter:

| Key | What it carries |
|-----|-----------------|
| `readings` | the raw JEPA dial readings the thought was born under |
| `direction` | per-dial movement (last TWO readings) |
| `rate_of_change` | per-dial acceleration (last THREE+, the second difference) |
| `warmth` / `warmth_direction` / `warmth_rate` | the field's temperature and its kinematics |
| `confidence` | how loud the loudest hand was this pulse — the prisoner's estimate of how well it saw the shadow (0–1) |
| `ts`, `space_id`, `topic` | when, where, and about what |
| the body | the monologue itself — the shadow |

**`write_monologue(text, readings, ts, topic, ...)`** — writes the
thought as a numbered file (`0001-warm-night.md`) with the readings as
front-matter. If the path is inside a git repo, it **commits** each
entry (`murmur: <topic>`) — murmur's pattern, verbatim: every thought
a commit. If not, it is a plain directory of witness files (the /tmp
overnight case; git failure is never fatal — the wall always works).

**`read_room()`** — the journal as a room: every entry becomes
`Message(author="murmur", text=..., ts=..., channel=topic)`. The
elephant reads the shadow-trail with the DialBank like any other room
— a field comes out of it. The trail is also a first-class space
adapter:

**`MurmurSpace(journal)`** — a DocSpace-style room (`kind="doc"`), the
same adapter shape `space_more.py` uses for repos: witness marks as
messages, tint target = a status line ("the wall is bright — warm
thoughts bubbling all night"). Read-only, always current — the wall
re-renders on every `room` access.

**`retrieve(query, ...)`** — retrieval by feeling: *"when did murmur
last feel like this?"* The query maps reading names to targets
(`{"panic": 0.7}`) or **deadband gates** (`{"panic": (0.5, 1.0)}`):
an entry outside the band does not ring — it is excluded; entries
inside rank by nearness to any scalar targets. The profile searched is
the full front-matter — readings, warmth kinematics, and the macro
read (`d_*` direction, `r_*` rate) — so you can retrieve by level *or*
by movement ("the spike"). This deterministic path is the deadband's:
**it rings on LEVELS** — a range is a gate, a target is a distance,
and the answer is the same every run. When `elephant/jepa_rag.py` is
present (the JEPA-RAG — retrieval where JEPA readings are first-class
citizens), the witness marks also index into its `JepaMemory` as
moments (the shadow + readings + ts + space) through `to_memory()`, and
`retrieve_feeling()` retrieves BY VIBE — cosine similarity in reading
space, the moment that felt most like the query. Vibe and level can
disagree (a quiet room whose only signal is a little panic can
out-vibe a loud warm room that also carries panic): that is the
difference between feeling and ringing. The deadband rings; the
feeling remembers.

## The seam — the pulse feeds the wall

`elephant/pulse.py` already runs the internal monologue on a constant
heartbeat: agents think even when silent, and every pulse takes a
perception check — the macro read of the room as a whole hand (two
numbers show direction, three+ show rate of change). The seam closes
the loop:

```python
loop = PulseLoop("murmur", tap, period=2.0)
# ...the room lives, the loop ticks...
murmurize_pulse(loop, journal, topic="warm night", ts=12.0)
```

`murmurize_pulse()` fires the pulse when due, runs
`internal_monologue()`, and commits that silent thinking **with** the
pulse's perception readings as front-matter — raw readings, direction,
rate of change, warmth kinematics, and a `confidence` (how loud the
loudest hand was this pulse). The wall writes itself, informed by the
terrain. This is the self-populating sustaining system the captain
named: murmur + elephant = the cave wall, writing its own shadows.

Every witness mark is therefore enough to *agree on the action*: the
monologue is the shadow (what the agent noticed), the front-matter is
the terrain context (what the room actually felt like) — the two
together are what two agents (or an agent and a captain) need to align
on what to do next.

## The reverse seam — the elephant reads the wall

- **As a room** — `journal.read_room()` / `journal.read_field()`: the
  elephant reads the shadow-trail with the same bank that reads the
  bar. A field comes out: warmth, κ, the dials. The trail of a night's
  thinking has a temperature.
- **As a doc space** — `MurmurSpace(journal)`: the trail as a
  DocSpace-style room with a status-line tint, so the elephant's
  readout writes back in the doc idiom.
- **As a git log** — because every witness mark in a git repo IS a
  commit, the existing `DocSpace.ingest_git_log(repo)` reads a
  git-backed murmur journal with **zero new code**: authors become
  committers, subjects become text, and the field reads the repo's
  *shape*. Two adapters, one trail.
- **By feeling** — `retrieve()`: the elephant doesn't have to re-read
  the whole wall; it asks "when did murmur last feel like this?" and
  the deadband decides what rings.

## The demo — the cave wall

`examples/demo_murmur.py` — one evening at The Tap, eight pulses, a
plain-dir journal in `/tmp` (no git needed):

1. **THE TERRAIN** — the room lives: a warm room (cheers, laughter,
   reactions), then a fire ("FIRE in the galley! evacuate now!!
   mayday"), then the agent walks to a quiet deck.
2. **THE OVERNIGHT WATCH** — every pulse's monologue is committed
   with its readings: `panic 0.05 → 0.67 → 0.13`, `Δpanic +0.48/pulse`
   at the fire. The agent says nothing all night; the wall writes
   everything.
3. **THE ELEPHANT READS THE WALL** — the shadow-trail as a room:
   `warmth −0.27, κ 3.59`, the dials, and the doc-room tint
   ("❄ a cold night — the wall is blank and sharp").
4. **RETRIEVAL BY FEELING** — `{"panic": 0.7}` finds the panic
   witnesses; `{"mood": 1.0}` finds the warm night; and
   `{"panic": (0.5, 1.0)}` — the deadband ring — returns exactly the
   two fire witnesses. The quiet hours never ring.
5. **THE WITNESSES** — every mark with its terrain front-matter,
   including one raw witness file, `---` block and all.

## Tests

`tests/test_murmur.py` (7 tests):

- write → entry with readings front-matter intact (file + dict);
- write → read → field round-trip with real numbers (a fresh journal
  over the same path recovers everything; the DialBank feels a field);
- retrieval by reading finds the target profile (scalar, range gate,
  topic filter, threshold, k-ordering);
- retrieval by movement (`d_panic` finds the spike);
- `MurmurSpace` reads like a room (field, tint, always-current wall);
- the pulse seam: three pulses self-populate the journal, each witness
  carrying raw readings + macro read, and the panic pulse feels more
  panic than the warm one;
- in a git repo, each witness mark is a commit (`git log` shows
  `murmur: <topic>`, one commit per thought).

## The critique, answered

Asked of a 405B critic: *does this honor the cave reframing, or is it
overclaiming?* The honest answers, folded in:

- **The wall does not write itself — a prisoner writes it.** The
  captain's phrase names the *system* (murmur-agent is a
  self-populating sustaining system), but the critic is right that the
  elephant is the prisoner: it observes shadows and scribbles its
  interpretations. Every witness is therefore **labeled as a witness**:
  the monologue is the interpretation, the front-matter is the data it
  was interpreting — the two are never confused, because they live in
  the same file, separated by the `---` of the front-matter.
- **The shadow is filtered through perception — so each witness
  carries its confidence.** `confidence` = how loud the loudest hand
  was this pulse. A flat room yields a low-confidence read; a panic
  spike yields a confident one. The prisoner knows how sure it is.
- **Traceability.** Every witness links its narrative to the raw data
  that informed it — the readings are *in the file*. To trace a shadow
  back to the object casting it, read the front-matter.
- **The deadband is the discipline.** Retrieval ranges are gates, not
  suggestions: outside the band, the witness does not ring. The wall
  does not shout everything it knows — only the moves that matter.

## The one-line version

> Murmur is the shadow writer — every thought a witness mark. The
> elephant is the terrain — what the thinking is really about. The
> pulse is the seam — every silent monologue committed with its
> readings. And the wall is a room the elephant reads and remembers —
> the shadow-trail, retrieved by feeling.

*The shadow is not the thinking. The shadow is the witness. Enough to
agree on the action — and when the terrain crosses the band, the
witness rings.*
