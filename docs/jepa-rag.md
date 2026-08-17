# JEPA-RAG — JEPA readings as first-class citizens, beside time and space stamps

*2026-08-17 · the captain's directive: "Think about a RAG with Jepa
readings as first-class citizens along side time and space stamps."*

---

## The problem with a normal RAG

A normal RAG indexes text embeddings and retrieves similar text. Ask
it "what did the room feel like during the fight?" and it does its
best — but the feeling was never indexed. The words are the shadow;
the feeling is the terrain. Retrieving by words alone is retrieving
shadows without their terrain.

This system indexes **moments** instead. A moment is a shadow with its
terrain context:

| field      | what it is                                              | the Terrain level |
|------------|---------------------------------------------------------|-------------------|
| `text`     | the shadow — the witness words (transcript chunk, bar line, watch log) | Shadow |
| `readings` | the JEPA reading vector — what the room FELT (the 9 dials) | Dial (the terrain, vectorized) |
| `ts`       | the time stamp — when it happened                       | Room / Fleet over time |
| `space_id` | the space stamp — which room it happened in             | Room / Fleet |
| `meta`     | anything else worth riding along                        | — |

The reading vector is not metadata on the text. It is a **first-class
retrieval dimension**, no less than the words. Time and space stamps
ride beside it as dimensions too. That is the captain's "alongside"
made concrete.

## The queries

All queries return ranked `MomentHit`s — the witness text **with** its
reading vector, ts, and space. The first-class citizens ride along on
every hit; that is the honesty guarantee.

| query | what it does | the idiom |
|-------|--------------|----------|
| `query_text(q)` | bag-of-words cosine against the shadows | the normal RAG way — what the words say |
| `query_readings(readings)` | cosine in JEPA space to a target reading profile — or (lo, hi) RANGE constraints per dial ("mood > 0.6, panic < 0.2" made literal) | **the first-class-citizen query** — what the room FELT |
| `query_field(field)` | nearest neighbors in JEPA space to a field (a `RoomField`, a readings dict, a vector) | the perfume query — "find the moment that felt most like right now" |
| `query_time(window)` | hard filter on the time stamp, ranked by proximity to the window's center | "what happened here yesterday at this hour?" |
| `query_space(space_id)` | hard filter on the space stamp, ranked newest-first | "what did the wheelhouse feel like last week?" |
| `query_combined(parts, weights)` | weighted sum of every present dimension — text 0.3, readings 0.5, time 0.1, space 0.1 by default, renormalized over what you give it | the full RAG query — the captain's "alongside" |

The math is deliberately small + honest: numpy matrices (a moments
matrix of reading vectors, a field matrix of warmth/κ, a timestamp
array, a space array) and a bag-of-words TF matrix for the lexical
side. No learned embeddings, no vector database — a few dozen moments,
nine dials of meaning, all in plain arrays.

Two design choices are worth naming, because they are choices:

- **The reading vector is raw, not centered.** Cosine runs on the
  dials as the bank reads them, exactly like `RoomField.vector()` /
  `normalize()` — the fleet's existing field math. A centered or
  variance-whitened reading space is a future refinement; the query
  API does not change when it lands.
- **The TF matrix is dense.** For a transcript corpus (thousands of
  moments, a few thousand tokens) that is a few tens of MB in
  memory. At true fleet scale (10k+ moments) the lexical side swaps
  to a sparse posting list behind the same `query_text` API.

`top_k=None` means *every* matching moment, ranked — for the stamp
queries that is the honest "show me the whole room" answer.

### The first-class-citizen query

`query_readings({"panic": 0.9, "mood": -0.6, "volume": 0.8})` is the
heart of this design: the JEPA reading **is** the query, exactly as
the text is in a normal RAG. Partial profiles are fine — unspecified
dials read 0.0 (the vector's origin, like `RoomField.vector()`). The
ranking is cosine similarity in reading space (the raw cosine:
negative means the moment is the *opposite* feeling, and that is
honest information).

For the captain's threshold idiom, pass ranges: `query_readings({"mood":
(0.6, 1.0), "panic": (0.0, 0.2)})` ranks by the fraction of dials
inside their bounds — a literal "mood > 0.6 and panic < 0.2" that
never lets a panicky moment sneak in because it is otherwise close.

### The combined query — weights

`query_combined` merges every dimension into one score:

```
score = w_text·text_sim  +  w_readings·reading_sim
      + w_time·time_proximity  +  w_space·space_match
```

Default weights are the captain's proportions: readings 0.5, text 0.3,
time 0.1, space 0.1. Weights renormalize over the dimensions actually
present, so a pure feeling query ranks on the full reading weight.
Space and time are **soft** inside the combination (a wrong-space
moment scores 0 on that dimension but can still rank on the others) —
for hard filters, use `query_space` / `query_time` alone.

## The Terrain connection

> "The shadow is not the thinking. The shadow is the *witness*."

Retrieval returns a witness **with its terrain context**: the shadow
text plus the reading vector, the time stamp, the space stamp —
enough to agree on the action. Two agents (or an agent and a captain)
do not need the full terrain; they need enough shared witness to
align on what to do next. A retrieved moment is exactly that: a
shadow you can act on, with the terrain numbers beside it so the
agreement is grounded, not vibes.

The deadband rings up the chain in the same vocabulary: when the
terrain crosses significance, the witness mark rings up. JEPA-RAG is
the memory side of that architecture — when a moment rings, you can
retrieve every other moment that felt like it, in any room, at any
time the elephant has stood in.

## The fleet's shared memory

This is how the elephant becomes the fleet's shared memory: **the
elephant remembers every room it has ever been in, and retrieves by
FEELING.** The Tap's trade nights, the captain's speeches, the
wheelhouse's storm watch and dawn watch and the galley fight — every
one is a moment with its reading vector. Ask for the fight by its
panic, ask for a good night by its warmth, ask for the wheelhouse by
its name, ask for last week by its stamps — or ask for the moment that
feels most like right now, and the perfume takes you to grandma's
shop.

## Building moments

- `moment_from_text(text, space_id, ts)` — one speech act: the dial
  bank reads the text, the readings are computed, not hand-set.
- `moment_from_room(room, space_id, ts)` — a whole conversation (a
  fight, a watch with several voices): the dials feel density, ripple,
  and presence, which one message cannot carry.
- `moments_from_markdown(path, space_id, base_ts)` — chunk a
  transcript into moments (headings start a chunk; long sections
  break on paragraph boundaries), each chunk read into its own
  vector. Timestamps keep chunk order, so a file's moments stay a
  stamp dimension.

## Files

- `elephant/jepa_rag.py` — the memory, the queries, the moment builders
- `tests/test_jepa_rag.py` — moments built from real fleet data
  (tap-trades 2026-08-16, the speeches, the boats); reading queries,
  field queries, stamp queries, the weighted combination, and the
  honesty guarantee (every hit carries its readings)
- `examples/demo_jepa_rag.py` — the fleet's memory in action: the
  warm night, the fight, the wheelhouse, the moment like now, last
  week by the stamps

---

*Retrieval by feeling. The shadow with its terrain. Enough to agree
on the action.*

— *the elephant's memory engineer, 2026-08-17*
