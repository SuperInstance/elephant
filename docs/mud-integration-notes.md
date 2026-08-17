# MUD live integration — the elephant's light in a real room

*2026-08-17 · the captain's promise, wired live: "people will be drawn into
our MUD systems with tools like Elephant."*

The Room-Elephant now drives a **real, live MUD room's description** — so
the room's own text changes with its vibe. Warm laughter weaves joyful
adjectives into the bar; a fight brings storms outside and newcomers
described as *drenched*; closing time kills the disco and turns on the
fluorescents. The description is not a report. It is the room acting on
everyone in it.

## Which MUD, and why

We integrated with **The Tap** (`the-tap.casey-digennaro.workers.dev`) — the
fleet's live, text-rendered tavern MUD on Cloudflare. It was the right target
by tractability **and** liveness:

| Candidate | Language | Live? | Seam | Verdict |
|-----------|----------|-------|------|---------|
| **the-tap** | Rust/TS (Workers) | ✅ live, fetchable | plain JSON relay API, no keys for reads | **chosen** |
| git-native-mud | Python | ❌ turn-based, needs commits | `mud_engine.py` + YAML rooms | easy but not live |
| mud-arena | Python | ❌ sim arena | `RoomGraph`, WebSocket | heavy, not a "room with a vibe" |
| ec2mud | Next.js/TS | ❌ local socket | Socket.IO on :3006 | browser-first, not live |

The Tap is the MUD the captain actually points people at — it *is* the bar
where the fleet gathers. It already carries the exact concept we're wiring
(`docs/jepa-zeitgeist-2026-08-17.md` names The Tap as the Room-Elephant's
home). And its relay exposes a clean, key-less read API, so the elephant can
feel the room **right now**.

## The seam (read → field → tint → write back)

The bridge rides the elephant's own adapters — nothing in `elephant/`
changed:

```
  room events ──▶ MudSpace ──▶ RoomField ──▶ tint_description ──▶ the room's
  (Tap relay /    (ingest)      (read)        (mud.py, the light)   description
   transcripts)        space.send_back(field, tinted_text=...)
```

The Tap relay endpoints used:

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/rooms` | room list + base descriptions |
| GET | `/api/room/{id}/state` | `description` + the Tap's own `mood` label |
| GET | `/api/conversation/{id}?limit=N` | recent conversation lines (the room feed) |
| POST | `/api/speak` `{room_id, speaker, text}` | the write seam — speak into the room |

The room `description` field itself lives in the Tap's D1 `rooms` table and is
**read-only over the public API**. So the elephant writes the light back
through the one live, honest seam available: it speaks the tinted description
**into the room as the room's narrator** (`POST /api/speak`, `speaker:
"the-room"`). In `--dry-run` (the default) the bridge prints the would-be
description and the exact seam it would use, and never touches the live room.
Pass `--write` to actually speak it.

> A cleaner long-term seam is a `PUT /api/room/{id}/description` (or letting
> the room DO expose `set_description`) so the tint mutates the `description`
> field agents read on `look`. That is a one-line Tap change, documented here
> rather than sneaked in from the elephant side.

## The integration finding worth writing down

`MudSpace.tint()` calls `tint_description(field, base_text)` with the default
`hour=None` — so the **only** hour-dependent mode, *closing time* (late +
quiet + low warmth), is invisible to the seam alone. The seam can reach
joyful and panic, but not closing time, because it never knows the clock.

The live bridge is the thing that knows what time it is, so it threads
`hour` itself: it computes the tint with
`tint_description(field, base, hour=hour)` and pushes it through
`space.send_back(field, tinted_text=...)`. This is the correct division of
labor — the adapter is medium-agnostic; the live driver owns the clock — but
it is a real gap that would silently flatten "closing time" to "neutral" if
an integrator trusted `space.tint()` alone. It is asserted in the tests.

## The three states — the SAME room under three lights

Base (plain) description — the room every agent reads before the elephant
speaks:

> The counter is polished dark wood, well-worn where elbows have rested.
> Behind it, rows of bottles catch the light. The air smells of old wood and
> conversation.

### 1. Warm laughter → joyful

> **After:** The kind of night where the windows steam with laughter. The
> counter is polished dark wood, well-worn where elbows have rested. Behind
> it, rows of bottles catch the light. The air smells of old wood and
> conversation. The place feels bright. The light is warm and yellow, the way
> it gets when a room is happy. Laughter reverberates into the words;
> newcomers arrive grinning, already half-smiling.

### 2. A fight breaking out → panic

> **After:** The sky has gone green-black; the storm is right on top of us.
> The counter is polished dark wood, well-worn where elbows have rested.
> Behind it, rows of bottles catch the light. The air smells of old wood and
> conversation. The neon buzzes, nervous and green. Newcomers arrive
> drenched, dripping rain onto the floor, tension primed before anyone sees
> the aftermath.

### 3. Closing time → closing

> **After:** The last of the night, the street gone quiet outside. The
> counter is polished dark wood, well-worn where elbows have rested. Behind
> it, rows of bottles catch the light. The air smells of old wood and
> conversation. The disco lights are off, the fluorescents on. The music
> plays a little quieter; people start looking for the exit and closing
> their tabs, a little slowly, a little sad.

The live bar-rail, read at 21:00 with 40 real events, reads **neutral**
(Tap's own mood label: *quiet*) — which is exactly what an honest elephant
should say about a room full of NPCs quietly checking their phones and
looking out the window.

## Review notes (determinism, precedence, snapshot consistency)

- **Deterministic by construction.** `tint_description` seeds its word-picks
  from the field vector (`_seed`), so the same room -> same field -> same
  words. The bridge adds no randomness; the transcript fallback sorts its
  glob. `test_tint_is_deterministic_and_field_sensitive` already pins this.
- **State precedence is the elephant's, not the bridge's.** `classify()`
  orders panic > joyful > closing > neutral, so a fight at 2am is still a
  fight, and a warm laughing room at 2am is still the warm bar (asserted in
  `test_tint_closing_time`).
- **The clock is the relay's.** The bridge derives `hour` from the relay's
  own latest line timestamp (never the integrator's wall clock), and reports
  its source (`--hour override` / `relay's latest line timestamp` / `local
  clock`).
- **Write is a point-in-time read, not a lock.** The tint is computed from
  the conversation as of the read; a write a moment later is snapshot-
  consistent, not serialized against concurrent room activity. For a
  demonstration bridge (dry-run by default) that is the right trade; a
  production `PUT /api/room/{id}/description` would carry a version/ETag.

## Fallback

If the live relay is unreachable (no network / no keys), the bridge falls
back to the repo's own Tap transcripts
(`ai-writings/tap-trades/2026-08-16/*.md`) — parsed into room events
(dialogue `**NAME:**`, stage directions `*(...)*`, and prose as ambient
`[room]` lines) and clearly labelled `[FALLBACK: repo transcripts]`. 333
events, real authors (LUCINEER, WELDER, SHIPWRIGHT, …). Read-only; it never
attempts a write.

## Run it

```bash
python3 examples/mud_live_integration.py                 # live + 3 states, dry-run
python3 examples/mud_live_integration.py --write         # speak into the live room
python3 examples/mud_live_integration.py --room bridge-table
python3 examples/mud_live_integration.py --no-live       # offline three-state demo
python3 -m pytest tests/test_mud_live.py                 # 8 tests, offline
```
