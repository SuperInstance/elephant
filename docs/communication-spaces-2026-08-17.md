# Communication Spaces — the elephant works everywhere

*2026-08-17 · the captain's modularity directive.*

The elephant must not be coupled to the MUD system. People will be
drawn into the MUD *because* of tools like the elephant — but the
elephant itself works in **any communication space between agents,
humans, bots, or sensor arrays**: a message board, a chatroom, a
messenger conversation, an X thread, a Discord server, a Slack
channel, an email thread, a team radio channel, a sensor telemetry
bus, a fish-finder feed, a camera watch. One sense, many rooms.

## The Space abstraction

Every communication space is normalized into the same thing the
elephant already reads:

```
Space (anything)  ──adapter──►  Room (messages / frames, normalized)
                                        │
                        ┌───────────────┼────────────────┐
                        ▼               ▼                ▼
                   DialBank      RoomField           Presets
                (7+ dials)   (warmth, κ, gap)   (Room-Elephant /
                                                 Personal-Elephant)
                        │               │                │
                        └───────► tint_description ◄────┘
                              (the space's own body
                               language — tokens every
                               participant sees)
```

The core (`room.py`, `dial.py`, `field.py`, `presets.py`, `nudge.py`)
never knows what the space *is*. It only knows Rooms, Messages,
Frames, DialBank, RoomField. Everything space-specific lives behind a
thin adapter.

## Adapters (each ~50-150 lines)

| Space | Adapter | Normalizes to |
|-------|---------|---------------|
| MUD room (our Tap, other MUDs) | `MudSpace` | messages from room events + NPC chatter; the room description becomes the tint target |
| Chatroom / messenger / X thread | `ChatSpace` | messages with authors, reactions, reply trees (gravity/reverb/ripple work as-is) |
| Multi-agent channel (agent bars, CNS bus) | `AgentSpace` | agent messages + system events on the same clock |
| Human + bot mixed channel | `HumanBotSpace` | same as chat; presence dial reads humans vs bots distinctly |
| Sensor arrays (radar, sounder, cameras, nav, autopilot) | `SensorSpace` | SensorFrames — already supported by `SignalRoom` |
| Email / async threads | `AsyncSpace` | messages with long time-deltas; gravity half-life stretches |
| File / doc workspace (ai-writings, repos) | `DocSpace` | commits, file events, review comments as messages |

Every adapter must provide:
1. `ingest(...)` — pull/accept events from the native space
2. normalized `Room` or `SignalRoom` access (same timestamped sequence)
3. `tint_target()` — what the elephant's description-mutation writes
   back into the native space (MUD description text, a chat topic, a
   pinned message, a status line, sensor alert phrasing...)
4. `send_back(...)` — optional: push the elephant's readout (numbers
   or tint) back into the space in the space's own idiom

## The rule

**JEPA correlates; it never replaces.** The vision model on the radar
screen is not replaced by the elephant — the elephant nudges what it
compares. The MUD's procedural generation is not replaced — the room's
field changes the description tokens everyone reads. A chatroom's
people are not replaced — the vibe shifts the air and the air shifts
the words. The elephant is the light, and the light works in every
room that has one.

## Build order

1. `elephant/space.py` — the `Space` protocol + adapter registry +
   the generic adapter scaffolding.
2. Adapters one at a time, starting with `MudSpace` (already in
   progress via `mud.py`) and `ChatSpace` (proves it works outside
   the MUD), then `SensorSpace` (reuses SignalRoom), then the rest.
3. `examples/demo_spaces.py` — the SAME elephant reading three
   different spaces: a MUD bar, a chat thread, a sensor array —
   showing one sense, many rooms.

---

*The elephant doesn't care if the room is made of oak, pixels, or
telemetry. It only cares how warm the room is — and how the room's
light changes everyone in it.*
