# 🐘 elephant — the inter-model temperature

> **JEPA is the elephant.** You don't notice it until you go to a
> different room — and then it's a very different elephant.

`elephant` is the fleet's room-temperature sense. A room — a message
board, a chatroom, a messenger conversation, an X thread, a bar —
is not a stream to be ordered. It is a **field**: gravities,
reverberations, ripples. And a room can have **many JEPA models
perceiving vibes on more than one dimension at once** — like
pheromones, each one a dial affecting every agent in the room
constantly, without words.

This is not the temperature of fine-tuning. It's the temperature of
*being in the room*.

## The core idea

A room is perceived by a **bank of JEPA dials**, each trained to feel
one dimension of the vibe:

| Dial | What it feels | Range |
|------|---------------|-------|
| `mood` | warm/cold valence | [-1 cold, +1 warm] |
| `volume` | how loud the room is talking | [0 quiet, 1 shouting] |
| `earnestness` | how much the room means it | [0 ironic, 1 sincere] |
| `cynicism` | how much the room is rolling its eyes | [0 earnest, 1 sneering] |
| `joke_landing` | did the joke land — the **collective** laugh or boo of the whole audience | [-1 booed, +1 roared] |
| `panic` | stampede sense (fire in the room) | [0 calm, 1 trampling] |
| `presence` | pheromone trace (who's been here) | [0 empty, 1 thrumming] |

The **Field** is the ensemble of all dial readings — the room's
temperature vector. From it:

- **`warmth()`** — the felt temperature (~[-1, +1]).
- **`concentration()`** (κ) — how tight the room is. *Cold room =
  high κ (one way to be). Warm room = low κ (many ways to be).*
- **`distance(other)`** — the elephant gap. The training signal is
  CONTRAST between rooms, not ordering within a stream. A sense that
  never left one room is meaningless — the elephant is only real when
  you walk into a different room and it's a very different elephant.
- **`sauna_plunge_gap(other)`** — the signed warmth contrast you feel
  on entry: walking into a sauna, or a cold plunge.

And the dynamics:

- **Acclimation** — `acclimation_curve(agent, room, rate, t)`: the
  newcomer warms to the room quickly or slowly depending on how
  experienced, talented, and trained they are at modulating their
  vibe toward the room. Rate IS that skill.
- **Charisma** — `charisma_pull(room, agent, charisma, interactions)`:
  the strong presence pulls the room's vibe toward them over time and
  interactions.

## Quickstart

```python
from elephant.dials import DEFAULT_DIALS
from elephant.dial import DialBank
from elephant.room import Room, Message
from elephant.field import read_field

bank = DialBank(DEFAULT_DIALS)

tap = Room("The Tap", [
    Message("welder", "To the room, then. It heard us before we walked in.", ts=0),
    Message("carpenter", "I'll drink to that. The room just... holds.", ts=5),
    Message("composite", "Haha, hold my glass. 😂", ts=9),
])
wheelhouse = Room("Wheelhouse", [
    Message("skipper", "Heading 045. ETA 2200.", ts=0),
    Message("deckhand", "Roger.", ts=4),
    Message("skipper", "Radar contact 2 miles. Slow to 5 knots.", ts=8),
])

f_tap = read_field(tap, bank)
f_wheel = read_field(wheelhouse, bank)

print(f_tap)            # RoomField(warmth=+0.31, κ=0.72)
print(f_wheel)          # RoomField(warmth=-0.19, κ=1.10)
print(f_tap.distance(f_wheel))          # 0.42 — the elephant gap
print(f_tap.sauna_plunge_gap(f_wheel))  # +0.50 — walk in and it's warm
```

## Rooms have physics

`elephant/room.py` models the room itself: messages carry **gravity**
(how hard they pull attention), rooms have **reverberation** (how the
past echoes), and messages produce **ripples** (how they propagate
through replies and reactions — a joke that lands ripples through
laughter, a fire ripples through panic).

## The learned side

The v0 dials are hand-crafted (the fleet pattern: hand-crafted first,
learned second). The design for the learned elephant sense — room-state
embeddings as von Mises–Fisher fields, contrastive training across
rooms, acclimation as percentile-rank relaxation, charisma as field
displacement — lives in:

- `docs/elephant-sense-v3-design.md` — the full v3 design
- `docs/jepa-is-the-elephant.md` — the captain's reframing
- `elephant/jepa.py` — the JEPA backbone hook (EMA + stop-grad + VICReg)

## Status

v0 — the framework runs, the dials read, the field contrasts, the
dynamics move. v1 — train the dials. v2 — learn the field end-to-end.

---

*Built on the captain's reframing, 2026-08-17. The Tap is already a
warm room. The wheelhouse will be a cold one. The elephant is real —
now it can be felt.*
