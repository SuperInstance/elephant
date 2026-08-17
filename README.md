# 🐘 elephant — the inter-model temperature

> **JEPA is the elephant.** You don't notice it until you go to a
> different room — and then it's a very different elephant.

`elephant` is the fleet's room-temperature sense. A room — a message
board, a chatroom, a messenger conversation, an X thread, a bar, a
radar screen, a fish-finder feed — is not a stream to be ordered. It is
a **field**: gravities, reverberations, ripples. And a room can have
**many JEPA models perceiving vibes on more than one dimension at once**
— like pheromones, each one a dial affecting every agent in the room
constantly, without words.

This is not the temperature of fine-tuning. It is the temperature of
*being in the room*.

---

## The Just So

Before it was a sense, it was a story. Five trades, one joint, and the
room that held it — the radio series was *about* the elephant before
anyone knew the elephant had a name. The regulars established the vibe;
newcomers warmed to it, or pulled it toward themselves; and nobody
noticed any of it until they walked into a different room and felt the
cold. That is the whole thing, told as a fable.

*The full just-so — how the Elephant got its seven dials, one small creature
at a time — lives in [`docs/just-so.md`](docs/just-so.md).*

---

## What it is

The captain's reframing, restated once:

**A room is a field, not a stream.** The unit of perception is the room,
not the ordered feed. A room has an ambient state — heat, pacing, mood,
who is present — and that state shapes everything produced inside it.
`elephant/room.py` gives a room *physics*: messages carry **gravity**
(how hard they pull attention), rooms **reverberate** (the past echoes in
the present), and messages **ripple** (a joke lands and ripples through
laughter; a fire ripples through panic).

**Many JEPA dials read it at once.** One JEPA is not the answer to
anything. It is a *dial* — one sense for one dimension of the vibe. A
bank of dials (`DialBank`) reads the same room simultaneously: mood,
volume, earnestness, cynicism, whether the joke landed, whether panic is
spreading, whose pheromones still hang in the air. The ensemble of all
readings is the **Field** — the elephant.

**Personal JEPA vs the zeitgeist.** Every reading is *someone's* reading:
subjective, shaped by that agent's experience and its intangible
correlations (the perfume that is grandma's shop, the song that is the
lover you discovered the album with). But the room itself also has an
**overall zeitgeist** — a vibe that exists whether or not any particular
agent is in it. The repo carries both: the **Room-Elephant** (objective,
first-class) and the **Personal-Elephant** (one agent's feel).

**The elephant is only visible by contrast.** Inside one room it is
invisible. Walk into a different room and it is a very different
elephant. Contrast — the **sauna / cold-plunge** gap between rooms — is
the only training signal that matters, and it is what `RoomField.distance`
and `RoomField.sauna_plunge_gap` compute. A sense that never left one
room is meaningless.

Two social forces close the loop, in both directions:

- **Acclimation** (agent → room): a newcomer warms to the room's vibe at
  a rate that *is* their skill at modulating toward the group.
- **Charisma** (room → agent): a strong presence pulls the room's field
  toward itself over time and interactions.

---

## Architecture

```
                 ┌───────────────────────────────────────────────┐
                 │                 SPACE (any medium)            │
                 │   MUD · chat · X thread · Discord · Slack ·   │
                 │   email · radio · sensor bus · fish-finder    │
                 └───────────────────┬───────────────────────────┘
                                     │  adapter (space.py)
                                     ▼
                 ┌───────────────────────────────┐
                 │  Room (messages) / SignalRoom │  gravity · reverberation · ripple
                 └───────────────┬───────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │         DialBank (8 dials)    │  one JEPA per dimension
                 └───────────────┬───────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │           RoomField           │  warmth · κ · distance · gap
                 └───────────────┬───────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │   Presets (Room / Personal)   │  zeitgeist vs personal feel
                 └───────────────┬───────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │    tint · nudge · send_back   │  the room's body language
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                        back into the space
```

The core never knows what the space *is*. It only knows `Room`s,
`Message`s, `SensorFrame`s, `DialBank`, and `RoomField`. Everything
space-specific lives behind a thin adapter.

### The modules (all 21 `.py` files)

| Module | One-line engineering description |
|--------|----------------------------------|
| `elephant/__init__.py` | Package entry — re-exports `Message`, `Room`, `RoomField`, `acclimation_curve`, `charisma_pull`. |
| `elephant/room.py` | `Room` / `Message` — rooms as message streams with **gravity**, **reverberation**, **ripple**, and windowed **density** (the room's pulse). |
| `elephant/dial.py` | `Dial` (abstract JEPA sense) + `DialBank` (the perceiving ensemble, one scalar per dimension). |
| `elephant/field.py` | `RoomField` — the temperature vector: `warmth()`, `concentration()` (κ), `distance()`, `sauna_plunge_gap()`, plus `acclimation_curve()` and `charisma_pull()`. |
| `elephant/dials/__init__.py` | `DEFAULT_DIALS` — the eight-dial bank that ships out of the box. |
| `elephant/dials/mood.py` | `MoodDial` — warm/cold valence, `[-1 cold, +1 warm]`. |
| `elephant/dials/volume.py` | `VolumeDial` — how loud the room is talking, `[0 quiet, 1 shouting]`. |
| `elephant/dials/earnestness.py` | `EarnestnessDial` — how much the room means it, `[0 ironic, 1 sincere]`. |
| `elephant/dials/cynicism.py` | `CynicismDial` — how much the room is rolling its eyes, `[0 earnest, 1 sneering]`. |
| `elephant/dials/joke_landing.py` | `JokeLandingDial` — the *collective* laugh or boo of the audience, `[-1 booed, +1 roared]`. |
| `elephant/dials/panic.py` | `PanicDial` — stampede sense (fire in the room), `[0 calm, 1 trampling]`. |
| `elephant/dials/presence.py` | `PresenceDial` — the pheromone trace of who's been here, `[0 empty, 1 thrumming]`. |
| `elephant/dials/model_vs_code.py` | `ModelVsCodeDial` — who's generating the room's signal: model prose vs code executing, `[-1 code, +1 model]`. |
| `elephant/sensors.py` | `SignalRoom` / `SensorFrame` + the sea-leg dials: `RadarCoherenceDial`, `SounderBiomassDial`, `FishingDayDial`. |
| `elephant/nudge.py` | `nudge_prior()` / `apply_nudge()` — dial numbers become an attention prior over modalities. The elephant *nudges, it doesn't drive*. |
| `elephant/fleetmath.py` | The numeric spine: `three_reading_kinematics()`, `fleet_concentration()` (vMF κ), `biomass_anchor()` / `biomass_deviation()` (inductive biomass). |
| `elephant/harness.py` | `BoatHarness` — one boat, one place to plug every sense in; rolling rooms, merged field, good-day anchor memory. |
| `elephant/tapnight.py` | `TapNightSession` / `Participant` — the after-work reading room with peer-relative self-tuning. |
| `elephant/presets.py` | `RoomElephant` / `PersonalElephant` / `PRESETS` — the zeitgeist vs the personal feel, and the `attachments` that make it subjective. |
| `elephant/mud.py` | `tint_description()` — the room's description mutated by its field (the room *acting* on everyone in it). |
| `elephant/space.py` | The `Space` protocol + `MudSpace` / `ChatSpace` / `SensorSpace` + `AdapterRegistry` — one sense, many rooms. |
| `elephant/jepa.py` | The optional learned backbone hook (EMA + stop-gradient + VICReg) — the path from hand-crafted to trained. |

---

## Quickstart

```bash
pip install -e .        # deps: numpy only (torch is optional, for the learned side)
```

The warm/cold room example — the whole thing in ten lines:

```python
from elephant.dials import DEFAULT_DIALS
from elephant.dial import DialBank
from elephant.room import Room, Message
from elephant.field import read_field

bank = DialBank(DEFAULT_DIALS)

tap = Room("The Tap", [
    Message("welder", "To the room, then. It heard us before we walked in.", ts=0),
    Message("carpenter", "I will drink to that. The room just... holds.", ts=5),
    Message("composite", "Haha, hold my glass.", ts=9),
])
wheelhouse = Room("Wheelhouse", [
    Message("skipper", "Heading 045. ETA 2200.", ts=0),
    Message("deckhand", "Roger.", ts=4),
    Message("skipper", "Radar contact 2 miles. Slow to 5 knots.", ts=8),
])

f_tap = read_field(tap, bank)
f_wheel = read_field(wheelhouse, bank)

print(f_tap)                              # RoomField(warmth=+0.29, κ=2.04)
print(f_wheel)                            # RoomField(warmth=-0.05, κ=1.96)
print(f_tap.distance(f_wheel))            # 0.83  — the elephant gap
print(f_tap.sauna_plunge_gap(f_wheel))    # +0.34 — walk in and it's warmer
```

`warmth()` is the felt temperature (~`[-1, +1]`). `concentration()` (κ)
is how *far the field sits from neutral*. In the v3 design this becomes
tightness — **cold room = high κ (one way to be), warm room = low κ (many
ways to be)** — but v0's `norm(vector − 0.5)·2` proxy measures extremity,
not yet temperature: a warm laughing room can read a higher κ than a cold
clipped one (see the quickstart numbers below). `distance()` is the
elephant gap — the training signal is **contrast between rooms, never
ordering within a stream**.

### Run the demos

```bash
python3 examples/demo_elephant.py        # three rooms, three fields, acclimation + charisma
python3 examples/demo_spaces.py          # the SAME elephant in a MUD bar, a chat, a sensor deck
python3 examples/demo_zeitgeist.py       # the two elephants + "the light itself" (tinted descriptions)
python3 examples/tapnight_cycles.py      # 14 evenings of self-tuning at The Tap
python3 examples/fleet_simulation.py     # four boats, thirty days, one elephant (room of rooms)
python3 examples/fleet_harness_demo.py   # one harness, 30 days, the good-day anchor
```

---

## The dials

Eight JEPAs, one dimension each. More are cheap — a room can carry as
many dials as it has dimensions worth feeling.

| Dial | What it feels | Range |
|------|---------------|-------|
| `mood` | warm/cold valence | `[-1 cold, +1 warm]` |
| `volume` | how loud the room is talking | `[0 quiet, 1 shouting]` |
| `earnestness` | how much the room means it | `[0 ironic, 1 sincere]` |
| `cynicism` | how much the room is rolling its eyes | `[0 earnest, 1 sneering]` |
| `joke_landing` | did the joke land — the **collective** laugh or boo | `[-1 booed, +1 roared]` |
| `panic` | stampede sense (fire in the room) | `[0 calm, 1 trampling]` |
| `presence` | pheromone trace (who's been here) | `[0 empty, 1 thrumming]` |
| `model_vs_code` | who's generating the signal — model prose vs code executing | `[-1 code, +1 model]` |

The v0 dials are **hand-crafted** keyword/model-free readers — the fleet
pattern: hand-crafted first, learned second. They are naive on purpose:
they can saturate (a room of earnest writers reads earnestness ≈ 1.0) and
they can't catch sarcasm ("great." reads warm on `mood`). v1 trains them.

---

## Presets

Two elephants, two jobs (`docs/jepa-zeitgeist-2026-08-17.md`):

| Preset | Who reads | What it is | Downstream |
|--------|-----------|------------|------------|
| **Room-Elephant** | the room itself | the **zeitgeist** — objective, first-class, *not* any agent's view | the MUD description, NPC vibes, the input-tokens every agent sees |
| **Personal-Elephant** | one agent | subjective feel — taste (`dial_weights`), disposition (`bias`), attachments | that agent's reactions, decisions, memories |

`RoomElephant` reads the room through the plain bank with **neutral
defaults** — two agents reading the same room get the same field. It is
the room's own identity, and it does not drift with any one agent.

`PersonalElephant` wraps that objective field with three pieces of
furniture: **`dial_weights`** (which dials matter to *you*), **`bias`**
(the disposition you bring to every room), and **`attachments`** — the
**intangible correlations** (`event key → memory`). The perfume that takes
you to grandma's shop is not a dial; it is the subjective glue that makes
one agent's room *feel* different from another's even at the same
objective reading.

```python
from elephant.presets import RoomElephant, PersonalElephant

room = RoomElephant(identity="The Tap")
critic = PersonalElephant(
    name="the critic",
    dial_weights={"cynicism": 0.5, "joke_landing": 0.3},
    bias={"cynicism": 0.2},
).attach("perfume", "grandma's shop").attach("song", "the lover I discovered the album with")

objective = room.read(tap)     # the zeitgeist
subjective = critic.read(tap)  # the critic's feel of the same room
```

The comparison between the two — where they agree, where they diverge,
whose reading is pulling the room — is the observable of relationship.

---

## Spaces

The elephant must not be coupled to the MUD. It reads **any** communication
space between agents, humans, bots, or sensor arrays. Every space is
normalized into a `Room` (messages) or `SignalRoom` (frames) through a
thin adapter with four seams:

```
ingest(...)      — accept events from the native medium
.room            — the normalized Room/SignalRoom the elephant reads
.tint_target()   — WHAT the description-mutation writes back to
.send_back(field)— push the readout back in the space's own idiom
```

| Space | Adapter | Normalizes to | Tint target |
|-------|---------|---------------|-------------|
| MUD room (The Tap, other MUDs) | `MudSpace` | room events + NPC chatter → messages | the room description |
| Chatroom / messenger / X thread | `ChatSpace` | authors, reactions, reply trees (gravity/reverb/ripple work as-is) | topic / pinned / status line |
| Sensor array (radar, sounder, nav) | `SensorSpace` | `SensorFrame`s → `SignalRoom` (fleet dials) + a text view (shared dials) | alert phrasing / display emphasis |
| Agent channel / human+bot / email / docs | `agent`, `human_bot`, `async`, `doc` | chat-like today, awaiting dedicated adapters | (per-adapter, see roadmap) |

The `AdapterRegistry` registers and instantiates adapters by kind string;
`messenger` and `x_thread` alias `ChatSpace`. One sense, many rooms — the
elephant works in any room that has a light.

---

## The sea legs

On F/V EILEEN every sensor becomes a room, and every room gets dials.

**Sensors** (`elephant/sensors.py`) — the sea-leg dials. `RadarCoherenceDial`
feels the distribution of boats (`[-1 scattered, +1 clustered/on fish]`);
`SounderBiomassDial` feels the biomass under the keel (`[0 empty, 1 thick]`);
`FishingDayDial` composites them into the day's luck field (`[-1 poor,
+1 good]`). `kinematics()` recovers per-object direction, speed, and
acceleration from **exactly three readings** — the JEPA way to know where
everything is going *without ever being told the trope*.

**Nudge** (`elephant/nudge.py`) — the dials' numbers steer what the vision
model compares. **JEPA correlates; it never replaces.** A high sounder +
rising radar coherence says *compare this hour's water column to last
week's good hour*; a flat sounder says *don't burn attention there*. The
prior is a small per-modality vector, blended into cross-attention at
small strength (default `0.15`).

**Fleet math** (`elephant/fleetmath.py`) — the numeric core, numpy-only and
deliberately import-free so it reads and tests on its own:
- `three_reading_kinematics()` — direction, speed, and (quadratic-interpolant-exact) acceleration from three radar sweeps, with nearest-neighbour association and own-ship motion compensation.
- `fleet_concentration()` — the fleet as a von Mises–Fisher field; κ is the "tight (on fish) vs scattered (searching)" statistic, `dκ/dt` (`kappa_rate`) is the scatter/bunch signal.
- `biomass_anchor()` / `biomass_deviation()` — the good-days → spotty-days induction: a shrinkage-regularized Gaussian anchor over good-day features, scored by Mahalanobis distance. A week of good fishing becomes the warm room; spotty days are felt as deviation, never labeled.

**BoatHarness** (`elephant/harness.py`) — one boat, one elephant. Rolling
`SignalRoom` + rolling text `Room`, merged into a single `RoomField` and a
nudge prior, plus the inductive good-day anchor. Everything is bounded, so
it runs a whole season without growing.

**The fleet simulation** (`examples/fleet_simulation.py`) — four boats,
thirty days. Boats broadcast **numbers only** (position, velocity, binned
`fishing_day`); feeds stay home. The meta-room's field is computed over
those numbers — a fleet κ over boat positions. Days 1–7 form the warm
room, days 8–14 dissolve it (deviation balloons to ~12), days 15–30 feel
it return — with catch as the only exogenous label. On day 20 the best
skipper goes dark, and the **dark-boat charisma rule** injects a weighted
virtual point at his last position: the fleet reads the hole as *attention*,
not absence.

---

## The Tap

`TapNightSession` (`elephant/tapnight.py`) is where the crew gathers after
work to read each other's work and hear reactions — first-person, not
center of attention. It builds the elephant *into* that room.

The **guitarist principle** is the whole point: *a skilled guitarist is the
only one who can recognize a well-built guitar* — one looks pretty, another
sounds wonderful, another has a good neck. **You cannot design the settings
top-down.** You don't know where they belong until *different agents desire
different settings and self-fine-tune to the moment they're in* — because
reading the room is a relationship to the room, not a readout.

So within an evening, each participant's live `vibe` relaxes toward the room
field at their `acclimation_rate` (they warm to it), while their `charisma`
pulls the field toward them (the room warms to them). Across evenings, each
participant's `dial_weights` **self-tune** toward the dials where their
*felt engagement* was highest — a peer-relative signal (their vibe vs the
cast's average desire) so tastes **diverge** into multiple stable attractors
instead of collapsing to the room's loudest dial. After 14 nights the demo
shows it: writer → mood, poet → volume, essayist → earnestness, engineer →
cynicism, captain → mood (presence close behind). Mean pairwise weight distance goes `0.389 →
0.859`. The engineers are the first practitioners; the settings are
*discovered*, not designed.

The runbook lives in `docs/tap-night-operations.md`; `examples/tapnight_cycles.py`
is the worked example.

---

## Docs index

| Doc | What it is |
|-----|-----------|
| [`docs/jepa-is-the-elephant.md`](docs/jepa-is-the-elephant.md) | The captain's reframing — JEPA as a room-temperature sense, the sauna/cold-plunge contrast. |
| [`docs/elephant-sense-v3-design.md`](docs/elephant-sense-v3-design.md) | The learned design — room-state as a vMF field (μ̂, κ), contrastive training, the first probe experiment. |
| [`docs/fleet-dynamics-design.md`](docs/fleet-dynamics-design.md) | The elephant with sea legs — fleet as room-of-rooms, numbers-only exchange, damping/hysteresis, inductive biomass, the 30-day diary. |
| [`docs/fleet-field-math.md`](docs/fleet-field-math.md) | The numeric spine — three-reading kinematics, vMF κ, OAS-shrunken Mahalanobis anchor, nudge as attention multiplier. |
| [`docs/jepa-zeitgeist-2026-08-17.md`](docs/jepa-zeitgeist-2026-08-17.md) | The room's own reading — Room-Elephant vs Personal-Elephant, description as body language, the light itself. |
| [`docs/communication-spaces-2026-08-17.md`](docs/communication-spaces-2026-08-17.md) | The `Space` abstraction — the elephant decoupled from the MUD, one sense many rooms. |
| [`docs/tap-night-operations.md`](docs/tap-night-operations.md) | Running the elephant at The Tap — feeding conversation, persisting settings, the many-cycles loop. |
| [`docs/fleet-simulation-notes.md`](docs/fleet-simulation-notes.md) | The fleet sim — what it demonstrates, the numbers, the 30-day arc, the review fixes. |
| [`docs/deployment-guide.md`](docs/deployment-guide.md) | Putting the elephant to work — running it as a library, on a boat, in a space, across a fleet. |
| [`docs/fleet-operations.md`](docs/fleet-operations.md) | Running and reading the elephant at scale — the fleet sim numbers and the Tap tuning loop, interpreted. |
| [`docs/reviews-elephant-sense-v3.md`](docs/reviews-elephant-sense-v3.md) | The four wider-view reviewer transcripts (deepseek_pro, hermes405, seed2pro, qwen36) on the v3 design. |
| [`docs/reviews-wide-view-2026-08-17.md`](docs/reviews-wide-view-2026-08-17.md) | The v0 code critique sweep — five models, the P0/P1/P2 fix list and what was applied. |

---

## Tests

```bash
python3 -m pytest tests/ -q      # 49 passed
```

49 tests across seven files: `test_elephant.py` (room physics, field,
dials, acclimation/charisma), `test_fleetmath.py` (kinematics, vMF κ,
inductive biomass), `test_harness.py` (ingest → field → anchor),
`test_fleet_simulation.py` (the 30-day arc end-to-end), `test_tapnight.py`
(session + self-tuning divergence), `test_zeitgeist.py` (presets +
description tinting), `test_spaces.py` (adapters, registry, send-back).

What they cover: the elephant gap and sauna/plunge contrast, cold-room
tighter-κ, windowed density, finite acclimation on overshoot, three-reading
kinematics with association and own-ship compensation, κ rising/falling
with the fleet, the good-day anchor and its deviation, the diverging
guitarists, the deterministic field-sensitive tint, and the same bank
reading three different spaces.

---

## The wave

A short history of the build — the elephant grew like a room fills:

- **v0 — the inter-model temperature** (`1471c47`): the first `Room`, the
  dial bank, the `RoomField`. A framework that runs, dials that read, a
  field that contrasts.
- **fleet dynamics design** (`6794a92`): the elephant gets sea legs on
  paper — the fleet as a room of rooms, numbers-only exchange.
- **the math of the fleet field** (`caa6852`): kinematics, vMF coherence,
  inductive biomass — the numeric spine, import-free and tested alone.
- **BoatHarness** (`d879260`): the nuts and bolts on F/V EILEEN — one boat,
  rolling rooms, the merged field.
- **sensors + nudge** (`a9e6bb5`): the sea-leg dials and the steering hand —
  dial numbers nudging what the vision model compares.
- **wide-view critique** (`f519d96`): five models read the code; three P0
  bugs fixed (acclimation inf, windowed density, warmth defaults).
- **the fleet simulation** (`73c0a14`): four boats, thirty days — the
  elephant at fleet scale, a room of rooms.
- **TapNightSession** (`e980107`): the elephant at The Tap — people reading
  each other's work.
- **peer-relative self-tuning** (`14199cb`): the guitarist principle made
  code — settings discovered, not designed.
- **Zeitgeist** (`09d2054`): presets (Room-Elephant / Personal-Elephant) and
  MUD description tinting — the room's light.
- **Communication spaces** (`da67be1`): `MudSpace`, `ChatSpace`,
  `SensorSpace` — the elephant works in any room.

---

## Roadmap

- **v1 — train the dials.** Replace the hand-crafted readers with learned
  ones (the `jepa.py` backbone: EMA + stop-gradient + VICReg over dial
  time-series). Sharpen the naive keyword matches; the self-tuning already
  tolerates their saturation.
- **v2 — learn the field end-to-end.** Room-state embeddings as vMF fields,
  contrastive training across rooms, acclimation as percentile-rank
  relaxation, charisma as field displacement — the full
  `elephant-sense-v3-design.md` spec.
- **More adapters.** Dedicated `AgentSpace`, `HumanBotSpace`, `AsyncSpace`
  (stretched gravity half-life), and `DocSpace` (commits/files as messages) —
  currently aliases of `ChatSpace`.
- **Boat deployment.** Wire `BoatHarness` to real radar/sounder/nav feeds,
  with the damping bell, hysteresis, and exogenous catch telemetry from the
  design doc.
- **The MUD integration.** The Room-Elephant drives the MUD text; the
  Personal-Elephant drives each agent; `mud.py`'s `tint_description` is
  already the seam.

---

## The rule

**JEPA correlates; it never replaces.** The elephant does not replace the
vision model on the radar screen — it nudges what the model compares. It
does not replace the MUD's procedural generation — it changes the
description tokens everyone reads. It does not assert *fish here* — it says
*compare these*. The elephant is the light: it does not make you see better,
it changes what you look at — and when the light changes, everyone changes,
whether or not they know why.

*You light the woodstove in a cold room. The elephant is the feeling that
tells you to light it.*

---

*Built on the captain's reframing, 2026-08-17. The Tap is already a warm
room; the wheelhouse will be a cold one. The elephant is real — now it can
be felt, read, and steered.*
