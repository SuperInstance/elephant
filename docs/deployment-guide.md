# Deployment Guide — putting the elephant to work

This is the operations side of the elephant: how to run it as a modular
tool in your own system, on a fishing boat, inside a chat/MUD/agent bar,
and across a whole fleet. It is grounded in the code as it exists today
(`elephant/` v0, numpy-only) — every number below was produced by running
the shipped examples, not by reading the design docs and hoping.

Companion reading:

- `docs/fleet-dynamics-design.md` — why the fleet is a room of rooms, and
  why boats share numbers, not feeds.
- `docs/fleet-field-math.md` — the vMF κ and inductive-biomass mathematics.
- `docs/fleet-simulation-notes.md` — what the sim demonstrates.
- `docs/tap-night-operations.md` — the Tap practice-room runbook.
- `docs/communication-spaces-2026-08-17.md` — the Space abstraction.

---

## 1. Standalone harness — the elephant as a library

The elephant is not coupled to any one system. The core is `room.py`,
`dial.py`, `dials/`, `field.py`, `nudge.py`; everything else is an adapter.
The fastest way in is a bare `Room` + the default dial bank:

```python
from elephant.dial import DialBank
from elephant.dials import DEFAULT_DIALS
from elephant.room import Room, Message
from elephant.field import read_field

bank = DialBank(DEFAULT_DIALS)
room = Room("The Tap", [
    Message("welder", "To the room, then. It heard us before we walked in.", ts=0),
    Message("carpenter", "I'll drink to that. The room just... holds.", ts=5),
    Message("comic", "Haha, hold my glass. 😂", ts=9),
])
field = read_field(room, bank)          # RoomField(warmth=+0.29, κ=2.04)
print(field.warmth(), field.concentration())
```

That is the whole contract: **build a Room, read the field.** From the
field you get `warmth()` (~[-1,+1]), `concentration()` (κ, the tightness),
`distance(other)` (the elephant gap between rooms), and
`sauna_plunge_gap(other)` (the signed warmth contrast you feel on entry).

When you need sensors (a boat, a deck, any array), swap the bare `Room`
for a `BoatHarness`:

```python
from elephant.harness import BoatHarness

h = BoatHarness(name="EILEEN")           # rolling rooms, bounded by default
h.ingest_radar([(0, 0), (1, 1), (0.5, 0.5)])   # list of (x, y) km
h.ingest_sounder(0.72)                    # biomass in [0, 1]
h.ingest_nav(45.0, 2.5)                   # heading deg, speed kts
h.ingest_conversation("skipper", "We're on them. Good mark.")
print(h.readings())                       # merged dial dict
print(h.current_nudge())                  # 7-vector attention prior
```

`BoatHarness` constructor knobs: `max_signal_frames=256`,
`max_messages=400`, `step=3600.0` (auto-timestamp increment in seconds),
`nav_speed_ref=10.0` (knots that saturate the nav feature at 1.0). The
rolling rooms are trimmed after every ingest, so the harness can run a
whole season without growing.

---

## 2. On a boat — F/V EILEEN

### Wiring sensors → SignalRoom frames

Every sensor becomes a frame in one `SignalRoom` (`harness.signal`); the
fleet dials read that room. The ingest helpers map one-to-one onto the
physical instruments:

| Instrument | Helper | `data` shape | Fleet dial it feeds |
|------------|--------|--------------|---------------------|
| Radar | `ingest_radar(targets)` | list of `(x, y)` km (boat-relative) | `radar_coherence` |
| Sounder | `ingest_sounder(biomass)` | float `[0, 1]` | `sounder_biomass` |
| Nav / AP | `ingest_nav(heading, speed)` | `{"heading", "speed"}` | nav feature (day vector) |
| Cameras | `ingest_camera(meta)` | `data=None`, detail in `meta` | vision model's room |
| Crew talk | `ingest_conversation(author, text)` | a `Message` | the 7 vibe dials |

Three composite readings fall out of the fleet dials (see
`elephant/sensors.py`):

- `fleet_kappa()` → `radar_coherence`, **[-1 scattered .. +1 clustered]**
  (`1 − clip(mean_spread/4)` plus a ±0.30 closing/scattering trend term).
- `biomass()` → `sounder_biomass`, **[0 empty .. 1 thick]** (mean of last 5,
  plus a ±0.20 trend).
- `fishing_day()` → `0.55·radar + 0.45·(2·biomass − 1)`, clipped to
  **[-1 poor .. +1 good]** — the day's luck field.

Radar also carries kinematics: `radar_kinematics()` recovers per-object
direction, speed, and acceleration from the last three sweeps by
nearest-neighbour association (gate 2 km). It returns
`{"objects", "fleet_mean_speed", "spread_rate"}` where `spread_rate`
positive = scattering, negative = bunching.

### The LOCAL-ONLY rule

> **Conversation JEPA stays on the boat. Only numbers leave.**

The crew's raw conversation (`harness.conversation`) is read by the vibe
dials *on the boat*. What crosses the wire to the fleet is the distilled
scalar — `fishing_day` binned to `{-1, 0, +1}`, position, velocity. Never
the feed. This is the load-bearing constraint of the whole design: enough
for the meta-room's temperature, not enough to fingerprint. Treat any
`ingest_conversation` text as private; the number is public, the words are
not.

### The nudge loop — dial numbers → attention prior → vision comparison

The elephant does not replace the vision model on the radar/sounder screen;
it **correlates**. Each dial reading nudges the vision model about *what to
compare together* (`elephant/nudge.py`):

```
readings()  ──►  nudge_prior(readings, modalities)  ──►  7-vector in [-1,+1]
                    │
                    └── apply_nudge(attention, prior, strength=0.15)
```

The signed map (`NUDGE_MAP`): `radar_coherence → radar (+1)`,
`sounder_biomass → sounder (+1)`, `fishing_day → nav (+0.5)`,
`mood → conversation (+1)`, `volume → camera_deck (+0.5)`,
`panic → camera_out (+1)`, `presence → camera_deck (+0.3)`.

A high sounder biomass plus rising radar coherence therefore pushes the
prior toward `sounder` and `radar` — "look at the water column under the
cluster, compare this hour to last week's good hour." A flat sounder pulls
those toward zero: don't burn attention there. `apply_nudge` blends the
prior into existing cross-attention at a small strength (default 0.15) —
the elephant nudges, it doesn't drive.

### The inductive biomass anchor

The harness remembers what a good day *felt like* without being told the
trope (`harness.py` → `day_features`, `day_memory`, `inductive_signal`):

- `day_features()` → the 3-vector `[fleet_kappa, biomass, nav]` where nav is
  mean nav speed divided by `nav_speed_ref`, clipped to [0, 1].
- `day_memory(good_day_threshold=0.2)` → when `fishing_day() >= 0.2`, store
  today's features as a good-day anchor (append to `good_days`); else `None`.
- `inductive_signal()` → `{"total", "radar", "biomass", "nav", "n_anchor_days",
  "anchor"}`: `total` is the L2 norm of `(features − mean(anchor))`, the
  per-channel entries are the absolute deviations.

**A week of good fishing = the anchor; spotty days = felt deviation.** In the
single-boat demo (`examples/fleet_harness_demo.py`, seed 7) the good hour
sits `total=0.032` from its anchor (biomass channel `0.031`), the spotty
hour sits `total=1.314` (biomass `0.568`) — a >40× jump in the biomass
channel, with **no** label saying "fishing got bad." The harness feels the
difference from the shape of the field alone.

---

## 3. In a chat / MUD / agent bar — the Space adapters

The elephant works in any communication space. `elephant/space.py` normalizes
every medium into the one thing the core already reads (a `Room` or
`SignalRoom`) through a thin adapter, then writes the elephant's readout
back in the space's own idiom.

### The built-in adapters + registry

| `kind` | Adapter | Normalizes to | Tint target |
|--------|---------|---------------|-------------|
| `mud` | `MudSpace` | messages (room events + NPC chatter) | the room description |
| `chat` | `ChatSpace` | messages with authors, reactions, reply trees | the channel topic / status line |
| `sensor` | `SensorSpace` | `SensorFrame`s (radar/sounder/nav/camera) **+** a text rendering | sensor alert phrasing |

Aliases (`messenger`, `x_thread`, `agent`, `human_bot`, `async`, `doc`) are
registered to `ChatSpace` today until dedicated adapters land. The registry
is a class with `register` / `get` / `kinds` / `has`:

```python
from elephant.space import AdapterRegistry, MudSpace
AdapterRegistry.register("mud", MudSpace)          # or as a @register decorator
tap = AdapterRegistry.get("mud", "The Tap")        # MudSpace("The Tap")
```

### The 4-method contract

A new adapter subclasses `Space` and provides four abstract members
(`space.py`):

1. `ingest(self, *events)` — accept events from the native space, return
   `self`.
2. `room` (property) — the normalized `Room` or `SignalRoom` the elephant
   reads (same timestamped sequence).
3. `tint_target(self)` — a *string naming* what the elephant writes back to
   (e.g. `"the room description"`).
4. `tint(self, field)` — transform a `RoomField` into the space's own idiom
   (the tinted description/topic/alert).

`read(bank)` and `send_back(field, tinted_text=None)` are already
implemented on the base class: `read` runs a dial bank over `.room`;
`send_back` calls `tint`, stashes it in `_last_tint`, and returns the text.
Override `send_back` only when the tint must be *applied* to a live object
(see `MudSpace.send_back`, which sets `self.description`).

Minimal new adapter (e.g. a Slack channel):

```python
from elephant.space import Space, ChatSpace

class SlackSpace(Space):
    kind = "slack"
    def __init__(self, name):
        super().__init__(name)
        self._room = Room(name)   # reuse ChatSpace internals in practice
    def ingest(self, *events):
        ...                       # coerce Slack events -> Messages
        return self
    @property
    def room(self):
        return self._room
    def tint_target(self):
        return "the channel topic"
    def tint(self, field):
        return f"{'✨' if field.warmth() >= 0.25 else '❄'} {self.name} — warmth {field.warmth():+.2f}"
```

### How the tint writes back

`send_back(field)` is the elephant acting on everyone in the room, not a
report. Three shipped behaviors:

- **MudSpace** — the room description. `tint` prefers
  `elephant/mud.py::tint_description(field, base_text, hour)` (the zeitgeist
  build): it classifies the field as `panic / joyful / closing / neutral`
  and weaves template weather + light + adjective banks into the description,
  deterministically seeded from the field. Same field → same words; a changed
  field → a changed room. The demo reads a warm bar as warmth **+0.45, κ 1.91**
  and the description opens *"A soft night, the harbor still and silver … the
  place feels ringing … laughter reverberates into the words."*
- **ChatSpace** — the topic line, e.g. the heated thread renders
  **`❄ crew-thread — cold — the room has gone flat and sharp (warmth -0.62, κ 3.68, 3 msgs)`**.
- **SensorSpace** — the alert line. With `radar_coherence=+0.92` and
  `sounder_biomass=+0.74`, it writes **`🟢 F/V EILEEN: fleet tight (κ +0.92) + biomass 0.74 — on fish, hold the drag`**.

`SensorSpace` is the two-room case worth internalizing: `.signal` holds the
raw `SensorFrame`s (read by the fleet dials), while `.room` holds a *text
rendering* of those frames (`_render_frame`) plus any crew chatter — so the
seven shared vibe dials can feel the same array. `full_read()` merges the
two, shared dials winning ties.

---

## 4. As a fleet — the meta-room

`examples/fleet_simulation.py` is the worked proof. Each boat runs a
`BoatHarness`; the fleet itself is one meta-room whose field is computed
over the boats' **broadcast numbers only** — position, velocity,
`radar_coherence`, and `fishing_day` binned to `{-1, 0, +1}`. Feeds stay
home.

The fleet field (`fleet_simulation.py`):

- `fleet_centroid` — the **weighted median**, x and y separately (one crazy
  boat far out does not drag the middle of the room).
- `fleet_spread_km` — weighted mean radial distance from that centroid.
- `fleet_kappa` — `1 / (1 + spread_km)`, in **[0, 1]**: bunched on the drag →
  ~0.9, scattered over the horizon → ~0.05.
- `meta_room_warmth` — `0.5·fishing_day + 0.3·(2·biomass − 1) +
  0.2·radar_coherence`, clipped to [-1, +1].

### Driving behavior — hold the drag, or scatter

The field does not command the autopilot; it shapes it the way room
temperature shapes conversation (design §4):

- **Field says fish** (κ high, sounder thick, luck positive) → **hold the
  drag**. The nudge says compare the column under the cluster to yesterday's
  good hour; the boat works the same water.
- **Field says searching** (κ low and falling, sounder thin) → **scatter**.
  The room is cold and *dissolving*; the boat moves, searches, re-prospects.

The loop is **damped** to keep the herd from running away (the
`_damping_bell`, design §4.2): `effective_kappa = κ·(1 − clip(κ, 0.2, 0.8))`.
A perfectly bunched fleet (κ → 1) or a fully scattered one (κ → 0) carries
less usable signal than the bell's middle; clustered fleets get a weak probe
nudge at the top, scattered fleets a weak hold at the bottom, and the real
signal lives in the untouched middle. In the sim the raw κ of the good week
is **0.63** but the effective κ is **0.22** — the bell is doing its job.

### The dark-boat charisma rule

If a **high-reputation boat goes dark** (stops broadcasting), the fleet
must not read it as thinning. The rule (design §2.5, flagged in the sim)
injects a **virtual point at the boat's last observed position, weighted
3× its reputation**, so the fleet holds attention *toward the hole* — the
charismatic boat went quiet because it's on fish, not because it left. In
the seeded run, `EILEEN` (reputation 1.7, the best skipper) goes dark on
day 20 at `(12.3, 8.3)` km; the remaining boats' recovery target becomes
that position and they re-group on **her** mark, not the old ground point.

---

## 5. Ops checklist

- **Bounded windows.** Both rolling rooms are trimmed to
  `max_signal_frames=256` / `max_messages=400` after every ingest (oldest
  dropped, lists kept sorted by `ts`). Memory is O(chunk), never O(season).
  Do not raise these without a reason — a season of frames must not grow
  unbounded.
- **Deterministic seeds.** Every demo draws from one
  `np.random.default_rng(seed)` (sim seed 7, Tap seed 42), consumed in a fixed
  order. Reproduce the numbers in this guide with the same seed; change the
  seed and the arc changes. Seed your own runs if you want reproducible
  operations.
- **The anchor must have an outside.** Do not build the inductive anchor
  from `fishing_day` — it is itself a composite of the dials, and that would
  be self-confirming. The sim uses **exogenous catch** (landed fish) as the
  ground-truth third feature. Your production anchor needs its own outside
  (a logged catch, a scale weight, a buyer's receipt) or the deviation will
  measure shared noise instead of fleet drift.
- **`pull --rebase` discipline for parallel agents.** Several agents write
  docs against this repo. Always `git pull --rebase origin main` before
  committing; on conflict, resolve and re-run the rebase. Add only your own
  files — do not sweep up other agents' untracked work.
- **Run the test suite.** `python3 -m pytest tests/ -q` — **49 passed in
  0.42s** as of this writing (`test_elephant`, `test_fleetmath`,
  `test_fleet_simulation`, `test_harness`, `test_spaces`, `test_tapnight`,
  `test_zeitgeist`). Don't merge a deployment doc whose examples you haven't
  run against the shipped code.
