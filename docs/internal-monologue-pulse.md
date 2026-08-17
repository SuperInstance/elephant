# The Internal Monologue Pulse — agents sense on a constant heartbeat

*2026-08-17 · the captain's directive, made code.*

---

## The captain's insight (the design core)

> Agents using internal monologues on constant pulses even if they aren't
> talking. These internal monologues take a Perception check as part of
> their looking around and thinking. They look at the table's conversation
> as a whole hand and see JEPA perceptions — like macro-economic currency
> exchange changes, where the number doesn't matter but TWO numbers show
> DIRECTION and MORE THAN TWO show RATE OF CHANGE.

An agent is **always sensing, even when silent**. The pulse is the
heartbeat of that sensing. Each pulse reads the room's field and, from the
*sequence* of readings, extracts the macro sense — the way a trader reads a
currency pair. The number alone is nothing; the movement is the perception.

This is the elephant's `pulse.py`: the internal monologue as a constant
heartbeat, with perception checks as the agent's looking-around.

## What a pulse is

`PulseLoop` gives an agent a heartbeat. It ticks on an interval
(`.tick(now)` or `.pulse()`), **even when the agent isn't speaking** — the
silence is not empty. Each tick:

1. **Reads the room** — the dial bank over the room's current state (the
   same `DialBank` the room field uses; a `Space` adapter's own bank, a
   `TapNightSession`'s bank, or the default eight dials).
2. **Records the reading** into a rolling history (bounded by `history`).
3. **Runs the perception check** — the macro read of the room over the
   pulse history, returned as a `PerceptionReport`.
4. **Thinks, silently** — `internal_monologue()` is 1-3 sentences of what
   the agent is noticing *without speaking*. This is the part that runs
   even when the agent says nothing in the room.

The loop tracks whether the agent itself has spoken since the last pulse
(`agent_said`) and how many messages crossed the table (`traffic`) — so a
pulse knows it is a *silent* one, and the monologue runs anyway.

## The perception check — looking around, reading the whole hand

`perception_check()` is the agent's looking-around: it reads the table's
conversation **as a whole hand**, not as any single message. The report
carries the macro read:

| Field | What it is | From |
|-------|-----------|------|
| `warmth_direction` | is the room warming or cooling? the headline | last TWO field vectors |
| `warmth_rate` | is that movement accelerating or easing? | last THREE+ (second difference) |
| `direction` | per-dial movement (mood +0.31/pulse, cynicism −0.05/pulse...) | last two, per dial |
| `rate_of_change` | per-dial acceleration (the same read, second difference) | last three+, per dial |
| `dial_deltas` | per-dial `{"direction": ..., "rate": ...}` pairs | both of the above |
| `whole_hand` | the table's conversation read AS A WHOLE — the macro in words | the whole report |
| `traffic` / `agent_said` | did the table move since my last pulse? did I speak? | the room, this tick |

`whole_hand` is the key observable: not "welder said X" but *"the table is
warming, and the movement is accelerating — mood rising 0.96/pulse,
joke_landing rising 1.00/pulse; 1 new message crossed the table."* The
macro, not the message.

## The math — two numbers show direction, three+ show rate of change

Both functions are standalone and numpy-only, in the fleetmath tradition
(they are `three_reading_kinematics` generalized from radar objects to any
dial series):

```python
direction(series, ts=None, noise_floor=0.02)      # per-dial movement
rate_of_change(series, ts=None, noise_floor=0.02) # per-dial second difference
```

- **`direction`** — the last TWO readings, per dial: `d = x[-1] − x[-2]`
  (per second if `ts` is given, per pulse otherwise). One number is
  nothing; two numbers show *which way and how fast*.
- **`rate_of_change`** — the last THREE readings, per dial: the central
  second difference, i.e. the exact acceleration of the quadratic
  interpolant through them:
  `a = 2·(v23 − v12) / (t3 − t1)` with `v_ij = (x_j − x_i)/(t_j − t_i)`.
  More than two numbers show *whether the movement itself is changing* —
  the warming is accelerating, or the fall is easing. A constant-speed
  room has **no** rate (only a change in the movement is a rate).

Robustness rules, in the captain's spirit:

- **Short series** — fewer than two readings: no direction (zeros). Fewer
  than three: no rate (zeros). The ear is still warming; it says so.
- **NaN** — carried forward from the last valid reading. A glitch is NOT a
  movement; the number doesn't matter.
- **Constant rooms** — no direction, no rate. Zero is a real read: *the
  room is holding steady*.
- **Noisy rooms** — movements below the noise floor (default 0.02 per
  pulse) read as 0. Small wiggles are not perceptions; only movement above
  the floor is a hand on the table.

Per-pulse units: `PulseLoop` reports direction per `period` and rate per
`period²` — the "mood warming +0.1/pulse, cynicism cooling −0.05/pulse²"
dial-delta language of the captain's brief.

## The internal monologue — the silence is not empty

`internal_monologue(prompt=None)` is the agent's silent thinking, composed
deterministically from the latest perception report:

> *I haven't said a word, but the room is warming — and the momentum is
> still building. Mood is the loudest hand on the table — rising 0.96 per
> pulse.*

- 1-3 sentences, always. Before the first two pulses: *"Only one pulse in —
  my ear is still warming to this room. Nothing to hold, nothing to say."*
- The headline comes from `warmth_direction` (warming / cooling / holding)
  and `warmth_rate` (momentum building / easing / steady).
- The loudest hand comes from the top per-dial direction — the dial the
  agent is actually watching.
- An optional `prompt` weaves the agent's task into the thinking: *"Asked
  'is the room warming?': mood tells the story."*

This is the part that runs even when the agent never speaks. `PulseLoop`
tracks `agent_said` so a caller can prove the monologue exists in silence.

## Design notes — the critique, answered

- **The number doesn't matter — so why keep magnitudes?** Because the
  captain's own language carries them: *"mood warming +0.1/pulse, cynicism
  cooling −0.05/pulse²"*. "The number doesn't matter" means **no single
  reading matters** — the *movement* (direction) and the *change in the
  movement* (rate) are the perceptions. That is exactly what the report
  carries: the absolute readings are relegated to `last_readings()` ("the
  numbers that don't matter individually"), and everything the agent
  thinks or says is built from deltas. Small movements below the noise
  floor read as 0 — magnitude only survives when it is a hand on the
  table.
- **The monologue IS the perception check.** They are two renderings of
  one perception: `PerceptionReport` is the machine-readable check;
  `internal_monologue()` is the same check rendered as an agent's
  thought (it is composed directly from the report). One perception, two
  faces — the report for systems, the monologue for the agent.
- **Constant pulses, or attention-triggered?** The captain said constant
  pulses — a heartbeat, not a metronome you must attend to. The interval
  default is the heartbeat; `due(now)` is the seam for callers who want
  to fire pulses on attention triggers (a message that lands, a reaction
  spike, a turn boundary) while still pulsing silently in between.
- **The loop closes.** The pulse observes; what believes it is the
  consumer. The `whole_hand` is the observable another system can act on
  — the agent's response policy (should I speak? what about?), the
  nudge prior (`elephant/nudge.py`), the plato prediction DB. Sensing
  without believing is only half the elephant; the closed loop is the
  next seam.

## Where it plugs in

- **TapNightSession** — participants pulse even when silent. A
  `PulseLoop("writer", session)` picks up the session's own `.bank` and
  `.room` automatically; between readings of each other's works, every
  participant is still sensing the evening as a whole hand. The pulse is
  the *between-the-lines* of the Tap: the guitarist listening while the
  next guitarist plays.
- **Spaces** — any `Space` adapter (`ChatSpace`, `MudSpace`,
  `SensorSpace`, and the chat-like aliases) exposes a normalized `.room`;
  the loop reads through it. The same pulse that reads a chat thread reads
  a MUD bar or a sensor deck — one heartbeat, many rooms.
- **BoatHarness / any room-bearing object** — anything with `.room` (and
  optionally `.bank`) is a room the loop can pulse.

## Cross-pollination: the pulse and the Plato JEPA encodings

The plato stack (`plato-perception`, `plato-prediction`) keeps a Dual-DB of
**perception vectors** (Z_in: what the room is) and **prediction vectors**
(Z_out: what the room is becoming). The pulse is exactly the series those
DBs want to hold:

- each pulse is a **perception vector** — the room's field at time t,
  stored in the perception DB;
- `direction` and `rate_of_change` are the **prediction features** — the
  macro read is a `Trend`/`MultiTarget` prediction in the prediction DB:
  *"warming at +0.1/pulse, accelerating at +0.05/pulse²"* is a forecast
  encoded in the same shape the prediction encoder already produces;
- the Z_in/Z_out contrast is the elephant gap applied to time: what the
  room *is* (last reading) vs what it *is becoming* (direction + rate).

Hand-crafted first, learned second — the fleet pattern. The pulse makes
the macro read a first-class object *now*; the learned dials (v1) and the
learned field (v2) can feed the same loop without changing its shape.

## The demo

`examples/demo_pulse.py` — the currency-exchange demo. One room, three
phases, one silent trader:

1. **WARM** — the room climbs out of the cold. Direction goes positive,
   then the rate **spikes** as the mood surges (and eases as the dial
   saturates): `Δ +0.55/p, Δ² +0.50/p²`, mood `-1.00 → +0.53`.
2. **FLAT** — the table goes quiet. Direction flattens to zero, rate to
   zero: `mood +0.00/pulse, rate +0.00/pulse²`. The trader holds.
3. **COOL** — the room turns. Direction flips negative (cynicism spikes
   +0.73/pulse), then the fall **eases** — the rate decays back toward
   zero.

Each pulse prints the trader's board (warmth, Δ, Δ², the moving dials)
and the internal monologue. The trader never says a word all evening; the
pulses never stop. **The macro read emerges from the numbers.**

## Tests

`tests/test_pulse.py` (18 tests):

- direction from the last two; rate from the last three (accelerating /
  decelerating / constant-speed); `ts`-normalized and per-pulse units;
- short series, NaN carry-forward, noise floor, constant rooms;
- a warming room reports warming direction **and** positive rate;
- a cooling-then-stabilizing room: direction flips, then the rate decays
  toward 0;
- `internal_monologue()` returns a string even when the agent has said
  nothing — the silence is not empty;
- the loop ticks without the agent speaking and still produces
  perceptions, with bounded history, stale-tick immunity, and `due()`;
- integration: the loop reads `ChatSpace` and a `TapNightSession`
  (bank + room resolved from the session).

---

*The number doesn't matter. Two numbers show direction; three show rate of
change. An agent is always sensing, even when silent — the pulse is the
heartbeat of that sensing, and the perception check is its looking-around.
The silence is not empty: it is full of macro reads.*
