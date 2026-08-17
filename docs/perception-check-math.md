# The Perception Check — Two Numbers Show Direction, More Than Two Show Rate of Change

**Author:** fleet perception-math lead (subagent)
**Status:** v0 — implemented in `elephant/perception_math.py`, tested in
`tests/test_perception_math.py`
**Reviewed by:** Qwen3.6-35B-A3B (definitions + quadratic-exactness),
DeepSeek V4-Pro (self) — see "Review notes" at the end.

This is the mathematics of the elephant's *internal-monologue pulse*:
the perception check the skipper runs a few times a day, without
thinking, on the room as a whole. The captain said it in one line:

> *Like macro-economic currency-exchange changes, where the NUMBER
> doesn't matter but TWO numbers show DIRECTION and MORE THAN TWO show
> RATE OF CHANGE.*

Nobody cares that EUR/USD sits at 1.08. They care that it *moved*, and
then whether that movement is *speeding up or slowing down*. A dial
reading is the same: its absolute level is noise; its difference is
signal; its second difference is *how the signal is changing*. This
document pins down those three ideas as mathematics, and shows they are
the same one-math as `fleetmath.three_reading_kinematics` — boats on
radar and dials in a room are two domains of one derivative ladder.

---

## 1. Direction — two readings

Given two consecutive readings of a dial, `x_{t−1}` and `x_t`, the
**direction** is the first difference:

```
d = x_t − x_{t−1}
```

- `d > 0` — the dial is rising (room warming, volume climbing, joke
  landing).
- `d < 0` — the dial is falling (room cooling, panic ebbing).
- `d = 0` — the dial is flat (nothing moved).

**Why the number alone doesn't matter.** Add any constant `c` to every
reading and `d` is unchanged — `(x_t + c) − (x_{t−1} + c) = d`. This is
the currency-pair insight exactly: the exchange rate's absolute level
carries no tradeable information (it is a unit convention — a
numéraire), while its *movement* is the entire signal. A room's "mood"
of 0.7 is meaningless by itself; a mood that *moved* from 0.2 to 0.7 is
a room that just warmed up. The difference is the perception; the level
is the thing you perceive it *against*.

Two readings give exactly one direction — the minimum information needed
to say "it moved." One reading gives nothing (no *pair* to difference).

---

## 2. Rate of change — three+ readings

Three consecutive readings `x_{t−2}, x_{t−1}, x_t` give **two**
directions, and therefore a **rate of change** — the second difference:

```
a = (x_t − x_{t−1}) − (x_{t−1} − x_{t−2})
  = x_t − 2 x_{t−1} + x_{t−2}
```

Equivalently, `a` is the *difference of the two directions*: is the
room moving faster or slower than it was?

- `a > 0` — **accelerating / cascading.** The move is gathering. The
  room is warming *faster*, panic is spreading *faster* — this is the
  cascade, the stampede.
- `a < 0` — **decelerating / exhausting.** The move is running out of
  fuel. The room is still warming but *slower*; the joke is still
  landing but the laugh is trailing off. This is the move exhausting
  itself.
- `a = 0` — **steady.** The direction is constant; the room moves at a
  constant rate.

**What three readings buy that two cannot.** Two readings tell you
*where* the room is going. Three tell you *whether it's getting there
faster or slower*. This is the whole point of the captain's "more than
two" clause: the rate of change is a strictly higher derivative, and it
carries a distinct, earlier signal.

### The macro insight — the slowdown shows up before the move ends

Consider a room warming toward a plateau — say mood rising `1, 0.9,
0.8, 0.7, 0.6` per tick (a warming that is exhausting itself):

```
x    = [0.0, 1.0, 1.9, 2.7, 3.4, 4.0]
d    = [1.0, 0.9, 0.8, 0.7, 0.6]      # direction: still positive, end to end
a    = [−0.1, −0.1, −0.1, −0.1]        # rate: negative the whole way
```

Every direction is positive — the room is *still warming* at the last
reading. But every rate is negative — the warming has been *decelerating
the whole time*. The skipper who reads only direction sees "warming" and
stays; the skipper who reads the rate sees "warming, but the warming is
tiring" and starts to move *before* the direction flattens to 0.

This is the operational value of the perception check: **the rate of
change leads the direction.** The second difference turns negative
(decelerating) while the first difference is still positive — the
macro-read sees the slowdown *before* the move ends, exactly as a
currency trader reads a decelerating rally and lightens up before the
pair tops out.

---

## 3. Noise — the noise floor (small moves are not moves)

A dial reading carries measurement noise: lexical counting, message-rate
sampling, the finite window of a JEPA dial. Differencing amplifies that
noise (a first difference of two independent `σ`-noisy readings has
error `√2 σ`; a second difference has error `√6 σ`, exactly as in
`fleet-field-math.md` §1.4). The result: a "move" of `±0.002` on a
dial that never actually moved is indistinguishable from zero, and
reporting it as signal makes the elephant jitter.

The fix is a **deadband** — the same idea as the VisionDeadband in the
plato work — a margin below which a difference reads as **0**:

```
if |d| < floor:  d ← 0
```

The deadband is where the elephant has no opinion. It is not a filter
(it does not average); it is a *commitment to indifference* below a
threshold.

**How to set the floor.** Two defensible choices:

1. **Per-dial variance estimate.** Run the dial in a quiet room, take
   the sample standard deviation `σ̂` of its *differences*, and set
   `floor = k σ̂` with `k ≈ 2–3`. This is the principled choice: the
   floor is a multiple of the dial's own noise.
2. **A fixed fraction of the dial range.** Dials are bounded
   (`mood ∈ [−1, 1]`, the rest `∈ [0, 1]`), so a floor of a few percent
   of range — e.g. `0.02` for a `[0, 1]` dial — is a cheap, robust
   default. This is the pragmatic v0 choice; it says "a move smaller
   than 2% of the dial's full swing is noise."

The floor may be a single scalar for all dials, or a per-dial mapping
(the quiet presence dial may need a tighter floor than the jumpy volume
dial). `perception_math` supports both.

---

## 4. Non-uniform pulses — `dt` matters

The pulse loop does not tick on a metronome. If the ticks are uneven,
then a *per-tick* difference is a lie: a `+0.5` between two readings
2 seconds apart is not the same speed as a `+0.5` between two readings
0.5 seconds apart. The differences must be **time-normalized** — per
second, not per tick.

Let `Δt_i = t_{i+1} − t_i` be the gap between consecutive readings. The
time-normalized direction (a *rate*, units of dial-units/second) is

```
d_i = (x_{i+1} − x_i) / Δt_i
```

and the time-normalized rate of change (units of dial-units/second²) is

```
a_i = (d_{i+1} − d_i) / Δt̄_i,    Δt̄_i = (Δt_i + Δt_{i+1}) / 2
```

`Δt̄_i` is the *midpoint* of the two adjacent gaps — the time between
the two velocity samples. This is not an ad-hoc choice: it is exactly
the central second divided difference, which is the **exact
acceleration of the quadratic interpolant** through any three
consecutive readings, for *arbitrary* (non-uniform) spacing. For
uniform `Δt = 1` it collapses back to the plain second difference of
§2.

**Why the midpoint, not `Δt_{i+1}`.** The same subtlety is documented in
`fleet-field-math.md` §1.3: dividing the velocity difference by `Δt_{i+1}`
(or `Δt_i`) alone is correct only for uniform ticks and otherwise biases
the acceleration toward the shorter interval. Dividing by `Δt̄_i` — i.e.
`2 (d_{i+1} − d_i) / (Δt_i + Δt_{i+1})` — is the exact form, and it is
what `perception_math.rate_of_change` implements (and what
`fleetmath.three_reading_kinematics` uses as its acceleration
denominator `(t3 − t1)`). One formula, both domains.

In the code, `dt=None` means unit ticks (so §1–§2 hold verbatim), a
scalar means a uniform gap, and a sequence means the per-step gaps.

---

## 5. The composite read — the whole hand

The room is not one dial; it is the ensemble — the table's conversation
as a whole. The perception check must fold per-dial directions and rates
into a single macro picture. `composite_read` returns four things:

1. **`macro_direction`** — the weighted mean of each dial's *latest*
   direction. The room's overall drift; the warmth trend. With weights
   `w_i` (default equal), `macro_direction = Σ w_i d_i`.

2. **`macro_rate`** — the weighted mean of each dial's *latest* rate.
   The room's overall acceleration: is the drift itself speeding up or
   exhausting? This is the second-order version of the first, and — per
   §2 — it leads it.

3. **`fastest_dial`** — `argmax_i |d_i|`: the dial moving fastest *right
   now*. This is **what is driving the room** — the one dimension
   carrying the conversation at this instant.

4. **`accelerating_dials`** — the dials whose `|a_i|` exceeds the noise
   floor, sorted by `|a_i|` descending. These are **what is about to
   matter** — the dials whose *rate of change* is non-negligible, so
   their direction is still being discovered. A dial that is flat but
   *accelerating upward* is exactly what the skipper wants flagged
   before it becomes the fastest dial.

Together these answer the skipper's three questions — *which way is the
room drifting? is that drift tiring or gathering? what's moving it, and
what's about to?* — all from differences, never from levels.

**Dead dials.** A NaN reading (a dial that didn't observe this tick) is
*dropped, not averaged in as 0*: the surviving dials re-normalize, so a
room with one silent dial still reads as a full room.

---

## 6. One math, two domains

This is not a new derivative. It is `fleetmath.three_reading_kinematics`
with the radar stripped off:

| concept | fleet (radar) | perception (room) |
|---------|---------------|-------------------|
| position | boat `(x, y)` | dial reading `x` |
| velocity | `(p2 − p1)/Δt` | direction `d` |
| acceleration | `2 (v23 − v12)/(t3 − t1)` | rate `a` |
| association | nearest-neighbour gate | (none — dials are tracked by name) |
| own-ship correction | lever arm | (none — dials are absolute) |

The room is the *easy* case: a dial is already a named track, so there
is no association problem and no own-ship frame to compensate. What
remains is the pure derivative ladder — first difference = direction,
second difference = rate — and the same `σ/Δ²` noise blow-up, the same
non-uniform-spacing exactness, and the same "three readings are the
minimum for acceleration" limit. The `test_rate_of_change_matches_fleetmath_acceleration`
test pins this down numerically: a 1-D quadratic fed through both
pipelines returns the same acceleration.

---

## Implementation notes

`elephant/perception_math.py` (numpy-only, no heavier package imports):

- `direction(series, dt=None, noise_floor=0.0) -> {name: [d_0, …]}`
  — per-dimension first difference, time-normalized, deadbanded.
- `rate_of_change(series, dt=None, noise_floor=0.0) -> {name: [a_0, …]}`
  — per-dimension second difference, time-normalized, deadbanded.
- `composite_read(series, weights=None, noise_floor=0.0) -> dict`
  — the whole-hand read (macro_direction, macro_rate, fastest_dial,
  accelerating_dials, plus the raw per-dial directions/rates).

`series` is either a mapping `{dial_name: [r0, r1, …]}` or a bare 1-D
sequence (wrapped as a single unnamed dial). Edge cases handled: 1 point
→ no direction and no rate; 2 points → direction only; constant series
→ all zeros; NaN → propagated through the per-dial differences and
excluded at the aggregate; non-uniform `dt` → per-second rates. The
deadband zeros magnitudes *strictly below* the floor (a move exactly at
the floor stays), and the floor is interpreted in the units of the
output (dial-units/second for direction, /second² for rate) — which for
the ~1 s pulse ticks of this system is effectively per-tick.

---

## Review notes

- **Qwen3.6-35B-A3B** (DeepInfra) confirmed the forward-difference and
  second-difference definitions and the quadratic-exactness claim, and
  independently recomputed the non-uniform-dt acceleration of
  `x(t) = 0.5·3 t² + 2t + 1` at `t = 0, 2, 5` as `a = 3`.
- **DeepSeek V4-Pro** (self) verified numerically: second difference of
  the quadratic equals `a` exactly (uniform and non-uniform), and the
  decelerating-rise example (`rate < 0` while `direction > 0`) — the
  "slowdown before the move ends" insight of §2 — holds exactly.
