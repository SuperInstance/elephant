# The Dual-DB bridge — the room perceives itself, and predicts itself

> The elephant reads a room. The Dual-DB bridge lets the room read
> *itself* — twice. Once for what it is (Z_in), once for where it's
> going (Z_out).

Every room in the elephant now carries two databases. **Z_in** is what
the room *perceives* — sensor frames, dial readings, field vectors,
each pressed into a vector. **Z_out** is what the room *predicts* about
itself — the next reading, the trend, the anomaly score. The elephant
has always read the room (Z_in). The bridge adds the second half: the
room's own predictions, a *trend dial* that reads where the room is
going rather than where it stands.

This module is cross-pollinated from two sibling JEPA repos, and it
gives back. That is the whole point of the captain's directive: all our
JEPA repos should cross-pollinate, because each one has an insight the
others haven't grown yet.

---

## The two parents

### Z_in — what plato-perception taught the elephant

`plato-perception` (`/home/eileen/projects/plato-perception/src/lib.rs`)
turns a sensor reading into a raw triple:

```
[value, confidence, timestamp_norm]
```

and then runs that triple through an encoding ladder — **Raw →
Normalized → HashProjection → RandomProjection → LearnedProjection** —
so the same reading can live at any fidelity, from raw numbers to a
unit-norm hashed projection. The projections are deterministic (FNV-1a
mixing), which matters: the same room read twice yields the same
vector, so a perception history is a *series you can reason over*, not
a bag of random embeddings.

The elephant already had frames (`SensorFrame`) and dials. What
plato-perception gave it was the *shape*: a reading is a reading,
whether it is a sounder's biomass, a radar's spread, a mood dial, or a
timestamp. `ZInEncoder` maps them all onto the same triple, and the
five `EncodingMethod`s ride along unchanged. The only deliberate
divergence is in the *bridge* (see below), not the encoder: the raw
triple still carries `timestamp_norm`, exactly as plato defines it.

### Z_out — what plato-prediction taught the elephant

`plato-prediction` (`/home/eileen/projects/plato-prediction/src/lib.rs`)
carries a `PredictionOutput` — a *typed* prediction (`ValuePrediction`,
`Classification`, `AnomalyScore`, `Action`, `Trend`, `MultiTarget`)
with a value, a confidence, a model name, and a latency. It encodes
those with **Raw / Confidence / Hierarchical / MultiHead** methods, so
a prediction is itself a vector that can be stored, compared, and
batch-queried.

What plato-prediction gave the elephant was the *vocabulary*: a
prediction isn't a bare float, it is a *typed* claim with a confidence.
`PredictionOutput` and `PredictionEncoder` port that over unchanged —
same type codes, same encode methods. The elephant's `ZOutPredictor`
then *makes* those claims, simply and honestly: linear extrapolation,
rolling statistics, no learned model (that is the `LearnedProjection`
stub's job, seeded later from `elephant/learned.py`).

---

## The Dual-DB architecture

```
                    ┌─────────────────────────────────────┐
                    │              the room               │
                    │  SignalRoom (frames)  ·  Room (msgs)│
                    └───────────────┬─────────────────────┘
                                    │  perceive()
                                    ▼
                    ┌─────────────────────────────────────┐
                    │  Z_in  — perception history          │
                    │  [sounder, radar, camera, nav]       │
                    │   × [value, conf, ts]  + 3 fleet     │
                    │   dials (radar_coherence,            │
                    │   sounder_biomass, fishing_day)      │
                    └───────────────┬─────────────────────┘
                                    │  predict()
                                    ▼
                    ┌─────────────────────────────────────┐
                    │  Z_out — prediction history          │
                    │  TREND             [-1 .. +1]        │
                    │  VALUE_PREDICTION  (next field)      │
                    │  ANOMALY_SCORE     [0 .. 1]          │
                    └───────────────┬─────────────────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
               trend_dial()                   anomaly()
               (where it's going)             (is something off?)
```

The bridge is `DualDBRoom`. It wraps a `SignalRoom` (frames) or a
`Room` (messages), keeps the Z_in history, and from it produces Z_out
predictions. `perceive()` snapshots the room's current state into a
Z_in vector; `predict()` runs the predictor over the history. The two
new senses — `trend_dial()` and `anomaly()` — are ordinary methods that
also cast into real `Dial`s (`TrendDial`, `AnomalyDial`) so they drop
into any `DialBank` unchanged.

The math is deliberately simple + honest. Per-dimension least-squares
slopes for trend, linear extrapolation for the next value, and a
regularized Mahalanobis deviation for the anomaly — the *same* math as
`fleetmath`'s good-days anchor (`biomass_anchor` / `biomass_deviation`).
The anomaly is not a new distance; it is the old one, turned inward:
"does this stretch of the room feel like the room's own good kind?"

---

## The trend dial — the room predicting itself

Most dials read what the room *is*. Mood, volume, panic — all
backward-looking, a photograph of the present. The trend dial is the
first dial that reads what the room *will be*:

```
trend_dial() -> float   # [-1 cooling .. +1 warming]
```

It is a new *kind* of dial, not a new *reading*. The room is the sensor,
the predictor is the perceiver, and the room's own direction is the
dimension being felt. In the tests, a warming sounder room reads
`+0.117` and the identical cooling room reads `-0.117` — symmetric, the
way a real dial ought to be. (The sign is symmetric only because the
bridge deliberately *pins the clock out* of the Z_in vector — see
below — so the recency ramp can't plant a phantom +warming on an empty
room.)

It slots into the field like any other dial: `DualDBRoom.dial()`
returns `{"trend_dial", "anomaly"}`, and `TrendDial` / `AnomalyDial`
wrap them as first-class `Dial`s. The field can now contain the room's
*anticipation*, not just its state.

---

## The anomaly sense — the room noticing something is off

```
anomaly() -> float   # [0, 1]
```

The anomaly is the Mahalanobis distance of the newest state from the
room's own recent pattern, mapped smoothly to a graded "weirdness"
meter:

```
D     = sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))     # same as fleetmath.biomass_deviation
typical = sqrt(d - 0.5)                 # the Nakagami mean of D, not sqrt(d)
anomaly = 1 - exp(-max(D - typical, 0)² / (2d))
```

Normal reads ~0, ~2× typical reads ~0.45, ~3× reads ~0.99 — no dead
zones, no binary trigger. In the tests, a perfectly steady room reads
`0.000`; a room where one dimension spikes 10× reads `1.000`. A steady
warming ramp reads `0.076` — *not* anomalous, because a trend is not a
deviation, it is the pattern itself.

When the room's recent history is so still that the covariance is
degenerate, the bridge falls back to a diagonal Mahalanobis with a tiny
floor — so a real jump still reads as anomalous, but a truly identical
state reads 0.

---

## The pulse seam

`elephant/pulse.py` is the internal-monologue engine: a `PulseLoop`
ticks on a constant heartbeat and reads a room's field through a
`DialBank`, computing *direction* from the last two readings and *rate
of change* from the last three. It is backward-looking — it reads where
the room *has* been.

The Dual-DB bridge is its forward-looking other half. They feed each
other two ways, with no change to `pulse.py`:

1. **`on_pulse` callback** — `DualDBRoom(room, on_pulse=...)` fires the
   callback with the latest Z_in vector and Z_out outputs after each
   perceive+predict. Perception in, prediction out. A `PulseLoop.tick()`
   can be driven from it.
2. **`TrendDial` / `AnomalyDial`** — the two new senses are real `Dial`s,
   so they join a `PulseLoop`'s `DialBank` unchanged. The pulse's
   `direction()` (past) and the trend dial (future) then sit side by
   side in the same board: the last two readings tell you where the
   room *was going*, the trend dial tells you where it *will go*.

---

## The reverse — what the elephant gives plato

Cross-pollination is bidirectional. The elephant's field is a *room
context* that plato's predictors never had:

1. **Room context as features.** `RoomField` / dial readings are prior
   context a prediction model can condition on — "given this warm,
   high-κ room, here is my prediction." plato's predictions are
   currently made in a vacuum; the elephant hands them the room.
2. **Free contrastive pairs.** Every `Z_in(t) → Z_in(t+1)` transition
   every room observes is a labelled training pair for a *real* learned
   projection — the missing piece behind the `LearnedProjection` stub.
   No labelling required; the room generates its own curriculum.
3. **Ground truth for projection quality.** A fleet of rooms lets plato
   measure when a projection lands out-of-distribution against the
   fleet's own anchor — an honest calibration signal no lab dataset
   provides. And confidence stops being a formula: the fleet can
   measure the actual hit rate of each prediction type.

This is the deeper point of the critique that shaped the bridge: Z_out
as summaries of Z_in is a *mirror*, not yet a predictive model. The
elephant gives plato the raw material — the contrastive pairs and the
room context — to turn the mirror into a predictor.

---

## What the reviews fixed

The design went through two external passes (Seed-2.0-pro critique,
DeepSeek review) and three honest corrections landed:

1. **The clock was leaking.** The raw triple's `timestamp_norm` is a
   recency ramp — metadata *about* the vector, not a property of the
   room's state. Fed to the trend it planted a permanent +warming bias.
   `ZInEncoder` keeps the faithful triple, but the bridge pins the
   timestamp component to 0.0.
2. **The anomaly had dead zones.** `clip((D - sqrt(d)) / 2√d, 0, 1)`
   compressed 99.9% of normal behaviour into 0–0.25 — a fire alarm, not
   a vibe meter. Replaced with the smooth Nakagami-excess mapping above.
3. **Confidence was overconfident.** `0.5 + 0.5·R²` never went low on
   noise, and constant dimensions (missing sensors) inflated it further.
   Now it is `clip(R², 0.05, 0.99)` over only the dimensions that are
   actually moving.

---

## Numbers (from the tests)

| input | TREND | confidence | VALUE_PRED | ANOMALY |
|-------|-------|-----------|-----------|---------|
| warming series `[i, i/2, i/4]` (8 vectors) | **+0.436** | 0.990 | +4.667 (> last 4.08) | 0.076 |
| steady room, no change | — | — | — | **0.000** |
| steady room, one dim spikes 10× | — | — | — | **1.000** |
| warming sounder room | **+0.117** | — | — | 0.000 |
| cooling sounder room | **−0.117** | — | — | 0.000 |

The Z_in vector is 15-dimensional: `[sounder, radar, camera, nav] ×
[value, confidence, ts]` plus the three fleet dial readings
(`radar_coherence`, `sounder_biomass`, `fishing_day`).

---

## Known limits, held honestly

- **Missing sensors encode as zero.** Zero is not "no reading", it is a
  strong signal; a sensor dropping offline will read as a jump. Native
  NaN/masking and per-channel fleet scaling are the next rung.
- **The trend is a mean of normalized slopes.** A more human
  aggregation (magnitude-weighted, or a 90th-percentile of movement)
  and per-channel global scaling would let big real changes dominate
  many flat dials.
- **Nominal dead time.** Engines blind sensors for seconds, waves move
  every dial at once, radar/sounder/nav update at different rates. A
  boat-ready bridge needs those maps; this is the room-scale v0.
- **`LearnedProjection` is a stub.** It is deterministic, like the hash
  projection, until the contrastive pairs above train a real backbone.

The bridge is not the finished instrument. It is the *seam* — the first
place where the room perceives itself and predicts itself, and the hook
where a learned predictor, a pulse engine, and the fleet's own
calibration can all grow into it.
