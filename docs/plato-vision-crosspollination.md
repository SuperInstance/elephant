# plato-vision-jepa × elephant — the vision dial

> **One sense, two sources.** Plato reads the room from camera frames;
> the elephant reads the room from vibes. They were always reading the
> same room. The vision dial is the elephant borrowing plato's eyes.

The captain's directive: *all our JEPA repos should cross-pollinate.*
`plato-vision-jepa` is the PLATO nervous system's perception layer — it
turns raw camera frames into a 16-dimensional `RoomVisionState` (brightness,
motion, occupancy, anomaly, quadrant activity, temporal trends). The
elephant is the fleet's room-temperature sense — a bank of dials, each one
JEPA perceiving one dimension of a room's vibe.

One sense, two sources. The vision dial (`elephant/dials/vision.py`) is
where they meet: the elephant gains a sense that reads camera frames into
the room's field.

---

## The signal chains, side by side

```
plato-vision-jepa          Camera → Frame Histogram → VisionDeadband → JEPA → RoomVisionState (16-dim)
elephant VisionDial        SensorFrame(camera) → room-state extract → deadband → weighted energy → reading
```

Same shape, different skin. Plato's chain ends in a 16-dim vector that
flows into `plato-nervous` for fusion; the elephant's chain ends in one
scalar reading on the `[0 dark+empty .. 1 bright+alive]` range that joins
the other eight dials in the field. Plato perceives; the elephant feels.

---

## What plato taught the elephant

### 1. The deadband insight (the big one)

Plato's `VisionDeadband` is a histogram diff filter: only frames whose
histogram changed significantly are passed to the JEPA. A camera pushing
the same room state every second is not telling you anything new — process
it and you waste compute; worse, you let the *repetition* dominate the
perception.

The elephant's version: `VisionDial(deadband=0.05)` applies the same filter
over the frame sequence **at read time**. A frame whose room-state differs
from the previous frame by less than the threshold (mean absolute
difference over the four fields, plato's histogram-intersection distance
reborn) is skipped. First frame always counts. The reading is the mean
energy over the last 8 processed frames — the recent visual field.

Consequences, both intended:

- **Redundancy can't lie.** 50 identical frames read exactly the same as
  1 frame. A camera that spams the same state cannot inflate the dial.
- **The filter is the compute budget.** Downstream, a camera feed that
  says nothing new costs nothing — the deadband is where the elephant
  decides what deserves attention, exactly as plato decides what deserves
  JEPA inference.

The threshold is a parameter, defaulting to plato's 0.05, and it is the
dial's primary tuning surface: raise it to make the dial calmer (only big
visual shifts move the needle), lower it to make the dial jumpy.

### 2. The 16-dim room-state layout

Plato's `RoomVisionState` is a fixed vector with a documented meaning per
index — this is the *protocol*, and the elephant now speaks it:

| Index | Field | This dial |
|-------|-------|-----------|
| 0 | brightness (0–1) | weight 0.40 |
| 1 | motion_level (0–1) | weight 0.35 |
| 2 | occupancy (0–1, normalized count) | weight 0.25 |
| 3 | anomaly_score (0–1) | bonus spike (0.5 × headroom) |
| 4–7 | quadrant activity (TL, TR, BL, BR) | *future dials* |
| 8–11 | temporal patterns | *future dials* |
| 12–15 | reserved | — |

The dial accepts **both** documented forms of frame `data`:

- a **full 16-dim vector** (list/tuple of 16 floats, plato's
  `RoomVisionState.to_vector()` layout — indices 0–3 used);
- a **dict** `{brightness, motion, occupancy, anomaly}` — plato's struct
  spellings `motion_level` / `anomaly_score` accepted, missing keys read
  0.0. A dict that names none of the room-state fields is treated as
  unreadable, not as "dark empty".

A room with **no camera frames reads 0.5** — neutral, the room has no
visual opinion. A plain text `Room` (the shared bank reads those too)
also reads 0.5 rather than crashing: the vision sense simply can't see it.

### 3. The energy formula

```
base   = 0.40·brightness + 0.35·motion + 0.25·occupancy
energy = clip(base + 0.5·anomaly·(1 − base), 0, 1)
```

Anomaly is a **bonus spike**, not a fourth weight: it pushes the reading
toward 1.0 by half the headroom the base energy left. A dark empty room
with a high anomaly reads 0.5 — *something is off in the dark*, which is
exactly the feeling, not a measurement. Bright + active + occupied reads
high; dark + still + empty reads low; the two rooms' gap is the vision
sense's contribution to the elephant's contrast.

---

## What the elephant gives plato-vision (the reverse)

The debt runs both ways. Plato's `anomaly_score` is a scalar with no
valence — it says *something changed*, never *what kind of something*.
The elephant's field is exactly the missing context:

### Context for its anomalies

A dark room that's calm and a dark room that's tense can carry identical
vision states — same brightness, same motion, same occupancy. The
elephant's other dials tell them apart:

| Vision state | mood | panic | volume | presence | What it is |
|---|---|---|---|---|---|
| dark, empty, anomaly 0.8 | 0.0 | 0.0 | 0.0 | 0.1 | *power cut, cat knocked something over* |
| dark, empty, anomaly 0.8 | −0.6 | 0.9 | 0.7 | 0.4 | *something is wrong and the room knows it* |

Feed the field as a context vector alongside plato's 16-dim state and the
anomaly stops being a threshold and becomes a *question with an answer in
the room*: anomaly + warm field = curiosity; anomaly + panic field = alarm.
That is the elephant's whole job, restated — it changes what you compare,
it never replaces the reading.

### The nudge, pointed at the camera

The elephant's numbers steer what downstream models compare (`nudge.py`).
The vision dial joins `volume` and `presence` as a camera-facing sense —
high visual energy says *compare this hour's frames to the good hour*;
a deadband-quiet camera says *don't burn attention there*. The dial's
reading is a prior over camera attention, exactly as `sounder_biomass` is
a prior over the water column.

### Room-relative calibration

The elephant's sauna/plunge contrast — the gap between rooms — is the
training signal, and it applies to plato's thresholds too. A busy Tap and
a dark wheelhouse have different *normals*; the same anomaly score means
different things in each. The elephant's field is how a vision system
learns a room's baseline: calibrate the anomaly threshold (and even the
deadband — tense room, pay attention to every change; calm room, save the
compute) against the room's own field history.

---

## The seam

```
plato-vision-jepa (camera) ─┐
                            ├→ the room's field → nudge → what gets compared
elephant dials (vibes) ─────┘
```

The seam is a `SensorFrame` with `sensor="camera"` whose `data` is plato's
own 16-dim vector — the elephant can now be fed by plato directly, or by
any camera adapter that speaks the same protocol. And plato can be given
the elephant's field as anomaly context. Two JEPA repos, one sense, both
directions.

## Next crosses

- **Quadrant dial** (indices 4–7): where in the room the energy is — a
  spatial sense plato already computes.
- **Temporal trend dial** (indices 8–11): is the room warming or cooling
  visually — motion trends over recent frames, plato's `temporal_patterns`.
- **Field-adaptive deadband**: the elephant's own tension (panic/mood)
  adjusting the vision deadband — attentive when the room is tense,
  thrifty when it's calm.
- **Anomaly context vector**: plato's `anomaly_score` + elephant field as
  input to a learned fusion in `plato-nervous`.
