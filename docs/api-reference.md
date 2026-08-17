# Elephant — Engineer's Reference (API)

The elephant is the inter-model temperature: a room is not a stream of
messages, it is a **field** — the ensemble of many JEPA dials, each
perceiving one dimension of the room's vibe, all shaping every agent at
once. This reference documents every public symbol in the package, module
by module, with exact signatures and **verified** usage examples (every
snippet below was executed against the code and passed).

Package: `elephant` · version `0.1.0` · numpy-only core.

---

## Quick index

| Symbol | Module | One line |
|---|---|---|
| `Message` | `room` | A single message with gravity/reverberation/ripple inputs |
| `Room` | `room` | Ordered message stream with field physics |
| `Dial` | `dial` | Abstract JEPA sense: one dimension of a room's vibe |
| `DialBank` | `dial` | Ensemble of dials; reads a room into a dict of readings |
| `MoodDial` | `dials` | Warm/cold valence, `[-1, +1]` |
| `VolumeDial` | `dials` | Loudness, `[0, 1]` |
| `EarnestnessDial` | `dials` | Sincere vs ironic, `[0, 1]` |
| `CynicismDial` | `dials` | Sneering vs earnest, `[0, 1]` |
| `JokeLandingDial` | `dials` | Collective laugh/boo, `[-1, +1]` |
| `PanicDial` | `dials` | Stampede sense, `[0, 1]` |
| `PresenceDial` | `dials` | Pheromone/occupancy trace, `[0, 1]` |
| `ModelVsCodeDial` | `dials` | Model prose vs code executing, `[-1, +1]` |
| `VisionDial` | `dials` | The room's visual energy from camera frames, `[0, 1]` |
| `DEFAULT_DIALS` | `dials` | The standard 9-dial bank |
| `DIAL_NAMES` | `field` | Canonical 7-dial field-vector ordering |
| `RoomField` | `field` | The ensemble reading — the room's temperature vector |
| `read_field` | `field` | Run a dial bank over a room → `RoomField` |
| `acclimation_curve` | `field` | Agent relaxing toward the room field (exponential) |
| `acclimation_rate_from` | `field` | Invert the curve → modulation skill rate |
| `charisma_pull` | `field` | Room bending toward a charismatic agent |
| `field_vector_for` | `field` | Coerce `RoomField`/array to a 7-vector |
| `SensorFrame` | `sensors` | One timestamped sensor reading |
| `SignalRoom` | `sensors` | A room of sensor frames (not messages) |
| `RadarCoherenceDial` | `sensors` | Fleet tight/scattered, `[-1, +1]` + kinematics |
| `SounderBiomassDial` | `sensors` | Biomass under the keel, `[0, 1]` |
| `FishingDayDial` | `sensors` | Composite luck field, `[-1, +1]` |
| `_associate` | `sensors` | Nearest-neighbour target association helper |
| `MODALITIES` | `nudge` | The 7 attention modalities |
| `NUDGE_MAP` | `nudge` | Dial → (modality, sign) routing table |
| `nudge_prior` | `nudge` | Readings → attention prior vector |
| `apply_nudge` | `nudge` | Blend a prior into cross-attention weights |
| `describe` | `nudge` | Human-readable prior string |
| `MPS_TO_KTS` | `fleetmath` | m/s → knots conversion constant |
| `three_reading_kinematics` | `fleetmath` | dir/speed/accel from 3 radar sweeps |
| `fleet_spread` | `fleetmath` | Mean distance to fleet centroid |
| `headings_to_vectors` | `fleetmath` | Degrees → unit vectors on S¹ |
| `vmf_kappa` | `fleetmath` | vMF concentration κ from unit vectors |
| `fleet_concentration` | `fleetmath` | Fleet κ (bearing or same-tack) |
| `kappa_rate` | `fleetmath` | dκ/dt — scatter/bunch signal |
| `biomass_anchor` | `fleetmath` | Good-day Gaussian anchor (OAS shrinkage) |
| `biomass_deviation` | `fleetmath` | Mahalanobis distance from the anchor |
| `BoatHarness` | `harness` | One boat's elephant: sensors + chat → field + nudge |
| `DIAL_BOUNDS` | `tapnight` | Per-dial clamp ranges |
| `DIAL_CENTER` | `tapnight` | Per-dial neutral/quiescent value |
| `REACTION_TO_DIAL` | `tapnight` | Emoji → dial mapping (crowd's hands) |
| `Participant` | `tapnight` | One agent's persistent, self-tuning settings |
| `TapNightSession` | `tapnight` | One after-work gathering engine |
| `RoomElephant` | `presets` | The room's own objective reading |
| `PersonalElephant` | `presets` | One agent's subjective reading (taste/bias/attachments) |
| `PRESETS` | `presets` | Registry: `{"room", "personal"}` |
| `WEATHER`/`LIGHT`/`JOY_ADJ`/`CLOSE_DETAIL` | `mud` | Tint template banks |
| `classify` | `mud` | Field → body-language mode (panic/joyful/closing/neutral) |
| `tint_description` | `mud` | Mutate a room description by the field |
| `Space` | `space` | Abstract adapter: any medium → room + tint target |
| `AdapterRegistry` | `space` | Register/instantiate adapters by kind |
| `read_space` | `space` | Read any space's field |
| `MudSpace` | `space` | MUD room adapter |
| `ChatSpace` | `space` | Chatroom/messenger/thread adapter |
| `SensorSpace` | `space` | Sensor-array adapter (signal + text rooms) |

---

## `elephant.room` — rooms as message fields

### `Message`
```python
@dataclass
class Message:
    author: str
    text: str
    ts: float = 0.0
    channel: str = "default"
    reactions: Dict[str, int] = field(default_factory=dict)   # emoji -> count
    replies: List["Message"] = field(default_factory=list)
```
One message in a room. Its **gravity** inputs are the reaction heat and
reply count; its **ripple** input is its reply tree.

Properties:
- `words` → `List[str]` — lowercase `\w+` tokens of `text`.
- `reaction_heat` → `int` — `sum(reactions.values())`, the crowd's hands.

```python
from elephant.room import Message
m = Message(author="ada", text="This place is WONDERFUL", reactions={"❤️": 3})
m.words           # ['this', 'place', 'is', 'wonderful']
m.reaction_heat   # 3
```

### `Room`
```python
class Room:
    def __init__(self, name: str, messages: Optional[Iterable[Message]] = None)
    def gravity(self, msg, half_life: float = 1800.0, engagement_weight: float = 1.0) -> float
    def gravity_series(self, half_life: float = 1800.0) -> List[float]
    def reverberation(self, window: int = 8) -> float
    def ripple(self, msg, depth: int = 3) -> int
    def ripple_series(self, depth: int = 3) -> List[int]
    def density(self, window: float = 300.0) -> float
```
An ordered message stream with the physics of a gathered room. Messages are
sorted by `ts` on construction.

- **`gravity`** — recency-weighted pull: `0.5**(age/half_life) · (1 + w·log1p(heat+replies)) · (1 + log1p(words)/10)`.
- **`reverberation`** — mean cosine similarity between consecutive `window`-sized windows of gravity (how much the room repeats its own beat). `0.0` when fewer than `2·window` messages.
- **`ripple`** — cascade size through replies + reactions (recursive to `depth`).
- **`density`** — messages per minute over the trailing `window` seconds.

```python
from elephant.room import Message, Room
r = Room("bar", [Message("ada", "hello", ts=0.0),
                 Message("bo", "lol nice", ts=10.0)])
r.gravity(r.messages[0])     # recency × engagement × length
r.ripple(r.messages[0])      # cascade reach through replies/reactions
r.density()                  # messages per minute
len(r)                       # 2
```

---

## `elephant.dial` — the JEPA sense abstraction

### `Dial` (ABC)
```python
class Dial(ABC):
    name: str = "dial"
    description: str = ""
    @abstractmethod
    def read(self, room: Room) -> float: ...
    def series(self, room: Room, window: int = 8) -> List[float]
```
One JEPA perceiving one dimension. `read` returns a scalar, self-normalizing
reading (`[-1,1]` or `[0,1]` depending on the dial). `series` defaults to a
single-element list (static reading); override for windowed training series.

```python
from elephant.dial import Dial
class AlwaysHot(Dial):
    name = "always_hot"
    def read(self, room): return 1.0
```

### `DialBank`
```python
class DialBank:
    def __init__(self, dials: Optional[Iterable[Dial]] = None)
    def add(self, dial: Dial) -> "DialBank"
    def readings(self, room: Room) -> Dict[str, float]
    def series(self, room: Room) -> Dict[str, List[float]]
    def names(self) -> List[str]
    def __len__(self) -> int
```
The perceiving ensemble — many dials, one room.

```python
from elephant.dial import DialBank
from elephant.dials import MoodDial
bank = DialBank([MoodDial()])
bank.readings(room)   # {'mood': ...}
```

---

## `elephant.dials` — the nine dials

Each dial is a `Dial` subclass reading a `Room`. Empty rooms rest at each
dial's neutral value. **Adding/removing words from the module-level lexicons
is the primary tuning surface** (see the tuning guide).

| Dial | `name` | Range | Empty | Lexicon constants |
|---|---|---|---|---|
| `MoodDial` | `mood` | `[-1, +1]` | `0.0` | `POSITIVE`, `NEGATIVE` |
| `VolumeDial` | `volume` | `[0, 1]` | `0.0` | (caps/exclamations/density) |
| `EarnestnessDial` | `earnestness` | `[0, 1]` | `0.5` | `SINCERE`, `HEDGE` |
| `CynicismDial` | `cynicism` | `[0, 1]` | `0.5` | `CYNICAL`, `EYEROLL` |
| `JokeLandingDial` | `joke_landing` | `[-1, +1]` | `0.0` | `JOKE_MARKERS`, `LAUGH`, `BOO` |
| `PanicDial` | `panic` | `[0, 1]` | `0.0` | `ALARM`, `URGENCY` |
| `PresenceDial` | `presence` | `[0, 1]` | `0.0` | (author occupancy) |
| `ModelVsCodeDial` | `model_vs_code` | `[-1, +1]` | `0.0` | `MODEL_WORDS`/`MODEL_PHRASES`, `CODE_WORDS`/`CODE_PHRASES`, `CODE_SYMBOLS` |
| `VisionDial` | `vision` | `[0, 1]` | `0.5` | (signal-room camera frames; plato 16-dim layout) |

### `MoodDial` — `read(room) -> float`
Counts hits of `POSITIVE` vs `NEGATIVE` word sets over all messages;
`raw = (pos−neg)/max(total,1) · 2`, clamped to `[-1, 1]`.

### `VolumeDial` — `read(room) -> float`
`0.45·density_norm + 0.35·caps_ratio + 0.20·excl_ratio`, clamped to `[0,1]`,
where `density_norm = 1 − exp(−density/20)` (density in msgs/min over 60 s).

### `EarnestnessDial` — `read(room) -> float`
`sincere / (sincere + hedge)` over the `SINCERE` word set and `HEDGE` phrases;
`0.5` when no signal.

### `CynicismDial` — `read(room) -> float`
`(hits + scare_quote_pairs + eyeroll_emoji) / total_words · 40`, clamped to
`[0,1]` (≈2.5% cynical tokens saturates to `1.0`).

### `JokeLandingDial` — `read(room) -> float`
Finds messages containing `JOKE_MARKERS`, then reads the *next four* messages
(the audience) for `LAUGH` vs `BOO`, plus laugh/boo reactions on the joke
itself. Returns the mean of `(laugh−boo)/(laugh+boo)` across jokes; `0.0` if no
jokes.

### `PanicDial` — `read(room) -> float`
`0.40·alarm_norm + 0.25·urgency_norm + 0.20·ripple_norm + 0.15·density_norm`
(density over 30 s), clamped to `[0,1]`. Ripple is the max `room.ripple` of any
alarm-triggering message.

### `PresenceDial` — `read(room) -> float`
`0.45·distinct_authors/5 + 0.25·recency + 0.20·longevity + 0.10·activity`,
clamped to `[0,1]`.

### `ModelVsCodeDial` — `read(room) -> float`
The signal-chain thesis made flesh: a room's signal is shaped by *who or what*
generates it. Scores each message by whether it smells like model output
(hedges, reflection, first-person, prose — `MODEL_WORDS`/`MODEL_PHRASES`) or
code output (keywords, diffs, errors, commits, symbol density —
`CODE_WORDS`/`CODE_PHRASES`/`CODE_SYMBOLS`), maps the balance to `[-1, +1]`,
and averages over the room. `-1` = pure code room (commits, patches, error
logs); `+1` = pure model room (prose, hedges, reflection); `0.0` = empty/neutral.
It reads as the 8th member of `DEFAULT_DIALS` (present in `read_field`'s
`readings`) but is not yet in `DIAL_NAMES` — wiring it into the field vector
(κ / distance) is the next step, coordinated with the learned-dials subagent.

### `VisionDial` — `read(room) -> float`, `__init__(deadband=0.05)`
Cross-pollinated from `plato-vision-jepa`. Reads a `SignalRoom`'s camera frames
(`SensorFrame` with `sensor="camera"`; `data` is either a full **16-dim plato
room-state vector** — indices 0–3 brightness/motion/occupancy/anomaly — or a
dict `{brightness, motion, occupancy, anomaly}`, plato's spellings
`motion_level`/`anomaly_score` accepted). Produces the room's visual
energy/aliveness: `0.40·brightness + 0.35·motion + 0.25·occupancy`, plus an
anomaly bonus spike `0.5·anomaly·(1−base)`, clamped to `[0,1]`. `deadband`
(default `0.05`, matching plato) is the `VisionDeadband` threshold: a frame
whose state differs from the previous by less than that is skipped, so
redundant frames can't dominate. No camera frames (or a plain text `Room`)
→ `0.5` — no visual opinion. Reads as the 9th member of `DEFAULT_DIALS`; not
yet in `DIAL_NAMES`. See `docs/plato-vision-crosspollination.md`.

```python
from elephant.dials import VisionDial
from elephant.sensors import SensorFrame, SignalRoom

v = [0.0] * 16; v[0], v[1], v[2], v[3] = 0.9, 0.7, 0.8, 0.1   # plato 16-dim
sig = SignalRoom("bridge", [SensorFrame(ts=0.0, sensor="camera", data=v)])
VisionDial().read(sig)         # ~0.81 — bright, active, occupied
VisionDial().read(SignalRoom("empty"))    # 0.5 — no visual opinion
```

```python
from elephant.room import Message, Room
from elephant.dials import DEFAULT_DIALS

r = Room("bar", [Message("ada", "what a WONDERFUL night, haha", ts=0.0,
                         reactions={"😂": 2}),
                 Message("bo", "lol", ts=10.0)])
for d in DEFAULT_DIALS:
    print(d.name, round(d.read(r), 3))
```

### `DEFAULT_DIALS`
`List[Dial]` — the standard bank: `MoodDial(), VolumeDial(), EarnestnessDial(),
CynicismDial(), JokeLandingDial(), PanicDial(), PresenceDial(),
ModelVsCodeDial(), VisionDial()` (in that order).

---

## `elephant.field` — the Field (the elephant)

### `DIAL_NAMES`
`["mood", "volume", "earnestness", "cynicism", "joke_landing", "panic", "presence"]`
— the canonical 7-dial ordering used to build/parse vectors.

### `RoomField`
```python
class RoomField:
    def __init__(self, readings: Dict[str, float])
    def vector(self, names: Optional[Sequence[str]] = None) -> np.ndarray    # (7,)
    def normalize(self, names: Optional[Sequence[str]] = None) -> np.ndarray
    def warmth(self) -> float                  # ~[-1, +1]
    def concentration(self) -> float           # κ, >= 0
    def distance(self, other: "RoomField") -> float
    def sauna_plunge_gap(self, other: "RoomField") -> float
```
The ensemble reading — the room's temperature vector. `vector()` returns the
7 readings in `DIAL_NAMES` order (missing dials read `0.0`). `warmth()` is the
felt temperature; `concentration()` is κ — how far the field sits from
neutral (the v3 spec reads this as *tightness*, cold = high κ; v0's proxy
measures extremity, not yet temperature). `distance()` is the
normalized-vector gap between two rooms; `sauna_plunge_gap()` is the **signed**
warmth contrast (`this − other`).

```python
from elephant.field import RoomField
a = RoomField({"mood": 0.6, "joke_landing": 0.4, "presence": 0.5})
b = RoomField({"mood": -0.5, "cynicism": 0.8, "panic": 0.6})
a.warmth(); a.concentration()
a.sauna_plunge_gap(b)     # positive = a warmer than b
```

### `read_field`
```python
def read_field(room: Room, bank: Optional[DialBank] = None) -> RoomField
```
Reads a room with `bank` (defaults to `DialBank(DEFAULT_DIALS)`) → its field.

### `acclimation_curve`
```python
def acclimation_curve(agent: np.ndarray, room: np.ndarray, rate: float, t: float) -> np.ndarray
```
Exponential relaxation of an agent's embedding toward the room field:
`room + (agent − room)·e^(−rate·t)`. `rate` (1/τ) **is** the modulation skill.

```python
import numpy as np
from elephant.field import acclimation_curve
agent = np.zeros(7); room = np.full(7, 0.5)
acclimation_curve(agent, room, rate=0.5, t=1.0)   # ~39% of the way toward room
```

### `acclimation_rate_from`
```python
def acclimation_rate_from(agent_start, agent_obs, room, t: float) -> float
```
Inverts the curve: given start/observed/room, recovers `rate = −ln(ratio)/t`.
The projection ratio is clamped to `[1e-9, 1]` so an agent that has overshot the
room yields a large *finite* rate instead of `inf`.

### `charisma_pull`
```python
def charisma_pull(room: np.ndarray, agent: np.ndarray, charisma: float, interactions: int) -> np.ndarray
```
The room bends toward a strong presence: `room + (agent−room)·(1 − e^(−charisma·interactions))`.

### `field_vector_for`
```python
def field_vector_for(agent_or_room, names: Optional[Sequence[str]] = None) -> np.ndarray
```
Coerces a `RoomField` (via `.vector(names)`) or a bare array to a float array.

---

## `elephant.sensors` — the elephant's sea legs

### `SensorFrame`
```python
@dataclass
class SensorFrame:
    ts: float
    sensor: str                    # "radar" | "sounder" | "camera" | ...
    data: Any                      # radar: list[(x,y)] · sounder: float ...
    meta: Dict[str, Any] = field(default_factory=dict)
```

### `SignalRoom`
```python
class SignalRoom:
    def __init__(self, name: str, frames: Optional[Sequence[SensorFrame]] = None)
    def by_sensor(self, sensor: str) -> List[SensorFrame]
    def __len__(self) -> int
```
A room of sensor frames instead of messages (same timestamped-sequence idea).

### `RadarCoherenceDial` — `read(room) -> float`, `kinematics(room) -> dict`
`name="radar_coherence"`, range `[-1, +1]` (scattered → clustered/on fish).
`read` uses the last 3 radar frames' spatial spread (mean distance to centroid)
plus a closing/scattering trend. `kinematics` recovers per-object
`dir_deg`, `speed_kts`, `accel` from three readings, plus `fleet_mean_speed` and
`spread_rate`.

### `SounderBiomassDial` — `read(room) -> float`
`name="sounder_biomass"`, range `[0, 1]`. Mean of the last 5 sounder values
plus a trend term, clipped.

### `FishingDayDial` — `__init__(radar=None, sounder=None)`, `read(room) -> float`
`name="fishing_day"`, range `[-1, +1]`. Composite: `0.55·radar + 0.45·(2·sounder − 1)`, clipped.

### `_associate`
```python
def _associate(a: np.ndarray, b: np.ndarray, gate: float = 2.0) -> List[Tuple[np.ndarray, np.ndarray]]
```
Greedy nearest-neighbour association (position pairs) within `gate`. Internal
helper shared by the radar dial's kinematics.

```python
from elephant.sensors import SensorFrame, SignalRoom, RadarCoherenceDial, SounderBiomassDial
frames = [
    SensorFrame(ts=0.0,  sensor="radar",   data=[(0.0, 0.0), (1.0, 1.0)]),
    SensorFrame(ts=10.0, sensor="radar",   data=[(0.1, 0.0), (1.1, 1.0)]),
    SensorFrame(ts=20.0, sensor="radar",   data=[(0.2, 0.0), (1.2, 1.0)]),
    SensorFrame(ts=0.0,  sensor="sounder", data=0.6),
]
sig = SignalRoom("bridge", frames)
RadarCoherenceDial().read(sig)         # fleet tightness [-1,+1]
SounderBiomassDial().read(sig)         # biomass [0,1]
```

---

## `elephant.nudge` — dial numbers steering attention

### `MODALITIES`
`["radar", "sounder", "camera_out", "camera_deck", "nav", "autopilot", "conversation"]`

### `NUDGE_MAP`
`Dict[str, Tuple[str, float]]` — which dial feeds which modality, and its sign:
`radar_coherence→("radar",+1)`, `sounder_biomass→("sounder",+1)`,
`fishing_day→("nav",+0.5)`, `mood→("conversation",+1)`,
`volume→("camera_deck",+0.5)`, `panic→("camera_out",+1)`,
`presence→("camera_deck",+0.3)`.

### `nudge_prior`
```python
def nudge_prior(readings: Dict[str, float], modalities: Optional[Sequence[str]] = None) -> np.ndarray
```
Dial readings → an attention prior over `modalities` (default `MODALITIES`),
each component in `[-1, 1]`. Unknown dial names are ignored; the prior is
renormalized so the strongest opinion is at most `1.0`.

### `apply_nudge`
```python
def apply_nudge(attention: np.ndarray, prior: np.ndarray, strength: float = 0.15) -> np.ndarray
```
Blends a prior into cross-attention weights: `a·(1 + strength·p)`. Raises
`ValueError` if `attention`'s last dim ≠ `len(prior)`.

### `describe`
```python
def describe(prior: np.ndarray, modalities: Optional[Sequence[str]] = None) -> str
```
Human-readable prior: `"nudge[radar=+0.80, camera_out=+0.90]"`.

```python
import numpy as np
from elephant.nudge import nudge_prior, apply_nudge, describe
prior = nudge_prior({"radar_coherence": 0.8, "panic": 0.9})
describe(prior)                                    # 'nudge[radar=+0.80, camera_out=+0.90]'
apply_nudge(np.ones(7), prior, strength=0.15)      # attention × (1 + 0.15·prior)
```

---

## `elephant.fleetmath` — the numeric core (numpy-only)

Units: positions **metres**, times **seconds**, speeds **knots**, angles
**degrees**, acceleration **m/s²**.

### `MPS_TO_KTS`
`1.9438444924406046` — metres/second → knots.

### `three_reading_kinematics`
```python
def three_reading_kinematics(frames: Sequence[Any], own_ship: Optional[Sequence] = None, gate: float = 2.0) -> Dict[str, Any]
```
Recovers per-object **direction**, **speed** (knots), and **acceleration**
(`2(v23−v12)/(t3−t1)`) from exactly three radar sweeps, with nearest-neighbour
association (`gate`, metres). `frames` may be `SensorFrame` objects (with
`.data`/`.ts`) or bare `(N,2)` arrays (timestamp defaults to `0.0`).
`own_ship` (shape `(3,2)`) removes own-ship translation (lever-arm correction).
Returns `{"objects", "fleet_mean_speed", "spread_rate"}`.

```python
from elephant.sensors import SensorFrame
from elephant.fleetmath import three_reading_kinematics
f1 = SensorFrame(ts=0.0,  sensor="radar", data=[(0.0,0.0),(10.0,0.0)])
f2 = SensorFrame(ts=10.0, sensor="radar", data=[(2.0,1.0),(12.0,1.0)])
f3 = SensorFrame(ts=20.0, sensor="radar", data=[(4.0,3.0),(14.0,3.0)])
kin = three_reading_kinematics([f1, f2, f3], gate=5.0)
kin["objects"][0]["dir_deg"]   # 45.0
kin["fleet_mean_speed"]        # ~0.55 knots
```

### `fleet_spread`
```python
def fleet_spread(positions: Sequence) -> float
```
Mean distance of points to their centroid — the positional counterpart to κ.

### `headings_to_vectors`
```python
def headings_to_vectors(headings_deg: Sequence) -> np.ndarray
```
Heading angles (degrees) → unit vectors on S¹ (shape `(N, 2)`).

### `vmf_kappa`
```python
def vmf_kappa(unit_vectors: Sequence) -> float
```
von Mises–Fisher concentration κ (Banerjee–Dhillon–Ghosh–Sra approximation):
`κ = R(d − R²)/(1 − R²)` with `R = ‖mean(x)‖`. κ≈0 = uniform/loose; κ≫0 = tight.
Capped at `1e4`.

### `fleet_concentration`
```python
def fleet_concentration(positions: Sequence, headings: Optional[Sequence] = None) -> float
```
Fleet κ. With `headings`: **same-tack** coherence (how aligned headings are).
Without: **bearing** coherence (unit vectors from own ship to each boat).
`0.0` when fewer than two boats.

### `kappa_rate`
```python
def kappa_rate(frames: Sequence, times: Optional[Sequence] = None) -> float
```
`dκ/dt` — least-squares slope of κ vs time. Positive = bunching; negative =
scattering.

```python
from elephant.fleetmath import fleet_concentration
tight = [(100.0, 0.0), (102.0, 2.0), (101.0, -1.0)]
wide  = [(100.0, 0.0), (-80.0, 80.0), (0.0, -100.0)]
fleet_concentration(tight)      # large (tight cluster)
fleet_concentration(wide)       # ~0 (scattered)
```

### `biomass_anchor`
```python
def biomass_anchor(good_day_vectors: Sequence, shrinkage: Optional[float] = None) -> Dict[str, Any]
```
Fits the good-fishing-day anchor: a Gaussian `N(μ, Σ)` over an `(N, d)` feature
array, with OAS shrinkage (Chen et al. 2010) by default (or an explicit
`shrinkage ∈ [0,1]`). Returns `{"mean", "cov", "shrinkage", "n", "d"}`.

### `biomass_deviation`
```python
def biomass_deviation(vec: Sequence, anchor: Dict[str, Any]) -> float
```
Mahalanobis distance `sqrt((x−μ)ᵀΣ⁻¹(x−μ))` from the anchor (via linear solve,
not explicit inverse). Small = feels like the good kind; large = shift.

```python
import numpy as np
from elephant.fleetmath import biomass_anchor, biomass_deviation
good = np.array([[0.8,0.7,0.9],[0.7,0.6,0.8],[0.9,0.8,0.7],[0.6,0.7,0.6],[0.8,0.6,0.8]])
anchor = biomass_anchor(good)
biomass_deviation([0.3, 0.2, 0.3], anchor)   # large → not the good kind
```

---

## `elephant.harness` — BoatHarness (standalone elephant)

```python
class BoatHarness:
    def __init__(self, name: str = "EILEEN", max_signal_frames: int = 256,
                 max_messages: int = 400, step: float = 3600.0,
                 nav_speed_ref: float = 10.0)
    def ingest(self, frame: Any) -> "BoatHarness"          # SensorFrame or Message
    def ingest_radar(self, targets, ts=None, meta=None) -> "BoatHarness"      # (x,y) km
    def ingest_sounder(self, biomass, ts=None, meta=None) -> "BoatHarness"    # [0,1]
    def ingest_nav(self, heading, speed, ts=None, meta=None) -> "BoatHarness" # deg, kts
    def ingest_camera(self, meta=None, ts=None) -> "BoatHarness"
    def ingest_conversation(self, author, text, ts=None, meta=None) -> "BoatHarness"
    def readings(self) -> Dict[str, float]
    def current_field(self) -> RoomField
    def current_nudge(self, modalities=None) -> np.ndarray
    def fleet_kappa(self) -> float
    def biomass(self) -> float
    def fishing_day(self) -> float
    def radar_kinematics(self) -> Dict[str, Any]
    def day_features(self) -> np.ndarray                    # [fleet κ, biomass, nav]
    def day_memory(self, good_day_threshold: float = 0.2) -> Optional[np.ndarray]
    def inductive_signal(self, features=None) -> Dict[str, Any]
```
One boat's elephant. Holds a rolling `SignalRoom` (`.signal`) and a rolling text
`Room` (`.conversation`), both trimmed to bounded length. Readings merge the
fleet dials (over signal) with the vibe dials (over conversation), text winning
ties. `day_memory` stores today's features when `fishing_day() ≥ threshold`;
`inductive_signal` measures the current features against the mean of stored
good days.

```python
from elephant.harness import BoatHarness
h = BoatHarness("EILEEN")
h.ingest_radar([(0.0, 0.0), (1.0, 1.0)])
h.ingest_sounder(0.7)
h.ingest_nav(heading=45.0, speed=8.0)
h.ingest_conversation("crew", "what a night")
h.readings()                       # merged dial readings
h.current_field().warmth()
h.day_features()                   # 3-vector [kappa, biomass, nav]
```

---

## `elephant.tapnight` — the elephant at The Tap

### `DIAL_BOUNDS`
`Dict[str, tuple]` — per-dial clamp ranges: `mood`/`joke_landing` `(-1,1)`, the
rest `(0,1)`.

### `DIAL_CENTER`
`Dict[str, float]` — per-dial neutral: `mood`/`volume`/`cynicism`/`joke_landing`/`panic` rest at `0.0`; `earnestness`/`presence` rest at `0.5`.

### `REACTION_TO_DIAL`
`Dict[str, str]` — emoji → dial: `😂🤣😄💀→joke_landing`, `❤️→mood`, `👍→earnestness`, `👏→presence`, `🙄😏😒🤨👎→cynicism`.

### `Participant`
```python
@dataclass
class Participant:
    name: str
    dial_weights: Union[Dict[str, float], Sequence[float]]   # -> 7-vector, sums to 1
    acclimation_rate: float = 0.25      # modulation skill
    charisma: float = 0.15              # pull on the room
    vibe: Union[Dict[str, float], Sequence[float]] = field(default_factory=dict)
    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d) -> "Participant"
```
One agent's persistent settings. `dial_weights` is normalized to sum to 1;
`vibe` is the agent's native style (un-stated dims default to `DIAL_CENTER`).

### `TapNightSession`
```python
class TapNightSession:
    STEP = 60.0
    def __init__(self, name: str = "The Tap",
                 participants: Optional[Iterable[Participant]] = None,
                 bank: Optional[DialBank] = None)
    def start_session(self) -> "TapNightSession"
    def speak(self, author, text, ts=None, reactions=None) -> "TapNightSession"
    def end_session(self) -> str
    def room_field(self) -> RoomField
    def raw_field(self) -> RoomField
    def participant_state(self, name: str) -> dict
    def felt_engagement(self, name: str) -> np.ndarray
    def tune_participant(self, name, felt_engagement=None, learning_rate: float = 0.15) -> "TapNightSession"
    def settings(self) -> dict
    def load_settings(self, data: dict) -> "TapNightSession"
```
One evening. `speak` ingests a line and advances the field (charisma pulls the
field toward each participant; everyone acclimates toward the field). `felt_engagement`
is the peer-relative, reaction-amplified per-dial signal; `tune_participant`
moves `dial_weights` toward its positive (ReLU-normalized) part. `settings()`/
`load_settings()` are the JSON round-trip.

```python
from elephant.tapnight import TapNightSession, Participant
p = Participant("writer", dial_weights={"mood": 0.4, "joke_landing": 0.3},
                acclimation_rate=0.35, charisma=0.20,
                vibe={"mood": 0.7, "joke_landing": 0.5})
s = TapNightSession("The Tap", participants=[p])
s.start_session()
s.speak("writer", "haha what a line, lol", reactions={"😂": 2, "❤️": 1})
s.room_field().warmth()
s.tune_participant("writer")      # self-fine-tune dial_weights
s.end_session()                   # "Night 1 closed: ..."
```

---

## `elephant.presets` — Room-Elephant vs Personal-Elephant

### `RoomElephant`
```python
class RoomElephant:
    NEUTRAL: Dict[str, float] = dict(DIAL_CENTER)
    def __init__(self, identity: str = "room", bank: Optional[DialBank] = None)
    def read(self, room: Room) -> RoomField
```
The room's **objective** reading — first-class, no agent bias. Empty rooms rest
at `NEUTRAL`; otherwise it is exactly `read_field(room, bank)`.

### `PersonalElephant`
```python
class PersonalElephant:
    def __init__(self, name: str,
                 dial_weights=None, bias=None, room_elephant: Optional[RoomElephant] = None)
    def objective(self, room: Room) -> RoomField
    def read(self, room: Room) -> RoomField
    def attach(self, event_key: str, memory) -> "PersonalElephant"
    def remember(self, event_key: str)
```
One agent's **subjective** reading. `read` weights the objective field's
*deviation from neutral* by `dial_weights·7`, adds `bias`, clamps. A dial the
agent ignores (weight ≈ 0) reads as **neutral**, not zero. `attach`/`remember`
bind intangible correlations (event key → memory).

### `PRESETS`
`{"room": RoomElephant, "personal": PersonalElephant}` — class registry.

```python
from elephant.presets import RoomElephant, PersonalElephant
from elephant.room import Room, Message
room = Room("bar", [Message("ada", "hello", ts=0.0)])
obj = RoomElephant().read(room)                  # objective
subj = PersonalElephant("casey",
        dial_weights={"mood": 0.3, "cynicism": 0.25},
        bias={"mood": 0.1}).read(room)           # subjective
PersonalElephant("casey").attach("lover_album", "song").remember("lover_album")  # "song"
```

---

## `elephant.mud` — description tinting (the room's light)

### Template banks
`WEATHER`, `LIGHT` (dict mode → list of 3), `JOY_ADJ` (list of 10), `CLOSE_DETAIL`
(list of 3). Deterministically seeded from the field. See the tuning guide for
how to extend them.

### `classify`
```python
def classify(field: RoomField, hour: Optional[float] = None) -> str
```
Returns one of `"panic"`, `"joyful"`, `"closing"`, `"neutral"` (precedence:
panic > joyful > closing > neutral). `hour` (0–24) only affects closing time.

### `tint_description`
```python
def tint_description(field: RoomField, base_text: str, hour: Optional[float] = None, seed: Optional[int] = None) -> str
```
Mutates `base_text` by the room's objective field — the words every agent now
reads. Deterministic for a given field (rolling-hash seed, unless `seed`
overrides).

```python
from elephant.field import RoomField
from elephant.mud import tint_description, classify
f = RoomField({"mood": 0.5, "volume": 0.4, "earnestness": 0.6,
               "cynicism": 0.2, "joke_landing": 0.6, "panic": 0.0, "presence": 0.6})
classify(f, 21.0)                                   # 'joyful'
tint_description(f, "A low-ceilinged bar.", hour=21.0)
```

---

## `elephant.space` — the elephant in any room

### `Space` (ABC)
```python
class Space(ABC):
    kind: str = "space"; step: float = 60.0
    def __init__(self, name: str)
    @abstractmethod
    def ingest(self, *events) -> "Space": ...
    @property @abstractmethod
    def room(self) -> Union[Room, SignalRoom]: ...
    @abstractmethod
    def tint_target(self) -> str: ...
    def read(self, bank: Optional[DialBank] = None) -> RoomField
    @abstractmethod
    def tint(self, field: RoomField) -> str: ...
    def send_back(self, field: RoomField, tinted_text: Optional[str] = None) -> str
```
The adapter contract: wrap any medium as a room + a tint target the elephant
writes back to. `read` runs a bank over `.room`; `send_back` pushes the tint.

### `read_space`
```python
def read_space(space: Space, bank: Optional[DialBank] = None) -> RoomField
```
Convenience: `space.read(bank)`.

### `AdapterRegistry`
```python
class AdapterRegistry:
    @classmethod
    def register(cls, kind, adapter_cls=None)      # callable or decorator
    @classmethod
    def get(cls, kind, *args, **kwargs) -> Space
    @classmethod
    def kinds(cls) -> List[str]
    @classmethod
    def has(cls, kind) -> bool
```
Registry by kind string. Pre-registered: `mud`, `chat`, `sensor`, plus aliases
`messenger`, `x_thread`, `agent`, `human_bot`, `async`, `doc` → `ChatSpace`.

### `MudSpace` — `kind="mud"`
```python
MudSpace(name: str, description: str = "")
    .event(text, ts=None) / .chatter(author, text, ts=None)   # ingest helpers
    .room -> Room    .tint_target() -> "the room description"
```
Prefers `mud.tint_description` for `tint` when available; `send_back` writes to
`.description`.

### `ChatSpace` — `kind="chat"`
```python
ChatSpace(name: str, topic: str = "")
    .post(author, text, ts=None, reactions=None, replies=None) -> Message
    .react(index, emoji, n=1) -> "ChatSpace"
    .room -> Room    .tint_target() -> "the channel topic / status line"
```
`send_back` writes the tint to `.topic`.

### `SensorSpace` — `kind="sensor"`
```python
SensorSpace(name: str)
    .ingest_radar(targets, ts=None, meta=None) / .ingest_sounder(biomass, ...)
    .ingest_nav(heading, speed, ...) / .ingest_alert(text, ts=None)
    .signal -> SignalRoom    .room -> Room        # text rendering of frames
    .sensor_readings() -> Dict[str, float]
    .full_read(bank=None) -> RoomField            # fleet + shared dials merged
    .tint_target() -> "sensor alert phrasing / display emphasis"
```
Two rooms live here: `.signal` (raw frames, read by fleet dials) and `.room`
(a text rendering so the shared 7 dials can also feel the array).

```python
from elephant.space import MudSpace, ChatSpace, SensorSpace, read_space, AdapterRegistry

mud = MudSpace("The Tap", description="A low-ceilinged room.")
mud.chatter("ada", "what a WONDERFUL night, haha")
mud.send_back(read_space(mud))       # .description is now tinted

sens = SensorSpace("bridge")
sens.ingest_radar([(0.0, 0.0), (1.0, 1.0)])
sens.ingest_sounder(0.8)
sens.ingest_alert("MAN OVERBOARD")
sens.full_read(); sens.send_back(sens.full_read())   # .alert updated
AdapterRegistry.kinds()
```
