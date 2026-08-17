# Fleet Operations — running and reading the elephant at scale

This is the interpretation side of the elephant: how to run the fleet
simulation, how to read the numbers it produces, and how to run the Tap
practice loop that tunes the whole thing. Every number quoted here is from
the shipped examples, run as-is (`seed 7` for the fleet, `seed 42` for the
Tap).

Companion reading: `docs/fleet-simulation-notes.md` (the sim), `docs/
deployment-guide.md` (how to wire it), `docs/tap-night-operations.md` (the
Tap runbook).

---

## 1. Running the fleet simulation

```bash
cd /home/eileen/projects/elephant
python3 examples/fleet_simulation.py      # 4 boats, 30 days, seed 7
python3 tests/test_fleet_simulation.py    # the assertions over the arc
```

The sim drives four boats (`EILEEN`, `PETREL`, `FULMAR`, `SHEARWATER`,
reputations 1.7 / 1.0 / 0.9 / 0.6) through a 30-day fishing arc, one
`BoatHarness` per boat, and computes the fleet field over their broadcast
scalars. The anchor feature is `[kappa, mean_biomass, catch]` where **catch
is exogenous** (drawn from the phase schedule, never a dial).

### What each phase means

**Days 1–7 — GOOD.** Boats bunch on a slowly-tightening drag around the
ground point (12, 8) km. The daily line reads, e.g. day 1 `κ 0.44`, day 7
`κ 0.88`. The meta-room warms, the sounder runs thick, catch is high. Seven
catch-good days become the anchor (a Gaussian over the `[κ, biomass, catch]`
features).

**Days 8–14 — SPOTTY.** Each boat drifts outward from its day-7 position
along its home bearing, searching at 6–8.5 kts over thin marks. κ collapses
(day 8 `κ 0.26` → day 14 `κ 0.05`), warmth goes deeply negative, catch dies.
`deviation` balloons — the elephant feels "this is not the warm room"
*without ever being told fishing is bad*.

**Days 15–30 — RECOVERY.** Boats blend from their day-14 positions back to a
tight home formation. κ climbs (day 18 `κ 0.77` → day 30 `κ 0.93`), warmth
recovers, and `deviation` drops back toward the anchor. This is the payoff:
the elephant recognizes the warm room returning *from the shape of the
field*, before it can be explained.

### How to read the per-phase numbers

The seeded run reports these phase means:

| Quantity | Good (1–7) | Spotty (8–14) | Recovery (15–30) |
|----------|-----------|---------------|------------------|
| fleet κ (raw / effective) | **0.63 / 0.22** | **0.11 / 0.08** | **0.76 / 0.17** |
| meta-room warmth | **+0.66** | **−0.83** | **+0.41** |
| exogenous catch | **0.81** | **0.12** | **0.71** |
| per-boat `fishing_day` | +0.64…+0.69 | −0.82…−0.88 | +0.40…+0.46 |
| inductive deviation (σ) | **1.35** | **12.17** | **3.50** |

How to read each column:

- **κ** — fleet tightness, `1/(1+spread_km)`. Near 1 = boats agreeing on
  where the fish are (warm); near 0 = disagreement (uncertain). The
  *effective* κ applies the damping bell (`κ·(1−clip(κ,0.2,0.8))`) — it is
  always smaller and is the number the driving loop actually uses.
- **warmth** — the meta-room temperature, `0.5·fishing_day + 0.3·(2·biomass−1)
  + 0.2·radar_coherence`. A warm fleet runs positive; the spotty week dives
  to **−0.83** (the cold plunge).
- **fishing_day** — each boat's *local* composite luck dial. It never leaves
  the boat as a float; only its bin `{-1, 0, +1}` crosses the wire.
- **deviation** — the Mahalanobis distance of today's field from the
  good-week anchor, in "good-day sigmas." **1.35** (good) → **12.17**
  (spotty) → **3.50** (recovery). The report labels a phase "feels like the
  good kind" when `deviation < 0.5 × spotty_deviation`.

The 30-day arc in one breath: **κ 0.79 (days 1–7) → 0.06 (days 12–14) →
0.94 (days 24–30)**, deviation **spotty 12.2 vs recovery 3.5**, anchor mean
**`[0.632, 0.819, 0.813]`** from 7 good days.

The per-day nudge prior (7 modalities) tells the same story: the good week
averages `radar=+0.69, sounder=+0.82, conversation=+1.00`; the spotty week
flips to `radar=−0.90, conversation=+0.30`. The elephant stops telling the
vision model to look at the radar cluster, and starts saying there is nothing
there worth comparing.

### What the dark-boat event demonstrates

On **day 20**, `EILEEN` (reputation 1.7, the best skipper) stops
broadcasting at `(12.3, 8.3)` km. The fleet field then uses the **3 active
boats' positions + one virtual point at her last position, weighted
3 × 1.7 = 5.1**. The remaining boats' recovery target becomes that last
position — they re-group on **her** mark, not the old ground point.

What it proves: **charisma is field displacement.** A strong presence that
goes quiet should still pull the room toward where it was — not read as a
thinning of the meta-room. The virtual point is reported explicitly (it is
the charisma rule working, not a hidden metric distortion). Set
`dark_boat_event=False` and the rule never fires; all four boats broadcast
all 30 days.

### How to extend the sim

- **More boats** — `FleetSimulation(n_boats=6, seed=7)`; home geometry
  generalizes (`_home_dir` places boat i at angle `2πi/n`), and boat names /
  reputations wrap by index modulo.
- **Real tracks** — the clean seam is `_simulate_day`: replace the
  phase-scheduled `pos`/`heading`/`speed`/`biomass` draws with your own AIS /
  log data, keep the harness feeding (3 sensor reads at 0h/8h/16h + one
  conversation line/day), and keep `catch` exogenous. The fleet field,
  anchor, and deviation machinery are unchanged.
- **A different anchor window** — `run(days=30)` fits the anchor from
  `features[:min(7, len(features))]`. Change that slice to match your real
  good-week horizon; the covariance uses OAS shrinkage
  (`elephant/fleetmath.py::_oas_shrinkage`) so it stays well-conditioned for
  N ≲ d (a week of good days).

---

## 2. Interpreting the elephant in production

### A warm room vs a cold one

The field is the ensemble of all seven dials. In the single-boat demo
(`examples/fleet_harness_demo.py`), the good hour reads
`RoomField(warmth=+0.50, κ=2.09)` with `mood=1.0, joke_landing=1.0,
cynicism=0.0, fishing_day=+0.83`; the spotty hour reads
`RoomField(warmth=−0.55, κ=3.62)` with `mood=−1.0, cynicism=1.0,
fishing_day=−0.29`. The two rooms are the sauna and the cold plunge — the
same dials, inverted.

Operationally: a **warm room** feels `mood` and `joke_landing` up, `cynicism`
and `panic` down, warmth positive; a **cold room** is the mirror. The demo
spaces (`examples/demo_spaces.py`) give three worked contrasts in one run:

- warm MUD bar — warmth **+0.45**, κ **1.91**, `mood=+1.0, joke_landing=+1.0`
- heated chat thread — warmth **−0.62**, κ **3.68**, `mood=−1.0, cynicism=+1.0`
- quiet sensor deck — warmth **−0.05**, κ **2.02**, all seven dials near rest

### When to trust κ vs warmth

They measure different things and disagree usefully.

- **κ (`concentration()`)** is *tightness* — how many ways there are to be in
  the room. A cold room is one way (high κ, rigid); a warm room is many ways
  (low κ, loose). Watch κ **fall** as regulars settle a vibe: at the Tap it
  drifts from ~1.7 down toward ~1.5 on the good nights.
- **warmth** is *valence* — whether that one-or-many ways is good or bad.

They can diverge: a room can be tight *and* warm (a crew locked onto a hot
drag — high κ, positive warmth) or loose *and* cold (a scattered fleet
searching alone — low κ, negative warmth). **Trust warmth for "is this good
right now," trust κ for "how rigid/settled is this room."** The single-boat
good hour is κ 2.09 *and* warm +0.50 — tight, and good. Do not treat κ alone
as health.

### How acclimation and charisma show up in the numbers

Both are dynamics, and both are visible:

- **Acclimation** (`field.py::acclimation_curve`) — an agent relaxes toward
  the room at rate = their `acclimation_rate` (1/τ). A high-rate agent is
  indistinguishable from the room within an evening; a low-rate agent keeps
  their own shape. In the Tap the captain (rate 0.40) warms to the room
  fastest; the engineer (rate 0.15) stays the dry voice longest.
- **Charisma** (`field.py::charisma_pull`) — a strong presence pulls the room
  toward *them* over interactions. Measure it as the gap between
  `room_field()` (effective, charisma-displaced) and `raw_field()` (dial-only):
  that gap is the sum of the room's strong presences bending the field toward
  themselves. In the Tap the captain (charisma 0.30) is the one whose dial
  "pulled the field" on 12 of 14 nights.

### The chemistry of agents — why two agents get along one day and not the next

Two agents can read warm together one evening and cold the next with no
change in either's settings, because **engagement is peer-relative and the
room field is what moves**. A single evening's field is the sum of
everyone's vibe, displaced by charisma and shaped by the works read aloud;
that aggregate swings night to night.

The Tap demo shows it in the raw record: night 2 `warmth +0.13, κ 1.53`
(the room warmed), night 3 `warmth −0.04, κ 1.74` (the room cooled), night 8
`−0.10, κ 1.95`, night 13 `−0.14, κ 2.13`. The same six people, the same
priors — and the room moves. **The married-couple off day is not a bug in
either agent; it is the field they are jointly generating.** If you want
stability, look at the *inputs* (what was read, what reactions landed) and
the *long-run* divergence (below), not at any one night's warmth. Do not
hand-tune a participant because one night read cold — the whole design is
that the settings are *discovered* by running cycles, not designed top-down.

---

## 3. The practice loop — running Tap nights

The elephant becomes a production tool only through **many cycles**. Each
evening: feed the works → read the field → self-tune once per participant →
persist. `examples/tapnight_cycles.py` runs 14 evenings with 6 personalities
(writer, poet, essayist, engineer, critic, captain) and prints the diverged
taste table. The loop is the one from `docs/tap-night-operations.md` §4:

```python
for evening in range(1, N + 1):
    session.start_session()
    for author, text, reactions in load_evening(evening):
        session.speak(author, text, reactions=reactions)
    field = session.room_field()
    for name in session.participants:
        session.tune_participant(name)     # self-fine-tune dial_weights
    save_settings(session.settings())      # persist between evenings
```

### Reading the diverged taste table

After 14 nights the table reads (top dial + weight):

| personality | top dial (weight) | accl | charisma |
|-------------|-------------------|------|----------|
| writer | mood **0.53**, joke_landing 0.44 | 0.35 | 0.20 |
| poet | volume **0.60**, mood 0.29 | 0.25 | 0.15 |
| essayist | earnestness **0.74** | 0.30 | 0.10 |
| engineer | cynicism **0.69** | 0.15 | 0.25 |
| critic | cynicism **0.60**, joke_landing 0.35 | 0.20 | 0.18 |
| captain | mood **0.73**, presence 0.21 | 0.40 | 0.30 |

The tell is the **mean pairwise distance of `dial_weights`**:
**initial 0.389 → final 0.859 (tastes diverged)**. Each personality settles
onto a different (or at most lightly shared) dial — the captain's
"different guitarists" made visible. Note the interesting overlap: engineer
and critic both land on cynicism, but the engineer's top weight (0.69) is
sharper than the critic's (0.60), and their second dials differ (earnestness
vs joke_landing) — divergence is about the *whole* 7-vector, not a single
argmax.

### Folding insights back into tuning

- **Watch the *trend*, not a night.** `learning_rate=0.15` is an exponential
  moving average: each evening moves a dial a little, and structure (who is
  *systematically* distinctive on which dial) is what survives the noise.
  Judge after dozens of evenings, not two or three.
- **If a taste looks "wrong" after many cycles, fix the inputs, not the
  knobs.** The self-tuning is driven by `felt_engagement()` — peer-relative
  (`vibe − cast_mean_vibe`) amplified by reaction heat (😂→joke_landing,
  ❤️→mood, 🙄→cynicism…). If a participant isn't diverging, the works or the
  reactions aren't giving them a distinctive signal.
- **Feed reactions in.** `joke_landing` and the self-tuning both lean on
  reaction heat. Without reactions the system still works (engagement rests
  on the peer-relative vibe alone), but "did the joke land" becomes real
  only when the crowd's hands are in the log.
- **Keep regulars registered.** Auto-registration (neutral defaults) is for
  drop-ins; the regulars' settings must persist to accumulate learning.
- **Seed for reproducibility.** The demo seeds `np.random.default_rng(42)`;
  seed your own run to reproduce the table above.
- **Know the v0 caveats.** The dials are hand-crafted and keyword-matching:
  they can saturate (a room of earnest writers reads `earnestness ≈ 1.0`)
  and cannot detect sarcasm ("great." reads warm). The peer-relative signal
  is designed so divergence survives that naivety; v1 trained dials will
  sharpen it.

---

*Run the demos before trusting any of this — the elephant is only real when
you walk into a different room and it is a very different elephant.*
