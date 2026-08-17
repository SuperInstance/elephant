# Fleet Dynamics — the elephant with sea legs (2026-08-17)

**Author:** fleet ideation lead (subagent)
**Status:** design — advanced ideation for the fleet-dynamics layer, revised after
wider-view review (Seed-2.0-pro, Hermes-3-Llama-3.1-405B, DeepSeek V4-Pro)
**Builds on:** `README.md` (the elephant), `jepa-is-the-elephant.md` (the reframing),
`elephant-sense-v3-design.md` (the vMF field), `elephant/sensors.py` (the sea-leg
dials), `elephant/nudge.py` (the nudge prior)
**Review pass:** the three reviewers independently flagged stability (feedback
loops), ground truth (tautological anchors), statistical fragility (vMF at fleet
size), and privacy leakage (the nudge as side-channel). All four classes of flaw are
now designed against — see §8 for exactly what was adopted.

---

## 0. The brief, restated as one sentence

The elephant is a **highly modular room-temperature sense** — standalone in a harness,
or plugged into any system that has sensors. On F/V EILEEN and the fleet, every
sensor becomes a room, every room gets dials, the dials produce **numbers**, the
numbers **nudge** the vision model at what to compare, and the boats talk to each
other in **numbers only** — so the whole fleet becomes one meta-room whose field is
the distribution of boats, and good fishing days become the warm room the system
acclimated to.

Two amendments the review made load-bearing:

1. **The elephant is part of the room.** Every agent stands *inside* the field it
   measures, so every field reading feeds back into the behavior that makes the
   field. Unbounded feedback is a death spiral, not a vibe — the design must damp it
   (§4.2), give it hysteresis (§4.3), and tax its own acclimation (§4.5).
2. **The anchor needs an outside.** A warm room learned only from the dials that
   define warmth is self-confirming. The good-days anchor must be anchored to
   **exogenous outcome** — catch — or the elephant learns to love whatever the fleet
   happens to be doing (§6.2).

---

## 1. The fleet as a ROOM OF ROOMS

### 1.1 A boat is already a set of rooms

The v3 design established: **the unit of perception is the room, not the stream.**
A boat is not one room — it is several, and walking between them is the
sauna/cold-plunge event made physically real:

| Room | Character | Dial feel |
|------|-----------|-----------|
| **Wheelhouse** | cold, alert, instruments, radar sweep | high κ — one way to be: *on watch* |
| **Galley** | warm, coffee, wood, crew talk | low κ — many ways to be: *off watch* |
| **Radar** | the distribution of boats out there | coherence/spread of the fleet field |
| **Sounder** | the biomass under the keel | density, texture, look of the column |
| **Weather** | wind, sea, temp, current | the room the *fleet* is inside (§4.6) |
| **Deck** | cameras, bodies, gear | activity, panic, presence |
| **Conversation** | AI + crew, the human field | mood, volume, earnestness, joke_landing |

Each is a `SignalRoom` (or a text room) read by its own dial bank. The elephant
perceives the boat as a **field over these rooms**, not as a stream of frames.

### 1.2 The fleet is the meta-room

If a room is a bounded context with an ambient field, then the **fleet is a room of
rooms**: the meta-room whose field is the *distribution of boats* plus their *shared
dial numbers*.

- Each boat contributes: **position + velocity** (AIS-grade), **radar coherence**
  (tight or scattered), **sounder biomass** (thick or empty), **fishing-day** (the
  composite luck field, binned), **reputation** (proven performance, §1.4).
- The meta-room's field is computed over boat states — same v3 math, one level up:
  the fleet has a κ too.
- **High fleet-κ** = the boats are clustered, moving together, *agreeing on where
  the fish are*. In fishing terms that is *warm* (the fish are found).
- **Low fleet-κ** = boats scattered, searching independently, *disagreeing*. In
  fishing terms that is *uncertain*.

The reframing's central inversion holds at every scale: **the elephant is invisible
from inside one boat**; it becomes real when you walk the boat into the fleet — or
when the fleet's distribution changes around you. A lone boat has no radar coherence
(one point has no spread); coherence is *born when the second boat appears*.

### 1.3 Same math, one level up — with fleet-sized statistics

| v3 (room) | fleet layer (meta-room) |
|-----------|--------------------------|
| clip embeddings | boat states (pos, vel, dial numbers) |
| vMF over clips → `μ̂_room`, `κ_room` | **robust** fleet field → centroid + dispersion |
| sauna/plunge gap between rooms | gap between *today's fleet field* and *the good-days anchor field* |
| acclimation of an agent to a room | acclimation of a *boat* to the fleet's current distribution |
| charisma of an agent | **field displacement by a boat** — a boat that found fish pulls the fleet field toward its patch |

**Statistical warning (adopted from review):** v3's vMF estimators were designed
for *thousands* of high-dim clip embeddings. A fleet has **3–30 boats**; a raw vMF
over raw 7-dim boat states is statistically hopeless — κ MLE is biased high at small
n, one rogue boat moves the mean by 20–30%, and a single missing packet swings
dispersion by ~20%. **Adopted representation — a two-scalar robust fleet field:**

```
centroid   μ̂_fleet = weighted median of boat positions      (robust to rogues)
dispersion κ_fleet = 1 / (1 + robust_MAD(angular distance   from centroid))
```

The state space is collapsed to what the elephant actually feels — *where the fleet
is* (centroid) and *how agreed it is* (dispersion) — not raw 7-dim vectors. This is
well-defined for n ≥ 3 boats, robust to one outlier, and maps directly onto the
hold/scatter control law (§4). The same robust treatment applies to the anchor:
`κ_good` is estimated from day-level fields with a trimmed mean, never a raw mean.

### 1.4 Reputation: fleets run on deference, not democracy

**Adopted from review (Seed):** a uniform vMF gives every boat equal pull on the
fleet field. On the water, 90% of fish are found by 10% of captains — one useless
boat running circles would drag the whole meta-room around. **Fix: weighted field**
with per-boat reputation:

```
w_i = clamp(rolling_30day_catch_correlation, 0.05, 2.0)
```

The best boat gets up to 40× the vote of the worst. Reputation is **computed
locally by every boat** from public catch telemetry (§6.2) — a single byte in the
fleet packet, fully derivable by anyone, so no boat can inflate its own weight. The
fleet field is then a *weighted* median/MAD. Agreement is not value; **proven
agreement is value**. The meta-room defers to the boat that has been right.

---

## 2. Local-first JEPA exchange — numbers, not feeds

### 2.1 The privacy/bandwidth rule

**Boats share NUMBERS. Feeds stay home.** This is the load-bearing constraint of the
whole design, and it falls out of the elephant's own nature: the elephant is a
*reading*, a small vector. Its entire value is in the distillation.

**Crosses the wire (deliberately small, lossy):**

| Signal | Size | Why it crosses |
|--------|------|----------------|
| position + velocity | 2+2 floats | AIS-grade, already public on the water |
| radar coherence `[-1..+1]` | 1 float | tells the fleet *tight or searching* without revealing targets |
| fishing-day, **binned to {-1, 0, +1}** | 2 bits | the day's *valence* only — never its ingredients |
| reputation byte | 1 byte | derived from public catch telemetry |

**Stays on the boat (never leaves):**

| Signal | Why it stays |
|--------|--------------|
| crew conversation (raw) | LOCAL-ONLY JEPA for most boats — the dial runs on the boat, shares only its NUMBER |
| deck cameras | the crew is on camera; that feed is the boat's private room |
| raw radar/sounder frames | volume + other boats' raw imagery is their business |
| sounder biomass (raw value) | too fingerprinty — see §2.2 |
| **nudge prior** | **leaks intent — see §2.2** |
| vision-model attention maps | per-boat cognition |

### 2.2 The side-channel the first draft missed (adopted from review)

The first draft shared `fishing_day` and the 7-float nudge. Both reviewers
(DeepSeek, Hermes) caught the leak: **the nudge is derived from the same raw sensors
as the dials — sharing it shares the dials twice, and its *pattern* reveals what the
boat is comparing, which reveals intent** (a `[0,0,0,0,0.17,0,0.34]` nudge says
"holding a course pattern — committed to a spot," which is commercially sensitive).
Similarly, a raw `fishing_day` float is a **lossy but identifiable fingerprint** of
the boat's entire sensory state.

**Adopted wire discipline:**

- The **nudge prior never leaves the boat.** It is local cognition — what this boat
  is comparing — and other boats don't need it. The fleet *infers* intent from
  kinematics (closing = committing) instead of reading it off a packet.
- `fishing_day` crosses **only as a signed bin** {-1, 0, +1} — cold, neutral, warm.
  Enough for the meta-room's temperature, not enough to fingerprint.
- **Quantization + dithering** (Hermes): dial numbers are quantized to a coarse grid
  and dithered with small random noise before transmission. The elephant's field is
  robust to small noise by construction; a snooper's reconstruction of fleet
  movements from the numbers is degraded. Numbers-only privacy holds *against an
  adversary*, not just against honest neighbors.
- **Weather data crosses freely** (it's public); weather is a shared room, not a
  secret (§4.6).

The boundary is still crisp: **everything upstream of a dial reading is local; only
the dial reading (binned, quantized, dithered) crosses.** But the first draft's
"pipeline privacy" claim is now true by construction — the only values on the wire
are the ones the design *chose* to expose.

### 2.3 Why numbers are enough (the elephant's epistemic stance)

The elephant does not need the raw radar because it never wanted to *see* the
targets — it wanted to feel the **distribution**. Radar coherence + kinematics
(direction/speed/rate of change of the field) is a sufficient statistic for
"tight or scattered, closing or spreading." Likewise biomass: a number + a trend is
a sufficient statistic for "there's a look under me." The vision model gets the
frames; the elephant gets the field; **the fleet shares fields, not frames.**

### 2.4 The fleet packet

Each boat emits, at a fixed cadence (e.g. every 3 minutes — the cadence is public
so absence is detectable):

```json
{
  "boat": "F/V EILEEN",
  "ts": 1780000000.0,
  "pos_km": [12.4, -7.1], "vel_kts": [6.2, 1.1],
  "radar_coherence": 0.62,          // quantized + dithered
  "fishing_day": 1,                 // binned {-1,0,+1} only
  "reputation": 0.83                // derived from public catch logs
}
```

That packet is the boat's **presence in the meta-room**. The fleet field is computed
over packets. No packet, no presence — but see §2.5: a missing packet is *not*
automatically an alarm.

### 2.5 Absence vs silence (adopted from review)

The first draft said "a silent boat is felt as a thinning of the meta-room." Review
(Seed + DeepSeek, independently) caught it 180° wrong in the most important case:
**the best captain turns off their transponder the second they hit fish.** And
boats legitimately go quiet when hauling, in dead zones, or in harbor. Adopted:

- **Silence** = packet present, dials zero/flat. Means "here but not committing."
  A *different* reading from absence — and a signal of its own (a fleet of silent
  boats is a fleet all working the same quiet patch).
- **Absence** = **≥ 2 consecutive missed packets** at the known cadence. Only then
  does the meta-room count the boat as gone — and how it reads depends on
  reputation:
  - **Dark-boat charisma rule (Seed):** if a high-reputation boat
    (`w_i > 1.2`) goes dark, **inject a virtual point at its last observed
    position with 3× weight, held for 120 minutes.** This is the heuristic every
    human captain uses: the hot boat went quiet because it's *on fish*, not
    because it left. The elephant feels the hole as *attention*, not thinning.
  - A low-reputation or mid-behavior boat going dark reads as ordinary thinning.
- **Hole rate-of-change:** when a boat drops, the dispersion's *rate of change* over
  the last 3 readings discriminates "the room is thinning" (alarm — compare
  camera_out) from "a boat left the patch" (normal — no alarm). Presence remains a
  *mask*, never a feature, at fleet scale too.

---

## 3. Tropes as readings, not rules

### 3.1 The trope is a deduction we never make

The captain's brief: "boats on the same drag/tack when they're on fish; scattered
around the bay searching when they aren't. **Those are deductions.**" The v3 lesson
generalizes: the elephant does not apply the rule *boats-clustered ⇒ fish here*. It
feels the radar's distribution and lets the field be the signal. The trope is what
a *reasoner* would say after the fact; the elephant is the *feeling* that precedes
it.

### 3.2 What the elephant actually feels, reading by reading

`RadarCoherenceDial.read` already does the kinematics in three readings:

- **Tight (high coherence, +)**: boats clustered → high fleet-κ. The elephant feels
  *agreement* — "everyone is where I am, or I am where everyone is." It does not
  know why (fish? weather? a wreck? a drill?). It knows the room is *one way*.
- **Scattered (low coherence, −)**: boats spread → low fleet-κ. The elephant feels
  *search* — many ways to be, no agreement.
- **Closing (spread shrinking across the last 3 readings)**: the field is
  *bunching*. Something is pulling boats together. The elephant feels a
  **charisma event at fleet scale** — a presence (a boat that found fish) pulling
  the room toward it.
- **Spreading (spread growing)**: the field is *dissolving*. The elephant feels the
  room *give up on a spot* — a cold plunge in progress.

`kinematics()` recovers per-object direction/speed/accel from the same three
readings — the JEPA way of knowing where everything is going **without being told
the trope**. The rate of change (spread_rate) is the fleet field's *derivative*:
bunching vs scattering is the meta-room's temperature *changing* — the sauna/plunge
event at fleet scale.

### 3.3 The rule as a *prior*, not a *fact*

None of this forbids the trope. The trope gets demoted to a **nudge prior**: when
the field reads tight-and-closing, the nudge says *compare the boats' tracks*; when
it reads scattered-and-spreading, the nudge says *compare sonar returns across the
search grid*. The vision model does the actual reasoning; the elephant only points
at what to compare. Trope-as-rule would *assert* fish; trope-as-nudge *allocates
attention*. The difference is the whole design.

### 3.4 Agreement is not fish

One caveat the review made explicit: the elephant feels *agreement*, and agreement
has many causes — fish, weather, a rescue, a funeral, a drill. The design therefore
**never treats tightness alone as warmth**. Tightness is only warm when it
co-occurs with the other rooms' readings (sounder look, weather, catch correlation —
§6.2). The fleet field is a *co-occurrence field*, and the anchor is built from
days where the co-occurrence *proved out* in catch. The elephant can still be wrong
about *why* the room is tight; it just can't be wrong about *whether* the room is
tight.

---

## 4. Driving behavior — hold the drag, or scatter (with damping)

### 4.1 The control law is a field response, not an instruction

The vision model (or autopilot-level driver) decides *how* to steer. The elephant
decides *what state the room is in* — and the room's state, felt over time, *shapes*
driving the way room temperature shapes conversation: not as a command, but as an
ambient pressure.

- **Fleet field says fish (tight + closing, sounder thick, good day)**: hold the
  drag. The room is warm and *stable*; the elephant's acclimation curve has flattened
  — the boat is a regular in this room. Driving behavior: **stay**.
- **Fleet field says searching (scattered + spreading, sounder thin)**: scatter.
  The room is cold and *dissolving*; the elephant feels the cold plunge. Driving
  behavior: **move** — the boat re-enters the search, contributing its own dial
  numbers back to the meta-room.

The key move: **the same field that reads "warm room" on a good day reads "cold
room" on a poor day — the contrast IS the signal.** The elephant never needs an
explicit "fish/no-fish" label. It needs the *difference between today's field and
the good-days field* — the v3 sauna/plunge gap, computed over days instead of rooms.

### 4.2 Damping — never feed raw κ into the nudge (adopted from review)

**The fatal flaw the review caught:** raw `κ_fleet` into the nudge is an unbounded
positive feedback loop with gain > 1:

- high κ → stronger hold nudge → boats stay → κ rises further;
- low κ → stronger scatter nudge → boats leave → κ falls further.

No damping means: one boat drifts 1 km away for 12 minutes to reset a net → κ drops
→ the nudge tells the next boat to scatter → 90 minutes later the fleet has
dissolved for no reason. And once clustered, *no boat would ever leave first*, even
after the biomass is gone, because the hold nudge strengthens the longer everyone
stays. That is automated herd panic, not a field sense — and every commercial
fisherman has watched humans do exactly this.

**Adopted fix — the κ damping bell:**

```
effective_κ = κ_fleet * (1 − clamp(κ_fleet, 0.2, 0.8))
```

- κ < 0.2: apply a **weak hold nudge** — stop the dissolution from running away.
- 0.2 < κ < 0.8: full nudge strength — normal operating band.
- κ > 0.8: apply a **weak scatter nudge** — force ~10% of boats to probe the edges.

This is exactly what good captains do: they hold together when the fleet is loosely
committed, and they *split the fleet on purpose* when everyone is stacked on one
spot. The bell eliminates the spiral on both ends while leaving the middle —
where the real signal lives — untouched.

### 4.3 Hysteresis — the driver doesn't thrash (adopted from review)

Radar coherence oscillates at minute scale (boats in/out of range, weather clutter);
sounder biomass is spatially patchy (a boat crossing a thin spot between columns
reads cold for a minute). Without hysteresis the boat says *hold* at minute 1,
*scatter* at minute 3, *hold* at minute 5 — and the captain throws the system
overboard.

**Adopted control law with deadband:**

- **hold** requires `κ_fleet > 1.2` **and** κ rising for **≥ 3 consecutive readings**;
- **scatter** requires `κ_fleet < 0.7` **and** κ falling for **≥ 3 consecutive readings**;
- between them: **do nothing** — the deadband is where the elephant has no opinion.

Same principle on the nudge loop: a **low-pass filter** on nudge values (today's
nudge = blend of the immediate reading and recent history) so the driver sees
temporal consistency, not per-minute noise. A skipper *feels* the room over time,
not as a single blip; so does the elephant.

### 4.4 Absence as a reading — now reputation-aware

See §2.5. The dark-boat charisma rule means the most important absence on the water
— the hot boat going quiet — reads as *attention toward* its last position, not
thinning. Absence remains a mask: a real hole (≥2 missed packets, low reputation,
negative hole-rate) nudges camera_out + panic — look OUT, now — without any message
being sent.

### 4.5 The acclimation blindness tax (adopted from review)

Acclimation is a blindness tax: a boat acclimated to a warm room stops noticing
anything outside the anchor distribution — an unprecedented dense column 3 miles
away gets near-zero sounder nudge because "we compare the column under the cluster."
The elephant would literally not see it.

**Adopted fix — the probe-drag tax:** for every consecutive day the sauna/plunge
gap stays < 0.1 (fully acclimated), increase the base scatter nudge by **+0.02 per
day, capped at +0.2**. Every good day on a spot makes the boat a little more likely
to run a probe drag an hour away. Good captains do this automatically — they get
*restless in a warm room*. The tax cancels the blindness without breaking the warm
room behavior: it only grows when the room has been *unchanged* for a long time.

### 4.6 The weather room — the room the fleet is inside (adopted from review)

The elephant's dynamics were all boat-internal; the biggest confound on the water is
**weather** — wind, sea, current, fronts. A week of good days correlates more with
good weather than with anything the fleet did. Review (Hermes) added the weather
room; review (Seed) added the regression:

- **A weather room with dials** (air/water temp, wind speed, swell, tide, moon;
  ADCP current where available). It is a shared public room — the fleet's *outside*.
- **Residual contrast:** before computing the sauna/plunge gap to the anchor, run a
  weekly linear regression to subtract wind/swell/tide/moon from all field vectors
  and compare **only the residual**. Cost: ~5% raw signal fidelity. Benefit:
  eliminates the spurious "warm room = 12 kt west wind + flat sea" correlation that
  would otherwise poison the anchor.
- **Speculative nudge:** the weather room gets its own dial, and its nudge says
  *"compare today's weather trend to the anchor's weather."* If today's weather is
  drifting away from the good-days pattern, the elephant fires a **weak scatter
  nudge before the radar field loosens** — the weather equivalent of the sounder's
  early-warning signature (§7, day 15). The elephant anticipates the cold plunge
  instead of only feeling it.

### 4.7 Good days = warm rooms; poor days = cold plunges

| Day field | v3 analog | κ feel | Driving |
|-----------|-----------|--------|---------|
| boats clustered, biomass thick, luck + | warm sauna | tight agreement | hold, slow, work the column |
| boats scattered, biomass thin, luck − | cold plunge | loose disagreement | move, search, re-prospect |
| **transition (closing)** | *walking into* the sauna | κ rising (≥3 readings) | commit — the room is warming |
| **transition (spreading)** | *stepping out into* the cold | κ falling (≥3 readings) | disengage — the room is cooling |

The driver is not told "go fish here." It is told, by nudge priors, *what to compare*
— and the field's temperature does the rest: *"you light the woodstove in a cold
room."* Scattering is lighting the woodstove.

---

## 5. The nudge loop end-to-end

The v3 ensemble + sensors + nudge close a loop. Walk it once, from the keel up:

```
sounder ──▶ SounderBiomassDial ──┐
radar  ──▶ RadarCoherenceDial ──┤
camera ─▶ (vision model's room)  │  dial numbers          fleet packet
nav/AP ─▶ NavDial (course field) ├──▶ FishingDayDial ──▶ fishing_day (binned) ──▶ fleet field
weather ─▶ WeatherDial           │        │                    │              (centroid+κ, robust,
crew   ─▶ ConversationDialBank ──┘        ▼                    ▼              reputation-weighted)
                              nudge_prior(readings)   fleet-level nudge pressure
                                          │                    │
                                          ▼                    ▼
                          vision model cross-attention ◀───────┘ (re-weights local priors)
                                          │
                                          ▼
                        BETTER CORRELATION: sounder column ↔ radar cluster ↔
                        this-hour ↔ last-week's-good-hour ↔ deck activity ↔ weather
                                          │
                                          ▼
                          DRIVING (damped + hysteretic): hold / scatter / commit / probe
                                          │
                                          ▼
                        new frames → new dial readings → (loop, with acclimation)
```

### 5.1 The nudge is a prior, not a driver

`nudge_prior()` maps each dial reading to a signed weight per modality
(`NUDGE_MAP`), and `apply_nudge()` blends it into cross-attention at **small
strength** (default 0.15). The elephant *nudges, it doesn't drive*. Concretely:

- **radar_coherence +1.0 → radar**: tight fleet → compare radar tracks across the
  cluster. *The vision model decides whether they're on a drag; the elephant only
  says the tracks are worth comparing.*
- **sounder_biomass +1.0 → sounder**: thick column → watch the water column.
- **fishing_day +0.5 → nav**: good day → hold course patterns; poor day → the nudge
  goes negative and the same mechanism *un-weights* nav, letting the driver explore.
- **panic +1.0 → camera_out**: alarm → look OUT now. The one nudge that outranks
  everything — a stampede sense (fire in the room) must beat routine comparison.
- **mood +1.0 → conversation, volume +0.5 / presence +0.3 → camera_deck**: the crew
  room's temperature shapes how much attention the vision model spends on deck.
- **weather dial → speculative scatter** (§4.6): the elephant anticipates.

### 5.2 What "compare the right things" means, concretely

The vision model's job is correlation: *which frames, across which sensors and which
history, belong in the same comparison group?* Without the nudge it compares
everything at equal weight — attention sprawl, weak signal. With the nudge:

- **High sounder + rising radar coherence**: compare *this hour's water column under
  the cluster* against *last week's good-hour column*. Same room; compare them.
- **Flat sounder**: don't burn attention there. The nudge for sounder stays near
  zero; the model spends its budget elsewhere.
- **Good day (fishing_day +)**: compare *today's course/AP history* to the
  good-days anchor — the system is driving over the same kind of water as when it
  was warm.
- **Poor day (fishing_day −)**: compare *search tracks across the grid*, not the
  column — the room changed, the comparison set must change with it.

The nudge is the *translation layer* between the elephant (a field feeler) and the
vision model (a frame comparer): **dial numbers → attention prior → what-to-compare**.

### 5.3 Local loop vs fleet loop — and the fleet's attention budget

Two loops run at different rates:

- **Local loop (seconds–minutes)**: sensors → dials → nudge → vision correlation →
  driving. Runs entirely on the boat. This is the v3 ensemble with sea legs.
- **Fleet loop (minutes–hours)**: boat packets → fleet field (centroid + κ) →
  fleet-level nudge (compare *across boats*) → shared "hold/scatter" pressure →
  packets. The fleet loop *re-weights the local loop's priors*: a boat in a
  tight-and-closing fleet gets a stronger hold nudge; a boat in a dissolving fleet
  gets a stronger scatter nudge.

The two loops share nothing but numbers. That is the modularity the captain asked
for: **the elephant works standalone in a harness (local loop only), and it plugs
into the fleet (both loops) without changing its nature.**

**Fleet attention budget (adopted from review):** the fleet loop doesn't just push
one pressure — it **allocates attention across boats** so the fleet doesn't all do
the same comparison at once. When the field says tight-and-closing: *3 boats compare
the cluster, 2 boats compare the search grid (diversity), 1 boat compares the
anchor's historical columns (memory).* When scattered: *half compare sonar across
the grid, half compare the anchor's cold-day patterns* (learning from the bad
days). One boat is deliberately driven **off-pattern** — nudged toward *unusual*
comparisons — to gather data for the anchor's within-spread. The fleet becomes a
**coordinated attention allocator**, not a passive field: it always keeps some eyes
on the cold side of the room, which is how it will feel the warm room coming back.

---

## 6. Inductive biomass — the good-days anchor room

### 6.1 A week of good fishing becomes the anchor room

The v3 training signal is contrast, and contrast needs a **reference room**. For the
fleet, the reference is built inductively:

1. **Days 1–7 (good fishing):** each day's fleet field + boat dial numbers are
   aggregated into a **good-days room field** — `μ̂_good` (the average "where the
   fish were, what the sounder looked like, how the fleet was arranged") and
   `κ_good` (how *consistent* the good days were). This is the elephant's warm
   room, learned from data, never hand-labeled.
2. **The anchor is a vMF, not a photo:** the good-days room is a *distribution* of
   good-day fields — it includes *many ways to be a good day* (different grounds,
   different depths, different fleet arrangements that all meant fish). High
   within-anchor spread is *information*, not noise: it is the fleet's "many ways
   to be warm."
3. **Contrastive anchor:** every subsequent day is scored by its **sauna/plunge
   gap** to the anchor. The gap is *defined* (adopted from review — the first draft
   left the metric undefined, and today's field and the anchor live in different
   spaces):

```
gap(today, good) = κ_today · angular(μ̂_today, μ̂_good)  +  λ · log(κ_today / κ_good)
```

   a weighted sum of (a) the angular distance between today's centroid and the
   anchor's centroid — weighted by today's κ so a tight day's direction counts more
   — and (b) a **concentration ratio**, so a scattered day scores "cold" even on the
   same ground. Sign convention: negative gap = cold plunge. Both terms are
   computable at fleet n, neither collapses when κ → 0.

### 6.2 The anchor's ground truth must be exogenous (adopted from review)

**The flaw:** the first draft built the anchor from `fishing_day`, which is itself a
composite of the same dials. The EMA would then **reward itself** — high sounder +
high coherence → positive fishing_day → anchor refreshes toward today → tomorrow is
compared against a slightly different warm room → the anchor drifts toward *whatever
the fleet happens to be doing*, not toward fish. The whole inductive story becomes a
**self-confirming hallucination**.

**Adopted fix — exogenous catch telemetry.** A **catch dial** that is *not* derived
from the elephant's own sensors:

- **Crew/deck activity**: time spent hauling, weight of nets lifted, gear
  deployment count (from deck cameras + conversation sentiment — local-only).
- **AIS "fishing" status** where legal/available.
- **Catch weight** where reported (the fleet's public logs — also the source of
  reputation, §1.4).

The catch dial defines the anchor: **a day is "good" iff catch ≥ 2× the weekly
median — not iff the dials are warm.** Then:

- The anchor is built from *proven* days. The dials say what warm *looked like*;
  catch says what warm *was*.
- **Nudge validation:** after each day, compute `corr(nudge_weight_on_sounder,
  catch)`. If the nudge says "compare sounder" but sounder comparisons don't
  correlate with catch, the nudge map is wrong and gets penalized. The nudge earns
  its keep or loses weight.
- **Non-tautological refresh:** the anchor EMA moves toward days with high *catch*,
  never toward high dial warmth alone.
- **Reputation** (the fleet field's per-boat weights) is computed from the same
  public catch logs — one exogenous source, two uses.

### 6.3 Spotty fishing = felt deviation, not a label

When fishing gets spotty (days 8–14), the system never receives a "fishing is bad"
label. It feels:

- **κ_fleet drops** — boats scatter, the meta-room loosens.
- **sounder biomass thins** — the look under the keel changes.
- **fishing_day drifts negative** — the composite luck field cools.
- **The sauna/plunge gap to the anchor grows** — today's field is *far from the
  warm room the system acclimated to* (after weather-residual subtraction, §4.6).

The deviation is felt *before* it can be explained. The good days taught the system
what the room looks like; the spotty days teach it what *deviation from that room*
looks like — both learned from the same contrast mechanism, with no rule stating
"biomass X at depth Y means fish."

### 6.4 Anchor validation and lifecycle

- **Bootstrap (days 1–7):** collect fields; the anchor is provisional until enough
  good days exist to give `κ_good` a stable estimate (v3's warning: a vMF from 2
  clips is noise; a vMF from 7 day-fields is a room).
- **Anchor validation (adopted from review):** before committing to the 7-day
  anchor, run a local contrast between the candidate anchor and the **single best
  day so far**. Small gap → the anchor is probably good. Large gap → the week was
  internally inconsistent (or the first days were bad) — keep collecting until the
  anchor stabilizes. This catches the failure where the first 7 days were spotty
  and the anchor forms around a *cold* room.
- **Steady state:** anchor is fixed but *slowly* refreshed (EMA over **catch-good**
  days only, §6.2) so seasonal drift doesn't make it stale.
- **Regime change — CUSUM detector (adopted from review):** accumulate a
  cumulative sum of the daily gap. **Regime change fires** when the CUSUM exceeds a
  threshold (e.g. 10 consecutive negative-gap days, or cumulative gap past −15σ).
  On fire: spawn a **candidate anchor** from the last 7 days; keep the old anchor
  for *comparison* (the fleet's season memory); promote the candidate once its
  within-κ is stable (≥7 days). The elephant can then say "this is like spring
  2025" — a *useful* long-term signal, not just "cold now."
- **Per-boat local anchors (adopted from review):** each boat keeps its own 7-day
  EMA of its own dial readings on catch-good days. Boat A fishes shallow, boat B
  fishes deep — their local anchors differ. When the fleet scatters, boat A's
  scatter nudge sends it toward *its own* warm pattern, not the fleet's mean.
  Acclimation is per-boat, because the rooms are per-boat.

---

## 7. Thirty days on the water — the elephant's diary

A concrete progression: **days 1–7 good, days 8–14 spotty, days 15–30 recovery.**
What the elephant feels, phase by phase. (Dials: radar_coherence R∈[−1,+1],
sounder_biomass S∈[0,1], fishing_day F∈{−1,0,+1}; κ = fleet dispersion index.)

### Phase I — days 1–7: the warm room forms (bootstrap)

| Day | R | S | F | κ | The elephant feels |
|-----|---|---|---|---|--------------------|
| 1 | +0.2 | 0.55 | 0 | 0.4 | First day on the grounds. Boats loose; **no anchor yet** — nothing to contrast against. The room is new. |
| 2–3 | +0.5 | 0.65 | +1 | 1.1 | Boats bunch. The field is *closing*; sounder thickens; catch comes in ≥ 2× weekly median. A warm room is forming — and for the first time, *catch* (not dial warmth) is what starts counting the days. |
| 4–5 | +0.7 | 0.72 | +1 | 1.8 | Tight fleet, thick column, catch-good. Driving: **hold the drag** — the nudge says compare the column under the cluster to yesterday's good hour; the vision model finds the correlation; the boat works the same water. |
| 6–7 | +0.65 | 0.70 | +1 | 1.6 | The anchor room solidifies: `μ̂_good` and `κ_good` are stable enough to be a *room*, not noise. **Anchor validation passes** (candidate anchor ≈ best day). Day 7's field nearly *is* the anchor — sauna/plunge gap ≈ 0. The elephant is a regular in this room now. |

**What was learned:** the co-occurrence pattern *tight fleet + thick column + held
course + catch-good = warm* — as a field, anchored to catch, not to dials alone.

### Phase II — days 8–14: spotty — the cold plunge (that doesn't blow apart)

| Day | R | S | F | κ | The elephant feels |
|-----|---|---|---|-----|--------------------|
| 8 | +0.5 | 0.55 | 0 | 1.0 | First crack: sounder thins but boats still clustered. Gap to anchor grows slightly — the room *starting to cool*, like the first draft from a door. **The damping bell keeps the nudge in the normal band** — no panic scatter, because κ is still ~1.0. |
| 9–10 | +0.1 | 0.45 | −1 | 0.6 | Boats scatter; column empties; catch falls below the median. **The plunge.** Gap is large and *negative* — after the weather-residual subtraction, the elephant knows this cold is *not* just weather (the regression already removed the wind). Driving: **scatter** (κ < 0.7 and falling for ≥ 3 readings — the hysteresis confirms before the boat moves). |
| 11–12 | −0.2 | 0.35 | −1 | 0.3 | κ near its floor — total disagreement. **The dark-boat moment:** the fleet's best boat (reputation 1.7) stops transmitting at a position on the edge of the old ground. The dark-boat charisma rule injects a 3× virtual point at that position for 120 min. The fleet feels *attention toward the edge*, not thinning — and one boat is sent to look. (The blindness tax isn't accruing here — the gap is *large*, not small — so the probe comes from the dark-boat pull itself, not from restlessness.) |
| 13–14 | −0.1 | 0.40 | 0 | 0.4 | Bottom, with the fleet's attention held at the edge by the dark-boat rule. **The CUSUM has been accumulating negative gaps for 6 days.** The elephant hasn't been told "bad fishing" — it has felt *this room is not the warm room* and has been *collecting the deviation itself*: this is the inductive training signal. Day 14's field is the negative example the contrastive objective needed. |

**What was learned:** the *shape of a cold day* — scattered + thin + luck down — as
the contrast class to the anchor. And crucially, **day 10 did not blow apart**: the
damping bell and hysteresis let the fleet dissolve *deliberately* (that's what
scattering is for) without the death-spiral panic a raw-κ loop would have produced.
The silence/absence rule (≥ 2 missed packets, reputation-aware) means the dark boat
reads as *the most valuable signal on the water*, not as a hole.

### Phase III — days 15–21: the first sign of warmth

| Day | R | S | F | κ | The elephant feels |
|-----|---|---|---|-----|--------------------|
| 15 | 0.0 | 0.50 | 0 | 0.5 | A whisper: **sounder ticks up at the dark boat's position — before the boats bunch.** The elephant feels the column's *look* approaching the anchor's distribution, and the dark-boat virtual point is now 2 days old — the real boat, or its patch, is confirming. Nudge: compare this column to the good-days columns — *before any catch data*. **This is the day the design is built to hit.** |
| 16–17 | +0.35 | 0.60 | +1 | 0.9 | Boats begin closing again — the field is *bunching* (spread_rate negative, κ rising ≥ 3 readings). A fleet-scale charisma event: the dark boat's patch is pulling the room back together. Driving: **commit** — the room is warming; the nudge re-weights the column. |
| 18–21 | +0.55 | 0.68 | +1 | 1.4 | The warm room *reconstitutes*. Each day's field lands closer to the anchor; catch crosses the 2× median again, so the anchor EMA refreshes toward *proven* warmth. The elephant is re-acclimating — faster than the first time (the anchor already exists; the system knows what warm looks like). |

**What was learned:** the **early-warning signature** — sounder look recovering ahead
of fleet coherence, at a position the dark-boat rule was already holding attention
on. The elephant fires the "compare to good days" nudge at day 15's whisper, not
day 18's confirmation. This is the payoff of inductive biomass: *driving over the
right biomass starts before the fleet agrees it exists.*

### Phase IV — days 22–30: the warm room again, plus one lesson

| Day | R | S | F | κ | The elephant feels |
|-----|---|---|---|-----|--------------------|
| 22–26 | +0.6 | 0.70 | +1 | 1.5 | Back in the anchor's neighborhood. Driving: hold the drag. The sauna/plunge gap is small again — the elephant is *home* — and the blindness tax is doing its job: by day 24 it has ticked +0.02 → +0.04, and one boat runs an hour-away probe drag on day 25 (finds nothing; the tax resets when the gap moves). The warm room no longer makes the elephant blind to the rest of it. |
| 27–30 | +0.7 | 0.74 | +1 | 1.7 | A *better* day than the anchor's mean — day 29's field sits past `μ̂_good`, and catch confirms it. The anchor refreshes (EMA toward the catch-good day): the warm room just got a little warmer. The elephant also kept days 8–14 in memory — it now knows both the warm room *and* the cold plunge, and the *walk between them*. |

**The 30-day lesson, in one line:** the elephant learned a room, felt it dissolve,
felt the *dark boat* point at where it would return, learned the dissolution,
recognized the room when it came back — and the whole arc was one long contrastive
training run with catch as the only label.

### 7.1 Making the diary falsifiable (adopted from review)

The review's last objection: a diary is a narrative unless it has thresholds. The
evaluation criteria for the 30-day arc, made concrete:

1. **Day-15 whisper:** the "compare to good days" nudge fires on day 15 (sounder
   rises ≥ 0.05 over the 3-reading window **and** gap is trending negative —
   closing), **and** κ_fleet has not yet risen above 1.0. False alarms counted: the
   same nudge must *not* have fired on days 9–14 (when the column was thinning) —
   i.e. hit-rate vs false-alarm measured over the whole arc, not just the one good
   day.
2. **No blowout:** the fleet must not fully dissolve on day 8–10 (κ never falls
   below 0.2 without ≥ 3 consecutive falling readings *and* the scatter command
   being deliberate). The damping bell + hysteresis are the mechanism; the metric
   is κ-floor behavior during the spotty phase.
3. **Dark-boat hold:** on day 11, the injected virtual point must hold fleet
   attention at the edge position for ≥ 90 of the 120 minutes (i.e. ≥ 1 boat's
   nudge keeps weighting that region).
4. **Anchor truth:** `corr(anchor_similarity, catch)` across days 15–30 must exceed
   `corr(dial_warmth, catch)` — the anchor must beat raw dial warmth, or the anchor
   is not earning its keep.
5. **Commit timing:** the hold/commit command on day 16–17 must arrive *after*
   κ rising ≥ 3 readings (hysteresis respected), and scatter on day 10 must arrive
   *after* κ falling ≥ 3 readings.

If any criterion fails, the failing mechanism is identified by name — this is an
evaluation, not an illustration.

---

## 8. What the wider view added — and what was adopted

Three models reviewed the brief (Seed-2.0-pro, Hermes-3-Llama-3.1-405B, DeepSeek
V4-Pro). Their critiques converged on four classes of flaw, all now designed
against:

**1. Feedback stability (Seed — the decisive catch).** Raw κ into the nudge is an
unbounded positive feedback loop; the 30-day diary would blow apart on day 10, not
bottom out. **Adopted:** the κ damping bell (§4.2), the probe-drag blindness tax
(§4.5), and the dark-boat charisma rule (§2.5). Seed also caught that the silent-boat
heuristic was 180° wrong — the best captain goes dark *on fish* — and that
uniform vMF is democracy where fleets run on deference. **Adopted:** reputation-
weighted field (§1.4), dark-boat rule (§2.5).

**2. Ground truth (DeepSeek — the tautology catch).** An anchor built from dials
that define warmth is self-confirming; the EMA rewards whatever the fleet happens to
be doing. **Adopted:** exogenous catch telemetry (§6.2), catch-defined good days,
nudge validation via `corr(nudge, catch)`, reputation from public catch logs (§1.4),
and the CUSUM regime-change detector with dual anchors (§6.4). DeepSeek also caught
the statistical fragility of vMF at n=3–30 boats → **adopted** the robust
two-scalar fleet field (§1.3), the defined gap metric (§6.1), the nudge side-channel
leak → **adopted** the wire discipline (§2.2), and hysteresis/deadband (§4.3).
DeepSeek's per-boat nudge budgets + fleet attention allocation (§5.3) and per-boat
local anchors (§6.4) are in.

**3. Privacy (Hermes + DeepSeek).** Numbers-only still leaks: the nudge reveals
intent; a raw fishing_day is a fingerprint. **Adopted:** nudge stays local, fishing_
day binned to {-1,0,+1}, quantization + dithering (§2.2). Hermes' weather room and
speculative nudge (§4.6) are in, as are Hermes' anchor-validation step (§6.4),
nudge low-pass smoothing (§4.3), and the falsifiability demand (§7.1).

**4. Evaluation (DeepSeek + Hermes).** A narrative diary is unfalsifiable.
**Adopted:** concrete thresholds, hit-rate vs false-alarm, and five named criteria
(§7.1).

**Not adopted:** nothing was dismissed outright. The only deliberate restraint:
reputation and catch telemetry assume boats share *public catch logs* — on boats
where even that is commercially sensitive, the design degrades gracefully (reputation
falls back to 1.0 flat weights; the anchor falls back to crew-observed deck activity
as its exogenous signal). The elephant still works; it just defers less.

---

## 9. What this enables (and what it deliberately doesn't)

**Enables:**

- A standalone harness elephant (local loop only) that becomes a fleet elephant by
  *sharing numbers* — same code, same dials, same nature.
- Driving on good days vs poor days as a *damped, hysteretic field response*
  (hold/scatter/commit/probe) rather than a rules engine — and a field that cannot
  spiral into herd panic.
- Inductive biomass: the good-days anchor is learned and **validated against
  catch**, never hand-labeled; deviation is felt, never reported.
- Privacy and bandwidth by construction: the wire only ever carries the values the
  design chose to expose — binned, quantized, dithered.
- The dark-boat rule: the most important signal on the water (the best captain going
  quiet on fish) reads as *attention*, not absence.
- A 30-day evaluation arc that is *already the training data* — day 14 is the
  negative class, day 29 refreshes the anchor, the whole diary is one contrastive
  run with catch as the only label.

**Deliberately does not:**

- Replace the vision model — the elephant correlates, it never concludes.
- Send raw feeds between boats — numbers only, forever; the nudge stays home.
- Assert "fish here" — it says *compare these*, and lets the field do the rest.
- Treat agreement as fish — tightness is only warmth when catch says so.
- Forget the captain's rule: *you light the woodstove in a cold room.* The scatter
  day is the woodstove — and the dark boat is the match.

---

*The elephant got sea legs. It reads the radar like a room, the sounder like a
memory, the weather like the room it's inside, and the fleet like a meta-room — and
on day 15, when the sounder whispers at the dark boat's last position, it will feel
the warm room coming back before the fleet agrees it exists. And this time it won't
blow apart on day 10: the elephant is part of the room, and it knows it.*
