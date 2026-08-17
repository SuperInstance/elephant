# Fleet Simulation Notes — the elephant at fleet scale (a room of rooms)

**File:** `examples/fleet_simulation.py`
**Companion design:** `docs/fleet-dynamics-design.md`
**Tests:** `tests/test_fleet_simulation.py`

## What this demonstrates

The elephant is a **room-temperature sense**: standalone in a harness (one boat,
`BoatHarness`), or plugged into any system with sensors. The fleet simulation is the
proof that the same mechanism scales from a **room** (one boat) to a **room of rooms**
(the fleet) without changing its nature.

Four fishing boats each run an elephant. Every boat perceives *its own* rooms
(radar = the distribution of the other boats, sounder = the biomass under the keel,
nav = course, conversation = the crew). The fleet is the **meta-room**: boats
broadcast **numbers only** (position, velocity, `radar_coherence`, `fishing_day`
binned to `{-1, 0, +1}`), and the meta-room's field is computed over those numbers —
a fleet concentration κ over boat positions, plus a warmth over the shared dials.
Feeds stay home; only the distilled readings cross the wire.

The 30-day arc is the demonstration: a warm room forms (good fishing), it dissolves
(spotty fishing), the fleet feels the *deviation* from the warm room it acclimated
to, and then it feels the warm room return — all through **contrast**, with catch as
the only exogenous label.

## The numbers it produces

Per day, the sim reports:

| Quantity | Meaning | Range |
|----------|---------|-------|
| `kappa` | fleet concentration — `1 / (1 + weighted mean radial distance from the weighted-median centroid)`. Clustered on the drag → near 1; scattered searching → near 0. | [0, 1] |
| `effective_kappa` | the **damping bell** (§4.2): `κ·(1 − clamp(κ, 0.2, 0.8))` — kills the herd-panic feedback loop at both ends. | [0, ~0.25] |
| `warmth` | meta-room temperature — `0.5·fishing_day + 0.3·(2·biomass − 1) + 0.2·radar_coherence`. | [−1, +1] |
| `spread_km` | the fleet's spatial spread (mean distance to centroid). | ≥ 0 km |
| `catch` | **exogenous** catch telemetry — the ground truth the anchor is built from (never derived from the dials). | [0, 1] |
| `deviation` | Mahalanobis distance of today's field from the good-week anchor — "does this stretch feel like the good kind?" | ≥ 0 |
| `fishing_day_per_boat` | each boat's composite luck dial (local, never leaves the boat). | [−1, +1] |
| `fishing_day_binned` | the signed bin that *does* cross the wire. | {−1, 0, +1} |
| `nudge_mean` | the fleet-mean nudge prior over 7 modalities (local cognition; only its number is reported). | [−1, +1]⁷ |

### The 30-day arc, phase by phase

- **Days 1–7 (good):** boats bunch on a slowly-drifting drag → `kappa` climbs from
  ~0.35 to ~0.85, `warmth` rises, `catch` high. Seven catch-good days become the
  **anchor** (`biomass_anchor` → a Gaussian over `[κ, biomass, catch]`).
- **Days 8–14 (spotty):** boats scatter searching → `kappa` falls toward ~0.1,
  `warmth` goes negative, `catch` collapses. `deviation` balloons — the elephant
  feels "this is not the warm room" *without ever being told fishing is bad*.
- **Days 15–30 (recovery):** boats re-group → `kappa` climbs back, `warmth`
  recovers, and `deviation` drops back toward the good-week anchor. Day 15+ is the
  payoff: the elephant recognizes the warm room returning from the *shape* of the
  field, before it can be explained.
- **Day 20 (dark-boat charisma, optional):** the highest-reputation boat goes dark.
  The meta-room injects a 3×-weighted virtual point at its last position, so the
  fleet holds attention *toward* the hole instead of reading it as thinning — the
  design's "hot boat went quiet because it's on fish, not because it left" heuristic.

## What it proves about the elephant at fleet scale

1. **The math scales.** The same vMF/field idea that reads one room reads the
   fleet. Fleet κ is the room temperature one level up: high κ = boats agreeing on
   where the fish are (warm); low κ = disagreement (uncertain). No new machinery.
2. **Numbers are enough.** The whole fleet runs on broadcast scalars. The meta-room
   never sees a feed; it feels the distribution. Privacy and bandwidth fall out by
   construction.
3. **The anchor must have an outside.** The review caught the tautology — an anchor
   built from `fishing_day` (itself a composite of the dials) would be
   self-confirming. The sim uses **exogenous catch** as the anchor's ground truth,
   so "good day" is defined by what was *landed*, not by what the dials felt.
4. **Deviation is felt, not reported.** The elephant is never handed a "fishing is
   bad" label. It feels the Mahalanobis gap between today's field and the warm room,
   and the gap is the whole training signal.
5. **The feedback loop is damped.** `effective_kappa` (the damping bell) shows the
   hold/scatter loop can't run away — clustered fleets get a weak probe nudge at the
   top, scattered fleets get a weak hold at the bottom, and the real signal lives in
   the un-touched middle.

## Review pass (Seed-2.0-pro)

Three fixes were adopted from a Seed-2.0-pro critique before implementation:

- **Exogenous catch in the anchor** (was `fishing_day`) — breaks the
  biomass↔fishing-day collinearity that would have made the Mahalanobis deviation
  measure shared noise instead of fleet drift.
- **Smooth κ** — `1/(1 + weighted mean spread)` uses every boat and varies
  continuously, rather than a median-of-4 MAD that would quantize the concentration
  into a handful of steps at fleet size.
- **Flagged dark boat** — the virtual point is reported explicitly (it *is* the
  charisma rule working, not a hidden metric distortion).

## Running it

```bash
cd /home/eileen/projects/elephant
python3 examples/fleet_simulation.py
python3 tests/test_fleet_simulation.py
```
