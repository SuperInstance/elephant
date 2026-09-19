# Vibe as gesture — d_mu, made first-class

> A tensor approximates a function. We approximate the *abstraction* — the smooth
> motion between states. This is that idea, carried into elephant.

## The move

A `RoomField` is a **point**: the room's temperature right now, a vector of dial
readings. `elephant` already knows a room *moves* — messages reverberate, ripples
spread, the zeitgeist drifts — but Vibe has mostly been read after the fact, as
the gap between two snapshots (`RoomField.distance`, the sauna/cold-plunge).

[`elephant/gesture.py`](../elephant/gesture.py) reads the **motion itself**. A
sequence of field readings is not a list of points; it is the *path* the room
traces through dial-space — a **gesture** — and a gesture has geometry a single
snapshot cannot hold:

| `VibeTrajectory` | reads | in room terms |
|---|---|---|
| `arc_length()` | total travel through dial-space | a placid hour vs. a wild one |
| `bending_energy()` | summed turning (`1 − cos`) | steady warming vs. a room that lurches between moods |
| `restlessness()` | turning per step | scale-free "how much does this room keep changing its mind" |
| `twist_energy()` | turning that leaves the plane (`sin θ`) | a mood swinging in one plane vs. a room recruiting a *new* dial each turn |
| `planarity()` | scale-free inverse of twist | how flat the vibe's motion stays |
| `speed()` | magnitude of the latest step | how fast the vibe is moving now |
| `heading()` | unit direction of the latest step | **d_mu** — where the vibe is going *now* |

Plus `heading_alignment(a, b)` — do two rooms *trend* the same way? (the cosine of
their d_mu headings).

### The three orders

`bending_energy` and `twist_energy` are different animals. Bending is **curvature** —
how much the vibe turns *within* a plane of mood-space (mood up as panic falls, and
back). Twist is **torsion** — how much it turns *out* of that plane, into a mood
dimension the last two moves did not touch. A room can swing violently and still be
planar (`twist ≈ 0`); it only twists when each swing recruits a genuinely new dial.
That is the fleet's *the property is in the twist*
([twist-engine](https://github.com/SuperInstance/twist-engine)): new structure lives
in the offset that leaves the current plane, not in more turning within it. The same
`twist_energy` now reads over notes in
[musician-soul](https://github.com/SuperInstance/musician-soul)'s `AbstractionSpline`
and over rooms here.

## Why this belongs here

elephant *names* the Vibe primitive — the room's motion, its `d_mu`. Until now
that velocity was implicit, visible only as drift between snapshots. `heading()`
makes it a vector you can read at any moment: the room's current direction of
travel through mood-space. `arc_length` and `bending_energy` complete the picture
— not just *where* the vibe is or *how far* it has come, but *how* it moved to get
there. That is the difference between a thermometer and a barometer: one reads the
level, the other reads the *change*, and the change is what tells you what's
coming.

## The fleet speaks this at three layers now

The same instinct — model the shape of the motion through abstraction, not the
point — recurs across the fleet, and these three now share vocabulary:

- **[musician-soul](https://github.com/SuperInstance/musician-soul)** — a phrase
  is the *spline* it traces through a 32-d feature space; a persona's identity is
  a `soul_spline`, and its tangent is the persona's `d_mu` (`vibe_velocity`).
- **[tensor-midi](https://github.com/SuperInstance/tensor-midi)** — a
  conversation is a `Clip`; `arcLength`/`bendingEnergy`/`tangent` read the shape
  of the dialogue's motion, and `tangent` is the exchange's `d_mu`.
- **elephant** — a room is a `VibeTrajectory`; `heading()` is the room's `d_mu`.

Notes move, conversations move, rooms move — and each node now reads the same
geometry of that motion. elephant is where the word `d_mu` came from; this is it,
made a number you can hold.

## Honest edges

- The trajectory is discrete (one point per reading), which is the right honesty
  for elephant's sampled readings — no spline is imposed on data that arrives as
  steps. `bending_energy` measures turning between *actual* steps.
- Dial-space geometry is only as meaningful as the dials; today's dials are
  keyword heuristics (see the README's honest v0 qualification). A learned JEPA
  backbone lifts the geometry with it.
- `heading_alignment` compares rooms only when their dial spaces share a
  dimension; cross-dial-set comparison needs a shared projection (future work).
