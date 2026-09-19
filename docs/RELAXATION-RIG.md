# The relaxation rig — a physics of state, tested with parts that exist

> Can software state be governed by *constrained optimization* instead of
> `if/else` branching — so a bad input is **smoothly deflected** rather than
> caught by an exception? [`elephant/relaxation.py`](../elephant/relaxation.py)
> is the smallest honest experiment that answers it, and it needs **no new
> model** — no "Jev", no learned barrier.

## The idea being tested

A proposed fleet paradigm ("the Spline-Stretched State Hull" / continuous
manifold relaxation) says: collapse `elephant` (anchors), `pincher` (a driving
field), a barrier model, and `quilt` (the solver) into one geometric system where
the whole application state is a point on a smooth energy surface, and every tick
relaxes it toward the lowest-energy valid configuration. Type errors become
*physically high-energy*, not caught exceptions.

Two of that essay's named pillars don't exist yet (`Jev` is not a repo; `pincher`
is a discrete reflex matcher, not a streaming field). But the **math** is
testable today, because the one genuinely missing piece — the smooth barrier — is
just a Boyd log-barrier, which needs no model.

## What the rig is

| Essay role | Rig piece | Reality |
|---|---|---|
| `elephant` anchors | `Box` from `dial_box()` — the feasible box `mood∈[-1,1]`, rest `∈[0,1]` | elephant's own dial invariants |
| `pincher` field | `attractor(target)` — a shove pulling state toward a target (maybe out of bounds) | synthetic (real pincher is event-based) |
| `Jev` barrier | `log_barrier` / `grad_log_barrier` — a hand-coded log-barrier | **no model needed** |
| `quilt` solver | `relax()` — gradient descent with an Armijo line search, kept strictly interior | a single-cell stand-in for the sheet |
| readout | `VibeTrajectory` over the settle path + peak `‖∇Φ‖` | elephant's own gesture module |

## What it found (`python -m elephant.relaxation`)

- **Deflection works, exactly as claimed.** A black-swan shove targeting `mood = +3`
  (illegal) settles at `mood = 0.988` — hard against the `+1` wall, **strictly
  inside the box, never illegal**. It's a smooth monotone descent, not a clip.
- **The deflection is visible as barrier stress.** Peak `‖∇Φ‖` is `~2` for a
  feasible shove vs `~220` for the black-swan — a clean, order-of-magnitude
  signal, exactly where real-world context pushes against a rigid constraint.
  *This is the "vibrating hull" stress line the essay wants in `quilt-flow`, and
  it is the load-bearing diagnostic of the whole rig.*
- **The state never leaves the box** — asserted over the entire path, for shoves
  as absurd as `9.0`.

## The honest failure mode (this is the point of a cheap experiment)

- **Relaxation is only as convex as the context field.** With a non-convex shove
  (`two_well_potential`, two Gaussian wells), the settle is **path-dependent**:
  start near well 1 and you land in well 1; start near well 2 and you land in well
  2. The essay's "settles into the safest state" is false in general — it settles
  into *a* local minimum, which depends on where you began. Warm-starting from the
  last equilibrium (which `quilt` does for free) makes this *stable* but also
  *history-dependent* — a feature (hysteresis/memory) only if you own it.
- **The solver matters.** A naive fixed-step descent on so stiff a barrier
  *thrashes* near the wall (oscillates, never converges); the Armijo line search
  is what makes the settle smooth. The gesture readout is what tells the two
  apart — which is why a branchless engine is **unobservable without it**.

## Verdict

The core claim holds with parts that already pass tests: **constrained
optimization deflects invalid state smoothly, and the strain is readable.** What
the paradigm actually needs next is not a new model but the two things this rig
exposes: (1) a strategy for **non-convex** context fields (convexify, or accept
and audit local minima), and (2) a **distributed** solver — `relax()` here is one
cell; `quilt`'s reactive graph is the real, incremental version, and
`si-conservation-diffusion` already implements the projection variant of this
step. "Jev" then becomes an *optimization* of a working system (a learned barrier
replacing the hand-coded one), not a prerequisite.

## Where this sits in the fleet

- **`elephant`** — supplies the state (`RoomField`), the anchors (`dial_box`), and
  the readout (`VibeTrajectory`: arc/bending/twist of the settle path).
- **`si-conservation-diffusion`** — already does projection-onto-manifold
  relaxation; the natural partner for a multi-agent version.
- **`constraint-theory-core` / `agent-manifold` / `lau-variational-methods`** —
  the geometry, simplex projection, and variational machinery to graduate this
  single-cell rig into a real manifold engine.
- **`quilt`** — the distributed solver substrate this stands in for.

*A tensor approximates a function; we approximate the shape of the state's motion
toward equilibrium. Here, for once, the whole loop is a physics you can watch.*
