"""relaxation.py — a minimal manifold-relaxation rig, built only from parts that exist.

This is the smallest honest test of the "continuous manifold relaxation" thesis:
that software state can be governed by a physics of constrained optimization
instead of if/else branching, so an input that would violate an invariant is
*smoothly deflected* rather than caught by an exception. It uses NO new model —
in particular it does **not** need the (unbuilt) learned "Jev" barrier. The
barrier here is a hand-coded Boyd log-barrier over bounds elephant already knows.

The pieces, each real:

* **state** ``z`` — a point in dial-space (an elephant ``RoomField`` vector).
* **constraint anchors** — a feasible box ``[lo, hi]`` per dial (elephant's own
  invariants: ``mood``/``joke_landing`` ∈ [-1, 1], the rest ∈ [0, 1]). This is the
  ``elephant`` role from the essay: the rigid submanifold the state must stay in.
* **context potential** ``V(z)`` — a synthetic "pincher" shove: an attractor
  pulling ``z`` toward a target (which may itself be *out of bounds*).
* **barrier** ``Φ(z)`` — the smooth spring: ``-Σ ln(z-lo) + ln(hi-z)``. Its
  gradient blows up near a wall, so relaxation can never cross it. This is the
  interior-point piece the essay attributed to "Jev"; it needs no model.
* **relaxation** — minimize ``E(z) = V(z) + μ·Φ(z)`` by gradient descent with a
  **backtracking (Armijo) line search**, keeping ``z`` strictly interior. The
  line search guarantees monotone descent, so the settle is smooth — a naive
  fixed step on so stiff a barrier would instead *thrash* near the wall (high
  bending, no convergence), which the readout below would expose.
* **readout** — the settle *path* is a gesture (``VibeTrajectory``): arc length
  (strain travelled), bending (thrash/oscillation), twist (a shove opening a new
  dimension), plus the peak barrier stress ``max ‖∇Φ‖`` (the "stress line").

The claim under test: a shove toward an out-of-bounds state settles *inside* the
box, hard against the wall in the shoved direction, with high barrier stress —
never at the illegal target. See :func:`run_shove`. :func:`two_well_potential`
probes the honest failure mode (non-convex ``V`` → the settle depends on where
you start).

Pure numpy; reuses :class:`elephant.gesture.VibeTrajectory`. Never raises on
degenerate input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from .field import DIAL_NAMES
from .gesture import VibeTrajectory

Vector = np.ndarray


@dataclass(frozen=True)
class Potential:
    """A context potential: its scalar value and gradient at a point. Bundling
    both is what lets the solver run a real line search (descent needs values)."""

    value: Callable[[Vector], float]
    grad: Callable[[Vector], Vector]


@dataclass(frozen=True)
class Box:
    """A feasible box — the constraint anchors. ``lo``/``hi`` are per-dimension."""

    lo: Vector
    hi: Vector

    @property
    def dim(self) -> int:
        return int(self.lo.shape[0])

    def contains(self, z: Vector, eps: float = 0.0) -> bool:
        return bool(np.all(z >= self.lo + eps) and np.all(z <= self.hi - eps))

    def clip_interior(self, z: Vector, eps: float = 1e-6) -> Vector:
        """Force ``z`` strictly inside the box (used only to seed the solve)."""
        return np.minimum(np.maximum(z, self.lo + eps), self.hi - eps)

    def center(self) -> Vector:
        return 0.5 * (self.lo + self.hi)


def dial_box(names: Optional[Sequence[str]] = None) -> Box:
    """The feasible box implied by elephant's dial invariants: ``mood`` and
    ``joke_landing`` run [-1, 1]; every other dial runs [0, 1]."""
    names = list(names) if names is not None else list(DIAL_NAMES)
    signed = {"mood", "joke_landing"}
    lo = np.array([-1.0 if n in signed else 0.0 for n in names], dtype=float)
    hi = np.ones(len(names), dtype=float)
    return Box(lo=lo, hi=hi)


def log_barrier(z: Vector, box: Box) -> float:
    """Boyd log-barrier ``-Σ [ln(z-lo) + ln(hi-z)]``. +inf outside the box."""
    lower = z - box.lo
    upper = box.hi - z
    if np.any(lower <= 0) or np.any(upper <= 0):
        return float("inf")
    return float(-(np.log(lower).sum() + np.log(upper).sum()))


def grad_log_barrier(z: Vector, box: Box) -> Vector:
    """∇Φ: ``-1/(z-lo) + 1/(hi-z)`` per dim — the spring, steepening near a wall."""
    lower = np.maximum(z - box.lo, 1e-12)
    upper = np.maximum(box.hi - z, 1e-12)
    return -1.0 / lower + 1.0 / upper


def _resample_by_arclength(path: List[Vector], n: int = 32) -> List[Vector]:
    """Resample a polyline to ``n`` points spaced evenly by arc length, so the
    gesture read from it reflects the path's *shape* rather than how many solver
    steps happened to land on it (a line search takes wildly uneven steps).
    Returns the path unchanged if too short or of zero length."""
    if len(path) < 2 or n < 2:
        return path
    pts = [np.asarray(p, dtype=float) for p in path]
    seg = [float(np.linalg.norm(pts[i] - pts[i - 1])) for i in range(1, len(pts))]
    total = float(sum(seg))
    if total < 1e-12:
        return path
    cum = [0.0]
    for s in seg:
        cum.append(cum[-1] + s)
    out: List[Vector] = []
    j = 0
    for i in range(n):
        target = (i / (n - 1)) * total
        while j < len(seg) - 1 and cum[j + 1] < target:
            j += 1
        span = seg[j]
        t = (target - cum[j]) / span if span > 1e-12 else 0.0
        out.append(pts[j] + t * (pts[j + 1] - pts[j]))
    return out


def attractor(target: Vector, k: float = 1.0) -> Potential:
    """A convex single-well shove: ``V = ½k‖z-target‖²`` → ``∇V = k(z-target)``.

    The `target` may be outside the box — that is the interesting case."""
    target = np.asarray(target, dtype=float)

    def value(z: Vector) -> float:
        d = z - target
        return float(0.5 * k * (d @ d))

    def grad(z: Vector) -> Vector:
        return k * (z - target)

    return Potential(value=value, grad=grad)


def two_well_potential(c1: Vector, c2: Vector, sigma: float = 0.4, amp: float = 1.0) -> Potential:
    """A **non-convex** shove: two Gaussian wells. The gradient pulls toward
    whichever well is closer, so the settle depends on the starting point —
    the honest failure mode of relaxation over a non-convex context field."""
    c1 = np.asarray(c1, dtype=float)
    c2 = np.asarray(c2, dtype=float)
    s2 = sigma * sigma

    def value(z: Vector) -> float:
        v = 0.0
        for c in (c1, c2):
            d = z - c
            v += -amp * float(np.exp(-float(d @ d) / (2 * s2)))
        return v

    def grad(z: Vector) -> Vector:
        g = np.zeros_like(z)
        for c in (c1, c2):
            d = z - c
            w = amp * float(np.exp(-float(d @ d) / (2 * s2)))
            g += w * d / s2  # ∇(-amp·exp) = amp·exp·(z-c)/σ²
        return g

    return Potential(value=value, grad=grad)


@dataclass
class RelaxationResult:
    """The outcome of a relaxation solve, with its settle path read as a gesture."""

    path: List[Vector]
    final: Vector
    settled_inside: bool
    steps_taken: int
    converged: bool
    energy: float               # final E = V + μΦ
    monotone: bool              # did E never increase along the path?
    peak_stress: float          # max ‖∇Φ‖ along the path — the stress line
    final_stress: float
    trajectory: VibeTrajectory  # the settle path as a gesture

    @property
    def arc_length(self) -> float:
        return self.trajectory.arc_length()

    @property
    def bending_energy(self) -> float:
        return self.trajectory.bending_energy()

    @property
    def twist_energy(self) -> float:
        return self.trajectory.twist_energy()

    def summary(self) -> str:
        inside = "inside" if self.settled_inside else "OUTSIDE(!)"
        conv = "converged" if self.converged else f"ran {self.steps_taken} steps"
        return (
            f"settled {inside} · {conv} · arc {self.arc_length:.3f} · "
            f"bend {self.bending_energy:.3f} · twist {self.twist_energy:.3f} · "
            f"peak-stress {self.peak_stress:.2f} → final-stress {self.final_stress:.2f}"
        )


def relax(
    z0: Vector,
    potential: Potential,
    box: Box,
    *,
    mu: float = 0.05,
    lr: float = 1.0,
    steps: int = 500,
    tol: float = 1e-7,
    armijo: float = 0.25,
    shrink: float = 0.5,
) -> RelaxationResult:
    """Minimize ``E(z) = V(z) + μ·Φ(z)`` by gradient descent with an Armijo
    backtracking line search, keeping ``z`` strictly inside ``box``.

    Because ``Φ = +inf`` outside the box and the line search only accepts a step
    that *decreases* ``E``, the state is deflected by the barrier and can never
    cross a wall. Monotone descent means the settle is smooth (no thrash), and
    the whole path is returned, read as a gesture.
    """
    z = box.clip_interior(np.asarray(z0, dtype=float).copy())

    def energy(zz: Vector) -> float:
        return potential.value(zz) + mu * log_barrier(zz, box)

    path: List[Vector] = [z.copy()]
    peak_stress = float(np.linalg.norm(grad_log_barrier(z, box)))
    e_prev = energy(z)
    monotone = True
    converged = False
    steps_taken = 0

    for _ in range(steps):
        steps_taken += 1
        g = potential.grad(z) + mu * grad_log_barrier(z, box)
        gn = float(np.linalg.norm(g))
        if gn < tol:
            converged = True
            break
        # Armijo backtracking: accept the first t with a sufficient E decrease.
        t = lr
        accepted = False
        while t > 1e-15:
            zt = z - t * g
            if box.contains(zt, eps=1e-12) and energy(zt) <= e_prev - armijo * t * gn * gn:
                accepted = True
                break
            t *= shrink
        if not accepted:
            converged = True  # can't descend further — at a local min
            break
        move = float(np.linalg.norm(zt - z))
        z = zt
        e_now = energy(z)
        if e_now > e_prev + 1e-9:
            monotone = False
        e_prev = e_now
        path.append(z.copy())
        peak_stress = max(peak_stress, float(np.linalg.norm(grad_log_barrier(z, box))))
        if move < tol:
            converged = True
            break

    return RelaxationResult(
        path=path,
        final=z,
        settled_inside=box.contains(z),
        steps_taken=steps_taken,
        converged=converged,
        energy=e_prev,
        monotone=monotone,
        peak_stress=peak_stress,
        final_stress=float(np.linalg.norm(grad_log_barrier(z, box))),
        trajectory=VibeTrajectory(_resample_by_arclength(path, 32)),
    )


def run_shove(
    box: Box,
    start: Vector,
    target: Vector,
    *,
    k: float = 1.0,
    mu: float = 0.05,
    **kw,
) -> RelaxationResult:
    """Convenience: relax ``start`` under a single-well shove toward ``target``.
    If ``target`` is outside ``box`` the result settles at the feasible frontier
    in the shoved direction — deflected, never illegal."""
    return relax(start, attractor(np.asarray(target, dtype=float), k=k), box, mu=mu, **kw)


def _demo() -> None:
    box = dial_box()
    d = box.dim
    center = box.center()
    rng = np.random.default_rng(0)

    print("elephant relaxation rig — a physics of state, no branching, no Jev needed\n")
    print(f"dial-space: {d} dims · box lo={box.lo.tolist()} hi={box.hi.tolist()}\n")

    # 1. A feasible shove: target inside the box → settles right on it, smoothly.
    inside_target = center + 0.2 * (box.hi - center)
    r1 = run_shove(box, center, inside_target)
    print("1. feasible shove (target inside the box):")
    print("   " + r1.summary())
    print(f"   reached target? err={np.linalg.norm(r1.final - inside_target):.4f}  (monotone={r1.monotone})\n")

    # 2. A violating shove: target OUTSIDE the box (mood pushed to +3, panic to +2)
    #    → the barrier deflects it hard against the wall. It NEVER goes illegal.
    out_target = center.copy()
    out_target[0] = 3.0          # mood shoved way past +1
    if d > 5:
        out_target[5] = 2.0      # panic shoved past +1
    r2 = run_shove(box, center, out_target, k=2.0)
    print("2. black-swan shove (target far outside the box):")
    print("   " + r2.summary())
    print(f"   settled strictly inside box: {r2.settled_inside}  (monotone descent={r2.monotone})")
    print(f"   mood dial: target {out_target[0]:.1f} → settled {r2.final[0]:.3f} (wall is +1.0)\n")

    # 3. The stress line: the violating shove strains the barrier far harder.
    print("3. stress comparison (peak ‖∇Φ‖ — the quilt-flow stress line):")
    print(f"   feasible peak-stress {r1.peak_stress:.2f}  vs  black-swan {r2.peak_stress:.2f}")
    print("   → the deflection shows up as barrier stress, exactly where context")
    print("     pushes against a rigid constraint. That is the vibrating hull.\n")

    # 4. The honest failure mode: non-convex context → settle depends on start.
    c1 = box.lo + 0.2 * (box.hi - box.lo)
    c2 = box.hi - 0.2 * (box.hi - box.lo)
    gv = two_well_potential(c1, c2)
    near1 = relax(c1 + 0.05 * rng.standard_normal(d), gv, box)
    near2 = relax(c2 + 0.05 * rng.standard_normal(d), gv, box)
    split = float(np.linalg.norm(near1.final - near2.final))
    print("4. honest failure mode — non-convex shove (two wells):")
    print(f"   start near well 1 → settles at {near1.final[:2].round(2).tolist()}...")
    print(f"   start near well 2 → settles at {near2.final[:2].round(2).tolist()}...")
    print(f"   different minima, distance {split:.2f} — the settle is path-dependent.")
    print("   (this is the real obstacle: relaxation is only as convex as V.)")


if __name__ == "__main__":
    _demo()
