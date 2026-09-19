"""Tests for elephant.relaxation — the minimal manifold-relaxation rig.

These assert the robust claims of the thesis: a shove that would violate a bound
is deflected (never illegal), the deflection shows up as barrier stress, the
solve is a monotone descent, and a non-convex context is honestly path-dependent.
"""

import numpy as np

from elephant.relaxation import (
    Box,
    attractor,
    dial_box,
    grad_log_barrier,
    log_barrier,
    relax,
    run_shove,
    two_well_potential,
    _resample_by_arclength,
)


def _unit_box(d=3):
    return Box(lo=np.zeros(d), hi=np.ones(d))


def test_dial_box_matches_elephant_invariants():
    box = dial_box(["mood", "volume", "joke_landing", "panic"])
    assert box.lo.tolist() == [-1.0, 0.0, -1.0, 0.0]  # mood & joke_landing signed
    assert box.hi.tolist() == [1.0, 1.0, 1.0, 1.0]
    # The real elephant dial set produces a well-formed box.
    real = dial_box()
    assert real.dim >= 7 and np.all(real.hi > real.lo)


def test_log_barrier_finite_inside_inf_outside():
    box = _unit_box(2)
    assert np.isfinite(log_barrier(np.array([0.5, 0.5]), box))
    assert log_barrier(np.array([0.5, 1.5]), box) == float("inf")  # above hi
    assert log_barrier(np.array([-0.1, 0.5]), box) == float("inf")  # below lo


def test_barrier_gradient_points_inward_near_a_wall():
    box = _unit_box(1)
    # Near the upper wall, ∇Φ is large and positive → descent (−∇Φ) pushes down,
    # away from the wall.
    g_hi = grad_log_barrier(np.array([0.99]), box)[0]
    assert g_hi > 0
    # Near the lower wall, ∇Φ is large and negative → descent pushes up.
    g_lo = grad_log_barrier(np.array([0.01]), box)[0]
    assert g_lo < 0
    # At the center it vanishes by symmetry.
    assert abs(grad_log_barrier(np.array([0.5]), box)[0]) < 1e-9


def test_feasible_shove_settles_on_target_smoothly():
    box = _unit_box(3)
    target = np.array([0.7, 0.3, 0.6])
    r = run_shove(box, box.center(), target, k=1.0, mu=0.02)
    assert r.settled_inside
    assert r.monotone
    assert np.linalg.norm(r.final - target) < 0.1  # small barrier offset only
    assert np.isfinite(r.arc_length) and r.arc_length > 0


def test_black_swan_shove_is_deflected_never_illegal():
    box = _unit_box(3)
    # Target far outside the box on dim 0.
    out = np.array([5.0, 0.5, 0.5])
    r = run_shove(box, box.center(), out, k=2.0, mu=0.02)
    # THE claim: it settles strictly inside, hard against the wall, not at 5.0.
    assert r.settled_inside
    assert r.final[0] < box.hi[0]          # never reaches the illegal target
    assert r.final[0] > box.center()[0]    # but it did move toward the wall
    assert r.final[0] > 0.9                # pinned near the +1 wall
    assert r.monotone


def test_deflection_shows_up_as_barrier_stress():
    box = _unit_box(3)
    feasible = run_shove(box, box.center(), np.array([0.7, 0.5, 0.5]), mu=0.02)
    blackswan = run_shove(box, box.center(), np.array([5.0, 0.5, 0.5]), k=2.0, mu=0.02)
    # The violating shove strains the barrier far harder — the stress line.
    assert blackswan.peak_stress > 10 * feasible.peak_stress


def test_relaxation_never_leaves_the_box():
    box = dial_box()
    out = box.center().copy()
    out[0] = 9.0  # absurd shove
    r = run_shove(box, box.center(), out, k=3.0)
    for p in r.path:
        assert box.contains(p), f"path left the box at {p}"


def test_planar_shove_has_no_twist():
    box = _unit_box(4)
    # Shove only in dims 0,1 → motion stays in a plane → zero torsion.
    r = run_shove(box, box.center(), np.array([5.0, 5.0, 0.5, 0.5]), k=2.0)
    assert r.twist_energy < 1e-6


def test_non_convex_context_is_path_dependent():
    box = _unit_box(2)
    c1 = np.array([0.2, 0.2])
    c2 = np.array([0.8, 0.8])
    gv = two_well_potential(c1, c2, sigma=0.3)
    near1 = relax(np.array([0.22, 0.22]), gv, box)
    near2 = relax(np.array([0.78, 0.78]), gv, box)
    # Different starts settle into different wells — the honest failure mode.
    assert np.linalg.norm(near1.final - near2.final) > 0.5
    assert near1.settled_inside and near2.settled_inside


def test_resample_returns_n_points_and_keeps_endpoints():
    path = [np.array([0.0, 0.0]), np.array([0.1, 0.0]), np.array([1.0, 0.0])]
    out = _resample_by_arclength(path, 5)
    assert len(out) == 5
    assert np.allclose(out[0], [0.0, 0.0])
    assert np.allclose(out[-1], [1.0, 0.0])
    # Even arc-length spacing on this straight path → equal x gaps of 0.25.
    for i in range(1, len(out)):
        assert abs((out[i][0] - out[i - 1][0]) - 0.25) < 1e-9
