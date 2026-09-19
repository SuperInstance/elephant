"""Tests for elephant.gesture — the geometry of a room's vibe motion."""

import numpy as np

from elephant.field import RoomField
from elephant.gesture import VibeTrajectory, heading_alignment


def _vecs(rows):
    return [np.array(r, dtype=float) for r in rows]


def test_empty_and_single_are_graceful():
    empty = VibeTrajectory([])
    assert len(empty) == 0
    assert empty.arc_length() == 0.0
    assert empty.speed() == 0.0
    assert empty.bending_energy() == 0.0
    assert np.allclose(empty.heading(), np.zeros(empty.dim if empty.dim else 0))

    single = VibeTrajectory(_vecs([[0.2, 0.3]]))
    assert len(single) == 1
    assert single.arc_length() == 0.0
    assert np.allclose(single.heading(), np.zeros(2))


def test_arc_length_counts_travel():
    # Three steps of length 1 along axis 0 → arc length 3.
    traj = VibeTrajectory(_vecs([[0, 0], [1, 0], [2, 0], [3, 0]]))
    assert abs(traj.arc_length() - 3.0) < 1e-9
    # A still room travels nowhere.
    still = VibeTrajectory(_vecs([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]))
    assert still.arc_length() == 0.0


def test_straight_drift_has_no_bending_but_a_wanderer_does():
    straight = VibeTrajectory(_vecs([[0, 0], [1, 0], [2, 0], [3, 0]]))
    assert straight.bending_energy() < 1e-9
    # Lurching: direction flips each step.
    lurch = VibeTrajectory(_vecs([[0, 0], [1, 0], [1, 1], [2, 1], [2, 2]]))
    assert lurch.bending_energy() > straight.bending_energy()
    assert 0.0 <= lurch.restlessness() <= 2.0


def test_heading_is_unit_and_is_the_last_direction():
    traj = VibeTrajectory(_vecs([[0, 0], [0, 0.5], [0, 2.0]]))
    h = traj.heading()
    assert abs(np.linalg.norm(h) - 1.0) < 1e-9
    # Last step moved along +axis1.
    assert h[1] > 0.99 and abs(h[0]) < 1e-9


def test_speed_is_last_step_magnitude():
    traj = VibeTrajectory(_vecs([[0, 0], [0, 0], [3, 4]]))
    assert abs(traj.speed() - 5.0) < 1e-9


def test_heading_alignment_same_vs_opposite():
    up_a = VibeTrajectory(_vecs([[0, 0], [1, 1]]))
    up_b = VibeTrajectory(_vecs([[5, 5], [6, 6]]))  # parallel heading
    down = VibeTrajectory(_vecs([[0, 0], [-1, -1]]))  # opposite heading
    assert heading_alignment(up_a, up_b) > 0.99
    assert heading_alignment(up_a, down) < -0.99


def test_integrates_with_roomfield():
    # A room warming: mood rising, panic falling, over three readings.
    fields = [
        RoomField({"mood": -0.5, "panic": 0.8}),
        RoomField({"mood": 0.0, "panic": 0.4}),
        RoomField({"mood": 0.6, "panic": 0.1}),
    ]
    traj = VibeTrajectory(fields)
    assert len(traj) == 3
    assert traj.arc_length() > 0.0
    # It moved somewhere, so there is a heading.
    assert np.linalg.norm(traj.heading()) > 0.99
