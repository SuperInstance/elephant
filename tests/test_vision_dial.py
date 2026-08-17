"""elephant — tests: the vision dial (the 9th sense, from plato-vision-jepa)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elephant.dial import Dial, DialBank
from elephant.dials import DEFAULT_DIALS, VisionDial
from elephant.sensors import SensorFrame, SignalRoom


# --------------------------------------------------------------------- #
# Frame builders — both documented data forms                           #
# --------------------------------------------------------------------- #
def _camera(ts, data, sensor="camera"):
    return SensorFrame(ts=ts, sensor=sensor, data=data)


def _bright_active_room(n=6):
    """Bright, moving, occupied room — dict form."""
    frames = [_camera(i, {
        "brightness": 0.9, "motion": 0.7, "occupancy": 0.8, "anomaly": 0.1,
    }) for i in range(n)]
    return SignalRoom("lit", frames)


def _dark_empty_room(n=6):
    """Dark, still, empty room — dict form with plato's field spellings."""
    frames = [_camera(i, {
        "brightness": 0.05, "motion_level": 0.05, "occupancy": 0.0,
        "anomaly_score": 0.02,
    }) for i in range(n)]
    return SignalRoom("dark", frames)


def _bright_active_16d(n=6):
    """Same bright room as a full 16-dim plato room-state vector."""
    frames = []
    for i in range(n):
        v = [0.0] * 16
        v[0], v[1], v[2], v[3] = 0.9, 0.7, 0.8, 0.1
        v[4:8] = [0.8, 0.7, 0.6, 0.9]          # quadrant activity
        v[8:12] = [0.5, 0.6, 0.7, 0.8]         # temporal trends
        frames.append(_camera(i, tuple(v)))
    return SignalRoom("lit16", frames)


def _mixed_room():
    """Radar frames (ignored) + one camera frame, to prove sensor gating."""
    return SignalRoom("bridge", [
        _camera(0.0, [(0.0, 0.0), (1.0, 1.0)], sensor="radar"),
        _camera(1.0, {"brightness": 0.6, "motion": 0.5,
                      "occupancy": 0.4, "anomaly": 0.1}),
    ])


# --------------------------------------------------------------------- #
# The tests                                                             #
# --------------------------------------------------------------------- #
def test_bright_active_reads_high():
    r = VisionDial().read(_bright_active_room())
    assert isinstance(r, float)
    assert r > 0.6, r


def test_dark_empty_reads_low():
    r = VisionDial().read(_dark_empty_room())
    assert r < 0.3, r


def test_bright_beats_dark():
    assert VisionDial().read(_bright_active_room()) > VisionDial().read(_dark_empty_room())


def test_dict_form_and_16d_form_agree():
    d = VisionDial().read(_bright_active_room())
    v = VisionDial().read(_bright_active_16d())
    assert abs(d - v) < 1e-9, (d, v)


def test_plato_spellings_accepted():
    # _dark_empty_room uses motion_level/anomaly_score keys; must read low.
    r = VisionDial().read(_dark_empty_room())
    assert r < 0.3, r


def test_no_frames_is_neutral():
    assert VisionDial().read(SignalRoom("empty")) == 0.5


def test_text_room_has_no_visual_opinion():
    # The shared bank reads plain text Rooms via read_field; the vision
    # sense must rest at neutral there, not crash.
    from elephant.room import Message, Room
    r = Room("tap", [Message("ada", "hello", ts=0.0)])
    assert VisionDial().read(r) == 0.5


def test_only_camera_frames_count():
    # Same single camera frame, with and without radar noise.
    r = VisionDial().read(_mixed_room())
    solo = VisionDial().read(SignalRoom("solo", [_camera(1.0, {
        "brightness": 0.6, "motion": 0.5, "occupancy": 0.4, "anomaly": 0.1,
    })]))
    assert abs(r - solo) < 1e-9, (r, solo)


def test_anomaly_is_a_bonus_spike():
    quiet = {"brightness": 0.0, "motion": 0.0, "occupancy": 0.0, "anomaly": 0.0}
    spooky = dict(quiet, anomaly=1.0)
    rq = VisionDial().read(SignalRoom("q", [_camera(0, quiet)]))
    rs = VisionDial().read(SignalRoom("s", [_camera(0, spooky)]))
    assert rs > rq, (rq, rs)
    assert rs <= 1.0 and rq == 0.0


def test_deadband_collapses_identical_frames():
    # 50 identical frames must read the same as a single frame: after the
    # first, every frame repeats the previous state (diff 0 <= 0.05).
    one = VisionDial().read(SignalRoom("one", [_camera(0, {
        "brightness": 0.8, "motion": 0.6, "occupancy": 0.5, "anomaly": 0.0,
    })]))
    many = VisionDial().read(SignalRoom("many", [_camera(i, {
        "brightness": 0.8, "motion": 0.6, "occupancy": 0.5, "anomaly": 0.0,
    }) for i in range(50)]))
    assert abs(one - many) < 1e-9, (one, many)


def test_deadband_skips_small_changes_processes_large():
    dial = VisionDial(deadband=0.05)
    a = {"brightness": 0.5, "motion": 0.5, "occupancy": 0.5, "anomaly": 0.0}
    small = dict(a, brightness=0.52)               # diff 0.005 — skipped
    big = {"brightness": 1.0, "motion": 1.0, "occupancy": 1.0, "anomaly": 0.0}

    skipped = dial.read(SignalRoom("s", [_camera(0, a), _camera(1, small)]))
    alone = dial.read(SignalRoom("a", [_camera(0, a)]))
    assert abs(skipped - alone) < 1e-9, (skipped, alone)

    processed = dial.read(SignalRoom("p", [_camera(0, a), _camera(1, big)]))
    assert processed > skipped + 0.2, (processed, skipped)


def test_deadband_zero_processes_everything():
    dial = VisionDial(deadband=0.0)
    a = {"brightness": 0.5, "motion": 0.5, "occupancy": 0.5, "anomaly": 0.0}
    b = dict(a, brightness=0.6)
    both = dial.read(SignalRoom("b", [_camera(0, a), _camera(1, b)]))
    first = dial.read(SignalRoom("f", [_camera(0, a)]))
    assert both > first, (both, first)


def test_unreadable_frames_are_neutral():
    room = SignalRoom("bad", [
        _camera(0, "not a state"),
        _camera(1, 42),
        _camera(2, {"foo": 1}),          # no recognized room-state keys
    ])
    assert VisionDial().read(room) == 0.5


def test_garbage_field_values_clamp_to_zero():
    # A dict that NAMES the fields is a room-state frame; garbage values
    # clamp to 0 rather than breaking the read.
    room = SignalRoom("garbage", [
        _camera(0, {"brightness": "oops", "motion": None,
                    "occupancy": float("nan"), "anomaly": -3.0}),
    ])
    assert VisionDial().read(room) == 0.0


def test_reading_is_bounded():
    rooms = [_bright_active_room(), _dark_empty_room(), SignalRoom("empty"),
             SignalRoom("weird", [_camera(0, [1.0] * 16),
                                  _camera(1, {"brightness": 5.0, "motion": -1.0})])]
    for room in rooms:
        r = VisionDial().read(room)
        assert 0.0 <= r <= 1.0, r


def test_first_frame_always_processed():
    # A lone frame always counts — plato's deadband processes the first
    # frame unconditionally (prev is None → no diff to compare).
    room = SignalRoom("one", [_camera(0, {"brightness": 0.9, "motion": 0.9,
                                          "occupancy": 0.9, "anomaly": 0.0})])
    assert VisionDial().read(room) > 0.8


def test_deadband_exact_threshold_is_skipped():
    # diff == deadband is NOT significant (plato's is_significant_change is
    # strictly `diff > threshold`): a frame exactly at the threshold skips.
    dial = VisionDial(deadband=0.1)
    a = {"brightness": 0.5, "motion": 0.5, "occupancy": 0.5, "anomaly": 0.0}
    exact = dict(a, brightness=0.9)          # diff = 0.4 / 4 = 0.1 == deadband
    alone = dial.read(SignalRoom("a", [_camera(0, a)]))
    at_threshold = dial.read(SignalRoom("t", [_camera(0, a), _camera(1, exact)]))
    assert abs(alone - at_threshold) < 1e-9, (alone, at_threshold)


def test_occupancy_above_one_clamps():
    # Raw person counts clamp to 1 (plato: occupancy is a normalized count).
    r = VisionDial().read(SignalRoom("crowd", [_camera(0, {
        "brightness": 0.0, "motion": 0.0, "occupancy": 3, "anomaly": 0.0,
    })]))
    # occupancy 3 → clamped 1 → base = 0.25 (occupancy weight alone).
    assert 0.2 < r < 0.3, r


def test_dict_missing_keys_default_to_zero():
    # A dict naming only some fields reads the missing ones as 0 (not
    # unreadable): brightness 0.8, rest 0 → base = 0.40·0.8 = 0.32.
    r = VisionDial().read(SignalRoom("partial", [_camera(0, {"brightness": 0.8})]))
    assert 0.3 < r < 0.34, r


def test_read_is_stateless():
    # Two reads of the same room agree — no hidden state carried between
    # read() calls (the deadband history resets each read).
    dial = VisionDial()
    room = _bright_active_room()
    assert dial.read(room) == dial.read(room)


def test_satisfies_dial_abc_and_registered_ninth():
    assert isinstance(VisionDial(), Dial)
    assert len(DEFAULT_DIALS) == 9, [d.name for d in DEFAULT_DIALS]
    names = [d.name for d in DEFAULT_DIALS]
    assert names[-1] == "vision"
    assert names.index("vision") == 8
    v = [d for d in DEFAULT_DIALS if d.name == "vision"][0]
    assert isinstance(v, VisionDial)
    # It reads cleanly through the bank on a signal room.
    readings = DialBank([VisionDial()]).readings(_bright_active_room())
    assert "vision" in readings
    assert readings["vision"] > 0.6


if __name__ == "__main__":
    fns = [test_bright_active_reads_high, test_dark_empty_reads_low,
           test_bright_beats_dark, test_dict_form_and_16d_form_agree,
           test_plato_spellings_accepted, test_no_frames_is_neutral,
           test_text_room_has_no_visual_opinion, test_only_camera_frames_count,
           test_anomaly_is_a_bonus_spike,
           test_deadband_collapses_identical_frames,
           test_deadband_skips_small_changes_processes_large,
           test_deadband_zero_processes_everything,
           test_unreadable_frames_are_neutral,
           test_garbage_field_values_clamp_to_zero,
           test_first_frame_always_processed,
           test_deadband_exact_threshold_is_skipped,
           test_occupancy_above_one_clamps, test_dict_missing_keys_default_to_zero,
           test_read_is_stateless, test_reading_is_bounded,
           test_satisfies_dial_abc_and_registered_ninth]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print("\nAll vision-dial tests passed.")
