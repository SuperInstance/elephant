"""elephant — tests: the Dual-DB bridge (Z_in perception, Z_out prediction).

The room perceives itself (Z_in, from plato-perception) and predicts
itself (Z_out, from plato-prediction). The trend dial — the room's own
predicted direction — and the anomaly sense join the field.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elephant.dualdb import (
    DualDBRoom, EncodingMethod, PredEncodingMethod, PredictionEncoder,
    PredictionOutput, PredictionType, ZInEncoder, ZOutPredictor,
    TrendDial, AnomalyDial,
)
from elephant.dial import DialBank
from elephant.room import Message, Room
from elephant.sensors import SensorFrame, SignalRoom


def _sounder_frame(ts: float, value: float) -> SensorFrame:
    return SensorFrame(ts=ts, sensor="sounder", data=value)


def _radar_frame(ts: float, targets) -> SensorFrame:
    return SensorFrame(ts=ts, sensor="radar", data=targets)


# --------------------------------------------------------------------- #
# Z_in — perception encoding                                            #
# --------------------------------------------------------------------- #
def test_zin_raw_encoding_sane():
    enc = ZInEncoder()
    f = _sounder_frame(ts=1000.0, value=0.7)
    v = enc.encode(f, EncodingMethod.RAW)
    assert v.shape == (3,)
    assert abs(v[0] - 0.7) < 1e-12          # value
    assert abs(v[1] - 1.0) < 1e-12          # confidence
    assert abs(v[2] - 0.001) < 1e-12        # timestamp_norm = ts/1e6
    # Point-cloud radar frames scalarize to a spread >= 0.
    r = _radar_frame(2000.0, [(0.0, 0.0), (0.0, 4.0), (3.0, 0.0)])
    spread = ZInEncoder.scalarize(r.data)
    assert spread > 0.0
    assert spread < 4.0


def test_zin_encoding_methods():
    enc = ZInEncoder()
    frames = [_sounder_frame(1000.0 + i * 1000.0, 0.1 * i) for i in range(3)]
    # Raw with no output_dim: 3 readings x 3 = 9.
    assert enc.encode(frames, EncodingMethod.RAW).shape == (9,)
    # Projections resize to output_dim.
    for method in (EncodingMethod.HASH_PROJECTION,
                   EncodingMethod.RANDOM_PROJECTION,
                   EncodingMethod.LEARNED_PROJECTION):
        proj = ZInEncoder(output_dim=8, seed=42)
        v = proj.encode(frames, method)
        assert v.shape == (8,)
        assert np.all(np.isfinite(v))
    # Determinism: same seed -> identical vectors.
    a = ZInEncoder(output_dim=8, seed=7).encode(frames, EncodingMethod.RANDOM_PROJECTION)
    b = ZInEncoder(output_dim=8, seed=7).encode(frames, EncodingMethod.RANDOM_PROJECTION)
    assert np.array_equal(a, b)
    # Different seeds -> different projections.
    c = ZInEncoder(output_dim=8, seed=99).encode(frames, EncodingMethod.RANDOM_PROJECTION)
    assert not np.array_equal(a, c)


def test_zin_normalized_zscore():
    enc = ZInEncoder()
    frames = [_sounder_frame(ts, v) for ts, v in ((0, 10.0), (1, 20.0), (2, 30.0))]
    v = enc.encode(frames, EncodingMethod.NORMALIZED)
    vals = v[0::3]
    assert abs(vals.mean()) < 1e-9
    assert abs(vals.std() - 1.0) < 1e-9
    # Sign pattern preserved: 10 < 20 < 30.
    assert vals[0] < vals[1] < vals[2]


def test_zin_dial_readings_dict():
    enc = ZInEncoder()
    v = enc.encode({"mood": 0.8, "panic": 0.1}, EncodingMethod.RAW)
    assert v.shape == (6,)
    assert abs(v[0] - 0.8) < 1e-12     # mood value
    assert abs(v[3] - 0.1) < 1e-12     # panic value


def test_zin_confidence_from_meta():
    enc = ZInEncoder()
    f = SensorFrame(ts=500.0, sensor="sounder", data=0.5,
                    meta={"confidence": 0.6})
    v = enc.encode(f, EncodingMethod.RAW)
    assert abs(v[1] - 0.6) < 1e-12


# --------------------------------------------------------------------- #
# Z_out — prediction                                                    #
# --------------------------------------------------------------------- #
def test_prediction_warming_series():
    pred = ZOutPredictor()
    series = [np.array([i, i * 0.5, i * 0.25], dtype=float) for i in range(8)]
    outs = pred.predict(series)
    by_type = {o.prediction_type: o for o in outs}
    assert by_type[PredictionType.TREND].value > 0.0           # warming
    assert by_type[PredictionType.TREND].confidence > 0.7      # high confidence
    assert by_type[PredictionType.VALUE_PREDICTION].value > np.mean(series[-1])
    assert 0.0 <= by_type[PredictionType.ANOMALY_SCORE].value <= 0.5  # steady ramp is not anomalous
    assert len(outs) == 3


def test_prediction_cooling_series():
    pred = ZOutPredictor()
    series = [np.array([10 - i, 5 - i * 0.5], dtype=float) for i in range(8)]
    trend = {o.prediction_type: o for o in pred.predict(series)}[PredictionType.TREND]
    assert trend.value < 0.0                                    # cooling


def test_anomaly_spikes_on_deviation():
    pred = ZOutPredictor()
    stable = [np.array([5.0, 5.0, 5.0]) for _ in range(10)] + [np.array([5.0, 5.0, 5.0])]
    spike = [np.array([5.0, 5.0, 5.0]) for _ in range(10)] + [np.array([50.0, 5.0, 5.0])]
    a_stable = {o.prediction_type: o for o in pred.predict(stable)}[PredictionType.ANOMALY_SCORE].value
    a_spike = {o.prediction_type: o for o in pred.predict(spike)}[PredictionType.ANOMALY_SCORE].value
    assert a_spike > 0.5
    assert a_spike > a_stable


def test_prediction_encoding_methods():
    po = PredictionOutput(PredictionType.VALUE_PREDICTION, 1.0, 0.5)
    raw = PredictionEncoder(method=PredEncodingMethod.RAW).encode(po)
    assert raw.shape == (3,) and abs(raw[0] - 1.0) < 1e-12
    conf = PredictionEncoder(method=PredEncodingMethod.CONFIDENCE).encode(po)
    assert abs(conf[0] - 0.5) < 1e-12          # value * confidence
    hier = PredictionEncoder(method=PredEncodingMethod.HIERARCHICAL).encode(po)
    assert hier.shape == (6,)                  # 3 raw + mean/max/min
    head = PredictionEncoder(method=PredEncodingMethod.MULTI_HEAD, heads=2).encode(po)
    assert head.shape == (6,)
    assert np.array_equal(head[:3], head[3:])  # repeated
    padded = PredictionEncoder(output_dim=4, method=PredEncodingMethod.RAW).encode(po)
    assert padded.shape == (4,) and padded[3] == 0.0


def test_prediction_output_vector():
    po = PredictionOutput(PredictionType.TREND, -0.5, 0.8)
    assert np.allclose(po.to_vector(), [-0.5, 0.8, 5.0])
    mt = PredictionOutput(PredictionType.MULTI_TARGET, 0.0, 1.0, targets=(1.0, 2.0, 3.0))
    assert np.allclose(mt.to_vector(), [0.0, 1.0, 6.0, 1.0, 2.0, 3.0])


# --------------------------------------------------------------------- #
# DualDBRoom — the bridge                                               #
# --------------------------------------------------------------------- #
def _warming_room(values=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)) -> SignalRoom:
    room = SignalRoom("sounder")
    for i, v in enumerate(values):
        room.frames.append(_sounder_frame(ts=1000.0 * (i + 1), value=v))
    room.frames.sort(key=lambda f: f.ts)
    return room


def test_trend_dial_signs():
    # Warming room -> positive trend dial.
    warm = DualDBRoom(_warming_room())
    for i in range(6):
        warm.room.frames.append(_sounder_frame(ts=1000.0 * (7 + i), value=0.6 + 0.05 * i))
        warm.room.frames.sort(key=lambda f: f.ts)
        warm.perceive()
    warm.predict()
    assert warm.trend_dial() > 0.0
    assert -1.0 <= warm.trend_dial() <= 1.0

    # Cooling room -> negative trend dial.
    cool = DualDBRoom(_warming_room())
    for i in range(6):
        cool.room.frames.append(_sounder_frame(ts=1000.0 * (7 + i), value=0.6 - 0.05 * i))
        cool.room.frames.sort(key=lambda f: f.ts)
        cool.perceive()
    cool.predict()
    assert cool.trend_dial() < 0.0

    # Flat room -> near-zero trend dial.
    flat = DualDBRoom(_warming_room((0.4,) * 6))
    for i in range(6):
        flat.room.frames.append(_sounder_frame(ts=1000.0 * (7 + i), value=0.4))
        flat.room.frames.sort(key=lambda f: f.ts)
        flat.perceive()
    flat.predict()
    assert abs(flat.trend_dial()) < 0.2, flat.trend_dial()


def test_dualdb_room_flow():
    room = _warming_room()
    room.frames.append(_radar_frame(1000.0, [(0.0, 0.0), (0.0, 4.0)]))
    room.frames.append(_radar_frame(2000.0, [(0.0, 0.0), (0.0, 3.0)]))
    room.frames.append(_radar_frame(3000.0, [(0.0, 0.0), (0.0, 2.0)]))
    room.frames.sort(key=lambda f: f.ts)

    bridge = DualDBRoom(room)
    for i in range(6):
        # The room evolves between perceptions: biomass thickens and the
        # fleet tightens, so the Z_in series carries a real trend.
        t = 4000.0 + i * 1000.0
        room.frames.append(_sounder_frame(t, 0.6 + 0.05 * i))
        room.frames.append(_radar_frame(t, [(0.0, 0.0), (0.0, 2.0 - 0.2 * i)]))
        room.frames.sort(key=lambda f: f.ts)
        bridge.perceive()
    outs = bridge.predict()
    assert len(outs) == 3
    assert len(bridge.zin) == 6
    assert bridge.perception_history().shape == (6, bridge.zin[0].shape[0])
    assert len(bridge.prediction_history()) == 3
    assert len(bridge.prediction_history(PredictionType.TREND)) == 1
    assert 0.0 <= bridge.anomaly() <= 1.0
    dial = bridge.dial()
    assert set(dial) == {"trend_dial", "anomaly"}
    assert "DualDBRoom" in repr(bridge)
    # Radar tightening (targets closing in) contributes to the warming feel.
    assert bridge.trend_dial() > 0.0


def test_dualdb_room_text_room():
    room = Room("warm", [
        Message("a", "I love this place, it's warm and kind.", ts=0),
        Message("b", "Haha, truly! cheers everyone", ts=2),
        Message("a", "we built it together, honestly", ts=4),
    ])
    bridge = DualDBRoom(room)
    for _ in range(4):
        bridge.perceive()
        room.messages.append(Message("c", "another warm line in the tap room", ts=10))
        room.messages.sort(key=lambda m: m.ts)
    outs = bridge.predict()
    assert len(outs) == 3
    assert -1.0 <= bridge.trend_dial() <= 1.0
    assert 0.0 <= bridge.anomaly() <= 1.0


def test_on_pulse_callback():
    seen = {}

    def pulse(zin_vec, zout_outs):
        seen["zin"] = zin_vec
        seen["zout"] = zout_outs

    bridge = DualDBRoom(_warming_room(), on_pulse=pulse)
    for i in range(6):
        bridge.perceive()
    bridge.predict()
    assert seen["zin"] is not None and seen["zin"].shape[0] > 0
    assert len(seen["zout"]) == 3


def test_insufficient_history_honest():
    bridge = DualDBRoom(_warming_room())
    bridge.perceive()
    assert bridge.predict() == []          # one perception: nothing to predict
    assert bridge.trend_dial() == 0.0      # auto-predict still honest


def test_dials_join_bank_pulse_seam():
    # The trend + anomaly senses cast as real Dials — they join a
    # DialBank unchanged (the seam to pulse.py's PulseLoop / read_field).
    bridge = DualDBRoom(_warming_room())
    for i in range(6):
        bridge.room.frames.append(_sounder_frame(ts=1000.0 * (7 + i), value=0.6 + 0.05 * i))
        bridge.room.frames.sort(key=lambda f: f.ts)
        bridge.perceive()
    bank = DialBank([TrendDial(bridge), AnomalyDial(bridge)])
    readings = bank.readings(bridge.room)
    assert set(readings) == {"trend_dial", "anomaly"}
    assert readings["trend_dial"] == bridge.trend_dial() > 0.0
    assert 0.0 <= readings["anomaly"] <= 1.0
