"""elephant — tests: the decomposition harness (the doctrine in code).

The captain's doctrine: a large model running a narrow task for a long
time can be decomposed into components that look like other components,
each with a simpler function, tuned over time by an algorithmic learning
mechanism (the guitarist principle) and varied by a stochastic mechanism
(temperature) when the application wants it. These tests pin the three
moves — distill, decompose, learn + vary — to code.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from elephant.decompose import (
    REWARD_THRESHOLD,
    Component,
    DecompositionHarness,
    input_features,
    output_features,
    pair_features,
)


# ---------------------------------------------------------------------- #
# Fixtures — the narrow task: a bar answering two ways                   #
# ---------------------------------------------------------------------- #
SHORT_OUT = "ok"
LONG_OUT = ("The tide came in slowly, and the boat rocked, and the crew "
            "watched the horizon for a long time, and it was a very "
            "elaborate and detailed story indeed.")
LONG_IN = ("please tell me a long elaborate story about the sea and "
           "the tide")


def two_behavior_trace(n=30):
    """Short terse inputs get short terse outputs; long inputs get long
    elaborate outputs — one trace, two clear behaviors."""
    trace = [(f"ok {i}", SHORT_OUT) for i in range(n)]
    trace += [(f"{LONG_IN} {i}", LONG_OUT) for i in range(n)]
    return trace


def warm_reward(inp, out):
    """A reward that prefers long elaborate answers (the application's
    taste)."""
    return 1.0 if len(out) > 30 or "elaborate" in out else 0.0


# ---------------------------------------------------------------------- #
# DISTILL — the trace decomposes into components that look alike         #
# ---------------------------------------------------------------------- #
def test_distill_two_behaviors_into_components_that_look_alike():
    h = DecompositionHarness(seed=7)
    h.ingest(two_behavior_trace())
    comps = h.distill(k=2)

    assert len(comps) == 2
    # Same shape: every organ has the same fields of the same types.
    shape0 = {k: type(v) for k, v in vars(comps[0]).items()}
    shape1 = {k: type(v) for k, v in vars(comps[1]).items()}
    assert shape0 == shape1
    for name in ("id", "prototype", "door", "output", "table",
                 "learning_rate", "temperature", "hits", "correct",
                 "score"):
        assert hasattr(comps[0], name)

    # Different specialization: one door faces short inputs, one long;
    # one answers terse, one elaborate.
    doors = sorted(round(float(c.door[0]), 6) for c in comps)
    assert doors[0] < doors[1]
    outs = {c.output for c in comps}
    assert outs == {SHORT_OUT, LONG_OUT}
    lens = sorted(len(c.output) for c in comps)
    assert lens[1] > lens[0] + 20  # terse vs elaborate, not alike in taste


def test_distill_caps_k_at_trace_length():
    h = DecompositionHarness(seed=1)
    h.ingest([("a", "b"), ("c", "d")])
    comps = h.distill(k=10)
    assert len(comps) == 2


def test_components_share_the_simple_interface():
    c = Component(id=0, prototype=np.zeros(4))
    assert c.accuracy() == 0.0
    c.hits, c.correct = 4, 3
    assert c.accuracy() == pytest.approx(0.75)
    # simple function: nearest stored input, fallback to representative
    assert c.respond(np.zeros(2)) == c.output
    c.table = [(np.zeros(2), "stored answer")]
    assert c.respond(np.zeros(2)) == "stored answer"


# ---------------------------------------------------------------------- #
# RESPOND — routing, and the stochastic knob                             #
# ---------------------------------------------------------------------- #
def test_respond_routes_to_the_right_component():
    h = DecompositionHarness(seed=7)
    h.ingest(two_behavior_trace())
    h.distill(k=2)

    assert h.respond("ok 5") == SHORT_OUT
    assert h.respond(f"{LONG_IN} 99") == LONG_OUT

    short_comp = h.route("ok 5")
    long_comp = h.route(f"{LONG_IN} 99")
    assert short_comp.output == SHORT_OUT
    assert long_comp.output == LONG_OUT


def test_temperature_zero_is_deterministic():
    h = DecompositionHarness(seed=7)
    h.ingest(two_behavior_trace())
    h.distill(k=2)
    outs = {h.respond("ok 5") for _ in range(50)}
    assert outs == {SHORT_OUT}


def test_temperature_above_zero_varies_output():
    # The stochastic mechanism: softmax over the nearest doors, so the
    # same body speaks in more than one voice when the app wants it.
    h = DecompositionHarness(seed=3)
    h.ingest(two_behavior_trace())
    h.distill(k=2)
    h.temperature = 1.0
    outs = {h.respond("ok 5") for _ in range(100)}
    assert len(outs) >= 2
    assert outs <= {SHORT_OUT, LONG_OUT}


def test_respond_temperature_override():
    h = DecompositionHarness(seed=3)
    h.ingest(two_behavior_trace())
    h.distill(k=2)
    # per-call override beats the harness default
    outs = {h.respond("ok 5", temperature=2.0) for _ in range(100)}
    assert len(outs) >= 2


# ---------------------------------------------------------------------- #
# LEARN — the algorithmic learning mechanism over time                   #
# ---------------------------------------------------------------------- #
def test_learn_improves_winning_component_over_epochs():
    h = DecompositionHarness(seed=1, temperature=0.3)
    h.ingest(two_behavior_trace())
    h.distill(k=2)

    h.learn(warm_reward, epochs=1)
    s1 = max(c.score for c in h.components)
    proto1 = {c.id: c.prototype.copy() for c in h.components}

    h.learn(warm_reward, epochs=4)
    s5 = max(c.score for c in h.components)
    winner = max(h.components, key=lambda c: c.score)

    # The algorithmic mechanism: the winner's score grew with running,
    # its prototype moved (settings discovered by running), and it
    # converged on the rewarded behavior.
    assert s5 > s1
    assert winner.accuracy() == pytest.approx(1.0)
    assert not np.allclose(proto1[winner.id], winner.prototype)


def test_learn_returns_epoch_curve():
    h = DecompositionHarness(seed=1, temperature=0.3)
    h.ingest(two_behavior_trace())
    h.distill(k=2)
    curve = h.learn(warm_reward, epochs=3)
    assert len(curve) == 3
    assert all(0.0 <= r <= 1.0 for r in curve)
    assert curve[-1] >= curve[0]  # the body warms, not cools


def test_learn_handles_nan_reward():
    h = DecompositionHarness(seed=1)
    h.ingest(two_behavior_trace())
    h.distill(k=2)
    # a broken reward_fn must not poison the learning loop
    def broken(inp, out):
        return float("nan") if "ok" in inp else 1.0
    h.learn(broken, epochs=2)
    assert all(math_isfinite(c.score) for c in h.components)


def math_isfinite(x):
    return x == x and abs(x) != float("inf")


# ---------------------------------------------------------------------- #
# SPECIALIZATION — the body after running                                #
# ---------------------------------------------------------------------- #
def test_specialization_shows_divergence():
    h = DecompositionHarness(seed=1, temperature=0.3)
    h.ingest(two_behavior_trace())
    h.distill(k=2)

    before = h.specialization()
    assert before["n"] == 2
    assert before["divergence"] == 0.0  # uniform organs at birth
    assert all(c["hits"] == 0 for c in before["components"])

    h.learn(warm_reward, epochs=5)
    after = h.specialization()

    # The organs have grown apart: scores differ, accuracy differs,
    # divergence is real.
    scores = {c["score"] for c in after["components"]}
    accs = {c["accuracy"] for c in after["components"]}
    assert len(scores) == 2
    assert len(accs) == 2
    assert after["divergence"] > 0.0
    # every organ still reports the same shape of facts
    for c in after["components"]:
        assert set(c) == {"id", "score", "hits", "correct", "accuracy",
                          "temperature", "door", "style", "output"}


# ---------------------------------------------------------------------- #
# Guards — NaN and empty traces                                          #
# ---------------------------------------------------------------------- #
def test_empty_trace_guarded():
    h = DecompositionHarness()
    with pytest.raises(ValueError):
        h.ingest([])


def test_distill_before_ingest_guarded():
    h = DecompositionHarness()
    with pytest.raises(ValueError):
        h.distill(k=2)


def test_respond_before_distill_guarded():
    h = DecompositionHarness()
    h.ingest(two_behavior_trace(3))
    with pytest.raises(ValueError):
        h.respond("ok 1")


def test_learn_before_distill_guarded():
    h = DecompositionHarness()
    h.ingest(two_behavior_trace(3))
    with pytest.raises(ValueError):
        h.learn(warm_reward, epochs=1)


def test_bad_k_guarded():
    h = DecompositionHarness()
    h.ingest(two_behavior_trace(3))
    with pytest.raises(ValueError):
        h.distill(k=0)
    with pytest.raises(ValueError):
        h.distill(k=-2)


def test_malformed_trace_item_guarded():
    h = DecompositionHarness()
    with pytest.raises(ValueError):
        h.ingest([("only one field")])
    with pytest.raises(ValueError):
        h.ingest([("", "")])  # a teacher that says nothing teaches nothing


def test_empty_output_trace_stays_finite():
    # outputs that are empty strings (or input-less) must not produce
    # NaN features or NaN prototypes
    h = DecompositionHarness(seed=2)
    h.ingest([("hi", ""), ("hello there", "ok"), ("...", ""), ("!", "y")])
    comps = h.distill(k=2)
    for c in comps:
        assert np.all(np.isfinite(c.prototype))
    # respond still works — returns strings, never NaN garbage
    for _ in range(3):
        assert isinstance(h.respond("hi"), str)
        assert isinstance(h.respond("hello there"), str)


# ---------------------------------------------------------------------- #
# The feature functions — simple, finite, monotone in the obvious ways   #
# ---------------------------------------------------------------------- #
def test_features_are_finite_and_sensible():
    assert np.all(np.isfinite(input_features("hello")))
    assert np.all(np.isfinite(output_features("a b", "a b c")))
    assert np.all(np.isfinite(pair_features("a b", "a b c")))
    # longer inputs/outputs -> longer features; overlap detects shared
    # tokens; entropy rewards varied language
    assert input_features("a b c d")[0] > input_features("a")[0]
    assert output_features("a", "a b c d e")[0] > output_features("a", "a")[0]
    assert output_features("a b", "a b c")[1] > output_features("a b", "z z")[1]
    assert output_features("a", "abracadabra abracadabra")[2] > \
        output_features("a", "aaaaaa")[2]
    # punctuation energy is per-word, so it stays bounded
    assert input_features("WAIT!!!")[1] > input_features("wait")[1]
