"""tests/test_contrast.py — unit tests for the v3 contrast machinery."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from elephant.contrast import (
    condition_edge, contrast_loss, parse_script_blocks, probe_report,
    room_spread, sample_room_batches, separability, speaker_key_from_filename,
    spread_hinge, text_clips_from_room, vmf_fit_generic,
)

torch = pytest.importorskip("torch")


def test_contrast_loss_prefers_separated_rooms():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(6, 8, generator=g)
    z[3:] += 3.0
    rooms = ["a", "a", "a", "b", "b", "b"]
    assert contrast_loss(z, rooms) < contrast_loss(torch.randn(6, 8, generator=g), rooms)


def test_contrast_loss_needs_same_room_partner():
    with pytest.raises(ValueError):
        contrast_loss(torch.randn(3, 8), ["a", "b", "c"])


def test_spread_hinge_fires_on_collapse():
    zc = torch.ones(3, 8)
    assert spread_hinge(zc, ["a", "a", "a"], {"a": 0.4}) > 0.0
    zs = torch.randn(3, 8, generator=torch.Generator().manual_seed(1)) * 3
    assert spread_hinge(zs, ["a", "a", "a"], {"a": 0.4}) == 0.0


def test_vmf_fit_generic_recovers_direction():
    rng = np.random.default_rng(0)
    mu = np.zeros(16)
    mu[2] = 1.0
    X = mu + 0.05 * rng.standard_normal((40, 16))
    f = vmf_fit_generic(X)
    assert f is not None
    assert f["mu_hat"][2] > 0.99
    assert f["rho"] > 0.9


def test_condition_edge_deadband_logic():
    rng = np.random.default_rng(0)
    A = np.stack([np.eye(16)[0] + 0.05 * rng.standard_normal(16)
                  for _ in range(30)])
    B = np.stack([np.eye(16)[1] + 0.05 * rng.standard_normal(16)
                  for _ in range(30)])
    fa = vmf_fit_generic(A, seed=0, B=20)
    fb = vmf_fit_generic(B, seed=0, B=20)
    e = condition_edge(fa, fb)
    assert e["real"] is True
    assert e["d_mu"] > e["deadband"]
    e_self = condition_edge(fa, fa)
    assert e_self["real"] is False  # zero displacement can't clear the band


def test_separability_gap_positive_for_clustered_rooms():
    rng = np.random.default_rng(0)
    z = np.vstack([
        rng.normal([5, 0], 0.1, size=(10, 2)),
        rng.normal([-5, 0], 0.1, size=(10, 2)),
    ])
    rooms = ["a"] * 10 + ["b"] * 10
    sep = separability(z, rooms)
    assert sep["gap"] > 0.5


def test_speaker_key_from_filename():
    assert speaker_key_from_filename("carpenter-build.mp3") == "carpenter"
    assert speaker_key_from_filename("episode-2-lucineer-intro.mp3") == "lucineer"


def test_parse_script_blocks():
    txt = "# T\n\n**WELDER (heat):** line0\n\n> `welder-scam`\n> So we all do it right.\n\n"
    with open("/tmp/_script_test.md", "w") as f:
        f.write(txt)
    blocks = parse_script_blocks("/tmp/_script_test.md")
    assert len(blocks) == 1
    slug, spk, text = blocks[0]
    assert slug == "welder-scam"
    assert spk == "WELDER"
    assert "do it right" in text


def test_sample_room_batches_complete_and_seeded():
    rooms = ["a"] * 3 + ["b"] * 4 + ["c"] * 2
    b1 = sample_room_batches(rooms, 5, __import__("random").Random(0))
    b2 = sample_room_batches(rooms, 5, __import__("random").Random(0))
    assert b1 == b2
    for idx in b1:
        rs = {rooms[i] for i in idx}
        # every room in the batch contributes ALL of its clips
        for r in rs:
            assert sorted(i for i in idx if rooms[i] == r) == \
                sorted(i for i, x in enumerate(rooms) if x == r)


def test_text_clips_nonoverlapping_and_speaker():
    from elephant.room import Message
    msgs = [Message(author=("a" if i % 2 else "b"), text=f"word{i} " * 6,
                    ts=float(i)) for i in range(16)]
    clips = text_clips_from_room("r", msgs, window=8)
    assert len(clips) == 2          # non-overlapping
    spk0 = clips[0][0].speaker
    assert spk0 in {"a", "b"}
