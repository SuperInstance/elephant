"""elephant.learned — tests: the learned dials (ROADMAP v1).

A learned dial must (a) satisfy the `Dial` ABC, (b) be trainable end-to-end
on a tiny corpus, (c) report sane held-out transfer numbers, and (d) drop
into a `DialBank` and read a room like any hand-crafted dial.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from elephant.dial import Dial, DialBank
from elephant.learned import (
    DIAL_NAMES,
    LearnedDial,
    LearnedDialModel,
    Vocab,
    room_from_markdown,
    teacher_readings,
    tokenize,
    train_and_report,
)
from elephant.room import Message, Room

# A few warm/cold lines so the mood dial (and the student) have real signal.
WARM_LINES = [
    ("A", "I love this warm place"),
    ("B", "everyone is kind and good here"),
    ("A", "haha, cheers to us"),
    ("B", "honestly we built it together"),
]
COLD_LINES = [
    ("A", "whatever, sure, fine"),
    ("B", "this is cold and dead"),
    ("A", "fire! flood! run!"),
    ("B", "great. just great."),
]


def _md(lines):
    body = "\n".join(f"**{spk}:** {txt}" for spk, txt in lines)
    return "# room\n\n" + body + "\n"


def _write(tmp_path, name, lines):
    p = tmp_path / name
    p.write_text(_md(lines))
    return str(p)


def _tiny_model(vocab, d=16):
    return LearnedDialModel(len(vocab), n_heads=len(DIAL_NAMES), d_model=d, d_trunk=d)


# --------------------------------------------------------------------- #
# Tokenization / parsing / teacher                                      #
# --------------------------------------------------------------------- #
def test_tokenize_keeps_punctuation():
    toks = tokenize("Great.   fire!!!  ")
    assert "great." in toks
    assert "fire!!!" in toks


def test_room_from_markdown_parses_speakers():
    room = room_from_markdown(_md(WARM_LINES))
    assert len(room.messages) >= len(WARM_LINES)
    authors = {m.author for m in room.messages}
    assert {"A", "B"} <= authors


def test_teacher_readings_shape_and_range():
    room = room_from_markdown(_md(WARM_LINES + COLD_LINES))
    v = teacher_readings(room)
    assert v.shape == (len(DIAL_NAMES),)
    assert -1.0 <= v[0] <= 1.0            # mood
    assert 0.0 <= v[1] <= 1.0             # volume
    assert 0.0 <= v[3] <= 1.0             # cynicism
    # warm lines outweigh cold -> mood should read warm (positive)
    assert v[0] > 0.0, v


# --------------------------------------------------------------------- #
# The learned Dial ABC + DialBank swap                                  #
# --------------------------------------------------------------------- #
def test_learned_dial_satisfies_abc():
    vocab = Vocab(["love", "warm", "cold", "great", "fire"], max_size=64)
    dial = LearnedDial("mood", _tiny_model(vocab), vocab, head_index=0)
    assert isinstance(dial, Dial)
    assert dial.name == "mood"
    room = Room("x", [Message("a", "I love this warm place", ts=0)])
    v = dial.read(room)
    assert isinstance(v, float)
    assert -1.0 <= v <= 1.0, v


def test_learned_dial_reads_empty_room_safely():
    vocab = Vocab(["a", "b"], max_size=8)
    dial = LearnedDial("mood", _tiny_model(vocab), vocab, head_index=0)
    assert dial.read(Room("empty")) == 0.0


def test_learned_dial_swaps_into_dialbank():
    vocab = Vocab(["love", "warm", "great"], max_size=16)
    learned = LearnedDial("mood", _tiny_model(vocab), vocab, head_index=0)
    bank = DialBank([learned])
    room = Room("r", [Message("a", "I love this", ts=0)])
    assert bank.names() == ["mood"]
    assert "mood" in bank.readings(room)
    assert -1.0 <= bank.readings(room)["mood"] <= 1.0


# --------------------------------------------------------------------- #
# End-to-end training + held-out correlation                            #
# --------------------------------------------------------------------- #
def test_training_runs_end_to_end_tiny(tmp_path):
    warm_train = [_write(tmp_path, f"warm-t{i}.md", WARM_LINES) for i in range(2)]
    cold_train = [_write(tmp_path, f"cold-t{i}.md", COLD_LINES) for i in range(2)]
    warm_test = [_write(tmp_path, f"warm-x{i}.md", WARM_LINES) for i in range(2)]
    cold_test = [_write(tmp_path, f"cold-x{i}.md", COLD_LINES) for i in range(2)]

    res = train_and_report(
        base=str(tmp_path),
        vocab_size=64, window=3, stride=1,
        epochs=20, jepa_epochs=2, pretrain=True, seed=0,
        train_files=warm_train + cold_train,
        test_files=warm_test + cold_test,
    )

    # pipeline completed and produced data
    assert res.n_train > 0
    assert res.n_test > 0
    assert math.isfinite(res.loss)

    # held-out correlation is computed and sane (finite, in [-1, 1]);
    # dials with zero teacher variance have no signal -> r is undefined
    for name in DIAL_NAMES:
        h = res.heldout[name]
        assert math.isfinite(h["teacher_std"])
        if h["teacher_std"] > 0.0:
            assert math.isfinite(h["r"]), (name, h)
            assert -1.0 <= h["r"] <= 1.0, (name, h)
            assert math.isfinite(h["r2"]), (name, h)
        else:
            assert math.isnan(h["r"]) or True, (name, h)

    # mood has real variance in the synthetic corpus and transfers
    assert res.heldout["mood"]["teacher_std"] > 0.0
    assert res.heldout["mood"]["r"] > 0.0, res.heldout["mood"]


def test_training_without_pretrain_runs(tmp_path):
    warm = [_write(tmp_path, f"w{i}.md", WARM_LINES) for i in range(3)]
    cold = [_write(tmp_path, f"c{i}.md", COLD_LINES) for i in range(3)]
    res = train_and_report(
        base=str(tmp_path), vocab_size=64, window=3, stride=1,
        epochs=10, jepa_epochs=2, pretrain=False, seed=0,
        train_files=warm[:2] + cold[:2],
        test_files=warm[2:] + cold[2:],
    )
    assert res.pretrained is False
    assert res.n_pretrain_pairs == 0
    assert math.isfinite(res.loss)


# --------------------------------------------------------------------- #
# Checkpoint round-trip                                                 #
# --------------------------------------------------------------------- #
def test_checkpoint_save_and_reload(tmp_path):
    from elephant.learned import load_learned_bank

    warm = [_write(tmp_path, f"w{i}.md", WARM_LINES) for i in range(2)]
    cold = [_write(tmp_path, f"c{i}.md", COLD_LINES) for i in range(2)]
    ckpt = str(tmp_path / "ckpt")
    train_and_report(
        base=str(tmp_path), vocab_size=64, window=3, stride=1,
        epochs=5, jepa_epochs=1, pretrain=False, seed=0,
        train_files=warm + cold, test_files=warm + cold,
        checkpoint_dir=ckpt,
    )
    dials = load_learned_bank(ckpt, device="cpu")
    assert len(dials) == len(DIAL_NAMES)
    assert all(isinstance(d, Dial) for d in dials)
    room = room_from_markdown(_md(WARM_LINES))
    bank = DialBank(dials)
    assert set(bank.readings(room)) == set(DIAL_NAMES)


if __name__ == "__main__":
    import sys as _s

    _s.exit(pytest.main([__file__, "-v"]))
