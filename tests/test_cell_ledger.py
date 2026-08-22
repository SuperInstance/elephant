"""elephant — tests: the cell-ledger producer (quilt bridge, synergy missing-link ②)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elephant.cell_ledger import CellLedgerProducer, genesis_commit
from elephant.roomd import DialBank, DEFAULT_DIALS  # noqa: F401 (import guard)


def test_genesis_commit_is_stable():
    g1 = genesis_commit("room.field.living-room", 1000.0)
    g2 = genesis_commit("room.field.living-room", 1000.0)
    assert g1 == g2
    assert len(g1) == 64


def test_record_chains_seals_in_order():
    p = CellLedgerProducer("room.field.living-room")
    e1 = p.record({"cell_id": "room.field.living-room", "kind": "field",
                   "ts": 1.0, "warmth": 0.1, "kappa": 24.0})
    e2 = p.record({"cell_id": "room.field.living-room", "kind": "field",
                   "ts": 2.0, "warmth": 0.2, "kappa": 23.0})
    assert e1 is not None and e2 is not None
    assert e1["seq"] == 1 and e2["seq"] == 2
    assert e1["prev_hash"] == genesis_commit("room.field.living-room", 1.0)
    assert e2["prev_hash"] == e1["hash"]
    assert e1["hash"] != e2["hash"]
    assert len(e1["hash"]) == 64 and len(e2["hash"]) == 64


def test_record_none_is_noop():
    p = CellLedgerProducer("room.field.x")
    assert p.record(None) is None
    assert p._seq == 0
    assert p._head == ""


def test_two_producers_independent_chains():
    a = CellLedgerProducer("room.a")
    b = CellLedgerProducer("room.b")
    ea = a.record({"cell_id": "room.a", "ts": 1.0})
    eb = b.record({"cell_id": "room.b", "ts": 1.0})
    assert ea["prev_hash"] != eb["prev_hash"]
    assert ea["hash"] != eb["hash"]
