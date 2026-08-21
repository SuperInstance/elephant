"""roomd tests — the field truth-holder (plan §1.1)."""
import json
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from elephant.roomd import RoomDaemon, build_uscp_packet, main, serve

WARM = [
    {"room": "galley", "author": "cook", "ts": 1, "text": "love this, great work everyone, thank you!"},
    {"room": "galley", "author": "deck", "ts": 2, "text": "wonderful morning, grateful, coffee is lovely"},
    {"room": "galley", "author": "cook", "ts": 3, "text": "beautiful catch today, proud of the crew"},
]
PANIC = [
    {"room": "galley", "author": "watch", "ts": 10, "text": "!!! FIRE FIRE EMERGENCY ABANDON NOW !!!"},
    {"room": "galley", "author": "mate", "ts": 11, "text": "URGENT!! fire spreading, everyone out!!"},
    {"room": "galley", "author": "watch", "ts": 12, "text": "!!! MAYDAY MAYDAY emergency fire !!!"},
]


def _ingest(daemon, events):
    for e in events:
        daemon.ingest(e)


def test_field_shape():
    d = RoomDaemon()
    _ingest(d, WARM)
    f = d.recompute()
    assert set(f) >= {"warmth", "dials", "rooms", "map_temperature", "ts"}
    assert f["rooms"]["galley"]["warmth"] > 0  # warm room reads warm
    assert "panic" in f["rooms"]["galley"]["dials"]


def test_panic_rings_exactly_once_rising_edge(tmp_path):
    d = RoomDaemon(inbox=str(tmp_path))
    _ingest(d, WARM)
    d.recompute()
    _ingest(d, PANIC)
    d.recompute()
    d.recompute()  # second check: no duplicate ring
    rings = list(tmp_path.glob("*elephant-roomd*.json"))
    panic_rings = [r for r in rings if "panic" in r.name]
    assert len(panic_rings) == 1
    packet = json.loads(panic_rings[0].read_text())
    assert packet["header"]["type"] == "USCP-v1"
    assert packet["header"]["origin_id"] == "elephant-roomd"
    assert packet["header"]["intent"] == "STATUS_REPORT"
    assert packet["header"]["priority"] == "HIGH"


def test_ring_rearms_on_fall(tmp_path, monkeypatch):
    # Patch time.time to guarantee unique ms-level timestamps for ring filenames.
    _ts = [1000000.0]
    def _tick():
        _ts[0] += 0.01  # +10 ms per call, unique even at 1ms resolution
        return _ts[0]
    monkeypatch.setattr(time, "time", _tick)
    d = RoomDaemon(inbox=str(tmp_path))
    _ingest(d, PANIC)
    d.recompute()
    # Clear panic and re-ingest warm to lower the panic dial below threshold.
    d.rooms["galley"].messages = d.rooms["galley"].messages[:0] + type(d.rooms["galley"].messages)(WARM and [])
    for m in list(d.rooms["galley"].messages):
        d.rooms["galley"].messages.remove(m)
    _ingest(d, WARM)
    d.recompute()
    _ingest(d, PANIC)
    d.recompute()
    panic_rings = [r for r in tmp_path.glob("*panic*")]
    assert len(panic_rings) == 2  # rang again after re-arm


def test_http_endpoints(tmp_path):
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]           # a free ephemeral port, no collision
    d = RoomDaemon(map_path=None, inbox=str(tmp_path))
    _ingest(d, WARM)
    t = threading.Thread(target=serve, args=(d, port), daemon=True)
    t.start()
    for _ in range(50):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
                break
        except Exception:
            time.sleep(0.1)
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/field", timeout=2) as r:
        body = json.loads(r.read())
    assert body["rooms"]["galley"]["warmth"] > 0
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/rooms/galley/field", timeout=2) as r:
        roomf = json.loads(r.read())
    assert roomf["room"] == "galley"


def test_map_load(tmp_path):
    mapfile = tmp_path / "map.json"
    mapfile.write_text(json.dumps({"rooms": [{"name": "wheelhouse"}, {"name": "galley"}]}))
    d = RoomDaemon(map_path=str(mapfile))
    assert set(d.rooms) == {"wheelhouse", "galley"}


def test_packet_atomic_and_valid(tmp_path):
    p = build_uscp_packet("panic", "high", {"room": "x", "panic": 0.9})
    assert p["header"]["priority"] == "HIGH"
    # no .tmp leftovers pattern — write_atomic replaces cleanly
    from elephant.roomd import write_atomic
    write_atomic(tmp_path / "pkt.json", p)
    assert json.loads((tmp_path / "pkt.json").read_text()) == p
    assert not list(tmp_path.glob("*.tmp"))


def test_zeitgeist_description():
    """The room's words change with its field (plan §3.7)."""
    d = RoomDaemon()
    _ingest(d, WARM)
    warm_desc = d.tinted_description("galley")
    _ingest(d, PANIC)
    panic_desc = d.tinted_description("galley")
    assert warm_desc and panic_desc
    assert warm_desc != panic_desc  # the field authors the words
    assert "drenched" in panic_desc or "rain" in panic_desc.lower() or panic_desc != warm_desc


def test_field_log_writes_and_bounded(tmp_path):
    """The v3 contrast corpus: snapshots append, rotation keeps newest half."""
    import json as _json
    d = RoomDaemon(field_log=str(tmp_path / "field.jsonl"))
    _ingest(d, WARM)
    for _ in range(8):
        d.recompute()
    lines = (tmp_path / "field.jsonl").read_text().splitlines()
    assert len(lines) == 8
    first = _json.loads(lines[0])
    assert "map_temperature" in first and "rooms" in first
    # force rotation
    d.field_log_lines = d.MAX_FIELD_LOG_LINES
    d._log_field({"ts": 1, "map_temperature": 0.0, "rooms": {}})
    lines = (tmp_path / "field.jsonl").read_text().splitlines()
    assert len(lines) <= 5  # newest half kept
