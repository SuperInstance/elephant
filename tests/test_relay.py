"""elephant — tests: the crab-traps relay limb (sealed reads → POST /edge)."""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elephant.relay import CrabTrapRelay, build_edge


READING = {"room": "living-room", "warmth": 0.10, "kappa": 1.20,
           "dials": {"mood": 0.5}, "messages": 3, "ts": 1_000.0}


def test_build_edge_genesis_is_unscored():
    e = build_edge("room.field.living-room", None, READING, 1_000.0, None)
    assert e["v"] == 1
    assert e["cell"] == "room.field.living-room"
    assert e["ts"] == 1_000_000                     # epoch millis
    assert e["before"] is None
    assert e["after"] is READING
    assert e["imbalance"] is None                    # never fake a number
    assert e["delta"]["changed"] is False
    assert e["chain"] is None                        # genesis edge
    assert e["provenance"]["caller"] == "elephant-roomd"


def test_build_edge_persistence_prior_scores_the_edge():
    before = dict(READING, warmth=0.10)
    after = dict(READING, warmth=0.35)
    e = build_edge("room.field.living-room", before, after, 2_000.0, "ab" * 32)
    assert e["imbalance"] == 0.25                    # surprise IS the edge
    assert e["delta"]["magnitude"] == 0.25
    assert e["delta"]["changed"] is True
    assert e["chain"] == "ab" * 32


class _RelayDouble(BaseHTTPRequestHandler):
    """Fake crab-traps relay: records edges, hands back chain_head.

    Modes: ok (always 201); conflict_once (one 409 carrying expected_head,
    then back to ok — a lost response healed by the retry); conflict
    (409 forever — a wedged ledger must not wedge the limb).
    """
    edges = []
    mode = "ok"

    def log_message(self, *args):
        pass

    def _respond(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _RelayDouble.edges.append(body)
        if _RelayDouble.mode == "conflict":
            return self._respond(409, {"success": False, "error": "chain broken",
                                       "expected_head": "ff" * 32})
        if _RelayDouble.mode == "conflict_once":
            _RelayDouble.mode = "ok"
            return self._respond(409, {"success": False, "error": "chain broken",
                                       "expected_head": "ef" * 32})
        return self._respond(201, {"success": True, "recorded": True,
                                   "chain_head": f"{len(_RelayDouble.edges):064x}"})


def _serve():
    srv = HTTPServer(("127.0.0.1", 0), _RelayDouble)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _await(cond, tries=100):
    for _ in range(tries):
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_submit_forwards_and_tracks_chain_head():
    _RelayDouble.edges.clear()
    _RelayDouble.mode = "ok"
    srv = _serve()
    try:
        relay = CrabTrapRelay(f"http://127.0.0.1:{srv.server_address[1]}").start()
        assert relay.submit("room.field.a", None, READING, 1_000.0)
        assert relay.submit("room.field.a", READING, dict(READING, warmth=0.2), 2_000.0)
        assert _await(lambda: relay.sent == 2)
        assert len(_RelayDouble.edges) == 2
        assert _RelayDouble.edges[0]["chain"] is None            # genesis
        # edge 2 carried the head the relay handed back for edge 1
        assert _RelayDouble.edges[1]["chain"] == "0" * 63 + "1"
        assert relay.last_error is None
        relay.stop()
    finally:
        srv.shutdown()


def test_submit_heals_a_lost_response_via_expected_head():
    _RelayDouble.edges.clear()
    _RelayDouble.mode = "ok"
    srv = _serve()
    try:
        relay = CrabTrapRelay(f"http://127.0.0.1:{srv.server_address[1]}").start()
        assert relay.submit("room.field.a", None, READING, 1_000.0)
        assert _await(lambda: relay.sent == 1)
        # Simulate a lost 201: our head is stale, the next POST 409s once.
        _RelayDouble.mode = "conflict_once"
        assert relay.submit("room.field.a", READING, dict(READING, warmth=0.3), 3_000.0)
        assert _await(lambda: relay.sent == 2)                    # healed on retry
        adopted = _RelayDouble.edges[-1]["chain"] == "ef" * 32
        assert adopted
        assert relay.last_error is None
        relay.stop()
    finally:
        srv.shutdown()


def test_permanent_conflict_records_error_without_wedging():
    _RelayDouble.edges.clear()
    _RelayDouble.mode = "conflict"
    srv = _serve()
    try:
        relay = CrabTrapRelay(f"http://127.0.0.1:{srv.server_address[1]}").start()
        assert relay.submit("room.field.a", None, READING, 1_000.0)
        assert _await(lambda: len(_RelayDouble.edges) == 2)       # attempt + one retry
        assert relay.sent == 0
        assert relay.last_error is not None
        assert relay._heads["room.field.a"] == "ff" * 32          # adopted, ready to heal
        relay.stop()
    finally:
        srv.shutdown()


def test_unreachable_relay_never_blocks_or_raises():
    # port 1 on localhost: refuses near-instantly; nothing may propagate
    relay = CrabTrapRelay("http://127.0.0.1:1", timeout=0.2).start()
    t0 = time.monotonic()
    assert relay.submit("room.field.a", None, READING, 1.0) is True
    assert time.monotonic() - t0 < 0.05              # fire-and-forget
    assert _await(lambda: relay.last_error is not None)
    relay.stop()


def test_status_reports_limb_state():
    relay = CrabTrapRelay("http://127.0.0.1:8787")
    s = relay.status()
    assert s["relay"] == "http://127.0.0.1:8787"
    assert s["sent"] == 0 and s["dropped"] == 0


def test_roomd_pushes_every_field_read_to_the_relay():
    """roomd GET /field is the producer: every read sealed then submitted."""
    from elephant.roomd import RoomDaemon

    class StubRelay:
        def __init__(self):
            self.submitted = []

        def submit(self, cell, before, after, ts):
            self.submitted.append((cell, before, after))
            return True

        def status(self):
            return {"relay": "stub", "sent": len(self.submitted)}

    stub = StubRelay()
    d = RoomDaemon(relay=stub)
    for i in range(3):
        d.ingest({"room": "sauna", "author": "a", "text": "warm toast", "ts": i})
        d.room_field("sauna")
    cells = [c for c, _, _ in stub.submitted]
    assert cells == ["room.field.sauna"] * 3
    # before/after chain across reads: read N's after is read N+1's before
    first, second, third = stub.submitted
    assert first[1] is None                       # genesis read
    assert second[1] == first[2] and third[1] == second[2]
    # the wire payload is the pure reading — no local-ledger hash inside
    assert "ledger" not in second[2] and "warmth" in second[2]
