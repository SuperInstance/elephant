#!/usr/bin/env python3
"""roomd — the field truth-holder.

The elephant's daemon (plan §1.1, fleet-next-level-plan.md). Loads an
eisenstein ``map.json`` export and an events JSONL file, runs the real
``DialBank(DEFAULT_DIALS)`` over each room, serves the field over HTTP,
and rings the deadband as a USCP-v1 STATUS_REPORT packet written
atomically into the CNS inbox — one packet per rising edge.

One truth-holder per quantity: the elephant owns the field. Everything
served here is the truth; terrain's POSTed deltas are its shadow.

Usage:
    python3 -m elephant.roomd --map map.json --events events.jsonl \
        [--port 4073] [--inbox ~/.hermes/cns_inbox] \
        [--relay http://127.0.0.1:8787]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

from .dial import DialBank
from .dials import DEFAULT_DIALS
from .cell_ledger import CellLedgerProducer
from .field import read_field
from .room import Message, Room
from .terrain import Deadband, Terrain
from .mud import tint_description

__all__ = ["RoomDaemon", "build_uscp_packet", "main"]

DEFAULT_PORT = 4073
WARMTH_RING_HIGH = 0.45   # |Δwarmth| that counts as a real swing
PANIC_RING_LEVEL = 0.55   # panic reading that counts as a stampede


def build_uscp_packet(ring_kind: str, severity: str, payload: Dict, now: Optional[float] = None) -> Dict:
    """A minimal USCP-v1 STATUS_REPORT packet (mirrors cns-echo's shape)."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now if now is not None else time.time()))
    return {
        "header": {
            "type": "USCP-v1",
            "origin_id": "elephant-roomd",
            "timestamp": ts,
            "priority": "HIGH" if severity == "high" else "MEDIUM",
            "intent": "STATUS_REPORT",
        },
        "body": {
            "subject": f"deadband ring: {ring_kind}",
            "source": "elephant-roomd",
            "content": json.dumps(payload),
            "sections": payload,
        },
    }


def write_atomic(path: Path, packet: Dict) -> None:
    """temp + rename — a packet is never half-visible (cns-echo pattern)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(packet, fh)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


class RoomDaemon:
    """Holds rooms, ingests events, serves the field, rings the deadband."""

    def __init__(self, map_path: Optional[str] = None, inbox: Optional[str] = None,
                 field_log: Optional[str] = None, relay: Any = None):
        self.bank = DialBank(DEFAULT_DIALS)
        self.rooms: Dict[str, Room] = {}
        self.terrains: Dict[str, Terrain] = {}
        self.deadbands: Dict[str, Deadband] = {}
        self.rings_fired: set = set()          # rising-edge memory
        self.ring_log: List[Dict] = []           # observability: every ring, bounded
        self.inbox = Path(inbox) if inbox else None
        self.descriptions: Dict[str, str] = {}   # base text per room (zeitgeist)
        self.field_log_path = Path(field_log) if field_log else None
        self.field_log_lines = 0                    # bounded rotation counter
        self.map_temperature: Optional[float] = None
        self._ledger = None                        # CellLedgerProducer (quilt seam); set via enable_ledger()
        self._relay = relay                        # CrabTrapRelay (crab-traps seam); the limb
        self._last_reading: Dict[str, Dict] = {}   # per-room prior — the relay's `before`
        if map_path:
            self.load_map(map_path)

    def enable_ledger(self, producer: Any) -> None:
        """Attach the quilt cell-ledger producer (synergy missing-link ②)."""
        self._ledger = producer

    # -- map ----------------------------------------------------------- #
    def load_map(self, map_path: str) -> None:
        data = json.loads(Path(map_path).read_text())
        rooms = data.get("rooms", data if isinstance(data, list) else [])
        for r in rooms:
            name = r.get("name") or r.get("id")
            if name:
                self.ensure_room(name, description=r.get("description", ""))

    def ensure_room(self, name: str, description: str = "") -> Room:
        if name not in self.rooms:
            self.rooms[name] = Room(name)
            self.terrains[name] = Terrain(space_id=name)
            self.deadbands[name] = Deadband()
            # zeitgeist: the room's own words, tinted by its field
            self.descriptions[name] = description or (
                "A low-ceilinged room, warm wood and a long counter.")
        return self.rooms[name]

    # -- ingest ---------------------------------------------------------#
    def ingest(self, event: Dict) -> None:
        room = self.ensure_room(str(event.get("room", "default")))
        room.messages.append(Message(
            author=str(event.get("author", "anon")),
            text=str(event.get("text", "")),
            ts=float(event.get("ts", time.time())),
        ))
        # keep rooms bounded — the elephant's windows are bounded by law
        if len(room.messages) > 512:
            del room.messages[: len(room.messages) - 512]

    def ingest_file(self, path: str) -> int:
        n = 0
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.ingest(json.loads(line))
                    n += 1
                except json.JSONDecodeError:
                    continue
        return n

    # -- field ----------------------------------------------------------#
    def room_field(self, name: str) -> Optional[Dict]:
        room = self.rooms.get(name)
        if room is None or not room.messages:
            return None
        field = read_field(room, self.bank)
        reading = {
            "room": name,
            "warmth": round(field.warmth(), 4),
            "kappa": round(field.concentration(), 4),
            "dials": {k: round(v, 4) for k, v in field.readings.items()},
            "messages": len(room.messages),
            "ts": time.time(),
        }
        # Cell-ledger bridge (quilt seam): seal every field read into the
        # shared chain so the grid can consume the elephant's readings.
        if self._ledger is not None:
            sealed = self._ledger.record({
                "cell_id": f"room.field.{name}",
                "kind": "field",
                "ts": time.time(),
                "reading": reading,
            })
            reading["ledger"] = {"seq": sealed["seq"], "hash": sealed["hash"],
                                 "prev_hash": sealed["prev_hash"]}
        # Crab-traps seam: push the sealed pair to the shared D1 ledger
        # (fire-and-forget — the limb never blocks the truth-holder).
        if self._relay is not None:
            wire = {k: v for k, v in reading.items() if k != "ledger"}
            self._relay.submit(f"room.field.{name}", self._last_reading.get(name),
                               wire, ts=reading.get("ts", time.time()))
            self._last_reading[name] = wire
        return reading

    def recompute(self) -> Dict:
        """Recompute every room + the map temperature; ring on crossings."""
        fields = {}
        for name, room in self.rooms.items():
            if room.messages:
                f = read_field(room, self.bank)
                fields[name] = f
        if fields:
            warmth = sum(f.warmth() for f in fields.values()) / len(fields)
            self.map_temperature = round(warmth, 4)
        payload = {
            "warmth": self.map_temperature,
            "kappa": None,
            "dials": {},
            "rooms": {n: self.room_field(n) for n in fields},
            "map_temperature": self.map_temperature,
            "ts": time.time(),
        }
        self._log_field(payload)
        self.check_rings(fields)
        return payload

    # -- field log (the training corpus, plan §3.1) ---------------------#
    MAX_FIELD_LOG_LINES = 10000

    def _log_field(self, payload: Dict) -> None:
        """Append one snapshot per recompute — the v3 contrast corpus.

        Bounded: at MAX lines, rotate (keep newest half). O(1) amortized,
        one jsonl line per recompute.
        """
        if not self.field_log_path:
            return
        try:
            self.field_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.field_log_path, "a") as fh:
                fh.write(json.dumps({"ts": payload["ts"],
                                     "map_temperature": payload["map_temperature"],
                                     "rooms": payload["rooms"]}) + "\n")
            self.field_log_lines += 1
            if self.field_log_lines >= self.MAX_FIELD_LOG_LINES:
                self._rotate_field_log()
        except OSError:
            pass  # a full disk must never kill the truth-holder

    def _rotate_field_log(self) -> None:
        try:
            lines = self.field_log_path.read_text().splitlines()
            keep = lines[len(lines) // 2:]
            tmp = self.field_log_path.with_suffix(".tmp")
            tmp.write_text("\n".join(keep) + "\n")
            tmp.replace(self.field_log_path)
            self.field_log_lines = len(keep)
        except OSError:
            pass

    # -- zeitgeist -------------------------------------------------------#
    def tinted_description(self, name: str) -> Optional[str]:
        """The room's own words, tinted by its live field (plan §3.7).

        The zeitgeist authors the room text: the description IS the room's
        body language. Deterministic per field (tint_description seeds
        from it), so the same field always speaks the same words.
        """
        room = self.rooms.get(name)
        base = self.descriptions.get(name)
        if room is None or base is None or not room.messages:
            return base
        field = read_field(room, self.bank)
        hour = time.localtime().tm_hour + time.localtime().tm_min / 60.0
        return tint_description(field, base, hour=hour)

    # -- deadband -------------------------------------------------------#
    def check_rings(self, fields) -> None:
        for name, f in fields.items():
            panic = f.readings.get("panic", 0.0)
            warmth = f.warmth()
            self._ring(name, "panic", panic >= PANIC_RING_LEVEL, "high",
                       {"room": name, "panic": round(panic, 4)})
            self._ring(name, "warmth_swing", abs(warmth) >= WARMTH_RING_HIGH, "medium",
                       {"room": name, "warmth": round(warmth, 4)})

    def _ring(self, key: str, kind: str, condition: bool, severity: str, extra: Dict) -> None:
        edge = f"{key}:{kind}"
        if condition and edge not in self.rings_fired:      # rising edge only
            self.rings_fired.add(edge)
            self.ring_log.append({"ts": time.time(), "room": key,
                                  "kind": kind, "severity": severity, **extra})
            if len(self.ring_log) > 256:
                del self.ring_log[: len(self.ring_log) - 256]
            if self.inbox:
                packet = build_uscp_packet(kind, severity, extra)
                fname = f"{int(time.time() * 1000)}-elephant-roomd-{kind}.json"
                write_atomic(self.inbox / fname, packet)
        elif not condition and edge in self.rings_fired:    # re-arm on fall
            self.rings_fired.discard(edge)


class _Handler(BaseHTTPRequestHandler):
    daemon_ref: RoomDaemon = None  # injected by serve()

    def log_message(self, *args):  # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        from urllib.parse import unquote
        d = self.daemon_ref
        if self.path in ("/field", "/"):
            return self._json(d.recompute())
        if self.path.startswith("/rooms/") and self.path.endswith("/field"):
            name = unquote(self.path[len("/rooms/"):-len("/field")])
            f = d.room_field(name)
            return self._json(f if f is not None else {"error": "unknown or empty room"}, 200 if f else 404)
        if self.path.startswith("/rooms/") and self.path.endswith("/description"):
            name = unquote(self.path[len("/rooms/"):-len("/description")])
            if name in d.rooms:
                return self._json({"room": name,
                                   "description": d.tinted_description(name)})
            return self._json({"error": "unknown room"}, 404)
        if self.path == "/health":
            return self._json({"ok": True, "rooms": list(d.rooms)})
        if self.path == "/relay":
            return self._json(d._relay.status() if d._relay is not None
                              else {"relay": None, "note": "roomd started without --relay"})
        if self.path == "/rings":
            return self._json({"rings": d.ring_log[-20:]})
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        d = self.daemon_ref
        if self.path == "/ingest":
            length = int(self.headers.get("Content-Length", 0))
            try:
                event = json.loads(self.rfile.read(length))
                d.ingest(event)
                return self._json({"ok": True, "rooms": len(d.rooms)})
            except Exception as e:
                return self._json({"error": str(e)}, 400)
        return self._json({"error": "not found"}, 404)


def serve(daemon: RoomDaemon, port: int = DEFAULT_PORT) -> None:
    _Handler.daemon_ref = daemon
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    print(f"roomd: serving /field on 127.0.0.1:{httpd.server_address[1]} "
          f"({len(daemon.rooms)} rooms, inbox={daemon.inbox})", flush=True)
    httpd.serve_forever()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="roomd", description="the field truth-holder")
    ap.add_argument("--map", help="eisenstein map.json export (rooms list)")
    ap.add_argument("--events", help="events.jsonl to ingest at startup")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--inbox", default=os.path.expanduser("~/.hermes/cns_inbox"),
                    help="USCP inbox for deadband rings (empty string disables)")
    ap.add_argument("--field-log", default=None,
                    help="append field snapshots to this jsonl (the v3 training corpus)")
    ap.add_argument("--relay", default="",
                    help="crab-traps edge-ledger base URL (e.g. http://127.0.0.1:8787) — "
                         "pushes every sealed field read to the shared D1 ledger "
                         "(empty string disables; GET /relay shows limb status)")
    args = ap.parse_args(argv)

    relay = None
    if args.relay:
        from .relay import CrabTrapRelay
        relay = CrabTrapRelay(args.relay).start()
    daemon = RoomDaemon(map_path=args.map,
                        inbox=args.inbox or None,
                        field_log=args.field_log or None,
                        relay=relay)
    if args.relay:
        daemon.enable_ledger(CellLedgerProducer("roomd.field"))
    if args.events:
        n = daemon.ingest_file(args.events)
        print(f"roomd: ingested {n} events", flush=True)
    if not args.map and not args.events:
        daemon.ensure_room("default")
    serve(daemon, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
