"""The elephant's limb — sealed readings pushed to the crab-traps relay.

The consumer half of the cell-ledger seam (synergy missing-link ②). The
producer (:mod:`elephant.cell_ledger`) seals every field read into the
local chain; this module forwards each sealed read to the crab-traps
Worker's edge-ledger relay (``POST /edge``), where it lands in the shared
D1 ledger the dial-dashboard reads.

Wire contract (crab-traps worker/src/edge-ledger.ts, cell-ledger.md §4):

    POST /edge { v:1, cell, ts, before, after, delta, imbalance,
                 provenance, chain }

The relay is the *sealing authority* for the remote chain: every 201
response returns the new ``chain_head``, and the next edge for that cell
must carry it in ``chain``. The limb never blocks — submissions are
enqueued and a daemon thread does the talking; a relay that is down or
slow can never stall ``room_field()``.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

RELAY_TIMEOUT_S = 5.0
MAX_QUEUE = 1024          # bounded limb memory — a long outage drops oldest
ELEPHANT_CALLER = "elephant-roomd"


def build_edge(cell: str, before: Optional[Dict], after: Dict, ts: float,
               chain: Optional[str]) -> Dict[str, Any]:
    """Shape one reading-pair into the relay's edge payload.

    Persistence prior (cell-ledger.md §3): the forecast is the prior
    reading, so surprise is the edge — ``imbalance = |Δwarmth|``. The
    genesis read has no prior, so it is unscored (``null``): never fake
    a number.
    """
    w0 = before.get("warmth") if isinstance(before, dict) else None
    w1 = after.get("warmth")
    scored = w0 is not None and w1 is not None
    magnitude = round(abs(w1 - w0), 6) if scored else None
    return {
        "v": 1,
        "cell": cell,
        "ts": int(ts * 1000),
        "before": before,
        "after": after,
        "delta": {
            "before": w0,
            "after": w1,
            "changed": bool(magnitude),
            "magnitude": magnitude,
        },
        "imbalance": magnitude,
        "provenance": {
            "origin": "push",
            "caller": ELEPHANT_CALLER,
            "trace": ["room_field"],
        },
        "chain": chain,
    }


class CrabTrapRelay:
    """Forwards sealed field reads to the crab-traps edge-ledger relay.

    Usage::

        relay = CrabTrapRelay("http://127.0.0.1:8787").start()
        relay.submit("room.field.living-room", before, after, ts)
        ...
        relay.stop()

    ``submit`` is fire-and-forget (bounded queue, daemon thread) — the
    limb never blocks. Chain recovery: on a 409 the relay's
    ``expected_head`` is adopted and the edge retried once, so a lost
    response (or a restarted ledger) heals instead of wedging.
    """

    def __init__(self, base_url: str, timeout: float = RELAY_TIMEOUT_S) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._heads: Dict[str, str] = {}          # cell -> relay chain_head
        self._queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=MAX_QUEUE)
        self._thread: Optional[threading.Thread] = None
        self.sent = 0
        self.dropped = 0
        self.last_error: Optional[str] = None

    # -- lifecycle ------------------------------------------------------ #
    def start(self) -> "CrabTrapRelay":
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name="crab-trap-relay")
            self._thread.start()
        return self

    def stop(self) -> None:
        try:
            self._queue.put_nowait(None)          # sentinel: drain then exit
        except queue.Full:
            pass

    # -- limb ------------------------------------------------------------ #
    def submit(self, cell: str, before: Optional[Dict], after: Dict,
               ts: float) -> bool:
        """Enqueue one sealed read; returns False (and drops) if the
        limb's queue is full — a stalled relay must never stall the room.

        The chain link is resolved at POST time by the serial talker, not
        here — two reads enqueued back-to-back must not both carry the
        same (stale) head.
        """
        item = {"cell": cell, "before": before, "after": after, "ts": ts}
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            self.dropped += 1
            self.last_error = "relay queue full — reading dropped (local seal intact)"
            return False

    # -- the talker ------------------------------------------------------#
    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            self._post(item, retry=True)

    def _post(self, item: Dict[str, Any], retry: bool) -> None:
        cell = item["cell"]
        edge = build_edge(cell, item["before"], item["after"], item["ts"],
                          self._heads.get(cell))
        req = urllib.request.Request(
            f"{self.base_url}/edge",
            data=json.dumps(edge).encode(),
            headers={
                "Content-Type": "application/json",
                # urllib's default UA is edge-banned (CF error 1010) — the
                # limb identifies itself instead.
                "User-Agent": f"{ELEPHANT_CALLER}/0.1 (cell-ledger limb)",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode() or "{}")
                head = body.get("chain_head")
                if head:
                    self._heads[cell] = head
                self.sent += 1
                self.last_error = None
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read().decode() or "{}")
            except Exception:
                body = {}
            if e.code == 409 and retry and body.get("expected_head"):
                # The ledger moved without us (lost response, restarted
                # cell) — adopt the relay's head and append once.
                self._heads[cell] = body["expected_head"]
                return self._post(item, retry=False)
            if e.code == 409 and body.get("error") == "duplicate edge":
                self.last_error = None           # already there — fine
                return
            self.last_error = f"relay 409: {body.get('error') or e}"
            log.debug("relay rejected edge for %s: %s", cell, self.last_error)
        except Exception as e:                    # network down, timeout…
            self.last_error = f"relay unreachable: {e}"
            log.debug("relay unreachable: %s", e)

    # -- introspection ---------------------------------------------------#
    def status(self) -> Dict[str, Any]:
        return {
            "relay": self.base_url,
            "sent": self.sent,
            "queued": self._queue.qsize(),
            "dropped": self.dropped,
            "heads": dict(self._heads),
            "last_error": self.last_error,
        }
