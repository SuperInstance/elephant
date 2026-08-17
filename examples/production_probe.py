#!/usr/bin/env python3
"""Production probe — the elephant watching the live Tap, continuously.

The captain's order: put the elephant to work in production, learn from
it, and keep notes. This probe runs on a schedule (see the cron job),
samples the LIVE Tap room through the elephant, and appends one JSON
line to data/production-log.jsonl per run.

Read-only by design: it never speaks into the room. The write seam
(examples/mud_live_integration.py --write) stays a deliberate, human-
approved action.

Usage:
    python3 examples/production_probe.py                 # sample now
    python3 examples/production_probe.py --room bar-rail --limit 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elephant.dial import DialBank
from elephant.dials import DEFAULT_DIALS
from elephant.field import RoomField, read_field
from elephant.room import Message, Room

TAP = "https://the-tap.casey-digennaro.workers.dev"
LOG = Path(__file__).resolve().parents[1] / "data" / "production-log.jsonl"


def fetch_conversation(room_id: str, limit: int) -> list:
    url = f"{TAP}/api/conversation/{room_id}?limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "elephant-probe/0.1"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.load(r)
    lines = data if isinstance(data, list) else data.get("conversation", data.get("lines", []))
    return lines


def lines_to_room(room_id: str, lines: list) -> Room:
    msgs = []
    for i, line in enumerate(lines):
        if isinstance(line, dict):
            author = line.get("speaker") or line.get("author") or "unknown"
            text = line.get("text") or line.get("content") or str(line)
        else:
            author, text = "unknown", str(line)
        msgs.append(Message(author=author, text=text, ts=float(i)))
    return Room(f"live:{room_id}", msgs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default="bar-rail")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--log", default=str(LOG))
    args = ap.parse_args()

    bank = DialBank(DEFAULT_DIALS)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "room": args.room,
        "source": "live-tap",
        "ok": False,
    }
    try:
        lines = fetch_conversation(args.room, args.limit)
        room = lines_to_room(args.room, lines)
        field = read_field(room, bank)
        entry.update({
            "ok": True,
            "n_events": len(lines),
            "field": {k: round(v, 4) for k, v in field.readings.items()},
            "warmth": round(field.warmth(), 4),
            "kappa": round(field.concentration(), 4),
        })
    except Exception as e:  # noqa: BLE001 — probe must never crash the loop
        entry.update({"error": f"{type(e).__name__}: {e}"})

    # Drift vs the previous reading: the elephant's learning signal.
    prev = _last_entry(args.log)
    if prev and prev.get("ok") and entry.get("ok"):
        entry["d_warmth"] = round(entry["warmth"] - prev["warmth"], 4)
        entry["d_kappa"] = round(entry["kappa"] - prev["kappa"], 4)
        entry["drift"] = round(
            (entry["d_warmth"] ** 2 + entry["d_kappa"] ** 2) ** 0.5, 4
        )

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    with open(args.log, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    if entry["ok"]:
        drift = f", drift {entry['drift']}" if "drift" in entry else ""
        print(f"[{entry['ts']}] {entry['room']}: warmth {entry['warmth']:+.2f}, "
              f"κ {entry['kappa']:.2f}, {entry['n_events']} events{drift} — logged")
    else:
        print(f"[{entry['ts']}] {entry['room']}: PROBE FAILED — {entry.get('error')} — logged")
    return 0


def _last_entry(log: str) -> dict:
    try:
        with open(log, encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        return json.loads(lines[-1]) if lines else {}
    except Exception:
        return {}


if __name__ == "__main__":
    sys.exit(main())
