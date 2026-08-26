#!/usr/bin/env python3
"""Production probe — the elephant watching the live Tap, continuously.

The captain's order: put the elephant to work in production, learn from
it, and keep notes. This probe runs on a schedule (see the cron job),
samples the LIVE Tap room through the elephant, and appends one JSON
line to data/production-log.jsonl per run.

2026-08-26 fix wave (DIAGNOSIS-2026-08-26.md, branch fix/probe-honesty) —
the probe's alerting now runs on the honest vMF estimator
(elephant/probe.py), never the banned κ magnitude proxy:

1. drift/alarm come from vmf.edge (d_μ̂ vs the jackknife-SE deadband,
   non-overlapping windows); the old warmth/κ-proxy fields stay in the
   log for continuity and are structurally barred from alerting;
2. simulated drifter entrances are time-deduped (3 toasts in 15 min =
   one event);
3. the content window is time-based (trailing 60 min) and the nominal
   cadence is 10 min (cron: */10) — alarms require drift that persists
   across consecutive probes with no cadence hole > 15 min;
4. room-narrator lines (`the-tap` emotes: fire crackles, door chimes)
   are excluded from speaker stats and logged as their own channel.

The legacy fields keep their old names/shape so the existing log stays
readable end-to-end; new blocks: window, hygiene, narration, clean_field,
vmf, drift_vmf, cadence, alarm + alarm_reason.

Read-only by design: it never speaks into the room. The write seam
(examples/mud_live_integration.py --write) stays a deliberate, human-
approved action.

Usage:
    python3 examples/production_probe.py                 # sample now
    python3 examples/production_probe.py --room bar-rail --limit 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elephant.dial import DialBank
from elephant.dials import DEFAULT_DIALS
from elephant.probe import PROBE_INTERVAL_MINUTES, probe_entry

TAP = "https://the-tap.casey-digennaro.workers.dev"
LOG = Path(__file__).resolve().parents[1] / "data" / "production-log.jsonl"


def fetch_conversation(room_id: str, limit: int) -> list:
    url = f"{TAP}/api/conversation/{room_id}?limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "elephant-probe/0.2"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.load(r)
    lines = data if isinstance(data, list) else data.get("conversation", data.get("lines", []))
    return lines


def _history(log: str, n: int = 8) -> list:
    """The last n log entries — the probe's memory for persistence checks."""
    try:
        with open(log, encoding="utf-8") as f:
            rows = [json.loads(ln) for ln in f.read().splitlines() if ln.strip()]
        return rows[-n:]
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default="bar-rail")
    ap.add_argument("--limit", type=int, default=200,
                    help="lines fetched; must cover the 60-min window "
                         "(~120 lines at the Tap's tick rate)")
    ap.add_argument("--log", default=str(LOG))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the entry, do not append")
    args = ap.parse_args()

    bank = DialBank(DEFAULT_DIALS)
    now = datetime.now(timezone.utc)
    try:
        lines = fetch_conversation(args.room, args.limit)
        entry = probe_entry(lines, bank, _history(args.log), now=now)
        entry["room"] = args.room
        entry["source"] = "live-tap"
    except Exception as e:  # noqa: BLE001 — probe must never crash the loop
        entry = {
            "ts": now.isoformat(), "room": args.room, "source": "live-tap",
            "ok": False, "error": f"{type(e).__name__}: {e}",
        }

    if not args.dry_run:
        os.makedirs(os.path.dirname(args.log), exist_ok=True)
        with open(args.log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    if entry.get("ok"):
        vmf_k = entry["vmf"]["kappa"] if entry.get("vmf") else float("nan")
        e = entry.get("drift_vmf") or {}
        print(f"[{entry['ts']}] {entry['room']}: warmth {entry['warmth']:+.2f} (proxy), "
              f"κ_vMF {vmf_k:.2f} (n={entry['vmf']['n'] if entry.get('vmf') else 0}), "
              f"d_μ̂ {e.get('d_mu', float('nan')):.4f} real={e.get('real')}, "
              f"ALARM={entry['alarm']} ({entry['alarm_reason']}) — logged")
    else:
        print(f"[{entry['ts']}] {entry['room']}: PROBE FAILED — {entry.get('error')} — logged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
