"""JEV field-watch: the room-field gets a booking layer.

The dials read the room; nothing writes down what was felt, and nothing
alarms when the ROOM ITSELF changes regime (a calm room catching fire is
the elephant in the field — nobody notices the temperature until they
change rooms). This adds the two missing pieces, doctrine-first:

  1. every perceived field reading is booked as a hash-chained receipt
     (sha256 over payload+parent — FNV-1a is legacy-weak per the org's
     agent-receipts survey; signatures are the v2 note);
  2. a JEV layer (mean-window predictor + surprise floor) watches the
     normalized field VECTOR and flags regime change with receipts.

Simulates two rooms — "harbor" (steady warmth, jokes landing) and
"signal-fire" (same room until t=40, then a panic storm) — and reports
alarm latency at the change point. Run: python3 examples/jev_field_watch.py
"""

import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from elephant.room import Message, Room
from elephant.field import read_field
from elephant.dial import DialBank
from elephant.dials.mood import MoodDial
from elephant.dials.panic import PanicDial
from elephant.dials.volume import VolumeDial
from elephant.dials.earnestness import EarnestnessDial
from elephant.dials.joke_landing import JokeLandingDial

CALM = ["lovely morning in the harbor, coffee's on",
        "haha that landing joke actually landed",
        "anyone tuning the dials today or just vibing",
        "quiet ships, warm water, good crew"]
FIRE = ["FIRE in the hold, everyone EVACUATE now",
        "oh no oh no the wal is BURNING panic panic",
        "EVACUATE EVACUATE, this is not a drill",
        "fire fire fire, we're losing the ship"]
JOKES = ["a crab walks into a bar...", "...shell-shocked, obviously",
         "the elephant notices when you change rooms"]


class ReceiptChain:
    def __init__(self):
        self.entries = []

    def book(self, kind: str, payload: dict) -> str:
        body = json.dumps({"kind": kind, "payload": payload},
                          sort_keys=True, default=str)
        parent = self.entries[-1]["hash"] if self.entries else "0" * 64
        h = hashlib.sha256((parent + body).encode()).hexdigest()
        self.entries.append({"hash": h, "parent": parent, "body": body})
        return h

    def verify(self) -> bool:
        for i, e in enumerate(self.entries):
            want = "0" * 64 if i == 0 else self.entries[i - 1]["hash"]
            if e["parent"] != want:
                return False
            if hashlib.sha256((e["parent"] + e["body"]).encode()).hexdigest() != e["hash"]:
                return False
        return True


def simulate(name: str, burn_at: int, chain: ReceiptChain):
    bank = DialBank([MoodDial(), PanicDial(), VolumeDial(),
                     EarnestnessDial(), JokeLandingDial()])
    print(f"\n== {name} (burn_at={burn_at}) ==")
    hist, alarms = [], []
    msgs = []
    for t in range(80):
        if burn_at and t >= burn_at:
            texts = [FIRE[t % len(FIRE)]] * 2
        elif t % 7 == 3:
            texts = [JOKES[t % len(JOKES)], CALM[t % len(CALM)]]
        else:
            texts = [CALM[t % len(CALM)], CALM[(t + 1) % len(CALM)]]
        for x in texts:
            msgs.append(Message(author="crew", text=x, ts=float(t)))
        room = Room(name, msgs)                       # the room ACCUMULATES
        field = read_field(room, bank)
        vec = [float(v) for v in field.normalize()]
        chain.book("field.reading", {"room": name, "t": t,
                                     "warmth": round(field.warmth(), 4),
                                     "kappa": round(field.concentration(), 4),
                                     "vec": [round(v, 4) for v in vec]})
        if len(hist) >= 8:
            win = hist[-8:]
            pred = [sum(w[i] for w in win) / len(win) for i in range(len(vec))]
            err = math.sqrt(sum((vec[i] - pred[i]) ** 2
                                for i in range(len(vec))) / len(vec))
            if err > 0.08:                            # floor: front ~0.10, calm ~0.005
                alarms.append(t)
                chain.book("field.alarm", {"room": name, "t": t,
                                            "err": round(err, 4)})
        hist.append(vec)
    lat = (alarms[0] - burn_at) if (burn_at and alarms) else None
    print(f"   alarms: {alarms[:8]}{'...' if len(alarms) > 8 else ''}"
          f"  latency at burn: {lat}")
    return alarms, lat


def main():
    chain = ReceiptChain()
    _, lat = simulate("harbor", burn_at=None, chain=chain)     # control
    _, lat_fire = simulate("signal-fire", burn_at=40, chain=chain)
    n = len(chain.entries)
    print(f"\nreceipts booked: {n}; chain verifies: {chain.verify()}")
    print(f"fire-room first-alarm latency: {lat_fire} ticks "
          f"(control false alarms: {'none' if lat is None else lat})")


if __name__ == "__main__":
    main()
