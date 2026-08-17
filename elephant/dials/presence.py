"""Presence dial — the pheromone trace.

"Like pheromones a little — but that's just one dimension for one of
many JEPAs affecting agents constantly without words." Presence is the
scent of the room: who has been here, how recently, how long they
lingered. Warm rooms hold scent; empty rooms lose it. This dial reads
the occupancy trail — the memory of bodies in the room.
"""
from __future__ import annotations

import math

from ..dial import Dial
from ..room import Room


class PresenceDial(Dial):
    name = "presence"
    description = "pheromone trace of the room, [0 empty .. 1 thrumming]"

    def read(self, room: Room) -> float:
        if not room.messages:
            return 0.0
        authors: dict = {}
        t0 = room.messages[0].ts
        t1 = room.messages[-1].ts
        span = max(t1 - t0, 1e-9)
        for m in room.messages:
            a = m.author
            entry = authors.setdefault(a, {"first": m.ts, "last": m.ts, "n": 0})
            entry["first"] = min(entry["first"], m.ts)
            entry["last"] = max(entry["last"], m.ts)
            entry["n"] += 1
        distinct = len(authors)
        # Recency: how fresh is the most recent scent?
        recency = 1.0 - math.exp(-(t1 - t0) / max(span, 1e-9))
        # Longevity: average fraction of the room's life each author was present.
        longevity = 0.0
        for a, e in authors.items():
            life = (e["last"] - e["first"]) / span
            longevity += min(1.0, life * 2.0)
        longevity /= max(distinct, 1)
        activity = min(1.0, len(room.messages) / 40.0)
        presence = 0.45 * distinct / 5.0 + 0.25 * recency + 0.20 * longevity + 0.10 * activity
        return max(0.0, min(1.0, presence))
