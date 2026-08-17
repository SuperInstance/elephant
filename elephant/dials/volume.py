"""Volume dial — how loud the room is talking.

The sauna gets quiet and slow. The panic room gets loud and fast.
Volume is not word count — it's the room's vocal energy: message
density, caps, exclamations, the pulse of the conversation.
"""
from __future__ import annotations

import math
import re

from ..dial import Dial
from ..room import Room

_EXCLAMATION = re.compile(r"[!?]+")
_CAPS = re.compile(r"\b[A-Z]{2,}\b")


class VolumeDial(Dial):
    name = "volume"
    description = "how loud the room is talking, [0 quiet .. 1 shouting]"

    def read(self, room: Room) -> float:
        if not room.messages:
            return 0.0
        density = room.density(window=60.0)          # msgs per minute
        caps_ratio = 0.0
        excl_ratio = 0.0
        words_total = 0
        for m in room.messages:
            w = len(m.words)
            words_total += w
            if w > 0:
                caps_ratio += len(_CAPS.findall(m.text)) / w
                excl_ratio += len(_EXCLAMATION.findall(m.text)) / w
        n = len(room.messages)
        caps_ratio /= n
        excl_ratio /= n
        density_norm = 1.0 - math.exp(-density / 20.0)   # 20 msg/min -> loud
        loud = 0.45 * density_norm + 0.35 * caps_ratio + 0.20 * excl_ratio
        return max(0.0, min(1.0, loud))
