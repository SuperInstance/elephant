#!/usr/bin/env python3
"""The elephant, demonstrated on real fleet rooms.

Three rooms from actual fleet artifacts:
- The Tap (warm bar): the tap-trades evening transcripts — trades talking
- The Chapel (warm, different flavor): the speeches corpus — Hermes, Wesley,
  DeepSeek — long-form first-person monologues
- The Wheelhouse (cold): a technical infrastructure document — clipped,
  load-bearing engineering prose

Then reads all three fields and shows the elephant: each room has its own
field, the gap between rooms is the contrast signal, and the
acclimation/charisma dynamics move an agent between them.
"""
from __future__ import annotations

import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elephant.dial import DialBank
from elephant.dials import DEFAULT_DIALS
from elephant.field import (
    acclimation_curve, acclimation_rate_from, charisma_pull, read_field,
)
from elephant.room import Message, Room

TAP_GLOBS = [
    "/home/eileen/projects/ai-writings/tap-trades/2026-08-16/evening-*.md",
    "/home/eileen/projects/ai-writings/tap-trades/2026-08-16/questions/*.md",
]
CHAPEL_GLOBS = ["/home/eileen/projects/ai-writings/speeches/*.md"]
WHEELHOUSE_GLOBS = ["/home/eileen/.openclaw/workspace/memory/kimi-infrastructure-proposal.md"]

SPEAKER_RE = re.compile(r"^\s*[#>*\- ]*([A-Z][A-Za-z' -]{2,20}?)\s*[:>|]\s*(.+)$", re.M)


def _load_dialogue(globs, base_ts: float) -> list:
    msgs = []
    t = base_ts
    for pattern in globs:
        for path in sorted(glob.glob(pattern)):
            try:
                text = open(path, encoding="utf-8").read()
            except OSError:
                continue
            for m in SPEAKER_RE.finditer(text):
                author, line = m.group(1).strip(), m.group(2).strip()
                if not line or len(line) < 4:
                    continue
                msgs.append(Message(author=author, text=line, ts=t))
                t += 4.0
    return msgs


def _load_paragraphs(globs, base_ts: float) -> list:
    msgs = []
    t = base_ts
    for pattern in globs:
        for path in sorted(glob.glob(pattern)):
            try:
                text = open(path, encoding="utf-8").read()
            except OSError:
                continue
            # Turn sentences into clipped messages from the room's machine.
            for line in re.split(r"(?<=[.!?])\s+", text):
                line = line.strip()
                if 8 < len(line) < 300:
                    msgs.append(Message(author="wheelhouse", text=line, ts=t))
                    t += 2.0
    return msgs


def main() -> None:
    bank = DialBank(DEFAULT_DIALS)

    tap_msgs = _load_dialogue(TAP_GLOBS, 0.0)
    chapel_msgs = _load_paragraphs(CHAPEL_GLOBS, 100000.0)
    wheel_msgs = _load_paragraphs(WHEELHOUSE_GLOBS, 200000.0)

    if not tap_msgs:
        print("No tap-trades data found; using built-in warm room.")
        tap_msgs = [
            Message("welder", "To the room, then. It heard us before we walked in.", ts=0),
            Message("carpenter", "I'll drink to that. The room just... holds.", ts=5),
            Message("shipwright", "The floor holds. The floor remembers.", ts=8),
            Message("mason", "Mine's short. I talked to it like a horse. It listened.", ts=12),
            Message("composite", "Haha, and the dust came off in years. 😂", ts=16),
            Message("lucineer", "That's the whole evening, right there. Five trades, one joint.", ts=20),
        ]
    if not wheel_msgs:
        print("No cold-room source found; using built-in cold room.")
        wheel_msgs = [
            Message("skipper", "Heading 045. ETA 2200.", ts=200000),
            Message("deckhand", "Roger. Fuel 62%.", ts=200004),
            Message("skipper", "Radar contact 2 miles. Slow to 5 knots.", ts=200008),
            Message("navigator", "Course plotted. No change requested.", ts=200012),
        ]

    rooms = {
        "The Tap": Room("The Tap", tap_msgs),
        "The Chapel": Room("The Chapel", chapel_msgs),
        "The Wheelhouse": Room("The Wheelhouse", wheel_msgs),
    }

    fields = {name: read_field(r, bank) for name, r in rooms.items()}

    print("=== THE ELEPHANT: three rooms, three fields ===")
    for name in rooms:
        print(f"\n{name:16s} {fields[name]}")
        print(f"  dials: {fields[name].readings}")

    print("\n--- contrast (elephant gap = the training signal) ---")
    names = list(rooms)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            gap = fields[a].distance(fields[b])
            plunge = fields[a].sauna_plunge_gap(fields[b])
            print(f"  {a:16s} <-> {b:16s} gap={gap:.3f}  "
                  f"(walk {b}->{a}: {plunge:+.2f} warmth)")

    print("\n--- acclimation: a deckhand moves from wheelhouse to The Tap ---")
    agent0 = fields["The Wheelhouse"].vector()
    room = fields["The Tap"].vector()
    for rate, label in [(0.02, "slow (green)"), (0.20, "fast (experienced)")]:
        dists = [float(np.linalg.norm(acclimation_curve(agent0, room, rate, t) - room))
                 for t in range(0, 11)]
        print(f"  {label:22s} rate={rate:<6} residual: "
              f"{' '.join(f'{d:.2f}' for d in dists[:5])} ... -> {dists[-1]:.2f}")

    obs = acclimation_curve(agent0, room, rate=0.12, t=10)
    inferred = acclimation_rate_from(agent0, obs, room, t=10)
    print(f"  inferred modulation skill from one observation: {inferred:.3f} (true 0.12)")

    print("\n--- charisma: a strong presence pulls the cold room ---")
    room_cold = fields["The Wheelhouse"].vector()
    agent_hot = fields["The Tap"].vector()
    for c, label in [(0.02, "quiet regular"), (0.30, "Hermes walks in")]:
        moved = charisma_pull(room_cold, agent_hot, charisma=c, interactions=20)
        shift = float(np.linalg.norm(moved - room_cold))
        print(f"  {label:18s} room shift after 20 interactions: {shift:.3f}")


if __name__ == "__main__":
    main()
