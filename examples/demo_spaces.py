"""One sense, many rooms — the SAME elephant reading three spaces.

A warm bar, a heated chat thread, and a quiet sensor deck. The elephant
doesn't care if the room is made of oak, pixels, or telemetry — it only
cares how warm the room is, and how the room's light changes everyone in
it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elephant.dial import DialBank
from elephant.dials import DEFAULT_DIALS
from elephant.space import ChatSpace, MudSpace, SensorSpace


def main():
    bank = DialBank(DEFAULT_DIALS)

    # 1. A warm bar.
    mud = MudSpace("The Tap")
    mud.chatter("marlo", "I love this place — it's warm and the company's good.", ts=0)
    mud.chatter("pincher", "Haha, why did the elephant cross the road? lol", ts=2)
    mud.chatter("marlo", "haha 😂😂", ts=3)
    mud.chatter("pincher", "we built it together, honestly", ts=4)
    mud.event("The jukebox clicks over to something slow and easy.", ts=6)

    # 2. A heated chat thread.
    chat = ChatSpace("crew-thread")
    chat.post("comic", "Why did the elephant cross the road? lol", ts=0,
              reactions={"😂": 5, "🤣": 3})
    chat.post("skeptic", "boo. that was bad. who let him cook 😬", ts=2)
    chat.post("comic", "ok sure, tough crowd 🙄", ts=3, reactions={"🙄": 2})

    # 3. A quiet sensor deck.
    sensor = SensorSpace("F/V EILEEN")
    sensor.ingest_radar([(0, 0), (1, 1), (0.5, 0.5)], ts=0)
    sensor.ingest_radar([(0.2, 0.1), (0.9, 0.9), (0.4, 0.5)], ts=10)
    sensor.ingest_radar([(0.3, 0.2), (0.8, 0.8), (0.5, 0.6)], ts=20)
    sensor.ingest_sounder(0.72, ts=5)
    sensor.ingest_sounder(0.75, ts=15)

    spaces = [
        ("a warm bar", mud),
        ("a heated chat", chat),
        ("a quiet sensor deck", sensor),
    ]
    for label, space in spaces:
        field = space.read(bank)
        print("=" * 68)
        print(f"{space.kind.upper()}  —  {label}: {space.name}")
        print(f"  tint target : {space.tint_target()}")
        print(f"  field       : warmth {field.warmth():+.2f}   "
              f"κ {field.concentration():.2f}")
        for d in bank.names():
            print(f"      {d:13s} {field.readings[d]:+.3f}")
        print(f"  readout     : {space.send_back(field)}")
        print()

    # The sensor deck ALSO carries its fleet dials (radar/sounder/fishing)
    # — and its alert phrasing uses them.
    sensor_field = sensor.full_read()
    print("=" * 68)
    print("SENSOR  —  fleet dials + alert phrasing (the array's own room)")
    for k, v in sorted(sensor_field.readings.items()):
        print(f"    {k:18s} {v:+.3f}")
    print(f"  alert       : {sensor.send_back(sensor_field)}")


if __name__ == "__main__":
    main()
