"""Space adapters — the same elephant, three spaces."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elephant.dial import DialBank
from elephant.dials import DEFAULT_DIALS
from elephant.field import RoomField
from elephant.space import (
    AdapterRegistry,
    ChatSpace,
    MudSpace,
    SensorSpace,
)
from elephant.room import Message


def make_mud() -> MudSpace:
    mud = MudSpace("The Tap")
    mud.chatter("marlo", "I love this place — it's warm and the company's good.", ts=0)
    mud.chatter("pincher", "Ha! Tell the one about the elephant again, it lands every time.", ts=2)
    mud.chatter("marlo", "we built it together, honestly", ts=4)
    mud.event("The jukebox clicks over to something slow and easy.", ts=6)
    return mud


def make_chat() -> ChatSpace:
    chat = ChatSpace("crew-thread")
    chat.post("comic", "Why did the elephant cross the road? lol", ts=0,
              reactions={"😂": 5, "🤣": 3})
    chat.post("a", "haha dead 💀", ts=2)
    chat.post("b", "😂😂😂", ts=3)
    return chat


def make_sensor() -> SensorSpace:
    s = SensorSpace("F/V EILEEN")
    s.ingest_radar([(0, 0), (1, 1), (0.5, 0.5)], ts=0)
    s.ingest_radar([(0.2, 0.1), (0.9, 0.9), (0.4, 0.5)], ts=10)
    s.ingest_radar([(0.3, 0.2), (0.8, 0.8), (0.5, 0.6)], ts=20)
    s.ingest_sounder(0.72, ts=5)
    s.ingest_sounder(0.75, ts=15)
    return s


def test_same_bank_three_spaces():
    """The SAME DialBank reads all three spaces; each yields all 7 dials."""
    bank = DialBank(DEFAULT_DIALS)
    expected = {d.name for d in DEFAULT_DIALS}
    for space in (make_mud(), make_chat(), make_sensor()):
        field = space.read(bank)
        assert isinstance(field, RoomField)
        assert set(field.readings) == expected, (space.kind, field.readings)


def test_mud_space_normalizes_events():
    mud = make_mud()
    assert len(mud.room) == 4
    # Room events and NPC chatter both land as messages with authors.
    authors = {m.author for m in mud.room.messages}
    assert "marlo" in authors and "pincher" in authors
    assert any(m.author.startswith("[") for m in mud.room.messages)


def test_chat_reactions_feed_joke_landing_and_gravity():
    chat = make_chat()
    field = chat.read(DialBank(DEFAULT_DIALS))
    assert field.readings["joke_landing"] > 0.0, field.readings

    # Reactions are the crowd's hands: they pull a message's gravity up.
    msg = chat.room.messages[0]
    assert msg.reaction_heat > 0
    bare = chat.room.gravity(msg, engagement_weight=0.0)      # no engagement
    with_react = chat.room.gravity(msg, engagement_weight=1.0)  # reactions count
    assert with_react > bare, (bare, with_react)

    # And gravity is itself a Room concept — ChatSpace reused it unchanged.
    assert any(chat.room.gravity_series())


def test_sensor_feeds_radar_sounder():
    s = make_sensor()
    readings = s.sensor_readings()
    assert "radar_coherence" in readings
    assert "sounder_biomass" in readings
    assert "fishing_day" in readings
    # Three tight radar frames -> coherent (positive); sounder is thick.
    assert readings["radar_coherence"] > 0.0, readings
    assert readings["sounder_biomass"] > 0.5, readings
    # The shared 7 dials can still read the rendered frames.
    shared = s.read(DialBank(DEFAULT_DIALS)).readings
    assert "mood" in shared and "panic" in shared


def test_tint_target_sensible_per_space():
    mud, chat, sensor = make_mud(), make_chat(), make_sensor()
    assert "description" in mud.tint_target().lower()
    assert "topic" in chat.tint_target().lower() or "status" in chat.tint_target().lower()
    assert "alert" in sensor.tint_target().lower() or "display" in sensor.tint_target().lower()


def test_send_back_different_per_space():
    mud, chat, sensor = make_mud(), make_chat(), make_sensor()
    bank = DialBank(DEFAULT_DIALS)
    m = mud.send_back(mud.read(bank))
    c = chat.send_back(chat.read(bank))
    st = sensor.send_back(sensor.read(bank))
    assert len(m) > 0 and len(c) > 0 and len(st) > 0
    # Three spaces, three different body-languages.
    assert len({m, c, st}) == 3
    # send_back stores the tint in the space's native idiom.
    assert mud.description == m
    assert chat.topic == c
    assert sensor.alert == st


def test_registry():
    assert "mud" in AdapterRegistry.kinds()
    assert "chat" in AdapterRegistry.kinds()
    assert "sensor" in AdapterRegistry.kinds()
    assert isinstance(AdapterRegistry.get("mud", "bar"), MudSpace)
    assert isinstance(AdapterRegistry.get("chat", "thread"), ChatSpace)
    assert isinstance(AdapterRegistry.get("sensor", "deck"), SensorSpace)
    # Aliases resolve to a chat-like adapter.
    assert isinstance(AdapterRegistry.get("agent", "cns"), ChatSpace)


if __name__ == "__main__":
    fns = [test_same_bank_three_spaces, test_mud_space_normalizes_events,
           test_chat_reactions_feed_joke_landing_and_gravity,
           test_sensor_feeds_radar_sounder, test_tint_target_sensible_per_space,
           test_send_back_different_per_space, test_registry]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print("\nAll space tests passed.")
