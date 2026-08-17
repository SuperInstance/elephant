"""elephant — tests: the murmur shadow journal.

The captain's Terrain reframing: murmur's commits ARE shadows on the
cave wall — witness marks of an agent's internal monologue. This suite
tests the self-populating shadow journal (`elephant/murmur.py`):

- every monologue becomes an entry with its JEPA readings intact as
  terrain front-matter (write → read round-trips with real numbers);
- the journal reads as a Room the DialBank can feel (a field is
  produced from the shadow-trail);
- retrieval by feeling finds the witness with the target profile
  (a panic reading finds the panic monologue; ranges work);
- the pulse seam: PulseLoop's internal_monologue() self-populates the
  journal, each witness carrying that pulse's perception readings;
- the shadow-trail is a room the elephant walks into (MurmurSpace);
- in a git repo, each witness mark becomes a commit.
"""
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from elephant.dial import DialBank
from elephant.dials import DEFAULT_DIALS
from elephant.field import RoomField, read_field
from elephant.murmur import MurmurJournal, MurmurSpace, murmurize_pulse
from elephant.pulse import PulseLoop
from elephant.room import Room
from elephant.space import ChatSpace

BANK = DialBank(DEFAULT_DIALS)
EXPECTED = {d.name for d in DEFAULT_DIALS}


def make_journal(tmp_path) -> MurmurJournal:
    return MurmurJournal(str(tmp_path / "journal"), space_id="the-tap")


def test_write_monologue_creates_entry_with_readings_frontmatter(tmp_path):
    j = make_journal(tmp_path)
    readings = {"mood": 0.42, "panic": 0.05, "volume": 0.6}
    entry = j.write_monologue(
        "I haven't said a word, but the room is warming.",
        readings, ts=12.5, topic="warm night",
        direction={"mood": 0.1}, warmth=0.33,
    )
    # The entry dict carries the terrain front-matter intact.
    assert entry["text"] == "I haven't said a word, but the room is warming."
    assert entry["readings"] == readings
    assert entry["direction"] == {"mood": 0.1}
    assert entry["warmth"] == pytest.approx(0.33)
    assert entry["ts"] == pytest.approx(12.5)
    assert entry["topic"] == "warm night"
    assert entry["space_id"] == "the-tap"
    # The witness file exists and its front-matter is human-visible.
    files = os.listdir(j.path)
    assert len(files) == 1 and files[0].endswith(".md")
    with open(os.path.join(j.path, files[0])) as f:
        raw = f.read()
    assert raw.startswith("---")
    assert "readings:" in raw and '"panic": 0.05' in raw
    assert "topic:" in raw and '"warm night"' in raw


def test_write_read_round_trip_with_real_numbers(tmp_path):
    j = make_journal(tmp_path)
    r1 = {"mood": 0.55, "panic": 0.02, "volume": 0.7, "earnestness": 0.8}
    r2 = {"mood": -0.6, "panic": 0.72, "volume": 0.95, "earnestness": 0.2}
    j.write_monologue("the room is warm and bright", r1, ts=1.0, topic="warm")
    j.write_monologue("FIRE — everything is wrong", r2, ts=2.5, topic="panic")

    # A fresh journal over the same path recovers everything.
    reopened = MurmurJournal(j.path, space_id="the-tap")
    entries = reopened.entries()
    assert len(entries) == 2
    assert [e["index"] for e in entries] == [1, 2]
    assert entries[0]["readings"] == r1
    assert entries[1]["readings"] == r2
    assert entries[1]["ts"] == pytest.approx(2.5)
    assert entries[1]["text"] == "FIRE — everything is wrong"
    # The journal reads as a Room with real timestamps, and the DialBank
    # feels a field from the shadow-trail.
    room = reopened.read_room()
    assert isinstance(room, Room) and len(room) == 2
    assert room.messages[0].author == "murmur"
    assert room.messages[0].ts == pytest.approx(1.0)
    field = read_field(room, BANK)
    assert isinstance(field, RoomField)
    assert set(field.readings) == EXPECTED
    assert isinstance(field.warmth(), float)


def test_retrieve_by_reading_finds_target_profile(tmp_path):
    j = make_journal(tmp_path)
    j.write_monologue("a quiet hour, nothing moving", {"mood": 0.1, "panic": 0.02},
                      ts=0.0, topic="quiet")
    j.write_monologue("the panic is real, everyone is running",
                      {"mood": -0.7, "panic": 0.85}, ts=10.0, topic="panic spike")
    j.write_monologue("the room is warm and everyone is laughing",
                      {"mood": 0.9, "panic": 0.03}, ts=20.0, topic="warm night")

    # Retrieval by feeling: a high panic reading finds the panic witness.
    hits = j.retrieve({"panic": 0.8})
    assert len(hits) == 1
    assert hits[0]["topic"] == "panic spike"
    assert hits[0]["readings"]["panic"] == pytest.approx(0.85)

    # Range query — "did panic ever cross the band?" (the deadband ring).
    hits = j.retrieve({"panic": (0.5, 1.0)})
    assert [h["topic"] for h in hits] == ["panic spike"]
    # Inside the band nothing rings: the in-band witnesses all sit at
    # distance 0 (quiet + warm night); the out-of-band one trails.
    hits = j.retrieve({"panic": (0.0, 0.05)}, k=3)
    in_band = [h["topic"] for h in hits if h["_distance"] == 0.0]
    assert set(in_band) == {"quiet", "warm night"}

    # Topic filter narrows the trail first; a threshold rejects the
    # mismatch — the quiet entry is nearest, but it is not close.
    hits = j.retrieve({"panic": 0.8}, topic="quiet", threshold=0.5)
    assert hits == []

    # k lets the elephant see the whole wall at once, nearest first.
    hits = j.retrieve({"panic": 0.8}, k=3)
    assert [h["topic"] for h in hits] == ["panic spike", "warm night", "quiet"]
    assert all("_distance" in h for h in hits)
    assert hits[0]["_distance"] < hits[1]["_distance"]


def test_retrieve_by_movement_profile(tmp_path):
    """Retrieval works on the macro read too — direction, not just level."""
    j = make_journal(tmp_path)
    j.write_monologue("holding", {"panic": 0.3}, ts=0.0, topic="flat",
                      direction={"panic": 0.01})
    j.write_monologue("the spike is coming", {"panic": 0.3}, ts=1.0,
                      topic="spike", direction={"panic": 0.45})
    hits = j.retrieve({"d_panic": 0.5})
    assert hits[0]["topic"] == "spike"


def test_murmur_space_adapter_reads_like_a_room(tmp_path):
    j = make_journal(tmp_path)
    j.write_monologue("warm and bright all night", {"mood": 0.6, "panic": 0.02},
                      ts=0.0, topic="warm")
    j.write_monologue("something is wrong, very wrong",
                      {"mood": -0.8, "panic": 0.9}, ts=5.0, topic="alarm")
    space = MurmurSpace(j)
    assert space.kind == "doc"
    room = space.room
    assert isinstance(room, Room) and len(room) == 2
    assert room.messages[0].channel == "warm"
    field = space.read(BANK)
    assert set(field.readings) == EXPECTED
    # The elephant writes its readout back as a status line.
    tinted = space.send_back(field)
    assert isinstance(tinted, str) and tinted == space.status
    # The wall is always current: new witnesses appear on the next read.
    j.write_monologue("the room is quiet again", {"mood": 0.0, "panic": 0.01},
                      ts=10.0, topic="quiet")
    assert len(space.room) == 3


def test_pulse_seam_murmurizes_the_internal_monologue(tmp_path):
    """The self-populating sustaining system: every pulse's silent
    thinking becomes a witness mark carrying that pulse's readings."""
    chat = ChatSpace("the-tap-overnight")
    chat.post("marlo", "the room is warm, bright, and everyone is laughing",
              ts=0.0)
    chat.post("pincher", "cheers and thanks, this place glows", ts=1.0)
    loop = PulseLoop("murmur", chat, period=2.0)
    j = make_journal(tmp_path)

    # Three pulses: warm, then a panic spike, then quiet. Each witness
    # is written on the pulse that felt the room at that moment.
    e1 = murmurize_pulse(loop, j, topic="warm night", ts=0.0)
    chat.post("deck", "FIRE in the galley! evacuate now!! mayday", ts=2.0)
    e2 = murmurize_pulse(loop, j, topic="panic spike", ts=2.0)
    chat.post("deck", "the alarm is over, the room goes still", ts=4.0)
    e3 = murmurize_pulse(loop, j, topic="quiet hour", ts=4.0)

    assert e1 is not None and e2 is not None and e3 is not None
    assert len(j) == 3
    # Every witness carries the pulse's raw readings...
    assert set(e2["readings"]) == EXPECTED
    assert isinstance(e2["readings"]["panic"], float)
    # ...and the macro read (direction / rate / warmth kinematics).
    assert "direction" in e2 and "rate_of_change" in e2
    assert "warmth" in e2 and "warmth_direction" in e2 and "warmth_rate" in e2
    # The monologue is the witness's body — silent thinking, committed.
    assert isinstance(e2["text"], str) and len(e2["text"]) > 0
    # The panic pulse felt more panic than the warm pulse did.
    assert e2["readings"]["panic"] > e1["readings"]["panic"]
    # The elephant can then retrieve the panic witness by feeling.
    hits = j.retrieve({"panic": e2["readings"]["panic"]})
    assert hits[0]["topic"] == "panic spike"


def test_jepa_memory_path_when_present(tmp_path):
    """When elephant/jepa_rag.py exists, the journal's witness marks
    index into a JepaMemory as moments, and retrieval BY VIBE works —
    the panic moment is the one that felt most like a panic reading.
    retrieve() itself stays deterministic (levels, not vibes).
    Skips cleanly if the memory is absent."""
    pytest.importorskip("elephant.jepa_rag")
    j = make_journal(tmp_path)
    j.write_monologue("a quiet hour, nothing moving",
                      {"mood": 0.1, "panic": 0.02},
                      ts=0.0, topic="quiet")
    j.write_monologue("the panic is real, everyone is running",
                      {"mood": -0.7, "panic": 0.85},
                      ts=10.0, topic="panic spike")
    j.write_monologue("the room is warm and everyone is laughing",
                      {"mood": 0.9, "panic": 0.03},
                      ts=20.0, topic="warm night")

    mem = j.to_memory()
    assert mem is not None and len(mem) == 3
    hits = j.retrieve_feeling({"panic": 0.8}, k=3)
    assert len(hits) == 3
    assert hits[0]["topic"] == "panic spike"
    assert "_distance" in hits[0]
    # The deterministic path is unchanged by the memory's presence:
    # deadband gates ring on levels, and movement profiles (d_*) are
    # not dials — they still find the spike.
    assert j.retrieve({"panic": 0.8})[0]["topic"] == "panic spike"
    j.write_monologue("holding", {"panic": 0.3}, ts=30.0, topic="flat",
                      direction={"panic": 0.01})
    j.write_monologue("the spike is coming", {"panic": 0.3}, ts=31.0,
                      topic="spike", direction={"panic": 0.45})
    assert j.retrieve({"d_panic": 0.5})[0]["topic"] == "spike"


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_witness_marks_become_commits_in_a_git_repo(tmp_path):
    """Murmur's pattern in elephant idiom: every thought a commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email",
                    "murmur@elephant.local"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name",
                    "murmur"], check=True)
    j = MurmurJournal(str(repo / "thinking"), space_id="the-tap")
    assert j.git_enabled is True
    j.write_monologue("the first witness mark", {"mood": 0.5, "panic": 0.02},
                      ts=1.0, topic="first light")
    proc = subprocess.run(["git", "-C", str(repo), "log",
                           "--format=%s", "-1"],
                          capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == "murmur: first light"
    # A second commit follows the first — the trail is version-controlled.
    j.write_monologue("and then the room went quiet", {"mood": 0.0},
                      ts=2.0, topic="quiet")
    proc = subprocess.run(["git", "-C", str(repo), "log", "--oneline"],
                          capture_output=True, text=True, check=True)
    assert len(proc.stdout.splitlines()) == 2
