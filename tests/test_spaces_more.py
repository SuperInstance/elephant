"""Four more space adapters — agent, human/bot, async, doc."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import elephant.space_more  # noqa: F401  (registers the four adapters)

from elephant.dial import DialBank
from elephant.dials import DEFAULT_DIALS
from elephant.field import RoomField
from elephant.space import AdapterRegistry, ChatSpace
from elephant.space_more import AgentSpace, AsyncSpace, DocSpace, HumanBotSpace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECTED_DIALS = {d.name for d in DEFAULT_DIALS}


def make_agent() -> AgentSpace:
    bus = AgentSpace("CNS-bus")
    bus.agent("agent-7", "radar cluster tight — we're on fish, hold the drag", ts=0)
    bus.agent("agent-3", "roger. sharing the drag with the fleet", ts=2)
    bus.system("scheduler: batch 14 dispatched to 6 agents", ts=4)
    bus.agent("agent-7", "joke's on the fish — they think they're safe lol", ts=5,
              reactions={"😂": 4})
    return bus


def make_human_bot() -> HumanBotSpace:
    hb = HumanBotSpace("support-desk")
    hb.human("casey", "is the elephant awake yet? the room feels warm already", ts=0)
    hb.bot("helpdesk", "I'm here — what can I fix?", ts=1)
    hb.human("marlo", "nothing to fix, just saying hi. good crowd tonight", ts=3)
    hb.human("casey", "agreed — the elephant's been warm all week", ts=6)
    return hb


def build_async(scale: float) -> AsyncSpace:
    a = AsyncSpace("quarterly-plan", half_life=1800.0, half_life_scale=scale)
    a.email("captain", "Q4 fleet plan",
            "Let's get the fleet tight before the ice comes.", ts=0)
    a.email("ops", "Re: Q4 fleet plan",
            "Radar arrays calibrated; biomass thick north of the bar.", ts=3600)
    a.email("captain", "Re: Q4 fleet plan",
            "Good. The elephant says the room is warm — trust it.", ts=172800)
    return a


def test_agent_space_normalizes_events():
    bus = make_agent()
    field = bus.read(DialBank(DEFAULT_DIALS))
    assert isinstance(field, RoomField)
    assert set(field.readings) == EXPECTED_DIALS
    authors = {m.author for m in bus.room.messages}
    assert "agent-7" in authors and "agent-3" in authors
    # System events share the same clock/room, authored by the bus itself.
    assert "[bus]" in authors
    ts = [m.ts for m in bus.room.messages]
    assert ts == sorted(ts)


def test_human_bot_tagging_ratio_and_presence():
    hb = make_human_bot()
    # 3 human messages / 1 bot message.
    assert hb.humans_vs_bots() == 3.0
    assert hb.kind_of("casey") == "human"
    assert hb.kind_of("helpdesk") == "bot"
    # The presence dial reads the two traces distinctly.
    by_kind = hb.presence_by_kind()
    assert set(by_kind) == {"human", "bot"}
    assert by_kind["human"] > by_kind["bot"] >= 0.0
    # Default read() swaps in the kind-aware presence dial — still 7 dials.
    field = hb.read()
    assert set(field.readings) == EXPECTED_DIALS
    assert 0.0 <= field.readings["presence"] <= 1.0


def test_async_half_life_scaling_changes_gravity():
    tight = build_async(1.0)
    loose = build_async(20.0)
    assert loose.effective_half_life == 20.0 * tight.effective_half_life
    # The trailing message is a long way out; under a tight half-life it has
    # cooled to nothing, under a stretched one it still pulls attention.
    g_tight = tight.gravity(tight.room.messages[-1])
    g_loose = loose.gravity(loose.room.messages[-1])
    assert g_loose > g_tight, (g_tight, g_loose)
    # The stretched series keeps more total pull across the thread.
    assert sum(loose.gravity_series()) > sum(tight.gravity_series())
    # Long-latency echo is a plain number (reverb over the stretched series).
    assert isinstance(loose.reverberation(), float)
    # Still a full 7-dial field.
    assert set(tight.read(DialBank(DEFAULT_DIALS)).readings) == EXPECTED_DIALS


def test_doc_space_ingests_real_git_log():
    doc = DocSpace("elephant-repo", repo_path=REPO)
    doc.ingest_git_log(max_count=20)
    assert len(doc.room) > 0
    # Authors are committers (real names, no synthetic [file]/[room] authors).
    authors = {m.author for m in doc.room.messages}
    assert len(authors) >= 2
    assert not any(a.startswith("[") for a in authors)
    # Normalized timestamps: monotonic, starting at 0.
    ts = [m.ts for m in doc.room.messages]
    assert ts == sorted(ts)
    assert ts[0] == 0.0
    field = doc.read(DialBank(DEFAULT_DIALS))
    assert set(field.readings) == EXPECTED_DIALS
    # A commit/file-event/review-comment can also land as messages.
    doc.file_event("elephant/space_more.py", "added")
    doc.review_comment("reviewer", "looks right — ship it")
    assert len(doc.room) > 0 and doc.room.messages[-1].author == "reviewer"


def test_four_registered_in_registry():
    assert isinstance(AdapterRegistry.get("agent", "bus"), AgentSpace)
    assert isinstance(AdapterRegistry.get("human_bot", "ch"), HumanBotSpace)
    assert isinstance(AdapterRegistry.get("async", "thread"), AsyncSpace)
    assert isinstance(AdapterRegistry.get("doc", "repo"), DocSpace)
    # The forward placeholders are superseded, but AgentSpace/HumanBotSpace/
    # AsyncSpace remain chat-like, so the old alias contract still holds.
    assert isinstance(AdapterRegistry.get("agent", "bus"), ChatSpace)
    assert isinstance(AdapterRegistry.get("human_bot", "ch"), ChatSpace)
    assert isinstance(AdapterRegistry.get("async", "thread"), ChatSpace)
    for kind in ("agent", "human_bot", "async", "doc"):
        assert kind in AdapterRegistry.kinds()


if __name__ == "__main__":
    fns = [test_agent_space_normalizes_events,
           test_human_bot_tagging_ratio_and_presence,
           test_async_half_life_scaling_changes_gravity,
           test_doc_space_ingests_real_git_log,
           test_four_registered_in_registry]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print("\nAll space_more tests passed.")
