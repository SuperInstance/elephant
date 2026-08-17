"""Four more rooms — the SAME elephant reading an agent bar, a human+bot
channel, an async email thread, and its own git history.

The elephant doesn't care if the room is made of agents, humans, bots,
email, or a commit log. It only cares how warm the room is — and how the
room's light changes everyone in it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elephant.dial import DialBank
from elephant.dials import DEFAULT_DIALS
from elephant.space_more import AgentSpace, AsyncSpace, DocSpace, HumanBotSpace


def main():
    bank = DialBank(DEFAULT_DIALS)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 1. An agent bar — the CNS bus.
    agents = AgentSpace("CNS-bus")
    agents.agent("agent-7", "radar cluster tight — we're on fish, hold the drag", ts=0)
    agents.agent("agent-3", "roger. sharing the drag with the fleet", ts=2)
    agents.system("scheduler: batch 14 dispatched to 6 agents", ts=4)
    agents.agent("agent-7", "joke's on the fish — they think they're safe lol", ts=5,
                 reactions={"😂": 4})

    # 2. A human + bot channel.
    hb = HumanBotSpace("support-desk")
    hb.human("casey", "is the elephant awake yet? the room feels warm already", ts=0)
    hb.bot("helpdesk", "I'm here — what can I fix? 🤖", ts=2)
    hb.human("marlo", "nothing to fix, just saying hi. good crowd tonight", ts=4)
    hb.human("casey", "agreed — the elephant's been warm all week", ts=6)

    # 3. An async email thread — long time-deltas, stretched half-life.
    email = AsyncSpace("quarterly-plan", half_life=1800.0, half_life_scale=20.0)
    email.email("captain", "Q4 fleet plan",
                "Let's get the fleet tight before the ice comes.", ts=0)
    email.email("ops", "Re: Q4 fleet plan",
                "Radar arrays calibrated; biomass thick north of the bar.", ts=86400)
    email.email("captain", "Re: Q4 fleet plan",
                "Good. The elephant says the room is warm — trust it.", ts=172800)

    # 4. The elephant's own git history as a room.
    doc = DocSpace("elephant-repo", repo_path=repo)
    doc.ingest_git_log(max_count=20)

    spaces = [
        ("an agent bar", agents),
        ("a human+bot channel", hb),
        ("an async email thread", email),
        ("its own git history", doc),
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

    # The extras each adapter carries beyond the shared seven dials.
    print("=" * 68)
    print("HUMAN/BOT  —  presence reads humans vs bots distinctly")
    print(f"  humans_vs_bots  : {hb.humans_vs_bots():.2f}")
    print(f"  presence_by_kind: { {k: round(v, 3) for k, v in hb.presence_by_kind().items()} }")
    print()
    print("ASYNC  —  long-latency echo (stretched half-life)")
    print(f"  effective half-life : {email.effective_half_life:,.0f}s")
    print(f"  gravity (oldest)    : {email.gravity(email.room.messages[0]):.4f}")
    print(f"  gravity (newest)    : {email.gravity(email.room.messages[-1]):.4f}")
    print()
    print("DOC  —  the repo's shape, read as a room")
    print(f"  commits ingested    : {len(doc.room)}")
    print(f"  committers          : {sorted({m.author for m in doc.room.messages})}")


if __name__ == "__main__":
    main()
