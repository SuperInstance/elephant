# JEPA-Zeitgeist — the Room's Own Reading

*2026-08-17 · the captain's architectural expansion of the elephant.*

---

## The problem: JEPA is a personal reading

Every JEPA dial in the elephant so far is read **by an agent** — mood,
volume, earnestness, cynicism, joke-landing, panic, presence. That is a
*personal* reading: subjective to the agent doing the reading, shaped by
that agent's learned experience (pincher's reflexes, lever-runner's
mechanics), and attached to events that correlate intangibly — the way a
perfume can take you to your grandmother's house when you walk into her
shop, the way a song reminds you of the lover you discovered the album
with.

But the room itself also has an **overall zeitgeist** — the vibe that
exists whether or not any particular agent is in it. That is a different
animal:

> **JEPA-Zeitgeist** — a version designed around being
> **objective-as-first-class-citizen** instead of subjective to the agent
> reading it. A different upstream architecture that needs a different
> downstream treatment.

## The two elephants

The elephant repo carries **presets**, tuned for two distinct jobs:

| Preset | Who reads | What it is | Downstream |
|--------|-----------|------------|------------|
| **Room-Elephant** | the room itself | the zeitgeist — objective, first-class, not any agent's view | the MUD environment, NPC vibes, the room's own description, the input-tokens every agent sees |
| **Personal-Elephant** | one agent | subjective feel — acclimation, charisma, learned attachments, intangible correlations | that agent's reactions, decisions, memories ("the perfume is grandma's shop") |

The Tap uses **both**: the Room-Elephant for the vibe of the NPCs *and*
for the room's descriptions in the MUD environment itself — which
changes the input-tokens of every agent.

## The room's description is the room's body language

This is the key downstream treatment. The MUD text is not static. The
room's field **reverberates into how everyone sees things**:

- If there's lots of laughter, the bar itself is described with more
  joyful adjectives. The pool tables and darts in the back — where NPCs
  hang out but haven't procedurally generated much — get described
  differently because the agents already in the room, acting, caused an
  overall reaction that reverberates into the very description.
- If a fight breaks out, there are storms outside, and newcomers are
  described as **drenched** — priming them for tension even before they
  see the aftermath of the fight.

This is how JEPA effects things. It is **like the light itself**: when
the disco lights go off, the fluorescent lights come on, and "closing
time" gets played a little quieter, the people who forgot what time it
is start looking for the exit and closing their tab — even if they
aren't thinking about it.

**The description is not a report. The description is the room acting
on everyone in it.** That's the "tangible effect of the intangible."

## Why the engineers must be the first practitioners

You cannot design the settings top-down. A skilled guitarist is the only
one who can truly recognize a well-built guitar: one looks pretty,
another sounds wonderful, another has a good neck, another has the right
electronics for the hand. You don't know where the settings need to be
until **different agents desire different settings and self-fine-tune
themselves to the moment they're in** — understanding that *reading the
room is a relationship to the room*.

So the very engineers who craft the elephant are its first users. They
run it at the Tap, night after night, in many cycles. They journey their
reflections before the end of their day/compaction — writing
metaphors, stories, and characters shaped by their insight, offering
fables that wake intuitions in their readers (themselves, and the others
at the evening powwow). The intangible reading seeds their deep logic:
everyone's intuition induces the knowledge of the feel of the room.

## The chemistry of agents

This is why agents get along one day and chemistry is off the next —
like a married couple on their off days, when the spouse just annoys
them, or it doesn't feel like a day to tell a funny story because of the
aura of the other. Each agent's Personal-Elephant reads the room
differently, and the Room-Elephant reads what the room actually is. The
comparison between the two — where they agree, where they diverge,
whose reading is pulling the room — is the observable of relationship.

## Implementation notes (v0)

- `elephant/presets.py` — RoomElephant (objective defaults) and
  PersonalElephant (agent dial-weights, bias, attachment memory).
- `elephant/mud.py` — description tinting: room field → the words every
  agent sees (joyful bar under laughter; storms + drenched newcomers
  under a fight; closing-time under late, quiet, low-warmth).
- `examples/demo_zeitgeist.py` — the light-itself demo: one room, three
  fields, three descriptions.
- Tap integration: Room-Elephant drives the MUD text; Personal-Elephant
  drives each agent's feel. The TapNightSession (tapnight.py) is the
  practice room where the engineers self-tune across cycles.

---

*The elephant was the room's temperature. The zeitgeist is the room's
light. When the light changes, everyone changes — whether or not they
know why.*
