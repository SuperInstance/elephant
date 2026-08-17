# Tap-Night Themes — four rooms, four elephants

The Tap isn't one room. It's four, depending on the night. The same seven
dials — mood, volume, earnestness, cynicism, joke_landing, panic, presence —
read a different room on open mic night than on trivia night, and a different
elephant walks out of each one.

`elephant/tapnight_themes.py` is the reusable presets for those nights. Each
`Theme` is a complete session recipe:

- **`cast()`** — the archetypes (`Participant`s), each with a starter
  `dial_weights` prior and a native `vibe`. These are *priors*, not answers:
  over many evenings the weights self-tune toward wherever each archetype
  feels engaged (see `tap-night-operations.md`). A theme just names the
  different guitarists and hands them their first guitar.
- **`room_tone`** — the seed: the opening messages that set the room's
  temperature before the night really starts.
- **`prompts`** — the starter prompt for each archetype.
- **`description`** — the intended vibe in one line.

The registry is `THEMES = {"open_mic", "trivia", "ttrpg", "singles"}`.

## The four nights

### `open_mic` — performers + a collective audience
**Vibe:** the room swings between the roar of a joke landing (high
`joke_landing`) and the hush of being taken seriously (`earnestness`).
Cast: a comic (joke_landing + mood), a poet (mood + earnestness), and an
**audience** that reads as a *collective* — low charisma (diffuse pull),
moderate acclimation, an ear for mood and whether the joke landed.

### `trivia` — a host + rival teams
**Vibe:** earnest and suspicious in equal measure. The room means it
(`earnestness`) but does not trust a wrong answer (`cynicism`); buzzer
moments spike `volume`, and a wrong answer cools `mood` and feeds the
eye-rolls. Cast: a host (earnest + presence), and two teams with *different*
leanings — team_north earnest-first, team_south cynicism-first — so the
rivalry is real, not a mirror.

### `ttrpg` — a GM + a party of players
**Vibe:** the room swings hard with the story. A tense roll spikes `panic`
and `volume`; a nat-20 spikes `mood` and `joke_landing`. The field *is* the
dice, felt by everyone at the table. Cast: a GM (volume + panic + presence,
the table's anchor) and three players who each work a different corner —
rogue (panic + volume), paladin (volume + presence), wizard (presence +
joke_landing, the one who laughs at the twenty).

### `singles` — a small mixed room
**Vibe:** warm-but-nervous — moderate `mood`, elevated `presence` (everyone
is watching everyone), tentative `joke_landing`. Cast: six people, all
leaning mood + presence + earnestness but with *different* dial_weights, so
the chemistry is observable — maya reads the room through `mood`, rowan
through `presence`, two agents reading the same warm room differently.

## How the rooms differ (measured)

Feeding each theme's `room_tone` into a session and reading the raw field
(the seed's own dial reading, before charisma/acclimation bends it):

| night     | signature dials                                | mood  | earnest | cynicism | joke_landing | panic | presence |
|-----------|------------------------------------------------|-------|---------|----------|--------------|-------|----------|
| open_mic  | joke_landing, mood                             | warm  | mid     | none     | **+0.81**    | low   | mid      |
| trivia    | earnestness, cynicism, volume                  | cold  | **+1.0**| **+1.0** | none         | low   | mid      |
| ttrpg     | panic, volume, presence                        | +0.67 | high    | none     | none         | **+0.61** | mid   |
| singles   | mood, presence, (tentative joke_landing)       | warm  | high    | low      | tentative    | low   | **+0.81** |

The four are measurably different rooms: trivia is the earnest/suspicious
room, open mic is the laughter room, TTRPG is the panic room, singles is the
thrumming warm room. (These are the raw dial readings; the v0 dials are naive
keyword matchers and can saturate — see `tap-night-operations.md` for the
caveats. The *differences* between rooms are the point, not the absolutes.)

## Running a themed night

```python
import sys
sys.path.insert(0, "/home/eileen/projects/elephant")

from elephant.tapnight_themes import THEMES
from elephant.tapnight import DIAL_NAMES

theme = THEMES["ttrpg"]                # open_mic | trivia | ttrpg | singles

session = theme.make_session()         # TapNightSession with the cast loaded
session.start_session()
theme.seed(session)                    # set the opening tone

# ... feed the night's real lines ...
session.speak("gm", "The door creaks, and behind it — a THING. Roll initiative.",
              reactions={"🔥": 1})

# read the room: the effective field (charisma + cast warming to it) ...
f = session.room_field()
print(f"warmth {f.warmth():+.2f}  κ {f.concentration():.2f}")

# ... and the raw field (the room before charisma bent it) ...
raw = session.raw_field()
for name, v in raw.readings.items():
    print(f"  {name:<13} {v:+.2f}")

# self-tune each archetype, then close the night
for name in session.participants:
    session.tune_participant(name)
print(session.end_session())
```

`make_session()` returns a `TapNightSession` (from `tapnight.py`) pre-loaded
with the theme's cast — so the regulars' settings persist and self-tune
across evenings exactly as the runbook describes. `seed()` just speaks the
opening tone; the two can be replayed as many nights as you like, and the
weights will diverge toward each archetype's own guitar.

## Which room am I in?

You only notice the elephant when you change rooms. The same seven dials, the
same `TapNightSession` engine, four casts and four opening tones — and four
different fields walk out. The `tests/test_tapnight_themes.py` suite pins
this down: each theme builds a valid, distinct cast; a themed session runs
end-to-end (speak → field → tune); and the four starter tones read as
measurably different rooms (`trivia.earnestness > open_mic.joke_landing`,
`singles.presence > ttrpg.panic`, and so on).
