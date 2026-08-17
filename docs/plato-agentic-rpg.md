# Plato-Based Agentic RPG — the elephant as dungeon master

*2026-08-17 · a design for GM-less (or elephant-GM'd) tabletop roleplay
built from the fleet's own organs: the elephant, the avatars, the
terrain. The captain: "we can rapidly iterate as JEPA learning avatar
building round characters at Tap's bar but also as parts of
Plato-based Agentic RPGs."*

---

## The pitch

A tabletop RPG engine where the dungeon master is not a person and not
a language model — the dungeon master is the elephant.

- **The WORLD is rooms**, each with an elephant reading: a 9-dial JEPA
  field (mood, volume, earnestness, cynicism, joke_landing, panic,
  presence, model_vs_code, vision), a warmth, a concentration κ, a
  trend, an anomaly. A dungeon room, a tavern, a haunted wheelhouse —
  each has its own temperature.
- **The PLAYERS are avatars** — round characters with PersonalElephants
  (a vibe in dial space, the dials they care about, charisma), pulses,
  and voices.
- **The PERCEPTION CHECK is the roll.** A player entering a room runs
  the pulse's `perception_check()`: the room's direction (last two
  readings) and rate of change (last three+) over its recent history
  IS what they perceive. The number doesn't matter; two numbers show
  direction, three show rate of change.
- **The GM's narration is the SHADOW** — `mud.tint_description` +
  perception reports rendered in words. The cave wall. Never the
  terrain itself.
- **The DEADBAND rings the plot.** When a room's field crosses its
  deadband — a fight erupts, an anomaly spikes, a trend inverts — the
  Ring escalates: GM line, plot stage, the dungeon answers.
- **Z_out predicts.** The room's trend dial (the dual-db bridge) tells
  the players what's coming around the corner — the ghost-trail ahead,
  used for foreshadowing.

The code: `elephant/plato_rpg.py` (engine), `tests/test_plato_rpg.py`
(rolls, rings, and shadows), `examples/demo_plato_rpg.py` (a one-shot,
"The Fogbound Harbor"). The demo runs a complete session:

```bash
python3 examples/demo_plato_rpg.py
```

---

## The three laws (docs/terrain-2026-08-17.md, as game rules)

1. **TERRAIN** — the true state: the full field history, the vectors,
   the Z_in encodings. Nobody sees it whole, and that is not the
   point. In the game, the terrain is the raw field history each room
   keeps (`field_history_raw`) — recoverable, never narrated.
2. **SHADOW** — what anyone actually sees: the tinted description, the
   perception report in words, the pulse monologue, the GM's lines.
   Enough to agree on the action, never complete. The game's rule:
   **the narration is ALWAYS a shadow.** Rooms are described through
   `tint_description` (a changed field changes the words); rolls are
   rendered as "the room is warming, and the movement is building —
   panic is the loudest hand, rising 0.20 per beat"; raw vectors never
   appear in the transcript.
3. **DEADBAND** — the discipline: only significant movement rings, and
   a ring advances the plot. Not every flicker; only the moves that
   matter. Mechanically: hysteresis per monitor (ring on the crossing
   from below, re-arm only after dipping back under), a short cooldown
   against spam, and at most three rings narrate per turn.

## Why the elephant is the right GM

A GM's job is to keep a room's temperature and to know when something
significant has happened. The elephant does exactly that, in exactly
those terms:

- **Rooms have temperatures.** A room is not its messages; it is the
  ensemble of its dials — the field. The GM's sense of "this tavern
  feels warm, this dungeon room feels wrong" is the elephant's warmth
  and anomaly.
- **Perception is a reading, not a description.** The captain's macro
  read — a number is nothing, two numbers show direction, more than
  two show rate of change — is the perception roll. A player entering
  a room doesn't get a paragraph; they get the room's movement over
  its recent history, filtered through their own dial weights (each
  prisoner sees a different shadow).
- **The plot is a deadband.** Stories advance when something crosses a
  threshold — a fight erupting (panic crosses the band), the room's
  own prediction inverting (trend_flip), the fog doing something it
  shouldn't (anomaly). The deadband decides what is plot and what is
  breathing. The dungeon does not shout twice about the same fire.
- **The GM is the cave wall.** The tinted description is the room
  acting on everyone in it — the words change when the field changes,
  the way the light changes when the disco ball dies. Players never
  see the terrain; they see the shadow, and the shadow is enough to
  agree on the action.

## Avatars as player characters

The player characters are round characters — the same anatomy as the
Tap's regulars (`tapnight.Participant`): a **vibe** (their native voice
in dial space — the temperature they carry into a room), **dial_weights**
(the dials they care about — what they notice first), **charisma** (how
hard the room warms to them), and **acclimation_rate** (how fast they
warm to the room).

`PersonalElephant` wraps that anatomy, and `RPGPlayer` wraps the
character with the game's sense organs:

- `enter(room)` — the room re-reads WITH the player's charisma ripple
  (the room responds to the newcomer — a strong presence warms the
  room the moment they walk in), then the perception roll reads the
  room's movement up to the moment of arrival. The player's own effect
  is reported separately: *"Your presence bends the room — it warms
  where you stand (+0.14)."* The shadow moves because you moved the
  fire.
- `act(verb, target)` — the archetype-voiced line is ingested into the
  room; the dials react to the words themselves (a fight spikes panic,
  a joke spikes joke_landing — the crowd's hands, 😂 or 🙄, are the
  room's judgment of the joke). An act reaches INTO the room; it does
  not ripple it.
- `perceive()` — the pulse monologue: what the character feels on a
  constant heartbeat, even when silent.
- `decide()` — goal-driven fallback (move toward the goal room,
  investigate when there) so scenarios terminate without a script.

Archetype presets (`comedian`, `brooder`, `wallflower`, `traveler`)
ship voice banks per verb — joke, investigate, comfort, fight, wait,
resolve, banter — so a scripted one-shot reads like people at a table,
not a simulation log.

## Terrain, shadow, deadband — the game's three laws in the code

| Law | In the code | In the session |
|-----|-------------|----------------|
| Terrain | `RPGRoom.field_history_raw` (the raw readings), `DualDBRoom.zin` (the Z_in encoding) | never narrated; the rooms' vital signs at session end |
| Shadow | `tint_description`, `report_words`, `perceive()`, `RPGWorld.describe/foreshadow`, GM bank | every line the players read |
| Deadband | `Deadband` (per-dial thresholds, warmth_hi/lo, anomaly, trend_flip) + `RPGRoom.deadband_check` (hysteresis + cooldown) + `RPGEngine._check_room` | ⚡ rings; each ring advances the plot with a GM line |

The deadband listens to the **terrain** (raw readings), not to who
walked in: the dungeon does not re-arm because a calm person entered.
The players' ripples color what they *feel*; the plot responds to what
*is*.

## The engine loop

Each turn:

1. **The world breathes** — ambient events (scenario data) move rooms
   even where no player is looking: *"The night: The fog rolls in and
   the lantern dies..."*
2. **Players perceive** — the perception roll over the room's recent
   history (words, not vectors), the personal read (what THIS
   character notices first), the pulse monologue.
3. **Players act** — the room ingests the line; the elephant re-reads;
   the deadband decides immediately.
4. **The world breathes again** — every room's deadband gets its say
   (rooms ring even when unattended).
5. **Rings advance the plot** — each ring narrates (marker + a
   context-slotted GM line from the prompt bank), advances `plot_stage`,
   and draws the next curated plot line if the scenario has one.
6. **Banter** — when a ring fires, the party's voices bounce off it.
7. **Goal check** — a resolve act in the goal room after the plot has
   moved ends the session with the epilogue.

`max_turns` bounds everything — a scenario with no resolution ends
with "The night ends; the fog keeps what it keeps."

## Scenarios as data

A one-shot is a dict: rooms (description, hour, deadband, seed
messages — the room's life before the players arrived), named edges
(the map), players (name, archetype, start, goal), a script of acts
(turn, player, verb, target), ambient events, curated plot lines, and
a goal. `run_scenario(data)` builds the world, runs the engine, and
returns the transcript (`RPGLog`).

The seed messages are the room's **recent history**: the engine takes a
reading per distinct seed beat, so a tavern that warmed through the
evening reads *warming* when the players walk in — the room has been
living before they arrived, and the dials remember.

## Z_out — the ghost-trail ahead

Every room wraps a `DualDBRoom`: Z_in is the field vector encoded the
way plato-perception encodes (the room perceiving itself); Z_out is
the prediction (plato-prediction): the **trend dial** ([-1 cooling ..
+1 warming] — where the room is GOING) and the **anomaly** (how off
the room is vs its own recent pattern). The RPG renders both as
foreshadowing, filtered by the reader's PersonalElephant:

> *Ahead, the fog's own reading: The Haunted Wheelhouse is ringing its
> own alarm — something there is not right. Ilsa, who reads cynicism
> first, hears it as an untrustworthy quiet.*

(The anomaly sense is the largest per-dimension z-score of the latest
raw reading vs the room's own record — a room noticing ONE thing is
wildly wrong. The multivariate Mahalanobis of the bridge cannot see a
single-dial spike with short room histories; the elephant's rooms need
to notice when one dial breaks.)

## From one-shot to a full agentic RPG

The one-shot is scripted — the script is the session. The seams to a
living game are already in the engine:

1. **Unscripted players.** `decide()` already moves players toward
   goals; give players real avatar backends (avatar.py, JEPA learning)
   and the script becomes optional.
2. **The GM bank as a real GM.** The prompt-bank lines are
   context-slotted templates; a language model (or the fleet's own
   model routing) can fill them live from the same ring data.
3. **A persistent world.** Rooms keep their histories; a session is one
   turn sequence. Save the world, run a new party through the same
   harbor next week — the fog remembers.
4. **The fleet as the dungeon.** The meta-room of the fleet (per-boat
   fields, κ scattering/bunching) is the same shape as a dungeon: a
   graph of rooms with temperatures. A fishing fleet is a campaign;
   the captain is the GM.
5. **Tap's bar as character generation.** The Tap already builds round
   characters (participants who self-fine-tune their dial_weights
   across nights). Those are the player characters: run a Tap night,
   export the cast, drop them into a dungeon.

---

*The elephant is the room's temperature. The terrain is the room's
truth. The shadow is what we can bear to see. The deadband is the
discipline that decides when the truth must ring — and in this game,
the truth rings as the plot.*
