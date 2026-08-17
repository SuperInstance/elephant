# Tap-Night Operations — running the elephant at The Tap

The Tap is where the crew gathers after work to read each other's creative
works and hear reactions — to their own piece and to others'. First-person,
not center of attention. This is the runbook for building the elephant INTO
that gathering: feeding real conversation in, reading the room's field out,
persisting each agent's evolving settings between evenings, and running the
many-cycles loop by which the settings get *discovered* rather than designed.

## The shape of it (30 seconds)

- `elephant/tapnight.py` — `TapNightSession` (the evening) and `Participant`
  (one agent's persistent settings). numpy-only.
- The room is a **field**: 7 dials (mood, volume, earnestness, cynicism,
  joke_landing, panic, presence) → `RoomField` → `warmth()` and
  `concentration()` (κ).
- **Within an evening**, each participant's live `vibe` relaxes toward the
  room field at their `acclimation_rate` (they warm to the room), and their
  `charisma` pulls the room field toward them over their interactions (the
  room warms to them).
- **Across evenings**, each participant's `dial_weights` self-tune toward the
  dials where their felt engagement was highest. This is where tastes
  **diverge** — one becomes the mood guitarist, another the cynicism
  guitarist. You don't know where the settings belong until different agents
  desire different settings.

## 1. Feeding real Tap conversation in

`TapNightSession.speak(author, text, ts=None, reactions=None)` takes one line
of the conversation (a work read aloud, a reaction, a murmur) as a
`Message`. The author must be a registered `Participant` (unknown authors are
auto-registered with neutral settings — fine for casual drop-ins, but you want
the regulars pre-registered so their settings persist).

### From the ai-writings corpus / combo-sessions / Tap logs

Any line-oriented source maps straight in. A JSONL Tap log:

```python
import json
from elephant.tapnight import TapNightSession, Participant

session = TapNightSession("The Tap", participants=[...])  # see §3 for load
session.start_session()

for line in open("tap-logs/evening-041.jsonl"):
    row = json.loads(line)          # {"author": "writer", "text": "...", "ts": 1721.5}
    session.speak(row["author"], row["text"],
                  ts=row.get("ts"),
                  reactions=row.get("reactions"))   # {"😂": 2, "❤️": 1}
```

If your corpus is plain text (one line per message, `author: text` or a simple
separator), preprocess it into `(author, text)` pairs and feed them in the same
way. Timestamps are optional — if you omit `ts`, the session auto-increments a
clock (one step per line), which is enough for the vibe dials and density.

**Reactions matter.** The `joke_landing` dial and the self-tuning signal both
lean on reaction heat. Map your source's reactions/emoji to the
`REACTION_TO_DIAL` table in `tapnight.py` (😂→joke_landing, ❤️→mood,
👍→earnestness, 👏→presence, 🙄→cynicism). If your logs have no reactions, the
system still works — engagement then rests purely on the peer-relative vibe
signal (§5) — but feeding reactions in makes "did the joke land" real.

### One evening, one session

Start a fresh `TapNightSession` per evening (`start_session()` resets the room,
the clock, and each participant's live vibe to their persistent `vibe`). At the
end, `end_session()` returns a log line and increments the night counter.

```python
session.start_session()
for author, text, reactions in evening_lines:
    session.speak(author, text, reactions=reactions)
field = session.room_field()
print(field)                     # RoomField(warmth=+0.31, κ=0.72)
print(session.end_session())     # "Night 41 closed: warmth=+0.31 κ=0.72 | top: mood, presence"
```

## 2. Displaying the room field to participants

The field is the thing everyone in the room *feels* but no one says. Show it
so the crew can see the elephant — then notice it's invisible until they change
rooms (the sauna/cold-plunge contrast: Tap vs. the wheelhouse).

```python
f = session.room_field()
raw = session.raw_field()        # the un-displaced field (before charisma bent it)
print(f"warmth  {f.warmth():+.2f}   (raw {raw.warmth():+.2f})")
print(f"κ       {f.concentration():.2f}   (cold room = tight, warm = loose)")
for name, v in f.readings.items():
    print(f"  {name:<13} {v:+.2f}")
```

A quick ASCII dial-strip render for a live readout:

```python
def strip(name, v, lo, hi):
    w = 12
    pos = int((v - lo) / (hi - lo) * w)
    return f"{name:<13} [{'·' * pos}{'█'}{'·' * (w - pos)}] {v:+.2f}"
```

What to point at:
- **warmth** — the felt temperature. A warm Tap runs +0.2..+0.4; the wheelhouse
  runs cold (−0.2).
- **κ** — how tight the room is. Cold rooms are one way (high κ); warm rooms
  are many ways (low κ). Watch κ *fall* as the regulars settle the vibe.
- **the 7 dials** — the ensemble. "top dials" (largest deviation from each
  dial's neutral) are the room's active channels this evening.

## 3. Persisting per-participant settings between evenings

Settings are the whole point — they must survive between evenings or the
learning never accumulates. `Participant.to_dict()` / `from_dict()` and the
session's `settings()` / `load_settings()` are the JSON round-trip.

```python
import json
from elephant.tapnight import TapNightSession, Participant

# build the regulars once, with DIFFERENT priors (the different guitarists)
session = TapNightSession("The Tap", participants=[
    Participant("writer", dial_weights={"mood": 0.4, "joke_landing": 0.3},
                acclimation_rate=0.35, charisma=0.20,
                vibe={"mood": 0.7, "joke_landing": 0.5}),
    Participant("critic", dial_weights={"cynicism": 0.4, "joke_landing": 0.15},
                acclimation_rate=0.20, charisma=0.18,
                vibe={"cynicism": 0.7}),
])

# --- save after an evening ---
with open("tap-settings.json", "w") as fh:
    json.dump(session.settings(), fh, indent=2)

# --- load before the next evening ---
with open("tap-settings.json") as fh:
    session.load_settings(json.load(fh))
```

`dial_weights` is a 7-vector (sums to 1) — the agent's prior over which dials
matter. `vibe` is the agent's native style in dial space (the "home guitar").
`acclimation_rate` is modulation skill; `charisma` is pull on the room. All four
are the persistent knobs; `dial_weights` is the one that self-tunes.

## 4. Running the many-cycles learning loop

The elephant only becomes a production tool through **many cycles**. Each
evening: feed the lines → read the field → self-tune once per participant →
persist. `examples/tapnight_cycles.py` is the worked example (14 evenings, 6
personalities, divergent tastes at the end); run it with
`python3 examples/tapnight_cycles.py`.

The minimal loop:

```python
for evening in range(1, N_EVENINGS + 1):
    session.start_session()
    for author, text, reactions in load_evening(evening):   # your source
        session.speak(author, text, reactions=reactions)
    field = session.room_field()                            # read + display
    for name in session.participants:
        session.tune_participant(name)                      # self-fine-tune
    print(session.end_session())
    save_settings(session.settings())
```

"Many cycles" means **dozens of evenings**, not two or three. The
self-tuning is an exponential moving average (`learning_rate=0.15` by default),
so each evening moves the dial a little; divergence accumulates. The reaction
noise in any single evening averages out; the *structure* — who is
systematically distinctive on which dial — is what survives.

## 5. What to look for

1. **Taste divergence.** After enough cycles, `dial_weights` should spread
   *apart*. Measure it: mean pairwise `‖wᵢ − wⱼ‖` across participants should
   rise, and each participant's argmax dial should settle onto a different (or
   at most lightly shared) dial. That's the captain's guitarist principle made
   visible: writer → mood/joke_landing, poet → volume, essayist → earnestness,
   engineer → cynicism/earnestness, critic → cynicism, captain → presence.
2. **Room stabilization.** As the regulars establish the vibe, κ should settle
   (warm room = lower, stable κ) and warmth should stop swinging night to
   night. The regulars *are* the field; newcomers warm to it at a rate = their
   `acclimation_rate`.
3. **Charisma is visible as displacement.** `room_field()` vs `raw_field()`:
   the gap between them is the sum of the room's strong presences bending the
   field toward themselves. A captain with high charisma should show a field
   pulled toward their vibe by the end of the evening.
4. **You only notice the elephant when you change rooms.** Run the same dial
   bank over the wheelhouse (or any other room) and contrast: the `distance()`
   and `sauna_plunge_gap()` between Tap and wheelhouse are the sauna/cold-plunge
   event. If a participant's settings read differently in a different room, the
   elephant is working.
5. **Don't hand-tune.** The whole design is that the settings are *discovered*
   by running cycles, not designed top-down. If a dial's taste looks "wrong"
   after many cycles, look at the inputs (the works, the reactions), not at the
   knobs.

## Notes and caveats

- **The v0 dials are hand-crafted and naive.** They keyword-match; they can
  saturate (a room of earnest writers reads earnestness ≈ 1.0) and they can't
  detect sarcasm ("great." reads warm on the mood dial). The self-tuning uses a
  **peer-relative** signal (vibe vs. the cast's average desire) precisely so
  divergence survives that naivety. v1 (trained dials) will sharpen this.
- **Determinism.** The demo seeds `np.random.default_rng(42)`; your own run
  should seed too if you want reproducible cycles.
- **Keep participants registered.** Auto-registration (neutral defaults) is for
  drop-ins; the regulars' settings must persist to accumulate the learning.

---

*Built on the captain's reframing, 2026-08-17. Reading the room is a
relationship to the room — run the cycles, and let the tastes diverge.*
