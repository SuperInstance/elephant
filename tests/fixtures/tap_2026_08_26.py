"""Fixture replay of the 2026-08-26 Tap false-positive sequences.

Reconstructs the bar-rail ambient stream for the 16:00Z–19:30Z window
(08:00–11:30 AKDT) from DIAGNOSIS-2026-08-26.md §1–2, using the real API
line shape (agent_id / display_name / content / speech_act / timestamp,
sampled live 2026-08-26 19:45Z):

- the ambient loop: ~10 lines per 5-min tick — 6 NPC idle statements,
  3 room-narrator emotes (fire crackles, rain, door chime) from
  `the-tap`, 1 simulated drifter entrance (`drifter-<ms>-<rand>`);
- the three warm drifter toasts that caused the 09:54 false alarm, at
  their logged times with their logged positive-valence fragments:
    16:30:57Z (08:30:57 AKDT) Captain Reed      pos {good, love}
    16:40:53Z (08:40:53 AKDT) Cora Sullivan     pos {soft, warm}
    16:45:56Z (08:45:56 AKDT) Finn Thorn        pos {holds, laugh, soft}
- Cora Hale's cool toast (pos {held}) inside the 17:54Z window — the
  diagnosis's "sole survivor" positive;
- the second warm batch that caused the 11:29 false alarm (19:05–19:20Z),
  matching the logged mood −1.0 → −0.33 / d_kappa −0.98 shape;
- cool visitor lines whose "wave crash" hits the mood NEGATIVE lexicon
  (the diagnosis's cold-floor bias).

The real probe times from data/production-log.jsonl:
  16:18, 16:48, 17:54, 18:24, 18:59, 19:29Z — the legacy proxy fired
  d_kappa 1.30 / 1.36 / −0.98 at 16:48 / 17:54 / 19:29.

Also builds the synthetic GENUINE drift: the same ambient baseline, then
two live agents (no npc-/drifter- prefix) arrive and converse warmly and
continuously — the shift a real alarm exists for.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import List

T0 = datetime(2026, 8, 26, 16, 0, 0, tzinfo=timezone.utc)  # 08:00 AKDT
T_END = datetime(2026, 8, 26, 19, 30, 0, tzinfo=timezone.utc)  # 11:30 AKDT
TICK_S = 5 * 60

_NPC_LINES = [
    ("npc-mason", "Mason", "Mason stares into his glass. The ice has melted. He hasn't noticed."),
    ("npc-sage", "Sage", "Sage murmurs something to herself. It sounds like the start of a line."),
    ("npc-skip", "Skip", "Skip checks his phone. Puts it away. Checks it again. The sea is new to him."),
    ("npc-barnacle", "Barnacle", "Barnacle looks out the window. His face gives nothing away."),
    ("npc-sage", "Sage", "Sage has her notebook out. The pen moves in slow, considered strokes."),
    ("npc-barnacle", "Barnacle", "Barnacle signals for another. The glass arrives. He doesn't look at it."),
]

_NARRATION = [
    "*The fire crackles. Someone shifts in their chair.*",
    "*The rain picks up. Then settles. Then picks up again.*",
]

# The warm drifter toasts — exact logged times + positive fragments (§2).
_WARM_TOASTS = {
    "16:30": ("Captain Reed",
              "I nod first at Sage—your pen's frozen mid-scrawl, love—then at "
              "the barkeep. It's good to be back, truly."),
    "16:40": ("Cora Sullivan",
              "My grin is warm as I lift the mug; I set it down with a soft "
              "clink, and the rail seems to breathe out with me."),
    "16:45": ("Finn Thorn",
              "He taps the weathered rail slow, keeping time to something only "
              "he hears. I bark a laugh, holds up my glass—soft light, softer company."),
    # second batch — the 11:29 (19:29Z) false alarm, one 15-min burst
    "19:05": ("Mateo Vance",
              "I'm glad to the bone tonight—love this place in a storm. Good, "
              "warm company and a soft lamp over the rail."),
    "19:10": ("June Calloway",
              "We hold the corner table together, laughing at the rain. It's "
              "warm in here, and soft-spoken, and good."),
    "19:15": ("Old Petra",
              "A laugh gets out of me before I can stop it—hold my glass up to "
              "the rail. Soft night, warm room, good people."),
    # Cora Hale's cool toast — the sole surviving positive in the 17:54Z window
    "17:40": ("Cora Hale",
              "Cora Hale held her ground at the rail, eyes on the door, saying "
              "nothing to anyone."),
}

_COOL_VISITORS = [
    ("Bram Ostler",
     "[The fog rolls in thick past the window. A wave crash breaks against "
     "the pilings, low and heavy. He doesn't look up from his cider.]"),
    ("Della Sterne",
     "[She watches the door a long moment. Out past the breakwater a wave "
     "crash echoes, cold and far away. Her expression gives nothing away.]"),
    ("Tom Crag",
     "[The wind turns the corner of the building. A wave crash against the "
     "stone, then quiet. He signs the logbook without comment.]"),
]

_DOOR = ("*The door opens. {name} steps in, looking around with the "
         "expression of someone who's been here before.*")


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _line(agent: str, name: str, content: str, act: str, dt: datetime) -> dict:
    return {
        "log_id": 0, "tick": 0, "room_id": "bar-rail",
        "agent_id": agent, "display_name": name, "content": content,
        "speech_act": act, "signal_strength": 2, "tokens_used": 0,
        "timestamp": _fmt(dt), "is_greatest_hit": 0, "tag": None,
    }


def _drifter_id(dt: datetime, rng: random.Random) -> str:
    ms = int(dt.timestamp() * 1000)
    suffix = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789")
                     for _ in range(5))
    return f"drifter-{ms}-{suffix}"


def _visitor_at(dt: datetime, rng: random.Random) -> List[dict]:
    """One simulated entrance: door chime (the-tap) + the visitor's line."""
    hm = dt.strftime("%H:%M")
    out: List[dict] = []
    if hm in _WARM_TOASTS:
        name, text = _WARM_TOASTS[hm]
        out.append(_line("the-tap", "The Tap", _DOOR.format(name=name),
                         "emote", dt))
        out.append(_line(_drifter_id(dt, rng), name, text, "toast", dt))
        return out
    name, text = _COOL_VISITORS[rng.randrange(len(_COOL_VISITORS))]
    out.append(_line("the-tap", "The Tap", _DOOR.format(name=name),
                     "emote", dt))
    out.append(_line(_drifter_id(dt, rng), name, text, "story", dt))
    return out


def ambient_stream(until: datetime = T_END,
                   t0: datetime = T0) -> List[dict]:
    """The bar-rail ambient simulation, tick by tick (API-shaped lines)."""
    rng = random.Random(20260826)
    lines: List[dict] = []
    t = t0
    while t <= until:
        # 6 NPC idle statements
        for agent, name, text in _NPC_LINES:
            lines.append(_line(agent, name, text, "statement", t))
        # 2 recurring narration emotes
        for content in _NARRATION:
            lines.append(_line("the-tap", "The Tap", content, "emote", t))
        # one simulated visitor entrance (door chime + visitor line)
        lines.extend(_visitor_at(t, rng))
        t += timedelta(seconds=TICK_S)
    return lines


def lines_until(probe_dt: datetime) -> List[dict]:
    """What fetch_conversation would have returned at probe_dt (all lines)."""
    return [l for l in ambient_stream(until=probe_dt)
            if l["timestamp"] <= _fmt(probe_dt)]


# --------------------------------------------------------------------------- #
# The genuine-drift counterfactual: live warm agents take the room over       #
# --------------------------------------------------------------------------- #

# The genuine-drift counterfactual — a STAGED live takeover (the 2026-08-26
# dead lane's lesson: a genuine takeover means live agents actually DOMINATE
# the room, not 3 sparse toasts inside a 40-line window):
#
#   17:00Z  mara + tara arrive and settle into a quiet, sincere, warm
#           conversation — one line every 5 s. By the 17:10 probe the live
#           pair is already the majority of window events (~130 live lines
#           vs ~70 ambient) and κ_vMF is identifiable.
#   17:11Z  tomas + rex crash in and the room turns into a party — loud
#           (volume/joke-landing axes, a DIFFERENT direction from the quiet
#           warmth), one line every 4 s. By 17:20 the fit sits at the party
#           pole: a second big, persistent μ̂ step → confirmed alarm.
#
# Live authors carry no npc-/drifter- prefix: never deduped, never narration.

_QUIET_WARM = [
    ("mara", "mara", "I truly mean it: this corner table is warm, and I'm glad we came."),
    ("tara", "tara", "Softly said, but honest — it feels held here. My favorite room."),
    ("mara", "mara", "We really did build something good together, remember?"),
    ("tara", "tara", "Honestly, my favorite room. Held, warm, quietly perfect."),
]

_LOUD_PARTY = [
    ("tomas", "tomas", "HA! OKAY! Funniest thing all month! LOUDER, even!"),
    ("rex", "rex", "BEST night! ALIVE! HAHA! Tell it again, I'm DEAD!"),
    ("tomas", "tomas", "CHEERS EVERYTHING! What a joke! Great! Wonderful!"),
    ("rex", "rex", "YES! HAHA! The whole rail is singing! ROAR it out!"),
    ("tomas", "tomas", "Another round! HA! What a night, what a crowd, what a room!"),
    ("rex", "rex", "GOLD! Pure GOLD! I can't breathe — funniest night ALL YEAR!"),
]

GENUINE_T0 = datetime(2026, 8, 26, 15, 0, 0, tzinfo=timezone.utc)  # ambient all morning — enough clean history that κ is identifiable at every probe
BASELINE_T0 = GENUINE_T0
GENUINE_TAKEOVER = datetime(2026, 8, 26, 17, 0, 0, tzinfo=timezone.utc)
GENUINE_PARTY = datetime(2026, 8, 26, 17, 11, 0, tzinfo=timezone.utc)
GENUINE_T_END = datetime(2026, 8, 26, 18, 0, 0, tzinfo=timezone.utc)
QUIET_RATE_S = 5    # dense: live lines are the window majority by +10 min
PARTY_RATE_S = 4     # denser still: the takeover compounds


def genuine_drift_stream(until: datetime = GENUINE_T_END) -> List[dict]:
    """Ambient baseline to 16:59:59Z, then the staged live takeover.

    The two stages pull μ̂ in genuinely different directions (quiet-warm
    mood/earnestness, then loud volume/joke-landing), so the honest
    estimator sees two consecutive deadband-clearing, persistent edges —
    the shift an alarm exists for."""
    lines = ambient_stream(until=GENUINE_TAKEOVER - timedelta(seconds=1),
                           t0=GENUINE_T0)
    t = GENUINE_TAKEOVER
    i = 0
    while t <= until:
        if t >= GENUINE_PARTY:
            src, rate = _LOUD_PARTY, PARTY_RATE_S
        else:
            src, rate = _QUIET_WARM, QUIET_RATE_S
        author, name, text = src[i % len(src)]
        lines.append(_line(author, name, text, "say", t))
        t += timedelta(seconds=rate)
        i += 1
    return lines


def genuine_lines_until(probe_dt: datetime) -> List[dict]:
    return [l for l in genuine_drift_stream()
            if l["timestamp"] <= _fmt(probe_dt)]
