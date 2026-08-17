"""Avatar — a round character in the making.

A character starts FLAT: a `persona_prompt` seed identity ("I am
flat.") plus a distinct initial dial-weight preset — the comedian, the
brooder, the wallflower. Flat characters are stamped from a mold;
round ones are grown. This module grows them the only way this repo
knows how: by sending them to The Tap.

An `Avatar` attends themed Tap nights. It speaks lines; its
`PersonalElephant` senses the room's field through its own taste and
disposition (the same deformation `PersonalElephant.read` applies, so
the avatar's read stays comparable to the room's own); it self-tunes
its `dial_weights` toward felt engagement by REUSING
`TapNightSession`'s tuning math — settings are discovered, not
designed; it records `attachments` — the moments that meant something,
bound on the elephant as intangible correlations (the perfume that
takes you to grandma's shop, the song that is the lover you
discovered the album with); and it enriches its persona with one-line
character notes distilled from each night.

The proof of roundness is the GUITARIST PRINCIPLE: two avatars with
different presets attending the SAME night end with DIFFERENT
profiles. One guitar for the looks, one for the sound, one for the
neck — tastes diverge under an identical room because the engagement
signal is anchored to each participant's own vibe against the cast's
average. They are not caricatures stamped from one mold.

Flat seed in, learned roundness out: `speak()` composes what the
character would say NOW (the seed still speaking, the tuned
sensitivities, a remembered attachment, the arc); `monologue()` runs
the silent pulse — what it feels but does not say, because silence is
not empty; `character_sheet()` and `through_line` show the drift from
the flat seed, which is the learning.

numpy-only and fully deterministic — no RNG anywhere. Roundness is
learned, not rolled.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from elephant.field import DIAL_NAMES, RoomField
from elephant.presets import PersonalElephant
from elephant.pulse import PulseLoop
from elephant.tapnight import DIAL_BOUNDS, DIAL_CENTER, Participant

__all__ = ["Avatar", "PRESETS", "FELT_PHRASES", "SENSITIVITY_PHRASES", "ARCHETYPES"]

# A moment "meant something" when the room moved this much in the avatar's
# subjective read, or when the crowd put at least this many hands on the line.
MOMENT_MOVE_FLOOR = 0.02
# When the room's biggest mover is below this, the attachment memory anchors
# on what the avatar FELT (its distinctive dial) instead of the room's.
MEMORY_FELT_FLOOR = 0.05


# Starter archetypes: the flat seeds. Each is a distinct prior — the taste
# the avatar walks in with — so identical rooms can grow different people.
PRESETS: Dict[str, Dict[str, Dict[str, float]]] = {
    # leans joke_landing + mood — works the laugh.
    "comedian": {
        "dial_weights": {
            "joke_landing": 0.40, "mood": 0.25, "presence": 0.10,
            "earnestness": 0.10, "volume": 0.05, "cynicism": 0.05,
            "panic": 0.05,
        },
        "bias": {"joke_landing": 0.15, "mood": 0.10},
        "vibe": {"joke_landing": 0.70, "mood": 0.55, "presence": 0.55},
    },
    # leans mood + panic + presence — sits with the heavy things.
    "brooder": {
        "dial_weights": {
            "mood": 0.35, "presence": 0.20, "panic": 0.15,
            "earnestness": 0.10, "cynicism": 0.10, "volume": 0.05,
            "joke_landing": 0.05,
        },
        "bias": {"mood": -0.15, "panic": 0.05},
        "vibe": {"mood": -0.20, "panic": 0.55, "presence": 0.60},
    },
    # leans presence — the shy one cares about presence but emits little.
    "wallflower": {
        "dial_weights": {
            "presence": 0.35, "panic": 0.20, "mood": 0.15,
            "earnestness": 0.15, "volume": 0.05, "cynicism": 0.05,
            "joke_landing": 0.05,
        },
        "bias": {"presence": -0.15, "volume": -0.10, "panic": 0.10},
        "vibe": {"presence": 0.45, "panic": 0.40, "mood": 0.45},
    },
}

# What a dial's felt engagement becomes in an attachment memory.
FELT_PHRASES: Dict[str, str] = {
    "joke_landing": "a joke that actually landed",
    "mood": "the room's temperature turned with me",
    "earnestness": "everyone meant it, and I meant it too",
    "cynicism": "the sneer I recognized as my own",
    "panic": "the fear I carry showed up for a second",
    "presence": "for a moment I was seen",
    "volume": "the room got loud and I was in it",
}

# What a tuned sensitivity sounds like in the character's own mouth.
SENSITIVITY_PHRASES: Dict[str, str] = {
    "joke_landing": "I've learned to listen for whether the joke lands "
                    "before I commit to the laugh",
    "mood": "I feel the room's temperature before I decide what to say",
    "earnestness": "I mean what I say — the room taught me that",
    "cynicism": "I watch for the sneer under the sweet words",
    "panic": "I carry the fear in the room, even when nobody names it",
    "presence": "I notice who's here — and who's about to leave",
    "volume": "I hear the room when it gets loud",
}

# The archetype each dial's lean becomes in the through-line.
ARCHETYPES: Dict[str, str] = {
    "joke_landing": "the one waiting for the laugh",
    "mood": "the one who feels the room's temperature",
    "presence": "the one who sees everyone at the table",
    "earnestness": "the one who means it all the way down",
    "cynicism": "the skeptic, the sneer seen plain",
    "panic": "the one who holds the fear",
    "volume": "the one who hears the room when it roars",
}


def _to_vector(x, default=0.0) -> np.ndarray:
    """Coerce a dial-space value into a 7-vector in DIAL_NAMES order.

    Accepts a dict {dial_name: value} (missing keys -> `default`, which
    may be a scalar or a per-dial dict), or any array-like of 7 floats.
    Mirrors presets.py's `_to_vector`.
    """
    if x is None:
        x = default
    if isinstance(x, dict):
        if isinstance(default, dict):
            return np.array([float(x.get(n, default[n])) for n in DIAL_NAMES],
                            dtype=float)
        return np.array([float(x.get(n, default)) for n in DIAL_NAMES],
                        dtype=float)
    if np.isscalar(x):
        return np.full(len(DIAL_NAMES), float(x), dtype=float)
    return np.asarray(list(x), dtype=float)


def _clamp(vec: np.ndarray) -> np.ndarray:
    """Clamp each dial to its bounds (mirrors presets.py's `_clamp`)."""
    out = vec.copy()
    for i, n in enumerate(DIAL_NAMES):
        lo, hi = DIAL_BOUNDS[n]
        out[i] = min(hi, max(lo, out[i]))
    return out


def _top_dials(field_vec: np.ndarray, k: int) -> List[str]:
    """The `k` dial names most deviated from neutral in a field vector
    (ties broken in DIAL_NAMES order — deterministic)."""
    dev = [(n, abs(float(field_vec[i]) - DIAL_CENTER[n]))
           for i, n in enumerate(DIAL_NAMES)]
    dev.sort(key=lambda t: -t[1])
    return [n for n, _ in dev[:k]]


class Avatar:
    """A round character in the making — a flat seed that learns its
    roundness at The Tap.

    The avatar carries a `PersonalElephant` (its subjective instrument:
    taste, disposition, attachments) and a stable `vibe` — its native
    style in dial space, its "home voice", used when it registers as a
    tapnight `Participant`. The vibe stays stable while the
    `dial_weights` learn; what changes across nights is what the avatar
    listens FOR, which is exactly what `TapNightSession`'s self-tuning
    moves.

    Two avatars with different presets attending the same night end
    with different profiles (the guitarist principle): the flat seed is
    the mold, the nights are what breaks it.
    """

    def __init__(self, name: str, persona_prompt: str,
                 elephant: Optional[PersonalElephant] = None,
                 preset: Optional[str] = None,
                 vibe: Optional[dict] = None):
        """Start flat: a seed identity, a preset prior (or a given
        elephant), and a home voice."""
        cfg = PRESETS.get(preset, {})
        if elephant is None:
            elephant = PersonalElephant(name,
                                        dial_weights=cfg.get("dial_weights"),
                                        bias=cfg.get("bias"))
        self.name = name
        self.persona_prompt = persona_prompt    # the flat seed
        self.elephant = elephant
        self._vibe = _to_vector(vibe or cfg.get("vibe", {}),
                                default=DIAL_CENTER)
        self.character_notes: List[str] = []    # one line per night
        self.nights: List[dict] = []            # night summaries
        self.monologue_log: List[dict] = []     # the silent pulse, logged
        self._started_with = self.elephant.dial_weights.copy()
        self._last_session = None
        self._pulse = None
        self._pulse_room = None
        self._speak_cursor = 0

    def __repr__(self) -> str:
        return (f"<Avatar {self.name!r}: {len(self.nights)} "
                f"night{'s' if len(self.nights) != 1 else ''}, "
                f"{len(self.elephant.attachments)} attachments>")

    # ------------------------------------------------------------------ #
    # Properties                                                         #
    # ------------------------------------------------------------------ #
    @property
    def dial_weights(self) -> Dict[str, float]:
        """The elephant's current dial weights — what the avatar listens
        for, as a dict in dial space."""
        return {n: float(self.elephant.dial_weights[i])
                for i, n in enumerate(DIAL_NAMES)}

    @property
    def persona(self) -> str:
        """The CURRENT persona: the flat seed enriched by one character
        note per night attended. With no notes, still just the seed."""
        if not self.character_notes:
            return self.persona_prompt
        return self.persona_prompt + "\n" + "\n".join(self.character_notes)

    @property
    def through_line(self) -> str:
        """The arc in one sentence: who walked in, what they lean toward
        now, and the moment they keep."""
        seed_first_sentence = self.persona_prompt.split(".")[0].strip()
        top_dial = DIAL_NAMES[int(np.argmax(self.elephant.dial_weights))]
        if self.elephant.attachments:
            last = list(self.elephant.attachments.values())[-1]
            snippet = str(last.get("line", "")).rstrip()
            if len(snippet) > 48:
                snippet = snippet[:48].rstrip()
            attachment_summary = f"the moment when {snippet!r}"
        else:
            attachment_summary = "no attachments yet — still flat"
        n = len(self.nights)
        nights = "night" if n == 1 else "nights"
        return (f"{self.name} walked in as {seed_first_sentence}. After "
                f"{n} {nights} at The Tap, {self.name} leans {top_dial} — "
                f"{ARCHETYPES.get(top_dial, top_dial)} — and keeps "
                f"{attachment_summary}.")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _ensure_participant(self, session) -> None:
        """Make sure the avatar is registered on the session with its
        CURRENT settings, and that the session's live `_vibe` has an
        entry (a missing one raises KeyError in `speak()`)."""
        if self.name not in session.participants:
            session.participants[self.name] = Participant(
                self.name,
                dial_weights=self.elephant.dial_weights.copy(),
                vibe=self._vibe,
                acclimation_rate=0.3,
                charisma=0.15)
            session._vibe[self.name] = self._vibe.copy()
            session._vibe_start[self.name] = self._vibe.copy()
        else:
            participant = session.participants[self.name]
            participant.dial_weights = self.elephant.dial_weights.copy()
            participant.vibe = self._vibe
            session._vibe[self.name] = self._vibe.copy()

    def _sense(self, session) -> RoomField:
        """The avatar's SUBJECTIVE read of the room right now: the
        session's effective (charisma-displaced) field deformed exactly
        like `PersonalElephant.read` — taste weights the deviation from
        neutral, disposition shifts it, bounds clamp it."""
        center = np.array([DIAL_CENTER[n] for n in DIAL_NAMES], dtype=float)
        base = RoomField(dict(zip(DIAL_NAMES, session.field))).vector()
        delta = base - center
        subj = (center + (self.elephant.dial_weights * len(DIAL_NAMES))
                * delta + self.elephant.bias)
        return RoomField(dict(zip(DIAL_NAMES, _clamp(subj))))

    def _distill_note(self, session, eng: np.ndarray) -> str:
        """One line per night: how the room ran, and what the avatar
        kept listening for (its top positively-felt dial)."""
        warm = session.room_field().warmth()
        top = _top_dials(session.field, 2)
        top1 = top[0]
        top2 = top[1] if len(top) > 1 else top[0]
        pos = np.maximum(np.asarray(eng, dtype=float), 0.0)
        if float(pos.max()) > 1e-9:
            engaged_dial = DIAL_NAMES[int(np.argmax(pos))]
            felt_phrase = FELT_PHRASES.get(engaged_dial,
                                           "something I can't name yet")
            listening = f"{engaged_dial} — {felt_phrase}"
        else:
            listening = "something I can't name yet"
        return (f"{session.name}: the room ran {warm:+0.2f} on "
                f"{top1}/{top2}, and I kept listening for {listening}.")

    def _salient_moments(self, moments: List[dict],
                         eng: np.ndarray) -> List[dict]:
        """The moments that become attachments: salience is reaction
        heat plus the max abs movement of the avatar's subjective read;
        take up to 3."""
        scored = [(float(m["heat"]) + float(np.max(np.abs(m["moved"]))), m)
                  for m in moments]
        scored.sort(key=lambda t: -t[0])
        return [m for _, m in scored[:3]]

    def _attach(self, night_key: str, idx: int, moment: dict,
                eng: np.ndarray) -> dict:
        """Bind one intangible correlation on the elephant and return
        the attachment dict — the line, the room it left, what moved,
        what it felt like."""
        moved = np.asarray(moment["moved"], dtype=float)
        pos = np.maximum(np.asarray(eng, dtype=float), 0.0)
        text = moment["text"]
        snippet = text if len(text) <= 48 else text[:48]
        i_moved = int(np.argmax(np.abs(moved)))
        # Anchor the memory on what the avatar FELT when the room barely
        # moved for it (saturated dials read as ~0 delta): the felt dial is
        # the avatar's own distinctive dial — that is the perfume, not the
        # room's. Otherwise the room's biggest mover tells the story.
        if float(pos.max()) > 1e-9 and abs(moved[i_moved]) < MEMORY_FELT_FLOOR:
            i_top = int(np.argmax(pos))
            top_mover = DIAL_NAMES[i_top]
            verb = "rose"
            mag = float(pos[i_top])
            memory = (f"when I said {snippet!r}, I felt {top_mover} "
                      f"{verb} {mag:.2f} — it felt like "
                      f"{FELT_PHRASES.get(top_mover, 'something')}.")
        else:
            i_top = i_moved
            top_mover = DIAL_NAMES[i_top]
            verb = "rose" if moved[i_top] > 0 else "fell"
            mag = float(abs(moved[i_top]))
            memory = (f"when I said {snippet!r}, the room's {top_mover} "
                      f"{verb} {mag:.2f} and it felt like "
                      f"{FELT_PHRASES.get(top_mover, 'something')}.")
        attachment = {
            "event_key": f"{night_key}#{idx}",
            "night": night_key,
            "line": text,
            "room": {n: float(moment["after"].readings[n])
                     for n in DIAL_NAMES},
            "moved": {DIAL_NAMES[i]: float(moved[i])
                      for i in range(len(DIAL_NAMES))
                      if abs(moved[i]) >= 0.02},
            "felt": {DIAL_NAMES[i]: float(eng[i])
                     for i in range(len(DIAL_NAMES)) if eng[i] > 0},
            "memory": memory,
        }
        # The FULL attachment (the moment, the room it left, what moved,
        # what it felt like) is what the elephant binds — the intangible
        # correlation is the whole moment, not a summary of it.
        self.elephant.attach(attachment["event_key"], attachment)
        return attachment

    # ------------------------------------------------------------------ #
    # The nights                                                         #
    # ------------------------------------------------------------------ #
    def attend(self, session,
               lines_spoken: Sequence[Union[str, Tuple[str, dict]]],
               reactions: Optional[Sequence[dict]] = None,
               night_key: Optional[str] = None,
               learning_rate: float = 0.15) -> dict:
        """Go to a themed Tap night: speak the lines, sense the room
        before and after each one, self-tune toward felt engagement
        (reusing the tapnight tuning math), bind attachments from the
        salient moments, distill one character note, and return the
        night's summary.

        Notes on the coupling: the avatar registers itself on the
        session by seeding `session._vibe`/`_vibe_start` directly — the
        same private access the repo's own examples use — because the
        session API has no public "add a participant mid-session"
        hook. `lines_spoken` may be empty: a silent night is still a
        night (the room is felt, the weights still tune, a note is
        still distilled)."""
        self._ensure_participant(session)
        if not session._vibe:
            session.start_session()
        night_key = night_key or f"{session.name}-night{len(self.nights) + 1}"

        moments: List[dict] = []
        for idx, line in enumerate(lines_spoken):
            if isinstance(line, tuple):
                text, reacs = line[0], (line[1] or {})
            else:
                # bare strings pair with the aligned `reactions` list
                text = line
                reacs = dict(reactions[idx]) if reactions else {}
            before = self._sense(session)
            session.speak(self.name, text, reactions=reacs)
            after = self._sense(session)
            moved = after.vector() - before.vector()
            heat = float(sum(reacs.values()))
            if heat > 0 or float(np.max(np.abs(moved))) > MOMENT_MOVE_FLOOR:
                moments.append({"idx": idx, "text": text, "before": before,
                                "after": after, "moved": moved,
                                "heat": heat})

        # The tuning only ever moves weight toward positively-felt dials;
        # the rest decay toward a trace (tapnight keeps an epsilon of
        # exploration on every dial). Over many nights a single dial can
        # dominate — that is a discovered guitarist, not a bug; the
        # attachments and notes keep the character round even then.
        session.tune_participant(self.name, learning_rate=learning_rate)
        self.elephant.dial_weights = \
            session.participants[self.name].dial_weights.copy()
        eng = np.asarray(session.felt_engagement(self.name), dtype=float)

        formed: List[dict] = []
        for moment in self._salient_moments(moments, eng):
            event_key = f"{night_key}#{moment['idx']}"
            if self.elephant.remember(event_key) is None:
                formed.append(self._attach(night_key, moment["idx"],
                                           moment, eng))

        note = self._distill_note(session, eng)
        self.character_notes.append(note)

        summary = {
            "night": night_key,
            "theme": session.name,
            "cycle": session.cycle,
            "warmth": session.room_field().warmth(),
            "top_dials": _top_dials(session.field, 3),
            "felt": {n: float(eng[i]) for i, n in enumerate(DIAL_NAMES)
                     if eng[i] > 0},
            "note": note,
            "attachments": [a["event_key"] for a in formed],
        }
        self.nights.append(summary)
        self._last_session = session
        return summary

    # ------------------------------------------------------------------ #
    # The silent pulse                                                   #
    # ------------------------------------------------------------------ #
    def monologue(self, room=None, prompt: Optional[str] = None) -> str:
        """The internal monologue on a pulse — what the avatar is
        feeling but NOT saying (silence is not empty). Uses a persistent
        PulseLoop per bound room, so the heartbeat keeps its history
        across calls."""
        target = room if room is not None else self._last_session
        if target is None:
            return "I haven't been to the Tap yet — no room to feel."
        if self._pulse_room is not target:
            self._pulse = PulseLoop(self.name, target, period=5.0)
            self._pulse_room = target
        self._pulse.pulse()
        base = self._pulse.internal_monologue(prompt)
        top_dial = DIAL_NAMES[int(np.argmax(self.elephant.dial_weights))]
        text = base + (f" {self.name}: my ear leans {top_dial} tonight, "
                       f"so {top_dial} is the dial I trust.")
        self.monologue_log.append({"ts": self._pulse.last_ts, "text": text})
        return text

    # ------------------------------------------------------------------ #
    # What the character says                                            #
    # ------------------------------------------------------------------ #
    def speak(self, prompt_context: str = "") -> str:
        """What the avatar SAYS now — the ROUND version. A deterministic
        composition that drifts from the flat seed as the character
        forms: the seed still speaks, the tuned sensitivities answer, a
        remembered attachment surfaces (round-robin, so different nights
        surface different memories), and the arc closes it."""
        frame = f"On {prompt_context}: " if prompt_context else ""
        parts: List[str] = [self.persona_prompt.strip().rstrip(".") + "."]
        top_two = np.argsort(-self.elephant.dial_weights, kind="stable")[:2]
        sensitivities = ", and ".join(
            SENSITIVITY_PHRASES.get(DIAL_NAMES[int(i)], DIAL_NAMES[int(i)])
            for i in top_two)
        parts.append(f"But {sensitivities}.")
        if self.elephant.attachments:
            keys = list(self.elephant.attachments.keys())
            key = keys[self._speak_cursor % len(keys)]
            self._speak_cursor += 1
            memory = self.elephant.attachments[key]
            parts.append(f"I keep remembering {memory['memory'].rstrip('.')}.")
        n = len(self.nights)
        parts.append(f"After {n} night{'s' if n != 1 else ''} at The Tap, "
                     f"that's who I am now.")
        return frame + " ".join(parts)

    # ------------------------------------------------------------------ #
    # The sheet                                                          #
    # ------------------------------------------------------------------ #
    def character_sheet(self) -> dict:
        """The whole character, on one sheet: seed and current persona,
        the dial drift (the learning), what it is sensitive to now, its
        attachments, its nights, and its through-line."""
        start = self._started_with
        now = self.elephant.dial_weights
        top_three = np.argsort(-now, kind="stable")[:3]
        return {
            "name": self.name,
            "persona": {
                "seed": self.persona_prompt,
                "notes": list(self.character_notes),
                "current": self.persona,
            },
            "dial_profile": {
                "started_with": {n: float(start[i])
                                 for i, n in enumerate(DIAL_NAMES)},
                "now": {n: float(now[i])
                        for i, n in enumerate(DIAL_NAMES)},
                "drift": {n: float(now[i] - start[i])
                          for i, n in enumerate(DIAL_NAMES)},
            },
            "sensitive_to": [{"dial": DIAL_NAMES[int(i)],
                              "weight": float(now[int(i)])}
                             for i in top_three],
            "attachments": list(self.elephant.attachments.values()),
            "nights_attended": list(self.nights),
            "through_line": self.through_line,
        }
