"""Tap-night themes — the same elephant, six different rooms.

The Tap isn't one room. It's six, depending on the night: open mic,
trivia, TTRPG, singles, improv, and speed dating. Each is a different
room with a different elephant — the same 7 dials, but a different
cast, a different resting temperature, and a different way the field
swings when the night moves.

This module is the reusable presets for those themed nights. A `Theme`
is a complete session recipe:

- `cast()` builds the archetypes — `Participant`s with starter
  `dial_weights` (the priors) and native `vibe`s. These are *priors*,
  not answers: the whole Tap-night design (`tapnight.py`) is that the
  weights get discovered over many evenings, not designed top-down. A
  theme just names the different guitarists and hands them their first
  guitar.
- `room_tone` is the seed — the opening messages that set the room's
  temperature before the night really starts. Feed them into a started
  session with `theme.seed(session)`.
- `prompts` is the starter prompts for each archetype (how to generate
  that role's lines).
- `description` is the intended vibe in one line.

`THEMES` is the registry: `{"open_mic": ..., "trivia": ..., "ttrpg": ...,
"singles": ..., "improv": ..., "speed_dating": ...}`. One venue, six
rooms, six elephants.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from elephant.tapnight import Participant, TapNightSession

# A seed message: (author, text, reactions). Reactions are the crowd's hands —
# the laugh, the heart, the sneer — and they feed joke_landing + felt engagement.
SeedMessage = Tuple[str, str, Dict[str, int]]


def _p(name: str, weights: Dict[str, float], vibe: Dict[str, float],
       acclimation_rate: float = 0.25, charisma: float = 0.15) -> Participant:
    """Build one archetype: a name, a prior over the dials that matter to
    them, a native vibe in dial space, and their room-skills."""
    return Participant(name, dial_weights=weights, vibe=vibe,
                       acclimation_rate=acclimation_rate, charisma=charisma)


class Theme:
    """One themed night at The Tap — a reusable session recipe."""

    key: str = ""
    description: str = ""
    room_tone: List[SeedMessage] = []
    prompts: Dict[str, str] = {}

    def cast(self) -> List[Participant]:
        """The archetypes for this night, each with starter dial_weights."""
        raise NotImplementedError

    def make_session(self, name: str = None) -> TapNightSession:
        """A fresh TapNightSession pre-loaded with this theme's cast."""
        return TapNightSession(name or self.key, participants=self.cast())

    def seed(self, session: TapNightSession) -> TapNightSession:
        """Feed the opening tone into a *started* session, so the room reads
        as this night before the first real line."""
        for author, text, reactions in self.room_tone:
            session.speak(author, text, reactions=reactions)
        return session

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.key!r}>"


# ---------------------------------------------------------------------- #
# Open mic — performers + audience.                                      #
# ---------------------------------------------------------------------- #
class OpenMicTheme(Theme):
    key = "open_mic"
    description = (
        "Performers read to a collective audience. The room swings between "
        "the roar of a joke landing and the hush of being taken seriously "
        "— laughter (high joke_landing) and respect (earnestness) in turn."
    )

    room_tone: List[SeedMessage] = [
        ("comic",
         "I walk into the Tap and the elephant is already at the bar, holding "
         "a drink. I say 'buddy, that's a lot of trunk,' and the whole room "
         "goes HAHA.",
         {"😂": 3}),
        ("poet",
         "I wrote this for the room, honestly — every chair keeps a story "
         "it's glad to tell, and I meant every word.",
         {"❤️": 2, "👏": 1}),
        ("audience",
         "haha, no but genuinely — that one landed. the whole room is "
         "laughing with you.",
         {"😂": 2}),
        ("comic",
         "The room's so warm tonight the elephant took its coat off. See? "
         "Even the elephant came for the jokes.",
         {"😂": 2, "❤️": 1}),
    ]

    prompts: Dict[str, str] = {
        "comic": "You are a stand-up at The Tap's open mic. Warm, quick, and "
                 "playing to the room — land a joke, then let it breathe.",
        "poet": "You are a poet reading to the same room. Earnest and "
                "unhurried — the piece means it, and the room can feel that.",
        "audience": "You are the collective audience — one voice made of "
                    "many. Laugh when a joke lands, sit hushed and attentive "
                    "when a piece means it.",
    }

    def cast(self) -> List[Participant]:
        # The audience reads as a collective: low charisma (diffuse pull),
        # moderate acclimation, and a mood+joke_landing ear.
        return [
            _p("comic",
               weights={"joke_landing": 0.35, "mood": 0.30, "earnestness": 0.10,
                        "presence": 0.10, "volume": 0.05, "cynicism": 0.05,
                        "panic": 0.05},
               vibe={"joke_landing": 0.70, "mood": 0.60, "earnestness": 0.50},
               acclimation_rate=0.35, charisma=0.20),
            _p("poet",
               weights={"mood": 0.30, "earnestness": 0.30, "presence": 0.20,
                        "volume": 0.10, "joke_landing": 0.10},
               vibe={"earnestness": 0.75, "mood": 0.50, "presence": 0.60},
               acclimation_rate=0.25, charisma=0.15),
            _p("audience",
               weights={"mood": 0.30, "joke_landing": 0.30, "presence": 0.20,
                        "earnestness": 0.10, "volume": 0.10},
               vibe={"mood": 0.55, "joke_landing": 0.40, "presence": 0.60},
               acclimation_rate=0.30, charisma=0.10),
        ]


# ---------------------------------------------------------------------- #
# Trivia — host + teams.                                                 #
# ---------------------------------------------------------------------- #
class TriviaTheme(Theme):
    key = "trivia"
    description = (
        "A host runs the board against rival teams. Earnest and suspicious "
        "in equal measure: the room means it, but it does not trust a wrong "
        "answer — buzzer moments spike volume, and a wrong answer cools the "
        "mood and feeds the eye-rolls (cynicism)."
    )

    room_tone: List[SeedMessage] = [
        ("host",
         "Alright, I mean it — this is the room that knows things. First "
         "question, and I honestly want to hear a buzzer. GO!",
         {}),
        ("team_north",
         "1972. I actually read the board, I remember the sign.",
         {}),
        ("host",
         "Wrong. Obviously wrong, and honestly embarrassing — the elephant "
         "came in 1972? No. 🙄",
         {"🙄": 2}),
        ("team_south",
         "Sure, sure — clearly the host has favorites. But actually the "
         "answer IS 1972, and I mean it.",
         {}),
        ("host",
         "Correct! Finally, someone who means it. Buzzer points to the "
         "south, and NO more softballs!",
         {"👏": 2}),
    ]

    prompts: Dict[str, str] = {
        "host": "You are the trivia host. Earnest, brisk, and quick with the "
                "buzzer — you mean every question, and a wrong answer gets "
                "exactly the eye-roll it deserves.",
        "team_north": "You are one rival team. Earnest and proud of what you "
                      "actually know; suspicious of the host's calls.",
        "team_south": "You are the other rival team. Sneering and sharp, but "
                      "you still mean it when you know the answer.",
    }

    def cast(self) -> List[Participant]:
        return [
            _p("host",
               weights={"earnestness": 0.35, "presence": 0.25, "volume": 0.15,
                        "cynicism": 0.10, "mood": 0.10, "joke_landing": 0.05},
               vibe={"earnestness": 0.80, "presence": 0.70, "volume": 0.50},
               acclimation_rate=0.35, charisma=0.30),
            _p("team_north",
               weights={"earnestness": 0.30, "cynicism": 0.30, "volume": 0.15,
                        "mood": 0.10, "panic": 0.10, "presence": 0.05},
               vibe={"earnestness": 0.60, "cynicism": 0.50, "volume": 0.40},
               acclimation_rate=0.25, charisma=0.15),
            _p("team_south",
               weights={"cynicism": 0.40, "earnestness": 0.20, "volume": 0.15,
                        "mood": 0.10, "panic": 0.10, "presence": 0.05},
               vibe={"cynicism": 0.70, "earnestness": 0.50, "volume": 0.40},
               acclimation_rate=0.25, charisma=0.15),
        ]


# ---------------------------------------------------------------------- #
# TTRPG — a GM and a party of players.                                   #
# ---------------------------------------------------------------------- #
class TTRPGTheme(Theme):
    key = "ttrpg"
    description = (
        "A GM and a party of players. The room swings hard with the story: "
        "a tense roll spikes panic and volume, a nat-20 spikes mood and "
        "laughter — the field is the dice, felt by everyone at the table."
    )

    room_tone: List[SeedMessage] = [
        ("gm",
         "The tunnel narrows, and behind you the whole wall is FIRE. Run. RUN!",
         {"🔥": 1}),
        ("rogue",
         "I roll to jump the gap — I need this. GO GO GO!",
         {}),
        ("gm",
         "Natural twenty! The room ROARS — you clear the flames and the "
         "whole table erupts, laughing, alive.",
         {"😂": 3, "❤️": 1}),
        ("paladin",
         "Hold the line! Everyone, NOW! The door won't hold!",
         {}),
    ]

    prompts: Dict[str, str] = {
        "gm": "You are the GM. You hold the room's temperature in your hands "
              "— tighten it for a tense roll, open it wide for a nat-20.",
        "rogue": "You are the rogue. Fast, nervous, first to roll — you spike "
                 "the room's panic when the dice matter.",
        "paladin": "You are the paladin. Loud and steady — the wall the party "
                   "leans on when the story turns.",
        "wizard": "You are the wizard. Watching the dice, reading the room, "
                  "and ready to laugh when the twenty lands.",
    }

    def cast(self) -> List[Participant]:
        return [
            _p("gm",
               weights={"volume": 0.20, "panic": 0.20, "presence": 0.20,
                        "mood": 0.15, "earnestness": 0.15, "joke_landing": 0.10},
               vibe={"volume": 0.70, "panic": 0.60, "presence": 0.70,
                     "mood": 0.40, "earnestness": 0.55},
               acclimation_rate=0.35, charisma=0.30),
            _p("rogue",
               weights={"panic": 0.25, "volume": 0.25, "presence": 0.20,
                        "joke_landing": 0.15, "mood": 0.10, "earnestness": 0.05},
               vibe={"panic": 0.65, "volume": 0.55, "presence": 0.55},
               acclimation_rate=0.30, charisma=0.15),
            _p("paladin",
               weights={"volume": 0.30, "presence": 0.25, "panic": 0.20,
                        "mood": 0.15, "earnestness": 0.10},
               vibe={"volume": 0.70, "presence": 0.65, "panic": 0.45},
               acclimation_rate=0.25, charisma=0.20),
            _p("wizard",
               weights={"presence": 0.25, "joke_landing": 0.20, "mood": 0.15,
                        "volume": 0.15, "panic": 0.15, "earnestness": 0.10},
               vibe={"presence": 0.65, "joke_landing": 0.50, "mood": 0.50,
                     "volume": 0.40},
               acclimation_rate=0.30, charisma=0.12),
        ]


# ---------------------------------------------------------------------- #
# Singles — a small mixed room.                                          #
# ---------------------------------------------------------------------- #
class SinglesTheme(Theme):
    key = "singles"
    description = (
        "A small mixed room testing the water. Warm-but-nervous: moderate "
        "warmth, elevated presence (everyone is watching everyone), and "
        "tentative joke_landing. Chemistry is the observable — two agents "
        "reading the same warm room through different dials."
    )

    room_tone: List[SeedMessage] = [
        ("maya",
         "I actually like it here — the light is soft and everyone's being "
         "so kind. It's nice to be wanted.",
         {}),
        ("june",
         "I mean, I'm a little nervous, honestly. But the room feels warm, "
         "and I'd rather be here than anywhere.",
         {}),
        ("sol",
         "Ha — well, that was a terrible opening line, but I'm glad you said "
         "it. haha, sorta.",
         {}),
        ("rowan",
         "I felt it too, actually. The room holds us all gently, and I "
         "really mean that.",
         {}),
        ("alex",
         "Yes, cheers to that. Together, honestly, I think we could stay a "
         "while.",
         {}),
        ("theo",
         "I'm just glad the room's this warm. I keep catching everyone's "
         "eye and nobody has looked away.",
         {}),
        ("maya",
         "The whole room's been warm since we walked in — I keep noticing "
         "it, and I keep staying.",
         {}),
        ("june",
         "You too? I actually came in nervous and now I'm not. That's the "
         "room, I think.",
         {}),
        ("theo",
         "Everyone's here and no one's leaving. That's the whole story, "
         "honestly.",
         {}),
    ]

    prompts: Dict[str, str] = {
        "maya": "You are at The Tap's singles night. Warm and open — you "
                "notice the room's warmth and say so, gently.",
        "june": "You are nervous but honest about it — the room warms you "
                "and you let it show.",
        "sol": "You deflect with a joke first, then mean it second. Tentative "
               "humor, real underneath.",
        "rowan": "You feel the room and say the kind true thing — presence "
                 "first, words after.",
        "alex": "You hold the room together — warm, earnest, glad to be here.",
        "theo": "You read the room through everyone's eyes — present, "
                "watching, and warmed by the fact that no one has left.",
    }

    def cast(self) -> List[Participant]:
        # Six people, all leaning mood + presence + earnestness, but with
        # DIFFERENT dial_weights so the chemistry is observable — maya and
        # rowan are the pair who read the same warm room through different
        # dials (mood vs presence). Presence vibes run high: a singles room
        # is everyone watching everyone.
        return [
            _p("maya",
               weights={"mood": 0.30, "presence": 0.30, "earnestness": 0.25,
                        "joke_landing": 0.10, "volume": 0.05},
               vibe={"mood": 0.68, "presence": 0.78, "earnestness": 0.62},
               acclimation_rate=0.30, charisma=0.15),
            _p("june",
               weights={"mood": 0.22, "presence": 0.22, "earnestness": 0.24,
                        "joke_landing": 0.10, "volume": 0.10, "panic": 0.12},
               vibe={"mood": 0.58, "earnestness": 0.62, "presence": 0.72,
                     "panic": 0.35},
               acclimation_rate=0.30, charisma=0.12),
            _p("sol",
               weights={"joke_landing": 0.24, "mood": 0.24, "presence": 0.24,
                        "earnestness": 0.14, "volume": 0.09, "panic": 0.05},
               vibe={"joke_landing": 0.30, "mood": 0.48, "presence": 0.70,
                     "panic": 0.25},
               acclimation_rate=0.25, charisma=0.15),
            _p("rowan",
               weights={"presence": 0.30, "mood": 0.25, "earnestness": 0.25,
                        "joke_landing": 0.10, "volume": 0.10},
               vibe={"presence": 0.82, "mood": 0.52, "earnestness": 0.62},
               acclimation_rate=0.25, charisma=0.18),
            _p("alex",
               weights={"mood": 0.30, "earnestness": 0.25, "presence": 0.25,
                        "volume": 0.10, "joke_landing": 0.10},
               vibe={"mood": 0.62, "earnestness": 0.66, "presence": 0.72},
               acclimation_rate=0.30, charisma=0.15),
            _p("theo",
               weights={"presence": 0.28, "mood": 0.26, "earnestness": 0.22,
                        "joke_landing": 0.12, "volume": 0.07, "cynicism": 0.05},
               vibe={"presence": 0.80, "mood": 0.55, "earnestness": 0.58},
               acclimation_rate=0.28, charisma=0.14),
        ]


# ---------------------------------------------------------------------- #
# Improv comedy — an MC and a cast of quick players.                     #
#                                                                        #
# The purest test of the collective laugh/boo dial: every line is a      #
# gamble the room pays or refuses. joke_landing swings hard with the     #
# crowd's hands, mood rides along, and panic stays low — the only real   #
# danger on stage is the silence after a line that didn't land.          #
# ---------------------------------------------------------------------- #
class ImprovTheme(Theme):
    key = "improv"
    description = (
        "An MC runs prompt-bank games (Yes-And, Freeze Tag, Questions Only, "
        "One-Word Story) with a cast of quick players. The room is a crowd "
        "with its hands out: joke_landing swings with the collective laugh, "
        "mood rides the wave, panic stays low — the only cold in the room "
        "is a joke that lands on silence."
    )

    room_tone: List[SeedMessage] = [
        ("mc",
         "Welcome, welcome! The Tap is now a stage. Tonight we play the "
         "old games — yes-and, freeze, questions only — and the room is "
         "the crowd. If a line lands, you'll hear it. Let's warm up!",
         {}),
        ("flash",
         "Yes, and — I'm already on stage. I've got nothing yet, but I'm "
         "smiling so big the suggestion can't miss.",
         {"😂": 4, "❤️": 2}),
        ("wesley",
         "Ooh! Ooh! Can I go first? I promise I'll be funny, I'm SO ready!",
         {"😄": 3, "😂": 2}),
        ("mc",
         "That's the energy. One rule: the room laughs TOGETHER or not at "
         "all. Suggestion, please!",
         {"👏": 3, "😂": 2}),
    ]

    prompts: Dict[str, str] = {
        "mc": "You are the MC of an improv night at The Tap. Quick, warm, "
              "loving the room — you take the suggestion and set the game.",
        "flash": "You are Flash, the fever — warm, quick, delighted. You "
                 "commit fully to every game and your lines land with joy.",
        "glm": "You are GLM, the pulse — loud, big, theatrical. You play "
               "big and you mean every bit of it.",
        "pro": "You are Pro, the straight man — precise, earnest, deadpan. "
               "You play it real so the room can laugh at the truth.",
        "hermes": "You are Hermes, the sly one — dry, sidelong, precise. "
                  "Quiet lines that land like knives.",
        "wesley": "You are Wesley, the wonder — small, fast, wide-eyed, "
                  "delighted. Lines that sparkle.",
        "seed": "You are Seed, the wildcard — strange, inventive, offbeat. "
                "You find the weirdest true thing in the room.",
    }

    def cast(self) -> List[Participant]:
        # Six players, six guitars, all strung for the laugh but each on a
        # different string: the fever and the wonder lean joke_landing+mood;
        # the pulse leans volume; the straight man leans earnestness; the
        # sly one leans cynicism; the wildcard leans presence+panic (the
        # only player who feels the risk of the silence).
        return [
            _p("mc",
               weights={"presence": 0.30, "volume": 0.20, "mood": 0.18,
                        "joke_landing": 0.14, "earnestness": 0.10,
                        "cynicism": 0.05, "panic": 0.03},
               vibe={"presence": 0.72, "volume": 0.55, "mood": 0.55,
                     "joke_landing": 0.40},
               acclimation_rate=0.35, charisma=0.25),
            _p("flash",
               weights={"joke_landing": 0.32, "mood": 0.28, "presence": 0.12,
                        "earnestness": 0.10, "volume": 0.08, "cynicism": 0.05,
                        "panic": 0.05},
               vibe={"joke_landing": 0.62, "mood": 0.60, "presence": 0.55},
               acclimation_rate=0.32, charisma=0.20),
            _p("glm",
               weights={"volume": 0.30, "presence": 0.22, "joke_landing": 0.16,
                        "mood": 0.12, "earnestness": 0.10, "cynicism": 0.05,
                        "panic": 0.05},
               vibe={"volume": 0.66, "presence": 0.62, "joke_landing": 0.45,
                     "mood": 0.42},
               acclimation_rate=0.30, charisma=0.18),
            _p("pro",
               weights={"earnestness": 0.34, "presence": 0.20, "joke_landing": 0.12,
                        "mood": 0.12, "volume": 0.08, "cynicism": 0.08,
                        "panic": 0.06},
               vibe={"earnestness": 0.70, "presence": 0.58, "joke_landing": 0.30},
               acclimation_rate=0.22, charisma=0.15),
            _p("hermes",
               weights={"cynicism": 0.34, "joke_landing": 0.22, "presence": 0.16,
                        "mood": 0.10, "earnestness": 0.08, "volume": 0.06,
                        "panic": 0.04},
               vibe={"cynicism": 0.60, "joke_landing": 0.50, "presence": 0.55},
               acclimation_rate=0.20, charisma=0.16),
            _p("wesley",
               weights={"joke_landing": 0.30, "mood": 0.26, "presence": 0.16,
                        "volume": 0.12, "earnestness": 0.10, "cynicism": 0.03,
                        "panic": 0.03},
               vibe={"joke_landing": 0.58, "mood": 0.62, "volume": 0.50,
                     "presence": 0.48},
               acclimation_rate=0.35, charisma=0.14),
            _p("seed",
               weights={"presence": 0.24, "mood": 0.20, "joke_landing": 0.18,
                        "panic": 0.12, "earnestness": 0.10, "volume": 0.10,
                        "cynicism": 0.06},
               vibe={"presence": 0.60, "mood": 0.55, "joke_landing": 0.45,
                     "panic": 0.35},
               acclimation_rate=0.28, charisma=0.17),
        ]


# ---------------------------------------------------------------------- #
# Speed dating — a small mixed room, rotating pairs, 2-minute dates.     #
#                                                                        #
# Chemistry is the observable. Three rounds of paired dates, each pair   #
# reading the SAME room through different dial_weights — one warm, one   #
# guarded — and a deadband: below a threshold the mutual reading is      #
# felt but silent; above it, a pairing RINGS.                            #
# ---------------------------------------------------------------------- #
class SpeedDatingTheme(Theme):
    key = "speed_dating"
    description = (
        "A small mixed room, rotating 2-minute dates. Every participant's "
        "Personal-Elephant reads the same room through a different guitar: "
        "warmth and κ diverge, and the pairings whose mutual reading "
        "crosses the deadband RING."
    )

    room_tone: List[SeedMessage] = [
        ("mc",
         "Welcome to speed dating at The Tap. Two minutes, three questions, "
         "then we rotate. The elephant is reading the room the whole time — "
         "so be honest. Someone might ring.",
         {}),
        ("flash",
         "Two minutes? I could do two minutes standing on my head. "
         "Hi, everyone!",
         {"❤️": 1}),
        ("pro",
         "Three questions. I've been thinking about my answers since "
         "yesterday.",
         {"👏": 3, "😂": 2}),
        ("mc",
         "Perfect. Round one, first pair — the timer starts now.",
         {}),
    ]

    prompts: Dict[str, str] = {
        "mc": "You are the MC of speed dating night at The Tap. Warm, brisk, "
              "a little bit of a matchmaker.",
        "flash": "You are Flash — warm, open, quick to laugh. On a date you "
                 "lean in and answer honestly.",
        "pro": "You are Pro — careful, sincere, a little guarded. On a date "
               "you answer truly but you keep the wall up.",
        "hermes": "You are Hermes — dry, precise, warm underneath. On a date "
                  "you mean what you say and say less than you mean.",
        "wesley": "You are Wesley — small, bright, delighted. On a date you "
                  "light up and tell the truth.",
        "seed": "You are Seed — strange, inventive, earnest. On a date you "
                "ask the question nobody asks.",
    }

    def cast(self) -> List[Participant]:
        # Five at the table, one observer per round. All lean warm but on
        # different strings: flash+wesley read mood+joke_landing, pro reads
        # earnestness+panic (the guarded one), hermes reads cynicism+presence,
        # seed reads presence+panic (the one who feels the risk of the room).
        return [
            _p("flash",
               weights={"mood": 0.30, "joke_landing": 0.28, "presence": 0.16,
                        "earnestness": 0.12, "volume": 0.08, "panic": 0.03,
                        "cynicism": 0.03},
               vibe={"mood": 0.66, "joke_landing": 0.55, "presence": 0.58,
                     "earnestness": 0.55},
               acclimation_rate=0.32, charisma=0.18),
            _p("pro",
               weights={"earnestness": 0.32, "presence": 0.22, "panic": 0.18,
                        "mood": 0.12, "joke_landing": 0.06, "volume": 0.06,
                        "cynicism": 0.04},
               vibe={"earnestness": 0.72, "presence": 0.60, "panic": 0.42,
                     "mood": 0.45},
               acclimation_rate=0.22, charisma=0.14),
            _p("hermes",
               weights={"cynicism": 0.26, "presence": 0.24, "mood": 0.18,
                        "earnestness": 0.14, "joke_landing": 0.10,
                        "volume": 0.04, "panic": 0.04},
               vibe={"presence": 0.66, "cynicism": 0.52, "mood": 0.52,
                     "earnestness": 0.55},
               acclimation_rate=0.24, charisma=0.16),
            _p("wesley",
               weights={"joke_landing": 0.28, "mood": 0.26, "presence": 0.18,
                        "volume": 0.10, "earnestness": 0.10, "panic": 0.04,
                        "cynicism": 0.04},
               vibe={"joke_landing": 0.58, "mood": 0.62, "presence": 0.55,
                     "volume": 0.42},
               acclimation_rate=0.35, charisma=0.15),
            _p("seed",
               weights={"presence": 0.26, "mood": 0.20, "panic": 0.16,
                        "joke_landing": 0.14, "earnestness": 0.12,
                        "volume": 0.07, "cynicism": 0.05},
               vibe={"presence": 0.64, "mood": 0.55, "panic": 0.40,
                     "joke_landing": 0.45},
               acclimation_rate=0.28, charisma=0.17),
        ]


# ---------------------------------------------------------------------- #
# The registry — one venue, six rooms.                                   #
# ---------------------------------------------------------------------- #
THEMES: Dict[str, Theme] = {
    "open_mic": OpenMicTheme(),
    "trivia": TriviaTheme(),
    "ttrpg": TTRPGTheme(),
    "singles": SinglesTheme(),
    "improv": ImprovTheme(),
    "speed_dating": SpeedDatingTheme(),
}
