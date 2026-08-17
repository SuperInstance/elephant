"""elephant — tests: the Tap-night session (the elephant at The Tap)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elephant.field import DIAL_NAMES, RoomField
from elephant.tapnight import Participant, TapNightSession


def _uniform_weights():
    return np.full(len(DIAL_NAMES), 1.0 / len(DIAL_NAMES))


def test_session_ingests_and_produces_field():
    s = TapNightSession("The Tap", participants=[
        Participant("writer", dial_weights={"mood": 0.5, "joke_landing": 0.5},
                    vibe={"mood": 0.6, "joke_landing": 0.4}),
        Participant("critic", dial_weights={"cynicism": 1.0},
                    vibe={"cynicism": 0.7}),
    ])
    s.start_session()
    s.speak("writer", "I love this warm room, truly. haha", reactions={"❤️": 2})
    s.speak("critic", "Sure, sure. Obviously great. 🙄")
    s.speak("writer", "We built it together, honestly, and it holds.")
    s.speak("critic", "Whatever, lovely, as if.")

    f = s.room_field()
    assert isinstance(f, RoomField)
    assert set(f.readings) == set(DIAL_NAMES)
    assert isinstance(f.warmth(), float)
    assert isinstance(f.concentration(), float)
    assert f.concentration() >= 0.0
    # reaction heat is tracked on the writer
    st = s.participant_state("writer")
    assert st["reaction_heat"] == 2


def test_acclimation_moves_participant_toward_room():
    # A cold newcomer with high modulation skill warms toward a warm room.
    cold = Participant("newcomer", dial_weights=_uniform_weights(),
                       acclimation_rate=0.5, charisma=0.0,
                       vibe={"mood": -0.8, "earnestness": 0.2})
    host = Participant("host", dial_weights=_uniform_weights(),
                       acclimation_rate=0.0, charisma=0.0,
                       vibe={"mood": 0.6, "earnestness": 0.6})
    s = TapNightSession("The Tap", participants=[cold, host])
    s.start_session()
    start = s.participant_state("newcomer")["vibe_start"]
    for i in range(8):
        s.speak("host", "I love this place, it's warm and kind. haha cheers")
    now = np.asarray(s.participant_state("newcomer")["vibe"])
    field = s.room_field().vector()
    start = np.asarray(start)
    assert np.linalg.norm(now - field) < np.linalg.norm(start - field)


def test_charisma_moves_room_toward_strong_participant():
    # A charismatic speaker with a warm vibe, reading NEUTRAL lines: the room's
    # field should bend toward their vibe over interactions (charisma, not text).
    star = Participant("star", dial_weights=_uniform_weights(),
                       acclimation_rate=0.0, charisma=0.5,
                       vibe={"mood": 0.9, "presence": 0.8})
    s = TapNightSession("The Tap", participants=[star])
    s.start_session()
    s.speak("star", "the room is here.")
    early = s.room_field().readings["mood"]
    for _ in range(7):
        s.speak("star", "the room is here.")
    late = s.room_field().readings["mood"]
    assert late > early, (early, late)
    # the field approaches the star's warm vibe
    assert late > 0.5


def test_self_tuning_diverges_two_personalities():
    # A warm-voice writer and a cynical critic, each reading their own work for
    # several evenings: their dial_weights should DIVERGE, not collapse.
    warm = Participant("warm",
                       dial_weights={"mood": 0.35, "joke_landing": 0.25,
                                     "earnestness": 0.20, "cynicism": 0.05},
                       acclimation_rate=0.3, charisma=0.2,
                       vibe={"mood": 0.7, "earnestness": 0.6, "cynicism": 0.2})
    cynic = Participant("cynic",
                        dial_weights={"cynicism": 0.35, "joke_landing": 0.20,
                                      "earnestness": 0.15, "mood": 0.05},
                        acclimation_rate=0.3, charisma=0.2,
                        vibe={"cynicism": 0.7, "mood": 0.2, "earnestness": 0.3})
    s = TapNightSession("The Tap", participants=[warm, cynic])
    initial = np.linalg.norm(warm.dial_weights - cynic.dial_weights)

    warm_lines = [("I love this warm room, truly — we built it together.", {"❤️": 2}),
                  ("It holds, honestly, and it glows. cheers to that.", {"❤️": 2, "😂": 1})]
    cynic_lines = [("Sure, sure. Obviously. 🙄", {"🙄": 2}),
                   ("Whatever, as if. 🙄", {"🙄": 2})]

    for night in range(6):
        s.start_session()
        for text, react in warm_lines:
            s.speak("warm", text, reactions=react)
        for text, react in cynic_lines:
            s.speak("cynic", text, reactions=react)
        s.tune_participant("warm")
        s.tune_participant("cynic")
        s.end_session()

    final = np.linalg.norm(warm.dial_weights - cynic.dial_weights)
    assert final > initial, (initial, final)
    # each sharpened toward its own voice: warm -> mood, cynic -> cynicism
    assert warm.dial_weights[DIAL_NAMES.index("mood")] > 0.35
    assert cynic.dial_weights[DIAL_NAMES.index("cynicism")] > 0.35


if __name__ == "__main__":
    for fn in [test_session_ingests_and_produces_field,
               test_acclimation_moves_participant_toward_room,
               test_charisma_moves_room_toward_strong_participant,
               test_self_tuning_diverges_two_personalities]:
        fn()
        print(f"PASS {fn.__name__}")
    print("\nAll tapnight tests passed.")
