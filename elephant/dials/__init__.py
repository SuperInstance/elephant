"""The dial bank — many JEPAs, many dimensions, one room.

Each module is one JEPA perceiving ONE dimension of the room's vibe,
like a dial on a thermostat you can't see but everyone feels. The
default bank ships seven dials:

- mood         — warm/cold valence of the room
- volume       — how loud the room is talking
- earnestness  — how much the room means it
- cynicism     — how much the room is rolling its eyes
- joke_landing — did the joke land? (the COLLECTIVE laugh or boo)
- panic        — stampede sense (fire in the room)
- presence     — pheromone trace (who's been here, how long)

More dials are cheap: a room can have many JEPA models perceiving
vibes on more than one dimension at once. Each dial is a sense, not a
description — it affects agents constantly, without words, because
it's a dial.
"""
from .mood import MoodDial
from .volume import VolumeDial
from .earnestness import EarnestnessDial
from .cynicism import CynicismDial
from .joke_landing import JokeLandingDial
from .panic import PanicDial
from .presence import PresenceDial

DEFAULT_DIALS = [
    MoodDial(),
    VolumeDial(),
    EarnestnessDial(),
    CynicismDial(),
    JokeLandingDial(),
    PanicDial(),
    PresenceDial(),
]

__all__ = [
    "MoodDial", "VolumeDial", "EarnestnessDial", "CynicismDial",
    "JokeLandingDial", "PanicDial", "PresenceDial", "DEFAULT_DIALS",
]
