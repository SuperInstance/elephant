"""elephant — the inter-model temperature.

A room is not a stream. It is a field. The elephant is the field:
the ensemble of many JEPA dials, each perceiving one dimension of a
room's vibe (mood, volume, earnestness, cynicism, joke-landing,
panic...), all shaping every agent in the room at once — like
pheromones, like room temperature. You don't notice the elephant
until you walk into a different room and it's a very different
elephant.

Core pieces:
- `room`  — rooms as message streams with gravity, reverberation, ripples
- `dials` — the bank of JEPA perceivers, one dimension each
- `field` — the room's temperature vector + acclimation/charisma math
- `jepa`  — optional learned backbone (EMA + stop-gradient + VICReg)
"""

from .room import Message, Room
from .field import RoomField, acclimation_curve, charisma_pull
from .gesture import VibeTrajectory, heading_alignment
from .relaxation import Box, Potential, dial_box, attractor, two_well_potential, relax, run_shove

__version__ = "0.2.0"
__all__ = [
    "Message",
    "Room",
    "RoomField",
    "acclimation_curve",
    "charisma_pull",
    "VibeTrajectory",
    "heading_alignment",
    # relaxation rig — a physics of constrained state (experimental)
    "Box",
    "Potential",
    "dial_box",
    "attractor",
    "two_well_potential",
    "relax",
    "run_shove",
]
