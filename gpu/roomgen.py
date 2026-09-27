"""gpu.roomgen — synthetic room-field data generator (v0, prototype).

Generates synthetic ROOMS in the spirit of the v3 spec
(docs/elephant-sense-v3-design.md) and the captain's reframing
(docs/jepa-is-the-elephant.md):

- every room has a latent ambient state (warmth, tightness κ, pacing,
  volume, and nuisance axes) — the "temperature of being in the room";
- cold rooms are TIGHT (high κ: one way to be), warm rooms are LOOSE
  (low κ: many ways to be) — warmth anti-correlates with tightness;
- the room emits anonymous "clips" through a fixed nonlinear mixture
  (the sensor channel). The encoder never sees the latent directly;
- ACCLIMATION traces: an agent entering a room relaxes toward the
  room latent with time constant τ (agent → room);
- CHARISMA passes: in charisma rooms the room latent itself drifts
  toward a strong agent over interactions (room → agent); matched
  control rooms keep the room latent static.

ALL DATA HERE IS SYNTHETIC. Nothing in this module touches fleet
corpora. It exists to exercise the training machinery and the metric
battery of the v3 spec on data where ground truth is known — see
docs/gpu-room-state-embed-2026-09-27.md for the honest booking.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

# ---------------------------------------------------------------------- #
# Latent layout                                                          #
# ---------------------------------------------------------------------- #
LATENT_NAMES = [
    "warmth",      # [-1, +1]: cold plunge -> sauna
    "tightness",  # κ proxy: cold rooms tight, warm rooms loose
    "pacing",      # slow (sauna) -> fast
    "volume",      # quiet -> loud
    "mood2",       # nuisance affect axis
    "topic1",      # nuisance topic axis
    "topic2",      # nuisance topic axis
    "energy",      # nuisance energy axis
]
LATENT_DIM = len(LATENT_NAMES)
WARMTH, TIGHTNESS = 0, 1

OBS_DIM = 48  # the synthetic sensor channel width


@dataclass
class RoomSpec:
    """Ground-truth latent state of one room."""

    room_id: str
    latent: torch.Tensor            # [LATENT_DIM]
    clip_sigma: float               # within-room latent jitter (the κ proxy)
    charisma: bool = False          # room runs a charismatic pass
    chi_max: float = 0.0            # charisma pull magnitude (0 for control)
    tau_charisma: float = 8.0


@dataclass
class RoomData:
    """One room's emitted clips + metadata."""

    spec: RoomSpec
    obs: torch.Tensor               # [n_clips, OBS_DIM] the sensor view
    latent_paths: torch.Tensor      # [n_clips, LATENT_DIM] ground truth (eval only)

    @property
    def room_id(self) -> str:
        return self.spec.room_id

    @property
    def warmth(self) -> float:
        return float(self.spec.latent[WARMTH].item())

    @property
    def clip_sigma(self) -> float:
        return self.spec.clip_sigma


@dataclass
class AcclimationTrace:
    """An agent entering a room and relaxing toward it (agent → room)."""

    room_id: str
    tau_true: float                 # ground-truth acclimation time constant
    turn_obs: torch.Tensor          # [T, OBS_DIM]
    z0: torch.Tensor                # agent's entry latent [LATENT_DIM]
    z_room: torch.Tensor            # the room latent [LATENT_DIM]


@dataclass
class World:
    """A whole synthetic world: rooms + traces + the fixed mixture."""

    train_rooms: List[RoomData]
    eval_rooms: List[RoomData]      # held out from training entirely
    acclimation: List[AcclimationTrace]
    charisma_rooms: List[RoomData]  # subset of eval rooms, with controls
    control_rooms: List[RoomData]
    seed: int


# ---------------------------------------------------------------------- #
# The fixed nonlinear mixture (room latent -> sensor view)               #
# ---------------------------------------------------------------------- #
class SensorMixture:
    """A seeded, frozen nonlinear map latent -> observation.

    x = gain * tanh(W @ z + b) + channel_noise, with random per-dim
    gain and mild channel dropout. Frozen at construction: the encoder
    must learn to invert it from contrast alone; the same frozen
    mixture generates train and eval rooms so held-out rooms are only
    "new latents", not "new physics" (an honest, stated limit).
    """

    def __init__(self, seed: int, obs_dim: int = OBS_DIM):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(LATENT_DIM, obs_dim, generator=g) / math.sqrt(LATENT_DIM)
        self.b = 0.25 * torch.randn(obs_dim, generator=g)
        self.gain = 0.8 + 0.6 * torch.rand(obs_dim, generator=g)
        self.noise_sigma = 0.05
        self.dropout_p = 0.05
        self._g = g  # channel noise stream, separate from data streams

    def observe(self, z: torch.Tensor) -> torch.Tensor:
        """z: [..., LATENT_DIM] -> [..., OBS_DIM]."""
        x = torch.tanh(z @ self.W + self.b) * self.gain
        x = x + self.noise_sigma * torch.randn(
            x.shape, generator=self._g
        )
        drop = (torch.rand(x.shape, generator=self._g) < self.dropout_p).float()
        return x * (1.0 - drop)


# ---------------------------------------------------------------------- #
# Room sampling                                                          #
# ---------------------------------------------------------------------- #
def sample_room_latent(gen: torch.Generator) -> torch.Tensor:
    """Sample a room latent. Warmth ~ U(-1,1); cold rooms run tighter
    (anti-correlated with warmth, per spec §2.2), the rest nuisance."""
    z = torch.zeros(LATENT_DIM)
    z[WARMTH] = 2.0 * torch.rand(1, generator=gen) - 1.0
    w = float(z[WARMTH])
    tight = 0.65 - 0.45 * w + 0.12 * torch.randn(1, generator=gen)
    z[TIGHTNESS] = float(min(max(tight, 0.05), 1.0))
    for i in range(2, LATENT_DIM):
        z[i] = 2.0 * torch.rand(1, generator=gen) - 1.0
    return z


def clip_sigma_from_latent(z: torch.Tensor) -> float:
    """Within-room latent jitter: tight (cold) rooms jitter little,
    loose (warm) rooms jitter a lot. This IS the κ proxy the room
    temperature metric must recover."""
    tight = float(z[TIGHTNESS])
    return 0.06 + 0.30 * (1.0 - tight)


def make_room(
    room_id: str,
    gen: torch.Generator,
    mixture: SensorMixture,
    n_clips: int,
    warmth_hint: Optional[float] = None,
) -> RoomData:
    z_room = sample_room_latent(gen)
    if warmth_hint is not None:
        z_room = z_room.clone()
        z_room[WARMTH] = warmth_hint
        tight = 0.65 - 0.45 * warmth_hint + 0.12 * torch.randn(1, generator=gen)
        z_room[TIGHTNESS] = float(min(max(tight, 0.05), 1.0))
    sigma = clip_sigma_from_latent(z_room)
    spec = RoomSpec(room_id=room_id, latent=z_room, clip_sigma=sigma)

    jitter = sigma * torch.randn(n_clips, LATENT_DIM, generator=gen)
    z_clips = z_room.unsqueeze(0) + jitter
    obs = mixture.observe(z_clips)
    return RoomData(spec=spec, obs=obs, latent_paths=z_clips)


def make_room_with_charisma_pass(
    room_id: str,
    gen: torch.Generator,
    mixture: SensorMixture,
    n_clips: int,
    chi_max: float,
    tau_c: float = 8.0,
) -> RoomData:
    """A room whose latent drifts toward a charismatic agent over
    n_clips interactions (room → agent). Control = chi_max 0."""
    z0 = sample_room_latent(gen)
    # the charismatic agent is COLD and tight (the wheelhouse presence)
    z_charm = sample_room_latent(gen).clone()
    z_charm[WARMTH] = -abs(float(z_charm[WARMTH])) - 0.3
    z_charm[TIGHTNESS] = 0.8

    sigma = clip_sigma_from_latent(z0)
    spec = RoomSpec(
        room_id=room_id, latent=z0, clip_sigma=sigma,
        charisma=chi_max > 0, chi_max=chi_max, tau_charisma=tau_c,
    )

    t = torch.arange(n_clips, dtype=torch.float32)
    chi_t = chi_max * (1.0 - torch.exp(-t / tau_c))
    drift = chi_t.unsqueeze(1) * (z_charm - z0).unsqueeze(0)  # [n, k]
    z_path = z0.unsqueeze(0) + drift                          # [n, k]
    jitter = sigma * torch.randn(n_clips, LATENT_DIM, generator=gen)
    z_clips = z_path + jitter
    obs = mixture.observe(z_clips)
    return RoomData(spec=spec, obs=obs, latent_paths=z_clips)


# ---------------------------------------------------------------------- #
# Acclimation traces (agent → room)                                      #
# ---------------------------------------------------------------------- #
def make_acclimation_trace(
    room: RoomData,
    gen: torch.Generator,
    mixture: SensorMixture,
    n_turns: int = 24,
    tau: Optional[float] = None,
) -> AcclimationTrace:
    """An agent enters `room` from a home room of OPPOSITE warmth
    (sauna -> plunge and plunge -> sauna) and relaxes toward the room
    latent: z(t) = z_room + (z0 - z_room) * exp(-t / tau)."""
    z_room = room.spec.latent
    z0 = z_room.clone()
    z0[WARMTH] = -float(z_room[WARMTH])  # arrive from the opposite pole
    z0[TIGHTNESS] = float(min(max(1.0 - float(z_room[TIGHTNESS]), 0.05), 1.0))
    for i in range(2, LATENT_DIM):
        z0[i] = 2.0 * torch.rand(1, generator=gen) - 1.0

    if tau is None:
        tau = float(torch.exp(torch.log(torch.tensor(2.0)) + 2.3 * torch.randn(1, generator=gen)))
        tau = max(1.0, min(30.0, tau))

    t = torch.arange(n_turns, dtype=torch.float32)
    decay = torch.exp(-t / tau).unsqueeze(1)
    z_t = z_room.unsqueeze(0) + decay * (z0 - z_room).unsqueeze(0)
    jitter = room.spec.clip_sigma * torch.randn(n_turns, LATENT_DIM, generator=gen)
    obs = mixture.observe(z_t + jitter)
    return AcclimationTrace(
        room_id=room.room_id, tau_true=tau, turn_obs=obs, z0=z0, z_room=z_room,
    )


# ---------------------------------------------------------------------- #
# World                                                                  #
# ---------------------------------------------------------------------- #
def build_world(
    seed: int = 2718,
    n_train_rooms: int = 36,
    n_eval_rooms: int = 12,
    n_clips: int = 48,
    n_acclimation: int = 16,
    n_charisma_pairs: int = 6,
    n_turns: int = 24,
) -> World:
    """Build the full synthetic world. Fully determined by `seed`."""
    gen = torch.Generator().manual_seed(seed)
    mixture = SensorMixture(seed=seed + 1)

    train_rooms = [
        make_room(f"train-{i:02d}", gen, mixture, n_clips) for i in range(n_train_rooms)
    ]
    eval_rooms = [
        make_room(f"eval-{i:02d}", gen, mixture, n_clips) for i in range(n_eval_rooms)
    ]

    # Acclimation traces live in HELD-OUT rooms only.
    acclimation = []
    for j in range(n_acclimation):
        room = eval_rooms[j % n_eval_rooms]
        acclimation.append(
            make_acclimation_trace(room, gen, mixture, n_turns=n_turns)
        )

    # Charisma vs matched control pairs, also held-out.
    charisma_rooms, control_rooms = [], []
    for p in range(n_charisma_pairs):
        chi = 0.25 + 0.35 * float(torch.rand(1, generator=gen))
        charisma_rooms.append(
            make_room_with_charisma_pass(f"charisma-{p:02d}", gen, mixture, n_clips, chi)
        )
        control_rooms.append(
            make_room_with_charisma_pass(f"control-{p:02d}", gen, mixture, n_clips, 0.0)
        )

    return World(
        train_rooms=train_rooms,
        eval_rooms=eval_rooms,
        acclimation=acclimation,
        charisma_rooms=charisma_rooms,
        control_rooms=control_rooms,
        seed=seed,
    )


def warmth_label(room: RoomData) -> Optional[int]:
    """Cold/warm label with an honest abstain band in the middle."""
    w = room.warmth
    if w < -0.2:
        return 0  # cold plunge
    if w > 0.2:
        return 1  # sauna
    return None
