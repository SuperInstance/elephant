"""gpu.model — the contrastive room encoder (v0).

A small MLP that maps a room clip's sensor view to an L2-normalized
embedding, trained with the v3 spec's adopted objective
(docs/elephant-sense-v3-design.md §2.3):

- hierarchical clip<->clip NT-Xent (SimCLR-style): positives are other
  clips from the SAME room, negatives are clips from OTHER rooms;
- batch structure: each batch is ALL clips from 2-3 rooms (many
  within-room positives, bounded negatives) — not one clip per room;
- temperature tau = 0.15 fixed (the spec's adopted value, not the
  SimCLR 0.07 default);
- an explicit within-room spread regularizer (VICReg-style variance
  floor) as the anti-collapse guard: the model is penalized when any
  embedding dimension's within-batch variance collapses.

No centroids are used as anchors (spec §2.3). The model is small by
design: the RTX 4050 has 6 GB and the point is the prototype, not
parameter count (~150k params).
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

TAU = 0.15           # spec §2.3: fixed contrastive temperature
SPREAD_LAMBDA = 1.0  # weight of the anti-collapse guard
COLLAPSE_COS = 0.90  # hinge: penalize a room only when its mean
                     # within-room pairwise cosine exceeds this — a
                     # FLOOR-style (VICReg variance) guard was tried first
                     # and it HOMOGENIZED within-room spread, destroying
                     # the κ/temperature signal (booked in the docs)


class RoomEncoder(nn.Module):
    """sensor view [B, obs_dim] -> unit embedding [B, emb_dim]."""

    def __init__(self, obs_dim: int = 48, emb_dim: int = 64,
                 hidden: Tuple[int, ...] = (256, 256, 128)):
        super().__init__()
        layers: List[nn.Module] = []
        d = obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, emb_dim)]
        self.net = nn.Sequential(*layers)
        self.emb_dim = emb_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


def nt_xent_room(z: torch.Tensor, room_index: torch.Tensor,
                 tau: float = TAU) -> torch.Tensor:
    """Clip<->clip NT-Xent where positives share a room (no centroids).

    z:           [B, d] unit embeddings
    room_index:  [B]    which room each clip belongs to
    """
    B = z.shape[0]
    if B < 4:
        return z.new_tensor(0.0, requires_grad=True)

    sim = (z @ z.t()) / tau                                   # [B, B]
    # mask self
    eye = torch.eye(B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)

    pos = (room_index.unsqueeze(0) == room_index.unsqueeze(1)) & ~eye  # [B,B]
    pos_count = pos.sum(dim=1).clamp(min=1)

    # log-sum-exp over ALL non-self entries (denominator), then average
    # log-prob over each anchor's positives (multi-positive NT-Xent)
    log_denom = torch.logsumexp(sim, dim=1)                   # [B]
    log_prob = sim - log_denom.unsqueeze(1)                   # [B,B]
    mean_log_prob_pos = (log_prob * pos).sum(dim=1) / pos_count
    return -mean_log_prob_pos.mean()


def spread_regularizer(z: torch.Tensor, room_index: torch.Tensor,
                       collapse_cos: float = COLLAPSE_COS) -> torch.Tensor:
    """Anti-collapse guard, hinge-style: penalize a room only when its
    mean within-room pairwise cosine exceeds `collapse_cos` (i.e. the
    room is collapsing to a point). Natural spread differences — the κ
    temperature signal — are left untouched. (A VICReg variance FLOOR
    was tried first: it homogenized spread and destroyed κ readout;
    see docs/gpu-room-state-embed-2026-09-27.md.)"""
    penalty = z.new_tensor(0.0)
    n_rooms = 0
    for r in room_index.unique():
        zr = F.normalize(z[room_index == r], dim=-1)
        if zr.shape[0] < 2:
            continue
        mean_cos = (zr @ zr.t()).sum() / (zr.shape[0] * (zr.shape[0] - 1))
        penalty = penalty + F.relu(mean_cos - collapse_cos) ** 2
        n_rooms += 1
    return penalty / max(n_rooms, 1)


def contrastive_loss(z: torch.Tensor, room_index: torch.Tensor,
                     tau: float = TAU,
                     spread_lambda: float = SPREAD_LAMBDA
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Total v0 objective: NT-Xent + within-room spread guard."""
    l_nce = nt_xent_room(z, room_index, tau)
    l_spread = spread_regularizer(z, room_index)
    return l_nce + spread_lambda * l_spread, l_nce


def make_batches(room_index: torch.Tensor, clips_per_room: int,
                 rooms_per_batch: int = 3, shuffle: bool = True,
                 generator: torch.Generator | None = None) -> List[torch.Tensor]:
    """Spec §2.3 batch structure: each batch = ALL clips from
    `rooms_per_batch` rooms."""
    n_rooms = room_index.max().item() + 1
    room_ids = torch.randperm(n_rooms, generator=generator) if shuffle \
        else torch.arange(n_rooms)
    batches = []
    for i in range(0, n_rooms, rooms_per_batch):
        rooms = room_ids[i:i + rooms_per_batch]
        idx = torch.cat([
            torch.arange(int(r) * clips_per_room, (int(r) + 1) * clips_per_room)
            for r in rooms
        ])
        batches.append(idx)
    return batches
