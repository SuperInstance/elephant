"""gpu — the GPU lane: room-state embedding prototype (v0, synthetic).

Self-contained prototype of the elephant-sense v3 direction
(docs/elephant-sense-v3-design.md): contrastively trained room-state
embeddings, acclimation curves (agent -> room), charisma as measurable
pull (room -> agent) — on SYNTHETIC room-field data with known ground
truth, trained on the RTX 4050.

Honest scope: this module touches no fleet corpora. It proves the
machinery and the metric battery; real-room claims stay unclaimed.
See docs/gpu-room-state-embed-2026-09-27.md.
"""
from .model import RoomEncoder, contrastive_loss, make_batches
from .roomgen import build_world

__all__ = ["RoomEncoder", "contrastive_loss", "make_batches", "build_world"]
