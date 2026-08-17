"""The nudge — dial numbers steering what the vision model compares.

The JEPA doesn't replace the vision model on the radar or sounder
screen. It CORRELATES: every sensor is a dial with a NUMBER, and those
numbers nudge the vision model at what to compare together. A high
sounder biomass plus rising radar coherence says: look at the water
column under the cluster, compare this hour to last week's good hour.
A flat sounder says: don't burn attention there.

This module is the framework hook: dial readings -> an attention prior
that any vision/correlation model can consume. The prior is a small
vector per modality, signed by convention (+ = pay attention here).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

MODALITIES = [
    "radar", "sounder", "camera_out", "camera_deck",
    "nav", "autopilot", "conversation",
]

# Which dial readings feed which modality's nudge, and with what sign.
NUDGE_MAP = {
    "radar_coherence": ("radar", +1.0),     # tight fleet -> compare radar tracks
    "sounder_biomass": ("sounder", +1.0),   # biomass -> watch the column
    "fishing_day": ("nav", +0.5),           # good day -> hold course patterns
    "mood": ("conversation", +1.0),         # warm room -> listen to the talk
    "volume": ("camera_deck", +0.5),        # loud deck -> look at the deck
    "panic": ("camera_out", +1.0),          # alarm -> look OUT, now
    "presence": ("camera_deck", +0.3),      # bodies on deck -> watch them
}


def nudge_prior(readings: Dict[str, float],
                modalities: Optional[Sequence[str]] = None) -> np.ndarray:
    """Dial readings -> attention prior over modalities.

    Each modality gets a nudge in [-1, 1]. The vision model multiplies
    or adds this into its cross-attention so it compares the frames
    the elephant says matter. Zero = no opinion = compare as usual.
    """
    mods = list(modalities or MODALITIES)
    prior = np.zeros(len(mods), dtype=float)
    for dial_name, value in readings.items():
        if dial_name not in NUDGE_MAP:
            continue
        mod, sign = NUDGE_MAP[dial_name]
        if mod not in mods:
            continue
        idx = mods.index(mod)
        prior[idx] += sign * float(np.clip(value, -1, 1))
    # normalize so the strongest opinion is at most 1.0
    m = float(np.max(np.abs(prior))) if len(prior) else 0.0
    if m > 1.0:
        prior = prior / m
    return prior


def apply_nudge(attention: np.ndarray, prior: np.ndarray,
                strength: float = 0.15) -> np.ndarray:
    """Blend a nudge prior into existing cross-attention weights.

    `attention` shape (..., n_modalities). Small strength by default —
    the elephant nudges, it doesn't drive.
    """
    a = np.asarray(attention, dtype=float)
    p = np.asarray(prior, dtype=float).reshape(-1)
    if a.shape[-1] != p.shape[0]:
        raise ValueError(f"attention last dim {a.shape[-1]} != prior {p.shape[0]}")
    return a * (1.0 + strength * p)


def describe(prior: np.ndarray, modalities: Optional[Sequence[str]] = None) -> str:
    mods = list(modalities or MODALITIES)
    parts = [f"{m}={v:+.2f}" for m, v in zip(mods, prior) if abs(v) > 1e-6]
    return "nudge[" + ", ".join(parts) + "]" if parts else "nudge[] (no opinion)"
