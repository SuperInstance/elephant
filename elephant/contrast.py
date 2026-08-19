"""elephant.contrast — the v3 room-contrast objective and probe metrics.

Implements elephant-sense-v3 §2.3 and the registered evaluation harness
(devils-advocate §3) at BOTH tiers:

  * **audio tier** — the frozen audio-JEPA v2 encoder's 384-dim clip
    embeddings (fleet-jepa-midi ConvEncoder; the tier the 0.015/0.271 probe
    numbers and the deadman were registered on). The encoder itself lives in
    the fleet-jepa-midi repo; this module consumes its embeddings.
  * **text tier** — the elephant `learned.py` TextEncoder trunk
    (checkpoints/learned_dials.pt), the v2-era learned encoder this head
    extends. Not a rewrite: the trunk is loaded and fine-tuned.

Objective (v3 §2.3, adopted from review — verbatim semantics):
  * hierarchical clip↔clip contrast, SimCLR-style with multi-positive
    InfoNCE: **anchor = a clip**, positives = *other clips from the same
    room*, negatives = clips from *other rooms*. No centroid is ever an
    anchor.
  * batch = **all clips from 2–3 rooms** (many within-room positives, a
    bounded negative set).
  * temperature τ = 0.15, fixed (registered).
  * explicit **within-room spread regularizer**: hinge penalty if a room's
    mean pairwise distance (1−cos) drops below `slack ×` its *frozen
    baseline* spread — the anti-collapse guard. Targets are measured on the
    frozen encoder BEFORE any training and frozen for the run, so
    "within-room spread preserved" is enforced, not assumed.

Probe metrics replicate `elephant_sense_probe.py` exactly (same
definitions, same kNN, same speaker-holdout) so before/after numbers are
apples-to-apples with checkpoints/elephant_probe.json.

Also here: `vmf_fit_generic` — a dimension-generic vMF MLE (the shipped
`vmf.py` is dial-space d=7 by design and is NOT touched). Used for
condition-level (sub-room) field fits of embedding windows, with a
message-level bootstrap SE that re-derives the drift deadband at the
condition estimator's scale (the deadband lesson: signal lives at condition
level, not message grain).
"""
from __future__ import annotations

import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .learned import Vocab, parse_document, tokenize

try:
    import torch
    import torch.nn.functional as F

    _TORCH = True
except Exception:  # pragma: no cover
    torch = F = None  # type: ignore
    _TORCH = False

TAU = 0.15            # v3 §2.3 — fixed, registered
SPREAD_SLACK = 0.9    # spread may tighten to 90% of frozen baseline, no further
SPREAD_LAMBDA = 5.0   # hinge weight (fixed a priori for all runs)


# --------------------------------------------------------------------- #
# Clips and corpora                                                     #
# --------------------------------------------------------------------- #
@dataclass
class Clip:
    """One contrast anchor. `key` is a stable id, `speaker` the holdout key."""
    room: str
    key: str
    speaker: str


@dataclass
class ClipBatch:
    clips: List[Clip]
    ids: torch.Tensor          # token ids / feature index — tier-agnostic use


def contrast_loss(
    z: "torch.Tensor",
    rooms: Sequence[str],
    tau: float = TAU,
) -> "torch.Tensor":
    """Multi-positive InfoNCE, anchor = clip (never a centroid).

    z: [B, d] (any scale; normalized inside). rooms: length-B room labels.
    For each anchor i: logits_j = cos(z_i, z_j)/tau over j != i;
    loss_i = -mean_pos log softmax. Batch must contain >= 2 rooms and every
    room >= 2 clips (positives must exist for every anchor).
    """
    zn = F.normalize(z, dim=-1)
    sim = zn @ zn.t() / tau
    B = z.shape[0]
    uniq = {r: i for i, r in enumerate(sorted(set(rooms)))}
    rid = torch.tensor([uniq[r] for r in rooms], dtype=torch.long, device=z.device)
    same = rid[:, None] == rid[None, :]
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    neg_mask = ~same & ~eye               # candidate logits: everything else
    pos_mask = same & ~eye                # positives: same room, other clips
    if not bool(pos_mask.any(dim=1).all()):
        raise ValueError("contrast batch: every clip needs >=1 same-room partner")
    # mask self to -inf so it never wins the softmax
    logits = sim.masked_fill(eye, float("-inf"))
    logp = torch.log_softmax(logits, dim=1)
    pos_terms = (logp.masked_fill(~pos_mask, 0.0)).sum(1) / pos_mask.sum(1).clamp(min=1)
    return -pos_terms.mean()


def spread_hinge(
    z: "torch.Tensor",
    rooms: Sequence[str],
    targets: Dict[str, float],
    slack: float = SPREAD_SLACK,
    weight: float = SPREAD_LAMBDA,
) -> "torch.Tensor":
    """Anti-collapse guard: within-room spread must stay >= slack*target."""
    zn = F.normalize(z, dim=-1)
    loss = z.new_zeros(())
    uniq = sorted(set(rooms))
    for r in uniq:
        if r not in targets:
            continue
        idx = (torch.tensor([1 if rr == r else 0 for rr in rooms],
                            dtype=torch.bool, device=z.device)).nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        zr = zn[idx]
        sim = zr @ zr.t()
        n = zr.shape[0]
        spread = (1.0 - sim).sum() / (n * (n - 1))   # mean pairwise 1-cos
        loss = loss + weight * F.relu(slack * targets[r] - spread) ** 2
    return loss


def room_spread(z: np.ndarray, rooms: Sequence[str]) -> Dict[str, float]:
    """Mean pairwise (1-cos) within each room — the collapse gauge."""
    zn = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-9, None)
    out: Dict[str, float] = {}
    for r in sorted(set(rooms)):
        idx = [i for i, rr in enumerate(rooms) if rr == r]
        if len(idx) < 2:
            continue
        s = zn[idx] @ zn[idx].T
        n = len(idx)
        out[r] = float((1.0 - s).sum() / (n * (n - 1)))
    return out


# --------------------------------------------------------------------- #
# The registered probe metrics (replicates elephant_sense_probe.py)      #
# --------------------------------------------------------------------- #
def _cosine_matrix(z: np.ndarray) -> np.ndarray:
    zn = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-9, None)
    return zn @ zn.T


def separability(z: np.ndarray, rooms: Sequence[str]) -> Dict[str, float]:
    """Same-room vs cross-room cosine: the FINE GAP lives here (probe-def)."""
    sim = _cosine_matrix(z)
    same, cross = [], []
    for a in range(len(rooms)):
        for b in range(a + 1, len(rooms)):
            (same if rooms[a] == rooms[b] else cross).append(sim[a, b])
    same = np.array(same)
    cross = np.array(cross)
    return {
        "same_room_mean": float(same.mean()),
        "cross_room_mean": float(cross.mean()),
        "gap": float(same.mean() - cross.mean()),
        "cross_sigma": float(cross.std()),
        "n_same": int(len(same)),
        "n_cross": int(len(cross)),
    }


def cross_group_gap(z: np.ndarray, idx_a: Sequence[int], idx_b: Sequence[int],
                    rooms: Sequence[str]) -> Dict[str, float]:
    """Coarse gap: mean within-A same-room cosine minus mean A-vs-B cosine.

    Registered coarse anchor = speech (A: the four tap rooms) vs music
    (B: the cold-plunge rooms) at the audio tier.
    """
    sim = _cosine_matrix(z)
    within = [sim[a, b] for ia, a in enumerate(idx_a)
              for b in idx_a[ia + 1:] if rooms[a] == rooms[b]]
    ab = [sim[a, b] for a in idx_a for b in idx_b]
    within = np.array(within)
    ab = np.array(ab)
    return {
        "within_a_mean": float(within.mean()),
        "cross_ab_mean": float(ab.mean()),
        "gap": float(within.mean() - ab.mean()),
        "cross_sigma": float(ab.std()),
        "n_within": int(len(within)),
        "n_cross": int(len(ab)),
    }


def room_discrimination(z: np.ndarray, rooms: Sequence[str],
                        speakers: Sequence[str],
                        holdout_speaker: bool = False) -> float:
    """k-NN (k=1) room accuracy over all clips (probe-def).

    holdout_speaker=True removes every clip sharing the query's speaker key
    from the candidate set — the decisive room-vs-voice control.
    """
    sim = _cosine_matrix(z)
    correct = total = 0
    for i in range(len(rooms)):
        cands = [j for j in range(len(rooms)) if j != i]
        if holdout_speaker:
            cands = [j for j in cands if speakers[j] != speakers[i]]
        if not cands:
            continue
        j = max(cands, key=lambda k: sim[i, k])
        correct += rooms[j] == rooms[i]
        total += 1
    return correct / total if total else 0.0


def probe_report(z: np.ndarray, clips: Sequence[Clip],
                 coarse_b_rooms: Sequence[str] = (),
                 ) -> Dict[str, object]:
    """The full registered metric block for one embedding matrix."""
    rooms = [c.room for c in clips]
    speakers = [c.speaker for c in clips]
    fine = separability(z, rooms)
    out: Dict[str, object] = {
        "n_clips": len(clips),
        "n_rooms": len(set(rooms)),
        "room_discrimination": room_discrimination(z, rooms, speakers),
        "room_discrimination_speaker_heldout": room_discrimination(
            z, rooms, speakers, holdout_speaker=True),
        "separability": fine,
        "spread": room_spread(z, rooms),
        "mean_spread": float(np.mean(list(room_spread(z, rooms).values()))),
    }
    if coarse_b_rooms:
        a = [i for i, c in enumerate(clips) if c.room not in coarse_b_rooms]
        b = [i for i, c in enumerate(clips) if c.room in coarse_b_rooms]
        out["coarse"] = cross_group_gap(z, a, b, rooms)
    return out


# --------------------------------------------------------------------- #
# Generic-d vMF fit (embeddings) — does NOT touch dial-space vmf.py     #
# --------------------------------------------------------------------- #
def vmf_fit_generic(X: np.ndarray, seed: int = 0, B: int = 0) -> Optional[dict]:
    """vMF MLE for arbitrary-d unit data. μ̂/ρ exact; κ by bisection on
    A_d(κ)=ρ using scipy Bessel ratios. Optional bootstrap SE(μ̂).

    Honesty note carried from vmf.py: at N ≈ 15, d = 384 the raw ρ is
    biased low under uniformity (√(N/d) noise floor) — κ there is a
    lower-bound-ish reading; μ̂ and its bootstrap SE are the quantities the
    condition-level analysis relies on.
    """
    from scipy.special import ive

    X = np.asarray(X, float)
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    N, d = X.shape
    r = X.mean(0)
    rho = float(np.linalg.norm(r))
    if rho < 1e-9:
        return None
    mu = r / rho

    def A(k: float) -> float:
        return float(ive(d / 2.0, k) / ive(d / 2.0 - 1.0, k))

    lo, hi = 1e-6, 1000.0
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if A(mid) < rho:
            lo = mid
        else:
            hi = mid
    kappa = math.sqrt(lo * hi)

    out = {"mu_hat": mu.tolist(), "rho": rho, "kappa": float(kappa),
           "n": int(N), "d": int(d), "mu_se": None}
    if B > 0 and N >= 4:
        rng = np.random.default_rng(seed)
        mus = []
        for _ in range(B):
            Xb = X[rng.integers(0, N, N)]
            rb = Xb.mean(0)
            nb = float(np.linalg.norm(rb))
            if nb > 1e-9:
                mus.append(rb / nb)
        if mus:
            M = np.stack(mus)
            out["mu_se"] = float(np.linalg.norm(M.mean(0) / max(
                np.linalg.norm(M.mean(0)), 1e-12) - mu))
            # mean chord distance is the more honest scale at small N:
            out["mu_se"] = float(np.mean(np.linalg.norm(M - mu, axis=1)))
    return out


def condition_edge(fit_a: Optional[dict], fit_b: Optional[dict],
                   db_factor: float = 2.0) -> Optional[dict]:
    """Displacement between two condition-level fits + re-derived deadband.

    `real` requires ‖Δμ̂‖ > db_factor · max(SE) where SE is the
    message-level bootstrap SE of each condition's μ̂ — the deadband
    re-derived at the condition estimator's scale (message-grain 0/50
    silence answered by construction, not by tuning).
    """
    if not fit_a or not fit_b:
        return None
    mu_a, mu_b = np.array(fit_a["mu_hat"]), np.array(fit_b["mu_hat"])
    d_mu = float(np.linalg.norm(mu_b - mu_a))
    d_cos = float(1.0 - mu_a @ mu_b)
    ses = [s for s in (fit_a.get("mu_se"), fit_b.get("mu_se")) if s]
    db = db_factor * max(ses) if ses else None
    return {
        "d_mu": d_mu,
        "d_cos": d_cos,
        "deadband": db,
        "real": bool(db is not None and d_mu > db),
        "kappa_a": fit_a["kappa"], "kappa_b": fit_b["kappa"],
        "se_a": fit_a.get("mu_se"), "se_b": fit_b.get("mu_se"),
    }


# --------------------------------------------------------------------- #
# Text corpora — the v3 §1.1 rooms as markdown                          #
# --------------------------------------------------------------------- #
def _plurality_author(messages) -> str:
    c = Counter(m.author for m in messages if m.author)
    return c.most_common(1)[0][0] if c else "room"


def text_clips_from_room(name: str, messages, window: int = 8,
                         stride: Optional[int] = None,
                         min_tokens: int = 5) -> List[Tuple[Clip, List[str]]]:
    """Non-overlapping windows as clips (overlap would leak kNN neighbors).

    stride defaults to `window` (non-overlapping). Each clip's speaker key =
    the plurality author of the window (the text-tier holdout key).
    """
    stride = stride or window
    out: List[Tuple[Clip, List[str]]] = []
    msgs = list(messages)
    for wi, i in enumerate(range(0, max(1, len(msgs) - window + 1), stride)):
        chunk = msgs[i: i + window]
        if not chunk:
            continue
        tokens = tokenize(" ".join(m.text for m in chunk))
        if len(tokens) < min_tokens:
            continue
        out.append((Clip(room=name, key=f"{name}#w{wi}",
                         speaker=_plurality_author(chunk)), tokens))
    return out


SCRIPT_SLUG_RE = re.compile(r"^`([^`]+)`\s*$")


def parse_script_blocks(path: str) -> List[Tuple[str, str, str]]:
    """Parse a tap broadcast SCRIPT.md into (slug, speaker, text) blocks.

    The slug (`> `lucineer-intro``) is the mp3 filename stem — this is the
    exact audio↔text alignment key for the fusion head.
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    blocks: List[Tuple[str, str, str]] = []
    author, slug, buf = "room", None, []

    def flush():
        nonlocal slug, buf
        if slug is not None:
            text = re.sub(r"\s+", " ", " ".join(buf)).strip()
            if text:
                blocks.append((slug, author, text))
        slug, buf = None, []

    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#") or s.startswith("|") or s.startswith("---"):
            continue
        m = re.match(r"^\*\*(.+?)\*\*\s*:?\s*(.*)$", s)
        if m and not s.startswith(">"):
            name = m.group(1).split("(", 1)[0].rstrip(" :,.").strip().upper()
            if name and name not in {"ALL"}:
                author = name
            body = re.sub(r"[*`]", "", m.group(2)).strip()
            if body and slug is not None:
                buf.append(body)
            continue
        if s.startswith("> "):
            body = s.lstrip(">").strip()
            sm = SCRIPT_SLUG_RE.match(body)
            if sm:
                flush()
                slug = sm.group(1)
                continue
            body = re.sub(r"[*`]", "", body).strip()
            if body and slug is not None:
                buf.append(body)
    flush()
    return blocks


def speaker_key_from_filename(filename: str) -> str:
    """Probe-def speaker key: leading token before the first '-'."""
    stem = os.path.basename(filename).rsplit(".", 1)[0].lower()
    stem = re.sub(r"^(episode-\d+[-_])", "", stem)
    return stem.split("-")[0]


# --------------------------------------------------------------------- #
# Batch sampling — batch = ALL clips of 2-3 rooms (v3 §2.3)             #
# --------------------------------------------------------------------- #
def sample_room_batches(rooms: Sequence[str], n_batches: int, rng: random.Random,
                        k_choices: Tuple[int, int] = (2, 3)) -> List[List[int]]:
    """Indices of batches, each = union of all clips of 2 or 3 rooms."""
    by_room: Dict[str, List[int]] = {}
    for i, r in enumerate(rooms):
        by_room.setdefault(r, []).append(i)
    room_names = sorted(by_room)
    eligible = [r for r in room_names if len(by_room[r]) >= 2]
    batches: List[List[int]] = []
    for _ in range(n_batches):
        k = rng.choice(k_choices)
        chosen = rng.sample(eligible, min(k, len(eligible)))
        idx = sorted(i for r in chosen for i in by_room[r])
        batches.append(idx)
    return batches
