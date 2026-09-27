"""gpu.evaluate — the elephant metrics, computed on HELD-OUT rooms.

Implements the v3 spec's evaluation battery (docs/elephant-sense-v3-design.md
§6) against frozen embeddings, on rooms the encoder never trained on:

1. room_discrimination_knn  — 1-NN: is a clip's nearest neighbor from
   its own room? (chance = fraction of the largest room)
2. sauna_plunge_gap         — mean same-room cosine minus mean
   cross-room cosine (the felt "different elephant")
3. silhouette               — room-identity silhouette on cosine
   distance (numpy implementation; no sklearn dependency)
4. coldwarm_probe           — logistic probe on frozen embeddings,
   cold vs warm, trained on TRAIN rooms, tested on HELD-OUT rooms;
   also probed on RAW observations as an honest baseline
5. kappa_correlation        — Pearson r between within-room embedding
   spread and the room's true clip_sigma (the room-temperature
   metric: can it feel how loose a room is?)
6. acclimation              — τ recovery (fit exp decay to
   distance-to-room-mean over turns) + rank-curve convergence
7. charisma                 — room-mean displacement along the agent
   direction after a charismatic pass vs matched controls (Cohen's d)

Every function is pure given (embeddings, labels) so the same battery
runs on the UNTRAINED encoder (the frozen baseline) and the trained
one — the before/after pair is the headline.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .model import RoomEncoder
from .roomgen import OBS_DIM, World, warmth_label


# ---------------------------------------------------------------------- #
# Embedding helpers                                                      #
# ---------------------------------------------------------------------- #
@torch.no_grad()
def encode(encoder: RoomEncoder, obs: torch.Tensor,
           device: torch.device, batch: int = 256) -> np.ndarray:
    encoder.eval()
    outs = []
    for i in range(0, obs.shape[0], batch):
        x = obs[i:i + batch].to(device)
        outs.append(encoder(x).cpu().numpy())
    return np.concatenate(outs, axis=0)


def _pairwise_cos(X: np.ndarray) -> np.ndarray:
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    return Xn @ Xn.T


# ---------------------------------------------------------------------- #
# 1-3: room discrimination, sauna/plunge gap, silhouette                 #
# ---------------------------------------------------------------------- #
def room_discrimination_knn(emb: np.ndarray, room_ids: Sequence[str]) -> float:
    S = _pairwise_cos(emb)
    np.fill_diagonal(S, -2.0)
    nn_room = np.asarray(room_ids)[S.argmax(axis=1)]
    return float((nn_room == np.asarray(room_ids)).mean())


def sauna_plunge_gap(emb: np.ndarray, room_ids: Sequence[str]) -> float:
    S = _pairwise_cos(emb)
    ids = np.asarray(room_ids)
    same = (ids[:, None] == ids[None, :])
    np.fill_diagonal(S, np.nan)
    same = same & ~np.eye(len(ids), dtype=bool)
    cross = ~same & ~np.eye(len(ids), dtype=bool)
    return float(np.nanmean(S[same]) - np.nanmean(S[cross]))


def silhouette_cosine(emb: np.ndarray, room_ids: Sequence[str]) -> float:
    """Standard silhouette with cosine distance. O(n^2); fine at the
    prototype's eval sizes (<= ~1500 clips)."""
    Xn = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    D = 1.0 - Xn @ Xn.T
    np.fill_diagonal(D, 0.0)
    ids = np.asarray(room_ids)
    uniq = np.unique(ids)
    sil = np.zeros(len(ids))
    for i in range(len(ids)):
        own = ids == ids[i]
        own[i] = False
        others = ids != ids[i]
        if own.sum() == 0 or len(np.unique(ids[others])) == 0:
            continue
        a = D[i, own].mean()
        b = min(D[i, (ids == r)].mean() for r in uniq if r != ids[i])
        sil[i] = (b - a) / max(a, b) if max(a, b) > 1e-12 else 0.0
    return float(sil.mean())


# ---------------------------------------------------------------------- #
# 4: cold/warm linear probe (train rooms -> held-out rooms)              #
# ---------------------------------------------------------------------- #
def _fit_logistic(X: np.ndarray, y: np.ndarray, iters: int = 400,
                  lr: float = 0.05, seed: int = 0) -> Tuple[np.ndarray, float]:
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    w = (0.01 * torch.randn(X.shape[1], generator=g)).requires_grad_(True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        logits = Xt @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, yt)
        loss.backward()
        opt.step()
    return w.detach().numpy(), float(b.detach()[0])


def coldwarm_probe(train_emb: np.ndarray, train_labels: np.ndarray,
                   test_emb: np.ndarray, test_labels: np.ndarray,
                   seed: int = 0) -> float:
    w, b = _fit_logistic(train_emb, train_labels, seed=seed)
    scores = test_emb @ w + b
    pred = (scores > 0).astype(int)
    return float((pred == test_labels).mean())


# ---------------------------------------------------------------------- #
# 5: room temperature (κ proxy correlation)                              #
# ---------------------------------------------------------------------- #
def kappa_correlation(emb_by_room: Dict[str, np.ndarray],
                      true_sigma_by_room: Dict[str, float]) -> float:
    """Pearson r between within-room embedding spread (1 - mean pairwise
    cosine) and the room's true clip_sigma."""
    rooms = sorted(emb_by_room)
    if len(rooms) < 3:
        return float("nan")
    spread, truth = [], []
    for r in rooms:
        E = emb_by_room[r]
        S = _pairwise_cos(E)
        n = S.shape[0]
        iu = np.triu_indices(n, k=1)
        spread.append(1.0 - S[iu].mean())
        truth.append(true_sigma_by_room[r])
    spread = np.array(spread)
    truth = np.array(truth)
    if spread.std() < 1e-9 or truth.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(spread, truth)[0, 1])


# ---------------------------------------------------------------------- #
# 6: acclimation — τ recovery + rank convergence                         #
# ---------------------------------------------------------------------- #
def _fit_tau(dist: np.ndarray) -> Tuple[float, float]:
    """Fit d(t) = floor + amp * exp(-t/tau).

    3-parameter exponential fit via scipy curve_fit when available;
    falls back to floor-refined log-linear regression otherwise.
    Returns (tau_hat, r2) with r2 measured in log-above-floor space.
    """
    t = np.arange(len(dist), dtype=float)
    try:
        from scipy.optimize import curve_fit

        def model(t, amp, tau, floor):
            return floor + amp * np.exp(-t / np.maximum(tau, 1e-3))

        p0 = (dist[0] - dist[-1], max(len(dist) / 4.0, 1.0), dist[-1])
        popt, _ = curve_fit(model, t, dist, p0=p0, maxfev=20000)
        amp, tau, floor = popt
        pred = model(t, *popt)
        ss_res = float(((dist - pred) ** 2).sum())
        ss_tot = float(((dist - dist.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return float(tau), float(max(r2, 0.0))
    except ImportError:
        # fallback: log-linear with a conservative floor estimate
        floor = float(dist[-max(len(dist) // 4, 1):].mean())
        y = np.log(np.maximum(dist - floor, 1e-6))
        if y.std() < 1e-9:
            return float("nan"), 0.0
        A = np.vstack([t, np.ones_like(t)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        slope = coef[0]
        if slope >= -1e-9:
            return float("nan"), 0.0
        pred = A @ coef
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return float(-1.0 / slope), float(max(r2, 0.0))


def acclimation_metrics(encoder: RoomEncoder, world: World,
                        device: torch.device) -> Dict[str, float]:
    """For each held-out trace: encode turns, measure cosine distance
    to the room's clip-embedding mean over turns, fit tau. Report the
    correlation between tau_hat and tau_true across traces and the
    mean fit R^2. Rank-curve convergence: fraction of traces whose
    last-third mean rank among room occupants beats the first-third."""
    taus_hat, taus_true, r2s, rank_gains = [], [], [], []
    for tr in world.acclimation:
        room = next(r for r in world.eval_rooms if r.room_id == tr.room_id)
        room_emb = encode(encoder, room.obs, device)
        room_mean = room_emb.mean(axis=0)
        room_mean /= max(np.linalg.norm(room_mean), 1e-9)

        turn_emb = encode(encoder, tr.turn_obs, device)
        turn_emb = turn_emb / np.maximum(
            np.linalg.norm(turn_emb, axis=1, keepdims=True), 1e-9)
        dist = 1.0 - turn_emb @ room_mean

        tau_hat, r2 = _fit_tau(dist)
        if not math.isnan(tau_hat):
            taus_hat.append(tau_hat)
            taus_true.append(tr.tau_true)
            r2s.append(r2)

        # percentile rank among room occupants (spec §3 observable):
        # a turn's rank = fraction of occupant clips whose cosine-to-mean
        # it beats. Invariant to drift/talkativeness/noise-floor.
        cos_to_mean = room_emb @ room_mean
        occ_sorted = np.sort(cos_to_mean)
        turn_scores = turn_emb @ room_mean
        turn_rank = np.searchsorted(occ_sorted, turn_scores) / len(occ_sorted)
        n3 = max(len(turn_rank) // 3, 1)
        rank_gains.append(float(turn_rank[-n3:].mean() - turn_rank[:n3].mean()))

    out: Dict[str, float] = {
        "n_fitted": float(len(taus_hat)),
        "mean_fit_r2": float(np.mean(r2s)) if r2s else float("nan"),
        "mean_rank_gain": float(np.mean(rank_gains)) if rank_gains else float("nan"),
    }
    if len(taus_hat) >= 3:
        lt = np.log(np.maximum(np.array(taus_hat), 1e-3))
        ltrue = np.log(np.array(taus_true))
        out["log_tau_correlation"] = float(np.corrcoef(lt, ltrue)[0, 1])
    else:
        out["log_tau_correlation"] = float("nan")
    return out


# ---------------------------------------------------------------------- #
# 7: charisma shift                                                       #
# ---------------------------------------------------------------------- #
def charisma_metrics(encoder: RoomEncoder, world: World,
                     device: torch.device) -> Dict[str, float]:
    """Spec §4 observable: displacement of the room-mean embedding
    along the agent's entry direction, last quarter of the pass minus
    first quarter. Compare charisma rooms vs matched controls."""
    def _shift(room) -> float:
        emb = encode(encoder, room.obs, device)
        n4 = max(len(emb) // 4, 1)
        early = emb[:n4].mean(axis=0)
        late = emb[-n4:].mean(axis=0)
        # agent direction: the room's own drift defines it; for controls
        # the room is static so the projection is pure noise. Use the
        # empirical direction (late - early) projected consistently:
        # measure ||late - early|| — displacement magnitude.
        return float(np.linalg.norm(late - early))

    chi = np.array([_shift(r) for r in world.charisma_rooms])
    ctl = np.array([_shift(r) for r in world.control_rooms])
    pooled = math.sqrt(
        ((len(chi) - 1) * chi.var(ddof=1) + (len(ctl) - 1) * ctl.var(ddof=1))
        / (len(chi) + len(ctl) - 2)
    ) if (len(chi) + len(ctl) > 2) else float("nan")
    return {
        "charisma_displacement_mean": float(chi.mean()),
        "control_displacement_mean": float(ctl.mean()),
        "cohens_d": float((chi.mean() - ctl.mean()) / pooled) if pooled and pooled > 1e-12 else float("nan"),
    }


# ---------------------------------------------------------------------- #
# The full battery                                                       #
# ---------------------------------------------------------------------- #
def evaluate_battery(encoder: RoomEncoder, world: World,
                     device: torch.device) -> Dict[str, float]:
    """All elephant metrics on held-out rooms, frozen encoder."""
    eval_obs = torch.cat([r.obs for r in world.eval_rooms])
    eval_ids = sum([[r.room_id] * r.obs.shape[0] for r in world.eval_rooms], [])
    emb = encode(encoder, eval_obs, device)

    knn = room_discrimination_knn(emb, eval_ids)
    gap = sauna_plunge_gap(emb, eval_ids)
    sil = silhouette_cosine(emb, eval_ids)

    # cold/warm probe: train on TRAIN rooms, test on HELD-OUT rooms
    tr_obs = torch.cat([r.obs for r in world.train_rooms])
    tr_emb = encode(encoder, tr_obs, device)
    tr_lab, tr_keep = [], []
    for i, r in enumerate(world.train_rooms):
        lab = warmth_label(r)
        if lab is not None:
            tr_lab += [lab] * r.obs.shape[0]
            tr_keep += list(range(i * r.obs.shape[0], (i + 1) * r.obs.shape[0]))
    te_lab, te_keep = [], []
    for i, r in enumerate(world.eval_rooms):
        lab = warmth_label(r)
        if lab is not None:
            te_lab += [lab] * r.obs.shape[0]
            te_keep += list(range(i * r.obs.shape[0], (i + 1) * r.obs.shape[0]))
    keep_tr = np.array(tr_keep)
    keep_te = np.array(te_keep)
    probe_emb = coldwarm_probe(
        tr_emb[keep_tr], np.array(tr_lab), emb[keep_te], np.array(te_lab))
    # honest baseline: the same probe on RAW sensor views
    probe_raw = coldwarm_probe(
        tr_obs.numpy()[keep_tr], np.array(tr_lab),
        eval_obs.numpy()[keep_te], np.array(te_lab))

    emb_by_room = {r.room_id: emb[i * r.obs.shape[0]:(i + 1) * r.obs.shape[0]]
                   for i, r in enumerate(world.eval_rooms)}
    kap = kappa_correlation(
        emb_by_room, {r.room_id: r.clip_sigma for r in world.eval_rooms})

    chance = max(
        np.unique(np.array(eval_ids), return_counts=True)[1].max()
        / len(eval_ids), 1e-9)

    return {
        "n_eval_clips": float(len(eval_ids)),
        "n_eval_rooms": float(len(world.eval_rooms)),
        "chance_knn": float(chance),
        "room_discrimination_knn": knn,
        "sauna_plunge_gap": gap,
        "silhouette": sil,
        "coldwarm_probe_embedding": probe_emb,
        "coldwarm_probe_raw_baseline": probe_raw,
        "kappa_correlation": kap,
    }
