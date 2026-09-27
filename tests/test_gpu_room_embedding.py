"""gpu — tests: the room-state embedding prototype (v0, synthetic).

Runs the real thing when torch is present (the venv / --extra 'learned'
install); SKIPS cleanly on the base environment (torch is an optional
dependency — pyproject [project.optional-dependencies] learned=["torch"]).
The rest of the suite never needs torch, so the suite stays green both
ways.

What is checked:
- generator determinism: same seed -> identical world (reproducibility);
- cold/warm structure: warmth_hint produces the labels we think it does;
- the contrastive objective actually descends on a tiny problem;
- the metric battery sanity: silhouette is ~1 for perfect clusters,
  ~<0 for scrambled ones; tau-fit recovers a known time constant;
  the cold/warm probe beats chance on clearly separated embeddings;
- charisma rooms really drift (latent ground truth) and controls don't.
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gpu.evaluate import (
    _fit_tau,
    coldwarm_probe,
    kappa_correlation,
    room_discrimination_knn,
    silhouette_cosine,
    sauna_plunge_gap,
)
from gpu.model import RoomEncoder, contrastive_loss, make_batches
from gpu.roomgen import build_world, make_room, warmth_label


# ---------------------------------------------------------------------- #
# Generator                                                              #
# ---------------------------------------------------------------------- #
def test_build_world_deterministic():
    w1 = build_world(seed=99, n_train_rooms=6, n_eval_rooms=3,
                     n_clips=10, n_acclimation=3, n_charisma_pairs=2)
    w2 = build_world(seed=99, n_train_rooms=6, n_eval_rooms=3,
                     n_clips=10, n_acclimation=3, n_charisma_pairs=2)
    assert torch.equal(w1.train_rooms[0].obs, w2.train_rooms[0].obs)
    assert torch.equal(w1.eval_rooms[2].obs, w2.eval_rooms[2].obs)
    assert w1.acclimation[0].tau_true == w2.acclimation[0].tau_true


def test_build_world_different_seeds_differ():
    w1 = build_world(seed=99, n_train_rooms=4, n_eval_rooms=2, n_clips=8,
                     n_acclimation=2, n_charisma_pairs=1)
    w2 = build_world(seed=100, n_train_rooms=4, n_eval_rooms=2, n_clips=8,
                     n_acclimation=2, n_charisma_pairs=1)
    assert not torch.equal(w1.train_rooms[0].obs, w2.train_rooms[0].obs)


def test_warmth_labels_abstain_band():
    gen = torch.Generator().manual_seed(5)
    from gpu.roomgen import SensorMixture, sample_room_latent, clip_sigma_from_latent
    mixture = SensorMixture(seed=5)
    cold = make_room("c", gen, mixture, 4, warmth_hint=-0.8)
    warm = make_room("w", gen, mixture, 4, warmth_hint=0.8)
    mid = make_room("m", gen, mixture, 4, warmth_hint=0.0)
    assert warmth_label(cold) == 0
    assert warmth_label(warm) == 1
    assert warmth_label(mid) is None


def test_warm_rooms_are_looser_than_cold():
    """The v3 temperature prior: cold = tight, warm = loose."""
    from gpu.roomgen import SensorMixture
    gen = torch.Generator().manual_seed(7)
    mixture = SensorMixture(seed=7)
    cold_sigmas = [make_room(f"c{i}", gen, mixture, 2, warmth_hint=-0.9).clip_sigma
                   for i in range(5)]
    warm_sigmas = [make_room(f"w{i}", gen, mixture, 2, warmth_hint=0.9).clip_sigma
                   for i in range(5)]
    assert np.mean(warm_sigmas) > np.mean(cold_sigmas)


def test_charisma_latents_drift_controls_do_not():
    gen = torch.Generator().manual_seed(11)
    from gpu.roomgen import SensorMixture, make_room_with_charisma_pass
    mixture = SensorMixture(seed=11)
    chi = make_room_with_charisma_pass("chi", gen, mixture, 40, chi_max=0.5)
    ctl = make_room_with_charisma_pass("ctl", gen, mixture, 40, chi_max=0.0)
    chi_drift = float((chi.latent_paths[-1] - chi.spec.latent).norm())
    ctl_drift = float((ctl.latent_paths[-1] - ctl.spec.latent).norm())
    assert chi_drift > 2 * ctl_drift  # the room really moved; control is jitter


# ---------------------------------------------------------------------- #
# Objective                                                              #
# ---------------------------------------------------------------------- #
def test_contrastive_loss_descends_on_tiny_problem():
    torch.manual_seed(3)
    world = build_world(seed=3, n_train_rooms=8, n_eval_rooms=4,
                        n_clips=12, n_acclimation=2, n_charisma_pairs=1)
    device = torch.device("cpu")
    enc = RoomEncoder()
    opt = torch.optim.Adam(enc.parameters(), lr=1e-2)
    obs = torch.cat([r.obs for r in world.train_rooms])
    ri = torch.cat([torch.full((r.obs.shape[0],), i)
                    for i, r in enumerate(world.train_rooms)])
    enc.train()
    first = last = None
    for _ in range(60):
        for idx in make_batches(ri, 12, rooms_per_batch=2,
                                generator=torch.Generator().manual_seed(1)):
            z = enc(obs[idx])
            loss, _ = contrastive_loss(z, ri[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if first is None:
                first = float(loss.item())
            last = float(loss.item())
    assert last < first * 0.9, (first, last)


def test_make_batches_cover_all_clips_exactly_once():
    ri = torch.arange(10 * 16) // 16
    batches = make_batches(ri, clips_per_room=16, rooms_per_batch=3,
                           generator=torch.Generator().manual_seed(0))
    flat = torch.cat(batches)
    assert len(flat) == 160
    assert len(set(flat.tolist())) == 160


# ---------------------------------------------------------------------- #
# Metric battery                                                         #
# ---------------------------------------------------------------------- #
def test_silhouette_perfect_vs_scrambled():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(20, 8)) + np.array([3.0] * 8)
    B = rng.normal(size=(20, 8)) - np.array([3.0] * 8)
    emb = np.vstack([A, B])
    ids = ["a"] * 20 + ["b"] * 20
    assert silhouette_cosine(emb, ids) > 0.9
    # scramble labels -> silhouette collapses
    shuffled = list(ids)
    rng.shuffle(shuffled)
    assert silhouette_cosine(emb, shuffled) < 0.3


def test_knn_and_gap_on_perfect_clusters():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(10, 4)) + 2.0
    B = rng.normal(size=(10, 4)) - 2.0
    emb = np.vstack([A, B])
    ids = ["a"] * 10 + ["b"] * 10
    assert room_discrimination_knn(emb, ids) == 1.0
    assert sauna_plunge_gap(emb, ids) > 0.5


def test_tau_fit_recovers_known_constant():
    t = np.arange(30, dtype=float)
    tau_true = 6.0
    dist = 0.5 * np.exp(-t / tau_true) + 0.05
    tau_hat, r2 = _fit_tau(dist)
    assert abs(tau_hat - tau_true) / tau_true < 0.15
    assert r2 > 0.95


def test_coldwarm_probe_generalizes_across_rooms():
    rng = np.random.default_rng(2)
    # two clouds, "train rooms" and disjoint "eval rooms" from the same clouds
    tr = np.vstack([rng.normal(size=(30, 6)) + 2.0,
                    rng.normal(size=(30, 6)) - 2.0])
    tr_y = np.array([1] * 30 + [0] * 30)
    te = np.vstack([rng.normal(size=(20, 6)) + 2.0,
                    rng.normal(size=(20, 6)) - 2.0])
    te_y = np.array([1] * 20 + [0] * 20)
    acc = coldwarm_probe(tr, tr_y, te, te_y)
    assert acc > 0.9


def test_kappa_correlation_tracks_true_spread():
    """The κ metric reads ANGULAR spread (embeddings live on the unit
    sphere — radial scale is invisible by design). Rooms are mean
    direction + angular noise scaled by the true spread."""
    rng = np.random.default_rng(3)
    emb_by_room, sigma_by_room = {}, {}
    mu = rng.normal(size=8)
    for i, s in enumerate([0.2, 0.6, 1.0, 1.4, 1.8]):
        pts = np.tile(mu, (25, 1)) + s * rng.normal(size=(25, 8))
        emb_by_room[f"r{i}"] = pts
        sigma_by_room[f"r{i}"] = s
    assert kappa_correlation(emb_by_room, sigma_by_room) > 0.9


# ---------------------------------------------------------------------- #
# End-to-end smoke (tiny, CPU): the whole train+eval path runs           #
# ---------------------------------------------------------------------- #
def test_end_to_end_smoke_cpu():
    from gpu.train import train as train_fn
    from gpu.evaluate import evaluate_battery
    world = build_world(seed=42, n_train_rooms=8, n_eval_rooms=4,
                        n_clips=12, n_acclimation=2, n_charisma_pairs=1)
    torch.manual_seed(42)
    enc = RoomEncoder()
    device = torch.device("cpu")
    base = evaluate_battery(enc, world, device)
    diag = train_fn(world, enc, device, epochs=10, lr=3e-3,
                    rooms_per_batch=2, seed=42)
    after = evaluate_battery(enc, world, device)
    assert diag["loss_last"] < diag["loss_first"]
    assert 0.0 <= base["room_discrimination_knn"] <= 1.0
    assert 0.0 <= after["room_discrimination_knn"] <= 1.0
    # NOTE (booked in docs): on this synthetic geometry the UNTRAINED
    # encoder is already near-ceiling for instance-level kNN, so we
    # assert no catastrophic regression rather than strict improvement;
    # the training win lives in sauna_plunge_gap / silhouette.
    assert after["room_discrimination_knn"] >= base["room_discrimination_knn"] - 0.05
    assert after["sauna_plunge_gap"] > base["sauna_plunge_gap"]
