#!/usr/bin/env python3
"""scripts/encoder_generalization_prototype.py — toy geometry demo for the
encoder-generalization objective (design: encoder_generalization_design.md).

Question (from RESEARCH-NOTE-MEMORIZATION-GEOMETRY-2026-08-19): the contrast
head's multi-positive InfoNCE built 17 memorized point-attractors (a Voronoi
tessellation); clips of UNSEEN nights fracture across / diffuse between the
attractors (held-out spread 4.5-7.6x train, held-out fine gaps 3 seeds vs the
0.05 floor: FAIL). Does making positives *independent stochastic views* of the
same night (split-half + fresh sampling, the token-dropout analog) turn the
point-attractors into smooth basins?

Toy analog (CPU-only; NOT the real head, NOT real text):
  * night centroid      m = u_cast + 0.55 * topic        in R^20
    (cast dirs ~orthogonal => same-cast nights sit ~2x closer than cross-cast)
  * clip = m + 0.35 * eps  (bag-of-words sampling noise around the centroid)
  * corpus: 5 casts x 3 train nights + the "tap" cast (nights 1, 2, 'mic' in
    train; nights 3, 4 HELD OUT and never trained on) = 18 train / 2 held-out
    docs -- same 17-ish-train / 2-held-out shape as the honest test.
  * encoder = over-capacity MLP R^20 -> R^8, unit-norm, cosine InfoNCE
    (tau = 0.15), batch = all clips of 2-3 docs (anchors = clips, as shipped).
  * REGIME plain = the CURRENT objective: one FIXED clip set sampled once;
    positives = other clips of the same doc (elephant/contrast.py semantics).
  * REGIME views = the PROPOSED objective: clips RESAMPLED fresh every step,
    split into two disjoint halves; a view's positives = the OTHER half of
    the same night only (content-disjoint independent views); negatives =
    clips of other docs (UNCHANGED).
  * eval on FRESH held-out clips: fine gap = mean within-night cos minus
    cross-night (3 vs 4) cos, floor 0.05, 3 seeds, PASS = 3/3 (the registered
    rule); plus spread ratio = held-out within-night spread / train spread.

Honest scope: this is a controlled geometry demonstration, not a field result
(see design doc section "what the prototype does not prove").
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-only, repo convention

import numpy as np
import torch
import torch.nn.functional as F

TAU, FLOOR = 0.15, 0.05          # registered temperature, registered floor
D_IN, D_EMB = 20, 8
CLIPS_PER_DOC, HELD_CLIPS = 12, 24
CLIP_SIGMA, TOPIC_SCALE, QUIRK_SCALE = 0.35, 0.9, 0.25
NOVELTY_SCALE, TAP_TRAIN_NIGHTS = 0.5, 12
THEME_DIM = 2
STEPS, LR, SEEDS = 2000, 1e-3, (0, 1, 2)


def unit(v):
    return v / np.linalg.norm(v)


def rand_ortho(rng, basis, d=D_IN):
    """Random unit vector orthogonal to the given orthonormal basis."""
    v = rng.normal(size=d)
    for b in basis:
        v -= (v @ b) * b
    return unit(v)


def build_corpus(rng):
    """24 train night-centroids + 2 held-out (tap cast nights 3,4).

    Each cast has a shared 3-dim THEME subspace (its recurring vocabulary);
    every night -- train or held-out -- is a fresh random mixture of the
    cast's themes plus residual novelty, so unseen nights live in the same
    theme span the encoder can learn from training nights of that cast.
    """
    m = []
    for c in range(6):
        u = rand_ortho(rng, [])                      # cast direction
        themes = [rand_ortho(rng, [u])]              # theme basis (build up)
        while len(themes) < THEME_DIM:
            themes.append(rand_ortho(rng, [u] + themes))

        def night():
            w = rng.normal(size=THEME_DIM)           # fresh theme mixture
            t = sum(wi * ti for wi, ti in zip(w, themes))
            t = t + NOVELTY_SCALE * rand_ortho(rng, [u] + themes)  # + novelty
            return u + TOPIC_SCALE * unit(t)

        m += [night() for _ in range(
            3 if c else TAP_TRAIN_NIGHTS)]            # 3/cast; tap dense
    m += [night() for _ in range(2)]                  # tap 3, 4 HELD OUT
    n_train = 15 + TAP_TRAIN_NIGHTS
    return np.stack(m[:n_train]), np.stack(m[n_train:])


def draw_clips(rng, m, k):
    """k clips of one night: theme signal (in m) + clip-unique idiosyncrasy
    (each window's own novelty, a fresh random direction) + sampling noise."""
    qs = np.stack([rand_ortho(rng, [m]) for _ in range(k)])
    return m + QUIRK_SCALE * qs + CLIP_SIGMA * rng.normal(size=(k, D_IN))


def new_mlp(seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(D_IN, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, D_EMB))


def infonce(z, pos, neg):
    """Multi-positive InfoNCE on cosine; pos/neg = boolean pair masks."""
    zn = F.normalize(z, dim=-1)
    logits = zn @ zn.t() / TAU
    eye = torch.eye(len(z), dtype=torch.bool)
    logits = logits.masked_fill(eye | ~(pos | neg), float("-inf"))
    logp = torch.log_softmax(logits, dim=1)
    return -(logp.masked_fill(~pos, 0.0).sum(1)
             / pos.sum(1).clamp(min=1)).mean()


def batch_docs(rng, n_docs):
    chosen = rng.choice(n_docs, size=int(rng.integers(2, 4)), replace=False)
    return sorted(chosen.tolist())


def train(regime, seed, rng, m_train):
    net = new_mlp(seed)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    if regime == "plain":                       # CURRENT: fixed clip set
        clips = np.stack([draw_clips(rng, m, CLIPS_PER_DOC) for m in m_train])
    for _ in range(STEPS):
        chosen = batch_docs(rng, len(m_train))
        if regime == "plain":
            x = np.concatenate([clips[d] for d in chosen])
            rid = np.repeat(np.arange(len(chosen)), CLIPS_PER_DOC)
            xt = torch.tensor(x, dtype=torch.float32)
        else:                                   # PROPOSED: fresh split-half views
            xs, rid, half = [], [], []
            for di, d in enumerate(chosen):
                fresh = draw_clips(rng, m_train[d], CLIPS_PER_DOC)
                xs.append(fresh)
                rid += [di] * CLIPS_PER_DOC
                half += [0] * (CLIPS_PER_DOC // 2) + [1] * (CLIPS_PER_DOC // 2)
            x = np.concatenate(xs)
            xt = torch.tensor(x, dtype=torch.float32)
            half = np.array(half)
        rid = torch.tensor(rid)
        same = rid[:, None] == rid[None, :]
        eye = torch.eye(len(rid), dtype=torch.bool)
        if regime == "plain":
            pos, neg = same & ~eye, ~same
        else:
            diff_half = torch.tensor(half[:, None] != half[None, :])
            pos, neg = same & diff_half, ~same
        loss = infonce(net(xt), pos, neg)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def embed(net, x):
    with torch.no_grad():
        return F.normalize(net(torch.tensor(x, dtype=torch.float32)),
                           dim=-1).numpy()


def mean_cos(a):
    s = a @ a.T
    n = len(a)
    return float((s.sum() - n) / (n * (n - 1)))


def run(regime, seed):
    rng = np.random.default_rng(seed)
    m_train, m_ho = build_corpus(rng)
    net = train(regime, seed, rng, m_train)
    z3 = embed(net, draw_clips(rng, m_ho[0], HELD_CLIPS))
    z4 = embed(net, draw_clips(rng, m_ho[1], HELD_CLIPS))
    within3, within4 = mean_cos(z3), mean_cos(z4)
    cross = float((z3 @ z4.T).mean())
    gap = 0.5 * (within3 + within4) - cross
    tr = [embed(net, draw_clips(rng, m, HELD_CLIPS)) for m in m_train]
    same_tr = float(np.mean([mean_cos(z) for z in tr]))
    cross_tr = float(np.mean([(tr[i] @ tr[j].T).mean()
                              for i in range(len(tr))
                              for j in range(i + 1, len(tr))]))
    spread_ratio = float(np.mean([1 - within3, 1 - within4])
                         / (1 - same_tr))
    return {"train_gap": same_tr - cross_tr, "gap": gap,
            "within3": within3, "within4": within4, "cross": cross,
            "spread_ratio": spread_ratio}


def main():
    for regime in ("plain", "views"):
        rs = [run(regime, s) for s in SEEDS]
        for s, r in zip(SEEDS, rs):
            verdict = "PASS" if r["gap"] > FLOOR else "FAIL"
            print(f"[{regime}] seed={s}: TRAIN gap={r['train_gap']:+.3f}  "
                  f"HELD-OUT fine gap={r['gap']:+.4f} vs floor {FLOOR}"
                  f" -> {verdict}  (within3={r['within3']:.3f} "
                  f"within4={r['within4']:.3f} cross={r['cross']:.3f}; "
                  f"spread_ratio={r['spread_ratio']:.1f}x train)")
        gaps = [r["gap"] for r in rs]
        n_pass = sum(g > FLOOR for g in gaps)
        tag = ("memorized: point-attractors, unseen nights fracture"
               if n_pass < len(SEEDS) else
               "basins: unseen nights land coherent")
        print(f"[{regime}] VERDICT: held-out fine gaps "
              f"[{', '.join(f'{g:+.4f}' for g in gaps)}] -> "
              f"{n_pass}/{len(SEEDS)} seeds PASS ({tag})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
