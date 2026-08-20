"""scripts/memorization_geometry.py — GEOMETRY of the held-out memorization failure.

Diagnosis-only companion to scripts/contrast_train_text.py (2026-08-19).
The held-out run (train on 17 rooms, hold out tap-night-3 / tap-night-4)
FAILED: fine gaps 0.0713 / 0.1073 / 0.0295 across seeds 0/1/2 vs the 0.05
noise floor. This script retrains nothing; it loads the seed-1 held-out
checkpoint (checkpoints/contrast/text_contrast_heldout_seed1.pt — the seed
whose held-out gap cleared 0.05), embeds ALL clips (training + the 18
held-out), and answers four geometric questions:

  a. nearest-neighbor confusion: which TRAINING room does each held-out
     clip snap to — same-cast nights (tap-night-1/2) or unrelated rooms?
  b. distance structure: within-night-3 / within-night-4 / night3×night4 /
     held-out→tap-train-nights / held-out→unrelated-rooms mean cosine dists
  c. spread/concentration: is the held-out region more diffuse than the
     training-room regions (per-room spread + mean within-room similarity)?
  d. rank defect: singular-value spectrum / effective dimensionality of the
     held-out embedding set vs the training set (PCA participation ratio,
     plus variance captured by the training PCA basis).

CPU-only (CUDA_VISIBLE_DEVICES="" before any torch import). Read-only wrt
checkpoints and elephant/ sources; prints numbers to stdout.
"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before torch import (registered CPU-only)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/

import numpy as np  # noqa: E402

from elephant import contrast  # noqa: E402
from elephant.contrast import text_clips_from_room  # noqa: E402
from elephant.learned import TextEncoder, Vocab  # noqa: E402

from contrast_train_text import (  # noqa: E402
    CKPT, OUT, WINDOW, MAX_LEN, build_text_corpus,
)

HOLDOUT_ROOMS = ["tap-night-3", "tap-night-4"]
SEED = 1  # the held-out seed that cleared the 0.05 noise floor
TAP_TRAIN = ["tap-night-1", "tap-night-2"]


def encode_tokens(encoder, tokens, vocab):
    """Identical to scripts/contrast_train_text.py::encode_tokens."""
    import torch
    ids = [vocab.encode(t, max_len=MAX_LEN) for t in tokens]
    X = torch.zeros((len(ids), MAX_LEN), dtype=torch.long)
    for i, x in enumerate(ids):
        X[i, : len(x)] = torch.tensor(x)
    with torch.no_grad():
        z = torch.nn.functional.normalize(encoder(X), dim=-1)
    return z.numpy()


def sim(a, b):
    """Mean cosine similarity between rows of a and rows of b (normalized)."""
    return float((a @ b.T).mean())


def dist(a, b):
    return 1.0 - sim(a, b)


def eff_rank(X):
    """Participation ratio of the PCA spectrum: (Σλ)²/Σλ², λ = σ² of centered X."""
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2
    return float(lam.sum() ** 2 / (lam ** 2).sum()), s


def main() -> int:
    import torch

    vocab = Vocab.load(os.path.join(CKPT, "learned_vocab.txt"))
    rooms = build_text_corpus()
    train_rooms = [(n, m) for n, m in rooms if n not in HOLDOUT_ROOMS]
    held_rooms = [(n, m) for n, m in rooms if n in HOLDOUT_ROOMS]

    def clips_for(room_list):
        cs, ts = [], []
        for name, msgs in room_list:
            for clip, toks in text_clips_from_room(name, msgs, window=WINDOW):
                cs.append(clip)
                ts.append(toks)
        return cs, ts

    tr_clips, tr_tokens = clips_for(train_rooms)
    ho_clips, ho_tokens = clips_for(held_rooms)
    tr_labels = np.array([c.room for c in tr_clips])
    ho_labels = np.array([c.room for c in ho_clips])
    print(f"train: rooms={len(set(tr_labels))} clips={len(tr_clips)}; "
          f"held-out: rooms={sorted(set(ho_labels))} clips={len(ho_clips)}")

    model = TextEncoder(len(vocab), d_model=64, d_trunk=64)
    sd = torch.load(os.path.join(
        OUT, f"text_contrast_heldout_seed{SEED}.pt"), map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    Ztr = encode_tokens(model, tr_tokens, vocab)   # [Ntr, 64] unit-norm
    Zho = encode_tokens(model, ho_tokens, vocab)   # [Nho, 64] unit-norm
    print(f"embeddings: train {Ztr.shape}, held-out {Zho.shape} "
          f"(norms: train {np.linalg.norm(Ztr, axis=1).mean():.4f}, "
          f"heldout {np.linalg.norm(Zho, axis=1).mean():.4f})")

    # ------------------------------------------------ (a) NN confusion ----
    print("\n=== (a) NEAREST-NEIGHBOR CONFUSION (held-out clip → training clips) ===")
    S = Zho @ Ztr.T                              # [Nho, Ntr] cosine sims
    same_cast = 0
    print(f"{'held-out clip':<28} {'top-1 room':<16} {'sim':>6}  top-3 rooms")
    for i, c in enumerate(ho_clips):
        order = np.argsort(-S[i])
        top1 = tr_labels[order[0]]
        top3 = [f"{tr_labels[j]}({S[i, j]:.3f})" for j in order[:3]]
        if top1 in TAP_TRAIN:
            same_cast += 1
        print(f"{c.key:<28} {top1:<16} {S[i, order[0]]:>6.3f}  {', '.join(top3)}")
    print(f"top-1 = same-cast tap night (1/2): {same_cast}/{len(ho_clips)}")
    top3_cast = sum(
        any(tr_labels[j] in TAP_TRAIN for j in np.argsort(-S[i])[:3])
        for i in range(len(ho_clips)))
    print(f"same-cast night within top-3:      {top3_cast}/{len(ho_clips)}")

    # ------------------------------------------------ (b) distance structure
    print("\n=== (b) DISTANCE STRUCTURE (mean cosine distance) ===")
    m3 = Zho[ho_labels == "tap-night-3"]
    m4 = Zho[ho_labels == "tap-night-4"]
    t1 = Ztr[tr_labels == "tap-night-1"]
    t2 = Ztr[tr_labels == "tap-night-2"]
    non_tap = Ztr[~np.isin(tr_labels, TAP_TRAIN)]
    iu = np.triu_indices(len(m3), 1)
    d33 = float((1.0 - (m3 @ m3.T)[iu]).mean())
    iu4 = np.triu_indices(len(m4), 1)
    d44 = float((1.0 - (m4 @ m4.T)[iu4]).mean())
    print(f"within tap-night-3 (held-out):      {d33:.4f}")
    print(f"within tap-night-4 (held-out):      {d44:.4f}")
    print(f"cross night-3 × night-4 (held-out): {dist(m3, m4):.4f}")
    print(f"held-out(all) → tap-night-1:        {dist(Zho, t1):.4f}")
    print(f"held-out(all) → tap-night-2:        {dist(Zho, t2):.4f}")
    print(f"held-out(all) → non-tap rooms:      {dist(Zho, non_tap):.4f}")
    print(f"  night-3 → tap-night-1 / night-2:  "
          f"{dist(m3, t1):.4f} / {dist(m3, t2):.4f}")
    print(f"  night-4 → tap-night-1 / night-2:  "
          f"{dist(m4, t1):.4f} / {dist(m4, t2):.4f}")
    tr_same = contrast.separability(Ztr, list(tr_labels))
    print(f"[ref] training rooms: same-room sim={tr_same['same_room_mean']:.4f} "
          f"cross-room sim={tr_same['cross_room_mean']:.4f}")
    # centroid geometry: is there a generic 'tap' blob?
    c3, c4 = m3.mean(0), m4.mean(0)
    c1, c2 = t1.mean(0), t2.mean(0)
    cn = non_tap.mean(0)
    n = lambda v: v / np.linalg.norm(v)
    print(f"centroid cos: c3·c1={n(c3) @ n(c1):.3f} c3·c2={n(c3) @ n(c2):.3f} "
          f"c4·c1={n(c4) @ n(c1):.3f} c4·c2={n(c4) @ n(c2):.3f} "
          f"c3·c4={n(c3) @ n(c4):.3f} c3·nonTap={n(c3) @ n(cn):.3f}")

    # ------------------------------------------------ (c) spread -----------
    print("\n=== (c) SPREAD / CONCENTRATION ===")
    ho_spread = contrast.room_spread(Zho, list(ho_labels))
    tr_spread = contrast.room_spread(Ztr, list(tr_labels))
    for r in sorted(ho_spread):
        print(f"  held-out {r:<16} spread={ho_spread[r]:.4f}")
    for r in TAP_TRAIN:
        print(f"  train    {r:<16} spread={tr_spread[r]:.4f}")
    tr_mean = float(np.mean(list(tr_spread.values())))
    print(f"  train-room mean spread:           {tr_mean:.4f}")
    s33 = float((m3 @ m3.T)[iu].mean())
    s44 = float((m4 @ m4.T)[iu4].mean())
    print(f"  mean within-room cosine sim: night-3={s33:.4f} night-4={s44:.4f} "
          f"(train same-room mean={tr_same['same_room_mean']:.4f})")

    # ------------------------------------------------ (d) rank defect ------
    print("\n=== (d) RANK / SPECTRUM ===")
    er_tr, s_tr = eff_rank(Ztr)
    er_ho, s_ho = eff_rank(Zho)
    er_tap, s_tap = eff_rank(np.vstack([t1, t2]))
    print(f"effective rank (participation ratio): train(all)={er_tr:.2f} "
          f"tap-night-1+2={er_tap:.2f} held-out(3+4)={er_ho:.2f}")
    print(f"train top-12 singular values:  {np.round(s_tr[:12], 2)}")
    print(f"held-out singular values:      {np.round(s_ho, 3)}")
    # variance of held-out set captured by the training PCA basis
    Xc = Ztr - Ztr.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Hc = Zho - Zho.mean(0, keepdims=True)
    tot = float((Hc ** 2).sum())
    for k in (2, 5, 10, 20):
        P = Vt[:k].T
        cap = float(((Hc @ P) ** 2).sum()) / tot
        print(f"held-out variance in top-{k:<2} train-PCA dims: {cap:.3f}")
    # off-origin displacement of the held-out blob (unit sphere => norm of mean)
    print(f"||mean|| (blob displacement): train={np.linalg.norm(Ztr.mean(0)):.4f} "
          f"held-out={np.linalg.norm(Zho.mean(0)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
