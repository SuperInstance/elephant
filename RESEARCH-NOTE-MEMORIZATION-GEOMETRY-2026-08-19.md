# RESEARCH NOTE — Geometry of the Held-Out Memorization Failure (2026-08-19)

Subject: why the text contrast head (seed-1 held-out checkpoint,
`checkpoints/contrast/text_contrast_heldout_seed1.pt`) FAILS the honest test
(train on 17 rooms, hold out tap-night-3/4 → held-out fine gaps
0.0713 / 0.1073 / 0.0295 vs 0.05 noise floor). Diagnosis only — no
retraining. Numbers from `scripts/memorization_geometry.py` (seed 1, the
seed whose gap cleared 0.05). All embeddings unit-norm; distances are cosine.

Corpus as built by `scripts/contrast_train_text.py`: 1097 training clips /
17 rooms, 18 held-out clips (13× night-3, 5× night-4).

## (a) Nearest-neighbor confusion

Held-out clips do **not** snap to their same-cast training nights
(tap-night-1/2): only **3/18** top-1 (and 3/18 top-3) hits. Dominant
attractors:

| top-1 training room | count (of 18) |
|---|---|
| tap-open-mic        | 12 |
| dogs-fell-in-love   | 2  |
| tap-night-1/2       | 3  |
| speeches            | 1  |

Top-1 sims are high (0.72–0.96) — the clips sit *inside* memorized foreign
clusters, not in a no-man's land. Notably the dominant attractor is
tap-open-mic: the same-venue/same-cast-family room. The head generalized
"the Tap" as a place, but attached that signal to the open-mic cluster, not
to the night-1/2 clusters.

## (b) Distance structure (mean cosine distance)

| comparison | dist |
|---|---|
| within tap-night-3 | 0.338 |
| within tap-night-4 | 0.566 |
| night-3 × night-4 (cross) | 0.471 |
| held-out → tap-night-1 | 0.885 |
| held-out → tap-night-2 | 0.536 |
| held-out → non-tap rooms | 0.876 |

Reference (training rooms): same-room sim 0.916, cross-room sim 0.475.
Centroid cosines: c3·c4 = 0.862; c3·tap-night-2 = 0.573, c4·tap-night-2 =
0.640; c3/c4·tap-night-1 = 0.139/0.177; c3·non-tap = 0.215.

Reading: the two held-out nights are **closer to each other than either is
to any training room**, and they lean toward the tap-night-2/open-mic side —
a weak, diffuse "tap blob" — but are as far from tap-night-1 as from random
rooms (0.885 ≈ 0.876). Night-4 is essentially unclustered (within-night
distance 0.566 exceeds its cross-night distance to night-3, 0.471): the
seed-1 "pass" (gap 0.107) is carried almost entirely by night-3.

## (c) Spread / concentration

| cluster | spread (mean pairwise cos dist) |
|---|---|
| held-out tap-night-3 | 0.338 |
| held-out tap-night-4 | 0.566 |
| train tap-night-1 | 0.121 |
| train tap-night-2 | 0.052 |
| train-room mean | 0.075 |

Within-room cosine sim: night-3 = 0.662, night-4 = 0.434, vs training
same-room mean 0.916. The held-out region is **4.5–7.6× more diffuse** than
the memorized training clusters — the head formed no attractor for unseen
text.

## (d) Rank / spectrum

- Effective rank (PCA participation ratio): train = 5.54, tap-night-1+2 =
  1.61, held-out = 4.89. **No rank collapse** on held-out vectors.
- Held-out variance captured by the training PCA basis: top-2 = 0.346,
  top-5 = 0.680, top-10 = 0.873, top-20 = 0.964. The held-out signal lives
  *inside* the low-rank subspace the head carved for the training rooms,
  but as diffuse variance *between* the clusters, not along any
  night-identity direction.
- ||mean||: train 0.856, held-out 0.783 — both blobs sit well off-origin on
  the sphere; the held-out blob is displaced toward the tap/open-mic
  sector.

## Verdict

The geometry is a **Voronoi tessellation by memorized point-clusters**, not
a collapsed map: multi-positive InfoNCE pulled every training clip onto one
of 17 tight, well-separated room attractors (spread 0.075, same-room sim
0.92), leaving the map between attractors unconstrained. Unseen nights of
the same cast land wherever their bag-of-words projection happens to fall —
a 5–7× more diffuse cloud that leans toward the same-venue open-mic cluster
(distance structure shows venue/cast generalization, zero night-identity
generalization: tap-night-1 is as far as a random room). What the head
actually learned to separate is *documents it has seen*; the only signal
that transfers is surface vocabulary shared with the open-mic night. To
make room-identity generalize rather than memorize, the objective needs
invariance pressure that exists at eval time: define positives as
independent views of the same identity (e.g. token-dropout / window-jitter
augmentations, or split each night into two disjoint halves and train
half-vs-half) so attractors become smooth basins that interpolate to unseen
text of the same room; better still, reframe the label to the level the
geometry says is learnable — venue/cast (tap, open-mic, nights 1–4 as one
room with night as a fine probe) — and, if night-level identity is truly
required, train episodically with leave-one-night-out batches so the loss
itself penalizes exactly this failure mode. Architectural/normalization
fixes are secondary: the rank data shows capacity is not the bottleneck —
the training signal contains no night-invariant feature to begin with.
