# GPU Lane v0 — Room-State Embeddings on Synthetic Room Fields (2026-09-27)

**Author:** overnight GPU lane lead (subagent, night watch 2026-09-27)
**Status:** prototype run complete; numbers below from the committed
`gpu/results/room-state-embed-v0.json`
**Branch:** `gpu/room-state-embed-v0`
**Implements (in miniature):** the v3 spec,
[`elephant-sense-v3-design.md`](elephant-sense-v3-design.md) — room-state
embeddings trained on cold/warm contrast, acclimation curves
(agent → room), charisma as measurable pull (room → agent)

---

## Read this first: what this is and is not

**Everything here runs on SYNTHETIC data.** `gpu/roomgen.py` generates rooms
with known ground-truth latents (warmth, tightness, pacing, nuisance axes)
pushed through a frozen nonlinear sensor mixture. No fleet corpus, no Tap
nights, no audio — nothing real. A model that separates synthetic rooms
proves the **training machinery and the metric battery work**; it proves
**nothing** about whether real fleet rooms are separable. That claim stays
unclaimed until this runs on real data (see "Next steps" and the spec's
§1.3 data wishlist).

This is v0: the smallest honest instantiation of the spec.

## What was built

A self-contained `gpu/` module (new; no existing module touched):

| File | Role |
|------|------|
| `gpu/roomgen.py` | Synthetic room-field generator: room latents, cold-tight/warm-loose prior (κ anti-correlates with warmth), clip emission through a frozen nonlinear mixture, acclimation traces `z(t) = z_room + (z0 − z_room)e^(−t/τ)`, charisma passes (room latent drifts toward a cold charismatic agent, χ(t) = χ_max(1 − e^(−t/τ_c))) with matched controls |
| `gpu/model.py` | Small MLP encoder (48 → 256 → 256 → 128 → 64, L2-normalized output, ~150k params) + the spec §2.3 objective: multi-positive NT-Xent, τ = 0.15 fixed, batches = all clips from 2–3 rooms, VICReg-style within-room spread guard |
| `gpu/train.py` | Seeded one-command train + eval; records device, CUDA flag, nvidia-smi samples taken *during* training, wall-clock, loss trace; writes results JSON |
| `gpu/evaluate.py` | The spec §6 metric battery, all on HELD-OUT rooms: 1-NN room discrimination, sauna/plunge gap, cosine silhouette, cold/warm linear probe (train rooms → held-out rooms, with a raw-observation baseline probe), κ-proxy correlation, acclimation τ-recovery + rank gain, charisma displacement vs controls |
| `gpu/results/room-state-embed-v0.json` | The committed evidence (baseline-untrained vs trained, environment, timings) |
| `tests/test_gpu_room_embedding.py` | 13 tests: determinism, cold/warm structure, loss descent, batch coverage, metric sanity, charisma drift ground truth, end-to-end CPU smoke. Skips cleanly without torch (torch is an optional dep — pyproject `learned` extra) |

Design choices taken directly from the spec (and its four-model review):
no centroids as anchors; τ = 0.15 not 0.07; per-room batches; explicit
within-room spread regularizer; charisma vs acclimation separated **only**
by experimental design (charisma rooms move the room latent; control rooms
don't — matched otherwise), never by statistics on a joint trajectory.

## The run (RTX 4050, WSL2 /dev/dxg)

- torch 2.14.0+cu126, **`torch.cuda.is_available() = True`**, device
  `cuda`, `NVIDIA GeForce RTX 4050 Laptop GPU` (6141 MiB)
- 36 train rooms / 12 held-out eval rooms × 48 clips, 80 epochs,
  ~150k-param encoder — **wall-clock 11.3 s** (the GPU was loafing:
  135 MiB / 14% util in the nvidia-smi samples taken during training;
  15 samples committed in the results JSON)
- seeded (default 2718): same seed → same world, same init, same batches

## Results (held-out rooms, frozen encoder, before → after training)

| Metric | Untrained | Trained | Read |
|--------|-----------|---------|------|
| Sauna/plunge gap (same − cross-room cosine) | **0.028** | **0.386** | **the win: the walk between rooms is now felt** |
| Cosine silhouette (room identity) | 0.463 | 0.518 | separation without full collapse |
| 1-NN room discrimination (chance ≈ 0.083) | 0.977 | 0.911 | *regressed* — untrained was already near ceiling (see below) |
| Cold/warm probe, embeddings (train→held-out) | 0.844 | 0.844 | **no embedding win** — raw obs probe = 0.930 both times |
| κ-proxy correlation (spread ↔ true σ) | **0.898** | **0.470** | *regressed* — contrastive tightening compresses temperature |
| Acclimation: log-τ correlation (true τ) | 0.765 | **0.894** | τ recovery improved |
| Acclimation: rank gain (last vs first third) | see JSON | see JSON | percentile-rank observable (spec §3) |
| Charisma: Cohen's d (χ-pass vs control) | 1.91 | **2.34** | the pull is more measurable after training |

### The honest reads

1. **The headline win is the sauna/plunge gap** (0.028 → 0.386): after
   contrast training, clips from the same held-out room are far closer to
   each other than to other rooms *in a space the training never saw*.
   That is the spec's core claim in miniature.
2. **Instance-level kNN regressed** (0.977 → 0.911). On this synthetic
   geometry the untrained encoder is already near ceiling — a random
   projection nearly preserves an 8-dim latent's metric structure
   (Johnson–Lindenstrauss). Training buys room-*scale* structure (the
   gap) at a small cost in instance geometry. Booked as a finding: **the
   synthetic world is too easy for instance separation to be the
   interesting metric here.**
3. **κ compression is real and structural.** NT-Xent pulls same-room
   clips together, which tightens within-room spread for every room —
   more for loose (warm) rooms — and compresses the temperature signal
   (r: 0.898 → 0.470). A VICReg-style variance **floor** was tried first
   and was *worse* (it homogenized spread by design); the committed
   guard is a collapse-only hinge. An epoch sweep (10/20/40/60/80) shows
   κ_r wandering 0.44–0.61 with no clean trend — the tension is in the
   objective, not the schedule. **Open problem, booked:** separating
   rooms while preserving how loose each room is.
4. **The embedding does not beat raw sensors for cold/warm probing**
   (0.844 vs 0.930). In this world warmth is nearly linearly present in
   the raw channel, so a probe has an easy job; the embedding's value is
   room-*level* structure, not the warmth axis. On real data (where the
   warmth axis is not handed to the sensor) this comparison is the one
   to watch.

## Booking — PROVEN / MODELED / SPECULATED

**PROVEN** (by this run + tests, on synthetic ground truth):
- The contrastive objective trains and descends; batches cover every clip
  exactly once per epoch (tested).
- The full metric battery runs end-to-end and its pieces pass sanity
  checks (silhouette ≈ 1 on perfect clusters; τ-fit recovers a known τ to
  <15%; probe generalizes across rooms on separable data).
- Charisma rooms' latents demonstrably drift and matched controls' don't
  (ground-truth test) — the experimental design is sound.
- CUDA training works on the RTX 4050 lane (torch + /dev/dxg), recorded
  with nvidia-smi evidence in the results JSON.
- Reproducibility: same seed → identical world (tested).

**MODELED** (implemented in synthetic miniature; real-data versions not yet run):
- Cold/warm separation, room discrimination, sauna/plunge gap, κ/spread
  temperature, acclimation τ-recovery, charisma displacement — every
  number above is a *simulation* result. The mapping from this synthetic
  world to fleet corpora is asserted only at the level of "the machinery
  exists and works."
- **The κ/separation tension** (honest read #3): contrastive separation
  compresses the temperature signal. Modeled, measured, and left open —
  the fix is a research question, not a config knob.

**SPECULATED** (not built, not run):
- That real rooms (Tap nights, music pole, Wesley's streams) will separate
  the way synthetic rooms do. The spec's own §8 probe says the fine
  contrast is nearly invisible (gap 0.015) to the frozen v2 encoder —
  whether contrast training opens it on real data is the open question,
  and this repo's answer must wait for that run.
- Speaker-identity confound: the synthetic world has **no cast identities
  at all**, so the speaker-heldout control (spec §6, mandatory on real
  data) is untested here by construction.
- Multimodal fusion (audio+text+pacing, §5): v0 is single-channel.

## Failures & friction, first-class

- The pre-existing background `pip install torch` (session clear-forest)
  **failed**: PEP 668 externally-managed environment, no `--break-system-packages`,
  nothing installed. Fixed by the distro's own recommended route: a venv
  at `~/venvs/elephant-gpu` (outside the repo), torch cu126 wheel. Logged
  in full in the session log; booking it here because failures are the
  record.
- torch on Python 3.14 required the cu126 wheel index (cu124 cp314 wheels
  were not consulted — cu126 was taken directly; noted, not swept).

## Next steps (not done in v0)

1. Same battery on real rooms: the four Tap nights + the music pole, using
   the frozen v2 encoder outputs as the sensor channel — the spec §8
   "next experiment" verbatim.
2. Speaker-heldout control becomes mandatory the moment real voices enter.
3. Cast identities in the synthetic generator (shared-cast confound,
   adversarial version).
4. Multimodal fusion per spec §5 (per-modality distance matching,
   modality dropout, presence-as-mask).

## How to run

```bash
# base env (no torch): the gpu tests skip, suite stays green
python3 -m pytest

# with the learned extra (torch), full run on GPU if present:
python3 -m gpu.train            # writes gpu/results/room-state-embed-v0.json
python3 -m pytest tests/test_gpu_room_embedding.py
```

---

*v0 walked the elephant around a synthetic room. The real sauna and the
real plunge are still ahead — but the scales it will wear there are cut.*
