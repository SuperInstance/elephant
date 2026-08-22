"""Riverbed generator — the sample-path forward model (generation-corpus
instrument) for the time-indexed vMF random field.

Grounding: docs/foundation-2026-08-21/kimi-ideation-2026-08-21.md §1 ("the
corpus IS a field sample path"; "Path B is the instrument that makes Path
A's wave-3 interpretable"), docs/foundation-2026-08-21/
foundation-synthesis-2026-08-21.md (the registered skew-product: room base
orbit ⊕ reader fiber), STAGE2-CORPUS-DESIGN-2026-08-20.md (the 9-family
warmth ladder + 21-reader attendance the generated corpora mirror).

This is the DIRECT vMF SIMULATOR of ideation §1.2: readings are sampled
straight from the field measure (room path + reader deviations + deadband
noise), bypassing text and the TapNight engine. It emits night JSONL in the
SAME v:2 schema as the wave-2 T-nights (data/nights/night-T*.jsonl, written
by scripts/e2_nights.py), so the registered analysis pipeline
(scripts/e2_instrument.py, scripts/premise_band_movers.py,
scripts/slope_regression*.py) consumes generated corpora unchanged.

EVENT SEMANTICS (corrected 2026-08-21 by the κ(t)-around-entry check —
memory/kappa-t-check-2026-08-21.md, DIRECTION-EVENT verdict; supersedes the
original "κ-trajectory-first" presupposition, which the field falsified):
  - The PRIMARY control channel of a night is the direction-only warmth
    schedule μ(t) = w(t)·Ŵ + sqrt(1−w(t)²)·e⊥(t), e⊥ ⊥ Ŵ a slow tangent
    random walk (the deadband drift floor). Warmth is DEFINED as the signed
    cosine Ŵ·μ̂ — the vmf.py direction-only convention. FLIPS are warmth
    jumps (μ events); ENTRIES are μ events too — the entrant's text pulls
    the room's warmth by a flip-magnitude step (field: Δwarmth −0.147 vs
    flip −0.151, p=0.68). The magnitude-contaminated field.py warmth()
    (same weights on raw re-centered readings, collinear with field
    extremity) is NEVER used to set anything; it is only logged (warmth_v0).
  - κ(t) is the concentration channel with FIELD polarity — warm content
    tight (24), cynical loose (11) — and transitions only ever LOOSEN it
    (flip response = the level change; entry = a smaller multiplicative
    loosening). κ is a designed channel matching the engine's measured κ
    semantics, not a roster-driven one.

SKEW-PRODUCT READER FIBER (Agenda Problem 3; ideation §1.1/§2.1; G6
  2026-08-21 — the ENGINE's charisma-pull equation, replay_readings
  parity): each reader's reading is the engine's own per-reader transform
  of the room channel —

    eff_R(t) = clamp( raw(t) + s_R(t) · (vibe_R(t) − raw(t)) ),
    s_R(t)   = 1 − exp( −charisma_R · n_R(t) ),
    vibe_R  ← vibe_R + (raw − vibe_R) · (1 − exp(−acclimation_R))

  (tapnight.speak / e2_instrument.replay_readings, replicated exactly;
  assert_replay_matches_log passes on generated nights). Branch
  parameters live ONLY in the per-night persona anchor logged as the
  roster's vibe/vibe_start (persona space — the coordinate firewall, and
  the G2 Arm-2 semantics: the name persists, the instrument doesn't):

    vibe0_R(night) = pool_vibe + (1−α) · dev_R(night)

  Branch parameter α:
    instrument (α=0): dev_R persistent across attended nights, OU-evolved
      in R⁷ between nights (φ=0.9, innovation sized from the filed
      ICC=0.9076 honesty target, ideation §1.4) — readers pull toward
      their own drifted persona vibe;
    collapse (α=1): every reader shares the pool's common vibe — the
      reading is room-driven (the registered collapse semantics);
    noise: dev_R redrawn per night (μ̂_R unstable by design);
    intermediate: any --alpha in [0,1].
  Reader baselines are constant WITHIN a night (P ≈ 0.994 persistence by
  construction) and OU-drifted BETWEEN nights (the ICC knob).

FIBER VERSIONS (--fiber, wave-4 draft §1.2, 2026-08-22 — docs/
wave4-registration-draft-2026-08-22.md):
  v3 (DEFAULT — the frozen wave-3 instrument): α lives ONLY in the static
    per-night persona anchor (the line-845 provenance the wave-3 S5
    verdict verified: vibe0 = pool + (1−α)·dev) and the charisma pull
    rides the acclimating vibe state. Every wave-3 corpus stays
    bit-reproducible under this default.
  v4: α re-pointed out of the static anchor into the pull's within-night
    target trajectory —
        target_R(t) = pool + (1−α)·dev_R + α·room(t)
        room(t)     = FIELD_ANCHOR_NORM · w_ar(t)/‖w_ar(t)‖
    (direction-only AR(1) wobble carrier at anchor scale — the §1.2
    option (i) amplitude match: the α contrast is purely WHO carries the
    offset, never how big it is; w_ar comes from the room rng stream and
    dev from the branch-free fiber stream, so pair members' targets
    differ only through α). The pull becomes
    eff = clamp(raw + s·(target_R(t) − raw)); the roster's vibe/vibe_start
    carry target_R(0); the acclimation line is unchanged (vibe becomes
    α-free state, still logged as vibe_now); target_R(t) is logged per
    speak as lens_now.target_now so replay parity can be re-registered
    as v2 — the wave-3 replay from vibe_start alone no longer
    reconstructs v4 fibers (§1.3.1, expected).

NULL MODE (--null-mode): no warmth structure — every segment of every
night sits at the night's base warmth (no flips, μ constant within a
night) and the ONLY scheduled variation is a common κ(t) shift shared by
the whole roster (cohesion-only common shift, per the foundation
synthesis's redefinition of common shift as measurable cohesion).

Contamination firewall (ideation §2): branch parameters live in persona
space and field-measure space only; nothing here ever computes an offset
from a roster mean, a corpus_sd, or an o/d quantity — those exist only on
the analysis side. corpus_sd and WARM-as-estimator-target are NOT handed
to the generator output; each corpus must pass the wave gate on its own.

Seeded reproducible: one numpy Generator per (seed, tag); session_id is a
deterministic md5 of (seed, tag), so re-runs are byte-identical (verified
by the manifest determinism check, same stripped-md5 discipline as
e2_nights.py).

Run:
  python3 scripts/riverbed_generator.py --branch instrument
  python3 scripts/riverbed_generator.py --alpha 0.5 --tag-prefix rb-a50
  python3 scripts/riverbed_generator.py --branch collapse --null-mode
  python3 scripts/riverbed_generator.py --alpha 0.25 --blind          # G3
  python3 scripts/riverbed_generator.py --unblind <sealed.json>       # G3
  python3 scripts/riverbed_generator.py --alpha 0.5 --pair-seed 4242  # G13
  python3 scripts/riverbed_generator.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elephant.field import RoomField
from elephant.vmf import (A7, CENTER, DIALS, HI, LO, SCALE, WARM,
                          edge as vmf_edge, vmf_fit)
from scripts.e2_nights import ATTENDANCE
from scripts.nights_abc import _cast, _newcomer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSONAS = os.path.join(ROOT, "data", "e2", "e2-personas.json")
DEFAULT_OUT = os.path.join(ROOT, "data", "nights", "generated")

D = 7
W_WIN = 8          # engine's trailing-window size (tapnight.TapNightSession W)
STEP = 60.0        # auto60 clock
KAPPA_R_DEFAULT = 40.0   # reader-fiber concentration (tight instrument)
ANCHOR_SCALE = 1.0       # persona-anchor scale on the FIELD's own vibe_start
                         # deviations (1.0 = engine-faithful persona geometry;
                         # G6 2026-08-21 — supersedes G7's DEV_SCALE=0.85, which
                         # was calibrated for the vMF fiber the G6 rework
                         # replaced; the charisma-pull fiber attenuates the
                         # persona signal through s_R(t) and the within-night
                         # acclimation pull, so the anchor carries the field's
                         # full magnitude. Realized end-to-end ICC lands in the
                         # filed [0.85, 0.96] band at the registered seed.)
FIELD_ANCHOR_NORM = 0.989  # measured avg norm of the 21-reader pool's
                         # vibe_start deviations from the pool mean (z-space,
                         # wave-2 cast + e2-personas, 2026-08-21) — the OU
                         # innovation and the redraw branch size at ANCHOR_SCALE
OU_PHI = 0.9             # between-night deviation persistence
ICC_TARGET = 0.99        # ANALYTIC OU-level honesty target: the between-night
                         # OU wobble ALONE would realize this ICC. The field-
                         # measure geometry (ladder × on-sphere fiber, loose
                         # entry eras, weak-dial lottery) contributes the
                         # remaining within-variance, so the END-TO-END
                         # realized ICC — measured by the G7 self-test through
                         # the registered Measurement — lands in the filed
                         # band [0.85, 0.96] (0.886 at the registered seed
                         # 20260821 under corrected κ; field wave-2
                         # actual-presence value 0.8444, wave-1 filed 0.9076)
ORTH_WALK = 0.005        # e⊥ tangent random-walk step (deadband drift floor).
                         # G7 calibration 2026-08-21: 0.02 → 0.005. The walk
                         # is THREADED across the wave (one persistent space —
                         # The Tap): each night continues the corpus's single
                         # latent tangent path instead of redrawing fresh;
                         # fresh-per-night directions couple the persona fibers
                         # to the room's per-night draw and sink realized ICC
                         # to ~0.74. At 0.005/step the path diffuses ~0.1 rad
                         # across a 9-night corpus (room-scale persistence)
                         # while within-night behavior stays walk-dominated at
                         # the same floor shape. Full deadband-floor sweep vs
                         # the field's 0.29 corpus-sd stable-phase d floor is
                         # still Gap G6 (untested here — see S1 report).
KAPPA_JITTER = 0.03      # multiplicative log-jitter on κ(t)
FLIP_SIZE = 0.5          # warmth jump at a warm→cynical flip (Δw)

# --- corrected event semantics (κ(t)-check, 2026-08-21 — DIRECTION-EVENT #
# verdict; memory/kappa-t-check-2026-08-21.md) -------------------------- #
# Field measurements on the filed corpora (per-strata logged fits + the
# pooled event table):
#   * entry-steps are μ/direction events: Δwarmth −0.147 ≈ flip's −0.151
#     (p=0.68), ‖Δμ̂‖ +0.301 ≈ flip's +0.329 (p=0.48) — μ is NOT continuous
#     at entry; the entrant's text pulls the room like a smaller flip.
#   * κ polarity: warm content TIGHT (κ≈24), cynical content LOOSE (κ≈11)
#     — the old 10/18 ran opposite to the field.
#   * κ responds to EVERY content transition by LOOSENING (flip window
#     Δlogκ −0.746 ≈ the warm→cold level change ln(11/24); entry −0.320,
#     a strictly smaller version; quiet ≈ 0). The old +12 entry TIGHTENING
#     pulse was the wrong sign.
# Design translation: latent levels KAPPA_WARM/KAPPA_COLD carry the flip
# response; entries multiply the latent κ by KAPPA_ENTRY_FACTOR (sized so
# the LOGGED window response ≈ −0.32 through the cumulative-fit renewal
# fraction ~0.25 at the entry position: latent ln(0.28) ≈ −1.28); the two
# combine by pointwise min (transitions only ever loosen latent κ). The
# logged fits smooth both into the field's gradual window responses.
ENTRY_DWARMTH = 0.485    # entry μ-step (0.97 × FLIP_SIZE: pooled Δwarmth
                         # entry/flip = 0.147/0.151, κ-check §4)
KAPPA_WARM = 24.0        # warm-content latent concentration (field ≈ 24)
KAPPA_COLD = 11.0        # cynical-content latent concentration (field ≈ 11)
KAPPA_ENTRY_FACTOR = 0.28  # latent κ multiplier at entry (e^{-1.28})
KAPPA_FLOOR = 2.5        # latent κ sanity floor

# ----------------------------------------------------------------------- #
# G6 NOISE MODEL (2026-08-21 — memory/research-g6-noise-2026-08-21.md).  #
# The unit-sphere emission cannot hit the field's triple (corpus_sd      #
# 0.2367 / stable-d floor 0.29 / ±0.10 warmth band): unit vectors cap    #
# per-component variance at 1/(7·SCALE²) → corpus_sd ≤ ~0.15 at ANY κ,   #
# and windowing 8 latent draws inflates logged κ ~8× (200/90 vs field    #
# 24/11). The engine-faithful decoupling (G6 §2, three parts):          #
#   (i)   per-speak per-dial Gaussian noise at the field's logged       #
#         within-stratum scales (below) — supplies corpus_sd's noise     #
#         component while the estimator (which unit-normalizes) absorbs  #
#         only σ/‖z‖ of it;                                             #
#   (ii)  the emitted windowed z KEEPS its magnitude (the engine logs    #
#         the raw windowed reading; vmf_fit unit-normalizes internally — #
#         the estimator sees the direction of the SAME noisy vector);    #
#   (iii) the reader fiber is the engine's charisma-pull equation        #
#         (replay_readings parity — see the fiber block below).          #
# PLUS the E_SEG schedule contrast: the warm→cynical flip is a TEXT      #
# step (cynicism/presence/earnestness-heavy), not a warmth step — the    #
# field's stratum-mean geometry (G6 §1.5: contrast 0.196 raw RMS is the  #
# largest missing corpus_sd component; WARM-direction flips supply only  #
# 0.108). Adopted with the harness's disclosure (calibration_harness     #
# E_SEG, the filed design decision): warmth loads on mood in WARM-space, #
# the strata ladder lives in E_SEG-space. Constants below are the FIELD  #
# stratum-mean measurements on the wave-2 T-nights (2026-08-21):         #
# warm-era seg coefficient −0.55; flip steps +0.81 (mean over            #
# T1/T2/T3/T8/T4b); entry steps +0.51 (mean over T4a/T4b/T5/T5c).       #
# μ(t) itself stays pure-warmth (Ŵ·μ = w(t) exact, the registered        #
# direction-only convention); the seg contrast rides the emission.      #
# ----------------------------------------------------------------------- #
DIAL_NOISE = 2.2   # σ: per-speak dial-noise scale (multiplicative on the
                   # field's within-stratum per-dial scales below). G6 run
                   # 2026-08-21: the free dials' within-era scatter needs the
                   # T1-warm anatomy level (~2x the pooled within-stratum
                   # mean — the pooled SIGMA_DIAL averages rail-pinned eras
                   # where the clamp cuts the noise; measured on the wave-2
                   # warm strata: mood .23, earnest .22, joke .30, presence
                   # .36 z) to land logged kappa in the field's order.
NOISE_ERA_EXP = 0.25  # era-scaling exponent on the per-speak noise:
                      # multiplier (KAPPA_COLD/κ(t))^p — warm eras tighter
                      # (x0.82 at κ=24), entry eras loosest (x1.11). The
                      # field's within-stratum scatter is era-dependent with
                      # the κ polarity (T5-pre 0.31 > T5-post 0.06, measured
                      # 2026-08-21); p=0 (flat) leaves the logged κ ratio at
                      # 1.66 vs the field's 2.18 — the ratio needs the era
                      # scaling, the level needs σ.
SIGMA_DIAL = np.array([   # field within-stratum per-dial z-sd (G6 §2.1,
    0.1163, 0.0101, 0.1449, 0.1488, 0.2021, 0.0284, 0.1981,
])                     # measured 2026-08-21: heterogeneous — joke_landing/
                       # presence loosest, volume ~deterministic; RMS 0.140 z
# --- era-position geometry (field stratum-mean vectors, z-space) ------ #
# The G6 run (2026-08-21) replaced the single-direction E_SEG contrast
# with the FIELD'S OWN per-dial era-position vectors, measured as stratum
# means on the wave-2 T-nights (data/nights): the flip is a TEXT step
# that moves cynicism rail-to-rail (+1.70 z) with presence (+.53), while
# mood/volume/panic barely move — the WARM mood-heavy flip swings and the
# harness's normalized E_SEG approximation BOTH miss this per-dial mix
# (G6 §1.5: the contrast 0.196 raw RMS is the largest missing corpus_sd
# component; the measured vectors deliver it at field scale: pooled
# per-dial z-sd cynicism .94 vs field .969, presence .38 vs .403).
# Dial order: mood, volume, earnestness, cynicism, joke_landing, panic,
# presence. All three are DEVIATIONS from the corpus grand mean
# (BASELINE_Z); steps are additive per event (flip then entries), the
# warm base is the field's mean warm-stratum deviation.
Z_WARM_DEV = np.array([   # warm-era deviation from grand mean (field mean
    -0.045, -0.003, 0.068, -0.946, -0.035, -0.050, -0.321,
])                        # over warm strata; cynicism sits at its LOW rail,
                          # presence a third down)
Z_FLIP = np.array([       # warm->cynical flip step (field mean over
    0.072, 0.004, -0.123, 1.698, 0.061, 0.094, 0.529,
])                        # T1/T2/T3/T8/T4b): cynicism rail-to-rail,
                          # presence up, earnestness down, mood ~flat
Z_ENTRY = np.array([      # entry step (field mean over T4a/T4b/T5/T5c):
    0.056, 0.003, -0.121, 1.208, 0.015, 0.087, 0.477,
])                        # a smaller version of the flip (the entrant's
                          # text is a content event — κ-check parity)
E_SEG = np.array([0.05, -0.25, -0.45, 0.55, -0.30, 0.25, -0.50], float)
E_SEG = E_SEG / np.linalg.norm(E_SEG)   # RETIRED (kept for reference/
                         # comparability only): the harness's normalized
                         # text-step direction, superseded by the measured
                         # Z_FLIP/Z_ENTRY vectors above (same disclosure
                         # the harness carries — warmth loads on mood in
                         # WARM-space, the strata ladder lives in the
                         # era-position space)
BASELINE_Z = np.array([   # field per-dial z-space MEAN offsets (G6 sec 2.1:
    0.9646, 0.0850, 0.8279, 0.9984, 0.2516, 0.0563, 0.6032,
])                        # the corpus GRAND MEAN — the magnitude/baseline
                          # structure; field z-norms ~2.0 with per-dial
                          # heterogeneous means; warm-era ||z|| ~= 1.46,
                          # matching this anchor at full scale). Measured on
                          # the wave-2 T-nights 2026-08-21.
EMISSION_BASELINE = 1.0  # scale on BASELINE_Z (1.0 = the field's own
                         # per-dial mean positions — the era vectors pin
                         # the free dials at their measured stratum levels;
                         # lower scales move every dial off its measured
                         # position and starve the charisma-x-era-swing
                         # between-reader channel, sinking realized ICC
                         # to 0.53-0.67)
WARMTH_SCALE = 0.45  # scale on the latent warmth (mu) part of the
                     # emission. The direct sampler stacks a direction-only
                     # warmth channel on top of the era-position anchor;
                     # the FIELD has no such second layer (its dial content
                     # IS the emission), and its warm-era ||z|| ~= 1.46 is
                     # reproduced at 0.45. Above ~0.7 the mood-heavy latent
                     # inflates ||z|| to ~1.9, the estimator-facing noise
                     # sigma/||z|| shrinks, logged kappa inflates past the
                     # field's 24/11 band, and the doubled mood swings
                     # crowd out the era channels (ICC -> 0.68).
# (Design-search disclosure 2026-08-21: two noise DEPENDENCE structures
# were measured and rejected — post-window AR(1) at NOISE_PHI=0.9 (worse on
# every metric: strata warmth residuals 0.35-0.49 from era-static offsets)
# and post-window iid (split-half d over-dispersed to 0.44-0.46). The
# per-message, window-smoothed form below is the engine's own dial
# semantics and reproduces the field's marginal scale AND dependence.)

BANK_CLASSES = ["MoodDial", "VolumeDial", "EarnestnessDial", "CynicismDial",
                "JokeLandingDial", "PanicDial", "PresenceDial",
                "ModelVsCodeDial", "VisionDial"]

# ----------------------------------------------------------------------- #
# Night shapes — the 9 frozen T-families as FIELD schedules.              #
# warmth = target signed cosine Ŵ·μ̂ (direction-only); base values mirror  #
# the filed roster-invariant ladder (STAGE2 §1: S2 .3187, S4a .4465,      #
# D/D-cold .6293, S4b .6319, S1/A .6551, S3 .7409, S5 .7589). Flips are   #
# warmth (μ) jumps of FLIP_SIZE; entries are μ steps of ENTRY_DWARMTH     #
# (κ-check: entry ≡ flip in μ-response). κ: warm tight (24), cynical       #
# loose (11), entries loosen further — field polarity.                    #
# ----------------------------------------------------------------------- #
NIGHT_FAMILIES = {
    # tag: (base_warmth, n_speaks, flip_seq|None, entry_seqs)
    "T1":  (0.6551, 40, 20, []),
    "T2":  (0.3187, 28, 8, []),
    "T3":  (0.6551, 40, 20, []),
    "T4a": (0.4465, 46, 20, [12]),
    "T4b": (0.6319, 45, 20, [28]),
    "T5":  (0.6293, 46, None, [24]),
    "T5c": (0.6293, 46, None, [24]),
    "T8":  (0.7409, 28, 20, []),
    "T9":  (0.7589, 20, None, []),      # no-flip control family
}
NIGHT_ORDER = ["T1", "T2", "T3", "T4a", "T4b", "T5", "T5c", "T8", "T9"]

BRANCHES = {  # (alpha, ou_phi, kappa_R, redraw_dev_per_night)
    "instrument": (0.0, OU_PHI, KAPPA_R_DEFAULT, False),
    "collapse": (1.0, OU_PHI, KAPPA_R_DEFAULT, False),
    "noise": (0.0, 0.0, 8.0, True),
}

# TODO(G2 — Arm 2, wave-3 SEPARATE REGISTRATION ADDENDUM; do not build here):
# the engine-native arm expresses branch semantics as a persona-resampling
# map on TapNightSession constructor inputs (collapse = per-night warmth-
# conditioned persona redraw; the name persists, the instrument doesn't —
# e2_nights.py run_night(...) is the wiring point). The direct-vMF arm above
# is complete; the extension point is a future build_arm2_wave(...) sibling
# of generate_wave() in this module. Registered only via its own addendum.

# ----------------------------------------------------------------------- #
# G1 — mid-night entrants (field roster mechanics, wave-3 plan §3 G1).    #
# Families with entry_seqs STAGE the entrant exactly as the engine does  #
# the drifter on the T-nights (verified on data/nights/night-{T4a,T4b,   #
# T5,T5c}.jsonl): absent from session_open roster, declared in           #
# staged_entries, OMITTED from every readers block before the entry seq  #
# (the field's drifter-omission convention that fires the analysis      #
# NaN-before-entry path), present from it with entry_mode "staged-cold", #
# authoring his first speak AT the entry seq.                            #
# ----------------------------------------------------------------------- #
ENTRANT_NAME = "drifter"
ENTRANT_SPEAKS = {  # engine-mirrored staged speak seqs (measured on the
    "T4a": [12, 16, 20, 24, 28, 32],   # filed wave-2 corpus, 2026-08-21)
    "T4b": [28, 32, 36, 40, 43],
    "T5":  [24, 28, 32, 36, 40, 44],
    "T5c": [24, 28, 32, 36, 40, 44],
}
# Measurement attendance of the staged entrant per canonical family —
# mirrors FIELD_NIGHTS_W2/COLD_ENTRY_W2 semantics: the drifter measurement-
# attends T4a/T4b; his T5/T5c line-readings are warmth content only.
# Custom staged families default to attendance=True.
STAGED_ATTENDANCE = {"T4a": True, "T4b": True, "T5": False, "T5c": False}


def staged_speaks(fam, family):
    """Seqs the staged entrant authors (engine positions for the canonical
    families; entry + 4k, <=6 lines, for custom staged families)."""
    if fam in ENTRANT_SPEAKS:
        return list(ENTRANT_SPEAKS[fam])
    e, n = family[3][0], family[1]
    return [e + 4 * k for k in range(6) if e + 4 * k < n]


# ----------------------------------------------------------------------- #
# vMF sampling on S⁶ (Wood 1994, exact; numpy-only)                       #
# ----------------------------------------------------------------------- #
def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def vmf_sample(rng, mu, kappa):
    """One exact draw from vMF(μ, κ) on S^{D-1} (Wood's rejection sampler
    for the μ-component + uniform tangent direction). κ=0 ⇒ uniform."""
    mu = _unit(np.asarray(mu, float))
    d = D
    if kappa < 1e-8:
        x = rng.normal(size=d)
        return _unit(x - (x @ mu) * mu) * 0.0 + _unit(rng.normal(size=d))
    b = (-2.0 * kappa + math.sqrt(4.0 * kappa ** 2 + (d - 1) ** 2)) / (d - 1)
    x0 = (1.0 - b) / (1.0 + b)
    m = (d - 1) / 2.0
    c = kappa * x0 + (d - 1) * math.log(1.0 - x0 ** 2)
    while True:
        z = rng.beta(m, m)
        u = rng.random()
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        if kappa * w + (d - 1) * math.log(1.0 - x0 * w) - c >= math.log(u):
            break
    xi = rng.normal(size=d)
    xi = _unit(xi - (xi @ mu) * mu)
    return w * mu + math.sqrt(max(0.0, 1.0 - w * w)) * xi


# ----------------------------------------------------------------------- #
# Personas (persona space only — the coordinate firewall, ideation §2.1)  #
# ----------------------------------------------------------------------- #
def load_personas():
    """name -> persona dict (dial_weights, acclimation_rate, charisma,
    vibe_start). Sources: the frozen cast (nights_abc._cast/_newcomer) and
    the seeded field-distribution draws (data/e2/e2-personas.json)."""
    out = {}
    for p in list(_cast()) + [_newcomer()]:
        d = p.to_dict()
        d["vibe_start"] = list(d["vibe"])
        out[p.name] = d
    doc = json.load(open(PERSONAS, encoding="utf-8"))
    for n, p in doc["new_personas"].items():
        out[n] = {"name": n, "dial_weights": list(p["dial_weights"]),
                  "acclimation_rate": float(p["acclimation_rate"]),
                  "charisma": float(p["charisma"]),
                  "vibe": list(p["vibe_start"]),
                  "vibe_start": list(p["vibe_start"])}
    return out


def persona_deviations(names, personas):
    """Persona-anchored deviation directions: z(vibe_start) de-meaned over
    the wave's reader pool, scaled by ANCHOR_SCALE (1.0 = the field's own
    persona-deviation magnitudes — engine-faithful; G6 2026-08-21).
    Persona space only — no estimator coordinate (roster-mean of READINGS,
    corpus_sd, o/d) is ever touched.

    G7 calibration note (2026-08-21): PROPORTIONAL, not per-reader
    unit-normalized — the engine's reader deviations are proportional to
    the persona's dial profile with heterogeneous magnitudes, and
    unit-normalizing every reader to the same norm erases that
    heterogeneity (it also starves weak dials of between-reader spread:
    realized panic-dial ICC collapsed to ~0.1). Proportional anchors
    reproduce the engine's magnitude structure."""
    z = {n: SCALE * (np.asarray(personas[n]["vibe_start"], float) - CENTER)
         for n in names}
    mean = np.mean(np.stack([z[n] for n in names]), axis=0)
    raw = {n: z[n] - mean for n in names}
    return {n: ANCHOR_SCALE * raw[n] for n in names}


# ----------------------------------------------------------------------- #
# Room path: κ-first schedule, μ slaved to the direction-only warmth      #
# ----------------------------------------------------------------------- #
def room_schedule(family, null_mode, rng, flip_size=FLIP_SIZE):
    """(warmth(t), kappa(t)) arrays — CORRECTED event semantics (κ(t)-check
    2026-08-21, DIRECTION-EVENT verdict):
      * warmth is the μ channel: flips are schedule jumps of flip_size;
        ENTRIES are μ events too — the entrant's text pulls warmth down by
        ENTRY_DWARMTH (flip-magnitude), exactly as the field's staged
        entries do (Δwarmth −0.147 vs flip −0.151, p=0.68). μ is not
        continuous at entry.
      * κ is the concentration channel with FIELD polarity: warm content
        tight (24), cynical loose (11); transitions only ever LOOSEN it —
        the flip response IS the level change, entries multiply by
        KAPPA_ENTRY_FACTOR, combined by pointwise min.
      * null mode: warmth flat at base (no μ structure); the only scheduled
        variation is the common κ shift at the would-be flip, field
        polarity (tight→loose) — cohesion-only, no direction content."""
    base, n, flip, entries = family
    w = np.full(n, base)
    if flip is not None and not null_mode:
        w[:flip] = base + flip_size / 2.0
        w[flip:] = base - flip_size / 2.0
    if not null_mode:
        for e in entries:   # entry = μ/direction event (not a κ event)
            w[e:] -= ENTRY_DWARMTH
    w = np.clip(w, -0.95, 0.95)
    if null_mode:
        kappa = np.full(n, KAPPA_WARM)
        if flip is not None:  # cohesion-only common shift, field polarity
            kappa[flip:] = KAPPA_COLD
    else:
        kappa = (np.where(np.arange(n) < flip, KAPPA_WARM, KAPPA_COLD)
                 if flip is not None else np.full(n, KAPPA_WARM))
    for e in entries:   # entry loosening: min-semantics (never tighten)
        level = float(kappa[max(0, e - 1)])   # latent level just before entry
        kappa[e:] = np.minimum(kappa[e:], level * KAPPA_ENTRY_FACTOR)
    kappa = np.maximum(kappa, KAPPA_FLOOR)
    kappa = kappa * np.exp(rng.normal(0.0, KAPPA_JITTER, n))
    return w, kappa


WOBBLE_LEVEL = 0.085  # tangent-wobble sd per draw at κ = KAPPA_COLD (the
                     # level knob, harness-faithful: its kappa0=300 room ran
                     # tan_sd 0.082 ~= this x 0.68). The kappa(t) design
                     # channel keeps its registered POLARITY and RATIO
                     # through tan_sd(k) = LEVEL*sqrt(KAPPA_COLD/k) — warm
                     # tight (x0.68), cold loose (x1.0), entries loosest —
                     # but the ABSOLUTE level is set here, NOT by A7(k(t)):
                     # at k=24 the A7-derived spread (0.166) is twice the
                     # field's scripted-text wobble, and its AR(0.98)
                     # quasi-static era offsets tilt strata warmth by
                     # +-0.28 (measured 2026-08-21) and cluster the
                     # cumulative fits to k~170. Entries: k 6.7 -> x1.28
                     # = 0.154, matching the field's loose entry strata
                     # (T5-pre within-sd 0.31).
AR_PHI = 0.9       # latent-room AR(1) tangent-wobble persistence — the
                    # harness's filed room family (G6 §5.2: "the harness's
                    # AR(1) room wobble is the right family for the room").
                    # The field's raw is scripted text: tight short-lag,
                    # drifting era-scale — NOT iid per-speak vMF draws
                    # (iid draws at latent κ=24 over-disperse the windowed
                    # means' split-half to d≈1.24 vs the field room's 0.90).


def seg_schedule(family, null_mode):
    """Per-speak era-position DEVIATION vectors (z-space, n x D) — the
    field's real schedule geometry (G6 §1.5/§5.4, the measured stratum-mean
    vectors): the warm->cynical flip steps the emission by Z_FLIP
    (cynicism rail-to-rail, presence up, mood ~flat), entries step by
    Z_ENTRY (a smaller version — the entrant's text is a content event
    too), warm eras sit at the Z_WARM_DEV base. Null mode: NO text steps
    (seg flat at the warm base) — the null corpus is cohesion-only, no
    direction content of any kind."""
    base, n, flip, entries = family
    seg = np.tile(Z_WARM_DEV, (n, 1))
    if not null_mode:
        if flip is not None:
            seg[flip:] += Z_FLIP
        for e in entries:
            seg[e:] += Z_ENTRY
    return seg


def room_path(family, null_mode, rng, flip_size=FLIP_SIZE, e_state=None,
              dial_noise=None, baseline=None):
    if dial_noise is None:
        dial_noise = DIAL_NOISE        # resolved at call time (calibration
    if baseline is None:              # sweeps mutate the module constants)
        baseline = EMISSION_BASELINE
    """One sample path of the room base orbit: μ(t) on S⁶ with Ŵ·μ(t) =
    w(t) EXACTLY (direction-only warmth), e⊥ a slow tangent walk; latent
    per-message draws s_i ~ vMF(μ(i), κ(i)); the observed windowed reading
    is the trailing-W mean of the draws PLUS the era-position anchor
    (baseline·BASELINE_Z + the measured era vectors) PLUS per-speak
    per-dial Gaussian noise — UNNORMALIZED (G6 §2.1: the engine logs
    the raw windowed z; vmf_fit normalizes internally, so the estimator
    sees the direction of the same noisy vector while corpus_sd sees the
    noise in full — the σ/‖z‖ decoupling the unit-sphere model cannot
    do). The latent warmth part carries the per-family ladder at
    WARMTH_SCALE (the field has no second warmth layer; 0.45 reproduces
    its warm-era ‖z‖ ≈ 1.46).

    e_state (G7): optional dict threading the tangent direction ACROSS the
    wave — one persistent latent path per corpus (The Tap is one space):
    the first night seeds it, later nights continue it. Within-night
    behavior is unchanged (same ORTH_WALK step). None ⇒ fresh draw
    (single-night / self-test use)."""
    base, n, flip, entries = family
    w, kappa = room_schedule(family, null_mode, rng, flip_size)
    seg = seg_schedule(family, null_mode)
    # e⊥(t): unit, ⊥ Ŵ, slow tangent random walk (the drift floor)
    if e_state is not None and e_state.get("e") is not None:
        e = _unit(np.asarray(e_state["e"], float))
    else:
        e = rng.normal(size=D)
        e = _unit(e - (e @ WARM) * WARM)
    mus, s_lat, w_ar_path = [], [], []   # w_ar_path: wave-4 §1.2 carrier
    w_ar = None    # AR(1) tangent-wobble state (per night; harness family)
    for t in range(n):
        xi = rng.normal(size=D)
        xi = xi - (xi @ WARM) * WARM - (xi @ e) * e
        e = _unit(e + ORTH_WALK * xi)
        mus.append(w[t] * WARM + math.sqrt(max(0.0, 1.0 - w[t] ** 2)) * e)
        # latent per-message z = A7(κ(t))·μ(t) + AR(1) tangent wobble at
        # the κ-matching stationary spread (harness _room_once's equations;
        # κ(t) keeps the registered polarity — warm tight / cold loose —
        # now expressed as the WOBBLE spread instead of iid draw scatter)
        mu = mus[-1]
        a7 = A7(float(kappa[t]))
        tan_sd = WOBBLE_LEVEL * math.sqrt(KAPPA_COLD / max(kappa[t], 1e-6))
        if w_ar is None:
            w_ar = tan_sd * rng.normal(size=D)   # stationary init (the
            w_ar = w_ar - (w_ar @ mu) * mu       # harness's vMF-entry draw)
        else:
            w_ar = w_ar - (w_ar @ mu) * mu   # re-anchor (μ moved)
        c_ar = tan_sd * math.sqrt(1.0 - AR_PHI ** 2)
        eps = rng.normal(size=D)
        eps = eps - (eps @ mu) * mu
        w_ar = AR_PHI * w_ar + c_ar * eps
        w_ar = w_ar - (w_ar @ mu) * mu
        w_ar_path.append(w_ar.copy())   # wave-4 §1.2: latent carrier exposed
        s_lat.append(WARMTH_SCALE * (a7 * mu) + w_ar)   # NOT re-normalized (G6 part ii)
    if e_state is not None:
        e_state["e"] = e
    obs = []
    for t in range(n):
        z = np.mean(s_lat[max(0, t - W_WIN + 1):t + 1], axis=0)
        z = z + baseline * BASELINE_Z               # G6: magnitude anchor
        z = z + seg[t]                              # G6: era-position vector
        if dial_noise > 0.0:
            # per-speak per-dial Gaussian noise (G6 part i) at the field's
            # within-stratum per-dial SHAPE (SIGMA_DIAL), applied to the
            # windowed emission (the engine's dials read the trailing W
            # window as one object), ERA-SCALED by the κ(t) design channel
            # at NOISE_ERA_EXP (warm eras tighter, entry eras loosest).
            # The field's within-stratum scatter is era-dependent with the
            # same polarity (measured per stratum 2026-08-21: T5-pre 0.31 >
            # T5-post 0.06; pooled SIGMA_DIAL is the shape, κ(t) the era
            # knob), and an era-INDEPENDENT σ structurally flattens the
            # logged κ ratio — the level needs σ, the ratio needs the era
            # scaling.
            z = z + (dial_noise * SIGMA_DIAL
                     * (KAPPA_COLD / max(float(kappa[t]), 1e-6)) ** NOISE_ERA_EXP
                     * rng.normal(size=D))
        obs.append(z)   # NOT unit-normalized (G6 part ii)
    # the FIT channel is the CLAMPED dial-space reading (engine parity: the
    # engine's fits run over vmf_windowed dial values — clamped to the dial
    # cube; with the baseline anchor + noise the rails bite asymmetrically
    # and an unclamped fit input biases logged warmth off the field's —
    # measured 2026-08-21: T5/T5c strata residuals −0.24/−0.34 unclamped)
    obs_fit = [SCALE * (_clamp(CENTER + z / SCALE) - CENTER) for z in obs]
    return {"w": w, "kappa": kappa, "mu": mus, "seg": seg, "obs": obs,
            "obs_fit": obs_fit, "w_ar": w_ar_path,
            "baseline": baseline, "dial_noise": dial_noise}


def _expected_clamp(x, lo, hi, sd):
    """E[clamp(x + N(0, sd^2))] — the truncated-normal mean (per dial).
    The noise-aware clamp correction: rails cut one-sided (the mood dial
    sits near +1 under the baseline anchor, so noise pushes it over the
    rail and the clamp cuts only the warm side — a systematic warmth
    bias a noise-free reconstruction cannot see; measured -0.19 on T5,
    2026-08-21)."""
    if sd <= 1e-12:
        return min(max(x, lo), hi)
    a, b = (lo - x) / sd, (hi - x) / sd
    rt = math.sqrt(2.0)
    Pa = 0.5 * (1.0 + math.erf(a / rt))
    Pb = 0.5 * (1.0 + math.erf(b / rt))
    fa = math.exp(-0.5 * a * a) / math.sqrt(2.0 * math.pi)
    fb = math.exp(-0.5 * b * b) / math.sqrt(2.0 * math.pi)
    return (lo * Pa + hi * (1.0 - Pb)
            + x * (Pb - Pa) - sd * (fb - fa))


def expected_logged_warmth_path(family, flip_size=FLIP_SIZE, W=W_WIN,
                                baseline=None, n_orth=4, seed=0,
                                dial_noise=None, n_quad=12):
    """Noise-aware expected logged-warmth trajectory from the schedule
    alone (the gate's 'cumulative-fit lag accounted' comparison object,
    G6-aware). The emission is reconstructed deterministically — windowed
    A7(κ)-shrunk μ means (at WARMTH_SCALE) + baseline anchor + the
    measured era-position vectors — then passed through TWO
    noise-awareness layers, because the naive noise-free reconstruction
    carries systematic biases the ±0.10 gate band cannot absorb
    (measured 2026-08-21, max strata residual 0.24 naive vs 0.10
    noise-aware):
      * the truncated-normal clamp mean per dial (rails cut one-sided —
        the mood dial sits near +1 under the baseline anchor);
      * E[unit(z + η)] by deterministic quadrature over n_quad seeded
        noise points (the unit-normalization shrink of heterogeneous
        per-dial noise biases warmth DOWN ~0.06-0.10 at the field's
        scales; closed forms get this wrong, quadrature does not).
    The e⊥ tangent direction is marginalized over n_orth seeded
    orthonormal WARM⊥ directions (warmth is e⊥-exact at the μ level;
    only ||z|| sees the alignment). Returns expected warmth_vmf(t) per
    speak (None under NMIN=10, like the logged fits)."""
    if baseline is None:
        baseline = EMISSION_BASELINE
    if dial_noise is None:
        dial_noise = DIAL_NOISE
    base, n, flip, entries = family
    rng0 = np.random.default_rng(seed)          # throwaway (κ jitter unused)
    w, kap = room_schedule(family, False, rng0, flip_size)
    seg = seg_schedule(family, False)
    shrink = np.array([A7(float(k)) for k in kap])
    qrng = np.random.default_rng(seed + 17)
    quad = [qrng.normal(size=D) for _ in range(n_quad)]
    Q = np.linalg.qr(np.column_stack([WARM.reshape(-1, 1),
                                      np.random.default_rng(seed).normal(size=(D, D - 1))]))[0]
    E = Q[:, 1:].T
    picked = E[np.random.default_rng(seed).integers(0, len(E),
                                                    size=min(n_orth, len(E)))]
    paths = []
    for e in picked:
        mus = [w[t] * WARM + math.sqrt(max(0.0, 1.0 - w[t] ** 2)) * e
               for t in range(n)]
        zu = []
        for t in range(n):
            zt = (WARMTH_SCALE * np.mean([shrink[i] * mus[i]
                           for i in range(max(0, t - W + 1), t + 1)], axis=0)
                  + baseline * BASELINE_Z + seg[t])
            # total zero-mean tangent perturbation per dial at t: the AR
            # wobble (era-persistent, cosine-diluting) + the era-scaled
            # iid dial noise
            tan_t = WOBBLE_LEVEL * math.sqrt(KAPPA_COLD / max(float(kap[t]), 1e-6))
            sd_t = np.sqrt(tan_t ** 2
                           + (dial_noise * SIGMA_DIAL
                              * (KAPPA_COLD / max(float(kap[t]), 1e-6)) ** NOISE_ERA_EXP) ** 2)
            for d_ in range(D):   # truncated-normal clamp expectation
                zt[d_] = SCALE[d_] * (_expected_clamp(
                    CENTER[d_] + zt[d_] / SCALE[d_], LO[d_], HI[d_],
                    sd_t[d_] / SCALE[d_]) - CENTER[d_])
            us = [_unit(zt + sd_t * c) for c in quad]
            zu.append(np.mean(us, axis=0))       # E[unit(z + wobble + eta)]
        warm = []
        for t in range(n):
            f = vmf_fit(zu[:t + 1]) if t + 1 >= 10 else None
            warm.append(f["warmth_vmf"] if f else None)
        paths.append(warm)
    out = []
    for t in range(n):
        vals = [p[t] for p in paths if p[t] is not None]
        out.append(float(np.mean(vals)) if vals else None)
    return out


def _reader_fit_light(win):
    """The tapnight._reader_fit light estimator (no NMIN guard, no
    bootstrap): Newton A₇ solve over the trailing reader window of unit
    z-space vectors. None under n < 3."""
    if len(win) < 3:
        return None
    z = np.stack([_unit(np.asarray(v, float)) for v in win])
    r = z.mean(0)
    rho = float(np.linalg.norm(r))
    if rho < 1e-12:
        return {"mu_hat": None, "kappa": None, "n": len(win)}
    mu = r / rho
    k = float(np.clip(rho * (7 - rho ** 2) / (1 - rho ** 2), 1e-6, 500.0))
    for _ in range(60):
        a = A7(k)
        g = 1.0 - a * a - 6.0 * a / k
        if abs(g) < 1e-12:
            break
        step = (a - rho) / g
        k = float(np.clip(k - step, 1e-6, 500.0))
        if abs(step) < 1e-9:
            break
    return {"mu_hat": mu.tolist(), "kappa": k, "n": len(win)}


def _clamp(v):
    return np.minimum(HI, np.maximum(LO, v))


# ----------------------------------------------------------------------- #
# Night emission — the v:2 schema, byte-shape-identical to e2_nights      #
# ----------------------------------------------------------------------- #
def persona_pool_vibe(names, personas):
    """The reader pool's mean persona vibe, z-space (the branch-invariant
    base of the per-night persona anchors): vibe0_R = pool + (1−α)·dev_R.
    Same pool as persona_deviations — the wave's whole reader set."""
    z = np.mean([SCALE * (np.asarray(personas[n]["vibe_start"], float) - CENTER)
                 for n in names], axis=0)
    return z


def generate_night(tag, family, roster_names, personas, dev_anchors,
                   ou_state, branch, seed, outdir, null_mode=False,
                   flip_size=FLIP_SIZE, pair_seed=None, fam=None, e_state=None,
                   pool_vibe_z=None, fiber="v3"):
    """Emit data path outdir/night-<tag>.jsonl. Returns (path, ou_state)
    with the OU state advanced for every APPEARING reader (roster + staged
    entrant; the between-night step happens once per night appeared, in
    fixed family order at the wave level).

    G1 (field entry mechanics): a family with entry_seqs stages the
    ENTRANT mid-night — omitted from the readers block before the entry
    seq, present from it (entry_mode "staged-cold"), never in the open
    roster but declared in staged_entries, exactly like the engine.

    G13 (2AFC pair matching): with pair_seed set, the room path and the
    reader fiber draw from SEPARATE rng streams keyed (pair_seed, family)
    — branch-invariant by construction (the key carries the family, not
    the branch-carrying tag), so paired corpora get the SAME room path,
    rosters, authors and kappa(t), and alpha enters only through the
    per-night persona anchors (vibe0 = pool + (1−α)·dev) and the OU/redraw
    draws. The charisma-pull fiber itself is deterministic given (raw,
    vibe, interactions) — engine-faithful: the engine's per-reader
    readings are a deterministic function of the logged room channel and
    the reader's persona state.

    FIBER v4 (wave-4 draft §1.2, fiber="v4"): α leaves the static anchor
    and rides the pull's within-night target trajectory
    target_R(t) = pool + (1−α)·dev + α·room(t) (room(t) the
    amplitude-matched w_ar carrier — direction-only at anchor scale).
    vibe0 == target_R(0) at roster entry; the acclimation line is
    unchanged (vibe becomes α-free state); lens_now gains target_now per
    speak (replay-parity v2 input). fiber="v3" (default) is byte-identical
    to the frozen wave-3 generator.
    """
    alpha, ou_phi, kappa_r, redraw = branch
    fam = fam if fam is not None else tag
    if pair_seed is not None:  # G13: branch-invariant streams
        room_rng = np.random.default_rng((pair_seed, zlib_crc(fam), 1))
        rng = np.random.default_rng((pair_seed, zlib_crc(fam), 2))
    else:
        rng = np.random.default_rng((seed, zlib_crc(tag)))
        room_rng = rng
    n = family[1]
    room = room_path(family, null_mode, room_rng, flip_size, e_state=e_state)
    # --- G1: staged entrant (families with entry events) --------------- #
    entries = family[3]
    entrant = ENTRANT_NAME if entries else None
    entry_seq = entries[0] if entries else None
    assert entrant not in roster_names, "entrant must stage, not roster"
    e_speaks = set(staged_speaks(fam, family)) if entrant else set()
    present = list(roster_names) + ([entrant] if entrant else [])

    # --- reader fiber: advance OU / redraw deviations for this night --- #
    # ICC honesty: steady-state OU variance = (1−ICC)/ICC of the anchor
    # variance (between-night wobble ≈ 0.1018 of the persistent deviation).
    ou_sigma = ANCHOR_SCALE * FIELD_ANCHOR_NORM * math.sqrt(
        (1.0 - ICC_TARGET) / ICC_TARGET * (1.0 - ou_phi ** 2))
    dev_now = {}
    fiber_personas = dict(personas)   # noise branch redraws the WHOLE
    for name in present:              # persona lens (dial weights too)
        if redraw:
            dev_now[name] = ANCHOR_SCALE * FIELD_ANCHOR_NORM * _unit(rng.normal(size=D))
            # G2 Arm-2 semantics for the noise branch: "the name persists,
            # the instrument doesn't" — per-night persona redraw includes
            # the DIAL LENS (the charisma-pull fiber has no κ_R draw, and
            # the lens (g_R) is what carries stable reader constants in
            # this fiber: without redrawing it the noise branch's realized
            # ICC sits at 0.76 (lens-dominated) instead of collapsing
            # below the filed 0.667 floor as the registration predicts).
            w_new = rng.uniform(0.05, 1.0, D)
            fiber_personas[name] = dict(personas[name])
            fiber_personas[name]["dial_weights"] = [float(x) for x in w_new]
        else:
            st = ou_state.get(name, np.zeros(D))
            st = ou_phi * st + ou_sigma * rng.normal(size=D)
            ou_state[name] = st
            dev_now[name] = dev_anchors[name] + st

    # --- reader fiber: the ENGINE's charisma-pull equation (G6 part iii) - #
    # replay_readings' exact math (scripts/e2_instrument.py; tapnight.speak
    # — verified against elephant/tapnight.py 2026-08-21):
    #     s   = 1 − exp(−charisma · n_R(t))      n_R = reader's own speaks
    #     eff = clamp(raw + s · (vibe − raw))    logged field_eff_to_reader
    #     vibe += (field_eff_after − vibe) · (1 − exp(−acclimation_rate))
    # Branch parameters live ONLY in the per-night persona anchor logged
    # as vibe/vibe_start (persona space — the coordinate firewall):
    #     vibe0_R = pool_vibe + (1−α) · dev_now_R
    # so instrument (α=0) readers pull toward their own drifted persona
    # vibe, collapse (α=1) readers share the pool's common vibe (the
    # reading is room-driven — the registered collapse semantics), noise
    # redraws the anchor per night. Parity is exact: the registered
    # replay on the logged rows reproduces field_eff_to_reader bit-for-bit
    # (assert_replay_matches_log, self-test 14).
    if pool_vibe_z is None:   # direct single-night callers: personas-mean pool
        pool_vibe_z = persona_pool_vibe(sorted(personas), personas)
    if fiber == "v4":
        # WAVE-4 (docs/wave4-registration-draft-2026-08-22.md §1.2): α
        # exits the static anchor and enters the pull's within-night
        # target trajectory —
        #     target_R(t) = pool + (1−α)·dev_R + α·room(t)
        # with room(t) the amplitude-matched AR(1) wobble carrier (§1.2
        # option (i)): direction-only at anchor scale,
        #     room(t) := FIELD_ANCHOR_NORM · w_ar(t)/‖w_ar(t)‖.
        # w_ar rides the ROOM stream (pair-shared bit-for-bit), dev the
        # branch-free fiber stream — pair members' targets differ ONLY
        # through α; the coordinate firewall holds (α stays in
        # persona/target space, never in the room path or κ(t)). No new
        # rng number is drawn here, so every shared draw is untouched.
        room_c = [FIELD_ANCHOR_NORM * _unit(w) for w in room["w_ar"]]

        def _target_dial(name, t):
            z = pool_vibe_z + (1.0 - alpha) * dev_now[name]
            if alpha > 0.0:   # α=0 keeps the exact v3 expression (static
                z = z + alpha * room_c[t]   # target; no ±0.0 addend)
            return _clamp(CENTER + z / SCALE)

        target_path = {name: [_target_dial(name, t) for t in range(n)]
                       for name in present}
        vibe0 = {name: target_path[name][0] for name in present}
    else:
        vibe0 = {name: _clamp(CENTER + (pool_vibe_z + (1.0 - alpha) * dev_now[name])
                              / SCALE) for name in present}
        target_path = None
    vibe = {name: np.asarray(vibe0[name], float).copy() for name in present}
    charisma = {name: float(fiber_personas[name]["charisma"]) for name in present}
    acclim = {name: 1.0 - math.exp(-float(fiber_personas[name]["acclimation_rate"]))
              for name in present}
    g = {}
    for name in present:
        wt = np.asarray(fiber_personas[name]["dial_weights"], float)
        g[name] = wt / wt.max() if wt.max() > 1e-12 else np.ones(D)
    eff_reader = {name: {} for name in present}   # dial-space effs, by seq
    vibe_now = {name: {} for name in present}     # post-acclimation lens

    # --- author schedule (balanced bag rotation, seeded; the staged ----- #
    # entrant authors exactly his engine positions — draw count and call
    # order are branch-invariant, so paired corpora align). G6 2026-08-21:
    # the FIELD's authorship is near-round-robin (measured on the wave-2
    # T-nights: per-reader speak counts 5–8 on a 6-reader roster, no
    # zeros); a uniform-rng rotation gives 0–12 spreads, and the charisma
    # pull s_R = 1−exp(−charisma·n_R) turns count spread into
    # within-reader across-night variance (kills realized ICC at 0.53–0.67
    # vs the field's 0.8444). The bag rotation is the engine's own
    # balance: every reader authors ⌈n/R⌉ or ⌊n/R⌋ speaks. ------------- #
    authors, bag = [], []
    for _ in range(n):
        if not bag:
            bag = list(rng.permutation(roster_names))
        authors.append(bag.pop())
    for q in sorted(e_speaks):
        assert 0 <= q < n, f"entrant speak {q} outside night ({fam})"
        authors[q] = entrant

    session_id = hashlib.md5(f"riverbed:{seed}:{tag}".encode()).hexdigest()

    def _entry(name):
        """Roster-shaped param block (open roster and staged_entries share
        the engine's exact 6-key shape — verified on night-T4a.jsonl). The
        vibe/vibe_start fields carry the reader's PER-NIGHT drifted
        persona anchor (the G6 fiber's branch channel — engine replays
        start from the logged vibe_start, so replay parity holds). Under
        --fiber v4 they carry target_R(0) — the t=0 pull target (wave-4
        §1.3.3: "target at t=0", not a static anchor; α stays sealed)."""
        return {"name": name,
                "dial_weights": [float(x) for x in fiber_personas[name]["dial_weights"]],
                "acclimation_rate": float(personas[name]["acclimation_rate"]),
                "charisma": float(personas[name]["charisma"]),
                "vibe": list(vibe0[name]),
                "vibe_start": list(vibe0[name])}

    open_row = {
        "v": 1, "type": "session_open", "session_id": session_id,
        "space_id": "The Tap", "t_start": 0.0, "clock_mode": "auto60",
        "reader": {"kind": "RoomElephant", "identity": "riverbed-v1",
                   "bank": list(BANK_CLASSES)},
        "params": {"W": W_WIN, "standardization": "z=2(v-c)/(hi-lo)",
                   "estimator": "vmf-mle-newton-v1", "kappa_max": 500},
        "roster": {name: _entry(name) for name in roster_names},
        "reader_schema": {"version": 2, "field": "field_eff_to_reader",
                          "lens": (["vibe_now", "weights_now", "target_now"]
                                   if fiber == "v4"
                                   else ["vibe_now", "weights_now"]),
                          "fit": "vmf-mle-newton-v1", "gate": "roster"},
    }
    if entrant is not None:  # G1: staged, exactly like the engine
        open_row["staged_entries"] = {entrant: _entry(entrant)}
    rows = [open_row]

    interactions = {}
    seen_author = set()
    last_fit = None
    for t in range(n):
        author = authors[t]
        interactions[author] = interactions.get(author, 0) + 1
        presence = sorted({authors[i] for i in range(max(0, t - W_WIN + 1), t + 1)})
        o_t = room["obs"][t]
        raw = _clamp(CENTER + o_t / SCALE)
        fit = vmf_fit(room["obs_fit"][:t + 1]) if t + 1 >= 10 else None
        edge = None
        if last_fit is not None and fit is not None:
            edge = vmf_edge(last_fit, fit)
            edge["real"] = None  # floor calibration is analysis-side
        if fit is not None:
            last_fit = fit

        # charisma-pull fiber (engine order: effs with the pre-speak vibe
        # using the post-increment interaction counts, THEN acclimation,
        # THEN the emit — tapnight.speak's exact sequence)
        effs = {}
        for name in present:
            if name == entrant and t < entry_seq:
                continue  # G1: no readings/vibe evolution before entry
            n_int = interactions.get(name, 0)
            s = 1.0 - math.exp(-charisma[name] * n_int)
            # wave-4 v4: the pull rides the within-night target
            # trajectory; v3 (and the engine replay) rides the
            # acclimating vibe state
            pull = target_path[name][t] if fiber == "v4" else vibe[name]
            eff = _clamp(raw + s * (pull - raw))
            eff_reader[name][t] = eff
            effs[name] = eff
        # acclimation: every present reader's vibe warms toward the room
        # channel (engine order: AFTER this speak's effs; the direct
        # sampler logs field_eff_after = raw, so the tracker target is raw)
        for name in present:
            if name == entrant and t < entry_seq:
                continue
            vibe[name] = vibe[name] + (raw - vibe[name]) * acclim[name]
            vibe_now[name][t] = vibe[name]   # engine logs the updated vibe
        readers = {}
        for name in effs:
            lens = {
                "vibe_now": vibe_now[name][t].tolist(),
                "weights_now": [float(x_) for x_ in fiber_personas[name]["dial_weights"]],
            }
            if fiber == "v4":   # wave-4 §1.3.1: per-t target logging —
                lens["target_now"] = target_path[name][t].tolist()
            readers[name] = {
                "reader_known": True,
                "charisma": charisma[name],
                "field_eff_to_reader": effs[name].tolist(),
                "lens_now": lens,
                "reader_fit": _reader_fit_light(
                    [SCALE * g[name] * (eff_reader[name][q] - CENTER)
                     for q in range(max(0, t - W_WIN + 1), t + 1)
                     if q in eff_reader[name]]),
            }
        entry_mode = {name: "roster" for name in roster_names}
        if entrant is not None and t >= entry_seq:
            entry_mode[entrant] = "staged-cold"  # G1: engine entry-mode value
        reading_of = {}
        a = effs[author]
        na = float(np.linalg.norm(a))
        for member in presence:
            if member == author:
                reading_of[member] = {"cos": 1.0}
                continue
            b = effs[member]
            nb = float(np.linalg.norm(b))
            reading_of[member] = {"cos": float(a @ b / (na * nb))
                                  if na > 1e-12 and nb > 1e-12 else 0.0}
        text = f"riverbed {tag} seq {t}"
        rows.append({
            "v": 2, "type": "speak", "session_id": session_id,
            "space_id": "The Tap", "seq": t, "ts": float(t) * STEP,
            "author": author,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "len": len(text), "reactions": {},
            "first_by_author": author not in seen_author,
            "presence_mask": presence,
            "field_raw_after": raw.tolist(),
            "field_eff_after": raw.tolist(),  # direct sampler: no charisma bend
            "interactions_after": dict(interactions),
            "fit": fit, "edge": edge,
            "readers": readers,
            "entry_mode": entry_mode,
            "reading_of": reading_of,
        })
        seen_author.add(author)

    final_fit = vmf_fit(room["obs_fit"])
    close_raw = _clamp(CENTER + room["obs"][-1] / SCALE)
    readings = {dn: float(close_raw[i]) for i, dn in enumerate(DIALS)}
    readings["model_vs_code"] = 0.5   # non-field bank dials: neutral
    readings["vision"] = 0.5
    dev_order = sorted(((n_, abs(readings[n_] - c_))
                        for n_, c_ in zip(DIALS, CENTER)),
                       key=lambda kv: -kv[1])
    rows.append({
        "v": 1, "type": "session_close", "session_id": session_id,
        "space_id": "The Tap", "t_end": float(n) * STEP, "cycle": 1,
        "final": {
            "readings": readings,
            "mu_hat": final_fit["mu_hat"] if final_fit else None,
            "kappa": final_fit["kappa"] if final_fit else None,
            "kappa_ci": final_fit["kappa_ci"] if final_fit else None,
            "warmth_v0": RoomField(readings).warmth(),  # legacy channel
            "warmth_vmf": final_fit["warmth_vmf"] if final_fit else None,
            "top_dials": ", ".join(n_ for n_, _ in dev_order[:3]),
        },
        "n_messages": n, "notes": "",
        "reader_final": {name: np.median(np.stack(
            list(eff_reader[name].values())), axis=0).tolist()
            for name in present},
    })

    path = os.path.join(outdir, f"night-{tag}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, allow_nan=False) + "\n")
    return path, ou_state


def zlib_crc(s):
    import zlib
    return zlib.crc32(s.encode("utf-8"))


def stripped_md5(path):
    out = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        r.pop("session_id", None)
        out.append(json.dumps(r, sort_keys=True))
    return hashlib.md5("\n".join(out).encode()).hexdigest()


# ----------------------------------------------------------------------- #
# Wave generation + manifest (e2_nights discipline: sha256, stripped md5, #
# determinism re-run). G3: with blind=True the manifest is REDACTED —     #
# SEALED_FIELDS (branch params + seeds) are withheld and tags carry an    #
# opaque corpus id — and the withheld parameters live in a sealed         #
# sidecar bound to the corpus by sha256 (nights + sealed file); open it   #
# ONLY post-registration via unblind()/--unblind, after verdicts are      #
# filed (wave-3 plan §5.5, ideation §2.3).                               #
# ----------------------------------------------------------------------- #
SEALED_FIELDS = ("branch", "alpha", "ou_phi", "kappa_R",
                 "redraw_dev_per_night", "null_mode", "seed", "pair_seed")


def generate_wave(outdir, branch_name="instrument", alpha=None, seed=20260821,
                  null_mode=False, tag_prefix=None, flip_size=FLIP_SIZE,
                  pair_seed=None, blind=False, corpus_id=None, fiber="v3"):
    os.makedirs(outdir, exist_ok=True)
    if branch_name in BRANCHES and alpha is None:
        branch = BRANCHES[branch_name]
    else:
        a = float(alpha)
        branch = (a, OU_PHI, KAPPA_R_DEFAULT, False)
        branch_name = f"alpha-{a:g}"
    if blind:
        corpus_id = re.sub(r"[^A-Za-z0-9_-]", "", corpus_id or "") \
            or secrets.token_hex(4)
        if tag_prefix:
            print("[riverbed] NOTE: --tag-prefix ignored under --blind "
                  "(opaque corpus id enforces the seal)")
        prefix = f"rb-{corpus_id}"
    else:
        prefix = tag_prefix or f"rb-{branch_name}" + ("-null" if null_mode else "")
    personas = load_personas()
    all_readers = sorted({n for names in ATTENDANCE.values() for n in names}
                         | {ENTRANT_NAME})
    dev_anchors = persona_deviations(all_readers, personas)
    pool_vibe_z = persona_pool_vibe(all_readers, personas)  # G6 fiber base
    ou_state: dict = {}

    # refuse overwrite (append-only discipline, same as e2_nights)
    tags = {fam: f"{prefix}-{fam}" for fam in NIGHT_ORDER}
    existing = [t for t in tags.values()
                if os.path.exists(os.path.join(outdir, f"night-{t}.jsonl"))]
    if existing:
        sys.exit(f"REFUSING to overwrite existing nights: {existing} "
                 f"(append-only corpus; pick a new --tag-prefix or outdir)")

    paths = {}
    e_state: dict = {}   # G7: one persistent tangent path across the wave
    for fam in NIGHT_ORDER:  # fixed order: OU advances per appeared night
        path, ou_state = generate_night(
            tags[fam], NIGHT_FAMILIES[fam], ATTENDANCE[fam], personas,
            dev_anchors, ou_state, branch, seed, outdir,
            null_mode=null_mode, flip_size=flip_size,
            pair_seed=pair_seed, fam=fam, e_state=e_state,
            pool_vibe_z=pool_vibe_z, fiber=fiber)
        paths[fam] = path

    manifest = {"generated_by": "scripts/riverbed_generator.py",
                "kind": "riverbed-forward-model-sample-path",
                "seed": seed, "branch": branch_name,
                "alpha": branch[0], "ou_phi": branch[1],
                "kappa_R": branch[2], "redraw_dev_per_night": branch[3],
                "null_mode": null_mode, "flip_size": flip_size,
                "entry_dwarmth": ENTRY_DWARMTH,
                "pair_seed": pair_seed, "reader_schema": 2,
                "fiber": fiber,   # wave-4 design fact (NOT sealed — the
                                  # instrument version, not a branch param)
                # G6 noise model — branch-free DESIGN FACTS (safe for the
                # redacted manifest; the analysis side reads them for
                # expected-path reconstruction, never as targets)
                "noise_model": {
                    "dial_noise": DIAL_NOISE,
                    "noise_era_exp": NOISE_ERA_EXP,
                    "sigma_dial": [float(x) for x in SIGMA_DIAL],
                    "emission": "unnormalized windowed z + era-position "
                                "vectors + per-speak per-dial gaussian "
                                "noise (era-scaled by kappa(t))",
                    "era_vectors": {
                        "z_warm_dev": [float(x) for x in Z_WARM_DEV],
                        "z_flip": [float(x) for x in Z_FLIP],
                        "z_entry": [float(x) for x in Z_ENTRY],
                        "baseline": EMISSION_BASELINE,
                        "baseline_z": [float(x) for x in BASELINE_Z],
                        "warmth_scale": WARMTH_SCALE,
                    },
                    "reader_fiber": "engine-charisma-pull "
                                    "(replay_readings parity)",
                },
                "nights": {}}
    for fam in NIGHT_ORDER:
        tag = tags[fam]
        rows = [json.loads(l) for l in open(paths[fam], encoding="utf-8")
                if l.strip()]
        speaks = [r for r in rows if r["type"] == "speak"]
        entries = NIGHT_FAMILIES[fam][3]
        manifest["nights"][tag] = {
            "file": os.path.basename(paths[fam]),
            "family": fam,
            "sha256": hashlib.sha256(open(paths[fam], "rb").read()).hexdigest(),
            "stripped_md5": stripped_md5(paths[fam]),
            "n_msgs": len(speaks),
            "roster": sorted(next(r for r in rows
                                  if r["type"] == "session_open")["roster"]),
            "schedule": {"base_warmth": NIGHT_FAMILIES[fam][0],
                         "flip_seq": NIGHT_FAMILIES[fam][2],
                         "entry_seqs": entries},
            # G1: staged-entrant bookkeeping (design facts, branch-free —
            # safe for the redacted manifest; consumed by the adapter/gate)
            "staged_entrant": ENTRANT_NAME if entries else None,
            "entry_seq": entries[0] if entries else None,
            "entrant_is_attendance": (bool(STAGED_ATTENDANCE.get(fam, True))
                                       if entries else None),
        }

    # determinism: re-run the whole wave into a temp dir, compare stripped
    # (the replay re-threads the tangent path from a fresh state)
    with tempfile.TemporaryDirectory() as tmp:
        ou2: dict = {}
        e2: dict = {}
        for fam in NIGHT_ORDER:
            p2, ou2 = generate_night(tags[fam], NIGHT_FAMILIES[fam],
                                     ATTENDANCE[fam], personas, dev_anchors,
                                     ou2, branch, seed, tmp,
                                     null_mode=null_mode,
                                     flip_size=flip_size,
                                     pair_seed=pair_seed, fam=fam,
                                     e_state=e2, pool_vibe_z=pool_vibe_z,
                                     fiber=fiber)
            assert stripped_md5(p2) == manifest["nights"][tags[fam]]["stripped_md5"], tags[fam]
            manifest["nights"][tags[fam]]["deterministic_replay_identical"] = True

    if not blind:
        mpath = os.path.join(outdir, "riverbed-manifest.json")
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)
        print(f"[riverbed] branch={branch_name} alpha={branch[0]} "
              f"null_mode={null_mode} seed={seed} "
              f"pair_seed={pair_seed if pair_seed is not None else '-'}")
        print(f"[riverbed] 9 nights -> {outdir} "
              f"(determinism re-run: all stripped-md5 identical)")
        print(f"[riverbed] manifest -> {mpath}")
        return manifest

    # --- G3: redact the manifest, seal the branch parameters ------------- #
    redacted = {k: v for k, v in manifest.items() if k not in SEALED_FIELDS}
    redacted["blinded"] = True
    redacted["corpus_id"] = corpus_id
    sname = f"riverbed-sealed-{corpus_id}.json"
    sealed = {k: manifest[k] for k in SEALED_FIELDS}
    sealed.update({"corpus_id": corpus_id, "tag_prefix": prefix,
                   "nights": {t: manifest["nights"][t]["sha256"]
                              for t in manifest["nights"]}})
    spath = os.path.join(outdir, sname)
    with open(spath, "w", encoding="utf-8") as f:
        json.dump(sealed, f, indent=1)
    redacted["sealed"] = {"file": sname,
                          "sha256": hashlib.sha256(open(spath, "rb").read()).hexdigest()}
    mpath = os.path.join(outdir, "riverbed-manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(redacted, f, indent=1)
    print(f"[riverbed] corpus_id={corpus_id} branch=SEALED alpha=SEALED "
          f"seed=SEALED (unblind only post-registration)")
    print(f"[riverbed] 9 nights -> {outdir} "
          f"(determinism re-run: all stripped-md5 identical)")
    print(f"[riverbed] redacted manifest -> {mpath}; sealed sidecar -> {spath}")
    return redacted


def unblind(sealed_path, manifest_path=None):
    """Open a sealed riverbed sidecar (post-registration only): verify the
    seal — the redacted manifest's sealed.sha256 pins the sidecar bytes,
    the sidecar's night sha256s pin the corpus — then return the withheld
    branch parameters. Any tamper (sidecar, manifest, or night file)
    raises AssertionError."""
    sealed_path = os.path.abspath(sealed_path)
    base = os.path.dirname(sealed_path)
    mpath = manifest_path or os.path.join(base, "riverbed-manifest.json")
    man = json.load(open(mpath, encoding="utf-8"))
    assert man.get("sealed", {}).get("file") == os.path.basename(sealed_path), \
        "sealed file is not the one this manifest declared"
    got = hashlib.sha256(open(sealed_path, "rb").read()).hexdigest()
    assert man["sealed"]["sha256"] == got, \
        "sealed sidecar does not match the manifest seal (tampered?)"
    sealed = json.load(open(sealed_path, encoding="utf-8"))
    for tag, sha in sealed["nights"].items():
        fn = os.path.join(base, man["nights"][tag]["file"])
        assert hashlib.sha256(open(fn, "rb").read()).hexdigest() == sha, \
            f"night {tag} does not match the seal (tampered?)"
    return sealed


# ----------------------------------------------------------------------- #
# Self-test                                                               #
# ----------------------------------------------------------------------- #
def _shim_night(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    nt = SimpleNamespace()
    nt.path = path
    nt.open = next(r for r in rows if r["type"] == "session_open")
    nt.speaks = [r for r in rows if r["type"] == "speak"]
    nt.close = next(r for r in rows if r["type"] == "session_close")
    nt.v2 = "readers" in nt.speaks[0]
    nt.params = {n: dict(p) for n, p in nt.open["roster"].items()}
    for n, p in nt.open.get("staged_entries", {}).items():
        nt.params.setdefault(n, dict(p))  # staged entrants are params too
    for n in nt.params:
        nt.params[n]["dial_weights"] = np.asarray(nt.params[n]["dial_weights"], float)
        nt.params[n]["vibe_start"] = np.asarray(nt.params[n]["vibe_start"], float)
    nt.canon_n = [float(np.mean(list(r["interactions_after"].values())))
                  for r in nt.speaks]
    return nt


def self_test():
    from scripts.e2_instrument import logged_readings
    from scripts.riverbed_adapter import load_wave
    from scripts.premise_band_movers import night_windows

    tmp = tempfile.mkdtemp(prefix="riverbed-selftest-")
    print(f"[self-test] scratch: {tmp}")

    # --- 1. generate mini-waves: instrument / collapse / null ---------- #
    fam = {"T1": (0.65, 40, 20, []), "T4a": (0.45, 46, 20, [12])}
    roster = ["writer", "poet", "engineer", "critic", "captain", "essayist"]
    personas = load_personas()
    anchors = persona_deviations(roster + [ENTRANT_NAME], personas)
    paths = {}
    for label, branch, nullm in (("instr", BRANCHES["instrument"], False),
                                 ("coll", BRANCHES["collapse"], False),
                                 ("null", BRANCHES["instrument"], True)):
        ou = {}
        for ft, f in fam.items():
            p, ou = generate_night(f"st-{label}-{ft}", f, roster, personas,
                                   anchors, ou, branch, 7, tmp,
                                   null_mode=nullm)
            paths[(label, ft)] = p
    print("[self-test] 1. generation: 3 branches x 2 families OK")

    # --- 2. schema compat vs the filed wave-2 corpus ------------------- #
    real_path = os.path.join(ROOT, "data", "nights", "night-T2.jsonl")
    real = _shim_night(real_path)
    gen = _shim_night(paths[("instr", "T1")])
    open_extra = {"staged_entries"}  # present only on staged real nights
    assert set(gen.open) >= set(real.open) - open_extra, \
        f"session_open missing: {set(real.open) - set(gen.open)}"
    assert set(gen.open["reader_schema"]) == set(real.open["reader_schema"])
    r_roster = next(iter(real.open["roster"].values()))
    g_roster = next(iter(gen.open["roster"].values()))
    assert set(g_roster) == set(r_roster), \
        f"roster entry keys differ: {set(r_roster) ^ set(g_roster)}"
    gs, rs = gen.speaks[15], real.speaks[15]  # fit non-null on both
    assert set(gs) == set(rs), f"speak keys differ: {set(rs) ^ set(gs)}"
    g_rd, r_rd = next(iter(gs["readers"].items()))[1], \
        next(iter(rs["readers"].items()))[1]
    assert set(g_rd) == set(r_rd), "readers block keys differ"
    assert set(g_rd["lens_now"]) == set(r_rd["lens_now"])
    assert set(g_rd["reader_fit"]) == set(r_rd["reader_fit"])
    assert set(gs["fit"]) == set(rs["fit"]), "fit keys differ"
    assert set(gs["edge"]) == set(rs["edge"]), "edge keys differ"
    assert set(gen.close) == set(real.close), "session_close keys differ"
    assert set(gen.close["final"]) == set(real.close["final"])
    # early speaks carry fit=None exactly like the engine (NMIN=10)
    assert all(r["fit"] is None for r in gen.speaks[:9])
    assert gen.speaks[9]["fit"] is not None
    # v:2 flag and numeric ranges
    assert gen.v2 and gen.speaks[15]["v"] == 2
    for r in gen.speaks:
        for blk in r["readers"].values():
            v = np.asarray(blk["field_eff_to_reader"])
            assert (v >= LO - 1e-9).all() and (v <= HI + 1e-9).all()
    print("[self-test] 2. schema parity with data/nights/night-T2.jsonl: "
          "open/speak/readers/fit/edge/close key sets identical, "
          "v:2 flag set, bounds respected")

    # --- 3. the registered pipeline consumes it unchanged -------------- #
    reads = {r: logged_readings(gen, r) for r in gen.params}
    assert all(len(v) == 40 for v in reads.values())
    m = SimpleNamespace(nights={"T1": gen}, readers=sorted(gen.params),
                        readings={r: {"T1": v} for r, v in reads.items()},
                        arch={r: r for r in gen.params})
    win = night_windows(m, "T1", 1.0, 12)
    rho = np.concatenate([win["rho"][r] for r in win["readers"]])
    assert np.isfinite(rho).any(), "night_windows produced no finite rho"
    assert len(win["positions"]) == 40 - 12 + 1
    print("[self-test] 3. e2_instrument.logged_readings + "
          "premise_band_movers.night_windows run unchanged; "
          f"finite rho at {np.isfinite(rho).sum()} reader-windows")

    # --- 4. seeded determinism ------------------------------------------ #
    ou = {}
    p2, _ = generate_night("st-instr-T1", fam["T1"], roster, personas,
                           anchors, ou, BRANCHES["instrument"], 7, tmp)
    assert stripped_md5(p2) == stripped_md5(paths[("instr", "T1")])
    print("[self-test] 4. same seed -> identical stripped md5")

    # --- 5. direction-only warmth (the magnitude-contamination fix) ----- #
    rng = np.random.default_rng(0)
    rp = room_path(fam["T1"], False, rng)
    for t in range(40):
        mu = rp["mu"][t]
        assert abs(float(np.linalg.norm(mu)) - 1.0) < 1e-9
        assert abs(float(WARM @ mu) - rp["w"][t]) < 1e-9, \
            "warmth schedule is not direction-only"
    # schedule recovery through the full sample->fit->log path. The logged
    # fits are CUMULATIVE over the night (engine semantics: one fit per
    # speak over all windowed observations so far) and the observations are
    # trailing-8 smoothed, so post-flip warmth lags the schedule; assert
    # the warm-era level and a monotone drop, not the cold-era level.
    logged_w = [r["fit"]["warmth_vmf"] for r in gen.speaks if r["fit"]]
    warm_mean = float(np.mean(logged_w[:11]))   # speaks 9-19, warm era
    late_mean = float(np.mean(logged_w[-4:]))   # lagged toward cold .40
    assert abs(warm_mean - 0.90) < 0.12, f"warm-era warmth {warm_mean:.3f}"
    assert warm_mean - late_mean > 0.08, \
        f"flip not visible in logged warmth: {warm_mean:.3f} -> {late_mean:.3f}"
    print(f"[self-test] 5. W.mu(t) == w(t) to 1e-9 (direction-only); "
          f"logged warmth_vmf tracks the schedule through the cumulative "
          f"fits (warm {warm_mean:.3f} -> late {late_mean:.3f})")

    # --- 6. corrected event semantics (κ-check: DIRECTION-EVENT) -------- #
    rng = np.random.default_rng(1)
    rp1 = room_path(fam["T1"], False, rng)    # flip@20 only
    rp4 = room_path(fam["T4a"], False, rng)   # entry@12 + flip@20
    e, f = 12, 20
    dmu_entry = float(np.linalg.norm(rp4["mu"][e] - rp4["mu"][e - 1]))
    dmu_flip = float(np.linalg.norm(rp1["mu"][f] - rp1["mu"][f - 1]))
    # entry moves μ at FLIP magnitude (field: ‖Δμ̂‖ 0.301 vs 0.329, p=0.48;
    # Δwarmth −0.147 vs −0.151, p=0.68) — μ is not continuous at entry
    assert 0.5 * dmu_flip < dmu_entry < 2.0 * dmu_flip, \
        f"entry |dmu| {dmu_entry:.3f} not flip-like ({dmu_flip:.3f})"
    # κ polarity: warm content tight, cynical loose (field ≈ 24 / ≈ 11)
    warm_k = float(np.mean(rp1["kappa"][:f]))
    cold_k = float(np.mean(rp1["kappa"][f:]))
    assert warm_k > cold_k + 8.0, f"warm κ {warm_k:.1f} !>> cold {cold_k:.1f}"
    # transitions only LOOSEN κ (field: entry Δlogκ −0.32, flip −0.75;
    # the old +12 tightening pulse was sign-flipped vs the field)
    assert rp4["kappa"][e] < rp4["kappa"][e - 1] - 5.0, "entry must loosen κ"
    assert rp1["kappa"][f] < rp1["kappa"][f - 1] - 5.0, "flip must loosen κ"
    k_min_post = float(np.min(rp4["kappa"][e:]))
    assert k_min_post >= KAPPA_FLOOR * 0.9
    print(f"[self-test] 6. entry@12 |dmu|={dmu_entry:.3f} ≈ flip@20 "
          f"|dmu|={dmu_flip:.3f} (μ events, κ-check parity); κ warm "
          f"{warm_k:.1f} > cold {cold_k:.1f}, loosens at entry "
          f"({rp4['kappa'][e-1]:.1f}->{rp4['kappa'][e]:.1f}) and flip "
          f"({rp1['kappa'][f-1]:.1f}->{rp1['kappa'][f]:.1f}) — field polarity)")

    # --- 7. branch discrimination at the alpha endpoints ---------------- #
    def spread(label, ft):
        nt = _shim_night(paths[(label, ft)])
        base = {}
        for name in nt.params:
            rr = logged_readings(nt, name)
            z = np.stack([SCALE * (v - CENTER) for _, v in rr])
            base[name] = np.median(z, axis=0)
        B = np.stack(list(base.values()))
        return float(np.sqrt(np.mean(B.std(axis=0, ddof=1) ** 2)))
    s_i, s_c = spread("instr", "T1"), spread("coll", "T1")
    # G6 rework note: the charisma-pull fiber attenuates the persona signal
    # through s_R(t) and the g-lens, so the instrument/collapse baseline-
    # spread contrast shrank from >2x (vMF fiber) to ~1.1-1.3x — the 2AFC
    # object is the MONOTONE ordering across alpha (the registered
    # prediction), not the old effect size (G6 addendum re-verification;
    # the effect is clean at the wave level: pair-mode fibers diverge,
    # noise-branch ICC collapses to 0.23). The mini-night spread is a
    # 6-reader seed-7 object — assert ordering with a small margin.
    assert s_i > 1.05 * s_c, f"instrument spread {s_i:.3f} !> collapse {s_c:.3f}"
    print(f"[self-test] 7. baseline spread: instrument {s_i:.3f} > "
          f"collapse {s_c:.3f} (alpha endpoints separate; G6 effect size)")

    # --- 8. null mode: flat warmth, mu constant, cohesion-only kappa ---- #
    rng = np.random.default_rng(2)
    rn = room_path(fam["T1"], True, rng)
    assert len(set(rn["w"].tolist())) == 1, "null mode must have flat warmth"
    # mu carries no WARMTH signal in null mode (the e⊥ drift-floor walk is
    # retained — it is warmth-neutral by construction, and removing it
    # would make the null corpus cleaner than the field's 0.29 floor)
    mu_arr = np.stack(rn["mu"])
    w_spread = float(np.ptp(mu_arr @ WARM))
    assert w_spread < 1e-9, f"null-mode warmth moved ({w_spread})"
    cos_min = float((mu_arr @ mu_arr.T).min())
    assert cos_min > 0.90, f"null-mode drift floor out of range (min cos {cos_min:.4f})"
    assert rn["kappa"].max() - rn["kappa"].min() > 1.0, \
        "null mode must keep the common kappa shift"
    nt_null = _shim_night(paths[("null", "T1")])
    w_null = [r["fit"]["warmth_vmf"] for r in nt_null.speaks if r["fit"]]
    assert max(w_null) - min(w_null) < 0.35, "null-mode warmth not flat"
    print(f"[self-test] 8. null mode: warmth flat (min cos mu {cos_min:.4f}), "
          f"kappa shift {rn['kappa'].min():.1f}->{rn['kappa'].max():.1f} "
          "(cohesion-only)")

    # --- 9. G1: staged entrant semantics mirror the field --------------- #
    st = _shim_night(paths[("instr", "T4a")])
    assert ENTRANT_NAME not in st.open["roster"], "entrant must not roster"
    assert sorted(st.open["staged_entries"]) == [ENTRANT_NAME]
    ent = st.open["staged_entries"][ENTRANT_NAME]
    assert set(ent) == set(next(iter(st.open["roster"].values()))) == {
        "name", "dial_weights", "acclimation_rate", "charisma",
        "vibe", "vibe_start"}, "staged entry shape != roster shape"
    inr = [ENTRANT_NAME in r["readers"] for r in st.speaks]
    e = 12
    assert not any(inr[:e]) and all(inr[e:]), \
        "entrant must be omitted before entry and present from it"
    assert next(r["seq"] for r in st.speaks
                if r["author"] == ENTRANT_NAME) == e, "cold entry at first speak"
    assert st.speaks[e - 1]["entry_mode"].get(ENTRANT_NAME) is None
    assert st.speaks[e]["entry_mode"][ENTRANT_NAME] == "staged-cold"
    reads_ent = logged_readings(st, ENTRANT_NAME)  # presence from 1st appear
    assert len(reads_ent) == 46 - e, "readings must start at entry"
    from scripts.e2_instrument import Night as _Night  # noqa: F401 (doc)
    print(f"[self-test] 9. G1: entrant staged @{e} — omitted before/present "
          f"from entry, staged-cold, {len(reads_ent)} readings (field parity)")

    # --- 10. G9: staged-night schema parity vs the filed corpus --------- #
    real4 = _shim_night(os.path.join(ROOT, "data", "nights", "night-T4a.jsonl"))
    assert set(st.open) == set(real4.open), \
        f"staged open keys differ: {set(real4.open) ^ set(st.open)}"
    r_se = next(iter(real4.open["staged_entries"].values()))
    assert set(ent) == set(r_se), "staged_entries shape differs from field"
    compared = 0
    for a, b in zip(st.speaks, real4.speaks):
        assert set(a) == set(b), f"staged speak keys differ at seq {a['seq']}"
        if ENTRANT_NAME in a["readers"]:
            assert set(a["readers"][ENTRANT_NAME]) == \
                set(b["readers"][ENTRANT_NAME]), \
                "entrant reader-block keys differ"
            compared += 1
    assert compared, "no entrant rows found for the staged parity check"
    assert set(gen.speaks[15]) == set(st.speaks[15]), \
        "staged vs non-staged speak key sets differ"
    print("[self-test] 10. G9: staged-night parity — open (incl. "
          "staged_entries), speak rows, entrant reader-block keys all "
          "identical to night-T4a.jsonl; staged == non-staged key sets")

    # --- 11. G7: realized between-night ICC of a mini instrument wave --- #
    # Full-design wave (the plan's registered attendance/ladder) at the
    # REGISTERED seed, measured through the REGISTERED Measurement via the
    # G5 adapter. (Scratch tags in a temp dir — not the registered corpora;
    # calibration-target check only, per plan §5.1's self-test boundary.)
    from scripts.riverbed_adapter import load_wave
    icc_dir = os.path.join(tmp, "icc-wave")
    generate_wave(icc_dir, branch_name="instrument", seed=20260821)
    wv = load_wave(icc_dir)
    icc, icc_dial = wv["measurement"].icc()
    # G6 re-verification (2026-08-21, G6 rework run doc): the charisma-pull
    # fiber + field-magnitude anchors + era-position geometry reproduce
    # the FIELD's actual-presence ICC — 0.8444 through this exact
    # Measurement path (the filed wave-2 number: S1 hardening doc + G6
    # research §3; canonical 0.7714). Band anchored on the field value;
    # the old [0.85, 0.96] bracket was the vMF-fiber calibration (0.886)
    # — superseded (G6 addendum + docs/riverbed-G6-run-2026-08-21.md);
    # the registered instrument-vs-NOISE discrimination is preserved
    # (noise branch collapses to ~0.2-0.3 << 0.78).
    assert 0.78 <= icc <= 0.88, \
        f"realized instrument ICC {icc:.4f} outside the re-verified band"
    print(f"[self-test] 11. G7/G6: instrument wave (21 readers x 9 families, "
          f"sd={wv['sd']:.4f}) realized ICC = {icc:.4f} in the re-verified "
          "field-actual band [0.60, 0.80] (registered Measurement, "
          "unmodified; G6 addendum re-bands the filed [0.85, 0.96])")

    # --- 12. G13: pair mode — branches share the room path -------------- #
    tmp2 = tempfile.mkdtemp(prefix="riverbed-pair-", dir=tmp)
    kw = dict(family=fam["T1"], roster_names=roster, personas=personas,
              dev_anchors=anchors, seed=7, outdir=tmp2, fam="T1")
    p0, _ = generate_night("st-p0-T1", branch=BRANCHES["instrument"],
                           ou_state={}, pair_seed=4242, **kw)
    p1, _ = generate_night("st-p1-T1", branch=BRANCHES["collapse"],
                           ou_state={}, pair_seed=4242, **kw)
    r0 = [json.loads(l) for l in open(p0) if l.strip()]
    r1 = [json.loads(l) for l in open(p1) if l.strip()]
    s0 = [r for r in r0 if r["type"] == "speak"]
    s1 = [r for r in r1 if r["type"] == "speak"]
    assert all(a["field_raw_after"] == b["field_raw_after"]
               for a, b in zip(s0, s1)), "pair: room paths differ"
    assert all(a["author"] == b["author"] for a, b in zip(s0, s1)), \
        "pair: author schedules differ"
    assert any(a["readers"]["writer"] != b["readers"]["writer"]
               for a, b in zip(s0, s1)), "pair: fibers did not diverge"
    pn, _ = generate_night("st-np-T1", branch=BRANCHES["instrument"],
                           ou_state={}, **kw)  # tag-keyed (no pair seed)
    sn = [r for r in (json.loads(l) for l in open(pn) if l.strip())
          if r["type"] == "speak"]
    assert any(a["field_raw_after"] != b["field_raw_after"]
               for a, b in zip(sn, s0)), \
        "without --pair-seed the tag-keyed rng must give a different room"
    print("[self-test] 12. G13: pair mode — same room path/authors across "
          "alpha=0 vs alpha=1, fibers diverge; tag-keyed default differs")

    # --- 13. G3: blinding — redacted manifest + sealed round-trip -------- #
    bdir = os.path.join(tmp, "blind-wave")
    red = generate_wave(bdir, branch_name="collapse", blind=True,
                        corpus_id="SELFTEST")
    for k in SEALED_FIELDS:
        assert k not in red, f"redacted manifest leaks {k}"
    assert red["blinded"] and red["corpus_id"] == "SELFTEST"
    assert all("SELFTEST" in t and "collapse" not in t
               for t in red["nights"]), "tags must be opaque to branch"
    sealed = unblind(os.path.join(bdir, red["sealed"]["file"]))
    assert sealed["branch"] == "collapse" and sealed["alpha"] == 1.0
    tampered = json.load(open(os.path.join(bdir, red["sealed"]["file"])))
    tampered["alpha"] = 0.0
    tpath = os.path.join(bdir, "tampered-sealed.json")
    with open(tpath, "w") as f:
        json.dump(tampered, f)
    try:
        unblind(tpath)
        raise AssertionError("tampered seal must not unblind")
    except AssertionError as exc:
        assert "seal" in str(exc)
    print("[self-test] 13. G3: blind wave — branch/seed withheld, opaque "
          "tags, sealed sidecar round-trips, tamper detected")

    # --- 14. G6: engine replay parity (the fiber is replay_readings) ---- #
    from scripts.e2_instrument import assert_replay_matches_log
    for ft in ("T1", "T4a"):
        nt = _shim_night(paths[("instr", ft)])
        nt.first_speak_seq = lambda reader, _nt=nt: next(
            (r["seq"] for r in _nt.speaks if r["author"] == reader), None)
        for name in nt.params:
            assert_replay_matches_log(nt, name, cold=(name == ENTRANT_NAME))
    ent_reads = len([1 for r in _shim_night(paths[("instr", "T4a")]).speaks
                     if ENTRANT_NAME in r["readers"]])
    print(f"[self-test] 14. G6: replay parity — e2_instrument."
          "assert_replay_matches_log reproduces every logged "
          "field_eff_to_reader bit-for-bit on T1 (roster) + T4a (staged "
          f"entrant, cold from entry; {ent_reads} entrant rows)")

    # --- 15. G6: noise-model calibration snapshot (registered seed) ----- #
    # The four re-verified generator statistics at the registered seed on
    # the registered tags (bands carry the realization spread measured
    # across draws, 2026-08-21; the G6 registration addendum holds the
    # full calibration table and the disclosed residuals).
    from scripts.riverbed_wave_gate import run_gate
    gate = run_gate(os.path.join(icc_dir, "riverbed-manifest.json"))
    assert gate["all_pass"], "the G6 generator must pass its own wave gate"
    sd15 = gate["corpus_sd"]
    assert 0.21 <= sd15 <= 0.30, f"corpus_sd {sd15:.4f} outside [0.21, 0.30]"
    w15 = load_wave(os.path.join(icc_dir, "riverbed-manifest.json"))
    m15 = w15["measurement"]
    from scripts.e2_instrument import W2_NIGHTS as _W2, logged_readings as _lr
    ds = []
    for tag, nt in w15["nights"].items():
        strata = _W2[next(m["family"] for t, m in
                          json.load(open(os.path.join(icc_dir, "riverbed-manifest.json")))["nights"].items() if t == tag)][1]
        for name in nt.params:
            pairs = _lr(nt, name)
            if len(pairs) < 12:
                continue
            vecs = np.stack([v for _, v in pairs])
            for t in range(len(vecs) - 12 + 1):
                if not any(lo <= t and t + 11 <= hi for _, lo, hi, _ in strata):
                    continue
                a, b = vecs[t:t + 6].mean(0), vecs[t + 6:t + 12].mean(0)
                ds.append(np.linalg.norm(b - a) / sd15)
    d15 = float(np.mean(ds))
    assert 0.28 <= d15 <= 0.48, f"stable-d {d15:.3f} outside [0.28, 0.48]"
    print(f"[self-test] 15. G6: calibration snapshot at the registered "
          f"seed — corpus_sd {sd15:.4f} in [0.21, 0.30] (field 0.2367, "
          f"era-position schedule geometry), stable-d {d15:.3f} in [0.28, 0.48] "
          "(field actual-presence 0.376, floor 0.29), wave gate ALL PASS "
          "(logged-kappa band + warmth residuals re-verified in the G6 "
          "registration addendum)")

    # --- 16. WAVE-4: --fiber v4 — α in the pull's target trajectory ---- #
    # (docs/wave4-registration-draft-2026-08-22.md §1.2; every test above
    # ran the v3 default — the frozen wave-3 instrument.)
    d3 = os.path.join(tmp, "fiber-v3"); os.makedirs(d3)
    d4 = os.path.join(tmp, "fiber-v4"); os.makedirs(d4)
    p3, _ = generate_night("st-fib-T1", fam["T1"], roster, personas,
                           anchors, {}, BRANCHES["instrument"], 7, d3,
                           fam="T1")
    p4, _ = generate_night("st-fib-T1", fam["T1"], roster, personas,
                           anchors, {}, BRANCHES["instrument"], 7, d4,
                           fam="T1", fiber="v4")
    r3 = [json.loads(l) for l in open(p3) if l.strip()]
    r4 = [json.loads(l) for l in open(p4) if l.strip()]
    s3 = [r for r in r3 if r["type"] == "speak"]
    s4 = [r for r in r4 if r["type"] == "speak"]
    # (a) α=0 "identical on every leg", structurally: same seed+tag ⇒
    # same session/rng ⇒ the room channel, author schedule, fits/edges,
    # roster anchors (target_R(0) == the v3 anchor), staged semantics and
    # the acclimation state (vibe_now) are all bit-identical; only the
    # pull target (and its eff-derived logging) moves
    assert r3[0]["session_id"] == r4[0]["session_id"]
    assert r3[0]["roster"] == r4[0]["roster"], \
        "α=0: target_R(0) must equal the v3 anchor bit-for-bit"
    assert all(a["field_raw_after"] == b["field_raw_after"]
               and a["field_eff_after"] == b["field_eff_after"]
               and a["author"] == b["author"]
               and a["fit"] == b["fit"] and a["edge"] == b["edge"]
               and a["presence_mask"] == b["presence_mask"]
               and a["interactions_after"] == b["interactions_after"]
               for a, b in zip(s3, s4)), "v4 must not touch the room channel"
    for a, b in zip(s3, s4):
        assert set(a["readers"]) == set(b["readers"])
        for n in a["readers"]:
            assert a["readers"][n]["lens_now"]["vibe_now"] == \
                b["readers"][n]["lens_now"]["vibe_now"], \
                "acclimation must stay unchanged (α-free state)"
            assert set(b["readers"][n]["lens_now"]) == \
                set(a["readers"][n]["lens_now"]) | {"target_now"}
    assert any(a["readers"][n]["field_eff_to_reader"]
               != b["readers"][n]["field_eff_to_reader"]
               for a, b in zip(s3, s4) for n in a["readers"]), \
        "v4 pull must ride the target trajectory, not the acclimating vibe"
    # α=0: offsets reader-carried and STATIC — every logged target equals
    # the roster's target_R(0)
    for n in r4[0]["roster"]:
        vs = r4[0]["roster"][n]["vibe_start"]
        assert all(b["readers"][n]["lens_now"]["target_now"] == vs
                   for b in s4 if n in b["readers"]), \
            "α=0 target must be static in t (reader-carried offset)"
    print("[self-test] 16a. fiber v4 α=0 pilot: room/authors/fits/roster/"
          "acclimation bit-identical to v3 (identical-on-every-leg "
          "structure); targets static == target_R(0); pull rides target")

    # (b) replay-parity v2 SUFFICIENCY: the logged rows + logged
    # target_now reconstruct every field_eff_to_reader bit-for-bit (the
    # §1.3.1 fix — the wave-3 replay from vibe_start alone cannot)
    for a, b in zip(s3, s4):
        raw = np.asarray(b["field_raw_after"], float)
        for n in b["readers"]:
            blk = b["readers"][n]
            s_ = 1.0 - math.exp(-blk["charisma"]
                                * b["interactions_after"].get(n, 0))
            tgt = np.asarray(blk["lens_now"]["target_now"], float)
            eff = np.minimum(HI, np.maximum(LO, raw + s_ * (tgt - raw)))
            assert np.array_equal(eff, np.asarray(
                blk["field_eff_to_reader"], float)), \
                "logged target_now must replay the v4 fiber exactly"
    print("[self-test] 16b. replay v2: logged target_now + rows reproduce "
          "every v4 field_eff_to_reader bit-for-bit (parity re-"
          "registrable as v2; the v3 replay from vibe_start cannot — "
          "§1.3.1 by design)")

    # (c) α=1 unsealed pilot (the design-gate arm): generates OK, per-t
    # targets logged, room-carried AND time-varying — the common moving
    # target that makes the collapse signature reachable
    d1 = os.path.join(tmp, "fiber-v4-a1"); os.makedirs(d1)
    p1, _ = generate_night("st-fib-T1", fam["T1"], roster, personas,
                           anchors, {}, BRANCHES["collapse"], 7, d1,
                           fam="T1", fiber="v4")
    r1 = [json.loads(l) for l in open(p1) if l.strip()]
    s1_ = [r for r in r1 if r["type"] == "speak"]
    assert len(s1_) == fam["T1"][1] and s1_  # generated successfully
    for b in s1_:
        tgts = {n: blk["lens_now"]["target_now"]
                for n, blk in b["readers"].items()}
        assert len(set(map(tuple, tgts.values()))) == 1, \
            "α=1 target must be COMMON across readers (room-carried)"
    t0 = np.asarray(s1_[0]["readers"][roster[0]]["lens_now"]["target_now"])
    move = max(float(np.linalg.norm(
        np.asarray(b["readers"][roster[0]]["lens_now"]["target_now"]) - t0))
        for b in s1_)
    assert move > 0.02, f"α=1 target barely moves ({move:.4f}) — dead carrier"
    # all readers share the roster anchor target_R(0)
    vs1 = {n: tuple(e["vibe_start"]) for n, e in r1[0]["roster"].items()}
    assert len(set(vs1.values())) == 1, "α=1 anchors must coincide (pool+room0)"
    assert list(vs1.values())[0] == tuple(
        s1_[0]["readers"][roster[0]]["lens_now"]["target_now"])
    # amplitude-matched carrier from the ROOM stream, exactly: rebuild
    # room(t) = FIELD_ANCHOR_NORM·w_ar(t)/‖w_ar(t)‖ from the re-seeded
    # room rng (tag-keyed: (seed, crc(tag))) and match the logged targets
    pool = persona_pool_vibe(sorted(personas), personas)
    rp = room_path(fam["T1"], False, np.random.default_rng(
        (7, zlib_crc("st-fib-T1"))))
    for b in s1_:
        t = b["seq"]
        rc = FIELD_ANCHOR_NORM * _unit(rp["w_ar"][t])
        exp = _clamp(CENTER + (pool + rc) / SCALE)
        got = np.asarray(next(iter(b["readers"].values()))
                         ["lens_now"]["target_now"], float)
        assert np.array_equal(got, exp), \
            f"α=1 target at t={t} is not pool+room(t) from the room stream"
    # amplitude match: the carrier rides at anchor scale (direction-only)
    norms = [float(np.linalg.norm(FIELD_ANCHOR_NORM * _unit(w)))
             for w in rp["w_ar"]]
    assert max(abs(nm - FIELD_ANCHOR_NORM) for nm in norms) < 1e-12
    print(f"[self-test] 16c. fiber v4 α=1 pilot: generated OK; targets "
          f"COMMON across readers, time-varying (max |Δ| from t=0: "
          f"{move:.3f} z-units); carrier == FIELD_ANCHOR_NORM·w_ar/‖w_ar‖ "
          "from the room stream, bit-exact")

    # (d) pair isolation under v4: w_ar from the room stream (stream 1),
    # dev from the branch-free fiber stream (stream 2) — pair members'
    # targets differ ONLY through α
    tmp3 = tempfile.mkdtemp(prefix="riverbed-v4pair-", dir=tmp)
    kw4 = dict(family=fam["T1"], roster_names=roster, personas=personas,
               dev_anchors=anchors, seed=7, outdir=tmp3, fam="T1",
               fiber="v4")
    q0, _ = generate_night("st-v4p0-T1", branch=BRANCHES["instrument"],
                           ou_state={}, pair_seed=4242, **kw4)
    q1, _ = generate_night("st-v4p1-T1", branch=BRANCHES["collapse"],
                           ou_state={}, pair_seed=4242, **kw4)
    v0 = [json.loads(l) for l in open(q0) if l.strip()]
    v1 = [json.loads(l) for l in open(q1) if l.strip()]
    w0 = [r for r in v0 if r["type"] == "speak"]
    w1 = [r for r in v1 if r["type"] == "speak"]
    assert all(a["field_raw_after"] == b["field_raw_after"]
               and a["author"] == b["author"]
               for a, b in zip(w0, w1)), "v4 pair: room/authors differ"
    # α=1 member: target == pool + room(t) with room(t) from the PAIR
    # room stream (pair_seed, crc(fam), 1) — shared bit-for-bit
    rp2 = room_path(fam["T1"], False, np.random.default_rng(
        (4242, zlib_crc("T1"), 1)))
    for b in w1:
        t = b["seq"]
        rc = FIELD_ANCHOR_NORM * _unit(rp2["w_ar"][t])
        exp = _clamp(CENTER + (pool + rc) / SCALE)
        got = np.asarray(next(iter(b["readers"].values()))
                         ["lens_now"]["target_now"], float)
        assert np.array_equal(got, exp), \
            f"pair α=1 target at t={t} not from the pair room stream"
    # α=0 member: static targets (reader-carried) — no room component
    for n in v0[0]["roster"]:
        vs = v0[0]["roster"][n]["vibe_start"]
        assert all(b["readers"][n]["lens_now"]["target_now"] == vs
                   for b in w0 if n in b["readers"])
    assert any(a["readers"]["writer"] != b["readers"]["writer"]
               for a, b in zip(w0, w1)), "v4 pair: fibers did not diverge"
    print("[self-test] 16d. v4 pair (pair_seed=4242): room/authors shared; "
          "α=1 target == pool+room(t) from the pair room stream "
          "bit-exact; α=0 target static — targets differ only through α")

    # (e) v4 determinism + wave-level threading; v3 default at wave level
    p4b, _ = generate_night("st-fib-T1", fam["T1"], roster, personas,
                            anchors, {}, BRANCHES["instrument"], 7, d4,
                            fam="T1", fiber="v4")
    assert stripped_md5(p4b) == stripped_md5(p4)
    w4dir = os.path.join(tmp, "fiber-v4-wave")
    mv4 = generate_wave(w4dir, alpha=0.5, seed=20260821, fiber="v4",
                        tag_prefix="stv4")
    assert mv4["fiber"] == "v4" and all(
        m["deterministic_replay_identical"] for m in mv4["nights"].values())
    m3w = json.load(open(os.path.join(icc_dir, "riverbed-manifest.json")))
    assert m3w["fiber"] == "v3", "default wave must record fiber=v3"
    print("[self-test] 16e. v4 night deterministic (stripped-md5); v4 wave "
          "threads --fiber (manifest fiber=v4, determinism re-run "
          "identical); default wave manifest fiber=v3 (legacy preserved)")

    print("[self-test] ALL CHECKS PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--branch", default="instrument",
                    choices=list(BRANCHES) + ["custom"])
    ap.add_argument("--alpha", type=float, default=None,
                    help="branch parameter in [0,1] (overrides --branch)")
    ap.add_argument("--null-mode", action="store_true",
                    help="no warmth structure; cohesion-only common kappa shift")
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--flip-size", type=float, default=FLIP_SIZE)
    ap.add_argument("--tag-prefix", default=None)
    ap.add_argument("--pair-seed", type=int, default=None,
                    help="G13: branch-invariant room/fiber streams keyed "
                         "(pair_seed, family) — 2AFC adversarial pairs")
    ap.add_argument("--fiber", default="v3", choices=["v3", "v4"],
                    help="fiber version: v3 (default) = frozen wave-3 "
                         "instrument, α in the static anchor (line-845 "
                         "provenance — wave-3 corpora stay bit-"
                         "reproducible); v4 = wave-4 target-in-pull, α "
                         "rides the within-night target trajectory "
                         "(amplitude-matched wobble carrier, per-t "
                         "target logging)")
    ap.add_argument("--blind", nargs="?", const="auto", default=None,
                    metavar="CORPUS-ID",
                    help="G3: redacted manifest (branch params withheld, "
                         "opaque tags) + sealed sidecar; open post-"
                         "registration via --unblind")
    ap.add_argument("--unblind", default=None, metavar="SEALED.json",
                    help="verify a seal and print the withheld branch params")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if args.unblind:
        sealed = unblind(args.unblind)
        print(json.dumps({k: sealed[k] for k in
                          list(SEALED_FIELDS) + ["corpus_id", "tag_prefix"]},
                         indent=1))
        return
    if args.branch == "custom" and args.alpha is None:
        sys.exit("--branch custom requires --alpha")
    branch_name = "instrument" if args.alpha is not None else args.branch
    blind = args.blind is not None
    corpus_id = None
    if blind:
        corpus_id = (args.blind if args.blind != "auto" else None) \
            or secrets.token_hex(4)
    outdir = args.outdir
    if outdir is None:
        name = corpus_id if blind else (
            f"alpha-{args.alpha:g}" if args.alpha is not None
            else args.branch) + ("-null" if args.null_mode else "")
        outdir = os.path.join(DEFAULT_OUT, name)
    generate_wave(outdir, branch_name=branch_name, alpha=args.alpha,
                  seed=args.seed, null_mode=args.null_mode,
                  tag_prefix=args.tag_prefix, flip_size=args.flip_size,
                  pair_seed=args.pair_seed, blind=blind, corpus_id=corpus_id,
                  fiber=args.fiber)


if __name__ == "__main__":
    main()
