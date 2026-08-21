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

SKEW-PRODUCT READER FIBER (Agenda Problem 3; ideation §1.1/§2.1):
  each reader R is a second-level vMF field whose mean deviates from the
  room by a PERSONA-ANCHORED direction (persona space only — vibe_start /
  dial_weights, never estimator coordinates):

    m_R(n,t) = normalize( μ_room(t) + (1−α)·dev_R(n) ),  x_R(t) ~ vMF(m_R, κ_R)

  Branch parameter α:
    instrument (α=0): dev_R persistent across attended nights, OU-evolved
      in R⁷ between nights (φ=0.9, innovation sized from the filed
      ICC=0.9076 honesty target, ideation §1.4);
    collapse (α=1): the reader's sampling distribution IS the room's;
    noise: dev_R redrawn per night, κ_R low (μ̂_R unstable by design);
    intermediate: any --alpha in [0,1].
  Reader baselines are constant WITHIN a night (P ≈ 0.994 persistence by
  construction) and OU-drifted BETWEEN nights (the ICC knob).

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
DEV_SCALE = 0.85         # norm of the persona-anchored deviation direction
                         # (G7 calibration 2026-08-21: 0.85 under the
                         # CORRECTED κ semantics — entry eras are now loose
                         # (κ≈6.7 latent), which adds within-night variance
                         # the persistent deviation must dominate; realized
                         # end-to-end ICC 0.886 at the registered seed)
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
    the wave's reader pool, rescaled so the AVERAGE norm is DEV_SCALE.
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
    avg = float(np.mean([np.linalg.norm(v) for v in raw.values()]))
    c = DEV_SCALE / (avg + 1e-12)
    return {n: c * raw[n] for n in names}


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


def room_path(family, null_mode, rng, flip_size=FLIP_SIZE, e_state=None):
    """One sample path of the room base orbit: μ(t) on S⁶ with Ŵ·μ(t) =
    w(t) EXACTLY (direction-only warmth), e⊥ a slow tangent walk; latent
    per-message draws s_i ~ vMF(μ(i), κ(i)); observed windowed samples
    o_t = normalize(mean of trailing W_WIN s_i) — the engine's
    windowed-reading analog (this smoothing is what the logged fits see).

    e_state (G7): optional dict threading the tangent direction ACROSS the
    wave — one persistent latent path per corpus (The Tap is one space):
    the first night seeds it, later nights continue it. Within-night
    behavior is unchanged (same ORTH_WALK step). None ⇒ fresh draw
    (single-night / self-test use)."""
    base, n, flip, entries = family
    w, kappa = room_schedule(family, null_mode, rng, flip_size)
    # e⊥(t): unit, ⊥ Ŵ, slow tangent random walk (the drift floor)
    if e_state is not None and e_state.get("e") is not None:
        e = _unit(np.asarray(e_state["e"], float))
    else:
        e = rng.normal(size=D)
        e = _unit(e - (e @ WARM) * WARM)
    mus, s_lat = [], []
    for t in range(n):
        xi = rng.normal(size=D)
        xi = xi - (xi @ WARM) * WARM - (xi @ e) * e
        e = _unit(e + ORTH_WALK * xi)
        mus.append(w[t] * WARM + math.sqrt(max(0.0, 1.0 - w[t] ** 2)) * e)
        s_lat.append(vmf_sample(rng, mus[-1], kappa[t]))
    if e_state is not None:
        e_state["e"] = e
    obs = [_unit(np.mean(s_lat[max(0, t - W_WIN + 1):t + 1], axis=0))
           for t in range(n)]
    return {"w": w, "kappa": kappa, "mu": mus, "obs": obs}


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
def generate_night(tag, family, roster_names, personas, dev_anchors,
                   ou_state, branch, seed, outdir, null_mode=False,
                   flip_size=FLIP_SIZE, pair_seed=None, fam=None, e_state=None):
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
    fiber mean m_R = mu + (1-alpha)*dev. At fixed kappa_R the per-draw
    fiber counts are kappa-determined, so streams stay aligned across
    the branches of a pair.
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
    ou_sigma = DEV_SCALE * math.sqrt((1.0 - ICC_TARGET) / ICC_TARGET
                                     * (1.0 - ou_phi ** 2))
    dev_now = {}
    for name in present:
        if redraw:
            dev_now[name] = DEV_SCALE * _unit(rng.normal(size=D))
        else:
            st = ou_state.get(name, np.zeros(D))
            st = ou_phi * st + ou_sigma * rng.normal(size=D)
            ou_state[name] = st
            dev_now[name] = dev_anchors[name] + st

    # --- author schedule (seeded rotation over the roster; the staged --- #
    # entrant authors exactly his engine positions — draw count and call
    # order are branch-invariant, so paired corpora align) -------------- #
    authors = [roster_names[i] for i in rng.integers(0, len(roster_names), n)]
    for q in sorted(e_speaks):
        assert 0 <= q < n, f"entrant speak {q} outside night ({fam})"
        authors[q] = entrant

    # --- reader fibers sampled against the room path ------------------- #
    g, denom = {}, {}
    for name in present:
        wt = np.asarray(personas[name]["dial_weights"], float)
        g[name] = wt / wt.max() if wt.max() > 1e-12 else np.ones(D)
        # The pipeline reads z_R = SCALE*g ⊙ (eff − CENTER); components with
        # a zero lens weight contribute 0 regardless of eff, so emit CENTER
        # there (never divide by zero) — engine-identical downstream values.
        dnm = SCALE * g[name]
        denom[name] = np.where(dnm > 1e-9, dnm, 1.0)
        denom[name] = (denom[name], dnm > 1e-9)
    x_reader = {name: {} for name in present}    # unit z-space draws, by seq
    eff_reader = {name: {} for name in present}  # dial-space images, by seq
    for t in range(n):
        for name in present:
            if name == entrant and t < entry_seq:
                continue  # G1: no fiber draws before entry
            m = _unit(room["mu"][t] + (1.0 - alpha) * dev_now[name])
            x = vmf_sample(rng, m, kappa_r)
            x_reader[name][t] = x
            dn, mask = denom[name]
            eff_reader[name][t] = _clamp(
                CENTER + np.where(mask, x / dn, 0.0))

    session_id = hashlib.md5(f"riverbed:{seed}:{tag}".encode()).hexdigest()

    def _entry(name):
        """Roster-shaped param block (open roster and staged_entries share
        the engine's exact 6-key shape — verified on night-T4a.jsonl)."""
        return {"name": name,
                "dial_weights": [float(x) for x in personas[name]["dial_weights"]],
                "acclimation_rate": float(personas[name]["acclimation_rate"]),
                "charisma": float(personas[name]["charisma"]),
                "vibe": list(personas[name]["vibe_start"]),
                "vibe_start": list(personas[name]["vibe_start"])}

    open_row = {
        "v": 1, "type": "session_open", "session_id": session_id,
        "space_id": "The Tap", "t_start": 0.0, "clock_mode": "auto60",
        "reader": {"kind": "RoomElephant", "identity": "riverbed-v1",
                   "bank": list(BANK_CLASSES)},
        "params": {"W": W_WIN, "standardization": "z=2(v-c)/(hi-lo)",
                   "estimator": "vmf-mle-newton-v1", "kappa_max": 500},
        "roster": {name: _entry(name) for name in roster_names},
        "reader_schema": {"version": 2, "field": "field_eff_to_reader",
                          "lens": ["vibe_now", "weights_now"],
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
        fit = vmf_fit(room["obs"][:t + 1]) if t + 1 >= 10 else None
        edge = None
        if last_fit is not None and fit is not None:
            edge = vmf_edge(last_fit, fit)
            edge["real"] = None  # floor calibration is analysis-side
        if fit is not None:
            last_fit = fit

        readers, effs = {}, {}
        for name in present:
            if name == entrant and t < entry_seq:
                continue  # G1: absent from the readers block before entry
            x = x_reader[name][t]
            eff = eff_reader[name][t]
            effs[name] = eff
            m = _unit(room["mu"][t] + (1.0 - alpha) * dev_now[name])
            readers[name] = {
                "reader_known": True,
                "charisma": float(personas[name]["charisma"]),
                "field_eff_to_reader": eff.tolist(),
                "lens_now": {
                    "vibe_now": _clamp(CENTER + m / SCALE).tolist(),
                    "weights_now": [float(x_) for x_ in personas[name]["dial_weights"]],
                },
                "reader_fit": _reader_fit_light(
                    [x_reader[name][s] for s in range(max(0, t - W_WIN + 1), t + 1)
                     if s in x_reader[name]]),
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

    final_fit = vmf_fit(room["obs"])
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
                  pair_seed=None, blind=False, corpus_id=None):
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
            pair_seed=pair_seed, fam=fam, e_state=e_state)
        paths[fam] = path

    manifest = {"generated_by": "scripts/riverbed_generator.py",
                "kind": "riverbed-forward-model-sample-path",
                "seed": seed, "branch": branch_name,
                "alpha": branch[0], "ou_phi": branch[1],
                "kappa_R": branch[2], "redraw_dev_per_night": branch[3],
                "null_mode": null_mode, "flip_size": flip_size,
                "entry_dwarmth": ENTRY_DWARMTH,
                "pair_seed": pair_seed, "reader_schema": 2, "nights": {}}
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
                                     e_state=e2)
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
    assert s_i > 2.0 * s_c, f"instrument spread {s_i:.3f} !> 2x collapse {s_c:.3f}"
    print(f"[self-test] 7. baseline spread: instrument {s_i:.3f} >> "
          f"collapse {s_c:.3f} (alpha endpoints separate)")

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
    assert 0.85 <= icc <= 0.96, \
        f"realized instrument ICC {icc:.4f} outside [0.85, 0.96]"
    print(f"[self-test] 11. G7: instrument wave (21 readers x 9 families, "
          f"sd={wv['sd']:.4f}) realized ICC = {icc:.4f} in [0.85, 0.96] "
          "(registered Measurement, unmodified)")

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
                  pair_seed=args.pair_seed, blind=blind, corpus_id=corpus_id)


if __name__ == "__main__":
    main()
