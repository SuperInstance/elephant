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

κ-TRAJECTORY-FIRST DESIGN (ideation §4.1: "entry steps are concentration
and roster-composition events; flips are mean-direction events"):
  - The PRIMARY control channel of a night is κ(t): piecewise per stratum,
    with κ-EVENTS (multiplicative pulses with exponential relaxation) at
    registered entry positions, plus small jitter.
  - μ_room(t) is SLAVED to a direction-only warmth schedule: μ(t) =
    w(t)·Ŵ + sqrt(1−w(t)²)·e⊥(t), e⊥ ⊥ Ŵ a slow tangent random walk (the
    deadband drift floor). Warmth is DEFINED as the signed cosine Ŵ·μ̂ —
    the vmf.py direction-only convention. The magnitude-contaminated
    field.py warmth() (same weights applied to raw re-centered readings,
    collinear with field extremity) is NEVER used to set anything; it is
    only logged (warmth_v0) as the legacy channel, computed from the
    emitted readings. Flips are warmth-schedule jumps (μ events); entries
    are κ events with μ continuous by construction.

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
  python3 scripts/riverbed_generator.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
DEV_SCALE = 0.55         # norm of the persona-anchored deviation direction
OU_PHI = 0.9             # between-night deviation persistence
ICC_TARGET = 0.9076      # filed field-corpus ICC (the honesty parameter)
ORTH_WALK = 0.02         # e⊥ tangent random-walk step (deadband drift floor)
KAPPA_JITTER = 0.03      # multiplicative log-jitter on κ(t)
FLIP_SIZE = 0.5          # warmth jump at a warm→cynical flip (Δw)

BANK_CLASSES = ["MoodDial", "VolumeDial", "EarnestnessDial", "CynicismDial",
                "JokeLandingDial", "PanicDial", "PresenceDial",
                "ModelVsCodeDial", "VisionDial"]

# ----------------------------------------------------------------------- #
# Night shapes — the 9 frozen T-families as FIELD schedules.              #
# warmth = target signed cosine Ŵ·μ̂ (direction-only); base values mirror  #
# the filed roster-invariant ladder (STAGE2 §1: S2 .3187, S4a .4465,      #
# D/D-cold .6293, S4b .6319, S1/A .6551, S3 .7409, S5 .7589). Flips are   #
# warmth (μ) jumps of FLIP_SIZE centered on the base; entries are κ       #
# events only. κ: warm strata loose (10), cold strata tight (18).         #
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
    the wave's reader pool, normalized, scaled to DEV_SCALE. Persona space
    only — no estimator coordinate (roster-mean of READINGS, corpus_sd,
    o/d) is ever touched."""
    z = {n: SCALE * (np.asarray(personas[n]["vibe_start"], float) - CENTER)
         for n in names}
    mean = np.mean(np.stack([z[n] for n in names]), axis=0)
    return {n: DEV_SCALE * _unit(z[n] - mean) for n in names}


# ----------------------------------------------------------------------- #
# Room path: κ-first schedule, μ slaved to the direction-only warmth      #
# ----------------------------------------------------------------------- #
def room_schedule(family, null_mode, rng, flip_size=FLIP_SIZE):
    """(warmth(t), kappa(t)) arrays. κ is the designed channel: per-stratum
    levels + multiplicative κ-events at entries with exponential relaxation
    + jitter. Warmth is flat at base in null mode (cohesion-only shift:
    the κ structure stays, the μ structure goes)."""
    base, n, flip, entries = family
    w = np.full(n, base)
    if flip is not None and not null_mode:
        w[:flip] = base + flip_size / 2.0
        w[flip:] = base - flip_size / 2.0
    w = np.clip(w, -0.95, 0.95)
    kappa = np.where(w >= base, 10.0, 18.0) if not null_mode \
        else np.full(n, 14.0)
    if null_mode and flip is not None:  # cohesion-only common shift
        kappa = np.full(n, 14.0)
        kappa[flip:] = 20.0
    for e in entries:  # entry = κ event (pulse + relaxation), μ untouched
        for t in range(e, n):
            kappa[t] += 12.0 * math.exp(-(t - e) / 6.0)
    kappa = kappa * np.exp(rng.normal(0.0, KAPPA_JITTER, n))
    return w, kappa


def room_path(family, null_mode, rng, flip_size=FLIP_SIZE):
    """One sample path of the room base orbit: μ(t) on S⁶ with Ŵ·μ(t) =
    w(t) EXACTLY (direction-only warmth), e⊥ a slow tangent walk; latent
    per-message draws s_i ~ vMF(μ(i), κ(i)); observed windowed samples
    o_t = normalize(mean of trailing W_WIN s_i) — the engine's
    windowed-reading analog (this smoothing is what the logged fits see).
    """
    base, n, flip, entries = family
    w, kappa = room_schedule(family, null_mode, rng, flip_size)
    # e⊥(t): unit, ⊥ Ŵ, slow tangent random walk (the drift floor)
    e = rng.normal(size=D)
    e = _unit(e - (e @ WARM) * WARM)
    mus, s_lat = [], []
    for t in range(n):
        xi = rng.normal(size=D)
        xi = xi - (xi @ WARM) * WARM - (xi @ e) * e
        e = _unit(e + ORTH_WALK * xi)
        mus.append(w[t] * WARM + math.sqrt(max(0.0, 1.0 - w[t] ** 2)) * e)
        s_lat.append(vmf_sample(rng, mus[-1], kappa[t]))
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
                   flip_size=FLIP_SIZE):
    """Emit data path outdir/night-<tag>.jsonl. Returns (path, ou_state)
    with the OU state advanced for every ATTENDING reader (between-night
    step happens once per attended night, in fixed NIGHT_ORDER)."""
    alpha, ou_phi, kappa_r, redraw = branch
    rng = np.random.default_rng((seed, zlib_crc(tag)))
    n = family[1]
    room = room_path(family, null_mode, rng, flip_size)

    # --- reader fiber: advance OU / redraw deviations for this night --- #
    # ICC honesty: steady-state OU variance = (1−ICC)/ICC of the anchor
    # variance (between-night wobble ≈ 0.1018 of the persistent deviation).
    ou_sigma = DEV_SCALE * math.sqrt((1.0 - ICC_TARGET) / ICC_TARGET
                                     * (1.0 - ou_phi ** 2))
    dev_now = {}
    for name in roster_names:
        if redraw:
            dev_now[name] = DEV_SCALE * _unit(rng.normal(size=D))
        else:
            st = ou_state.get(name, np.zeros(D))
            st = ou_phi * st + ou_sigma * rng.normal(size=D)
            ou_state[name] = st
            dev_now[name] = dev_anchors[name] + st

    # --- author schedule (seeded rotation over the roster) --- #
    authors = [roster_names[i] for i in rng.integers(0, len(roster_names), n)]

    # --- reader fibers sampled against the room path --- #
    g, denom = {}, {}
    for name in roster_names:
        wt = np.asarray(personas[name]["dial_weights"], float)
        g[name] = wt / wt.max() if wt.max() > 1e-12 else np.ones(D)
        # The pipeline reads z_R = SCALE*g ⊙ (eff − CENTER); components with
        # a zero lens weight contribute 0 regardless of eff, so emit CENTER
        # there (never divide by zero) — engine-identical downstream values.
        dnm = SCALE * g[name]
        denom[name] = np.where(dnm > 1e-9, dnm, 1.0)
        denom[name] = (denom[name], dnm > 1e-9)
    x_reader = {name: [] for name in roster_names}   # unit z-space draws
    eff_reader = {name: [] for name in roster_names}  # dial-space images
    for t in range(n):
        for name in roster_names:
            m = _unit(room["mu"][t] + (1.0 - alpha) * dev_now[name])
            x = vmf_sample(rng, m, kappa_r)
            x_reader[name].append(x)
            dn, mask = denom[name]
            eff_reader[name].append(_clamp(
                CENTER + np.where(mask, x / dn, 0.0)))

    session_id = hashlib.md5(f"riverbed:{seed}:{tag}".encode()).hexdigest()
    rows = []
    rows.append({
        "v": 1, "type": "session_open", "session_id": session_id,
        "space_id": "The Tap", "t_start": 0.0, "clock_mode": "auto60",
        "reader": {"kind": "RoomElephant", "identity": "riverbed-v1",
                   "bank": list(BANK_CLASSES)},
        "params": {"W": W_WIN, "standardization": "z=2(v-c)/(hi-lo)",
                   "estimator": "vmf-mle-newton-v1", "kappa_max": 500},
        "roster": {name: {"name": name,
                          "dial_weights": [float(x) for x in personas[name]["dial_weights"]],
                          "acclimation_rate": float(personas[name]["acclimation_rate"]),
                          "charisma": float(personas[name]["charisma"]),
                          "vibe": list(personas[name]["vibe_start"]),
                          "vibe_start": list(personas[name]["vibe_start"])}
                   for name in roster_names},
        "reader_schema": {"version": 2, "field": "field_eff_to_reader",
                          "lens": ["vibe_now", "weights_now"],
                          "fit": "vmf-mle-newton-v1", "gate": "roster"},
    })

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
        for name in roster_names:
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
                    x_reader[name][max(0, t - W_WIN + 1):t + 1]),
            }
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
            "entry_mode": {name: "roster" for name in roster_names},
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
        "reader_final": {name: np.median(np.stack(eff_reader[name]),
                                         axis=0).tolist()
                         for name in roster_names},
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
# determinism re-run; branch recorded but sealable for blinded analysis)  #
# ----------------------------------------------------------------------- #
def generate_wave(outdir, branch_name="instrument", alpha=None, seed=20260821,
                  null_mode=False, tag_prefix=None, flip_size=FLIP_SIZE):
    os.makedirs(outdir, exist_ok=True)
    if branch_name in BRANCHES and alpha is None:
        branch = BRANCHES[branch_name]
    else:
        a = float(alpha)
        branch = (a, OU_PHI, KAPPA_R_DEFAULT, False)
        branch_name = f"alpha-{a:g}"
    prefix = tag_prefix or f"rb-{branch_name}" + ("-null" if null_mode else "")
    personas = load_personas()
    all_readers = sorted({n for names in ATTENDANCE.values() for n in names})
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
    for fam in NIGHT_ORDER:  # fixed order: OU advances per attended night
        path, ou_state = generate_night(
            tags[fam], NIGHT_FAMILIES[fam], ATTENDANCE[fam], personas,
            dev_anchors, ou_state, branch, seed, outdir,
            null_mode=null_mode, flip_size=flip_size)
        paths[fam] = path

    manifest = {"generated_by": "scripts/riverbed_generator.py",
                "kind": "riverbed-forward-model-sample-path",
                "seed": seed, "branch": branch_name,
                "alpha": branch[0], "ou_phi": branch[1],
                "kappa_R": branch[2], "redraw_dev_per_night": branch[3],
                "null_mode": null_mode, "flip_size": flip_size,
                "reader_schema": 2, "nights": {}}
    for fam in NIGHT_ORDER:
        tag = tags[fam]
        rows = [json.loads(l) for l in open(paths[fam], encoding="utf-8")
                if l.strip()]
        speaks = [r for r in rows if r["type"] == "speak"]
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
                         "entry_seqs": NIGHT_FAMILIES[fam][3]},
        }

    # determinism: re-run the whole wave into a temp dir, compare stripped
    with tempfile.TemporaryDirectory() as tmp:
        ou2: dict = {}
        for fam in NIGHT_ORDER:
            p2, ou2 = generate_night(tags[fam], NIGHT_FAMILIES[fam],
                                     ATTENDANCE[fam], personas, dev_anchors,
                                     ou2, branch, seed, tmp,
                                     null_mode=null_mode,
                                     flip_size=flip_size)
            assert stripped_md5(p2) == manifest["nights"][tags[fam]]["stripped_md5"], tags[fam]
            manifest["nights"][tags[fam]]["deterministic_replay_identical"] = True
    mpath = os.path.join(outdir, "riverbed-manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"[riverbed] branch={branch_name} alpha={branch[0]} "
          f"null_mode={null_mode} seed={seed}")
    print(f"[riverbed] 9 nights -> {outdir} "
          f"(determinism re-run: all stripped-md5 identical)")
    print(f"[riverbed] manifest -> {mpath}")
    return manifest


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
    anchors = persona_deviations(roster, personas)
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

    # --- 6. kappa-first: entries are kappa events, flips are mu events -- #
    rng = np.random.default_rng(1)
    rp4 = room_path(fam["T4a"], False, rng)
    e = 12
    k_jump = abs(rp4["kappa"][e] - rp4["kappa"][e - 1])
    mu_jump_at_entry = float(np.linalg.norm(rp4["mu"][e] - rp4["mu"][e - 1]))
    assert k_jump > 5.0 and mu_jump_at_entry < 0.05
    f = 20
    mu_jump_at_flip = float(np.linalg.norm(rp4["mu"][f] - rp4["mu"][f - 1]))
    assert mu_jump_at_flip > 10 * mu_jump_at_entry
    print(f"[self-test] 6. entry@12: |dK|={k_jump:.1f} vs |dmu|="
          f"{mu_jump_at_entry:.4f}; flip@20: |dmu|={mu_jump_at_flip:.3f} "
          "(entries are kappa events; flips are mu events)")

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
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if args.branch == "custom" and args.alpha is None:
        sys.exit("--branch custom requires --alpha")
    branch_name = "instrument" if args.alpha is not None else args.branch
    outdir = args.outdir
    if outdir is None:
        name = (f"alpha-{args.alpha:g}" if args.alpha is not None
                else args.branch) + ("-null" if args.null_mode else "")
        outdir = os.path.join(DEFAULT_OUT, name)
    generate_wave(outdir, branch_name=branch_name, alpha=args.alpha,
                   seed=args.seed, null_mode=args.null_mode,
                   tag_prefix=args.tag_prefix, flip_size=args.flip_size)


if __name__ == "__main__":
    main()
