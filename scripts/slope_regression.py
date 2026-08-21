"""SLOPE REGRESSION — the registered decisive test of H-reader≡room.

Registration (advisor, 2026-08-19; research/topic.md v3 +
research/prototype/e2-e3-side-by-side.md §7):

  Regress each reader's baseline (mean reliable-subspace reading, per-reader,
  n_nights >= 3) on the measured warmth of the rooms they actually visited.
  Slope ≈ 0 ⇒ alignment (baseline is a reader-specific instrument constant).
  Slope ≈ 1 ⇒ collapse (baseline is slow warmth; nurse-as-index dissolves).

Reader side (registered default):
  - readings from the E2 field instrument, CANONICAL presence (the registered
    primary instrument of the E2 field arm), all attended primary nights;
  - baseline vector = mean of the reader's reading vectors (E-cont convention,
    reused verbatim from scripts/e5_identity_propagation.py::cont_baselines);
  - reliable subspace ONLY: mood, volume, earnestness, presence — the ICC
    reliable subspace of topic.md's two-line schema rule; panic excluded by
    rule (cynicism / joke_landing are likewise outside the reliable subspace);
  - scalar = direction cosine of the z-standardized reliable-subspace baseline
    against the warm direction RESTRICTED to the same subspace and
    renormalized (vmf.WARM sliced) — commensurable with the room side, whose
    warmth_vmf = Ŵ·μ̂ is also a direction cosine. Slope = 1 keeps its
    registered meaning only in matched units.

Room side (the dissertation's solid half, untouched):
  - per-night warmth = mean of the logged per-speak vMF fits' warmth_vmf
    (the room-field thermometer, μ̂-projection, vmf-mle-newton-v1), consumed
    read-only from the night logs; a room's warmth is reader-independent;
  - reader's x = mean warmth over the nights they actually attended.

Inference: OLS with intercept; bootstrap over READERS (B=2000, seed 20260820)
for the 95% CI; permutation null (10,000 shuffles of x across readers, seed
20260820) for "is the slope distinguishable from no relationship at all".

E5 discipline: the primary reader side is the UNRESIDUALIZED registered
baseline. The class-residual variant (E5-corrected clean group-centering —
never the buggy in-place mutation of e2_instrument.spread_seg) runs only as a
labeled SENSITIVITY, and only on the 15-reader set: on the 7-reader primary
set every reader is a singleton archetype, so residualization is identically
zero there (disclosed, not hidden).

Read-only against the corpus; numpy-only, CPU, deterministic seeds.
Run:  python3 scripts/slope_regression.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elephant.vmf import CENTER, DIALS, SCALE, WARM
from scripts.e2_field import field_readers
from scripts.e2_instrument import (Measurement, Night, PRIMARY_NIGHTS,
                                   corpus_sd)
from scripts.e5_identity_propagation import cont_baselines, cont_spread

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "slope", "slope-regression-results.json")

SEED = 20260820
B_BOOT = 2000
N_PERM = 10_000
MIN_NIGHTS = 3  # registered inclusion rule

# The ICC reliable subspace (topic.md two-line schema rule): panic excluded
# by rule; cynicism/joke_landing outside the reliable set (ICC ~.64).
RELIABLE = ["mood", "volume", "earnestness", "presence"]
RIDX = [DIALS.index(d) for d in RELIABLE]
WARM_REL = WARM[RIDX] / np.linalg.norm(WARM[RIDX])


def room_warmth(night: Night) -> float:
    """Mean logged per-speak warmth_vmf over the night (fits need >=10
    windows, so early speaks carry None and are skipped)."""
    vals = [r["fit"]["warmth_vmf"] for r in night.speaks if r.get("fit")]
    return float(np.mean(vals))


def close_warmth(night: Night) -> float:
    for row in reversed(list(open(night.path, encoding="utf-8"))):
        r = json.loads(row)
        if r["type"] == "session_close":
            return float(r["final"]["warmth_vmf"])
    raise ValueError(night.name)


def reader_scalar(base_vec: np.ndarray, normalize: bool = True) -> float:
    """Baseline vector -> reliable-subspace warmth. normalize=True: direction
    cosine (commensurable with warmth_vmf). False: raw z-projection."""
    z = SCALE * (np.asarray(base_vec, float) - CENTER)
    zr = z[RIDX]
    if normalize:
        n = np.linalg.norm(zr)
        if n < 1e-12:
            return float("nan")
        zr = zr / n
    return float(WARM_REL @ zr)


def ols(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    vx = float(np.var(x, ddof=1))
    if vx <= 0 or len(x) < 3:
        return float("nan"), float("nan")
    slope = float(np.cov(x, y, ddof=1)[0, 1] / vx)
    return slope, float(np.mean(y) - slope * np.mean(x))


def boot_ci(x, y, B=B_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    slopes, ints = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        s, a = ols(x[idx], y[idx])
        if not np.isnan(s):
            slopes.append(s)
            ints.append(a)
    ci = lambda xs: [float(np.percentile(xs, 2.5)),
                     float(np.percentile(xs, 97.5))]
    return ci(slopes), ci(ints)


def perm_p(x, y, slope_obs, n=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        s, _ = ols(x, y[rng.permutation(len(y))])
        if not np.isnan(s) and abs(s) >= abs(slope_obs):
            hits += 1
    return (hits + 1) / (n + 1)


def build_points(m: Measurement, warmth_by_night, normalize=True):
    """Per-reader (x, y) points from a fitted Measurement."""
    base = cont_baselines(m)  # E-cont convention, E5 machinery verbatim
    pts = {}
    for r in m.readers:
        nights = sorted(m.readings[r])
        if not nights or r not in base:
            continue
        x = float(np.mean([warmth_by_night[n] for n in nights]))
        y = reader_scalar(base[r], normalize=normalize)
        pts[r] = {"n_nights": len(nights), "nights": nights, "x": x, "y": y,
                  "archetype": m.arch[r]}
    return pts


def regress(pts, readers, y_override=None):
    rs = [r for r in readers if r in pts]
    x = np.array([pts[r]["x"] for r in rs])
    y = np.array([pts[r]["y"] for r in rs]) if y_override is None \
        else np.asarray(y_override, float)
    slope, intercept = ols(x, y)
    out = {"readers": rs, "n": len(rs), "slope": slope,
           "intercept": intercept}
    if not np.isnan(slope):
        out["slope_ci"], out["intercept_ci"] = boot_ci(x, y)
        out["perm_p_two_sided"] = perm_p(x, y, slope)
    return out


def class_resid(y, labels, readers):
    """E5-CORRECTED group-centering: group means computed from the ORIGINAL
    values first, then subtracted (never the in-place mutation of
    e2_instrument.spread_seg(class_residual=True))."""
    means = {}
    for a in set(labels[r] for r in readers):
        members = [y[readers.index(r)] for r in readers if labels[r] == a]
        means[a] = float(np.mean(members))
    return np.array([y[i] - means[labels[r]] for i, r in enumerate(readers)])


def main():
    nights = {n: Night(n) for n in PRIMARY_NIGHTS}
    sd, _ = corpus_sd(list(nights.values()))

    print("=" * 78)
    print("SLOPE REGRESSION — H-reader≡room (registered decisive test)")
    print("=" * 78)

    # reproduction guard against the filed E2 numbers (drift => corpus/instrument
    # drift shows up here, as in e5_identity_propagation.py)
    m = Measurement(field_readers(), sd, presence="canonical")
    drift = m.drift_mean()
    spread_c = cont_spread(cont_baselines(m), sd)
    print(f"\n[0] guard: corpus_sd={sd:.4f} (filed 0.2367)  "
          f"E-cont spread={spread_c:.4f} (filed 0.4556)  "
          f"drift={drift:.4f} (filed 0.7483)")
    assert abs(sd - 0.2367) < 1e-3, "corpus_sd drifted from filed value"
    assert abs(spread_c - 0.4556) < 1e-3, "E-cont spread drifted from filed"
    assert abs(drift - 0.7483) < 1e-3, "drift drifted from filed value"

    warmth_mean = {n: room_warmth(nights[n]) for n in PRIMARY_NIGHTS}
    warmth_close = {n: close_warmth(nights[n]) for n in PRIMARY_NIGHTS}
    print("\n[1] room warmth (room-field thermometer, logged warmth_vmf):")
    for n in PRIMARY_NIGHTS:
        print(f"    {n:<7} mean-per-speak {warmth_mean[n]:+.4f}   "
              f"session_close {warmth_close[n]:+.4f}")

    pts = build_points(m, warmth_mean)
    primary = sorted(r for r in pts if pts[r]["n_nights"] >= MIN_NIGHTS)
    excluded = sorted(r for r in pts if pts[r]["n_nights"] < MIN_NIGHTS)

    print(f"\n[2] readers: {len(pts)} total; primary n_nights>={MIN_NIGHTS}: "
          f"{len(primary)}; EXCLUDED (listed, n_nights<3): {len(excluded)}")
    for r in excluded:
        print(f"    excluded {r:<13} n_nights={pts[r]['n_nights']} "
              f"archetype={pts[r]['archetype']}")

    print("\n[3] per-reader points (primary):")
    print(f"    {'reader':<13} {'arch':<9} {'n':>2} {'x=roomwarm':>11} "
          f"{'y=baseline':>11}")
    for r in primary:
        p = pts[r]
        print(f"    {r:<13} {p['archetype']:<9} {p['n_nights']:>2} "
              f"{p['x']:>+11.4f} {p['y']:>+11.4f}")

    res = regress(pts, primary)
    print(f"\n[4] PRIMARY: slope = {res['slope']:.4f}  "
          f"bootstrap 95% CI [{res['slope_ci'][0]:.4f}, "
          f"{res['slope_ci'][1]:.4f}]  (B={B_BOOT} over readers, seed {SEED})")
    print(f"    intercept = {res['intercept']:.4f}  "
          f"CI [{res['intercept_ci'][0]:.4f}, {res['intercept_ci'][1]:.4f}]")
    print(f"    permutation null (shuffle x, n={N_PERM}): "
          f"two-sided p = {res['perm_p_two_sided']:.4f}")
    contains_0 = res["slope_ci"][0] <= 0 <= res["slope_ci"][1]
    contains_1 = res["slope_ci"][0] <= 1 <= res["slope_ci"][1]
    print(f"    CI contains 0: {contains_0}   CI contains 1: {contains_1}")

    # ---------------- sensitivities (labeled; never swapped) ------------- #
    print("\n[5] SENSITIVITIES (labeled; the registered primary stands):")
    sens = {}

    all15 = sorted(pts)
    sens["all_15_readers_n2"] = regress(pts, all15)
    print(f"    (a) all 15 readers (n_nights>=2, registration-relaxed): "
          f"slope {sens['all_15_readers_n2']['slope']:.4f} "
          f"CI {['%.4f' % v for v in sens['all_15_readers_n2']['slope_ci']]}")

    m_act = Measurement(field_readers(), sd, presence="actual")
    pts_act = build_points(m_act, warmth_mean)
    sens["actual_presence"] = regress(
        pts_act, sorted(r for r in pts_act
                        if pts_act[r]["n_nights"] >= MIN_NIGHTS))
    print(f"    (b) actual-presence instrument (participation-conflated, "
          f"+0.18 floor): slope {sens['actual_presence']['slope']:.4f} "
          f"CI {['%.4f' % v for v in sens['actual_presence']['slope_ci']]}")

    y15 = np.array([pts[r]["y"] for r in all15])
    y15r = class_resid(y15, m.arch, all15)
    sens["class_residual_15"] = regress(pts, all15, y_override=y15r)
    print(f"    (c) class-residual y (E5-CLEAN centering; 15 readers — the "
          f"7-reader primary set is all-singleton archetypes, residualization "
          f"is identically 0 there): slope "
          f"{sens['class_residual_15']['slope']:.4f} "
          f"CI {['%.4f' % v for v in sens['class_residual_15']['slope_ci']]}")

    pts_pos = build_points(m, warmth_mean, normalize=False)
    sens["unnormalized_z"] = regress(pts_pos, primary)
    print(f"    (d) reader side as raw z-projection (position, not direction"
          f" cosine): slope {sens['unnormalized_z']['slope']:.4f} "
          f"CI {['%.4f' % v for v in sens['unnormalized_z']['slope_ci']]}")

    pts_close = build_points(m, warmth_close)
    sens["session_close_warmth"] = regress(pts_close, primary)
    print(f"    (e) room side = session_close warmth_vmf (final-window fit): "
          f"slope {sens['session_close_warmth']['slope']:.4f} "
          f"CI {['%.4f' % v for v in sens['session_close_warmth']['slope_ci']]}")

    results = {
        "date": "2026-08-20",
        "test": "H-reader≡room registered slope regression "
                "(topic.md v3; side-by-side §7)",
        "corpus_sd": sd,
        "guard": {"cont_spread": spread_c, "drift": drift},
        "reader_side": {
            "instrument": "E2 field, canonical presence",
            "baseline": "mean reading vector (E-cont convention)",
            "subspace": RELIABLE,
            "scalar": "direction cosine of z-standardized reliable-subspace "
                      "baseline vs renormalized restricted vmf.WARM",
        },
        "room_side": {"per_night_warmth_mean_perspeak": warmth_mean,
                      "per_night_warmth_session_close": warmth_close},
        "min_nights": MIN_NIGHTS,
        "excluded_readers": {r: pts[r]["n_nights"] for r in excluded},
        "per_reader": pts,
        "primary": res,
        "sensitivities": sens,
        "seeds": {"bootstrap": SEED, "permutation": SEED},
        "branch_rule": {"slope≈0": "alignment (reader-specific instrument "
                                    "constant)",
                        "slope≈1": "collapse (baseline is slow warmth)"},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"\n[slope] results -> {OUT}")


if __name__ == "__main__":
    main()
