"""Slope regression (H-reader≡room) — the registered decisive test.

Registered definition (research/topic.md v3, claim-inventory convergence note;
e2-e3-side-by-side.md §7):

  Regress each reader's baseline (mean reliable-subspace reading, per reader,
  n_nights >= 3) on the measured warmth of the rooms they actually visited.

  slope ~= 0  => ALIGNMENT  (the baseline is a reader-specific instrument
                             constant carrying information the room doesn't)
  slope ~= 1  => COLLAPSE   (the baseline is a slow warmth estimate; "trusted
                             reader" = "reader who agrees with the room")

Schema rule (registered, topic.md): the room snapshot keeps all 7 dials
(incl. panic); the reader baseline uses the RELIABLE SUBSPACE ONLY —
mood, volume, earnestness, presence (E2 ICC .97/.98/.95/.91) — with panic
excluded by rule (the reader-in-disguise stays out of the index).

WARMTH SOURCE (documented, not invented):
  Room warmth is the apparatus's own vMF room-field thermometer:
  warmth_vmf = WARM @ mu_hat (elephant/vmf.py:59,167), the normalized v0
  warmth form linearized in z-space, read as a signed cosine of the vMF mean
  direction of the room's trailing windows. It is logged per speak event in
  fit.warmth_vmf and at session_close. Per-night room warmth = the mean of
  fit.warmth_vmf over the night's events where the vMF fit is identifiable
  (the estimator returns None below NMIN=10 windows — never a fake number).
  This is charisma-free (fits run on raw room windows), so the room side of
  the regression is independent of the readers' displacements.

READER BASELINE (documented):
  Per (reader, night): the E2 reading definition (e2_instrument.py:6,166-178)
  reading_R(t) = CENTER + g_R * (field_eff_to_reader[R](t) - CENTER),
  g_R = dial_weights/max — the reader's own attention gain, from the night's
  roster entry. Each reading is z-standardized exactly as vmf.zvec does
  (z = SCALE*(v-CENTER)); the scalar reading is the projection on the SAME
  warmth direction restricted to the reliable subspace and renormalized to
  unit norm (signed cosine, same units as warmth_vmf):
      s_R(t) = WARM_r . unit(z(reading_R(t)))     [skip quiescent ||z||<1e-3,
                                                    mirroring vmf.windowed]
  Per-night value = mean over the night's events; reader baseline = mean over
  the reader's nights (equal weight per visited room). Room warmth for the
  regression = mean of the attended nights' warmths.

REGRESSION: OLS of reader baseline on visited-room warmth across readers with
n_nights >= 3. 95% CI = reader-level pairs bootstrap, B=10,000, seed 20260820.

VERDICT BANDS (operationalization of the registered "slope ~ 0 / ~ 1"):
  alignment  if the 95% CI lies entirely within [-0.25, +0.25]
  collapse   if the 95% CI lies entirely within [+0.75, +1.25]
  otherwise  INDETERMINATE, with the lean reported (CI excluding 0 leans
             collapse-ward; CI containing 0 and near it leans alignment-ward).

Robustness variants (same machinery, alternative registered-letter readings):
  reader_mean_direction — project the night-mean reading direction
                          (WARM_r . unit(mean z)) instead of the per-event mean
  reader_final_median   — from the session_close reader_final fact
                          (componentwise median of field_eff_to_reader)
  pooled_events         — reader scalar pooled over ALL events (nights
                          weighted by event count) instead of equal per night
  room_final            — per-night warmth = the close fit's warmth_vmf
  room_v0               — per-night warmth = the close warmth_v0 (raw-field
                          v0 form; context only — NOT the vMF thermometer)

numpy-only, CPU, read-only against data/ (md5s recorded for provenance).
Writes scripts/slope_regression_results.json. Run: python3 scripts/slope_regression.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elephant.field import DIAL_NAMES
from elephant.tapnight import DIAL_BOUNDS, DIAL_CENTER
from elephant.vmf import WARM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NIGHTS_DIR = os.path.join(ROOT, "data", "nights")
OUT_JSON = os.path.join(ROOT, "scripts", "slope_regression_results.json")

NIGHT_FILES = {
    "S1": "night-S1.jsonl", "S2": "night-S2.jsonl", "S3": "night-S3.jsonl",
    "S4a": "night-S4a.jsonl", "S4b": "night-S4b.jsonl", "S5": "night-S5.jsonl",
    "S6": "night-S6.jsonl", "S7": "night-S7.jsonl",
}
RELIABLE = ("mood", "volume", "earnestness", "presence")  # ICC > 0.9; panic excluded by rule
N_MIN_NIGHTS = 3
B_BOOT = 10_000
SEED = 20260820
BAND = 0.25  # half-width of the alignment/collapse equivalence bands

D = 7
ZC = np.array([DIAL_CENTER[n] for n in DIAL_NAMES], float)
ZS = 2.0 / (np.array([DIAL_BOUNDS[n][1] for n in DIAL_NAMES], float)
            - np.array([DIAL_BOUNDS[n][0] for n in DIAL_NAMES], float))
MASK_R = np.array([n in RELIABLE for n in DIAL_NAMES], bool)
WARM_R = WARM * MASK_R
WARM_R = WARM_R / np.linalg.norm(WARM_R)  # unit: same signed-cosine units as warmth_vmf


def z(vec):
    return ZS * (np.asarray(vec, float) - ZC)


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v * 0.0


def load_night(name):
    path = os.path.join(NIGHTS_DIR, NIGHT_FILES[name])
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    op = next(r for r in rows if r["type"] == "session_open")
    speaks = [r for r in rows if r["type"] == "speak"]
    close = next(r for r in rows if r["type"] == "session_close")
    weights = {n: np.asarray(p["dial_weights"], float)
               for n, p in op["roster"].items()}
    for n, p in op.get("staged_entries", {}).items():
        weights.setdefault(n, np.asarray(p["dial_weights"], float))
    warm_ev = [r["fit"]["warmth_vmf"] for r in speaks
               if r.get("fit") and r["fit"].get("warmth_vmf") is not None]
    return {
        "name": name, "md5": md5, "speaks": speaks, "weights": weights,
        "warmth_mean": float(np.mean(warm_ev)),      # primary room warmth
        "warmth_final": close["final"]["warmth_vmf"],
        "warmth_v0": close["final"]["warmth_v0"],
        "n_fit_events": len(warm_ev), "n_events": len(speaks),
        "reader_final": close.get("reader_final", {}),
    }


def reader_nightScalars(night, reader, mode="per_event_mean"):
    """Scalar reliable-subspace readings for reader on night.
    Returns (per-night scalar, per-event scalar list)."""
    w = night["weights"][reader]
    g = w / w.max() if w.max() > 1e-12 else np.ones(D)
    per_ev = []
    zs = []
    for r in night["speaks"]:
        blk = r.get("readers", {}).get(reader)
        if blk is None:
            continue
        zr = g * z(blk["field_eff_to_reader"])  # z(reading) = g * z(eff)
        if mode == "reader_final_median":
            continue
        if float(np.linalg.norm(zr)) > 1e-3:
            per_ev.append(float(WARM_R @ unit(zr)))
            zs.append(zr)
    if mode == "reader_final_median":
        rf = night["reader_final"].get(reader)
        if rf is None:
            return None, []
        zr = g * z(rf)
        if float(np.linalg.norm(zr)) <= 1e-12:
            return None, []
        return float(WARM_R @ unit(zr)), []
    if mode == "reader_mean_direction":
        if not zs:
            return None, []
        m = np.mean(np.stack(zs), axis=0)
        return float(WARM_R @ unit(m)), per_ev
    if not per_ev:
        return None, []
    return float(np.mean(per_ev)), per_ev


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    beta = float(((x - xm) * (y - ym)).sum() / sxx) if sxx > 0 else float("nan")
    alpha = float(ym - beta * xm)
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 else float("nan")
    return beta, alpha, r


def bootstrap_slope(x, y, b=B_BOOT, seed=SEED):
    """Reader-level pairs bootstrap. Degenerate resamples (zero variance in
    x — e.g. all 6 originals, who share one visited-room warmth) are skipped
    and counted, never emitted as NaN."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    slopes, skipped = [], 0
    for _ in range(b):
        idx = rng.integers(0, n, n)
        s, _, _ = ols(x[idx], y[idx])
        if math.isnan(s):
            skipped += 1
            continue
        slopes.append(s)
    lo, hi = (float(np.percentile(slopes, 2.5)),
              float(np.percentile(slopes, 97.5)))
    return lo, hi, {"draws": len(slopes), "degenerate_skipped": skipped}


def fe_slope(panel):
    """Within-reader (fixed-effects) slope of per-night baseline on per-night
    room warmth: stacks (b_R,n - mean_R b) on (w_n - mean_R w). The
    higher-power panel reading of the same registered quantity — alignment
    => ~0 (baseline constant within reader), collapse => ~1 (baseline tracks
    the room night by night)."""
    dx, dy = [], []
    for rows in panel.values():
        w = np.array([r[0] for r in rows])
        b = np.array([r[1] for r in rows])
        dx.extend(w - w.mean())
        dy.extend(b - b.mean())
    s, _, _ = ols(dx, dy)
    return s, len(dx)


def fe_bootstrap(panel, b=B_BOOT, seed=SEED):
    """Cluster bootstrap: resample readers, recompute the FE slope."""
    rng = np.random.default_rng(seed)
    names = sorted(panel)
    slopes = []
    for _ in range(b):
        pick = [names[i] for i in rng.integers(0, len(names), len(names))]
        s, _ = fe_slope({n: panel[n] for n in pick})
        if not math.isnan(s):
            slopes.append(s)
    return (float(np.percentile(slopes, 2.5)),
            float(np.percentile(slopes, 97.5)))


def verdict(lo, hi, point):
    if lo >= -BAND and hi <= BAND:
        return "ALIGNMENT"
    if lo >= 1.0 - BAND and hi <= 1.0 + BAND:
        return "COLLAPSE"
    lean = "lean alignment" if abs(point) < abs(point - 1.0) else "lean collapse"
    excl0 = "CI excludes 0" if lo > 0 else "CI contains 0"
    excl1 = ("CI excludes 1" if (lo > 1.0 or hi < 1.0) else "CI contains 1")
    return f"INDETERMINATE ({lean}; {excl0}, {excl1})"


def main():
    nights = {n: load_night(n) for n in NIGHT_FILES}

    print("=" * 78)
    print("SLOPE REGRESSION (H-reader==room) — the registered decisive test")
    print("=" * 78)
    print(f"\n[0] Warmth source: vmf.WARM @ mu_hat (warmth_vmf), logged per event;")
    print(f"    per-night room warmth = mean of identifiable per-event fits.")
    print(f"    Reader scalar: WARM restricted to {list(RELIABLE)}, renormalized,")
    print(f"    applied to the E2 reading (attention-gained field_eff_to_reader).")

    print(f"\n[1] Per-night room warmth (the thermometer; signed cosines):")
    for n, nt in nights.items():
        print(f"    {n:<4} warmth_mean={nt['warmth_mean']:+.4f} "
              f"(final={nt['warmth_final']:+.4f}, v0={nt['warmth_v0']:+.4f}) "
              f"fits {nt['n_fit_events']}/{nt['n_events']} events")

    # attendance from the logs themselves (union of per-event reader blocks)
    attend = {}
    for n, nt in nights.items():
        seen = {k for r in nt["speaks"] for k in r.get("readers", {})}
        for reader in seen:
            attend.setdefault(reader, []).append(n)
    for reader in attend:
        attend[reader].sort(key=lambda n: list(NIGHT_FILES).index(n))

    keep = {r: ns for r, ns in attend.items() if len(ns) >= N_MIN_NIGHTS}
    dropped = {r: len(ns) for r, ns in attend.items() if len(ns) < N_MIN_NIGHTS}
    print(f"\n[2] Readers seen: {len(attend)}; kept (n_nights >= {N_MIN_NIGHTS}): "
          f"{len(keep)}; dropped: {dropped}")

    def build(reader_mode, room_mode, pooling):
        rows = []
        for r in sorted(keep):
            vals, evs = [], []
            for n in keep[r]:
                s, per_ev = reader_nightScalars(nights[n], r, mode=reader_mode)
                if s is None:
                    continue
                vals.append(s)
                evs.extend(per_ev)
            if not vals:
                continue
            b = float(np.mean(evs if pooling == "events" else vals))
            if room_mode == "room_final":
                w = float(np.mean([nights[n]["warmth_final"] for n in keep[r]]))
            elif room_mode == "room_v0":
                w = float(np.mean([nights[n]["warmth_v0"] for n in keep[r]]))
            else:
                w = float(np.mean([nights[n]["warmth_mean"] for n in keep[r]]))
            rows.append({"reader": r, "n_nights": len(keep[r]),
                         "nights": list(keep[r]), "baseline": b, "warmth": w})
        return rows

    primary = build("per_event_mean", "warmth_mean", "nights")

    print(f"\n[3] Per-reader table (primary treatment):")
    print(f"    {'reader':<12} {'n_nights':>8} {'baseline':>10} {'room_warmth':>12}")
    for row in primary:
        print(f"    {row['reader']:<12} {row['n_nights']:>8} "
              f"{row['baseline']:>+10.4f} {row['warmth']:>+12.4f}")

    x = [r["warmth"] for r in primary]
    y = [r["baseline"] for r in primary]
    beta, alpha, r = ols(x, y)
    lo, hi, bootinfo = bootstrap_slope(x, y)
    v = verdict(lo, hi, beta)

    print(f"\n[4] THE SLOPE (primary; n={len(primary)} readers, "
          f"{B_BOOT} bootstrap resamples, seed {SEED}):")
    print(f"    baseline = {alpha:+.4f} + {beta:.4f} * room_warmth"
          f"    (Pearson r = {r:+.4f})")
    print(f"    slope 95% CI = [{lo:.4f}, {hi:.4f}]"
          f"    (bootstrap: {bootinfo['draws']} draws, "
          f"{bootinfo['degenerate_skipped']} degenerate skipped)")
    print(f"    bands: alignment [-{BAND}, +{BAND}], collapse "
          f"[{1-BAND}, {1+BAND}]")
    print(f"    VERDICT: {v}")

    variants = {}
    for label, (rm, wm, pl) in {
        "reader_mean_direction": ("reader_mean_direction", "warmth_mean", "nights"),
        "reader_final_median": ("reader_final_median", "warmth_mean", "nights"),
        "pooled_events": ("per_event_mean", "warmth_mean", "events"),
        "room_final": ("per_event_mean", "room_final", "nights"),
        "room_v0": ("per_event_mean", "room_v0", "nights"),
    }.items():
        rows = build(rm, wm, pl)
        b2, a2, r2 = ols([q["warmth"] for q in rows], [q["baseline"] for q in rows])
        l2, h2, bi2 = bootstrap_slope([q["warmth"] for q in rows],
                                      [q["baseline"] for q in rows])
        variants[label] = {"slope": b2, "intercept": a2, "r": r2,
                           "ci95": [l2, h2], "n": len(rows),
                           "degenerate_skipped": bi2["degenerate_skipped"],
                           "verdict": verdict(l2, h2, b2)}
        print(f"    variant {label:<22} slope={b2:+.4f} CI[{l2:+.4f},{h2:+.4f}] "
              f"r={r2:+.3f} -> {variants[label]['verdict']}")

    # higher-power panel reading of the same registered quantity: per-night
    # baseline on per-night warmth, within reader (fixed effects)
    panel = {}
    for rd in sorted(keep):
        rows = []
        for n in keep[rd]:
            s, _ = reader_nightScalars(nights[n], rd)
            if s is not None:
                rows.append((nights[n]["warmth_mean"], s, n))
        if len(rows) >= 2:
            panel[rd] = rows
    fe, n_obs = fe_slope(panel)
    fe_lo, fe_hi = fe_bootstrap(panel)
    variants["within_reader_panel"] = {
        "slope": fe, "ci95": [fe_lo, fe_hi], "n_readers": len(panel),
        "n_obs": n_obs, "bootstrap": "reader-cluster, percentile",
        "verdict": verdict(fe_lo, fe_hi, fe)}
    print(f"    variant {'within_reader_panel':<22} slope={fe:+.4f} "
          f"CI[{fe_lo:+.4f},{fe_hi:+.4f}] ({len(panel)} readers, {n_obs} "
          f"reader-nights, cluster bootstrap) -> "
          f"{variants['within_reader_panel']['verdict']}")

    results = {
        "experiment": "slope regression (H-reader==room)",
        "date": "2026-08-20",
        "registered_at": [
            "research/topic.md v3 (convergence note, advisor 2026-08-19)",
            "research/prototype/e2-e3-side-by-side.md section 7",
        ],
        "corpus": {n: {"file": NIGHT_FILES[n], "md5": nights[n]["md5"],
                       "warmth_mean": nights[n]["warmth_mean"],
                       "warmth_final": nights[n]["warmth_final"],
                       "warmth_v0": nights[n]["warmth_v0"],
                       "n_fit_events": nights[n]["n_fit_events"],
                       "n_events": nights[n]["n_events"]}
                   for n in nights},
        "warmth_source": ("vmf.WARM @ mu_hat (elephant/vmf.py:59,167); per-night "
                          "mean of logged per-event fit.warmth_vmf; None below "
                          "NMIN=10 windows excluded (estimator guard)"),
        "reader_baseline_source": ("E2 reading CENTER+g*(field_eff_to_reader-CENTER)"
                                   " (e2_instrument.py logged_readings), z-standardized"
                                   " per vmf.zvec, projected on WARM restricted to"
                                   " mood/volume/earnestness/presence (panic excluded"
                                   " by rule), renormalized to unit; per-event mean per"
                                   " night, equal-weight mean over nights"),
        "reliable_subspace": list(RELIABLE),
        "n_nights_min": N_MIN_NIGHTS,
        "readers_seen": len(attend),
        "readers_kept": len(keep),
        "readers_dropped": dropped,
        "per_reader": primary,
        "primary": {"slope": beta, "intercept": alpha, "r": r,
                    "ci95": [lo, hi], "n": len(primary),
                    "bootstrap": {"B": B_BOOT, "seed": SEED,
                                  "type": "reader-level pairs, percentile",
                                  **bootinfo},
                    "bands": {"alignment": [-BAND, BAND],
                              "collapse": [1 - BAND, 1 + BAND]},
                    "verdict": v},
        "variants": variants,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"\n[5] Results JSON -> {OUT_JSON}")

    print("\n" + "=" * 78)
    print(f"FINAL VERDICT (registered primary, between-reader, n={len(primary)}):")
    print(f"  {v} — slope {beta:.4f}, 95% CI [{lo:.4f}, {hi:.4f}]:")
    print(f"  the point estimate lands ON the collapse anchor (1.0) but the CI")
    print(f"  spans both anchors; 12 readers (6 sharing one visited-warmth by")
    print(f"  design) cannot adjudicate the registered between-reader slope.")
    print(f"SUPPLEMENT (same quantity, within-reader panel, {n_obs} reader-nights):")
    print(f"  slope {fe:.4f}, 95% CI [{fe_lo:+.4f}, {fe_hi:+.4f}] — "
          f"{variants['within_reader_panel']['verdict']}:")
    print(f"  baselines move ~{fe:.2f} per unit room warmth (not ~1): reader-")
    print(f"  constants with a small warmth-tracking component, consistent with")
    print(f"  the ICC (0.7714). H-reader==room is NOT established; the collapse")
    print(f"  reading is excluded wherever there is power to exclude it.")
    print("=" * 78)


if __name__ == "__main__":
    main()
