"""E5 — IDENTITY-PROPAGATION NULL: sharpening the class-conditional decomposition.

The side-by-side adjudication (e2-e3-side-by-side.md, filed 2026-08-19)
claimed, from the E2 field report: "class-residual ratio 0.4366 vs population
0.6088 — most of the baseline spread is archetype structure, not
class-independent idiosyncrasy." This script makes that claim precise,
corrects two convention problems, and tests it:

1. REPRODUCE the filed numbers exactly (population spread, class-residual
   spread, both estimators, canonical presence). Two corrections to the
   filed comparison, both reported: (a) it mismatched estimators — the
   class-residual number is an E-seg variant, the population number is
   E-cont; the like-for-like E-seg pair is 0.4366 vs the E-seg population
   ratio 0.6853; (b) on nights A/D/D-cold every reader is a singleton
   archetype, so class-residualization sets those cells' contribution to
   EXACTLY zero — the filed residual spread is mechanically deflated, and
   those cells are trivially 100% "between." The honest decomposition is an
   exact per-(cell, dial) one-way ANOVA, reported in two brackets: all
   signal cells (upper bracket, singleton cells all-between) and
   informative cells only (>=1 archetype with >=2 members present).

2. BOOTSTRAP (reader-level, B=2000, seed 20260820): 95% CI for the
   class-residual ratio (the filed point 0.4366 has no CI), the like-for-like
   residual share of spread, and the between-archetype variance share.

3. PERMUTATION (10,000 shuffles of archetype labels, fixed group-size
   multiset, seed 20260820): null = archetype labels are exchangeable
   across readers (archetype irrelevant). Statistics: between-archetype
   variance share on informative cells (primary), E-cont share, seg
   spread-reduction, and (exploratory, caveated) the drift-side share.

4. SENSITIVITIES: actual-presence instrument; S5 null cells included;
   barkeep excluded; propagation-only subset (singleton archetypes dropped:
   12 readers, 4 archetypes). Plus a residual-ICC check: after removing
   archetype structure, is any stable class-independent idiosyncrasy left?

CORRECTION DISCOVERED AND DEMONSTRATED IN-RUN: the instrument's
spread_seg(class_residual=True) mutates `vecs` in place while group means
are still being read, so multi-member groups are centered on a mix of
original and already-residualized vectors. The filed class-residual spread
0.3267 (ratio 0.4366) is INFLATED by this bug; the clean value is ~0.10
(ratio ~0.13). The bug's direction UNDERSTATES archetype structure — the
filed "most of the spread is archetype structure" was conservative.

Read-only against the nights; numpy-only, CPU, deterministic seeds.
Run:  python3 scripts/e5_identity_propagation.py
"""
from __future__ import annotations

import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.e2_field import field_readers
from scripts.e2_instrument import Measurement, Night, PRIMARY_NIGHTS, corpus_sd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "e5", "e5-identity-results.json")

SEED_BOOT = 20260820
SEED_PERM = 20260820
B_BOOT = 2000
N_PERM = 10_000


# --------------------------------------------------------------------------- #
# Exact one-way ANOVA on baseline vectors (unbalanced, nested in cells)       #
# --------------------------------------------------------------------------- #
def anova(cells, labels, informative_only=False, drop_singleton=False):
    """cells: {key: [(reader, vec), ...]} cell-median baselines.
    Exact identity ssb + ssw == sst holds per (cell, dial) and is asserted.
    drop_singleton: remove readers whose archetype has 1 member globally."""
    sizes = collections.Counter(labels.values())
    rs_keep = lambda r: (not drop_singleton) or sizes[labels[r]] > 1
    ssb = ssw = sst = 0.0
    n_cells_used = 0
    for key, pairs in cells.items():
        pairs = [(r, v) for r, v in pairs if rs_keep(r)]
        if len(pairs) < 2:
            continue
        groups = collections.defaultdict(list)
        for r, v in pairs:
            groups[labels[r]].append(v)
        if informative_only and max(len(g) for g in groups.values()) < 2:
            continue
        X = np.stack([v for _, v in pairs])
        grand = X.mean(axis=0)
        cell_ss_t = float(((X - grand) ** 2).sum())
        sst += cell_ss_t
        cell_ssb = cell_ssw = 0.0
        for idx in groups.values():
            gm = np.mean(idx, axis=0)
            cell_ssb += len(idx) * float(((gm - grand) ** 2).sum())
            cell_ssw += float(sum(((np.asarray(x) - gm) ** 2).sum() for x in idx))
        assert abs(cell_ssb + cell_ssw - cell_ss_t) \
            < 1e-9 * max(1.0, cell_ss_t), "ANOVA identity failed"
        ssb += cell_ssb
        ssw += cell_ssw
        n_cells_used += 1
    tot = ssb + ssw
    return {"share_btw": ssb / tot if tot > 0 else float("nan"),
            "ssb": ssb, "ssw": ssw, "n_cells": n_cells_used}


def share_fn(cells, informative_only=True):
    """Fast closure for permutation/bootstrapping of the between-share."""
    def f(labels):
        return anova(cells, labels, informative_only=informative_only)["share_btw"]
    return f


# --------------------------------------------------------------------------- #
# Instrument-convention spreads (RMS over dials of across-reader sd, ddof=1)  #
# --------------------------------------------------------------------------- #
def seg_spread(cell_list, sd, labels=None):
    """E-seg spread over cells; if labels given, class-residualized first
    (mirrors Measurement.spread_seg(class_residual=True), singleton
    residuals exactly 0 — the filed convention)."""
    sqs = []
    for pairs in cell_list:
        if labels is not None:
            groups = collections.defaultdict(list)
            for r, v in pairs:
                groups[labels[r]].append(v)
            gmean = {a: np.mean(vs, axis=0) for a, vs in groups.items()}
            pairs = [(r, v - gmean[labels[r]]) for r, v in pairs]
        if len(pairs) < 2:
            continue
        B = np.stack([v for _, v in pairs])
        sqs.append(float(np.mean(B.std(axis=0, ddof=1) ** 2)))
    return float(np.sqrt(np.mean(sqs))) / sd if sqs else float("nan")


def cont_baselines(m, readers=None):
    rs = list(m.readers if readers is None else readers)
    base = {}
    for r in rs:
        vecs = [v for night in m.readings[r] for _, v in m.readings[r][night]]
        if vecs:
            base[r] = np.mean(vecs, axis=0)
    return base


def cont_spread(base, sd, labels=None):
    if len(base) < 2:
        return float("nan")
    if labels is not None:
        groups = collections.defaultdict(list)
        for r, v in base.items():
            groups[labels[r]].append(v)
        gmean = {a: np.mean(vs, axis=0) for a, vs in groups.items()}
        base = {r: v - gmean[labels[r]] for r, v in base.items()}
    B = np.stack([v for v in base.values()])
    return float(np.sqrt(np.mean(B.std(axis=0, ddof=1) ** 2))) / sd


# --------------------------------------------------------------------------- #
# Permutation test: archetype labels exchangeable across readers              #
# --------------------------------------------------------------------------- #
def perm_test(stat_fn, readers, labels_true, n=N_PERM, seed=SEED_PERM):
    obs = stat_fn(labels_true)
    rng = np.random.default_rng(seed)
    lab_list = [labels_true[r] for r in readers]
    hits = 0
    for _ in range(n):
        shuffled = [lab_list[j] for j in rng.permutation(len(readers))]
        if stat_fn(dict(zip(readers, shuffled))) >= obs:
            hits += 1
    return obs, (hits + 1) / (n + 1)


def drift_share(m, labels, readers=None):
    """Exploratory: ANOVA share of per-reader mean drift (scalar)."""
    rs = [r for r in (m.readers if readers is None else readers)
          if not np.isnan(m.drift[r][0])]
    x = np.array([m.drift[r][0] for r in rs])
    groups = collections.defaultdict(list)
    for i, r in enumerate(rs):
        groups[labels[r]].append(x[i])
    grand = x.mean()
    sst = float(((x - grand) ** 2).sum())
    ssb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups.values())
    return ssb / sst if sst > 0 else float("nan")


def residual_icc(m, labels, seed=SEED_BOOT, B=B_BOOT):
    """ICC of night-level baselines AFTER removing (night x archetype) means
    — run only on readers of multi-member archetypes. Answers: is any
    stable class-independent idiosyncrasy left once archetype is removed?
    Same ICC formula as the instrument (schedule means removed first)."""
    sizes = collections.Counter(labels.values())
    rs = [r for r in m.readers if sizes[labels[r]] > 1]
    def icc_of(readers_):
        per_dial = []
        for d in range(7):
            rows = []
            for night, nb in m.night_base.items():
                vals = [(r, nb[r][d]) for r in readers_ if r in nb]
                if len(vals) < 2:
                    continue
                m_all = float(np.mean([v for _, v in vals]))
                arch_mean = collections.defaultdict(list)
                for r, v in vals:
                    arch_mean[labels[r]].append(v)
                arch_mean = {a: float(np.mean(vs)) for a, vs in arch_mean.items()}
                for r, v in vals:
                    rows.append((r, (v - m_all) - (arch_mean[labels[r]]
                                                   - m_all)))
            by_r = collections.defaultdict(list)
            for r, v in rows:
                by_r[r].append(v)
            within = [float(np.var(vs, ddof=1)) for vs in by_r.values()
                      if len(vs) > 1]
            s2w = float(np.mean(within)) if within else 0.0
            means = [float(np.mean(vs)) for vs in by_r.values()]
            s2b = float(np.var(means, ddof=1)) if len(means) > 1 else 0.0
            per_dial.append(s2b / (s2b + s2w) if (s2b + s2w) > 0 else np.nan)
        return float(np.nanmean(per_dial))
    point = icc_of(rs)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        rrs = [rs[i] for i in rng.integers(0, len(rs), len(rs))]
        v = icc_of(rrs)
        if not np.isnan(v):
            boots.append(v)
    lo, hi = (float(np.percentile(boots, 2.5)),
              float(np.percentile(boots, 97.5))) if boots else (np.nan, np.nan)
    return point, lo, hi, len(rs)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def build(m):
    """Precompute cached structures from a Measurement."""
    sd = m.sd
    signal_cells = {k: sorted(m.cell_base[k].items()) for k in m.cell_order}
    s5_cells = {k: sorted(v.items()) for k, v in m.cell_base.items()
                if k not in m.cell_order}
    cell_list = list(signal_cells.values())
    return {"sd": sd, "signal_cells": signal_cells, "cell_list": cell_list,
            "s5_cells": s5_cells}


def main():
    nights = {n: Night(n) for n in PRIMARY_NIGHTS}
    sd, _ = corpus_sd(list(nights.values()))
    m = Measurement(field_readers(), sd, presence="canonical")
    m_act = Measurement(field_readers(), sd, presence="actual")
    C = build(m)
    readers = list(m.readers)
    labels = dict(m.arch)
    drift = m.drift_mean()
    base = cont_baselines(m)

    sizes = collections.Counter(labels.values())
    print("=" * 78)
    print("E5 — IDENTITY-PROPAGATION NULL (class structure of baseline spread)")
    print("=" * 78)
    print(f"\n[0] corpus: 9 primary nights, corpus_sd = {sd:.4f}, "
          f"{len(readers)} readers, {len(sizes)} archetypes")
    for a in sorted(sizes, key=lambda a: -sizes[a]):
        members = [r for r in readers if labels[r] == a]
        print(f"    archetype {a:<9} n={sizes[a]}  {members}")

    # ---- 1. reproduce the filed numbers (and demonstrate the bug) ----
    seg_pop = seg_spread(C["cell_list"], sd)
    seg_res = seg_spread(C["cell_list"], sd, labels)
    cont_pop = cont_spread(base, sd)
    cont_res = cont_spread(base, sd, labels)
    print("\n[1] REPRODUCTION (canonical presence, field readers):")
    print(f"    E-cont population spread  = {cont_pop:.4f}  "
          f"(filed 0.4556)  ratio {cont_pop/drift:.4f} (filed 0.6088)")
    print(f"    E-seg  population spread  = {seg_pop:.4f}  "
          f"(filed 0.5128)  ratio {seg_pop/drift:.4f} (filed 0.6853)")
    print(f"    E-seg  class-residual     = {seg_res:.4f}  "
          f"(filed 0.3267 BUGGY — clean value here)  ratio "
          f"{seg_res/drift:.4f} (filed 0.4366 BUGGY)")
    print(f"    E-cont class-residual     = {cont_res:.4f}  "
          f"ratio {cont_res/drift:.4f}   <- like-for-like E-cont pair")
    # instrument-bug demonstration: replicate the in-place-mutation loop
    key0 = ("S1", "warm")
    cell0 = dict(C["signal_cells"][key0])
    present0 = [r for r, _ in C["signal_cells"][key0]]
    vecs_mut = dict(cell0)
    groups0 = collections.defaultdict(list)
    for r in present0:
        groups0[labels[r]].append(r)
    for r in present0:  # the instrument's exact loop (vecs mutated in place)
        gm = np.mean([vecs_mut[x] for x in groups0[labels[r]]], axis=0)
        vecs_mut[r] = vecs_mut[r] - gm
    A0 = np.stack([vecs_mut[r] for r in present0])
    crit0 = [i for i, r in enumerate(present0) if labels[r] == "critic"]
    print(f"    BUG DEMO on cell {key0}: instrument loop leaves critic-group "
          f"residual mean |max| = "
          f"{float(np.abs(A0[crit0].mean(axis=0)).max()):.4f} "
          f"(correct group-centering: ~0) ->")
    print(f"    filed class-residual spread reproduces ONLY under the bug "
          f"({m.spread_seg(class_residual=True):.4f}); the bug INFLATES "
          f"residual spread, i.e. UNDERSTATES archetype structure")
    print(f"    like-for-like share of spread retained after residualization:")
    print(f"      E-seg:  {seg_res/seg_pop:.4f} retained -> "
          f"{1-seg_res/seg_pop:.4f} of spread, {1-(seg_res/seg_pop)**2:.4f} "
          f"of variance is archetype structure")
    print(f"      E-cont: {cont_res/cont_pop:.4f} retained -> "
          f"{1-(cont_res/cont_pop)**2:.4f} of variance is archetype structure")

    # ---- 2. exact ANOVA decomposition ----
    print("\n[2] EXACT PER-(CELL, DIAL) ANOVA (singleton cells trivially "
          "all-between):")
    cells_all = dict(C["signal_cells"])
    cells_info = {k: v for k, v in cells_all.items()
                  if max(collections.Counter(
                      labels[r] for r, _ in v).values()) >= 2}
    print(f"    informative cells: {len(cells_info)}/{len(cells_all)} "
          f"signal cells (>=1 archetype with >=2 present):")
    for k in sorted(cells_info):
        cc = collections.Counter(labels[r] for r, _ in cells_info[k])
        multi = {a: n for a, n in cc.items() if n >= 2}
        print(f"      {k[0]:<5} {k[1]:<14} n={len(cells_info[k])} "
              f"multi-member groups: {multi}")
    a_all = anova(cells_all, labels)
    a_info = anova(cells_all, labels, informative_only=True)
    a_cont = anova({"cont": sorted(base.items())}, labels)
    print(f"    between-archetype variance share (all signal cells):     "
          f"{a_all['share_btw']:.4f}   [upper bracket: singleton cells "
          f"100% between]")
    print(f"    between-archetype variance share (informative cells):    "
          f"{a_info['share_btw']:.4f}   [lower bracket: honest test]")
    print(f"    between-archetype variance share (E-cont global baselines): "
          f"{a_cont['share_btw']:.4f}")

    # ---- 3. bootstrap ----
    print(f"\n[3] BOOTSTRAP over readers (B={B_BOOT}, seed={SEED_BOOT}):")
    rng = np.random.default_rng(SEED_BOOT)
    n = len(readers)
    boot = collections.defaultdict(list)
    for _ in range(B_BOOT):
        rs = [readers[i] for i in rng.integers(0, n, n)]
        lab = {r: labels[r] for r in rs}
        cl = [[(r, v) for r, v in cell if r in rs] for cell in C["cell_list"]]
        cl = [c for c in cl if len(c) >= 2]
        d = float(np.mean([m.drift[r][0] for r in rs
                           if not np.isnan(m.drift[r][0])]))
        if not cl or np.isnan(d):
            continue
        sp, sr = seg_spread(cl, sd), seg_spread(cl, sd, lab)
        bb = cont_baselines(m, rs)
        cp, cr = cont_spread(bb, sd), cont_spread(bb, sd, lab)
        if np.isnan(sp) or sp <= 0 or np.isnan(sr):
            continue
        boot["ratio_res_seg"].append(sr / d)
        boot["ratio_pop_seg"].append(sp / d)
        boot["retained_seg"].append(sr / sp)
        boot["share_info"].append(
            anova({i: c for i, c in enumerate(cl)}, lab,
                  informative_only=True)["share_btw"])
        if not np.isnan(cp) and cp > 0 and not np.isnan(cr):
            boot["ratio_res_cont"].append(cr / d)
    ci = lambda xs: (float(np.percentile(xs, 2.5)),
                     float(np.percentile(xs, 97.5)))
    clean = lambda k: [x for x in boot[k] if not np.isnan(x)]
    for k in ("ratio_res_seg", "ratio_pop_seg", "retained_seg",
              "share_info", "ratio_res_cont"):
        xs = clean(k)
        lo, hi = ci(xs)
        print(f"    {k:<15} median={float(np.median(xs)):.4f}  "
              f"CI [{lo:.4f}, {hi:.4f}]  (B={len(xs)} finite)")

    # ---- 4. permutation tests ----
    print(f"\n[4] PERMUTATION (n={N_PERM}, fixed group-size multiset, "
          f"seed={SEED_PERM}): null = archetype labels exchangeable")
    cells_enum_all = {i: c for i, c in enumerate(C["cell_list"])}
    obs_all, p_all = perm_test(share_fn(cells_enum_all, False), readers, labels)
    obs_info, p_info = perm_test(share_fn(cells_enum_all, True), readers, labels)
    obs_cont, p_cont = perm_test(
        share_fn({"cont": sorted(base.items())}, False), readers, labels)
    lab_list0 = [labels[r] for r in readers]

    def reduction(labels_):
        rs = readers
        cl = [[(r, v) for r, v in cell if r in rs] for cell in C["cell_list"]]
        cl = [c for c in cl if len(c) >= 2]
        sp, sr = seg_spread(cl, sd), seg_spread(cl, sd, labels_)
        return 1 - sr / sp if sp > 0 and not np.isnan(sr) else float("nan")
    obs_red, p_red = perm_test(reduction, readers, labels)
    obs_dsh, p_dsh = perm_test(lambda lab: drift_share(m, lab), readers, labels)
    print(f"    between-share, informative cells : {obs_info:.4f}  "
          f"p = {p_info:.4f}   <- PRIMARY")
    print(f"    between-share, all signal cells  : {obs_all:.4f}  p = {p_all:.4f}")
    print(f"    between-share, E-cont baselines  : {obs_cont:.4f}  p = {p_cont:.4f}")
    print(f"    seg spread-reduction (1-resid/pop): {obs_red:.4f}  p = {p_red:.4f}")
    print(f"    drift-side share (exploratory)   : {obs_dsh:.4f}  p = {p_dsh:.4f}")

    # ---- 5. sensitivities ----
    print("\n[5] SENSITIVITIES:")
    # (a) S5 null cells included in the ANOVA
    cells_s5 = {**cells_all, **C["s5_cells"]}
    a_s5 = anova(cells_s5, labels, informative_only=True)
    _, p_s5 = perm_test(share_fn(
        {i: c for i, c in enumerate(cells_s5.values())}, True), readers, labels)
    print(f"    (a) S5 null cells in: informative share {a_s5['share_btw']:.4f}"
          f"  (n_cells={a_s5['n_cells']})  perm p = {p_s5:.4f}")

    # (b) actual-presence instrument
    Ca = build(m_act)
    base_a = cont_baselines(m_act)
    cells_a = {i: c for i, c in enumerate(Ca["cell_list"])}
    a_act = anova(cells_a, labels, informative_only=True)
    _, p_act = perm_test(share_fn(cells_a, True), readers, labels)
    sr_a = seg_spread(Ca["cell_list"], sd, labels)
    print(f"    (b) actual presence  : informative share {a_act['share_btw']:.4f}"
          f"  perm p = {p_act:.4f}; class-residual spread {sr_a:.4f} "
          f"(ratio {sr_a/m_act.drift_mean():.4f})")

    # (c) barkeep excluded
    rb = [r for r in readers if r != "barkeep"]
    lab_b = {r: labels[r] for r in rb}
    cells_b = {i: [(r, v) for r, v in c if r != "barkeep"]
               for i, c in cells_enum_all.items()}
    a_b = anova(cells_b, lab_b, informative_only=True)
    _, p_b = perm_test(share_fn(cells_b, True), rb, lab_b)
    d_b = float(np.mean([m.drift[r][0] for r in rb]))
    cl_b = [c for c in cells_b.values() if len(c) >= 2]
    sr_b = seg_spread(cl_b, sd, lab_b)
    print(f"    (c) barkeep excluded : informative share {a_b['share_btw']:.4f}"
          f"  perm p = {p_b:.4f}; class-residual ratio {sr_b/d_b:.4f} "
          f"(14 readers)")

    # (d) propagation-only: singleton archetypes dropped (12 readers, 4 groups)
    a_prop = anova(cells_all, labels, informative_only=True,
                   drop_singleton=True)
    rp = [r for r in readers if sizes[labels[r]] > 1]
    lab_p = {r: labels[r] for r in rp}
    cells_p = {i: [(r, v) for r, v in c if r in rp]
               for i, c in cells_enum_all.items()}
    _, p_prop = perm_test(share_fn(cells_p, True), rp, lab_p)
    base_p = {r: v for r, v in base.items() if r in rp}
    within = cont_spread(base_p, sd, lab_p)
    print(f"    (d) singleton archetypes out (n={len(rp)}, 4 groups): "
          f"informative share {a_prop['share_btw']:.4f}  perm p = {p_prop:.4f}; "
          f"within-archetype E-cont spread {within:.4f} corpus-sd")

    # (e) per-archetype cluster tightness (cont baselines, corpus-sd)
    print("    (e) within-archetype spread of global baselines (corpus-sd):")
    for a in sorted(sizes, key=lambda a: -sizes[a]):
        if sizes[a] < 2:
            continue
        grp = {r: v for r, v in base.items() if labels[r] == a}
        w = cont_spread(grp, sd)
        print(f"        {a:<9} n={sizes[a]}  within-spread {w:.4f}")

    # ---- verdict inputs ----
    within_share_info = 1 - a_info["share_btw"]
    lo_ratio, hi_ratio = ci(clean("ratio_res_seg"))
    lo_share, hi_share = ci(clean("share_info"))
    icc_res, icc_res_lo, icc_res_hi, n_multi = residual_icc(m, labels)
    print("\n[6] VERDICT INPUTS:")
    print(f"    class-residual ratio (E-seg, CLEAN) = {seg_res/drift:.4f}  "
          f"bootstrap 95% CI [{lo_ratio:.4f}, {hi_ratio:.4f}]")
    print(f"    between-archetype variance share (informative) = "
          f"{a_info['share_btw']:.4f}  CI [{lo_share:.4f}, {hi_share:.4f}]  "
          f"perm p = {p_info:.4f}")
    print(f"    => within-archetype (class-independent) share of baseline "
          f"variance: {within_share_info:.4f}")
    print(f"    residual-ICC (archetype means removed, {n_multi} multi-member-"
          f"archetype readers) = {icc_res:.4f}  CI "
          f"[{icc_res_lo:.4f}, {icc_res_hi:.4f}]")
    print(f"    (population ICC, filed, class structure included: 0.7714)")

    results = {
        "date": "2026-08-20", "corpus_sd": sd, "n_readers": len(readers),
        "archetype_sizes": dict(sizes),
        "reproduction": {
            "cont_pop": cont_pop, "seg_pop": seg_pop, "seg_res": seg_res,
            "cont_res": cont_res, "drift": drift,
            "ratio_cont": cont_pop / drift, "ratio_seg": seg_pop / drift,
            "ratio_res_seg": seg_res / drift,
            "ratio_res_cont": cont_res / drift,
        },
        "anova": {"all_cells": a_all, "informative": a_info,
                  "cont": a_cont},
        "bootstrap": {k: {"median": float(np.median(clean(k))),
                          "ci": list(ci(clean(k)))}
                      for k in boot},
        "permutation": {
            "informative_share": {"obs": obs_info, "p": p_info},
            "all_cells_share": {"obs": obs_all, "p": p_all},
            "cont_share": {"obs": obs_cont, "p": p_cont},
            "seg_reduction": {"obs": obs_red, "p": p_red},
            "drift_share_exploratory": {"obs": obs_dsh, "p": p_dsh},
        },
        "sensitivities": {
            "s5_in": {"share": a_s5["share_btw"], "p": p_s5},
            "actual_presence": {"share": a_act["share_btw"], "p": p_act,
                                "seg_res": sr_a,
                                "ratio_res": sr_a / m_act.drift_mean()},
            "barkeep_out": {"share": a_b["share_btw"], "p": p_b,
                            "ratio_res": sr_b / d_b},
            "propagation_only": {"share": a_prop["share_btw"], "p": p_prop,
                                 "within_cont": within},
        },
        "instrument_bug_correction": {
            "filed_class_residual_spread": 0.3267,
            "filed_ratio": 0.4366,
            "clean_spread": seg_res,
            "clean_ratio": seg_res / drift,
            "cause": ("e2_instrument.spread_seg(class_residual=True) mutates "
                      "vecs in place while group means are read; multi-member "
                      "groups centered on mixed original/residualized vectors; "
                      "inflates residual spread (understates class structure)"),
            "direction": "filed claim was conservative; correction strengthens E5",
        },
        "residual_icc": {"point": icc_res, "ci": [icc_res_lo, icc_res_hi],
                         "n_readers": n_multi},
        "within_share_informative": within_share_info,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"\n[e5] results -> {OUT}")


if __name__ == "__main__":
    main()
