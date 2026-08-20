#!/usr/bin/env python3
"""scripts/silence_test.py — the REGISTERED frame-level falsifier (silence test).

Registration: zeroclaw-dissertation/research/topic.md ~line 48 (hermes idea 1,
adopted as a test): "rerun the classification suite with inter-event silence
durations as the ONLY features; if scores match without any edge content, the
apparatus has been measuring the clock." Advisor constraint (pre-run): the
silence feature's SOURCE must be specified, and STEP=60 auto-clock sessions
are EXCLUDED — synthetic timestamps make silence trivially constant there.
The test only has teeth where the clock varies: real-timed sessions or a
text-length reading-time proxy.

SILENCE SOURCE SPECIFICATION (decided before any classifier ran):
  1. roomd-field-log.jsonl (real wall-clock, 2.005s poll): message arrivals
     derived from per-room message-count increments. TEETH CHECK run first
     (teeth_audit_roomd): arrivals are a +3-message generator tick every
     ~900.0s (8/11 gaps within poll-jitter of the median; the-bridge had 0
     arrivals in ~174 min) — near-constant silence is trivial by the
     advisor's own exclusion logic => EXCLUDED as primary, numbers reported.
  2. production-log.jsonl (real wall-clock): single room (bar-rail); gaps are
     the 30-min measurement poll cadence — the apparatus's own sampling
     clock, not conversation pacing; single class => cannot classify.
     EXCLUDED, numbers reported.
  3. PRIMARY (teeth verified): text-length reading-time proxy over the
     nights corpus speak events. Inter-event silence before message i is
         tau_i = T_BASE + T_CHAR * len_i
     (a message takes as long to arrive as its text takes to read). The
     proxy is affine in len, so "silence-only" == "length-only": length is
     NOT one of the 7 dials — exactly the non-field structure the falsifier
     must rule in or out. Teeth (measured): SEG1 lens 76.0+/-12.2 vs SEG2
     61.5+/-17.1 chars (Welch d ~ 0.98) and len sequences differ A vs H.

CORPUS: nights A (=B=C, byte-identical len/dial sequences by construction,
used ONCE per condition_eval.py convention) + night-H (independent
transition geometry, different len sequence). D/D' excluded per suite
convention; S-series excluded by the auto60 rule. No session timestamp is
ever used as a feature; only len-derived tau is.

CLASSIFICATION LEVELS (mirroring the suite):
  L1 condition-level: SEG1 (warm-earnest, seq 0-19) vs SEG2 (cynical-banter,
     seq 20-39) — the fine condition edge; dial-tier analog chord 1.229.
  L2 night-identity: A vs H (encoder-tier analog: room-identity in-sample).

ARMS (per window, W=4 messages, stride 1 — the suite's condition window):
  (a) CONTENT, two strengths:
      a1 "dials": per-window mean+sd of the 7 logged field_eff_after dials
         (14 feats).
      a2 "field+fit" (the room field AS SHIPPED, fit-level — the quantities
         behind the 1.229 fine gap): dial mean+sd PLUS the vMF fit of the
         window's dial readings (unit mu_hat 7-dim, log kappa) via the same
         estimator semantics as elephant.contrast.vmf_fit_generic
         (mu = resultant direction; kappa by bisection on A_d(kappa)=rho
         with scipy.special.ive — the shipped estimator's own Bessel
         ratio). 22 feats.
  (b) SILENCE — inter-event silence durations ONLY: per-window [mean, sd,
      min, max] of tau (4 feats). NO cumulative/absolute time: window
      position encodes "SEG2 comes later", which IS the clock — excluded by
      design (anti-rigging rule).
  (c) SILENCE+POSITION (diagnostic, rigged on purpose): silence + cumulative
      proxy time. Expected ~perfect; shows what clock leakage looks like.

CLASSIFIERS: (i) numpy logistic regression (deterministic: zero init,
full-batch GD, train-standardized, L2 1e-3); (ii) nearest-centroid
(train-standardized centroid distance) — no learning, variance-robust at
this n. Eval: leave-one-night-out (A<->H, primary, per-direction reported)
+ within-night half-split (windows wholly in msgs 0-9 vs 10-19,
message-disjoint; tests positional stationarity of each arm). Metrics:
accuracy, rank AUC, exact two-sided binomial p vs chance (lgamma).

PRE-REGISTERED DECISION RULE (fixed before running; pooled LONO logistic):
  * content discriminates : acc_content >= 0.60 AND p_content < 0.05
  * CLOCK   verdict       : content discriminates AND acc_silence >=
                            acc_content - 0.10
  * OBJECT  verdict       : content discriminates AND acc_silence <=
                            acc_content - 0.15
  * otherwise             : INDETERMINATE
  NOTE (honesty clause, declared now): the rule was written for the two
  anticipated outcomes. A third shape — content FAILS its gate while
  silence-only strongly discriminates — is direct evidence in the CLOCK
  direction and will be reported as such, not absorbed into "indeterminate"
  silently. No post-hoc threshold is moved to force either label.

Output: prints the verdict block; writes SILENCE-TEST-2026-08-20.json.
CPU, numpy (+scipy.special Bessel ratio, same as the shipped estimator).
No git operations.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter

import numpy as np
from scipy.special import ive

ELEPHANT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ELEPHANT, "data")
NIGHTS = os.path.join(DATA, "nights")
W = 4                          # the suite's condition window
T_BASE, T_CHAR = 0.20, 0.005   # tau affine in len; scale-invariant here
LR, ITERS, L2 = 0.5, 4000, 1e-3
DIALS = 7


# --------------------------------------------------------------------- #
# Teeth audit: the real-timed sources, measured not assumed              #
# --------------------------------------------------------------------- #
def teeth_audit_roomd() -> dict:
    rows = [json.loads(l) for l in
            open(os.path.join(DATA, "roomd-field-log.jsonl")) if l.strip()]
    ts = [r["ts"] for r in rows]
    rooms = sorted(rows[0]["rooms"].keys())
    out = {"n_snapshots": len(rows), "span_min": (ts[-1] - ts[0]) / 60.0,
           "poll_gap_med_s": float(np.median(np.diff(ts))), "rooms": {}}
    for rm in rooms:
        counts = [r["rooms"][rm]["messages"] for r in rows]
        arrivals = [ts[i] for i in range(1, len(rows))
                    if counts[i] > counts[i - 1]]
        jumps = Counter(counts[i] - counts[i - 1] for i in range(1, len(rows))
                        if counts[i] > counts[i - 1])
        gaps = np.diff(arrivals) if len(arrivals) > 1 else np.array([])
        near_const = 0
        if len(gaps):
            med = float(np.median(gaps))
            near_const = int(np.sum(np.abs(gaps - med) <= 2.5))
        out["rooms"][rm] = {
            "n_arrivals": len(arrivals), "jump_sizes": dict(jumps),
            "n_gaps": int(len(gaps)),
            "gap_med_s": float(np.median(gaps)) if len(gaps) else None,
            "gap_sd_s": float(np.std(gaps)) if len(gaps) else None,
            "gaps_within_2.5s_of_median": near_const,
            "teeth": bool(len(gaps) >= 8 and near_const < 0.5 * len(gaps))}
    return out


def teeth_audit_production() -> dict:
    from datetime import datetime
    rows = [json.loads(l) for l in
            open(os.path.join(DATA, "production-log.jsonl")) if l.strip()]
    t = [datetime.fromisoformat(r["ts"]).timestamp() for r in rows]
    gaps = np.diff(t)
    return {"n_rows": len(rows),
            "rooms": dict(Counter(r.get("room") for r in rows)),
            "gap_med_s": float(np.median(gaps)),
            "gap_sd_s": float(np.std(gaps)), "teeth": False,
            "reason": "single room; gaps = 30-min measurement poll cadence "
                      "(the apparatus's sampling clock, not conversation "
                      "pacing); single class => classification impossible"}


def teeth_audit_nights_proxy() -> dict:
    out, seqs = {}, {}
    for tag, fn in (("A", "night-A.jsonl"), ("B", "night-B.jsonl"),
                    ("C", "night-C.jsonl"), ("H", "night-H.jsonl")):
        ev = load_night(os.path.join(NIGHTS, fn))
        lens = np.array([e["len"] for e in ev])
        seqs[tag] = lens.tolist()
        out[tag] = {"n": len(ev),
                    "seg1_len_mean": float(lens[:20].mean()),
                    "seg1_len_sd": float(lens[:20].std()),
                    "seg2_len_mean": float(lens[20:].mean()),
                    "seg2_len_sd": float(lens[20:].std())}
    s1, s2 = out["A"]["seg1_len_sd"], out["A"]["seg2_len_sd"]
    d = (out["A"]["seg1_len_mean"] - out["A"]["seg2_len_mean"]) / \
        math.sqrt((s1 ** 2 + s2 ** 2) / 2.0)
    out["identical_len_seq_A_B"] = seqs["A"] == seqs["B"]
    out["identical_len_seq_A_C"] = seqs["A"] == seqs["C"]
    out["identical_len_seq_A_H"] = seqs["A"] == seqs["H"]
    out["welch_d_seg1_vs_seg2_len"] = d
    out["teeth"] = abs(d) >= 0.5 and not out["identical_len_seq_A_H"]
    return out


# --------------------------------------------------------------------- #
# Corpus + features                                                      #
# --------------------------------------------------------------------- #
def load_night(path: str):
    ev = []
    for line in open(path):
        r = json.loads(line)
        if r.get("type") == "speak":
            ev.append({"seq": r["seq"], "len": r["len"],
                       "dials": np.asarray(r["field_eff_after"], float)})
    ev.sort(key=lambda e: e["seq"])
    return ev


def vmf_fit(X: np.ndarray):
    """Window vMF fit, same semantics as elephant.contrast.vmf_fit_generic:
    unit-normalize rows; mu_hat = resultant direction; kappa by bisection on
    A_d(kappa) = rho (scipy Bessel ratios). Returns (mu_hat 7, log kappa)."""
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    d = X.shape[1]
    r = X.mean(0)
    rho = float(np.linalg.norm(r))
    if rho < 1e-9:
        return r, 0.0
    mu = r / rho

    def A(k):
        return float(ive(d / 2.0, k) / ive(d / 2.0 - 1.0, k))

    lo, hi = 1e-6, 1000.0
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if A(mid) < rho:
            lo = mid
        else:
            hi = mid
    return mu, math.log(math.sqrt(lo * hi))


def windows(night_ev, night_tag: str):
    out = []
    taus = [T_BASE + T_CHAR * e["len"] for e in night_ev]
    for stratum, lo, hi, label in (("SEG1", 0, 20, 0), ("SEG2", 20, 40, 1)):
        for i in range(lo, hi - W + 1):
            chunk = night_ev[i:i + W]
            t = np.array(taus[i:i + W])
            D = np.stack([e["dials"] for e in chunk])
            mu, logk = vmf_fit(D)
            out.append({
                "night": night_tag, "label": label, "stratum": stratum,
                "start_seq": i,
                # a1: dials only
                "content_dials":
                    np.concatenate([D.mean(0), D.std(0)]),
                # a2: dials + the shipped fit geometry (mu_hat, log kappa)
                "content_field_fit":
                    np.concatenate([D.mean(0), D.std(0), mu, [logk]]),
                # b: silence durations only (position-free by construction)
                "silence": np.array([t.mean(), t.std(), t.min(), t.max()]),
                # c: rigged diagnostic
                "silence+position": np.concatenate(
                    [[t.mean(), t.std(), t.min(), t.max()],
                     [sum(taus[:i])]])})
    return out


# --------------------------------------------------------------------- #
# Classifiers                                                            #
# --------------------------------------------------------------------- #
def _fit_logreg(X, y):
    n = len(X)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = np.concatenate([(X - mu) / sd, np.ones((n, 1))], 1)
    w = np.zeros(Z.shape[1])
    for _ in range(ITERS):
        p = 1.0 / (1.0 + np.exp(-(Z @ w)))
        w -= LR * (Z.T @ (p - y) / n + L2 * np.r_[w[:-1], 0.0])
    return w, mu, sd


def _pred_logreg(m, X):
    w, mu, sd = m
    Z = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1)
    return Z @ w


def _fit_centroid(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    return (Z[y == 1].mean(0), Z[y == 0].mean(0), mu, sd)


def _pred_centroid(m, X):
    c1, c0, mu, sd = m
    Z = (X - mu) / sd
    # positive = closer to class-1 centroid => predict label 1
    return np.linalg.norm(Z - c0, axis=1) - np.linalg.norm(Z - c1, axis=1)


def auc(scores, y):
    scores = np.asarray(scores, float)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def binom_p_two_sided(k, n, p=0.5):
    def pmf(i):
        return math.exp(math.lgamma(n + 1) - math.lgamma(i + 1)
                        - math.lgamma(n - i + 1)
                        + i * math.log(p) + (n - i) * math.log(1 - p))
    pk = pmf(k)
    tol = pk * (1.0 + 1e-9) if pk > 0 else 0.0
    return min(1.0, sum(pmf(i) for i in range(n + 1)
                        if pmf(i) <= tol))


def run_arm(win, arm: str, mode: str, clf: str) -> dict:
    X = np.stack([w[arm] for w in win])
    y = np.array([w["label"] for w in win], float)
    nights = np.array([w["night"] for w in win])
    starts = np.array([w["start_seq"] for w in win])
    fit, pred = (_fit_logreg, _pred_logreg) if clf == "logreg" \
        else (_fit_centroid, _pred_centroid)

    if mode == "lono":
        folds = [(nights != t, nights == t) for t in sorted(set(nights))]
    else:  # within-night, message-disjoint halves of EACH stratum
        # (within-stratum position, not global start_seq: SEG2 restarts at 0)
        pos = np.array([w["start_seq"] - (0 if w["stratum"] == "SEG1"
                                          else 20) for w in win])
        folds = [((nights == nt) & (pos <= 10 - W),
                  (nights == nt) & (pos >= 10))
                 for nt in sorted(set(nights))]

    scores = np.full(len(y), np.nan)
    for tr, te in folds:
        scores[te] = pred(fit(X[tr], y[tr]), X[te])
    ok = ~np.isnan(scores)
    correct = (scores[ok] > 0) == (y[ok] == 1)
    return {"arm": arm, "mode": mode, "clf": clf, "n": int(ok.sum()),
            "accuracy": float(correct.mean()), "auc": auc(scores[ok], y[ok]),
            "binom_p_vs_chance":
                binom_p_two_sided(int(correct.sum()), int(ok.sum()))}


def run_lono_directions(win, arm: str, clf: str) -> dict:
    """Per-direction LONO (train A test H; train H test A)."""
    X = np.stack([w[arm] for w in win])
    y = np.array([w["label"] for w in win], float)
    nights = np.array([w["night"] for w in win])
    fit, pred = (_fit_logreg, _pred_logreg) if clf == "logreg" \
        else (_fit_centroid, _pred_centroid)
    out = {}
    for tr_n, te_n in (("A", "H"), ("H", "A")):
        s = pred(fit(X[nights == tr_n], y[nights == tr_n]),
                 X[nights == te_n])
        c = (s > 0) == (y[nights == te_n] == 1)
        out[f"train{tr_n}_test{te_n}"] = {
            "n": int(len(s)), "accuracy": float(c.mean()),
            "auc": auc(s, y[nights == te_n])}
    return out


def night_identity(win, arm: str, clf: str) -> dict:
    """L2: classify windows by NIGHT (A vs H). With only two nights a
    leave-one-night-out split trains on a single class — degenerate by
    construction (the encoder-tier held-out failure's own shape). The
    honest eval is the message-disjoint half-split: train on first-half
    windows of BOTH nights, test on second halves."""
    X = np.stack([w[arm] for w in win])
    y = np.array([1.0 if w["night"] == "H" else 0.0 for w in win])
    pos = np.array([w["start_seq"] - (0 if w["stratum"] == "SEG1" else 20)
                    for w in win])
    tr, te = pos <= 10 - W, pos >= 10
    fit, pred = (_fit_logreg, _pred_logreg) if clf == "logreg" \
        else (_fit_centroid, _pred_centroid)
    scores = pred(fit(X[tr], y[tr]), X[te])
    c = (scores > 0) == (y[te] == 1)
    return {"arm": arm, "clf": clf, "mode": "half (msg-disjoint)",
            "n": int(te.sum()), "accuracy": float(c.mean()),
            "auc": auc(scores, y[te]),
            "binom_p_vs_chance": binom_p_two_sided(int(c.sum()),
                                                    int(te.sum()))}


# --------------------------------------------------------------------- #
def main() -> int:
    global W
    print("=" * 74)
    print("SILENCE TEST — registered frame-level falsifier (2026-08-20)")
    print("=" * 74)

    audit = {"roomd": teeth_audit_roomd(),
             "production": teeth_audit_production(),
             "nights_proxy": teeth_audit_nights_proxy()}
    print("\n[1] TEETH AUDIT (silence sources, measured not assumed)")
    r = audit["roomd"]
    print(f"  roomd-field-log: {r['n_snapshots']} snapshots, "
          f"{r['span_min']:.1f} min, poll {r['poll_gap_med_s']:.3f}s")
    for rm, d in r["rooms"].items():
        print(f"    {rm:14s} arrivals={d['n_arrivals']} "
              f"jumps={d['jump_sizes']} gap_med={d['gap_med_s']}s "
              f"gap_sd={d['gap_sd_s']}s "
              f"near-median={d['gaps_within_2.5s_of_median']}/{d['n_gaps']} "
              f"teeth={d['teeth']}")
    p = audit["production"]
    print(f"  production-log: {p['n_rows']} rows rooms={p['rooms']} "
          f"gap_med={p['gap_med_s']:.1f}s gap_sd={p['gap_sd_s']:.1f}s "
          f"teeth={p['teeth']} — {p['reason']}")
    npx = audit["nights_proxy"]
    print(f"  nights proxy (PRIMARY): SEG1 len {npx['A']['seg1_len_mean']:.1f}"
          f"+/-{npx['A']['seg1_len_sd']:.1f} vs SEG2 "
          f"{npx['A']['seg2_len_mean']:.1f}+/-{npx['A']['seg2_len_sd']:.1f} "
          f"Welch d={npx['welch_d_seg1_vs_seg2_len']:.2f} | len seq A==B:"
          f"{npx['identical_len_seq_A_B']} A==C:{npx['identical_len_seq_A_C']} "
          f"A==H:{npx['identical_len_seq_A_H']} | teeth={npx['teeth']}")

    A = windows(load_night(os.path.join(NIGHTS, "night-A.jsonl")), "A")
    H = windows(load_night(os.path.join(NIGHTS, "night-H.jsonl")), "H")
    assert len(A) == len(H) == 2 * (20 - W + 1)
    corpus = A + H

    print(f"\n[2] L1 CONDITION CLASSIFICATION (SEG1 warm-earnest vs SEG2 "
          f"cynical-banter), W={W} stride 1, n windows/night="
          f"{len(A)}")
    l1 = {}
    for arm in ("content_dials", "content_field_fit", "silence",
                "silence+position"):
        for clf in ("logreg", "centroid"):
            for mode in ("lono", "half"):
                res = run_arm(corpus, arm, mode, clf)
                l1[f"{arm}|{clf}|{mode}"] = res
                tag = "  (RIGGED diagnostic)" if arm == "silence+position" \
                    else ""
                print(f"  {arm:18s} {clf:8s} {mode:4s} n={res['n']:3d} "
                      f"acc={res['accuracy']:.3f} auc={res['auc']:.3f} "
                      f"p={res['binom_p_vs_chance']:.2e}{tag}")
    dirs = {arm: {clf: run_lono_directions(corpus, arm, clf)
                  for clf in ("logreg", "centroid")}
            for arm in ("content_field_fit", "silence")}
    print("  LONO per-direction (logreg):")
    for arm in ("content_field_fit", "silence"):
        for k, v in dirs[arm]["logreg"].items():
            print(f"    {arm:18s} {k}: acc={v['accuracy']:.3f} "
                  f"auc={v['auc']:.3f} (n={v['n']})")

    print("\n[3] L2 NIGHT-IDENTITY (A vs H; silence HAS teeth here — len "
          "sequences differ)")
    l2 = {}
    for arm in ("content_field_fit", "silence"):
        for clf in ("logreg", "centroid"):
            res = night_identity(corpus, arm, clf)
            l2[f"{arm}|{clf}"] = res
            print(f"  {arm:18s} {clf:8s} n={res['n']:3d} "
                  f"acc={res['accuracy']:.3f} auc={res['auc']:.3f} "
                  f"p={res['binom_p_vs_chance']:.2e}")

    try:
        s = json.load(open(os.path.join(NIGHTS, "summary.json")))
        fine_chord = float(np.linalg.norm(
            np.array(s["nights"]["A"]["seg1_mu_hat"])
            - np.array(s["nights"]["A"]["seg2_mu_hat"])))
    except Exception:
        fine_chord = None
    print(f"\n  context: dial-tier fine gap reference (vMF mu_hat chord "
          f"SEG1 vs SEG2, night A) = {fine_chord}")

    # ---- mechanism: which dials ride message length? ----------------- #
    evA = load_night(os.path.join(NIGHTS, "night-A.jsonl"))
    evH = load_night(os.path.join(NIGHTS, "night-H.jsonl"))
    lens = np.array([e["len"] for e in evA + evH])
    dialm = np.stack([e["dials"] for e in evA + evH])
    dial_names = ["mood", "volume", "earnestness", "cynicism",
                  "joke_landing", "panic", "presence"]
    lz = (lens - lens.mean()) / lens.std()
    r_len = {dial_names[j]: float(np.mean(lz * ((dialm[:, j]
              - dialm[:, j].mean()) / (dialm[:, j].std() + 1e-12))))
             for j in range(DIALS)}
    print("\n  mechanism: Pearson r(message len, dial) pooled A+H:")
    for k, v in r_len.items():
        print(f"    {k:12s} r={v:+.3f}")

    # ---- window-size sensitivity (is content just noisy at W=4?) ------ #
    sens = {}
    for W8 in (4, 8):
        W = W8
        wA = windows(load_night(os.path.join(NIGHTS, "night-A.jsonl")), "A")
        wH = windows(load_night(os.path.join(NIGHTS, "night-H.jsonl")), "H")
        corp = wA + wH
        row = {}
        for arm in ("content_dials", "content_field_fit", "silence"):
            for clf in ("logreg", "centroid"):
                res = run_arm(corp, arm, "lono", clf)
                row[f"{arm}|{clf}"] = {"acc": res["accuracy"],
                                       "p": res["binom_p_vs_chance"],
                                       "n": res["n"]}
                print(f"  W={W} {arm:18s} {clf:8s} lono n={res['n']:3d} "
                      f"acc={res['accuracy']:.3f} "
                      f"p={res['binom_p_vs_chance']:.2e}")
        sens[f"W{W}"] = row
    W = 4

    # ---- verdict: the pre-registered rule, applied literally ---------- #
    acc_c = l1["content_field_fit|logreg|lono"]["accuracy"]
    p_c = l1["content_field_fit|logreg|lono"]["binom_p_vs_chance"]
    acc_s = l1["silence|logreg|lono"]["accuracy"]
    p_s = l1["silence|logreg|lono"]["binom_p_vs_chance"]
    content_discriminates = acc_c >= 0.60 and p_c < 0.05
    if content_discriminates and acc_s >= acc_c - 0.10:
        verdict = ("CLOCK: silence-only matches content-only — the apparatus "
                   "may be measuring the clock, not the room")
        third = None
    elif content_discriminates and acc_s <= acc_c - 0.15:
        verdict = ("OBJECT: content discriminates while silence-only "
                   "collapses — the frame has an object beyond the "
                   "length/pacing clock")
        third = None
    else:
        verdict = "INDETERMINATE under the pre-registered branches"
        third = None
        if not content_discriminates and p_s < 0.05 and acc_s > acc_c:
            third = ("THIRD SHAPE (declared honesty clause): content FAILED "
                     "its own discrimination gate while silence-only "
                     "discriminates significantly and ABOVE content — "
                     "direct evidence in the CLOCK direction at this level; "
                     "the falsifier fired")

    print("\n" + "=" * 74)
    print(f"VERDICT (pre-registered rule, pooled LONO logreg): {verdict}")
    print(f"  content(field+fit) acc={acc_c:.3f} (p={p_c:.2e}) | "
          f"silence-only acc={acc_s:.3f} (p={p_s:.2e}) | chance=0.500")
    if third:
        print(f"  {third}")
    print("=" * 74)

    out = {"date": "2026-08-20", "teeth_audit": audit,
           "window": {"W": W, "stride": 1,
                      "tau_model": f"tau = {T_BASE} + {T_CHAR}*len s"},
           "L1_condition": l1, "L1_lono_directions": dirs, "L2": l2,
           "mechanism_r_len_dial": r_len,
           "window_size_sensitivity_lono": sens,
           "dial_tier_fine_gap_reference": fine_chord,
           "decision_rule": {
               "content_discriminates": "acc>=0.60 and p<0.05 (LONO logreg)",
               "clock": "acc_silence >= acc_content-0.10",
               "object": "acc_silence <= acc_content-0.15"},
           "verdict": verdict, "third_shape_note": third}
    with open(os.path.join(ELEPHANT, "SILENCE-TEST-2026-08-20.json"),
              "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote SILENCE-TEST-2026-08-20.json (the dated .md report is "
          "written by hand from these numbers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
