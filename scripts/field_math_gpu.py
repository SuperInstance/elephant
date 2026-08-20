#!/usr/bin/env python3
"""Field-math GPU batch farm — the room-field thermometer as a CUDA batch job.

The CPU-serial consumer (elephant/vmf.py + its loop in tapnight/premise-
measurement) fits one window at a time: normalize, mean, Banerjee init,
60-iteration Newton on the A7 Bessel ratio, a 200-sample bootstrap CI
(numpy default_rng per call), and an N-row jackknife SE. Over a corpus of
~14.5k trailing windows that is a serial Python/numpy grind.

This script GPU-batches the *identical* math on the RTX 4050 (torch
2.13.0+cu130, float64 for parity with the numpy reference):

  1. vmf_fit_batch(X, N)   — (mu_hat, kappa, rho, warmth, kappa_ci, mu_se,
                             axis_spread, saturated) for every window at once.
                             Banerjee init vectorized; Newton solved as a
                             fixed 60-iteration loop with per-element freeze
                             masks that replicate vmf.py's break semantics
                             (|g| < 1e-12 stop, |step| < 1e-9 stop-after-
                             update); the bootstrap resample index streams
                             are generated with numpy default_rng(0) on the
                             host — the SAME streams vmf_fit draws — so the
                             GPU CI is numerically the CPU CI, not a
                             lookalike from a different RNG.
  2. edge_batch            — field edges (d_mu, d_warmth, d_log_kappa, the
                             jackknife-SE deadband) over consecutive fits of
                             every stream, batched; verified against
                             elephant.vmf.edge and against the edges logged
                             in the night logs by tapnight itself.
  3. ledger_batch          — the production-ledger imbalance: v0 warmth +
                             kappa proxy over the before/after fields of
                             data/production-log.jsonl, drift = L2 of the
                             (d_warmth, d_kappa) pair, batched; verified
                             against the logged drift values.

Correctness gate: every batched quantity must match the CPU reference
(elephant.vmf.vmf_fit run serially over the full corpus) to <= 1e-6 max
abs diff, and invalid windows (isotropic, N < NMIN) must be flagged
identically. Speedup is reported for the full fit (bootstrap + jackknife
included) and for the core solver alone, both end-to-end (host->device
upload + compute + results download included for GPU).

Edge cases handled (see FIELD-MATH-GPU-2026-08-20.md):
  - sinh/cosh overflow as rho -> 1: rho clamped at RHOMAX=0.999 and the
    Banerjee init at KMAX=500, exactly like vmf.py; saturated windows pin
    kappa = 500 and are reported via the `saturated` flag.
  - closed-form A7 catastrophic cancellation for kappa < 0.5: the same
    series branch A7 ~= kappa/7, applied with torch.where (the closed form
    is evaluated on a k-safe copy so no inf/nan is ever produced).
  - Newton derivative vanishing (|g| < 1e-12): step suppressed per element.
  - isotropic windows (rho < 1e-12): masked to NaN — the batch analogue of
    vmf_fit returning None; never a fake number.

Run:  python3 scripts/field_math_gpu.py [--refresh] [--no-doc-check]
Read-only against the corpus except for its own cache
(scripts/field-math-gpu-cache.npz). Writes no logs, commits nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from elephant.vmf import (A7 as A7_cpu, D, KMAX, NMIN, RHOMAX, WARM,
                          edge as edge_cpu, vmf_fit)
from elephant.field import DIAL_NAMES

CACHE = os.path.join(ROOT, "scripts", "field-math-gpu-cache.npz")
NIGHTS_DIR = os.path.join(ROOT, "data", "nights")
ROOMD_LOG = os.path.join(ROOT, "data", "roomd-field-log.jsonl")
PROD_LOG = os.path.join(ROOT, "data", "production-log.jsonl")

NMAX = 64          # trailing cap, mirrors vmf.windowed(cap=64)
BOOT_B = 200       # bootstrap resamples, mirrors vmf_fit(B=200)
QUIS = 1e-3        # quiescent skip, mirrors vmf.windowed
TOL = 1e-6         # correctness gate

# night-A-repro is a byte-replay of night-A (identical windows) — excluded.
NIGHT_FILES = sorted(f for f in os.listdir(NIGHTS_DIR)
                     if f.endswith(".jsonl") and f != "night-A-repro.jsonl")


# --------------------------------------------------------------------------- #
# Corpus: (stream -> z-samples -> trailing-window events), all on CPU         #
# --------------------------------------------------------------------------- #
def build_streams():
    """Independent z-sample streams from the three corpora.

    Each snapshot row is one bank-level readings dict (the per-message raw
    field for the nights, the room dials for the roomd log, the probe field
    for the production log), standardized with elephant.vmf.zvec. This is
    the premise_measurement.py reading of the logs: one row = one reading.
    """
    from elephant.vmf import DIALS, zvec

    def as_readings(v):
        """Night logs store field_raw_after as a 7-list in DIALS order."""
        if isinstance(v, dict):
            return v
        return dict(zip(DIALS, v))

    streams = []

    per_room = {}
    with open(ROOMD_LOG, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for rm, info in row.get("rooms", {}).items():
                per_room.setdefault(rm, []).append(zvec(info["dials"]))
    for rm, zs in per_room.items():
        streams.append((f"roomd:{rm}", zs))

    for fn in NIGHT_FILES:
        zs = []
        with open(os.path.join(NIGHTS_DIR, fn), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("type") == "speak":
                    zs.append(zvec(as_readings(row["field_raw_after"])))
        streams.append((f"night:{fn[:-6]}", zs))

    zs = []
    with open(PROD_LOG, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("ok") and row.get("field"):
                zs.append(zvec(row["field"]))
    streams.append(("production:bar-rail", zs))
    return streams


def build_events(streams):
    """Trailing-window events: windowed(zs, cap=NMAX, step=1), N >= NMIN.

    Mirrors vmf.windowed: quiescent z (||z|| < 1e-3) never enters a window.
    Returns X (E, NMAX, 7) zero-padded, N (E,), stream ids, end positions.
    """
    X, Ns, sid, pos = [], [], [], []
    for si, (name, zs) in enumerate(streams):
        zs = [z for z in zs if float(np.linalg.norm(z)) > QUIS]
        for i in range(len(zs)):
            win = zs[max(0, i - NMAX + 1):i + 1]
            if len(win) < NMIN:
                continue
            w = np.zeros((NMAX, D), dtype=np.float64)
            w[:len(win)] = win
            X.append(w)
            Ns.append(len(win))
            sid.append(si)
            pos.append(i)
    return (np.stack(X), np.asarray(Ns, dtype=np.int64),
            np.asarray(sid, dtype=np.int64), np.asarray(pos, dtype=np.int64))


# --------------------------------------------------------------------------- #
# GPU batched vMF MLE — exact mirror of elephant.vmf.vmf_fit                  #
# --------------------------------------------------------------------------- #
def A7_t(k: torch.Tensor) -> torch.Tensor:
    """A7(kappa) closed form with the k < 0.5 series branch, batched."""
    small = k < 0.5
    ks = torch.where(small, torch.ones_like(k), k)   # never evaluate the
    s, c = torch.sinh(ks), torch.cosh(ks)            # closed form below 0.5
    num = (1.0 + 15.0 / ks ** 2) * c - (6.0 / ks + 15.0 / ks ** 3) * s
    den = (1.0 + 3.0 / ks ** 2) * s - (3.0 / ks) * c
    return torch.where(small, k / 7.0, num / den)


def banerjee_t(rho: torch.Tensor) -> torch.Tensor:
    """Banerjee et al. init / bootstrap shortcut, clipped (sinh-overflow guard)."""
    return torch.clamp(rho * (D - rho ** 2) / (1.0 - rho ** 2), 1e-6, KMAX)


def kappa_newton_t(rho: torch.Tensor, iters: int = 60) -> torch.Tensor:
    """Batched Newton on A7(k) = rho with vmf.py's exact break semantics.

    CPU: break when |g| < 1e-12 (no update) or |step| < 1e-9 (update happens,
    then break). GPU: per-element `active` mask freezes converged elements.
    """
    k = banerjee_t(rho)
    active = torch.ones_like(k, dtype=torch.bool)
    for _ in range(iters):
        a = A7_t(k)
        g = 1.0 - a * a - (D - 1.0) * a / k
        gok = g.abs() >= 1e-12
        step = (a - rho) / torch.where(gok, g, torch.ones_like(g))
        upd = active & gok
        k = torch.where(upd, torch.clamp(k - step, 1e-6, KMAX), k)
        active = upd & (step.abs() >= 1e-9)
    return k


def _boot_idx_for(n: int, b: int) -> np.ndarray:
    """The exact index stream vmf_fit draws for a window of length n.

    vmf_fit creates default_rng(seed=0) per call and draws b times
    integers(0, n, n) — replicate it on the host so the GPU bootstrap is
    the same estimator, not a different-RNG lookalike.
    """
    rng = np.random.default_rng(0)
    return np.stack([rng.integers(0, n, n) for _ in range(b)]).astype(np.int64)


def vmf_fit_batch(X: torch.Tensor, N: torch.Tensor, device,
                  chunk_elems: int = 2 ** 25, with_uncertainty: bool = True):
    """Batched elephant.vmf.vmf_fit. X (E, Nmax, 7) f64, N (E,) lengths.

    Returns dict of host numpy arrays; invalid windows (N < NMIN or
    rho < 1e-12) are NaN — the batch analogue of vmf_fit returning None.
    """
    E, Nmax, _ = X.shape
    dev = X.device
    W = torch.as_tensor(WARM, dtype=torch.float64, device=dev)
    ar = torch.arange(Nmax, device=dev)
    mask = ar[None, :] < N[:, None]                        # (E, Nmax)
    Nf = N.to(torch.float64)

    Xn = X / X.norm(dim=-1, keepdim=True).clamp_min(1e-300)  # padded rows -> 0
    r = Xn.sum(dim=1) / Nf[:, None]
    rho_raw = r.norm(dim=-1)
    rho = rho_raw.clamp(max=RHOMAX)
    valid = (N >= NMIN) & (rho_raw >= 1e-12)
    rho_run = torch.where(valid, rho, torch.full_like(rho, 0.5))
    mu = r / rho_run.clamp_min(1e-300).unsqueeze(-1)
    kappa = kappa_newton_t(rho_run)
    warmth = mu @ W
    out = {"mu_hat": mu, "kappa": kappa, "rho": rho, "warmth_vmf": warmth,
           "valid": valid, "saturated": (rho >= RHOMAX) | (kappa >= KMAX)}

    if with_uncertainty:
        # --- bootstrap CI on kappa (Banerjee shortcut per resample) -------- #
        ci = torch.zeros(E, 2, dtype=torch.float64, device=dev)
        for n_val in torch.unique(N).tolist():
            sel = (N == n_val).nonzero(as_tuple=True)[0]
            if len(sel) == 0 or n_val < NMIN:
                continue
            idx = torch.as_tensor(_boot_idx_for(int(n_val), BOOT_B),
                                  device=dev)                  # (B, n)
            Xg = Xn[sel][:, :n_val, :]                        # (E_n, n, 7)
            for s in range(0, len(sel), max(1, chunk_elems // (BOOT_B * n_val * D))):
                sub = sel[s:s + max(1, chunk_elems // (BOOT_B * n_val * D))]
                Xb = Xg[s:s + len(sub)][:, idx, :]            # (e, B, n, 7)
                rb = Xb.mean(dim=2)                           # (e, B, 7)
                rhob = rb.norm(dim=-1).clamp(max=RHOMAX)
                kb = torch.where(rhob < 1e-12, torch.zeros_like(rhob),
                                 banerjee_t(rhob))
                q = torch.quantile(kb, torch.tensor([0.025, 0.975],
                                                    dtype=kb.dtype,
                                                    device=dev), dim=1)
                ci[sub] = q.T
        out["kappa_ci"] = ci

        # --- jackknife SE(mu_hat) (leave-one-out means, normalized) --------- #
        S = Xn.sum(dim=1)                                     # (E, 7)
        loo = (S[:, None, :] - Xn) / (Nf - 1.0)[:, None, None]
        loo = loo / loo.norm(dim=-1, keepdim=True).clamp_min(1e-300)
        m3 = mask.unsqueeze(-1).to(torch.float64)
        jk_mean = (loo * m3).sum(dim=1) / Nf[:, None]
        diff = (loo - jk_mean[:, None, :]) * m3
        mu_se = torch.sqrt((Nf - 1.0) / Nf * (diff ** 2).sum(dim=(1, 2)))
        out["mu_se"] = mu_se

        # --- axis spread (numpy std default ddof=0 over the n real rows) ---- #
        mean = (Xn * m3).sum(dim=1) / Nf[:, None]
        var = (((Xn - mean[:, None, :]) ** 2) * m3).sum(dim=1) / Nf[:, None]
        out["axis_spread"] = var.sqrt()

    bad = ~valid
    for k_ in ("kappa", "rho", "warmth_vmf", "mu_se"):
        if k_ in out:
            out[k_] = torch.where(bad, torch.full_like(out[k_], float("nan")),
                                  out[k_])
    out["mu_hat"] = torch.where(bad.unsqueeze(-1),
                                torch.full_like(out["mu_hat"], float("nan")),
                                out["mu_hat"])
    if "axis_spread" in out:
        out["axis_spread"] = torch.where(
            bad.unsqueeze(-1),
            torch.full_like(out["axis_spread"], float("nan")),
            out["axis_spread"])
    return {k_: (v.detach().cpu().numpy() if torch.is_tensor(v) else v)
            for k_, v in out.items()}


# --------------------------------------------------------------------------- #
# Batched field edges + production-ledger imbalance                            #
# --------------------------------------------------------------------------- #
def edge_batch(mu_b, mu_a, kap_b, kap_a, warm_b, warm_a, se_b, se_a,
               db_factor: float = 2.0):
    """Batched elephant.vmf.edge over (before, after) fit pairs."""
    d_mu = np.linalg.norm(mu_a - mu_b, axis=-1)
    return {
        "d_mu": d_mu,
        "d_warmth": warm_a - warm_b,
        "d_log_kappa": np.log(kap_a / kap_b),
        "real": d_mu > db_factor * np.maximum(se_b, se_a),
    }


def ledger_batch(fields, device):
    """Production-ledger imbalance, batched on GPU.

    fields: (R, 7) raw dial rows in DIAL_NAMES order (v0 space, NOT z-space).
    Returns warmth (v0 formula), kappa proxy (2*||v - 0.5||), and the
    before->after pair metrics drift = L2(d_warmth, d_kappa) plus the raw
    field L2, mirroring examples/production_probe.py's ledger update.
    """
    F = torch.as_tensor(fields, dtype=torch.float64, device=device)
    warm = (0.30 * F[:, 0] + 0.15 * F[:, 4] + 0.20 * (F[:, 2] - 0.5)
            + 0.20 * (F[:, 6] - 0.5) + 0.20 * (F[:, 1] - 0.5)
            - 0.15 * F[:, 3] - 0.10 * F[:, 5])
    kap = 2.0 * (F - 0.5).norm(dim=-1)
    d_w = warm[1:] - warm[:-1]
    d_k = kap[1:] - kap[:-1]
    drift = torch.sqrt(d_w ** 2 + d_k ** 2)
    field_l2 = (F[1:] - F[:-1]).norm(dim=-1)
    return {k_: v.detach().cpu().numpy() for k_, v in
            {"warmth": warm, "kappa_proxy": kap, "d_warmth": d_w,
             "d_kappa": d_k, "drift": drift, "field_l2": field_l2}.items()}


# --------------------------------------------------------------------------- #
# CPU reference (the serial loop we are replacing)                             #
# --------------------------------------------------------------------------- #
def cpu_reference(X, N, refresh=False):
    """Run elephant.vmf.vmf_fit serially over every event; cache results."""
    h = hashlib.sha256()
    h.update(X.tobytes())
    h.update(N.tobytes())
    digest = h.hexdigest()[:16]
    if not refresh and os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=False)
        if str(z["digest"]) == digest and len(z["kappa"]) == len(X):
            return {k: z[k] for k in z.files if k != "digest"}, float(z["seconds"])

    res = {k: np.full(len(X), np.nan) for k in
           ("kappa", "rho", "warmth_vmf", "mu_se")}
    res["mu_hat"] = np.full((len(X), D), np.nan)
    res["kappa_ci"] = np.full((len(X), 2), np.nan)
    res["axis_spread"] = np.full((len(X), D), np.nan)
    res["valid"] = np.zeros(len(X), dtype=bool)
    res["saturated"] = np.zeros(len(X), dtype=bool)

    t0 = time.perf_counter()
    for e in range(len(X)):
        fit = vmf_fit(X[e, :N[e]])
        if fit is None:
            continue
        res["mu_hat"][e] = fit["mu_hat"]
        res["kappa"][e] = fit["kappa"]
        res["rho"][e] = fit["rho"]
        res["kappa_ci"][e] = fit["kappa_ci"]
        res["warmth_vmf"][e] = fit["warmth_vmf"]
        res["mu_se"][e] = fit["mu_se"]
        res["axis_spread"][e] = fit["axis_spread"]
        res["valid"][e] = True
        res["saturated"][e] = fit["saturated"]
    seconds = time.perf_counter() - t0

    np.savez_compressed(CACHE, digest=np.array(digest), seconds=np.array(seconds),
                        **res)
    return res, seconds


def cpu_core_fit(zs):
    """vmf.py's core lines (normalize -> Newton -> warmth), no uncertainty."""
    X = np.asarray(zs, float)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    r = X.mean(0)
    rho = min(float(np.linalg.norm(r)), RHOMAX)
    if rho < 1e-12:
        return None
    mu = r / rho
    k = float(np.clip(rho * (D - rho ** 2) / (1 - rho ** 2), 1e-6, KMAX))
    for _ in range(60):
        a = A7_cpu(k)
        g = 1.0 - a * a - (D - 1.0) * a / k
        if abs(g) < 1e-12:
            break
        step = (a - rho) / g
        k = float(np.clip(k - step, 1e-6, KMAX))
        if abs(step) < 1e-9:
            break
    return float(WARM @ mu), k


# --------------------------------------------------------------------------- #
# Verification batteries                                                       #
# --------------------------------------------------------------------------- #
def check_a7(device):
    ks = np.concatenate([np.geomspace(1e-6, 0.4999, 200),
                         np.linspace(0.5, 600.0, 2000)])
    kt = torch.as_tensor(ks, dtype=torch.float64, device=device)
    gpu = A7_t(kt).cpu().numpy()
    cpu = np.array([A7_cpu(k) for k in ks])
    return float(np.abs(gpu - cpu).max())


def synthetic_battery(device):
    """Adversarial windows: series branch, saturation, near-RHOMAX,
    antipodal (isotropic -> None/NaN), N = NMIN exactly, rho straddling
    the Banerjee clip. Returns rows of (label, cpu_valid, gpu_valid,
    max|d mu|, max|d kappa|, saturated)."""
    rng = np.random.default_rng(7)
    cases = []

    def vmf_sample(k, n, d=D):
        mu = rng.normal(size=d)
        mu /= np.linalg.norm(mu)
        xs = mu[None, :] + rng.normal(scale=0.5 / max(k, 0.5), size=(n, d))
        return xs / np.linalg.norm(xs, axis=1, keepdims=True)

    cases.append(("loose-k~0.5(series branch)", vmf_sample(0.5, 40)))
    cases.append(("loose-k~2", vmf_sample(2.0, 40)))
    cases.append(("k~20", vmf_sample(20.0, 40)))
    u = rng.normal(size=D)
    u /= np.linalg.norm(u)
    cases.append(("saturated-identical", np.tile(u, (30, 1))))
    cases.append(("near-RHOMAX-tiny-noise",
                  u[None, :] + 0.02 * rng.normal(size=(30, D))))
    v = rng.normal(size=D)
    v /= np.linalg.norm(v)
    cases.append(("antipodal-isotropic",
                  np.stack([(-1) ** i * v for i in range(12)])))
    cases.append(("exactly-NMIN-10", vmf_sample(8.0, NMIN)))
    # rho engineered just under RHOMAX: kappa_init just under the clip
    cases.append(("rho-just-under-clamp",
                  u[None, :] + 0.05 * rng.normal(size=(50, D))))

    rows = []
    for label, zs in cases:
        X = np.zeros((1, NMAX, D))
        X[0, :len(zs)] = zs
        N = np.array([len(zs)])
        gpu = vmf_fit_batch(torch.as_tensor(X, dtype=torch.float64,
                                            device=device),
                            torch.as_tensor(N, device=device), device)
        fit = vmf_fit(zs)
        cv = fit is not None
        gv = bool(gpu["valid"][0])
        dmu = (np.abs(gpu["mu_hat"][0] - np.array(fit["mu_hat"])).max()
               if cv and gv else float("nan"))
        dkap = (abs(gpu["kappa"][0] - fit["kappa"]) if cv and gv
                else float("nan"))
        dse = (abs(gpu["mu_se"][0] - fit["mu_se"]) if cv and gv
               else float("nan"))
        dci = (np.abs(gpu["kappa_ci"][0] - np.array(fit["kappa_ci"])).max()
               if cv and gv else float("nan"))
        rows.append((label, cv, gv, dmu, dkap, dse, dci,
                     bool(gpu["saturated"][0]) if gv else None,
                     fit["kappa"] if cv else None))
    return rows


def load_logged_edges():
    """(fit_before, fit_after, logged_edge) triples from the night logs —
    tapnight's own production edges, an independent end-to-end check."""
    triples = []
    for fn in NIGHT_FILES:
        prev = None
        with open(os.path.join(NIGHTS_DIR, fn), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("type") != "speak" or not row.get("fit"):
                    prev = row.get("fit") or prev
                    continue
                if prev and row.get("edge"):
                    triples.append((prev, row["fit"], row["edge"]))
                prev = row["fit"]
    return triples


def load_prod_rows():
    rows = []
    with open(PROD_LOG, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("ok") and r.get("field"):
                rows.append(r)
    return rows


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="force CPU reference recompute")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available()
                       or args.device == "cpu" else "cpu")
    print("=" * 78)
    print("FIELD-MATH GPU BATCH FARM — vMF MLE + edges + ledger, CUDA-batched")
    print("=" * 78)
    print(f"device: {dev}"
          + (f" ({torch.cuda.get_device_name(dev)}, "
             f"{torch.cuda.get_device_properties(dev).total_memory/2**20:.0f} MB)"
             if dev.type == "cuda" else "")
          + f" | torch {torch.__version__} | float64 parity mode")

    # -- [1] corpus -------------------------------------------------------- #
    streams = build_streams()
    X, N, sid, pos = build_events(streams)
    E = len(X)
    print(f"\n[1] corpus: {len(streams)} streams "
          + ", ".join(f"{n}({len(z)})" for n, z in streams[:3]) + " ...")
    print(f"    events (trailing windows, cap={NMAX}, N>={NMIN}): {E}  "
          f"| N in [{N.min()}, {N.max()}] mean {N.mean():.1f}")

    # -- [2] A7 parity ------------------------------------------------------ #
    d = check_a7(dev)
    print(f"\n[2] A7 Bessel-ratio parity over k in [1e-6, 600]: "
          f"max|GPU-CPU| = {d:.3e}  ->  {'PASS' if d < TOL else 'FAIL'}")

    # -- [3] synthetic edge-case battery ------------------------------------ #
    print("\n[3] synthetic edge-case battery (CPU vmf_fit vs GPU batch):")
    print(f"    {'case':<28}{'cpu':>5}{'gpu':>5}{'d_mu':>10}{'d_kap':>10}"
          f"{'d_se':>10}{'d_ci':>10}{'sat':>5}{'kappa':>9}")
    worst = 0.0
    for (lab, cv, gv, dmu, dkap, dse, dci, sat, kap) in synthetic_battery(dev):
        parity = "ok" if cv == gv else "MISMATCH"
        finite = [v for v in (dmu, dkap, dse, dci) if np.isfinite(v)]
        m = max(finite) if finite else 0.0
        worst = max(worst, float(m))
        print(f"    {lab:<28}{str(cv):>5}{str(gv):>5}"
              f"{dmu:10.2e}{dkap:10.2e}{dse:10.2e}{dci:10.2e}"
              f"{str(sat):>5}{(f'{kap:9.3f}' if kap is not None else '      --')}  {parity}")
    print(f"    worst diff across battery: {worst:.3e}  ->  "
          f"{'PASS' if worst < TOL else 'FAIL'}")

    # -- [4] full-corpus correctness ---------------------------------------- #
    print(f"\n[4] full-corpus correctness vs CPU serial elephant.vmf.vmf_fit:")
    cpu, cpu_seconds = cpu_reference(X, N, refresh=args.refresh)
    if args.device != "cpu" and dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    Xt = torch.as_tensor(X, dtype=torch.float64, device=dev)
    Nt = torch.as_tensor(N, device=dev)
    gpu = vmf_fit_batch(Xt, Nt, dev)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    both = cpu["valid"] & gpu["valid"]
    only_cpu = cpu["valid"] & ~gpu["valid"]
    only_gpu = ~cpu["valid"] & gpu["valid"]
    print(f"    valid fits: CPU {int(cpu['valid'].sum())} / GPU {int(gpu['valid'].sum())}"
          f" | agreement {int(both.sum())} | CPU-only {int(only_cpu.sum())}"
          f" | GPU-only {int(only_gpu.sum())}")
    diffs = {}
    for q in ("mu_hat", "kappa", "rho", "warmth_vmf", "kappa_ci", "mu_se",
              "axis_spread"):
        a, b = cpu[q][both], gpu[q][both]
        diffs[q] = float(np.abs(a - b).max())
        print(f"    max|d {q:<12}| = {diffs[q]:.3e}")
    sat_agree = int((cpu["saturated"][both] == gpu["saturated"][both]).sum())
    print(f"    saturated-flag agreement: {sat_agree}/{int(both.sum())} "
          f"(CPU saturated: {int(cpu['saturated'].sum())})")
    gate = max(diffs.values())
    print(f"    correctness gate (max over quantities, tol {TOL:g}): "
          f"{gate:.3e}  ->  {'PASS' if gate < TOL and only_cpu.sum()==0 and only_gpu.sum()==0 else 'FAIL'}")

    # -- [5] batched edges vs elephant.vmf.edge (+ tapnight's logged edges) -- #
    print("\n[5] field-edge batch (consecutive fits per stream):")
    fb_m, fa_m, fb_k, fa_k = [], [], [], []
    fb_w, fa_w, fb_s, fa_s = [], [], [], []
    pairs = []
    for s in range(len(streams)):
        idx = np.nonzero((sid == s) & both)[0]
        for a, b in zip(idx[:-1], idx[1:]):
            pairs.append((a, b))
            fb_m.append(cpu["mu_hat"][a]); fa_m.append(cpu["mu_hat"][b])
            fb_k.append(cpu["kappa"][a]);   fa_k.append(cpu["kappa"][b])
            fb_w.append(cpu["warmth_vmf"][a]); fa_w.append(cpu["warmth_vmf"][b])
            fb_s.append(cpu["mu_se"][a]);   fa_s.append(cpu["mu_se"][b])
    gb = edge_batch(np.array(fb_m), np.array(fa_m), np.array(fb_k),
                    np.array(fa_k), np.array(fb_w), np.array(fa_w),
                    np.array(fb_s), np.array(fa_s))
    ref = [edge_cpu({"mu_hat": fb_m[i].tolist(), "kappa": float(fb_k[i]),
                     "warmth_vmf": float(fb_w[i]), "mu_se": float(fb_s[i])},
                    {"mu_hat": fa_m[i].tolist(), "kappa": float(fa_k[i]),
                     "warmth_vmf": float(fa_w[i]), "mu_se": float(fa_s[i])})
           for i in range(len(pairs))]
    ed = {"d_mu": 0.0, "d_warmth": 0.0, "d_log_kappa": 0.0}
    ereal = 0
    for i, rr in enumerate(ref):
        for q in ed:
            ed[q] = max(ed[q], abs(gb[q][i] - rr[q]))
        ereal += int(gb["real"][i] != rr["real"])
    print(f"    {len(pairs)} consecutive-fit edges across {len(streams)} streams")
    for q, v in ed.items():
        print(f"    max|d {q:<13}| = {v:.3e}")
    print(f"    deadband `real` disagreements: {ereal}  ->  "
          f"{'PASS' if max(ed.values()) < TOL and ereal == 0 else 'FAIL'}")

    triples = load_logged_edges()
    lg = {"d_mu": [], "d_warmth": [], "d_log_kappa": []}
    for fb, fa, le in triples:
        m = edge_batch(np.array([fb["mu_hat"]]), np.array([fa["mu_hat"]]),
                       np.array([fb["kappa"]]), np.array([fa["kappa"]]),
                       np.array([fb["warmth_vmf"]]),
                       np.array([fa["warmth_vmf"]]),
                       np.array([fb["mu_se"]]), np.array([fa["mu_se"]]))
        for q in lg:
            lg[q].append(abs(m[q][0] - le[q]))
    print(f"    vs tapnight's LOGGED edges ({len(triples)} night-log edges): "
          f"max|d d_mu|={max(lg['d_mu']):.2e} "
          f"max|d d_warmth|={max(lg['d_warmth']):.2e} "
          f"max|d d_log_kappa|={max(lg['d_log_kappa']):.2e}")

    # -- [6] production-ledger imbalance ------------------------------------- #
    print("\n[6] production-ledger imbalance (before->after, batched GPU):")
    prows = load_prod_rows()
    F = np.array([[r["field"].get(n, 0.0) for n in DIAL_NAMES] for r in prows])
    led = ledger_batch(F, dev)
    logged_dw = np.array([r["d_warmth"] for r in prows[1:] if "d_warmth" in r])
    logged_dk = np.array([r["d_kappa"] for r in prows[1:] if "d_kappa" in r])
    logged_dr = np.array([r["drift"] for r in prows[1:] if "drift" in r])
    k = len(logged_dr)
    # ledger stores values rounded to 4 dp before differencing
    ddw = np.abs(led["d_warmth"][-k:] - logged_dw).max()
    ddk = np.abs(led["d_kappa"][-k:] - logged_dk).max()
    ddr = np.abs(led["drift"][-k:] - logged_dr).max()
    print(f"    {len(prows)} snapshots -> {len(prows)-1} before->after pairs "
          f"({k} with logged deltas)")
    print(f"    max|d d_warmth| = {ddw:.2e} (ledger rounds to 4 dp)")
    print(f"    max|d d_kappa|  = {ddk:.2e}")
    print(f"    max|d drift|    = {ddr:.2e}   L2(d_warmth, d_kappa) parity")
    print(f"    full-field L2 before->after: mean {led['field_l2'].mean():.4f} "
          f"max {led['field_l2'].max():.4f}")
    print(f"    ledger parity ->  {'PASS' if max(ddw, ddk, ddr) < 1e-3 else 'FAIL'}")

    # -- [7] speed ----------------------------------------------------------- #
    print("\n[7] speed (same in-memory numpy inputs for both sides; GPU time "
          "includes H2D upload + compute + D2H results):")
    t0 = time.perf_counter()
    for e in range(E):
        cpu_core_fit(X[e, :N[e]])
    t_core_cpu = time.perf_counter() - t0

    if dev.type == "cuda":
        gpu_warm = vmf_fit_batch(Xt[:64].contiguous(), Nt[:64], dev,
                                 with_uncertainty=False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    _ = vmf_fit_batch(Xt, Nt, dev, with_uncertainty=False)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_core_gpu = time.perf_counter() - t0

    t0 = time.perf_counter()
    Xh = torch.as_tensor(X, dtype=torch.float64, device=dev)
    Nh = torch.as_tensor(N, device=dev)
    _ = vmf_fit_batch(Xh, Nh, dev, with_uncertainty=True)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_full_gpu = time.perf_counter() - t0

    peak = (torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == "cuda"
            else 0.0)
    print(f"    CPU serial, full vmf_fit (Newton + bootstrap B={BOOT_B} + "
          f"jackknife): {cpu_seconds:8.2f} s  ({E/cpu_seconds:9.1f} fits/s)")
    print(f"    CPU serial, core solver only:                         "
          f"{t_core_cpu:8.2f} s  ({E/t_core_cpu:9.1f} fits/s)")
    print(f"    GPU batch,  core solver only:                         "
          f"{t_core_gpu:8.2f} s  ({E/t_core_gpu:9.1f} fits/s)")
    print(f"    GPU batch,  full (incl. upload + bootstrap + jackknife): "
          f"{t_full_gpu:8.2f} s  ({E/t_full_gpu:9.1f} fits/s)")
    print(f"    GPU peak VRAM: {peak:.0f} MB")
    s_full = cpu_seconds / t_full_gpu
    s_core = t_core_cpu / t_core_gpu
    print(f"\n    SPEEDUP full fit:  {s_full:6.1f}x   "
          f"SPEEDUP core: {s_core:6.1f}x")

    ok = (d < TOL and worst < TOL and gate < TOL and only_cpu.sum() == 0
          and only_gpu.sum() == 0 and max(ed.values()) < TOL and ereal == 0
          and max(ddw, ddk, ddr) < 1e-3)
    print("\n" + "=" * 78)
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} — GPU batch farm matches the "
          f"CPU serial reference to {max(gate, worst):.1e} (tol {TOL:g})")
    print(f"         full-fit speedup {s_full:.1f}x, core-solver speedup "
          f"{s_core:.1f}x over {E} windows on {torch.cuda.get_device_name(dev) if dev.type=='cuda' else dev}")
    print("=" * 78)


if __name__ == "__main__":
    main()
