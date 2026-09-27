"""gpu.train — seeded training + held-out evaluation, one command.

Usage (from repo root):

    python -m gpu.train                    # CUDA if available, else CPU
    python -m gpu.train --cpu --seed 2718  # force CPU (slower)
    python -m gpu.train --epochs 80

Writes a results JSON (default gpu/results/room-state-embed-v0.json)
containing: config, environment (torch version, device, CUDA used,
nvidia-smi memory samples taken DURING training), wall-clock, the
untrained-baseline metric battery, the trained metric battery, and
per-epoch loss trace. Fully reproducible from --seed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .evaluate import (
    acclimation_metrics,
    charisma_metrics,
    evaluate_battery,
)
from .model import RoomEncoder, contrastive_loss, make_batches
from .roomgen import build_world

NVIDIA_SMI_CANDIDATES = [
    shutil.which("nvidia-smi") or "",
    "/usr/lib/wsl/lib/nvidia-smi",
]
NVIDIA_SMI_QUERY = [
    "--query-gpu=name,memory.used,utilization.gpu",
    "--format=csv,noheader",
]


def sample_gpu() -> Dict[str, str]:
    """One nvidia-smi sample. subprocess list-form ONLY (fleet rule)."""
    for path in NVIDIA_SMI_CANDIDATES:
        if path and os.path.exists(path):
            try:
                r = subprocess.run([path] + NVIDIA_SMI_QUERY,
                                   capture_output=True, text=True, timeout=10)
                if r.returncode == 0:
                    return {"sample": r.stdout.strip()}
            except (OSError, subprocess.SubprocessError):
                continue
    return {"sample": "n/a"}


def train(
    world,
    encoder: RoomEncoder,
    device: torch.device,
    epochs: int,
    lr: float,
    rooms_per_batch: int,
    seed: int,
) -> Dict:
    """Seeded training loop. Returns diagnostics."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    clips_per_room = world.train_rooms[0].obs.shape[0]
    obs = torch.cat([r.obs for r in world.train_rooms])
    room_index = torch.cat([
        torch.full((r.obs.shape[0],), i) for i, r in enumerate(world.train_rooms)
    ])

    g = torch.Generator().manual_seed(seed + 7)
    encoder.train()
    losses: List[float] = []
    smi_samples: List[Dict[str, str]] = []
    t0 = time.perf_counter()

    for epoch in range(epochs):
        batches = make_batches(room_index, clips_per_room,
                               rooms_per_batch=rooms_per_batch,
                               shuffle=True, generator=g)
        epoch_loss = 0.0
        for idx in batches:
            x = obs[idx].to(device)
            ri = room_index[idx].to(device)
            z = encoder(x)
            loss, _ = contrastive_loss(z, ri)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
        sched.step()
        losses.append(epoch_loss / len(batches))
        if epoch % max(1, epochs // 12) == 0 or epoch == epochs - 1:
            smi_samples.append(sample_gpu())

    wall = time.perf_counter() - t0
    return {"loss_first": losses[0], "loss_last": losses[-1],
            "losses": losses, "wall_seconds": wall,
            "smi_samples": smi_samples,
            "steps_per_epoch": len(batches)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=2718)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--rooms-per-batch", type=int, default=3)
    ap.add_argument("--n-train-rooms", type=int, default=36)
    ap.add_argument("--n-eval-rooms", type=int, default=12)
    ap.add_argument("--n-clips", type=int, default=48)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", type=str,
                    default="gpu/results/room-state-embed-v0.json")
    args = ap.parse_args()

    # ---- seeding (all three RNG streams) --------------------------------
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    import random as _random
    _random.seed(seed)

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    cuda_used = device.type == "cuda"

    world = build_world(
        seed=seed,
        n_train_rooms=args.n_train_rooms,
        n_eval_rooms=args.n_eval_rooms,
        n_clips=args.n_clips,
    )

    # ---- baseline battery on the UNTRAINED encoder ----------------------
    torch.manual_seed(seed + 100)  # fixed init for the baseline comparison
    encoder = RoomEncoder().to(device)
    baseline = evaluate_battery(encoder, world, device)
    base_accl = acclimation_metrics(encoder, world, device)
    base_char = charisma_metrics(encoder, world, device)

    # ---- train -----------------------------------------------------------
    diag = train(world, encoder, device, args.epochs, args.lr,
                 args.rooms_per_batch, seed)

    # ---- trained battery -------------------------------------------------
    trained = evaluate_battery(encoder, world, device)
    tr_accl = acclimation_metrics(encoder, world, device)
    tr_char = charisma_metrics(encoder, world, device)

    # ---- environment evidence -------------------------------------------
    gpu_name = ""
    if cuda_used:
        gpu_name = torch.cuda.get_device_name(0)

    results = {
        "title": "gpu: room-state embedding v0 (synthetic prototype)",
        "honesty": "ALL DATA IS SYNTHETIC. This proves the training "
                   "machinery and metric battery on known ground truth; "
                   "it proves NOTHING about real fleet rooms yet.",
        "seed": seed,
        "config": {
            "epochs": args.epochs, "lr": args.lr,
            "rooms_per_batch": args.rooms_per_batch,
            "n_train_rooms": args.n_train_rooms,
            "n_eval_rooms": args.n_eval_rooms,
            "n_clips": args.n_clips,
            "emb_dim": encoder.emb_dim,
            "obs_dim": encoder.net[0].in_features,
        },
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
            "cuda_used": cuda_used,
            "gpu_name": gpu_name,
            "cuda_version_runtime": torch.version.cuda,
            "train_wall_seconds": diag["wall_seconds"],
            "steps_per_epoch": diag["steps_per_epoch"],
            "smi_during_training": diag["smi_samples"],
        },
        "loss_trace": {"first": diag["loss_first"], "last": diag["loss_last"]},
        "baseline_untrained": {**baseline,
                               "acclimation": base_accl,
                               "charisma": base_char},
        "trained": {**trained,
                    "acclimation": tr_accl,
                    "charisma": tr_char},
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print(f"[gpu.train] device={device} cuda={cuda_used} "
          f"({gpu_name or 'cpu'})")
    print(f"[gpu.train] loss {diag['loss_first']:.4f} -> "
          f"{diag['loss_last']:.4f} over {args.epochs} epochs, "
          f"wall {diag['wall_seconds']:.1f}s")
    print(f"[gpu.train] baseline  knn={baseline['room_discrimination_knn']:.3f} "
          f"gap={baseline['sauna_plunge_gap']:.3f} "
          f"sil={baseline['silhouette']:.3f} "
          f"probe={baseline['coldwarm_probe_embedding']:.3f} "
          f"(raw {baseline['coldwarm_probe_raw_baseline']:.3f}) "
          f"kappa_r={baseline['kappa_correlation']:.3f}")
    print(f"[gpu.train] trained   knn={trained['room_discrimination_knn']:.3f} "
          f"gap={trained['sauna_plunge_gap']:.3f} "
          f"sil={trained['silhouette']:.3f} "
          f"probe={trained['coldwarm_probe_embedding']:.3f} "
          f"(raw {trained['coldwarm_probe_raw_baseline']:.3f}) "
          f"kappa_r={trained['kappa_correlation']:.3f}")
    print(f"[gpu.train] acclimation log_tau_r: "
          f"{base_accl['log_tau_correlation']:.3f} -> "
          f"{tr_accl['log_tau_correlation']:.3f}")
    print(f"[gpu.train] charisma cohens_d: "
          f"{base_char['cohens_d']:.3f} -> {tr_char['cohens_d']:.3f}")
    print(f"[gpu.train] results -> {out}")


if __name__ == "__main__":
    main()
