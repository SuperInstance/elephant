"""scripts/contrast_audio.py — the REGISTERED gate-3 run (audio tier).

This is the tier the deadman was registered on: devils-advocate §3 —
"after the §2.3 contrastive head / fine-tune (fixed τ = 0.15, explicit
within-room spread regularizer)" on the trades-nights control where the
frozen v2 encoder measured fine gap 0.0146 / disc 0.339 / heldout 0.356
(checkpoints/elephant_probe.json, fleet-jepa-midi).

Encoder = the frozen audio-JEPA v2 ConvEncoder (fleet-jepa-midi
checkpoints/audio_jepa_v2.pt), fine-tuned — not rewritten. Embeddings and
metrics replicate elephant_sense_probe.py exactly; the harness validates
itself by reproducing the frozen probe numbers before any training.

Rooms (v3 §1.1): tap-trades episodes 1-4 (the killer control), open-mic,
named venues, fleet-radio series, and music/ split into 3 cold-plunge
rooms (the coarse pole).

Registered numbers come from seeds 0/1/2 (three consecutive runs).
"""
from __future__ import annotations

import json
import os
import random
import sys

# CPU-only hardening (2026-08-19): the box has a GPU-pinning bug that crashed
# three prior dispatches; this run is registered CPU-only.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ELEPHANT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ELEPHANT)
sys.path.insert(0, "/home/eileen/projects/fleet-jepa-midi")

import numpy as np
import torch
import torch.nn.functional as F

from elephant import contrast
from elephant.contrast import (
    Clip, contrast_loss, cross_group_gap, probe_report, sample_room_batches,
    spread_hinge,
)

import elephant_sense_probe as probe  # the frozen probe, verbatim

AI = "/home/eileen/projects/ai-writings"
CKPT = "/home/eileen/projects/fleet-jepa-midi/checkpoints/audio_jepa_v2.pt"
OUT = os.path.join(ELEPHANT, "checkpoints", "contrast")

SEEDS = (0, 1, 2)
EPOCHS = 120
N_BATCHES = 40
LR = 5e-5
WINDOW, HOP = 240, 120
MAX_TRAIN_WINDOWS = 16   # training-time cap per clip (eval uses ALL windows)


def build_rooms():
    """room -> list of audio paths. tap = the probe's own discovery."""
    rooms = probe.discover_rooms([os.path.join(AI, "tap-trades", "radio-theater")])
    rooms = {f"tap-{r.split('-')[-1]}": v for r, v in rooms.items()}

    def add_dir(name, path, min_size=50_000):
        from audio_jepa.dataset import discover_clips
        if os.path.isdir(path):
            clips = discover_clips(path)
            if len(clips) >= 2:
                rooms[name] = clips

    add_dir("open-mic", os.path.join(AI, "radio-theater",
                                   "tap-open-mic-night", "voices"))
    add_dir("front-door", os.path.join(AI, "radio-theater", "the-front-door"))
    add_dir("tavern-night", os.path.join(AI, "radio-theater", "tavern-night"))
    add_dir("dogs-fell-in-love", os.path.join(AI, "radio-theater", "dogs-fell-in-love"))
    add_dir("channel-42-dawn", os.path.join(AI, "radio-theater", "channel-42-dawn"))
    add_dir("hermes-jazz", os.path.join(AI, "radio-theater", "hermes-jazz-suite"))

    # fleet-radio: one room per series (004 / 005)
    ra = os.path.join(AI, "radio", "audio")
    if os.path.isdir(ra):
        for series in ("004", "005"):
            clips = sorted(p for p in os.listdir(ra)
                           if p.startswith(f"fleet-radio-{series}")
                           and p.lower().endswith((".mp3", ".wav"))
                           and os.path.getsize(os.path.join(ra, p)) >= 50_000)
            if len(clips) >= 2:
                rooms[f"fleet-radio-{series}"] = [os.path.join(ra, c) for c in clips]

    # the cold plunge: music/ top-level mp3s split into 3 rooms
    mus = os.path.join(AI, "music")
    mp3s = sorted(p for p in os.listdir(mus)
                  if p.lower().endswith((".mp3", ".wav"))
                  and os.path.getsize(os.path.join(mus, p)) >= 50_000)
    k = (len(mp3s) + 2) // 3
    for i, name in enumerate(("music-a", "music-b", "music-c")):
        chunk = mp3s[i * k: (i + 1) * k]
        if len(chunk) >= 2:
            rooms[name] = [os.path.join(mus, c) for c in chunk]
    return rooms


def embed_cached(encoder, mels, device):
    """Probe-exact embedding from cached mels (pad, window, mean-pool, L2)."""
    zs = []
    with torch.no_grad():
        for mel in mels:
            m = mel.to(device)
            T = m.shape[-1]
            if T < WINDOW:
                m = F.pad(m, (0, WINDOW - T))
                T = WINDOW
            embs = [encoder(m[:, s:s + WINDOW].unsqueeze(0).unsqueeze(0))[0]
                    for s in range(0, T - WINDOW + 1, HOP)] or [None]
            if embs[0] is None:
                embs = [encoder(m[:, :WINDOW].unsqueeze(0).unsqueeze(0))[0]]
            zs.append(F.normalize(torch.stack(embs).mean(0), dim=-1))
    return torch.stack(zs)


def main() -> int:
    import librosa

    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cpu")
    print(f"[audio] device={device}")

    rooms = build_rooms()
    print(f"[audio] rooms: " + ", ".join(f"{r}={len(v)}" for r, v in sorted(rooms.items())))

    encoder, cfg = probe.load_encoder(CKPT, device)
    frontend = probe.MelFrontend(
        sample_rate=cfg.get("sample_rate", 16_000),
        n_fft=cfg.get("n_fft", 400),
        hop_length=cfg.get("hop_length", 160),
        n_mels=cfg.get("n_mels", 64),
    )

    clips: list = []
    mels: list = []
    from elephant.contrast import speaker_key_from_filename
    for r in sorted(rooms):
        for p in rooms[r]:
            y, sr = librosa.load(str(p), sr=16_000, mono=True)
            mel = probe.compute_mel(torch.from_numpy(y.astype(np.float32)),
                                    frontend, 16_000)
            base = os.path.basename(str(p))
            clips.append(Clip(room=r,
                              key=os.path.splitext(base)[0],
                              speaker=speaker_key_from_filename(base)))
            mels.append(mel)
    print(f"[audio] clips={len(clips)}")

    # ---- harness fidelity check vs the frozen probe ------------------- #
    encoder.eval()
    z0 = embed_cached(encoder, mels, device).cpu().numpy()
    tap_idx = [i for i, c in enumerate(clips) if c.room.startswith("tap-")]
    z0_tap = z0[tap_idx]
    clips_tap = [clips[i] for i in tap_idx]
    rep0_tap = probe_report(z0_tap, clips_tap)
    print(f"[audio] FROZEN tap baseline: gap={rep0_tap['separability']['gap']:.4f} "
          f"(probe.json: 0.0146) disc={rep0_tap['room_discrimination']:.3f} "
          f"(0.339) heldout={rep0_tap['room_discrimination_speaker_heldout']:.3f} "
          f"(0.356)")

    room_names = [c.room for c in clips]
    music_rooms = [r for r in set(room_names) if r.startswith("music-")]
    rep0_full = probe_report(z0, clips, coarse_b_rooms=music_rooms)
    base_spread = contrast.room_spread(z0, room_names)
    # frozen baseline: committed at 77b8aa4 — NEVER regenerated/overwritten
    fb = os.path.join(OUT, "audio_frozen_baseline.json")
    if os.path.exists(fb):
        committed = json.load(open(fb))
        print(f"[audio] frozen baseline COMMITTED (kept): coarse "
              f"gap={committed['coarse']['gap']:.4f}; recomputed check "
              f"coarse={rep0_full['coarse']['gap']:.4f}")
    else:
        with open(fb, "w") as f:
            json.dump(rep0_full, f, indent=2, default=float)
    print(f"[audio] FROZEN coarse gap (speech vs music): "
          f"{rep0_full['coarse']['gap']:.4f} (probe-era scale: 0.271)")

    # ---- window tensors for training (CPU, capped) --------------------- #
    win_cache = []
    for mel in mels:
        T = mel.shape[-1]
        if T < WINDOW:
            w = F.pad(mel, (0, WINDOW - T))[:, :WINDOW]
            wins = [w]
        else:
            starts = list(range(0, T - WINDOW + 1, HOP))[:MAX_TRAIN_WINDOWS]
            wins = [mel[:, s:s + WINDOW] for s in starts]
        win_cache.append(torch.stack(wins))   # [n_win, 64, WINDOW]

    seeds = SEEDS
    if len(sys.argv) > 1:   # e.g. --seeds 0  (per-seed process, identical math)
        assert sys.argv[1] == "--seeds", "usage: contrast_audio.py [--seeds 0,1,2]"
        seeds = tuple(int(s) for s in sys.argv[2].split(","))
    results = {}
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = random.Random(seed)
        encoder, _ = probe.load_encoder(CKPT, device)   # fresh from v2 ckpt
        encoder.eval()   # BN stats frozen (fidelity to the frozen embedder)
        opt = torch.optim.AdamW(encoder.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=EPOCHS * N_BATCHES)
        final_loss = float("nan")
        for ep in range(EPOCHS):
            for idx in sample_room_batches(room_names, N_BATCHES, rng):
                # per-clip contiguous window span (jitter augmentation)
                zs = []
                for i in idx:
                    wins = win_cache[i]
                    n = wins.shape[0]
                    if n >= 4:
                        span = rng.randint(max(2, n // 2), n)
                        s0 = rng.randint(0, n - span)
                        w = wins[s0: s0 + span]
                    else:
                        w = wins
                    zs.append(encoder(w.unsqueeze(1).to(device))
                              .mean(dim=0))
                z = F.normalize(torch.stack(zs), dim=-1)
                loss = contrast_loss(z, [room_names[i] for i in idx])
                loss = loss + spread_hinge(z, [room_names[i] for i in idx],
                                           base_spread)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                final_loss = float(loss.item())
        encoder.eval()
        z1 = embed_cached(encoder, mels, device).cpu().numpy()
        rep1_full = probe_report(z1, clips, coarse_b_rooms=music_rooms)
        rep1_tap = probe_report(z1[tap_idx], clips_tap)
        post_spread = contrast.room_spread(z1, room_names)
        results[seed] = {
            "tap": rep1_tap,
            "full": rep1_full,
            "spread_preservation": {
                r: float(post_spread.get(r, float("nan")) / base_spread[r])
                for r in base_spread},
            "final_batch_loss": final_loss,
        }
        torch.save({
            "encoder": encoder.state_dict(),
            "config": cfg, "seed": seed,
            "objective": {"tau": contrast.TAU, "spread_slack": contrast.SPREAD_SLACK,
                          "spread_lambda": contrast.SPREAD_LAMBDA,
                          "epochs": EPOCHS, "n_batches": N_BATCHES,
                          "lr": LR, "window": WINDOW, "hop": HOP,
                          "max_train_windows": MAX_TRAIN_WINDOWS},
        }, os.path.join(OUT, f"audio_contrast_seed{seed}.pt"))
        # crash-safe: flush per-seed results as each seed lands
        with open(os.path.join(OUT, f"audio_contrast_results_seed{seed}.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"[audio] seed={seed}: fine gap={rep1_tap['separability']['gap']:.4f} "
              f"disc={rep1_tap['room_discrimination']:.3f} "
              f"heldout={rep1_tap['room_discrimination_speaker_heldout']:.3f} "
              f"coarse={rep1_full['coarse']['gap']:.4f} "
              f"mean_spread(full)={rep1_full['mean_spread']:.3f} "
              f"(frozen {rep0_full['mean_spread']:.3f})")
    with open(os.path.join(OUT, "audio_contrast_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("[audio] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
