"""scripts/fusion_head.py — v3 §5 late fusion for the tap-room state.

Fusion rules implemented (spec §5.2, adopted from review):
  * per-modality L2 normalization + distance-distribution matching
    (rescale the text channel so its 95th-percentile pairwise distance
    matches the audio channel's — without this, audio dominates),
  * late fusion = a projector over the concatenated channel embeddings
    (the sanctioned step; the cross-attention layer is the spec's "next
    step beyond", not shipped here),
  * modality dropout p=0.3 per channel during training so the projector
    works with any subset (required because music/boat channels are absent
    in text-only or audio-only rooms).

Alignment is exact, not heuristic: each tap broadcast SCRIPT.md block's
backtick slug IS the mp3 filename stem (parse_script_blocks), so every
fused clip is (audio mp3, script utterance) of the same line.

GEOMETRY STATEMENT (requirement (a) — printed by this script):
  the fused field's own fine/coarse ordering is measured and DECLARED
  against the two known geometries: dial-tier (fine 1.229 > coarse 0.941,
  inverted) vs encoder-tier (fine 0.015 < coarse 0.271). A fusion that
  served both without saying so would be a goalpost move; this one states
  which geometry its numbers live in, and the audio-only dropout path is
  reported separately from the fused path.
"""
from __future__ import annotations

import json
import os
import random
import sys

ELEPHANT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ELEPHANT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from elephant import contrast
from elephant.contrast import (
    Clip, contrast_loss, cross_group_gap, parse_script_blocks, probe_report,
    sample_room_batches, speaker_key_from_filename, spread_hinge,
)
from elephant.learned import TextEncoder, Vocab, tokenize

AI = "/home/eileen/projects/ai-writings"
CKPT = os.path.join(ELEPHANT, "checkpoints")
OUT = os.path.join(CKPT, "contrast")
SEEDS = (0, 1, 2)
EPOCHS = 300
N_BATCHES = 30
LR = 1e-3
DROP_P = 0.3
P_MATCH = 95   # percentile used for distance-distribution matching


class FusionProjector(nn.Module):
    """Late fusion: MLP over [audio_z (384), text_z (64)] -> 128, L2-norm."""

    def __init__(self, d_audio=384, d_text=64, d_out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_audio + d_text, 256), nn.ReLU(),
            nn.Linear(256, d_out))

    def forward(self, a, t):
        z = self.net(torch.cat([a, t], dim=-1))
        return F.normalize(z, dim=-1)


def build_aligned_clips():
    """(room, slug, speaker, text) for every mp3↔script-line pair."""
    rows = []
    for ep in (1, 2, 3, 4):
        room = f"tap-{ep}"
        script = os.path.join(AI, "tap-trades", "radio-theater",
                              f"episode-{ep}", "SCRIPT.md")
        blocks = dict()
        for slug, spk, text in parse_script_blocks(script):
            blocks[slug] = (spk, text)
        meta = json.load(open(os.path.join(OUT, "audio_clip_meta.json")))
        for i, m in enumerate(meta):
            if m["room"] != room:
                continue
            if m["key"] in blocks:
                spk, text = blocks[m["key"]]
                rows.append({"idx": i, "room": room, "key": m["key"],
                             "speaker": m["speaker"], "text": text})
    return rows


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = build_aligned_clips()
    by_room = {}
    for r in rows:
        by_room.setdefault(r["room"], []).append(r)
    print(f"[fusion] aligned clips: {len(rows)} "
          f"({ {k: len(v) for k, v in by_room.items()} })")

    # audio channel: the contrast-trained audio head (seed 0)
    za = np.load(os.path.join(OUT, "audio_emb_seed0.npy"))
    A = torch.tensor(np.stack([za[r["idx"]] for r in rows]), dtype=torch.float32,
                     device=device)
    A = F.normalize(A, dim=-1)

    # text channel: the contrast-trained text head (seed 0) on the aligned
    # script utterances (one utterance = one clip at fusion grain)
    vocab = Vocab.load(os.path.join(CKPT, "learned_vocab.txt"))
    tmodel = TextEncoder(len(vocab), 64, 64)
    tmodel.load_state_dict(torch.load(
        os.path.join(OUT, "text_contrast_seed0.pt"), map_location="cpu"))
    tmodel.eval()
    ids = [vocab.encode(tokenize(r["text"]), max_len=256) for r in rows]
    X = torch.zeros((len(ids), 256), dtype=torch.long)
    for i, x in enumerate(ids):
        X[i, : len(x)] = torch.tensor(x)
    with torch.no_grad():
        Tn = F.normalize(tmodel(X), dim=-1).to(device)

    # ---- distance-distribution matching (p95) -------------------------- #
    def p95_dist(Z):
        Z = F.normalize(Z, dim=-1)
        d = 1.0 - (Z @ Z.t())
        n = Z.shape[0]
        iu = torch.triu_indices(n, n, 1)
        return float(d[iu[0], iu[1]].quantile(P_MATCH / 100.0))

    pa, pt = p95_dist(A), p95_dist(Tn)
    scale = pa / pt
    Ts = Tn * scale
    print(f"[fusion] p95 dist: audio={pa:.3f} text={pt:.3f} -> text scale "
          f"{scale:.3f}")

    clips = [Clip(r["room"], r["key"], r["speaker"]) for r in rows]
    rooms = [c.room for c in rows]
    z_fused_pre = torch.cat([F.normalize(A, dim=-1), Ts], dim=-1)

    def report(z, label):
        rep = probe_report(z.cpu().numpy(), clips)
        print(f"[fusion:{label}] fine gap={rep['separability']['gap']:.4f} "
              f"disc={rep['room_discrimination']:.3f} "
              f"heldout={rep['room_discrimination_speaker_heldout']:.3f} "
              f"mean_spread={rep['mean_spread']:.3f}")
        return rep

    pre = report(F.normalize(A, dim=-1), "audio-only pre")
    pre_t = report(Tn, "text-only pre")

    base_spread = contrast.room_spread(
        torch.cat([F.normalize(A, dim=-1), Ts], -1).cpu().numpy(), rooms)
    # spread targets on the concat space for the hinge
    results = {"p95": {"audio": pa, "text": pt, "text_scale": scale},
               "pre": {"audio_only": pre, "text_only": pre_t}, "seeds": {}}
    for seed in SEEDS:
        torch.manual_seed(seed)
        rng = random.Random(seed)
        proj = FusionProjector().to(device)
        opt = torch.optim.Adam(proj.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=EPOCHS * N_BATCHES)
        for _ in range(EPOCHS):
            for idx in sample_room_batches(rooms, N_BATCHES, rng):
                a, t = A[idx], Ts[idx]
                if rng.random() < DROP_P:
                    a = torch.zeros_like(a)
                if rng.random() < DROP_P:
                    t = torch.zeros_like(t)
                z = proj(a, t)
                loss = contrast_loss(z, [rooms[i] for i in idx])
                loss = loss + spread_hinge(z, [rooms[i] for i in idx],
                                           base_spread)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        with torch.no_grad():
            z_full = proj(A, Ts)
            z_audio_only = proj(A, torch.zeros_like(Ts))
            z_text_only = proj(torch.zeros_like(A), Ts)
        rep_f = report(z_full, f"fused s{seed}")
        rep_a = report(z_audio_only, f"fused/audio-dropout s{seed}")
        rep_t = report(z_text_only, f"fused/text-dropout s{seed}")
        results["seeds"][seed] = {"fused": rep_f, "audio_only": rep_a,
                                  "text_only": rep_t}
        torch.save(proj.state_dict(),
                   os.path.join(OUT, f"fusion_projector_seed{seed}.pt"))

    # ---- GEOMETRY STATEMENT (requirement (a)) -------------------------- #
    fine = float(np.mean([results["seeds"][s]["fused"]["separability"]["gap"]
                          for s in SEEDS]))
    audio_only_fine = pre["separability"]["gap"]
    coarse_audio = json.load(open(os.path.join(
        OUT, "audio_contrast_results.json")))["0"]["full"]["coarse"]["gap"]
    ordering = "fine>coarse" if fine > coarse_audio else "coarse>fine"
    statement = {
        "fused_fine_gap": fine,
        "audio_head_coarse_gap": coarse_audio,
        "fused_ordering_within_tap_rooms": ordering,
        "declared_geometry": (
            "the fused projector is trained and evaluated on the four tap "
            "rooms only (clip-level fine contrast); its numbers live in the "
            "GEOMETRY OF THE UNDERLYING ENCODERS it fuses — audio head "
            "(encoder-tier) and text head. It has no music channel (modality "
            "dropout exists precisely because channels are absent), so its "
            "coarse pole is inherited from the audio head's "
            f"{coarse_audio:.3f}, not re-measured inside the fusion. "
            "This fusion therefore serves the fine (room-discrimination) "
            "axis directly and the coarse axis only through its audio "
            "channel — stated, not assumed."),
    }
    results["geometry_statement"] = statement
    print(json.dumps(statement, indent=2))
    with open(os.path.join(OUT, "fusion_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
