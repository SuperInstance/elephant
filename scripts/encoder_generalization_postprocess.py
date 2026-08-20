#!/usr/bin/env python3
"""scripts/encoder_generalization_postprocess.py — rebuild merged results.json
for encoder_generalization_upgrade.py modes whose seeds ran as PARALLEL
processes (each process wrote only its own seed; the last writer clobbered
the shared results.json). Loads each saved model_seed{k}.pt, re-embeds the
train + held-out clips with the SAME code paths, and writes the merged file.

Usage: python3 scripts/encoder_generalization_postprocess.py [mode ...]
(default: all modes with saved checkpoints under checkpoints/contrast/upgrade/).
"""
from __future__ import annotations

import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np  # noqa: E402

from elephant import contrast  # noqa: E402
from elephant.contrast import probe_report  # noqa: E402
from elephant.learned import TextEncoder, Vocab  # noqa: E402

from contrast_train_text import CKPT, MAX_LEN, WINDOW  # noqa: E402
from encoder_generalization_upgrade import (  # noqa: E402
    MODES, UP_DIR, build_clips, build_rooms_with_extras, encode_tokens,
    heldout_eval_block, venue_discrimination,
)


def rebuild(mode: str) -> bool:
    import torch

    out_dir = os.path.join(UP_DIR, mode)
    cfg = MODES[mode]
    holdout_rooms = cfg["holdout"]
    models = sorted(f for f in os.listdir(out_dir)
                    if f.startswith("model_seed") and f.endswith(".pt"))
    if not models:
        return False
    seeds = sorted(int(f[len("model_seed"):-len(".pt")]) for f in models)
    print(f"[post] {mode}: holdout={holdout_rooms} seeds={seeds}")

    vocab = Vocab.load(os.path.join(CKPT, "learned_vocab.txt"))
    rooms = build_rooms_with_extras(cfg)
    train_rooms = [(n, m) for n, m, *_ in rooms if n not in holdout_rooms]
    held_rooms = [(n, m) for n, m, *_ in rooms if n in holdout_rooms]
    clips, tokens_all = build_clips(train_rooms)
    ho_clips, ho_tokens = build_clips(held_rooms)
    ho_rooms = [c.room for c in ho_clips]

    base = TextEncoder(len(vocab), d_model=64, d_trunk=64)
    sd = torch.load(os.path.join(CKPT, "learned_dials.pt"), map_location="cpu")
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items()
              if k.startswith("encoder.")}
    base.load_state_dict(enc_sd)

    merged = {"mode": mode, "objective": cfg["objective"],
              "holdout_rooms": holdout_rooms, "seeds": {}}
    for seed in seeds:
        model = TextEncoder(len(vocab), d_model=64, d_trunk=64)
        model.load_state_dict(enc_sd)
        model.load_state_dict(torch.load(
            os.path.join(out_dir, f"model_seed{seed}.pt"), map_location="cpu"))
        model.eval()
        z1 = encode_tokens(model, tokens_all, vocab)
        rep = probe_report(z1, clips)
        zh = encode_tokens(model, ho_tokens, vocab)
        he = heldout_eval_block(zh, ho_clips)
        he["venue_discrimination_train_nn"] = venue_discrimination(
            zh, ho_clips, z1, clips)
        merged["seeds"][str(seed)] = {
            "train_report": {
                "fine_gap": rep["separability"]["gap"],
                "room_discrimination": rep["room_discrimination"],
                "room_discrimination_speaker_heldout":
                    rep["room_discrimination_speaker_heldout"],
                "mean_spread": rep["mean_spread"],
            },
            "heldout_eval": he,
        }
        print(f"[post] {mode} seed={seed}: TRAIN gap="
              f"{rep['separability']['gap']:.4f} HELD-OUT gap="
              f"{he['fine_gap']:.4f} disc={he['room_discrimination']:.3f} "
              f"venue={he['venue_discrimination_train_nn']:.3f}")

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(merged, f, indent=2, default=float)
    return True


def main() -> int:
    modes = sys.argv[1:] or sorted(
        d for d in os.listdir(UP_DIR)
        if os.path.isdir(os.path.join(UP_DIR, d)) and d in MODES)
    for m in modes:
        rebuild(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
