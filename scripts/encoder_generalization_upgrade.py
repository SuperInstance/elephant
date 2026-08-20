"""scripts/encoder_generalization_upgrade.py — the REGISTERED encoder-generalization
upgrades, retrieval half of the thesis (2026-08-20).

Honest-test background (CONTRAST-HELDOUT-ROOM-2026-08-19): the text contrast
head trained on 17 rooms (tap-night-1/2 + 15 non-tap) FAILS to separate
UNSEEN tap-night-3/4 (held-out fine gaps 0.0713/0.1073/0.0295 vs the 0.05
noise floor; seed 2 below floor -> not demonstrated at the registered
3-consecutive-seeds standard). The registered upgrades (open question #2 in
research/topic.md): (a) more training nights, (b) a second/different
train-test split, (c) a generalizing (contrastive) objective — per
RESEARCH-NOTE-MEMORIZATION-GEOMETRY-2026-08-19 the objective needs invariance
pressure that exists at eval time: positives as independent views of the
same identity (token-dropout / split-half views).

This script runs ALL of them against the same frozen v2 trunk, the same
clips, the same loss family, the same noise floor (fine gap > 0.05), and
reports honestly whether ANY clears the floor 3/3 seeds.

Modes (--mode):
  baseline      holdout [tap-night-3, tap-night-4], objective plain
                (reproduction of the committed FAIL, unmodified training)
  more-nights   holdout [tap-night-3] only (18 train rooms, 3 tap nights)
                -> attempt (a): MORE training nights of the same cast
  second-split  holdout [tap-night-1, tap-night-2] (the nights that were
                TRAIN in the baseline; swapped split) -> attempt (b)
  views         holdout [tap-night-3, tap-night-4], objective VIEWS:
                positives = content-disjoint HALF-SPLIT views of the same
                night (fresh random split every step) -> attempt (c1)
  dropout       holdout [tap-night-3, tap-night-4], objective DROPOUT:
                positives = same-room clips with fresh token-dropout masks
                (p=0.3) every step; eval on FULL text -> attempt (c2)

All modes: CPU-only, τ=0.15, spread hinge vs the frozen baseline, 200x60
batches, LR 1e-4 cosine, 3 seeds (0/1/2), per-seed crash-safe JSON under
checkpoints/contrast/upgrade/<mode>/. The committed checkpoints and results
files are NEVER touched.
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys
import time

# CPU-only hardening (registered): before any torch import.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np

from elephant import contrast
from elephant.contrast import (
    Clip, contrast_loss, probe_report, sample_room_batches, spread_hinge,
    text_clips_from_room,
)
from elephant.learned import TextEncoder, Vocab
from contrast_train_text import (
    CKPT, OUT, WINDOW, MAX_LEN, build_text_corpus,
)

UP_DIR = os.path.join(OUT, "upgrade")
NOISE_FLOOR = 0.05
SEEDS = (0, 1, 2)
EPOCHS = 200
N_BATCHES = 60
LR = 1e-4
DROPOUT_P = 0.3

# venue/cast families for the SECONDARY reading (what the geometry says is
# the learnable level): tap nights + open mic are one cast/venue family.
TAP_FAMILY = {"tap-night-1", "tap-night-2", "tap-night-3", "tap-night-4",
              "tap-open-mic"}

MODES = {
    "baseline":     {"holdout": ["tap-night-3", "tap-night-4"], "objective": "plain"},
    "more-nights":  {"holdout": ["tap-night-3", "tap-night-4"],
                      "objective": "plain",
                      "extra_nights": True},   # + tap-questions, tap-sequels
    "second-split": {"holdout": ["tap-night-1", "tap-night-2"], "objective": "plain"},
    "views":        {"holdout": ["tap-night-3", "tap-night-4"], "objective": "views"},
    "dropout":      {"holdout": ["tap-night-3", "tap-night-4"], "objective": "dropout"},
}


def parse_args(argv):
    mode, seeds = "baseline", SEEDS
    i = 1
    while i < len(argv):
        if argv[i] == "--mode":
            mode = argv[i + 1]
            if mode not in MODES:
                raise SystemExit(f"unknown mode {mode}; modes: {sorted(MODES)}")
        elif argv[i] == "--seeds":
            seeds = tuple(int(s) for s in argv[i + 1].split(","))
        else:
            raise SystemExit("usage: encoder_generalization_upgrade.py "
                             "[--mode baseline|more-nights|second-split|views|dropout] "
                             "[--seeds 0,1,2]")
        i += 2
    return mode, seeds


def build_clips(room_list):
    cs, ts = [], []
    for name, msgs in room_list:
        for clip, toks in text_clips_from_room(name, msgs, window=WINDOW):
            cs.append(clip)
            ts.append(toks)
    return cs, ts


def encode_tokens(encoder, tokens, vocab):
    import torch
    ids = [vocab.encode(t, max_len=MAX_LEN) for t in tokens]
    X = torch.zeros((len(ids), MAX_LEN), dtype=torch.long)
    for i, x in enumerate(ids):
        X[i, : len(x)] = torch.tensor(x)
    with torch.no_grad():
        z = torch.nn.functional.normalize(encoder(X), dim=-1)
    return z.numpy()


def heldout_eval_block(zh, ho_clips):
    ho_rooms = [c.room for c in ho_clips]
    ho_speakers = [c.speaker for c in ho_clips]
    hf = contrast.separability(zh, ho_rooms)
    hd = contrast.room_discrimination(zh, ho_rooms, ho_speakers)
    hsp = contrast.room_discrimination(zh, ho_rooms, ho_speakers,
                                       holdout_speaker=True)
    return {
        "holdout_rooms": sorted(set(ho_rooms)),
        "n_clips": len(ho_clips),
        "fine_gap": hf["gap"],
        "same_room_mean": hf["same_room_mean"],
        "cross_room_mean": hf["cross_room_mean"],
        "n_same": hf["n_same"], "n_cross": hf["n_cross"],
        "room_discrimination": hd,
        "room_discrimination_speaker_heldout": hsp,
        "noise_floor": NOISE_FLOOR,
        "fine_gap_beats_noise_floor": bool(hf["gap"] > NOISE_FLOOR),
    }


def venue_discrimination(zho, ho_clips, ztr, tr_clips):
    """SECONDARY reading: nearest TRAINING clip's venue/cast family.

    The geometry note's verdict: the only signal that transfers is the
    venue/cast family (tap vs not-tap). For each held-out clip, its top-1
    training neighbor; correct if same venue family. Reported, never
    promoted to the primary bar.
    """
    tr_rooms = np.array([c.room for c in tr_clips])
    S = zho @ ztr.T
    nn = tr_rooms[np.argmax(S, axis=1)]
    correct = sum(1 for c, r in zip(ho_clips, nn)
                  if ((c.room in TAP_FAMILY) == (r in TAP_FAMILY)))
    return correct / len(ho_clips)


def build_rooms_with_extras(cfg, tag=""):
    """build_text_corpus() + the (a) extra same-cast training nights."""
    import glob as _glob
    from elephant.learned import room_from_file
    rooms = build_text_corpus()
    if cfg.get("extra_nights"):
        d = "/home/eileen/projects/ai-writings/tap-trades/2026-08-16"
        extra = [
            ("tap-questions",
             sorted(_glob.glob(os.path.join(d, "questions", "*.md")))),
            ("tap-sequels",
             sorted(_glob.glob(os.path.join(d, "sequels", "*.md")))
             + sorted(_glob.glob(os.path.join(d, "sequels-night2", "*.md")))),
        ]
        for name, files in extra:
            msgs = []
            for f in files:
                msgs.extend(room_from_file(f, name).messages)
            if msgs:
                rooms.append((name, msgs))
        if tag:
            print(f"[{tag}] extra training nights added", flush=True)
    return rooms


def main() -> int:
    import torch

    mode, seeds = parse_args(sys.argv)
    cfg = MODES[mode]
    holdout_rooms = cfg["holdout"]
    objective = cfg["objective"]
    out_dir = os.path.join(UP_DIR, mode)
    os.makedirs(out_dir, exist_ok=True)
    results_file = os.path.join(out_dir, "results.json")

    tag = f"upgrade/{mode}/{objective}"
    print(f"[{tag}] holdout={holdout_rooms} objective={objective} seeds={seeds}",
          flush=True)

    vocab = Vocab.load(os.path.join(CKPT, "learned_vocab.txt"))
    rooms = build_rooms_with_extras(cfg, tag)
    known = {n for n, _m, *_ in rooms}
    missing = [r for r in holdout_rooms if r not in known]
    if missing:
        raise SystemExit(f"unknown holdout rooms {missing}")

    train_rooms = [(n, m) for n, m, *_ in rooms if n not in holdout_rooms]
    held_rooms = [(n, m) for n, m, *_ in rooms if n in holdout_rooms]

    clips, tokens_all = build_clips(train_rooms)
    room_names = [c.room for c in clips]
    ho_clips, ho_tokens = build_clips(held_rooms)
    ho_rooms = [c.room for c in ho_clips]
    ho_speakers = [c.speaker for c in ho_clips]
    print(f"[{tag}] train rooms={len(train_rooms)} clips={len(clips)}; "
          f"held-out rooms={sorted(set(ho_rooms))} clips={len(ho_clips)}",
          flush=True)

    # ---- frozen baseline (v2 trunk) + spread targets ------------------- #
    base = TextEncoder(len(vocab), d_model=64, d_trunk=64)
    sd = torch.load(os.path.join(CKPT, "learned_dials.pt"), map_location="cpu")
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items()
              if k.startswith("encoder.")}
    base.load_state_dict(enc_sd)
    z0 = encode_tokens(base, tokens_all, vocab)
    base_spread = contrast.room_spread(z0, room_names)
    base_report = probe_report(z0, clips)
    print(f"[{tag}] FROZEN baseline: gap={base_report['separability']['gap']:.4f} "
          f"disc={base_report['room_discrimination']:.3f}", flush=True)

    # trunk reference on held-out clips
    z0h = encode_tokens(base, ho_tokens, vocab)
    t_fine = contrast.separability(z0h, ho_rooms)
    t_disc = contrast.room_discrimination(z0h, ho_rooms, ho_speakers)
    t_sp = contrast.room_discrimination(z0h, ho_rooms, ho_speakers,
                                        holdout_speaker=True)
    print(f"[{tag}] trunk on held-out: gap={t_fine['gap']:.4f} disc={t_disc:.3f} "
          f"sp-heldout={t_sp:.3f}", flush=True)

    # ---- ids tensor once ------------------------------------------------ #
    ids_list = [vocab.encode(t, max_len=MAX_LEN) for t in tokens_all]
    Xt = torch.zeros((len(ids_list), MAX_LEN), dtype=torch.long)
    for i, x in enumerate(ids_list):
        Xt[i, : len(x)] = torch.tensor(x)

    results = {"mode": mode, "objective": objective,
               "holdout_rooms": holdout_rooms, "seeds": {}}
    for seed in seeds:
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = random.Random(seed)
        model = TextEncoder(len(vocab), d_model=64, d_trunk=64)
        model.load_state_dict(enc_sd)          # start from the v2 checkpoint
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=EPOCHS * N_BATCHES)
        model.train()
        for ep in range(EPOCHS):
            for idx in sample_room_batches(room_names, N_BATCHES, rng):
                xb = Xt[idx]
                if objective == "dropout":
                    # fresh token-dropout mask per step (independent views)
                    drop = np.random.rand(xb.shape[0], xb.shape[1]) < DROPOUT_P
                    drop &= (xb.numpy() != 0)
                    xb = xb.clone()
                    xb[torch.from_numpy(drop)] = 0
                z = model(xb)
                loss = contrast_loss(z, [room_names[i] for i in idx],
                                     tau=contrast.TAU)
                loss = loss + spread_hinge(z, [room_names[i] for i in idx],
                                           base_spread)
                if objective == "views":
                    # content-disjoint half-split views: positives = the
                    # OTHER half of the same night only (fresh split/step).
                    # Vectorized (the wesley-stream room is 826 clips/batch;
                    # a python loop version was ~5x slower).
                    room_of = np.array([room_names[i] for i in idx])
                    uniq = sorted(set(room_of.tolist()))
                    rid = np.searchsorted(uniq, room_of)
                    half = np.zeros(len(idx), dtype=int)
                    for r in range(len(uniq)):
                        ii = np.where(rid == r)[0]
                        perm = list(range(len(ii)))
                        rng.shuffle(perm)
                        half[ii[perm[len(ii) // 2:]]] = 1
                    same_room = rid[:, None] == rid[None, :]
                    diff_half = half[:, None] != half[None, :]
                    pos = torch.from_numpy(same_room & diff_half)
                    zn = torch.nn.functional.normalize(z, dim=-1)
                    sim = zn @ zn.t() / contrast.TAU
                    B = len(idx)
                    eye = torch.eye(B, dtype=torch.bool)
                    logits = sim.masked_fill(eye, float("-inf"))
                    logp = torch.log_softmax(logits, dim=1)
                    pos_terms = (logp.masked_fill(~pos, 0.0).sum(1)
                                 / pos.sum(1).clamp(min=1))
                    view_loss = -pos_terms.mean()
                    loss = view_loss + spread_hinge(
                        z, [room_names[i] for i in idx], base_spread)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        model.eval()
        dt = time.time() - t0
        z1 = encode_tokens(model, tokens_all, vocab)
        rep = probe_report(z1, clips)
        zh = encode_tokens(model, ho_tokens, vocab)
        he = heldout_eval_block(zh, ho_clips)
        vd = venue_discrimination(zh, ho_clips, z1, clips)
        he["venue_discrimination_train_nn"] = vd
        results["seeds"][seed] = {
            "train_report": {
                "fine_gap": rep["separability"]["gap"],
                "room_discrimination": rep["room_discrimination"],
                "room_discrimination_speaker_heldout":
                    rep["room_discrimination_speaker_heldout"],
                "mean_spread": rep["mean_spread"],
            },
            "heldout_eval": he,
            "seconds": round(dt, 1),
        }
        torch.save(model.state_dict(), os.path.join(
            out_dir, f"model_seed{seed}.pt"))
        with open(os.path.join(out_dir, f"results_seed{seed}.json"), "w") as f:
            json.dump({str(seed): results["seeds"][seed]}, f, indent=2,
                      default=float)
        with open(results_file, "w") as f:      # crash-safe per seed
            json.dump(results, f, indent=2, default=float)
        verdict = "PASS" if he["fine_gap"] > NOISE_FLOOR else "FAIL"
        print(f"[{tag}] seed={seed} ({dt:.0f}s): TRAIN gap="
              f"{rep['separability']['gap']:.4f} | HELD-OUT fine gap="
              f"{he['fine_gap']:.4f} vs floor {NOISE_FLOOR} -> {verdict}; "
              f"disc={he['room_discrimination']:.3f} "
              f"sp-heldout={he['room_discrimination_speaker_heldout']:.3f} "
              f"venue-disc={vd:.3f}", flush=True)

    gaps = {s: results["seeds"][s]["heldout_eval"]["fine_gap"] for s in seeds}
    n_pass = sum(g > NOISE_FLOOR for g in gaps.values())
    gap_str = ", ".join(f"seed{s}={gaps[s]:.4f}" for s in seeds)
    all_pass = n_pass == len(seeds)
    print(f"[{tag}] VERDICT: held-out fine gaps [{gap_str}] -> "
          f"{n_pass}/{len(seeds)} seeds > {NOISE_FLOOR}: "
          f"{'PASS — generalizes' if all_pass else 'FAIL — does not clear the floor 3/3'}",
          flush=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    # final merge: rebuild results.json from per-seed files (parallel-seed
    # processes each wrote their own seed; the last writer used to clobber)
    merged = {"mode": mode, "objective": objective,
              "holdout_rooms": holdout_rooms, "seeds": {}}
    for s in sorted(seeds):
        pf = os.path.join(out_dir, f"results_seed{s}.json")
        if os.path.exists(pf):
            merged["seeds"][str(s)] = json.load(open(pf))[str(s)]
    if len(merged["seeds"]) == len(seeds):
        with open(results_file, "w") as f:
            json.dump(merged, f, indent=2, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
