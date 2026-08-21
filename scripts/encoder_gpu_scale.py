"""scripts/encoder_gpu_scale.py — the CUDA port + corpus-scale probe (2026-08-20).

Extends the 2026-08-20 "more identity data clears the held-out floor" finding
(ENCODER-GENERALIZATION-UPGRADE-2026-08-20.md, mode `more-nights`: 3/3 PASS,
mean 0.1099) along TWO axes:

  1. DEVICE — the contrastive training loop ported to CUDA (RTX 4050 Laptop,
     6.4 GB VRAM). No CUDA_VISIBLE_DEVICES pin here (the registered scripts
     pin CPU-only at import; that bug-era hardening is intentionally NOT
     copied). Tensors + model live on the GPU; all probe metrics stay numpy
     on CPU, byte-identical definitions to the registered harness.
  2. CORPUS SCALE — more same-cast (trades cast: LUCINEER/WELDER/CARPENTER/
     SHIPWRIGHT/MASON/COMPOSITE/WESLEY) TRAINING nights. The held-out set is
     UNTOUCHED: tap-night-3 + tap-night-4, the same 18 clips, the same
     0.05 noise floor, the same schedule (200x60 batches, LR 1e-4 cosine,
     tau=0.15 + spread hinge), the same 3 seeds.

     New same-cast training material (the unused 2026-08-16 tap-trades
     filing, same cast as the held-out nights):
       * tap-pieces — the six SOURCE trade monologues (carpenter/mason/
         shipwright/welder/composite/wesley-the-room): the canonical
         per-trade vocabulary, never before in any condition.
       * tap-sequels-n1 / tap-sequels-n2 — the sequels split into their two
         real night filings (the joint-map's own reading: "the same
         argument, run a second night"). Only in scale-split.

     Dose axis (same-cast trades filings in TRAIN):
       baseline    2  (evening-1, evening-2)              [+ open-mic always]
       more        4  (+ questions, sequels-merged)       [the registered 0.1099]
       scale       5  (+ pieces)                          -> 6 same-cast rooms
       scale-split 6  (+ pieces, sequels split n1/n2)     -> 7 same-cast rooms

     NOT used: tap-joint-map/day-joint-map (meta-analysis docs, not night
     material) and the improv/speed-dating nights (different cast — the
     fleet-model cast, not the trades cast of the held-out nights — and
     their generated data does not exist; generating would spend external
     API calls for cast-mismatched nights).

Corpora (--corpus): baseline | more | scale | scale-split
Device (--device): auto (cuda if available) | cpu — cpu exists for the
GPU-vs-CPU speedup measurement on the identical code path.

Everything is additive: results under checkpoints/contrast/gpu_scale/<corpus>/;
the registered checkpoints/results are never touched. No git commit.
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# NOTE: deliberately NO os.environ["CUDA_VISIBLE_DEVICES"] = "" here — this
# is the GPU port. (contrast_train_text.py / encoder_generalization_upgrade.py
# pin CPU-only at import, which is why their builders are copied below.)

import numpy as np

from elephant import contrast
from elephant.contrast import (
    contrast_loss, probe_report, sample_room_batches, spread_hinge,
    text_clips_from_room,
)
from elephant.learned import TextEncoder, Vocab, room_from_file

AI = "/home/eileen/projects/ai-writings"
CKPT = os.path.join(_ROOT, "checkpoints")
OUT = os.path.join(CKPT, "contrast", "gpu_scale")

WINDOW = 8            # registered clip window (non-overlapping)
MAX_LEN = 256
EPOCHS = 200          # registered schedule
N_BATCHES = 60
LR = 1e-4
NOISE_FLOOR = 0.05
SEEDS = (0, 1, 2)
HOLDOUT = ("tap-night-3", "tap-night-4")   # NEVER touched, same 18 clips

TAP_D = os.path.join(AI, "tap-trades", "2026-08-16")
RT = os.path.join(AI, "tap-trades", "radio-theater")

# venue/cast family for the secondary reading (same as the upgrade run)
TAP_FAMILY = {"tap-night-1", "tap-night-2", "tap-night-3", "tap-night-4",
              "tap-open-mic"}

CORPORA = {
    "baseline":    {"extras": (),                      "nights": 2},
    "more":        {"extras": ("questions", "sequels"), "nights": 4},
    "scale":       {"extras": ("questions", "sequels", "pieces"), "nights": 5},
    "scale-split": {"extras": ("questions", "sequels-split", "pieces"),
                    "nights": 6},
}


# --------------------------------------------------------------------- #
# Corpus — build_text_corpus COPIED VERBATIM from contrast_train_text.py #
# (that module pins CUDA_VISIBLE_DEVICES="" at import; we must not)      #
# --------------------------------------------------------------------- #
def build_text_corpus() -> list:
    """The v3 §1.1 text rooms as (room_name, messages). Verbatim copy."""
    rooms = []

    def add_room(name, files):
        msgs = []
        for f in files:
            if os.path.exists(f):
                msgs.extend(room_from_file(f, name).messages)
        if msgs:
            rooms.append((name, msgs))

    d = TAP_D
    add_room("tap-night-1", [os.path.join(d, "evening-at-the-tap.md"),
                             os.path.join(RT, "episode-1", "SCRIPT.md")])
    add_room("tap-night-2", [os.path.join(d, "evening-2-open-question-night.md"),
                             os.path.join(RT, "episode-2", "SCRIPT.md")])
    add_room("tap-night-3", [os.path.join(d, "evening-3-adaptation-night.md"),
                             os.path.join(RT, "episode-3", "SCRIPT.md")]
             + sorted(glob.glob(os.path.join(d, "adaptations", "*.md"))))
    add_room("tap-night-4", [os.path.join(RT, "episode-4", "SCRIPT.md")]
             + sorted(glob.glob(os.path.join(d, "lenses", "*.md"))))
    for epd in sorted(glob.glob(os.path.join(
            AI, "radio-theater", "compass-head-radio-hour",
            "episode-*", "script", "episode-*-script.md"))):
        num = os.path.basename(epd).split("-")[1]
        add_room(f"compass-{num}", [epd])
    add_room("front-door", sorted(glob.glob(os.path.join(
        AI, "radio-theater", "the-front-door", "*.md"))))
    add_room("tap-open-mic", [os.path.join(AI, "radio-theater",
                                           "tap-open-mic-night.md")]
             + sorted(glob.glob(os.path.join(AI, "tap-trades", "open-mic",
                                             "2026-08-16", "*.md"))))
    add_room("tavern-night", sorted(glob.glob(os.path.join(
        AI, "radio-theater", "tavern-night", "*.md"))))
    add_room("front-door", sorted(glob.glob(os.path.join(
        AI, "radio-theater", "the-front-door", "*.md"))))
    add_room("dogs-fell-in-love", sorted(glob.glob(os.path.join(
        AI, "radio-theater", "dogs-fell-in-love", "episode-*.md"))))
    add_room("channel-42-dawn", [os.path.join(AI, "radio-theater",
                                              "channel-42-dawn",
                                              "morning-show-script.md")])
    add_room("fleet-radio-004", [os.path.join(
        AI, "radio", "fleet-radio-004-the-excavators-daughter.md")])
    add_room("fleet-radio-005", [os.path.join(
        AI, "radio", "fleet-radio-005-platos-shell.md")])
    add_room("speeches", sorted(glob.glob(os.path.join(AI, "speeches", "*.md"))))
    add_room("wesley-stream", sorted(glob.glob(os.path.join(
        AI, "wesley-stream", "*.md"))))
    return rooms


def add_extra_nights(rooms: list, extras) -> list:
    """Append the scale extra same-cast nights (trades-cast filings)."""
    def files_room(name, files):
        msgs = []
        for f in files:
            msgs.extend(room_from_file(f, name).messages)
        if msgs:
            rooms.append((name, msgs))

    if "questions" in extras:
        files_room("tap-questions",
                   sorted(glob.glob(os.path.join(TAP_D, "questions", "*.md"))))
    if "sequels" in extras:
        files_room("tap-sequels",
                   sorted(glob.glob(os.path.join(TAP_D, "sequels", "*.md")))
                   + sorted(glob.glob(os.path.join(TAP_D, "sequels-night2",
                                                    "*.md"))))
    if "sequels-split" in extras:
        files_room("tap-sequels-n1",
                   sorted(glob.glob(os.path.join(TAP_D, "sequels", "*.md"))))
        files_room("tap-sequels-n2",
                   sorted(glob.glob(os.path.join(TAP_D, "sequels-night2",
                                                    "*.md"))))
    if "pieces" in extras:
        files_room("tap-pieces", [
            os.path.join(TAP_D, f) for f in
            ("carpenter.md", "mason.md", "shipwright.md", "welder.md",
             "composite.md", "wesley-the-room.md")])
    return rooms


# --------------------------------------------------------------------- #
# Clips / encoding / eval — same definitions as the registered harness  #
# --------------------------------------------------------------------- #
def build_clips(room_list):
    cs, ts = [], []
    for name, msgs in room_list:
        for clip, toks in text_clips_from_room(name, msgs, window=WINDOW):
            cs.append(clip)
            ts.append(toks)
    return cs, ts


def encode_tokens(encoder, tokens, vocab, device):
    import torch
    ids = [vocab.encode(t, max_len=MAX_LEN) for t in tokens]
    X = torch.zeros((len(ids), MAX_LEN), dtype=torch.long, device=device)
    for i, x in enumerate(ids):
        X[i, : len(x)] = torch.tensor(x, device=device)
    with torch.no_grad():
        z = torch.nn.functional.normalize(encoder(X), dim=-1)
    return z.cpu().numpy()


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
    """Secondary reading: top-1 training neighbor's venue family (tapped)."""
    tr_rooms = np.array([c.room for c in tr_clips])
    S = zho @ ztr.T
    nn = tr_rooms[np.argmax(S, axis=1)]
    correct = sum(1 for c, r in zip(ho_clips, nn)
                  if ((c.room in TAP_FAMILY) == (r in TAP_FAMILY)))
    return correct / len(ho_clips)


# --------------------------------------------------------------------- #
# Training — the registered loop, device-parameterized                  #
# --------------------------------------------------------------------- #
def train_one(seed, model, Xt, room_names, base_spread, device, epochs=EPOCHS):
    """The registered 200x60 loop on `device`. Returns (seconds, n_steps)."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    model.load_state_dict(ENC_SD)          # fresh from the v2 checkpoint
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * N_BATCHES)
    model.train()
    t0 = time.time()
    n_steps = 0
    for ep in range(epochs):
        for idx in sample_room_batches(room_names, N_BATCHES, rng):
            xb = Xt[torch.tensor(idx, device=device)]
            z = model(xb)
            loss = contrast_loss(z, [room_names[i] for i in idx],
                                 tau=contrast.TAU)
            loss = loss + spread_hinge(z, [room_names[i] for i in idx],
                                       base_spread)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            n_steps += 1
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    return dt, n_steps


ENC_SD = None   # set in main (the frozen v2 trunk state dict)


def parse_args(argv):
    corpus, seeds, device = "scale", SEEDS, "auto"
    epochs = EPOCHS
    i = 1
    while i < len(argv):
        if argv[i] == "--corpus":
            corpus = argv[i + 1]
        elif argv[i] == "--seeds":
            seeds = tuple(int(s) for s in argv[i + 1].split(","))
        elif argv[i] == "--device":
            device = argv[i + 1]
        elif argv[i] == "--epochs":
            epochs = int(argv[i + 1])
        else:
            raise SystemExit("usage: encoder_gpu_scale.py "
                             "[--corpus baseline|more|scale|scale-split] "
                             "[--seeds 0,1,2] [--device auto|cuda|cpu] "
                             "[--epochs N]")
        i += 2
    if corpus not in CORPORA:
        raise SystemExit(f"unknown corpus {corpus}; known: {sorted(CORPORA)}")
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return corpus, seeds, device, epochs


def main() -> int:
    import torch

    global ENC_SD
    corpus, seeds, device, epochs = parse_args(sys.argv)
    out_dir = os.path.join(OUT, corpus)
    os.makedirs(out_dir, exist_ok=True)
    results_file = os.path.join(out_dir, f"results_{device}.json")
    tag = f"gpu_scale/{corpus}@{device}"

    if device == "cuda":
        free = torch.cuda.mem_get_info()[0] / 1e9
        print(f"[{tag}] CUDA device: {torch.cuda.get_device_name(0)} "
              f"({free:.1f} GB free)", flush=True)

    vocab = Vocab.load(os.path.join(CKPT, "learned_vocab.txt"))
    rooms = add_extra_nights(build_text_corpus(), CORPORA[corpus]["extras"])
    known = {n for n, _m, *_ in rooms}
    missing = [r for r in HOLDOUT if r not in known]
    if missing:
        raise SystemExit(f"unknown holdout rooms {missing}")

    train_rooms = [(n, m) for n, m, *_ in rooms if n not in HOLDOUT]
    held_rooms = [(n, m) for n, m, *_ in rooms if n in HOLDOUT]
    clips, tokens_all = build_clips(train_rooms)
    room_names = [c.room for c in clips]
    ho_clips, ho_tokens = build_clips(held_rooms)
    ho_rooms = [c.room for c in ho_clips]
    ho_speakers = [c.speaker for c in ho_clips]
    tap_train = sorted({r for r in room_names if r.startswith("tap-")})
    print(f"[{tag}] train rooms={len(train_rooms)} clips={len(clips)}; "
          f"HELD-OUT {sorted(set(ho_rooms))} clips={len(ho_clips)} "
          f"(UNTOUCHED); tap-cast train rooms: {tap_train}", flush=True)

    # ---- frozen v2 trunk + spread targets ------------------------------ #
    base = TextEncoder(len(vocab), d_model=64, d_trunk=64)
    sd = torch.load(os.path.join(CKPT, "learned_dials.pt"), map_location="cpu")
    ENC_SD = {k[len("encoder."):]: v for k, v in sd.items()
              if k.startswith("encoder.")}
    base.load_state_dict(ENC_SD)
    base.to(device)
    z0 = encode_tokens(base, tokens_all, vocab, device)
    base_spread = contrast.room_spread(z0, room_names)
    base_report = probe_report(z0, clips)
    z0h = encode_tokens(base, ho_tokens, vocab, device)
    t_fine = contrast.separability(z0h, ho_rooms)
    print(f"[{tag}] FROZEN trunk: train gap={base_report['separability']['gap']:.4f} "
          f"| held-out gap={t_fine['gap']:.4f} "
          f"(trunk anchor, matches registered 0.0019)", flush=True)

    # ---- ids tensor once, ON DEVICE ------------------------------------ #
    ids_list = [vocab.encode(t, max_len=MAX_LEN) for t in tokens_all]
    Xt = torch.zeros((len(ids_list), MAX_LEN), dtype=torch.long, device=device)
    for i, x in enumerate(ids_list):
        Xt[i, : len(x)] = torch.tensor(x, device=device)
    print(f"[{tag}] ids tensor on {device}: {tuple(Xt.shape)} "
          f"({Xt.numel() * 8 / 1e6:.1f} MB)", flush=True)

    results = {"corpus": corpus, "extras": list(CORPORA[corpus]["extras"]),
               "n_same_cast_train_nights": CORPORA[corpus]["nights"],
               "device": device, "epochs": epochs,
               "holdout_rooms": list(HOLDOUT), "seeds": {}}
    for seed in seeds:
        model = TextEncoder(len(vocab), d_model=64, d_trunk=64)
        dt, n_steps = train_one(seed, model, Xt, room_names, base_spread,
                                device, epochs)
        model.eval()
        z1 = encode_tokens(model, tokens_all, vocab, device)
        rep = probe_report(z1, clips)
        zh = encode_tokens(model, ho_tokens, vocab, device)
        he = heldout_eval_block(zh, ho_clips)
        vd = venue_discrimination(zh, ho_clips, z1, clips)
        he["venue_discrimination_train_nn"] = vd
        peak = (float(torch.cuda.max_memory_allocated()) / 1e9
                if device == "cuda" else None)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
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
            "steps_per_sec": round(n_steps / dt, 1),
            "peak_vram_gb": peak,
        }
        torch.save(model.state_dict(),
                   os.path.join(out_dir, f"model_seed{seed}_{device}.pt"))
        with open(os.path.join(out_dir, f"results_seed{seed}_{device}.json"),
                  "w") as f:
            json.dump({str(seed): results["seeds"][seed]}, f, indent=2,
                      default=float)
        with open(results_file, "w") as f:      # crash-safe per seed
            json.dump(results, f, indent=2, default=float)
        verdict = "PASS" if he["fine_gap"] > NOISE_FLOOR else "FAIL"
        print(f"[{tag}] seed={seed} ({dt:.1f}s, "
              f"{n_steps / dt:.0f} steps/s"
              + (f", peak {peak:.2f} GB" if peak else "") + "): TRAIN gap="
              f"{rep['separability']['gap']:.4f} | HELD-OUT fine gap="
              f"{he['fine_gap']:.4f} vs floor {NOISE_FLOOR} -> {verdict}; "
              f"disc={he['room_discrimination']:.3f} "
              f"sp-heldout={he['room_discrimination_speaker_heldout']:.3f} "
              f"venue-disc={vd:.3f}", flush=True)

    gaps = {s: results["seeds"][s]["heldout_eval"]["fine_gap"] for s in seeds}
    n_pass = sum(g > NOISE_FLOOR for g in gaps.values())
    gap_str = ", ".join(f"seed{s}={gaps[s]:.4f}" for s in seeds)
    all_pass = n_pass == len(seeds)
    mean_gap = float(np.mean(list(gaps.values())))
    print(f"[{tag}] VERDICT: held-out fine gaps [{gap_str}] mean={mean_gap:.4f} "
          f"-> {n_pass}/{len(seeds)} seeds > {NOISE_FLOOR}: "
          f"{'PASS — generalizes' if all_pass else 'FAIL — floor not cleared 3/3'}",
          flush=True)
    results["verdict"] = {
        "n_pass": n_pass, "n_seeds": len(seeds),
        "mean_fine_gap": mean_gap,
        "all_pass": all_pass,
    }
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
