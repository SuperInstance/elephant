"""scripts/contrast_train_text.py — the v3 contrast head on the learned.py trunk.

Gate-3 build (elephant-sense-v3 §2.3 + devils-advocate §3, text tier):
  * trunk = the v2-era learned encoder (checkpoints/learned_dials.pt
    TextEncoder), LOADED, not rewritten; vocab = learned_vocab.txt, frozen.
  * rooms = the v3 §1.1 text rooms: tap nights 1-4 (evening + broadcast
    script clips per night), compass-head episodes, the open-mic room, the
    named venues, fleet-radio series, the speeches room, the wesley-stream
    text-only room.
  * clips = NON-OVERLAPPING windows (W=8, stride=8) so kNN metrics are not
    inflated by shared content; speaker key = plurality author.
  * objective = multi-positive InfoNCE (τ=0.15, anchor=clip, batch = all
    clips of 2-3 rooms) + within-room spread hinge vs the FROZEN baseline
    spread (anti-collapse, targets precomputed and frozen).
  * 3 seeds (0/1/2) — the registered consecutive-run structure.

Outputs checkpoints/contrast/text_contrast_seed{k}.pt (+ trunk metrics json).
Evaluation of the trained heads lives in scripts/contrast_eval.py.

Room-heldout mode (2026-08-19, the honest test): `--holdout room1,room2`
EXCLUDES those rooms from the training corpus / clips / room_names / spread
targets entirely, trains on the rest, then evaluates fine gap +
room-discrimination + speaker-heldout on the EXCLUDED clips (re-embedded
with the trained model — data the model never trained on). Heldout
checkpoints/results go to text_contrast_heldout_seed{k}.pt /
text_contrast_heldout_results.json so the registered (full-corpus)
artifacts and the frozen baseline are never touched. Without --holdout the
behavior is identical to the registered training run.
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys

# CPU-only hardening (2026-08-19): the box has a GPU-pinning bug that crashed
# three prior dispatches; this run is registered CPU-only. Must be set before
# the first torch import (elephant.contrast imports torch at module level).
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elephant import contrast
from elephant.contrast import (
    Clip, contrast_loss, probe_report, sample_room_batches, spread_hinge,
    text_clips_from_room,
)
from elephant.learned import TextEncoder, Vocab, load_learned_bank  # noqa: F401
from elephant.learned import room_from_file

AI = "/home/eileen/projects/ai-writings"
CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "checkpoints")
OUT = os.path.join(CKPT, "contrast")

WINDOW = 8
SEEDS = (0, 1, 2)
EPOCHS = 200          # batches per epoch-schedule unit (see below)
N_BATCHES = 60        # contrast batches per epoch
LR = 1e-4
MAX_LEN = 256


def parse_args(argv):
    """[--seeds 0,1,2] [--holdout room1,room2]; seeds default SEEDS."""
    seeds = SEEDS
    holdout = []
    i = 1
    while i < len(argv):
        if argv[i] == "--seeds":
            seeds = tuple(int(s) for s in argv[i + 1].split(","))
        elif argv[i] == "--holdout":
            if i + 1 >= len(argv):
                raise SystemExit("--holdout needs a comma-separated room list")
            holdout = [s.strip() for s in argv[i + 1].split(",") if s.strip()]
        else:
            raise SystemExit("usage: contrast_train_text.py "
                             "[--seeds 0,1,2] [--holdout room1,room2]")
        i += 2
    return seeds, holdout


def build_text_corpus() -> list:
    """The v3 §1.1 text rooms as (room_name, messages, meta)."""
    from elephant.learned import parse_document
    from elephant.room import Message

    rooms = []

    def add_room(name, files):
        msgs = []
        for f in files:
            if os.path.exists(f):
                msgs.extend(room_from_file(f, name).messages)
        if msgs:
            rooms.append((name, msgs))

    d = os.path.join(AI, "tap-trades", "2026-08-16")
    rt = os.path.join(AI, "tap-trades", "radio-theater")
    # the killer control: 4 nights, same cast. Each night = evening doc +
    # its broadcast episode script (same room family, same night).
    add_room("tap-night-1", [os.path.join(d, "evening-at-the-tap.md"),
                             os.path.join(rt, "episode-1", "SCRIPT.md")])
    add_room("tap-night-2", [os.path.join(d, "evening-2-open-question-night.md"),
                             os.path.join(rt, "episode-2", "SCRIPT.md")])
    add_room("tap-night-3", [os.path.join(d, "evening-3-adaptation-night.md"),
                             os.path.join(rt, "episode-3", "SCRIPT.md")]
             + sorted(glob.glob(os.path.join(d, "adaptations", "*.md"))))
    # night 4 has no evening doc — it lives as the broadcast episode + the
    # lenses pieces (the same filing learned.py's tap_corpus uses).
    add_room("tap-night-4", [os.path.join(rt, "episode-4", "SCRIPT.md")]
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


def main() -> int:
    import torch

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    rng = random.Random(0)

    seeds, holdout_rooms = parse_args(sys.argv)
    tag = "text-heldout" if holdout_rooms else "text"

    vocab = Vocab.load(os.path.join(CKPT, "learned_vocab.txt"))

    # ---- corpus & clips ------------------------------------------------ #
    rooms = build_text_corpus()
    known = {name for name, _msgs, *_ in rooms}
    missing = [r for r in holdout_rooms if r not in known]
    if missing:
        raise SystemExit(f"--holdout unknown rooms {missing}; "
                         f"known rooms: {sorted(known)}")

    def clips_for(room_list):
        cs, ts = [], []
        for name, msgs, *_ in room_list:
            for clip, toks in text_clips_from_room(name, msgs, window=WINDOW):
                cs.append(clip)
                ts.append(toks)
        return cs, ts

    train_rooms = [(n, m) for n, m, *_ in rooms if n not in holdout_rooms]
    held_rooms = [(n, m) for n, m, *_ in rooms if n in holdout_rooms]

    if holdout_rooms:
        pre_clips, _ = clips_for(rooms)      # count-only before/after check
        print(f"[{tag}] pre-holdout corpus: rooms={len(rooms)} "
              f"clips={len(pre_clips)}")

    clips, tokens_all = clips_for(train_rooms)
    room_names = [c.room for c in clips]
    print(f"[{tag}] train rooms={len(train_rooms)} clips={len(clips)}"
          + (f"  (holdout EXCLUDED: {','.join(holdout_rooms)})" if holdout_rooms else ""))

    ho_clips: list = []
    ho_tokens: list = []
    if holdout_rooms:
        ho_clips, ho_tokens = clips_for(held_rooms)
        ho_found = sorted({c.room for c in ho_clips})
        if not ho_clips or set(ho_found) != set(holdout_rooms):
            raise SystemExit("holdout produced no clips for some rooms: "
                             f"asked={holdout_rooms} got={ho_found}")
        print(f"[{tag}] heldout (NEVER trained on): rooms={ho_found} "
              f"clips={len(ho_clips)}")

    def encode_tokens(encoder, tokens):
        ids = [vocab.encode(t, max_len=MAX_LEN) for t in tokens]
        L = MAX_LEN
        X = torch.zeros((len(ids), L), dtype=torch.long)
        for i, x in enumerate(ids):
            X[i, : len(x)] = torch.tensor(x)
        with torch.no_grad():
            z = torch.nn.functional.normalize(encoder(X), dim=-1)
        return z.numpy()

    def encode_all(encoder):
        return encode_tokens(encoder, tokens_all)

    # ---- frozen baseline (the v2 trunk, untouched) ---------------------- #
    base = TextEncoder(len(vocab), d_model=64, d_trunk=64)
    sd = torch.load(os.path.join(CKPT, "learned_dials.pt"), map_location="cpu")
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items()
              if k.startswith("encoder.")}
    base.load_state_dict(enc_sd)
    z0 = encode_all(base)
    base_report = probe_report(z0, clips)
    base_spread = contrast.room_spread(z0, room_names)
    print(f"[text] FROZEN baseline: fine gap={base_report['separability']['gap']:.4f} "
          f"disc={base_report['room_discrimination']:.3f} "
          f"heldout={base_report['room_discrimination_speaker_heldout']:.3f} "
          f"mean_spread={base_report['mean_spread']:.3f}")
    # frozen baseline: committed at 77b8aa4 — NEVER regenerated/overwritten
    fb = os.path.join(OUT, "text_frozen_baseline.json")
    if os.path.exists(fb):
        committed = json.load(open(fb))
        gap_c = committed["separability"]["gap"]
        gap_n = base_report["separability"]["gap"]
        print(f"[text] frozen baseline COMMITTED (kept): gap={gap_c:.4f}; "
              f"recomputed check gap={gap_n:.4f} "
              f"(delta {abs(gap_n - gap_c):.2e})")
    else:
        if not holdout_rooms:   # NEVER write the frozen baseline in
            with open(fb, "w") as f:   # room-heldout mode
                json.dump(base_report, f, indent=2, default=float)

    # trunk (frozen v2) reference numbers on the held-out clips — the
    # honest anchor: what the UNTRAINED head does on unseen nights
    if holdout_rooms:
        ho_rooms = [c.room for c in ho_clips]
        ho_speakers = [c.speaker for c in ho_clips]
        z0h = encode_tokens(base, ho_tokens)
        t_fine = contrast.separability(z0h, ho_rooms)
        t_disc = contrast.room_discrimination(z0h, ho_rooms, ho_speakers)
        t_disc_sp = contrast.room_discrimination(
            z0h, ho_rooms, ho_speakers, holdout_speaker=True)
        print(f"[{tag}] trunk(frozen v2) on heldout clips: fine gap="
              f"{t_fine['gap']:.4f} disc={t_disc:.3f} "
              f"speaker-heldout={t_disc_sp:.3f}")

    # ---- ids tensor once ------------------------------------------------ #
    ids_list = [vocab.encode(t, max_len=MAX_LEN) for t in tokens_all]
    Xt = torch.zeros((len(ids_list), MAX_LEN), dtype=torch.long)
    for i, x in enumerate(ids_list):
        Xt[i, : len(x)] = torch.tensor(x)

    results = {}
    results_file = ("text_contrast_heldout_results.json" if holdout_rooms
                    else "text_contrast_results.json")
    for seed in seeds:
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
                z = model(Xt[idx])
                loss = contrast_loss(z, [room_names[i] for i in idx],
                                     tau=contrast.TAU)
                loss = loss + spread_hinge(z, [room_names[i] for i in idx],
                                           base_spread)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        model.eval()
        z1 = encode_all(model)
        rep = probe_report(z1, clips)
        post_spread = contrast.room_spread(z1, room_names)
        results[seed] = {
            "report": rep,
            "spread_preservation": {
                r: float(post_spread.get(r, float("nan")) / base_spread[r])
                for r in base_spread},
            "final_batch_loss": float(loss.item()),
        }
        torch.save(model.state_dict(),
                   os.path.join(OUT,
                                f"text_contrast_heldout_seed{seed}.pt"
                                if holdout_rooms
                                else f"text_contrast_seed{seed}.pt"))
        # crash-safe: flush per-seed results as each seed lands
        with open(os.path.join(OUT, results_file), "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"[{tag}] seed={seed} TRAIN-corpus: fine gap={rep['separability']['gap']:.4f} "
              f"disc={rep['room_discrimination']:.3f} "
              f"heldout={rep['room_discrimination_speaker_heldout']:.3f} "
              f"mean_spread={rep['mean_spread']:.3f}")
        if holdout_rooms:
            zh = encode_tokens(model, ho_tokens)
            hf = contrast.separability(zh, ho_rooms)
            hd = contrast.room_discrimination(zh, ho_rooms, ho_speakers)
            hsp = contrast.room_discrimination(
                zh, ho_rooms, ho_speakers, holdout_speaker=True)
            results[seed]["heldout_eval"] = {
                "holdout_rooms": sorted(set(ho_rooms)),
                "n_clips": len(ho_clips),
                "fine_gap": hf["gap"],
                "same_room_mean": hf["same_room_mean"],
                "cross_room_mean": hf["cross_room_mean"],
                "n_same": hf["n_same"], "n_cross": hf["n_cross"],
                "room_discrimination": hd,
                "room_discrimination_speaker_heldout": hsp,
                "noise_floor": 0.05,
                "fine_gap_beats_noise_floor": bool(hf["gap"] > 0.05),
            }
            with open(os.path.join(OUT, results_file), "w") as f:
                json.dump(results, f, indent=2, default=float)
            verdict = "PASS" if hf["gap"] > 0.05 else "FAIL"
            print(f"[{tag}] seed={seed} HELD-OUT (unseen, n={len(ho_clips)}): "
                  f"fine gap={hf['gap']:.4f} vs noise floor 0.05 → {verdict}; "
                  f"disc={hd:.3f} speaker-heldout={hsp:.3f}")
    with open(os.path.join(OUT, results_file), "w") as f:
        json.dump(results, f, indent=2, default=float)
    if holdout_rooms:
        gaps = {s: results[s]["heldout_eval"]["fine_gap"] for s in seeds}
        all_pass = all(g > 0.05 for g in gaps.values())
        gap_str = ", ".join(f"seed{s}={gaps[s]:.4f}" for s in seeds)
        if all_pass:
            print(f"[{tag}] HELD-OUT VERDICT: fine gaps [{gap_str}] — "
                  "ALL > 0.05 noise floor: room identity IS recoverable "
                  "for unseen nights of the same cast")
        else:
            print(f"[{tag}] HELD-OUT VERDICT: fine gaps [{gap_str}] — "
                  "NOT all > 0.05: held-out claim FAILS (room identity not "
                  "demonstrated on unseen nights)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
