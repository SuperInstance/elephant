"""scripts/condition_eval.py — requirement (b): condition-level evaluation.

The deadband lesson (gate 2): signal lives at CONDITION level, not message
grain — the shipped per-message deadband fired 0/50 even at real
transitions. This script evaluates the trained contrast heads'
displacements on sub-room / condition segmentations, and RE-DERIVES the
deadband at the condition-estimator scale (message-level bootstrap SE of
μ̂, deadband = 2·max SE), rather than reusing the message-grain band.

Conditions (all text tier; audio tier gets the episode-half analog):
  * nights corpus: SEG1 (warm-earnest) vs SEG2 (cynical-banter) — the fine
    condition edge whose dial-tier analog is 1.229 chord; SEG1 vs TTRPG —
    the coarse condition anchor (dial-tier analog 0.941).
  * trades-night conditions: first half vs second half of each tap evening.
  * open-mic vs regular (tap night 1 room vs the open-mic room).
Nights D / D′ are NOT used (D′ carries the replay-honesty gap; D is
reserved). A/B/C are identical by construction — SEG1/SEG2 are their
conditions, used once.

Windows within a condition: W=4, stride 1 (the trajectory scale); the SE is
a bootstrap over MESSAGES (resample -> re-window -> refit), so overlap does
not fake precision.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys

ELEPHANT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ELEPHANT)

import numpy as np

from elephant import contrast
from elephant.contrast import condition_edge, vmf_fit_generic
from elephant.learned import TextEncoder, Vocab, room_from_file, tokenize
from elephant.room import Message

AI = "/home/eileen/projects/ai-writings"
CKPT = os.path.join(ELEPHANT, "checkpoints")
OUT = os.path.join(CKPT, "contrast")
W_COND = 4
B = 200
SEEDS = (0, 1, 2)


def _load_nights():
    spec = importlib.util.spec_from_file_location(
        "nights_abc", os.path.join(ELEPHANT, "scripts", "nights_abc.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["nights_abc"] = m
    spec.loader.exec_module(m)
    ttrpg = m.THEMES["ttrpg"].room_tone + m.TTRPG_EXTENSION
    def to_msgs(seq):
        return [Message(author=a, text=t, ts=float(i)) for i, (a, t, *_)
                in enumerate(seq)]
    return to_msgs(m.SEG1), to_msgs(m.SEG2), to_msgs(ttrpg)


def embed_windows(model, vocab, msgs, w=W_COND, stride=1):
    """Window embeddings + a message-level bootstrap refit closure."""
    def windows_of(messages):
        out = []
        for i in range(0, max(1, len(messages) - w + 1), stride):
            chunk = messages[i: i + w]
            if not chunk:
                continue
            toks = tokenize(" ".join(m.text for m in chunk))
            if len(toks) < 5:
                continue
            ids = vocab.encode(toks, max_len=256)
            out.append(ids)
        return out

    def embed(ids_list):
        import torch
        L = 256
        X = torch.zeros((len(ids_list), L), dtype=torch.long)
        for i, x in enumerate(ids_list):
            X[i, : len(x)] = torch.tensor(x)
        with torch.no_grad():
            z = torch.nn.functional.normalize(model(X), dim=-1)
        return z.numpy()

    base_ids = windows_of(msgs)
    Z = embed(base_ids)

    def fit_of(messages, seed=0):
        ids = windows_of(messages)
        if len(ids) < 3:
            return None
        return vmf_fit_generic(embed(ids), seed=seed, B=0)

    fit = vmf_fit_generic(Z, seed=0, B=0)
    # message-level bootstrap SE
    rng = np.random.default_rng(0)
    mus = []
    n = len(msgs)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        res = [msgs[i] for i in idx]
        f = fit_of(res)
        if f:
            mus.append(np.array(f["mu_hat"]))
    if mus:
        M = np.stack(mus)
        fit["mu_se"] = float(np.mean(np.linalg.norm(M - np.array(fit["mu_hat"]),
                                                    axis=1)))
    return fit


def main() -> int:
    import torch

    vocab = Vocab.load(os.path.join(CKPT, "learned_vocab.txt"))
    seg1, seg2, ttrpg = _load_nights()

    evenings = {
        "tap-1": os.path.join(AI, "tap-trades", "2026-08-16",
                              "evening-at-the-tap.md"),
        "tap-2": os.path.join(AI, "tap-trades", "2026-08-16",
                              "evening-2-open-question-night.md"),
        "tap-3": os.path.join(AI, "tap-trades", "2026-08-16",
                              "evening-3-adaptation-night.md"),
    }
    openmic_files = [os.path.join(AI, "radio-theater", "tap-open-mic-night.md")] + \
        sorted(glob.glob(os.path.join(AI, "tap-trades", "open-mic",
                                      "2026-08-16", "*.md")))

    results = {}
    for seed in SEEDS:
        model = TextEncoder(len(vocab), d_model=64, d_trunk=64)
        sd = torch.load(os.path.join(OUT, f"text_contrast_seed{seed}.pt"),
                        map_location="cpu")
        model.load_state_dict(sd)
        model.eval()

        pairs = {}
        f1 = embed_windows(model, vocab, seg1)
        f2 = embed_windows(model, vocab, seg2)
        ft = embed_windows(model, vocab, ttrpg)
        pairs["nights: SEG1->SEG2 (fine condition)"] = condition_edge(f1, f2)
        pairs["nights: SEG1->TTRPG (coarse condition)"] = condition_edge(f1, ft)

        for name, path in evenings.items():
            room = room_from_file(path, name)
            msgs = room.messages
            h = len(msgs) // 2
            fa = embed_windows(model, vocab, msgs[:h])
            fb = embed_windows(model, vocab, msgs[h:])
            pairs[f"{name}: first-half -> second-half"] = condition_edge(fa, fb)

        om = [m for f in openmic_files if os.path.exists(f)
              for m in room_from_file(f, "openmic").messages]
        tap1 = room_from_file(evenings["tap-1"], "tap1").messages
        pairs["tap-1 room -> open-mic room"] = condition_edge(
            embed_windows(model, vocab, tap1),
            embed_windows(model, vocab, om))

        results[seed] = pairs
        fired = sum(1 for v in pairs.values() if v and v["real"])
        print(f"[cond] seed={seed}: {fired}/{len(pairs)} condition edges "
              f"clear the re-derived deadband")
        for k, v in pairs.items():
            if v:
                print(f"    {k:42s} d_mu={v['d_mu']:.3f} db={v['deadband']:.3f} "
                      f"real={v['real']}")

    # ---- audio tier: episode-half conditions (slug-ordered clips) ------- #
    try:
        from elephant.contrast import parse_script_blocks
        audio_results = {}
        for seed in SEEDS:
            ck = torch.load(os.path.join(OUT, f"audio_contrast_seed{seed}.pt"),
                            map_location="cpu")
            emb = np.load(os.path.join(OUT, f"audio_emb_seed{seed}.npy")) \
                if os.path.exists(os.path.join(OUT, f"audio_emb_seed{seed}.npy")) \
                else None
            if emb is None:
                continue
            pairs_a = {}
            meta = json.load(open(os.path.join(OUT, "audio_clip_meta.json")))
            for ep in ("tap-1", "tap-2", "tap-3", "tap-4"):
                script = os.path.join(
                    AI, "tap-trades", "radio-theater",
                    f"episode-{ep.split('-')[1]}", "SCRIPT.md")
                blocks = parse_script_blocks(script)
                order = {slug: i for i, (slug, _, _) in enumerate(blocks)}
                idx = [i for i, m in enumerate(meta)
                       if m["room"] == ep and m["key"] in order]
                idx.sort(key=lambda i: order[meta[i]["key"]])
                if len(idx) < 8:
                    continue
                h = len(idx) // 2
                fa = vmf_fit_generic(emb[idx[:h]], seed=0, B=B)
                fb = vmf_fit_generic(emb[idx[h:]], seed=0, B=B)
                pairs_a[f"{ep}: first-half -> second-half"] = condition_edge(fa, fb)
            audio_results[seed] = pairs_a
            for k, v in pairs_a.items():
                if v:
                    print(f"    [audio s{seed}] {k:38s} d_mu={v['d_mu']:.3f} "
                          f"db={v['deadband']:.3f} real={v['real']}")
        if audio_results:
            results["audio"] = audio_results
    except FileNotFoundError as e:
        print(f"[cond] audio tier skipped: {e}")

    with open(os.path.join(OUT, "condition_eval.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("[cond] wrote condition_eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
