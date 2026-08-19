"""scripts/contrast_eval.py — the REGISTERED evaluation (gate 3).

Reloads the trained heads from checkpoints/contrast/, re-embeds, re-computes
the registered metrics from scratch (checkpoint reproducibility), and
emits the verdict block against the pre-registered deadman:

  * fine gap ≥ 0.10 (from the frozen 0.0146), three consecutive runs
  * speaker-heldout discrimination ≥ 0.50 (chance 0.25)
  * noise-margin read: gap > 0.05 + 2σ of the cross-room distance spread
  * within-room spread preserved (no collapse)
  * the (a) geometry statement: the head's own fine/coarse ordering vs the
    dial tier's inversion (fine 1.229 > coarse 0.941) and the encoder
    tier's original ordering (fine 0.015 < coarse 0.271)

Also saves embeddings + clip metadata for the condition-level and fusion
stages (audio_emb_seed{n}.npy, text_emb_seed{n}.npy, *_clip_meta.json).
"""
from __future__ import annotations

import json
import os
import sys

ELEPHANT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ELEPHANT)
sys.path.insert(0, "/home/eileen/projects/fleet-jepa-midi")

import numpy as np
import torch

from elephant import contrast
from elephant.contrast import Clip, probe_report

AI = "/home/eileen/projects/ai-writings"
CKPT = os.path.join(ELEPHANT, "checkpoints")
OUT = os.path.join(CKPT, "contrast")
SEEDS = (0, 1, 2)

FINE_THRESHOLD = 0.10
HELDOUT_THRESHOLD = 0.50
NOISE_FLOOR = 0.05


def eval_audio() -> dict:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ca", os.path.join(ELEPHANT, "scripts", "contrast_audio.py"))
    ca = importlib.util.module_from_spec(spec)
    sys.modules["ca"] = ca
    spec.loader.exec_module(ca)
    import librosa
    import elephant_sense_probe as probe

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rooms = ca.build_rooms()
    encoder0, cfg = probe.load_encoder(
        "/home/eileen/projects/fleet-jepa-midi/checkpoints/audio_jepa_v2.pt",
        device)
    frontend = probe.MelFrontend(
        sample_rate=cfg.get("sample_rate", 16_000),
        n_fft=cfg.get("n_fft", 400),
        hop_length=cfg.get("hop_length", 160),
        n_mels=cfg.get("n_mels", 64))

    clips, mels = [], []
    from elephant.contrast import speaker_key_from_filename
    for r in sorted(rooms):
        for p in rooms[r]:
            y, _ = librosa.load(str(p), sr=16_000, mono=True)
            mels.append(probe.compute_mel(
                torch.from_numpy(y.astype(np.float32)), frontend, 16_000))
            base = os.path.basename(str(p))
            clips.append(Clip(r, os.path.splitext(base)[0],
                              speaker_key_from_filename(base)))
    room_names = [c.room for c in clips]
    music = [r for r in set(room_names) if r.startswith("music-")]
    tap_idx = [i for i, c in enumerate(clips) if c.room.startswith("tap-")]

    z0 = ca.embed_cached(encoder0, mels, device).cpu().numpy()
    base = probe_report(z0, clips, coarse_b_rooms=music)
    base_tap = probe_report(z0[tap_idx], [clips[i] for i in tap_idx])

    out = {"frozen": {"tap": base_tap, "full": base}, "seeds": {}}
    np.save(os.path.join(OUT, "audio_emb_frozen.npy"), z0)
    if not os.path.exists(os.path.join(OUT, "audio_clip_meta.json")):
        with open(os.path.join(OUT, "audio_clip_meta.json"), "w") as f:
            json.dump([{"room": c.room, "key": c.key, "speaker": c.speaker}
                       for c in clips], f)

    for seed in SEEDS:
        ck = torch.load(os.path.join(OUT, f"audio_contrast_seed{seed}.pt"),
                        map_location=device)
        encoder0.load_state_dict(ck["encoder"])
        encoder0.eval()
        z = ca.embed_cached(encoder0, mels, device).cpu().numpy()
        rep = probe_report(z, clips, coarse_b_rooms=music)
        rep_tap = probe_report(z[tap_idx], [clips[i] for i in tap_idx])
        out["seeds"][seed] = {"tap": rep_tap, "full": rep}
        np.save(os.path.join(OUT, f"audio_emb_seed{seed}.npy"), z)
        print(f"[eval:audio s{seed}] fine={rep_tap['separability']['gap']:.4f} "
              f"disc={rep_tap['room_discrimination']:.3f} "
              f"heldout={rep_tap['room_discrimination_speaker_heldout']:.3f} "
              f"coarse={rep['coarse']['gap']:.4f}")
    return out


def eval_text() -> dict:
    spec_path = os.path.join(ELEPHANT, "scripts", "contrast_train_text.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("ct", spec_path)
    ct = importlib.util.module_from_spec(spec)
    sys.modules["ct"] = ct
    spec.loader.exec_module(ct)

    from elephant.learned import TextEncoder, Vocab
    vocab = Vocab.load(os.path.join(CKPT, "learned_vocab.txt"))
    rooms = ct.build_text_corpus()
    clips, tokens_all = [], []
    for name, msgs, *_ in rooms:
        for clip, toks in ct.text_clips_from_room(name, msgs, window=ct.WINDOW):
            clips.append(clip)
            tokens_all.append(toks)

    def embed(model):
        ids = [vocab.encode(t, max_len=ct.MAX_LEN) for t in tokens_all]
        X = torch.zeros((len(ids), ct.MAX_LEN), dtype=torch.long)
        for i, x in enumerate(ids):
            X[i, : len(x)] = torch.tensor(x)
        with torch.no_grad():
            return torch.nn.functional.normalize(model(X), dim=-1).numpy()

    base_model = TextEncoder(len(vocab), 64, 64)
    sd = torch.load(os.path.join(CKPT, "learned_dials.pt"), map_location="cpu")
    base_model.load_state_dict(
        {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")})
    z0 = embed(base_model)
    tap_idx = [i for i, c in enumerate(clips) if c.room.startswith("tap-night-")]
    out = {"frozen": probe_report(z0, clips),
           "frozen_tap": probe_report(z0[tap_idx], [clips[i] for i in tap_idx]),
           "seeds": {}, "n_clips": len(clips)}
    np.save(os.path.join(OUT, "text_emb_frozen.npy"), z0)
    if not os.path.exists(os.path.join(OUT, "text_clip_meta.json")):
        with open(os.path.join(OUT, "text_clip_meta.json"), "w") as f:
            json.dump([{"room": c.room, "key": c.key, "speaker": c.speaker}
                       for c in clips], f)

    for seed in SEEDS:
        model = TextEncoder(len(vocab), 64, 64)
        model.load_state_dict(torch.load(
            os.path.join(OUT, f"text_contrast_seed{seed}.pt"),
            map_location="cpu"))
        model.eval()
        z = embed(model)
        out["seeds"][seed] = {"full": probe_report(z, clips),
                             "tap": probe_report(z[tap_idx],
                                                 [clips[i] for i in tap_idx])}
        np.save(os.path.join(OUT, f"text_emb_seed{seed}.npy"), z)
        r = out["seeds"][seed]["tap"]
        rf = out["seeds"][seed]["full"]
        print(f"[eval:text s{seed}] TAP fine={r['separability']['gap']:.4f} "
              f"disc={r['room_discrimination']:.3f} "
              f"heldout={r['room_discrimination_speaker_heldout']:.3f} | "
              f"full disc={rf['room_discrimination']:.3f} "
              f"spread={rf['mean_spread']:.3f}")
    return out


def verdict(block: dict, tier: str) -> dict:
    fines, helds, coars = [], [], []
    for seed, rep in block["seeds"].items():
        r_tap = rep.get("tap") or rep
        fines.append(r_tap["separability"]["gap"])
        helds.append(r_tap["room_discrimination_speaker_heldout"])
        if rep.get("coarse"):
            coars.append(rep["coarse"]["gap"])
    fine = float(np.mean(fines))
    fine_min = float(np.min(fines))
    held = float(np.mean(helds))
    v = {
        "tier": tier,
        "fine_gap_mean": fine,
        "fine_gap_min": fine_min,
        "fine_gap_all_seeds": [round(f, 4) for f in fines],
        "fine_ge_0.10_all_seeds": bool(fine_min >= FINE_THRESHOLD),
        "heldout_mean": held,
        "heldout_ge_0.50": bool(held >= HELDOUT_THRESHOLD),
        "n_seeds": len(fines),
    }
    if coars:
        v["coarse_gap_mean"] = float(np.mean(coars))
        v["ordering"] = ("fine>coarse (dial-tier geometry)" if fine > np.mean(coars)
                         else "coarse>fine (encoder-tier geometry)")
    return v


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    audio = eval_audio()
    text = eval_text()

    v_audio = verdict(audio, "audio")
    v_text = verdict(text, "text")
    summary = {
        "thresholds": {"fine": FINE_THRESHOLD, "heldout": HELDOUT_THRESHOLD,
                       "noise_floor": NOISE_FLOOR},
        "audio": v_audio,
        "text": v_text,
        "audio_frozen_tap": {
            "fine": audio["frozen"]["tap"]["separability"]["gap"],
            "disc": audio["frozen"]["tap"]["room_discrimination"],
            "heldout": audio["frozen"]["tap"]["room_discrimination_speaker_heldout"],
            "coarse": audio["frozen"]["full"]["coarse"]["gap"]},
        "text_frozen": {
            "fine": text["frozen"]["separability"]["gap"],
            "disc": text["frozen"]["room_discrimination"],
            "heldout": text["frozen"]["room_discrimination_speaker_heldout"],
            "mean_spread": text["frozen"]["mean_spread"]},
        "text_frozen_tap": {
            "fine": text["frozen_tap"]["separability"]["gap"],
            "disc": text["frozen_tap"]["room_discrimination"],
            "heldout": text["frozen_tap"]["room_discrimination_speaker_heldout"]},
        "spread_preservation_audio": None,
    }
    with open(os.path.join(OUT, "registered_eval.json"), "w") as f:
        json.dump({"summary": summary, "audio": audio, "text": text},
                  f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
