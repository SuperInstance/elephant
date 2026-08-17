"""elephant.learned — the learned dials (ROADMAP v1).

The dials in `dials/` are hand-crafted v0 senses. This module grows the
learned side: a small text encoder plus **one regression head per dial**,
trained by *supervised distillation* — the v0 hand-crafted dial is the
teacher, the learned model is the student that must reproduce the teacher's
reading **from raw room text alone**.

That is the honest v1 contract: *learn to feel what the hand-crafted dials
feel, but from the text, not the recipe.* Once a student can do that on rooms
it has never seen, it has earned a dial of its own — and the dials that fail
to transfer are the seams the elephant still can't feel (see
`docs/dial-training-v1.md`).

The JEPA shape from `jepa.py`'s promise is implemented too, as an **optional
self-supervised pretraining stage**: an EMA target encoder + stop-gradient +
cosine predictor + VICReg variance/covariance, run over *unlabeled* room text
(window `t` predicts window `t+1`) before the supervised heads are fitted.
Even a few epochs counts; the question is whether pretraining helps the
held-out transfer.

Everything is kept deliberately small — word-level whitespace tokens, a few
thousand vocab, a 64-dim encoder, tiny batches — so a full train+pretrain+eval
runs in minutes, not hours.
"""
from __future__ import annotations

import glob
import math
import os
import random
import re
from dataclasses import dataclass, field as dc_field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .dial import Dial
from .field import DIAL_NAMES
from .room import Message, Room

# --------------------------------------------------------------------- #
# torch is optional for the package; learned dials need it               #
# --------------------------------------------------------------------- #
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH = True
except Exception:  # pragma: no cover - exercised only without torch installed
    torch = nn = F = None  # type: ignore
    _TORCH = False

# The v0 hand-crafted dials are the teachers. Imported lazily so the module
# (and the rest of the package) imports cleanly without torch/dials present.
_TEACHERS = None


def _teachers() -> Dict[str, Dial]:
    global _TEACHERS
    if _TEACHERS is None:
        from .dials import DEFAULT_DIALS

        _TEACHERS = {d.name: d for d in DEFAULT_DIALS}
    return _TEACHERS


# Each dial's native output range (the v0 convention the student must match).
# Dial names are looked up with a fallback so the module keeps working as the
# fleet grows new dials (e.g. the 8th, `model_vs_code`); unknown dials default
# to a [0,1] reading.
DIAL_RANGES: Dict[str, Tuple[float, float]] = {
    "mood": (-1.0, 1.0),
    "volume": (0.0, 1.0),
    "earnestness": (0.0, 1.0),
    "cynicism": (0.0, 1.0),
    "joke_landing": (-1.0, 1.0),
    "panic": (0.0, 1.0),
    "presence": (0.0, 1.0),
    "model_vs_code": (-1.0, 1.0),
}


def dial_range(name: str) -> Tuple[float, float]:
    return DIAL_RANGES.get(name, (0.0, 1.0))

# --------------------------------------------------------------------- #
# Tokenization — word-level, whitespace, tiny vocab                      #
# --------------------------------------------------------------------- #
_WS_RE = re.compile(r"\S+")


def tokenize(text: str) -> List[str]:
    """Word-level whitespace tokens, lowercased, punctuation attached.

    Punctuation stays glued to its word (`great.` != `great`) because the
    v0 cynicism dial literally keys on `great.` and the volume/panic dials
    key on `!`/`?`/caps. We want the student to *see* the same surface the
    teacher reads.
    """
    return _WS_RE.findall(text.lower())


class Vocab:
    """A small whitespace-token vocab built from training text only."""

    UNK = 1
    PAD = 0

    def __init__(self, tokens: Optional[Iterable[str]] = None, max_size: int = 4000):
        self.max_size = max_size
        self.itos: List[str] = ["<pad>", "<unk>"]
        self.stoi: Dict[str, int] = {"<pad>": self.PAD, "<unk>": self.UNK}
        if tokens is not None:
            self.fit(tokens)

    def fit(self, tokens: Iterable[str]) -> "Vocab":
        counts: Dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        for t, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if len(self.itos) >= self.max_size:
                break
            if t not in self.stoi:
                self.stoi[t] = len(self.itos)
                self.itos.append(t)
        return self

    def encode(self, tokens: Sequence[str], max_len: int = 256) -> List[int]:
        ids = [self.stoi.get(t, self.UNK) for t in tokens[:max_len]]
        return ids

    def __len__(self) -> int:
        return len(self.itos)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write("\n".join(self.itos))

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path) as f:
            itos = [l.rstrip("\n") for l in f]
        v = cls()
        v.itos = itos
        v.stoi = {t: i for i, t in enumerate(itos)}
        return v


# --------------------------------------------------------------------- #
# Markdown -> Room parser                                               #
# --------------------------------------------------------------------- #
_SPEAKER_RE = re.compile(r"^\*\*(.+?)\*\*\s*:?\s*(.*)$")


def _speaker_name(label: str) -> Optional[str]:
    """Extract the speaker's bare name from a `**LABEL**` chunk.

    Labels look like `LUCINEER (foreman), setting down the round:` — take the
    part before any parenthetical, drop trailing `:`/`,`, collapse whitespace.
    """
    name = label.split("(", 1)[0]
    name = name.rstrip(" :,.").strip()
    if not name or name.upper() in {"ALL", "ALL OF THEM"}:
        return None
    return name.upper()

_EMPH_ONLY_RE = re.compile(r"^\*[^*].*\*$")  # a whole italic stage-direction line


def _clean_utterance(s: str) -> str:
    s = s.replace("**", "").replace("*", "").replace("`", "")
    s = s.replace(">", "").replace("|", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_document(text: str) -> List[Tuple[str, str]]:
    """Turn raw markdown into a list of (author, utterance) tuples.

    Handles the three fleet prose shapes: radio-theater blockquotes
    (`> `speaker-line-id`` + prose), stage scripts (`**SPEAKER (role):** line`
    plus `*(stage direction)*`), and plain long-form (speeches, pieces) where
    every paragraph is an utterance by the title speaker.
    """
    lines = text.split("\n")
    utterances: List[Tuple[str, str]] = []
    author = "room"

    for raw in lines:
        line = raw.rstrip()
        s = line.strip()
        if not s:
            continue
        if s.startswith("<!--"):
            continue
        # Front matter is handled structurally below: headings, italic-only
        # metadata lines, `---`/`===` separators, and table rows are all
        # skipped inline, so we don't need a fragile "skip until ---" pass
        # (many pieces have no `---` at all).
        if s.startswith("---") or s.startswith("==="):
            continue
        if s.startswith("#"):
            continue
        if s.startswith("|"):          # tables (voice-cast etc.)
            continue
        if _EMPH_ONLY_RE.match(s) and not s.startswith("**"):
            continue                   # *italic stage direction on its own line*
        if s.startswith("> "):         # blockquote utterance
            body = s.lstrip(">").strip()
            body = re.sub(r"^`[^`]*`\s*", "", body)  # drop the `line-id`
            body = _clean_utterance(body)
            if body:
                utterances.append((author, body))
            continue
        m = _SPEAKER_RE.match(s)
        if m:
            spk = _speaker_name(m.group(1))
            body = _clean_utterance(m.group(2))
            if spk:
                author = spk
            if body:
                utterances.append((author, body))
            continue
        # plain prose line — part of the current speaker's speech
        body = _clean_utterance(s)
        if body:
            utterances.append((author, body))
    return utterances


def room_from_markdown(text: str, name: str = "room") -> Room:
    """Build a `Room` from raw markdown text.

    Utterances become `Message`s in document order; a real timestamp is faked
    from the index so the pacing dials (density) still have a span to read.
    """
    messages = [
        Message(author=author, text=body, ts=float(i * 30.0))
        for i, (author, body) in enumerate(parse_document(text))
    ]
    return Room(name, messages)


def room_from_file(path: str, name: Optional[str] = None) -> Room:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return room_from_markdown(text, name or os.path.basename(path))


# --------------------------------------------------------------------- #
# Teacher labels                                                        #
# --------------------------------------------------------------------- #
def teacher_readings(room: Room) -> np.ndarray:
    """The v0 hand-crafted readings for a room, in DIAL_NAMES order."""
    teachers = _teachers()
    return np.array([teachers[n].read(room) for n in DIAL_NAMES], dtype=float)


# --------------------------------------------------------------------- #
# Corpus — the Tap nights (same cast, different night) + speeches        #
# --------------------------------------------------------------------- #
def _tap_dir(base: str) -> str:
    return os.path.join(base, "tap-trades")


def tap_corpus(base: str) -> Tuple[List[str], List[str]]:
    """Return (train_files, test_files) for the held-out generalization check.

    **Train = nights 1–2**: the source evenings, their broadcast episodes,
    the six trade pieces, and the night-1/night-2 sequels and questions.
    **Test = nights 3–4**: adaptation night, its broadcast episode, the lens
    night, the adaptations/lenses pieces — plus the long-form speeches corpus
    (a genuinely different kind of room). Test rooms are never seen in
    training, in the vocab, or in the JEPA pretraining.
    """
    tap = _tap_dir(base)
    d = os.path.join(tap, "2026-08-16")
    rt = os.path.join(tap, "radio-theater")

    train = [
        os.path.join(d, "evening-at-the-tap.md"),             # night 1
        os.path.join(d, "evening-2-open-question-night.md"),  # night 2
        os.path.join(rt, "SCRIPT.md"),                        # episode 1 (broadcast)
        os.path.join(rt, "episode-2", "SCRIPT.md"),           # episode 2 (broadcast)
        os.path.join(d, "carpenter.md"),
        os.path.join(d, "composite.md"),
        os.path.join(d, "mason.md"),
        os.path.join(d, "shipwright.md"),
        os.path.join(d, "welder.md"),
        os.path.join(d, "wesley-the-room.md"),
    ]
    train += sorted(glob.glob(os.path.join(d, "sequels", "*.md")))
    train += sorted(glob.glob(os.path.join(d, "sequels-night2", "*.md")))
    train += sorted(glob.glob(os.path.join(d, "questions", "*.md")))

    test = [
        os.path.join(d, "evening-3-adaptation-night.md"),     # night 3
        os.path.join(rt, "episode-3", "SCRIPT.md"),           # episode 3 (broadcast)
        os.path.join(rt, "episode-4", "SCRIPT.md"),           # episode 4 (lens night)
    ]
    test += sorted(glob.glob(os.path.join(d, "adaptations", "*.md")))
    test += sorted(glob.glob(os.path.join(d, "lenses", "*.md")))
    test += sorted(glob.glob(os.path.join(base, "speeches", "*.md")))

    return [f for f in train if os.path.exists(f)], [f for f in test if os.path.exists(f)]


# --------------------------------------------------------------------- #
# Window sampling — many (text, label) samples per room                  #
# --------------------------------------------------------------------- #
@dataclass
class Sample:
    room: str
    tokens: List[str]
    labels: np.ndarray


def room_windows(
    room: Room, window: int = 8, stride: int = 1
) -> List[Sample]:
    """Slide a `window`-message view over a room; each view is a (text, labels)
    sample. A sense reads a *span* of conversation, so a room yields many spans
    — which is what gives the student enough samples to actually learn."""
    msgs = room.messages
    samples: List[Sample] = []
    if len(msgs) == 0:
        return samples
    for i in range(0, max(1, len(msgs) - window + 1), max(1, stride)):
        chunk = msgs[i : i + window]
        sub = Room(f"{room.name}#{i}", list(chunk))
        if sum(len(m.words) for m in chunk) < 5:
            continue
        tokens = tokenize(" ".join(m.text for m in chunk))
        if len(tokens) < 5:
            continue
        samples.append(
            Sample(room=room.name, tokens=tokens, labels=teacher_readings(sub))
        )
    return samples


def build_windows(rooms: Sequence[Room], **kw) -> List[Sample]:
    out: List[Sample] = []
    for r in rooms:
        out.extend(room_windows(r, **kw))
    return out


# --------------------------------------------------------------------- #
# Models                                                                #
# --------------------------------------------------------------------- #
if _TORCH:

    class TextEncoder(nn.Module):
        """Bag-of-words embedding + MLP trunk. Small on purpose."""

        def __init__(self, vocab_size: int, d_model: int = 64, d_trunk: int = 64):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.trunk = nn.Sequential(
                nn.Linear(2 * d_model, d_trunk),
                nn.ReLU(),
                nn.Linear(d_trunk, d_trunk),
            )

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            mask = (ids != 0).unsqueeze(-1).float()  # [B, L, 1]
            e = self.emb(ids) * mask                  # zero the pad rows
            cnt = mask.sum(1).clamp(min=1.0)          # [B, 1]
            mean = e.sum(1) / cnt
            max_ = e.masked_fill(ids.unsqueeze(-1) == 0, float("-inf")).max(1).values
            h = torch.cat([mean, max_], dim=-1)       # [B, 2d]
            return self.trunk(h)

    class LearnedDialModel(nn.Module):
        """Shared text encoder + one head per dial."""

        def __init__(self, vocab_size: int, n_heads: int = 7,
                     d_model: int = 64, d_trunk: int = 64):
            super().__init__()
            self.encoder = TextEncoder(vocab_size, d_model, d_trunk)
            self.heads = nn.ModuleList([nn.Linear(d_trunk, 1) for _ in range(n_heads)])

        def encode(self, ids: torch.Tensor) -> torch.Tensor:
            return self.encoder(ids)

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            h = self.encoder(ids)
            return torch.cat([head(h) for head in self.heads], dim=-1)  # [B, n_heads]

    class JEPAPretrainer(nn.Module):
        """The `jepa.py` promise: EMA target + stop-gradient + cosine predictor.

        Context encoder (student) predicts a future window's embedding through
        a predictor; the target encoder is an exponential moving average of the
        student, detached (stop-gradient); VICReg variance/covariance terms keep
        the representation from collapsing.
        """

        def __init__(self, vocab_size: int, d_model: int = 64, d_trunk: int = 64,
                     ema_tau: float = 0.996):
            super().__init__()
            self.context_encoder = TextEncoder(vocab_size, d_model, d_trunk)
            self.target_encoder = TextEncoder(vocab_size, d_model, d_trunk)
            self.predictor = nn.Sequential(
                nn.Linear(d_trunk, d_trunk),
                nn.ReLU(),
                nn.Linear(d_trunk, d_trunk),
            )
            self.ema_tau = ema_tau
            # target starts as a copy and is never trained by gradients.
            self.target_encoder.load_state_dict(self.context_encoder.state_dict())
            for p in self.target_encoder.parameters():
                p.requires_grad_(False)

        @torch.no_grad()
        def update_target(self) -> None:
            tau = self.ema_tau
            for tp, cp in zip(self.target_encoder.parameters(),
                              self.context_encoder.parameters()):
                tp.data.mul_(tau).add_(cp.data, alpha=1.0 - tau)

        def forward(self, ctx_ids: torch.Tensor, tgt_ids: torch.Tensor):
            z_c = self.context_encoder(ctx_ids)
            z_t = self.target_encoder(tgt_ids).detach()   # stop-gradient
            p = self.predictor(z_c)
            return z_c, z_t, p

    def _vicreg(z: torch.Tensor, var_w: float = 25.0, cov_w: float = 1.0) -> torch.Tensor:
        """VICReg variance + covariance regularizers on a batch of embeddings."""
        z = z - z.mean(0)
        std = torch.sqrt(z.var(0) + 1e-4)
        var_term = F.relu(1.0 - std).mean()
        n, d = z.shape
        if n > 1:
            cov = (z.t() @ z) / (n - 1)
            cov = cov - torch.diag(torch.diag(cov))
            cov_term = (cov ** 2).sum() / d
        else:
            cov_term = torch.zeros((), device=z.device)
        return var_w * var_term + cov_w * cov_term

else:  # pragma: no cover
    TextEncoder = LearnedDialModel = JEPAPretrainer = None  # type: ignore


def _need_torch():
    if not _TORCH:
        raise ImportError(
            "elephant.learned needs torch (pip install elephant[learned])."
        )


# --------------------------------------------------------------------- #
# Device / helpers                                                      #
# --------------------------------------------------------------------- #
def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _batch(ids_list: List[List[int]], max_len: int, pad: int) -> torch.Tensor:
    n = len(ids_list)
    L = min(max_len, max((len(x) for x in ids_list), default=1))
    out = torch.full((n, L), pad, dtype=torch.long)
    for i, x in enumerate(ids_list):
        out[i, : len(x[:L])] = torch.tensor(x[:L], dtype=torch.long)
    return out


def _to_ids(samples: Sequence[Sample], vocab: Vocab, max_len: int) -> List[List[int]]:
    return [vocab.encode(s.tokens, max_len=max_len) for s in samples]


# --------------------------------------------------------------------- #
# Metrics                                                               #
# --------------------------------------------------------------------- #
def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    a = a.astype(float)
    b = b.astype(float)
    if a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# --------------------------------------------------------------------- #
# Training orchestration                                                #
# --------------------------------------------------------------------- #
@dataclass
class TrainResult:
    """Everything the doc and tests need."""
    vocab_size: int
    n_train: int
    n_test: int
    n_pretrain_pairs: int
    device: str
    heldout: Dict[str, Dict[str, float]]      # dial -> {"r":.., "r2":.., "teacher_std":..}
    heldout_room: Dict[str, Dict[str, float]] # dial -> {"r":.., "r2":..} (room-mean)
    pretrained: bool
    epochs: int
    jepa_epochs: int
    loss: float


def train_and_report(
    base: str = "/home/eileen/projects/ai-writings",
    vocab_size: int = 4000,
    max_len: int = 256,
    d_model: int = 64,
    d_trunk: int = 64,
    window: int = 8,
    stride: int = 1,
    pretrain: bool = True,
    jepa_epochs: int = 5,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    train_files: Optional[Sequence[str]] = None,
    test_files: Optional[Sequence[str]] = None,
    checkpoint_dir: Optional[str] = None,
) -> TrainResult:
    """Run the whole ROADMAP v1 pipeline and return held-out transfer numbers.

    `train_files`/`test_files` override the corpus (used by tests with a tiny
    synthetic corpus; default = the Tap nights 1-2 vs 3-4 + speeches).
    """
    _need_torch()
    _seed(seed)
    device = _device()

    if train_files is None or test_files is None:
        train_files, test_files = tap_corpus(base)

    train_rooms = [room_from_file(f) for f in train_files if os.path.exists(f)]
    test_rooms = [room_from_file(f) for f in test_files if os.path.exists(f)]

    # ---- vocab from TRAIN text only ------------------------------------ #
    all_train_tokens = tokenize(" ".join(m.text for r in train_rooms for m in r.messages))
    vocab = Vocab(all_train_tokens, max_size=vocab_size)

    train_samples = build_windows(train_rooms, window=window, stride=stride)
    test_samples = build_windows(test_rooms, window=window, stride=stride)

    train_ids = _to_ids(train_samples, vocab, max_len)
    test_ids = _to_ids(test_samples, vocab, max_len)
    y_train = np.stack([s.labels for s in train_samples]) if train_samples else np.zeros((0, 7))
    y_test = np.stack([s.labels for s in test_samples]) if test_samples else np.zeros((0, 7))

    model = LearnedDialModel(len(vocab), n_heads=len(DIAL_NAMES),
                             d_model=d_model, d_trunk=d_trunk).to(device)

    # ---- optional JEPA self-supervised pretraining --------------------- #
    pretrained = False
    n_pretrain_pairs = 0
    if pretrain and len(train_samples) > 4:
        pretrained = True
        jepa = JEPAPretrainer(len(vocab), d_model=d_model, d_trunk=d_trunk).to(device)
        # context = window t, target = window t+1 within the same room
        pairs: List[Tuple[List[int], List[int]]] = []
        for r in train_rooms:
            ws = room_windows(r, window=window, stride=stride)
            ids = [vocab.encode(s.tokens, max_len=max_len) for s in ws]
            for a, b in zip(ids[:-1], ids[1:]):
                pairs.append((a, b))
        n_pretrain_pairs = len(pairs)
        opt = torch.optim.Adam(
            list(jepa.context_encoder.parameters()) + list(jepa.predictor.parameters()),
            lr=lr,
        )
        jepa.train()
        for _ in range(max(0, jepa_epochs)):
            random.shuffle(pairs)
            for i in range(0, len(pairs), batch_size):
                chunk = pairs[i : i + batch_size]
                ctx = _batch([c for c, _ in chunk], max_len, vocab.PAD).to(device)
                tgt = _batch([t for _, t in chunk], max_len, vocab.PAD).to(device)
                z_c, z_t, p = jepa(ctx, tgt)
                cos = 1.0 - F.cosine_similarity(p, z_t, dim=-1).mean()
                reg = _vicreg(z_c)
                loss = cos + 0.05 * reg
                opt.zero_grad()
                loss.backward()
                opt.step()
                jepa.update_target()
        # seed the supervised trunk with the pretrained context encoder
        model.encoder.load_state_dict(jepa.context_encoder.state_dict())

    # ---- supervised distillation --------------------------------------- #
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    n_train = len(train_ids)
    final_loss = float("nan")
    for _ in range(epochs):
        order = list(range(n_train))
        random.shuffle(order)
        for i in range(0, n_train, batch_size):
            idx = order[i : i + batch_size]
            xb = _batch([train_ids[j] for j in idx], max_len, vocab.PAD).to(device)
            yb = Yt[idx]
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            final_loss = float(loss.item())

    # ---- evaluate on held-out ------------------------------------------ #
    model.eval()
    heldout: Dict[str, Dict[str, float]] = {}
    heldout_room: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        preds = []
        for i in range(0, len(test_ids), batch_size):
            xb = _batch(test_ids[i : i + batch_size], max_len, vocab.PAD).to(device)
            preds.append(model(xb).cpu().numpy())
        pred_all = np.concatenate(preds) if preds else np.zeros((0, 7))
        for j, name in enumerate(DIAL_NAMES):
            t = y_test[:, j]
            p = pred_all[:, j]
            heldout[name] = {
                "r": _pearson_r(t, p),
                "r2": _r2(t, p),
                "teacher_std": float(t.std()) if t.size else float("nan"),
            }
            # room-mean aggregation: the elephant feels a *room*, not a span
            rooms = [s.room for s in test_samples]
            pm = _group_mean(rooms, p)
            tm = _group_mean(rooms, t)
            heldout_room[name] = {"r": _pearson_r(tm, pm), "r2": _r2(tm, pm)}

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "learned_dials.pt"))
        vocab.save(os.path.join(checkpoint_dir, "learned_vocab.txt"))
        with open(os.path.join(checkpoint_dir, "results.json"), "w") as f:
            import json
            json.dump({
                "heldout": heldout,
                "heldout_room": heldout_room,
                "n_train": n_train,
                "n_test": len(test_samples),
                "n_pretrain_pairs": n_pretrain_pairs,
                "pretrained": pretrained,
                "vocab_size": len(vocab),
                "device": device,
                "epochs": epochs,
                "jepa_epochs": jepa_epochs,
            }, f, indent=2, default=float)

    return TrainResult(
        vocab_size=len(vocab), n_train=n_train, n_test=len(test_samples),
        n_pretrain_pairs=n_pretrain_pairs, device=device, heldout=heldout,
        heldout_room=heldout_room, pretrained=pretrained, epochs=epochs,
        jepa_epochs=jepa_epochs, loss=final_loss,
    )


def _group_mean(keys: Sequence[str], values: np.ndarray) -> np.ndarray:
    """Mean of `values` grouped by `keys`, in order of first appearance."""
    order: List[str] = []
    acc: Dict[str, List[float]] = {}
    for k, v in zip(keys, values):
        if k not in acc:
            acc[k] = []
            order.append(k)
        acc[k].append(float(v))
    return np.array([float(np.mean(acc[k])) for k in order])


# --------------------------------------------------------------------- #
# The learned Dial — a swap-in replacement for any v0 dial                #
# --------------------------------------------------------------------- #
class LearnedDial(Dial):
    """A trained student for one dial dimension, satisfying the `Dial` ABC.

    Reads a room by tokenizing its text, encoding it with the shared trunk,
    and running its single head. The output is clamped to the v0 dial's
    native range so it is a drop-in for the hand-crafted dial in any
    `DialBank` / `RoomField`.
    """

    def __init__(self, name: str, model: "LearnedDialModel", vocab: Vocab,
                 head_index: int, device: str = "cpu", max_len: int = 256):
        _need_torch()
        assert name in DIAL_NAMES, f"unknown dial name {name!r}"
        self.name = name
        self._model = model
        self._vocab = vocab
        self._head = head_index
        self._device = device
        self._max_len = max_len
        self._lo, self._hi = dial_range(name)
        self.description = f"learned {name} dial (distilled from the v0 teacher)"

    def _encode_room(self, room: Room) -> torch.Tensor:
        tokens = tokenize(" ".join(m.text for m in room.messages))
        ids = self._vocab.encode(tokens, max_len=self._max_len)
        x = _batch([ids], self._max_len, self._vocab.PAD).to(self._device)
        return x

    def read(self, room: Room) -> float:
        _need_torch()
        if not room.messages:
            return float(np.clip(0.0, self._lo, self._hi))
        self._model.eval()
        with torch.no_grad():
            x = self._encode_room(room)
            h = self._model.encode(x)
            v = self._model.heads[self._head](h).item()
        return float(np.clip(v, self._lo, self._hi))


def load_learned_bank(checkpoint_dir: str, device: Optional[str] = None) -> List[Dial]:
    """Reconstruct the seven learned dials from a saved checkpoint + vocab."""
    _need_torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab.load(os.path.join(checkpoint_dir, "learned_vocab.txt"))
    model = LearnedDialModel(len(vocab), n_heads=len(DIAL_NAMES)).to(device)
    model.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, "learned_dials.pt"),
                   map_location=device)
    )
    model.eval()
    return [LearnedDial(n, model, vocab, i, device) for i, n in enumerate(DIAL_NAMES)]


# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #
def _fmt_results(res: TrainResult) -> str:
    lines = [
        f"device={res.device} vocab={res.vocab_size} train={res.n_train} "
        f"test={res.n_test} jepa_pairs={res.n_pretrain_pairs} "
        f"pretrained={res.pretrained} epochs={res.epochs} jepa_epochs={res.jepa_epochs}",
        "per-dial held-out transfer (window-level):",
        "  dial           r       r2      teacher_std",
    ]
    for n in DIAL_NAMES:
        h = res.heldout.get(n, {})
        lines.append(
            f"  {n:14s} {h.get('r', float('nan')):+6.3f}  "
            f"{h.get('r2', float('nan')):+6.3f}  {h.get('teacher_std', float('nan')):6.3f}"
        )
    lines.append("per-dial held-out transfer (room-mean):")
    lines.append("  dial           r       r2")
    for n in DIAL_NAMES:
        h = res.heldout_room.get(n, {})
        lines.append(f"  {n:14s} {h.get('r', float('nan')):+6.3f}  {h.get('r2', float('nan')):+6.3f}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Train the elephant's learned dials (ROADMAP v1)")
    p.add_argument("--base", default="/home/eileen/projects/ai-writings")
    p.add_argument("--vocab-size", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--jepa-epochs", type=int, default=5)
    p.add_argument("--no-pretrain", action="store_true")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    res = train_and_report(
        base=args.base, vocab_size=args.vocab_size, epochs=args.epochs,
        jepa_epochs=args.jepa_epochs, pretrain=not args.no_pretrain,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(_fmt_results(res))


if __name__ == "__main__":
    main()
