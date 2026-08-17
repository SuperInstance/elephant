"""Model-vs-code dial — who is generating the room's signal.

The signal-chain thesis (`docs/signal-chain-thesis.md`): a room's signal is
not only *what* is being said, but *who or what* is generating it. At one
end, a model thinking in the open — prose, hedges, reflection, first-person,
long-form, creative. At the other, code executing deterministically —
symbols, keywords, diffs, error messages, terse, commit-shaped. The ratio
between the two is itself a dial reading: it changes the room's temperature
and shapes what the elephant should nudge.

The ancestor of this repo is a Rust DSP pipeline that transformed a *stream*
of samples one at a time. This dial is the same idea, matured: instead of
transforming a sample, it reads who is *producing* the room's signal and
turns that into one scalar dimension of the field. `[-1 code .. +1 model]`.

A room full of commits is cold. A room full of prose is warm. This is the
8th sense — the one that can tell a ship talking to itself in diffs from a
room full of people thinking out loud.
"""
from __future__ import annotations

import re

from ..dial import Dial
from ..room import Room

# --------------------------------------------------------------------------- #
# Lexicons — the words that smell like model vs code.                         #
# --------------------------------------------------------------------------- #

# Model: hedges, reflection, first-person, prose, creativity, warmth.
MODEL_WORDS = {
    "i", "we", "my", "our", "me", "us", "you", "your",
    "maybe", "perhaps", "probably", "likely", "arguably", "possibly",
    "feel", "felt", "feels", "feeling", "think", "thinks", "believe",
    "wonder", "wondered", "imagine", "remember", "remembers", "sense",
    "seemed", "seems", "seem", "however", "moreover", "therefore", "thus",
    "indeed", "ultimately", "meanwhile", "furthermore", "nevertheless",
    "story", "voice", "warm", "warmth", "light", "gentle", "soft", "alive",
    "holds", "held", "together", "kind", "wonderful", "beautiful",
    "something", "someone", "everything", "nothing", "ourselves", "myself",
}
MODEL_PHRASES = {
    "i think", "i believe", "i wonder", "i feel", "it seems", "in a sense",
    "sort of", "kind of", "as if", "what if", "to me", "for me",
    "maybe we", "perhaps the", "we are", "we were",
}

# Code: keywords, determinism, diffs, errors, commit discipline.
CODE_WORDS = {
    "def", "fn", "function", "return", "import", "class", "struct",
    "impl", "let", "const", "var", "pub", "match", "enum", "trait",
    "elif", "else", "loop", "while", "typeof", "interface", "namespace",
    "static", "void", "mut", "traceback", "error", "exception", "assert",
    "undefined", "nan", "null", "none", "todo", "fixme", "hack",
    "deprecated", "refactor", "merge", "commit", "push", "rebase", "pull",
    "diff", "patch", "lint", "typecheck", "coverage", "dockerfile",
    "pipeline", "syntaxerror", "keyerror", "typeerror",
}
CODE_PHRASES = {
    "feat:", "fix:", "chore:", "docs:", "refactor:", "test:", "perf:",
    "build:", "ci:", "revert:", "style:", "release:", "merge ", "commit ",
    "push ", "pull request", "diff --git", "+++ b/", "--- a/", "@@ -",
    "at line", "syntax error", "merge conflict", "type error",
    "null pointer", "undefined behavior", "running tests",
}
# Symbols that read as code: braces, brackets, parens, semicolons, operators.
CODE_SYMBOLS = re.compile(
    r"[{}()\[\];]|->|=>|::|==|!=|<=|>=|\+=|-=|\*=|/=|&&|\|\|"
)


def _score(text: str, words: object) -> float:
    """Score one message on the model/code spectrum, in [-1, +1].

    `text` is the lowercased raw message; `words` is an iterable of its
    lowercase `\\w+` tokens. Counts model vs code lexicon hits (words +
    phrases) plus code symbol density, then maps the balance onto [-1, +1]:
    -1 = pure code, +1 = pure model, 0 = no signal (or perfectly balanced).
    """
    wset = set(words)
    model = sum(1 for w in wset if w in MODEL_WORDS)
    model += sum(1 for p in MODEL_PHRASES if p in text)
    code = sum(1 for w in wset if w in CODE_WORDS)
    code += sum(1 for p in CODE_PHRASES if p in text)
    code += len(CODE_SYMBOLS.findall(text))
    total = model + code
    if total == 0:
        return 0.0
    return (model - code) / total


class ModelVsCodeDial(Dial):
    name = "model_vs_code"
    description = "who is generating the room's signal, [-1 code .. +1 model]"

    def read(self, room: Room) -> float:
        if not room.messages:
            return 0.0
        scores = [_score(m.text.lower(), m.words) for m in room.messages]
        return max(-1.0, min(1.0, sum(scores) / len(scores)))
