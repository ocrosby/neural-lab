"""Embedded tiny-Shakespeare snippet + character-level encoder.

Public-domain text (Shakespeare's *Coriolanus*, opening scene). ~2.5 KB
included as a Python string so tests and demos run fully offline. This is
the same corpus Karpathy's char-rnn uses in miniature form.
"""

from __future__ import annotations

import numpy as np

CORPUS: str = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.

Second Citizen:
Would you proceed especially against Caius Marcius?

All:
Against him first: he's a very dog to the commonalty.

Second Citizen:
Consider you what services he has done for his country?

First Citizen:
Very well; and could be content to give him good
report fort, but that he pays himself with being proud.

Second Citizen:
Nay, but speak not maliciously.

First Citizen:
I say unto you, what he hath done famously, he did
it to that end: though soft-conscienced men can be
content to say it was for his country he did it to
please his mother and to be partly proud; which he
is, even till the altitude of his virtue.
"""


class TextEncoder:
    """Character-level tokenizer: builds vocab from a training string and (en|de)codes.

    Attributes:
        chars: Sorted list of unique characters in the training text.
        char_to_id: Char -> integer id.
        id_to_char: Integer id -> char.
    """

    def __init__(self, text: str) -> None:
        assert len(text) > 0, "TextEncoder needs non-empty training text"
        self.chars: list[str] = sorted(set(text))
        self.char_to_id: dict[str, int] = {c: i for i, c in enumerate(self.chars)}
        self.id_to_char: dict[int, str] = {i: c for c, i in self.char_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> np.ndarray:
        for c in text:
            if c not in self.char_to_id:
                raise ValueError(f"unseen character {c!r} — not in vocab")
        return np.array([self.char_to_id[c] for c in text], dtype=np.int64)

    def decode(self, ids: np.ndarray) -> str:
        assert ids.ndim == 1, f"expected 1D array of ids, got shape {ids.shape}"
        return "".join(self.id_to_char[int(i)] for i in ids)


def sample_sequence(ids: np.ndarray, seq_len: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Given a long id sequence, return an (input, target) pair with target = shift-by-1."""
    assert ids.ndim == 1, f"ids must be 1D, got shape {ids.shape}"
    assert seq_len > 0, f"seq_len must be positive, got {seq_len}"
    assert len(ids) > seq_len, f"corpus of length {len(ids)} too short for seq_len={seq_len}"

    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, len(ids) - seq_len - 1))
    inputs = ids[start : start + seq_len].copy()
    targets = ids[start + 1 : start + 1 + seq_len].copy()
    return inputs, targets
