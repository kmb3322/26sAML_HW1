"""Tokenizers and data generation for the sanity check and modular arithmetic tasks.

Two tokenizers are defined:

- ``ResidueTokenizer`` represents each integer 0..max_p-1 as a single token,
  yielding fixed-length sequences of the form
  ``<bos> <n:a> <op> <n:b> = <n:c> <eos>``.
- ``SanityTokenizer`` is a tiny word-level tokenizer used only for the
  "I love machine learning" sanity-check sentence.

Both tokenizers expose the same minimal interface (``vocab_size``, ``pad_id``,
``bos_id``, ``eos_id``, ``to_dict``/``from_dict``) so that the training loop
can be task-agnostic.
"""

from __future__ import annotations

import random
from typing import Dict, List, Literal, Optional, Tuple

Op = Literal["+", "-", "/"]


# ---------------------------------------------------------------------------
# Residue tokenizer
# ---------------------------------------------------------------------------


class ResidueTokenizer:
    """Single-token-per-residue tokenizer for modular arithmetic.

    With ``max_p=113`` the vocabulary contains one token for each of the
    integers 0..112, so the same tokenizer covers both ``p=97`` and
    ``p=113`` experiments.
    """

    KIND = "residue"

    def __init__(self, max_p: int = 113):
        self.max_p = max_p
        self.special_tokens = ["<pad>", "<bos>", "<eos>", "=", "+", "-", "/"]
        self.number_tokens = [f"<n:{i}>" for i in range(max_p)]
        self.tokens = self.special_tokens + self.number_tokens

        self.stoi = {tok: i for i, tok in enumerate(self.tokens)}
        self.itos = {i: tok for tok, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    @property
    def eq_id(self) -> int:
        return self.stoi["="]

    def op_id(self, op: Op) -> int:
        if op not in {"+", "-", "/"}:
            raise ValueError(f"Unknown op: {op}")
        return self.stoi[op]

    def num_id(self, n: int) -> int:
        if not (0 <= n < self.max_p):
            raise ValueError(f"Number {n} outside tokenizer range [0, {self.max_p}).")
        return self.stoi[f"<n:{n}>"]

    def id_to_num(self, token_id: int) -> int:
        tok = self.itos[int(token_id)]
        if not tok.startswith("<n:"):
            raise ValueError(f"Token id {token_id} ({tok}) is not a number token.")
        return int(tok[3:-1])

    def encode_problem(
        self,
        a: int,
        b: int,
        op: Op,
        c: Optional[int] = None,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        """Encode ``a op b [= c]``.

        If ``c`` is ``None`` only the prompt up through ``=`` is returned, which
        is the slice the model sees at inference time.
        """
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend([self.num_id(a), self.op_id(op), self.num_id(b), self.eq_id])
        if c is not None:
            ids.append(self.num_id(c))
            if add_eos:
                ids.append(self.eos_id)
        return ids

    def to_dict(self) -> Dict:
        return {"kind": self.KIND, "max_p": self.max_p}

    @classmethod
    def from_dict(cls, d: Dict) -> "ResidueTokenizer":
        return cls(max_p=int(d["max_p"]))


# ---------------------------------------------------------------------------
# Sanity tokenizer
# ---------------------------------------------------------------------------


class SanityTokenizer:
    """Tiny word-level tokenizer for the "I love machine learning" sentence."""

    KIND = "sanity"
    DEFAULT_WORDS: Tuple[str, ...] = ("I", "love", "machine", "learning")

    def __init__(self, words: Tuple[str, ...] = DEFAULT_WORDS):
        self.special_tokens = ["<pad>", "<bos>", "<eos>"]
        self.words = list(words)
        self.tokens = self.special_tokens + self.words

        self.stoi = {tok: i for i, tok in enumerate(self.tokens)}
        self.itos = {i: tok for tok, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    def encode_words(
        self,
        words: List[str],
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.stoi[w] for w in words)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.itos[int(i)] for i in ids)

    def to_dict(self) -> Dict:
        return {"kind": self.KIND, "words": list(self.words)}

    @classmethod
    def from_dict(cls, d: Dict) -> "SanityTokenizer":
        return cls(words=tuple(d.get("words", cls.DEFAULT_WORDS)))


def load_tokenizer_from_dict(d: Dict):
    """Reconstruct the right tokenizer from a saved dict."""
    kind = d.get("kind", "residue")
    if kind == "residue":
        return ResidueTokenizer.from_dict(d)
    if kind == "sanity":
        return SanityTokenizer.from_dict(d)
    raise ValueError(f"Unknown tokenizer kind: {kind}")


# ---------------------------------------------------------------------------
# Modular arithmetic data generation
# ---------------------------------------------------------------------------


def modular_answer(a: int, b: int, op: Op, p: int) -> Optional[int]:
    """Return ``a op b mod p`` or ``None`` if undefined (division by zero)."""
    if op == "+":
        return (a + b) % p
    if op == "-":
        return (a - b) % p
    if op == "/":
        if b % p == 0:
            return None
        # p is prime, so Fermat's little theorem gives b^{-1} = b^{p-2} mod p.
        return (a * pow(b, p - 2, p)) % p
    raise ValueError(f"Unknown op: {op}")


def make_modular_rows(
    op: Op,
    p: int,
    seed: int,
    train_frac: float = 0.50,
    val_frac: float = 0.10,
) -> Dict[str, List[Dict]]:
    """Build train/val/test row splits for ``a op b = c mod p``.

    For ``+`` and ``-`` the dataset has size ``p**2``.  For ``/`` we drop
    ``b == 0``, giving ``p * (p - 1)`` rows.

    Each row is a dict ``{a, b, op, p, c, text}`` where ``text`` is the
    human-readable expression (used only for printing/sanity).
    """
    rows: List[Dict] = []
    for a in range(p):
        for b in range(p):
            c = modular_answer(a, b, op, p)
            if c is None:
                continue
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "op": op,
                    "p": p,
                    "c": c,
                    "text": f"{a} {op} {b} = {c}",
                }
            )

    rng = random.Random(seed)
    rng.shuffle(rows)

    n = len(rows)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    return {
        "train": rows[:n_train],
        "val": rows[n_train : n_train + n_val],
        "test": rows[n_train + n_val :],
    }


def rows_to_examples(rows: List[Dict], tokenizer: ResidueTokenizer) -> List[Dict]:
    """Convert rows to next-token training examples.

    The full token sequence is ``[BOS, a, op, b, =, c, EOS]`` (length 7).
    For next-token training the input is ``ids[:-1]`` (length 6) and the
    target is ``ids[1:]`` (length 6).  We compute the loss only on the
    position whose target token is the answer ``c`` -- that is the position
    coming right after ``=``, which has target index ``len(prompt)-1``
    where ``prompt = [BOS, a, op, b, =]``.
    """
    examples: List[Dict] = []
    for r in rows:
        ids = tokenizer.encode_problem(r["a"], r["b"], r["op"], c=r["c"])
        prompt_ids = tokenizer.encode_problem(r["a"], r["b"], r["op"], c=None)

        target_mask = [0.0] * (len(ids) - 1)
        answer_target_index = len(prompt_ids) - 1
        target_mask[answer_target_index] = 1.0

        examples.append({"ids": ids, "target_mask": target_mask, "row": r})
    return examples


# ---------------------------------------------------------------------------
# Sanity-check data
# ---------------------------------------------------------------------------


def make_sanity_examples(
    tokenizer: SanityTokenizer,
    prompt_len: int = 0,
) -> List[Dict]:
    """Single training example for the memorization sanity check.

    Token sequence: ``[BOS, I, love, machine, learning, EOS]`` (length 6).

    ``prompt_len`` is the number of leading tokens that are treated as a
    prompt -- the loss is masked on every prediction *made from inside the
    prompt*.  With ``prompt_len=3`` the prompt is ``[BOS, I, love]`` and the
    loss is computed only on the targets ``[machine, learning, EOS]``, which
    is the second sanity check from the assignment.

    Concretely, target positions ``0 .. prompt_len-2`` are masked (they are
    the predictions of the 2nd, 3rd, ... prompt tokens).  ``prompt_len <= 1``
    means no masking.
    """
    ids = tokenizer.encode_words(list(tokenizer.words))
    target_mask = [1.0] * (len(ids) - 1)
    n_mask = max(prompt_len - 1, 0)
    for i in range(min(n_mask, len(target_mask))):
        target_mask[i] = 0.0
    return [{"ids": ids, "target_mask": target_mask, "prompt_len": prompt_len}]
