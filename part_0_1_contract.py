"""
CSE 493S/599S HW2: interface for Part 0 and Part 1.

We will be using an autograder for this part. For ease of grading, please fill in
these functions to evaluate your trained models. Do not rename the functions
or change their signatures.

You may import from other files in your repo. You may add helper functions.
Just make sure the three functions below work as specified.

We implement the three required functions as a thin wrapper around
``inference.py``.  The checkpoint directory must contain a ``ckpt.pt`` saved
by ``train.py``.  Both the sanity-check checkpoint (``SanityTokenizer``) and
the modular-arithmetic checkpoint (``ResidueTokenizer``) are supported.
"""

from inference import load_checkpoint, predict_modular_answer


def load_model_and_tokenizer(checkpoint_dir: str):
    """Load model + tokenizer from a directory containing ``ckpt.pt``.

    The model is moved onto an appropriate device and put into eval mode.
    """
    model, tokenizer, _device, _ckpt = load_checkpoint(checkpoint_dir)
    return model, tokenizer


def get_bos_token(tokenizer=None):
    """Return the BOS token id for the supplied tokenizer.

    All tokenizers in this repo expose ``bos_id`` (an int).  When no
    tokenizer is provided we cannot identify the right BOS, so we return
    ``None``.
    """
    if tokenizer is None:
        return None
    return tokenizer.bos_id


def predict_answer(model, tokenizer, a: int, b: int, op: str, p: int) -> int:
    """Predict ``c`` such that ``a op b ≡ c (mod p)`` using the trained model."""
    import torch  # local to avoid hard import for callers that only need bos.

    device = next(model.parameters()).device
    return predict_modular_answer(model, tokenizer, a, b, op, p, device=device)
