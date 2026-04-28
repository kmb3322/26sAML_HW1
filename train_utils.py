"""Batch collation and masked cross-entropy used by ``train.py``."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def collate_lm_batch(
    batch: List[Dict],
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences for next-token language model training.

    Each example in ``batch`` is a dict with keys:
      - ``ids``: list[int] of length ``L`` (token sequence)
      - ``target_mask``: list[float] of length ``L - 1`` (which target positions to count)

    Returns:
      - ``x``: ``[B, T-1]``, token ids padded with ``pad_id``
      - ``y``: ``[B, T-1]``, next-token targets padded with ``-100`` (ignored by CE)
      - ``target_mask``: ``[B, T-1]``, ``1.0`` at positions where the loss should count
    """
    max_len = max(len(ex["ids"]) for ex in batch)
    B = len(batch)
    T = max_len - 1

    x = torch.full((B, T), pad_id, dtype=torch.long)
    y = torch.full((B, T), -100, dtype=torch.long)
    target_mask = torch.zeros((B, T), dtype=torch.float32)

    for i, ex in enumerate(batch):
        ids = torch.tensor(ex["ids"], dtype=torch.long)
        m = torch.tensor(ex["target_mask"], dtype=torch.float32)

        seq_T = len(ids) - 1
        x[i, :seq_T] = ids[:-1]
        y[i, :seq_T] = ids[1:]
        target_mask[i, :seq_T] = m

    return x, y, target_mask


def masked_cross_entropy(
    logits: torch.Tensor,
    y: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean cross-entropy across positions where ``target_mask`` is non-zero.

    ``logits`` is ``[B, T, V]``, ``y`` is ``[B, T]``, ``target_mask`` is ``[B, T]``.
    Positions with ``y == -100`` are ignored by ``F.cross_entropy`` and
    additionally those (and any unmasked positions) get zero weight via
    ``target_mask``.
    """
    B, T, V = logits.shape
    per_token_loss = F.cross_entropy(
        logits.reshape(B * T, V),
        y.reshape(B * T),
        reduction="none",
        ignore_index=-100,
    ).reshape(B, T)

    denom = target_mask.sum().clamp_min(1.0)
    return (per_token_loss * target_mask).sum() / denom
