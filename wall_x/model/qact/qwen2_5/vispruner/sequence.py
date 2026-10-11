"""Sequence masking and repadding for visual token removal."""

from __future__ import annotations

from typing import Any

import torch


def _apply_one(
    value: torch.Tensor | None,
    keep_mask: torch.BoolTensor,
    *,
    pad_value: int | float = 0,
    sequence_dim: int = 1,
) -> torch.Tensor | None:
    if value is None:
        return None
    if value.shape[0] != keep_mask.shape[0] or value.shape[sequence_dim] != keep_mask.shape[1]:
        raise ValueError(
            f"sequence shape {tuple(value.shape)} does not match mask "
            f"{tuple(keep_mask.shape)}"
        )
    if sequence_dim != 1:
        value = value.movedim(sequence_dim, 1)

    lengths = keep_mask.sum(dim=1).tolist()
    max_length = max(lengths, default=0)
    output_shape = (value.shape[0], max_length, *value.shape[2:])
    output = torch.full(
        output_shape,
        pad_value,
        dtype=value.dtype,
        device=value.device,
    )
    for batch_idx, length in enumerate(lengths):
        indices = keep_mask[batch_idx].nonzero(as_tuple=True)[0]
        if length:
            output[batch_idx, :length] = value[batch_idx].index_select(0, indices)
    if sequence_dim != 1:
        output = output.movedim(1, sequence_dim)
    return output


def apply_keep_mask(
    *,
    keep_mask: torch.BoolTensor,
    input_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    labels: torch.Tensor | None,
    moe_token_types: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
    pad_token_id: int = 0,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Apply one batch sequence mask and right-pad the remaining tokens."""

    if keep_mask.ndim != 2 or keep_mask.dtype != torch.bool:
        raise ValueError("keep_mask must be a boolean tensor with shape [B, S]")
    if input_ids is None and inputs_embeds is None:
        raise ValueError("input_ids or inputs_embeds is required")
    if position_ids is not None and position_ids.ndim == 3:
        # Qwen M-RoPE uses [3, B, S].  Move the three spatial axes to the
        # feature dimension while gathering along the batch sequence axis.
        pruned_positions = _apply_one(
            position_ids.permute(1, 2, 0), keep_mask, pad_value=1
        ).permute(2, 0, 1)
    else:
        pruned_positions = _apply_one(position_ids, keep_mask, pad_value=1)
    return (
        _apply_one(input_ids, keep_mask, pad_value=pad_token_id),
        _apply_one(attention_mask, keep_mask, pad_value=0),
        _apply_one(labels, keep_mask, pad_value=-100),
        _apply_one(moe_token_types, keep_mask, pad_value=0),
        pruned_positions,
        _apply_one(inputs_embeds, keep_mask, pad_value=0.0),
    )


__all__ = ["apply_keep_mask"]
