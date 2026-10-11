"""Pure tensor algorithms used by VisPruner.

The functions in this file do not import Wall-X.  They operate on one image
at a time so that image boundaries remain explicit when a batch contains
multiple camera views.
"""

from __future__ import annotations

from typing import Any

import torch


def image_token_lengths(grid_thw: Any, spatial_merge_size: int) -> list[int]:
    """Return the merged visual-token count for each ``(t, h, w)`` image."""

    if spatial_merge_size < 1:
        raise ValueError("spatial_merge_size must be positive")
    if hasattr(grid_thw, "tolist"):
        grid_thw = grid_thw.tolist()

    lengths: list[int] = []
    for row in grid_thw:
        if len(row) != 3:
            raise ValueError(f"each grid row must have 3 values, got {row!r}")
        t, h, w = (int(value) for value in row)
        if min(t, h, w) <= 0:
            raise ValueError(f"grid values must be positive, got {row!r}")
        if h % spatial_merge_size or w % spatial_merge_size:
            raise ValueError(
                "height and width must be divisible by spatial_merge_size: "
                f"row={row!r}, spatial_merge_size={spatial_merge_size}"
            )
        lengths.append(t * (h // spatial_merge_size) * (w // spatial_merge_size))
    return lengths


def _stable_descending_argsort(values: torch.Tensor) -> torch.Tensor:
    """Sort scores deterministically when ties occur."""

    try:
        return torch.argsort(values, descending=True, stable=True)
    except TypeError:  # pragma: no cover - compatibility with old torch builds
        return torch.argsort(values, descending=True)


def _diverse_select(
    features_normalized: torch.Tensor,
    residual_indices: torch.Tensor,
    diverse_token_num: int,
) -> torch.Tensor:
    """Select diverse residual tokens using VisPruner's iterative rule."""

    if diverse_token_num <= 0 or residual_indices.numel() == 0:
        return residual_indices[:0]

    remaining = residual_indices
    target = min(int(diverse_token_num), int(remaining.numel()))
    while target > 0 and remaining.numel() > target:
        reduction = min(8, int(remaining.numel()) - target)
        if reduction <= 0 or remaining.numel() < 2:
            break

        residual_tokens = features_normalized[remaining]
        first = residual_tokens[::2]
        second = residual_tokens[1::2]
        if second.numel() == 0:
            break
        pair_scores = (first @ second.transpose(-1, -2)).max(dim=-1).values
        distinct = _stable_descending_argsort(pair_scores)[reduction:]
        remaining = torch.cat((remaining[::2][distinct], remaining[1::2]))

    if remaining.numel() > target:
        remaining = remaining[:target]
    return remaining


def select_indices(
    scores: torch.Tensor,
    features: torch.Tensor,
    keep_num: int,
    important_ratio: float,
    generator: torch.Generator | None = None,
) -> torch.LongTensor:
    """Select important and diverse visual-token indices for one image.

    ``scores`` must be ``[N]`` and ``features`` must be ``[N, C]``.  The
    returned indices are unique and sorted in original token order.
    """

    if scores.ndim != 1:
        raise ValueError(f"scores must have shape [N], got {tuple(scores.shape)}")
    if features.ndim != 2 or features.shape[0] != scores.shape[0]:
        raise ValueError(
            "features must have shape [N, C] with the same N as scores; "
            f"got scores={tuple(scores.shape)}, features={tuple(features.shape)}"
        )
    if keep_num < 1:
        raise ValueError("keep_num must be positive")
    if not 0.0 <= important_ratio <= 1.0:
        raise ValueError("important_ratio must be in [0, 1]")

    n_tokens = int(scores.shape[0])
    if n_tokens == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    keep_num = min(int(keep_num), n_tokens)
    if keep_num == n_tokens:
        return torch.arange(n_tokens, dtype=torch.long, device=scores.device)

    ranking = _stable_descending_argsort(scores.detach())
    important_num = min(keep_num, int(keep_num * important_ratio))
    important = ranking[:important_num]
    residual = ranking[important_num:]
    diverse_num = keep_num - important_num

    normalized = features.detach().float()
    normalized = normalized / normalized.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    if generator is not None and not torch.is_floating_point(scores):
        raise ValueError("scores must be floating point")
    selected_diverse = _diverse_select(normalized, residual, diverse_num)

    # The iterative rule can stop early for very small odd residual sets.  Fill
    # any remaining slots by the original score ranking so the budget stays
    # exact and the sequence planner never receives a short selection.
    selected = torch.cat((important, selected_diverse))
    if selected.numel() < keep_num:
        selected_set = set(selected.tolist())
        fill = [idx for idx in residual.tolist() if idx not in selected_set]
        selected = torch.cat(
            (
                selected,
                torch.tensor(
                    fill[: keep_num - selected.numel()],
                    dtype=torch.long,
                    device=scores.device,
                ),
            )
        )
    return torch.sort(selected[:keep_num]).values


__all__ = ["image_token_lengths", "select_indices"]
