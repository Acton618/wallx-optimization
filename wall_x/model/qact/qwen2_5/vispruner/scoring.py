"""Score reducers for visual-token selection."""

from __future__ import annotations

import torch


ATTN_REDUCERS = frozenset(("attn_mean", "attn_max"))
OUTPUT_REDUCERS = frozenset(("attn_out_norm",))
NO_ATTN_REDUCERS = frozenset(("random", "norm"))


def needs_attn_matrix(reduce_name: str) -> bool:
    return reduce_name in ATTN_REDUCERS


def needs_attn_output(reduce_name: str) -> bool:
    return reduce_name in OUTPUT_REDUCERS


def reduce_scores(
    reduce_name: str,
    *,
    attn: torch.Tensor | None = None,
    attn_output: torch.Tensor | None = None,
    image_embeds: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Reduce one image's attention/output/features to a ``[N]`` score vector."""

    if reduce_name == "attn_mean":
        if attn is None:
            raise ValueError("attn_mean requires an attention matrix")
        return attn.detach().float().mean(dim=0).mean(dim=0)
    if reduce_name == "attn_max":
        if attn is None:
            raise ValueError("attn_max requires an attention matrix")
        return attn.detach().float().mean(dim=0).amax(dim=0)
    if reduce_name == "attn_out_norm":
        if attn_output is None:
            raise ValueError("attn_out_norm requires attention output")
        return attn_output.detach().float().norm(dim=-1)
    if reduce_name == "random":
        if image_embeds is None:
            raise ValueError("random scoring requires image_embeds for shape/device")
        return torch.rand(
            image_embeds.shape[0],
            device=image_embeds.device,
            dtype=torch.float32,
            generator=generator,
        )
    if reduce_name == "norm":
        if image_embeds is None:
            raise ValueError("norm scoring requires image_embeds")
        return image_embeds.detach().float().norm(dim=-1)
    raise ValueError(f"unknown score reducer: {reduce_name!r}")


__all__ = [
    "ATTN_REDUCERS",
    "OUTPUT_REDUCERS",
    "NO_ATTN_REDUCERS",
    "needs_attn_matrix",
    "needs_attn_output",
    "reduce_scores",
]
