"""Build and apply a visual-token pruning plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .algorithms import image_token_lengths, select_indices
from .config import VisPrunerConfig
from .hook import pop_scores
from .scoring import reduce_scores
from .sequence import apply_keep_mask


@dataclass
class PrunePlan:
    pruned: bool
    image_embeds: torch.Tensor | None = None
    keep_mask: torch.BoolTensor | None = None
    keep_indices: list[torch.Tensor] | None = None
    pad_token_id: int = 0

    def apply(
        self,
        *,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
        moe_token_types: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if not self.pruned:
            return (
                input_ids,
                attention_mask,
                labels,
                moe_token_types,
                position_ids,
                inputs_embeds,
            )
        if self.keep_mask is None:
            raise ValueError("pruned plan is missing keep_mask")
        return apply_keep_mask(
            keep_mask=self.keep_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            moe_token_types=moe_token_types,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pad_token_id=self.pad_token_id,
        )


def _align_patch_scores(
    visual: torch.nn.Module,
    image_grid_thw: torch.Tensor,
    raw_scores: list[torch.Tensor],
    spatial_merge_size: int,
) -> list[torch.Tensor]:
    """Undo the visual window reorder and merge four patch scores per token."""

    merge_unit = spatial_merge_size * spatial_merge_size
    if not raw_scores:
        return []
    merged = torch.cat(
        [score.reshape(-1) for score in raw_scores], dim=0
    )
    if merged.numel() % merge_unit:
        raise ValueError("attention score count is not divisible by merge unit")
    merged = merged.reshape(-1, merge_unit).mean(dim=1)
    window_index, _ = visual.get_window_index(image_grid_thw.to(torch.int32))
    reverse = torch.argsort(window_index.to(merged.device))
    merged = merged.index_select(0, reverse)
    lengths = image_token_lengths(image_grid_thw, spatial_merge_size)
    if sum(lengths) != merged.shape[0]:
        raise ValueError("aligned score count does not match image grid")
    output: list[torch.Tensor] = []
    offset = 0
    for length in lengths:
        output.append(merged[offset : offset + length])
        offset += length
    return output


def _split_image_positions(
    input_ids: torch.Tensor,
    image_token_id: int,
    lengths: list[int],
) -> list[list[torch.Tensor]]:
    positions = (input_ids == image_token_id).nonzero(as_tuple=False)
    per_sample: list[list[torch.Tensor]] = [[] for _ in range(input_ids.shape[0])]
    row_offset = 0
    for batch_idx in range(input_ids.shape[0]):
        row_positions = positions[positions[:, 0] == batch_idx, 1]
        needed = 0
        row_groups: list[torch.Tensor] = []
        while (
            row_offset + len(row_groups) < len(lengths)
            and needed < row_positions.numel()
        ):
            length = lengths[row_offset + len(row_groups)]
            next_needed = needed + length
            if next_needed > row_positions.numel():
                break
            row_groups.append(row_positions[needed:next_needed])
            needed = next_needed
        if needed != row_positions.numel():
            raise ValueError(
                "image_grid_thw and image token positions disagree for batch "
                f"{batch_idx}: grid tokens={needed}, sequence tokens={row_positions.numel()}"
            )
        per_sample[batch_idx] = row_groups
        row_offset += len(row_groups)
    if row_offset != len(lengths):
        raise ValueError("not all image_grid_thw rows were assigned to a batch item")
    return per_sample


def maybe_prune_images(
    *,
    visual: torch.nn.Module,
    image_embeds: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    labels: torch.Tensor | None,
    moe_token_types: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
    spatial_merge_size: int,
    model_config: Any,
) -> PrunePlan:
    """Return a plan for the current image batch, or a disabled plan."""

    cfg = VisPrunerConfig.from_model_config(model_config)
    if not cfg.enabled or image_grid_thw is None:
        return PrunePlan(pruned=False)

    lengths = image_token_lengths(image_grid_thw, spatial_merge_size)
    if sum(lengths) != image_embeds.shape[0]:
        raise ValueError(
            "image_grid_thw token count does not match image_embeds: "
            f"grid={sum(lengths)}, embeds={image_embeds.shape[0]}"
        )

    reducer = cfg.score_reduce
    hook_data = pop_scores(visual)
    score_parts: list[torch.Tensor]
    if reducer in {"attn_mean", "attn_max"}:
        layer_data = hook_data.get(cfg.score_layer)
        if not layer_data or layer_data.get("attn") is None:
            raise RuntimeError(
                f"no attention scores collected for VisPruner layer {cfg.score_layer}"
            )
        score_parts = [
            reduce_scores(reducer, attn=part)
            for part in layer_data["attn"]
        ]
        score_parts = _align_patch_scores(
            visual, image_grid_thw, score_parts, spatial_merge_size
        )
    elif reducer == "attn_out_norm":
        layer_data = hook_data.get(cfg.score_layer)
        if not layer_data or layer_data.get("attn_output") is None:
            raise RuntimeError(
                f"no attention output collected for VisPruner layer {cfg.score_layer}"
            )
        raw = layer_data["attn_output"]
        patch_scores = [
            reduce_scores(
                reducer,
                attn_output=raw[start:end],
            )
            for start, end in _patch_boundaries(image_grid_thw)
        ]
        score_parts = _align_patch_scores(
            visual, image_grid_thw, patch_scores, spatial_merge_size
        )
    else:
        score_parts = []

    if reducer in {"random", "norm"}:
        score_parts = []
    generator = torch.Generator(device=image_embeds.device)
    generator.manual_seed(cfg.seed)

    keep_indices: list[torch.Tensor] = []
    selected_embeds: list[torch.Tensor] = []
    embed_offset = 0
    for image_idx, length in enumerate(lengths):
        features = image_embeds[embed_offset : embed_offset + length]
        if reducer in {"random", "norm"}:
            scores = reduce_scores(
                reducer,
                image_embeds=features,
                generator=generator,
            )
        else:
            scores = score_parts[image_idx]
        indices = select_indices(
            scores,
            features,
            cfg.keep_num,
            cfg.important_ratio,
            generator=generator,
        )
        keep_indices.append(indices)
        selected_embeds.append(features.index_select(0, indices))
        embed_offset += length

    return plan_from_keep_indices(
        image_embeds=torch.cat(selected_embeds, dim=0),
        keep_indices=keep_indices,
        image_grid_thw=image_grid_thw,
        input_ids=input_ids,
        spatial_merge_size=spatial_merge_size,
        model_config=model_config,
    )


def plan_from_keep_indices(
    *,
    image_embeds: torch.Tensor,
    keep_indices: list[torch.Tensor],
    image_grid_thw: torch.Tensor,
    input_ids: torch.Tensor,
    spatial_merge_size: int,
    model_config: Any,
) -> PrunePlan:
    """Build the sequence mask for indices selected inside the vision tower."""

    lengths = image_token_lengths(image_grid_thw, spatial_merge_size)
    if len(keep_indices) != len(lengths):
        raise ValueError("one keep-index tensor is required per image")
    if sum(int(indices.numel()) for indices in keep_indices) != image_embeds.shape[0]:
        raise ValueError("selected image embeddings do not match keep indices")
    for length, indices in zip(lengths, keep_indices):
        if indices.ndim != 1 or indices.numel() == 0:
            raise ValueError("each image must keep at least one visual token")
        if bool((indices < 0).any()) or bool((indices >= length).any()):
            raise ValueError("keep indices are outside image bounds")
        if indices.unique().numel() != indices.numel():
            raise ValueError("keep indices must be unique")
        if not torch.equal(indices, indices.sort().values):
            raise ValueError("keep indices must be in original image order")

    image_positions = _split_image_positions(
        input_ids,
        int(getattr(model_config, "image_token_id")),
        lengths,
    )
    keep_mask = torch.ones_like(input_ids, dtype=torch.bool)
    image_idx = 0
    for batch_idx, row_groups in enumerate(image_positions):
        for positions in row_groups:
            # ``positions`` contains sequence-column indices.  Index the
            # corresponding batch row explicitly; indexing ``keep_mask`` with
            # the columns alone would incorrectly address its batch axis.
            row_mask = keep_mask[batch_idx]
            row_mask[positions] = False
            row_mask[positions[keep_indices[image_idx]]] = True
            image_idx += 1

    if int(keep_mask.sum().item()) <= 0:
        raise ValueError("pruning removed the complete input sequence")
    return PrunePlan(
        pruned=True,
        image_embeds=image_embeds,
        keep_mask=keep_mask,
        keep_indices=keep_indices,
        pad_token_id=int(getattr(model_config, "pad_token_id", 0) or 0),
    )


def _patch_boundaries(image_grid_thw: torch.Tensor) -> list[tuple[int, int]]:
    lengths = [int(t) * int(h) * int(w) for t, h, w in image_grid_thw.tolist()]
    boundaries: list[tuple[int, int]] = []
    offset = 0
    for length in lengths:
        boundaries.append((offset, offset + length))
        offset += length
    return boundaries


__all__ = ["PrunePlan", "maybe_prune_images"]
