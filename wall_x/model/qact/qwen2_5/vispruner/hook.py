"""Optional attention hooks for VisPruner scoring.

Hooks only observe the visual attention modules.  They return ``None`` and
therefore leave the original FlashAttention/eager forward path untouched.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from wall_x.model.qact.qwen2_5.modeling_qwen2_5_vl import (
    apply_rotary_pos_emb_flashatt,
    apply_rotary_pos_emb_vision,
)

from .config import VisPrunerConfig
from .scoring import needs_attn_matrix, needs_attn_output, reduce_scores


def _segment_attention_scores(
    module: Any,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    flash_rope: bool,
) -> list[torch.Tensor]:
    """Recompute per-image Q/K attention scores without changing the module."""

    seq_len = hidden_states.shape[0]
    q, k, _ = (
        module.qkv(hidden_states)
        .reshape(seq_len, 3, module.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    # The wallx-0.5 vision attention implementation does not expose a
    # ``head_dim`` attribute.  The reshape above already gives us the exact
    # per-head dimension and also keeps this hook compatible with both the
    # eager and FlashAttention visual blocks.
    head_dim = q.shape[-1]
    if position_embeddings is None:
        return []
    cos, sin = position_embeddings
    if flash_rope:
        q, k = apply_rotary_pos_emb_flashatt(
            q.unsqueeze(0), k.unsqueeze(0), cos, sin
        )
        q, k = q.squeeze(0), k.squeeze(0)
    else:
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

    scores: list[torch.Tensor] = []
    boundaries = cu_seqlens.to(device=q.device, dtype=torch.long).tolist()
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        q_part = q[start:end].transpose(0, 1)
        k_part = k[start:end].transpose(0, 1)
        logits = torch.matmul(q_part, k_part.transpose(-1, -2)) / math.sqrt(head_dim)
        attn = logits.float().softmax(dim=-1).to(logits.dtype)
        scores.append(attn)
    return scores


def install_hooks(visual: torch.nn.Module, model_config: Any) -> None:
    """Install idempotent, opt-in hooks on the configured visual layers."""

    if hasattr(visual, "_vispruner_state"):
        return

    cfg = VisPrunerConfig.from_model_config(model_config)
    if not cfg.enabled:
        # Disabled means the original visual forward has no hook overhead.
        return
    state: dict[str, Any] = {"config": cfg, "scores": {}, "handles": []}
    visual._vispruner_state = state

    def clear_before_forward(*_args: Any, **_kwargs: Any) -> None:
        state["scores"].clear()

    # A hook on the visual module itself clears stale data before every image
    # batch, including the disabled path.
    state["handles"].append(visual.register_forward_pre_hook(clear_before_forward))

    for layer_idx in cfg.score_layers:
        if layer_idx < 0 or layer_idx >= len(visual.blocks):
            raise ValueError(
                f"vispruner score layer {layer_idx} is outside visual depth "
                f"{len(visual.blocks)}"
            )
        module = visual.blocks[layer_idx].attn
        is_flash = module.__class__.__name__ == "Qwen2_5_VLVisionFlashAttention2"

        if needs_attn_matrix(cfg.score_reduce):

            def pre_hook(
                hooked_module: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                block_idx: int = layer_idx,
                flash_rope: bool = is_flash,
            ) -> None:
                if not cfg.enabled or not args:
                    return None
                cu_seqlens = kwargs.get("cu_seqlens")
                if cu_seqlens is None:
                    return None
                with torch.no_grad():
                    attn_parts = _segment_attention_scores(
                        hooked_module,
                        args[0],
                        cu_seqlens,
                        kwargs.get("position_embeddings"),
                        flash_rope=flash_rope,
                    )
                    state["scores"][block_idx] = {
                        "attn": [part.detach() for part in attn_parts],
                        "attn_output": None,
                    }
                return None

            state["handles"].append(
                module.register_forward_pre_hook(pre_hook, with_kwargs=True)
            )
        elif needs_attn_output(cfg.score_reduce):

            def output_hook(
                hooked_module: Any,
                _args: tuple[Any, ...],
                output: torch.Tensor,
                *,
                block_idx: int = layer_idx,
            ) -> None:
                if not cfg.enabled:
                    return None
                # The output is in the same window-reordered sequence as the
                # attention input.  Planner alignment restores image order.
                state["scores"][block_idx] = {
                    "attn": None,
                    "attn_output": output.detach(),
                }
                return None

            state["handles"].append(module.register_forward_hook(output_hook))


def pop_scores(visual: torch.nn.Module) -> dict[int, dict[str, Any]]:
    """Return and clear scores collected by the most recent visual forward."""

    state = getattr(visual, "_vispruner_state", None)
    if state is None:
        return {}
    scores = state["scores"]
    state["scores"] = {}
    return scores


__all__ = ["install_hooks", "pop_scores"]
