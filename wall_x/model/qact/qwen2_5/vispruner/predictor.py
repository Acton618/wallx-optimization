"""Small visual-token scorer and its checkpoint loader."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class TokenScorePredictor(nn.Module):
    """Match the two-layer MLP used by the original Wall-X experiment."""

    def __init__(self, input_dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim or max(1, input_dim // 4))
        self.dropout = float(dropout)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        return self.net(token_features).squeeze(-1)

    def config_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
        }


def load_predictor(path: str, *, input_dim: int, early_layer: int) -> TokenScorePredictor:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("predictor checkpoint must contain model_state_dict and metadata")
    cfg = checkpoint.get("config") or {}
    meta = checkpoint.get("meta") or {}
    if int(cfg.get("input_dim", -1)) != input_dim:
        raise ValueError(f"predictor input_dim must be {input_dim}, got {cfg.get('input_dim')}")
    if meta.get("feature_source") != "early_hidden" or int(meta.get("early_layer", -1)) != early_layer:
        raise ValueError(
            "predictor checkpoint must be trained on early_hidden features "
            f"from layer {early_layer} of this vision tower"
        )
    model = TokenScorePredictor(**cfg)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


__all__ = ["TokenScorePredictor", "load_predictor"]
