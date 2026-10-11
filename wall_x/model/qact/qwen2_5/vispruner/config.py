"""Configuration for the optional VisPruner path.

This module deliberately does not modify the Wall-X model configuration.  The
model integration will read these fields through :meth:`from_model_config` in
a later stage, while the default remains disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


_SCORE_REDUCERS = {
    "attn_mean",
    "attn_max",
    "attn_out_norm",
    "random",
    "norm",
}


def _read(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


@dataclass(frozen=True)
class VisPrunerConfig:
    """Runtime-only VisPruner settings.

    ``enable`` is the sole switch controlling whether the future model
    integration is active.  In particular, ``keep_num`` must not implicitly
    enable or disable the path: ``keep_num == N`` is used to test the full
    sequence-rebuild path without removing tokens.
    """

    enable: bool = False
    score_layers: tuple[int, ...] = (31, 23)
    score_layer: int = 31
    score_reduce: str = "attn_max"
    keep_num: int = 20
    important_ratio: float = 0.5
    seed: int = 0

    def __post_init__(self) -> None:
        layers = tuple(int(layer) for layer in self.score_layers)
        object.__setattr__(self, "score_layers", layers)
        if not layers:
            raise ValueError("score_layers must contain at least one layer")
        if self.score_layer not in layers:
            raise ValueError("score_layer must be included in score_layers")
        if self.score_reduce not in _SCORE_REDUCERS:
            choices = ", ".join(sorted(_SCORE_REDUCERS))
            raise ValueError(f"unknown score_reduce={self.score_reduce!r}; use {choices}")
        if self.keep_num < 1:
            raise ValueError("keep_num must be positive")
        if not 0.0 <= self.important_ratio <= 1.0:
            raise ValueError("important_ratio must be in [0, 1]")

    @property
    def enabled(self) -> bool:
        """Alias used by integration code without changing the public switch."""

        return bool(self.enable)

    @classmethod
    def from_model_config(cls, model_config: Any) -> "VisPrunerConfig":
        """Read optional ``vispruner_*`` fields from a config-like object."""

        raw_layers = _read(model_config, "vispruner_score_layers", cls.score_layers)
        if isinstance(raw_layers, int):
            raw_layers = (raw_layers,)
        else:
            raw_layers = tuple(raw_layers)

        return cls(
            enable=bool(_read(model_config, "vispruner_enable", cls.enable)),
            score_layers=raw_layers,
            score_layer=int(
                _read(model_config, "vispruner_score_layer", cls.score_layer)
            ),
            score_reduce=str(
                _read(model_config, "vispruner_score_reduce", cls.score_reduce)
            ),
            keep_num=int(_read(model_config, "vispruner_keep_num", cls.keep_num)),
            important_ratio=float(
                _read(
                    model_config,
                    "vispruner_important_ratio",
                    cls.important_ratio,
                )
            ),
            seed=int(_read(model_config, "vispruner_seed", cls.seed)),
        )


__all__ = ["VisPrunerConfig"]
