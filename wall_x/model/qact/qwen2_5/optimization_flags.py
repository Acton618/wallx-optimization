"""Central switches for experimental inference features.

All switches default to ``False``.  The model integration will consume these
flags in later stages; keeping them together makes rollback explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _read(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


@dataclass(frozen=True)
class InferenceOptimizationFlags:
    vispruner_enable: bool = False
    predictor_enable: bool = False
    ode_optimization_enable: bool = False

    @classmethod
    def from_model_config(cls, config: Any) -> "InferenceOptimizationFlags":
        return cls(
            vispruner_enable=bool(_read(config, "vispruner_enable", False)),
            predictor_enable=bool(_read(config, "predictor_enable", False)),
            ode_optimization_enable=bool(
                _read(config, "ode_optimization_enable", False)
            ),
        )


__all__ = ["InferenceOptimizationFlags"]
