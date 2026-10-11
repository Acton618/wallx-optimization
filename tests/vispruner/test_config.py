import pytest

from wall_x.model.qact.qwen2_5.vispruner.config import VisPrunerConfig
from wall_x.model.qact.qwen2_5.optimization_flags import InferenceOptimizationFlags


def test_vispruner_is_disabled_by_default():
    config = VisPrunerConfig()
    assert config.enable is False
    assert config.enabled is False


def test_config_reads_optional_fields_from_mapping():
    config = VisPrunerConfig.from_model_config(
        {
            "vispruner_enable": True,
            "vispruner_score_layers": [31, 23],
            "vispruner_score_layer": 23,
            "vispruner_keep_num": 40,
        }
    )
    assert config.enabled is True
    assert config.score_layer == 23
    assert config.keep_num == 40


def test_other_experimental_features_are_disabled_by_default():
    flags = InferenceOptimizationFlags()
    assert flags == InferenceOptimizationFlags(False, False, False)


def test_ode_master_switch_is_read_from_model_config():
    flags = InferenceOptimizationFlags.from_model_config(
        {"ode_optimization_enable": True}
    )
    assert flags.ode_optimization_enable is True


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        VisPrunerConfig(score_reduce="unknown")
    with pytest.raises(ValueError):
        VisPrunerConfig(score_layer=7)
