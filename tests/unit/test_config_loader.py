"""
Unit tests: src/utils/config_loader.py.
"""

from src.utils.config_loader import load_config


def test_load_config_returns_a_dict():
    config = load_config()

    assert isinstance(config, dict)


def test_load_config_has_the_keys_the_rest_of_src_relies_on():
    config = load_config()

    for key in (
        "model_path",
        "encoder_path",
        "scaler_path",
        "imputer_path",
        "feature_list_path",
        "mlflow",
        "api",
        "logging",
    ):
        assert key in config, f"config.yaml is missing top-level key '{key}'"


def test_mlflow_block_has_the_keys_model_registry_relies_on():
    config = load_config()
    mlflow_cfg = config["mlflow"]

    for key in (
        "tracking_uri",
        "experiment_name",
        "registered_model_name",
        "model_alias",
        "model_stage",
    ):
        assert key in mlflow_cfg, f"config.yaml's mlflow: block is missing '{key}'"


def test_api_port_is_an_integer():
    config = load_config()

    assert isinstance(config["api"]["port"], int)
