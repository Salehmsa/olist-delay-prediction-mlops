"""
Model tests: src/inference/model_registry.py — the model resolves through
the registry correctly (not that it predicts well; that's test_predict.py).
"""

from src.inference.model_registry import (
    MODEL_ALIAS,
    MODEL_URI,
    MODEL_VERSION_INFO,
    REGISTERED_MODEL_NAME,
)
from src.utils.config_loader import load_config


def test_model_version_info_has_the_expected_shape():
    for key in ("model_name", "model_version", "stage", "run_id"):
        assert key in MODEL_VERSION_INFO
        assert MODEL_VERSION_INFO[key] not in (None, "")


def test_registered_name_matches_config():
    config = load_config()

    assert REGISTERED_MODEL_NAME == config["mlflow"]["registered_model_name"]
    assert MODEL_VERSION_INFO["model_name"] == config["mlflow"]["registered_model_name"]


def test_model_uri_is_the_alias_form_config_expects():
    config = load_config()
    expected_alias = config["mlflow"]["model_alias"]

    assert MODEL_ALIAS == expected_alias
    assert MODEL_URI == f"models:/{REGISTERED_MODEL_NAME}@{expected_alias}"


def test_model_version_is_a_positive_integer_when_parsed():
    assert int(MODEL_VERSION_INFO["model_version"]) >= 1
