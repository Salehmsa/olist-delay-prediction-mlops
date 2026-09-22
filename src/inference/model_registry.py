"""
Single place that resolves "the current production model" against MLflow.

predict.py, preprocessing.py, pipeline.py, and app/main.py all import
MODEL_VERSION_INFO from here instead of each querying the registry
separately — one MLflow Model Registry lookup at startup, not four, and one
source of truth for which run's artifacts back the currently-loaded model
(preprocessing.py needs that run's run_id to pull the matching
encoder/scaler/imputer bundle — see src/training/log_model.py, which logs
the model and its preprocessing bundle to the same run on purpose, so the
two can never drift apart).
"""

import os

# Quiet two cosmetic-only MLflow behaviors, set before mlflow is imported so
# they take effect everywhere in the process (predict.py's load_model() and
# preprocessing.py's artifact download both run after this module):
# a tqdm progress bar on every artifact download (fine interactively, just
# log noise inside a container or CI run), and an unrelated hint suggesting
# an LLM-tracing skill that doesn't apply to this project.
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from src.utils.config_loader import load_config  # noqa: E402

config = load_config()
mlflow_config = config["mlflow"]

mlflow.set_tracking_uri(mlflow_config["tracking_uri"])

REGISTERED_MODEL_NAME = mlflow_config["registered_model_name"]
MODEL_ALIAS = mlflow_config["model_alias"]

_client = MlflowClient()

try:
    _model_version = _client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME, MODEL_ALIAS
    )
except MlflowException as exc:
    raise RuntimeError(
        f"No model found at models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS} "
        f"(tracking_uri={mlflow_config['tracking_uri']}). Run "
        "`python -m src.training.log_model` first to register one."
    ) from exc

MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"

MODEL_VERSION_INFO = {
    "model_name": REGISTERED_MODEL_NAME,
    "model_version": _model_version.version,
    "stage": _model_version.current_stage,
    "run_id": _model_version.run_id,
}
