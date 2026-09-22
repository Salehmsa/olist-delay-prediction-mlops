"""
Section 8: idempotent registration entrypoint for docker-compose's
`register` service.

Same bootstrap-only-if-missing check tests/conftest.py already uses
(Section 6): `docker compose up` can be run again later (a restart, or a
second `up` after `down` without `-v`) without spamming a new, pointless
registry version every time one already exists from an earlier run against
the same mlflow_data volume.

Only ever calls log_and_register(allow_placeholder=True) - this script's
whole job is a safe, unattended default for first boot, not a replacement
for the manual "fill in KNOWN_METRICS, then run
`python -m src.training.log_model` without the flag" step the README
documents for real evaluation numbers.

Usage (this is what docker-compose.yml's `register` service runs):

    python scripts/register_if_needed.py
"""

import sys

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.training.log_model import log_and_register
from src.utils.config_loader import load_config
from src.utils.logger import logger

config = load_config()["mlflow"]
mlflow.set_tracking_uri(config["tracking_uri"])

try:
    MlflowClient().get_model_version_by_alias(
        config["registered_model_name"], config["model_alias"]
    )
    logger.info(
        "models:/%s@%s already registered - skipping (the mlflow_data "
        "volume already has a production model from an earlier "
        "`docker compose up`). Delete the volume, or run "
        "`python -m src.training.log_model` yourself inside the api "
        "container, to register a new version on purpose.",
        config["registered_model_name"],
        config["model_alias"],
    )
    sys.exit(0)
except MlflowException:
    pass

log_and_register(allow_placeholder=True)
