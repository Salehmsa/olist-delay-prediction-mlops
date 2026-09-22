import os
import yaml
from pathlib import Path


def load_config():

    config_file = Path(__file__).parents[2] / "config" / "config.yaml"

    with open(config_file, "r", encoding="utf-8") as file:

        config = yaml.safe_load(file)

    # Section 8: the one setting that legitimately differs between "running
    # on your laptop" (config.yaml's local sqlite:///mlflow.db) and "running
    # in a container" (docker-compose's own mlflow service, reached over the
    # network at http://mlflow:5000, its data in a Docker-managed volume -
    # not a path on this laptop). Everything else in config.yaml stays the
    # same in both places, so only this one value gets an override, and only
    # when it's actually set - unset locally, so local behavior is unchanged.
    tracking_uri_override = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri_override:
        config["mlflow"]["tracking_uri"] = tracking_uri_override

    return config
