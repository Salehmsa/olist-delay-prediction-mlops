"""
Section 5: log the already-trained model (from Notebooks 05/06) into MLflow
- tracking (params, metrics, artifacts), registry (a version), and a
"production" alias/stage so the service can load it without ever touching
models/*.pkl directly again.

This does NOT train anything ("No training inside the inference pipeline" -
Task 3, Notes). It loads what Notebook 06 already fit and saved, and hands
it to MLflow. Re-run it any time you swap in a new models/*.pkl from a fresh
notebook run - each run creates a new registry version, and the alias below
moves to point at whichever version you last ran this against.

Usage (from the project root, inside the activated environment):

    python -m src.training.log_model

    # or, to unblock immediately without the real numbers on hand yet:
    python -m src.training.log_model --allow-placeholder
"""

import argparse
import os
import warnings

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib  # noqa: E402
import mlflow  # noqa: E402

from src.utils.config_loader import load_config
from src.utils.logger import logger

# -----------------------------------------------------------------------------
# Real numbers from Notebook 06's final evaluation - classification_report(
# y_test, best_pred) + roc_auc_score(y_test, best_prob) - on the held-out
# test set (n=14472). This is what "you can explain your metric" (Task 3,
# Where we are now) means in practice: MLflow shows the metric the model
# was actually picked on, not a placeholder.
#
# is_delayed is heavily imbalanced (957 delayed vs 13515 on-time, ~6.6%
# positive rate - Notebook 02's label definition: 1 = Delayed, 0 = On Time,
# confirmed in src/inference/predict.py's own docstring). That imbalance is
# exactly why precision/recall/f1 below are the MACRO average, not accuracy
# or the weighted average: weighted avg (0.90/0.93/0.91) is dominated by
# the 13515 easy negatives and would hide how the model does on the class
# that actually matters here; macro avg (0.63/0.54/0.55) weighs both
# classes equally, so it doesn't let the majority class launder the
# minority-class result.
#
# The three *_delayed entries go further and log the delayed-class (1) row
# on its own, unaveraged - the single most business-relevant number for a
# delay predictor: of orders the model FLAGS as delayed, only 31% actually
# are (precision_delayed), and it only CATCHES 8% of the delays that
# actually happen (recall_delayed). accuracy=0.93 alone would hide this
# completely - the textbook trap of reporting accuracy on imbalanced data.
# roc_auc=0.61 (barely above 0.50 random) corroborates the same story from
# a threshold-independent angle. Worth stating plainly in any write-up of
# this project: strong on Section 8's engineering, weak on Notebook 06's
# actual predictive power for the event that matters - a real limitation,
# not a deployment bug, and out of scope for Task 3 to fix (no training
# inside the inference pipeline).
# -----------------------------------------------------------------------------
KNOWN_METRICS = {
    "accuracy": 0.93,
    "precision": 0.63,  # macro avg
    "recall": 0.54,  # macro avg
    "f1": 0.55,  # macro avg
    "roc_auc": 0.6105108647035242,
    "precision_delayed": 0.31,  # class 1 (is_delayed) only
    "recall_delayed": 0.08,  # class 1 (is_delayed) only
    "f1_delayed": 0.13,  # class 1 (is_delayed) only
}

_PLACEHOLDER_METRICS = {key: 0.0 for key in KNOWN_METRICS}

_LOGGABLE_PARAM_TYPES = (str, int, float, bool, type(None))


def _loggable_params(model) -> dict:
    """
    model.get_params() for most sklearn estimators, filtered to values
    MLflow can actually store as params (simple scalars). A nested object
    (e.g. a base_estimator inside an ensemble) is skipped rather than
    crashing the run - MLflow params are strings/scalars, not objects.
    """

    if not hasattr(model, "get_params"):
        return {"model_type": type(model).__name__}

    params = {
        key: value
        for key, value in model.get_params().items()
        if isinstance(value, _LOGGABLE_PARAM_TYPES)
    }
    params["model_type"] = type(model).__name__

    return params


def log_and_register(allow_placeholder: bool = False) -> dict:
    config = load_config()
    mlflow_config = config["mlflow"]

    metrics = KNOWN_METRICS
    # Whether placeholder metrics actually get used - independent of the
    # allow_placeholder ARGUMENT. register_if_needed.py always calls this
    # with allow_placeholder=True (Section 8) as a safe unattended default,
    # but now that KNOWN_METRICS is fully filled in, that call logs the
    # real numbers below, not zeros. This flag (not the argument) is what
    # the metrics_source tag reflects, so the tag never claims "placeholder"
    # while real numbers are sitting right next to it.
    using_placeholder = any(value is None for value in metrics.values())
    if using_placeholder:
        if not allow_placeholder:
            missing = [key for key, value in metrics.items() if value is None]
            raise SystemExit(
                f"KNOWN_METRICS has unfilled value(s) for {missing}. Open "
                "Notebook 06's evaluation output and fill in the real "
                "numbers at the top of src/training/log_model.py, or "
                "re-run with --allow-placeholder to log clearly-tagged "
                "placeholder metrics instead."
            )
        metrics = _PLACEHOLDER_METRICS
        logger.warning(
            "Logging PLACEHOLDER metrics (--allow-placeholder). Replace "
            "KNOWN_METRICS in src/training/log_model.py with real values "
            "from Notebook 06 and re-run when you have them."
        )

    model = joblib.load(config["model_path"])

    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])

    registered_name = mlflow_config["registered_model_name"]

    with mlflow.start_run(run_name="notebook_05_06_model") as run:
        mlflow.set_tags(
            {
                "source": "Notebooks 05-06 (trained offline, not by this script)",
                "metrics_source": (
                    "placeholder" if using_placeholder else "notebook_06_evaluation"
                ),
            }
        )

        mlflow.log_params(_loggable_params(model))
        mlflow.log_metrics(metrics)

        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=registered_name,
            # MLflow 3.x serializes sklearn models with skops, which by
            # default refuses to save any type it doesn't already know is
            # safe - including sklearn.tree._tree.Tree, the node-storage
            # object every DecisionTree/RandomForest/ExtraTrees/
            # GradientBoosting model uses internally. That's a real
            # precaution against a maliciously crafted *downloaded* model
            # file, not a signal that anything is wrong with yours: this is
            # the model Notebook 06 trained and models/final_model.pkl
            # already held before this script touched it, so trusting it
            # here is safe. Only needed at save time - load_model() in
            # predict.py needs no equivalent flag.
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        # Logged to the SAME run as the model, under preprocessing/ - so the
        # registered version and its exact encoder/scaler/imputer/feature
        # list can never drift apart. preprocessing.py downloads this whole
        # folder at startup using the run_id below.
        for artifact_key in (
            "encoder_path",
            "scaler_path",
            "imputer_path",
            "feature_list_path",
        ):
            mlflow.log_artifact(config[artifact_key], artifact_path="preprocessing")

        run_id = run.info.run_id

    version = model_info.registered_model_version
    client = mlflow.MlflowClient()

    # Modern API - what the service actually loads through
    # (models:/<name>@<alias>, config.yaml's mlflow.model_alias).
    client.set_registered_model_alias(
        name=registered_name,
        alias=mlflow_config["model_alias"],
        version=version,
    )

    # Classic "stage" - the exact word Task 3 uses ("a version and a
    # stage"). Deprecated since MLflow 2.9 in favor of the alias above, but
    # still functional, so it's set too for anyone reviewing this under the
    # older vocabulary. The running service reads the alias, not this.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        client.transition_model_version_stage(
            name=registered_name,
            version=version,
            stage=mlflow_config["model_stage"],
        )

    logger.info(
        "Registered %s v%s (run_id=%s) as alias '%s' / stage '%s'",
        registered_name,
        version,
        run_id,
        mlflow_config["model_alias"],
        mlflow_config["model_stage"],
    )

    print(f"Registered {registered_name} v{version} (run_id={run_id})")
    print(f"  alias: {mlflow_config['model_alias']}")
    print(f"  stage: {mlflow_config['model_stage']}")
    print(
        "  MLflow UI: mlflow ui --backend-store-uri "
        f"{mlflow_config['tracking_uri']}"
    )

    return {"registered_name": registered_name, "version": version, "run_id": run_id}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-placeholder",
        action="store_true",
        help="Log clearly-tagged placeholder metrics instead of refusing to run.",
    )
    args = parser.parse_args()

    log_and_register(allow_placeholder=args.allow_placeholder)
