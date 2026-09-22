"""
Qafza's Definition of Done, "لازم تقدر" (must be able to): "تثبت إن مخرج
البايبلاين مطابق لمخرج النوتبوك على نفس المدخل" - prove the serving
pipeline's output matches the notebook's output for the same input.

Notebook 06 itself never engineers features - it loads already-engineered
train/validation/test_features.parquet and calls rf_model.predict(X_test) /
rf_model.predict_proba(X_test) directly on them (its final cells, right
before joblib.dump(rf_model, "artifacts/final_model.pkl")). The feature
engineering that PRODUCES that 42-column matrix in the first place is
Notebook 05's job - mirrored in this repo by
src/features/feature_engineering.py (review_missing) and
src/data/preprocessing.py (impute -> scale -> encode -> reindex), both
built on the exact fitted imputer/encoder/scaler/model this repo ships
under models/*.pkl (config.yaml: "Fitted artifacts saved by Notebooks 05 &
06" - log_model.py uploads them into MLflow verbatim, it does not refit
anything: "No training inside the inference pipeline").

So structurally, "the notebook's output" for a given raw order reduces to:
the SAME fitted model, called on a feature row built from the SAME fitted
imputer/encoder/scaler - wired together by hand, the way a notebook cell
would, not through this project's own orchestration code
(create_features() / preprocess()).

_hand_wired_prediction() below builds exactly that second path,
independently of src/features/feature_engineering.py's and
src/data/preprocessing.py's own function bodies: it imports only the
fitted objects themselves (model, imputer, encoder, scaler) plus each
one's OWN metadata (*.feature_names_in_, and model.feature_names_in_ for
the final column order/set - not preprocessing.py's FEATURE_LIST constant
or feature_list.json), then does impute -> scale -> encode -> reindex ->
predict here, from scratch. The one piece of actual derivation logic it
cannot pull from fitted-object metadata is review_missing =
review_score.isna() - feature_engineering.py's own docstring attributes
that exact rule to Notebook 05, so it is repeated here rather than
imported.

If run_pipeline() (the real code /predict calls) ever disagrees with this
hand-wired reconstruction for the same input, that disagreement IS a
pipeline/notebook mismatch by definition: both paths load the identical
trained artifacts (same MLflow-registered run - src/inference/model_registry.py
resolves both to the same run_id), so the only thing that CAN differ
between them is a bug in the orchestration code wrapping those artifacts -
exactly the "training-serving skew" class of bug this proof exists to
catch.

HONEST SCOPE - what this does NOT prove: that review_missing's derivation,
or the imputer/encoder/scaler's fitted parameters themselves, are faithful
to Notebook 05's ORIGINAL computation. Notebook 05 was not available when
this test was written - only Notebook 06 was (see README's "Model
provenance" section). What IS independently confirmed (see README): the
shipped models/final_model.pkl loads as RandomForestClassifier(
n_estimators=100, random_state=42, n_jobs=-1) with n_features_in_=42 and
classes_=[0, 1] - an exact match to Notebook 06's own cells 10 and 27 -
and src/training/log_model.py's KNOWN_METRICS (roc_auc=0.6105108647035242
down to the last digit) matches Notebook 06's own printed test-set
evaluation character for character, so this is genuinely that notebook's
model, not a stand-in.
"""

import pandas as pd
import pytest

from src.data.preprocessing import encoder, imputer, scaler
from src.inference.input_builder import build_input_dataframe
from src.inference.pipeline import model, run_pipeline
from src.schemas.request_schema import OrderRequest


def _hand_wired_prediction(order: OrderRequest):
    """
    "If you had the fitted objects and did this by hand in a notebook
    cell" - independent of create_features() / preprocess()'s own code.
    Returns (prediction, probability), the same shape run_pipeline() does.
    """

    df = build_input_dataframe(order).copy()

    # The one derivation rule not recoverable from fitted-object metadata
    # alone - feature_engineering.py attributes it to Notebook 05.
    df["review_missing"] = df["review_score"].isna().astype(float)

    numeric_columns = list(imputer.feature_names_in_)
    df[numeric_columns] = imputer.transform(df[numeric_columns])

    scaled_columns = list(scaler.feature_names_in_)
    scaled_df = pd.DataFrame(
        scaler.transform(df[scaled_columns]), columns=scaled_columns, index=df.index
    )

    categorical_columns = list(encoder.feature_names_in_)
    encoded = encoder.transform(df[categorical_columns])
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_columns),
        index=df.index,
    )

    combined = pd.concat([scaled_df, encoded_df], axis=1)

    # Ground truth for the final column order/set: the fitted MODEL's own
    # feature_names_in_ - not preprocessing.py's FEATURE_LIST constant or
    # feature_list.json, so this path shares no code and no config file
    # with the module under test.
    model_ready = combined[list(model.feature_names_in_)]

    prediction = model.predict(model_ready)
    probability = model.predict_proba(model_ready)[:, 1]

    return int(prediction[0]), float(probability[0])


def test_pipeline_matches_hand_wired_notebook_style_computation_full_order(
    valid_order_kwargs,
):
    """A fully-specified, realistic order - the project's standard sample."""

    order = OrderRequest(**valid_order_kwargs)

    pipeline_prediction, pipeline_probability = run_pipeline(
        order, source="notebook_parity_test"
    )
    notebook_prediction, notebook_probability = _hand_wired_prediction(order)

    assert pipeline_prediction == notebook_prediction
    assert pipeline_probability == pytest.approx(notebook_probability, abs=1e-12)


def test_pipeline_matches_hand_wired_notebook_style_computation_sparse_order():
    """
    Every optional field missing, including review_score - the imputer (and
    review_missing=1.0) do real work here, unlike the fully-specified case
    above, so both paths are also cross-checked through an actual impute
    step, not just pass-through values.
    """

    order = OrderRequest(order_status="delivered", customer_state="SP")

    pipeline_prediction, pipeline_probability = run_pipeline(
        order, source="notebook_parity_test"
    )
    notebook_prediction, notebook_probability = _hand_wired_prediction(order)

    assert pipeline_prediction == notebook_prediction
    assert pipeline_probability == pytest.approx(notebook_probability, abs=1e-12)
