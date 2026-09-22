"""
Model tests: the model loads, predicts the right shape, behaves on known
inputs. "Known inputs" here means schema-valid synthetic inputs with
asserted well-formed outputs, not ground-truth-labeled examples - this
project has no held-out labeled test set available to this task (the real
evaluation lives in Notebook 06, not copied into this repo). See
test_pipeline_e2e.py (integration/) for the same model exercised through
the full request -> prediction path.
"""

import numpy as np

from src.data.preprocessing import preprocess
from src.features.feature_engineering import create_features
from src.inference.input_builder import build_input_dataframe
from src.inference.predict import predict
from src.schemas.request_schema import OrderRequest


def _model_ready_row(**overrides):
    kwargs = dict(
        customer_zip_code_prefix=12345,
        total_payment=150.0,
        avg_payment=75.0,
        max_installments=3,
        payment_count=2,
        items_count=2,
        total_price=130.0,
        total_freight=20.0,
        unique_products=2,
        unique_sellers=1,
        review_score=5,
        approval_delay_hours=3,
        order_status="delivered",
        customer_state="SP",
    )
    kwargs.update(overrides)
    request = OrderRequest(**kwargs)

    return preprocess(create_features(build_input_dataframe(request)))


def test_model_loads(model):
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_model_is_a_binary_classifier(model):
    """
    predict.py's predict_proba(df)[:, 1] assumes exactly two classes with
    class 1 as the positive one it reports. If a future retrain ever
    produces a model with a different number of classes, this must fail
    loudly here instead of silently mis-indexing probability[:, 1].
    """

    assert len(model.classes_) == 2


def test_predict_returns_one_prediction_and_one_probability_per_row(model):
    df = _model_ready_row()

    prediction, probability = predict(model, df)

    assert len(prediction) == 1
    assert len(probability) == 1


def test_predicted_class_is_one_of_the_models_known_classes(model):
    df = _model_ready_row()

    prediction, _ = predict(model, df)

    assert prediction[0] in model.classes_


def test_probability_is_a_finite_value_between_0_and_1(model):
    df = _model_ready_row()

    _, probability = predict(model, df)

    assert np.isfinite(probability[0])
    assert 0.0 <= probability[0] <= 1.0


def test_same_input_twice_gives_identical_output(model):
    """
    Determinism sanity check - a model that isn't seeded/stateful
    consistently would be a real production bug (two identical requests
    getting different answers), not just a test-flakiness annoyance.
    """

    df = _model_ready_row()

    prediction_1, probability_1 = predict(model, df)
    prediction_2, probability_2 = predict(model, df)

    assert prediction_1[0] == prediction_2[0]
    assert probability_1[0] == probability_2[0]


def test_model_handles_a_request_with_every_optional_field_missing(model):
    """
    The imputer-heavy path: a brand-new order where every numeric field is
    unknown must still reach the model and produce a well-formed output,
    not a NaN-propagation crash.
    """

    request = OrderRequest(order_status="canceled", customer_state="RJ")
    df = preprocess(create_features(build_input_dataframe(request)))

    prediction, probability = predict(model, df)

    assert prediction[0] in model.classes_
    assert np.isfinite(probability[0])
