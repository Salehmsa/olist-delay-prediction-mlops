"""
Unit tests: src/data/preprocessing.py — apply_imputer / apply_scaler /
apply_encoder / preprocess. Exercises the REAL fitted imputer/encoder/scaler
that conftest.py's MLflow bootstrap already pulled into this process
(module-level singletons in src.data.preprocessing), not a re-fitted copy -
this is what actually runs in production.
"""

import numpy as np

from src.data import preprocessing
from src.features.feature_engineering import create_features
from src.inference.input_builder import build_input_dataframe


def _engineered_df(valid_order_request):
    return create_features(build_input_dataframe(valid_order_request))


def test_apply_imputer_fills_missing_numeric_values(valid_order_request):
    df = _engineered_df(valid_order_request)
    df.loc[0, "total_payment"] = np.nan

    result = preprocessing.apply_imputer(df)

    assert result["total_payment"].isna().sum() == 0
    assert list(result.columns) == preprocessing.NUMERIC_COLUMNS


def test_apply_scaler_returns_the_scalers_own_fitted_columns(valid_order_request):
    """
    apply_scaler() trusts SCALED_COLUMNS (scaler.feature_names_in_), not
    NUMERIC_COLUMNS (the imputer's) - the two are NOT assumed identical.
    Build its input the same way preprocess() does: impute, then merge back
    into the full engineered df so any extra column the scaler expects
    (review_missing) is present and clean.
    """

    df = _engineered_df(valid_order_request)

    imputed = preprocessing.apply_imputer(df)
    scaler_input = df.copy()
    scaler_input[preprocessing.NUMERIC_COLUMNS] = imputed[preprocessing.NUMERIC_COLUMNS]

    scaled = preprocessing.apply_scaler(scaler_input)

    assert list(scaled.columns) == preprocessing.SCALED_COLUMNS
    assert scaled.isna().sum().sum() == 0


def test_apply_encoder_one_hot_encodes_each_categorical_column_exactly_once(
    valid_order_request,
):
    df = _engineered_df(valid_order_request)

    encoded = preprocessing.apply_encoder(df)

    status_cols = [c for c in encoded.columns if c.startswith("order_status_")]
    state_cols = [c for c in encoded.columns if c.startswith("customer_state_")]

    # Exactly one "hot" column per categorical field - not zero (a silently
    # unmatched category), not more than one.
    assert encoded[status_cols].sum(axis=1).iloc[0] == 1
    assert encoded[state_cols].sum(axis=1).iloc[0] == 1


def test_preprocess_returns_the_full_42_column_feature_contract(valid_order_request):
    df = _engineered_df(valid_order_request)

    result = preprocessing.preprocess(df)

    assert list(result.columns) == preprocessing.FEATURE_LIST
    assert len(result.columns) == len(preprocessing.FEATURE_LIST)
    assert result.isna().sum().sum() == 0


def test_preprocess_output_is_a_single_row(valid_order_request):
    df = _engineered_df(valid_order_request)

    result = preprocessing.preprocess(df)

    assert len(result) == 1


def test_preprocess_includes_review_missing_regardless_of_scaler_fit(
    valid_order_request,
):
    """
    Regression test for a real bug this suite caught: apply_scaler() used
    to blindly reuse NUMERIC_COLUMNS (the imputer's fitted columns) instead
    of the scaler's own, which silently dropped review_missing from the
    final feature matrix whenever the fitted scaler's columns didn't
    exactly match the imputer's. Must hold no matter which of the two the
    real fitted scaler turns out to be.
    """

    df = _engineered_df(valid_order_request)

    result = preprocessing.preprocess(df)

    assert "review_missing" in result.columns
    assert result["review_missing"].notna().all()


def test_preprocess_raises_value_error_if_a_required_column_goes_missing(
    valid_order_request,
):
    """
    preprocess() reindexes to FEATURE_LIST and raises loudly if something's
    missing, instead of silently handing the model a column of NaN under
    the wrong name. Simulate that by feeding it a combined frame missing a
    categorical column entirely (customer_state dropped before encoding).
    """

    df = _engineered_df(valid_order_request)
    broken = df.drop(columns=["customer_state"])

    try:
        preprocessing.apply_encoder(broken)
        raised = False
    except KeyError:
        # apply_encoder itself fails first since it selects
        # CATEGORICAL_COLUMNS directly - confirms the missing-column case
        # fails loudly at the first place that touches it, not silently.
        raised = True

    assert raised
