"""
Unit tests: src/inference/input_builder.py — build_input_dataframe().
"""

import pandas as pd

from src.inference.input_builder import build_input_dataframe
from src.schemas.request_schema import OrderRequest

RAW_COLUMNS = [
    "customer_zip_code_prefix",
    "total_payment",
    "avg_payment",
    "max_installments",
    "payment_count",
    "items_count",
    "total_price",
    "total_freight",
    "unique_products",
    "unique_sellers",
    "review_score",
    "approval_delay_hours",
    "order_status",
    "customer_state",
]


def test_returns_single_row_dataframe(valid_order_request):
    df = build_input_dataframe(valid_order_request)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


def test_has_exactly_the_raw_columns_preprocessing_expects(valid_order_request):
    df = build_input_dataframe(valid_order_request)

    assert set(df.columns) == set(RAW_COLUMNS)
    # review_missing is deliberately NOT here - feature_engineering.py adds
    # it downstream. A regression here would mean preprocess() silently
    # gets a column it didn't expect from this function.
    assert "review_missing" not in df.columns


def test_values_round_trip_from_the_request(valid_order_kwargs, valid_order_request):
    df = build_input_dataframe(valid_order_request)

    assert df.loc[0, "total_payment"] == valid_order_kwargs["total_payment"]
    assert df.loc[0, "order_status"] == valid_order_kwargs["order_status"]
    assert df.loc[0, "customer_state"] == valid_order_kwargs["customer_state"]


def test_missing_optional_numeric_fields_become_nan():
    """
    Every numeric field is Optional in OrderRequest - a brand-new order with
    unknown payment/freight/review details must still produce a row (NaN,
    not a crash), because apply_imputer() downstream is what's supposed to
    handle this, not this function.
    """

    request = OrderRequest(order_status="delivered", customer_state="SP")
    df = build_input_dataframe(request)

    assert df["total_payment"].isna().all()
    assert df["review_score"].isna().all()
    assert df.loc[0, "order_status"] == "delivered"
