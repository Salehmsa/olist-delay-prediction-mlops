"""
Data tests: src/validation/expectations.py + data_validation.py — schema,
ranges, and nulls, enforced by the Great Expectations suite. Tested against
raw DataFrames built directly here (NOT through OrderRequest/Pydantic), on
purpose: this suite must be proven correct as its own independent layer, not
only reachable through a client that already blocks most of these cases
earlier. See test_request_schema.py for the Pydantic-layer tests, and
test_pipeline_e2e.py (integration/) for a case that only THIS layer catches.
"""

import pandas as pd
import pytest

from src.validation.data_validation import DataValidationError, validate_request_data

VALID_ROW = {
    "customer_zip_code_prefix": 12345.0,
    "total_payment": 150.0,
    "avg_payment": 75.0,
    "max_installments": 3.0,
    "payment_count": 2.0,
    "items_count": 2.0,
    "total_price": 130.0,
    "total_freight": 20.0,
    "unique_products": 2.0,
    "unique_sellers": 1.0,
    "review_score": 5.0,
    "approval_delay_hours": 3.0,
    "order_status": "delivered",
    "customer_state": "SP",
}


def _row(**overrides):
    data = {**VALID_ROW, **overrides}
    return pd.DataFrame({k: [v] for k, v in data.items()})


def test_valid_row_passes():
    validate_request_data(_row())  # does not raise


def test_null_review_score_is_allowed():
    """
    review_missing exists precisely because a null review_score is normal,
    not an error - the suite must not flag it.
    """

    validate_request_data(_row(review_score=None))


@pytest.mark.parametrize(
    "column",
    [
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
        "approval_delay_hours",
    ],
)
def test_negative_value_is_rejected(column):
    with pytest.raises(DataValidationError) as exc_info:
        validate_request_data(_row(**{column: -1.0}))

    assert any(column in failure for failure in exc_info.value.failures)


@pytest.mark.parametrize("bad_score", [0, 6, -1, 9])
def test_review_score_out_of_range_is_rejected(bad_score):
    with pytest.raises(DataValidationError) as exc_info:
        validate_request_data(_row(review_score=bad_score))

    assert any("review_score" in failure for failure in exc_info.value.failures)


def test_unknown_order_status_is_rejected():
    with pytest.raises(DataValidationError) as exc_info:
        validate_request_data(_row(order_status="processing"))

    assert any("order_status" in failure for failure in exc_info.value.failures)


def test_unknown_customer_state_is_rejected():
    with pytest.raises(DataValidationError) as exc_info:
        validate_request_data(_row(customer_state="ZZ"))

    assert any("customer_state" in failure for failure in exc_info.value.failures)


def test_null_order_status_is_rejected():
    with pytest.raises(DataValidationError) as exc_info:
        validate_request_data(_row(order_status=None))

    assert any("order_status" in failure for failure in exc_info.value.failures)


def test_null_customer_state_is_rejected():
    with pytest.raises(DataValidationError) as exc_info:
        validate_request_data(_row(customer_state=None))

    assert any("customer_state" in failure for failure in exc_info.value.failures)


def test_data_validation_error_carries_one_readable_failure_per_problem():
    """
    A row with TWO problems at once must report both, not just the first -
    app/main.py's 400 handler surfaces exc.failures verbatim to the client,
    so a partial failure list means a confusing "fix it, resubmit, get told
    about a second problem you weren't warned about" loop.
    """

    with pytest.raises(DataValidationError) as exc_info:
        validate_request_data(_row(total_payment=-1.0, review_score=9))

    assert len(exc_info.value.failures) >= 2
