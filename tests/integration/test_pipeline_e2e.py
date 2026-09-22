"""
Integration tests: src/inference/pipeline.run_pipeline() - the full chain
end to end (build raw row -> Great Expectations validation -> feature
engineering -> preprocess -> predict), exactly the sequence /predict will
call once Section 7 adds it. Exercising it directly here (no HTTP layer)
means Section 7's route tests will mostly just wrap these same cases in
`client.post(...)`.
"""

import pytest

from src.inference.pipeline import run_pipeline
from src.schemas.request_schema import OrderRequest
from src.validation.data_validation import DataValidationError


def test_valid_request_produces_a_prediction_and_probability(valid_order_kwargs):
    request = OrderRequest(**valid_order_kwargs)

    prediction, probability = run_pipeline(request)

    assert prediction is not None
    assert 0.0 <= probability <= 1.0


def test_request_with_every_optional_field_missing_still_predicts():
    """
    Full round trip through the imputer for a brand-new order with no
    payment/review/freight history yet - the case request_schema.py's
    Optional fields exist to support.
    """

    request = OrderRequest(order_status="delivered", customer_state="SP")

    prediction, probability = run_pipeline(request)

    assert prediction is not None
    assert 0.0 <= probability <= 1.0


def test_negative_price_passes_pydantic_but_is_rejected_one_layer_deeper(
    valid_order_kwargs,
):
    """
    The exact case request_schema.py has no ge=0 guard for (see
    tests/data/test_request_schema.py's
    test_negative_prices_are_NOT_caught_here_by_design) - OrderRequest
    happily constructs it, so this is the test proving the Great
    Expectations layer in run_pipeline() actually catches what Pydantic
    doesn't. If this ever starts passing without raising, the two
    validation layers have silently stopped agreeing on who's responsible
    for what.
    """

    valid_order_kwargs["total_payment"] = -50.0
    request = OrderRequest(**valid_order_kwargs)  # constructs fine

    with pytest.raises(DataValidationError) as exc_info:
        run_pipeline(request)

    assert any("total_payment" in failure for failure in exc_info.value.failures)


def test_out_of_range_review_score_construction_is_blocked_by_pydantic_first(
    valid_order_kwargs,
):
    """
    Unlike total_payment, review_score DOES have a Pydantic-level ge=1/le=5
    guard - confirms that layer fires before run_pipeline() is ever
    reached, i.e. this case never becomes a DataValidationError at all.
    """

    from pydantic import ValidationError

    valid_order_kwargs["review_score"] = 9

    with pytest.raises(ValidationError):
        OrderRequest(**valid_order_kwargs)
