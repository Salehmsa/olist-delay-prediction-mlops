"""
Data tests: src/schemas/request_schema.py — the Pydantic-level schema
contract. This is the FIRST line of defense (Section 3); Great Expectations
(test_expectations_suite.py) is the second, deeper layer.
"""

import pytest
from pydantic import ValidationError

from src.schemas.request_schema import OrderRequest


def test_valid_request_is_accepted(valid_order_kwargs):
    request = OrderRequest(**valid_order_kwargs)

    assert request.order_status == "delivered"
    assert request.customer_state == "SP"


def test_all_numeric_fields_are_optional():
    """
    A brand-new order where nothing but the two required categorical fields
    is known yet must still construct - that's the whole point of routing
    every numeric field through the imputer instead of rejecting it here.
    """

    request = OrderRequest(order_status="delivered", customer_state="SP")

    assert request.total_payment is None
    assert request.review_score is None


@pytest.mark.parametrize("bad_status", ["shipped", "invoiced", "", "DELIVERED"])
def test_order_status_outside_the_trained_categories_is_rejected(
    valid_order_kwargs, bad_status
):
    valid_order_kwargs["order_status"] = bad_status

    with pytest.raises(ValidationError):
        OrderRequest(**valid_order_kwargs)


@pytest.mark.parametrize("bad_state", ["ZZ", "sp", "California", ""])
def test_customer_state_outside_the_27_trained_states_is_rejected(
    valid_order_kwargs, bad_state
):
    valid_order_kwargs["customer_state"] = bad_state

    with pytest.raises(ValidationError):
        OrderRequest(**valid_order_kwargs)


@pytest.mark.parametrize("bad_score", [0, 6, -1, 10])
def test_review_score_outside_1_to_5_is_rejected(valid_order_kwargs, bad_score):
    valid_order_kwargs["review_score"] = bad_score

    with pytest.raises(ValidationError):
        OrderRequest(**valid_order_kwargs)


def test_missing_required_categorical_field_is_rejected(valid_order_kwargs):
    del valid_order_kwargs["order_status"]

    with pytest.raises(ValidationError):
        OrderRequest(**valid_order_kwargs)


def test_negative_prices_are_NOT_caught_here_by_design(valid_order_kwargs):
    """
    Documents a real, deliberate gap: request_schema.py has no ge=0 on
    total_payment/total_freight/etc. - that non-negativity check lives one
    layer deeper, in the Great Expectations suite (test_expectations_suite.py
    covers it). This test exists so that if someone "fixes" this gap by
    adding validation here, they see this test and update it on purpose
    instead of the GE layer silently becoming redundant.
    """

    valid_order_kwargs["total_payment"] = -50.0

    request = OrderRequest(**valid_order_kwargs)  # does NOT raise

    assert request.total_payment == -50.0
