"""
Data tests: leakage checks.

A feature "leaks" when it encodes information that would not actually be
available at prediction time - most often, anything derived from the
delivery outcome itself. This is a regression test, not a research
decision: it locks in that models/feature_list.json (the model's real
42-column contract) and the raw request schema stay free of the obvious
post-outcome signal names, so a future retrain/feature addition that
accidentally reintroduces one fails CI loudly instead of shipping quietly.

NOTE (worth a deliberate look, not just this test): review_score is one of
the 42 trained features, and in the real Olist dataset a review is
typically written AFTER delivery - which is the classic, dataset-specific
leakage risk here, distinct from the naming-convention check below. That
call belongs to Notebook 02 (label creation) / Notebook 05 (feature
engineering), which this task doesn't have copied in to inspect. Worth
confirming there before trusting this feature at face value.
"""

from src.data.preprocessing import FEATURE_LIST
from src.inference.input_builder import build_input_dataframe
from src.schemas.request_schema import OrderRequest

# Substrings that would signal a post-outcome / label-derived column if they
# ever showed up in a feature name. Deliberately specific (not just
# "delay", which legitimately appears in approval_delay_hours - a
# PRE-delivery, payment-approval signal, not the delivery delay itself).
LEAKAGE_SIGNAL_SUBSTRINGS = [
    "delivered_date",
    "actual_delivery",
    "delivery_date",
    "delivery_delay",
    "is_late",
    "is_delayed",
    "eta_diff",
    "delay_days",
    "label",
    "target",
]


def test_feature_list_contains_no_post_outcome_signal_names():
    lowered = [name.lower() for name in FEATURE_LIST]

    offenders = [name for name in lowered for signal in LEAKAGE_SIGNAL_SUBSTRINGS if signal in name]

    assert offenders == [], f"possible leakage in feature_list.json: {offenders}"


def test_the_deny_list_is_not_accidentally_too_broad():
    """
    Sanity check on the check itself: approval_delay_hours legitimately
    contains "delay" and MUST stay in the feature list - if this ever
    fails, the deny-list above got too aggressive, not the feature set.
    """

    assert "approval_delay_hours" in FEATURE_LIST


def test_raw_request_schema_never_asks_for_a_post_outcome_field(valid_order_kwargs):
    """
    input_builder.py's raw columns are the only thing a client ever
    supplies - confirms none of THOSE carry a post-outcome name either, one
    layer before feature_list.json.
    """

    request = OrderRequest(**valid_order_kwargs)
    raw_columns = [c.lower() for c in build_input_dataframe(request).columns]

    offenders = [
        name for name in raw_columns for signal in LEAKAGE_SIGNAL_SUBSTRINGS if signal in name
    ]

    assert offenders == []
