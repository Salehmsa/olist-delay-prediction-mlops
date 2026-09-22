"""
Defines the Great Expectations suite that validates a raw order request
before it reaches the feature/preprocessing pipeline (Section 4).

Ranges, allowed categories, and required (non-null) columns are recovered
from the same fitted objects the rest of src/ trusts (models/encoder.pkl)
rather than retyped by hand, so this suite can't silently drift from what
the model actually expects - the same discipline preprocessing.py uses for
NUMERIC_COLUMNS / CATEGORICAL_COLUMNS.
"""

import joblib
import great_expectations as gx
import great_expectations.expectations as gxe
from great_expectations.core.expectation_suite import ExpectationSuite

from src.utils.config_loader import load_config

config = load_config()
encoder = joblib.load(config["encoder_path"])

ORDER_STATUS_CATEGORIES = list(encoder.categories_[0])
CUSTOMER_STATE_CATEGORIES = list(encoder.categories_[1])

# Columns that can never legitimately be negative on a real order. Nulls are
# NOT flagged here - SimpleImputer(strategy="median") is what handles a
# missing value; Great Expectations' job is catching a present-but-wrong
# value (a negative price, a fat-fingered figure), not standing in for the
# imputer.
NON_NEGATIVE_COLUMNS = [
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
]


def build_context():
    """
    An ephemeral (in-memory) Data Context: no .great_expectations/ project
    scaffolding written to disk, no progress-bar spam in the logs - safe to
    build once per process and reuse for every request.
    """

    context = gx.get_context(mode="ephemeral")

    context.variables.progress_bars = {
        "globally": False,
        "metric_calculations": False,
    }
    context.variables.save()

    return context


def build_suite(context):
    suite = context.suites.add(ExpectationSuite(name="order_request_suite"))

    for column in NON_NEGATIVE_COLUMNS:
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeBetween(column=column, min_value=0)
        )

    # review_score: null is fine (most new orders have none yet - that's
    # exactly what review_missing encodes) but a value that IS present must
    # be a real 1-5 rating, not e.g. a 9 from a client bug.
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="review_score", min_value=1, max_value=5
        )
    )

    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="order_status", value_set=ORDER_STATUS_CATEGORIES
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="customer_state", value_set=CUSTOMER_STATE_CATEGORIES
        )
    )

    # order_status / customer_state have no imputer fallback - a real order
    # always has both, so missing here is a genuine problem, not the
    # ordinary missingness the numeric fields tolerate. (Pydantic already
    # requires these at the schema layer; declaring it again here documents
    # the expectation explicitly and keeps the GE suite self-describing on
    # its own, independent of request_schema.py.)
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="order_status"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="customer_state"))

    return suite


def build_batch_definition(context):
    data_source = context.data_sources.add_pandas("orders")
    data_asset = data_source.add_dataframe_asset(name="order_requests")

    return data_asset.add_batch_definition_whole_dataframe("batch")


# Built once at import time, reused for every request (src/inference/
# pipeline.py) - the same pattern preprocessing.py and predict.py use for
# the fitted model/encoder/scaler, instead of rebuilding a Data Context on
# every prediction.
context = build_context()
suite = build_suite(context)
batch_definition = build_batch_definition(context)
