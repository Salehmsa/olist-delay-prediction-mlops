from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Exactly encoder.categories_[0] / [1] from models/encoder.pkl (verified by
# loading it — see project notes). A value outside these lists is rejected
# by FastAPI with a clear 422 before it ever reaches the model, instead of
# silently falling through OneHotEncoder's handle_unknown="ignore" and
# producing an all-zero row the model was never shown at training time.
OrderStatus = Literal["canceled", "delivered"]

CustomerState = Literal[
    "AC",
    "AL",
    "AM",
    "AP",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MG",
    "MS",
    "MT",
    "PA",
    "PB",
    "PE",
    "PI",
    "PR",
    "RJ",
    "RN",
    "RO",
    "RR",
    "RS",
    "SC",
    "SE",
    "SP",
    "TO",
]

# One realistic, schema-valid order - reused as-is below AND as the
# tests/conftest.py fixture's own sample data, so the interactive API docs
# (Section 7: "check the automatic API docs and make sure the examples
# work") and the test suite exercise literally the same values.
_EXAMPLE_ORDER = {
    "order_id": "e481f51cbdc54678b7cc49136f2d6af7",
    "customer_zip_code_prefix": 12345,
    "total_payment": 150.0,
    "avg_payment": 75.0,
    "max_installments": 3,
    "payment_count": 2,
    "items_count": 2,
    "total_price": 130.0,
    "total_freight": 20.0,
    "unique_products": 2,
    "unique_sellers": 1,
    "review_score": 5,
    "approval_delay_hours": 3,
    "order_status": "delivered",
    "customer_state": "SP",
}


class OrderRequest(BaseModel):
    """
    Raw order fields, in exactly the shape imputer.pkl / scaler.pkl were fit
    on (models/*.pkl, feature_names_in_). All numeric fields are optional:
    the fitted SimpleImputer(strategy="median") fills a missing value
    exactly like it filled missing rows at training time, so the API leans
    on it instead of rejecting a request just because one field isn't known
    yet — that's what the imputer is *for*.

    review_score is the field most often genuinely unknown at prediction
    time (a brand-new order has no review yet). review_missing is NOT
    collected from the client — src/features/feature_engineering.py derives
    it from review_score being null, the same signal Notebook 05 engineered
    it from.
    """

    model_config = ConfigDict(json_schema_extra={"examples": [_EXAMPLE_ORDER]})

    # Section 10, bullet 3: optional, and never used by the model itself
    # (not in feature_list.json) - it exists only so src/monitoring/
    # prediction_log.py can tag a logged prediction with something that
    # matches a REAL Olist order, so it can be evaluated later once the
    # real delivery date arrives. Omit it and logging still works - it
    # just means that particular row can't be joined back to a real
    # outcome afterward. See README's "Monitoring" section.
    order_id: Optional[str] = None

    customer_zip_code_prefix: Optional[float] = None
    total_payment: Optional[float] = None
    avg_payment: Optional[float] = None
    max_installments: Optional[float] = None
    payment_count: Optional[float] = None
    items_count: Optional[float] = None
    total_price: Optional[float] = None
    total_freight: Optional[float] = None
    unique_products: Optional[float] = None
    unique_sellers: Optional[float] = None
    review_score: Optional[float] = Field(default=None, ge=1, le=5)
    approval_delay_hours: Optional[float] = None

    order_status: OrderStatus
    customer_state: CustomerState


class BatchPredictionRequest(BaseModel):
    """
    Section 7: /predict/batch's request body. Capped at 500 orders per call
    - a batch endpoint that accepts unbounded input is a memory/latency
    liability, and 500 is generous for what this service is for (scoring a
    day's/hour's worth of new orders, not a bulk data migration).
    """

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"orders": [_EXAMPLE_ORDER, _EXAMPLE_ORDER]}]}
    )

    orders: List[OrderRequest] = Field(..., min_length=1, max_length=500)
