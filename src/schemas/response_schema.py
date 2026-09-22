"""
Section 7: response contracts for /predict and /predict/batch. Declared and
validated the same way the request side is (Task 3: "Validate the request
and the response with schemas"), and passed to FastAPI as each route's
response_model so a shape bug here fails loudly in the OpenAPI docs / a 500,
not by silently serializing whatever the route function happened to return.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """
    One order's prediction result.

    prediction: 1 = the model flags this order as delayed/late, 0 = on
    time. Confirmed against Notebook 02 (Label Creation), which defines
    the target variable is_delayed as 1 = Delayed, 0 = On Time - this is
    the model's raw class label (a 2-class model, see
    tests/model/test_predict.py), not a separately-mapped verdict.

    probability: P(prediction == 1), i.e. the model's estimated chance
    THIS order is delayed - not "P(the predicted class)" - so it stays
    informative even when prediction == 0.
    """

    prediction: int
    probability: float = Field(..., ge=0.0, le=1.0)
    model_version: str


class BatchPredictionItem(BaseModel):
    """
    One order's outcome inside a batch call. Exactly one of `result` /
    `error` is set - a batch response uses this per-item split instead of
    failing the whole call for one bad row (see BatchPredictionResponse).
    """

    index: int
    success: bool
    result: Optional[PredictionResponse] = None
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """
    /predict/batch's response. A malformed request body (wrong type, bad
    Literal) never reaches here - FastAPI rejects the whole request with a
    422 before the route runs, same as /predict. What DOES land here is a
    well-formed batch where one or more orders individually fail Section 4's
    Great Expectations layer (a negative price, say): the request as a
    whole still succeeds (200), and each order's own outcome is reported in
    `results` - so one bad row in a batch of 500 doesn't cost the other 499
    their predictions.
    """

    results: List[BatchPredictionItem]
    succeeded: int
    failed: int
