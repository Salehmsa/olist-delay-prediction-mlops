"""
Runs an incoming order request's raw DataFrame through the Great
Expectations suite in expectations.py before it reaches feature engineering
or preprocessing - Section 4's "validate incoming data ... before it
reaches the model" requirement.

Decision on failure: REJECT. A failed expectation means the request itself
is suspect (a negative price, a review score of 9, a customer_state the
model never saw at training time) - the same "reject bad payloads clearly"
posture Section 3's exception handlers already use for schema-level
problems (RequestValidationError), just one layer deeper than a Pydantic
type/Literal can express.
"""

import pandas as pd

from src.utils.logger import logger
from src.validation.expectations import batch_definition, suite


class DataValidationError(Exception):
    """Raised when a request fails the Great Expectations suite."""

    def __init__(self, failures):
        self.failures = failures
        super().__init__("; ".join(failures))


def validate_request_data(df: pd.DataFrame) -> None:
    """
    Validates df (the raw request DataFrame from input_builder.py) against
    the Great Expectations suite.

    Returns None on success. Raises DataValidationError with one readable
    message per failed expectation on failure - app/main.py's exception
    handler turns that into a clean 400 with the specific reasons.
    """

    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    result = batch.validate(suite)

    if result.success:
        return

    failures = []

    for expectation_result in result.results:
        if expectation_result.success:
            continue

        exp_config = expectation_result.expectation_config
        column = exp_config.kwargs.get("column", "?")
        failures.append(f"{exp_config.type} failed on column '{column}'")

    logger.warning("data validation failed | failures=%s", failures)

    raise DataValidationError(failures)
