"""
Section 6: shared pytest bootstrap and fixtures.

WHY THIS FILE HAS MODULE-LEVEL CODE, NOT JUST FIXTURES
--------------------------------------------------------
Several modules under test resolve the MLflow Model Registry AT IMPORT TIME,
not lazily inside a function:

    src/inference/model_registry.py   -> looks up models:/<name>@production
    src/data/preprocessing.py         -> downloads that run's artifacts
    src/inference/pipeline.py         -> calls load_model() at import time

That lookup happens the moment ANY test file does
`from src.inference... import ...` — which, in pytest's collection order,
happens AFTER conftest.py's own module-level code has already run (pytest
loads conftest.py first, before importing the test files below it). That
makes this the one place a "nothing registered yet" problem can be fixed
for the whole suite at once, instead of becoming a separate import error in
every test file that touches src/.

This does NOT run on every `pytest` invocation — only when nothing is
registered yet. If you already ran `python -m src.training.log_model` (the
normal Section 5 workflow), your real registered model is left exactly as
it is and every test below exercises it. Only a completely fresh
clone/CI runner with an empty mlflow.db falls through to the placeholder
branch, which registers one the same way `log_model.py --allow-placeholder`
would if you ran it by hand.
"""

import mlflow
import pytest
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.schemas.request_schema import OrderRequest
from src.utils.config_loader import load_config

_config = load_config()
_mlflow_config = _config["mlflow"]

mlflow.set_tracking_uri(_mlflow_config["tracking_uri"])

_client = MlflowClient()

try:
    _client.get_model_version_by_alias(
        _mlflow_config["registered_model_name"], _mlflow_config["model_alias"]
    )
except MlflowException:
    from src.training.log_model import log_and_register

    log_and_register(allow_placeholder=True)


# -----------------------------------------------------------------------------
# Shared fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def valid_order_kwargs() -> dict:
    """
    One realistic, schema-valid order. Same sample values used consistently
    across the project's earlier smoke scripts, kept here as the one shared
    source instead of re-typed in every test file.
    """

    return dict(
        customer_zip_code_prefix=12345,
        total_payment=150.0,
        avg_payment=75.0,
        max_installments=3,
        payment_count=2,
        items_count=2,
        total_price=130.0,
        total_freight=20.0,
        unique_products=2,
        unique_sellers=1,
        review_score=5,
        approval_delay_hours=3,
        order_status="delivered",
        customer_state="SP",
    )


@pytest.fixture
def valid_order_request(valid_order_kwargs) -> OrderRequest:
    return OrderRequest(**valid_order_kwargs)


@pytest.fixture(scope="session")
def model():
    """
    Session-scoped so the ~100MB skops model is deserialized ONCE for the
    whole test run, not once per test. Reuses the same singleton
    src/inference/pipeline.py already loads at import time rather than
    calling load_model() again — one load, shared by every test that needs
    a real model.
    """

    from src.inference.pipeline import model as loaded_model

    return loaded_model


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient against the real app - no running server needed."""

    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)
