"""
Unit tests for src/monitoring/prediction_log.py - the SQLite prediction
log Section 10, bullet 3 writes to.

Each test gets its own throwaway DB file (tmp_path, monkeypatched in
place of config.yaml's real prediction_log_path) - fully isolated from
the shared SESSION-scoped `client` fixture (tests/conftest.py) that most
of the rest of the suite uses. Every /predict or /predict/batch call any
OTHER test makes through that shared client also writes to the real
logs/predictions.db as a side effect (src/inference/pipeline.py) - these
tests would become order-dependent on whatever the rest of the suite
already logged if they touched that same file.
"""

import pytest

from src.monitoring import prediction_log


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "predictions.db"

    monkeypatch.setattr(
        prediction_log,
        "load_config",
        lambda: {"monitoring": {"prediction_log_path": str(db_path)}},
    )
    return db_path


def test_log_prediction_persists_a_row():
    prediction_log.log_prediction(
        order_id="abc123",
        source="predict",
        prediction=1,
        probability=0.83,
        model_version="olist_delay_classifier v1 (Production)",
        latency_ms=12.5,
    )

    assert prediction_log.total_count() == 1

    row = prediction_log.recent_predictions(10)[0]
    assert row["order_id"] == "abc123"
    assert row["source"] == "predict"
    assert row["prediction"] == 1
    assert row["probability"] == pytest.approx(0.83)
    assert row["model_version"] == "olist_delay_classifier v1 (Production)"
    assert row["latency_ms"] == pytest.approx(12.5)
    assert row["ts_utc"]  # non-empty, set automatically


def test_log_prediction_allows_missing_order_id():
    """
    Optional by design (src/schemas/request_schema.py's own docstring) -
    a caller that never supplies one still gets a logged, drift-countable
    row, just one that can never be joined back to a real order later.
    """

    prediction_log.log_prediction(
        order_id=None,
        source="predict_batch",
        prediction=0,
        probability=0.1,
        model_version="v1",
        latency_ms=5.0,
    )

    row = prediction_log.recent_predictions(1)[0]
    assert row["order_id"] is None


def test_recent_predictions_returns_newest_first():
    for i in range(5):
        prediction_log.log_prediction(
            order_id=f"order-{i}",
            source="predict",
            prediction=i % 2,
            probability=0.5,
            model_version="v1",
            latency_ms=1.0,
        )

    rows = prediction_log.recent_predictions(3)

    assert [row["order_id"] for row in rows] == ["order-4", "order-3", "order-2"]


def test_total_count_reflects_every_logged_row():
    assert prediction_log.total_count() == 0

    for _ in range(7):
        prediction_log.log_prediction(
            order_id=None,
            source="predict",
            prediction=0,
            probability=0.2,
            model_version="v1",
            latency_ms=1.0,
        )

    assert prediction_log.total_count() == 7


def test_log_prediction_never_raises_even_if_the_write_fails(monkeypatch):
    """
    Best-effort by design (this module's own docstring): a logging
    failure must never take down a real prediction response that has
    already succeeded. Simulates that failure deterministically (no
    reliance on any particular filesystem's error behavior) by making
    the underlying sqlite3.connect itself raise.
    """

    def _boom(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(prediction_log.sqlite3, "connect", _boom)

    prediction_log.log_prediction(
        order_id=None,
        source="predict",
        prediction=0,
        probability=0.2,
        model_version="v1",
        latency_ms=1.0,
    )  # must not raise
