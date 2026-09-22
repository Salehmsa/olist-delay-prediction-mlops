"""
Unit tests for src/monitoring/drift.py.

Same isolation pattern as test_prediction_log.py (a throwaway DB per
test) plus a fixed, small monitoring config (window_size=10,
min_sample_size=5) so each test can log an exact, known number of rows
and assert an exact expected verdict - drift.py's whole job is counting
and thresholds, so the tests are written around round numbers that make
the arithmetic checkable by hand.
"""

import pytest

from src.monitoring import drift, prediction_log

_TEST_CONFIG = {
    "monitoring": {
        "prediction_log_path": None,  # filled in by the fixture below
        "baseline_positive_rate": 0.066,
        "drift_window_size": 10,
        "drift_min_sample_size": 5,
        "drift_high_threshold": 0.20,
        "drift_low_threshold": 0.01,
    }
}


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "predictions.db"

    config = {**_TEST_CONFIG, "monitoring": {**_TEST_CONFIG["monitoring"]}}
    config["monitoring"]["prediction_log_path"] = str(db_path)

    monkeypatch.setattr(prediction_log, "load_config", lambda: config)
    monkeypatch.setattr(drift, "load_config", lambda: config)
    return db_path


def _log_n(n, prediction, probability=0.5):
    for _ in range(n):
        prediction_log.log_prediction(
            order_id=None,
            source="predict",
            prediction=prediction,
            probability=probability,
            model_version="v1",
            latency_ms=1.0,
        )


def test_insufficient_data_below_min_sample_size():
    _log_n(3, prediction=0)  # min_sample_size is 5

    report = drift.compute_drift_report()

    assert report["status"] == "insufficient_data"
    assert report["predictions_logged_total"] == 3


def test_status_ok_when_positive_rate_within_bounds():
    # window_size=10: 1 positive / 10 = 10% - inside [1%, 20%]
    _log_n(9, prediction=0)
    _log_n(1, prediction=1)

    report = drift.compute_drift_report()

    assert report["status"] == "ok"
    assert report["current_positive_rate"] == pytest.approx(0.10)
    assert report["baseline_positive_rate"] == 0.066


def test_status_drift_detected_when_positive_rate_too_high():
    # 5 positive / 10 = 50% - above the 20% high threshold
    _log_n(5, prediction=0)
    _log_n(5, prediction=1)

    report = drift.compute_drift_report()

    assert report["status"] == "drift_detected"
    assert report["current_positive_rate"] == pytest.approx(0.50)


def test_status_drift_detected_when_positive_rate_too_low():
    # 0 positive / 10 = 0% - below the 1% low threshold. The scenario
    # this rule exists for (see monitoring/prometheus/alerts.yml's
    # comment): recall_delayed is already only 8% on the real model, so
    # a further collapse toward 0% predicted-delayed is a strong signal
    # something broke, not that deliveries genuinely improved overnight.
    _log_n(10, prediction=0)

    report = drift.compute_drift_report()

    assert report["status"] == "drift_detected"
    assert report["current_positive_rate"] == pytest.approx(0.0)


def test_window_uses_only_the_most_recent_predictions():
    """
    21 rows logged in three batches; window_size=10 means only the
    newest 10 should count. recent_predictions() is newest-first, so
    with batches logged in this order - ten 1s, then ten 0s, then one
    final 1 - the newest 10 rows are [that final 1, then the nine
    newest 0s from the middle batch] = 1 positive / 10 = 10%, safely
    "ok". If the window leaked older rows in, the extra 1s from the
    first batch would push this to "drift_detected" instead.
    """

    _log_n(10, prediction=1)
    _log_n(10, prediction=0)
    _log_n(1, prediction=1)

    report = drift.compute_drift_report()

    assert report["status"] == "ok"
    assert report["current_positive_rate"] == pytest.approx(0.10)


def test_psi_absent_until_a_reference_window_exists():
    _log_n(10, prediction=0, probability=0.1)  # exactly one window, no reference yet

    report = drift.compute_drift_report()

    assert report["psi_probability_drift"] is None


def test_psi_near_zero_for_identical_distributions():
    _log_n(10, prediction=0, probability=0.1)  # reference window
    _log_n(10, prediction=0, probability=0.1)  # current window, identical

    report = drift.compute_drift_report()

    assert report["psi_probability_drift"] is not None
    assert report["psi_probability_drift"] == pytest.approx(0.0, abs=1e-3)


def test_psi_detects_a_real_distribution_shift():
    _log_n(10, prediction=0, probability=0.1)  # reference: all low scores
    _log_n(10, prediction=1, probability=0.9)  # current: all high scores

    report = drift.compute_drift_report()

    # A textbook complete shift (all mass moves to a different bin) - by
    # the usual PSI convention, anything above ~0.25 already means
    # "significant shift"; this scenario lands far higher than that, so
    # the assertion stays well clear of the epsilon-smoothing noise
    # floor instead of pinning an exact fragile value.
    assert report["psi_probability_drift"] > 1.0
