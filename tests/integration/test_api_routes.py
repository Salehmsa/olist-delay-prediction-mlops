"""
Integration tests: API routes, end to end, through FastAPI's TestClient
(no separate running server / uvicorn process needed - TestClient drives
the real ASGI app in-process).
"""


def test_root_reports_running(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "running"}


def test_health_reports_healthy(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_model_info_matches_the_resolved_registry_state(client):
    from src.inference.model_registry import MODEL_VERSION_INFO

    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json() == MODEL_VERSION_INFO


def test_unknown_route_returns_404(client):
    response = client.get("/this-route-does-not-exist")

    assert response.status_code == 404


# -----------------------------------------------------------------------------
# /predict
# -----------------------------------------------------------------------------


def test_predict_valid_order_returns_200_with_prediction_shape(
    client, valid_order_kwargs
):
    response = client.post("/predict", json=valid_order_kwargs)

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"prediction", "probability", "model_version"}
    assert isinstance(body["prediction"], int)
    assert 0.0 <= body["probability"] <= 1.0
    assert isinstance(body["model_version"], str) and body["model_version"]


def test_predict_missing_required_field_returns_422(client, valid_order_kwargs):
    del valid_order_kwargs["order_status"]

    response = client.post("/predict", json=valid_order_kwargs)

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"


def test_predict_bad_category_returns_422_not_500(client, valid_order_kwargs):
    """
    order_status outside the trained Literal values must be rejected at the
    schema layer (422), never reach the encoder's handle_unknown="ignore"
    fallback.
    """

    valid_order_kwargs["order_status"] = "processing"

    response = client.post("/predict", json=valid_order_kwargs)

    assert response.status_code == 422


def test_predict_negative_price_passes_schema_but_returns_400(
    client, valid_order_kwargs
):
    """
    The exact case request_schema.py has no ge=0 guard for - well-typed per
    Pydantic, caught one layer deeper by Great Expectations
    (src/validation/expectations.py), surfaced here as a clean 400, not a
    422 and not a 500.
    """

    valid_order_kwargs["total_payment"] = -50.0

    response = client.post("/predict", json=valid_order_kwargs)

    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "data_validation_failed"
    assert any("total_payment" in detail for detail in body["details"])


def test_predict_with_every_optional_field_missing_still_returns_200(client):
    response = client.post(
        "/predict", json={"order_status": "delivered", "customer_state": "SP"}
    )

    assert response.status_code == 200


def test_predict_example_from_the_openapi_schema_actually_works(client):
    """
    Section 7: "check the automatic API docs and make sure the examples
    work" - reads the example straight out of the live OpenAPI schema
    (the same JSON Swagger UI's "Try it out" pre-fills) and POSTs it for
    real, instead of trusting that the example value looks plausible.
    """

    openapi_schema = client.get("/openapi.json").json()
    order_schema = openapi_schema["components"]["schemas"]["OrderRequest"]
    example = order_schema["examples"][0]

    response = client.post("/predict", json=example)

    assert response.status_code == 200


# -----------------------------------------------------------------------------
# /predict/batch
# -----------------------------------------------------------------------------


def test_predict_batch_all_valid_returns_200_with_one_result_per_order(
    client, valid_order_kwargs
):
    response = client.post(
        "/predict/batch", json={"orders": [valid_order_kwargs, valid_order_kwargs]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["succeeded"] == 2
    assert body["failed"] == 0
    assert len(body["results"]) == 2
    assert all(item["success"] for item in body["results"])
    assert all(item["result"]["probability"] is not None for item in body["results"])


def test_predict_batch_mixed_valid_and_invalid_returns_200_with_per_item_errors(
    client, valid_order_kwargs
):
    """
    The behavior that justifies a per-item response shape at all: one bad
    row in a batch must NOT cost the other rows their predictions.
    """

    bad_order = {**valid_order_kwargs, "total_payment": -50.0}

    response = client.post(
        "/predict/batch", json={"orders": [valid_order_kwargs, bad_order]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["succeeded"] == 1
    assert body["failed"] == 1

    good, bad = body["results"]
    assert good["success"] is True and good["error"] is None
    assert bad["success"] is False and bad["result"] is None
    assert "total_payment" in bad["error"]


def test_predict_batch_preserves_original_order_via_index(client, valid_order_kwargs):
    bad_order = {**valid_order_kwargs, "total_payment": -50.0}

    response = client.post(
        "/predict/batch", json={"orders": [bad_order, valid_order_kwargs, bad_order]}
    )

    indices = [item["index"] for item in response.json()["results"]]
    assert indices == [0, 1, 2]


def test_predict_batch_malformed_order_shape_rejects_the_whole_request(
    client, valid_order_kwargs
):
    """
    Unlike a Great-Expectations-level failure, a bad SHAPE (here: an
    order_status the schema never trained on) is a client bug in the
    request itself - the whole batch is rejected with a 422 before the
    route runs, not reported as a per-item error.
    """

    bad_shape_order = {**valid_order_kwargs, "order_status": "processing"}

    response = client.post(
        "/predict/batch", json={"orders": [valid_order_kwargs, bad_shape_order]}
    )

    assert response.status_code == 422


def test_predict_batch_empty_orders_list_is_rejected(client):
    response = client.post("/predict/batch", json={"orders": []})

    assert response.status_code == 422


def test_predict_batch_over_the_size_cap_is_rejected(client, valid_order_kwargs):
    response = client.post(
        "/predict/batch", json={"orders": [valid_order_kwargs] * 501}
    )

    assert response.status_code == 422


# -----------------------------------------------------------------------------
# Section 10: monitoring
#
# `client` is session-scoped (tests/conftest.py) and shared with every
# /predict(/batch) test above, all of which also write to the SAME real
# logs/predictions.db as a side effect (src/inference/pipeline.py). These
# tests can never assert an exact predictions_logged_total or drift
# status for that reason - only that logging actually happened (a
# BEFORE/AFTER comparison around a call made inside the test itself) and
# that the response shapes are correct. Exact-count behavior belongs to
# tests/monitoring/test_prediction_log.py and test_drift.py instead,
# which each use a fully isolated, empty database.
# -----------------------------------------------------------------------------


def test_monitoring_summary_shape(client):
    response = client.get("/monitoring/summary")

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"predictions_logged_total", "drift"}
    assert isinstance(body["predictions_logged_total"], int)
    assert body["predictions_logged_total"] >= 0
    assert body["drift"]["status"] in {"insufficient_data", "ok", "drift_detected"}


def test_predict_increments_the_monitoring_summary_count(client, valid_order_kwargs):
    before = client.get("/monitoring/summary").json()["predictions_logged_total"]

    response = client.post("/predict", json=valid_order_kwargs)
    assert response.status_code == 200

    after = client.get("/monitoring/summary").json()["predictions_logged_total"]

    assert after == before + 1


def test_predict_batch_increments_the_monitoring_summary_count_per_order(
    client, valid_order_kwargs
):
    before = client.get("/monitoring/summary").json()["predictions_logged_total"]

    response = client.post(
        "/predict/batch", json={"orders": [valid_order_kwargs, valid_order_kwargs]}
    )
    assert response.status_code == 200

    after = client.get("/monitoring/summary").json()["predictions_logged_total"]

    assert after == before + 2


def test_a_request_that_fails_validation_is_not_logged_as_a_prediction(
    client, valid_order_kwargs
):
    """
    Only a SUCCESSFUL prediction is logged (src/inference/pipeline.py) -
    a request that fails Great Expectations was never actually predicted
    on, so it has no business inflating predictions_logged_total or
    skewing the drift-rate denominator.
    """

    before = client.get("/monitoring/summary").json()["predictions_logged_total"]

    valid_order_kwargs["total_payment"] = -50.0
    response = client.post("/predict", json=valid_order_kwargs)
    assert response.status_code == 400

    after = client.get("/monitoring/summary").json()["predictions_logged_total"]

    assert after == before


def test_metrics_endpoint_exposes_prometheus_format(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

    body = response.text
    # The generic RED metrics from app/main.py's Instrumentator...
    assert "http_requests_total" in body
    assert "http_request_duration_highr_seconds_bucket" in body
    # ...and the ML-specific ones from src/monitoring/metrics.py.
    assert "olist_predictions_total" in body
    assert "olist_prediction_latency_seconds" in body
    assert "olist_prediction_positive_rate" in body
    assert "olist_prediction_drift_status" in body


def test_metrics_reflects_predictions_made_through_predict(client, valid_order_kwargs):
    response = client.post("/predict", json=valid_order_kwargs)
    predicted_class = str(response.json()["prediction"])

    metrics_body = client.get("/metrics").text

    assert (
        f'olist_predictions_total{{predicted_class="{predicted_class}"}}'
        in metrics_body
    )
