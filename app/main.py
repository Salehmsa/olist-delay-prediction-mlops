import time

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from src.inference.model_registry import MODEL_VERSION_INFO
from src.inference.pipeline import MODEL_VERSION, run_pipeline
from src.monitoring.drift import compute_drift_report
from src.monitoring.metrics import register_drift_collector
from src.monitoring.prediction_log import total_count
from src.schemas.request_schema import BatchPredictionRequest, OrderRequest
from src.schemas.response_schema import (
    BatchPredictionItem,
    BatchPredictionResponse,
    PredictionResponse,
)
from src.utils.config_loader import load_config
from src.utils.logger import logger
from src.validation.data_validation import DataValidationError

config = load_config()

app = FastAPI(
    title=config["api"]["title"],
)

# Section 10, bullet 1: "expose metrics for the service - request count,
# latency, error rate". Instrumentator gives the generic RED metrics for
# every route for free (request count/latency/status, broken down by
# path and method) at GET /metrics, in the standard Prometheus text
# format - monitoring/prometheus/prometheus.yml scrapes it from there.
# excluded_handlers keeps /metrics itself out of its own request-count
# metrics (scraping /metrics every few seconds would otherwise show up
# as "traffic" in its own numbers). The ML-specific metrics on top of
# this (predictions by class, model-pipeline latency, drift) are
# src/monitoring/metrics.py - registered onto the SAME registry
# Instrumentator exposes here, so one /metrics endpoint serves all of it.
Instrumentator(excluded_handlers=["/metrics"]).instrument(app).expose(app)
register_drift_collector()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    General observability for every request: method, path, status, latency.
    Prediction-specific input/output/model-version logging lives in
    src/inference/pipeline.run_pipeline() instead - this middleware is the
    safety net that also covers /, /health, /model-info, and gives Section
    10 (monitoring) one place to add request-count/latency metrics later.
    """

    start = time.perf_counter()

    response = await call_next(request)

    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "%s %s | status=%s | latency_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )

    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    A malformed payload (wrong type, missing required field, a value outside
    the categories the model was trained on) is a normal, expected case -
    not a server bug. Reject it clearly with a 422 and the specific field
    errors instead of letting an unhandled exception take the worker down.
    Also what rejects a malformed /predict/batch body outright (Section 7):
    one bad order shape fails the WHOLE request here, before the route
    runs - a well-formed batch where one order fails Great Expectations
    instead gets the per-item handling in the /predict/batch route itself.
    """

    logger.warning("validation error on %s | errors=%s", request.url.path, exc.errors())

    return JSONResponse(
        status_code=422,
        content={"error": "invalid_request", "details": exc.errors()},
    )


@app.exception_handler(DataValidationError)
async def data_validation_exception_handler(request: Request, exc: DataValidationError):
    """
    Section 4: a request that is well-typed (passed Pydantic) but fails the
    Great Expectations suite - a negative price, an out-of-range review
    score, or similar. One layer deeper than RequestValidationError. Only
    reached from /predict (a single order) - /predict/batch catches this
    itself per order so one bad row doesn't fail the whole batch.
    """

    logger.warning("data validation error on %s | %s", request.url.path, exc.failures)

    return JSONResponse(
        status_code=400,
        content={"error": "data_validation_failed", "details": exc.failures},
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """
    Raised by src/data/preprocessing.preprocess() if the pipeline ever fails
    to produce a required column - a data/config problem, not a client
    mistake, but still handled cleanly rather than crashing the service.
    """

    logger.error("value error on %s | %s", request.url.path, str(exc))

    return JSONResponse(
        status_code=400,
        content={"error": "bad_request", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Last-resort catch-all: an unexpected bug returns a clean 500 instead of
    crashing the worker or leaking a stack trace to the client. The full
    traceback goes to the log, not the response.
    """

    logger.error("unhandled exception on %s", request.url.path, exc_info=True)

    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error"},
    )


@app.get("/")
def root():
    return {"status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/model-info")
def model_info():
    """
    Section 5: the real MLflow registry state (name, version, stage, run_id)
    - resolved once at startup in src/inference/model_registry.py, not a
    static string in config.yaml.
    """

    return MODEL_VERSION_INFO


@app.get("/monitoring/summary")
def monitoring_summary():
    """
    Section 10: a human-readable complement to /metrics - the same
    underlying data (Prometheus's format is built for machines/Grafana,
    not for opening in a browser or Swagger's "Try it out"), so "expose
    metrics for the service" (bullet 1) has an answer either way you want
    to look at it. drift/positive-rate figures here come straight from
    src/monitoring/drift.compute_drift_report() - the exact same function
    the /metrics DriftCollector calls, so this endpoint and Grafana's
    dashboard can never disagree with each other.
    """

    return {
        "predictions_logged_total": total_count(),
        "drift": compute_drift_report(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_single(request: OrderRequest):
    """
    Section 7: one order in, one prediction out. All the actual work -
    validate, engineer features, preprocess, predict, log - happens in
    src/inference/pipeline.run_pipeline(), the exact same function
    tests/integration/test_pipeline_e2e.py already exercises directly. A
    bad payload never reaches this function (422, via the
    RequestValidationError handler above); a well-typed payload that fails
    Great Expectations raises DataValidationError, which propagates out of
    this route uncaught and is turned into a 400 by that handler too - this
    route does no error handling of its own on purpose.
    """

    prediction, probability = run_pipeline(request, source="predict")

    return PredictionResponse(
        prediction=int(prediction),
        probability=probability,
        model_version=MODEL_VERSION,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    """
    Section 7: many orders in, one result per order out - each order run
    through the exact same run_pipeline() as /predict, one at a time.

    Unlike /predict, a single order failing Great Expectations does NOT
    fail the whole call: it's caught here and reported as that order's own
    `error`, so one bad row in a batch of 500 doesn't cost the other 499
    their predictions (see BatchPredictionResponse's docstring). A
    malformed order SHAPE (wrong type, bad Literal) still fails the whole
    request with a 422, before this function ever runs - that's a client
    bug, not a per-row data-quality question.
    """

    results = []
    succeeded = 0
    failed = 0

    for index, order in enumerate(request.orders):
        try:
            prediction, probability = run_pipeline(order, source="predict_batch")

            results.append(
                BatchPredictionItem(
                    index=index,
                    success=True,
                    result=PredictionResponse(
                        prediction=int(prediction),
                        probability=probability,
                        model_version=MODEL_VERSION,
                    ),
                )
            )
            succeeded += 1

        except DataValidationError as exc:
            results.append(
                BatchPredictionItem(
                    index=index,
                    success=False,
                    error=f"data_validation_failed: {'; '.join(exc.failures)}",
                )
            )
            failed += 1

        except ValueError as exc:
            results.append(
                BatchPredictionItem(
                    index=index, success=False, error=f"bad_request: {exc}"
                )
            )
            failed += 1

    logger.info(
        "batch predict | total=%s | succeeded=%s | failed=%s",
        len(request.orders),
        succeeded,
        failed,
    )

    return BatchPredictionResponse(results=results, succeeded=succeeded, failed=failed)
