import time

from src.data.preprocessing import preprocess
from src.features.feature_engineering import create_features
from src.inference.input_builder import build_input_dataframe
from src.inference.model_registry import MODEL_VERSION_INFO
from src.inference.predict import load_model, predict
from src.schemas.request_schema import OrderRequest
from src.utils.logger import logger
from src.validation.data_validation import validate_request_data

model = load_model()

# Section 5: the real registered name/version/stage from MLflow, not a
# static config string - if you register a new version, every log line
# reflects it automatically after the next service restart.
MODEL_VERSION = (
    f"{MODEL_VERSION_INFO['model_name']} v{MODEL_VERSION_INFO['model_version']}"
    f" ({MODEL_VERSION_INFO['stage']})"
)


def run_pipeline(request: OrderRequest):
    """
    Single entry point: one OrderRequest in, (prediction, probability) out.
    Order mirrors what a real deployment should do, in the sequence Section
    4 adds validation to Section 2's transform chain:

        build raw row -> Great Expectations validation -> engineer
        review_missing -> impute -> scale -> encode -> predict

    Logs input, output, latency, and model version for every call
    (Section 3) here rather than in the API layer, so it covers /predict,
    /predict/batch, and any direct/CLI/test call alike.
    """

    start = time.perf_counter()

    try:
        raw_df = build_input_dataframe(request)

        validate_request_data(raw_df)

        engineered_df = create_features(raw_df)
        model_ready_df = preprocess(engineered_df)

        prediction, probability = predict(model, model_ready_df)

        result_prediction = prediction[0]
        result_probability = float(probability[0])

        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "prediction request | input=%s | prediction=%s | probability=%.4f "
            "| latency_ms=%.2f | model_version=%s",
            request.model_dump(),
            result_prediction,
            result_probability,
            latency_ms,
            MODEL_VERSION,
        )

        return result_prediction, result_probability

    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000

        logger.error(
            "prediction request FAILED | input=%s | latency_ms=%.2f | model_version=%s",
            request.model_dump(),
            latency_ms,
            MODEL_VERSION,
            exc_info=True,
        )

        raise
