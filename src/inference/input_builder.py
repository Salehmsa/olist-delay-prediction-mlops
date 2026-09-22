import pandas as pd

from src.schemas.request_schema import OrderRequest


def build_input_dataframe(request: OrderRequest) -> pd.DataFrame:
    """
    One order request -> one-row DataFrame with exactly the raw columns
    imputer.pkl / scaler.pkl / encoder.pkl expect (src/data/preprocessing.py).
    Column names must match feature_names_in_ on those fitted objects — they
    do here because request_schema.py was written directly from that
    metadata, not guessed.

    review_missing is intentionally absent: feature_engineering.create_features()
    adds it downstream, derived from review_score.
    """

    data = {
        "customer_zip_code_prefix": [request.customer_zip_code_prefix],
        "total_payment": [request.total_payment],
        "avg_payment": [request.avg_payment],
        "max_installments": [request.max_installments],
        "payment_count": [request.payment_count],
        "items_count": [request.items_count],
        "total_price": [request.total_price],
        "total_freight": [request.total_freight],
        "unique_products": [request.unique_products],
        "unique_sellers": [request.unique_sellers],
        "review_score": [request.review_score],
        "approval_delay_hours": [request.approval_delay_hours],
        "order_status": [request.order_status],
        "customer_state": [request.customer_state],
    }

    return pd.DataFrame(data)
