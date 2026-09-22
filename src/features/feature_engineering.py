import pandas as pd


def add_review_missing_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Notebook 05 engineered `review_missing` as 1 when an order has no
    review_score yet, 0 otherwise — then let the imputer fill review_score
    itself with the training median. Reproduce both steps here, in the same
    order, so a new order with no review yet is handled exactly like the
    missing reviews the model was trained on.
    """

    df = df.copy()

    df["review_missing"] = df["review_score"].isna().astype(float)

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Entry point the inference pipeline calls after build_input_dataframe()
    and before preprocess(). Previously this also derived purchase_month /
    purchase_weekday / purchase_hour from an order_purchase_timestamp — that
    was dead code: models/feature_list.json (the model's real 42-column
    contract) has no such columns, and no caller ever passed a timestamp in.
    Removed rather than left unused.
    """

    df = add_review_missing_flag(df)

    return df
