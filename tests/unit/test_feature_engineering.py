"""
Unit tests: src/features/feature_engineering.py.
"""

import numpy as np
import pandas as pd

from src.features.feature_engineering import add_review_missing_flag, create_features


def test_review_missing_is_zero_when_review_score_present():
    df = pd.DataFrame({"review_score": [5.0, 3.0]})

    result = add_review_missing_flag(df)

    assert list(result["review_missing"]) == [0.0, 0.0]


def test_review_missing_is_one_when_review_score_is_nan():
    df = pd.DataFrame({"review_score": [np.nan, 4.0, np.nan]})

    result = add_review_missing_flag(df)

    assert list(result["review_missing"]) == [1.0, 0.0, 1.0]


def test_add_review_missing_flag_does_not_mutate_the_input():
    """
    Both functions df.copy() before adding a column on purpose - a caller
    holding the original raw_df (src/inference/pipeline.py does, for
    logging) must not see it grow an extra column as a side effect.
    """

    df = pd.DataFrame({"review_score": [5.0]})

    add_review_missing_flag(df)

    assert "review_missing" not in df.columns


def test_create_features_adds_review_missing_and_keeps_other_columns():
    df = pd.DataFrame({"review_score": [np.nan], "order_status": ["delivered"]})

    result = create_features(df)

    assert set(result.columns) == {"review_score", "order_status", "review_missing"}
    assert result.loc[0, "review_missing"] == 1.0
    assert result.loc[0, "order_status"] == "delivered"
