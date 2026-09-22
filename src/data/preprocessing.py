import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from src.inference.model_registry import MODEL_VERSION_INFO

# Section 5: encoder/scaler/imputer/feature_list no longer come from
# models/*.pkl directly - they're pulled from the SAME MLflow run that
# produced the currently-aliased model (src/training/log_model.py logs the
# model and this bundle together on purpose), so a model version and its
# preprocessing can never drift apart even if models/ changes locally later.
_preprocessing_dir = Path(
    mlflow.artifacts.download_artifacts(
        run_id=MODEL_VERSION_INFO["run_id"], artifact_path="preprocessing"
    )
)

imputer = joblib.load(_preprocessing_dir / "imputer.pkl")
encoder = joblib.load(_preprocessing_dir / "encoder.pkl")
scaler = joblib.load(_preprocessing_dir / "scaler.pkl")

with open(_preprocessing_dir / "feature_list.json", "r", encoding="utf-8") as f:
    FEATURE_LIST = json.load(f)

# Recovered from the fitted objects' own feature_names_in_ rather than
# retyped by hand. NUMERIC_COLUMNS and SCALED_COLUMNS are deliberately kept
# separate rather than assumed identical: the imputer only needs columns
# that can genuinely be missing, while review_missing - an engineered flag
# added by src/features/feature_engineering.py, never itself null by
# construction - has no reason to have been part of the imputer's fit, even
# though feature_list.json places it inside the scaled numeric block
# (suggesting the scaler WAS fit on it, possibly by StandardScaler().fit()
# on a DataFrame that included it).
NUMERIC_COLUMNS = list(imputer.feature_names_in_)
CATEGORICAL_COLUMNS = list(encoder.feature_names_in_)

# The scaler only gets feature_names_in_ if it was fit directly on a
# DataFrame - fitting it on imputer.transform(...)'s plain ndarray output
# (an easy, common pattern when chaining impute -> scale) leaves no
# column-name metadata at all. When that metadata is missing, the only safe
# assumption is that the scaler was fit on the same columns as the imputer
# (NUMERIC_COLUMNS) - preprocess() below still correctly recovers
# review_missing either way via its "still_missing" pass-through; without
# this metadata it just can't also tell whether review_missing itself was
# part of the scaled block.
SCALED_COLUMNS = list(getattr(scaler, "feature_names_in_", NUMERIC_COLUMNS))


def load_preprocessors():
    """Load all preprocessing objects."""

    return imputer, encoder, scaler


def apply_imputer(df: pd.DataFrame) -> pd.DataFrame:
    """Median-impute the imputer's own fitted columns, in the order it was fit on."""

    transformed_data = imputer.transform(df[NUMERIC_COLUMNS])

    return pd.DataFrame(transformed_data, columns=NUMERIC_COLUMNS, index=df.index)


def apply_scaler(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standard-scale the scaler's own fitted columns (SCALED_COLUMNS) out of
    df. Expects df to already carry clean (non-null) values for every one
    of those columns - preprocess() below is what guarantees that, by
    imputing NUMERIC_COLUMNS first and merging the result back into df
    before calling this, so review_missing (never null) rides along
    untouched if the scaler expects it too.
    """

    transformed_data = scaler.transform(df[SCALED_COLUMNS])

    return pd.DataFrame(transformed_data, columns=SCALED_COLUMNS, index=df.index)


def apply_encoder(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode order_status + customer_state. The fitted encoder uses
    handle_unknown="ignore" as a pickled-in safety net, but request_schema.py
    already restricts both fields to the encoder's exact known categories via
    Literal types, so the API rejects a bad value before it gets here.
    """

    transformed_data = encoder.transform(df[CATEGORICAL_COLUMNS])

    if hasattr(transformed_data, "toarray"):
        transformed_data = transformed_data.toarray()

    columns = encoder.get_feature_names_out(CATEGORICAL_COLUMNS)

    return pd.DataFrame(transformed_data, columns=columns, index=df.index)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main preprocessing pipeline, in the exact order Notebook 06 applied it:
    impute -> scale the numeric columns, one-hot encode the categorical
    columns, concatenate, then reindex to the registered model's
    feature_list.json so a column-order mistake fails loudly here instead of
    silently feeding the model the wrong value under the wrong feature name.

    review_missing is threaded through whichever way the real fitted scaler
    actually expects: included in the scaled block if SCALED_COLUMNS covers
    it, or carried through unscaled at the end below if it doesn't. Either
    way, a genuinely unresolvable gap still raises loudly rather than
    silently dropping a column the model needs.
    """

    df = df.copy()

    imputed_numeric = apply_imputer(df)

    scaler_input = df.copy()
    scaler_input[NUMERIC_COLUMNS] = imputed_numeric[NUMERIC_COLUMNS]
    scaled_numeric = apply_scaler(scaler_input)

    encoded_categorical = apply_encoder(df)

    combined = pd.concat([scaled_numeric, encoded_categorical], axis=1)

    # Anything feature_list.json still needs that neither the scaler nor
    # the encoder produced (review_missing, if SCALED_COLUMNS didn't cover
    # it) is already a complete, engineered column sitting in df untouched.
    still_missing = [
        column
        for column in FEATURE_LIST
        if column not in combined.columns and column in df.columns
    ]
    if still_missing:
        combined = pd.concat([combined, df[still_missing]], axis=1)

    missing = set(FEATURE_LIST) - set(combined.columns)
    if missing:
        raise ValueError(
            f"preprocess() did not produce required column(s): {sorted(missing)}"
        )

    return combined[FEATURE_LIST]
