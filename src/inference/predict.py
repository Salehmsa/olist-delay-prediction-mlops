import mlflow

from src.inference.model_registry import MODEL_URI


def load_model():
    """
    Section 5: loads the model MLflow currently has aliased as "production"
    (config.yaml's mlflow.model_alias) — not models/final_model.pkl directly
    anymore. If this raises "not found", run
    `python -m src.training.log_model` first.
    """

    return mlflow.sklearn.load_model(MODEL_URI)


def predict(model, df):
    """
    df must already be the fully preprocessed, 42-column output of
    src.data.preprocessing.preprocess() - this function does no
    preprocessing itself.

    predict_proba(df)[:, 1] is P(is_delayed == 1). Confirmed against
    Notebook 02 (Label Creation), which defines the target variable
    is_delayed as 1 = Delayed, 0 = On Time - so prediction == 1 means the
    model flags this order as late, prediction == 0 means on time, and
    probability is specifically the model's estimated chance THIS order
    is delayed, not "confidence in whichever class it picked".
    """

    prediction = model.predict(df)

    probability = (model.predict_proba(df))[:, 1]

    return prediction, probability
