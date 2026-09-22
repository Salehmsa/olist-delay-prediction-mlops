# Training notebooks

Task 3's own framing: training happens only in the notebooks, this repo's
`src/`/`app/` code only ever *serves* what they already fit (see the main
README's "Model provenance" and "Proof: the pipeline's output matches the
notebook's output on the same input").

**Included here:**

- `06_Model Training  Evaluation.ipynb` — trains the baseline (Dummy),
  Logistic Regression, and Random Forest models on the pre-engineered
  `train_features.parquet` / `validation_features.parquet` /
  `test_features.parquet`, compares them, evaluates the selected model
  (Random Forest) on the held-out test set, and saves
  `artifacts/final_model.pkl` — the exact file this repo ships as
  `models/final_model.pkl` (confirmed independently: same
  `RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)`,
  same 42/42 features, same `roc_auc=0.6105108647035242` down to the last
  digit as `src/training/log_model.py`'s `KNOWN_METRICS`).

**Not yet included:** notebooks 01–05 (data loading/cleaning, label
creation, EDA, and — most relevantly — the feature-engineering notebook
that turns a raw order into the 42-column matrix Notebook 06 trains on).
`src/features/feature_engineering.py` and `src/data/preprocessing.py`
document, in their own docstrings, which specific notebook each piece of
logic is meant to mirror (e.g. `review_missing` → Notebook 05), but
without that notebook itself in this repo, that's a traceable claim, not
an independently re-checked one. `tests/model/test_notebook_parity.py`
proves what's provable without it: the SAME fitted model + fitted
preprocessing objects give identical output through this repo's pipeline
as they would called directly, by hand, on the same input — see that
file's own docstring for exactly what that does and doesn't cover.

None of these notebooks are executed by the service, the Docker image, or
CI — they're here for lineage and review only (`.dockerignore` excludes
this whole directory from the API/register image build context).
