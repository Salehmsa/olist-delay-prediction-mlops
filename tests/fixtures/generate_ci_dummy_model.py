"""
Section 9: generates tests/fixtures/ci_dummy_model.pkl - a tiny, schema-
correct binary classifier used ONLY by the CI workflow, never by a human or
by Docker.

Why this exists: the real models/final_model.pkl is DVC-managed (Section 4),
and its DVC remote is a local folder on Saleh's own machine - unreachable
from a GitHub Actions runner. But every import of src.inference.pipeline /
src.inference.model_registry resolves a registered model AT IMPORT TIME
(see tests/conftest.py's own docstring), so pytest collection itself needs
*something* loadable at models/final_model.pkl to avoid crashing before a
single test runs.

tests/model/test_predict.py already documents its own philosophy: "schema-
valid synthetic inputs with asserted well-formed outputs, not ground-truth-
labeled examples." None of the model/integration tests assert a specific
prediction - only structural contracts (binary classifier, one prediction
per row, probability in [0, 1], deterministic, handles missing fields).
A dummy model with the right shape satisfies all of that.

This is NOT a replacement for the real model anywhere real predictions
matter (local dev, Docker, the actual registered "production" version) -
those all still use the genuine DVC-managed models/final_model.pkl. CI
verifies the CODE wires together correctly; it does not and cannot verify
real predictive performance without reaching Saleh's machine, which is by
design (Section 4's chosen DVC remote is local-only).

Re-run this only if models/feature_list.json's column set ever changes:

    python tests/fixtures/generate_ci_dummy_model.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

_HERE = Path(__file__).parent
_FEATURE_LIST_PATH = _HERE.parent.parent / "models" / "feature_list.json"
_OUTPUT_PATH = _HERE / "ci_dummy_model.pkl"


def main() -> None:
    with open(_FEATURE_LIST_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    rng = np.random.default_rng(seed=42)
    n_samples = 200

    x = rng.standard_normal((n_samples, len(feature_names)))
    # A real (if meaningless) decision boundary, not pure label noise -
    # keeps predict_proba from degenerating to a constant 0.0/1.0, which
    # would still pass every test above but is a more honest stand-in for
    # "a classifier that actually looks at its inputs."
    y = (x[:, 0] + x[:, 1] > 0).astype(int)

    model = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
    model.fit(x, y)
    model.feature_names_in_ = np.array(feature_names)

    joblib.dump(model, _OUTPUT_PATH)

    print(f"Wrote {_OUTPUT_PATH} ({_OUTPUT_PATH.stat().st_size} bytes)")
    print(f"  features: {len(feature_names)}, classes: {model.classes_.tolist()}")


if __name__ == "__main__":
    main()
