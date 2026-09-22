"""
Section 10, bullet 2: "track the distribution of predictions over time,
and watch for drift" - computed from the SQLite log prediction_log.py
writes to, since drift is a property of the FLOW of predictions over
time, not any single one.

Two independent signals, on purpose:

1. Positive-rate drift - the rolling proportion of predictions flagging
   "delayed" (prediction == 1) over the last drift_window_size rows,
   compared against config.yaml's baseline_positive_rate: the REAL
   positive rate Notebook 06's held-out test set showed (~6.6% - see
   src/training/log_model.py's KNOWN_METRICS comment and README's "Model
   provenance" table). Simple, interpretable, and anchored to a number
   this project can actually stand behind - this is the signal
   monitoring/prometheus/alerts.yml's PredictionDriftDetected rule fires
   on (via the Prometheus gauge src/monitoring/metrics.py exposes).

2. PSI (Population Stability Index) on predicted probabilities - a more
   sensitive secondary signal, comparing the CURRENT window's score
   distribution against a REFERENCE window drawn from this service's OWN
   earlier predictions (the drift_window_size rows immediately before the
   current ones). Deliberately NOT compared against Notebook 06's actual
   training/validation scores: this service only ever received the
   summary metrics in KNOWN_METRICS, never the raw prediction array, and
   Section 5's own boundary is "no training inside the inference
   pipeline" - reaching back into notebook internals to fetch a reference
   array would cross that same line. A self-referential reference window
   is a real, commonly used pattern for exactly this gap, and still
   catches what matters most in production - today's traffic looking
   different from last week's. Exposed for diagnosis (Grafana panel,
   /monitoring/summary) but deliberately NOT wired to its own alert rule
   - see README's "Alerting" section for why.
"""

from typing import Optional

import numpy as np

from src.monitoring.prediction_log import recent_predictions, total_count
from src.utils.config_loader import load_config

_PSI_BINS = np.linspace(0.0, 1.0, 11)  # 10 equal-width bins over [0, 1]
_PSI_EPSILON = 1e-4  # avoids log(0) / divide-by-0 when a bin is empty


def _positive_rate(rows: list[dict]) -> Optional[float]:
    if not rows:
        return None
    return sum(row["prediction"] for row in rows) / len(rows)


def _psi(reference_scores: np.ndarray, current_scores: np.ndarray) -> float:
    """
    Standard PSI: sum over bins of (cur% - ref%) * ln(cur% / ref%). 10
    equal-width bins over [0, 1] is a common default for a bounded
    probability score - fine-grained enough to catch a real shift,
    coarse enough that a bin isn't just noise at this service's traffic
    volume.
    """

    ref_counts, _ = np.histogram(reference_scores, bins=_PSI_BINS)
    cur_counts, _ = np.histogram(current_scores, bins=_PSI_BINS)

    ref_pct = ref_counts / max(len(reference_scores), 1) + _PSI_EPSILON
    cur_pct = cur_counts / max(len(current_scores), 1) + _PSI_EPSILON

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_drift_report() -> dict:
    """
    The single function everything else in Section 10 calls: the
    DriftCollector Prometheus reads on every /metrics scrape
    (src/monitoring/metrics.py) and the /monitoring/summary route
    (app/main.py) both call this directly, so there is exactly one
    place this logic lives.
    """

    config = load_config()
    mon_cfg = config["monitoring"]

    baseline_rate = mon_cfg["baseline_positive_rate"]
    window_size = mon_cfg["drift_window_size"]
    min_sample = mon_cfg["drift_min_sample_size"]
    high_threshold = mon_cfg["drift_high_threshold"]
    low_threshold = mon_cfg["drift_low_threshold"]

    total_logged = total_count()

    # One fetch, newest-first; current = the first window_size rows,
    # reference (for PSI below) = the window_size rows right before
    # those - both slices of the SAME fetch, so they can never overlap
    # or race against a write landing between two separate queries.
    rows = recent_predictions(2 * window_size)
    current_rows = rows[:window_size]
    reference_rows = rows[window_size : 2 * window_size]

    current_rate = _positive_rate(current_rows)

    if current_rate is None or len(current_rows) < min_sample:
        return {
            "status": "insufficient_data",
            "reason": (
                f"only {len(current_rows)} prediction(s) logged so far, need "
                f"at least {min_sample} before a drift verdict is meaningful"
            ),
            "predictions_logged_total": total_logged,
            "baseline_positive_rate": baseline_rate,
        }

    is_drifting = not (low_threshold <= current_rate <= high_threshold)
    status = "drift_detected" if is_drifting else "ok"

    psi_value = None
    psi_note = (
        f"needs {2 * window_size} logged predictions to compute (has {total_logged})"
    )
    if len(reference_rows) >= min_sample:
        reference_scores = np.array([row["probability"] for row in reference_rows])
        current_scores = np.array([row["probability"] for row in current_rows])
        psi_value = round(_psi(reference_scores, current_scores), 4)
        psi_note = (
            "compares the current window to this service's OWN earlier "
            "predictions, not Notebook 06's training scores - see this "
            "module's docstring"
        )

    return {
        "status": status,
        "predictions_logged_total": total_logged,
        "window_size": len(current_rows),
        "current_positive_rate": round(current_rate, 4),
        "baseline_positive_rate": baseline_rate,
        "drift_bounds": {"low": low_threshold, "high": high_threshold},
        "psi_probability_drift": psi_value,
        "psi_note": psi_note,
    }
