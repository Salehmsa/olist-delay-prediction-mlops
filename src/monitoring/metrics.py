"""
Section 10, bullet 1: the custom, ML-specific Prometheus metrics this
service exposes on top of what app/main.py's Instrumentator already gives
for free (HTTP request count / latency / status per route - the generic
RED metrics every service gets). These are specific to what THIS service
actually does:

- predictions_total: predictions made, BY PREDICTED CLASS - so "request
  count" can be read as "predictions by outcome", not just raw HTTP
  traffic.
- prediction_latency_seconds: the MODEL PIPELINE's own latency (Great
  Expectations -> features -> preprocess -> predict), separate from the
  Instrumentator's HTTP-level latency (which also includes FastAPI/
  Starlette/JSON overhead) - isolates the number that actually matters
  if inference itself gets slow.
- DriftCollector: exposes the positive-rate and drift-status gauges
  src/monitoring/drift.py computes.

DriftCollector is a CUSTOM Prometheus collector, not a plain Gauge you
.set() - these two values are DERIVED from the prediction log (a SQLite
query), not incremented inline during a request the way predictions_total
is. A custom collector computes them fresh every time Prometheus actually
scrapes /metrics, instead of a Gauge sitting at whatever value it was
last set to - the textbook-correct Prometheus pattern for a metric backed
by external state, and it means the drift verdict Grafana shows is never
more stale than Prometheus's own scrape interval
(monitoring/prometheus/prometheus.yml).

Registered once, at import time, onto prometheus_client's default global
REGISTRY - the same registry app/main.py's Instrumentator().expose(app)
serves /metrics from, so nothing extra is needed to wire this in beyond
importing this module.
"""

from prometheus_client import REGISTRY, Counter, Histogram
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

predictions_total = Counter(
    "olist_predictions_total",
    "Total predictions made, by predicted class",
    ["predicted_class"],
)

prediction_latency_seconds = Histogram(
    "olist_prediction_latency_seconds",
    "Model pipeline latency (validate -> features -> preprocess -> predict), seconds",
)


class DriftCollector(Collector):
    def collect(self):
        from src.monitoring.drift import (
            compute_drift_report,
        )  # local: avoids import cycle

        report = compute_drift_report()

        rate_gauge = GaugeMetricFamily(
            "olist_prediction_positive_rate",
            "Rolling proportion of recent predictions flagging a delay "
            "(insufficient_data until drift_min_sample_size is reached)",
        )
        status_gauge = GaugeMetricFamily(
            "olist_prediction_drift_status",
            "0 = ok, 1 = drift_detected, -1 = insufficient_data",
        )

        if report["status"] == "insufficient_data":
            status_gauge.add_metric([], -1)
        else:
            rate_gauge.add_metric([], report["current_positive_rate"])
            status_gauge.add_metric(
                [], 1 if report["status"] == "drift_detected" else 0
            )

        yield rate_gauge
        yield status_gauge


_drift_collector_registered = False


def register_drift_collector() -> None:
    """
    Idempotent on purpose, via a plain module-level flag rather than
    reaching into prometheus_client's internals: Python only executes
    app/main.py's top level once per process (re-imports hit the
    sys.modules cache), so in practice this already only runs once - this
    guard is the cheap extra safety net for the one case that wouldn't be
    true (a future importlib.reload, or an app-factory pattern that
    builds the app more than once), since a second REGISTRY.register()
    of the same collector name raises ValueError("Duplicated
    timeseries...") and would crash the whole app at import time.
    """

    global _drift_collector_registered
    if not _drift_collector_registered:
        REGISTRY.register(DriftCollector())
        _drift_collector_registered = True
