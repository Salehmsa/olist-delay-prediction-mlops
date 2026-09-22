"""
Section 10, bullet 3: "store prediction logs so you can evaluate them
later once the real delivery date arrives" - persist every prediction
this service makes, and give bullet 2 (drift) real data to compute over.

SQLite, not a flat file: this needs to be QUERIED later (rolling drift
stats now, a real ground-truth join once an order's actual delivery date
is known), not just archived - the same reasoning config.yaml's own
mlflow.db already uses for MLflow's tracking store. It lives under
logs/ specifically so it rides docker-compose.yml's EXISTING api_logs
volume - no new volume, no new service, persists across container
restarts for free.

order_id (src/schemas/request_schema.py) is optional and is the one
field that decides whether a row is actually useful for the "evaluate
later" half of this bullet: without it, this log can tell you WHAT the
service predicted, but never which real Olist order that was. When a
caller omits it, the row is still logged (drift tracking still works),
just with order_id = NULL - a real, visible limitation, not a hidden one.
"""

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from src.utils.config_loader import load_config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    order_id TEXT,
    source TEXT NOT NULL,
    prediction INTEGER NOT NULL,
    probability REAL NOT NULL,
    model_version TEXT NOT NULL,
    latency_ms REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(ts_utc);
"""


def _db_path() -> Path:
    config = load_config()
    path = Path(config["monitoring"]["prediction_log_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _connect():
    """
    Opens the DB, makes sure the table exists (cheap - CREATE TABLE IF
    NOT EXISTS - and idempotent, so every call paying this cost is fine
    at this service's scale), commits on a clean exit, always closes.
    An exception inside the `with` block skips the commit and propagates
    normally - nothing partial is ever persisted.
    """

    conn = sqlite3.connect(_db_path(), timeout=5)
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_prediction(
    *,
    order_id: Optional[str],
    source: str,
    prediction: int,
    probability: float,
    model_version: str,
    latency_ms: float,
) -> None:
    """
    Best-effort by design: a logging failure must never take down a real
    prediction response that has already succeeded. Errors are logged,
    not raised - the inverse of run_pipeline()'s own try/except, which
    re-raises because there the caller hasn't got a result yet.
    """

    from src.utils.logger import logger  # local import avoids an import cycle

    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO predictions "
                "(ts_utc, order_id, source, prediction, probability, "
                " model_version, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    order_id,
                    source,
                    int(prediction),
                    float(probability),
                    model_version,
                    float(latency_ms),
                ),
            )
    except Exception:
        logger.error("failed to write prediction log", exc_info=True)


def recent_predictions(limit: int) -> list[dict]:
    """Most recent `limit` rows, newest first - drift.py's raw material."""

    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


def total_count() -> int:
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
