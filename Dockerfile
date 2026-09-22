# Section 8: the API image. Two stages so the final image never carries
# pip's build cache or wheel downloads - only the installed packages
# themselves get copied forward.

# --- Stage 1: resolve dependencies into a throwaway virtualenv ---
FROM python:3.12-slim AS builder

# python:3.12-slim, not 3.11 - Section 7's scikit-learn version-skew fix
# pinned requirements.txt to the exact interpreter the fitted
# models/*.pkl were verified against (see the comment at the top of
# requirements.txt). Matching it here too means the container runs the
# same wheels that were already proven correct, not a different build.
WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Only requirements.txt, never requirements-dev.txt - pytest/ruff/black/
# pre-commit/dvc are dev-only tools (see requirements-dev.txt's own
# comments) and have no business in the image that actually serves
# predictions.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Stage 2: the runtime image ---
FROM python:3.12-slim AS runtime

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# PYTHONPATH=/app: uvicorn's own CLI inserts the cwd into sys.path when it
# loads app.main (so the api service's CMD below never needed this), but
# docker-compose.yml's `register` service overrides CMD to a plain
# `python scripts/register_if_needed.py` - and a bare `python script.py`
# (no -m) only puts that script's OWN directory (/app/scripts) on
# sys.path, not /app, so `from src.training.log_model import ...` failed
# with ModuleNotFoundError: No module named 'src'. Setting PYTHONPATH here
# fixes it for every entrypoint in this image, present or future, rather
# than special-casing register's own invocation.

# Never run the service as root inside the container.
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# .dockerignore keeps this from also picking up notebooks/, tests/,
# .venv/, mlflow.db, mlruns/, and anything else that either doesn't belong
# in the image or (mlflow.db/mlruns/) is superseded by docker-compose.yml's
# separate mlflow service + mlflow_data volume - see .dockerignore's own
# comments. models/*.pkl IS included on purpose: the register service
# (docker-compose.yml) needs it to log/register a model on first boot,
# the exact same way `python -m src.training.log_model` does locally.
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

# Container-level liveness, the same /health route Section 3's
# readiness design already exposes. A stdlib urllib call needs no extra
# OS package on top of what -slim already has (curl isn't in -slim).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

# No --reload - that's a dev-only uvicorn flag (it watches the filesystem
# for changes, which a built image never has). docker-compose.yml's
# `register` service reuses this same image with its command overridden
# to `python scripts/register_if_needed.py` instead of this CMD.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
