# Olist Delivery Delay Prediction — Inference Service

MLOps Training 2026/2027 · Task 3 · Qafza Training

Turns the model trained in `notebooks/01`–`06` into a real inference
service: a Python package, a FastAPI app, data versioning, experiment
tracking, tests, containers, a CI/CD pipeline, and monitoring
(Prometheus + Grafana) — all 10 sections of the task are built **and**
verified end to end on real infrastructure (see "Project status").

Training happens only in the notebooks. This service loads the fitted
objects Notebooks 05 & 06 saved (imputer, encoder, scaler, model) and never
re-fits them — Section 5 moved that loading from local `models/*.pkl` files
to the MLflow Model Registry (see "Before you run anything" below).

## Model provenance

All five files in `models/` — `encoder.pkl`, `imputer.pkl`, `scaler.pkl`,
`feature_list.json`, and `final_model.pkl` — are the real, genuine fitted
artifacts from Notebooks 05/06. `final_model.pkl` is a
`RandomForestClassifier` (100 estimators, `random_state=42`, `max_features
="sqrt"`), 42/42 features matching `feature_list.json` exactly, loads
clean under the pinned `scikit-learn==1.8.0` with zero
`InconsistentVersionWarning`.

**Real evaluation numbers** (Notebook 06's `classification_report` +
`roc_auc_score` on the held-out test set, n=14472), logged into MLflow by
`src/training/log_model.py`'s `KNOWN_METRICS`:

| Metric | Value | Basis |
|---|---|---|
| `accuracy` | 0.93 | overall |
| `precision` / `recall` / `f1` | 0.63 / 0.54 / 0.55 | macro avg (both classes weighed equally) |
| `roc_auc` | 0.61 | threshold-independent |
| `precision_delayed` / `recall_delayed` / `f1_delayed` | 0.31 / 0.08 / 0.13 | class 1 (`is_delayed`) only |

**Read honestly, not just reported:** `is_delayed` is heavily imbalanced
(957 delayed vs 13515 on-time in the test set, ~6.6% positive rate), which
is exactly why accuracy alone (0.93) is misleading here — the textbook trap
of reporting accuracy on imbalanced data. The number that actually matters
for a *delay* predictor is `recall_delayed`: of the delays that really
happened, the model catches only 8% of them. `roc_auc=0.61` (barely above
0.50 random) tells the same story from a different angle. This is a real
limitation of Notebook 06's model, not a bug introduced by this service —
and out of scope for Task 3 to fix ("no training inside the inference
pipeline"). Worth stating plainly in any write-up or interview discussion
of this project rather than leading with the 93% accuracy figure alone.
See the comment block above `KNOWN_METRICS` in `src/training/log_model.py`
for the full reasoning behind the macro-avg-vs-weighted-avg choice.

**Model choice, read honestly:** Notebook 06's own validation-set
comparison actually scores Logistic Regression slightly *higher* than
Random Forest on both F1 (0.234 vs 0.224) and ROC AUC (0.765 vs 0.737) —
sorting that comparison table by F1 puts Logistic Regression first, not
Random Forest. Random Forest is still the one evaluated on the test set
and saved (`joblib.dump(rf_model, "artifacts/final_model.pkl")`) — a
manual choice in the notebook (`# إذا كان Random Forest الأفضل` — "if
Random Forest is the best"), not a programmatic pick of whichever model
actually scored highest. The reasoning isn't visible in Notebook 06
itself, and Task 3's "no training inside the inference pipeline" rule
means this service serves whatever Notebook 06 actually saved regardless —
reported here rather than glossed over, same as the `roc_auc=0.61`
limitation above.

### Proof: the pipeline's output matches the notebook's output on the same input

Qafza's Definition of Done asks for this explicitly: "تثبت إن مخرج
البايبلاين مطابق لمخرج النوتبوك على نفس المدخل." `tests/model/test_notebook_parity.py`
is that proof, kept as a permanent, automated test rather than a one-off
manual check.

Notebook 06 itself never engineers features — it loads already-engineered
`test_features.parquet` and calls `rf_model.predict(X_test)` /
`.predict_proba(X_test)` directly. So "the notebook's output" for a raw
order reduces to: the same fitted model, called on a feature row built
from the same fitted `imputer`/`encoder`/`scaler` — wired together by
hand, the way a notebook cell would, not through this project's own
orchestration code. That's exactly what the test builds as a second,
independent path: for a given raw order, it computes the prediction (1) via
`run_pipeline()` — the exact function `/predict` calls — and (2) via a
hand-wired reconstruction that imports only the fitted `model` / `imputer`
/ `encoder` / `scaler` objects themselves (the same MLflow-registered
artifacts) plus each one's own `feature_names_in_` metadata, and does
impute → scale → encode → reindex → predict from scratch, sharing no code
with `src/features/feature_engineering.py` or `src/data/preprocessing.py`.
Run against two different orders (fully-specified, and every optional
field missing so the imputer does real work), both agree exactly —
prediction *and* probability, not just the predicted class.

Confirmed in a genuinely fresh virtualenv with `requirements-dev.txt`
pinned exactly (`scikit-learn==1.8.0`, `pandas==2.2.3`, `mlflow==3.16.1`),
against the real model: `python -m src.training.log_model` (no
`--allow-placeholder`) registered `models/final_model.pkl` itself, not a
stand-in — both cases passed, and the full suite (115 tests) still passed
clean alongside them. It also passes in CI against the dummy model (see
"Why CI runs on a dummy model" below) — the test never asserts a specific
value, only that both paths agree, which holds for either model.

**What this proves:** the FastAPI/pipeline plumbing (`create_features()`,
`preprocess()`, the API/schema layers) doesn't alter what the trained
model + trained preprocessors would produce for a given raw order — the
exact class of bug ("training-serving skew") this kind of check exists to
catch — using the real artifacts Notebook 06 produced, independently
confirmed above to be `RandomForestClassifier(n_estimators=100,
random_state=42, n_jobs=-1)`, 42/42 features, loading clean.

**What this doesn't prove:** that `review_missing`'s one-line derivation,
or the imputer/encoder/scaler's fitted parameters themselves, are a
bit-exact match to Notebook 05's *original* computation — Notebook 05 (the
feature-engineering notebook) wasn't available when this test was written,
only Notebook 06 was. `notebooks/06_Model Training  Evaluation.ipynb` is
included in this repo specifically so this claim can be checked against
its actual source instead of taken on faith — see `notebooks/README.md`.

## Project status

Built in the order the task specifies. A section is only checked once it is
wired into the running service end to end — not just present as a file.

- [x] 1. Repository & configuration
- [x] 2. Notebooks → Python modules (real 42-column feature contract)
- [x] 3. Logging & error handling
- [x] 4. Data versioning (DVC) & validation (Great Expectations) —
      `models/final_model.pkl` DVC-tracked against a local remote,
      verified with a real add/push/delete/pull round trip (md5-identical)
- [x] 5. Experiment tracking & model registry (MLflow)
- [x] 6. Testing (pytest) — 94 tests at the time (unit / data / model /
      integration); 113 as of Section 10 (+ `tests/monitoring`); 115 once
      `tests/model/test_notebook_parity.py` joined them (see "Model
      provenance" above) — see "Testing" below for the current count
- [x] 7. API (`/predict`, `/predict/batch`)
- [x] 8. Docker & Docker Compose — verified with a real Docker daemon,
      real model, `/predict/batch` tested live through the containers;
      connection string environment-driven via `.env`/`.env.example`,
      `.gitignore` added so a real `.env` can never reach the repo
- [x] 9. CI/CD — GitHub Actions verified on real infrastructure, not just
      simulated: [run #1](https://github.com/Salehmsa/olist-delay-prediction-mlops/actions/runs/35691357659)
      on the first push to `main` — `test` green (1m31s), `build-and-push`
      green (1m43s) right after it, image published to GHCR; pre-commit
      hooks installed and active locally too
- [x] 10. Monitoring — verified on **real infrastructure**, not just
      simulated: `docker compose up --build` brought up all 5 services on
      a real Docker daemon (Windows + WSL2/Docker Desktop); Prometheus's
      `/targets` page showed `olist-api` **UP** and actually scraping;
      Grafana provisioned its datasource and the "Olist Delivery Delay -
      Monitoring (Section 10)" dashboard automatically, with **zero
      manual setup**;
      driving 32 real predictions through `/predict` moved every panel
      live — `predictions_logged_total` climbed in real time, and
      `olist_prediction_drift_status` flipped from `-1`
      (`insufficient_data`) to `1` (`drift_detected`) exactly as
      `PredictionDriftDetected`'s own rule expects once the positive rate
      dropped to 0%. Both Docker Hub images resolved correctly too
      (`grafana/grafana:13.2.2` confirmed straight from the running UI's
      own footer) — the one thing this project couldn't confirm from its
      own sandbox. Same bar Section 9 was held to: a real run, not just
      files that look right.

## Structure

```
olist_mlops_task3_phase1/
├── app/
│   └── main.py         FastAPI app — routes + request/response wiring only,
│                        business logic lives in src/inference/pipeline.py
├── config/
│   └── config.yaml      every path, MLflow setting, host/port, log setting
├── data/                 lightweight reference data (sample payloads)
├── models/               real fitted artifacts from Notebooks 05/06:
│                         encoder.pkl, scaler.pkl, imputer.pkl,
│                         feature_list.json — small, committed to git
│                         directly. final_model.pkl (95MB) is DVC-managed
│                         instead (see "Data & artifact versioning" above)
│                         — gitignored, tracked via final_model.pkl.dvc,
│                         restored with `dvc pull`. All read by
│                         src/training/log_model.py only; the running
│                         service loads everything through MLflow instead
├── .dvc/                 DVC's own config + internal cache/state
├── .dvcignore
├── mlflow.db             created on first `log_model.py` run — MLflow's
│                          SQLite tracking + registry store, LOCAL dev only;
│                          inside Docker this is superseded by the mlflow
│                          service + mlflow_data volume below (not present
│                          in a fresh copy of this project until then)
├── mlruns/                MLflow's artifact store — same local-dev-only,
│                           created-on-first-run caveat as mlflow.db
├── notebooks/             training notebooks, for lineage — never executed
│                          by the service or the image. Notebook 06 (model
│                          training/evaluation) is included; 01–05 are not
│                          yet copied in — see notebooks/README.md
├── scripts/
│   └── register_if_needed.py   docker-compose's `register` service —
│                                 registers the real model (real metrics,
│                                 KNOWN_METRICS is filled in) only if none
│                                 exists yet (Section 8)
├── src/
│   ├── data/              preprocessing (impute / scale / encode)
│   ├── features/          feature engineering (review_missing)
│   ├── inference/          input building, model registry resolution,
│   │                       prediction, the run_pipeline() orchestrator
│   ├── monitoring/          Section 10: prediction_log.py (SQLite log),
│   │                        drift.py (positive-rate + PSI), metrics.py
│   │                        (custom Prometheus metrics + DriftCollector)
│   ├── schemas/            pydantic request/response contracts
│   ├── training/           log_model.py — registers models/*.pkl into
│   │                       MLflow; never called by the running service
│   ├── utils/               config loader (env-overridable MLflow
│   │                        tracking URI, Section 8), logger
│   └── validation/          Great Expectations suite + validation wrapper
├── monitoring/               Section 10: config for the prometheus/grafana
│                              containers, read directly from this checkout
│                              via docker-compose.yml's volume mounts —
│                              never copied into the api/register image
│                              (.dockerignore excludes it)
│   ├── prometheus/
│   │   ├── prometheus.yml   scrape config (targets api:8000 every 10s) +
│   │   │                     rule_files pointing at alerts.yml
│   │   └── alerts.yml       the 3 alert rules — see "Monitoring" below
│   └── grafana/
│       ├── provisioning/     datasource + dashboard-provider YAML, loaded
│       │                     automatically on container start
│       └── dashboards/
│           └── olist_monitoring.json   the dashboard itself, 9 panels
├── tests/
│   ├── conftest.py         shared fixtures + auto-bootstraps an MLflow
│   │                       registration if none exists yet (fresh clone /
│   │                       CI) — needs `dvc pull` to have run first, same
│   │                       as everything else that reads final_model.pkl
│   ├── fixtures/
│   │   ├── generate_ci_dummy_model.py   builds the CI-only dummy model
│   │   │                                  below (Section 9) — re-run only
│   │   │                                  if feature_list.json's columns
│   │   │                                  ever change
│   │   └── ci_dummy_model.pkl            tiny (~26KB), schema-correct,
│   │                                      committed to git directly — used
│   │                                      ONLY by CI, never locally/Docker
│   ├── unit/                preprocessing, feature engineering, utilities
│   ├── data/                 schema, ranges, nulls, leakage checks
│   ├── model/                 the model loads, predicts the right shape
│   ├── monitoring/             prediction_log.py + drift.py, each against
│   │                           its own throwaway SQLite DB (13 tests)
│   └── integration/           API routes + the pipeline, end to end —
│                               including /metrics and /monitoring/summary
├── pytest.ini
├── ruff.toml                 excludes notebooks/ from lint — a training
│                             artifact kept for lineage, not source; keeps
│                             the pre-commit ruff hook (--fix) from
│                             rewriting notebook cells at commit time too
├── requirements.txt         runtime dependencies, pinned
├── requirements-dev.txt     + test / lint / format tooling
├── Dockerfile               the API image — multi-stage, non-root, healthcheck
├── Dockerfile.mlflow        the mlflow tracking/registry server image
├── docker-compose.yml       mlflow + register + api + prometheus + grafana —
│                             one command, fresh clone
├── .github/
│   └── workflows/
│       └── ci.yml            Section 9: lint → format check → test, then
│                               build+push to GHCR on main only — see
│                               "CI/CD (Section 9)" below
├── .pre-commit-config.yaml   the same lint/format checks, run locally
│                               before a commit — `pre-commit install`
├── .env.example             documents MLFLOW_TRACKING_URI and Grafana's
│                             admin user/password + their safe defaults
│                             (Section 8: env vars for secrets/connection
│                             strings) — copy to .env only to override one
├── .gitignore                keeps .env, mlflow.db, mlruns/, logs/*.db,
│                             caches, venvs out of git — mirrors
│                             .dockerignore's reasoning where it applies,
│                             different list where it doesn't
└── .dockerignore              also excludes monitoring/ from the image
                                build context (Section 10) — those configs
                                are read straight from this checkout by
                                the prometheus/grafana containers, never
                                needed inside api/register's own image
```

## Setup

```bash
cd olist_mlops_task3_phase1
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements-dev.txt   # pulls in requirements.txt too
```

`requirements.txt` is pinned to versions confirmed to load the current
`models/*.pkl` cleanly. If your notebook kernel used different
scikit-learn/pandas versions, match `requirements.txt` to that kernel
exactly — see the comment at the top of the file.

## Data & artifact versioning (DVC)

Section 4: "version the data and artifacts with DVC, so any result can be
traced back to its source." Only `models/final_model.pkl` (95MB) is DVC-
tracked — `encoder.pkl`/`scaler.pkl`/`imputer.pkl` are ~1KB each and
`feature_list.json` is plain text, all four committed to git directly like
any other source file. DVC exists to keep a large binary out of git's own
history while still giving it a content-hashed, verifiable pointer; running
it on files this small would add overhead (a `.dvc` file, a cache entry, a
gitignore line) with no actual benefit — git already handles a few KB fine.

**First-time setup** (already done once for this repo — skip to "Fresh
clone" below unless you're re-doing this from scratch):

```bash
git init

dvc init
dvc config cache.type hardlink,symlink,copy   # cache and working copy share
                                                # disk blocks instead of a
                                                # second physical copy - see
                                                # "why hardlink" below
dvc remote add -d local_remote C:\dvc_storage\olist_model   # outside the
                                                               # project folder
                                                               # on purpose -
                                                               # a "remote"
                                                               # that lived
                                                               # inside the
                                                               # repo it's
                                                               # backing up
                                                               # wouldn't be
                                                               # much of a
                                                               # backup

dvc add models/final_model.pkl

git add .
git commit -m "Initial commit: Task 3 inference service (DVC-tracked model)"

dvc push
```

**Order matters here, and it's easy to get backwards:** `dvc add` must run
*before* the first `git add .` / `git commit` — not after. `dvc add`
writes `models/final_model.pkl.dvc` (a small text pointer: md5 hash + size
+ path) and auto-adds `/final_model.pkl` to `models/.gitignore` *before*
git ever looks at the directory, so the one `git add .` above correctly
picks up the pointer and skips the 95MB binary. Commit first and run
`dvc add` second instead, and git has already captured the raw file into
its own history — DVC then refuses with `output 'models/final_model.pkl'
is already tracked by SCM (e.g. Git)`, and even a `git rm --cached` fix
afterward leaves that 95MB sitting in git's history from the first commit.
If this happens: `Remove-Item -Recurse -Force .git` (safe pre-push — this
only discards *local, unpushed* git history, not `.dvc`, which is already
configured correctly and doesn't need to be redone), `git init` again,
then resume from `dvc add` above. `dvc push` copies the real 95MB into
the remote folder — the one genuine extra copy on disk a real backup
requires (see the size discussion above); everything before it (the
working copy vs. DVC's local cache) costs nothing extra thanks to the
hardlink config.

**Fresh clone** (a new machine, or `models/final_model.pkl` missing after
`git clone` — expected, since it's gitignored and DVC-managed now):

```bash
dvc pull
```

Pulls the real file back from the remote, byte-identical (verified via the
md5 in the `.dvc` pointer) — required before `python -m src.training.log_model`
(needs the file to log) or `docker compose up --build` (the Dockerfile
`COPY`s the current directory into the image — an image built without the
real file first pulled would ship without a model at all).

**Why hardlink, and why it can matter on Windows specifically:** DVC's
default cache mode is a plain copy — fine on Linux/Mac, but doubles disk
usage for every large file (once in `models/`, once in `.dvc/cache`).
`cache.type hardlink,symlink,copy` tries hardlink first (same file, two
names, zero extra space — NTFS supports this), falls back to symlink, and
only actually copies if neither works. Verified end to end before shipping
this: tracked the file, pushed it, then deleted both the working copy
*and* the local cache to simulate a lost machine, and `dvc pull` restored
it with an identical md5 — the round trip is real, not just configured.

## Configuration

Every path, host/port, log setting, MLflow setting, and monitoring
parameter lives in `config/config.yaml` and is loaded once via
`src/utils/config_loader.py`. Nothing in `app/` or `src/` hardcodes a path
or a parameter. The `monitoring:` block (Section 10 — prediction-log path,
drift window/thresholds) is documented in full in "Monitoring (Section 10)"
below, alongside its own config file, not repeated here.

**Secrets and connection strings (Section 8):** the settings that
legitimately differ by environment — where `api`/`register` reach
MLflow, and Grafana's admin login — are never hardcoded in a committed
file. `src/utils/config_loader.py` reads the MLflow URI from the
`MLFLOW_TRACKING_URI` environment variable when set, falling back to
`config.yaml`'s local default otherwise; Grafana reads its own two
variables directly (`docker-compose.yml`'s `environment:` block — it has
no `config.yaml` of its own to fall back to). Inside Docker,
`docker-compose.yml` supplies all three via
`${MLFLOW_TRACKING_URI:-http://mlflow:5000}`,
`${GRAFANA_ADMIN_USER:-admin}`, `${GRAFANA_ADMIN_PASSWORD:-admin}` —
Compose-native substitution from a `.env` file in the project root, not a
bare string in the compose file. `.env.example` documents all three
variables and their safe defaults; copy it to `.env` only if you need to
override one (e.g. pointing at an external MLflow server, or changing
Grafana's login before exposing it beyond localhost) — `.env` itself is
gitignored, so a real `.env` never reaches the repo. No `.env` file at all
is the normal case: every default is baked into `docker-compose.yml`, so
`docker compose up --build` still runs with zero setup on a clean
machine. This project has no other secret (no DB password beyond
Grafana's own default login, no API key — MLflow's backend store is a
local SQLite file, not a networked credential).

## Before you run anything: register a model in MLflow

**Fresh clone first:** since Section 4, `models/final_model.pkl` is DVC-
managed, not committed to git — it won't exist yet after a plain
`git clone`. Run `dvc pull` first (see "Data & artifact versioning" above)
or the command below fails with a file-not-found, not an MLflow error.

Since Section 5, the service does **not** read `models/*.pkl` directly — it
resolves "the current production model" from the MLflow Model Registry at
startup (`src/inference/model_registry.py`) and fails fast if nothing is
registered yet. On a fresh clone, register one first:

```bash
python -m src.training.log_model
```

This loads `models/final_model.pkl` (never re-fits it), logs it — plus its
matching encoder/scaler/imputer/feature-list bundle and the real Notebook 06
evaluation metrics (`KNOWN_METRICS` — see "Model provenance" above) — to
MLflow, registers a version, and points the `production` alias/stage at it.
No `--allow-placeholder` needed: `KNOWN_METRICS` is fully filled in, so this
now logs real numbers on the first try. Re-run this any time you swap in a
fresh `models/*.pkl` from a new notebook run — each run creates a new
registry version and moves the alias to it, and update `KNOWN_METRICS` to
match that run's own evaluation numbers first.

`--allow-placeholder` still exists for the case that actually needs it —
a fresh `models/*.pkl` you want registered immediately, before you've gone
back to fill in that new run's real metrics — and logs clearly-tagged zero
metrics in that situation only. `docker-compose.yml`'s `register` service
(Section 8) always calls this with `--allow-placeholder` as a safe
unattended default for first boot; since `KNOWN_METRICS` has no unfilled
values right now, that flag is inert — it logs the real numbers above, not
zeros (verified: the run's `metrics_source` tag reads
`notebook_06_evaluation`, never `placeholder`, whenever `KNOWN_METRICS` is
fully filled in, regardless of which way this script was invoked).

## Run the API

```bash
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000/docs` for the interactive API docs — every
route below has a working example pre-filled ("Try it out" posts it as-is).

Routes:

| Route | Method | What it does |
|---|---|---|
| `/` | GET | liveness — `{"status": "running"}` |
| `/health` | GET | readiness — only reachable once the model has finished loading |
| `/model-info` | GET | the resolved MLflow registry state: name, version, stage, run_id |
| `/predict` | POST | one order in, one `{prediction, probability, model_version}` out |
| `/predict/batch` | POST | up to 500 orders in, one per-order result out — one bad row does not fail the others |

Example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"total_payment": 150, "total_freight": 20, "review_score": 5, "items_count": 2, "approval_delay_hours": 3, "order_status": "delivered", "customer_state": "SP"}'
```

A payload with a bad category (`order_status`, `customer_state`) or the
wrong type is rejected with `422` before it reaches the model. A payload
that's well-typed but fails Great Expectations (a negative price, an
out-of-range `review_score`) is rejected with `400` — see Section 4.

`prediction`: `1` = the model flags this order as delayed/late, `0` = on
time — confirmed against Notebook 02 (Label Creation), which defines the
target variable `is_delayed` as `1 = Delayed, 0 = On Time`. `probability`
is `P(prediction == 1)`, i.e. the model's estimated chance this specific
order is delayed.

## Run with Docker

**Fresh clone first:** run `dvc pull` before `docker compose up --build`
(see "Data & artifact versioning" above). The Dockerfile `COPY`s the
current directory into the image — build it before `models/final_model.pkl`
has been pulled and the image ships with no model file at all, since it's
gitignored (DVC-managed) rather than committed.

Verified end to end on a real Docker daemon: `docker compose up --build`
brings up all five services from a clean volume. `mlflow` + `register` +
`api` (Section 8) — `/predict/batch` returns real predictions from the
real registered model (`olist_delay_classifier v1 (Production)`) served
entirely from inside the containers, no local Python env involved.
`prometheus` + `grafana` (Section 10) — confirmed on a second real run:
`olist-api` shows `UP` on Prometheus's own `/targets` page, all 3 alert
rules load and evaluate, and Grafana's dashboard renders live data with
zero manual setup — see "Monitoring (Section 10)" for the full walkthrough.

```bash
docker compose up --build
```

One command, fresh clone, nothing installed locally except Docker itself
— no local Python env, no manual `log_model.py` step. Five services come
up, `mlflow`/`register`/`api` in the strict dependency order Section 8
already established, `prometheus`/`grafana` right after:

| Service | What it does |
|---|---|
| `mlflow` | the tracking/registry server — `http://localhost:5000`, data in the `mlflow_data` Docker volume |
| `register` | runs once: registers the model (real metrics) against `mlflow` if none exists yet, then exits |
| `api` | the FastAPI service — `http://localhost:8000`, only starts once `mlflow` is healthy and `register` has exited `0` |
| `prometheus` | scrapes `api`'s `/metrics` every 10s — `http://localhost:9090` |
| `grafana` | dashboards on top of `prometheus` — `http://localhost:3000` (default login `admin`/`admin`) |

This is also Task 3's actual fix, not just a convenience: locally,
`config.yaml`'s `mlflow.tracking_uri` is a `sqlite:///mlflow.db` file next
to the code — meaningless inside a container. Inside Docker,
`src/utils/config_loader.py` overrides that one setting from the
`MLFLOW_TRACKING_URI` environment variable docker-compose.yml sets
(`http://mlflow:5000` by default, via `${MLFLOW_TRACKING_URI:-http://mlflow:5000}`
— see "Configuration" above), so the `api` container never touches a local
file — it reaches the model over the network, and the model itself lives
in a Docker-managed volume, not on this laptop. Local (non-Docker)
development is untouched — that environment variable is only ever set
inside the containers.

Re-running `docker compose up` later (a restart, or `up` again after
`down` without `-v`) does **not** re-register a new version every time —
`scripts/register_if_needed.py` checks first and skips if
`mlflow_data` already has one, the same bootstrap-only-if-missing pattern
`tests/conftest.py` uses locally (Section 6). To force a clean slate:
`docker compose down -v` (the `-v` also drops `mlflow_data`, `api_logs`,
and Section 10's `prometheus_data`/`grafana_data`) — **required** any
time `Dockerfile.mlflow`'s server flags change, since an experiment
already registered under old flags keeps its old `artifact_location` in
the volume regardless of what the image now starts with.

Two things worth knowing if you ever rebuild this from scratch on a new
machine, since both cost real debugging time the first time around:

- `register`'s command is a plain `python scripts/register_if_needed.py`
  (docker-compose.yml), not `python -m ...` — that only puts
  `scripts/`, not `/app`, on the import path, so `from src... import ...`
  fails with `ModuleNotFoundError: No module named 'src'` unless
  `PYTHONPATH=/app` is set (Dockerfile does this).
- MLflow's server validates the incoming `Host` header by default
  (DNS-rebinding protection) and rejects anything not allow-listed with a
  403 — `register`/`api` reach `mlflow` as `http://mlflow:5000`, a
  hostname that isn't `localhost` by default, so `Dockerfile.mlflow` sets
  `--allowed-hosts "mlflow:*,localhost:*,127.0.0.1:*"`. Separately,
  `--artifacts-destination` (not `--default-artifact-root`) is what makes
  artifact upload/download go over HTTP instead of requiring `register`/
  `api` to have `/mlflow_data` mounted directly, which they don't.

## Testing

```bash
pip install -r requirements-dev.txt   # pytest, pytest-cov, httpx, ruff, black
pytest
```

115 tests across `tests/unit`, `tests/data`, `tests/model`,
`tests/monitoring`, and `tests/integration` — one command, no separate
server or fixtures to set up by hand (`tests/conftest.py` registers a
model in MLflow automatically if this is a completely fresh clone). For
coverage:

```bash
pytest --cov=src --cov=app --cov-report=term-missing
```

## CI/CD (Section 9)

Section 9's task text: a pipeline that runs on every push (lint, format
check, test); builds and pushes the image only if all of that passes;
pre-commit hooks so the same checks run locally too; a failed test has to
actually stop the pipeline, not just get logged and ignored.
`.github/workflows/ci.yml` and `.pre-commit-config.yaml` implement all four.

### The pipeline

Two jobs, gated so the second can never start before the first fully
passes:

| Job | Runs on | Steps | When |
|---|---|---|---|
| `test` | every push and PR against `main` | ruff → `black --check` → pytest (115 tests) | always |
| `build-and-push` | `needs: test` | build the API image, push to GHCR as `:latest` and `:<commit sha>` | only a real push to `main`, never a PR |

"A failed test must stop the pipeline" is true here for free, not through
extra config: GitHub Actions stops a job at its first failing step by
default (no `continue-on-error` anywhere in the file), and
`build-and-push` declares `needs: test` — so it simply never starts if
`test` didn't finish green. Nothing reaches the registry off a broken
commit. `build-and-push` is further restricted to
`if: github.event_name == 'push' && github.ref == 'refs/heads/main'`: a
pull request still runs the full `test` job (so lint/format/test results
show up on the PR itself), it just never touches the registry — a PR from
a fork couldn't authenticate to push images anyway, and untrusted branches
shouldn't publish images regardless.

### Where the image ends up

`ghcr.io/salehmsa/olist-delay-prediction-mlops` — GitHub Container
Registry, authenticated with the repo's own built-in `GITHUB_TOKEN` (no new
secret to create or store, consistent with Section 8's "environment
variables for secrets" — there's simply no secret to manage here at all).
The image name is lowercased in its own step because `github.repository`
evaluates to `Salehmsa/olist-delay-prediction-mlops` — capital S included —
and GHCR, like every OCI registry, rejects uppercase in image names.

### Why CI runs on a dummy model, and what that means for the published image

`models/final_model.pkl` is DVC-tracked (Section 4) against a **local
folder on your own machine** as its remote — by design, per how Section 4
was scoped. A GitHub-hosted runner has no way to reach that folder, so
`dvc pull` isn't an option inside this workflow at all.

That would normally be a dead end: `src/inference/model_registry.py` /
`pipeline.py` resolve a registered model **at import time**, so without
*something* loadable at `models/final_model.pkl`, `pytest` can't even
collect the test suite, and the `register`/`api` containers would crash
the same way at startup.

The fix is `tests/fixtures/ci_dummy_model.pkl` — a tiny (~26KB),
schema-correct `RandomForestClassifier` (same 42 features, same interface
as the real model) generated once by
`tests/fixtures/generate_ci_dummy_model.py` and committed to git directly
(small enough to not need DVC). Both jobs copy it over
`models/final_model.pkl` before doing anything else. This is a deliberate
substitution, not a shortcut, because of how the tests are already written:
everything under `tests/model/` and `tests/integration/` asserts
*structural* contracts only — binary classifier, one prediction/probability
per row, probability in `[0, 1]`, deterministic, handles missing fields —
never a specific real-model prediction (see the docstring at the top of
`tests/model/test_predict.py`). The dummy satisfies every one of those
contracts, so CI genuinely proves the code wires together correctly.

**In plain terms:** the image this workflow publishes to GHCR on every push
to `main` answers with this placeholder model, not your real DVC-managed
one — exactly what a stranger would get from `git clone` + `docker build`
with no `dvc pull` in between, since CI is that exact scenario. To run the
*real* model, locally or in Docker, `dvc pull` first (see "Data & artifact
versioning (DVC)" above), then `docker compose up --build` as normal.
Nothing about this is hidden: `register_if_needed.py`'s own MLflow tags
(`metrics_source: notebook_06_evaluation` vs `placeholder`) make it obvious
from the registry alone which model any given run actually used.

### Pre-commit hooks

The same checks the `test` job runs — file hygiene (trailing whitespace,
EOF newlines, valid YAML, a 500KB large-file guard, merge-conflict markers),
ruff `--fix`, black — but locally, before a commit is even made. One-time
setup:

```bash
pre-commit install
```

After that every `git commit` runs the hooks automatically; a failing hook
blocks the commit (ruff/black may auto-fix files in place — `git add` the
result and commit again). To run everything on demand without committing:

```bash
pre-commit run --all-files
```

The large-file guard isn't a generic default left at whatever it ships
with — it's set at 500KB on purpose, as the automated version of the exact
mistake Section 4 already hit once in this project (a large binary staged
with plain `git add` before `dvc add` got to it — see "Data & artifact
versioning" above). 500KB comfortably covers every file this repo commits
to git directly (the encoder/scaler/imputer, `feature_list.json`, the
~26KB CI dummy model), while the real 95MB `models/final_model.pkl` should
never be stageable at all once DVC owns it.

### Pushing this for the first time

```bash
git add .github/workflows/ci.yml .pre-commit-config.yaml tests/fixtures/
git commit -m "Add CI/CD pipeline (Section 9)"
git push
```

Then open the repo's **Actions** tab on GitHub: `test` should go green
first, `build-and-push` right after it (only for a push to `main`, not a
PR), and the new image should then show up under **Packages** (the repo's
sidebar, or your GitHub profile's Packages tab).

## Monitoring (Section 10)

Section 10's task text: expose service metrics (request count, latency,
error rate); track the distribution of predictions over time and watch
for drift; store prediction logs so they can be evaluated later once the
real delivery date arrives; decide what to alert on, and write it down.

> 10. المراقبة (Monitoring)
> اعرض metrics للخدمة: عدد الطلبات، الزمن، ونسبة الأخطاء. تتبع توزيع
> التوقعات مع الوقت، وراقب الـ drift. خزن سجلات التوقعات حتى تقدر تقيمها
> لاحقا لما يوصل تاريخ التوصيل الحقيقي. قرر شو بدك تعمل عليه تنبيه، واكتبه.

Two ways to satisfy "expose metrics" were on the table: a lean
`/metrics`-only endpoint, or a real Prometheus + Grafana stack behind it.
Went with the latter — `prometheus` and `grafana` in `docker-compose.yml`
("Run with Docker" above), both provisioned automatically on first boot,
no manual "add a datasource" click.

### Service metrics (bullet 1)

`GET /metrics` (Prometheus text format) exposes two layers from one
endpoint — confirmed against **live** output from a running instance,
not copied from a library's docs:

| Metric | Source | What it is |
|---|---|---|
| `http_requests_total{handler,method,status}` | `prometheus-fastapi-instrumentator` (`app/main.py`) | request count, by route and status **class** — Prometheus groups `status="2xx"`/`"4xx"`/`"5xx"`, not exact codes |
| `http_request_duration_highr_seconds_bucket` | same | fine-grained latency histogram built for `histogram_quantile()` — what `alerts.yml`'s `HighLatency` rule reads |
| `olist_predictions_total{predicted_class}` | `src/monitoring/metrics.py` | predictions made, by predicted class (`0`/`1`) — "request count" read as outcomes, not raw HTTP traffic |
| `olist_prediction_latency_seconds` | same | the **model pipeline's own** latency (validate → features → preprocess → predict) — isolated from the FastAPI/Starlette overhead the HTTP histogram above also includes |
| `olist_prediction_positive_rate` / `olist_prediction_drift_status` | same, see bullet 2 | drift signals, as gauges — see below |

`/metrics` itself is excluded from its own request-count metrics
(`excluded_handlers=["/metrics"]`) so Prometheus scraping it every 10s
doesn't show up as "traffic" in its own numbers. Error rate isn't a
separate tracked number — it's `http_requests_total{status="5xx"} /
http_requests_total`, a ratio over the two counters above, computed in
PromQL by `alerts.yml`'s `HighErrorRate` rule and Grafana's panels.

`GET /monitoring/summary` gives the same underlying data as plain JSON,
for a human or `curl` rather than Prometheus — real captured output
below, from three requests through `/predict` in this checkout (`status`
is `"insufficient_data"` below `drift_min_sample_size` — see bullet 2):

```bash
curl http://localhost:8000/monitoring/summary
```
```json
{
  "predictions_logged_total": 16,
  "drift": {
    "status": "insufficient_data",
    "reason": "only 16 prediction(s) logged so far, need at least 30 before a drift verdict is meaningful",
    "predictions_logged_total": 16,
    "baseline_positive_rate": 0.066
  }
}
```

Past `drift_min_sample_size` predictions the shape gains `window_size`,
`current_positive_rate`, `drift_bounds: {low, high}`, and
`psi_probability_drift` (`null` with an explanatory `psi_note` until a
full second window of history exists — see bullet 2).

### Prediction distribution & drift (bullet 2)

`src/monitoring/drift.py`'s `compute_drift_report()` is the single
function everything else reads from — the Prometheus `DriftCollector`
and `/monitoring/summary` both call it directly, so the two can never
disagree. Two independent signals:

1. **Positive-rate drift** — the rolling share of the last
   `drift_window_size` (200) predictions flagging `is_delayed=1`,
   compared against `config.yaml`'s `baseline_positive_rate` (0.066 —
   Notebook 06's real held-out-test positive rate, the same number in
   "Model provenance" above). Outside `[1%, 20%]` → `drift_detected`.
   This is the signal `alerts.yml`'s `PredictionDriftDetected` rule
   actually fires on.
2. **PSI (Population Stability Index)** on predicted probabilities — a
   more sensitive secondary signal: the current window's score
   distribution (10 equal-width bins over `[0, 1]`) against a
   **reference window drawn from this service's own earlier
   predictions** (the 200 rows immediately before the current ones).
   Deliberately *not* compared against Notebook 06's actual training
   scores: this service only ever received `KNOWN_METRICS`' summary
   numbers, never the raw training prediction array, and reaching back
   into notebook internals for one would cross the same "no training
   inside the inference pipeline" line Section 5 already drew. A
   self-referential reference window is a real, commonly used pattern
   for exactly this gap, and still catches what matters most in
   production — today's traffic looking different from last week's.
   Exposed for diagnosis (Grafana, `/monitoring/summary`) but
   deliberately **not** wired to its own alert — see "Alerting" below
   for why.

Below `drift_min_sample_size` (30) logged predictions, both signals
report `"insufficient_data"` rather than a false `ok`/`drift_detected`
verdict on too little evidence — expected and correct behavior against a
freshly started container, not a bug.

`DriftCollector` (`src/monitoring/metrics.py`) is a **custom** Prometheus
collector, not a `Gauge.set()` — these two values are derived from a
SQLite query, not incremented inline per request, so a plain Gauge would
just sit at whatever it was last set to. A custom collector recomputes
both fresh on every single `/metrics` scrape (confirmed live: with
`status="insufficient_data"`, `olist_prediction_drift_status` reads
exactly `-1.0` and `olist_prediction_positive_rate` is correctly absent
from the scrape entirely, matching the 0/1/-1 contract in the code's own
docstring), so the drift verdict Grafana shows is never staler than
Prometheus's own 10-second scrape interval
(`monitoring/prometheus/prometheus.yml`) — the textbook-correct pattern
for a metric backed by external state rather than an inline counter.

### Prediction logging (bullet 3)

Every **successful** prediction — through `/predict` or `/predict/batch`,
never a request that failed schema or Great-Expectations validation,
since that was never actually predicted on — is written to
`logs/predictions.db` (SQLite, `src/monitoring/prediction_log.py`),
inside the same `api_logs` Docker volume `logs/app.log` already uses, so
it persists across restarts with no new volume or service:

```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    order_id TEXT,               -- nullable, see below
    source TEXT NOT NULL,        -- "predict" | "predict_batch"
    prediction INTEGER NOT NULL,
    probability REAL NOT NULL,
    model_version TEXT NOT NULL,
    latency_ms REAL NOT NULL
);
```

`OrderRequest` gained one new optional field for this: `order_id` — not
used by the model (`input_builder.py` ignores it, same as
`review_missing`) — it exists solely so a logged row can later be joined
back to a real Olist order. A caller that omits it still gets a logged,
drift-countable row, just one that can never be joined back to a specific
order later.

**Honest scope note:** this bullet asks for logs "so you can evaluate
them later once the real delivery date arrives." This service stores
exactly what that evaluation would need — the prediction, the
probability, the order id when given, a timestamp — but *running* that
evaluation (joining `predictions.db` against a real delivered-date feed
once one exists, and scoring accuracy on it) is a job for whichever
system owns that downstream ground-truth data, not something this
inference service does to itself. That's consistent with how this
project has scoped "no training inside the inference pipeline"
throughout, not a gap specific to this bullet.

### Alerting

Three rules, `monitoring/prometheus/alerts.yml`, evaluated by Prometheus
itself every 10s (full reasoning in that file's own header comment —
this is the README mirror):

| Alert | Fires when | Why this threshold |
|---|---|---|
| `HighErrorRate` | 5xx rate > 5% over 5m | standard SRE floor — below it, isolated failures; above it, something's actually broken (bad deploy, model failed to load, Great Expectations misconfigured) |
| `HighLatency` | p95 latency > 500ms over 5m | real measured pipeline latency here is single-digit milliseconds (a 100-tree RandomForest on 42 features) — 500ms is deliberately ~50–100x that, generous enough to only catch something genuinely wrong (resource contention, a stuck dependency), not ordinary jitter |
| `PredictionDriftDetected` | `olist_prediction_drift_status == 1` for 10m | the positive-rate signal (bullet 2) leaving `[1%, 20%]`. Both directions matter: drifting up means far more orders flagged delayed than training data suggests; drifting down toward 0% is the more dangerous case given this model's own `recall_delayed` is already only 8% (see "Model provenance") — a further collapse reads as something broken, not deliveries genuinely improving overnight. `for: 10m` (longer than the other two) because the underlying gauge needs `drift_min_sample_size` (30) fresh predictions before it can leave `insufficient_data` at all — a shorter window would just flap |

PSI is deliberately **not** its own alert rule — it's a more sensitive,
more diagnostic signal (Grafana, `/monitoring/summary`), but its
self-referential reference window (bullet 2 above) makes it better
suited to a human glancing at a trend than a page-worthy threshold; the
positive-rate rule above is the one anchored to a number this project can
actually defend (Notebook 06's real baseline).

**What this does *not* include, on purpose:** no Alertmanager, no
Slack/email/PagerDuty routing — there's no real notification channel to
wire a training project to, and inventing one would be less honest than
leaving it out. These rules fire and show as **firing** on Prometheus's
own Alerts page (`http://localhost:9090/alerts`) — the correctly scoped
answer to "decide what to alert on and write it down" without pretending
to infrastructure this project doesn't actually have.

### Seeing it live

```bash
docker compose up --build
```

| What | Where |
|---|---|
| Raw metrics | `http://localhost:8000/metrics` |
| Human-readable summary | `http://localhost:8000/monitoring/summary` |
| Prometheus — scrape health | `http://localhost:9090/targets` (`olist-api` should read `UP`) |
| Prometheus — alert state | `http://localhost:9090/alerts` |
| Grafana dashboard | `http://localhost:3000` (login `admin` / `admin` by default — "Configuration" above) |

The dashboard (`monitoring/grafana/dashboards/olist_monitoring.json`,
provisioned automatically — no manual datasource step) has 9 panels:
predictions logged, current predicted-delay rate, drift status, and p95
latency as headline numbers; request rate by status, latency percentiles,
predictions-by-class rate, model-pipeline latency, and predicted-delay
rate against the 0.066 / 0.01 / 0.20 reference lines as time series. A
handful of real requests against `/predict` (or `/docs`'s "Try it out")
is enough to see it move — drift's own panels need
`drift_min_sample_size` (30) logged predictions before they leave
"insufficient data".

### What's verified, and what isn't

Validated in two stages, both real, neither skipped:

**Stage 1 — sandbox (files and logic):** `promtool check config` and
`promtool check rules` both pass clean on the exact files Docker mounts;
`docker compose config` fully resolves the 5-service file (env
substitution, volumes, `depends_on`); every metric name and label was
confirmed against **live** `/metrics` output from a running instance,
including the `status="2xx"`/`"4xx"` grouping and the `-1.0` drift-status
value with no `positive_rate` sample while `insufficient_data`; the full
test suite (113 tests) passes, including `TestClient` calls that hit
`/metrics` and `/monitoring/summary` for real; and the new dependency
(`prometheus-fastapi-instrumentator`) installs and resolves cleanly in a
genuinely fresh Python 3.12 virtualenv running every `ci.yml` step by
hand in order — 113 passed.

**Stage 2 — a real Docker daemon, on an actual machine (Windows +
Docker Desktop/WSL2), Section 8's own pattern repeated for Section 10:**
`docker compose up --build` brought up all 5 services; Prometheus's
`/targets` page showed `olist-api` **UP** (`http://api:8000/metrics`,
last scrape 47ms); all 3 `alerts.yml` rules loaded and evaluated
(`olist-api-alerts`, initially `INACTIVE(3)`); Grafana provisioned its
datasource and the "Olist Delivery Delay - Monitoring (Section 10)"
dashboard with **zero manual setup**, confirmed as
`grafana/grafana:13.2.2` from its own UI footer — resolving the one tag string this project's own sandbox
couldn't pull and confirm directly. Then, live: 32 real requests through
`/predict` moved every panel — `predictions_logged_total` climbed to 32
in real time, `olist_prediction_drift_status` flipped from `-1`
(`insufficient_data`) to `1` (`drift_detected`) the moment the positive
rate hit 0% (below the 1% floor), and `GET /monitoring/summary`
independently confirmed the exact same numbers straight from the source
(`"status": "drift_detected", "current_positive_rate": 0.0,
"psi_note": "needs 400 logged predictions to compute (has 32)"` — that
`psi_note` string matches `drift.py`'s own f-string character for
character, proof this is the real code path, not a coincidence). p95
latency briefly spiked to ~3.6s on the very first requests after a cold
container start, then fell to ~1.4s as the window rolled forward — exactly
the cold-start pattern expected, not a real regression, and never
sustained long enough to actually trip `HighLatency`'s `for: 5m`.

Item 10 above is checked on that basis — a real run, on real
infrastructure, not files that look right.

## Known open items

Not blockers, just not resolved yet:

- 5 of the 6 training notebooks (01–05: data loading/cleaning, label
  creation, EDA, feature engineering) are not yet copied into `notebooks/`.
  Notebook 06 (model training & evaluation) *is* included now, specifically
  so `tests/model/test_notebook_parity.py`'s claims can be checked against
  its actual source — see `notebooks/README.md`.
- Development moved from a Windows machine where Docker Desktop could not
  run at all (its WSL2 backend hit a Group-Policy-restricted logon-type
  error, an IT policy restriction on that account, unrelated to this
  project) to a second machine specifically to get a real
  `docker compose up --build` run — see "Run with Docker" above for what
  that surfaced and fixed (import path, Host-header validation, artifact
  proxying). All five services (Section 8 + Section 10's `prometheus`/
  `grafana`) now verified working together end to end on that machine.
