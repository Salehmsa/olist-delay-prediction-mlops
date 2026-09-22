# Olist Delivery Delay Prediction — Inference Service

MLOps Training 2026/2027 · Task 3 · Qafza Training

Turns the model trained in `notebooks/01`–`06` into a real inference
service: a Python package, a FastAPI app, data versioning, experiment
tracking, tests, containers, and a CI/CD pipeline — monitoring is the one
section still to land.

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
- [x] 6. Testing (pytest) — 94 tests, unit / data / model / integration
- [x] 7. API (`/predict`, `/predict/batch`)
- [x] 8. Docker & Docker Compose — verified with a real Docker daemon,
      real model, `/predict/batch` tested live through the containers;
      connection string environment-driven via `.env`/`.env.example`,
      `.gitignore` added so a real `.env` can never reach the repo
- [ ] 9. CI/CD — pipeline + pre-commit hooks written and locally validated
      (`actionlint` on the workflow, the exact CI steps re-run end to end
      against a simulated fresh checkout, `pre-commit run --all-files`
      against the real hook repos) — left unchecked on purpose until a real
      push shows green on GitHub's own Actions tab, not just simulated here;
      see "CI/CD (Section 9)" below
- [ ] 10. Monitoring

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
├── notebooks/             the 6 training notebooks, for lineage — never
│                          executed by the service or the image
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
│   ├── schemas/            pydantic request/response contracts
│   ├── training/           log_model.py — registers models/*.pkl into
│   │                       MLflow; never called by the running service
│   ├── utils/               config loader (env-overridable MLflow
│   │                        tracking URI, Section 8), logger
│   └── validation/          Great Expectations suite + validation wrapper
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
│   └── integration/           API routes + the pipeline, end to end
├── pytest.ini
├── requirements.txt         runtime dependencies, pinned
├── requirements-dev.txt     + test / lint / format tooling
├── Dockerfile               the API image — multi-stage, non-root, healthcheck
├── Dockerfile.mlflow        the mlflow tracking/registry server image
├── docker-compose.yml       mlflow + register + api — one command, fresh clone
├── .github/
│   └── workflows/
│       └── ci.yml            Section 9: lint → format check → test, then
│                               build+push to GHCR on main only — see
│                               "CI/CD (Section 9)" below
├── .pre-commit-config.yaml   the same lint/format checks, run locally
│                               before a commit — `pre-commit install`
├── .env.example             documents MLFLOW_TRACKING_URI + its safe default
│                             (Section 8: env vars for secrets/connection
│                             strings) — copy to .env only to override it
├── .gitignore                keeps .env, mlflow.db, mlruns/, caches, venvs
│                             out of git — mirrors .dockerignore's reasoning
│                             where it applies, different list where it doesn't
└── .dockerignore
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

Every path, host/port, log setting, and MLflow setting lives in
`config/config.yaml` and is loaded once via `src/utils/config_loader.py`.
Nothing in `app/` or `src/` hardcodes a path or a parameter.

**Secrets and connection strings (Section 8):** the one setting that
legitimately differs by environment — where `api`/`register` reach
MLflow — is never hardcoded in a committed file. `src/utils/config_loader.py`
reads it from the `MLFLOW_TRACKING_URI` environment variable when set,
falling back to `config.yaml`'s local default otherwise. Inside Docker,
`docker-compose.yml` supplies that variable via `${MLFLOW_TRACKING_URI:-http://mlflow:5000}`
— Compose-native substitution from a `.env` file in the project root, not
a bare string in the compose file. `.env.example` documents the variable
and its safe default; copy it to `.env` only if you need to override that
default (e.g. pointing at an external MLflow server) — `.env` itself is
gitignored, so a real `.env` never reaches the repo. No `.env` file at all
is the normal case: every default is baked into `docker-compose.yml`, so
`docker compose up --build` still runs with zero setup on a clean
machine. This project has no other secret (no DB password, no API key —
MLflow's backend store is a local SQLite file, not a networked
credential), so `MLFLOW_TRACKING_URI` is the only entry in `.env.example`.

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
brings up all three services from a clean volume, and `/predict/batch`
returns real predictions from the real registered model
(`olist_delay_classifier v1 (Production)`) served entirely from inside the
containers — no local Python env involved.

```bash
docker compose up --build
```

One command, fresh clone, nothing installed locally except Docker itself
— no local Python env, no manual `log_model.py` step. Three services come
up in order:

| Service | What it does |
|---|---|
| `mlflow` | the tracking/registry server — `http://localhost:5000`, data in the `mlflow_data` Docker volume |
| `register` | runs once: registers the model (real metrics) against `mlflow` if none exists yet, then exits |
| `api` | the FastAPI service — `http://localhost:8000`, only starts once `mlflow` is healthy and `register` has exited `0` |

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
`docker compose down -v` (the `-v` also drops `mlflow_data` and
`api_logs`) — **required** any time `Dockerfile.mlflow`'s server flags
change, since an experiment already registered under old flags keeps its
old `artifact_location` in the volume regardless of what the image now
starts with.

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

94 tests across `tests/unit`, `tests/data`, `tests/model`, and
`tests/integration` — one command, no separate server or fixtures to set up
by hand (`tests/conftest.py` registers a model in MLflow automatically if
this is a completely fresh clone). For coverage:

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
| `test` | every push and PR against `main` | ruff → `black --check` → pytest (94 tests) | always |
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

## Known open items

Not blockers, just not resolved yet:

- The six training notebooks are not yet copied into `notebooks/` (kept out
  so far — see `notebooks/TODO_COPY_NOTEBOOKS.txt`).
- Development moved from a Windows machine where Docker Desktop could not
  run at all (its WSL2 backend hit a Group-Policy-restricted logon-type
  error, an IT policy restriction on that account, unrelated to this
  project) to a second machine specifically to get a real
  `docker compose up --build` run — see "Run with Docker" above for what
  that surfaced and fixed (import path, Host-header validation, artifact
  proxying). All three now verified working together end to end.
