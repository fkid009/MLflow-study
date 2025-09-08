# 03 – Models (pyfunc)

## Why MLflow?

MLflow standardizes **experiment tracking, reproducibility, and deployment**. It shines when:

* Multiple people iterate on the same model line (dev → QA → prod).
* You must **reproduce past results** or **roll back** quickly.
* You need **lineage** for audits (who trained what, when, with which data/code/deps).

**Therefore, this folder** teaches how to make a model **re-runnable on demand** by focusing on:

* **`signature`** — the input/output schema contract
* **`mlflow.pyfunc.load_model()`** — a framework-agnostic way to load and use the saved model

---

## What you’ll learn here

1. Save a model with **signature + input\_example** via `mlflow.sklearn.log_model()`.
2. Load the exact model instance via **`mlflow.pyfunc.load_model(MODEL_URI)`**.


---

## Core concepts

### Signature = **IO schema contract**

Describes **which inputs** (column names / dtypes / shapes) the model expects and **which outputs** it returns.
At inference/serving time, MLflow **validates** the request against the signature and **fails fast** on mismatches.

### `mlflow.pyfunc.load_model()` = **Unified loader**

Load any MLflow-saved model with a consistent `predict()` API.

**URI types**

* `runs:/<run_id>/model` — freeze to a specific training run (great for experiments/debugging).
* `models:/<name>/Production` — pointer to the **current prod** version (great for hot-swaps).
* `models:/<name>/<version>` — pin a specific version (great for backfills/repro).

*(Remember: to resolve `runs:/` and `models:/`, your `MLFLOW_TRACKING_URI` must point to the right MLflow server. For S3/MinIO artifacts, provide credentials/environment as needed.)*

---

## Caveats & gotchas

### Signature

* **Column names are part of the contract.** Typos, reordering, or dtype changes cause validation errors.
* **Include preprocessing inside the model** (e.g., `sklearn.Pipeline` or a pyfunc custom model).
  External preprocessing is not enforced by signature.
* Be explicit with tricky types (datetimes/timezones).
* While optional, **saving signature is practically mandatory** for safe serving & collaboration.

### `load_model()`

* Ensure **`MLFLOW_TRACKING_URI`** is set so `runs:/`/`models:/` can be resolved.
* For S3/MinIO artifact access, set the appropriate **credentials and endpoint**.
* To load `models:/...`, the model must be **registered** and the target Stage/version must **exist**.
* For services, load once at startup and **cache**; reload on a deployment hook if needed.


---

## Run snippets

```bash
# Enter the container
docker compose exec trainer bash

# 1) Train & save (with signature)
python tutorial/03_models_pyfunc/run.py

# 2) Load & smoke test
#   a) auto-pick latest run
python examples/03_models_pyfunc/reproduce_run.py
#   b) or explicitly point to prod
export MODEL_URI="models:/iris_clf/Production"
python examples/03_models_pyfunc/reproduce_run.py
```

---

### One-liner takeaway

Use **Signature** to lock in the IO contract, and **`load_model()`** to reliably revive “that exact model,” making your experiments **reproducible and safe to serve**.
