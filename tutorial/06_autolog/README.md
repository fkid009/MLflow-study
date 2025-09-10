# 06 – Autolog (Use Autolog Only)

**Goal**

* Use `mlflow.autolog()` / `mlflow.<framework>.autolog()` to automatically log training.
* Understand **what autolog does** and **what it stores**.
* Learn the **model artifact folder structure** that gets saved with a run.

**Files**

* `run.py` — scikit-learn pipeline with `mlflow.sklearn.autolog()` (no manual logging).
* `eval_run.py` — XGBoost with `mlflow.xgboost.autolog()` (no manual logging; logs training curves when `eval_set` is provided).

---

## 0) What Autolog Does

Call autolog **once before training**. When your code hits `fit()`, MLflow hooks into the framework and **automatically logs** to the active run:

* **Params**: estimator/pipeline hyperparameters
* **Model artifact**: trained model + environment files (see “Model Artifact Folder”)
* **Framework metrics**: if the framework exposes metrics during training, autolog captures them

  * e.g., XGBoost with `eval_set` → **per-iteration eval metrics** (training curves)

> Autolog captures what the library itself exposes. **Custom business metrics** (e.g., F1 macro, CTR, p95 latency) are not automatically known; you’d log those manually with `mlflow.log_metric()`.
> In this lesson we **use autolog only** (no manual logging).

### Framework Notes (for these examples)

* **scikit-learn (`mlflow.sklearn.autolog`)**

  * Logs model **parameters** and the trained **model**.
  * Unless you implement your own evaluation loop, **validation/test metrics may not be auto-logged**. Calling `pipe.score(...)` does not necessarily create logged metrics; you may see no metrics in the UI (by design here, since we’re autolog-only).
* **XGBoost (`mlflow.xgboost.autolog`)**

  * Logs **parameters** and **model**.
  * If you pass `eval_set=[(X_val, y_val)]` (and optionally `early_stopping_rounds`), it **automatically logs per-iteration eval metrics** (e.g., `mlogloss`). View the curve in the **Metrics** tab.

---

## 1) How to Run

```bash
docker compose exec trainer bash

# scikit-learn (autolog only)
python examples/06_autolog/run.py

# XGBoost (autolog only; training curves logged if eval_set is provided)
python examples/06_autolog/eval_run.py
```

Open MLflow UI: [http://localhost:5000](http://localhost:5000)

* **Experiment**: `06-autolog`
* **Runs**:

  * `sklearn_pipeline_autolog_only` — check **Params** and the **model** artifact.
  * `xgb_autolog_only` — check **Params**, **model**, and the **per-iteration metric curve** in **Metrics**.

---

## 2) Useful Autolog Options (FYI)

Exact options vary by framework/version, but commonly:

* `log_models=True|False` — whether to save the trained model
* `log_input_examples=True|False` — attempt to save an input example
* `registered_model_name="..."` — log and **register** the model in the registry by name
* `disable=True|False` — turn autolog off
* (XGBoost) using `eval_set`/`early_stopping_rounds` enables **iteration metrics** logging

> In this folder, we intentionally **do not** use manual logging (`log_metric(s)`, `log_artifact`) to keep it autolog-only.
