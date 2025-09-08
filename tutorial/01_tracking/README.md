# 01 — Tracking (MLflow)

Learn what **Tracking** means in MLflow and practice logging a simple run with **params**, **metrics (with steps)**, **artifacts**, and **tags**.

---

## What is “Tracking” in MLflow?

**Tracking** is MLflow’s capability to **record and organize the results of ML experiments** so you can compare, reproduce, and share them later.
It captures:

* **Parameters (params)** — the inputs you chose (e.g., `C=1.0`, `max_iter=200`)
* **Metrics** — numeric outcomes over time/steps (e.g., validation accuracy per epoch)
* **Artifacts** — files produced by your run (plots, logs, reports, models)
* **Tags** — searchable labels for runs (e.g., `stage=demo`, `owner=hb`)

Tracking writes:

* **Run metadata** (who/when/what) and values (params/metrics/tags) to a **tracking store**
* **Artifacts** (files) to an **artifact store**

> By default (no special configuration), MLflow uses a local `./mlruns` folder for everything.
> If you set a `MLFLOW_TRACKING_URI`, metadata and artifacts can live in remote stores too.

---

## Core objects & terms

* **Experiment**
  A named collection of runs (think “project folder”). You select one with `mlflow.set_experiment(name)`.
* **Run**
  A single execution you log under an experiment: `with mlflow.start_run(): ...`
* **Params**
  Key–value pairs describing your choices (hyps, seeds, data version).
* **Metrics**
  Numeric values logged over **time/step** (e.g., per epoch).
  Use `mlflow.log_metric("val_acc", value, step=epoch)` to get time-series charts.
* **Artifacts**
  Arbitrary files (images, JSON, text reports, models). Organize with `artifact_path="reports"` etc.
* **Tags**
  Free-form labels that make searching and filtering easier.

---

## What this example does

* Creates/uses the experiment **`01-tracking-basics`**
* Starts a run named **`hello_tracking`**
* Logs:

  * `params`: `{"C": 1.0, "max_iter": 200}`
  * `metrics`: `val_acc` at steps `0, 1, 2` (to demonstrate time-series)
  * `artifact`: `reports/report.txt`
  * `tag`: `stage=demo`

---

## How to run

```bash
python examples/01_tracking_basics/run.py
```

To explore results in the UI (if you haven’t already launched one):

```bash
mlflow ui
# open http://127.0.0.1:5000
```

