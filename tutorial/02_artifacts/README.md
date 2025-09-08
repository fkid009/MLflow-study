Here you go—drop this into `examples/02_artifacts/README.md`.

---

# 02 – Artifacts (Logging text/dict/CSV/table/image)

The goal of this example is to **learn how to log artifacts to MLflow correctly**.

---

## 1) What you’ll learn

* What artifacts are and where they’re stored
* Practical patterns for `log_text`, `log_dict`, `log_table`, `log_image`, and `log_artifact`
* When local files appear vs when they don’t

---

## 2) Quick run

```bash
# enter the container
docker compose exec trainer bash

# run the example
python examples/02_artifacts/run.py

# open the UI
# MLflow UI: http://localhost:5000
# Experiment: 02-artifacts → Run: artifact_examples
# Expand the Artifacts tree to see: reports/, configs/, data/, tables/, images/
```

**Our compose setup**

* Artifact store: **MinIO (S3-compatible)**
* So `log_text` / `log_dict` / `log_table` / `log_image` **do not create local files**—they upload directly to the MinIO bucket.

---

## 3) Core concepts

### Artifacts

* Files produced during training/evaluation (models, plots, CSVs, reports, etc.).
* MLflow stores **metadata** in a DB (Postgres) and **files** in an **artifact store** (S3/MinIO/local).

### `artifact_file` (path + filename)

* Means **“within this run’s artifact store, save under this path/filename.”**
* Used by **functions that upload content directly** (no local file needed):

  * `mlflow.log_text(text, artifact_file="reports/notes.md")`
  * `mlflow.log_dict(obj, artifact_file="configs/train.json")`
  * `mlflow.log_table(df, artifact_file="tables/metrics.json")`
  * `mlflow.log_image(img, artifact_file="images/plot.png")`
* Notes:

  * **Uploads straight to the artifact store**, no local file created
  * Intermediate folders are created automatically; **same path overwrites**
  * Use **relative paths** with `/` separators

### `artifact_path` (folder only)

* Used when **you already have a local file** and want to upload it:

  * `mlflow.log_artifact("local/plot.png", artifact_path="images")`
  * The artifact becomes `images/plot.png` (filename comes from the local file)
* The **local file remains** (MLflow doesn’t delete it).

---

## 4) Why did `report.txt` show up locally in example 01?

* In 01 we did: **(1) create a local file → (2) upload with `log_artifact()`**.
* Because the `trainer` container mounts your project directory (`.:/work`), any files created there **also appear on your host**.

If you **don’t** want local files, use `log_text` / `log_dict` / `log_table` / `log_image` with `artifact_file`.

---

## 5) Common patterns

### (A) Upload directly (no local file)

```python
with mlflow.start_run():
    mlflow.log_text("done!", artifact_file="reports/summary.txt")
    mlflow.log_dict({"lr": 1e-3, "bs": 64}, artifact_file="configs/train.json")
    mlflow.log_table(df, artifact_file="tables/metrics.json")
    mlflow.log_image(img, artifact_file="images/gradient.png")
```

### (B) Upload an existing local file (keep local copy)

```python
from pathlib import Path
Path("outputs").mkdir(exist_ok=True)
Path("outputs/report.txt").write_text("local copy kept")
mlflow.log_artifact("outputs/report.txt", artifact_path="reports")
# Artifact: reports/report.txt (local file remains)
```

### (C) Use a temporary file and clean up automatically

```python
from tempfile import TemporaryDirectory
from pathlib import Path

with mlflow.start_run(), TemporaryDirectory() as tmp:
    p = Path(tmp) / "report.txt"
    p.write_text("temporary")
    mlflow.log_artifact(str(p), artifact_path="reports")
# temp folder auto-deletes here
```

### (D) See where artifacts go

```python
with mlflow.start_run():
    print("artifact_uri:", mlflow.get_artifact_uri())
    # Example: s3://mlflow-artifacts/<exp_id>/<run_id>/artifacts
```

### (E) Download an artifact later

```python
from mlflow.artifacts import download_artifacts
local_path = download_artifacts(artifact_uri="runs:/<run_id>/reports/summary.txt")
print("downloaded to:", local_path)
```

---

## 6) Safety checklist

* `artifact_file` = **path + filename** (e.g., `reports/notes.md`)
  `artifact_path` = **folder only** (e.g., `reports`)
* Always use `/` as the separator (even on Windows).
* **No absolute paths**: use `reports/notes.md`, not `/reports/notes.md`.
* Pick helpful extensions:

  * Tables → `.json` (nice preview in UI)
  * Images → `.png` / `.jpg`
* Large files already on disk → prefer `log_artifact()`.
* Models are usually saved with **`log_model(..., artifact_path="model")`** (folder-level).

---

## 7) Files in this folder

* `run.py`
  Demonstrates logging text, dict, CSV, table, and image artifacts in one go.
* (From the parent) `common/mlflow_env.py`
  Minimal wrapper to call `mlflow.set_experiment(...)`.

After running, the Artifacts tree should include:

```
reports/notes.md
configs/train_config.json
data/metrics.csv
tables/metrics.json
images/gradient.png
```

---


