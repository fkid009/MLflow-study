# 04 – Model Registry (Register · Version · Stage · Alias)

## Objectives

* Register a trained model in the **Model Registry** to create a **version**.
* Move a version through **stages** (`None → Staging → Production → Archived`).
* (Optional) Assign **aliases** (e.g., `prod`, `champion`) for stable references.

---

## Prerequisites

* MLflow stack is running:

  ```bash
  docker compose up -d --build
  ```
* MLflow UI available at: **[http://localhost:5000](http://localhost:5000)**
* Open a shell in the trainer container:

  ```bash
  docker compose exec trainer bash
  ```

---

## Key Concepts

* **Registry**: The official catalog for your models. A **Registered Model** (e.g., `iris_clf`) contains multiple **model versions**.
* **Model Version**: An incrementing number (`1, 2, 3, ...`) assigned when a model is registered.
* **Stage**: The lifecycle state of a version:

  * `None` – just registered (unclassified)
  * `Staging` – under validation/QA
  * `Production` – serving in production
  * `Archived` – stored but not active
* **Alias**: A custom label bound to exactly one version within a registered model (e.g., `prod`, `champion`). Independent of stage.
* **Useful URIs**

  * By version: `models:/iris_clf/1`
  * By stage:   `models:/iris_clf/Production`
  * By alias:   `models:/iris_clf@prod`

---

## Files in This Folder

### `log_and_register.py`

* **What it does**: Trains a simple classifier (Iris + `LogisticRegression`) and **registers** it as `iris_clf`, creating a **new model version**.
* **Why it matters**: Saves **signature** and **input example** for reproducibility and safer deployment.

### `list_versions.py`

* **What it does**: Lists all versions for `iris_clf` with **version**, **stage**, **aliases**, **created time**, and **run\_id** in a compact table.

### `promote_stage.py`

* **What it does**: Transitions a **version** to a target **stage** (`None`, `Staging`, `Production`, `Archived`), and (optionally) sets an **alias**.
* **Default behavior**: Archives any existing versions in the target stage to ensure safe handover (blue/green-style swaps).

---

## How to Run

```bash
# Enter the trainer container
docker compose exec trainer bash

# 1) Train + Register (create a model version)
python tutorial/04_registry/log_and_register.py

# 2) List versions
python tutorial/04_registry/list_versions.py

# 3) Promote a version to Staging (example: v1)
python tutorial/04_registry/promote_stage.py --version 1 --stage Staging

# 4) Promote to Production (existing Production is archived automatically)
python tutorial/04_registry/promote_stage.py --version 1 --stage Production

# 5) (Optional) Assign an alias (e.g., 'prod') to v1
python tutorial/04_registry/promote_stage.py --version 1 --stage Production --alias prod
```

---

## What to Check in the UI

1. Go to **Models** → select `iris_clf`.
2. Verify **Versions** table (stage and aliases per version).
3. When promoting a new version to **Production**, the previous Production typically moves to **Archived** (default behavior in the script).

---

## Recommended Usage Patterns

* **Consumers / Deployment**

  * Stage-based reference: `models:/iris_clf/Production`
  * Alias-based reference: `models:/iris_clf@prod`
    Pick one convention for your team to keep deployment scripts simple.
* **Rollback**

  * Promote a different version to `Production` for immediate rollback, or re-promote the previous one.