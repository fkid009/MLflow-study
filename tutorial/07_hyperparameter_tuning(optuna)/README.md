# 07 – Optuna Tuning

**Purpose of this folder**

* Map Optuna’s core concepts (**Study / Trial / Sampler / Pruner / Objective**) to the actual code flow
* After tuning completes, **automatically register the best Trial’s model** in the **Model Registry**

---

## What is Optuna?

* A **hyperparameter optimization library**. You define an objective function (train → evaluate → **return a scalar score**). Optuna then performs **sampling, exploration, and pruning** to find **strong configurations** efficiently.
* Core components:

  * **Study**: The container for an optimization session
  * **Trial**: One execution with a specific set of parameters (one training run)
  * **Sampler**: The strategy for proposing the next parameters (recommended default: **TPE**)
  * **Pruner**: **Early-stops** underperforming Trials (saves time/cost)
  * **Objective**: A user-defined function of the form `(trial) -> score`
* Strengths:

  * **Efficient search** (often finds strong results with fewer trials than grid search)
  * **Dynamic/conditional search spaces** (e.g., if model A then tune `lr`, if model B then tune `depth`)
  * **Fast failure** via **pruning** (early stopping)
  * Practical features for **parallel/distributed runs**, **resuming**, and **fixed seeds** for reproducibility

---
