from src.mlflow_utils import setup_mlflow
from src.utils import get_metrics

import os
import numpy as np

import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import mlflow, optuna
from mlflow.models.signature import infer_signature
from optuna.integration.mlflow import MLflowCallback

EXP_NAME = "07-optuna-tuning"
RUN_NAME = "optuna_tuning"
TAGS = {"stage": "tuning"}
N_TRIALS = 25

X, y = load_iris(return_X_y=True, as_frame=True)
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
NUM_CLASS = len(np.unique(y))

def objective(trial: optuna.Trial) -> float:
    """
    목적함수: 검증 정확도(val_accuracy) 최대화
    - autolog는 XGBoost의 eval metric 히스토리를 자동 기록
    - MLflowCallback이 trial.value를 metric_name으로 자동 로깅
    """
    # 탐색 하이퍼파라미터
    search_params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }
    # 고정 하이퍼파라미터
    static_params = {
        "tree_method": "hist",        # GPU면 "gpu_hist"
        "objective": "multi:softprob",
        "num_class": NUM_CLASS,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "early_stopping_rounds": 30,
    }
    params = {**search_params, **static_params}

    with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):

        clf = xgb.XGBClassifier(**params)
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        y_pred = clf.predict(X_val)

        acc, precision, recall, f1 = get_metrics(y_val, y_pred, "macro")
        mlflow.log_metrics({
            "val_accuracy": acc,
            f"val_precision": precision,
            f"val_recall": recall,
            f"val_f1": f1,
        })

        mlflow.log_params(params)

        signature = infer_signature(X_tr, clf.predict(X_tr))

        mlflow.xgboost.log_model(
            clf, 
            name = "model",
            signature = signature,
            input_example = X_val.head(2)
        )
        return f1

setup_mlflow(EXP_NAME)

optuna_sampler = optuna.samplers.TPESampler(seed = 42)

with mlflow.start_run(run_name = RUN_NAME, tags = TAGS):

    study = optuna.create_study(
        study_name="iris_xgb_acc",
        direction="maximize",
        sampler=optuna_sampler,
    )

    study.optimize(objective, n_trials=N_TRIALS)


    best = study.best_trial
    best_acc, best_prec, best_rec, best_f1 = None, None, None, None

    print("\n[Optuna] Best Trial")
    print(f"- number: {best.number}")
    print(f"- value of f1-score: {study.best_value:.4f}")
    print("- params:")
    for k, v in best.params.items():
        print(f"  - {k}: {v}")