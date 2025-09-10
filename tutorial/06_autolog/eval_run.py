from src.mlflow_utils import setup_mlflow

import numpy as np

import mlflow

import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

EXP_NAME = "06-autolog"
RUN_NAME = "xgb_autolog_only"
TAGS = {"stage": "train"}

setup_mlflow(EXP_NAME)

mlflow.xgboost.autolog(log_models=True)

X, y = load_iris(return_X_y=True, as_frame=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run(run_name = RUN_NAME, tags=TAGS):
    params = {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "tree_method": "hist",                # GPU면 "gpu_hist"
        "objective": "multi:softprob",
        "num_class": len(np.unique(y)),
        "eval_metric": "mlogloss",
        "random_state": 42,
        "early_stopping_rounds": 30,
    }
    clf = xgb.XGBClassifier(**params)

    # eval_set/early_stopping을 주면 스텝별 eval metric이 자동 로깅됨
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        verbose=False
    )
