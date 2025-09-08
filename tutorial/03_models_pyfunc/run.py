from src.mlflow_utils import setup_mlflow
from src.utils import get_metrics
from src.path import TUTORIAL_DIR

import os, logging
import pandas as pd

import mlflow
from mlflow.models.signature import infer_signature

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

EXP_NAME = "03-models-pyfunc"
RUN_NAME = "logreg_v1"
TAGS = {"stage": "train"}
REGISTER_NAME = "iris_clf"
CURRENT_PATH = TUTORIAL_DIR / "03_models_pyfunc"

setup_mlflow(EXP_NAME)

X, y = load_iris(return_X_y = True, as_frame = True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 42,
    stratify = y
)
X_te.to_csv(CURRENT_PATH / "X_te.csv", index=False) # 재현 확인용
pd.DataFrame({"y_true": y_te}).to_csv(CURRENT_PATH / "y_te.csv", index=False) # 재현 확인용

with mlflow.start_run(run_name = RUN_NAME, tags = TAGS) as run:
    # 1) 학습 & 로깅
    params = {"C": 1.0, "max_iter": 200, "solver": "lbfgs", "random_state": 42}
    mlflow.log_params(params)

    clf = LogisticRegression(**params)
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    acc, precision, recall, f1 = get_metrics(y_te, y_pred, "macro")
    metrics = {
        "val_acc": acc,
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": f1
    }
    mlflow.log_metrics(metrics)

    # 2) signature / input_example
    signature = infer_signature(X_tr, clf.predict(X_tr))
    input_example = X_te.head(2)

    # 3) 모델 저장 (+ 선택: 레지스트리 등록)
    do_register = str(os.environ.get("REGISTER_MODEL", "")).lower() in {"1","true","yes"}
    registered_model_name = REGISTER_NAME if do_register else None

    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        registered_model_name=registered_model_name,
    )

    # 4) 간단 리포트 아티팩트
    report = "\n".join([
        "Model: LogisticRegression",
        f"val_accuracy: {acc:.4f}",
        f"val_precision: {precision:.4f}",
        f"val_recall: {recall:.4f}",
        f"val_f1_macro: {f1:.4f}",
        f"registered: {do_register} ({registered_model_name})"
    ])
    mlflow.log_text(report, artifact_file="reports/summary.txt")

    if do_register:
        logging.info(f"[OK] Registered to Model Registry as '{REGISTER_NAME}'")