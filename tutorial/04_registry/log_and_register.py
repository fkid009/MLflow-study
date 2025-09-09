import logging

from src.utils import get_metrics
from src.mlflow_utils import setup_mlflow

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

EXP_NAME   = "04-registry"
RUN_NAME   = "logreg_register"
MODEL_NAME = "iris_clf"

setup_mlflow(EXP_NAME)

X, y = load_iris(return_X_y=True, as_frame=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run(run_name=RUN_NAME, tags={"stage": "train"}) as run:
    # 1) 학습 + 로깅
    params = {"C": 1.0, "max_iter": 200, "solver": "lbfgs", "random_state": 42}
    mlflow.log_params(params)

    clf = LogisticRegression(**params).fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    acc, precision, recall, f1 = get_metrics(y_te, y_pred, "macro")
    mlflow.log_metrics({
        "val_acc": acc, "val_precision": precision, "val_recall": recall, "val_f1": f1
    })

    # 2) 시그니처/인풋 예시 추천
    signature = infer_signature(X_tr, clf.predict(X_tr))
    input_example = X_te.head(2)

    # 3) 모델을 '현재 런의 아티팩트(model/)'로만 기록 (등록은 나중에)
    mlflow.sklearn.log_model(
        sk_model=clf,
        name="model",                 
        signature=signature,
        input_example=input_example,
        registered_model_name=None,   
    )

    # 4) 레지스트리에 명시적으로 버전 생성
    source = mlflow.get_artifact_uri("model")  # 예: s3://.../artifacts/model
    client = MlflowClient()
    try:
        client.create_registered_model(MODEL_NAME)
    except Exception:
        pass 

    mv = client.create_model_version(
        name=MODEL_NAME,
        source=source,
        run_id=run.info.run_id,
        description="registered from run artifacts/model",
    )
    logging.info(f"Registered '{MODEL_NAME}' version: {mv.version}")