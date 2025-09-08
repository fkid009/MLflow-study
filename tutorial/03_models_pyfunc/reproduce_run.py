import os
import pandas as pd

from src.path import TUTORIAL_DIR
from src.utils import get_metrics

import mlflow 
from mlflow.tracking import MlflowClient


EXP_NAME = "03-models-pyfunc" 
model_uri = os.environ.get("MODEL_URI")


# 1) 모델 URI 결정
# - MODEL_URI 환경변수가 있으면 우선 사용
# - 없으면 Experiment "03-models-pyfunc"의 최신 run에서 runs:/<run_id>/model 자동 선택
if not model_uri:
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXP_NAME)
    if not exp:
        raise RuntimeError(
            f"Experiment '{EXP_NAME}' not found. "
            "Run the training script first (examples/03_models_pyfunc/run.py)."
        )
    
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    print(f"[INFO] MODEL_URI not set; using latest run model: {model_uri}")
else:
    print(f"[INFO] Using MODEL_URI from env: {model_uri}")

# 2) 모델 로드
model = mlflow.pyfunc.load_model(model_uri)

# 3) 예시 입력 (Iris feature 스키마에 맞춤)
sample = pd.DataFrame(
    [
        [5.1, 3.5, 1.4, 0.2],
        [6.2, 2.8, 4.8, 1.8],
    ],
    columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ],
)

# 4) 예측
pred = model.predict(sample)

# 5) 출력
print("Input:")
print(sample)
print("\nPredictions:")
print(pred)

# 6) 재현 확인 (테스트 성능 확안)

X_TEST_PATH = TUTORIAL_DIR / "03_models_pyfunc" / "X_te.csv"
Y_TEST_PATH = TUTORIAL_DIR / "03_models_pyfunc" / "y_te.csv"

X_te = pd.read_csv(X_TEST_PATH)
y_te = pd.read_csv(Y_TEST_PATH)["y_true"].to_numpy()

y_pred = model.predict(X_te)
acc, precision, recall, f1 = get_metrics(y_te, y_pred, "macro")
print(f"accuracy: {acc}") # 0.9666666666666667
print(f"precision: {precision}") # 0.9696969696969697
print(f"recall: {recall}") # 0.9666666666666667
print(f"f1-score: {f1}") # 0.9665831244778612