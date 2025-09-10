from src.mlflow_utils import setup_mlflow

import mlflow

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

EXP_NAME = "06-autolog"
RUN_NAME = "sklean_pipeline_autolog_only"
TAGS = {"stage": "train"}

setup_mlflow(EXP_NAME)

mlflow.sklearn.autolog(log_models = True) # Auto logging start

X, y = load_iris(return_X_y = True, as_frame = True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run(run_name = RUN_NAME, tags = TAGS) as run:
    pipe = Pipeline(
        steps = [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression())
        ]
    )
    pipe.fit(X_tr, y_tr)