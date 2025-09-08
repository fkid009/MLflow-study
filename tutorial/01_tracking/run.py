from src.mlflow_utils import setup_mlflow
from src.path import TUTORIAL_DIR

import mlflow
import time

EXP_NAME = "01-tracking-basics"
RUN_NAME = "hello_tracking"
TAGS = {"stage": "demo"}

CURRENT_PATH = TUTORIAL_DIR / "01_tracking"

setup_mlflow(EXP_NAME)

with mlflow.start_run(run_name=RUN_NAME, tags=TAGS):
        # 1) Params
        params = {"C": 1.0, "max_iter": 200}
        mlflow.log_params(params)

        # 2) Metrics 
        acc_history = [0.70, 0.76, 0.79]
        for step, acc in enumerate(acc_history):
            mlflow.log_metric("val_acc", acc, step=step)
            time.sleep(0.2) 

        # 3) Artifact (간단한 텍스트 리포트)
        report_path = CURRENT_PATH / "report.txt"
        report_path.write_text(
            "Tracking basics demo completed.\n"
            f"final_val_acc={acc_history[-1]:.4f}\n"
            f"params={params}\n"
        )
        mlflow.log_artifact(str(report_path), artifact_path="reports")