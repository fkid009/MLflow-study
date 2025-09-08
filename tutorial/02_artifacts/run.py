from src.mlflow_utils import setup_mlflow
from src.path import TUTORIAL_DIR

import pandas as pd, numpy as np
import mlflow

EXP_NAME = "02-artifacts"
RUN_NAME = "artifact_examples"
TAGS = {"stage": "demo"}
CURRENT_PATH = TUTORIAL_DIR / "02_artifacts"

setup_mlflow(EXP_NAME)

with mlflow.start_run(run_name = RUN_NAME, tags = TAGS):
        # 1) 텍스트 (파일 없이 바로 업로드)
        mlflow.log_text(
            text=(
                "# Notes\n"
                "- This run demonstrates various MLflow artifact logging patterns.\n"
                "- Text, dict(JSON), CSV, table, and image are included.\n"
            ),
            artifact_file="reports/notes.md",
        )

        # 2) 딕셔너리 → JSON
        train_config = {
            "seed": 42,
            "epochs": 5,
            "batch_size": 64,
            "lr": 0.001,
            "comment": "baseline config for demo",
        }
        mlflow.log_dict(train_config, artifact_file="configs/train_config.json")

        # 3) CSV 파일 (로컬에 저장 후 업로드)
        df = pd.DataFrame(
            {
                "step": [0, 1, 2, 3, 4],
                "val_acc": [0.72, 0.75, 0.78, 0.80, 0.81],
                "tag": ["warmup", "phase1", "phase1", "phase2", "phase2"],
            }
        )
        csv_path = CURRENT_PATH / "metrics.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path), artifact_path="data")

        # 4) 테이블로 직접 로깅 (UI에서 미리보기 가능)
        mlflow.log_table(data=df, artifact_file="tables/metrics.json")

        # 5) 이미지 (넘파이 배열 → PNG)
        #    - 의존성 없이 간단한 그래디언트 이미지를 만들어 저장
        h, w = 120, 200
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # R 채널: 좌→우로 선형 증가, G: 고정값, B: 우→좌로 감소
        gradient = np.linspace(0, 255, w, dtype=np.uint8)
        img[:, :, 0] = gradient  # R
        img[:, :, 1] = 160       # G
        img[:, :, 2] = 255 - gradient  # B
        mlflow.log_image(img, artifact_file="images/gradient.png")