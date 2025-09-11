from src.mlflow_utils import setup_mlflow

import os

import mlflow
from mlflow.tracking import MlflowClient


EXP_NAME        = "07-optuna-tuning"
PARENT_RUN_NAME = "optuna_tuning"
METRIC_KEY      = "val_f1_macro"
MODEL_NAME      = "iris_xgb"
PROMOTE_STAGE   = "Production"
ALIAS           = os.getenv("ALIAS")
ARCHIVE_EXISTING = os.getenv("ARCHIVE_EXISTING", "true").lower() in {"1", "true", "yes"}

setup_mlflow(EXP_NAME)
client = MlflowClient()

exp = client.get_experiment_by_name(EXP_NAME)


# 1) 최신 parent run(optuna_search) 찾기
parent_runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string=(
        "attributes.status = 'FINISHED' "
        f"and tags.mlflow.runName = '{PARENT_RUN_NAME}' "
        "and tags.stage = 'tuning'"
    ),
    order_by=["attributes.start_time DESC"],
    max_results=1,
)

parent = parent_runs[0]
parent_id = parent.info.run_id
print(f"[INFO] Parent run: {parent_id} (name={PARENT_RUN_NAME})")

# 2) child runs 중에서 METRIC_KEY 기준 최고 run 선정
#    (metrics.<key> DESC로 정렬)
child_runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string=(
        f"tags.mlflow.parentRunId = '{parent_id}' "
        "and attributes.status = 'FINISHED'"
    ),
    order_by=[f"metrics.{METRIC_KEY} DESC", "attributes.start_time DESC"],
    max_results=1,
)
best = child_runs[0]

# 메트릭 값 확인(없을 수도 있으니 get)
best_metric = None
if best.data.metrics and METRIC_KEY in best.data.metrics:
    best_metric = best.data.metrics[METRIC_KEY]
print(f"[INFO] Best trial run: {best.info.run_id}  "
        f"(metric {METRIC_KEY}={best_metric})")

# 3) 모델 아티팩트 경로 조립 (각 trial에서 artifact_path='model'로 저장했음)
source = f"{best.info.artifact_uri}/model"
print(f"[INFO] Source artifact path: {source}")

# 4) Registered Model 생성(없으면)
try:
    client.create_registered_model(MODEL_NAME)
    print(f"[INFO] Created registered model '{MODEL_NAME}'.")
except Exception:
    print(f"[INFO] Registered model '{MODEL_NAME}' already exists.")

# 5) 버전 생성
mv = client.create_model_version(
    name=MODEL_NAME,
    source=source,
    run_id=best.info.run_id,
    description=f"Registered from best trial (metric={METRIC_KEY}) under parent {parent_id}",
)
print(f"[OK] Registered '{MODEL_NAME}' version: {mv.version}")

# 6) (옵션) Stage 승급
if PROMOTE_STAGE:
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage=PROMOTE_STAGE,
        archive_existing_versions=ARCHIVE_EXISTING,
    )
    print(f"[OK] Promoted version {mv.version} -> stage '{PROMOTE_STAGE}' "
            f"(archive_existing={ARCHIVE_EXISTING})")

# 7) (옵션) Alias 부여
if ALIAS:
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=ALIAS,
        version=str(mv.version),
    )
    print(f"[OK] Set alias '{ALIAS}' -> version {mv.version}")

print("[DONE] Best-trial registration completed.")

