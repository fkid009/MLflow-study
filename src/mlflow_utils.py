import mlflow

from typing import Optional, Dict, Any


def setup_mlflow(experiment_name: str, tracking_uri: Optional[str] = None):
    """
    MLflow 환경 설정 헬퍼:
    - (옵션) tracking_uri가 주어지면 먼저 set_tracking_uri
    - experiment_name이 없으면 생성, 있으면 해당 실험으로 설정
    - Experiment 객체(또는 None)를 반환
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow.get_experiment_by_name(experiment_name)