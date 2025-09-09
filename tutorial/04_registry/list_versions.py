from datetime import datetime

from mlflow.tracking import MlflowClient

MODEL_NAME = "iris_clf"

c = MlflowClient()
versions = c.search_model_versions(f"name='{MODEL_NAME}'")
if not versions:
    print(f"No versions found for model '{MODEL_NAME}'. Run log_and_register.py first.")

print(f"Model: {MODEL_NAME}")
print("-" * 80)
print(f"{'VER':<5} {'STAGE':<12} {'ALIAS':<12} {'CREATED':<20} {'RUN_ID':<32}")
print("-" * 80)
for v in sorted(versions, key=lambda x: int(x.version)):
    created = datetime.fromtimestamp(v.creation_timestamp/1000).strftime("%Y-%m-%d %H:%M:%S")
    # aliases: list[str] (None일 수 있어 방어)
    aliases = ",".join(v.aliases) if getattr(v, "aliases", None) else "-"
    print(f"{v.version:<5} {v.current_stage:<12} {aliases:<12} {created:<20} {v.run_id:<32}")