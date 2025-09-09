import argparse
from mlflow.tracking import MlflowClient

MODEL_NAME = "iris_clf"

def parse_args():
    p = argparse.ArgumentParser(description="Promote/Demote a model version's stage and (optionally) set alias.")
    p.add_argument("--version", type=int, required=True, help="Model version number (e.g., 1)")
    p.add_argument("--stage", type=str, required=True,
                   choices=["None", "Staging", "Production", "Archived"],
                   help="Target stage")
    p.add_argument("--alias", type=str, default=None,
                   help="Optional alias to set on this version (e.g., 'prod' or 'champion')")
    p.add_argument("--no-archive-existing", action="store_true",
                   help="Do NOT archive existing versions in target stage (default: archive)")
    return p.parse_args()

def main():
    args = parse_args()
    c = MlflowClient()

    # Stage 전환
    c.transition_model_version_stage(
        name=MODEL_NAME,
        version=args.version,
        stage=args.stage,
        archive_existing_versions=not args.no_archive_existing,
    )
    print(f"[OK] '{MODEL_NAME}' version {args.version} -> stage '{args.stage}'")

    # (선택) Alias 부여
    if args.alias:
        # 동일 alias가 기존에 다른 버전에 연결되어 있다면 새로운 버전으로 재지정됨
        c.set_registered_model_alias(
            name=MODEL_NAME,
            alias=args.alias,
            version=str(args.version),
        )
        print(f"[OK] alias '{args.alias}' -> version {args.version}")

if __name__ == "__main__":
    main()
