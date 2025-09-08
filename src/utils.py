from pathlib import Path

def get_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)