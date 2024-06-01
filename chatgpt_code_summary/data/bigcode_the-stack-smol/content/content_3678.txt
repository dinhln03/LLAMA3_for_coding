import os

def ensure_dir(path: str) -> str:
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    return path