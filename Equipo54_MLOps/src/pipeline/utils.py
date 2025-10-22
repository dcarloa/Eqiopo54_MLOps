import json
from pathlib import Path

try:
    import pandas as pd
except Exception:
    pd = None


def read_csv(path):
    p = Path(path)
    if not p.exists():
        print(f"File not found: {p}")
        return None
    if pd is None:
        # Fallback simple reader
        return [line.strip().split(',') for line in p.read_text(encoding='utf-8').splitlines()]
    return pd.read_csv(p)


def save_metrics(metrics: dict, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
