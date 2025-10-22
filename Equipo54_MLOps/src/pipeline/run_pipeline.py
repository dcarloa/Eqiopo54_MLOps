"""
Script principal del pipeline. Ejecuta etapas básicas según `params.yaml`.
"""
import yaml
from pathlib import Path
from .utils import read_csv, save_metrics

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).resolve().parent / 'params.yaml'


def load_config(path=CONFIG_PATH):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run():
    cfg = load_config()
    data_path = ROOT / cfg['data']['raw']
    print(f"Leyendo datos desde {data_path}")
    df = read_csv(data_path)

    metrics = {'rows': len(df)} if df is not None else {'rows': 0}
    out = Path(cfg['outputs']['metrics'])
    out.parent.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, out)
    print(f"Métricas guardadas en {out}")


if __name__ == '__main__':
    run()
