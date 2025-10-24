"""
Script principal del pipeline. Ejecuta etapas básicas según `params.yaml`.
"""
import json
from pathlib import Path
from datetime import datetime

import yaml
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).resolve().parent / 'params.yaml'


def load_config(path=CONFIG_PATH):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def choose_input(cfg):
    primary = ROOT / cfg['paths']['input']
    fallback = ROOT / cfg['paths'].get('fallback_input', '')
    if primary.exists():
        return primary
    if fallback and fallback.exists():
        return fallback
    raise FileNotFoundError(f"No input features file found. Checked: {primary} and fallback: {fallback}")


def validate_dataframe(df, cfg):
    target = cfg.get('target')
    numeric = cfg['features'].get('numeric', [])
    categorical = cfg['features'].get('categorical', [])

    missing = [c for c in numeric + categorical + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    min_rows = cfg.get('validations', {}).get('min_rows', 1)
    if len(df) < min_rows:
        raise ValueError(f"Not enough rows: {len(df)} found, min required is {min_rows}")

    # Check nulls in critical columns
    critical = [target] + numeric + categorical
    nulls = {c: int(df[c].isnull().sum()) for c in critical}
    null_cols = [c for c, n in nulls.items() if n > 0]
    if null_cols and not cfg.get('validations', {}).get('impute_missing', False):
        raise ValueError(f"Null values present in columns: {null_cols}")


def run():
    cfg = load_config()
    input_path = choose_input(cfg)
    print(f"Using input: {input_path}")

    df = pd.read_csv(input_path)

    # Basic validation
    validate_dataframe(df, cfg)

    target = cfg['target']
    numeric = cfg['features'].get('numeric', [])
    categorical = cfg['features'].get('categorical', [])

    X = df[numeric + categorical]
    y = df[target]

    # splitting
    test_size = cfg.get('split', {}).get('test_size', 0.2)
    random_state = cfg.get('split', {}).get('random_state', 42)

    stratify = None
    if cfg.get('split', {}).get('stratify', True):
        counts = y.value_counts()
        n_classes = counts.size
        # compute intended test set size (at least 1)
        test_n = max(1, int(len(y) * test_size))
        # if test set would have fewer samples than number of classes or some class <2, disable stratify
        if test_n < n_classes or (counts < 2).any():
            print(f"Warning: disabling stratify because test_n={test_n} < n_classes={n_classes} or some class has <2 samples")
            stratify = None
        else:
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Preprocessing
    transformers = []
    if numeric:
        transformers.append(('num', StandardScaler(), numeric))
    if categorical:
        # support different sklearn versions: sparse_output (newer) vs sparse (older)
        try:
            cat_enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError:
            cat_enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        transformers.append(('cat', cat_enc, categorical))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Estimator
    model_params = cfg.get('model', {}).get('params', {})
    clf = DecisionTreeClassifier(**model_params)
    clf.fit(X_train_t, y_train)

    y_pred = clf.predict(X_test_t)

    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average='macro'))

    # Persist model
    model_out = ROOT / cfg['paths']['outputs'].get('model_out', 'models/decision_tree.joblib')
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': clf, 'preprocessor': preprocessor}, model_out)

    # Metrics
    metrics_out = ROOT / cfg['paths']['outputs'].get('metrics_out', 'reports/metrics/metrics.json')
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    support = {str(k): int(v) for k, v in y_test.value_counts().to_dict().items()}
    metrics = {
        'accuracy': acc,
        'f1_macro': f1,
        'support_per_class': support,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }

    with open(metrics_out, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"Input used: {input_path}")
    print(f"accuracy: {acc:.4f}")
    print(f"f1_macro: {f1:.4f}")


if __name__ == '__main__':
    run()
