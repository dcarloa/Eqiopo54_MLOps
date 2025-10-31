# src/pipeline/mlflow_runner.py
import json
import subprocess
from pathlib import Path
import mlflow
import mlflow.sklearn
import joblib
import argparse

# --- Config MLflow (ajusta si usas server remoto) ---
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_EXPERIMENT = "MLFlow_StudentsExperiment"
REGISTERED_MODEL_NAME = "student-performance-dt"

# --- Rutas del proyecto (coherentes con tu pipeline) ---
ROOT = Path(__file__).resolve().parents[2]  # .../Equipo54_MLOps
MODELS_DIR = ROOT / "models" / "latest"  # Carpeta fija para artefactos
PARAMS_YAML = ROOT / "src" / "pipeline" / "params.yaml"

def run_pipeline(optimize: bool = False, start_from: int = 1, stop_at: int = 3):
    cmd = [
        "python",
        str(ROOT / "src" / "pipeline" / "run_pipeline.py"),
        "--start-from", str(start_from),
        "--stop-at", str(stop_at),
    ]
    if optimize:
        cmd.append("--optimize")

    print(f" Ejecutando pipeline: {' '.join(cmd)}")
    # Capturamos salida para dejarla como artefacto si quieres
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError(f"Pipeline falló con código {res.returncode}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    ap.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--no-register", action="store_true",
                    help="No registrar el modelo en el Model Registry; solo log_model")
    args = ap.parse_args()

    # Config MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name="pipeline_run_dt"):
        # 1) Ejecutar pipeline (1→3; la evaluación ya vive dentro de train_model.py)
        run_pipeline(optimize=args.optimize, start_from=1, stop_at=3)

        # 2) Usar directorio fijo models/latest/ para artefactos
        print(f"📂 Usando artefactos desde: {MODELS_DIR}")

        metrics_json = MODELS_DIR / "model_metrics.json"
        model_pkl = MODELS_DIR / "decision_tree_model.pkl"

        # 3) Log de artefactos "crudos": params y métricas
        if PARAMS_YAML.exists():
            mlflow.log_artifact(str(PARAMS_YAML), artifact_path="config")
        else:
            print(f" No encontrado: {PARAMS_YAML}")

        if metrics_json.exists():
            # Registra métricas "clave" como mlflow.log_metric
            with open(metrics_json, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            # Escoge llaves estándar si existen
            for k in ("train_accuracy", "test_accuracy", "f1_macro", "accuracy"):
                if k in metrics:
                    mlflow.log_metric(k, float(metrics[k]))
            # Sube el archivo completo como artefacto
            mlflow.log_artifact(str(metrics_json), artifact_path="metrics")
        else:
            print(f" No encontrado: {metrics_json}")

        # 4) Log del modelo
        if model_pkl.exists():
            model_obj = joblib.load(model_pkl)
            # Nota: tu joblib guarda SOLO el estimador o incluye más? (en tu caso, DecisionTree puro)
            # Si en el futuro guardas dicts, ajusta a model_obj['model'].
            registered_name = None if args.no_register else REGISTERED_MODEL_NAME
            model_info = mlflow.sklearn.log_model(
                sk_model=model_obj,
                artifact_path="model",
                registered_model_name=registered_name
            )
            print(f" Modelo registrado/loggeado: {model_info.model_uri}")
        else:
            print(f" No encontrado: {model_pkl}")

        # 5) Params del entrenamiento (opcionales)
        # Si quieres también "parametrizar" lo corrido:
        mlflow.log_param("optimize", args.optimize)
        mlflow.log_param("pipeline_steps", "1-3 (train incluye eval)")

if __name__ == "__main__":
    main()
