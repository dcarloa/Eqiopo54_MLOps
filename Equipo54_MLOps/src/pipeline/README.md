# Pipeline — Uso rápido

Este README contiene instrucciones mínimas para ejecutar el pipeline de entrenamiento y evaluación desde la carpeta `src/pipeline`.

Propósito
- Orquestar los pasos de limpieza, generación de features, entrenamiento y evaluación.

Requisitos mínimos
- Python 3.10+ (se probó con Python 3.11)
- Virtualenv/venv y dependencias instaladas (ver `requirements.txt` en la raíz del repo)
- Ejecutar desde la raíz del repositorio: `C:\git\Eqiopo54_MLOps`

Comandos rápidos (PowerShell)
1. Cambiar a la rama de trabajo y crear/activar entorno:

```powershell
git fetch origin
git checkout feat/pipeline-refactor
# Crear venv (si aún no existe)
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Ejecutar el pipeline (todo el flujo):

```powershell
# Ejecutar todo (pasos 1..4)
python src/pipeline/run_pipeline.py

# O ejecutar hasta un paso específico (ej: solo hasta entrenamiento)
python src/pipeline/run_pipeline.py --stop-at 3
```

Notas importantes
- El `run_pipeline.py` invoca scripts externos ubicados en `src/data/`, `src/features/` y `src/models/`:
  1. `src/data/make_dataset.py`
  2. `src/features/build_features.py`
  3. `src/models/train_model.py`
  4. `src/models/evaluate_model.py`

- Si el script de evaluación (`src/models/evaluate_model.py`) no está presente, el pipeline fallará en el paso 4. Para evitar el fallo temporalmente, ejecuta el pipeline con `--stop-at 3` (solo hasta entrenamiento) o ejecuta manualmente los pasos 1..3.

- El runner usa rutas relativas al ejecutar los scripts; por eso es importante ejecutar los comandos desde la raíz del repositorio.

Configuración
- Existe un archivo de configuración en `src/pipeline/params.yaml` que contiene parámetros y rutas, pero el runner actual (`run_pipeline.py`) no lo lee automáticamente; la configuración usada por el runner proviene de los valores por defecto en el script y de los argumentos CLI que recibe.

Salidas generadas (no versionar)
- Modelos: `models/` (p. ej. `models/decision_tree_model.pkl`)
- Encoders / splits: `models/label_encoders.pkl`, `models/train_test_split.pkl`
- Métricas: `reports/metrics/metrics.json`

Buenas prácticas
- No incluyas `.venv/`, modelos ni `reports/metrics/*` en los commits. Añade esos patrones a `.gitignore` si no están presentes.
- Si deseas que el runner use `params.yaml`, considera crear `src/pipeline/params.py` que cargue y valide el YAML y que `run_pipeline.py` lo importe.

Contacto
- Si algo no funciona, consulta con la persona encargada del pipeline (owner) antes de mover o renombrar scripts fuera de `src/pipeline`.

