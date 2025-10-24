# Especificación del pipeline — Contratos e interfaces

Este documento define los contratos (entradas, salidas), precondiciones, errores esperados y flujo de alto nivel para el pipeline de entrenamiento/evaluación. Está orientado a que la orquestación del Paso 3 pueda ejecutarlo sin ambigüedades.

## 1. Entradas

- Dataset de features (CSV). Ruta primaria esperada: `data/processed/student_features.csv`.
	- Fallback para pruebas locales: `src/pipeline/data/student_features_dummy.csv` (solo para testing / no en producción).
- Columna objetivo: nombre definido en `params.yaml` (ej.: `Performance`).
- Listas de variables: `features.numeric` y `features.categorical` definidas en `params.yaml`.

### Tipos esperados
- Columnas numéricas: integer/float (edad, horas de estudio, etc.).
- Columnas categóricas: string/enum (gender, internet_access, Performance).

## 2. Precondiciones (validaciones previas)

Antes de lanzar el pipeline se deben validar estas condiciones (si no se cumplen, abortar con error claro):

- Archivo de entrada existe y es legible.
- Columnas obligatorias presentes: todas las columnas listadas en `features.numeric`, `features.categorical` y la columna `target`.
- Mínimo de filas: `>= 10` (configurable); si menos, emitir warning y abortar/continuar según parámetro `allow_small_dataset`.
- No nulos en columnas clave (target y columnas necesarias). Si hay nulos, el pipeline debe:
	- Opción A: fallar y emitir error `MissingValuesError` indicando columnas con nulos.
	- Opción B: si `impute_missing: true` está activado en `params.yaml`, aplicar imputation y continuar.
- Dominio de categorías válido: las columnas categóricas deben contener valores que el preprocesador reconoce o se puede mapear; si aparecen categorías nuevas, loggear y (por defecto) incluirlas en LabelEncoder/OneHotEncoder.

## 3. Salidas

- Modelo serializado:
	- Ruta: `models/<model_name>_<version_or_timestamp>.pkl` (por convención). Formato: joblib / pickle compatible con scikit-learn.
- Métricas JSON:
	- Ruta: `reports/metrics/<run_id>_metrics.json` o `reports/metrics/pipeline_metrics.json`.
	- Contenido: cumple `src/pipeline/schema_metrics.json` (accuracy, precision_macro, recall_macro, f1_macro, support_per_class, timestamp, etc.).
- Opcional: figuras y reportes en `reports/figures/` o `reports/reports/`.

## 4. Errores tempranos (catálogo)

- MissingFileError: `Input features file not found: <path>` — cuando falta el CSV.
- MissingColumnError: `Column(s) missing: [col1, col2]` — cuando faltan columnas requeridas.
- NoSamplesForClassError: `No samples for class '<class>' in target` — al intentar stratify o al calcular métricas.
- TypeError: `Column '<col>' expected numeric but found non-numeric values` — al validar tipos.
- MissingValuesError: `Null values present in columns: [...]` — si no se permite imputación.
- ModelPersistError: `Failed to persist model at <path>: <reason>`

En todos los casos, los mensajes deben ser claros y contener el path y el nombre de la columna/entidad afectada.

## 5. Flujo del pipeline (alto nivel)

1. load_features: leer CSV de `data.raw` (o dummy si se usa modo test).
2. validate_schema: comprobar columnas/tipos/nulos/dominios.
3. split: `train_test_split` (usar `stratify` cuando `stratify: true`).
4. preprocessing:
	 - num_transformer: imputación (opcional) + escalado (StandardScaler o similar).
	 - cat_transformer: imputación (opcional) + OneHotEncoder / OrdinalEncoder / LabelEncoder.
	 - combinar con ColumnTransformer.
5. fit_estimator: entrenar el estimador (scikit-learn estimator definido en params).
6. predict & evaluate: generar predicciones en test y calcular métricas.
7. persist_artifacts: serializar modelo, encoders, y guardar métricas JSON y reportes.

## 6. Política de nombres y versionado

- Por defecto, no sobrescribir modelos anteriores. Usar sufijo con timestamp (ISO 8601) o un `run_id`:
	- ejemplo: `models/random_forest_2025-10-23T15-04-05.pkl`.
- Para métricas: `reports/metrics/metrics_2025-10-23T15-04-05.json`.
- Opcional: mantener `latest` symlink/archivo `models/latest.json` apuntando a la última versión.

## 7. Parámetros centralizados

- El pipeline usará `src/pipeline/params.yaml` (archivo local al pipeline) con:
	- paths: input, processed, outputs (model, metrics)
	- target
	- features: numeric, categorical
	- split: test_size, random_state, stratify
	- model: type and params
	- validations: min_rows, allow_small_dataset, impute_missing

## 8. Notas de compatibilidad

- Todo diseñado para integrarse con scikit-learn (ColumnTransformer + Estimator). Los estimadores deben seguir la API `fit(X,y)`/`predict(X)`.

---

Versión: 1.0 — especificación para implementar la orquestación en Paso 3.
