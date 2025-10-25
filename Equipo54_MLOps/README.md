# 🎓 Student Performance Prediction - MLOps Project

Proyecto de Machine Learning para predecir el rendimiento académico de estudiantes usando Árboles de Decisión.

**Equipo**: EQUIPO54_MLOps  
**Modelo**: Decision Tree Classifier

La idea principal de este proyecto es desarrollar un modelo de Machine Learning de clasificación que prediga el rendimiento académico potencial de un estudiante (Performance) basándose en sus datos demográficos, antecedentes educativos y contexto socioeconómico disponibles al momento de su ingreso. Para lograrlo, se tómo la decisión de utilizar un modelo basado en Árboles de decisión, puesto que la principal ventaja de los árboles de decisión en un dataset puramente categórico es su capacidad para manejar este tipo de variables de forma nativa, eliminando la necesidad de preprocesamiento complejo.

---

## 📋 Tabla de Contenidos

- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Uso Rápido](#uso-rápido)
- [Pipeline Completo](#pipeline-completo)
- [Hacer Predicciones](#hacer-predicciones)
- [Métricas del Modelo](#métricas-del-modelo)
- [Troubleshooting](#troubleshooting)

---

## 📁 Estructura del Proyecto

```
EQUIPO54_MLOps/
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── make_dataset.py              # Procesamiento de datos
│   │   ├── raw/                         # Datos originales
│   │   │   └── student_entry_performance_modified.csv
│   │   └── processed/                   # Datos procesados
│   │       └── student_performance_clean.csv
│   └── models/
│       ├── __init__.py
│       ├── train_model.py               # Entrenamiento
│       └── predict_model.py             # Predicciones
│
├── models/                               # Modelos entrenados
│   ├── decision_tree_model.pkl
│   ├── label_encoders.pkl
│   ├── model_metrics.pkl
│   └── model_metrics.json
│
├── notebooks/                            # Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_decision_tree_training.ipynb
│
├── requirements.txt                      # Dependencias
├── Makefile                             # Automatización
├── .gitignore
└── README.md
```

---

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/EQUIPO54_MLOps.git
cd EQUIPO54_MLOps/Equipo54_MLOps
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv

# En Windows
venv\Scripts\activate

# En Mac/Linux
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## ⚡ Uso Rápido

### Opción 1: Usar modelo pre-entrenado

Si ya tienes el modelo entrenado en `models/`:

```bash
# Predicción interactiva
python src/models/predict_model.py models/ --interactive

# Predicción desde CSV
python src/models/predict_model.py models/ new_students.csv --output predictions.csv
```

### Opción 2: Entrenar desde cero

```bash
# 1. Procesar datos
python src/data/make_dataset.py src/data/raw/student_entry_performance_modified.csv src/data/processed/student_performance_clean.csv

# 2. Entrenar modelo
python src/models/train_model.py src/data/processed/student_performance_clean.csv models/ --no-optimize

# 3. Hacer predicciones
python src/models/predict_model.py models/ --interactive
```

---

## 🚀 Ejecución del Pipeline

### **Pipeline Completo (Recomendado)**
bash
python src/pipeline/run_pipeline.py

Con optimización:
bash
python src/pipeline/run_pipeline.py --optimize

---

### **Scripts Individuales (Paso a Paso)**

#### Paso 1: Limpieza de Datos
bash
python src/data/make_dataset.py src/data/raw/student_entry_performance_modified.csv src/data/processed/student_clean.csv

#### Paso 2: Generación de Features
bash
python src/features/build_features.py src/data/processed/student_clean.csv src/data/processed/student_features.csv

#### Paso 3: Entrenamiento del Modelo
bash
# Sin optimización (rápido)
python src/models/train_model.py src/data/processed/student_features.csv models/ --no-optimize

# Con optimización (recomendado)
python src/models/train_model.py src/data/processed/student_features.csv models/

#### Paso 4: Evaluación del Modelo
bash
python src/models/evaluate_model.py models/decision_tree_model.pkl models/train_test_split.pkl models/label_encoders.pkl reports/metrics/

---

### **Todo el Pipeline en Bloque**
bash
python src/data/make_dataset.py src/data/raw/student_entry_performance_modified.csv src/data/processed/student_clean.csv

python src/features/build_features.py src/data/processed/student_clean.csv src/data/processed/student_features.csv

python src/models/train_model.py src/data/processed/student_features.csv models/ --no-optimize

python src/models/evaluate_model.py models/decision_tree_model.pkl models/train_test_split.pkl models/label_encoders.pkl reports/metrics/

---

## 📊 Archivos Generados

src/data/processed/student_clean.csv - Datos limpios
src/data/processed/student_features.csv - Features codificadas
models/decision_tree_model.pkl - Modelo entrenado
models/label_encoders.pkl - Encoders
models/train_test_split.pkl - Splits
reports/metrics/metrics.json - Métricas

---

## 🎓 Documentación Adicional

### Notebooks

- `01_exploratory_data_analysis.ipynb` - Análisis exploratorio de datos
- `02_decision_tree_training.ipynb` - Entrenamiento completo con visualizaciones

Para ejecutar notebooks:

```bash
jupyter notebook notebooks/
```

### Scripts

**`src/data/make_dataset.py`**
- Procesa datos crudos
- Estandariza formato
- Guarda datos limpios

**`src/models/train_model.py`**
- Entrena modelo de árbol de decisión
- Optimiza hiperparámetros con GridSearchCV
- Guarda modelo y métricas

**`src/models/predict_model.py`**
- Carga modelo entrenado
- Hace predicciones en modo interactivo o batch
- Proporciona probabilidades y confianza

---



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
