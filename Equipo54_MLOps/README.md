# 🎓 Student Performance Prediction - MLOps Project

Proyecto de Machine Learning para predecir el rendimiento académico de estudiantes usando Árboles de Decisión.

**Equipo**: EQUIPO54_MLOps  
**Modelo**: Decision Tree Classifier

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

## 🔄 Pipeline Completo

### 1. Procesamiento de Datos

```bash
python src/data/make_dataset.py \
    src/data/raw/student_entry_performance_modified.csv \
    src/data/processed/student_performance_clean.csv
```

**Lo que hace:**
- ✅ Limpia y estandariza texto (mayúsculas, espacios)
- ✅ Normaliza categorías de Performance
- ✅ Elimina columnas sin valor predictivo
- ✅ Maneja valores faltantes
- ✅ Guarda datos limpios

### 2. Entrenamiento del Modelo

```bash
# Con optimización (recomendado, ~5-10 min)
python src/models/train_model.py \
    src/data/processed/student_performance_clean.csv \
    models/

# Sin optimización (rápido, ~30 seg)
python src/models/train_model.py \
    src/data/processed/student_performance_clean.csv \
    models/ \
    --no-optimize
```

**Opciones adicionales:**

```bash
# Cambiar tamaño del test set
python src/models/train_model.py src/data/processed/student_performance_clean.csv models/ --test-size 0.3

# Cambiar random state
python src/models/train_model.py src/data/processed/student_performance_clean.csv models/ --random-state 123
```

**Lo que genera:**
- `decision_tree_model.pkl` - Modelo entrenado
- `label_encoders.pkl` - Encoders para variables categóricas
- `model_metrics.pkl` - Métricas del modelo (pickle)
- `model_metrics.json` - Métricas del modelo (JSON legible)

### 3. Hacer Predicciones

#### **Modo Interactivo** (una predicción)

```bash
python src/models/predict_model.py models/ --interactive
```

**Ejemplo de interacción:**

```
Ingresa los datos del estudiante:

Gender
  Ejemplos válidos: MALE, FEMALE
  → Gender: MALE

Caste
  Ejemplos válidos: GENERAL, OBC, SC, ST
  → Caste: GENERAL

...

📊 RESULTADO DE LA PREDICCIÓN
=================================
🎯 Rendimiento Predicho: Excellent
📈 Confianza: 85.32%
```

#### **Modo Batch** (múltiples predicciones)

```bash
python src/models/predict_model.py models/ new_students.csv --output predictions.csv
```

**El CSV de salida incluirá:**
- Todas las columnas originales
- `Predicted_Performance` - Predicción del modelo
- `Prob_Excellent`, `Prob_Very Good`, `Prob_Good`, `Prob_Average` - Probabilidades
- `Confidence` - Nivel de confianza de la predicción

---

## 📊 Métricas del Modelo

Ver métricas guardadas:

```bash
cat models/model_metrics.json
```

**Ejemplo de estructura:**

```json
{
  "train_accuracy": 0.9234,
  "test_accuracy": 0.8567,
  "cv_score": 0.8521,
  "best_params": {
    "max_depth": 15,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "criterion": "gini"
  },
  "classification_report": {
    "Excellent": {
      "precision": 0.87,
      "recall": 0.89,
      "f1-score": 0.88
    }
  },
  "feature_importance_top10": {
    "Class_XII_Percentage": 0.3245,
    "Class_ X_Percentage": 0.2891,
    "coaching": 0.1456
  }
}
```

**Features más importantes:**
1. Class_XII_Percentage - Rendimiento en 12° grado
2. Class_X_Percentage - Rendimiento en 10° grado
3. coaching - Acceso a tutorías

---

## 🔧 Troubleshooting

### Error: "Modelo no encontrado"

```bash
# Entrenar el modelo primero
python src/models/train_model.py src/data/processed/student_performance_clean.csv models/
```

### Error: "Archivo CSV no existe"

```bash
# Verificar ruta del archivo
ls src/data/raw/

# Crear carpeta si no existe
mkdir -p src/data/raw
mkdir -p src/data/processed
```

### Error: "Valores no vistos"

El modelo maneja valores nuevos asignándolos a una categoría por defecto. Para mejores resultados, asegúrate de que los datos nuevos usen las mismas categorías que los datos de entrenamiento.

### Error de importación

```bash
# Reinstalar dependencias
pip install -r requirements.txt

# Verificar instalación
python -c "import sklearn, pandas, numpy; print('OK')"
```

---

## 🛠️ Uso del Makefile (Opcional)

Si tienes `make` instalado:

```bash
make install    # Instalar dependencias
make data       # Procesar datos
make train      # Entrenar modelo
make predict    # Predicción interactiva
make all        # Pipeline completo
make clean      # Limpiar archivos generados
```

---

## 📈 Próximos Pasos (Roadmap)

- [ ] API REST con FastAPI
- [ ] Dockerización del proyecto
- [ ] CI/CD con GitHub Actions
- [ ] Monitoring con MLflow
- [ ] Dashboard interactivo con Streamlit
- [ ] Pruebas unitarias con pytest
- [ ] Modelos ensemble (Random Forest, XGBoost)

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

## 👥 Contribuidores

**EQUIPO54_MLOps**

- ML Engineer: [Tu Nombre]
- Data Scientist: [Nombre]
- MLOps Engineer: [Nombre]

---

## 📄 Licencia

Este proyecto es para fines educativos.

---

## 📞 Contacto

Para preguntas o sugerencias:
- Email: [tu-email]
- GitHub: [tu-usuario]
- LinkedIn: [tu-perfil]

---

## 🙏 Agradecimientos

- Dataset: [Fuente del dataset]
- Herramientas: scikit-learn, pandas, matplotlib
- Inspiración: MLOps best practices

---

**¿Preguntas?** Revisa la sección de [Troubleshooting](#troubleshooting) o abre un issue en GitHub.


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
