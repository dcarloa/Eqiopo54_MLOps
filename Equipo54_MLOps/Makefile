.PHONY: help install data train predict clean all test

# Variables
PYTHON := python
DATA_RAW := data/raw/student_entry_performance_modified.csv
DATA_PROCESSED := data/processed/student_performance_clean.csv
MODEL_DIR := models/

help:
	@echo "🎓 Student Performance MLOps Project"
	@echo ""
	@echo "Comandos disponibles:"
	@echo "  make install        - Instalar dependencias"
	@echo "  make data          - Procesar datos"
	@echo "  make train         - Entrenar modelo"
	@echo "  make train-fast    - Entrenar modelo sin optimización"
	@echo "  make predict       - Hacer predicción interactiva"
	@echo "  make all           - Pipeline completo (data + train)"
	@echo "  make clean         - Limpiar archivos generados"
	@echo "  make test          - Ejecutar tests (futuro)"
	@echo ""

install:
	@echo "📦 Instalando dependencias..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "✅ Dependencias instaladas"

data:
	@echo "🔄 Procesando datos..."
	$(PYTHON) src/data/make_dataset.py $(DATA_RAW) $(DATA_PROCESSED)
	@echo "✅ Datos procesados"

train:
	@echo "🚀 Entrenando modelo con optimización..."
	$(PYTHON) src/models/train_model.py $(DATA_PROCESSED) $(MODEL_DIR)
	@echo "✅ Modelo entrenado"

train-fast:
	@echo "⚡ Entrenando modelo SIN optimización (rápido)..."
	$(PYTHON) src/models/train_model.py $(DATA_PROCESSED) $(MODEL_DIR) --no-optimize
	@echo "✅ Modelo entrenado"

predict:
	@echo "🔮 Modo predicción interactiva..."
	$(PYTHON) src/models/predict_model.py $(MODEL_DIR) --interactive

predict-csv:
	@echo "📊 Hacer predicciones desde CSV..."
	@echo "Uso: make predict-csv INPUT=data/new_students.csv OUTPUT=predictions.csv"
	$(PYTHON) src/models/predict_model.py $(MODEL_DIR) $(INPUT) --output $(OUTPUT)

all: data train
	@echo "✅ Pipeline completo ejecutado"

clean:
	@echo "🧹 Limpiando archivos generados..."
	rm -rf $(DATA_PROCESSED)
	rm -rf $(MODEL_DIR)*.pkl
	rm -rf $(MODEL_DIR)*.json
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf src/data/__pycache__
	rm -rf src/models/__pycache__
	@echo "✅ Archivos limpiados"

test:
	@echo "🧪 Ejecutando tests..."
	@echo "⚠️  Tests aún no implementados"
	# pytest tests/

.DEFAULT_GOAL := help