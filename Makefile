.PHONY: setup run clean

setup:
	python -m venv .venv || echo "venv exists"
	.venv\Scripts\python.exe -m pip install --upgrade pip
	.venv\Scripts\python.exe -m pip install -r requirements.txt

run:
	.venv\Scripts\python.exe -m Equipo54_MLOps.src.pipeline.run_pipeline

clean:
	rem Remove generated model and metrics (safe: don't touch data or DVC)
	if exist models\decision_tree.joblib del /q models\decision_tree.joblib
	if exist reports\metrics\metrics.json del /q reports\metrics\metrics.json
