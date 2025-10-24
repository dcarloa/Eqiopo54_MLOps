# Minimal convenience script to run the pipeline on Windows
if (Test-Path .\.venv\Scripts\Activate.ps1) {
    Write-Host "Activating .venv"
    . .\.venv\Scripts\Activate.ps1
    $python = ".\.venv\Scripts\python.exe"
} else {
    Write-Host ".venv not found — using system python"
    $python = "python"
}

Write-Host "Running pipeline..."
& $python -m Equipo54_MLOps.src.pipeline.run_pipeline
