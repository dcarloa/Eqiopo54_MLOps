# MLOps Pipeline with DVC and Docker

This document explains the automated MLOps pipeline for this project, which uses DVC for data versioning and Docker for environment management.

## Components

Three key files have been added to automate and manage the pipeline:

1.  **`dvc.yaml`**: This is the core DVC pipeline definition file. It defines the stages of our machine learning workflow (`process_data`, `train`), their dependencies (input data, scripts), and their outputs (processed data, models). DVC uses this file to track and reproduce the pipeline.

2.  **`Dockerfile`**: This file defines a portable Docker environment for our project. It specifies the base Python image, installs all necessary dependencies from `requirements.txt`, and copies the project code into the image. This guarantees that the code runs in the exact same environment, whether on a local machine or in a CI/CD pipeline.

3.  **`.github/workflows/main.yml`**: This GitHub Actions workflow automates the entire process. On every push to the `main` branch, it performs the following steps:
    *   Builds the Docker image using the `Dockerfile`.
    *   Creates a `.env` file from GitHub Secrets.
    *   Runs a container from the image, loading the `.env` file to execute the DVC pipeline (`dvc repro`).
    *   Pushes any new data or model versions to the DVC remote storage.
    *   Commits the updated `dvc.lock` file back to the Git repository.

## How to Replicate the Pipeline Locally

You can run the entire pipeline on your local machine, just like GitHub Actions does. This is useful for testing changes before pushing them.

### Prerequisites

1.  **Install Docker**: Make sure you have Docker installed and running on your system. [Install Docker](https://docs.docker.com/get-docker/).
2.  **DVC Remote Storage Credentials**: You need a local copy of the credentials file for your DVC remote storage (e.g., the `gdrive-creds.json` for a Google Drive remote).

### Running the Local Pipeline

Follow these steps to replicate the CI/CD process locally.

**Step 1: Build the Docker Image**

This command builds the Docker image from the `Dockerfile` and tags it as `mlops-pipeline`.

```bash
docker build -t mlops-pipeline .
```

**Step 2: Create the `.env` file**

The GitHub workflow creates a `.env` file from secrets. You can do the same locally. This file will contain the credentials needed by DVC inside the container.

First, read your credentials file into a variable.

```bash
# For Linux/macOS:
export CREDS_CONTENT=$(cat /path/to/your/gdrive-creds.json)

# For Windows (PowerShell):
$env:CREDS_CONTENT = Get-Content C:\path\to\your\gdrive-creds.json -Raw
```

Now, create the `.env` file from that variable.

```bash
# For Linux/macOS:
echo "GDRIVE_CREDENTIALS_DATA=${CREDS_CONTENT}" > .env

# For Windows (PowerShell):
"GDRIVE_CREDENTIALS_DATA=$($env:CREDS_CONTENT)" | Out-File -FilePath .env -Encoding utf8
```

**Step 3: Run the DVC Pipeline Inside the Container**

This command now uses `--env-file .env` to securely load the credentials into the container.

```bash
docker run \
  --env-file .env \
  -v ./.git:/app/.git \
  mlops-pipeline \
  /bin/bash -c "
    # 1. Configure DVC from environment variable (loaded from .env)
    echo '$GDRIVE_CREDENTIALS_DATA' > gdrive-creds.json
    dvc remote modify myremote gdrive_service_account_json_file_path gdrive-creds.json

    # 2. Reproduce the DVC pipeline
    dvc pull
    dvc repro
    dvc push

    # 3. The workflow would normally commit dvc.lock here.
    # When running locally, you will do this manually in the next step.
    echo 'Pipeline finished. Check for changes in dvc.lock and commit them.'
  "
```

**Step 4: Commit Changes**

After the Docker command finishes, your `dvc.lock` file may have been updated by the pipeline. You should commit this file to Git.

```bash
git add dvc.lock
git commit -m "Ran pipeline locally, updated model"
```
