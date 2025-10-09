
## 📘 Data Version Control (DVC) + AWS S3 Setup

### Overview

This project uses **[DVC](https://dvc.org/)** for versioning large data files and synchronizing them with **AWS S3**.
DVC works together with **Git** to make our ML/data pipeline reproducible, without bloating the repo.

---

## 🚀 Quick Start for New Team Members

### 1️⃣ Prerequisites

* [Git](https://git-scm.com/)
* [DVC](https://dvc.org/doc/install)
* [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

Confirm they’re installed:

```bash
git --version
dvc --version
aws --version
```

---

### 2️⃣ Configure AWS Credentials

You’ll need access to the project’s S3 bucket.
Use one of the following options:

#### **Option A – Recommended for Local Dev**

Ask your admin for **temporary credentials** (Access key ID + Secret access key), then create a local `.env` file:

```
# .env (not committed to git)
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxx
AWS_DEFAULT_REGION=us-east-1
```

Then run the setup script:

- **Linux users**
```bash
./scripts/setup_aws.sh
```

- **Windows users**
```bash
.\scripts\setup_aws.ps1
```

This will configure the AWS CLI automatically.

#### **Option B – Using `aws configure` manually**

```bash
aws configure
```

Then enter:

```
AWS Access Key ID: ...
AWS Secret Access Key: ...
Default region name: us-east-1
```

---
### 3️⃣ Initialize DVC

```bash
dvc init
```

If this is your first time setting up the project, configure the remote:

```bash
dvc remote add -d myremote s3://my-dvc-study-entry-performance-data
dvc remote modify myremote region us-east-1
```

> ⚠️ Do this **only once** — the remote will be shared in the repo’s `.dvc/config`.

---

### 4️⃣ Pull the Latest Data


After cloning the repo and setting up AWS:

```bash
git pull
dvc pull
```

This will download the latest version of the datasets from S3.

---
### 5️⃣ Adding or Updating Data

If you generate new data or update an existing dataset:

```bash
dvc add Equipo54_MLOps/src/data
git add Equipo54_MLOps/src/data
git commit -m "Add raw dataset"
dvc push     # upload data to S3
git push     # commit metadata to Git
```

---

## 🧰 Folder & File Conventions

| File / Folder          | Purpose               | Commit to Git?                  |
|------------------------| --------------------- | ------------------------------- |
| `.dvc/`                | DVC internal metadata | ✅ Yes                           |
| `.dvc/cache/`          | Local cache of data   | ❌ No                            |
| `data/`                | Local working data    | ❌ No (DVC tracks it indirectly) |
| `*.dvc`                | Data pointer files    | ✅ Yes                           |
| `dvc.yaml`, `dvc.lock` | Pipeline definitions  | ✅ Yes                           |
| `.env`                 | Local credentials     | ❌ No                            |
| `scripts/*`            | AWS setup helper      | ✅ Yes                           |

---