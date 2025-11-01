# Stage 1: Builder
# ============================
FROM python:3.9-slim as builder

WORKDIR /app

# --- Copy only what's needed for installing dependencies ---
COPY Equipo54_MLOps/setup.py Equipo54_MLOps/requirements.txt ./

# --- Install system dependencies for compiling certain packages ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# --- Install Python dependencies (production only via setup.py) ---
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

# --- Log installed dependencies (for debugging builds) ---
RUN echo "=== Installed dependencies ===" && pip freeze && echo "============================"

# --- Now copy the rest of the application code ---
COPY . .

# ============================
# Stage 2: Runtime Image
# ============================
FROM python:3.9-slim

WORKDIR /app

# --- Install lightweight runtime tools (e.g., git for DVC) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git && \
    rm -rf /var/lib/apt/lists/*

# --- Copy installed packages from builder ---
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# --- Copy the entire project from builder ---
COPY --from=builder /app /app

# --- Install the local package (to ensure imports work) ---
RUN cd /app/Equipo54_MLOps && pip install --no-cache-dir .

# Test AWS and DVC setups
RUN echo "Testing DVC and AWS setup..." && \
    if [ -f /app/scripts/setup_aws.sh ]; then \
        chmod +x /app/scripts/setup_aws.sh && \
        echo "✅ setup_aws.sh ready"; \
    else \
        echo "⚠️ setup_aws.sh not found, skipping"; \
    fi && \
    dvc --version && echo "✅ DVC installed successfully"

# Set working directory to the project root
WORKDIR /app/

# Default command
CMD ["/bin/bash"]