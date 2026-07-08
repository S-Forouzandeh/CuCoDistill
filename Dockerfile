# CuCoDistill — reproducible GPU environment (paper App. B.4 stack)
#   docker build -t cucodistill .
#   docker run --gpus all -it cucodistill                 # default: accuracy table
#   docker run --gpus all -it cucodistill \
#     python scripts/synthetic_sweep.py                    # any other command
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace
# core deps first (better layer caching); DHG baselines are optional
COPY requirements.txt .
RUN pip install --no-cache-dir numpy>=1.23 scipy>=1.10 && \
    pip install --no-cache-dir dhg>=0.9.5 optuna scikit-learn || \
    echo "DHG install skipped; falling back to in-repo reference baselines"
COPY . .

# default entry point: reproduce the node-classification accuracy table
CMD ["python", "scripts/run_table.py", "--table", "accuracy"]
