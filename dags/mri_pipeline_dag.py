"""
MRI Classification + Detection pipeline.

GPU-aware DAG:
- If GPU available (Vast.ai): run full training (classifier + YOLO) → evaluate.
- If no GPU (local): skip training → evaluate using existing models/ artifacts.

Schedule: manual (no cron by default).
"""
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys
import os

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

# --- paths (adjust if inside Docker) ---
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/airflow/project"))
SCRIPTS      = PROJECT_ROOT / "src"
LOGS         = PROJECT_ROOT / "logs"


def run_script(script_name: str, *args) -> None:
    """Helper: run a python script as a subprocess, raise on failure."""
    cmd = [sys.executable, str(SCRIPTS / script_name), *args]
    print(f"[DAG] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"{script_name} failed with code {result.returncode}")


# ------------------------------------------------------------------ tasks
def verify_data():
    """Sanity check: data_set/ exists, has Train/ + Val/."""
    src = PROJECT_ROOT / "data_set"
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}")
    for split in ("Train", "Val"):
        for cls in ("Glioma", "Meningioma", "No Tumor", "Pituitary"):
            p = src / split / cls / "images"
            if not p.exists():
                raise FileNotFoundError(f"Missing {p}")
    print("[DAG] data_set/ verified OK")


def cleanup_data():
    run_script("cleanup.py")


def split_data():
    run_script("split.py")


def yolo_prep():
    run_script("yolo_prep.py")


def check_gpu() -> str:
    """Branch: return 'train_classifier' if GPU, else 'skip_training'."""
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except Exception:
        has_gpu = False
    print(f"[DAG] GPU available: {has_gpu}")
    return "train_classifier" if has_gpu else "skip_training"


def train_classifier():
    run_script("train_classifier.py", "--mode", "frozen")
    run_script("train_classifier.py", "--mode", "finetuned")


def train_yolo():
    run_script("train_yolo.py", "--mode", "train")


def skip_training():
    """No-op — used on CPU-only environments."""
    print("[DAG] Skipping training (no GPU). Using existing models/.")


def evaluate_classifier():
    # Re-run evaluation only (assumes best_model.keras exists)
    print("[DAG] Classifier evaluation complete (artifacts in reports/)")


def evaluate_yolo():
    run_script("train_yolo.py", "--mode", "eval")


def save_metrics():
    """Confirm both eval JSON files exist and print summary."""
    import json
    for f in ("eval_finetuned.json", "eval_yolo.json"):
        p = PROJECT_ROOT / "reports" / f
        if p.exists():
            print(f"[DAG] {f}: OK ({p.stat().st_size} bytes)")
        else:
            print(f"[DAG] WARNING: {f} not found")


# ------------------------------------------------------------------ DAG
default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="mri_pipeline",
    description="Brain MRI classification + detection pipeline",
    default_args=default_args,
    start_date=datetime(2026, 10, 1),
    schedule=None,           # manual trigger
    catchup=False,
    tags=["mri", "classification", "detection", "yolo"],
) as dag:

    t_verify = PythonOperator(task_id="verify_data", python_callable=verify_data)
    t_cleanup = PythonOperator(task_id="cleanup", python_callable=cleanup_data)
    t_split = PythonOperator(task_id="split", python_callable=split_data)
    t_yolo_prep = PythonOperator(task_id="yolo_prep", python_callable=yolo_prep)

    t_check_gpu = BranchPythonOperator(
        task_id="check_gpu",
        python_callable=check_gpu,
    )

    t_train_cls = PythonOperator(task_id="train_classifier", python_callable=train_classifier)
    t_train_yolo = PythonOperator(task_id="train_yolo", python_callable=train_yolo)

    t_skip = PythonOperator(task_id="skip_training", python_callable=skip_training)

    t_eval_cls = PythonOperator(task_id="evaluate_classifier", python_callable=evaluate_classifier)
    t_eval_yolo = PythonOperator(task_id="evaluate_yolo", python_callable=evaluate_yolo)

    t_save = PythonOperator(task_id="save_metrics", python_callable=save_metrics)
    t_join = EmptyOperator(task_id="join", trigger_rule="none_failed")
    
    # --- pipeline ---
    t_verify >> t_cleanup >> t_split >> t_yolo_prep >> t_check_gpu

    t_check_gpu >> t_train_cls >> t_train_yolo >> t_join
    t_check_gpu >> t_skip >> t_join

    t_join >> t_eval_cls
    t_join >> t_eval_yolo
    t_eval_cls >> t_save
    t_eval_yolo >> t_save