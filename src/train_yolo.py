"""
YOLOv8 training + evaluation for brain MRI tumor detection.

Modes:
    python src/train_yolo.py --mode train     # fine-tune yolov8n on data_set_yolo/
    python src/train_yolo.py --mode eval      # evaluate saved model on test split
    python src/train_yolo.py --mode predict   # run inference on 10 test images (sanity)
"""
import argparse, json, logging, sys, shutil
from pathlib import Path
from datetime import datetime

import torch
from ultralytics import YOLO
import cv2
import numpy as np

ROOT        = Path(__file__).resolve().parent.parent
DATA_YAML   = ROOT / "data_set_yolo" / "data.yaml"
MODELS_DIR  = ROOT / "models"
PRETRAINED  = MODELS_DIR / "pretrained"
REPORTS     = ROOT / "reports"
LOGS        = ROOT / "logs"
RUNS_DIR    = ROOT / "runs"
for d in (MODELS_DIR, PRETRAINED, REPORTS, LOGS, RUNS_DIR):
    d.mkdir(exist_ok=True)

CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

def setup_logging(mode):
    p = LOGS / f"yolo_{mode}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler(p, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("yolo"), p


# ------------------------------------------------------------------ train
def train(log):
    if not DATA_YAML.exists():
        log.error(f"Missing {DATA_YAML}")
        sys.exit(1)

    device = 0 if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}  |  CUDA available: {torch.cuda.is_available()}")

    weights = PRETRAINED / "yolov8n.pt"
    if not weights.exists():
        log.error(f"Missing pretrained weights: {weights}")
        sys.exit(1)

    model = YOLO(str(weights))
    log.info(f"Loaded YOLOv8n base weights from {weights}")

    results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=16,
        patience=10,
        device=device,
        project=str(RUNS_DIR),
        name="yolo_brain_mri",
        exist_ok=True,
        # augmentation (Ultralytics defaults + tuned)
        mosaic=1.0,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        # logging
        plots=True,
        save=True,
        verbose=True,
    )

    # copy best.pt into models/ for a stable path
    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        dst = MODELS_DIR / "detector_best.pt"
        shutil.copy2(best, dst)
        log.info(f"Best YOLO weights copied to {dst}")
    else:
        log.warning(f"best.pt not found in {run_dir}/weights")

    log.info("Training complete")


# ------------------------------------------------------------------ eval
def evaluate(log):
    weights = MODELS_DIR / "detector_best.pt"
    if not weights.exists():
        log.error(f"Missing {weights}. Run --mode train first.")
        sys.exit(1)

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(weights))

    log.info("Evaluating on test split...")
    results = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=16,
        device=device,
        project=str(RUNS_DIR),
        name="yolo_eval_test",
        exist_ok=True,
    )

    # metrics dict
    metrics = {
        "mAP50":     float(results.box.map50),
        "mAP50_95":  float(results.box.map),
        "per_class": {},
    }
    for i, name in enumerate(CLASSES):
        metrics["per_class"][name] = {
            "precision": float(results.box.p[i]) if i < len(results.box.p) else None,
            "recall":    float(results.box.r[i]) if i < len(results.box.r) else None,
            "mAP50":     float(results.box.ap50[i]) if i < len(results.box.ap50) else None,
            "mAP50_95":  float(results.box.ap[i]) if i < len(results.box.ap) else None,
        }

    out = REPORTS / "eval_yolo.json"
    out.write_text(json.dumps(metrics, indent=2))
    log.info(f"mAP50:    {metrics['mAP50']:.4f}")
    log.info(f"mAP50-95: {metrics['mAP50_95']:.4f}")
    for name, m in metrics["per_class"].items():
        if m["mAP50"] is not None:
            log.info(f"  {name:12s}: P={m['precision']:.3f} R={m['recall']:.3f} "
                     f"mAP50={m['mAP50']:.3f}")
    log.info(f"Saved {out}")


# ------------------------------------------------------------------ predict (visual)
def predict(log, n=10):
    weights = MODELS_DIR / "detector_best.pt"
    if not weights.exists():
        log.error(f"Missing {weights}")
        sys.exit(1)

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(weights))

    test_imgs = sorted((ROOT / "data_set_yolo" / "images" / "test").glob("*.jpg"))[:n]
    out_dir = REPORTS / "yolo_predictions"
    out_dir.mkdir(exist_ok=True)

    log.info(f"Running inference on {len(test_imgs)} test images...")
    for img in test_imgs:
        res = model.predict(str(img), imgsz=640, conf=0.25, device=device, verbose=False)[0]
        out_path = out_dir / f"pred_{img.name}"
        cv2.imwrite(str(out_path), res.plot())
    log.info(f"Saved {len(test_imgs)} prediction images to {out_dir}")


# ------------------------------------------------------------------ entry
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "predict"], required=True)
    args = parser.parse_args()

    log, log_path = setup_logging(args.mode)
    log.info(f"=== YOLO mode: {args.mode} ===")
    log.info(f"Log: {log_path}")

    if args.mode == "train":     train(log)
    elif args.mode == "eval":    evaluate(log)
    elif args.mode == "predict": predict(log)


if __name__ == "__main__":
    main()