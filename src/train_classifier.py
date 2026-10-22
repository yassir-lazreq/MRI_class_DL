"""
Brain MRI 4-class classifier — EfficientNet-B0 transfer learning.

Modes:
    python src/train_classifier.py --mode frozen
        Stage 1: backbone frozen, train head only. LR=1e-3.
    python src/train_classifier.py --mode finetuned
        Stage 2: unfreeze top 30% of backbone. LR=1e-5. Starts from Stage 1 weights.

Common pipeline:
    data_set_split/{train,val,test}/{Glioma,Meningioma,No Tumor,Pituitary}
    → resize 224x224 → augment (train only) → normalize → EfficientNet-B0 → softmax(4)
"""
import argparse, json, logging, sys
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------------------------------------------ paths
ROOT       = Path(__file__).resolve().parent.parent
DATA_ROOT  = ROOT / "data_set_split"
MODELS_DIR = ROOT / "models"
REPORTS    = ROOT / "reports"
LOGS       = ROOT / "logs"
for d in (MODELS_DIR, REPORTS, LOGS):
    d.mkdir(exist_ok=True)

CLASSES     = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
AUTOTUNE    = tf.data.AUTOTUNE
SEED        = 42

# ------------------------------------------------------------------ logging
def setup_logging(mode):
    p = LOGS / f"train_{mode}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler(p, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("train"), p


# ------------------------------------------------------------------ data
def load_split(split, augment):
    """Load one split as tf.data.Dataset of (image, label) pairs."""
    return tf.keras.utils.image_dataset_from_directory(
        DATA_ROOT / split,
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,            # enforces ordering, matches label ids
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        seed=SEED,
    )


def build_augmentation():
    """Applied on the fly, train only."""
    return tf.keras.Sequential([
        layers.RandomRotation(0.04),                 # ~15°
        layers.RandomTranslation(0.10, 0.10),        # ±10%
        layers.RandomZoom(0.15),                     # ±15%
        layers.RandomContrast(0.15),                 # ±15%
        layers.RandomFlip("horizontal"),
    ], name="augmentation")


def prepare(ds, augment):
    """Apply augmentation (train only), preprocess_input, prefetch."""
    aug = build_augmentation() if augment else None
    def _map(x, y):
        if aug is not None:
            x = aug(x, training=True)
        x = preprocess_input(x)
        return x, y
    return ds.map(_map, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


def get_datasets():
    train = load_split("train", augment=True)
    val   = load_split("val",   augment=False)
    test  = load_split("test",  augment=False)
    return (prepare(train, augment=True),
            prepare(val,   augment=False),
            prepare(test,  augment=False))