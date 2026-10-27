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
def load_split(split):
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
    train = load_split("train")
    val   = load_split("val")
    test  = load_split("test")
    return (prepare(train, augment=True),
            prepare(val,   augment=False),
            prepare(test,  augment=False))
    
# ------------------------------------------------------------------ model
def build_model(mode):
    """
    mode = "frozen"    : backbone fully frozen, train head only, LR=1e-3
    mode = "finetuned" : unfreeze top 30%, LR=1e-5, head already warm
    """
    base = EfficientNetB0(include_top=False, weights="imagenet",
                          input_shape=(*IMG_SIZE, 3))
    base.trainable = False

    if mode == "finetuned":
        n = len(base.layers)
        cut = int(n * 0.70)          # unfreeze last 30%
        for layer in base.layers[cut:]:
            layer.trainable = True

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name="input")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="drop1")(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4),
                     name="fc128")(x)
    x = layers.Dropout(0.3, name="drop2")(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax", name="logits")(x)

    model = models.Model(inputs, outputs, name=f"efficientnet_b0_{mode}")

    lr = 1e-3 if mode == "frozen" else 1e-5
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ------------------------------------------------------------------ class weights
def compute_class_weights(train_ds):
    """Counter-balance the ~2:1 class imbalance."""
    counts = np.zeros(len(CLASSES), dtype=np.int64)
    for _, y in train_ds:
        for label in y.numpy():
            counts[int(label)] += 1
    total = counts.sum()
    weights = total / (len(CLASSES) * counts)
    return {i: float(weights[i]) for i in range(len(CLASSES))}, counts


# ------------------------------------------------------------------ callbacks
def get_callbacks(mode):
    ckpt = MODELS_DIR / f"efficientnet_{mode}.keras"
    return [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(ckpt), monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-7, verbose=1,
        ),
    ], ckpt
    
# ------------------------------------------------------------------ train + eval
def evaluate_and_save(model, test_ds, mode, log):
    """Compute test metrics + confusion matrix, save to JSON."""
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt

    log.info(f"[{mode}] evaluating on test set...")
    y_true, y_pred = [], []
    for x, y in test_ds:
        p = model.predict(x, verbose=0)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(p.argmax(axis=1).tolist())

    report = classification_report(
        y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    # save metrics
    eval_path = REPORTS / f"eval_{mode}.json"
    eval_path.write_text(json.dumps({
        "mode": mode,
        "test_metrics": {
            "accuracy":    report["accuracy"],
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall":    report["macro avg"]["recall"],
            "macro_f1":        report["macro avg"]["f1-score"],
            "weighted_f1":     report["weighted avg"]["f1-score"],
        },
        "per_class": {c: report[c] for c in CLASSES},
        "confusion_matrix": cm.tolist(),
        "class_order": CLASSES,
    }, indent=2))

    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {mode} (test)")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig_path = REPORTS / f"confusion_{mode}.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()

    log.info(f"[{mode}] test accuracy = {report['accuracy']:.4f}")
    log.info(f"[{mode}] test macro F1 = {report['macro avg']['f1-score']:.4f}")
    log.info(f"[{mode}] saved {eval_path} and {fig_path}")

    return report


def plot_history(history, mode, log):
    """Save loss/accuracy curves."""
    import matplotlib.pyplot as plt
    h = history.history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(h["loss"], label="train"); ax1.plot(h["val_loss"], label="val")
    ax1.set_title(f"Loss — {mode}"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(h["accuracy"], label="train"); ax2.plot(h["val_accuracy"], label="val")
    ax2.set_title(f"Accuracy — {mode}"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Acc")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    p = REPORTS / f"curves_{mode}.png"
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"[{mode}] saved {p}")


def train(mode):
    log, log_path = setup_logging(mode)
    log.info(f"=== Mode: {mode} ===")
    log.info(f"Log file: {log_path}")

    train_ds, val_ds, test_ds = get_datasets()
    weights, counts = compute_class_weights(train_ds)
    log.info(f"Class weights: {weights}")
    log.info(f"Class counts : {counts.tolist()}")

    model = build_model(mode)
    cbs, ckpt = get_callbacks(mode)
    epochs = 15 if mode == "frozen" else 20

    # For fine-tuned mode: start from stage-1 weights
    if mode == "finetuned":
        stage1 = MODELS_DIR / "classifier_frozen.keras"
        if not stage1.exists():
            log.error(f"Stage 1 checkpoint not found at {stage1}. Run --mode frozen first.")
            sys.exit(1)
        log.info(f"Loading Stage 1 weights from {stage1}")
        model.load_weights(str(stage1))

    log.info(f"Training {mode} for up to {epochs} epochs...")
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs, callbacks=cbs,
        class_weight=weights,
    )

    # Save history
    (REPORTS / f"history_{mode}.json").write_text(
        json.dumps({k: [float(v) for v in vs] for k, vs in history.history.items()}, indent=2)
    )
    plot_history(history, mode, log)

    # Final evaluate on test (best weights restored by EarlyStopping)
    evaluate_and_save(model, test_ds, mode, log)
    log.info(f"[{mode}] done. Model saved at {ckpt}")


# ------------------------------------------------------------------ entry
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["frozen", "finetuned"], required=True,
                        help="frozen: train head only; finetuned: unfreeze top 30%")
    args = parser.parse_args()
    train(args.mode)


if __name__ == "__main__":
    main()