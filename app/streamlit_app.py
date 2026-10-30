"""
Brain MRI — Classification + Detection Demo.
Loads models once at startup, serves predictions for uploaded images.
"""
import json
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO

# ------------------------------------------------------------------ paths
ROOT     = Path(__file__).resolve().parent.parent
MODELS   = ROOT / "models"
REPORTS  = ROOT / "reports"
CLASSES  = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
IMG_SIZE = (224, 224)

# ------------------------------------------------------------------ models
@st.cache_resource
def load_classifier():
    return tf.keras.models.load_model(str(MODELS / "classifier_best.keras"))

@st.cache_resource
def load_detector():
    return YOLO(str(MODELS / "detector_best.pt"))


@st.cache_data
def load_metrics():
    try:
        return {
            "cls": json.loads((REPORTS / "eval_finetuned.json").read_text()),
            "det": json.loads((REPORTS / "eval_yolo.json").read_text()),
        }
    except FileNotFoundError:
        return None

# ------------------------------------------------------------------ inference
def predict_classification(model, img: Image.Image):
    """Resize → preprocess → predict. Returns (class_name, probs_dict)."""
    img_resized = img.resize(IMG_SIZE).convert("RGB")
    arr = np.array(img_resized, dtype=np.float32)          # EfficientNet: keep [0,255]
    arr = np.expand_dims(arr, axis=0)                       # (1,224,224,3)
    probs = model.predict(arr, verbose=0)[0]
    return CLASSES[int(np.argmax(probs))], {CLASSES[i]: float(probs[i]) for i in range(4)}

def predict_detection(model, img: Image.Image):
    """Run YOLO. Returns list of boxes: [{class, conf, xyxy}]."""
    results = model.predict(np.array(img.convert("RGB")), verbose=False, conf=0.25)[0]
    out = []
    for box in results.boxes:
        out.append({
            "class": CLASSES[int(box.cls)],
            "conf":  float(box.conf),
            "xyxy":  box.xyxy[0].tolist(),
        })
    return out

def draw_boxes(img: Image.Image, detections):
    """Draw boxes on the PIL image using matplotlib + return as PIL."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    colors = {"Glioma": "red", "Meningioma": "lime",
              "No Tumor": "orange", "Pituitary": "cyan"}
    for d in detections:
        x1, y1, x2, y2 = d["xyxy"]
        ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                       linewidth=2, edgecolor=colors[d["class"]], fill=False))
        ax.text(x1, y1-5, f"{d['class']} {d['conf']:.2f}",
                color=colors[d["class"]], fontsize=10, backgroundcolor="black")
    ax.axis("off")
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# ------------------------------------------------------------------ UI
st.set_page_config(page_title="Brain MRI — Tumor Detection", page_icon="🧠", layout="wide")

st.title("🧠 Brain MRI — Tumor Classification & Detection")
st.caption("EfficientNet-B0 (classification) + YOLOv8n (detection) — pedagogical demo, not a medical device.")

with st.spinner("Loading models..."):
    classifier = load_classifier()
    detector   = load_detector()
    metrics    = load_metrics()

# sidebar — metrics
with st.sidebar:
    st.header("Model metrics")
    if metrics:
        st.subheader("Classification (test)")
        c = metrics["cls"]["test_metrics"]
        st.metric("Accuracy", f"{c['accuracy']*100:.1f}%")
        st.metric("Macro F1", f"{c['macro_f1']*100:.1f}%")
        st.subheader("Detection (test)")
        st.metric("mAP@0.5", f"{metrics['det']['mAP50']:.3f}")
        st.metric("mAP@0.5:0.95", f"{metrics['det']['mAP50_95']:.3f}")
    else:
        st.warning("Metrics not found in reports/")

# main
uploaded = st.file_uploader("Upload an MRI image (.jpg / .png)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an MRI image to run the pipeline.")
else:
    img = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Classification")
        cls_name, probs = predict_classification(classifier, img)
        st.success(f"**Prediction: {cls_name}**")
        for c, p in sorted(probs.items(), key=lambda x: -x[1]):
            st.progress(p, text=f"{c:12s} {p*100:5.1f}%")

    with col2:
        st.subheader("Detection")
        dets = predict_detection(detector, img)
        if dets:
            annotated = draw_boxes(img, dets)
            st.image(annotated, use_container_width=True)
            for d in dets:
                st.write(f"- **{d['class']}**  conf: {d['conf']:.2f}")
        else:
            st.warning("No tumors detected above 0.25 confidence.")
            st.image(img, use_container_width=True)

    st.divider()
    st.caption("⚠️ Pedagogical prototype — not a medical diagnostic tool.")