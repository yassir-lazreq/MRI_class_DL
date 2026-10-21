"""
Build YOLO-ready dataset from data_set_split/.
- Flattens per-class folders into images/{split}/ + labels/{split}/
- Strips class-2 boxes (No Tumor -> empty labels)
- Strips class-3 boxes from Glioma/Meningioma folders (anatomical noise)
- Keeps class-3 in Pituitary folder (target tumor)
- Writes data.yaml
"""
import shutil, logging
from pathlib import Path
from datetime import datetime

ROOT  = Path(__file__).resolve().parent.parent
SRC   = ROOT / "data_set_split"
DST   = ROOT / "data_set_yolo"
LOGS  = ROOT / "logs"; LOGS.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s",
    handlers=[logging.FileHandler(LOGS / f"yolo_prep_{datetime.now():%Y%m%d_%H%M%S}.log", encoding="utf-8"),
              logging.StreamHandler()],
)
log = logging.getLogger("yolo_prep")

CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
SPLITS  = ["train", "val", "test"]

if DST.exists():
    log.info(f"Removing {DST}")
    shutil.rmtree(DST)
for sp in SPLITS:
    (DST / "images" / sp).mkdir(parents=True, exist_ok=True)
    (DST / "labels" / sp).mkdir(parents=True, exist_ok=True)

def filter_label(text, folder_class):
    """Return filtered label text based on folder class."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if folder_class == "No Tumor":
        return ""  # drop all boxes -> negative example
    out = []
    for ln in lines:
        cid = int(float(ln.split()[0]))
        if folder_class in ("Glioma", "Meningioma") and cid == 3:
            continue  # strip anatomical pituitary from non-pituitary folders
        out.append(ln)
    return "\n".join(out) + ("\n" if out else "")

stats = {"kept": 0, "emptied": 0, "stripped_lines": 0}
for sp in SPLITS:
    for c in CLASSES:
        img_dir = SRC / sp / c / "images"
        lbl_dir = SRC / sp / c / "labels"
        for img in img_dir.glob("*.jpg"):
            lbl = lbl_dir / f"{img.stem}.txt"
            original = lbl.read_text() if lbl.exists() else ""
            filtered = filter_label(original, c)

            # collision-free naming: prefix with folder class
            safe_stem = f"{c.replace(' ', '_')}__{img.stem}"

            dst_img = DST / "images" / sp / f"{safe_stem}.jpg"
            dst_lbl = DST / "labels" / sp / f"{safe_stem}.txt"
            shutil.copy2(img, dst_img)
            dst_lbl.write_text(filtered)

            n_orig = len([ln for ln in original.splitlines() if ln.strip()])
            n_new  = len([ln for ln in filtered.splitlines() if ln.strip()])
            stats["stripped_lines"] += (n_orig - n_new)
            if n_new == 0 and n_orig > 0:
                stats["emptied"] += 1
            stats["kept"] += 1

# ---- data.yaml ----
yaml_path = DST / "data.yaml"
yaml_text = (
    f"path: {DST.resolve()}\n"
    f"train: images/train\n"
    f"val: images/val\n"
    f"test: images/test\n\n"
    f"nc: 4\n"
    f"names:\n"
    f"  0: Glioma\n"
    f"  1: Meningioma\n"
    f"  2: No Tumor\n"
    f"  3: Pituitary\n"
)
yaml_path.write_text(yaml_text)

log.info(f"Total images written: {stats['kept']}")
log.info(f"Boxes stripped: {stats['stripped_lines']}")
log.info(f"Files emptied (No Tumor): {stats['emptied']}")
log.info(f"data.yaml written to {yaml_path}")

# ---- final counts per split ----
for sp in SPLITS:
    n_img = len(list((DST / "images" / sp).glob("*.jpg")))
    n_lbl = len(list((DST / "labels" / sp).glob("*.txt")))
    log.info(f"{sp:6s}: {n_img:5d} images / {n_lbl:5d} labels")