"""
Stratified train/test split for the classification dataset.
Source:  data_set_clean/{Train,Val}/{class}/{images,labels}
Dest:    data_set_split/{train,val,test}/{class}/{images,labels}

- Train: 85% stays train, 15% becomes test (stratified per class).
- Val:   copied as-is (already held out).

Deterministic: fixed seed.
"""
import shutil, logging, random
from pathlib import Path
from datetime import datetime

ROOT  = Path(__file__).resolve().parent.parent
SRC   = ROOT / "data_set_clean"
DST   = ROOT / "data_set_split"
LOGS  = ROOT / "logs"; LOGS.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s",
    handlers=[logging.FileHandler(LOGS / f"split_{datetime.now():%Y%m%d_%H%M%S}.log", encoding="utf-8"),
              logging.StreamHandler()],
)
log = logging.getLogger("split")

CLASSES   = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
TEST_FRAC = 0.15
SEED      = 42
random.seed(SEED)

if DST.exists():
    log.info(f"Removing {DST}")
    shutil.rmtree(DST)

def copy_pair(img, lbl, split, cls):
    tgt_img = DST / split / cls / "images" / img.name
    tgt_lbl = DST / split / cls / "labels" / lbl.name
    tgt_img.parent.mkdir(parents=True, exist_ok=True)
    tgt_lbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, tgt_img)
    shutil.copy2(lbl, tgt_lbl)

# ---- TRAIN -> train/test (stratified) ----
for cls in CLASSES:
    imgs = sorted((SRC / "Train" / cls / "images").glob("*.jpg"))
    random.shuffle(imgs)
    n_test  = round(len(imgs) * TEST_FRAC)
    n_train = len(imgs) - n_test

    for img in imgs[:n_train]:
        copy_pair(img, SRC/"Train"/cls/"labels"/f"{img.stem}.txt", "train", cls)
    for img in imgs[n_train:]:
        copy_pair(img, SRC/"Train"/cls/"labels"/f"{img.stem}.txt", "test", cls)

    log.info(f"{cls:12s} | total={len(imgs):5d} | train={n_train:5d} | test={n_test:4d}")

# ---- VAL copied as-is ----
for cls in CLASSES:
    imgs = sorted((SRC / "Val" / cls / "images").glob("*.jpg"))
    for img in imgs:
        copy_pair(img, SRC/"Val"/cls/"labels"/f"{img.stem}.txt", "val", cls)
    log.info(f"{cls:12s} | val={len(imgs):5d}")

log.info("Split complete")
print(f"\nLog: {LOGS}")