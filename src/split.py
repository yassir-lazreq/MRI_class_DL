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
import hashlib
from collections import defaultdict

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
    
    
def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

def split_by_hash(imgs: list[Path], test_frac: float, seed: int):
    """Group images by content hash, then split groups to avoid leakage."""
    random.seed(seed)
    groups = defaultdict(list)
    for img in imgs:
        groups[file_hash(img)].append(img)

    group_list = list(groups.values())
    random.shuffle(group_list)

    n_test_groups = round(len(group_list) * test_frac)
    test_imgs  = [img for g in group_list[:n_test_groups]  for img in g]
    train_imgs = [img for g in group_list[n_test_groups:] for img in g]
    return train_imgs, test_imgs

# ---- TRAIN -> train/test (stratified) ----
for cls in CLASSES:
    imgs = sorted((SRC / "Train" / cls / "images").glob("*.jpg"))
    train_imgs, test_imgs = split_by_hash(imgs, TEST_FRAC, SEED)
    for img in train_imgs:
        copy_pair(img, SRC/"Train"/cls/"labels"/f"{img.stem}.txt", "train", cls)
    for img in test_imgs:
        copy_pair(img, SRC/"Train"/cls/"labels"/f"{img.stem}.txt", "test", cls)
    log.info(f"{cls:12s} | total={len(imgs)} | train={len(train_imgs)} | test={len(test_imgs)}")

# ---- VAL copied as-is ----
# ---- VAL: copy as-is BUT drop any image whose hash is already in Train ----
train_hashes = set()
for split in ("train", "test"):
    for cls in CLASSES:
        for p in (DST / split / cls / "images").glob("*.jpg"):
            train_hashes.add(file_hash(p))
log.info(f"Train+test hashes: {len(train_hashes)}")

for cls in CLASSES:
    imgs = sorted((SRC / "Val" / cls / "images").glob("*.jpg"))
    kept = 0
    dropped = 0
    for img in imgs:
        if file_hash(img) in train_hashes:
            dropped += 1
            continue
        copy_pair(img, SRC/"Val"/cls/"labels"/f"{img.stem}.txt", "val", cls)
        kept += 1
    log.info(f"{cls:12s} | val kept={kept:4d} | dropped (train dup)={dropped}")

log.info("Split complete")
print(f"\nLog: {LOGS}")