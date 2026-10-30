"""
Cleanup for MRI_class_DL dataset.
Reads data_set/, writes data_set_clean/. Original is never modified.

Fixes applied:
1. Move 14 misfiled images (labels match image, folder does not).
2. Relabel 1 file in place (m3_120: class 3 -> class 1).
3. Delete 2 orphan labels in Val/No Tumor.
4. Create 2 empty labels for unlabeled Val/No Tumor images.

Detection-specific filtering (class-2/class-3 stripping) is NOT done here —
that happens later in the YOLO dataset builder.
"""
import shutil, logging
from pathlib import Path
from datetime import datetime

ROOT  = Path(__file__).resolve().parent.parent
SRC   = ROOT / "data_set"
DST   = ROOT / "data_set_clean"
LOGS  = ROOT / "logs"
LOGS.mkdir(exist_ok=True)

log_path = LOGS / f"cleanup_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s",
    handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger("cleanup")

# ============================================================
# 1. Fresh copy
# ============================================================
if DST.exists():
    log.info(f"Removing existing {DST}")
    shutil.rmtree(DST)
log.info(f"Copying {SRC} -> {DST}")
shutil.copytree(SRC, DST)
log.info("Copy done")


# ---- 1b. Deduplicate by content hash ----
import hashlib
seen_hashes = {}

def file_hash(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()

dropped_dupes = 0
for split in ["Train", "Val"]:
    for cls in ["Glioma", "Meningioma", "No Tumor", "Pituitary"]:
        for img in sorted((DST / split / cls / "images").glob("*.jpg")):
            h = file_hash(img)
            if h in seen_hashes:
                lbl = DST / split / cls / "labels" / f"{img.stem}.txt"
                img.unlink()
                if lbl.exists(): lbl.unlink()
                dropped_dupes += 1
            else:
                seen_hashes[h] = (split, cls, img.name)

log.info(f"DEDUP: dropped {dropped_dupes} duplicate images")
log.info(f"DEDUP: unique images kept: {len(seen_hashes)}")

# ============================================================
# 2. Move/relabel decisions
# ============================================================
#   (split, src_cls, stem, action, target)
#   action "move"    -> move image+label to target folder
#   action "relabel" -> keep in place, rewrite all class_ids to target
MOVES = [
    ("Train", "Glioma",     "gg (278)",   "move",    "Pituitary"),
    ("Train", "Glioma",     "Te-gl_0043", "move",    "Meningioma"),
    ("Train", "Glioma",     "Tr-gl_1167", "move",    "Meningioma"),
    ("Train", "Meningioma", "m (49)",     "move",    "Pituitary"),
    ("Train", "Meningioma", "m (50)",     "move",    "Pituitary"),
    ("Train", "Meningioma", "m (51)",     "move",    "Pituitary"),
    ("Train", "Meningioma", "m (52)",     "move",    "Pituitary"),
    ("Train", "Meningioma", "m (53)",     "move",    "Pituitary"),
    ("Train", "Meningioma", "m (54)",     "move",    "Pituitary"),
    ("Train", "Meningioma", "m (55)",     "move",    "Pituitary"),
    ("Train", "Meningioma", "m3 (120)",   "relabel", 1),
    ("Train", "Meningioma", "Tr-me_0023", "move",    "Glioma"),
    ("Train", "Meningioma", "Tr-me_0119", "move",    "Pituitary"),
    ("Train", "Meningioma", "Tr-me_0385", "move",    "Glioma"),
    ("Train", "Pituitary",  "p (564)",    "move",    "Meningioma"),
]

def unique_path(folder, stem, suffix):
    p = folder / f"{stem}{suffix}"
    if not p.exists(): return p
    n = 1
    while True:
        p = folder / f"{stem}__fromMoved_{n}{suffix}"
        if not p.exists(): return p
        n += 1

for split, src_cls, stem, action, target in MOVES:
    img = DST / split / src_cls / "images" / f"{stem}.jpg"
    lbl = DST / split / src_cls / "labels" / f"{stem}.txt"
    if not (img.exists() and lbl.exists()):
        log.warning(f"SKIP {split}/{src_cls}/{stem}: source missing")
        continue

    if action == "move":
        dst_img_dir = DST / split / target / "images"
        dst_lbl_dir = DST / split / target / "labels"
        final_img = unique_path(dst_img_dir, stem, ".jpg")
        final_lbl = unique_path(dst_lbl_dir, stem, ".txt")
        shutil.move(str(img), str(final_img))
        shutil.move(str(lbl), str(final_lbl))
        log.info(f"MOVED {split}/{src_cls}/{stem} -> {split}/{target}/{final_img.name}")

    elif action == "relabel":
        lines = [ln for ln in lbl.read_text().splitlines() if ln.strip()]
        new_lines = []
        for ln in lines:
            parts = ln.split()
            parts[0] = str(target)
            new_lines.append(" ".join(parts))
        lbl.write_text("\n".join(new_lines) + "\n")
        log.info(f"RELABELED {split}/{src_cls}/{stem}: all boxes -> class {target}")



# ============================================================
# 3. Drop tumor-folder images with empty/missing labels
#    (e.g. Val/Glioma/gg (342): looks like No Tumor, mislabeled folder)
# ============================================================
TUMOR_CLASSES = ["Glioma", "Meningioma", "Pituitary"]
for split in ["Train", "Val"]:
    for cls in TUMOR_CLASSES:
        img_dir = DST / split / cls / "images"
        lbl_dir = DST / split / cls / "labels"
        if not img_dir.exists():
            continue
        for img in img_dir.glob("*.jpg"):
            lbl = lbl_dir / f"{img.stem}.txt"
            content = lbl.read_text().strip() if lbl.exists() else ""
            if not content:
                log.info(f"DROPPED unlabeled {split}/{cls}/{img.name} (no boxes)")
                img.unlink()
                if lbl.exists():
                    lbl.unlink()
# ============================================================
# 4. Val/No Tumor orphans
# ============================================================
for stem in ["image(53)", "image(60)"]:
    p = DST / "Val" / "No Tumor" / "labels" / f"{stem}.txt"
    if p.exists():
        p.unlink()
        log.info(f"DELETED orphan Val/No Tumor/{stem}.txt")

for stem in ["image(55)", "image(61)"]:
    img = DST / "Val" / "No Tumor" / "images" / f"{stem}.jpg"
    lbl = DST / "Val" / "No Tumor" / "labels" / f"{stem}.txt"
    if img.exists() and not lbl.exists():
        lbl.write_text("")
        log.info(f"CREATED empty label Val/No Tumor/{stem}.txt")

log.info("Cleanup complete")
print(f"\nLog written to: {log_path}")