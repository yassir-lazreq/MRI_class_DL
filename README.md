# MRI Brain Tumor Classification and Detection

Deep-learning project for four-class brain MRI analysis:

- Glioma
- Meningioma
- No Tumor
- Pituitary

The project contains two modeling pipelines:

1. **Image classification** with EfficientNet-B0.
2. **Tumor detection** with YOLOv8.

## Project pipeline

```text
data_set/
    │
    ▼
01_data_exploration.ipynb
    │
    ▼
src/cleanup.py
    │
    ▼
data_set_clean/
    │
    ▼
src/split.py
    │
    ├──► data_set_split/ ──► src/train_classifier.py ──► models/, reports/
    │
    └──► src/yolo_prep.py ──► data_set_yolo/ ──► src/train_yolo.py ──► runs/
```

## Environment setup

Create or activate the project virtual environment, then install the dependency group you need:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements/requirements_train.txt
```

For notebook development:

```powershell
pip install -r requirements/requirements_dev.txt
```

For deployment or inference:

```powershell
pip install -r requirements/requirements_inference.txt
```

## Data preparation

Run the preparation scripts from the project root:

```powershell
python src/cleanup.py
python src/split.py
python src/yolo_prep.py
```

The original `data_set/` directory is not modified. The scripts create:

- `data_set_clean/`: corrected image-label pairs.
- `data_set_split/`: classification `train`, `val`, and `test` folders.
- `data_set_yolo/`: flat YOLO images, labels, and `data.yaml`.

## Classification

Train the frozen-backbone EfficientNet-B0 model:

```powershell
python src/train_classifier.py --mode frozen
```

Fine-tune the top part of the backbone using the frozen-stage checkpoint:

```powershell
python src/train_classifier.py --mode finetuned
```

Classification models are saved in `models/`. Evaluation JSON files, confusion matrices, and training curves are saved in `reports/`.

## YOLO detection

Train YOLOv8:

```powershell
python src/train_yolo.py --mode train
```

Evaluate the saved detector on the test split:

```powershell
python src/train_yolo.py --mode eval
```

Run the prediction sanity check:

```powershell
python src/train_yolo.py --mode predict
```

YOLO training runs are stored in `runs/`. The selected plots and evaluation summaries are stored in `reports/`.

## Notebooks

The notebooks document the workflow:

1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_data_cleaning.ipynb`
3. `notebooks/03_data_preparation.ipynb`
4. `notebooks/04_classification.ipynb`
5. `notebooks/05_detection.ipynb`

## Repository notes

Large datasets, archives, logs, model files, reports, and run outputs are excluded from Git by `.gitignore`. The pretrained YOLOv8 checkpoint is stored at `models/pretrained/yolov8n.pt`.
