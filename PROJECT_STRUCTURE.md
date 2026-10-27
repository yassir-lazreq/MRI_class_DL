# Project structure

```text
MRI_class_DL/
├── README.md
├── PROJECT_STRUCTURE.md
├── .gitignore
│
├── app/
│   └── .gitkeep                  # Streamlit application location
├── dags/
│   └── .gitkeep                  # Airflow DAG location
├── docker/
│   └── .gitkeep                  # Docker configuration location
│
├── data/
│   └── archives/                 # Local ZIP archives
│
├── data_set/                     # Original Train/Val dataset
├── data_set_clean/               # Cleaned image-label dataset
├── data_set_split/               # Classification train/val/test dataset
├── data_set_yolo/                # YOLO images, labels, and data.yaml
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_classification.ipynb
│   └── 05_detection.ipynb
│
├── requirements/
│   ├── requirements_train.txt
│   ├── requirements_inference.txt
│   └── requirements_dev.txt
│
├── src/
│   ├── cleanup.py                # Copy and correct the original dataset
│   ├── split.py                  # Create classification train/val/test splits
│   ├── yolo_prep.py              # Build the YOLO dataset
│   ├── train_classifier.py       # EfficientNet-B0 training and evaluation
│   └── train_yolo.py             # YOLOv8 training and evaluation
│
├── models/                       # Generated classification and detection models
│   ├── classifier_best.keras
│   ├── classifier_frozen.keras
│   ├── classifier_finetuned.keras
│   ├── detector_best.pt
│   └── pretrained/
│       └── yolov8n.pt            # YOLOv8 base checkpoint
├── reports/                      # Evaluation JSON files and plots
├── runs/                         # Ultralytics training run outputs
├── logs/                         # Script and training logs
└── .venv/                        # Local Python environment
```

## Data flow

```text
data_set/
    │
    ├── 01_data_exploration.ipynb
    │
    ▼
cleanup.py
    │
    ▼
data_set_clean/
    │
    ▼
split.py
    │
    ▼
data_set_split/
    ├── train/
    ├── val/
    └── test/
         │
         ├── train_classifier.py
         │       ├── models/
         │       └── reports/
         │
         └── yolo_prep.py
                 │
                 ▼
             data_set_yolo/
                 │
                 ▼
             train_yolo.py
                 └── runs/
```

## Dataset classes

The class order is consistent across classification and detection:

```text
0: Glioma
1: Meningioma
2: No Tumor
3: Pituitary
```

## Generated outputs

- `models/`: Keras classification checkpoints and detector weights.
- `reports/`: classification metrics, YOLO metrics, confusion matrices, precision-recall curves, and training curves.
- `runs/`: complete Ultralytics run artifacts, including plots, logs, validation images, and weights.
- `logs/`: timestamped execution logs from the preparation and training scripts.
