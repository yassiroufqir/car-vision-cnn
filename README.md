# Car Vision CNN

Freelance-style computer vision project: identify a car's make/model from a photo and detect visible damage.

## Client use case
Garages, used-car platforms and insurers upload a photo and get:
1. Predicted make/model + confidence
2. Detected damage zones (scratch, dent, crack, ...)

## Models
- **Model A (classification):** transfer learning, EfficientNet-B0 fine-tuned on Stanford Cars.
- **Model B (damage detection):** YOLOv8 trained on CarDD.

## Structure
```
data/classification/   Stanford Cars (train/ val/ test/)
data/damage/           CarDD in YOLO format
src/                   training and evaluation scripts
app/                   Gradio demo
reports/               metrics, confusion matrix, final report
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python src/train_classifier.py --data data/classification --epochs 15
python src/train_damage.py --data data/damage/data.yaml --epochs 50
python app/app.py
```

## Metrics (to fill)
| Model | Metric | Value |
|---|---|---|
| A | Top-1 / Top-5 accuracy | - |
| B | mAP@0.5 | - |

## Author
Yassir Oufqir, UEMF
