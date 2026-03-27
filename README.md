# RetinaIQ — DR Grading Model

5-class diabetic retinopathy grading (No DR → Proliferative DR) using transfer learning.

## Setup

```bash
pip install -r requirements.txt
```

## Dataset Format (APTOS-style)

```
<dataset_path>/
    train.csv          # columns: id_code, diagnosis (0-4)
    train_images/      # or images/
        <id_code>.png
        ...
```

## Train

```bash
python train.py --dataset_path /path/to/dataset
```

Options:
- `--model` — backbone: `efficientnetv2b3` (default), `efficientnetb0`, `mobilenetv3`, `densenet121`
- `--epochs` — max epochs (default: 5)
- `--batch_size` — default: 16
- `--output` — weights output path (default: `ml_models/retinaiq_<model>.h5`)

Outputs saved to `ml_models/`:
- `retinaiq_<model>.h5` — trained weights
- `training_history.json` — per-epoch metrics + final test scores (loss, accuracy, AUC)

## Evaluate

```bash
python evaluate.py --dataset_path /path/to/dataset --weights ml_models/retinaiq_efficientnetv2b3.h5
```

Options:
- `--model` — must match the backbone used during training (default: `efficientnetv2b3`)
- `--batch_size` — default: 16

Outputs saved to `ml_models/evaluation/`:
- `metrics.json` — accuracy, AUC (macro OvR), F1-macro, QWK
- `confusion_matrix.png` — confusion matrix heatmap

## Medical Disclaimer

AI-generated output must be reviewed by a qualified ophthalmologist before any clinical decision.
