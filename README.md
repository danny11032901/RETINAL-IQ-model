# RetinaIQ

Explainable Deep Learning Framework for Automated Diabetic Retinopathy Screening.

## Quick Start

1. Copy `.env.example` to `.env`.
2. Set `DATASET_PATH` in `.env` to your APTOS-style dataset folder containing:
	- `train.csv`
	- `train_images/` (or `images/`)
3. Run:

```bash
docker-compose up --build
```

## Auto Training (No Manual Weight Steps)

If `AUTO_TRAIN_IF_MISSING=true` and the primary weight file is missing, backend startup will:

1. Read `DATASET_PATH`
2. Run `backend/ml_models/train.py`
3. Save weights to `MODEL_DIR/EFFICIENTNETV2B3_WEIGHTS`
4. Load the trained weights automatically and start inference

This means you only need to configure the dataset path once; no manual training command is required.

## Services

- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs
- MinIO Console: http://localhost:9001

## Medical Disclaimer

This output is AI-generated and must be reviewed by a qualified ophthalmologist before any clinical decision.
