# RetinaIQ — ULTIMATE MASTER PROMPT
## Explainable Deep Learning Framework for Automated Diabetic Retinopathy Screening
### Version 2.0 | For: Kiro / GitHub Copilot Workspace / Cursor AI / Any Agentic Coding Tool

> **HOW TO USE THIS PROMPT:** Feed this entire document as your project specification.
> Every section is a hard requirement. Nothing is optional unless marked [OPTIONAL].
> Build every file, every function, every pixel as described. Zero placeholders.

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION 0 — PROJECT IDENTITY & MISSION
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
Project Name:       RetinaIQ
Subtitle:           Explainable Deep Learning Framework for Automated DR Screening
Institution:        Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad
Affiliated To:      Osmania University
Supervisor:         Mr. Naveen Raja SM (Dept. of AI & Data Science)
Student IDs:        1601-22-771-080 | 160-122-771-116 | 160-122-771-131
Domain:             Medical AI | Ophthalmology | Explainable AI | Edge Deployment
Target Users:       Ophthalmologists, General Practitioners, Screening Technicians,
                    Rural Healthcare Workers
Deployment Targets: Hospital clinics, Telemedicine platforms, Rural screening vans,
                    Mobile health units, Primary care centers
Clinical Standard:  Results must meet AAO (American Academy of Ophthalmology)
                    and NICE (UK) referral guideline thresholds
```

**Mission Statement (embed in app):**
"Diabetic Retinopathy (DR) is one of the leading causes of preventable blindness worldwide,
caused by damage to retinal blood vessels from diabetes. In early stages, DR shows no
noticeable symptoms — making timely, scalable, AI-assisted screening critical to prevent
irreversible vision loss."

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION 1 — ABSOLUTE NON-NEGOTIABLE RULES
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. **ZERO bugs, ZERO placeholder functions, ZERO `TODO` comments in production code.**
2. **ZERO black-box outputs.** Every single prediction MUST be accompanied by Grad-CAM + LIME visuals.
3. **EVERY file listed** in the directory trees below must be created, complete, and runnable.
4. **The entire system** must boot end-to-end with `docker-compose up --build`. No manual steps post-boot.
5. **The React app** must build with `npm run build` without a single TypeScript error.
6. **The FastAPI backend** must start with `uvicorn app.main:app --reload` without errors.
7. **The model** must produce a valid Grad-CAM heatmap base64 PNG on every inference call — no exceptions.
8. **The adaptive preprocessor** must handle ALL of: JPEG, PNG, TIFF, BMP, DICOM, dark images, overexposed images, blurry images, low-resolution images, images with uneven illumination, and fundus images from different camera models.
9. **The PDF report** must generate and download successfully from the `/reports/{id}/download` endpoint.
10. **No hardcoded secrets.** All credentials, keys, URLs from `.env` only.
11. **Medical disclaimer** must appear on every prediction result: *"This output is AI-generated and must be reviewed by a qualified ophthalmologist before any clinical decision."*
12. **Audit trail is mandatory.** Every inference logged: user ID, timestamp, model version, preprocessing config, result, confidence.

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION 2 — COMPLETE TECHNOLOGY STACK
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 2.1 Backend Stack
```
Language:           Python 3.11+
Web Framework:      FastAPI (async-first, Pydantic v2)
ASGI Server:        Uvicorn (production) with Gunicorn worker manager
ML Primary:         TensorFlow 2.15+ with Keras 3
ML Secondary:       PyTorch 2.x (for LIME superpixel computation)
Explainability:     tf-keras-vis (Grad-CAM), LIME (lime library), SHAP
Image Processing:   OpenCV 4.x, Pillow 10+, scikit-image, pydicom (DICOM support)
CLAHE:              OpenCV cv2.createCLAHE()
Database:           PostgreSQL 15 via SQLAlchemy 2.0 (async) + Alembic migrations
Caching:            Redis 7 (inference cache, session store)
Task Queue:         Celery 5 + Redis broker (async heavy inference)
Object Storage:     MinIO (self-hosted S3-compatible) — fallback to AWS S3 via env flag
Auth:               JWT (python-jose) + OAuth2PasswordBearer + bcrypt (passlib)
Rate Limiting:      slowapi (Starlette middleware)
PDF Generation:     ReportLab 4.x
Logging:            Loguru (structured JSON to stdout)
Monitoring:         prometheus-fastapi-instrumentator (Prometheus metrics at /metrics)
Testing:            pytest + pytest-asyncio + httpx (async test client)
Linting:            ruff + mypy (strict)
```

### 2.2 Frontend Stack
```
Framework:          React 18.3 + TypeScript 5.x (strict mode: true)
Build Tool:         Vite 5.x
Styling:            Tailwind CSS 3.x + custom CSS design tokens
State:              Zustand 4.x
HTTP:               Axios 1.x with request/response interceptors
Charts:             Recharts 2.x (donut, bar, line charts)
Heatmap Canvas:     Native HTML5 Canvas API (custom component, no library)
Upload:             react-dropzone 14.x
Routing:            React Router 6.x (nested routes, protected routes)
Animation:          Framer Motion 11.x
Notifications:      react-hot-toast 2.x
Icons:              Lucide React 0.4x
PDF Preview:        react-pdf (for in-browser report preview)
Testing:            Vitest + React Testing Library + MSW (mock service worker)
```

### 2.3 Infrastructure
```
Containerization:   Docker 24+ + Docker Compose v2
Reverse Proxy:      Nginx 1.25 (SSL termination, static file serving, API proxy)
SSL:                Let's Encrypt (certbot) — auto-configured for production
CI/CD:              GitHub Actions workflow file (lint → test → build → deploy)
Secrets:            .env file (never committed); .env.example committed
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION 3 — MODEL ARCHITECTURE (HIGHEST PRIORITY)
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

> This is the most critical section. The model is the heart of the system.
> Implement every detail. No approximations.

### 3.1 Three-Tier Model Strategy
*(Derived directly from the project presentation — implement all three tiers)*

```
TIER 1 — Lightweight / Edge Deployment (Rural Healthcare, Low-Resource)
  Primary:  EfficientNet-B0
  Fallback: MobileNetV3-Large
  Use Case: Edge devices, CPU-only servers, rural screening vans
  Target:   < 500ms inference on CPU, < 50MB model size

TIER 2 — Advanced / Hospital-Grade (Primary Active Model)
  Primary:  EfficientNetV2-B3  ← MAIN PRODUCTION MODEL
  Fallback: DenseNet121
  Use Case: Hospital clinics, telemedicine, GPU-enabled servers
  Target:   < 300ms inference on GPU, highest possible accuracy

TIER 3 — Future Extension (Implement architecture, mark as experimental)
  Model:    CNN + Vision Transformer Hybrid (EfficientNet backbone + ViT attention head)
  Purpose:  Captures global retinal structural patterns
  Status:   Implement the architecture class; training pipeline ready but not default
```

**Auto-Tier Selection Logic (implement in `model_router.py`):**
```python
def select_model_tier(image: np.ndarray, server_context: ServerContext) -> ModelTier:
    """
    Automatically selects the optimal model tier based on:
    1. Image size / resolution
    2. Server hardware (GPU available vs CPU-only)
    3. User's model_preference override (if provided)
    4. Current server load (queue depth from Redis)
    """
    if server_context.gpu_available and image.size > 50000:
        return ModelTier.TIER2_EFFICIENTNETV2_B3
    elif server_context.gpu_available:
        return ModelTier.TIER2_DENSENET121
    elif image.size > 10000:
        return ModelTier.TIER1_EFFICIENTNET_B0
    else:
        return ModelTier.TIER1_MOBILENETV3
```

---

### 3.2 Primary Model: EfficientNetV2-B3 (Full Architecture)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT LAYER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Retinal Fundus Image
  - Any resolution (handled by adaptive preprocessing)
  - Any contrast, brightness, angle, noise profile
  - Formats: JPEG, PNG, TIFF, BMP, DICOM
  - Output of preprocessing: 300×300×3 float32 tensor, normalized [0,1]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE A: ADAPTIVE PREPROCESSING PIPELINE
(Runs before every inference AND during training)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1: Format Detection & DICOM Extraction
  - Detect file format via python-magic (not file extension)
  - If DICOM: extract pixel array via pydicom, handle 16-bit grayscale → 8-bit RGB
  - If standard: load via Pillow → convert to RGB

Step 2: Image Quality Assessment (runs BEFORE any processing)
  - Blur Score: cv2.Laplacian(gray, cv2.CV_64F).var()  → sharpness (higher=sharper)
  - Contrast Score: gray.std() / 255.0                  → RMS contrast [0,1]
  - Brightness Mean: gray.mean() / 255.0                → brightness [0,1]
  - SNR Score: signal-to-noise ratio via scikit-image
  - Composite Quality Score: weighted average [0–100]
  - STORE all scores — they go into the API response and DB

Step 3: Auto-Configure Preprocessing (based on quality scores above)
  - CLAHE Clip Limit:
      contrast < 0.25 → clip_limit = 5.0  (very low contrast, aggressive)
      contrast < 0.45 → clip_limit = 3.0  (low contrast, moderate boost)
      contrast < 0.65 → clip_limit = 2.0  (normal, standard)
      contrast ≥ 0.65 → clip_limit = 1.0  (good contrast, minimal)
  - CLAHE Grid Size: always (8, 8)
  - Noise Reduction:
      blur_score < 30  → Bilateral filter (d=9, sigmaColor=75, sigmaSpace=75)
      blur_score < 100 → Gaussian blur (kernel=3×3, sigma=0.8)
      blur_score ≥ 100 → No denoising (image is sharp)
  - Brightness Correction:
      brightness < 0.2 → gamma correction (gamma=0.5, brightens dark images)
      brightness > 0.8 → gamma correction (gamma=1.5, reduces overexposure)
      else             → no correction

Step 4: Green Channel Extraction Enhancement
  - Extract green channel (B, G, R = cv2.split(image))
  - Apply CLAHE to green channel (highest lesion contrast in fundus images)
  - Merge back into 3-channel RGB

Step 5: Optic Disc & Vessel Suppression (quality_score > 60 only)
  - Use circular Hough transform to detect optic disc region
  - Soft-mask disc region to reduce background bias during classification
  - Do NOT mask if quality score too low (disc may not be detectable)

Step 6: Resize
  - Smart resize to 300×300 using cv2.INTER_AREA (downscale) or cv2.INTER_CUBIC (upscale)
  - Maintain aspect ratio with center-crop, not stretch

Step 7: Normalization
  - Pixel values: [0, 255] → [0.0, 1.0] (divide by 255.0)
  - Apply EfficientNetV2 preprocessing mean/std normalization

Step 8: Training-Only Augmentation (NOT applied at inference)
  - Rotation: ±15° (reflect fill mode)
  - Horizontal Flip (p=0.5)
  - Vertical Flip (p=0.3)
  - Zoom: ±10%
  - Brightness jitter: ±20%
  - Contrast jitter: ±15%
  - Hue shift: ±0.05
  - MixUp (alpha=0.2): blends two training images + labels
  - CutMix: cuts patch from one image, pastes into another with label mix
  - GridMask: randomly masks grid regions (improves small lesion detection)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE B: EfficientNetV2-B3 BACKBONE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Load: tf.keras.applications.EfficientNetV2B3(
    weights='imagenet',        # pretrained on ImageNet-21k
    include_top=False,
    input_shape=(300, 300, 3)
  )
- Phase 1 Training: FREEZE all backbone layers
  → Train only classification head for 10 epochs
- Phase 2 Fine-tuning: UNFREEZE top 30% of backbone layers
  → Train with LR reduced by 10x for 40+ epochs
- Progressive Unfreezing Schedule:
    Epoch 0–10:   Only head trains (LR = 1e-4)
    Epoch 11–25:  Top 30% backbone unfrozen (LR = 1e-5)
    Epoch 26–50:  Top 60% backbone unfrozen (LR = 5e-6)
    Epoch 51+:    Full backbone trainable (LR = 1e-6)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE C: CUSTOM CLASSIFICATION HEAD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
backbone_output
  → GlobalAveragePooling2D()
  → BatchNormalization()
  → Dense(512, activation='relu', kernel_regularizer=l2(1e-4))
  → BatchNormalization()
  → Dropout(0.5)
  → Dense(256, activation='relu', kernel_regularizer=l2(1e-4))
  → BatchNormalization()
  → Dropout(0.4)
  → Dense(128, activation='relu')
  → Dropout(0.3)
  → Dense(5, activation='softmax')   ← 5-class output

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE D: OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Softmax probabilities for:
  Grade 0 → No DR           (class weight: auto-computed from dataset)
  Grade 1 → Mild NPDR       (class weight: auto-computed)
  Grade 2 → Moderate NPDR   (class weight: auto-computed)
  Grade 3 → Severe NPDR     (class weight: auto-computed)
  Grade 4 → Proliferative DR (class weight: auto-computed)

DR Label Mapping (5 classes from APTOS 2019):
  0 = No DR
  1 = Mild
  2 = Moderate
  3 = Severe
  4 = Proliferative DR
```

---

### 3.3 Complete Training Pipeline (`train.py`)

```python
"""
RetinaIQ Model Training Pipeline
Dataset: Kaggle APTOS 2019 Blindness Detection + EyePACS (optional)
Run: python train.py --dataset_path ./data/aptos2019 --model efficientnetv2b3 --epochs 100
"""

# IMPLEMENT ALL OF THE FOLLOWING — every item is mandatory:

# ── Data Loading ──────────────────────────────────────────────────────────────
# 1. Load train.csv from APTOS 2019 dataset
# 2. Validate all image files exist; log and skip missing files
# 3. Print class distribution before and after balancing
# 4. Stratified train/val/test split: 70% / 15% / 15%
#    Use sklearn.model_selection.StratifiedShuffleSplit
# 5. Create tf.data.Dataset pipelines with:
#    - AUTOTUNE prefetch
#    - Parallel map for preprocessing
#    - Shuffle buffer = len(train_set)
#    - Repeat for multi-epoch training

# ── Class Imbalance Handling ───────────────────────────────────────────────────
# 6. Compute class weights:
#    from sklearn.utils.class_weight import compute_class_weight
#    class_weights = compute_class_weight('balanced', classes=..., y=...)
# 7. Implement Focal Loss:
#    class FocalLoss(tf.keras.losses.Loss):
#        def __init__(self, gamma=2.0, alpha=0.25):
#            → reduces loss weight for well-classified easy samples
#            → forces model to focus on hard-to-classify mild/moderate cases
# 8. Implement SMOTE-like oversampling for Grade 1 (Mild) — most confused class

# ── Model Construction ────────────────────────────────────────────────────────
# 9.  Build EfficientNetV2-B3 with full classification head (see Section 3.2)
# 10. Build MobileNetV3-Large with same head (Tier 1 fallback)
# 11. Build EfficientNet-B0 with same head (Tier 1 primary)
# 12. [EXPERIMENTAL] Build CNN+ViT hybrid (Tier 3):
#     EfficientNet-B3 backbone → feature map → ViT patch attention → classification head

# ── Training Strategy ─────────────────────────────────────────────────────────
# 13. Optimizer: Adam(learning_rate=1e-4, weight_decay=1e-4)
# 14. LR Scheduler: CosineAnnealingLR (T_max=50, eta_min=1e-7)
#     Implement as tf.keras.callbacks.LearningRateScheduler
# 15. Warmup: linear LR warmup for first 5 epochs (0 → 1e-4)
# 16. Progressive Unfreezing: implement as custom Keras callback
#     class ProgressiveUnfreezeCallback(tf.keras.callbacks.Callback):
#         → unfreezes layers according to epoch schedule in Section 3.2
# 17. Augmentation: apply MixUp and CutMix as custom tf.data map functions
# 18. EarlyStopping: monitor='val_auc', patience=15, restore_best_weights=True
# 19. ModelCheckpoint: save best val_auc model as SavedModel + .h5
# 20. ReduceLROnPlateau: monitor='val_loss', factor=0.5, patience=7

# ── k-Fold Cross-Validation ────────────────────────────────────────────────────
# 21. Implement 5-fold stratified cross-validation
#     Report mean ± std for: Accuracy, AUC, F1-macro, QWK across all folds
#     Save best fold model as production model

# ── Evaluation ────────────────────────────────────────────────────────────────
# 22. Final evaluation on held-out test set:
#     - Accuracy (overall)
#     - AUC-ROC (macro OvR)
#     - F1-Score (macro)
#     - Quadratic Weighted Kappa (QWK) — primary clinical metric
#     - Per-class precision, recall, F1
#     - Confusion matrix (saved as PNG)
#     - ROC curve per class (saved as PNG)
# 23. Calibration check: plot reliability diagram (predicted probability vs actual)
# 24. Grad-CAM validation: generate Grad-CAM for 20 test samples, save as grid PNG

# ── Export ─────────────────────────────────────────────────────────────────────
# 25. Save final model:
#     - TensorFlow SavedModel format: ./ml_models/retinaiq_efficientnetv2b3/
#     - Keras .h5: ./ml_models/retinaiq_efficientnetv2b3.h5
#     - TFLite quantized: ./ml_models/retinaiq_lite.tflite (for edge deployment)
# 26. Save training history as JSON (loss, accuracy, AUC, LR per epoch)
# 27. Save all evaluation plots to ./ml_models/evaluation_plots/
# 28. Write model card: ./ml_models/MODEL_CARD.md
#     (dataset, metrics, training config, intended use, limitations)
```

---

### 3.4 Training Hyperparameters (Final Reference Table)

```
┌─────────────────────────────────┬──────────────────────────────────────┐
│ Parameter                       │ Value                                │
├─────────────────────────────────┼──────────────────────────────────────┤
│ Optimizer                       │ Adam (weight_decay=1e-4)             │
│ Initial LR (frozen head)        │ 1e-4                                 │
│ LR (top-30% unfrozen)           │ 1e-5                                 │
│ LR (top-60% unfrozen)           │ 5e-6                                 │
│ LR (full model)                 │ 1e-6                                 │
│ LR Schedule                     │ Cosine Annealing + Warmup (5 epoch)  │
│ Epochs (max)                    │ 100                                  │
│ EarlyStopping patience          │ 15 (monitor: val_auc)                │
│ Batch Size                      │ 32                                   │
│ Loss Function                   │ Focal Loss (γ=2.0, α=0.25)          │
│ Class Weights                   │ Auto (sklearn balanced)              │
│ Dropout                         │ 0.5 → 0.4 → 0.3 (head layers)      │
│ L2 Regularization               │ 1e-4 (Dense layers)                 │
│ Validation Split                │ 0.15 (stratified)                   │
│ Input Resolution                │ 300×300×3                            │
│ Augmentation (train)            │ Rotation, Flip, Zoom, MixUp, CutMix │
│ Cross-Validation                │ 5-fold stratified                    │
│ Primary Metric                  │ QWK (Quadratic Weighted Kappa)       │
│ Secondary Metrics               │ AUC-ROC macro, F1 macro, Accuracy    │
│ Dataset                         │ APTOS 2019 + EyePACS                 │
│ Train/Val/Test Split            │ 70% / 15% / 15%                      │
│ GPU Environment                 │ CUDA-enabled (auto-detected)         │
└─────────────────────────────────┴──────────────────────────────────────┘
```

---

### 3.5 Complete Explainability Pipeline

#### 3.5.1 Grad-CAM (Primary — IMPLEMENT FULLY)
```python
# File: backend/app/services/ml/gradcam.py

import tensorflow as tf
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

class GradCAMExplainer:
    """
    Gradient-weighted Class Activation Mapping.
    Generates lesion heatmaps highlighting:
      - Microaneurysms (small red dots)
      - Hemorrhages (blot/flame-shaped)
      - Exudates (bright yellow deposits)
      - Neovascularization (new vessel growth in Grade 4)
    Used AFTER prediction, at inference time.
    Overlays colored heatmap on original retinal image.
    """

    def __init__(self, model: tf.keras.Model, last_conv_layer_name: str):
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        # For EfficientNetV2-B3: last_conv_layer_name = 'top_conv'

    def compute_heatmap(
        self,
        image_array: np.ndarray,  # shape: (1, 300, 300, 3)
        pred_class_idx: int,
        sensitivity: float = 1.0
    ) -> np.ndarray:
        """Returns heatmap array [0,1], shape (300, 300)"""
        grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[
                self.model.get_layer(self.last_conv_layer_name).output,
                self.model.output
            ]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_array, training=False)
            loss = predictions[:, pred_class_idx]

        grads = tape.gradient(loss, conv_outputs)
        # Importance weights via global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        # Weight feature maps by gradient importance
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        # ReLU: only positive activations matter
        heatmap = np.maximum(heatmap, 0)
        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        # Apply sensitivity scaling
        heatmap = np.power(heatmap, 1.0 / sensitivity)
        return heatmap

    def overlay_on_image(
        self,
        original_image: np.ndarray,  # shape: (H, W, 3), uint8
        heatmap: np.ndarray,          # shape: (h, w), float [0,1]
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """Returns BGR overlay image as uint8 numpy array"""
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        # Convert original to BGR for OpenCV
        original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay

    def generate_base64_overlay(
        self,
        original_image: np.ndarray,
        pred_class_idx: int,
        preprocessed_array: np.ndarray
    ) -> dict:
        """
        Returns dict with:
          - heatmap_only_b64: pure heatmap (JET colormap) as base64 PNG
          - overlay_b64: original + heatmap overlay as base64 PNG
          - original_b64: original image as base64 PNG (for side-by-side UI)
          - attention_regions: list of (x, y, w, h) bboxes of top-3 attention regions
        """
        heatmap = self.compute_heatmap(preprocessed_array, pred_class_idx)

        # Generate all three views
        overlay = self.overlay_on_image(original_image, heatmap)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * cv2.resize(
            heatmap, (original_image.shape[1], original_image.shape[0])
        )), cv2.COLORMAP_JET)

        def to_base64(img_bgr: np.ndarray) -> str:
            _, buffer = cv2.imencode('.png', img_bgr)
            return 'data:image/png;base64,' + base64.b64encode(buffer).decode()

        # Detect top attention regions (for UI bounding box overlay)
        attention_regions = self._extract_attention_regions(heatmap, original_image.shape)

        return {
            'overlay_b64': to_base64(overlay),
            'heatmap_only_b64': to_base64(heatmap_colored),
            'original_b64': to_base64(cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)),
            'attention_regions': attention_regions
        }

    def _extract_attention_regions(self, heatmap: np.ndarray, img_shape: tuple) -> list:
        """Find top-3 bounding boxes of high-attention regions"""
        threshold = 0.6
        binary = (cv2.resize(heatmap, (img_shape[1], img_shape[0])) > threshold).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = [cv2.boundingRect(c) for c in contours]
        regions.sort(key=lambda r: r[2]*r[3], reverse=True)  # sort by area
        return [{'x': x, 'y': y, 'w': w, 'h': h} for x, y, w, h in regions[:3]]
```

#### 3.5.2 LIME (Secondary — IMPLEMENT FULLY)
```python
# File: backend/app/services/ml/lime_explainer.py

from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np

class LIMEExplainer:
    """
    Local Interpretable Model-Agnostic Explanations.
    Segments retinal image into superpixels.
    Shows WHICH superpixel regions contributed POSITIVELY (green)
    and NEGATIVELY (red) to the predicted DR grade.
    More granular than Grad-CAM for clinical review.
    """

    def __init__(self, predict_fn):
        self.explainer = lime_image.LimeImageExplainer()
        self.predict_fn = predict_fn  # model.predict wrapper returning numpy array

    def explain(
        self,
        image: np.ndarray,       # (H, W, 3) float32 [0,1]
        pred_class_idx: int,
        num_samples: int = 1000,
        num_features: int = 10,
        hide_rest: bool = False
    ) -> str:
        """Returns base64 PNG of LIME superpixel explanation"""
        explanation = self.explainer.explain_instance(
            image,
            self.predict_fn,
            top_labels=5,
            hide_color=0,
            num_samples=num_samples
        )
        temp, mask = explanation.get_image_and_mask(
            pred_class_idx,
            positive_only=False,
            num_features=num_features,
            hide_rest=hide_rest
        )
        # Mark boundaries between superpixels
        marked = mark_boundaries(temp, mask)
        marked_uint8 = (marked * 255).astype(np.uint8)
        # Return as base64
        img_pil = Image.fromarray(marked_uint8)
        buffer = BytesIO()
        img_pil.save(buffer, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
```

#### 3.5.3 SHAP (Tertiary — for metadata/tabular explanation)
```python
# File: backend/app/services/ml/shap_explainer.py
# Explain which patient metadata features (HbA1c, diabetes duration) increase DR risk
# Use shap.DeepExplainer or shap.GradientExplainer
# Output: feature importance bar chart as base64 PNG
# Only shown if patient metadata is provided with the scan
```

---

### 3.6 Model Loader (Singleton Pattern)
```python
# File: backend/app/services/ml/model_loader.py

class ModelRegistry:
    """
    Loads all model tiers ONCE at application startup.
    Provides thread-safe singleton access during inference.
    Auto-detects available GPU and logs hardware context.
    """
    _instance = None
    _models = {}

    @classmethod
    def get_instance(cls) -> 'ModelRegistry':
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_all_models()
        return cls._instance

    def _load_all_models(self):
        # Load Tier 1: EfficientNet-B0
        # Load Tier 1: MobileNetV3-Large
        # Load Tier 2: EfficientNetV2-B3  ← primary
        # Load Tier 2: DenseNet121
        # [EXPERIMENTAL] Load Tier 3: CNN+ViT hybrid
        # Log GPU availability, VRAM, inference benchmark time
        # Store GradCAMExplainer and LIMEExplainer per model
        pass

    def get_model(self, tier: ModelTier) -> tf.keras.Model:
        return self._models[tier]
```

---

### 3.7 Full Inference Pipeline
```python
# File: backend/app/services/ml/inference.py

async def run_full_inference(
    image_bytes: bytes,
    patient_id: Optional[str],
    model_preference: str,
    user_id: str,
    db: AsyncSession,
    storage: StorageService
) -> PredictionResponse:
    """
    Complete inference pipeline. Steps:
    1.  Decode image bytes → detect format → convert to RGB numpy array
    2.  Run image quality assessment → compute quality scores
    3.  Auto-configure preprocessing parameters
    4.  Apply adaptive preprocessing pipeline
    5.  Select model tier (auto or user preference)
    6.  Run model inference → get 5-class probabilities
    7.  Determine predicted grade (argmax)
    8.  Compute confidence (max probability)
    9.  Generate Grad-CAM overlay (all 3 views)
    10. Generate LIME explanation (async, may take 2–5s)
    11. Map grade → clinical recommendation text
    12. Upload: original image + Grad-CAM to MinIO/S3
    13. Write prediction record to PostgreSQL
    14. Write to inference audit log
    15. Cache result in Redis (key: hash of image bytes, TTL=1hr)
    16. Return full PredictionResponse
    """
    start_time = time.perf_counter()
    # ... full implementation
    processing_ms = int((time.perf_counter() - start_time) * 1000)
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION 4 — BACKEND COMPLETE STRUCTURE
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
backend/
├── app/
│   ├── main.py                         # FastAPI app: CORS, middleware, routers, startup events
│   ├── core/
│   │   ├── config.py                   # Pydantic BaseSettings (all from .env)
│   │   ├── security.py                 # JWT (create, verify, decode), bcrypt hashing
│   │   ├── logging.py                  # Loguru JSON logger, request ID middleware
│   │   └── exceptions.py               # Custom exception classes + handlers
│   ├── api/
│   │   └── v1/
│   │       ├── router.py               # Aggregates all v1 routes
│   │       └── routes/
│   │           ├── auth.py             # POST /auth/login, /register, /refresh, /logout
│   │           ├── predict.py          # POST /predict/analyze (multipart upload)
│   │           │                       # GET  /predict/{prediction_id}
│   │           ├── patients.py         # GET/POST/PUT/DELETE /patients
│   │           │                       # GET /patients/{id}/history
│   │           ├── reports.py          # GET /reports/{prediction_id}/download (PDF)
│   │           │                       # GET /reports/patient/{patient_id}
│   │           ├── dashboard.py        # GET /dashboard/stats
│   │           │                       # GET /dashboard/recent-scans
│   │           │                       # GET /dashboard/grade-distribution
│   │           ├── annotations.py      # POST /predict/{id}/annotate (doctor notes)
│   │           └── health.py           # GET /health (liveness), GET /ready (readiness)
│   ├── models/
│   │   ├── db/
│   │   │   ├── user.py                 # SQLAlchemy User ORM model
│   │   │   ├── patient.py              # SQLAlchemy Patient ORM model
│   │   │   ├── prediction.py           # SQLAlchemy Prediction ORM model
│   │   │   └── audit_log.py            # SQLAlchemy AuditLog ORM model
│   │   └── schemas/
│   │       ├── auth.py                 # LoginRequest, RegisterRequest, TokenResponse
│   │       ├── predict.py              # PredictionResponse (full), PredictionSummary
│   │       ├── patient.py              # PatientCreate, PatientUpdate, PatientOut
│   │       ├── dashboard.py            # DashboardStats, RecentScan, GradeDistribution
│   │       └── annotation.py           # DoctorAnnotation
│   ├── services/
│   │   ├── ml/
│   │   │   ├── model_loader.py         # ModelRegistry singleton (all tiers)
│   │   │   ├── model_router.py         # Tier selection logic
│   │   │   ├── preprocessor.py         # AdaptivePreprocessor class (full pipeline)
│   │   │   ├── quality_check.py        # ImageQualityAssessor (blur, contrast, SNR)
│   │   │   ├── inference.py            # run_full_inference() orchestrator
│   │   │   ├── gradcam.py              # GradCAMExplainer class
│   │   │   ├── lime_explainer.py       # LIMEExplainer class
│   │   │   └── shap_explainer.py       # SHAPExplainer (metadata features)
│   │   ├── patient_service.py          # CRUD operations for patients
│   │   ├── report_service.py           # PDF generation via ReportLab
│   │   ├── storage_service.py          # MinIO/S3 upload, download, presigned URLs
│   │   └── recommendation_engine.py   # Grade → clinical text recommendation
│   ├── tasks/
│   │   ├── celery_app.py               # Celery app config
│   │   └── inference_task.py           # @celery_app.task: async_analyze_image()
│   ├── db/
│   │   ├── session.py                  # AsyncSession factory, get_db dependency
│   │   └── migrations/                 # Alembic: versions/, env.py, alembic.ini
│   └── middleware/
│       ├── request_id.py               # Attach X-Request-ID to every request/response
│       └── timing.py                   # Log request processing time
├── ml_models/
│   ├── train.py                        # Full training script (see Section 3.3)
│   ├── evaluate.py                     # Standalone evaluation on test set
│   ├── convert_tflite.py               # Export to TFLite quantized
│   ├── MODEL_CARD.md                   # Model documentation
│   └── [weights are gitignored]        # .h5 and SavedModel dirs in .gitignore
├── tests/
│   ├── conftest.py                     # Pytest fixtures: test DB, test client, mock models
│   ├── test_quality_check.py           # 10+ test images (blurry, dark, overexposed, normal)
│   ├── test_preprocessor.py            # Adaptive preprocessing for all image types
│   ├── test_inference.py               # End-to-end: bytes → PredictionResponse
│   ├── test_gradcam.py                 # Grad-CAM output validation (shape, dtype, base64)
│   ├── test_lime.py                    # LIME output validation
│   ├── test_auth.py                    # JWT: create, verify, expire, role check
│   ├── test_api_predict.py             # Integration: POST /predict/analyze
│   ├── test_api_patients.py            # CRUD patient API tests
│   ├── test_api_reports.py             # PDF download endpoint test
│   └── test_recommendation_engine.py  # Grade → recommendation text mapping
├── Dockerfile
├── requirements.txt                    # All pinned dependencies
├── .env.example                        # All required env vars with descriptions
└── pyproject.toml                      # ruff + mypy config
```

---

### 4.1 Complete API Specification

#### POST `/api/v1/predict/analyze`
```
Request (multipart/form-data):
  image:            File    REQUIRED — JPG/PNG/TIFF/BMP/DICOM, max 20MB
  patient_id:       UUID    OPTIONAL
  model_preference: string  OPTIONAL: 'auto' | 'tier1' | 'tier2' | 'efficientnetv2b3'
                                     | 'mobilenetv3' | 'efficientnetb0' | 'densenet121'

Response 200 (JSON):
{
  "prediction_id":        "550e8400-e29b-41d4-a716-446655440000",
  "patient_id":           "uuid | null",
  "analyzed_by":          "user_uuid",
  "timestamp":            "2025-02-28T10:50:37Z",
  "model_used":           "EfficientNetV2-B3",
  "model_tier":           2,
  "image_quality": {
    "blur_score":         142.7,
    "contrast_score":     0.58,
    "brightness_mean":    0.43,
    "snr_score":          28.4,
    "composite_score":    74.2,
    "quality_label":      "Good",     // "Poor" | "Acceptable" | "Good"
    "warning":            null        // or "Low quality image ..."
  },
  "preprocessing_applied": {
    "clahe_clip_limit":   2.0,
    "clahe_grid_size":    "8x8",
    "denoise_method":     "gaussian",
    "denoise_strength":   "moderate",
    "brightness_correction": false,
    "optic_disc_masked":  true,
    "resize":             "300x300",
    "augmentations":      []
  },
  "dr_grade":             2,
  "dr_label":             "Moderate Diabetic Retinopathy",
  "dr_label_short":       "Moderate NPDR",
  "confidence":           0.874,
  "class_probabilities": {
    "0_no_dr":            0.021,
    "1_mild":             0.048,
    "2_moderate":         0.874,
    "3_severe":           0.041,
    "4_proliferative":    0.016
  },
  "explainability": {
    "gradcam_overlay_b64":    "data:image/png;base64,...",
    "gradcam_heatmap_b64":    "data:image/png;base64,...",
    "gradcam_original_b64":   "data:image/png;base64,...",
    "attention_regions": [
      {"x": 120, "y": 90, "w": 45, "h": 38},
      {"x": 210, "y": 155, "w": 28, "h": 22}
    ],
    "lime_explanation_b64":   "data:image/png;base64,...",
    "shap_available":         false   // true only if patient metadata provided
  },
  "clinical": {
    "recommendation":     "Moderate NPDR detected. Refer to ophthalmologist within 1 month.",
    "urgency":            "moderate",   // "none" | "routine" | "moderate" | "urgent" | "emergency"
    "referral_guideline": "AAO 2023",
    "follow_up_months":   1,
    "disclaimer":         "This report is AI-generated and must be reviewed by a qualified ophthalmologist before any clinical decision."
  },
  "performance": {
    "preprocessing_ms":   42,
    "inference_ms":       218,
    "gradcam_ms":         67,
    "lime_ms":            1840,
    "total_ms":           2167
  }
}

Errors:
  400: Image format not supported / file too large / corrupted
  401: Unauthorized (no/invalid JWT)
  413: File size exceeds 20MB limit
  422: Missing required field
  429: Rate limit exceeded (30/min per user)
  500: Model inference failure (logged + alerted)
```

#### GET `/api/v1/reports/{prediction_id}/download`
```
Response: application/pdf
Content-Disposition: attachment; filename="retinaiq_report_{prediction_id}.pdf"

PDF Contents (see Section 4.2):
  Page 1: Patient info + scan details + DR grade
  Page 2: Original image + Grad-CAM overlay side by side
  Page 3: Class probabilities table + clinical recommendation + disclaimer
```

#### GET `/api/v1/dashboard/stats`
```
Response:
{
  "total_scans":          1247,
  "scans_this_week":      43,
  "scans_today":          7,
  "average_confidence":   0.831,
  "grade_distribution":   {"0": 412, "1": 198, "2": 367, "3": 156, "4": 114},
  "model_status": {
    "tier2_loaded":       true,
    "tier1_loaded":       true,
    "gpu_available":      true,
    "avg_inference_ms":   284
  }
}
```

---

### 4.2 Clinical Recommendation Engine
```python
# File: backend/app/services/recommendation_engine.py

CLINICAL_RECOMMENDATIONS = {
    0: {
        "text": "No diabetic retinopathy detected. Continue routine annual screening.",
        "urgency": "none",
        "follow_up_months": 12,
        "guideline": "AAO 2023 — annual screening for all diabetic patients"
    },
    1: {
        "text": "Mild non-proliferative diabetic retinopathy (NPDR) detected. "
                "Optimize glycemic control and blood pressure. Repeat screening in 12 months.",
        "urgency": "routine",
        "follow_up_months": 12,
        "guideline": "AAO 2023 — mild NPDR: annual follow-up"
    },
    2: {
        "text": "Moderate non-proliferative diabetic retinopathy (NPDR) detected. "
                "Refer to ophthalmologist within 1 month for comprehensive dilated examination.",
        "urgency": "moderate",
        "follow_up_months": 1,
        "guideline": "AAO 2023 — moderate NPDR: ophthalmology referral within 1 month"
    },
    3: {
        "text": "Severe non-proliferative diabetic retinopathy (NPDR) detected. "
                "URGENT referral to ophthalmologist required within 1 week. "
                "High risk of progression to proliferative DR.",
        "urgency": "urgent",
        "follow_up_months": 0.25,  # ~1 week
        "guideline": "AAO 2023 / NICE NG28 — severe NPDR: urgent ophthalmology referral"
    },
    4: {
        "text": "Proliferative diabetic retinopathy (PDR) detected. "
                "IMMEDIATE specialist referral required. Risk of severe vision loss. "
                "Do not delay — contact ophthalmology department today.",
        "urgency": "emergency",
        "follow_up_months": 0,
        "guideline": "AAO 2023 / NICE NG28 — PDR: same-day or next-day referral"
    }
}
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION 5 — UI/UX DESIGN SYSTEM (HIGHEST DETAIL)
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

> The UI must be UNFORGETTABLE. A clinician who sees this for the first time
> must immediately feel: "This was built by people who understand medicine AND design."
> Reference: Viz.ai, IDx-DR, Butterfly Network, Tempus — but exceed them all.

### 5.1 Design Identity

```
Theme:          "Bioluminescent Precision"
Concept:        The deep-ocean bioluminescence metaphor — data glowing in dark space.
                Like looking through a fundus camera in a darkened clinical room.
                Information is precious, revealed carefully, with surgical precision.

Tone:           Refined clinical minimalism with high-impact data moments.
                NOT sterile/cold. NOT playful. NOT corporate.
                EXACTLY: confident, trustworthy, sophisticated, alive.

Differentiator: The retinal scan "pulse" animation that runs as a background motif
                throughout the app — a slow sine-wave scan line that sweeps across
                visualizations, echoing real fundus camera behavior.
                This is the ONE thing every user will remember.
```

### 5.2 Complete Design Token System
```css
/* File: frontend/src/design-system/tokens.css */

@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=DM+Sans:ital,wght@0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  /* ── BACKGROUND LAYERS ───────────────────────────────── */
  --bg-void:          #060A14;   /* Deepest: page background */
  --bg-base:          #0A0F1E;   /* App shell background */
  --bg-surface:       #0F1829;   /* Sidebar, nav backgrounds */
  --bg-card:          #152038;   /* Card/panel surfaces */
  --bg-elevated:      #1C2B47;   /* Modal, dropdown, tooltip */
  --bg-input:         #111E33;   /* Input field background */
  --bg-hover:         #1A2B42;   /* Hover state backgrounds */

  /* ── BIOLUMINESCENT ACCENT PALETTE ──────────────────── */
  --accent-primary:   #00D4AA;   /* Teal-green: primary CTAs, active states */
  --accent-bright:    #00F5C8;   /* Brighter: hover highlights */
  --accent-dim:       #00A882;   /* Dimmer: secondary emphasis */
  --accent-glow:      rgba(0, 212, 170, 0.12);  /* Ambient glow */
  --accent-glow-strong: rgba(0, 212, 170, 0.25); /* Strong glow (active) */

  /* ── SEVERITY SCALE ─────────────────────────────────── */
  --grade-0-bg:       rgba(34, 197, 94, 0.12);
  --grade-0-text:     #22C55E;    /* No DR — green */
  --grade-0-border:   rgba(34, 197, 94, 0.4);

  --grade-1-bg:       rgba(132, 204, 22, 0.12);
  --grade-1-text:     #84CC16;    /* Mild — lime */
  --grade-1-border:   rgba(132, 204, 22, 0.4);

  --grade-2-bg:       rgba(245, 158, 11, 0.12);
  --grade-2-text:     #F59E0B;    /* Moderate — amber */
  --grade-2-border:   rgba(245, 158, 11, 0.4);

  --grade-3-bg:       rgba(249, 115, 22, 0.12);
  --grade-3-text:     #F97316;    /* Severe — orange */
  --grade-3-border:   rgba(249, 115, 22, 0.4);

  --grade-4-bg:       rgba(239, 68, 68, 0.12);
  --grade-4-text:     #EF4444;    /* Proliferative — red */
  --grade-4-border:   rgba(239, 68, 68, 0.5);

  /* ── TEXT SCALE ─────────────────────────────────────── */
  --text-primary:     #EEF2FF;    /* High-contrast body text */
  --text-secondary:   #94A3B8;    /* Supporting text */
  --text-tertiary:    #64748B;    /* Labels, captions */
  --text-disabled:    #334155;    /* Disabled state */
  --text-accent:      #00D4AA;    /* Accent text */
  --text-inverse:     #060A14;    /* Text on light backgrounds */

  /* ── BORDERS ────────────────────────────────────────── */
  --border-subtle:    rgba(255,255,255,0.06);
  --border-default:   rgba(255,255,255,0.10);
  --border-strong:    rgba(255,255,255,0.18);
  --border-accent:    rgba(0, 212, 170, 0.35);
  --border-focus:     rgba(0, 212, 170, 0.7);

  /* ── TYPOGRAPHY ─────────────────────────────────────── */
  --font-display:     'Sora', sans-serif;
  --font-body:        'DM Sans', sans-serif;
  --font-mono:        'JetBrains Mono', monospace;

  /* Type Scale */
  --text-xs:    11px;  --leading-xs:  16px;
  --text-sm:    13px;  --leading-sm:  20px;
  --text-base:  15px;  --leading-base: 24px;
  --text-lg:    17px;  --leading-lg:  26px;
  --text-xl:    20px;  --leading-xl:  28px;
  --text-2xl:   24px;  --leading-2xl: 32px;
  --text-3xl:   30px;  --leading-3xl: 38px;
  --text-4xl:   38px;  --leading-4xl: 46px;
  --text-5xl:   48px;  --leading-5xl: 56px;
  --text-hero:  72px;  --leading-hero: 78px;

  /* ── SPACING (8pt grid) ─────────────────────────────── */
  --s-1: 4px;   --s-2: 8px;   --s-3: 12px;  --s-4: 16px;
  --s-5: 20px;  --s-6: 24px;  --s-7: 28px;  --s-8: 32px;
  --s-10: 40px; --s-12: 48px; --s-16: 64px; --s-20: 80px;
  --s-24: 96px; --s-32: 128px;

  /* ── BORDER RADIUS ──────────────────────────────────── */
  --r-xs:   4px;   --r-sm:  8px;
  --r-md:   12px;  --r-lg:  16px;
  --r-xl:   20px;  --r-2xl: 28px;
  --r-full: 9999px;

  /* ── SHADOWS ────────────────────────────────────────── */
  --shadow-sm:     0 1px 3px rgba(0,0,0,0.5), 0 1px 2px rgba(0,0,0,0.4);
  --shadow-md:     0 4px 16px rgba(0,0,0,0.5), 0 2px 6px rgba(0,0,0,0.3);
  --shadow-lg:     0 8px 32px rgba(0,0,0,0.6), 0 4px 12px rgba(0,0,0,0.4);
  --shadow-xl:     0 16px 56px rgba(0,0,0,0.7);
  --shadow-glow:   0 0 24px rgba(0, 212, 170, 0.18), 0 0 48px rgba(0, 212, 170, 0.08);
  --shadow-glow-lg: 0 0 48px rgba(0, 212, 170, 0.25), 0 0 96px rgba(0, 212, 170, 0.12);
  --shadow-card:   0 2px 8px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);

  /* ── TRANSITIONS ────────────────────────────────────── */
  --ease-out:   cubic-bezier(0.16, 1, 0.3, 1);
  --ease-in:    cubic-bezier(0.7, 0, 0.84, 0);
  --ease-inout: cubic-bezier(0.87, 0, 0.13, 1);
  --duration-fast:   120ms;
  --duration-base:   200ms;
  --duration-slow:   350ms;
  --duration-slower: 500ms;
}
```

---

### 5.3 Page-by-Page Implementation Specifications

#### 5.3.1 Landing Page (`Landing.tsx`)

```
HERO SECTION (100vh):
  Background: --bg-void
  Animated SVG: Slow rotating concentric circles (retinal cross-section visual)
    - 5 rings, each slightly different stroke dash offset
    - Outermost ring: --accent-dim, 0.3 opacity
    - Inner rings: progressively brighter toward center
    - Central element: stylized eye/retina SVG icon, pulsing glow animation
    - Sweep line: thin horizontal line that scans top-to-bottom (4s loop, ease-in-out)
    - All animation: CSS keyframes only, no JS needed

  Headline: "See What Others Miss."
    - Font: Sora 700, 72px, letter-spacing: -0.04em
    - Color: --text-primary
    - "Miss." has color: --accent-primary
    - Framer Motion: slide up + fade in, delay 200ms

  Sub-headline (below, 24px gap):
    "AI-powered diabetic retinopathy screening with clinical-grade explainability."
    - Font: DM Sans 400, 20px, --text-secondary
    - Framer Motion: slide up + fade in, delay 350ms

  CTA Row (48px gap below sub-headline):
    Button 1: "Begin Screening" — filled teal, Sora 600, 17px
      - Background: --accent-primary
      - Color: --text-inverse
      - Border-radius: --r-full
      - Padding: 14px 36px
      - Hover: --accent-bright, box-shadow: --shadow-glow
      - Arrow icon (Lucide ArrowRight) slides in from left on hover

    Button 2: "Read the Research" — ghost border
      - Border: 1px solid --border-accent
      - Color: --accent-primary
      - Same size as Button 1
      - Hover: bg --accent-glow

  Framer Motion: staggered children, delay 500ms

STATS STRIP (below hero, dark band):
  4 stats in a row, separated by subtle vertical dividers:
    "93%+ AUC" | "<300ms Inference" | "5-Grade Classification" | "Grad-CAM Verified"
  Each stat:
    - Number: JetBrains Mono 600, 28px, --accent-primary
    - Label: DM Sans 400, 13px, --text-tertiary, uppercase, letter-spacing: 0.08em
  Background: --bg-surface with 1px border top/bottom --border-subtle

FEATURE CARDS (3-column grid, 96px top padding):
  Card 1 — "Precision Diagnosis"
    Icon: Target-like SVG (retina bullseye)
    Heading: "93%+ AUC on APTOS 2019"
    Body: "EfficientNetV2-B3 trained on 5-class DR grading..."

  Card 2 — "Instant Explainability"
    Icon: Eye with heatmap layers
    Heading: "Grad-CAM + LIME Fusion"
    Body: "Every prediction shows exactly which retinal regions..."

  Card 3 — "Edge-Ready"
    Icon: Signal/wifi-style radiating circles
    Heading: "3-Tier Model Architecture"
    Body: "From rural clinic smartphones to hospital GPU servers..."

  Card style:
    Background: --bg-card
    Border: 1px solid --border-subtle
    Border-radius: --r-xl
    Padding: 40px 32px
    Top: glowing accent line (2px, gradient: transparent → --accent-primary → transparent)
    Hover: border-color --border-accent, translateY(-4px), --shadow-glow

METHODOLOGY SECTION:
  Show the system flow diagram (from the presentation):
    Patient → Retinal Fundus Image → ML Model on the Edge →
    Predictions available on dashboard → Alert Patients / Doctor
    "Send appointment request based on our analysis + doctors call"
  Implement as animated SVG flow with connecting arrows that draw on scroll

HOW IT WORKS (3-step cards, numbered):
  1. Upload Retinal Image
  2. AI Analyzes & Explains
  3. Clinician Reviews & Acts

LITERATURE BASIS SECTION:
  "Built on peer-reviewed research" heading
  Reference cards for the 6 studies from the paper:
    Show as horizontal scroll of citation cards with methodology + accuracy

FOOTER:
  CBIT logo placeholder + project details
  Links: Dashboard, About, Research Paper, Contact
  Disclaimer text (small, --text-tertiary)
```

---

#### 5.3.2 Login Page (`Login.tsx`)

```
Full-screen split layout:
  Left 55%: The hero animation from landing (retinal sweep) — subtle, scaled down
  Right 45%: Login card

Login Card:
  Background: --bg-card
  Border: 1px solid --border-subtle
  Border-radius: --r-2xl
  Padding: 48px
  Shadow: --shadow-xl

  "Welcome back" — Sora 600, 28px
  "RetinaIQ Clinical Portal" — DM Sans 400, 14px, --text-secondary

  Role selector (3 pills: Admin | Doctor | Technician)
    - Selected: --accent-primary bg, --text-inverse
    - Unselected: --bg-input border --border-default
    - Changes which dashboard view loads after login

  Email input, Password input (custom styled, NOT browser default)
  "Forgot password?" link — --accent-primary, right-aligned
  "Sign In" button — full width, teal filled
  Divider: "Don't have an account? Contact your administrator."
```

---

#### 5.3.3 Dashboard Page (`Dashboard.tsx`)

```
LAYOUT:
  Left sidebar (240px fixed): Sidebar.tsx
  Main content: fluid, padding 32px

SIDEBAR:
  Top: RetinaIQ logo (teal eye icon + wordmark)
  Nav items (with Lucide icons):
    - Dashboard (LayoutDashboard)
    - Analyze (ScanEye)
    - Patients (Users)
    - Reports (FileText)
    - Settings (Settings)
  Bottom: User avatar, name, role badge, logout button
  Active item: --accent-primary left border (3px), bg --bg-hover

STATS ROW (4 cards, equal width):
  Card 1: Total Scans — large number (JetBrains Mono 700, 42px), sparkline below
  Card 2: This Week — number + trend arrow (up/down, colored)
  Card 3: Avg Confidence — percentage with circular progress ring
  Card 4: Pending Review — count (predictions with is_reviewed=false)
  Each card: --bg-card, --border-subtle, --r-xl, --shadow-card

GRADE DISTRIBUTION CHART:
  Recharts DonutChart, 300px diameter
  Colors match severity scale exactly
  Center: total scan count
  Legend: grade labels with counts and percentages
  Right of chart: ranked severity list with colored bars

WEEKLY VOLUME CHART:
  Recharts BarChart, 7-day rolling window
  Bars: --accent-primary fill, rounded tops
  Tooltip: custom styled matching design system
  X-axis: Mon–Sun labels, --text-tertiary

RECENT SCANS TABLE:
  Columns: Patient | Grade | Confidence | Model | Time | Actions
  Grade column: colored badge matching severity scale
  Confidence column: mini progress bar
  Actions: "View" (opens prediction) | "Download PDF"
  Hover row: bg --bg-hover
  Pagination: 10 items per page

SYSTEM STATUS PANEL (bottom right):
  "Model Status" card
  Indicators (green/amber/red dots):
    - Tier 2 Model: Loaded / Loading / Error
    - Tier 1 Model: Loaded / Loading / Error
    - GPU: Available / CPU-only
    - Database: Connected / Disconnected
    - Redis: Connected / Disconnected
  Avg inference time: JetBrains Mono, --accent-primary
```

---

#### 5.3.4 Analyze Page (`Analyze.tsx`) — THE CROWN JEWEL

```
LAYOUT: Two-column (50/50 desktop), single column (mobile)
Both columns: --bg-card cards, full height viewport minus topbar

════════════════════════════════════════
LEFT PANEL: "UPLOAD & CONFIGURE"
════════════════════════════════════════

DROPZONE (primary interactive area):
  Height: 320px
  Style: dashed border (2px, --border-accent, 8px dash-gap)
  Background: subtle gradient (--bg-card to --bg-input)
  Border-radius: --r-xl

  CENTER CONTENT (when empty):
    Large retina icon (SVG, 64px, --accent-dim)
    "Drop retinal image here" — Sora 600, 18px, --text-secondary
    "or click to browse files" — DM Sans 400, 14px, --text-tertiary
    Accepted formats: "JPG · PNG · TIFF · BMP · DICOM" — mono tags

  HOVER STATE:
    Border: --accent-primary (solid)
    Background: --accent-glow
    Icon: scales to 1.1x, color --accent-primary
    Transition: 200ms --ease-out

  DRAG-OVER STATE:
    Border: --accent-bright (solid, 3px)
    Background: --accent-glow-strong
    Text changes to: "Release to analyze"
    Animated ring pulse effect (CSS @keyframes)

  AFTER IMAGE DROP:
    Image preview fills the dropzone (object-fit: cover)
    Overlay (bottom): dark gradient with image metadata
      - Filename, format, resolution, file size
      - All in JetBrains Mono, small

QUALITY INDICATOR (below dropzone):
  When image loaded:
    Left: circular gauge (SVG arc, 80px)
      - Score 0-100 rendered as arc fill
      - Color: grade-0 (>70) | grade-2 (40-70) | grade-4 (<40)
      - Number inside: JetBrains Mono 600, 24px
      - Animates from 0 to score on reveal (1s ease-out)
    Right: breakdown table
      - Blur: [score] [Good/Poor]
      - Contrast: [score] [Good/Low]
      - Brightness: [score] [Good/Dark/Bright]
      - SNR: [score] [Good/Poor]

WARNING BANNER (if quality score < 40):
  Background: rgba(239, 68, 68, 0.10)
  Border-left: 3px solid --grade-4-text
  Icon: AlertTriangle (Lucide, --grade-4-text)
  Text: "Low quality image detected. AI results may have reduced accuracy. Consider re-capturing."

PREPROCESSING PREVIEW (collapsible, shown after quality assessment):
  Shows what preprocessing will be applied:
  "CLAHE: clip_limit 3.0 | Denoising: Bilateral | Brightness correction: Yes"
  All as mono tags

MODEL PREFERENCE (optional controls):
  Label: "Model Selection" — DM Sans 500, 13px, --text-secondary
  Radio pills: "Auto (Recommended)" | "Hospital Grade (V2-B3)" | "Edge/Fast (B0)"
  Default: Auto
  Auto shows tooltip: "Will select EfficientNetV2-B3 on this server"

PATIENT ASSOCIATION (optional):
  Patient search (typeahead): search by name or patient code
  "New patient" quick-add inline

ANALYZE BUTTON:
  Full width, 56px height
  Text: "Analyze Image" — Sora 600, 16px
  Background: --accent-primary
  Border-radius: --r-full
  Icon: ScanEye (Lucide), left side
  Hover: --accent-bright, --shadow-glow-lg, icon pulses
  Disabled (no image): --bg-surface, --text-disabled
  Loading state: spinner + sequential text changes (see below)

PROCESSING STEPS INDICATOR (shown during inference):
  6-step vertical timeline, each step:
    Pending: dim circle, --text-tertiary
    Active: --accent-primary circle, pulsing, --text-primary
    Done: green checkmark, --grade-0-text

  Steps:
    1. ✓ Image Received
    2. ⟳ Quality Assessment (150ms)
    3. ⟳ Adaptive Preprocessing (200ms)
    4. ⟳ AI Inference (300ms)
    5. ⟳ Generating Grad-CAM (100ms)
    6. ⟳ LIME Explanation (2000ms)
    7. ✓ Results Ready

  Each step completion triggers a micro-animation (checkmark draws in)

════════════════════════════════════════
RIGHT PANEL: "RESULTS" (shows after inference)
════════════════════════════════════════

Initial state: centered placeholder
  "Results will appear here after analysis"
  Faint retinal scan illustration

AFTER INFERENCE — animate in with Framer Motion (staggered children):

━━━ GRADE CARD (top, full width) ━━━
  Background: linear-gradient using grade color (very subtle)
  Border: 1px solid grade border color
  Left: Grade badge
    Large circle/shield icon, grade color
    "Grade 2" — JetBrains Mono 700, 36px
    "MODERATE NPDR" — Sora 700, 22px, grade text color
  Right: Confidence ring
    SVG arc showing confidence %
    JetBrains Mono 700, 32px — "87.4%"
    "Confidence" label below, --text-tertiary
  Animation: card slides in from right, 400ms

━━━ PROBABILITY BARS (5 classes) ━━━
  Section title: "Class Probabilities"
  5 horizontal bars, each:
    Left: grade label ("No DR", "Mild", "Moderate", "Severe", "Proliferative")
    Center: animated bar (width = probability %, fills on render)
      Each bar: its grade color
      Predicted grade bar: full opacity, others: 40% opacity
    Right: JetBrains Mono percentage value
  Animation: bars expand from 0 width, staggered 80ms delays

━━━ GRAD-CAM VIEWER ━━━
  Section title: "AI Attention Map (Grad-CAM)"
  Subtitle: "Highlighted regions show lesions influencing the prediction"

  Toggle bar (3 segments):
    [Original] [Heatmap] [Overlay]
    Active segment: --accent-primary bg, pill shape
    Inactive: --bg-input, --text-secondary

  Image display area (400px × 400px on desktop):
    Smooth crossfade transition between views (300ms opacity)
    Zoom on hover (CSS: transform scale 1.8, centered, cursor zoom-in)
    Bounding boxes: when 'Overlay' selected, show attention_regions as
      animated dashed rectangles (--accent-primary, animated dash-offset)

  Heatmap Legend (shown when Heatmap or Overlay selected):
    Horizontal gradient bar: blue → cyan → green → yellow → red
    Labels: "Low Attention" ← → "High Attention"
    Font: DM Sans 400, 11px, --text-tertiary

  LIME Toggle (separate smaller button):
    "Show LIME Explanation" — secondary ghost button
    Expands section below with superpixel explanation
    Caption: "Green regions support the diagnosis; red regions oppose it"

━━━ CLINICAL RECOMMENDATION ━━━
  Card with left border (4px, urgency color):
    urgency-color mapping:
      none/routine → --grade-0-text
      moderate     → --grade-2-text
      urgent       → --grade-3-text
      emergency    → --grade-4-text (+ subtle pulsing border animation)

  Icon: Stethoscope (Lucide), urgency color
  Heading: "Clinical Recommendation" — DM Sans 600, 15px, --text-secondary
  Recommendation text: DM Sans 500, 16px, --text-primary
  Guideline source: "AAO 2023" — mono tag, small
  Follow-up: "Recommended follow-up: 1 month" — --text-secondary

  EMERGENCY GRADE (Grade 4 only):
    Extra red warning block:
    "⚠ IMMEDIATE ACTION REQUIRED" — bold, red, pulsing

━━━ PROCESSING METADATA (collapsed accordion) ━━━
  "Technical Details ▾"
  When expanded:
    Model: EfficientNetV2-B3 (Tier 2)
    Preprocessing: CLAHE 2.0 | Gaussian denoising | Optic disc masked
    Inference: 218ms | Grad-CAM: 67ms | LIME: 1840ms | Total: 2167ms
    Prediction ID: [uuid] (copyable)
  All in JetBrains Mono, 12px, --text-tertiary

━━━ ACTION BAR (sticky bottom of results panel) ━━━
  3 buttons:
    "Save to Patient Record" — filled teal (disabled if no patient linked)
    "Download PDF Report" — ghost, with FileDown icon
    "New Analysis" — text link, --text-secondary
```

---

#### 5.3.5 Patients Page (`Patients.tsx`)

```
SEARCH & FILTER BAR:
  Search input: name or patient code
  Filter: All | With Scans | Pending Review
  "Add Patient" button: teal filled, right-aligned

PATIENT CARDS GRID (3 columns desktop, 2 tablet, 1 mobile):
  Each card: --bg-card, --r-xl, --shadow-card
    Avatar: initials in colored circle (grade color of latest scan)
    Name: Sora 600
    Code: JetBrains Mono, --text-secondary
    Latest grade badge
    Scan count: "4 scans"
    Last scan date
    "View History" button on hover
```

---

#### 5.3.6 Patient Detail Page (`PatientDetail.tsx`)

```
TOP: Patient info card
  Name, DOB, Gender, Diabetes Duration, HbA1c
  All from DB; editable inline (click to edit)

SCAN HISTORY TIMELINE:
  Chronological list of all scans for this patient
  Each scan: timestamp, grade badge, confidence, model used
  Grade TREND indicator: "↑ Worsening" | "→ Stable" | "↓ Improving"
  Click any scan: expands to show Grad-CAM thumbnails + recommendation

GRADE PROGRESSION CHART:
  Recharts LineChart: X=date, Y=grade (0-4)
  Points: colored circles matching grade
  Trend line: dashed
  Tooltip: hover shows full details

DOCTOR ANNOTATION SECTION (per scan):
  TextArea: "Add clinical notes..."
  "Save Notes" button
  Saved notes shown with timestamp and author name
  Notes are stored in DB and appear in PDF report
```

---

#### 5.3.7 Reports Page (`Reports.tsx`)

```
LIST VIEW:
  All predictions with download status
  Columns: Date | Patient | Grade | Confidence | Reviewed | Actions
  "Reviewed" toggle: doctor marks as reviewed (calls PATCH /predict/{id})
  Bulk actions: "Download Selected as ZIP"

REPORT PREVIEW (in-browser):
  Click "Preview" → modal with react-pdf rendering the generated PDF
  "Download" button inside modal
```

---

### 5.4 PDF Report Specification (`report_service.py`)

```
PAGE 1 — COVER / SUMMARY:
  Header band: --bg-base color (or dark navy)
    Left: RetinaIQ logo (text-based, no image dependency)
    Right: Report date, Prediction ID (mono font)
  
  Patient Information Table:
    Name | Patient Code | DOB | Gender | Diabetes Duration | HbA1c
  
  Scan Information:
    Date/Time | Model Used | Processing Time | Quality Score
  
  GRADE DISPLAY (large, centered):
    Grade number (72pt, bold) + Label ("Moderate NPDR")
    Color: grade-appropriate
    Confidence: "Confidence: 87.4%"

PAGE 2 — VISUAL EVIDENCE:
  Side-by-side images (equal width, 250pt each):
    Left: "Original Retinal Image" + image
    Right: "Grad-CAM Attention Map" + overlay image
  
  Caption: "Highlighted regions indicate lesion areas influencing the AI prediction"
  
  Attention Regions Table:
    Lists top-3 detected attention bounding boxes with coordinates

PAGE 3 — ANALYSIS & RECOMMENDATION:
  5-Class Probability Table:
    Grade | Label | Probability | Bar visualization
    Predicted row: highlighted
  
  Clinical Recommendation (large box):
    Full recommendation text
    Urgency level: colored badge
    Referral guideline source
    Follow-up timeline
  
  Doctor Notes (if any):
    Author, timestamp, notes text
  
  Disclaimer (boxed, prominent):
    "This report has been generated by RetinaIQ AI system and must be
    reviewed and verified by a qualified ophthalmologist before any
    clinical decisions are made. This tool is intended as decision
    support, not as a standalone diagnostic instrument."
  
  Preprocessing Transparency:
    Small table: CLAHE params, denoise method, model version

FOOTER (all pages):
  "RetinaIQ | CBIT, Hyderabad | Powered by EfficientNetV2-B3"
  Page number
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION 6 — DATABASE SCHEMA (COMPLETE)
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```sql
-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── USERS ─────────────────────────────────────────────────────────────────────
CREATE TABLE users (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email           VARCHAR(255) UNIQUE NOT NULL,
  hashed_password VARCHAR(255) NOT NULL,
  full_name       VARCHAR(255) NOT NULL,
  role            VARCHAR(50) NOT NULL DEFAULT 'technician',
                  -- ENUM: 'admin' | 'doctor' | 'technician'
  clinic_name     VARCHAR(255),
  clinic_id       UUID,                   -- for multi-clinic deployments
  is_active       BOOLEAN DEFAULT TRUE,
  last_login_at   TIMESTAMP,
  created_at      TIMESTAMP DEFAULT NOW(),
  updated_at      TIMESTAMP DEFAULT NOW()
);

-- ── PATIENTS ──────────────────────────────────────────────────────────────────
CREATE TABLE patients (
  id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_code            VARCHAR(100) UNIQUE,
  full_name               VARCHAR(255) NOT NULL,
  date_of_birth           DATE,
  gender                  VARCHAR(20),     -- 'male' | 'female' | 'other'
  phone                   VARCHAR(30),
  diabetes_type           VARCHAR(10),     -- 'type1' | 'type2' | 'other'
  diabetes_duration_years FLOAT,
  hba1c_percent           FLOAT,           -- glycated hemoglobin (key clinical feature)
  systolic_bp             INTEGER,         -- blood pressure
  diastolic_bp            INTEGER,
  notes                   TEXT,
  created_by              UUID REFERENCES users(id),
  created_at              TIMESTAMP DEFAULT NOW(),
  updated_at              TIMESTAMP DEFAULT NOW()
);

-- ── PREDICTIONS ───────────────────────────────────────────────────────────────
CREATE TABLE predictions (
  id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id           UUID REFERENCES patients(id) ON DELETE SET NULL,
  analyzed_by          UUID REFERENCES users(id),

  -- Storage paths (MinIO/S3)
  original_image_path  VARCHAR(500),
  gradcam_overlay_path VARCHAR(500),
  gradcam_heatmap_path VARCHAR(500),
  lime_image_path      VARCHAR(500),

  -- Image Quality
  blur_score           FLOAT,
  contrast_score       FLOAT,
  brightness_mean      FLOAT,
  snr_score            FLOAT,
  composite_quality    FLOAT,
  quality_label        VARCHAR(20),

  -- Preprocessing
  preprocessing_config JSONB,           -- full auto-config params
  model_used           VARCHAR(100),    -- 'EfficientNetV2-B3' etc.
  model_tier           INTEGER,         -- 1 | 2 | 3

  -- Results
  dr_grade             INTEGER NOT NULL, -- 0-4
  confidence           FLOAT NOT NULL,
  class_probabilities  JSONB,           -- {0: 0.02, 1: 0.05, ...}
  attention_regions    JSONB,           -- [{x,y,w,h}, ...]

  -- Clinical
  clinical_recommendation TEXT,
  urgency_level        VARCHAR(20),
  follow_up_months     FLOAT,

  -- Performance
  preprocessing_ms     INTEGER,
  inference_ms         INTEGER,
  gradcam_ms           INTEGER,
  lime_ms              INTEGER,
  total_ms             INTEGER,

  -- Review workflow
  doctor_notes         TEXT,
  reviewed_by          UUID REFERENCES users(id),
  is_reviewed          BOOLEAN DEFAULT FALSE,
  reviewed_at          TIMESTAMP,

  created_at           TIMESTAMP DEFAULT NOW()
);

-- ── AUDIT LOG ─────────────────────────────────────────────────────────────────
CREATE TABLE audit_log (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       UUID REFERENCES users(id),
  action        VARCHAR(100) NOT NULL,  -- 'inference', 'login', 'delete_patient', etc.
  resource_type VARCHAR(50),            -- 'prediction', 'patient', 'user'
  resource_id   UUID,
  ip_address    INET,
  user_agent    TEXT,
  metadata      JSONB,                  -- additional context
  created_at    TIMESTAMP DEFAULT NOW()
);

-- ── INDEXES ───────────────────────────────────────────────────────────────────
CREATE INDEX idx_predictions_patient_id  ON predictions(patient_id);
CREATE INDEX idx_predictions_analyzed_by ON predictions(analyzed_by);
CREATE INDEX idx_predictions_created_at  ON predictions(created_at DESC);
CREATE INDEX idx_predictions_dr_grade    ON predictions(dr_grade);
CREATE INDEX idx_audit_log_user_id       ON audit_log(user_id);
CREATE INDEX idx_patients_patient_code   ON patients(patient_code);
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION 7 — SECURITY (COMPLETE)
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
Authentication:
  - JWT access tokens: 30-minute expiry
  - JWT refresh tokens: 7-day expiry, rotation on use
  - bcrypt password hashing: cost factor 12
  - Failed login limit: 5 attempts → 15-minute lockout (stored in Redis)

Authorization (Role Matrix):
  Endpoint                        | Admin | Doctor | Technician
  ─────────────────────────────── |───────|────────|───────────
  POST /predict/analyze           |  ✓    |   ✓    |    ✓
  GET  /predict/{id}              |  ✓    |   ✓    |    ✓
  POST /predict/{id}/annotate     |  ✗    |   ✓    |    ✗
  PATCH /predict/{id} (reviewed)  |  ✗    |   ✓    |    ✗
  GET/POST /patients              |  ✓    |   ✓    |    ✓
  DELETE /patients/{id}           |  ✓    |   ✗    |    ✗
  GET /reports/*/download         |  ✓    |   ✓    |    ✓
  GET /dashboard/stats            |  ✓    |   ✓    |    ✓
  GET /users (admin panel)        |  ✓    |   ✗    |    ✗

File Upload Security:
  - python-magic for MIME type detection (not file extension)
  - Allowed MIME types: image/jpeg, image/png, image/tiff, image/bmp,
    application/dicom, image/x-dicom
  - Max file size: 20MB (enforced at Nginx AND FastAPI)
  - All files stored with UUID filenames (never original)
  - Virus scanning hook: optional ClamAV integration (env flag)

Network Security:
  - CORS: restricted to FRONTEND_URL from .env
  - Rate limiting: 30 requests/min per user on /predict/analyze
  - General rate limit: 200 requests/min per IP on all routes
  - HTTPS: Nginx SSL termination (Let's Encrypt in production)
  - Security headers: X-Content-Type-Options, X-Frame-Options,
    Content-Security-Policy, Strict-Transport-Security
  - SQL injection: SQLAlchemy ORM only, NO raw SQL strings
  - XSS: Pydantic strict validation on all text inputs

Data Privacy:
  - Patient data: stored encrypted at rest (PostgreSQL encryption)
  - Images: stored in private MinIO bucket (no public access)
  - Presigned URLs for image access (60-minute expiry)
  - GDPR-ready: patient data deletion endpoint (cascades to predictions)
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION 8 — DOCKER & DEPLOYMENT
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```yaml
# docker-compose.yml

version: '3.9'
services:

  backend:
    build:
      context: ./backend
      target: production
    ports: ["8000:8000"]
    env_file: .env
    depends_on:
      postgres: {condition: service_healthy}
      redis:    {condition: service_healthy}
      minio:    {condition: service_healthy}
    volumes:
      - ./ml_models:/app/ml_models:ro   # read-only model weights
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  celery_worker:
    build:
      context: ./backend
      target: production
    command: celery -A app.tasks.celery_app worker --loglevel=info --concurrency=4
    env_file: .env
    depends_on: [redis, backend]
    volumes:
      - ./ml_models:/app/ml_models:ro
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      target: production
    ports: ["3000:80"]
    depends_on: [backend]
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB:       ${DB_NAME}
      POSTGRES_USER:     ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./backend/db/migrations/init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER} -d ${DB_NAME}"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redisdata:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER:     ${MINIO_ACCESS_KEY}
      MINIO_ROOT_PASSWORD: ${MINIO_SECRET_KEY}
    volumes:
      - miniodata:/data
    ports: ["9000:9000", "9001:9001"]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]

  nginx:
    image: nginx:1.25-alpine
    ports: ["80:80", "443:443"]
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on: [backend, frontend]
    restart: unless-stopped

volumes:
  pgdata:
  redisdata:
  miniodata:
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION 9 — NOVELTY FEATURES (7 DIFFERENTIATORS)
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Every one of the following must be implemented. These features make this system
superior to all 7 papers reviewed in the literature survey.

```
#1 ADAPTIVE PREPROCESSING INTELLIGENCE
   No prior system in literature auto-configures CLAHE clip limits,
   denoise strength, and brightness correction per image at runtime.
   This reduces technician dependency and handles diverse imaging hardware.

#2 THREE-TIER MODEL AUTO-ROUTING
   Automatic selection between EfficientNet-B0 (Tier 1/edge),
   EfficientNetV2-B3 (Tier 2/hospital), and CNN+ViT (Tier 3/experimental)
   based on image quality, file size, and GPU availability.
   No system in the literature review implements automatic tier routing.

#3 MULTI-XAI FUSION PANEL
   Simultaneous Grad-CAM + LIME presentation with toggleable views.
   Literature [3] used only attention maps; [5] used Grad-CAM only.
   No system combines both with bounding box region highlighting.

#4 CLINICAL RECOMMENDATION ENGINE
   Automatic mapping from DR grade to AAO/NICE-aligned recommendation.
   Specifies urgency level, follow-up timeline, and referral wording.
   No existing system in the review includes guideline-aligned text output.

#5 PHYSICIAN ANNOTATION LAYER
   Doctor can add clinical notes post-prediction, stored in DB,
   included in the PDF report alongside AI results.
   Creates a hybrid AI+human clinical record.

#6 REAL-TIME IMAGE QUALITY GATING
   Images below composite quality threshold trigger user warnings
   before inference begins. Prevents reliance on results from
   corrupt or low-quality fundus photographs.
   Not implemented in any of the 7 reviewed systems.

#7 COMPLETE AUDIT TRAIL + REPRODUCIBILITY LOG
   Every prediction logs: user ID, timestamp, model version,
   preprocessing config, all metrics, hardware context.
   Enables full reproducibility — any prediction can be re-explained.
   Mandatory for clinical governance and regulatory compliance.
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION 10 — FINAL DELIVERY CHECKLIST
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before marking the project complete, verify ALL of the following:

```
BACKEND:
[ ] uvicorn app.main:app --reload starts without errors
[ ] All 7 route files exist and are fully implemented (no NotImplementedError)
[ ] POST /predict/analyze returns a valid PredictionResponse with base64 images
[ ] Grad-CAM overlay is a valid PNG for every inference call
[ ] LIME explanation is generated and returned
[ ] PDF download endpoint returns a valid PDF with images
[ ] JWT auth works: login → token → protected route access
[ ] Role enforcement: technician cannot annotate predictions
[ ] Rate limiting: 31st request/min returns 429
[ ] Adaptive preprocessor handles: blurry JPEG, dark PNG, DICOM, TIFF, overexposed
[ ] ModelRegistry loads all model tiers at startup (logs on boot)
[ ] PostgreSQL: all 4 tables created via Alembic migration on first run
[ ] Redis: prediction results cached with 1hr TTL
[ ] MinIO: images uploaded with UUID filenames, accessible via presigned URLs
[ ] All audit log entries written for every inference
[ ] CORS blocked for unlisted origins
[ ] ruff: zero linting errors
[ ] mypy: zero type errors (strict mode)
[ ] pytest: all tests pass

FRONTEND:
[ ] npm run build completes with zero TypeScript errors
[ ] Landing page: hero animation renders, Framer Motion works
[ ] Login page: form submits, JWT stored, redirects to dashboard
[ ] Dashboard: stats load from API, charts render with real data
[ ] Analyze page: drag & drop works, image preview shows
[ ] Analyze page: quality gauge animates to correct score
[ ] Analyze page: processing steps animate sequentially
[ ] Analyze page: grade badge color matches severity
[ ] Analyze page: probability bars animate in on result load
[ ] Analyze page: Grad-CAM toggle (Original/Heatmap/Overlay) works
[ ] Analyze page: LIME explanation shows on toggle
[ ] Analyze page: PDF download triggers file save
[ ] Analyze page: bounding boxes overlay on attention regions
[ ] Patients page: search, add, view history all work
[ ] Patient detail page: grade progression chart renders
[ ] Doctor annotation: text saves and appears in subsequent loads
[ ] All Framer Motion animations play (no janky/missing)
[ ] Responsive: all pages usable on 375px mobile width
[ ] All Axios errors show react-hot-toast notifications

DOCKER:
[ ] docker-compose up --build starts all 7 services without error
[ ] Frontend accessible at http://localhost:3000
[ ] Backend API accessible at http://localhost:8000
[ ] Swagger docs at http://localhost:8000/docs
[ ] MinIO console at http://localhost:9001
[ ] No hardcoded secrets anywhere (all from .env)

MODEL:
[ ] train.py runs end-to-end on APTOS 2019 dataset
[ ] All 3 model tiers have architecture classes defined
[ ] Focal Loss implemented correctly
[ ] Progressive unfreezing callback implemented
[ ] k-fold cross-validation produces per-fold metrics
[ ] Model exported in SavedModel + .h5 + TFLite formats
[ ] MODEL_CARD.md written with all required sections

DOCUMENTATION:
[ ] README.md complete with all 10 required sections
[ ] .env.example has all variables with descriptions
[ ] API documented at /docs (auto-Swagger from FastAPI)
[ ] MODEL_CARD.md present in ml_models/
[ ] Inline code comments for all non-obvious logic
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTION 11 — ENVIRONMENT VARIABLES (.env.example)
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```bash
# ── Application ──────────────────────────────────────────
APP_NAME=RetinaIQ
APP_ENV=development              # development | production
DEBUG=true
FRONTEND_URL=http://localhost:3000

# ── Security ─────────────────────────────────────────────
SECRET_KEY=your-256-bit-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# ── Database ──────────────────────────────────────────────
DB_HOST=postgres
DB_PORT=5432
DB_NAME=retinaiq
DB_USER=retinaiq_user
DB_PASSWORD=your-strong-password-here
DATABASE_URL=postgresql+asyncpg://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}

# ── Redis ────────────────────────────────────────────────
REDIS_URL=redis://redis:6379/0
CACHE_TTL_SECONDS=3600

# ── MinIO / S3 ───────────────────────────────────────────
STORAGE_BACKEND=minio           # minio | s3
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
MINIO_BUCKET_NAME=retinaiq-images
MINIO_SECURE=false              # true for HTTPS MinIO

# ── Model ────────────────────────────────────────────────
MODEL_DIR=/app/ml_models
DEFAULT_MODEL_TIER=2            # 1 | 2 | 3
EFFICIENTNETV2B3_WEIGHTS=retinaiq_efficientnetv2b3.h5
EFFICIENTNETB0_WEIGHTS=retinaiq_efficientnetb0.h5
MOBILENETV3_WEIGHTS=retinaiq_mobilenetv3.h5
IMAGE_QUALITY_THRESHOLD=40      # below this → show quality warning
MAX_UPLOAD_SIZE_MB=20

# ── Celery ───────────────────────────────────────────────
CELERY_BROKER_URL=redis://redis:6379/1
CELERY_RESULT_BACKEND=redis://redis:6379/2

# ── Logging ──────────────────────────────────────────────
LOG_LEVEL=INFO
LOG_FORMAT=json                 # json | text
```

---

**BUILD THIS. EVERY LINE. EVERY FILE. NO SHORTCUTS.**
**This is a medical AI system. Quality, reliability, and trustworthiness are non-negotiable.**
**The deliverable must be deployable in a real clinic on day one.**
