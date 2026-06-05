# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PhD research on automated prostate cancer grading from histopathology images using the PANDAS dataset (ISUP grades 0–5). The codebase implements patch-level CNNs and Multiple Instance Learning (MIL) with attention pooling.

All experiments are run as Jupyter notebooks — there is no CLI training entry point.

```bash
jupyter notebook          # launch notebooks
jupyter lab               # alternative UI
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Architecture

### `utils/` — Shared Library

| File | Purpose |
|------|---------|
| `models.py` | All model wrappers and ensemble logic |
| `dataset.py` | Dataset classes and color-space transforms (Albumentations-compatible) |
| `train.py` | `training_step`, `validation_step`, `train_model` loop with early stopping |
| `metrics.py` | Bootstrap metrics (QWK, accuracy, F1, recall, precision) + `model_checkpoint` |
| `layer.py` | Custom layers: `GeM`, `SEBlock`, `MixUp`, `CutMix`, `DeformableConv2d` |
| `mil.py` | Canonical MIL dataset (`PandasWithMilDataset`) and MIL models |

### Model Classes

**Patch-level (standard classification)**
- `EfficientNetApi` — wraps `efficientnet_pytorch.EfficientNet`, freezes all but last 150 params, adds dropout + linear head
- `ConvNeXtApi` — wraps torchvision ConvNeXt, unfreezes last N stages
- `EfficientNetApiA` — alternate wrapper with configurable `fine_tune` param count
- `ViTApi`, `SwinApi` — ViT and Swin Transformer wrappers with the same pattern

**MIL (Multiple Instance Learning)** — all in `utils/mil.py`
- `EfficientNetMIL` — backbone + `GatedAttention` pooling; input `(B, N, C, H, W)`, output dict `{'logits', 'attn', 'features'}`
- `ConvNeXtMIL` — same pattern; unfreezes last N `model.features` blocks
- `SwinMIL` — same pattern; always unfreezes norm/permute/avgpool/flatten layers in addition to last N blocks
- `ViTMIL` — same pattern; freezes all but last `fine_tune` params
- `AttentionMIL`, `SwinAttentionMIL` (in `utils/models.py`) — older/simpler MIL variants

**Ensemble**
- `EnsembleEfficientNet` — combines multiple models with `mean`, `weighted_mean`, `max`, `majority_vote`, or `weighted_vote`

### Dataset Classes

- `PandasDataset` (in `utils/dataset.py`) — single-image dataset; expects `image_id` and `isup_grade` columns
- `PandasWithMilDataset` — two versions exist:
  - `utils/mil.py` — **canonical version**: validates inputs, handles alpha channels, raises on missing folders
  - `utils/dataset.py` — older version without those guards; prefer the `mil.py` version
- `SicapDataset` — same as `PandasDataset` but uses `slide_id` column (SICAPv2 external dataset)
- `PatchBagDataset` — creates bags by tiling a single image into a 6×6 grid

Bag structure: pads to `max_patches=36` with white patches; returns `(bag, mask, label, img_id)`. The `mask` tensor marks real vs. padding patches (1/0).

### Color Space Transforms (Albumentations-compatible)

Defined in `utils/dataset.py`; all accept and return `image` arrays:
- `RGB2XYZTransform`, `RGB2HedTransform`, `RGB2LABTransform`, `RGB2LUVTransform`, `RGB2HSVTransform`
- `RGB2YHUTransform`, `RGB2YHVTransform` — custom 3-channel fusions from HED/XYZ/LUV
- `RGB2Fusion` — fuses multiple color spaces with `sum`/`mean`/`max` modes
- `RemovePenMarkAlbumentations` — pre-processing to remove pen artifacts

### Label Encoding

All models use **ordinal encoding** for ISUP grades 0–5 (5-dimensional binary vector):
- Grade 0 → `[0,0,0,0,0]`, Grade 3 → `[1,1,1,0,0]`, Grade 5 → `[1,1,1,1,1]`
- `encode_ordinal_labels` / `decode_ordinal_predictions` in `utils/models.py`
- Loss: `BCEWithLogitsLoss` (via `OrdinalRegressionLoss`)
- Decoding: `sigmoid → sum thresholded at 0.5`

### Training Loop

`train_model` in `utils/train.py`:
- Primary metric for checkpointing: **QWK (Quadratic Weighted Kappa)** via `val_kappa['mean']`
- Early stopping with configurable `patience` (default 20 epochs)
- Metrics reported with 95% CI via bootstrap (1000 samples) in `utils/metrics.py`
- `apply_active_learning` — runs additional training epochs on low-confidence samples without checkpointing
- MixUp/CutMix augmentation is implemented in `layer.py` and imported but currently commented out in `training_step`

### Evaluation

`evaluation()` in `utils/metrics.py` — used post-training in notebooks; returns `(metrics_dict, (preds, targets, img_ids))`. `format_metrics()` prints the bootstrap CI table.

## Data

```
data/
├── train_5fold.csv      # 5-fold CV splits (fold column)
├── test.csv             # held-out test set
├── train.csv, val.csv   # alternative splits
├── entropy.csv          # pre-computed per-image entropy scores
├── without_pen_mask.csv # images after pen-mark removal
└── binary/              # per-grade binary CSVs (isup_grade_0_binary.csv … _5_)
```

Images are not in this repo — paths are configured per-notebook. The PANDAS dataset is the primary source; SICAPv2 is used for cross-dataset evaluation.

Pre-trained EfficientNet weights are stored at repo root (`efficientnet-b0-*.pth`).

## Notebook Organization

```
code/tests/baseline/     # main experiments: EfficientNet variants, ConvNeXt
code/tests/transformers/ # MIL experiments (b0-mil.ipynb, b0-mil-49.ipynb)
code/providers/          # specialist/provider-specific models
code/fusion/             # color-space fusion experiments
analyse/                 # GradCAM, FiftyOne visualization, color analysis
```

Training logs and confusion matrices are saved under `code/tests/baseline/logs/`.

## Experiment Dashboard

`experiment-dashboard/` — React + Vite + TypeScript app for visualizing experiment results. Currently at scaffold stage (default Vite template). Run with:
```bash
cd experiment-dashboard && npm install && npm run dev
```

## Key Design Decisions

- **Backbone freezing**: all wrappers freeze most params and unfreeze the last N (150 params for EfficientNet/ViT, last N blocks for ConvNeXt/Swin)
- **MIL mask**: always pass the `mask` tensor to MIL model `forward()` — without it, softmax runs over padding patches and attention scores are meaningless
- **Entropy-based active learning**: `remove_images_by_entropy` in `utils/train.py` identifies low-confidence samples for retraining
- `utils/mil.py` contains the canonical, more robust implementations; `utils/models.py` contains older/alternative versions of some classes
