# %% [markdown]
# # Swin Transformer Family (tiny/small/base) — Ordinal Focal Loss + Optuna HPO
# 
# This notebook sweeps the **Swin Transformer family (tiny, small, base)** from torchvision on the
# PANDAS ISUP-grading task, mirroring the EfficientNet/ConvNeXt notebooks. For **each** backbone it:
# 
# 1. **Cleans noise** — removes the noisiest 20% of images (highest `difficulty_score` in `entropy.csv`).
# 2. **Searches hyperparameters with Optuna** — `lr`, `dropout`, `focal_gamma`, `focal_alpha`,
#    `unfreeze_blocks`, `weight_decay`, `batch_size`, and `use_ordinal_loss` (focal-only vs focal+ordinal) and `ordinal_weight` (with `focal_weight = 1 - ordinal_weight`) are tuned to maximise validation QWK on a short proxy run.
# 3. **Trains fully** with the best params (cosine schedule + warm-up, AMP, early stopping on QWK).
# 4. **Evaluates on the validation set** with bootstrap 95% CIs, a confusion matrix and a classification report.
# 
# A final cell compares all three models and saves a summary (`logs/swin-family-summary.{csv,txt,png}`).
# 
# > **Note on methodology.** As in the EfficientNet/ConvNeXt notebooks, the held-out fold (`VAL_FOLD`) is
# > used both for Optuna selection *and* for the final reported metrics. The independent `data/test.csv`
# > set is available if you later want a leakage-free estimate — just point the eval loader at it.
# 
# > **Note on Swin specifics.** `torchvision`'s `SwinTransformer` doesn't expose a `classifier` attribute
# > like EfficientNet/ConvNeXt — the pooled features go through `norm → permute → avgpool → flatten → head`,
# > where `head` is a plain `nn.Linear` (not a `Sequential`). The wrapper below swaps both `head` **and**
# > the pre-pooling `norm` for `Identity`, then applies its own `LayerNorm → Dropout → Linear` on the pooled
# > features — the same pattern used for ConvNeXt's `classifier`, just adapted to Swin's slightly different
# > module layout. `features` is still a flat `nn.Sequential` of patch-embed + 4 stages + 3 patch-merging
# > (downsampling) layers, so the "unfreeze last N blocks" trick from the other notebooks applies unchanged.
# >
# > Swin variants also expect roughly `224×224` inputs divisible by the patch/window sizes — if your
# > `PandasOverlapDataset` pipeline already feeds 224×224 (or another size divisible by 32) to the other
# > backbones, no changes should be needed here.
# 

# %%
import os
import gc
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.amp import autocast, GradScaler
import torchvision.models as tvm
import albumentations as Albu
import optuna
from optuna.samplers import TPESampler
from warmup_scheduler import GradualWarmupScheduler
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score,
    classification_report, confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from IPython.display import display
import sys
sys.path.append('..')
from utils.dataset import PandasOverlapDataset

optuna.logging.set_verbosity(optuna.logging.WARNING)

# %% [markdown]
# ## Fixed Configuration

# %%
SEED           = 42
NUM_WORKERS    = 4
OUTPUT_CLASSES = 5       # ordinal thresholds for ISUP 0-5
WARMUP_EPOCHS  = 1
WARMUP_FACTOR  = 2
USE_AMP        = True    # mixed precision (fp16) — major VRAM saver
VAL_FOLD       = 3       # held-out fold used for HPO + final evaluation
ENTROPY_DROP_FRAC = 0.20 # fraction of noisiest images removed

# --- Optuna / training budget (tune to taste; a full sweep is heavy) ---
N_TRIALS       = 20      # Optuna trials per backbone
N_EPOCHS_TRIAL = 6       # epochs per Optuna trial (short proxy)
N_EPOCHS_FULL  = 40      # epochs for the final training of each backbone
PATIENCE       = 8       # early-stopping patience (full run)

# Which Swin Transformer variants to sweep (tiny/small/base; torchvision has no "large" for Swin v1).
MODELS_TO_RUN  = ['tiny', 'small', 'base']

# Per-backbone batch-size search candidates (larger nets get smaller batches to fit VRAM).
# Attention over windows is more memory-hungry than pure conv nets at a comparable stage,
# so batches are set conservatively, especially for base.
BATCH_CANDIDATES = {
    'tiny':  [8, 16],
    'small': [8, 16],
    'base':  [4, 8],
}

REPO_DIR   = '..'                                   # pos-propose/family -> repo root
DATA_DIR   = os.path.join(REPO_DIR, 'data')
IMAGES_DIR = os.path.join(REPO_DIR, 'dataset')

LOG_DIR    = 'logs'
MODEL_DIR  = 'models'
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')


# %% [markdown]
# ## VRAM Utility Helpers

# %%
def free_vram(*objs):
    """Aggressive cleanup: delete refs, run gc, empty CUDA cache."""
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def vram_report(tag=''):
    if not torch.cuda.is_available():
        return
    alloc    = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak     = torch.cuda.max_memory_allocated() / 1e9
    print(f'  [VRAM {tag}] allocated={alloc:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB')

# %% [markdown]
# ## Loss Function — Ordinal Focal Loss
# 
# `ordinal_weight` and `focal_weight` are a convex combination that always sums to 1 (`focal_weight = 1 - ordinal_weight`, computed internally — never set independently). `use_ordinal` (wired from Optuna's `use_ordinal_loss`) forces `ordinal_weight=0` (pure focal loss), so the search can compare **focal-only** vs **focal + ordinal**.

# %%
class OrdinalFocalLoss(nn.Module):
    """Focal BCE over ordinal thresholds + an optional soft penalty on the
    expected-class distance (the "ordinal" term).

    The two terms are a convex combination that always sums to 1:
        total_loss = (1 - ordinal_weight) * focal_loss + ordinal_weight * ordinal_loss

    Only `ordinal_weight` is a free parameter — `focal_weight` is always
    `1 - ordinal_weight`, so it can never be set independently and the two
    can never sum to anything other than 1.

    Setting `use_ordinal=False` forces `ordinal_weight` to 0 (i.e. `focal_weight=1`),
    which reduces this to a pure focal loss — lets Optuna compare "focal only"
    vs "focal + ordinal" cleanly.
    """

    def __init__(self, alpha=0.25, gamma=2.0, ordinal_weight=0.2,
                 use_ordinal=True, reduction='mean'):
        super().__init__()
        if not 0.0 <= ordinal_weight <= 1.0:
            raise ValueError(f'ordinal_weight must be in [0, 1], got {ordinal_weight}')

        self.alpha = alpha
        self.gamma = gamma
        self.use_ordinal = use_ordinal
        # hard-zero the ordinal term when disabled, regardless of what was passed in
        self.ordinal_weight = ordinal_weight if use_ordinal else 0.0
        self.focal_weight   = 1.0 - self.ordinal_weight   # always sums to 1 with ordinal_weight
        self.reduction = reduction

    def forward(self, logits, targets):
        # numerical stability under AMP
        logits  = logits.float()
        targets = targets.float()

        probs = torch.sigmoid(logits)
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # focal term
        p_t  = probs * targets + (1 - probs) * (1 - targets)
        loss = self.alpha * ((1 - p_t) ** self.gamma) * bce

        if self.reduction == 'mean':
            focal_loss = loss.mean()
        elif self.reduction == 'sum':
            focal_loss = loss.sum()
        else:
            focal_loss = loss

        total = self.focal_weight * focal_loss

        if self.ordinal_weight != 0.0:
            # ordinal penalty on the (soft) predicted class index
            expected_class = probs.sum(dim=1)
            target_class   = targets.sum(dim=1)
            max_class      = logits.shape[1]
            ordinal_loss   = ((expected_class - target_class) ** 2).mean() / (max_class ** 2)
            total = total + self.ordinal_weight * ordinal_loss

        return total


def decode_ordinal_predictions(logits):
    """sigmoid -> threshold @ 0.5 -> sum  ==>  ISUP grade 0..5."""
    return (torch.sigmoid(logits.float()) > 0.5).sum(dim=1)


# %% [markdown]
# ## Swin Transformer Wrapper + Backbone Registry (tiny, small, base)

# %%
SWIN_REGISTRY = {
    'tiny':  (tvm.swin_t, tvm.Swin_T_Weights.DEFAULT),
    'small': (tvm.swin_s, tvm.Swin_S_Weights.DEFAULT),
    'base':  (tvm.swin_b, tvm.Swin_B_Weights.DEFAULT),
}


class SwinApi(nn.Module):
    """
    Generic torchvision Swin Transformer (tiny/small/base) wrapper.

    Freezes the whole backbone, then unfreezes the last `unfreeze_blocks` entries of
    `features` (torchvision's SwinTransformer.features is a flat nn.Sequential of
    patch-embed, stage, patch-merging, stage, patch-merging, ..., last stage — 8 entries
    for the standard configs, same count as ConvNeXt's `features`).

    Unlike EfficientNet/ConvNeXt, Swin has no `classifier` attribute: pooled features flow
    through `norm -> permute -> avgpool -> flatten -> head`, where `head` is a plain
    `nn.Linear`. Both `head` and the pre-pooling `norm` are replaced with `Identity` here,
    and the ordinal-regression head applies its own LayerNorm on the pooled features
    instead — the same design as the ConvNeXt wrapper's replaced `classifier`, just
    adapted to Swin's module layout.
    """
    def __init__(self, model, output_dimensions, dropout_rate=0.4, unfreeze_blocks=2):
        super().__init__()
        self.model = model

        for p in self.model.parameters():
            p.requires_grad = False

        if unfreeze_blocks > 0:
            for block in self.model.features[-unfreeze_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

        # torchvision's Swin `head` is a plain nn.Linear (no Sequential wrapper)
        if isinstance(self.model.head, nn.Sequential):
            in_features = self.model.head[-1].in_features
        else:
            in_features = self.model.head.in_features
        self.model.head = nn.Identity()
        self.model.norm = nn.Identity()  # our own head normalizes instead, mirrors the ConvNeXt design

        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, output_dimensions),
        )

    def extract(self, x):
        x = self.model(x)
        if x.ndim == 4:                 # safety net, shouldn't trigger since norm/head are Identity
            x = x.mean(dim=[2, 3])
        return x

    def forward(self, x):
        return self.head(self.extract(x))


def build_model(name, dropout_rate, unfreeze_blocks):
    ctor, weights = SWIN_REGISTRY[name]
    backbone = ctor(weights=weights)
    return SwinApi(backbone, OUTPUT_CLASSES, dropout_rate, unfreeze_blocks).to(device)


# %% [markdown]
# ## Data Loading with Entropy-based Noise Cleaning

# %%
df_all = pd.read_csv(os.path.join(DATA_DIR, 'train_5fold.csv'))
df_all.columns = df_all.columns.str.strip()
print(f'Total records: {len(df_all)}')

# --- Noise cleaning: drop the noisiest images by difficulty score ---
df_entropy = pd.read_csv(os.path.join(DATA_DIR, 'entropy.csv'))
df_entropy = df_entropy.sort_values('difficulty_score', ascending=False)
n_remove   = int(len(df_entropy) * ENTROPY_DROP_FRAC)
noisy_ids  = set(df_entropy.head(n_remove)['image_id'])
df_all     = df_all[~df_all['image_id'].isin(noisy_ids)].reset_index(drop=True)
print(f'After entropy filter (removed {n_remove} noisiest): {len(df_all)}')


def drop_missing(df):
    exists = df['image_id'].apply(lambda x: os.path.isdir(os.path.join(IMAGES_DIR, str(x))))
    return df[exists].reset_index(drop=True)


df_train = drop_missing(df_all[df_all['fold'] != VAL_FOLD].reset_index(drop=True))
df_val   = drop_missing(df_all[df_all['fold'] == VAL_FOLD].reset_index(drop=True))

print(f'Train: {len(df_train)}   Val: {len(df_val)}')
print('Val class distribution:')
print(df_val['isup_grade'].value_counts().sort_index())

# %% [markdown]
# ## Augmentation

# %%
train_transforms = Albu.Compose([
    Albu.Transpose(p=0.5),
    Albu.VerticalFlip(p=0.5),
    Albu.HorizontalFlip(p=0.5),
    Albu.RandomBrightnessContrast(p=0.3),
    Albu.HueSaturationValue(p=0.2),
])

val_transforms = None

# %% [markdown]
# ## Data Loaders + Train/Val Epoch Helpers (AMP-enabled)

# %%
def make_loaders(batch_size):
    train_ds = PandasOverlapDataset(IMAGES_DIR, df_train, transforms=train_transforms, overlap=0)
    val_ds   = PandasOverlapDataset(IMAGES_DIR, df_val,   transforms=val_transforms,   overlap=0)

    common = dict(
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=RandomSampler(train_ds), **common)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader


def run_epoch_train(model, loader, optimizer, loss_fn, device, scaler, accum_steps=1):
    model.train()
    losses = []
    optimizer.zero_grad(set_to_none=True)

    for step, (imgs, targets, _) in enumerate(tqdm(loader, desc='Train', leave=False)):
        imgs    = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(device_type='cuda', enabled=USE_AMP, dtype=torch.float16):
            logits = model(imgs)
            loss   = loss_fn(logits, targets) / accum_steps

        scaler.scale(loss).backward()
        if (step + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item() * accum_steps)

    return float(np.mean(losses))


def run_epoch_val(model, loader, loss_fn, device):
    model.eval()
    losses, preds, gts = [], [], []
    with torch.no_grad():
        for imgs, targets, _ in tqdm(loader, desc='Val', leave=False):
            imgs    = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast(device_type='cuda', enabled=USE_AMP, dtype=torch.float16):
                logits = model(imgs)
                loss   = loss_fn(logits, targets)
            losses.append(loss.item())
            preds.append(decode_ordinal_predictions(logits).cpu())
            gts.append(targets.sum(1).long().cpu())

    preds = torch.cat(preds).numpy()
    gts   = torch.cat(gts).numpy()
    return dict(
        val_loss=float(np.mean(losses)),
        val_kappa=cohen_kappa_score(gts, preds, weights='quadratic'),
        val_acc=accuracy_score(gts, preds),
        val_f1=f1_score(gts, preds, average='macro', zero_division=0),
    )

# %% [markdown]
# ## Optuna Objective & Study
# 
# Each trial trains a short proxy (`N_EPOCHS_TRIAL`) and returns the best validation QWK reached.
# OOM is caught and the trial is pruned rather than crashing the whole sweep.
# 
# Search space knobs (same convex-combination loss design as the other family notebooks):
# - **`use_ordinal_loss`** (bool) — whether to use the ordinal-distance term at all, or plain focal loss only.
# - **`ordinal_weight`** — weight of the ordinal term, searched in `[0.05, 0.95]` only when `use_ordinal_loss=True`
#   (fixed at 0 otherwise). `focal_weight = 1 - ordinal_weight` is derived automatically, so the two weights
#   always sum to 1.
# - **`unfreeze_blocks`** — searched in `[1, 6]`, same range as the ConvNeXt notebook since Swin's `features`
#   also has 8 flat entries (patch-embed, 4 stages, 3 patch-merging layers).
# 

# %%
def objective(trial, name):
    use_ordinal_loss = trial.suggest_categorical('use_ordinal_loss', [True, False])

    params = dict(
        lr              = trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        dropout_rate    = trial.suggest_float('dropout_rate', 0.1, 0.6),
        focal_gamma     = trial.suggest_float('focal_gamma', 1.0, 4.0),
        focal_alpha     = trial.suggest_float('focal_alpha', 0.1, 0.9),
        use_ordinal_loss= use_ordinal_loss,
        unfreeze_blocks = trial.suggest_int('unfreeze_blocks', 1, 6),
        weight_decay    = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        batch_size      = trial.suggest_categorical('batch_size', BATCH_CANDIDATES[name]),
    )

    if use_ordinal_loss:
        params['ordinal_weight'] = trial.suggest_float('ordinal_weight', 0.05, 0.95)
    else:
        params['ordinal_weight'] = 0.0
    params['focal_weight'] = 1.0 - params['ordinal_weight']  # for logging only

    free_vram()
    model = optimizer = None
    try:
        model     = build_model(name, params['dropout_rate'], params['unfreeze_blocks'])
        loss_fn   = OrdinalFocalLoss(
            alpha=params['focal_alpha'], gamma=params['focal_gamma'],
            ordinal_weight=params['ordinal_weight'], use_ordinal=params['use_ordinal_loss'],
        )
        scaler    = GradScaler(enabled=USE_AMP)
        train_loader, val_loader = make_loaders(params['batch_size'])
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=params['lr'], weight_decay=params['weight_decay'],
        )

        best_kappa = -1.0
        for epoch in tqdm(range(1, N_EPOCHS_TRIAL + 1), desc="Epoch"):
            print("="*20)
            print("Epoch", epoch)
            print("="*20)
            run_epoch_train(model, train_loader, optimizer, loss_fn, device, scaler)
            metrics    = run_epoch_val(model, val_loader, loss_fn, device)
            best_kappa = max(best_kappa, metrics['val_kappa'])
            trial.report(metrics['val_kappa'], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return best_kappa

    except torch.cuda.OutOfMemoryError:
        print(f'  [OOM] {name} trial pruned (batch_size={params["batch_size"]})')
        raise optuna.TrialPruned()
    finally:
        free_vram(model, optimizer)


def run_study(name):
    sampler = TPESampler(seed=SEED)
    pruner  = optuna.pruners.MedianPruner(n_warmup_steps=2)
    study   = optuna.create_study(
        direction='maximize', sampler=sampler, pruner=pruner,
        study_name=f'swin-{name}-ordinal-focal',
    )
    study.optimize(
        lambda t: objective(t, name),
        n_trials=N_TRIALS, gc_after_trial=True, show_progress_bar=True,
    )
    return study


# %% [markdown]
# ## Full Training (best params) + Validation Evaluation

# %%
def run_full_training(name, best_params, log_path, model_path):
    free_vram()
    model   = build_model(name, best_params['dropout_rate'], best_params['unfreeze_blocks'])
    loss_fn = OrdinalFocalLoss(
        alpha=best_params['focal_alpha'], gamma=best_params['focal_gamma'],
        ordinal_weight=best_params.get('ordinal_weight', 0.2),
        use_ordinal=best_params.get('use_ordinal_loss', True),
    )
    scaler  = GradScaler(enabled=USE_AMP)
    train_loader, val_loader = make_loaders(best_params['batch_size'])

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=best_params['lr'] / WARMUP_FACTOR, weight_decay=best_params['weight_decay'],
    )
    scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS_FULL - WARMUP_EPOCHS)
    scheduler     = GradualWarmupScheduler(optimizer, multiplier=WARMUP_FACTOR,
                                           total_epoch=WARMUP_EPOCHS, after_scheduler=scheduler_cos)

    history = dict(train_loss=[], val_loss=[], val_kappa=[], val_acc=[], val_f1=[])
    best_kappa, best_epoch, no_improve = -1.0, 0, 0
    open(log_path, 'w').close()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, N_EPOCHS_FULL + 1):
        train_loss = run_epoch_train(model, train_loader, optimizer, loss_fn, device, scaler)
        metrics    = run_epoch_val(model, val_loader, loss_fn, device)
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        for k in ('val_loss', 'val_kappa', 'val_acc', 'val_f1'):
            history[k].append(metrics[k])

        with open(log_path, 'a') as f:
            f.write(
                f'epoch: {epoch} | lr: {lr_now:.7f} | train_loss: {train_loss:.5f} | '
                f'val_loss: {metrics["val_loss"]:.5f} | val_kappa: {metrics["val_kappa"]:.4f} | '
                f'val_acc: {metrics["val_acc"]:.4f}\n'
            )
        print(f'  [{name}] epoch {epoch:02d}/{N_EPOCHS_FULL}  train={train_loss:.4f}  '
              f'val_loss={metrics["val_loss"]:.4f}  QWK={metrics["val_kappa"]:.4f}  '
              f'acc={metrics["val_acc"]*100:.2f}%')

        if metrics['val_kappa'] > best_kappa:
            best_kappa, best_epoch, no_improve = metrics['val_kappa'], epoch, 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f'  [{name}] early stop @ epoch {epoch} (best QWK={best_kappa:.4f} @ epoch {best_epoch})')
                break

    free_vram(model, optimizer)
    return history, best_kappa, best_epoch


def evaluate_on_val(name, model_path, best_params, n_boot=1000):
    free_vram()
    model = build_model(name, best_params['dropout_rate'], best_params['unfreeze_blocks'])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    _, val_loader = make_loaders(best_params['batch_size'])

    preds, gts = [], []
    with torch.no_grad():
        for imgs, targets, _ in tqdm(val_loader, desc=f'Eval {name}', leave=False):
            imgs = imgs.to(device, non_blocking=True)
            with autocast(device_type='cuda', enabled=USE_AMP, dtype=torch.float16):
                logits = model(imgs)
            preds.append(decode_ordinal_predictions(logits).cpu())
            gts.append(targets.sum(1).long())
    preds = torch.cat(preds).numpy()
    gts   = torch.cat(gts).numpy()

    rng = np.random.default_rng(SEED)
    n   = len(gts)
    bacc = np.empty(n_boot); bkap = np.empty(n_boot); bf1 = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        bacc[i] = accuracy_score(gts[idx], preds[idx])
        bkap[i] = cohen_kappa_score(gts[idx], preds[idx], weights='quadratic')
        bf1[i]  = f1_score(gts[idx], preds[idx], average='macro', zero_division=0)

    res = dict(
        acc=accuracy_score(gts, preds),
        kappa=cohen_kappa_score(gts, preds, weights='quadratic'),
        f1=f1_score(gts, preds, average='macro', zero_division=0),
        acc_ci=(np.percentile(bacc, 2.5), np.percentile(bacc, 97.5)),
        kappa_ci=(np.percentile(bkap, 2.5), np.percentile(bkap, 97.5)),
        f1_ci=(np.percentile(bf1, 2.5), np.percentile(bf1, 97.5)),
        acc_std=bacc.std(ddof=1), kappa_std=bkap.std(ddof=1), f1_std=bf1.std(ddof=1),
        preds=preds, gts=gts,
    )
    free_vram(model)
    return res

# %% [markdown]
# ## Per-model Artifacts (curves, confusion matrix, results txt)

# %%
LABELS = [f'ISUP {i}' for i in range(6)]


def save_model_artifacts(name, history, best_epoch, res, best_params, study_best):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'],   label='Val')
    axes[0, 0].set_title('Loss'); axes[0, 0].legend(); axes[0, 0].grid(True)

    axes[0, 1].plot(history['val_kappa'], color='orange')
    axes[0, 1].axvline(best_epoch - 1, color='red', ls='--', label=f'best ep {best_epoch}')
    axes[0, 1].set_title('Val QWK'); axes[0, 1].legend(); axes[0, 1].grid(True)

    axes[1, 0].plot(history['val_acc'], color='green')
    axes[1, 0].set_title('Val Accuracy'); axes[1, 0].grid(True)

    axes[1, 1].plot(history['val_f1'], color='red')
    axes[1, 1].set_title('Val Macro F1'); axes[1, 1].grid(True)

    plt.suptitle(f'Swin-{name.upper()} — Ordinal Focal Loss (Optuna)', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, f'swin-family-{name}-training.png'), dpi=200, bbox_inches='tight')
    plt.show()

    cm      = confusion_matrix(res['gts'], res['preds'])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABELS, yticklabels=LABELS, ax=axes[0])
    axes[0].set_title('Confusion (counts)'); axes[0].set_ylabel('True'); axes[0].set_xlabel('Pred')
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=LABELS, yticklabels=LABELS, ax=axes[1])
    axes[1].set_title('Confusion (normalized)'); axes[1].set_ylabel('True'); axes[1].set_xlabel('Pred')
    plt.suptitle(f'Swin-{name.upper()} — Validation', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, f'swin-family-{name}-confusion-matrix.png'), dpi=200, bbox_inches='tight')
    plt.show()

    with open(os.path.join(LOG_DIR, f'swin-family-{name}-results.txt'), 'w') as f:
        f.write(f'Swin-{name.upper()} + Ordinal Focal Loss (Optuna HPO)\n')
        f.write('=' * 70 + '\n\nBest Optuna hyperparameters:\n')
        for k, v in best_params.items():
            f.write(f'  {k}: {v}\n')
        f.write(f'  (study best val QWK during HPO: {study_best:.4f})\n\n')
        f.write('Validation-set results (bootstrap 1000, 95% CI):\n')
        f.write(f'  Accuracy : {res["acc"]*100:.2f}% \u00b1 {res["acc_std"]*100:.2f}%  '
                f'[{res["acc_ci"][0]*100:.2f}%-{res["acc_ci"][1]*100:.2f}%]\n')
        f.write(f'  QW Kappa : {res["kappa"]:.4f} \u00b1 {res["kappa_std"]:.4f}  '
                f'[{res["kappa_ci"][0]:.4f}-{res["kappa_ci"][1]:.4f}]\n')
        f.write(f'  Macro F1 : {res["f1"]:.4f} \u00b1 {res["f1_std"]:.4f}  '
                f'[{res["f1_ci"][0]:.4f}-{res["f1_ci"][1]:.4f}]\n\n')
        f.write('Classification Report:\n')
        f.write(classification_report(res['gts'], res['preds'],
                                      target_names=LABELS, digits=4, zero_division=0))
        f.write('\nConfusion Matrix:\n' + str(cm) + '\n')


# %% [markdown]
# ## Run the Full Sweep — Optuna + Training + Evaluation for every backbone
# 
# This is the heavy cell. It iterates over `MODELS_TO_RUN` (`tiny`, `small`, `base`), and for each
# backbone runs the Optuna search, the full training, and the validation evaluation, saving all per-model
# artifacts.
# 

# %%
summary = {}

for name in MODELS_TO_RUN:
    print('\n' + '#' * 72)
    print(f'#  Swin-{name.upper()}  \u2014  Optuna HPO + full training + validation eval')
    print('#' * 72)

    model_path = os.path.join(MODEL_DIR, f'swin-family-{name}.pth')
    log_path   = os.path.join(LOG_DIR,   f'swin-family-{name}.txt')

    study       = run_study(name)
    best_params = study.best_params
    print(f'  best trial QWK={study.best_value:.4f}')
    print(f'  best params: {best_params}')

    history, best_val_kappa, best_epoch = run_full_training(name, best_params, log_path, model_path)

    res = evaluate_on_val(name, model_path, best_params)
    print(f'  >>> VAL  QWK={res["kappa"]:.4f}  acc={res["acc"]*100:.2f}%  macroF1={res["f1"]:.4f}')

    summary[name] = dict(
        best_params=best_params, study_best=study.best_value, best_epoch=best_epoch,
        acc=res['acc'], kappa=res['kappa'], f1=res['f1'],
        acc_ci=res['acc_ci'], kappa_ci=res['kappa_ci'], f1_ci=res['f1_ci'],
        acc_std=res['acc_std'], kappa_std=res['kappa_std'], f1_std=res['f1_std'],
    )

    save_model_artifacts(name, history, best_epoch, res, best_params, study.best_value)
    free_vram()

print('\nSweep complete.')


# %% [markdown]
# ## Final Comparison — all Swin tiny/small/base on the validation set

# %%
rows = []
for name, s in summary.items():
    rows.append(dict(
        model=f'Swin-{name}',
        qwk=s['kappa'], acc=s['acc'], macro_f1=s['f1'],
        qwk_lo=s['kappa_ci'][0], qwk_hi=s['kappa_ci'][1],
        study_best=s['study_best'], best_epoch=s['best_epoch'],
    ))

df_summary = pd.DataFrame(rows).sort_values('qwk', ascending=False).reset_index(drop=True)
df_summary.to_csv(os.path.join(LOG_DIR, 'swin-family-summary.csv'), index=False)
display(df_summary)

best_row = df_summary.iloc[0]
print(f'\nBest model on validation: {best_row["model"]}  '
      f'(QWK={best_row["qwk"]:.4f}, acc={best_row["acc"]*100:.2f}%, macroF1={best_row["macro_f1"]:.4f})')

order = df_summary.sort_values('qwk')
yerr  = np.vstack([order['qwk'] - order['qwk_lo'], order['qwk_hi'] - order['qwk']])
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(order['model'], order['qwk'], xerr=yerr, color='steelblue', alpha=0.85, capsize=4)
ax.set_xlabel('Validation QWK')
ax.set_title('Swin tiny/small/base \u2014 Validation QWK (Ordinal Focal Loss + Optuna)')
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, 'swin-family-comparison.png'), dpi=200, bbox_inches='tight')
plt.show()

with open(os.path.join(LOG_DIR, 'swin-family-summary.txt'), 'w') as f:
    f.write('Swin Transformer family (tiny/small/base) \u2014 Ordinal Focal Loss + Optuna HPO\n')
    f.write('Validation-set comparison (noise-cleaned, entropy filter '
            f'{ENTROPY_DROP_FRAC:.0%})\n')
    f.write('=' * 72 + '\n\n')
    f.write(df_summary.to_string(index=False) + '\n\n')
    f.write('Best hyperparameters per model:\n')
    for name, s in summary.items():
        f.write(f'  {name}: {json.dumps(s["best_params"])}\n')

print('\nSaved: logs/swin-family-summary.csv, logs/swin-family-summary.txt, logs/swin-family-comparison.png')


# %%
# --- Test Set Evaluation & Ensemble ---
print('\n' + '=' * 72)
print('=' * 20 + ' TEST SET EVALUATION & ENSEMBLE ' + '=' * 20)
print('=' * 72)

df_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
df_test.columns = df_test.columns.str.strip()
df_test = drop_missing(df_test)
print(f'Test records: {len(df_test)}')

test_ds = PandasOverlapDataset(IMAGES_DIR, df_test, transforms=None, overlap=0)
test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

test_gts = []
all_test_logits = {name: [] for name in summary.keys()}

for name in summary.keys():
    print(f'\nEvaluating Swin-{name.upper()} on Test Set...')
    best_params = summary[name]['best_params']

    free_vram()
    model = build_model(name, best_params['dropout_rate'], best_params['unfreeze_blocks'])
    model_path = os.path.join(MODEL_DIR, f'swin-family-{name}.pth')
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    preds, logits_list = [], []
    gts = []

    with torch.no_grad():
        for imgs, targets, _ in tqdm(test_loader, desc=f'Test {name}', leave=False):
            imgs = imgs.to(device, non_blocking=True)
            with autocast(device_type='cuda', enabled=USE_AMP, dtype=torch.float16):
                logits = model(imgs)
            logits_list.append(logits.float().cpu())
            preds.append(decode_ordinal_predictions(logits).cpu())

            if name == list(summary.keys())[0]:
                gts.append(targets.sum(1).long().cpu())

    preds = torch.cat(preds).numpy()
    if name == list(summary.keys())[0]:
        test_gts = torch.cat(gts).numpy()

    all_test_logits[name] = torch.cat(logits_list)

    acc = accuracy_score(test_gts, preds)
    kappa = cohen_kappa_score(test_gts, preds, weights='quadratic')
    f1 = f1_score(test_gts, preds, average='macro', zero_division=0)

    print(f'  >>> TEST  Swin-{name.upper()}  QWK={kappa:.4f}  acc={acc*100:.2f}%  macroF1={f1:.4f}')

print('\n' + '-' * 72)
print('--- Ensemble Evaluation ---')

ensemble_logits = torch.zeros_like(all_test_logits[list(summary.keys())[0]])
for name in summary.keys():
    ensemble_logits += all_test_logits[name]
ensemble_logits /= len(summary.keys())

ensemble_probs = torch.zeros_like(all_test_logits[list(summary.keys())[0]])
for name in summary.keys():
    ensemble_probs += torch.sigmoid(all_test_logits[name])
ensemble_probs /= len(summary.keys())

ensemble_preds_logits = decode_ordinal_predictions(ensemble_logits).numpy()
ensemble_preds_probs = (ensemble_probs > 0.5).sum(dim=1).numpy()

acc_ens = accuracy_score(test_gts, ensemble_preds_logits)
kappa_ens = cohen_kappa_score(test_gts, ensemble_preds_logits, weights='quadratic')
f1_ens = f1_score(test_gts, ensemble_preds_logits, average='macro', zero_division=0)
print(f'  >>> TEST ENSEMBLE (Mean Logits) QWK={kappa_ens:.4f}  acc={acc_ens*100:.2f}%  macroF1={f1_ens:.4f}')

acc_ens_p = accuracy_score(test_gts, ensemble_preds_probs)
kappa_ens_p = cohen_kappa_score(test_gts, ensemble_preds_probs, weights='quadratic')
f1_ens_p = f1_score(test_gts, ensemble_preds_probs, average='macro', zero_division=0)
print(f'  >>> TEST ENSEMBLE (Mean Probs)  QWK={kappa_ens_p:.4f}  acc={acc_ens_p*100:.2f}%  macroF1={f1_ens_p:.4f}')

with open(os.path.join(LOG_DIR, 'swin-family-test-results.txt'), 'w') as f:
    f.write('Test Set Results\n')
    f.write('=' * 50 + '\n')
    for name in summary.keys():
        preds = decode_ordinal_predictions(all_test_logits[name]).numpy()
        k = cohen_kappa_score(test_gts, preds, weights='quadratic')
        a = accuracy_score(test_gts, preds)
        f1_sc = f1_score(test_gts, preds, average='macro', zero_division=0)
        f.write(f'Swin-{name.upper()}: QWK={k:.4f}, Acc={a*100:.2f}%, F1={f1_sc:.4f}\n')

    f.write('-' * 50 + '\n')
    f.write(f'Ensemble (Mean Logits): QWK={kappa_ens:.4f}, Acc={acc_ens*100:.2f}%, F1={f1_ens:.4f}\n')
    f.write(f'Ensemble (Mean Probs):  QWK={kappa_ens_p:.4f}, Acc={acc_ens_p*100:.2f}%, F1={f1_ens_p:.4f}\n')



