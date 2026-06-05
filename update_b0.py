import json
import copy

def modify():
    path = 'code/tests/transformers/b0-mil-focal.ipynb'
    with open(path, 'r') as f:
        nb = json.load(f)

    # 1. Imports
    nb['cells'][1]['source'] = [
        "from torch import optim\n",
        "from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights\n",
        "import torch\n",
        "import random\n",
        "import numpy as np\n",
        "import albumentations as Albu\n",
        "import pandas as pd\n",
        "from torch.utils.data import DataLoader\n",
        "from torch.utils.data.sampler import RandomSampler, SequentialSampler\n",
        "from warmup_scheduler import GradualWarmupScheduler\n",
        "from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, recall_score, precision_score\n",
        "from tqdm import tqdm\n",
        "import os\n",
        "import sys\n",
        "import optuna\n",
        "from optuna.pruners import MedianPruner\n",
        "sys.path.append('../../../')\n",
        "from utils.dataset import PandasWithMilDataset\n",
        "from utils.models import EfficientNetMIL\n"
    ]

    # 2. Config
    nb['cells'][3]['source'] = [
        "# Training parameters\n",
        "seed = 42\n",
        "batch_size = 8\n",
        "num_workers = 4\n",
        "output_classes = 5  # For ordinal encoding: ISUP 0-5 → 5 thresholds\n",
        "init_lr = 3e-4\n",
        "warmup_factor = 2\n",
        "warmup_epochs = 1\n",
        "dropout_rate = 0.4\n",
        "patience = 7\n",
        "\n",
        "# Optuna Configuration\n",
        "OPTUNA_SUBSET_FRAC = 0.2\n",
        "OPTUNA_BATCH_SIZE = batch_size\n",
        "N_OPTUNA_EPOCHS = 10\n",
        "N_TRIALS = 30\n",
        "\n",
        "# Device\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "print(f\"Using device: {device}\")\n",
        "\n",
        "# Set seeds\n",
        "torch.manual_seed(seed)\n",
        "random.seed(seed)\n",
        "np.random.seed(seed)\n",
        "if torch.cuda.is_available():\n",
        "    torch.cuda.manual_seed_all(seed)\n",
        "\n",
        "# Paths\n",
        "ROOT_DIR = '../../..'\n",
        "data_dir = '../../../..'\n",
        "images_dir = '/home/woshington/Projects/Doutorado/bag_of_patches'\n",
        "\n",
        "print(f\"Data directory: {images_dir}\")\n",
        "\n",
        "# Output paths\n",
        "os.makedirs('../logs', exist_ok=True)\n",
        "os.makedirs('../models', exist_ok=True)\n",
        "model_path = 'models/b0-mil-focal-optuna.pth'\n",
        "log_path = 'logs/b0-mil-focal-optuna.txt'\n",
        "STUDY_DB = 'sqlite:///../logs/b0-mil-focal-optuna.db'\n",
        "STUDY_NAME = 'b0-mil-focal-optuna'\n"
    ]

    # 3. Remove loss initialization from global scope
    # Cell 5 is the code for BCEFocalLoss
    loss_cell_source = []
    for line in nb['cells'][5]['source']:
        if "loss_function = " in line or "print(\"Ordinal Regression Loss initialized\")" in line:
            continue
        loss_cell_source.append(line)
    nb['cells'][5]['source'] = loss_cell_source

    # 4. In Cell 11, dataloaders
    # We will modify it to create optuna_train_loader and optuna_val_loader
    nb['cells'][11]['source'] = [
        "from torch.utils.data import SequentialSampler\n",
        "\n",
        "# Subset aleatório fixo para o Optuna\n",
        "rng_optuna = np.random.default_rng(seed)\n",
        "optuna_idx = rng_optuna.choice(len(df_train), size=int(len(df_train) * OPTUNA_SUBSET_FRAC), replace=False)\n",
        "df_optuna = df_train.iloc[optuna_idx].reset_index(drop=True)\n",
        "\n",
        "print(f'Optuna subset: {len(df_optuna)} imgs ({OPTUNA_SUBSET_FRAC:.0%})')\n",
        "\n",
        "optuna_train_ds = PandasWithMilDataset(images_dir, df_optuna, transforms=train_transforms)\n",
        "optuna_val_ds = PandasWithMilDataset(images_dir, df_val, transforms=val_transforms)\n",
        "\n",
        "optuna_train_loader = DataLoader(\n",
        "    optuna_train_ds,\n",
        "    batch_size=OPTUNA_BATCH_SIZE,\n",
        "    num_workers=num_workers,\n",
        "    sampler=RandomSampler(optuna_train_ds),\n",
        "    pin_memory=True,\n",
        "    prefetch_factor=2,\n",
        "    persistent_workers=True,\n",
        "    drop_last=True,\n",
        ")\n",
        "\n",
        "optuna_val_loader = DataLoader(\n",
        "    optuna_val_ds,\n",
        "    batch_size=OPTUNA_BATCH_SIZE * 2,\n",
        "    num_workers=num_workers,\n",
        "    sampler=SequentialSampler(optuna_val_ds),\n",
        "    pin_memory=True,\n",
        "    prefetch_factor=2,\n",
        "    persistent_workers=True,\n",
        ")\n",
        "\n",
        "print(f\"Optuna Train batches: {len(optuna_train_loader)}\")\n",
        "print(f\"Optuna Validation batches: {len(optuna_val_loader)}\")\n"
    ]

    # 5. Model Setup
    nb['cells'][13]['source'] = [
        "def build_model():\n",
        "    load_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)\n",
        "    model = EfficientNetMIL(\n",
        "        model=load_model,\n",
        "        output_classes=output_classes,\n",
        "        dropout_rate=dropout_rate,\n",
        "        hidden_dim=512,\n",
        "        gated=True,\n",
        "        pool=\"att\",\n",
        "    )\n",
        "    return model.to(device)\n"
    ]

    # 6. We will drop cell 15 (Optimizer/Scheduler) entirely because it moves to objective
    
    # 7. Keep Cell 17 (Training step / val step)
    
    # 8. Re-write the training loop to be the objective function
    nb['cells'][19]['source'] = [
        "def objective(trial: optuna.Trial) -> float:\n",
        "    gamma   = trial.suggest_float('gamma',   0.5, 3.0)\n",
        "    w_focal = trial.suggest_float('w_focal', 0.1, 2.0)\n",
        "    w_ord   = trial.suggest_float('w_ord',   0.1, 2.0)\n",
        "    lr      = trial.suggest_float('lr',      1e-5, 1e-3, log=True)\n",
        "\n",
        "    model = build_model()\n",
        "    loss_function = BCEFocalLoss(gamma=gamma, w_focal=w_focal, w_ord=w_ord)\n",
        "    \n",
        "    optimizer = optim.Adam(model.parameters(), lr=lr / warmup_factor)\n",
        "    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_OPTUNA_EPOCHS - warmup_epochs)\n",
        "    scheduler = GradualWarmupScheduler(\n",
        "        optimizer,\n",
        "        multiplier=warmup_factor,\n",
        "        total_epoch=warmup_epochs,\n",
        "        after_scheduler=scheduler_cosine\n",
        "    )\n",
        "\n",
        "    scaler = torch.amp.GradScaler()\n",
        "    best_kappa = 0.0\n",
        "    \n",
        "    for epoch in range(N_OPTUNA_EPOCHS):\n",
        "        training_step(model, optuna_train_loader, optimizer, device, loss_function, scaler)\n",
        "        metrics = validation_step(model, optuna_val_loader, device, loss_function)\n",
        "        kappa = metrics['val_kappa']\n",
        "        \n",
        "        scheduler.step()\n",
        "\n",
        "        trial.report(kappa, epoch)\n",
        "        if trial.should_prune():\n",
        "            del model\n",
        "            torch.cuda.empty_cache()\n",
        "            raise optuna.exceptions.TrialPruned()\n",
        "\n",
        "        best_kappa = max(best_kappa, kappa)\n",
        "\n",
        "    del model\n",
        "    torch.cuda.empty_cache()\n",
        "    \n",
        "    return best_kappa\n"
    ]

    # Add a new cell to run optuna
    optuna_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "optuna.logging.set_verbosity(optuna.logging.WARNING)\n",
            "\n",
            "study = optuna.create_study(\n",
            "    study_name=STUDY_NAME,\n",
            "    direction='maximize',\n",
            "    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),\n",
            "    storage=STUDY_DB,\n",
            "    load_if_exists=True,\n",
            ")\n",
            "\n",
            "print(\"Starting Optuna optimization...\")\n",
            "study.optimize(objective, n_trials=N_TRIALS)\n",
            "\n",
            "print(\"\\n=========================================\")\n",
            "print(\"Best trial:\")\n",
            "best = study.best_trial\n",
            "print(f\"  Value (Kappa): {best.value:.4f}\")\n",
            "print(\"  Params:\")\n",
            "for key, value in best.params.items():\n",
            "    print(f\"    {key}: {value}\")\n"
        ]
    }
    nb['cells'].insert(20, optuna_cell)

    # Remove the plot / evaluate cells as Optuna replaces full training
    # Or keep them but mark them as Markdown or full retrain
    # Actually, swin-t-mil-optuna just added optuna importances and visualization
    vis_cell_1 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "importances = optuna.importance.get_param_importances(study)\n",
            "print('Importância dos hiperparâmetros:')\n",
            "for param, imp in importances.items():\n",
            "    print(f'  {param}: {imp:.4f}')\n"
        ]
    }
    
    vis_cell_2 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from optuna import visualization as optvis\n",
            "\n",
            "fig = optvis.plot_optimization_history(study)\n",
            "fig.update_layout(title='Histórico de Otimização — Kappa por Trial', height=450)\n",
            "fig.show()\n"
        ]
    }

    nb['cells'].insert(21, vis_cell_1)
    nb['cells'].insert(22, vis_cell_2)

    # delete old cells (cell 15, and cells after 22)
    # The cell indices:
    # 15 was Optimizer, we clear it
    nb['cells'][15]['source'] = []
    
    # We should delete the old plot training history (now cell 23, 24) and test sets
    # Rather than mapping indices which might shift, let's build a new cells array
    
    # New cells list
    new_cells = []
    for i, cell in enumerate(nb['cells']):
        # If it's cell 15 (empty now), skip
        if i == 15:
            continue
        # We don't need the Plot Training History and Eval on test set cells.
        if i >= 23: 
            break
        new_cells.append(cell)

    dashboard_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(f'Storage: {STUDY_DB}')\n",
            "print(f'Study  : {STUDY_NAME}')\n",
            "print('\\nPara abrir o dashboard, execute no terminal:')\n",
            "print(f'  optuna-dashboard {STUDY_DB}')\n"
        ]
    }
    new_cells.append(dashboard_cell)

    nb['cells'] = new_cells

    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    modify()
