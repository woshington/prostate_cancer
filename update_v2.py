import json
import numpy as np

def modify():
    path = 'code/tests/baseline/efficientnet-v2-entropy-ordinal-focal.ipynb'
    with open(path, 'r') as f:
        nb = json.load(f)

    # Modify Imports
    nb['cells'][0]['source'] = [
        "from torch import optim\n",
        "from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights\n",
        "import torch\n",
        "import random\n",
        "import numpy as np\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
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
        "sys.path.append('../../..')\n",
        "\n",
        "from utils.dataset import PandasDataset\n",
        "from utils.models import EfficientNetApi\n"
    ]

    # Modify Config
    nb['cells'][1]['source'] = [
        "# Training parameters\n",
        "seed = 42\n",
        "batch_size = 3\n",
        "num_workers = 4\n",
        "output_classes = 5  # For ordinal encoding: ISUP 0-5 → 5 thresholds\n",
        "warmup_factor = 2\n",
        "warmup_epochs = 1\n",
        "\n",
        "# Optuna Config\n",
        "OPTUNA_SUBSET_FRAC = 0.25\n",
        "N_OPTUNA_EPOCHS = 10\n",
        "N_TRIALS = 25\n",
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
        "images_dir = os.path.join(data_dir, 'tiles')\n",
        "\n",
        "# Output paths\n",
        "os.makedirs('../logs', exist_ok=True)\n",
        "os.makedirs('../models', exist_ok=True)\n",
        "model_path = 'models/efficientnet-v2-entropy-ordinal-focal-optuna.pth'\n",
        "log_path = 'logs/efficientnet-v2-entropy-ordinal-focal-optuna.txt'\n",
        "STUDY_DB = 'sqlite:///../logs/efficientnet-v2-optuna.db'\n",
        "STUDY_NAME = 'efficientnet-v2-optuna'\n"
    ]

    # Cell 2 is FocalOrdinalRegressionLoss
    source2 = nb['cells'][2]['source']
    new_source2 = []
    for line in source2:
        if "loss_function = FocalOrdinalRegressionLoss" in line:
            break
        new_source2.append(line)
    
    decode_fn = [
        "def decode_ordinal_predictions(logits):\n",
        "    probs = torch.sigmoid(logits)\n",
        "    predictions = (probs > 0.5).sum(dim=1)\n",
        "    return predictions\n"
    ]
    nb['cells'][2]['source'] = new_source2 + ["\n"] + decode_fn

    # Dataloaders (Cell 5)
    nb['cells'][5]['source'] = [
        "# Create datasets\n",
        "rng_optuna = np.random.default_rng(seed)\n",
        "optuna_idx = rng_optuna.choice(len(df_train), size=int(len(df_train) * OPTUNA_SUBSET_FRAC), replace=False)\n",
        "df_optuna = df_train.iloc[optuna_idx].reset_index(drop=True)\n",
        "print(f'Optuna subset: {len(df_optuna)} imgs ({OPTUNA_SUBSET_FRAC:.0%})')\n",
        "\n",
        "optuna_train_dataset = PandasDataset(images_dir, df_optuna, transforms=train_transforms, format=\"png\")\n",
        "optuna_valid_dataset = PandasDataset(images_dir, df_val, transforms=val_transforms, format=\"png\")\n",
        "\n",
        "optuna_train_loader = DataLoader(\n",
        "    optuna_train_dataset,\n",
        "    batch_size=batch_size,\n",
        "    num_workers=num_workers,\n",
        "    sampler=RandomSampler(optuna_train_dataset),\n",
        "    pin_memory=True,\n",
        "    persistent_workers=True\n",
        ")\n",
        "\n",
        "optuna_valid_loader = DataLoader(\n",
        "    optuna_valid_dataset,\n",
        "    batch_size=batch_size,\n",
        "    num_workers=num_workers,\n",
        "    sampler=SequentialSampler(optuna_valid_dataset),\n",
        "    pin_memory=True,\n",
        "    persistent_workers=True\n",
        ")\n",
        "\n",
        "test_dataset = PandasDataset(images_dir, df_test, transforms=val_transforms, format=\"png\")\n",
        "test_loader = DataLoader(\n",
        "    test_dataset,\n",
        "    batch_size=batch_size,\n",
        "    num_workers=num_workers,\n",
        "    shuffle=False\n",
        ")\n",
        "\n",
        "print(f\"Optuna Train batches: {len(optuna_train_loader)}\")\n",
        "print(f\"Validation batches: {len(optuna_valid_loader)}\")\n",
        "print(f\"Test batches: {len(test_loader)}\")\n"
    ]

    # Model Setup (Cell 6) - wrap in build_model(trial)
    nb['cells'][6]['source'] = [
        "def build_model(trial):\n",
        "    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.6)\n",
        "    load_model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)\n",
        "    model = EfficientNetApi(model=load_model, output_dimensions=output_classes, dropout_rate=dropout_rate)\n",
        "    return model.to(device)\n"
    ]

    # Instead of clearing Cell 7, set scaler globally
    nb['cells'][7]['source'] = [
        "scaler = torch.amp.GradScaler()\n"
    ]

    # Keep train/val steps (Cell 8)

    # Rewrite Training Loop to Objective (Cell 9)
    nb['cells'][9]['source'] = [
        "def objective(trial: optuna.Trial) -> float:\n",
        "    alpha = trial.suggest_float('alpha', 0.1, 2.0)\n",
        "    beta = trial.suggest_float('beta', 0.1, 2.0)\n",
        "    gamma = trial.suggest_float('gamma', 0.5, 3.0)\n",
        "    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)\n",
        "    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)\n",
        "\n",
        "    model = build_model(trial)\n",
        "    loss_function = FocalOrdinalRegressionLoss(alpha=alpha, beta=beta, gamma=gamma).to(device)\n",
        "\n",
        "    optimizer = optim.AdamW(model.parameters(), lr=lr / warmup_factor, weight_decay=weight_decay)\n",
        "    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_OPTUNA_EPOCHS - warmup_epochs)\n",
        "    scheduler = GradualWarmupScheduler(\n",
        "        optimizer, multiplier=warmup_factor, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine\n",
        "    )\n",
        "\n",
        "    best_kappa = 0.0\n",
        "    \n",
        "    for epoch in range(N_OPTUNA_EPOCHS):\n",
        "        training_step(model, optuna_train_loader, optimizer, device, loss_function)\n",
        "        metrics = validation_step(model, optuna_valid_loader, device, loss_function)\n",
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

    vis_cell_1 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "importances = optuna.importance.get_param_importances(study)\n",
            "print('Hyperparameter importance:')\n",
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
            "fig.update_layout(title='Optimization History', height=450)\n",
            "fig.show()\n"
        ]
    }

    dashboard_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(f'Storage: {STUDY_DB}')\n",
            "print(f'Study  : {STUDY_NAME}')\n",
            "print('\\nTo open dashboard, run in terminal:')\n",
            "print(f'  optuna-dashboard {STUDY_DB}')\n"
        ]
    }

    new_cells = []
    for i in range(10):
        new_cells.append(nb['cells'][i])
    
    new_cells.extend([optuna_cell, vis_cell_1, vis_cell_2, dashboard_cell])

    nb['cells'] = new_cells

    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    modify()
