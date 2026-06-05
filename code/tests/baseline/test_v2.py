# Training parameters
seed = 42
batch_size = 3
num_workers = 4
output_classes = 5  # For ordinal encoding: ISUP 0-5 → 5 thresholds
warmup_factor = 2
warmup_epochs = 1

# Optuna Config
OPTUNA_SUBSET_FRAC = 0.25
N_OPTUNA_EPOCHS = 10
N_TRIALS = 25

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set seeds
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Paths
ROOT_DIR = '../../..'
data_dir = '../../../..'
images_dir = os.path.join(data_dir, 'tiles')

# Output paths
os.makedirs('../logs', exist_ok=True)
os.makedirs('../models', exist_ok=True)
model_path = 'models/efficientnet-v2-entropy-ordinal-focal-optuna.pth'
log_path = 'logs/efficientnet-v2-entropy-ordinal-focal-optuna.txt'
STUDY_DB = 'sqlite:///../logs/efficientnet-v2-optuna.db'
STUDY_NAME = 'efficientnet-v2-optuna'


# Training parameters
seed = 42
batch_size = 3
num_workers = 4
output_classes = 5  # For ordinal encoding: ISUP 0-5 → 5 thresholds
init_lr = 3e-4
warmup_factor = 2
warmup_epochs = 1
n_epochs = 50
dropout_rate = 0.2
patience = 7

# Focal loss parameters
focal_alpha = 0.25  # Weighting factor for focal loss
focal_gamma = 2.0   # Focusing parameter (higher = more focus on hard examples)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set seeds
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Paths
ROOT_DIR = '../../..'
data_dir = '../../../..'
images_dir = os.path.join(data_dir, 'tiles')

# Output paths
os.makedirs('../logs', exist_ok=True)
os.makedirs('../models', exist_ok=True)
model_path = 'models/efficientnet-v2-entropy-ordinal-focal.pth'
log_path = 'logs/efficientnet-v2-entropy-ordinal-focal.txt'

# Create datasets
rng_optuna = np.random.default_rng(seed)
optuna_idx = rng_optuna.choice(len(df_train), size=int(len(df_train) * OPTUNA_SUBSET_FRAC), replace=False)
df_optuna = df_train.iloc[optuna_idx].reset_index(drop=True)
print(f'Optuna subset: {len(df_optuna)} imgs ({OPTUNA_SUBSET_FRAC:.0%})')

optuna_train_dataset = PandasDataset(images_dir, df_optuna, transforms=train_transforms, format="png")
optuna_valid_dataset = PandasDataset(images_dir, df_val, transforms=val_transforms, format="png")

optuna_train_loader = DataLoader(
    optuna_train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=RandomSampler(optuna_train_dataset),
    pin_memory=True,
    persistent_workers=True
)

optuna_valid_loader = DataLoader(
    optuna_valid_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=SequentialSampler(optuna_valid_dataset),
    pin_memory=True,
    persistent_workers=True
)

test_dataset = PandasDataset(images_dir, df_test, transforms=val_transforms, format="png")
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False
)

print(f"Optuna Train batches: {len(optuna_train_loader)}")
print(f"Validation batches: {len(optuna_valid_loader)}")
print(f"Test batches: {len(test_loader)}")


scaler = torch.amp.GradScaler()


def objective(trial: optuna.Trial) -> float:
    alpha = trial.suggest_float('alpha', 0.1, 2.0)
    beta = trial.suggest_float('beta', 0.1, 2.0)
    gamma = trial.suggest_float('gamma', 0.5, 3.0)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    model = build_model(trial)
    loss_function = FocalOrdinalRegressionLoss(alpha=alpha, beta=beta, gamma=gamma).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr / warmup_factor, weight_decay=weight_decay)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_OPTUNA_EPOCHS - warmup_epochs)
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=warmup_factor, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine
    )

    best_kappa = 0.0
    
    for epoch in range(N_OPTUNA_EPOCHS):
        training_step(model, optuna_train_loader, optimizer, device, loss_function)
        metrics = validation_step(model, optuna_valid_loader, device, loss_function)
        kappa = metrics['val_kappa']
        
        scheduler.step()

        trial.report(kappa, epoch)
        if trial.should_prune():
            del model
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()

        best_kappa = max(best_kappa, kappa)

    del model
    torch.cuda.empty_cache()
    
    return best_kappa


optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(
    study_name=STUDY_NAME,
    direction='maximize',
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    storage=STUDY_DB,
    load_if_exists=True,
)

print("Starting Optuna optimization...")
study.optimize(objective, n_trials=N_TRIALS)

print("\n=========================================")
print("Best trial:")
best = study.best_trial
print(f"  Value (Kappa): {best.value:.4f}")
print("  Params:")
for key, value in best.params.items():
    print(f"    {key}: {value}")


importances = optuna.importance.get_param_importances(study)
print('Hyperparameter importance:')
for param, imp in importances.items():
    print(f'  {param}: {imp:.4f}')


from optuna import visualization as optvis

fig = optvis.plot_optimization_history(study)
fig.update_layout(title='Optimization History', height=450)
fig.show()


print(f'Storage: {STUDY_DB}')
print(f'Study  : {STUDY_NAME}')
print('\nTo open dashboard, run in terminal:')
print(f'  optuna-dashboard {STUDY_DB}')
