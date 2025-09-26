#%%
from torch import optim
from torchvision.models import efficientnet_b0, efficientnet_b1, EfficientNet_B0_Weights, EfficientNet_B1_Weights
import torch
import random
import numpy as np
import torch.nn as nn
import albumentations as Albu
import pandas as pd
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
from warmup_scheduler import GradualWarmupScheduler
import os
from utils.dataset import PandasDataset
from utils.metrics import model_checkpoint
from utils.train import train_model
from utils.models import EfficientNetApi
from utils.models import FixedScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
#%%
seed = 42
shuffle = True
batch_size = 4
num_workers = 4
output_classes = 5
init_lr = 2e-6
warmup_factor = 5
warmup_epochs = 2
n_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = '../..'

data_dir = '../../../dataset'
images_dir = os.path.join(data_dir, 'tiles')
#%%
load_model = efficientnet_b0(
     weights=EfficientNet_B0_Weights.DEFAULT
)

model = EfficientNetApi(model=load_model, output_dimensions=output_classes)
model = model.to(device)

#%%
print("Using device:", device)
loss_function = nn.BCEWithLogitsLoss()

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
#%%
df_train_ = pd.read_csv(f"{ROOT_DIR}/data/train_5fold.csv")
df_train_.columns = df_train_.columns.str.strip()
train_indexes = np.where((df_train_['fold'] != 3))[0]
valid_indexes = np.where((df_train_['fold'] == 3))[0]
#
df_train = df_train_.loc[train_indexes]
df_val = df_train_.loc[valid_indexes]
df_test = pd.read_csv(f"{ROOT_DIR}/data/test.csv")
#%% md
# #### view data
#%%
(df_train.shape, df_val.shape, df_test.shape)
#%%
transforms = Albu.Compose([
    Albu.Transpose(p=0.5),
    Albu.VerticalFlip(p=0.5),
    Albu.HorizontalFlip(p=0.5),
])
#%%
df_train.columns = df_train.columns.str.strip()

train_dataset = PandasDataset(images_dir, df_train, transforms=transforms)
valid_dataset = PandasDataset(images_dir, df_val, transforms=None)
test_dataset = PandasDataset(images_dir, df_test, transforms=None)
#%%
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=RandomSampler(train_dataset)
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=batch_size, num_workers=num_workers, sampler = RandomSampler(valid_dataset)
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, num_workers=num_workers, sampler = RandomSampler(test_dataset)
)
#%%
optimizer = optim.Adam(model.parameters(), lr=init_lr)
scheduler = ReduceLROnPlateau(optimizer, patience=3)
#%%
train_model(
    model=model,
    epochs=n_epochs,
    optimizer=optimizer,
    scheduler=scheduler,
    train_dataloader=train_loader,
    valid_dataloader=valid_loader,
    checkpoint=model_checkpoint,
    device=device,
    loss_function=loss_function,
    path_to_save_metrics="logs/with-noise-rgb.txt",
    path_to_save_model="models/efficientnet.pth",
    patience=5,
)
#%% md
# # tests
#%%
# from utils.metrics import evaluation, format_metrics
# model.load_state_dict(
#     torch.load(f"models/efficientnet.pth")
# )
# response = evaluation(model, test_loader, device)
# result = format_metrics(response[0])
# print(result)