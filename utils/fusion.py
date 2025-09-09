import torch
import torch.nn as nn
import efficientnet_pytorch as efficientnet_model
from torch.utils.data import Dataset
import numpy as np
from skimage import io as skio, color

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        out = avg_out + max_out
        return self.sigmoid(out).unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_concat)
        return self.sigmoid(x_out)


class CombinedAttention(nn.Module):
    def __init__(self, in_channels):
        super(CombinedAttention, self).__init__()
        self.channel_att = ChannelAttention(in_channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x_channel = self.channel_att(x) * x
        x_spatial = self.spatial_att(x) * x
        # Uma estratégia de fusão simples é somar. Você pode experimentar outras.
        return x_channel + x_spatial


class EfficientNetFusion(nn.Module):
    def __init__(self, backbone, output_dimensions, attention_type='channel', num_modalities=3):
        super(EfficientNetFusion, self).__init__()

        self.backbones = nn.ModuleList([
            efficientnet_model.EfficientNet.from_pretrained(backbone)
            for _ in range(num_modalities)
        ])

        for i in range(num_modalities):
            self.backbones[i]._fc = nn.Identity()

        feature_size = self.backbones[0]._fc.in_features

        # Módulos de atenção
        attention_module = {
            'channel': ChannelAttention(feature_size),
            'spatial': SpatialAttention(),
            'combined': CombinedAttention(feature_size)
        }.get(attention_type)

        self.attention = attention_module

        self.fully_connected = nn.Linear(feature_size * num_modalities, output_dimensions)

    def forward(self, inputs):
        features = [self.backbones[i](inputs[i]) for i in range(len(inputs))]

        weighted_features = []
        for feat in features:
            if isinstance(self.attention, ChannelAttention) or isinstance(self.attention, CombinedAttention):
                weights = self.attention(feat)
                weighted_features.append(feat * weights)
            elif isinstance(self.attention, SpatialAttention):
                weights = self.attention(feat)
                weighted_features.append(feat * weights)

        fused_features = torch.cat(weighted_features, dim=1)

        output = self.fully_connected(fused_features)
        return output


class PandasFusionDataset(Dataset):
    def __init__(self, image_dir, dataframe, transforms=None, normalize=False):
        self.image_dir = image_dir
        self.dataframe = dataframe
        self.transforms = transforms
        self.normalize = normalize

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        img_id = row.image_id.strip()
        file_path = f"{self.image_dir}/{img_id}.jpg"

        image = skio.imread(file_path)
        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        if self.normalize:
            image = image.astype(np.float32) / 255.0

        img_rgb = image.copy()
        img_xyz = color.rgb2xyz(image)
        img_lab = color.rgb2lab(image)

        img_rgb = np.transpose(img_rgb, (2, 0, 1))
        img_xyz = np.transpose(img_xyz, (2, 0, 1))
        img_lab = np.transpose(img_lab, (2, 0, 1))

        label = np.zeros(5).astype(np.float32)
        label[:row.isup_grade] = 1.

        return (
            torch.tensor(img_rgb, dtype=torch.float32),
            torch.tensor(img_xyz, dtype=torch.float32),
            torch.tensor(img_lab, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            img_id
        )
